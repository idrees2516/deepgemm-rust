//! SM100 (Blackwell) host-side operations: the new-generation
//! `fp8_fp4_gemm` family (all operand dtype combinations, all grouped
//! layouts, all epilogues) plus the device-side quantization data path.
//!
//! Everything dispatches to the tcgen05 kernels in
//! [`crate::cuda::fp8_fp4_gemm_1d1d`] and [`crate::cuda::bf16_gemm_sm100`],
//! configured by [`crate::heuristics_sm100`].

use cudarc::driver::safe::{CudaFunction, CudaSlice, CudaStream, DevicePtr};
use cudarc::driver::sys::{self, CUtensorMap, CUtensorMapDataType, CUtensorMapSwizzle};

use crate::cuda::bf16_gemm_sm100::build_bf16_source;
use crate::cuda::fp8_fp4_gemm_1d1d::build_fp8_fp4_source;
use crate::cuda::sm100_cast::build_sm100_cast_source;
use crate::device::DgContext;
use crate::heuristics_sm100::{best_config, to_bf16_kernel_cfg, to_fp8fp4_kernel_cfg, Sm100Desc};
use crate::launch::{ArgBuilder, DevPtr, LaunchGrid};
use crate::types::{DgError, DgResult, GemmType};

// ===========================================================================
// operand / output types
// ===========================================================================

/// A device-resident low-precision operand (e4m3 FP8 or e2m1 FP4) with
/// packed-UE8M0 scale factors in the 1d1d layout.
///
/// * `data`: `(rows, k)` K-major storage — e4m3 bytes, or **packed e2m1**
///   (two 4-bit codes per byte) for FP4. `ld` is in **logical** K elements.
/// * `sf`: `(ceil(k / (4 * gran_k)) [* num_groups], tma_aligned(rows))`
///   `u32` words (see [`crate::cuda::sm100_cast`] for the exact layout).
/// * FP4 operands must be K-major (MXF4 is K-major only).
#[derive(Debug, Clone, Copy)]
pub struct LpOperand<'a> {
    pub data: &'a CudaSlice<u8>,
    pub sf: &'a CudaSlice<u32>,
    pub rows: u32,
    pub k: u32,
    pub ld: u32,
    /// 8 = e4m3, 4 = e2m1 (packed).
    pub bits: u32,
    /// Scale-factor granularity along K: 32 (MXFP4/MXFP8) or 128.
    pub gran_k: u32,
    /// MN-major storage (`data` is `(k, rows)` with `ld` the MN stride).
    pub major_mn: bool,
}

impl<'a> LpOperand<'a> {
    pub fn check(&self) -> DgResult<()> {
        if self.bits != 8 && self.bits != 4 {
            return Err(DgError::Shape(format!("invalid bits {}", self.bits)));
        }
        if self.gran_k != 32 && self.gran_k != 128 {
            return Err(DgError::Shape(format!("invalid gran_k {}", self.gran_k)));
        }
        if self.bits == 4 && self.major_mn {
            return Err(DgError::Shape("FP4 operands must be K-major".into()));
        }
        if self.bits == 4 && (self.k % 128 != 0 || self.ld % 128 != 0) {
            return Err(DgError::Shape(
                "FP4 k and ld must be multiples of 128".into(),
            ));
        }
        if self.k % self.gran_k != 0 {
            return Err(DgError::Shape(format!(
                "k ({}) must be a multiple of the SF granularity ({})",
                self.k, self.gran_k
            )));
        }
        let storage =
            self.rows as usize * self.ld.div_ceil(if self.bits == 4 { 2 } else { 1 }) as usize;
        if self.data.len() < storage {
            return Err(DgError::Shape(format!(
                "data buffer too small: {} < {}",
                self.data.len(),
                storage
            )));
        }
        let cols = self.k.div_ceil(self.gran_k * 4) as usize;
        let aligned = crate::types::tma_aligned_size(self.rows, 4) as usize;
        if self.sf.len() < cols * aligned {
            return Err(DgError::Shape(format!(
                "sf buffer too small: {} < {} x {}",
                self.sf.len(),
                cols,
                aligned
            )));
        }
        Ok(())
    }
}

/// The GEMM output: bf16 / fp32 direct store, or e4m3 with dynamic
/// per-row per-32-column UE8M0 output scale factors (`sfd`, packed layout
/// identical to [`LpOperand::sf`]; batched GEMMs only, like upstream).
pub enum Sm100Out<'a> {
    Bf16 {
        buf: &'a mut CudaSlice<u16>,
        ld: u32,
    },
    F32 {
        buf: &'a mut CudaSlice<f32>,
        ld: u32,
    },
    /// `sfd`: `(ceil(n / 128), tma_aligned(m))` u32 words.
    Fp8 {
        buf: &'a mut CudaSlice<u8>,
        ld: u32,
        sfd: &'a mut CudaSlice<u32>,
    },
}

/// Epilogue options.
#[derive(Debug, Clone, Copy)]
pub struct Sm100Epilogue {
    /// Multiply the product before storing (BLAS `alpha`).
    pub alpha: Option<f32>,
    /// `D += A @ B` — the output buffer must already contain the addend.
    pub accumulate: bool,
    /// Tensor-core utilization control in percent (100 = unthrottled).
    pub tc_util: u32,
}

impl Default for Sm100Epilogue {
    fn default() -> Self {
        Self {
            alpha: None,
            accumulate: false,
            tc_util: 100,
        }
    }
}

fn cd_dtype_id(out: &Sm100Out) -> u32 {
    match out {
        Sm100Out::Bf16 { .. } => 0,
        Sm100Out::F32 { .. } => 1,
        Sm100Out::Fp8 { .. } => 2,
    }
}

fn epilogue_op(epi: &Sm100Epilogue, out: &Sm100Out) -> DgResult<u32> {
    if let Sm100Out::Fp8 { .. } = out {
        if epi.alpha.is_some() || epi.accumulate {
            return Err(DgError::Shape(
                "the FP8-quantized output cannot combine alpha/accumulation".into(),
            ));
        }
        return Ok(3);
    }
    if epi.accumulate && epi.alpha.is_some() {
        return Err(DgError::Shape(
            "alpha and accumulation are mutually exclusive".into(),
        ));
    }
    Ok(if epi.alpha.is_some() { 1 } else { 0 })
}

/// Upstream `DG_PRINT_CONFIGS` parity: print each unique problem -> config
/// mapping once, so benchmark runs can be correlated with tile choices.
fn print_config_once(problem: &str, config: &str) {
    use std::sync::atomic::{AtomicBool, Ordering};
    static ENABLED: AtomicBool = AtomicBool::new(false);
    static INIT: AtomicBool = AtomicBool::new(false);
    if !INIT.swap(true, Ordering::Relaxed) {
        ENABLED.store(
            std::env::var("DG_PRINT_CONFIGS").is_ok_and(|v| v != "0"),
            Ordering::Relaxed,
        );
    }
    if !ENABLED.load(Ordering::Relaxed) {
        return;
    }
    use std::collections::HashSet;
    use std::sync::Mutex;
    static PRINTED: std::sync::OnceLock<Mutex<HashSet<String>>> = std::sync::OnceLock::new();
    let set = PRINTED.get_or_init(|| Mutex::new(HashSet::new()));
    if let Ok(mut guard) = set.lock() {
        if guard.insert(problem.to_string()) {
            println!("[deepgemm] {problem}: {config}");
        }
    }
}

// ===========================================================================
// TMA descriptor builders (SM100 semantics, incl. the FP4 packed dtypes)
// ===========================================================================

fn tma_check(res: sys::CUresult) -> DgResult<()> {
    if res == sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(DgError::Driver(format!(
            "cuTensorMapEncodeTiled failed: {res:?}"
        )))
    }
}

fn mode_to_swizzle(mode: u32) -> DgResult<CUtensorMapSwizzle> {
    Ok(match mode {
        0 | 16 => CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        32 => CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_32B,
        64 => CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B,
        128 => CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
        m => return Err(DgError::Shape(format!("unsupported swizzle mode {m}"))),
    })
}

/// 2-D tiled TMA descriptor with an explicit dtype and box (SM100 variant:
/// FP4 operands use the special `16U4_ALIGN*` dtypes with logical dims).
#[allow(clippy::too_many_arguments)]
fn tma_2d_sm100(
    addr: u64,
    dtype: CUtensorMapDataType,
    gmem_inner: u32,
    gmem_outer: u32,
    gmem_outer_stride_bytes: u64,
    box_inner: u32,
    box_outer: u32,
    swizzle_mode: u32,
) -> DgResult<CUtensorMap> {
    if gmem_outer_stride_bytes % 16 != 0 {
        return Err(DgError::Shape(format!(
            "TMA outer stride ({gmem_outer_stride_bytes}B) must be 16B-aligned"
        )));
    }
    let mut tm: CUtensorMap = unsafe { std::mem::zeroed() };
    let gdim = [gmem_inner as u64, gmem_outer as u64];
    let gstride = [gmem_outer_stride_bytes];
    let bdim = [box_inner, box_outer];
    let estride = [1u32, 1u32];
    tma_check(unsafe {
        sys::cuTensorMapEncodeTiled(
            &mut tm,
            dtype,
            2,
            addr as *mut _,
            gdim.as_ptr(),
            gstride.as_ptr(),
            bdim.as_ptr(),
            estride.as_ptr(),
            sys::CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            mode_to_swizzle(swizzle_mode)?,
            sys::CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            sys::CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
    })?;
    Ok(tm)
}

/// 3-D tiled TMA descriptor (batched/grouped operands).
#[allow(clippy::too_many_arguments)]
fn tma_3d_sm100(
    addr: u64,
    dtype: CUtensorMapDataType,
    gmem_dims: [u32; 3],
    gmem_strides_bytes: [u64; 2],
    box_dims: [u32; 3],
    swizzle_mode: u32,
) -> DgResult<CUtensorMap> {
    let mut tm: CUtensorMap = unsafe { std::mem::zeroed() };
    let gdim = [
        gmem_dims[0] as u64,
        gmem_dims[1] as u64,
        gmem_dims[2] as u64,
    ];
    let bdim = [box_dims[0], box_dims[1], box_dims[2]];
    let estride = [1u32, 1u32, 1u32];
    tma_check(unsafe {
        sys::cuTensorMapEncodeTiled(
            &mut tm,
            dtype,
            3,
            addr as *mut _,
            gdim.as_ptr(),
            gmem_strides_bytes.as_ptr(),
            bdim.as_ptr(),
            estride.as_ptr(),
            sys::CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            mode_to_swizzle(swizzle_mode)?,
            sys::CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            sys::CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
    })?;
    Ok(tm)
}

/// Data-operand TMA descriptor (2-D or batched 3-D).
///
/// * K-major: gmem `(k, mn [* groups])`, box `(swizzle_atoms..., block_mn)`.
/// * MN-major: gmem `(mn, k [* groups])`, box `(swizzle/elem, block_k)`.
/// * Packed FP4 uses `16U4_ALIGN8B` (MXF4, packed SMEM) or `16U4_ALIGN16B`
///   (mixed MXF8F6F4, unpacked SMEM) — dims in logical 4-bit elements.
#[allow(clippy::too_many_arguments)]
fn make_operand_tma(
    stream: &CudaStream,
    op: &LpOperand,
    num_groups: u32,
    load_block_mn: u32,
    block_k: u32,
    swizzle: u32,
    batched_3d: bool,
    fp4_packed_smem: bool,
) -> DgResult<CUtensorMap> {
    if op.bits == 4 && op.major_mn {
        return Err(DgError::Shape("FP4 operands must be K-major".into()));
    }
    let addr = dp(stream, op.data);
    let ld_bytes = if op.bits == 4 {
        // packed gmem storage: one byte per two logical elements
        op.ld as u64 / 2
    } else {
        op.ld as u64
    };
    let dtype = if op.bits == 4 {
        if fp4_packed_smem {
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B
        } else {
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B
        }
    } else {
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8
    };
    // box inner in *logical* elements: FP4 packed = 2 per storage byte,
    // FP4 unpacked = 1 byte per element, FP8 = 1 byte per element.
    let box_inner = if op.bits == 4 {
        if fp4_packed_smem {
            swizzle * 2
        } else {
            swizzle
        }
    } else {
        swizzle
    };

    if !batched_3d {
        if op.major_mn {
            // gmem (mn, k), stride = ld (MN-stride) bytes
            tma_2d_sm100(
                addr, dtype, op.rows, op.k, ld_bytes, box_inner, block_k, swizzle,
            )
        } else {
            tma_2d_sm100(
                addr,
                dtype,
                op.k,
                op.rows * num_groups,
                ld_bytes,
                box_inner,
                load_block_mn,
                swizzle,
            )
        }
    } else {
        // batched: gmem (inner, mn, groups)
        if op.major_mn {
            tma_3d_sm100(
                addr,
                dtype,
                [op.rows, op.k, num_groups],
                [ld_bytes, op.rows as u64 * ld_bytes],
                [box_inner, block_k, 1],
                swizzle,
            )
        } else {
            tma_3d_sm100(
                addr,
                dtype,
                [op.k, op.rows, num_groups],
                [ld_bytes, op.rows as u64 * ld_bytes],
                [box_inner, load_block_mn, 1],
                swizzle,
            )
        }
    }
}

/// Packed-UE8M0 SF TMA descriptor: gmem `(tma_aligned(mn), cols * groups)`
/// u32 words with mn contiguous.
#[allow(clippy::too_many_arguments)]
fn make_sf_tma(
    stream: &CudaStream,
    sf: &CudaSlice<u32>,
    mn: u32,
    k: u32,
    gran_k: u32,
    num_groups: u32,
    block_mn: u32,
    sf_block_k: u32,
) -> DgResult<CUtensorMap> {
    let aligned = crate::types::tma_aligned_size(mn, 4);
    let cols = k.div_ceil(gran_k * 4);
    tma_2d_sm100(
        dp(stream, sf),
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT32,
        aligned,
        cols * num_groups,
        aligned as u64 * 4,
        block_mn,
        sf_block_k,
        0,
    )
}

/// CD output TMA descriptor (bf16/f32/fp8, `(n, m [* groups])` n-major).
#[allow(clippy::too_many_arguments)]
fn make_cd_tma(
    stream: &CudaStream,
    out: &Sm100Out,
    m: u32,
    n: u32,
    num_groups: u32,
    store_block_m: u32,
    store_block_n: u32,
    swizzle_cd: u32,
    batched_3d: bool,
) -> DgResult<CUtensorMap> {
    let (addr, dtype, elem, ld) = match out {
        Sm100Out::Bf16 { buf, ld } => (
            dp(stream, buf),
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2u64,
            *ld as u64,
        ),
        Sm100Out::F32 { buf, ld } => (
            dp(stream, buf),
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
            4,
            *ld as u64,
        ),
        Sm100Out::Fp8 { buf, ld, .. } => (
            dp(stream, buf),
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
            1,
            *ld as u64,
        ),
    };
    let ld_bytes = ld * elem;
    let box_inner = if swizzle_cd == 0 {
        store_block_n
    } else {
        swizzle_cd / elem as u32
    };
    if !batched_3d {
        tma_2d_sm100(
            addr,
            dtype,
            n,
            m,
            ld_bytes,
            box_inner,
            store_block_m,
            swizzle_cd,
        )
    } else {
        tma_3d_sm100(
            addr,
            dtype,
            [n, m, num_groups],
            [ld_bytes, m as u64 * ld_bytes],
            [box_inner, store_block_m, 1],
            swizzle_cd,
        )
    }
}

fn dp<T>(stream: &CudaStream, s: &CudaSlice<T>) -> u64 {
    s.device_ptr(stream).0
}
// ===========================================================================
// the core launcher
// ===========================================================================

const FP8FP4_KERNEL_NAME: &str = "deepgemm_sm100_fp8_fp4_gemm";
const BF16_KERNEL_NAME: &str = "deepgemm_sm100_bf16_gemm";

/// Resolve the kernel template configuration and launch the FP8/FP4 GEMM.
#[allow(clippy::too_many_arguments)]
fn launch_fp8_fp4(
    ctx: &DgContext,
    a: &LpOperand,
    b: &LpOperand,
    out: &Sm100Out,
    gemm_type: GemmType,
    use_psum: bool,
    m: u32,
    n: u32,
    k: u32,
    num_groups: u32,
    expected_m: u32,
    expected_num_groups: u32,
    grouped_layout: Option<u64>,
    epi: &Sm100Epilogue,
) -> DgResult<()> {
    a.check()?;
    b.check()?;
    if a.k != k || b.k != k {
        return Err(DgError::Shape("A/B K dimensions must match".into()));
    }
    let epi_op = epilogue_op(epi, out)?;
    let is_mxf4 = a.bits == 4 && b.bits == 4;

    let desc = Sm100Desc {
        gemm_type,
        use_psum_layout: use_psum,
        m,
        n,
        k,
        num_groups,
        expected_m,
        expected_num_groups,
        a_bits: a.bits,
        b_bits: b.bits,
        major_a_mn: a.major_mn,
        major_b_mn: b.major_mn,
        cd_dtype: cd_dtype_id(out),
        with_accumulation: epi.accumulate,
        num_sms: ctx.arch.num_sms,
        tc_util: if epi.tc_util == 0 {
            100
        } else {
            epi.tc_util.min(100)
        },
        k_grouped: false,
    };
    let hcfg = best_config(&desc);
    let cfg = to_fp8fp4_kernel_cfg(
        &hcfg,
        &desc,
        a.gran_k,
        b.gran_k,
        num_groups,
        0,
        0,
        0,
        epi_op,
        epi.accumulate,
        is_mxf4,
    );
    print_config_once(
        &format!(
            "fp8_fp4_1d1d m={m} n={n} k={k} groups={num_groups} a_bits={} b_bits={} gran=({},{}) type={:?}{}",
            a.bits, b.bits, a.gran_k, b.gran_k, gemm_type,
            if use_psum { "+psum" } else { "" }
        ),
        &format!(
            "block={}x{}x{} stages={} store_stages={} cluster={}x{} swap_ab={} swizzle=({},{},{}) smem={}",
            cfg.block_m,
            cfg.block_n,
            cfg.block_k,
            cfg.num_stages,
            cfg.num_tma_store_stages,
            if cfg.is_multicast_on_a { 1 } else { cfg.multicast },
            if cfg.is_multicast_on_a { cfg.multicast } else { 1 },
            cfg.swap_ab,
            cfg.swizzle_a,
            cfg.swizzle_b,
            cfg.swizzle_cd,
            hcfg.smem_size
        ),
    );

    let load_block_m = cfg.block_m
        / if cfg.is_multicast_on_a {
            cfg.multicast
        } else {
            1
        };
    let load_block_n = cfg.block_n
        / if cfg.is_multicast_on_a {
            1
        } else {
            cfg.multicast
        };
    let sf_block_m = cfg.block_m.div_ceil(128) * 128;
    let sf_block_n = cfg.block_n.div_ceil(128) * 128;
    let sf_block_k = cfg.block_k / 128;
    let batched_3d = cfg.gemm_type == 4;

    let tm_a = make_operand_tma(
        &ctx.stream,
        a,
        num_groups,
        load_block_m,
        cfg.block_k,
        cfg.swizzle_a,
        batched_3d,
        is_mxf4,
    )?;
    let tm_b = make_operand_tma(
        &ctx.stream,
        b,
        num_groups,
        load_block_n,
        cfg.block_k,
        cfg.swizzle_b,
        batched_3d,
        is_mxf4,
    )?;
    // A's SF: Normal/contiguous share one flat M arena (1 column group);
    // masked and batched stack the groups along the SF column dimension
    // (the kernel indexes SFA columns with `group * shape_sfa_k + col`).
    let sfa_groups = match gemm_type {
        GemmType::Batched | GemmType::MGroupedMasked => num_groups,
        _ => 1,
    };
    let tm_sfa = make_sf_tma(
        &ctx.stream,
        a.sf,
        m,
        k,
        a.gran_k,
        sfa_groups,
        sf_block_m,
        sf_block_k,
    )?;
    let tm_sfb = make_sf_tma(
        &ctx.stream,
        b.sf,
        n,
        k,
        b.gran_k,
        num_groups,
        sf_block_n,
        sf_block_k,
    )?;

    let store_block_m = if cfg.swap_ab {
        16
    } else {
        cfg.block_m.min(128)
    };
    let store_block_n = if cfg.swap_ab {
        cfg.block_n
    } else {
        cfg.swizzle_cd / cd_elem_size(cd_dtype_id(out))
    };
    let tm_cd = make_cd_tma(
        &ctx.stream,
        out,
        m,
        n,
        num_groups,
        store_block_m,
        store_block_n,
        cfg.swizzle_cd,
        batched_3d,
    )?;

    // Epilogue runtime args
    let (sfd_ptr, sfd_stride) = match out {
        Sm100Out::Fp8 { sfd, .. } => (dp(&ctx.stream, sfd), crate::types::tma_aligned_size(m, 4)),
        _ => (0u64, 0u32),
    };
    let alpha = epi.alpha.unwrap_or(1.0);

    let source = build_fp8_fp4_source(&cfg);
    let module = ctx.jit.module(&ctx.device, &source)?;
    let func = module
        .load_function(FP8FP4_KERNEL_NAME)
        .map_err(|e| DgError::Driver(format!("{e:?}")))?;

    let layout_ptr = grouped_layout.unwrap_or(0);

    let mut args = ArgBuilder::new();
    args.push(&DevPtr(layout_ptr));
    args.push(&m);
    args.push(&n);
    args.push(&k);
    args.push(&DevPtr(sfd_ptr));
    args.push(&sfd_stride);
    args.push(&m);
    args.push(&n);
    args.push(&alpha);
    args.push_aligned(&tm_a, 64);
    args.push_aligned(&tm_b, 64);
    args.push_aligned(&tm_sfa, 64);
    args.push_aligned(&tm_sfb, 64);
    args.push_aligned(&tm_cd, 64);

    let cluster = if cfg.multicast > 1 {
        Some((cfg.multicast, 1, 1))
    } else {
        None
    };
    unsafe {
        args.launch(
            &func,
            &ctx.stream,
            LaunchGrid {
                grid_x: cfg.num_sms,
                block_x: cfg.num_non_epilogue_threads + cfg.num_epilogue_threads,
                smem: hcfg.smem_size,
                cluster,
                ..Default::default()
            },
        )
    }
}

fn cd_elem_size(dt: u32) -> u32 {
    match dt {
        1 => 4,
        2 => 1,
        _ => 2,
    }
}

// ===========================================================================
// public API: the new-generation fp8_fp4_gemm family
// ===========================================================================

/// Dense low-precision GEMM: `D = alpha * (A @ B^T) (+ C)`.
///
/// Operand majors select the logical layout natively (no transposes):
/// * `nt`: A `(m, k)` and B `(n, k)`, both K-major.
/// * `nn`: B `(k, n)` MN-major. `tn`: A `(k, m)` MN-major. `tt`: both MN-major.
/// * FP4 operands are K-major only (MXF4 constraint).
#[allow(clippy::too_many_arguments)]
pub fn fp8_fp4_gemm_dev(
    ctx: &DgContext,
    a: &LpOperand,
    b: &LpOperand,
    out: &Sm100Out,
    epi: &Sm100Epilogue,
) -> DgResult<()> {
    let (m, n, k) = (a.rows, b.rows, a.k);
    if let Sm100Out::Fp8 { .. } = out {
        return Err(DgError::Shape(
            "the dynamic FP8 output is only supported by the batched API".into(),
        ));
    }
    launch_fp8_fp4(
        ctx,
        a,
        b,
        out,
        GemmType::Normal,
        false,
        m,
        n,
        k,
        1,
        m,
        1,
        None,
        epi,
    )
}

/// MoE grouped GEMM, contiguous (psum) layout.
///
/// * `a`: `(m_total, k)` — tokens of all experts concatenated, each group's
///   start aligned to `psum` semantics; `m_indices[i]` = expert of row `i`
///   (`-1` for padding).
/// * `b`: `(num_groups * n, k)` — expert `g`'s weights at rows
///   `[g * n, (g + 1) * n)`; `b.sf` covers `(num_groups, n)` rows.
/// * `out`: `(m_total, n)` with each group starting at a 128-aligned row.
#[allow(clippy::too_many_arguments)]
pub fn m_grouped_fp8_fp4_gemm_contiguous_dev(
    ctx: &DgContext,
    a: &LpOperand,
    m_indices: &CudaSlice<i32>,
    b: &LpOperand,
    out: &Sm100Out,
    num_groups: u32,
    epi: &Sm100Epilogue,
) -> DgResult<()> {
    if let Sm100Out::Fp8 { .. } = out {
        return Err(DgError::Shape(
            "the dynamic FP8 output is only supported by the batched API".into(),
        ));
    }
    if epi.accumulate {
        return Err(DgError::Shape(
            "accumulation is not supported by the grouped contiguous layout".into(),
        ));
    }
    let (m, n, k) = (a.rows, b.rows / num_groups.max(1), a.k);
    // psum[g] = one-past-last row of group g
    let mut psum = unsafe {
        ctx.stream
            .alloc::<u32>(num_groups as usize)
            .map_err(|e| DgError::Driver(format!("{e:?}")))?
    };
    ctx.stream
        .memset_zeros(&mut psum)
        .map_err(|e| DgError::Driver(format!("{e:?}")))?;
    run_psum_kernel(ctx, m_indices, &psum, num_groups, m)?;
    launch_fp8_fp4(
        ctx,
        a,
        b,
        out,
        GemmType::MGroupedContiguous,
        true,
        m,
        n,
        k,
        num_groups,
        m,
        1,
        Some(dp(&ctx.stream, &psum)),
        epi,
    )
}

/// MoE grouped GEMM, masked layout.
///
/// * `a`: `(num_groups, m_max, k)`; `b`: `(num_groups * n, k)`;
///   `masked_m[g]` = valid rows of group g.
/// * `out`: `(num_groups * m_max, n)` bf16 — padding rows are written with
///   the same values as row 0 of the group (upstream behavior: only the
///   leading `masked_m[g]` rows are meaningful).
#[allow(clippy::too_many_arguments)]
pub fn m_grouped_fp8_fp4_gemm_masked_dev(
    ctx: &DgContext,
    a: &LpOperand,
    masked_m: &CudaSlice<i32>,
    b: &LpOperand,
    out: &Sm100Out,
    num_groups: u32,
    expected_m: u32,
    epi: &Sm100Epilogue,
) -> DgResult<()> {
    if let Sm100Out::Fp8 { .. } = out {
        return Err(DgError::Shape(
            "the dynamic FP8 output is only supported by the batched API".into(),
        ));
    }
    let m_max = a.rows;
    let n = b.rows / num_groups.max(1);
    let k = a.k;
    let a_sf_rows = m_max;
    // A's SF covers (groups, m_max): rows dimension is groups * m_max
    let a_full = LpOperand {
        rows: a_sf_rows,
        ..*a
    };
    launch_fp8_fp4(
        ctx,
        &a_full,
        b,
        out,
        GemmType::MGroupedMasked,
        false,
        m_max,
        n,
        k,
        num_groups,
        expected_m.max(1),
        num_groups,
        Some(dp(&ctx.stream, masked_m)),
        epi,
    )
}

/// Batched low-precision GEMM (bmm): `D[g] = A[g] @ B[g]^T`.
///
/// * `a`: `(num_groups, m, k)`; `b`: `(num_groups, n, k)`;
///   SFs cover the flattened `(groups * rows)` M/N dimension.
/// * The dynamic FP8 output (with `sfd`) is only available here, matching
///   upstream.
#[allow(clippy::too_many_arguments)]
pub fn fp8_fp4_bmm_dev(
    ctx: &DgContext,
    a: &LpOperand,
    b: &LpOperand,
    out: &Sm100Out,
    num_groups: u32,
    epi: &Sm100Epilogue,
) -> DgResult<()> {
    let (m, n, k) = (a.rows, b.rows, a.k);
    launch_fp8_fp4(
        ctx,
        a,
        b,
        out,
        GemmType::Batched,
        false,
        m,
        n,
        k,
        num_groups,
        m,
        1,
        None,
        epi,
    )
}

/// SM100-native BF16 GEMM: `D = alpha * (A @ B^T) (+ C)`.
#[allow(clippy::too_many_arguments)]
pub fn bf16_gemm_dev(
    ctx: &DgContext,
    a: &CudaSlice<u16>,
    a_ld: u32,
    m: u32,
    b: &CudaSlice<u16>,
    b_ld: u32,
    n: u32,
    k: u32,
    out: &Sm100Out,
    epi: &Sm100Epilogue,
) -> DgResult<()> {
    let epi_op = epilogue_op(epi, out)?;
    let desc = Sm100Desc {
        gemm_type: GemmType::Normal,
        use_psum_layout: false,
        m,
        n,
        k,
        num_groups: 1,
        expected_m: m,
        expected_num_groups: 1,
        a_bits: 16,
        b_bits: 16,
        major_a_mn: false,
        major_b_mn: false,
        cd_dtype: cd_dtype_id(out),
        with_accumulation: epi.accumulate,
        num_sms: ctx.arch.num_sms,
        tc_util: if epi.tc_util == 0 {
            100
        } else {
            epi.tc_util.min(100)
        },
        k_grouped: false,
    };
    let hcfg = best_config(&desc);
    let cfg = to_bf16_kernel_cfg(&hcfg, &desc, 1, 0, 0, 0, epi_op, epi.accumulate);
    print_config_once(
        &format!("bf16_sm100 m={m} n={n} k={k}"),
        &format!(
            "block={}x{}x{} stages={} cluster={}x{} swap_ab={} tc_util={}",
            cfg.block_m,
            cfg.block_n,
            cfg.block_k,
            cfg.num_stages,
            if cfg.is_multicast_on_a {
                1
            } else {
                cfg.multicast
            },
            if cfg.is_multicast_on_a {
                cfg.multicast
            } else {
                1
            },
            cfg.swap_ab,
            cfg.tc_util
        ),
    );

    let load_block_m = cfg.block_m
        / if cfg.is_multicast_on_a {
            cfg.multicast
        } else {
            1
        };
    let load_block_n = cfg.block_n
        / if cfg.is_multicast_on_a {
            1
        } else {
            cfg.multicast
        };
    let cd_esz = cd_elem_size(cfg.cd_dtype);
    let store_block_m = if cfg.swap_ab {
        16
    } else {
        cfg.block_m.min(128)
    };
    let store_block_n = if cfg.swap_ab {
        cfg.block_n
    } else {
        cfg.swizzle_cd / cd_esz
    };

    let tm_a = tma_2d_sm100(
        dp(&ctx.stream, a),
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        k,
        m,
        a_ld as u64 * 2,
        if cfg.swizzle_a != 0 {
            cfg.swizzle_a / 2
        } else {
            cfg.block_k
        },
        load_block_m,
        cfg.swizzle_a,
    )?;
    let tm_b = tma_2d_sm100(
        dp(&ctx.stream, b),
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        k,
        n,
        b_ld as u64 * 2,
        if cfg.swizzle_b != 0 {
            cfg.swizzle_b / 2
        } else {
            cfg.block_k
        },
        load_block_n,
        cfg.swizzle_b,
    )?;
    let tm_cd = make_cd_tma(
        &ctx.stream,
        out,
        m,
        n,
        1,
        store_block_m,
        store_block_n,
        cfg.swizzle_cd,
        false,
    )?;

    let (sfd_ptr, sfd_stride) = match out {
        Sm100Out::Fp8 { sfd, .. } => (dp(&ctx.stream, sfd), crate::types::tma_aligned_size(m, 4)),
        _ => (0u64, 0u32),
    };

    let source = build_bf16_source(&cfg);
    let module = ctx.jit.module(&ctx.device, &source)?;
    let func = module
        .load_function(BF16_KERNEL_NAME)
        .map_err(|e| DgError::Driver(format!("{e:?}")))?;

    let mut args = ArgBuilder::new();
    args.push(&DevPtr(0u64));
    args.push(&m);
    args.push(&n);
    args.push(&k);
    args.push(&DevPtr(sfd_ptr));
    args.push(&sfd_stride);
    args.push(&m);
    args.push(&n);
    args.push(&epi.alpha.unwrap_or(1.0));
    args.push_aligned(&tm_a, 64);
    args.push_aligned(&tm_b, 64);
    args.push_aligned(&tm_cd, 64);

    let cluster = if cfg.multicast > 1 {
        Some((cfg.multicast, 1, 1))
    } else {
        None
    };
    unsafe {
        args.launch(
            &func,
            &ctx.stream,
            LaunchGrid {
                grid_x: cfg.num_sms,
                block_x: cfg.num_non_epilogue_threads + cfg.num_epilogue_threads,
                smem: hcfg.smem_size,
                cluster,
                ..Default::default()
            },
        )
    }
}
// ===========================================================================
// device-side quantization data path
// ===========================================================================

fn cast_module_fn(ctx: &DgContext, name: &str) -> DgResult<CudaFunction> {
    let module = ctx.jit.module(&ctx.device, &build_sm100_cast_source())?;
    module
        .load_function(name)
        .map_err(|e| DgError::Driver(format!("{e:?}")))
}

fn run_psum_kernel(
    ctx: &DgContext,
    m_indices: &CudaSlice<i32>,
    psum: &CudaSlice<u32>,
    num_groups: u32,
    m: u32,
) -> DgResult<()> {
    let func = cast_module_fn(ctx, "deepgemm_psum_from_m_indices")?;
    let m_u32 = m;
    let mut args = ArgBuilder::new();
    args.push(&DevPtr(dp(&ctx.stream, m_indices)));
    args.push(&DevPtr(dp(&ctx.stream, psum)));
    args.push(&num_groups);
    args.push(&m_u32);
    unsafe { args.launch(&func, &ctx.stream, LaunchGrid::new(m.div_ceil(256), 256, 0)) }
}

/// The number of `u32` SF words along the packed-UE8M0 K dimension.
pub fn sf_packed_cols(k: u32, gran_k: u32) -> u32 {
    k.div_ceil(gran_k * 4)
}

/// The TMA-aligned row count of a packed SF matrix.
pub fn sf_aligned_rows(rows: u32) -> u32 {
    crate::types::tma_aligned_size(rows, 4)
}

/// Allocate the packed SF buffer for `(rows, k)` with granularity `gran_k`
/// (zero-initialized — padding words read as 2^0 scales).
///
/// `rows` is the **total** row count over `num_groups` groups
/// (`rows = num_groups * group_rows`); the buffer takes the grouped layout
/// `(tma_aligned(group_rows), cols * num_groups)` the SM100 SF TMA expects.
/// `num_groups = 1` gives the flat `(tma_aligned(rows), cols)` layout.
pub fn alloc_sf(
    ctx: &DgContext,
    rows: u32,
    k: u32,
    gran_k: u32,
    num_groups: u32,
) -> DgResult<CudaSlice<u32>> {
    let groups = num_groups.max(1);
    let group_rows = rows / groups;
    let mut buf = unsafe {
        ctx.stream
            .alloc::<u32>(
                sf_packed_cols(k, gran_k) as usize
                    * groups as usize
                    * sf_aligned_rows(group_rows) as usize,
            )
            .map_err(|e| DgError::Driver(format!("{e:?}")))?
    };
    ctx.stream
        .memset_zeros(&mut buf)
        .map_err(|e| DgError::Driver(format!("{e:?}")))?;
    Ok(buf)
}

/// Quantize a BF16 tensor into packed e2m1 (FP4) + packed UE8M0 per-32 SFs
/// (the MXFP4 recipe). `src` is `(rows, src_ld)` bf16 with
/// `rows = num_groups * group_rows`; returns `(packed_data (rows, k/2), sf)`
/// in the grouped-packed SF layout (see [`alloc_sf`]).
pub fn cast_bf16_to_fp4_packed_sf_dev(
    ctx: &DgContext,
    src: &CudaSlice<u16>,
    src_ld: u32,
    rows: u32,
    k: u32,
    num_groups: u32,
) -> DgResult<(CudaSlice<u8>, CudaSlice<u32>)> {
    if k % 128 != 0 {
        return Err(DgError::Shape("fp4 k must be a multiple of 128".into()));
    }
    let groups = num_groups.max(1);
    let group_rows = rows / groups;
    let dst = unsafe {
        ctx.stream
            .alloc::<u8>(rows as usize * k as usize / 2)
            .map_err(|e| DgError::Driver(format!("{e:?}")))?
    };
    let sf = alloc_sf(ctx, rows, k, 32, groups)?;
    let func = cast_module_fn(ctx, "deepgemm_cast_bf16_to_fp4_packed_sf")?;
    let mut args = ArgBuilder::new();
    args.push(&DevPtr(dp(&ctx.stream, src)));
    args.push(&src_ld);
    args.push(&DevPtr(dp(&ctx.stream, &dst)));
    args.push(&(k / 2));
    args.push(&DevPtr(dp(&ctx.stream, &sf)));
    args.push(&sf_aligned_rows(group_rows));
    args.push(&rows);
    args.push(&k);
    args.push(&groups);
    unsafe {
        args.launch(
            &func,
            &ctx.stream,
            LaunchGrid {
                grid_x: sf_packed_cols(k, 32),
                grid_y: rows.div_ceil(4),
                block_x: 1,
                block_y: 4,
                smem: 0,
                ..Default::default()
            },
        )?;
    }
    Ok((dst, sf))
}

/// Quantize a BF16 tensor into e4m3 + packed UE8M0 SFs.
/// `gran_k` = 32 (MXFP8) or 128 (the DeepSeek recipe); `rows` is the total
/// row count over `num_groups` groups (grouped-packed SF output).
#[allow(clippy::too_many_arguments)]
pub fn cast_bf16_to_fp8_sf_dev(
    ctx: &DgContext,
    src: &CudaSlice<u16>,
    src_ld: u32,
    rows: u32,
    k: u32,
    gran_k: u32,
    num_groups: u32,
) -> DgResult<(CudaSlice<u8>, CudaSlice<u32>)> {
    if gran_k != 32 && gran_k != 128 {
        return Err(DgError::Shape("gran_k must be 32 or 128".into()));
    }
    if k % (gran_k * 4) != 0 {
        return Err(DgError::Shape(format!(
            "k must be a multiple of {} for gran {gran_k}",
            gran_k * 4
        )));
    }
    let groups = num_groups.max(1);
    let group_rows = rows / groups;
    let dst = unsafe {
        ctx.stream
            .alloc::<u8>(rows as usize * k as usize)
            .map_err(|e| DgError::Driver(format!("{e:?}")))?
    };
    let sf = alloc_sf(ctx, rows, k, gran_k, groups)?;
    let name = if gran_k == 32 {
        "deepgemm_cast_bf16_to_fp8_sf_gran32"
    } else {
        "deepgemm_cast_bf16_to_fp8_sf_gran128"
    };
    let func = cast_module_fn(ctx, name)?;
    let mut args = ArgBuilder::new();
    args.push(&DevPtr(dp(&ctx.stream, src)));
    args.push(&src_ld);
    args.push(&DevPtr(dp(&ctx.stream, &dst)));
    args.push(&k);
    args.push(&DevPtr(dp(&ctx.stream, &sf)));
    args.push(&sf_aligned_rows(group_rows));
    args.push(&rows);
    args.push(&k);
    args.push(&groups);
    let elems_per_thread = gran_k * 4;
    unsafe {
        args.launch(
            &func,
            &ctx.stream,
            LaunchGrid {
                grid_x: k / elems_per_thread,
                grid_y: rows.div_ceil(4),
                block_x: 1,
                block_y: 4,
                smem: 0,
                ..Default::default()
            },
        )?;
    }
    Ok((dst, sf))
}

/// Transform power-of-two FP32 SFs `(rows, k/gran)` (K-major, stride
/// `src_ld`; `rows` = total over `num_groups` groups) into the packed
/// UE8M0 1d1d layout (grouped when `num_groups > 1`).
#[allow(clippy::too_many_arguments)]
pub fn transform_sf1d_packed_ue8m0_dev(
    ctx: &DgContext,
    src: &CudaSlice<f32>,
    src_ld: u32,
    rows: u32,
    k: u32,
    gran_k: u32,
    num_groups: u32,
) -> DgResult<CudaSlice<u32>> {
    let groups = num_groups.max(1);
    let group_rows = rows / groups;
    let dst = alloc_sf(ctx, rows, k, gran_k, groups)?;
    let func = cast_module_fn(ctx, "deepgemm_transform_sf1d_packed_ue8m0")?;
    let mut args = ArgBuilder::new();
    args.push(&DevPtr(dp(&ctx.stream, src)));
    args.push(&src_ld);
    args.push(&DevPtr(dp(&ctx.stream, &dst)));
    args.push(&sf_aligned_rows(group_rows));
    args.push(&rows);
    args.push(&k);
    args.push(&gran_k);
    args.push(&groups);
    unsafe {
        args.launch(
            &func,
            &ctx.stream,
            LaunchGrid {
                grid_x: sf_packed_cols(k, gran_k),
                grid_y: rows.div_ceil(4),
                block_x: 1,
                block_y: 4,
                smem: 0,
                ..Default::default()
            },
        )?;
    }
    Ok(dst)
}

/// Transform power-of-two FP32 2-D SFs `(ceil(rows/128), k/128)` (the
/// DeepSeek weight recipe) into the packed UE8M0 1d1d layout (gran 128).
/// `rows` = total over `num_groups` groups; each group must own a whole
/// number of 128-row tiles (`rows / num_groups % 128 == 0`).
#[allow(clippy::too_many_arguments)]
pub fn transform_sf2d_packed_ue8m0_dev(
    ctx: &DgContext,
    src: &CudaSlice<f32>,
    src_ld: u32,
    rows: u32,
    k: u32,
    num_groups: u32,
) -> DgResult<CudaSlice<u32>> {
    let groups = num_groups.max(1);
    let group_rows = rows / groups;
    if groups > 1 && group_rows % 128 != 0 {
        return Err(DgError::Shape(format!(
            "2-D SF groups must own whole 128-row tiles (group rows {group_rows})"
        )));
    }
    let dst = alloc_sf(ctx, rows, k, 128, groups)?;
    let func = cast_module_fn(ctx, "deepgemm_transform_sf2d_packed_ue8m0")?;
    let mut args = ArgBuilder::new();
    args.push(&DevPtr(dp(&ctx.stream, src)));
    args.push(&src_ld);
    args.push(&DevPtr(dp(&ctx.stream, &dst)));
    args.push(&sf_aligned_rows(group_rows));
    args.push(&rows);
    args.push(&k);
    args.push(&groups);
    unsafe {
        args.launch(
            &func,
            &ctx.stream,
            LaunchGrid {
                grid_x: sf_packed_cols(k, 128),
                grid_y: rows.div_ceil(4),
                block_x: 1,
                block_y: 4,
                smem: 0,
                ..Default::default()
            },
        )?;
    }
    Ok(dst)
}

/// Unpack packed e2m1 into one 4-bit code per byte (low nibble) — the raw
/// storage view for FP4 operands (utility / debugging).
pub fn unpack_fp4_raw_dev(
    ctx: &DgContext,
    src: &CudaSlice<u8>,
    packed_len: usize,
) -> DgResult<CudaSlice<u8>> {
    let dst = unsafe {
        ctx.stream
            .alloc::<u8>(packed_len * 2)
            .map_err(|e| DgError::Driver(format!("{e:?}")))?
    };
    let func = cast_module_fn(ctx, "deepgemm_unpack_fp4_raw")?;
    let len = packed_len as u32;
    let mut args = ArgBuilder::new();
    args.push(&DevPtr(dp(&ctx.stream, src)));
    args.push(&DevPtr(dp(&ctx.stream, &dst)));
    args.push(&len);
    unsafe {
        args.launch(
            &func,
            &ctx.stream,
            LaunchGrid::new(packed_len.div_ceil(256) as u32, 256, 0),
        )?;
    }
    Ok(dst)
}

// ===========================================================================
// old-API bridges: the SM90-era entry points dispatch here on Blackwell
// ===========================================================================

/// `transform_sf`-layout (transposed TMA-aligned f32) -> packed UE8M0.
/// `rows` = total over `num_groups` groups; the source carries the groups
/// stacked along its leading dim — `(groups * k/gran, tma_aligned(group_rows))`,
/// exactly what the SM90 [`crate::ops::transform_sf`] produces for batches.
pub fn transform_sf_t_packed_ue8m0_dev(
    ctx: &DgContext,
    src: &CudaSlice<f32>,
    rows: u32,
    k: u32,
    gran_k: u32,
    num_groups: u32,
) -> DgResult<CudaSlice<u32>> {
    let groups = num_groups.max(1);
    let group_rows = rows / groups;
    let dst = alloc_sf(ctx, rows, k, gran_k, groups)?;
    let func = cast_module_fn(ctx, "deepgemm_transform_sf_t_f32_to_packed_ue8m0")?;
    let stride = crate::types::tma_aligned_size(group_rows, 4);
    let mut args = ArgBuilder::new();
    args.push(&DevPtr(dp(&ctx.stream, src)));
    args.push(&stride);
    args.push(&DevPtr(dp(&ctx.stream, &dst)));
    args.push(&sf_aligned_rows(group_rows));
    args.push(&rows);
    args.push(&k);
    args.push(&gran_k);
    args.push(&groups);
    unsafe {
        args.launch(
            &func,
            &ctx.stream,
            LaunchGrid {
                grid_x: sf_packed_cols(k, gran_k),
                grid_y: rows.div_ceil(4),
                block_x: 1,
                block_y: 4,
                smem: 0,
                ..Default::default()
            },
        )?;
    }
    Ok(dst)
}

/// Bridge for the SM90-era `fp8_gemm_nt_dev` (1d2d recipe -> SM100 1d1d,
/// gran 128 on both sides).
#[allow(clippy::too_many_arguments)]
pub(crate) fn bridge_fp8_gemm_nt_dev(
    ctx: &DgContext,
    a: &crate::ops::Fp8Tensor,
    a_sf_t: &CudaSlice<f32>,
    b: &crate::ops::Fp8Tensor,
    sfb: &CudaSlice<f32>,
    out: &mut CudaSlice<u16>,
    out_ld: u32,
) -> DgResult<()> {
    let (m, n, k) = (a.rows, b.rows, a.k);
    let sf_a = transform_sf_t_packed_ue8m0_dev(ctx, a_sf_t, m, k, 128, 1)?;
    // sfb is (num_groups * n_tiles, k_blocks) f32 k-contiguous: tile rows
    let sf_b = transform_sf2d_packed_ue8m0_dev(ctx, sfb, k.div_ceil(128), n, k, 1)?;
    let a_op = LpOperand {
        data: a.data,
        sf: &sf_a,
        rows: m,
        k,
        ld: a.ld,
        bits: 8,
        gran_k: 128,
        major_mn: false,
    };
    let b_op = LpOperand {
        data: b.data,
        sf: &sf_b,
        rows: n,
        k,
        ld: b.ld,
        bits: 8,
        gran_k: 128,
        major_mn: false,
    };
    let o = Sm100Out::Bf16 {
        buf: out,
        ld: out_ld,
    };
    fp8_fp4_gemm_dev(ctx, &a_op, &b_op, &o, &Sm100Epilogue::default())
}

/// Bridge for the SM90-era contiguous grouped GEMM (psum layout on SM100).
#[allow(clippy::too_many_arguments)]
pub(crate) fn bridge_m_grouped_fp8_contiguous_dev(
    ctx: &DgContext,
    a: &crate::ops::Fp8Tensor,
    a_sf_t: &CudaSlice<f32>,
    b: &crate::ops::Fp8Tensor,
    sfb: &CudaSlice<f32>,
    out: &mut CudaSlice<u16>,
    out_ld: u32,
    m_indices: &CudaSlice<i32>,
    num_groups: u32,
) -> DgResult<()> {
    let (m, n, k) = (a.rows, b.rows / num_groups.max(1), a.k);
    // A's SF: one flat M arena (the contiguous kernel applies no group
    // offset to SFA columns); B's SF: groups stacked along the columns.
    let sf_a = transform_sf_t_packed_ue8m0_dev(ctx, a_sf_t, m, k, 128, 1)?;
    let sf_b = transform_sf2d_packed_ue8m0_dev(ctx, sfb, k.div_ceil(128), b.rows, k, num_groups)?;
    let a_op = LpOperand {
        data: a.data,
        sf: &sf_a,
        rows: m,
        k,
        ld: a.ld,
        bits: 8,
        gran_k: 128,
        major_mn: false,
    };
    let b_op = LpOperand {
        data: b.data,
        sf: &sf_b,
        rows: n,
        k,
        ld: b.ld,
        bits: 8,
        gran_k: 128,
        major_mn: false,
    };
    let o = Sm100Out::Bf16 {
        buf: out,
        ld: out_ld,
    };
    m_grouped_fp8_fp4_gemm_contiguous_dev(
        ctx,
        &a_op,
        m_indices,
        &b_op,
        &o,
        num_groups,
        &Sm100Epilogue::default(),
    )
}

/// Bridge for the SM90-era masked grouped GEMM.
#[allow(clippy::too_many_arguments)]
pub(crate) fn bridge_m_grouped_fp8_masked_dev(
    ctx: &DgContext,
    a: &crate::ops::Fp8Tensor,
    a_sf_t: &CudaSlice<f32>,
    b: &crate::ops::Fp8Tensor,
    sfb: &CudaSlice<f32>,
    out: &mut CudaSlice<u16>,
    out_ld: u32,
    masked_m: &CudaSlice<i32>,
    num_groups: u32,
    expected_m: u32,
) -> DgResult<()> {
    let m_max = a.rows;
    let n = b.rows / num_groups.max(1);
    let k = a.k;
    // Masked: both A's and B's SFs stack the groups along the SF columns
    // (the kernel indexes SFA/SFB columns with `group * shape_sf_k + col`).
    let sf_a =
        transform_sf_t_packed_ue8m0_dev(ctx, a_sf_t, m_max * num_groups, k, 128, num_groups)?;
    let sf_b = transform_sf2d_packed_ue8m0_dev(ctx, sfb, k.div_ceil(128), b.rows, k, num_groups)?;
    let a_op = LpOperand {
        data: a.data,
        sf: &sf_a,
        rows: m_max,
        k,
        ld: a.ld,
        bits: 8,
        gran_k: 128,
        major_mn: false,
    };
    let b_op = LpOperand {
        data: b.data,
        sf: &sf_b,
        rows: n,
        k,
        ld: b.ld,
        bits: 8,
        gran_k: 128,
        major_mn: false,
    };
    let o = Sm100Out::Bf16 {
        buf: out,
        ld: out_ld,
    };
    m_grouped_fp8_fp4_gemm_masked_dev(
        ctx,
        &a_op,
        masked_m,
        &b_op,
        &o,
        num_groups,
        expected_m,
        &Sm100Epilogue::default(),
    )
}

/// Bridge for the SM90-era batched GEMM.
#[allow(clippy::too_many_arguments)]
pub(crate) fn bridge_fp8_bmm_dev(
    ctx: &DgContext,
    a: &crate::ops::Fp8Tensor,
    a_sf_t: &CudaSlice<f32>,
    b: &crate::ops::Fp8Tensor,
    sfb: &CudaSlice<f32>,
    out: &mut CudaSlice<u16>,
    out_ld: u32,
    num_groups: u32,
) -> DgResult<()> {
    let (m, n, k) = (a.rows, b.rows, a.k);
    let sf_a = transform_sf_t_packed_ue8m0_dev(ctx, a_sf_t, m * num_groups, k, 128, num_groups)?;
    let sf_b = transform_sf2d_packed_ue8m0_dev(ctx, sfb, k.div_ceil(128), b.rows, k, num_groups)?;
    let a_op = LpOperand {
        data: a.data,
        sf: &sf_a,
        rows: m,
        k,
        ld: a.ld,
        bits: 8,
        gran_k: 128,
        major_mn: false,
    };
    let b_op = LpOperand {
        data: b.data,
        sf: &sf_b,
        rows: n,
        k,
        ld: b.ld,
        bits: 8,
        gran_k: 128,
        major_mn: false,
    };
    let o = Sm100Out::Bf16 {
        buf: out,
        ld: out_ld,
    };
    fp8_fp4_bmm_dev(ctx, &a_op, &b_op, &o, num_groups, &Sm100Epilogue::default())
}

/// Bridge for the SM90-era BF16 GEMM.
#[allow(clippy::too_many_arguments)]
pub(crate) fn bridge_bf16_gemm_nt_dev(
    ctx: &DgContext,
    a: &CudaSlice<u16>,
    b: &CudaSlice<u16>,
    out: &mut CudaSlice<u16>,
    m: u32,
    n: u32,
    k: u32,
    a_ld: u32,
    b_ld: u32,
    out_ld: u32,
) -> DgResult<()> {
    let o = Sm100Out::Bf16 {
        buf: out,
        ld: out_ld,
    };
    bf16_gemm_dev(
        ctx,
        a,
        a_ld,
        m,
        b,
        b_ld,
        n,
        k,
        &o,
        &Sm100Epilogue::default(),
    )
}

// ===========================================================================
// host-tensor sugar: end-to-end MXFP4
// ===========================================================================

/// End-to-end MXFP4 GEMM from BF16 host tensors: quantize both operands to
/// packed e2m1 + UE8M0 per-32 scale factors, run the Blackwell-native MXF4
/// tcgen05 kernel, and download the BF16 result.
pub fn fp4_gemm_nt_native(
    ctx: &DgContext,
    a_bf16: &[u16],
    b_bf16: &[u16],
    m: u32,
    n: u32,
    k: u32,
) -> DgResult<Vec<u16>> {
    let a_dev = crate::ops::upload(ctx, a_bf16)?;
    let b_dev = crate::ops::upload(ctx, b_bf16)?;
    let (a_data, a_sf) = cast_bf16_to_fp4_packed_sf_dev(ctx, &a_dev, k, m, k, 1)?;
    let (b_data, b_sf) = cast_bf16_to_fp4_packed_sf_dev(ctx, &b_dev, k, n, k, 1)?;
    let mut out = unsafe {
        ctx.stream
            .alloc::<u16>(m as usize * n as usize)
            .map_err(|e| DgError::Driver(format!("{e:?}")))?
    };
    let a_op = LpOperand {
        data: &a_data,
        sf: &a_sf,
        rows: m,
        k,
        ld: k,
        bits: 4,
        gran_k: 32,
        major_mn: false,
    };
    let b_op = LpOperand {
        data: &b_data,
        sf: &b_sf,
        rows: n,
        k,
        ld: k,
        bits: 4,
        gran_k: 32,
        major_mn: false,
    };
    let o = Sm100Out::Bf16 {
        buf: &mut out,
        ld: n,
    };
    fp8_fp4_gemm_dev(ctx, &a_op, &b_op, &o, &Sm100Epilogue::default())?;
    crate::ops::download(ctx, &out)
}

/// Device-resident MXFP4 GEMM over pre-quantized operands.
#[allow(clippy::too_many_arguments)]
pub fn fp4_gemm_packed_dev(
    ctx: &DgContext,
    a_data: &CudaSlice<u8>,
    a_sf: &CudaSlice<u32>,
    b_data: &CudaSlice<u8>,
    b_sf: &CudaSlice<u32>,
    out: &mut CudaSlice<u16>,
    m: u32,
    n: u32,
    k: u32,
) -> DgResult<()> {
    let a_op = LpOperand {
        data: a_data,
        sf: a_sf,
        rows: m,
        k,
        ld: k,
        bits: 4,
        gran_k: 32,
        major_mn: false,
    };
    let b_op = LpOperand {
        data: b_data,
        sf: b_sf,
        rows: n,
        k,
        ld: k,
        bits: 4,
        gran_k: 32,
        major_mn: false,
    };
    let o = Sm100Out::Bf16 { buf: out, ld: n };
    fp8_fp4_gemm_dev(ctx, &a_op, &b_op, &o, &Sm100Epilogue::default())
}
