//! Public operations: the DeepGEMM API surface in Rust.
//!
//! Device-resident core ops (suffixed `_dev`) plus host-convenience
//! wrappers. All heavy kernels are NVRTC-JITed SM90a CUDA; modules are
//! cached per (config, source) across calls and processes.

use cudarc::driver::safe::{CudaSlice, DevicePtr, DeviceRepr};

use crate::cuda;
use crate::device::DgContext;
use crate::heuristics::{best_bf16_config, best_fp8_config, GemmConfig};
use crate::launch::{ArgBuilder, DevPtr, LaunchGrid};
use crate::tma;
use crate::types::{DgError, DgResult, GemmType};

const GEMM_KERNEL_NAME: &str = "deepgemm_gemm_kernel";

// ===========================================================================
// upload / download helpers
// ===========================================================================

pub fn upload<T: DeviceRepr>(ctx: &DgContext, host: &[T]) -> DgResult<CudaSlice<T>> {
    ctx.stream
        .clone_htod(host)
        .map_err(|e| DgError::Driver(format!("{e:?}")))
}

pub fn download<T: DeviceRepr + Default + Clone>(
    ctx: &DgContext,
    dev: &CudaSlice<T>,
) -> DgResult<Vec<T>> {
    let mut out = vec![T::default(); dev.len()];
    ctx.stream
        .memcpy_dtoh(dev, &mut out)
        .map_err(|e| DgError::Driver(format!("{e:?}")))?;
    Ok(out)
}

pub(crate) fn dev_ptr<T: DeviceRepr>(ctx: &DgContext, s: &CudaSlice<T>) -> u64 {
    s.device_ptr(&ctx.stream).0
}

// ===========================================================================
// transform_sf
// ===========================================================================

/// Transform scale factors into the MN-major TMA-aligned layout required by
/// the SM90 kernels (upstream `transform_sf` / `get_mn_major_tma_aligned_tensor`).
///
/// Input: `(num_batches, mn, sf_k)` f32, `sf_k` contiguous.
/// Output: `(num_batches * sf_k, tma_aligned(mn))` f32, `mn` contiguous.
pub fn transform_sf_dev(
    ctx: &DgContext,
    sf: &CudaSlice<f32>,
    mn: u32,
    sf_k: u32,
    num_batches: u32,
) -> DgResult<CudaSlice<f32>> {
    let aligned_mn = crate::types::tma_aligned_size(mn, 4);
    let mut dst = unsafe {
        ctx.stream
            .alloc::<f32>(aligned_mn as usize * sf_k as usize * num_batches as usize)
            .map_err(|e| DgError::Driver(format!("{e:?}")))?
    };
    // Zero-fill (padding columns must be 0)
    ctx.stream
        .memset_zeros(&mut dst)
        .map_err(|e| DgError::Driver(format!("{e:?}")))?;

    let module = ctx
        .jit
        .module(&ctx.device, &cuda::layout::build_layout_kernel_source())?;
    let func = module
        .load_function("deepgemm_transpose_fp32")
        .map_err(|e| DgError::Driver(format!("{e:?}")))?;

    let num_mn_tiles = mn.div_ceil(64);
    let num_k_tiles = sf_k.div_ceil(128);
    let mut args = ArgBuilder::new();
    args.push(&DevPtr(dev_ptr(ctx, sf)));
    args.push(&DevPtr(dev_ptr(ctx, &dst)));
    args.push(&mn);
    args.push(&sf_k);
    args.push(&aligned_mn);
    args.push(&num_batches);
    unsafe {
        args.launch(
            &func,
            &ctx.stream,
            LaunchGrid {
                grid_x: num_mn_tiles * num_k_tiles,
                grid_y: num_batches,
                block_x: 512,
                smem: 64 * 129 * 4,
                ..Default::default()
            },
        )?;
    }
    Ok(dst)
}

/// Host convenience: upload + transform.
pub fn transform_sf(
    ctx: &DgContext,
    sf_host: &[f32],
    mn: u32,
    sf_k: u32,
    num_batches: u32,
) -> DgResult<CudaSlice<f32>> {
    let expected = mn as usize * sf_k as usize * num_batches as usize;
    if sf_host.len() < expected {
        return Err(DgError::Shape(format!(
            "sf buffer too small: {} < {expected}",
            sf_host.len()
        )));
    }
    let dev = upload(ctx, sf_host)?;
    transform_sf_dev(ctx, &dev, mn, sf_k, num_batches)
}

// ===========================================================================
// GEMM launcher (shared by fp8 / bf16 kernels)
// ===========================================================================

#[allow(clippy::too_many_arguments)]
fn launch_gemm(
    ctx: &DgContext,
    cfg: &GemmConfig,
    gemm_type: GemmType,
    is_bf16: bool,
    shape_m: u32,
    shape_n: u32,
    shape_k: u32,
    num_groups: u32,
    // fp8-only:
    sfb_ptr: u64,
    sfb_layout: Option<&SfbLayout>,
    grouped_layout_ptr: u64,
    // tensor maps
    tm_a: cudarc::driver::sys::CUtensorMap,
    tm_b: cudarc::driver::sys::CUtensorMap,
    tm_d: cudarc::driver::sys::CUtensorMap,
    tm_sfa: Option<cudarc::driver::sys::CUtensorMap>,
) -> DgResult<()> {
    let gemm_type_id = match gemm_type {
        GemmType::Normal => 0,
        GemmType::MGroupedContiguous => 1,
        GemmType::MGroupedMasked => 2,
        GemmType::Batched => 3,
    };

    let source = cuda::fp8_gemm_1d2d::build_gemm_kernel_source(
        cfg.block_m,
        cfg.block_n,
        cfg.block_k,
        cfg.num_stages,
        cfg.swizzle_a,
        cfg.swizzle_b,
        cfg.swizzle_cd,
        cfg.num_tma_threads,
        cfg.num_math_threads,
        cfg.cluster_size(),
        cfg.is_multicast_on_a(),
        cfg.num_sms,
        gemm_type_id,
        num_groups,
        is_bf16,
    );
    let module = ctx.jit.module(&ctx.device, &source)?;
    let func = module
        .load_function(GEMM_KERNEL_NAME)
        .map_err(|e| DgError::Driver(format!("{e:?}")))?;

    // Exact dynamic smem (must match the kernel's internal layout math).
    let smem = exact_smem_size(cfg, shape_k, is_bf16);
    let max_smem = 232448 - 1024;
    if smem > max_smem {
        return Err(DgError::Shape(format!(
            "required smem {smem} exceeds {max_smem}"
        )));
    }

    let mut args = ArgBuilder::new();
    if !is_bf16 {
        let layout = sfb_layout.ok_or_else(|| DgError::Shape("missing sfb layout".into()))?;
        // K-contiguous sfb is what the kernel expects (stride_k = 1).
        if !layout.k_contiguous {
            return Err(DgError::Shape(
                "sfb must be k-contiguous: (num_groups*n_tiles, k_blocks) f32".into(),
            ));
        }
        args.push(&DevPtr(sfb_ptr));
    }
    args.push(&DevPtr(grouped_layout_ptr));
    args.push(&shape_m);
    args.push(&shape_n);
    args.push(&shape_k);
    args.push_aligned(&tm_a, 64);
    args.push_aligned(&tm_b, 64);
    args.push_aligned(&tm_d, 64);
    if let Some(tm) = tm_sfa {
        args.push_aligned(&tm, 64);
    }

    let cluster = if cfg.cluster_size() > 1 {
        Some((cfg.cluster_size(), 1, 1))
    } else {
        None
    };
    unsafe {
        args.launch(
            &func,
            &ctx.stream,
            LaunchGrid {
                grid_x: cfg.num_sms,
                block_x: cfg.num_threads(),
                smem,
                cluster,
                ..Default::default()
            },
        )
    }
}

/// Exact dynamic shared-memory bytes for one config (mirrors the kernel layout).
pub fn exact_smem_size(cfg: &GemmConfig, shape_k: u32, is_bf16: bool) -> u32 {
    let elem = if is_bf16 { 2 } else { 1 };
    let smem_d = (cfg.block_m * cfg.block_n * 2).div_ceil(1024) * 1024;
    let smem_a = cfg.block_m * cfg.block_k * elem;
    let smem_b = cfg.block_n * cfg.block_k * elem;
    let smem_sfa = if is_bf16 {
        0
    } else {
        (cfg.block_m * 4).div_ceil(128) * 128
    };
    let smem_sfb = if is_bf16 {
        0
    } else {
        (shape_k.div_ceil(cfg.block_k) * 4).div_ceil(8) * 8
    };
    let barriers = cfg.num_stages * 8 * 2;
    smem_d + cfg.num_stages * (smem_a + smem_b + smem_sfa) + smem_sfb + barriers
}

/// 2-D (128 x 128) scale factor layout of B (weights).
#[derive(Debug, Clone, Copy)]
pub struct SfbLayout {
    pub n_tiles: u32,
    pub k_blocks: u32,
    pub num_groups: u32,
    /// true: `(groups*n_tiles, k_blocks)` with k contiguous (kernel-native).
    /// false: `(groups*n_tiles, k_blocks)` with n-tiles contiguous — needs a
    /// transpose first (callers should prefer k-contiguous).
    pub k_contiguous: bool,
}

/// A device-resident FP8 operand pair (data + scales).
pub struct Fp8Tensor<'a> {
    pub data: &'a CudaSlice<u8>,
    pub sf: &'a CudaSlice<f32>,
    pub rows: u32,
    pub k: u32,
    pub ld: u32,
    /// sf rows (== rows for 1d scaling)
    pub sf_rows: u32,
}

impl<'a> Fp8Tensor<'a> {
    pub fn new(data: &'a CudaSlice<u8>, sf: &'a CudaSlice<f32>, rows: u32, k: u32) -> Self {
        Self {
            data,
            sf,
            rows,
            k,
            ld: k,
            sf_rows: rows,
        }
    }
    pub fn with_ld(mut self, ld: u32) -> Self {
        self.ld = ld;
        self
    }
}

/// Device-resident BF16 operand.
pub struct Bf16Tensor<'a> {
    pub data: &'a CudaSlice<u16>,
    pub rows: u32,
    pub k: u32,
    pub ld: u32,
}

impl<'a> Bf16Tensor<'a> {
    pub fn new(data: &'a CudaSlice<u16>, rows: u32, k: u32) -> Self {
        Self {
            data,
            rows,
            k,
            ld: k,
        }
    }
}

// ===========================================================================
// fp8_gemm_nt (+ grouped variants, device tensors)
// ===========================================================================

fn check_fp8_shapes(_m: u32, n: u32, k: u32) -> DgResult<()> {
    if k % 128 != 0 {
        return Err(DgError::Shape(format!(
            "k ({k}) must be a multiple of 128 (per-128-channel scaling)"
        )));
    }
    if n % 8 != 0 {
        return Err(DgError::Shape(format!("n ({n}) must be a multiple of 8")));
    }
    Ok(())
}

/// The core FP8 GEMM: `out[m, n] = (a[m, k] * sfa) @ (b[n, k] * sfb)^T` in bf16.
///
/// * `a`: `(m, k)` e4m3, K-major, row stride `a.ld`.
/// * `a_sf_t`: **transformed** SFA from [`transform_sf_dev`] — `(k_blocks, tma_aligned(m))`.
/// * `b`: `(n, k)` e4m3, K-major.
/// * `sfb`: `(ceil(n / 128), k_blocks)` f32, k-contiguous (128x128 tile scales).
/// * `out`: `(m, n)` bf16, row stride `out_ld`.
#[allow(clippy::too_many_arguments)]
pub fn fp8_gemm_nt_dev(
    ctx: &DgContext,
    a: &Fp8Tensor,
    a_sf_t: &CudaSlice<f32>,
    b: &Fp8Tensor,
    sfb: &CudaSlice<f32>,
    out: &mut CudaSlice<u16>,
    out_ld: u32,
) -> DgResult<()> {
    if ctx.arch.is_blackwell() {
        return crate::ops_sm100::bridge_fp8_gemm_nt_dev(ctx, a, a_sf_t, b, sfb, out, out_ld);
    }
    let (m, n, k) = (a.rows, b.rows, a.k);
    check_fp8_shapes(m, n, k)?;

    let cfg = best_fp8_config(&ctx.arch, GemmType::Normal, m, n, k, 1, m);
    let k_blocks = k.div_ceil(128);

    let tm_a = tma::make_tma_2d(
        dev_ptr(ctx, a.data) as *const _,
        tma::TMA_ELEM_FP8,
        1,
        k,
        m,
        a.ld as u64,
        cfg.block_k,
        cfg.block_m,
        cfg.swizzle_a,
    )?;
    let tm_b = tma::make_tma_2d(
        dev_ptr(ctx, b.data) as *const _,
        tma::TMA_ELEM_FP8,
        1,
        k,
        n,
        b.ld as u64,
        cfg.block_k,
        cfg.block_n,
        cfg.swizzle_b,
    )?;
    let tm_d = tma::make_tma_2d(
        dev_ptr(ctx, out) as *const _,
        tma::TMA_ELEM_BF16,
        2,
        n,
        m,
        out_ld as u64,
        tma_d_inner(&cfg),
        cfg.block_m,
        cfg.swizzle_cd,
    )?;
    let tm_sfa = tma::make_tma_sf(
        dev_ptr(ctx, a_sf_t) as *const _,
        m,
        k_blocks,
        1,
        cfg.block_m,
    )?;

    let sfb_layout = SfbLayout {
        n_tiles: n.div_ceil(128),
        k_blocks,
        num_groups: 1,
        k_contiguous: true,
    };
    launch_gemm(
        ctx,
        &cfg,
        GemmType::Normal,
        false,
        m,
        n,
        k,
        1,
        dev_ptr(ctx, sfb),
        Some(&sfb_layout),
        0,
        tm_a,
        tm_b,
        tm_d,
        Some(tm_sfa),
    )
}

/// MoE grouped GEMM (contiguous layout).
///
/// * `a`: `(m_total, k)` — tokens of all experts concatenated, each group's
///   start aligned to 128 rows; padding rows may contain garbage.
/// * `m_indices`: `(m_total)` i32 — expert id per row, `-1` for padding.
/// * `b`: `(num_groups * n, k)` — expert `g`'s weights at rows `[g*n, (g+1)*n)`.
/// * `sfb`: `(num_groups * n_tiles, k_blocks)` f32 k-contiguous.
/// * `a_sf_t`: transformed SFA over all `m_total` rows.
/// * `out`: `(m_total, n)` bf16.
#[allow(clippy::too_many_arguments)]
pub fn m_grouped_fp8_gemm_nt_contiguous_dev(
    ctx: &DgContext,
    a: &Fp8Tensor,
    a_sf_t: &CudaSlice<f32>,
    m_indices: &CudaSlice<i32>,
    b: &Fp8Tensor,
    sfb: &CudaSlice<f32>,
    num_groups: u32,
    out: &mut CudaSlice<u16>,
    out_ld: u32,
) -> DgResult<()> {
    if ctx.arch.is_blackwell() {
        return crate::ops_sm100::bridge_m_grouped_fp8_contiguous_dev(
            ctx, a, a_sf_t, b, sfb, out, out_ld, m_indices, num_groups,
        );
    }
    let (m, n, k) = (a.rows, b.rows / num_groups, a.k);
    check_fp8_shapes(m, n, k)?;
    if b.rows % num_groups != 0 {
        return Err(DgError::Shape("b.rows must be num_groups * n".into()));
    }
    if m_indices.len() < m as usize {
        return Err(DgError::Shape("m_indices too small".into()));
    }

    let cfg = best_fp8_config(
        &ctx.arch,
        GemmType::MGroupedContiguous,
        m,
        n,
        k,
        num_groups,
        m,
    );
    let k_blocks = k.div_ceil(128);

    let tm_a = tma::make_tma_2d(
        dev_ptr(ctx, a.data) as *const _,
        tma::TMA_ELEM_FP8,
        1,
        k,
        m,
        a.ld as u64,
        cfg.block_k,
        cfg.block_m,
        cfg.swizzle_a,
    )?;
    let tm_b = tma::make_tma_2d(
        dev_ptr(ctx, b.data) as *const _,
        tma::TMA_ELEM_FP8,
        1,
        k,
        b.rows,
        b.ld as u64,
        cfg.block_k,
        cfg.block_n,
        cfg.swizzle_b,
    )?;
    let tm_d = tma::make_tma_2d(
        dev_ptr(ctx, out) as *const _,
        tma::TMA_ELEM_BF16,
        2,
        n,
        m,
        out_ld as u64,
        tma_d_inner(&cfg),
        cfg.block_m,
        cfg.swizzle_cd,
    )?;
    let tm_sfa = tma::make_tma_sf(
        dev_ptr(ctx, a_sf_t) as *const _,
        m,
        k_blocks,
        1,
        cfg.block_m,
    )?;

    let sfb_layout = SfbLayout {
        n_tiles: n.div_ceil(128),
        k_blocks,
        num_groups,
        k_contiguous: true,
    };
    launch_gemm(
        ctx,
        &cfg,
        GemmType::MGroupedContiguous,
        false,
        m,
        n,
        k,
        num_groups,
        dev_ptr(ctx, sfb),
        Some(&sfb_layout),
        dev_ptr(ctx, m_indices),
        tm_a,
        tm_b,
        tm_d,
        Some(tm_sfa),
    )
}

/// MoE grouped GEMM (masked layout, DeepEP dispatch output).
///
/// * `a`: `(num_groups * m, k)` — group `g` occupies rows `[g*m, (g+1)*m)`;
///   only the first `masked_m[g]` rows of each group are valid.
/// * `a_sf_t`: transformed SFA with per-group stacking:
///   `transform_sf_dev(..., m, k_blocks, num_groups)`.
/// * `b`, `sfb`: like the contiguous variant.
/// * `out`: `(num_groups * m, n)` bf16.
/// * `expected_m`: upper bound on `max(masked_m)` used for scheduling (the
///   actual values are read on device — zero CPU synchronization).
#[allow(clippy::too_many_arguments)]
pub fn m_grouped_fp8_gemm_nt_masked_dev(
    ctx: &DgContext,
    a: &Fp8Tensor,
    a_sf_t: &CudaSlice<f32>,
    masked_m: &CudaSlice<i32>,
    b: &Fp8Tensor,
    sfb: &CudaSlice<f32>,
    num_groups: u32,
    out: &mut CudaSlice<u16>,
    out_ld: u32,
    expected_m: u32,
) -> DgResult<()> {
    if ctx.arch.is_blackwell() {
        return crate::ops_sm100::bridge_m_grouped_fp8_masked_dev(
            ctx, a, a_sf_t, b, sfb, out, out_ld, masked_m, num_groups, expected_m,
        );
    }
    let m = a.rows / num_groups;
    let (n, k) = (b.rows / num_groups, a.k);
    check_fp8_shapes(m, n, k)?;

    let cfg = best_fp8_config(
        &ctx.arch,
        GemmType::MGroupedMasked,
        expected_m,
        n,
        k,
        num_groups,
        expected_m,
    );
    let k_blocks = k.div_ceil(128);

    let tm_a = tma::make_tma_2d(
        dev_ptr(ctx, a.data) as *const _,
        tma::TMA_ELEM_FP8,
        1,
        k,
        a.rows,
        a.ld as u64,
        cfg.block_k,
        cfg.block_m,
        cfg.swizzle_a,
    )?;
    let tm_b = tma::make_tma_2d(
        dev_ptr(ctx, b.data) as *const _,
        tma::TMA_ELEM_FP8,
        1,
        k,
        b.rows,
        b.ld as u64,
        cfg.block_k,
        cfg.block_n,
        cfg.swizzle_b,
    )?;
    let tm_d = tma::make_tma_2d(
        dev_ptr(ctx, out) as *const _,
        tma::TMA_ELEM_BF16,
        2,
        n,
        out.len() as u32 / n.max(1),
        out_ld as u64,
        tma_d_inner(&cfg),
        cfg.block_m,
        cfg.swizzle_cd,
    )?;
    let tm_sfa = tma::make_tma_sf(
        dev_ptr(ctx, a_sf_t) as *const _,
        m,
        k_blocks,
        num_groups,
        cfg.block_m,
    )?;

    let sfb_layout = SfbLayout {
        n_tiles: n.div_ceil(128),
        k_blocks,
        num_groups,
        k_contiguous: true,
    };
    launch_gemm(
        ctx,
        &cfg,
        GemmType::MGroupedMasked,
        false,
        m,
        n,
        k,
        num_groups,
        dev_ptr(ctx, sfb),
        Some(&sfb_layout),
        dev_ptr(ctx, masked_m),
        tm_a,
        tm_b,
        tm_d,
        Some(tm_sfa),
    )
}

/// Batched FP8 GEMM (`bmm`): `out[g, m, n] = a[g, m, k] @ b[g, n, k]^T`.
#[allow(clippy::too_many_arguments)]
pub fn fp8_bmm_nt_dev(
    ctx: &DgContext,
    a: &Fp8Tensor,
    a_sf_t: &CudaSlice<f32>,
    b: &Fp8Tensor,
    sfb: &CudaSlice<f32>,
    batch: u32,
    out: &mut CudaSlice<u16>,
    out_ld: u32,
) -> DgResult<()> {
    if ctx.arch.is_blackwell() {
        return crate::ops_sm100::bridge_fp8_bmm_dev(ctx, a, a_sf_t, b, sfb, out, out_ld, batch);
    }
    let m = a.rows / batch;
    let n = b.rows / batch;
    let k = a.k;
    check_fp8_shapes(m, n, k)?;

    let cfg = best_fp8_config(&ctx.arch, GemmType::Batched, m, n, k, batch, m);
    let k_blocks = k.div_ceil(128);

    let tm_a = tma::make_tma_3d(
        dev_ptr(ctx, a.data) as *const _,
        tma::TMA_ELEM_FP8,
        1,
        [k, m, batch],
        [a.ld as u64, a.ld as u64 * m as u64],
        [cfg.block_k, cfg.block_m, 1],
        cfg.swizzle_a,
    )?;
    let tm_b = tma::make_tma_3d(
        dev_ptr(ctx, b.data) as *const _,
        tma::TMA_ELEM_FP8,
        1,
        [k, n, batch],
        [b.ld as u64, b.ld as u64 * n as u64],
        [cfg.block_k, cfg.block_n, 1],
        cfg.swizzle_b,
    )?;
    let tm_d = tma::make_tma_3d(
        dev_ptr(ctx, out) as *const _,
        tma::TMA_ELEM_BF16,
        2,
        [n, m, batch],
        [out_ld as u64, out_ld as u64 * m as u64],
        [tma_d_inner(&cfg), cfg.block_m, 1],
        cfg.swizzle_cd,
    )?;
    let tm_sfa = tma::make_tma_sf(
        dev_ptr(ctx, a_sf_t) as *const _,
        m,
        k_blocks,
        batch,
        cfg.block_m,
    )?;

    let sfb_layout = SfbLayout {
        n_tiles: n.div_ceil(128),
        k_blocks,
        num_groups: batch,
        k_contiguous: true,
    };
    launch_gemm(
        ctx,
        &cfg,
        GemmType::Batched,
        false,
        m,
        n,
        k,
        batch,
        dev_ptr(ctx, sfb),
        Some(&sfb_layout),
        0,
        tm_a,
        tm_b,
        tm_d,
        Some(tm_sfa),
    )
}

fn tma_d_inner(cfg: &GemmConfig) -> u32 {
    if cfg.swizzle_cd == 0 {
        cfg.block_n
    } else {
        cfg.swizzle_cd / 2
    }
}

// ===========================================================================
// Host convenience wrappers
// ===========================================================================

/// Host-input FP8 NT GEMM. Returns the bf16 output on device.
#[allow(clippy::too_many_arguments)]
pub fn fp8_gemm_nt(
    ctx: &DgContext,
    a: &[u8],
    sfa: &[f32],
    m: u32,
    b: &[u8],
    sfb: &[f32],
    n: u32,
    k: u32,
) -> DgResult<CudaSlice<u16>> {
    let k_blocks = k.div_ceil(128);
    let a_dev = upload(ctx, a)?;
    let b_dev = upload(ctx, b)?;
    let sfb_dev = upload(ctx, sfb)?;
    let sfa_t = transform_sf(ctx, sfa, m, k_blocks, 1)?;

    let mut out = unsafe {
        ctx.stream
            .alloc::<u16>(m as usize * n as usize)
            .map_err(|e| DgError::Driver(format!("{e:?}")))?
    };
    fp8_gemm_nt_dev(
        ctx,
        &Fp8Tensor::new(&a_dev, &sfa_t, m, k),
        &sfa_t,
        &Fp8Tensor::new(&b_dev, &sfb_dev, n, k),
        &sfb_dev,
        &mut out,
        n,
    )?;
    Ok(out)
}

// ===========================================================================
// BF16 GEMM (same warp-specialized kernel, no scale factors)
// ===========================================================================

/// BF16 NT GEMM: `out[m, n] = a[m, k] @ b[n, k]^T`, bf16 in/out.
#[allow(clippy::too_many_arguments)]
pub fn bf16_gemm_nt_dev(
    ctx: &DgContext,
    a: &Bf16Tensor,
    b: &Bf16Tensor,
    out: &mut CudaSlice<u16>,
    out_ld: u32,
) -> DgResult<()> {
    let (m, n, k) = (a.rows, b.rows, a.k);
    if ctx.arch.is_blackwell() {
        return crate::ops_sm100::bridge_bf16_gemm_nt_dev(
            ctx, a.data, b.data, out, m, n, k, a.ld, b.ld, out_ld,
        );
    }
    if k % 64 != 0 {
        return Err(DgError::Shape(format!(
            "bf16 k ({k}) must be a multiple of 64"
        )));
    }
    let cfg = best_bf16_config(&ctx.arch, GemmType::Normal, m, n, k, 1, m);

    let tm_a = tma::make_tma_2d(
        dev_ptr(ctx, a.data) as *const _,
        tma::TMA_ELEM_BF16,
        2,
        k,
        m,
        a.ld as u64,
        cfg.block_k,
        cfg.block_m,
        cfg.swizzle_a,
    )?;
    let tm_b = tma::make_tma_2d(
        dev_ptr(ctx, b.data) as *const _,
        tma::TMA_ELEM_BF16,
        2,
        k,
        n,
        b.ld as u64,
        cfg.block_k,
        cfg.block_n,
        cfg.swizzle_b,
    )?;
    let tm_d = tma::make_tma_2d(
        dev_ptr(ctx, out) as *const _,
        tma::TMA_ELEM_BF16,
        2,
        n,
        m,
        out_ld as u64,
        tma_d_inner(&cfg),
        cfg.block_m,
        cfg.swizzle_cd,
    )?;

    // BF16 via cuBLASLt when available (Normal shapes); the custom kernel is
    // the fallback and the grouped-GEMM engine.
    if cfg.cluster_size() == 1 && m >= 256 && n >= 256 {
        if let Ok(()) = crate::cublaslt::bf16_gemm_nt(
            ctx,
            dev_ptr(ctx, a.data),
            dev_ptr(ctx, b.data),
            dev_ptr(ctx, out),
            m,
            n,
            k,
            a.ld as i64,
            b.ld as i64,
            out_ld as i64,
        ) {
            return Ok(());
        }
    }

    launch_gemm(
        ctx,
        &cfg,
        GemmType::Normal,
        true,
        m,
        n,
        k,
        1,
        0,
        None,
        0,
        tm_a,
        tm_b,
        tm_d,
        None,
    )
}

/// Grouped BF16 GEMM, masked layout (per-expert m read on device).
#[allow(clippy::too_many_arguments)]
pub fn m_grouped_bf16_gemm_nt_masked_dev(
    ctx: &DgContext,
    a: &Bf16Tensor,
    masked_m: &CudaSlice<i32>,
    b: &Bf16Tensor,
    num_groups: u32,
    out: &mut CudaSlice<u16>,
    out_ld: u32,
    expected_m: u32,
) -> DgResult<()> {
    let m = a.rows / num_groups;
    let (n, k) = (b.rows / num_groups, a.k);
    let cfg = best_bf16_config(
        &ctx.arch,
        GemmType::MGroupedMasked,
        expected_m,
        n,
        k,
        num_groups,
        expected_m,
    );

    let tm_a = tma::make_tma_2d(
        dev_ptr(ctx, a.data) as *const _,
        tma::TMA_ELEM_BF16,
        2,
        k,
        a.rows,
        a.ld as u64,
        cfg.block_k,
        cfg.block_m,
        cfg.swizzle_a,
    )?;
    let tm_b = tma::make_tma_2d(
        dev_ptr(ctx, b.data) as *const _,
        tma::TMA_ELEM_BF16,
        2,
        k,
        b.rows,
        b.ld as u64,
        cfg.block_k,
        cfg.block_n,
        cfg.swizzle_b,
    )?;
    let tm_d = tma::make_tma_2d(
        dev_ptr(ctx, out) as *const _,
        tma::TMA_ELEM_BF16,
        2,
        n,
        a.rows,
        out_ld as u64,
        tma_d_inner(&cfg),
        cfg.block_m,
        cfg.swizzle_cd,
    )?;

    launch_gemm(
        ctx,
        &cfg,
        GemmType::MGroupedMasked,
        true,
        m,
        n,
        k,
        num_groups,
        0,
        None,
        dev_ptr(ctx, masked_m),
        tm_a,
        tm_b,
        tm_d,
        None,
    )
}

// ===========================================================================
// Layout adapters: nn / tt / tn
// ===========================================================================

/// Transpose an fp8 device matrix (rows x cols) into (cols x rows).
pub fn transpose_fp8_dev(
    ctx: &DgContext,
    src: &CudaSlice<u8>,
    rows: u32,
    cols: u32,
    ld_src: u32,
) -> DgResult<(CudaSlice<u8>, u32)> {
    let dst = unsafe {
        ctx.stream
            .alloc::<u8>(rows as usize * cols as usize)
            .map_err(|e| DgError::Driver(format!("{e:?}")))?
    };
    let module = ctx
        .jit
        .module(&ctx.device, &cuda::layout::build_layout_kernel_source())?;
    let func = module
        .load_function("deepgemm_transpose_fp8")
        .map_err(|e| DgError::Driver(format!("{e:?}")))?;
    let ld_dst = rows;
    let mut args = ArgBuilder::new();
    args.push(&DevPtr(dev_ptr(ctx, src)));
    args.push(&DevPtr(dev_ptr(ctx, &dst)));
    args.push(&rows);
    args.push(&cols);
    args.push(&ld_src);
    args.push(&ld_dst);
    unsafe {
        args.launch(
            &func,
            &ctx.stream,
            LaunchGrid {
                grid_x: cols.div_ceil(32),
                grid_y: rows.div_ceil(32),
                block_x: 256,
                smem: 32 * 33,
                ..Default::default()
            },
        )?;
    }
    Ok((dst, ld_dst))
}

/// `nn`: B given as `(k, n)` k-major-rows → transpose to `(n, k)` then NT.
#[allow(clippy::too_many_arguments)]
pub fn fp8_gemm_nn(
    ctx: &DgContext,
    a: &[u8],
    sfa: &[f32],
    m: u32,
    b_kn: &[u8],
    sfb: &[f32],
    n: u32,
    k: u32,
) -> DgResult<CudaSlice<u16>> {
    let a_dev = upload(ctx, a)?;
    let b_dev = upload(ctx, b_kn)?;
    let (b_t, _) = transpose_fp8_dev(ctx, &b_dev, k, n, n)?;
    let sfb_dev = upload(ctx, sfb)?;
    let sfa_t = transform_sf(ctx, sfa, m, k.div_ceil(128), 1)?;
    let mut out = unsafe {
        ctx.stream
            .alloc::<u16>(m as usize * n as usize)
            .map_err(|e| DgError::Driver(format!("{e:?}")))?
    };
    fp8_gemm_nt_dev(
        ctx,
        &Fp8Tensor::new(&a_dev, &sfa_t, m, k),
        &sfa_t,
        &Fp8Tensor::new(&b_t, &sfb_dev, n, k),
        &sfb_dev,
        &mut out,
        n,
    )?;
    Ok(out)
}

/// `tn`: A given as `(k, m)` → transpose to `(m, k)` then NT.
#[allow(clippy::too_many_arguments)]
pub fn fp8_gemm_tn(
    ctx: &DgContext,
    a_km: &[u8],
    sfa: &[f32],
    m: u32,
    b: &[u8],
    sfb: &[f32],
    n: u32,
    k: u32,
) -> DgResult<CudaSlice<u16>> {
    let a_dev = upload(ctx, a_km)?;
    let b_dev = upload(ctx, b)?;
    let (a_t, _) = transpose_fp8_dev(ctx, &a_dev, k, m, m)?;
    let sfb_dev = upload(ctx, sfb)?;
    let sfa_t = transform_sf(ctx, sfa, m, k.div_ceil(128), 1)?;
    let mut out = unsafe {
        ctx.stream
            .alloc::<u16>(m as usize * n as usize)
            .map_err(|e| DgError::Driver(format!("{e:?}")))?
    };
    fp8_gemm_nt_dev(
        ctx,
        &Fp8Tensor::new(&a_t, &sfa_t, m, k),
        &sfa_t,
        &Fp8Tensor::new(&b_dev, &sfb_dev, n, k),
        &sfb_dev,
        &mut out,
        n,
    )?;
    Ok(out)
}

/// `tt`: both operands transposed.
#[allow(clippy::too_many_arguments)]
pub fn fp8_gemm_tt(
    ctx: &DgContext,
    a_km: &[u8],
    sfa: &[f32],
    m: u32,
    b_kn: &[u8],
    sfb: &[f32],
    n: u32,
    k: u32,
) -> DgResult<CudaSlice<u16>> {
    let a_dev = upload(ctx, a_km)?;
    let b_dev = upload(ctx, b_kn)?;
    let (a_t, _) = transpose_fp8_dev(ctx, &a_dev, k, m, m)?;
    let (b_t, _) = transpose_fp8_dev(ctx, &b_dev, k, n, n)?;
    let sfb_dev = upload(ctx, sfb)?;
    let sfa_t = transform_sf(ctx, sfa, m, k.div_ceil(128), 1)?;
    let mut out = unsafe {
        ctx.stream
            .alloc::<u16>(m as usize * n as usize)
            .map_err(|e| DgError::Driver(format!("{e:?}")))?
    };
    fp8_gemm_nt_dev(
        ctx,
        &Fp8Tensor::new(&a_t, &sfa_t, m, k),
        &sfa_t,
        &Fp8Tensor::new(&b_t, &sfb_dev, n, k),
        &sfb_dev,
        &mut out,
        n,
    )?;
    Ok(out)
}

// ===========================================================================
// FP4 (e2m1) GEMM: unpack to FP8, reuse the FP8 engine (Hopper path)
// ===========================================================================

/// Unpack packed-FP4 (2 values/byte, low nibble first) into FP8 e4m3.
pub fn unpack_fp4_dev(
    ctx: &DgContext,
    src: &CudaSlice<u8>,
    packed_len: usize,
) -> DgResult<CudaSlice<u8>> {
    let dst = unsafe {
        ctx.stream
            .alloc::<u8>(packed_len * 2)
            .map_err(|e| DgError::Driver(format!("{e:?}")))?
    };
    let module = ctx
        .jit
        .module(&ctx.device, &cuda::layout::build_layout_kernel_source())?;
    let func = module
        .load_function("deepgemm_unpack_fp4")
        .map_err(|e| DgError::Driver(format!("{e:?}")))?;
    let packed_len_u32 = packed_len as u32;
    let mut args = ArgBuilder::new();
    args.push(&DevPtr(dev_ptr(ctx, src)));
    args.push(&DevPtr(dev_ptr(ctx, &dst)));
    args.push(&packed_len_u32);
    unsafe {
        args.launch(
            &func,
            &ctx.stream,
            LaunchGrid::new(packed_len.div_ceil(256) as u32, 256, 0),
        )?;
    }
    Ok(dst)
}

/// FP4 NT GEMM (host tensors). `a_packed`/`b_packed` hold two e2m1 values per
/// byte along K; scales follow the same recipes as the FP8 path. On Hopper,
/// FP4 has no native tensor-core path, so operands are unpacked to FP8 first.
#[allow(clippy::too_many_arguments)]
pub fn fp4_gemm_nt(
    ctx: &DgContext,
    a_packed: &[u8],
    sfa: &[f32],
    m: u32,
    b_packed: &[u8],
    sfb: &[f32],
    n: u32,
    k: u32,
) -> DgResult<CudaSlice<u16>> {
    if k % 2 != 0 {
        return Err(DgError::Shape("fp4 k must be even".into()));
    }
    let a_packed_dev = upload(ctx, a_packed)?;
    let b_packed_dev = upload(ctx, b_packed)?;
    let a_dev = unpack_fp4_dev(ctx, &a_packed_dev, a_packed.len())?;
    let b_dev = unpack_fp4_dev(ctx, &b_packed_dev, b_packed.len())?;
    let sfb_dev = upload(ctx, sfb)?;
    let sfa_t = transform_sf(ctx, sfa, m, k.div_ceil(128), 1)?;
    let mut out = unsafe {
        ctx.stream
            .alloc::<u16>(m as usize * n as usize)
            .map_err(|e| DgError::Driver(format!("{e:?}")))?
    };
    fp8_gemm_nt_dev(
        ctx,
        &Fp8Tensor::new(&a_dev, &sfa_t, m, k),
        &sfa_t,
        &Fp8Tensor::new(&b_dev, &sfb_dev, n, k),
        &sfb_dev,
        &mut out,
        n,
    )?;
    Ok(out)
}

// ===========================================================================
// MQA logits (DeepEP prefill attention)
// ===========================================================================

/// MQA logits, bf16: `out[d, t] = dot(q[d, :], k[d, t, :])`.
///
/// * `q`: `(num_dpus, H)` bf16
/// * `k`: `(num_dpus, max_tokens, H)` bf16
/// * `num_valid`: optional `(num_dpus)` i32 — tokens beyond it get -inf.
/// * `out`: `(num_dpus, max_tokens)` bf16.
#[allow(clippy::too_many_arguments)]
pub fn mqa_logits_bf16_dev(
    ctx: &DgContext,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    num_valid: Option<&CudaSlice<i32>>,
    out: &mut CudaSlice<u16>,
    num_dpus: u32,
    max_tokens: u32,
    h: u32,
) -> DgResult<()> {
    let module = ctx
        .jit
        .module(&ctx.device, &cuda::mqa_logits::build_mqa_logits_source())?;
    let func = module
        .load_function("deepgemm_mqa_logits_bf16")
        .map_err(|e| DgError::Driver(format!("{e:?}")))?;
    let valid_ptr = num_valid.map(|v| dev_ptr(ctx, v)).unwrap_or(0);
    let mut args = ArgBuilder::new();
    args.push(&DevPtr(dev_ptr(ctx, q)));
    args.push(&DevPtr(dev_ptr(ctx, k)));
    args.push(&DevPtr(valid_ptr));
    args.push(&DevPtr(dev_ptr(ctx, out)));
    args.push(&num_dpus);
    args.push(&max_tokens);
    args.push(&h);
    unsafe {
        args.launch(
            &func,
            &ctx.stream,
            LaunchGrid {
                grid_x: max_tokens.div_ceil(4),
                grid_y: num_dpus,
                block_x: 256,
                ..Default::default()
            },
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn mqa_logits_bf16(
    ctx: &DgContext,
    q: &[u16],
    k: &[u16],
    num_valid: Option<&[i32]>,
    num_dpus: u32,
    max_tokens: u32,
    h: u32,
) -> DgResult<CudaSlice<u16>> {
    let q_dev = upload(ctx, q)?;
    let k_dev = upload(ctx, k)?;
    let valid_dev = match num_valid {
        Some(v) => Some(upload(ctx, v)?),
        None => None,
    };
    let mut out = unsafe {
        ctx.stream
            .alloc::<u16>(num_dpus as usize * max_tokens as usize)
            .map_err(|e| DgError::Driver(format!("{e:?}")))?
    };
    mqa_logits_bf16_dev(
        ctx,
        &q_dev,
        &k_dev,
        valid_dev.as_ref(),
        &mut out,
        num_dpus,
        max_tokens,
        h,
    )?;
    Ok(out)
}
