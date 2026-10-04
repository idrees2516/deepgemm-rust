//! SM100 (Blackwell) kernel-configuration heuristics — a faithful port of
//! upstream DeepGEMM's `SM100ArchSpec` (layout candidate enumeration,
//! storage/pipeline/launch configs, TMEM capacity checks).

use crate::cuda::bf16_gemm_sm100::Bf16Config;
use crate::cuda::fp8_fp4_gemm_1d1d::Fp8Fp4Config;
use crate::types::GemmType;

/// SM100a per-SM shared-memory capacity (227 KB, minus 1 KB reserved).
pub const SM100_SMEM_CAPACITY: u32 = 232_448;

/// Upstream `get_mk_alignment_for_contiguous_layout`.
pub const MK_ALIGNMENT_FOR_CONTIGUOUS: u32 = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sm100Layout {
    pub swap_ab: bool,
    pub block_m: u32,
    pub block_n: u32,
    pub block_k: u32,
    pub cluster_m: u32,
    pub cluster_n: u32,
}

impl Sm100Layout {
    pub fn cluster_size(&self) -> u32 {
        self.cluster_m * self.cluster_n
    }
    pub fn is_multicast_on_a(&self) -> bool {
        // cluster_n > 1 => the pair shares B tiles and splits A along M
        self.cluster_n > 1
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Sm100Storage {
    pub load_block_m: u32,
    pub load_block_n: u32,
    pub store_block_m: u32,
    pub store_block_n: u32,
    pub swizzle_a: u32,
    pub swizzle_b: u32,
    pub swizzle_cd: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct Sm100Config {
    pub layout: Sm100Layout,
    pub storage: Sm100Storage,
    pub num_stages: u32,
    pub num_tma_store_stages: u32,
    pub smem_size: u32,
    pub num_non_epilogue_threads: u32,
    pub num_epilogue_threads: u32,
}

/// The problem description consumed by the heuristics.
#[derive(Debug, Clone, Copy)]
pub struct Sm100Desc {
    pub gemm_type: GemmType,
    /// Upstream `GemmType::MGroupedContiguousWithPsumLayout` is used for the
    /// contiguous layout in the SM100 kernel.
    pub use_psum_layout: bool,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub num_groups: u32,
    pub expected_m: u32,
    pub expected_num_groups: u32,
    /// Operand element widths in bits: 8 (e4m3) or 4 (e2m1).
    pub a_bits: u32,
    pub b_bits: u32,
    pub major_a_mn: bool,
    pub major_b_mn: bool,
    /// 0 = bf16, 1 = fp32, 2 = e4m3 (dynamic-SF output).
    pub cd_dtype: u32,
    pub with_accumulation: bool,
    pub num_sms: u32,
    /// Tensor-core utilization percentage (100 = off).
    pub tc_util: u32,
}

fn is_m_grouped(t: GemmType) -> bool {
    matches!(t, GemmType::MGroupedContiguous | GemmType::MGroupedMasked)
}

fn num_element_bits(a_bits: u32, b_bits: u32) -> u32 {
    if a_bits == 4 && b_bits == 4 {
        4
    } else if a_bits == 16 || b_bits == 16 {
        16
    } else {
        8
    }
}

fn smem_pack_factor(bits: u32) -> u32 {
    // Packed FP4 stores two logical elements per byte (MXF4 path only).
    if bits == 4 {
        2
    } else {
        1
    }
}

/// UTCCP-aligned SF block sizes (align up to 128 rows).
pub fn sf_utccp_aligned_block_sizes(block_m: u32, block_n: u32, bf16: bool) -> (u32, u32) {
    if bf16 {
        (0, 0)
    } else {
        (block_m.div_ceil(128) * 128, block_n.div_ceil(128) * 128)
    }
}

fn get_swizzle_mode(block_size: u32, elem_size: u32) -> u32 {
    for mode in [128, 64, 32, 16] {
        if block_size * elem_size % mode == 0 {
            return mode;
        }
    }
    unreachable!("no swizzle mode fits")
}

fn get_storage_config(desc: &Sm100Desc, layout: &Sm100Layout) -> Sm100Storage {
    let load_block_m = layout.block_m / layout.cluster_n;
    let load_block_n = layout.block_n / layout.cluster_m;
    let store_block_m = if layout.swap_ab {
        16
    } else {
        layout.block_m.min(128)
    };
    let store_block_n = layout.block_n;

    let a_store_elem = if desc.a_bits == 4 { 1 } else { desc.a_bits / 8 }; // storage bytes
    let b_store_elem = if desc.b_bits == 4 { 1 } else { desc.b_bits / 8 };
    let cd_store_elem = match desc.cd_dtype {
        1 => 4,
        2 => 1,
        _ => 2,
    };
    let is_bf16 = desc.a_bits == 16;
    let (a_sz, b_sz) = if is_bf16 {
        (2, 2)
    } else {
        (a_store_elem, b_store_elem)
    };
    let pack_a = if is_bf16 {
        1
    } else {
        smem_pack_factor(desc.a_bits)
    };
    let pack_b = if is_bf16 {
        1
    } else {
        smem_pack_factor(desc.b_bits)
    };

    let inner_a = if desc.major_a_mn {
        load_block_m
    } else {
        layout.block_k
    };
    let inner_b = if desc.major_b_mn {
        load_block_n
    } else {
        layout.block_k
    };
    let swizzle_a = get_swizzle_mode(inner_a / pack_a, a_sz);
    let swizzle_b = get_swizzle_mode(inner_b / pack_b, b_sz);
    let swizzle_cd = get_swizzle_mode(store_block_n, cd_store_elem);

    Sm100Storage {
        load_block_m,
        load_block_n,
        store_block_m,
        store_block_n,
        swizzle_a,
        swizzle_b,
        swizzle_cd,
    }
}

fn get_num_tma_store_stages(desc: &Sm100Desc, layout: &Sm100Layout, bf16: bool) -> u32 {
    if bf16 || layout.swap_ab || !matches!(desc.gemm_type, GemmType::MGroupedContiguous) {
        return 2;
    }
    // K-grouped GEMMs with many K blocks per group benefit from single-stage
    // stores (pacing DRAM traffic); requires the k-grouped layout.
    let num_k_blocks_per_group = desc.k / desc.num_groups.max(1) / layout.block_k;
    let min_k_blocks = if desc.cd_dtype != 1 {
        16
    } else if desc.with_accumulation {
        24
    } else {
        32
    };
    if num_k_blocks_per_group >= min_k_blocks {
        1
    } else {
        2
    }
}

fn get_pipeline_config(
    desc: &Sm100Desc,
    layout: &Sm100Layout,
    storage: &Sm100Storage,
    bf16: bool,
) -> (u32, u32, u32) {
    const MAX_STAGES: u32 = 32;

    let num_tma_store_stages = get_num_tma_store_stages(desc, layout, bf16);
    let cd_store_elem = match desc.cd_dtype {
        1 => 4,
        2 => 1,
        _ => 2,
    };
    let smem_cd = if layout.swap_ab {
        storage.store_block_m * storage.store_block_n * cd_store_elem * num_tma_store_stages
    } else {
        storage.store_block_m * storage.swizzle_cd * num_tma_store_stages
    };

    // full barriers for A/B+SF-transpose arrivals, SF-TMA full, empty, tmem
    // full/empty/overlap, plus the tensor-core control barrier
    let smem_barriers = MAX_STAGES * 8 * 3 + 2 * 8 * 3 + 8;
    let smem_tmem_ptr = 4;

    let is_bf16 = bf16;
    let (a_sz, b_sz) = if is_bf16 { (2, 2) } else { (1, 1) };
    let pack_a = if is_bf16 {
        1
    } else {
        smem_pack_factor(desc.a_bits)
    };
    let pack_b = if is_bf16 {
        1
    } else {
        smem_pack_factor(desc.b_bits)
    };
    let smem_a_per_stage = storage.load_block_m * layout.block_k * a_sz / pack_a;
    let smem_b_per_stage = storage.load_block_n * layout.block_k * b_sz / pack_b;

    let mut smem_sfa_per_stage = 0;
    let mut smem_sfb_per_stage = 0;
    if !is_bf16 {
        let (sf_block_m, sf_block_n) =
            sf_utccp_aligned_block_sizes(layout.block_m, layout.block_n, false);
        smem_sfa_per_stage = sf_block_m * layout.block_k / 32;
        smem_sfb_per_stage = sf_block_n * layout.block_k / 32;
    }

    let smem_extra = smem_cd + smem_barriers + smem_tmem_ptr;
    let smem_per_stage =
        smem_a_per_stage + smem_b_per_stage + smem_sfa_per_stage + smem_sfb_per_stage;
    let num_stages = if smem_per_stage == 0 {
        MAX_STAGES
    } else {
        ((SM100_SMEM_CAPACITY - smem_extra) / smem_per_stage).min(MAX_STAGES)
    };
    (
        smem_extra + num_stages * smem_per_stage,
        num_stages,
        num_tma_store_stages,
    )
}

#[derive(Debug, Clone, Copy)]
struct LayoutInfo {
    num_waves: u32,
    last_wave_util: u32,
    layout: Sm100Layout,
}

fn get_layout_info(desc: &Sm100Desc, layout: &Sm100Layout) -> LayoutInfo {
    let num_blocks = desc.expected_m.div_ceil(layout.block_m)
        * desc.n.div_ceil(layout.block_n)
        * desc.expected_num_groups;
    let num_waves = num_blocks.div_ceil(desc.num_sms);
    let num_last_blocks = num_blocks % desc.num_sms;
    let last_wave_util = if num_last_blocks == 0 {
        desc.num_sms
    } else {
        num_last_blocks
    };
    LayoutInfo {
        num_waves,
        last_wave_util,
        layout: *layout,
    }
}

fn compare(a: &LayoutInfo, b: &LayoutInfo) -> bool {
    // "a is better than b" (upstream `SM100ArchSpec::compare`)
    if (a.num_waves == 1 || b.num_waves == 1) && a.num_waves != b.num_waves {
        return a.num_waves < b.num_waves;
    }
    if a.layout.cluster_size() != b.layout.cluster_size() {
        return a.layout.cluster_size() > b.layout.cluster_size();
    }
    if a.num_waves != b.num_waves {
        return a.num_waves < b.num_waves;
    }
    if a.last_wave_util != b.last_wave_util {
        return a.last_wave_util > b.last_wave_util;
    }
    if a.layout.block_m + a.layout.block_n != b.layout.block_m + b.layout.block_n {
        return a.layout.block_m + a.layout.block_n < b.layout.block_m + b.layout.block_n;
    }
    a.layout.block_m * a.layout.block_n < b.layout.block_m * b.layout.block_n
}

fn get_layout_candidates(desc: &Sm100Desc, bf16: bool) -> Vec<Sm100Layout> {
    let bits = num_element_bits(
        if bf16 { 16 } else { desc.a_bits },
        if bf16 { 16 } else { desc.b_bits },
    );
    let block_k = 128 * 8 / bits;

    // m-grouped GEMMs: always swap-AB, block N = 128 (the UMMA M)
    if is_m_grouped(desc.gemm_type) {
        let swap_ab = true;
        let block_n = 128;
        let block_m = MK_ALIGNMENT_FOR_CONTIGUOUS;
        let cluster_m = 1;
        let cluster_n = if desc.n.div_ceil(block_n) % 2 == 0 && desc.num_sms % 2 == 0 {
            2
        } else {
            1
        };
        return vec![Sm100Layout {
            swap_ab,
            block_m,
            block_n,
            block_k,
            cluster_m,
            cluster_n,
        }];
    }

    let mut candidates = Vec::new();
    for swap_ab in [false, true] {
        let (block_m_candidates, block_n_candidates): (Vec<u32>, Vec<u32>) = if swap_ab {
            let step: u32 = 16; // lcm(16, block_m_multiple_of = 1)
            let m: Vec<u32> = ((step..=256).step_by(step as usize)).collect();
            (m, vec![128])
        } else {
            let m: Vec<u32> = if desc.m <= 32 {
                vec![32]
            } else if desc.m <= 64 {
                vec![64]
            } else {
                vec![128]
            };
            let mut n = Vec::new();
            n.push(16); // block_n_multiple_of = 1
            let step: u32 = 32;
            let end = if desc.k <= 256 { 128 } else { 256 };
            let mut v = step;
            while v <= end {
                n.push(v);
                v += step;
            }
            (m, n)
        };

        for cluster_m in 1..=2u32 {
            if swap_ab && cluster_m > 1 {
                continue;
            }
            for cluster_n in 1..=2u32 {
                if cluster_m * cluster_n > 2 {
                    continue;
                }
                if !swap_ab && cluster_n > 1 {
                    continue; // only layout A/D (multicast on A)
                }
                if desc.num_sms % (cluster_m * cluster_n) != 0 {
                    continue;
                }
                for &block_m in &block_m_candidates {
                    let swizzle_a_req = if !bf16 && desc.a_bits == 4 { 128 } else { 64 };
                    let load_block_m_req = if desc.major_a_mn { swizzle_a_req } else { 8 };
                    if (block_m / cluster_n) % load_block_m_req != 0 {
                        continue;
                    }
                    if desc.m.div_ceil(block_m) % cluster_m != 0 {
                        continue;
                    }
                    for &block_n in &block_n_candidates {
                        let swizzle_b_req = if !bf16 && desc.b_bits == 4 { 128 } else { 64 };
                        let load_block_n_req = if desc.major_b_mn { swizzle_b_req } else { 8 };
                        if (block_n / cluster_m) % load_block_n_req != 0 {
                            continue;
                        }
                        if desc.n.div_ceil(block_n) % cluster_n != 0 {
                            continue;
                        }
                        if swap_ab && block_n != 128 {
                            continue;
                        }
                        // Tensor memory capacity
                        let (sf_block_m, sf_block_n) =
                            sf_utccp_aligned_block_sizes(block_m, block_n, bf16);
                        let sf_block_k = block_k / 128;
                        let tmem_sf_cols = if bf16 {
                            0
                        } else {
                            sf_block_m * sf_block_k / 32 + sf_block_n * sf_block_k / 32
                        };
                        let umma_n = if swap_ab { block_m } else { block_n };
                        if umma_n + tmem_sf_cols > 512 {
                            continue;
                        }
                        let layout = Sm100Layout {
                            swap_ab,
                            block_m,
                            block_n,
                            block_k,
                            cluster_m,
                            cluster_n,
                        };
                        // K-major operands always get 128B swizzle
                        if !desc.major_a_mn || !desc.major_b_mn {
                            let storage = get_storage_config(desc, &layout);
                            if storage.swizzle_a != 128 || storage.swizzle_b != 128 {
                                continue;
                            }
                        }
                        candidates.push(layout);
                    }
                }
            }
        }
    }

    // FP8 output requires complete per-32 SF groups per store atom
    if desc.cd_dtype == 2 {
        let owns_partial = |l: &Sm100Layout| {
            let storage = get_storage_config(desc, l);
            let store_block_n = if l.swap_ab {
                l.block_n
            } else {
                storage.swizzle_cd
            };
            store_block_n % 32 != 0
        };
        candidates.retain(|l| !owns_partial(l));
    }

    candidates
}

/// Select the best SM100 configuration (upstream `get_best_config<SM100ArchSpec>`).
pub fn best_config(desc: &Sm100Desc) -> Sm100Config {
    let bf16 = desc.a_bits == 16;
    let candidates = get_layout_candidates(desc, bf16);
    assert!(
        !candidates.is_empty(),
        "no SM100 layout candidate for the given problem"
    );
    let mut best = candidates[0];
    let mut best_info = get_layout_info(desc, &best);
    for cand in &candidates[1..] {
        let info = get_layout_info(desc, cand);
        if compare(&info, &best_info) {
            best = *cand;
            best_info = info;
        }
    }
    let storage = get_storage_config(desc, &best);
    let (smem_size, num_stages, num_tma_store_stages) =
        get_pipeline_config(desc, &best, &storage, bf16);
    Sm100Config {
        layout: best,
        storage,
        num_stages,
        num_tma_store_stages,
        smem_size,
        num_non_epilogue_threads: 128,
        num_epilogue_threads: 128,
    }
}

/// Map a crate `GemmType` (+ psum flag) to the SM100 kernel's gemm-type id.
pub fn sm100_gemm_type_id(t: GemmType, use_psum: bool) -> u32 {
    match t {
        GemmType::Normal => 0,
        GemmType::MGroupedContiguous => {
            if use_psum {
                5
            } else {
                1
            }
        }
        GemmType::MGroupedMasked => 2,
        GemmType::Batched => 4,
    }
}

/// Convert a heuristic config into the FP8/FP4 kernel template configuration.
#[allow(clippy::too_many_arguments)]
pub fn to_fp8fp4_kernel_cfg(
    cfg: &Sm100Config,
    desc: &Sm100Desc,
    gran_k_a: u32,
    gran_k_b: u32,
    num_groups: u32,
    shape_m: u32,
    shape_n: u32,
    shape_k: u32,
    epilogue_op: u32,
    with_accumulation: bool,
    is_mxf4: bool,
) -> Fp8Fp4Config {
    let block_k = if is_mxf4 { 256 } else { 128 };
    Fp8Fp4Config {
        gemm_type: sm100_gemm_type_id(desc.gemm_type, desc.use_psum_layout),
        major_a: desc.major_a_mn as u32,
        major_b: desc.major_b_mn as u32,
        gran_k_a,
        gran_k_b,
        shape_m,
        shape_n,
        shape_k,
        block_m: cfg.layout.block_m,
        block_n: cfg.layout.block_n,
        block_k,
        num_groups,
        swizzle_a: cfg.storage.swizzle_a,
        swizzle_b: cfg.storage.swizzle_b,
        swizzle_cd: cfg.storage.swizzle_cd,
        num_stages: cfg.num_stages,
        num_tma_store_stages: cfg.num_tma_store_stages,
        num_non_epilogue_threads: cfg.num_non_epilogue_threads,
        num_epilogue_threads: cfg.num_epilogue_threads,
        multicast: cfg.layout.cluster_size(),
        is_multicast_on_a: cfg.layout.is_multicast_on_a(),
        num_sms: desc.num_sms,
        swap_ab: cfg.layout.swap_ab,
        ensure_zero_padding: false,
        k_alignment: block_k.max(gran_k_a.max(gran_k_b)),
        with_accumulation,
        a_bits: desc.a_bits,
        b_bits: desc.b_bits,
        cd_dtype: desc.cd_dtype,
        epilogue_op,
    }
}

/// Convert a heuristic config into the BF16 kernel template configuration.
#[allow(clippy::too_many_arguments)]
pub fn to_bf16_kernel_cfg(
    cfg: &Sm100Config,
    desc: &Sm100Desc,
    num_groups: u32,
    shape_m: u32,
    shape_n: u32,
    shape_k: u32,
    epilogue_op: u32,
    with_accumulation: bool,
) -> Bf16Config {
    Bf16Config {
        gemm_type: sm100_gemm_type_id(desc.gemm_type, desc.use_psum_layout),
        major_a: desc.major_a_mn as u32,
        major_b: desc.major_b_mn as u32,
        shape_m,
        shape_n,
        shape_k,
        block_m: cfg.layout.block_m,
        block_n: cfg.layout.block_n,
        block_k: 64,
        num_groups,
        swizzle_a: cfg.storage.swizzle_a,
        swizzle_b: cfg.storage.swizzle_b,
        swizzle_cd: cfg.storage.swizzle_cd,
        num_stages: cfg.num_stages,
        num_non_epilogue_threads: cfg.num_non_epilogue_threads,
        num_epilogue_threads: cfg.num_epilogue_threads,
        multicast: cfg.layout.cluster_size(),
        is_multicast_on_a: cfg.layout.is_multicast_on_a(),
        num_sms: desc.num_sms,
        k_alignment: 128,
        swap_ab: cfg.layout.swap_ab,
        ensure_zero_padding: false,
        with_accumulation,
        cd_dtype: desc.cd_dtype,
        epilogue_op,
        tc_util: desc.tc_util,
    }
}
