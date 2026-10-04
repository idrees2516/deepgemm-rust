//! Kernel configuration heuristics — a port of upstream DeepGEMM's
//! `SM90ArchSpec` (block sizes, pipeline stages, cluster shape), simplified
//! where the upstream heuristics rely on torch-side runtime knobs.

use crate::device::Arch;
use crate::types::GemmType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmConfig {
    pub block_m: u32,
    pub block_n: u32,
    pub block_k: u32,
    pub num_stages: u32,
    pub swizzle_a: u32,
    pub swizzle_b: u32,
    pub swizzle_cd: u32,
    pub cluster_m: u32,
    pub cluster_n: u32,
    pub num_tma_threads: u32,
    pub num_math_threads: u32,
    pub num_sms: u32,
    pub smem_size: u32,
}

impl GemmConfig {
    pub fn cluster_size(&self) -> u32 {
        self.cluster_m * self.cluster_n
    }
    pub fn num_threads(&self) -> u32 {
        self.num_tma_threads + self.num_math_threads
    }
    pub fn is_multicast_on_a(&self) -> bool {
        // cluster_m > 1 => the same A tile feeds multiple CTAs => multicast A
        self.cluster_m > 1
    }
    pub fn num_multicast(&self) -> u32 {
        self.cluster_size()
    }
}

/// Compute the number of pipeline stages for a given layout (upstream
/// `get_pipeline_config`). `cd_smem` is the shared-memory cost of the
/// epilogue buffer; `smem_per_stage` the per-stage A+B(+SF) cost.
fn num_stages_for(smem_capacity: u32, smem_extra: u32, smem_per_stage: u32) -> u32 {
    const MAX_STAGES: u32 = 16;
    // Reserve for barriers: MAX_STAGES * 8 bytes * 2 (full/empty)
    let smem_extra = smem_extra + MAX_STAGES * 8 * 2;
    if smem_per_stage == 0 {
        return MAX_STAGES;
    }
    ((smem_capacity - smem_extra) / smem_per_stage).min(MAX_STAGES)
}

/// Select the best config for an FP8 GEMM (1d2d kernel) on SM90a.
///
/// Mirrors upstream candidate enumeration + the analytic L1/L2 cost model,
/// with the following deliberate restrictions for the 1d2d scale-factor
/// layout: `block_n >= 128` (B has 128-wide 2D scale granularity) and
/// `block_k == 128`.
pub fn best_fp8_config(
    arch: &Arch,
    gemm_type: GemmType,
    m: u32,
    n: u32,
    k: u32,
    num_groups: u32,
    expected_m: u32,
) -> GemmConfig {
    let num_sms = arch.num_sms;

    // Block M candidates (upstream)
    let mut block_m_candidates: Vec<u32> = Vec::new();
    match gemm_type {
        GemmType::Normal | GemmType::Batched => {
            block_m_candidates.extend_from_slice(&[64, 128]);
            if m <= 16 {
                block_m_candidates.push(16);
            }
            if m <= 32 {
                block_m_candidates.push(32);
            }
            block_m_candidates.push(256);
        }
        GemmType::MGroupedContiguous => {
            // Upstream: BLOCK_M fixed to the M/K alignment of the contiguous layout.
            block_m_candidates.push(crate::types::get_mk_alignment_for_contiguous_layout());
        }
        GemmType::MGroupedMasked => {
            block_m_candidates.extend_from_slice(&[64, 128]);
        }
    }

    // Block N candidates: >= 128, multiples of 16, <= 192 (register pressure)
    let mut block_n_candidates: Vec<u32> = Vec::new();
    {
        let mut bn = 128u32;
        while bn <= 192 {
            block_n_candidates.push(bn);
            bn += 16;
        }
    }

    let block_k = 128u32;
    let num_k_blocks = k.div_ceil(block_k);

    let mut best: Option<(u64, GemmConfig)> = None;
    let push = |cost: u64, cfg: GemmConfig, best: &mut Option<(u64, GemmConfig)>| {
        if best.map(|(c, _)| cost < c).unwrap_or(true) {
            *best = Some((cost, cfg));
        }
    };

    for &cluster_m in &[1u32, 2] {
        for &cluster_n in &[1u32, 2] {
            if cluster_m * cluster_n > 2 {
                continue;
            }
            if num_sms % (cluster_m * cluster_n) != 0 {
                continue;
            }
            for &block_m in &block_m_candidates {
                for &block_n in &block_n_candidates {
                    // 1d2d unroll requirement (upstream)
                    if block_n > block_k {
                        let d = block_n - block_k;
                        if block_n % d != 0 && block_k % d != 0 {
                            continue;
                        }
                    }
                    // Masked-grouped multicast legality
                    if matches!(gemm_type, GemmType::MGroupedMasked)
                        && n.div_ceil(block_n) % (cluster_m * cluster_n) != 0
                    {
                        continue;
                    }
                    // Register pressure: at least one dim <= 128
                    if block_m > 128 && block_n > 128 {
                        continue;
                    }

                    // Storage config
                    let swizzle_a = get_swizzle_mode_fp8(block_k); // 128B
                    let swizzle_b = get_swizzle_mode_fp8(block_k); // 128B
                    let swizzle_cd = get_swizzle_mode_bf16(block_n);

                    // SMEM budget
                    let smem_cd = align_up(block_m * block_n * 2, 1024);
                    // SFA per stage + uniform SFB (k_blocks floats)
                    let smem_sfa_per_stage = align_up(block_m * 4, 128);
                    let smem_sfb = align_up(num_k_blocks * 4, 8);
                    let smem_extra = smem_cd + smem_sfb;
                    let smem_per_stage = block_m * block_k + block_n * block_k + smem_sfa_per_stage;
                    let num_stages = num_stages_for(arch.smem_capacity, smem_extra, smem_per_stage);

                    // Pipeline depth requirements (upstream)
                    if num_stages < 3 {
                        continue;
                    }
                    if block_m * block_n < 128 * 192 && num_stages < 4 {
                        continue;
                    }

                    let num_math_threads = if block_m <= 64 { 128 } else { 256 };
                    let cfg = GemmConfig {
                        block_m,
                        block_n,
                        block_k,
                        num_stages,
                        swizzle_a,
                        swizzle_b,
                        swizzle_cd,
                        cluster_m,
                        cluster_n,
                        num_tma_threads: 128,
                        num_math_threads,
                        num_sms,
                        smem_size: smem_extra + num_stages * smem_per_stage,
                    };

                    // ---- Upstream analytic cost model (L1/L2 bandwidth cycles) ----
                    let (em, en, ek, eg) = match gemm_type {
                        GemmType::MGroupedMasked => (expected_m, n, k, num_groups),
                        GemmType::MGroupedContiguous => (expected_m, n, k, 1),
                        _ => (m, n, k, num_groups),
                    };
                    let num_blocks = em.div_ceil(block_m) * en.div_ceil(block_n) * eg;
                    let num_waves = num_blocks.div_ceil(num_sms);
                    let last_wave_util = {
                        let r = num_blocks % num_sms;
                        if r == 0 {
                            num_sms
                        } else {
                            r
                        }
                    };
                    let _ = last_wave_util;

                    // B/cycle estimates from upstream
                    let l2_bw: f64 = 64.0f64.min(num_sms as f64).max(1.0) * 1024.0;
                    let l1_bw: f64 = 128.0 * num_sms as f64;

                    let bytes_l2_ab: f64 = ek as f64
                        * (block_m as f64 / cluster_n as f64 + block_n as f64 / cluster_m as f64);
                    let bytes_l1_ab: f64 = ek as f64 * (block_m + block_n) as f64;
                    let bytes_l1_tc: f64 = ek as f64 * (64.max(block_m) + block_n) as f64
                        + (block_m * block_n * 2) as f64;
                    let bytes_l1_l2_cd: f64 = (block_m * block_n * 2) as f64;

                    let num_l2_cycles =
                        ((bytes_l2_ab + bytes_l1_l2_cd) * num_blocks as f64 / l2_bw).ceil();
                    let num_l1_cycles =
                        ((bytes_l1_ab + bytes_l1_tc + bytes_l1_l2_cd) * num_blocks as f64 / l1_bw)
                            .ceil();
                    let wave_eff = num_blocks as f64 / (num_waves as f64 * num_sms as f64);
                    let mut num_cycles = num_l1_cycles.max(num_l2_cycles) / wave_eff;

                    // Disable multicast if only one wave
                    if cluster_m * cluster_n > 1 && num_waves <= 1 {
                        num_cycles = f64::INFINITY;
                    }

                    push(num_cycles as u64, cfg, &mut best);
                }
            }
        }
    }

    best.map(|(_, c)| c).unwrap_or(GemmConfig {
        block_m: 64,
        block_n: 128,
        block_k: 128,
        num_stages: 4,
        swizzle_a: 128,
        swizzle_b: 128,
        swizzle_cd: 128,
        cluster_m: 1,
        cluster_n: 1,
        num_tma_threads: 128,
        num_math_threads: 128,
        num_sms,
        smem_size: 64 * 1024,
    })
}

/// BF16 (no scale factor) config: upstream `sm90_bf16_gemm` uses BLOCK_K = 64
/// elements (128 bytes) and the same warp-specialized structure.
pub fn best_bf16_config(
    arch: &Arch,
    gemm_type: GemmType,
    m: u32,
    n: u32,
    k: u32,
    num_groups: u32,
    expected_m: u32,
) -> GemmConfig {
    let num_sms = arch.num_sms;

    let mut block_m_candidates: Vec<u32> = Vec::new();
    match gemm_type {
        GemmType::Normal | GemmType::Batched => {
            block_m_candidates.extend_from_slice(&[64, 128]);
            if m <= 16 {
                block_m_candidates.push(16);
            }
            if m <= 32 {
                block_m_candidates.push(32);
            }
            block_m_candidates.push(256);
        }
        GemmType::MGroupedContiguous => {
            block_m_candidates.push(crate::types::get_mk_alignment_for_contiguous_layout());
        }
        GemmType::MGroupedMasked => {
            block_m_candidates.extend_from_slice(&[64, 128]);
        }
    }

    let mut block_n_candidates: Vec<u32> = Vec::new();
    {
        let mut bn = 128u32;
        while bn <= 256 {
            block_n_candidates.push(bn);
            bn += 16;
        }
    }

    // BLOCK_K for bf16: 64 elements = 128 bytes (same atom as fp8's 128 x 1B)
    let block_k = 64u32;

    let mut best: Option<(u64, GemmConfig)> = None;
    let push = |cost: u64, cfg: GemmConfig, best: &mut Option<(u64, GemmConfig)>| {
        if best.map(|(c, _)| cost < c).unwrap_or(true) {
            *best = Some((cost, cfg));
        }
    };

    for &cluster_m in &[1u32, 2] {
        for &cluster_n in &[1u32, 2] {
            if cluster_m * cluster_n > 2 || num_sms % (cluster_m * cluster_n) != 0 {
                continue;
            }
            for &block_m in &block_m_candidates {
                for &block_n in &block_n_candidates {
                    if matches!(gemm_type, GemmType::MGroupedMasked)
                        && n.div_ceil(block_n) % (cluster_m * cluster_n) != 0
                    {
                        continue;
                    }
                    if block_m > 128 && block_n > 128 {
                        continue;
                    }
                    let swizzle_a = 128;
                    let swizzle_b = 128;
                    let swizzle_cd = get_swizzle_mode_bf16(block_n);

                    let smem_cd = align_up(block_m * block_n * 2, 1024);
                    let smem_per_stage = (block_m * block_k + block_n * block_k) * 2;
                    let num_stages = num_stages_for(arch.smem_capacity, smem_cd, smem_per_stage);
                    if num_stages < 3 {
                        continue;
                    }
                    if block_m * block_n < 128 * 192 && num_stages < 4 {
                        continue;
                    }

                    let num_math_threads = if block_m <= 64 { 128 } else { 256 };
                    let cfg = GemmConfig {
                        block_m,
                        block_n,
                        block_k,
                        num_stages,
                        swizzle_a,
                        swizzle_b,
                        swizzle_cd,
                        cluster_m,
                        cluster_n,
                        num_tma_threads: 128,
                        num_math_threads,
                        num_sms,
                        smem_size: smem_cd + num_stages * smem_per_stage,
                    };

                    let (em, en, ek, eg) = match gemm_type {
                        GemmType::MGroupedMasked => (expected_m, n, k, num_groups),
                        GemmType::MGroupedContiguous => (expected_m, n, k, 1),
                        _ => (m, n, k, num_groups),
                    };
                    let num_blocks = em.div_ceil(block_m) * en.div_ceil(block_n) * eg;
                    let num_waves = num_blocks.div_ceil(num_sms);
                    let l2_bw: f64 = 64.0f64.min(num_sms as f64).max(1.0) * 1024.0;
                    let l1_bw: f64 = 128.0 * num_sms as f64;
                    let bytes_l2_ab: f64 =
                        ek as f64 * ((block_m / cluster_n + block_n / cluster_m) * 2) as f64;
                    let bytes_l1_ab: f64 = (ek * (block_m + block_n) * 2) as f64;
                    let bytes_l1_tc: f64 =
                        (ek * (64.max(block_m) + block_n) * 2 + block_m * block_n * 2) as f64;
                    let bytes_cd: f64 = (block_m * block_n * 2) as f64;
                    let num_l2_cycles =
                        ((bytes_l2_ab + bytes_cd) * num_blocks as f64 / l2_bw).ceil();
                    let num_l1_cycles =
                        ((bytes_l1_ab + bytes_l1_tc + bytes_cd) * num_blocks as f64 / l1_bw).ceil();
                    let wave_eff = num_blocks as f64 / (num_waves as f64 * num_sms as f64);
                    let mut num_cycles = num_l1_cycles.max(num_l2_cycles) / wave_eff;
                    if cluster_m * cluster_n > 1 && num_waves <= 1 {
                        num_cycles = f64::INFINITY;
                    }
                    push(num_cycles as u64, cfg, &mut best);
                }
            }
        }
    }

    best.map(|(_, c)| c).unwrap_or(GemmConfig {
        block_m: 64,
        block_n: 128,
        block_k: 64,
        num_stages: 4,
        swizzle_a: 128,
        swizzle_b: 128,
        swizzle_cd: 128,
        cluster_m: 1,
        cluster_n: 1,
        num_tma_threads: 128,
        num_math_threads: 128,
        num_sms,
        smem_size: 64 * 1024,
    })
}

fn get_swizzle_mode_fp8(block_k: u32) -> u32 {
    // FP8 = 1 byte: block_k=128 -> 128B swizzle
    crate::tma::get_swizzle_mode(block_k, 1)
}

fn get_swizzle_mode_bf16(block_n: u32) -> u32 {
    crate::tma::get_swizzle_mode(block_n, 2)
}

pub(crate) fn align_up(x: u32, a: u32) -> u32 {
    x.div_ceil(a) * a
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h100() -> Arch {
        Arch {
            major: 9,
            minor: 0,
            nvrtc_arch: "sm_90a",
            smem_capacity: 232448 - 1024,
            num_sms: 132,
        }
    }

    #[test]
    fn normal_fp8_picks_reasonable_config() {
        let cfg = best_fp8_config(&h100(), GemmType::Normal, 4096, 4096, 7168, 1, 4096);
        assert_eq!(cfg.block_k, 128);
        assert!(cfg.block_n >= 128 && cfg.block_n <= 192);
        assert!(cfg.smem_size <= 232448 - 1024);
        assert!(cfg.num_stages >= 3);
    }

    #[test]
    fn grouped_contiguous_uses_128_block_m() {
        let cfg = best_fp8_config(
            &h100(),
            GemmType::MGroupedContiguous,
            8192,
            4096,
            7168,
            256,
            8192,
        );
        assert_eq!(cfg.block_m, 128);
    }

    #[test]
    fn small_m_picks_small_block() {
        let cfg = best_fp8_config(&h100(), GemmType::Normal, 16, 7168, 7168, 1, 16);
        assert_eq!(cfg.block_m, 16);
        assert_eq!(cfg.num_math_threads, 128);
    }
}
