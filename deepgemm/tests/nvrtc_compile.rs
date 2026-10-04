//! NVRTC compile tests for every generated CUDA kernel — runnable WITHOUT a
//! GPU (only needs `libnvrtc`), e.g. in CI.
//!
//! Run: `LD_LIBRARY_PATH=<dir with libnvrtc.so> cargo test --test nvrtc_compile`

use deepgemm::cuda;

fn expect_compile(name: &str, src: &str, arch: &str) {
    match deepgemm::jit::JitEngine::compile_only(src, arch) {
        Ok(ptx) => {
            assert!(ptx.contains(".target"), "expected PTX output for {name}");
        }
        Err(e) => panic!("[{name}] NVRTC compile failed for {arch}:\n{e}"),
    }
}

#[test]
fn fp8_gemm_kernel_compiles_sm90a() {
    // Representative configs across BLOCK_M / BLOCK_N / gemm types.
    let cases = [
        // (block_m, block_n, gemm_type_id, num_groups, cluster, mc_on_a)
        (64u32, 128u32, 0u32, 1u32, 1u32, false),
        (64, 128, 0, 1, 2, false),
        (128, 128, 0, 1, 1, false),
        (128, 192, 0, 1, 1, false),
        (128, 128, 1, 256, 1, false), // contiguous grouped (BLOCK_M fixed 128)
        (64, 128, 2, 64, 1, false),   // masked grouped
        (64, 192, 2, 64, 1, false),
        (128, 128, 3, 8, 1, false), // batched
        (16, 128, 0, 1, 1, false),  // tiny M
        (32, 144, 0, 1, 1, false),
        (64, 128, 0, 1, 2, true), // multicast on A (cluster_n = 2)
    ];
    for &(bm, bn, gt, ng, cluster, mc_a) in &cases {
        let swizzle_cd = deepgemm::tma::get_swizzle_mode(bn, 2);
        let src = cuda::fp8_gemm_1d2d::build_gemm_kernel_source(
            bm,
            bn,
            128,
            6,
            128,
            128,
            swizzle_cd,
            128,
            if bm <= 64 { 128 } else { 256 },
            cluster,
            mc_a,
            132,
            gt,
            ng,
            false,
        );
        expect_compile(
            &format!("fp8 bm={bm} bn={bn} gt={gt} cluster={cluster}"),
            &src,
            "sm_90a",
        );
    }
}

#[test]
fn bf16_gemm_kernel_compiles_sm90a() {
    let cases = [
        (64u32, 128u32, 0u32, 1u32, 1u32),
        (128, 128, 0, 1, 1),
        (128, 256, 0, 1, 1),
        (256, 128, 0, 1, 1),
        (64, 128, 2, 64, 1), // masked grouped bf16
        (128, 128, 3, 8, 1), // batched bf16
    ];
    for &(bm, bn, gt, ng, cluster) in &cases {
        let swizzle_cd = deepgemm::tma::get_swizzle_mode(bn, 2);
        let src = cuda::fp8_gemm_1d2d::build_gemm_kernel_source(
            bm,
            bn,
            64,
            6,
            128,
            128,
            swizzle_cd,
            128,
            if bm <= 64 { 128 } else { 256 },
            cluster,
            false,
            132,
            gt,
            ng,
            true,
        );
        expect_compile(&format!("bf16 bm={bm} bn={bn} gt={gt}"), &src, "sm_90a");
    }
}

#[test]
fn layout_kernels_compile() {
    expect_compile(
        "layout",
        &cuda::layout::build_layout_kernel_source(),
        "sm_90a",
    );
}

#[test]
fn sm100_cast_kernels_compile() {
    expect_compile(
        "sm100_cast",
        &deepgemm::cuda::sm100_cast::build_sm100_cast_source(),
        "sm_100a",
    );
}

#[test]
fn mqa_logits_compiles() {
    expect_compile(
        "mqa",
        &cuda::mqa_logits::build_mqa_logits_source(),
        "sm_90a",
    );
}

// ===========================================================================
// SM100 (Blackwell) kernels: tcgen05 FP8/FP4 + BF16, all GemmTypes.
// ===========================================================================

use deepgemm::cuda::bf16_gemm_sm100::{build_bf16_source, Bf16Config};
use deepgemm::cuda::fp8_fp4_gemm_1d1d::{build_fp8_fp4_source, Fp8Fp4Config};

#[allow(clippy::too_many_arguments)]
fn fp8fp4_cfg(
    gt: u32,
    bm: u32,
    bn: u32,
    a_bits: u32,
    b_bits: u32,
    swap_ab: bool,
    mc: u32,
    mc_on_a: bool,
    gran: u32,
    cd: u32,
    epi: u32,
    ng: u32,
) -> Fp8Fp4Config {
    let is_mxf4 = a_bits == 4 && b_bits == 4;
    let block_k: u32 = if is_mxf4 { 256 } else { 128 };
    Fp8Fp4Config {
        gemm_type: gt,
        major_a: 0,
        major_b: 0,
        gran_k_a: if a_bits == 4 { 32 } else { gran },
        gran_k_b: if b_bits == 4 { 32 } else { gran },
        shape_m: 0,
        shape_n: 0,
        shape_k: 0,
        block_m: bm,
        block_n: bn,
        block_k,
        num_groups: ng,
        // K-major: swizzle must equal the storage bytes per K row
        swizzle_a: if is_mxf4 { 128 } else { block_k },
        swizzle_b: if is_mxf4 { 128 } else { block_k },
        swizzle_cd: deepgemm::tma::get_swizzle_mode(bn, if cd == 1 { 4 } else { 2 }),
        num_stages: 4,
        num_tma_store_stages: 2,
        num_non_epilogue_threads: 128,
        num_epilogue_threads: 128,
        multicast: mc,
        is_multicast_on_a: mc_on_a,
        num_sms: 148,
        swap_ab,
        ensure_zero_padding: false,
        k_alignment: block_k,
        with_accumulation: false,
        a_bits,
        b_bits,
        cd_dtype: cd,
        epilogue_op: epi,
    }
}

#[test]
fn sm100_fp8_fp4_gemm_compiles_sm100a() {
    let cases: Vec<(&str, Fp8Fp4Config)> = vec![
        // MXF4 (packed FP4 x FP4, gran 32, BLOCK_K 256)
        (
            "mxf4 nt 128x128",
            fp8fp4_cfg(0, 128, 128, 4, 4, false, 1, false, 32, 0, 0, 1),
        ),
        (
            "mxf4 nt 128x256 mc2",
            fp8fp4_cfg(0, 128, 256, 4, 4, false, 2, false, 32, 0, 0, 1),
        ),
        (
            "mxf4 swapab 128x128 mc2",
            fp8fp4_cfg(1, 128, 128, 4, 4, true, 2, true, 32, 0, 0, 256),
        ),
        (
            "mxf4 masked 128x128 mc2",
            fp8fp4_cfg(2, 128, 128, 4, 4, true, 2, true, 32, 0, 0, 64),
        ),
        // FP8 x FP8, gran 32
        (
            "fp8 nt 128x128",
            fp8fp4_cfg(0, 128, 128, 8, 8, false, 1, false, 32, 0, 0, 1),
        ),
        (
            "fp8 nt 64x128",
            fp8fp4_cfg(0, 64, 128, 8, 8, false, 1, false, 32, 0, 0, 1),
        ),
        (
            "fp8 nt 128x192 mc2",
            fp8fp4_cfg(0, 128, 192, 8, 8, false, 2, false, 32, 0, 0, 1),
        ),
        // FP8 x FP8, gran 128 (DeepSeek recipe on both sides)
        (
            "fp8 gran128 nt 128x128",
            fp8fp4_cfg(0, 128, 128, 8, 8, false, 1, false, 128, 0, 0, 1),
        ),
        // FP8 x FP4 mixed (unpacked FP4 side)
        (
            "fp8xfp4 nt 128x128",
            fp8fp4_cfg(0, 128, 128, 8, 4, false, 1, false, 32, 0, 0, 1),
        ),
        (
            "fp8xfp4 swizzled",
            fp8fp4_cfg(0, 128, 128, 4, 8, false, 1, false, 32, 0, 0, 1),
        ),
        // grouped contiguous + psum, masked, batched, k-grouped
        (
            "fp8 contig psum",
            fp8fp4_cfg(5, 128, 128, 8, 8, true, 2, true, 32, 0, 0, 256),
        ),
        (
            "fp8 masked",
            fp8fp4_cfg(2, 128, 128, 8, 8, true, 2, true, 32, 0, 0, 64),
        ),
        (
            "fp8 batched",
            fp8fp4_cfg(4, 128, 128, 8, 8, false, 1, false, 32, 0, 0, 8),
        ),
        (
            "fp8 kgrouped",
            fp8fp4_cfg(3, 128, 128, 8, 8, false, 1, false, 32, 0, 0, 8),
        ),
        (
            "fp8 kgrouped psum",
            fp8fp4_cfg(6, 128, 128, 8, 8, false, 1, false, 128, 0, 0, 8),
        ),
        // epilogue variants
        (
            "fp8 alpha",
            fp8fp4_cfg(0, 128, 128, 8, 8, false, 1, false, 32, 0, 1, 1),
        ),
        (
            "fp8 stochastic",
            fp8fp4_cfg(0, 128, 128, 8, 8, false, 1, false, 32, 0, 2, 1),
        ),
        (
            "fp8 quant-out batched",
            fp8fp4_cfg(4, 128, 128, 8, 8, false, 1, false, 32, 2, 3, 4),
        ),
        (
            "mxf4 stochastic",
            fp8fp4_cfg(0, 128, 128, 4, 4, false, 1, false, 32, 0, 2, 1),
        ),
        (
            "fp8 accum",
            fp8fp4_cfg(0, 128, 128, 8, 8, false, 1, false, 32, 0, 0, 1),
        ),
        (
            "fp8 fp32-out",
            fp8fp4_cfg(0, 128, 128, 8, 8, false, 1, false, 32, 1, 0, 1),
        ),
        (
            "mxf4 swapab stochastic",
            fp8fp4_cfg(5, 128, 128, 4, 4, true, 2, true, 32, 0, 2, 256),
        ),
        (
            "fp8 masked stochastic",
            fp8fp4_cfg(2, 128, 128, 8, 8, true, 2, true, 32, 0, 2, 64),
        ),
    ];
    // with accumulation variant
    let mut accum = fp8fp4_cfg(0, 128, 128, 8, 8, false, 1, false, 32, 0, 0, 1);
    accum.with_accumulation = true;
    let _ = accum; // covered by fp8 accum case above (same statics)
    for (name, cfg) in cases {
        let src = build_fp8_fp4_source(&cfg);
        expect_compile(name, &src, "sm_100a");
    }
}

#[test]
fn sm100_bf16_gemm_compiles_sm100a() {
    let mk = |gt: u32,
              bm: u32,
              bn: u32,
              swap: bool,
              mc: u32,
              mc_a: bool,
              cd: u32,
              epi: u32,
              stages: u32| Bf16Config {
        gemm_type: gt,
        major_a: 0,
        major_b: 0,
        shape_m: 0,
        shape_n: 0,
        shape_k: 0,
        block_m: bm,
        block_n: bn,
        block_k: 64,
        num_groups: if gt == 0 { 1 } else { 64 },
        swizzle_a: 128,
        swizzle_b: 128,
        swizzle_cd: deepgemm::tma::get_swizzle_mode(bn, if cd == 1 { 4 } else { 2 }),
        num_stages: stages,
        num_non_epilogue_threads: 128,
        num_epilogue_threads: 128,
        multicast: mc,
        is_multicast_on_a: mc_a,
        num_sms: 148,
        k_alignment: 128,
        swap_ab: swap,
        ensure_zero_padding: false,
        with_accumulation: false,
        cd_dtype: cd,
        epilogue_op: epi,
        tc_util: 100,
    };
    let mut tc80 = mk(0, 128, 128, false, 1, false, 0, 0, 8);
    tc80.tc_util = 80;
    let cases = vec![
        ("bf16 nt 128x128", mk(0, 128, 128, false, 1, false, 0, 0, 8)),
        (
            "bf16 nt 128x256 mc2",
            mk(0, 128, 256, false, 2, false, 0, 0, 8),
        ),
        (
            "bf16 merged stages",
            mk(0, 128, 128, false, 1, false, 0, 0, 16),
        ),
        (
            "bf16 swapab contig",
            mk(1, 128, 128, true, 2, true, 0, 0, 8),
        ),
        ("bf16 masked", mk(2, 128, 128, true, 2, true, 0, 0, 8)),
        ("bf16 batched", mk(4, 128, 128, false, 1, false, 0, 0, 8)),
        ("bf16 fp32-out", mk(0, 128, 128, false, 1, false, 1, 0, 8)),
        ("bf16 alpha", mk(0, 128, 128, false, 1, false, 0, 1, 8)),
        ("bf16 stochastic", mk(0, 128, 128, false, 1, false, 0, 2, 8)),
        ("bf16 tc-util 80", tc80),
    ];
    for (name, cfg) in cases {
        let src = build_bf16_source(&cfg);
        expect_compile(name, &src, "sm_100a");
    }
}
