//! Debug helper: compile an SM100 fp8-quantize-out config and dump the PTX.

fn main() {
    let cfg = deepgemm::cuda::fp8_fp4_gemm_1d1d::Fp8Fp4Config {
        gemm_type: 4,
        major_a: 0,
        major_b: 0,
        gran_k_a: 32,
        gran_k_b: 32,
        shape_m: 0,
        shape_n: 0,
        shape_k: 0,
        block_m: 128,
        block_n: 128,
        block_k: 128,
        num_groups: 4,
        swizzle_a: 128,
        swizzle_b: 128,
        swizzle_cd: 128,
        num_stages: 4,
        num_tma_store_stages: 2,
        num_non_epilogue_threads: 128,
        num_epilogue_threads: 128,
        multicast: 1,
        is_multicast_on_a: false,
        num_sms: 148,
        swap_ab: false,
        ensure_zero_padding: false,
        k_alignment: 128,
        with_accumulation: false,
        a_bits: 8,
        b_bits: 8,
        cd_dtype: 2,
        epilogue_op: 3,
    };
    let src = deepgemm::cuda::fp8_fp4_gemm_1d1d::build_fp8_fp4_source(&cfg);
    std::fs::write("/tmp/sm100_quant.cu", &src).unwrap();
    match deepgemm::jit::JitEngine::compile_only(&src, "sm_100a") {
        Ok(ptx) => {
            std::fs::write("/tmp/sm100_quant.ptx", &ptx).unwrap();
            println!("ptx written ({} lines)", ptx.lines().count());
        }
        Err(e) => println!("compile err: {e}"),
    }
}
