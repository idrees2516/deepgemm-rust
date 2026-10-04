//! Debug helper: dump a generated SM100 kernel source to /tmp for inspection.

fn main() {
    let which = std::env::args().nth(1).unwrap_or_else(|| "bf16".into());
    let src = match which.as_str() {
        "bf16" => {
            let cfg = deepgemm::cuda::bf16_gemm_sm100::Bf16Config {
                gemm_type: 0,
                major_a: 0,
                major_b: 0,
                shape_m: 0,
                shape_n: 0,
                shape_k: 0,
                block_m: 128,
                block_n: 128,
                block_k: 64,
                num_groups: 1,
                swizzle_a: 128,
                swizzle_b: 128,
                swizzle_cd: 128,
                num_stages: 8,
                num_non_epilogue_threads: 128,
                num_epilogue_threads: 128,
                multicast: 1,
                is_multicast_on_a: false,
                num_sms: 148,
                k_alignment: 128,
                swap_ab: false,
                ensure_zero_padding: false,
                with_accumulation: false,
                cd_dtype: 0,
                epilogue_op: 0,
                tc_util: 100,
            };
            deepgemm::cuda::bf16_gemm_sm100::build_bf16_source(&cfg)
        }
        _ => {
            let cfg = deepgemm::cuda::fp8_fp4_gemm_1d1d::Fp8Fp4Config {
                gemm_type: 0,
                major_a: 0,
                major_b: 0,
                gran_k_a: 32,
                gran_k_b: 32,
                shape_m: 0,
                shape_n: 0,
                shape_k: 0,
                block_m: 128,
                block_n: 128,
                block_k: 256,
                num_groups: 1,
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
                k_alignment: 256,
                with_accumulation: false,
                a_bits: 4,
                b_bits: 4,
                cd_dtype: 0,
                epilogue_op: 0,
            };
            deepgemm::cuda::fp8_fp4_gemm_1d1d::build_fp8_fp4_source(&cfg)
        }
    };
    let path = format!("/tmp/sm100_{which}.cu");
    std::fs::write(&path, &src).unwrap();
    println!("written {path} ({} lines)", src.lines().count());
}
