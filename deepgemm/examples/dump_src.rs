fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bm: u32 = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(64);
    let bn: u32 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(128);
    let bk: u32 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(128);
    let st: u32 = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(6);
    let src = deepgemm::cuda::fp8_gemm_1d2d::build_gemm_kernel_source(
        bm,
        bn,
        bk,
        st,
        128,
        128,
        128,
        128,
        if bm <= 64 { 128 } else { 256 },
        1,
        false,
        132,
        0,
        1,
        false,
    );
    std::fs::write("/tmp/kernel_gen.cu", src).unwrap();
    println!("written bm={bm} bn={bn}");
}
