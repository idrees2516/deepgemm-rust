//! deepgemm-bench: correctness + benchmark CLI.
//!
//! ```sh
//! # benchmark the DeepSeek-style FP8 GEMM shapes
//! cargo run --release -p deepgemm-bench -- bench --op fp8_nt --m 4096 --n 7168 --k 7168
//!
//! # MoE grouped GEMM (contiguous), 256 experts
//! cargo run --release -p deepgemm-bench -- bench --op grouped --groups 256 --n 7168 --k 7168
//!
//! # verify against the CPU reference
//! cargo run --release -p deepgemm-bench -- verify --op fp8_nt --m 256 --n 512 --k 768
//! ```

use clap::{Parser, Subcommand};
use deepgemm::prelude::*;
use deepgemm::reference as rf;

#[derive(Parser)]
#[command(
    name = "deepgemm-bench",
    about = "DeepGEMM-Rust benchmark & verification CLI"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Benchmark a kernel
    Bench {
        #[arg(long, default_value = "fp8_nt")]
        op: String,
        #[arg(long, default_value = "4096")]
        m: u32,
        #[arg(long, default_value = "7168")]
        n: u32,
        #[arg(long, default_value = "7168")]
        k: u32,
        #[arg(long, default_value = "256")]
        groups: u32,
        #[arg(long, default_value = "50")]
        iters: u32,
        #[arg(long, default_value = "10")]
        warmup: u32,
    },
    /// Verify correctness against the CPU reference
    Verify {
        #[arg(long, default_value = "fp8_nt")]
        op: String,
        #[arg(long, default_value = "256")]
        m: u32,
        #[arg(long, default_value = "512")]
        n: u32,
        #[arg(long, default_value = "768")]
        k: u32,
        #[arg(long, default_value = "8")]
        groups: u32,
    },
    /// List the device and the chosen kernel configs
    Info {
        #[arg(long, default_value = "4096")]
        m: u32,
        #[arg(long, default_value = "7168")]
        n: u32,
        #[arg(long, default_value = "7168")]
        k: u32,
    },
}

struct Timing {
    ctx: std::sync::Arc<DgContext>,
}

impl Timing {
    fn run<F: FnMut()>(&self, label: &str, flops: f64, warmup: u32, iters: u32, mut f: F) {
        for _ in 0..warmup {
            f();
        }
        self.ctx.sync().unwrap();
        let start = std::time::Instant::now();
        for _ in 0..iters {
            f();
        }
        self.ctx.sync().unwrap();
        let ms = start.elapsed().as_secs_f64() * 1e3 / iters as f64;
        let tflops = flops / (ms / 1e3) / 1e12;
        println!("{label:<44} {ms:>10.3} ms   {tflops:>8.2} TFLOPS");
    }
}

fn main() {
    let cli = Cli::parse();
    let ctx = DgContext::new(0).expect("failed to init CUDA context");
    println!(
        "device: sm_{}{}, {} SMs, smem {} KB",
        ctx.arch.major,
        ctx.arch.minor,
        ctx.arch.num_sms,
        ctx.arch.smem_capacity / 1024
    );

    match cli.cmd {
        Cmd::Info { m, n, k } => {
            let cfg =
                deepgemm::heuristics::best_fp8_config(&ctx.arch, GemmType::Normal, m, n, k, 1, m);
            println!("fp8_nt({m}x{n}x{k}) config:");
            println!(
                "  block: {}x{}x{}, stages: {}",
                cfg.block_m, cfg.block_n, cfg.block_k, cfg.num_stages
            );
            println!(
                "  threads: {} (tma {} + math {}), cluster: {}x{}",
                cfg.num_threads(),
                cfg.num_tma_threads,
                cfg.num_math_threads,
                cfg.cluster_m,
                cfg.cluster_n
            );
            println!(
                "  swizzle: a={} b={} d={}",
                cfg.swizzle_a, cfg.swizzle_b, cfg.swizzle_cd
            );
            println!(
                "  smem: {} KB",
                deepgemm::ops::exact_smem_size(&cfg, k, false) / 1024
            );
        }
        Cmd::Bench {
            op,
            m,
            n,
            k,
            groups,
            iters,
            warmup,
        } => {
            let t = Timing {
                ctx: std::sync::Arc::new(ctx),
            };
            match op.as_str() {
                "fp4_nt_native" => {
                    // Blackwell-native MXFP4 (e2m1 + UE8M0 per-32) end to end
                    let flops = 2.0 * m as f64 * n as f64 * k as f64;
                    let bf16 = |seed: u32, len: u32| -> Vec<u16> {
                        (0..len)
                            .map(|i| {
                                let v = (((i.wrapping_mul(seed)) % 17) as f32 - 8.0) / 8.0;
                                ((v.to_bits() + 0x7FFF) >> 16) as u16
                            })
                            .collect()
                    };
                    let a = bf16(3, m * k);
                    let b = bf16(5, n * k);
                    let ctx2 = &t.ctx;
                    let mut out: Option<Vec<u16>> = None;
                    t.run(
                        "fp4_nt_native (incl. quantize)",
                        flops,
                        warmup,
                        iters,
                        || {
                            out =
                                deepgemm::ops_sm100::fp4_gemm_nt_native(ctx2, &a, &b, m, n, k).ok();
                        },
                    );
                }
                "bf16_nt_sm100" => {
                    let flops = 2.0 * m as f64 * n as f64 * k as f64;
                    let a = (0..m * k)
                        .map(|i| ((((i % 7) as f32 - 3.0) / 8.0).to_bits() >> 16) as u16)
                        .collect::<Vec<_>>();
                    let b = (0..n * k)
                        .map(|i| ((((i % 5) as f32 - 2.0) / 8.0).to_bits() >> 16) as u16)
                        .collect::<Vec<_>>();
                    let ctx2 = &t.ctx;
                    let a_dev = deepgemm::ops::upload(ctx2, &a).unwrap();
                    let b_dev = deepgemm::ops::upload(ctx2, &b).unwrap();
                    let mut out = unsafe { ctx2.stream.alloc::<u16>((m * n) as usize).unwrap() };
                    t.run("bf16_nt_sm100", flops, warmup, iters, || {
                        deepgemm::ops_sm100::bf16_gemm_dev(
                            ctx2,
                            &a_dev,
                            k,
                            m,
                            &b_dev,
                            k,
                            n,
                            k,
                            &mut deepgemm::ops_sm100::Sm100Out::Bf16 {
                                buf: &mut out,
                                ld: n,
                            },
                            &deepgemm::ops_sm100::Sm100Epilogue::default(),
                        )
                        .unwrap();
                    });
                }
                "fp8_nt" => {
                    let flops = 2.0 * m as f64 * n as f64 * k as f64;
                    let (a, sfa) = rand_operand(m, k);
                    let (b, sfb) = rand_operand(n, k);
                    let ctx = &t.ctx;
                    let a_dev = upload(ctx, &a).unwrap();
                    let b_dev = upload(ctx, &b).unwrap();
                    let sfb_dev = upload(ctx, &sfb).unwrap();
                    let sfa_t = transform_sf(ctx, &sfa, m, k.div_ceil(128), 1).unwrap();
                    let mut out = unsafe { ctx.stream.alloc::<u16>((m * n) as usize).unwrap() };
                    let (mut f, label) = {
                        let ctx = ctx.clone();
                        let label = format!("fp8_gemm_nt {m}x{n}x{k}");
                        (
                            move || {
                                deepgemm::ops::fp8_gemm_nt_dev(
                                    &ctx,
                                    &Fp8Tensor::new(&a_dev, &sfa_t, m, k),
                                    &sfa_t,
                                    &Fp8Tensor::new(&b_dev, &sfb_dev, n, k),
                                    &sfb_dev,
                                    &mut out,
                                    n,
                                )
                                .unwrap();
                            },
                            label,
                        )
                    };
                    t.run(&label, flops, warmup, iters, &mut f);
                }
                "bf16_nt" => {
                    let flops = 2.0 * m as f64 * n as f64 * k as f64;
                    let a: Vec<u16> = (0..m * k)
                        .map(|i| rf::f32_to_bf16((i % 13) as f32 / 13.0 - 0.5))
                        .collect();
                    let b: Vec<u16> = (0..n * k)
                        .map(|i| rf::f32_to_bf16((i % 11) as f32 / 11.0 - 0.5))
                        .collect();
                    let ctx = &t.ctx;
                    let a_dev = upload(ctx, &a).unwrap();
                    let b_dev = upload(ctx, &b).unwrap();
                    let mut out = unsafe { ctx.stream.alloc::<u16>((m * n) as usize).unwrap() };
                    let label = format!("bf16_gemm_nt {m}x{n}x{k}");
                    let mut f = || {
                        deepgemm::ops::bf16_gemm_nt_dev(
                            ctx,
                            &Bf16Tensor::new(&a_dev, m, k),
                            &Bf16Tensor::new(&b_dev, n, k),
                            &mut out,
                            n,
                        )
                        .unwrap();
                    };
                    t.run(&label, flops, warmup, iters, &mut f);
                }
                "grouped" => {
                    // Contiguous grouped GEMM: `groups` experts of 128 tokens each.
                    let gm = 128u32;
                    let m_total = groups * gm;
                    let flops = 2.0 * m_total as f64 * n as f64 * k as f64;
                    let (a, sfa) = rand_operand(m_total, k);
                    let (b, sfb) = rand_operand(groups * n, k);
                    let m_indices: Vec<i32> = (0..m_total).map(|r| (r / gm) as i32).collect();
                    let ctx = &t.ctx;
                    let a_dev = upload(ctx, &a).unwrap();
                    let b_dev = upload(ctx, &b).unwrap();
                    let sfb_dev = upload(ctx, &sfb).unwrap();
                    let m_idx = upload(ctx, &m_indices).unwrap();
                    let sfa_t = transform_sf(ctx, &sfa, m_total, k.div_ceil(128), 1).unwrap();
                    let mut out =
                        unsafe { ctx.stream.alloc::<u16>((m_total * n) as usize).unwrap() };
                    let label =
                        format!("m_grouped contiguous g={groups} {gm}x{n}x{k} (per expert)");
                    let mut f = || {
                        deepgemm::ops::m_grouped_fp8_gemm_nt_contiguous_dev(
                            ctx,
                            &Fp8Tensor::new(&a_dev, &sfa_t, m_total, k),
                            &sfa_t,
                            &m_idx,
                            &Fp8Tensor::new(&b_dev, &sfb_dev, groups * n, k),
                            &sfb_dev,
                            groups,
                            &mut out,
                            n,
                        )
                        .unwrap();
                    };
                    t.run(&label, flops, warmup, iters, &mut f);
                }
                other => eprintln!("unknown op {other:?} (fp8_nt | bf16_nt | grouped)"),
            }
        }
        Cmd::Verify {
            op,
            m,
            n,
            k,
            groups,
        } => match op.as_str() {
            "fp8_nt" => {
                let ctx_ref = &ctx;
                let (a, sfa) = rand_operand(m, k);
                let (b, sfb) = rand_operand(n, k);
                let out_dev = fp8_gemm_nt(ctx_ref, &a, &sfa, m, &b, &sfb, n, k).unwrap();
                let got = download(ctx_ref, &out_dev).unwrap();
                let want =
                    rf::ref_fp8_gemm_nt(&a, &sfa, m as usize, &b, &sfb, n as usize, k as usize);
                let mut worst = 0f32;
                for (g, w) in got.iter().zip(want.iter()) {
                    let g = rf::bf16_to_f32(*g);
                    worst = worst.max((g - w).abs() / w.abs().max(1.0));
                }
                println!(
                    "fp8_nt verify: max rel err = {worst:.4} -> {}",
                    if worst < 0.05 { "PASS" } else { "FAIL" }
                );
            }
            "grouped" => {
                let gm = 128u32;
                let m_total = groups * gm;
                let (a, sfa) = rand_operand(m_total, k);
                let (b, sfb) = rand_operand(groups * n, k);
                let m_indices: Vec<i32> = (0..m_total).map(|r| (r / gm) as i32).collect();
                let a_dev = upload(&ctx, &a).unwrap();
                let b_dev = upload(&ctx, &b).unwrap();
                let sfb_dev = upload(&ctx, &sfb).unwrap();
                let m_idx = upload(&ctx, &m_indices).unwrap();
                let sfa_t = transform_sf(&ctx, &sfa, m_total, k.div_ceil(128), 1).unwrap();
                let mut out = unsafe { ctx.stream.alloc::<u16>((m_total * n) as usize).unwrap() };
                deepgemm::ops::m_grouped_fp8_gemm_nt_contiguous_dev(
                    &ctx,
                    &Fp8Tensor::new(&a_dev, &sfa_t, m_total, k),
                    &sfa_t,
                    &m_idx,
                    &Fp8Tensor::new(&b_dev, &sfb_dev, groups * n, k),
                    &sfb_dev,
                    groups,
                    &mut out,
                    n,
                )
                .unwrap();
                let got = download(&ctx, &out).unwrap();
                let want = rf::ref_m_grouped_fp8_gemm_nt_contiguous(
                    &a,
                    &sfa,
                    &m_indices,
                    m_total as usize,
                    &b,
                    &sfb,
                    n as usize,
                    k as usize,
                );
                let mut worst = 0f32;
                for (g, w) in got.iter().zip(want.iter()) {
                    let g = rf::bf16_to_f32(*g);
                    worst = worst.max((g - w).abs() / w.abs().max(1.0));
                }
                println!(
                    "grouped verify: max rel err = {worst:.4} -> {}",
                    if worst < 0.05 { "PASS" } else { "FAIL" }
                );
            }
            other => eprintln!("unknown op {other:?} (fp8_nt | grouped)"),
        },
    }
}

fn rand_operand(rows: u32, k: u32) -> (Vec<u8>, Vec<f32>) {
    let mut vals = Vec::with_capacity((rows * k) as usize);
    let mut s = 0x853c49e6748fea9bu64;
    for _ in 0..rows * k {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        vals.push(((s >> 33) as f32 / u32::MAX as f32 - 0.5) * 0.5);
    }
    let data = rf::quantize_e4m3(&vals);
    let sfa = vec![0.01f32; (rows * k.div_ceil(128)) as usize];
    (data, sfa)
}
