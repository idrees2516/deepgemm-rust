//! End-to-end correctness tests — require a CUDA GPU (Hopper for the FP8
//! kernels). Run on a GPU machine with:
//!
//! ```sh
//! cargo test -p deepgemm --features e2e --test e2e -- --nocapture
//! ```

use deepgemm::prelude::*;
use deepgemm::reference as rf;

fn make_operand(rows: u32, k: u32, seed: u32) -> (Vec<u8>, Vec<f32>) {
    let mut vals = Vec::with_capacity((rows * k) as usize);
    let mut s = seed as u64 * 6364136223846793005 + 1442695040888963407;
    let scale = if seed % 2 == 0 { 0.5 } else { 1.0 };
    for _ in 0..rows * k {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let v = ((s >> 33) as f32 / u32::MAX as f32 - 0.5) * scale;
        vals.push(v);
    }
    let data = rf::quantize_e4m3(&vals);
    // Per-128-block scales for activations; block scales for weights.
    let sf = if seed % 2 == 0 {
        vec![0.01f32; (rows * k.div_ceil(128)) as usize]
    } else {
        vec![0.02f32; (rows.div_ceil(128) * k.div_ceil(128)) as usize]
    };
    (data, sf)
}

fn max_rel_err(got: &[u16], want: &[f32]) -> (f32, usize) {
    let mut worst = 0f32;
    let mut at = 0usize;
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let g = rf::bf16_to_f32(g);
        let denom = w.abs().max(1.0);
        let e = (g - w).abs() / denom;
        if e > worst {
            worst = e;
            at = i;
        }
    }
    (worst, at)
}

#[test]
fn e2e_fp8_gemm_nt_correct() {
    let ctx = DgContext::new(0).expect("need a CUDA device");
    if !ctx.arch.has_wgmma() {
        eprintln!(
            "skipping: FP8 kernels need Hopper; got sm_{}{}",
            ctx.arch.major, ctx.arch.minor
        );
        return;
    }
    let (m, n, k) = (256u32, 512u32, 768u32);
    let (a, sfa) = make_operand(m, k, 0);
    let (b, sfb) = make_operand(n, k, 1);

    let out_dev = fp8_gemm_nt(&ctx, &a, &sfa, m, &b, &sfb, n, k).unwrap();
    let out = download(&ctx, &out_dev).unwrap();
    let want = rf::ref_fp8_gemm_nt(&a, &sfa, m as usize, &b, &sfb, n as usize, k as usize);

    let (err, at) = max_rel_err(&out, &want);
    let m_at = at / n as usize;
    println!(
        "fp8_nt max rel err = {err:.4} at ({m_at}, {})",
        at % n as usize
    );
    // bf16 rounding + fp8 quantization: generous tolerance
    assert!(err < 0.05, "fp8_gemm_nt error too large: {err}");
}

#[test]
fn e2e_m_grouped_contiguous_correct() {
    let ctx = DgContext::new(0).expect("need a CUDA device");
    if !ctx.arch.has_wgmma() {
        eprintln!("skipping: needs Hopper");
        return;
    }
    let (num_groups, n, k) = (4u32, 256u32, 512u32);
    let group_m = 128u32; // aligned to BLOCK_M(128)
    let m = num_groups * group_m;

    let (a, sfa) = make_operand(m, k, 0);
    let (b, sfb) = make_operand(num_groups * n, k, 1);

    // Tokens of group g occupy rows [g*group_m, (g+1)*group_m)
    let m_indices: Vec<i32> = (0..m).map(|r| (r / group_m) as i32).collect();

    // Upload + transform
    let a_dev = upload(&ctx, &a).unwrap();
    let b_dev = upload(&ctx, &b).unwrap();
    let sfb_dev = upload(&ctx, &sfb).unwrap();
    let m_indices_dev = upload(&ctx, &m_indices).unwrap();
    let sfa_t = transform_sf(&ctx, &sfa, m, k.div_ceil(128), 1).unwrap();

    let mut out = unsafe { ctx.stream.alloc::<u16>((m * n) as usize).unwrap() };
    deepgemm::ops::m_grouped_fp8_gemm_nt_contiguous_dev(
        &ctx,
        &Fp8Tensor::new(&a_dev, &sfa_t, m, k),
        &sfa_t,
        &m_indices_dev,
        &Fp8Tensor::new(&b_dev, &sfb_dev, num_groups * n, k),
        &sfb_dev,
        num_groups,
        &mut out,
        n,
    )
    .unwrap();

    let got = download(&ctx, &out).unwrap();
    let want = rf::ref_m_grouped_fp8_gemm_nt_contiguous(
        &a, &sfa, &m_indices, m as usize, &b, &sfb, n as usize, k as usize,
    );
    let (err, _) = max_rel_err(&got, &want);
    println!("grouped_contiguous max rel err = {err:.4}");
    assert!(err < 0.05, "grouped contiguous error too large: {err}");
}

#[test]
fn e2e_m_grouped_masked_correct() {
    let ctx = DgContext::new(0).expect("need a CUDA device");
    if !ctx.arch.has_wgmma() {
        eprintln!("skipping: needs Hopper");
        return;
    }
    let (num_groups, n, k) = (8u32, 256u32, 512u32);
    let m_per_group = 128u32;
    let masked_m: Vec<i32> = vec![37, 128, 1, 64, 15, 100, 128, 3];
    let m_total = num_groups * m_per_group;

    let (a, sfa) = make_operand(m_total, k, 0);
    let (b, sfb) = make_operand(num_groups * n, k, 1);

    let a_dev = upload(&ctx, &a).unwrap();
    let b_dev = upload(&ctx, &b).unwrap();
    let sfb_dev = upload(&ctx, &sfb).unwrap();
    let masked_m_dev = upload(&ctx, &masked_m).unwrap();
    let sfa_t = transform_sf(&ctx, &sfa, m_per_group, k.div_ceil(128), num_groups).unwrap();

    let mut out = unsafe { ctx.stream.alloc::<u16>((m_total * n) as usize).unwrap() };
    deepgemm::ops::m_grouped_fp8_gemm_nt_masked_dev(
        &ctx,
        &Fp8Tensor::new(&a_dev, &sfa_t, m_total, k),
        &sfa_t,
        &masked_m_dev,
        &Fp8Tensor::new(&b_dev, &sfb_dev, num_groups * n, k),
        &sfb_dev,
        num_groups,
        &mut out,
        n,
        128, // expected_m
    )
    .unwrap();

    let got = download(&ctx, &out).unwrap();
    let want = rf::ref_m_grouped_fp8_gemm_nt_masked(
        &a,
        &sfa,
        &masked_m,
        num_groups as usize,
        m_per_group as usize,
        &b,
        &sfb,
        n as usize,
        k as usize,
    );
    // Only compare valid rows
    let mut worst = 0f32;
    for g in 0..num_groups as usize {
        for r in 0..masked_m[g] as usize {
            let row = g * m_per_group as usize + r;
            for c in 0..n as usize {
                let gv = rf::bf16_to_f32(got[row * n as usize + c]);
                let wv = want[row * n as usize + c];
                let e = (gv - wv).abs() / wv.abs().max(1.0);
                worst = worst.max(e);
            }
        }
    }
    println!("grouped_masked max rel err = {worst:.4}");
    assert!(worst < 0.05, "grouped masked error too large: {worst}");
}

#[test]
fn e2e_bf16_gemm_correct() {
    let ctx = DgContext::new(0).expect("need a CUDA device");
    if !ctx.arch.has_wgmma() {
        eprintln!("skipping: needs Hopper");
        return;
    }
    let (m, n, k) = (128u32, 256u32, 512u32);
    let a: Vec<u16> = (0..m * k)
        .map(|i| rf::f32_to_bf16((i as i32 % 13) as f32 / 13.0 - 0.5))
        .collect();
    let b: Vec<u16> = (0..n * k)
        .map(|i| rf::f32_to_bf16((i as i32 % 11) as f32 / 11.0 - 0.5))
        .collect();

    let a_dev = upload(&ctx, &a).unwrap();
    let b_dev = upload(&ctx, &b).unwrap();
    let mut out = unsafe { ctx.stream.alloc::<u16>((m * n) as usize).unwrap() };
    deepgemm::ops::bf16_gemm_nt_dev(
        &ctx,
        &Bf16Tensor::new(&a_dev, m, k),
        &Bf16Tensor::new(&b_dev, n, k),
        &mut out,
        n,
    )
    .unwrap();
    let got = download(&ctx, &out).unwrap();
    let want = rf::ref_bf16_gemm_nt(&a, m as usize, &b, n as usize, k as usize);
    let (err, _) = max_rel_err(&got, &want);
    println!("bf16_nt max rel err = {err:.4}");
    assert!(err < 0.02, "bf16 error too large: {err}");
}

#[test]
fn e2e_mqa_logits_correct() {
    let ctx = DgContext::new(0).expect("need a CUDA device");
    let (num_dpus, max_tokens, h) = (8u32, 64u32, 576u32);
    let q: Vec<u16> = (0..num_dpus * h)
        .map(|i| rf::f32_to_bf16((i % 17) as f32 / 17.0 - 0.5))
        .collect();
    let k: Vec<u16> = (0..num_dpus * max_tokens * h)
        .map(|i| rf::f32_to_bf16((i % 19) as f32 / 19.0 - 0.5))
        .collect();

    let out = mqa_logits_bf16(&ctx, &q, &k, None, num_dpus, max_tokens, h).unwrap();
    let got = download(&ctx, &out).unwrap();

    // Reference
    let mut worst = 0f32;
    for d in 0..num_dpus as usize {
        for t in 0..max_tokens as usize {
            let mut acc = 0f32;
            for hh in 0..h as usize {
                acc += rf::bf16_to_f32(q[d * h as usize + hh])
                    * rf::bf16_to_f32(k[(d * max_tokens as usize + t) * h as usize + hh]);
            }
            let g = rf::bf16_to_f32(got[d * max_tokens as usize + t]);
            worst = worst.max((g - acc).abs() / acc.abs().max(1.0));
        }
    }
    println!("mqa_logits max rel err = {worst:.4}");
    assert!(worst < 0.02, "mqa error too large: {worst}");
}

// ===========================================================================
// SM100 (Blackwell): end-to-end MXFP4 / FP8 / BF16 — gated on sm_100a
// ===========================================================================

fn is_blackwell(ctx: &DgContext) -> bool {
    ctx.arch.is_blackwell()
}

fn bf16_bits(v: f32) -> u16 {
    ((v.to_bits() + 0x7FFF + ((v.to_bits() >> 16) & 1)) >> 16) as u16
}
fn bf16_from_bits(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

#[test]
#[cfg(feature = "e2e")]
fn sm100_fp4_gemm_nt_native() {
    let ctx = DgContext::new(0).expect("cuda");
    if !is_blackwell(&ctx) {
        eprintln!("skipping: needs Blackwell");
        return;
    }
    let (m, n, k) = (256u32, 256u32, 512u32);
    let mut a = Vec::with_capacity((m * k) as usize);
    let mut b = Vec::with_capacity((n * k) as usize);
    let mut s = 12345u64;
    let mut rnd = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 33) as f32 / u32::MAX as f32 - 0.5) * 2.0
    };
    for _ in 0..m * k {
        a.push(bf16_bits(rnd()));
    }
    for _ in 0..n * k {
        b.push(bf16_bits(rnd()));
    }

    let out = deepgemm::ops_sm100::fp4_gemm_nt_native(&ctx, &a, &b, m, n, k).expect("fp4 gemm");

    // CPU reference over the bf16-rounded inputs
    let af: Vec<Vec<f32>> = a
        .chunks(k as usize)
        .map(|r| r.iter().map(|&x| bf16_from_bits(x)).collect())
        .collect();
    let bf: Vec<Vec<f32>> = b
        .chunks(k as usize)
        .map(|r| r.iter().map(|&x| bf16_from_bits(x)).collect())
        .collect();
    let mut a_pack = Vec::new();
    let mut a_sfs = Vec::new();
    for r in &af {
        let (p, s) = rf::quantize_bf16_row_to_mxfp4(r);
        a_pack.extend(p);
        a_sfs.push(s);
    }
    let mut b_pack = Vec::new();
    let mut b_sfs = Vec::new();
    for r in &bf {
        let (p, s) = rf::quantize_bf16_row_to_mxfp4(r);
        b_pack.extend(p);
        b_sfs.push(s);
    }
    let pack_sf = |sfs: &[Vec<u8>], rows: usize| -> Vec<u32> {
        let aligned = rows.div_ceil(4) * 4;
        let cols = k as usize / 128;
        let mut out = vec![0u32; cols * aligned];
        for (r, s) in sfs.iter().enumerate() {
            for (j, &e) in s.iter().enumerate() {
                out[(j / 4) * aligned + r] |= (e as u32) << ((j % 4) * 8);
            }
        }
        out
    };
    let want = rf::mxfp4_gemm_reference(
        &a_pack,
        &pack_sf(&a_sfs, m as usize),
        m as usize,
        &b_pack,
        &pack_sf(&b_sfs, n as usize),
        n as usize,
        k as usize,
    );

    let mut max_err = 0.0f32;
    for i in 0..m as usize {
        for j in 0..n as usize {
            let got = bf16_from_bits(out[i * n as usize + j]);
            let err = (got - want[i * n as usize + j]).abs();
            // e2m1 quantization error dominates: allow a per-element bound
            max_err = max_err.max(err);
        }
    }
    // With amax ~1 and sf=1, per-element error bound ~ k * 0.25 * 0.25 (very loose)
    let bound = (k as f32) * 0.125;
    assert!(max_err < bound, "fp4 gemm max err {max_err} >= {bound}");
    eprintln!("sm100 fp4 gemm: max err {max_err:.4} (bound {bound})");
}

#[test]
#[cfg(feature = "e2e")]
fn sm100_bf16_gemm_nt() {
    let ctx = DgContext::new(0).expect("cuda");
    if !is_blackwell(&ctx) {
        eprintln!("skipping: needs Blackwell");
        return;
    }
    let (m, n, k) = (128u32, 128u32, 128u32);
    let mut a = vec![0u16; (m * k) as usize];
    let mut b = vec![0u16; (n * k) as usize];
    for i in 0..m * k {
        a[i as usize] = bf16_bits(((i % 7) as f32 - 3.0) / 8.0);
    }
    for i in 0..n * k {
        b[i as usize] = bf16_bits(((i % 5) as f32 - 2.0) / 8.0);
    }
    let a_dev = upload(&ctx, &a).unwrap();
    let b_dev = upload(&ctx, &b).unwrap();
    let mut out = unsafe { ctx.stream.alloc::<u16>((m * n) as usize).unwrap() };
    deepgemm::ops_sm100::bf16_gemm_dev(
        &ctx,
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
    .expect("bf16 gemm");
    let got = download(&ctx, &out).unwrap();
    for i in 0..m as usize {
        for j in 0..n as usize {
            let mut want = 0.0f32;
            for kk in 0..k as usize {
                want +=
                    bf16_from_bits(a[i * k as usize + kk]) * bf16_from_bits(b[j * k as usize + kk]);
            }
            let err = (bf16_from_bits(got[i * n as usize + j]) - want).abs();
            assert!(
                err < 0.5,
                "bf16 ({i},{j}): {} vs {want}",
                bf16_from_bits(got[i * n as usize + j])
            );
        }
    }
    eprintln!("sm100 bf16 gemm ok");
}
