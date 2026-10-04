//! CPU reference implementations and quantization helpers — used by the
//! correctness tests and to build inputs without a GPU-side stack.

/// Round-to-nearest-even f32 -> fp8 e4m3 (the `__nv_fp8_e4m3` conversion).
pub fn f32_to_e4m3(v: f32) -> u8 {
    // Handle specials
    if v.is_nan() {
        return 0x7F; // NaN in e4m3 (S.1111.111)
    }
    let sign = if v < 0.0 { 1u8 } else { 0u8 };
    let a = v.abs();

    // e4m3: bias 7, exp bits 3, mantissa 3 bits; max finite = 448
    // value = m.ddd * 2^(e-7) for e in [1,7]; subnormals e=0.
    let mut best = 0u8;
    let mut best_err = f32::INFINITY;
    // brute-force over the 127 non-NaN encodings with the sign bit cleared
    for code in 0u16..0x7F {
        let c = code as u8;
        let val = e4m3_to_f32(c);
        let err = (val - a).abs();
        if err < best_err || (err == best_err && (c & 1) == 0) {
            best_err = err;
            best = c;
        }
    }
    // Saturate to +-448 like __nv_fp8_e4m3 conversions (torch default)
    if best_err > 0.0 && a > 448.0 {
        return sign << 7 | 0x7E; // +-448
    }
    best | (sign << 7)
}

/// Decode an e4m3 byte to f32 (S EEEE MMM, exponent bias 7).
pub fn e4m3_to_f32(code: u8) -> f32 {
    let sign = (code >> 7) & 1;
    let exp = ((code >> 3) & 0xF) as i32; // 4 exponent bits
    let mant = (code & 0x7) as u32;

    let val = if exp == 0 {
        // Subnormal: (m / 8) * 2^-6
        (mant as f32) / 8.0 * 2f32.powi(-6)
    } else if exp == 15 && mant == 7 {
        // 0x7F (and its negation) encodes NaN
        return f32::NAN;
    } else {
        (1.0 + (mant as f32) / 8.0) * 2f32.powi(exp - 7)
    };
    if sign == 1 {
        -val
    } else {
        val
    }
}

/// Quantize a slice of f32 to e4m3 bytes.
pub fn quantize_e4m3(vals: &[f32]) -> Vec<u8> {
    vals.iter().map(|&v| f32_to_e4m3(v)).collect()
}

/// Dequantize e4m3 bytes to f32.
pub fn dequantize_e4m3(bytes: &[u8]) -> Vec<f32> {
    bytes.iter().map(|&b| e4m3_to_f32(b)).collect()
}

/// bf16 <-> f32.
pub fn f32_to_bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    // Round to nearest even on the lower 16 bits.
    let mut b = (bits >> 16) as u16;
    let lsb = (bits & 0xFFFF) + 0x7FFF + ((bits >> 16) & 1);
    if lsb > 0xFFFF {
        b += 1;
    }
    b
}

pub fn bf16_to_f32(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
}

/// Reference FP8 GEMM with fine-grained scaling (1d A scales, 2d B scales):
/// `out[m, n] = sum_k sfa[m, k/128] * sfb[n/128, k/128] * a[m, k] * b[n, k]`.
#[allow(clippy::too_many_arguments)]
pub fn ref_fp8_gemm_nt(
    a: &[u8],
    sfa: &[f32],
    m: usize,
    b: &[u8],
    sfb: &[f32],
    n: usize,
    k: usize,
) -> Vec<f32> {
    let k_blocks = k.div_ceil(128);
    let mut out = vec![0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0f32;
            let mut block_acc = 0f32;
            for ki in 0..k {
                block_acc += e4m3_to_f32(a[mi * k + ki]) * e4m3_to_f32(b[ni * k + ki]);
                if (ki + 1) % 128 == 0 || ki + 1 == k {
                    let kb = ki / 128;
                    let sa = sfa[mi * k_blocks + kb];
                    let sb = sfb[(ni / 128) * k_blocks + kb];
                    acc += block_acc * sa * sb;
                    block_acc = 0.0;
                }
            }
            out[mi * n + ni] = acc;
        }
    }
    out
}

/// Reference bf16 GEMM NT.
pub fn ref_bf16_gemm_nt(a: &[u16], m: usize, b: &[u16], n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0f64;
            for ki in 0..k {
                acc += bf16_to_f32(a[mi * k + ki]) as f64 * bf16_to_f32(b[ni * k + ki]) as f64;
            }
            out[mi * n + ni] = acc as f32;
        }
    }
    out
}

/// Reference contiguous grouped GEMM: applies the same math per valid row.
#[allow(clippy::too_many_arguments)]
pub fn ref_m_grouped_fp8_gemm_nt_contiguous(
    a: &[u8],
    sfa: &[f32],
    m_indices: &[i32],
    m: usize,
    b: &[u8],
    sfb: &[f32],
    n: usize,
    k: usize,
) -> Vec<f32> {
    let k_blocks = k.div_ceil(128);
    let mut out = vec![0f32; m * n];
    for mi in 0..m {
        let g = m_indices[mi];
        if g < 0 {
            continue;
        }
        let g = g as usize;
        for ni in 0..n {
            let mut acc = 0f32;
            let mut block_acc = 0f32;
            for ki in 0..k {
                block_acc += e4m3_to_f32(a[mi * k + ki]) * e4m3_to_f32(b[(g * n + ni) * k + ki]);
                if (ki + 1) % 128 == 0 || ki + 1 == k {
                    let kb = ki / 128;
                    let sa = sfa[mi * k_blocks + kb];
                    let sb = sfb[(ni / 128) * k_blocks + kb];
                    acc += block_acc * sa * sb;
                    block_acc = 0.0;
                }
            }
            out[mi * n + ni] = acc;
        }
    }
    out
}

/// Reference masked grouped GEMM.
#[allow(clippy::too_many_arguments)]
pub fn ref_m_grouped_fp8_gemm_nt_masked(
    a: &[u8],
    sfa: &[f32],
    masked_m: &[i32],
    num_groups: usize,
    m_per_group: usize,
    b: &[u8],
    sfb: &[f32],
    n: usize,
    k: usize,
) -> Vec<f32> {
    let k_blocks = k.div_ceil(128);
    let mut out = vec![0f32; num_groups * m_per_group * n];
    for g in 0..num_groups {
        let valid = masked_m[g] as usize;
        for mi in 0..valid {
            let row = g * m_per_group + mi;
            for ni in 0..n {
                let mut acc = 0f32;
                let mut block_acc = 0f32;
                for ki in 0..k {
                    block_acc +=
                        e4m3_to_f32(a[row * k + ki]) * e4m3_to_f32(b[(g * n + ni) * k + ki]);
                    if (ki + 1) % 128 == 0 || ki + 1 == k {
                        let kb = ki / 128;
                        let sa = sfa[row * k_blocks + kb];
                        let sb = sfb[(ni / 128) * k_blocks + kb];
                        acc += block_acc * sa * sb;
                        block_acc = 0.0;
                    }
                }
                out[row * n + ni] = acc;
            }
        }
    }
    out
}

// ===========================================================================
// FP4 (e2m1) + UE8M0 reference codecs (CPU)
// ===========================================================================

/// The 16 e2m1 values (index = 4-bit code).
pub const E2M1_VALUES: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Decode a packed e2m1 byte pair (low nibble = element 2i, high = 2i+1).
pub fn decode_e2m1_pair(byte: u8) -> (f32, f32) {
    (
        E2M1_VALUES[(byte & 0x0F) as usize],
        E2M1_VALUES[(byte >> 4) as usize],
    )
}

/// Round-to-nearest-even e2m1 encode of a scaled value (matches
/// `cvt.rn.satfinite.e2m1x2.f32` on the representable set).
pub fn encode_e2m1(v: f32) -> u8 {
    let magnitudes = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let a = if v < 0.0 { -v } else { v };
    let a = if a.is_nan() { 0.0 } else { a };
    // saturating at 6.0
    let a = if a > 6.0 { 6.0 } else { a };
    let mut best = 0u8;
    let mut best_err = f32::INFINITY;
    for (i, &m) in magnitudes.iter().enumerate() {
        let err = (a - m).abs();
        // ties -> even code (rn)
        let better = err < best_err || (err == best_err && (i as u32 & 1) == 0);
        if better {
            best_err = err;
            best = i as u8;
        }
    }
    let sign = if v < 0.0 || (v == 0.0 && v.is_sign_negative()) {
        1u8
    } else {
        0u8
    };
    best | (sign << 3)
}

/// UE8M0 scale from a power-of-two f32 (exponent byte, bias 127).
pub fn f32_to_ue8m0(sf: f32) -> u8 {
    (((sf.to_bits() >> 23) & 0xFF) as u8).max(1)
}

/// UE8M0 scale to f32 (2^(e-127)); the UE8M0 byte IS the biased exponent.
pub fn ue8m0_to_f32(e: u8) -> f32 {
    f32::from_bits((e as u32) << 23)
}

/// Reference MXFP4 quantization of a bf16 row slice:
/// returns (packed bytes, ue8m0 exponents per 32-element group).
pub fn quantize_bf16_row_to_mxfp4(row: &[f32]) -> (Vec<u8>, Vec<u8>) {
    assert!(row.len() % 32 == 0);
    let mut packed = Vec::with_capacity(row.len() / 2);
    let mut sfs = Vec::with_capacity(row.len() / 32);
    for g in row.chunks(32) {
        let amax = g.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        // e2m1 finite max = 6
        let exp = ((amax.to_bits() + (1 << 23) - 1 - (0x40u32 << 16)) >> 23).max(1 + 127);
        let exp = exp.clamp(1, 255);
        let sf = f32::from_bits((exp) << 23);
        let inv = 1.0 / sf;
        sfs.push(exp as u8);
        for pair in g.chunks(2) {
            let lo = encode_e2m1(pair[0] * inv);
            let hi = encode_e2m1(pair[1] * inv);
            packed.push(lo | (hi << 4));
        }
    }
    (packed, sfs)
}

/// Reference MXFP4 x MXFP4 GEMM from quantized operands.
/// * `a`/`b`: packed e2m1 rows (`(rows, k/2)` bytes)
/// * `a_sf`/`b_sf`: packed UE8M0 words in the 1d1d layout
///   (`(ceil(k/128), aligned_rows)` u32, byte j of word (col, r) covers
///   K slice `[(col*4+j)*32, +32)`).
#[allow(clippy::too_many_arguments)]
pub fn mxfp4_gemm_reference(
    a_packed: &[u8],
    a_sf: &[u32],
    m: usize,
    b_packed: &[u8],
    b_sf: &[u32],
    n: usize,
    k: usize,
) -> Vec<f32> {
    let sf_col = |sf: &[u32], rows: usize, r: usize, kk: usize| -> f32 {
        let word_col = kk / 128;
        let byte = (kk % 128) / 32;
        let aligned = rows.div_ceil(4) * 4;
        let w = sf[word_col * aligned + r];
        ue8m0_to_f32(((w >> (byte * 8)) & 0xFF) as u8)
    };
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for kk in (0..k).step_by(2) {
                let (a0, a1) = decode_e2m1_pair(a_packed[i * (k / 2) + kk / 2]);
                let (b0, b1) = decode_e2m1_pair(b_packed[j * (k / 2) + kk / 2]);
                let sfa0 = sf_col(a_sf, m, i, kk);
                let sfb0 = sf_col(b_sf, n, j, kk);
                acc += (a0 as f64) * (sfa0 as f64) * (b0 as f64) * (sfb0 as f64);
                if kk + 1 < k {
                    let sfa1 = sf_col(a_sf, m, i, kk + 1);
                    let sfb1 = sf_col(b_sf, n, j, kk + 1);
                    acc += (a1 as f64) * (sfa1 as f64) * (b1 as f64) * (sfb1 as f64);
                }
            }
            out[i * n + j] = acc as f32;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3_roundtrip_known_values() {
        assert_eq!(e4m3_to_f32(0x00), 0.0); // +-0
        assert_eq!(e4m3_to_f32(0x40), 2.0); // e=8: 2^(8-7)
        assert_eq!(e4m3_to_f32(0x38), 1.0); // e=7: 2^0
        assert_eq!(e4m3_to_f32(0x30), 0.5); // e=6
        assert_eq!(e4m3_to_f32(0x28), 0.25); // e=5
        assert_eq!(e4m3_to_f32(0x01), 2f32.powi(-9)); // smallest subnormal
        assert_eq!(e4m3_to_f32(0x7E), 448.0); // max finite
        assert!(e4m3_to_f32(0x7F).is_nan());
        assert_eq!(e4m3_to_f32(0xC0), -2.0);
    }

    #[test]
    fn quantize_roundtrip() {
        for x in [
            0.0,
            0.5,
            1.0,
            -1.0,
            2.0,
            0.25,
            -0.125,
            448.0,
            -448.0,
            2f32.powi(-9),
        ] {
            let q = f32_to_e4m3(x);
            let d = e4m3_to_f32(q);
            let scale = x.abs().max(1e-6);
            assert!((d - x).abs() / scale < 0.2, "x={x} q={q:#x} d={d}");
        }
    }

    #[test]
    fn bf16_roundtrip() {
        for x in [0.0, 1.0, -2.5, 0.25, 100.0] {
            let b = f32_to_bf16(x);
            let d = bf16_to_f32(b);
            assert!((d - x).abs() / x.abs().max(1e-6) < 0.01, "x={x} d={d}");
        }
    }
}
