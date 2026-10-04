// ===========================================================================
// SM100: heuristics + FP4 reference codecs (GPU-free)
// ===========================================================================

#[test]
fn sm100_heuristics_normal_gemm() {
    use deepgemm::heuristics_sm100::{best_config, Sm100Desc};
    use deepgemm::types::GemmType;
    let desc = Sm100Desc {
        gemm_type: GemmType::Normal,
        use_psum_layout: false,
        m: 4096,
        n: 7168,
        k: 7168,
        num_groups: 1,
        expected_m: 4096,
        expected_num_groups: 1,
        a_bits: 4,
        b_bits: 4,
        major_a_mn: false,
        major_b_mn: false,
        cd_dtype: 0,
        with_accumulation: false,
        num_sms: 148,
        tc_util: 100,
        k_grouped: false,
    };
    let cfg = best_config(&desc);
    assert_eq!(cfg.layout.block_k, 256, "MXF4 block K");
    assert!(cfg.num_stages >= 2, "need a deep pipeline");
    assert!(cfg.smem_size <= deepgemm::heuristics_sm100::SM100_SMEM_CAPACITY);

    // FP8
    let desc8 = Sm100Desc {
        a_bits: 8,
        b_bits: 8,
        ..desc
    };
    let cfg8 = best_config(&desc8);
    assert_eq!(cfg8.layout.block_k, 128);
}

#[test]
fn sm100_heuristics_grouped_swap_ab() {
    use deepgemm::heuristics_sm100::{best_config, Sm100Desc};
    use deepgemm::types::GemmType;
    let desc = Sm100Desc {
        gemm_type: GemmType::MGroupedContiguous,
        use_psum_layout: true,
        m: 8192,
        n: 7168,
        k: 7168,
        num_groups: 256,
        expected_m: 8192,
        expected_num_groups: 1,
        a_bits: 8,
        b_bits: 8,
        major_a_mn: false,
        major_b_mn: false,
        cd_dtype: 0,
        with_accumulation: false,
        num_sms: 148,
        tc_util: 100,
        k_grouped: false,
    };
    let cfg = best_config(&desc);
    assert!(cfg.layout.swap_ab, "m-grouped must swap AB");
    assert_eq!(cfg.layout.block_n, 128);
    assert_eq!(cfg.layout.block_m, 128);
}

#[test]
fn fp4_reference_codecs_roundtrip() {
    use deepgemm::reference::*;
    // every representable value round-trips exactly
    for (code, &v) in E2M1_VALUES.iter().enumerate() {
        assert_eq!(encode_e2m1(v) as usize, code, "code {code} ({v})");
    }
    // ties round to even codes
    assert_eq!(encode_e2m1(0.25), 0, "0.25 -> 0 (even)");
    assert_eq!(encode_e2m1(0.75), 2, "0.75 -> 1.0 (even code 2)");
    assert_eq!(encode_e2m1(5.0), 6, "5.0 ties 4.0/6.0 -> code 6 (even)");
    assert_eq!(encode_e2m1(100.0), 7, "saturates at code 7 (= 6.0)");
    // ue8m0
    assert_eq!(f32_to_ue8m0(1.0), 127);
    assert_eq!(ue8m0_to_f32(127), 1.0);
    assert_eq!(ue8m0_to_f32(130), 8.0);
}

#[test]
fn mxfp4_reference_gemm_matches_float() {
    use deepgemm::reference::*;
    let (m, n, k) = (8usize, 8usize, 128usize);
    let mut a_rows = Vec::new();
    let mut b_rows = Vec::new();
    for i in 0..m {
        a_rows.push(
            (0..k)
                .map(|j| (((i * 13 + j * 7) % 17) as f32 - 8.0) / 8.0)
                .collect::<Vec<_>>(),
        );
    }
    for i in 0..n {
        b_rows.push(
            (0..k)
                .map(|j| (((i * 11 + j * 5) % 19) as f32 - 9.0) / 9.0)
                .collect::<Vec<_>>(),
        );
    }
    let mut a_pack = Vec::new();
    let mut a_sfs = Vec::new();
    for r in &a_rows {
        let (p, s) = quantize_bf16_row_to_mxfp4(r);
        a_pack.extend(p);
        a_sfs.push(s);
    }
    let mut b_pack = Vec::new();
    let mut b_sfs = Vec::new();
    for r in &b_rows {
        let (p, s) = quantize_bf16_row_to_mxfp4(r);
        b_pack.extend(p);
        b_sfs.push(s);
    }
    // build packed SF words in the 1d1d layout
    let pack_sf = |sfs: &[Vec<u8>], rows: usize| -> Vec<u32> {
        let aligned = rows.div_ceil(4) * 4;
        let cols = k.div_ceil(128);
        let mut out = vec![0u32; cols * aligned];
        for (r, s) in sfs.iter().enumerate() {
            for (j, &e) in s.iter().enumerate() {
                out[(j / 4) * aligned + r] |= (e as u32) << ((j % 4) * 8);
            }
        }
        out
    };
    let a_sf = pack_sf(&a_sfs, m);
    let b_sf = pack_sf(&b_sfs, n);
    let got = mxfp4_gemm_reference(&a_pack, &a_sf, m, &b_pack, &b_sf, n, k);
    // compare against direct fp32 quantized math
    let dequant = |_row: &[f32], packed: &[u8], sfs: &[u8], r: usize| -> Vec<f32> {
        (0..k)
            .map(|kk| {
                let (lo, hi) = decode_e2m1_pair(packed[r * (k / 2) + kk / 2]);
                let v = if kk % 2 == 0 { lo } else { hi };
                v * ue8m0_to_f32(sfs[kk / 32])
            })
            .collect::<Vec<_>>()
    };
    for i in 0..m {
        let aq = dequant(&a_rows[i], &a_pack, &a_sfs[i], i);
        for j in 0..n {
            let bq = dequant(&b_rows[j], &b_pack, &b_sfs[j], j);
            let want: f32 = aq.iter().zip(&bq).map(|(x, y)| x * y).sum();
            let diff = (got[i * n + j] - want).abs();
            assert!(diff < 1e-3, "({i},{j}): {got:?} vs {want:?}");
        }
    }
}

#[test]
fn sm100_mk_alignment_knob_selects_block_m() {
    use deepgemm::heuristics_sm100::{best_config, Sm100Desc};
    use deepgemm::types::{
        get_mk_alignment_for_contiguous_layout, get_theoretical_mk_alignment_for_contiguous_layout,
        set_mk_alignment_for_contiguous_layout, GemmType,
    };
    let desc = Sm100Desc {
        gemm_type: GemmType::MGroupedContiguous,
        use_psum_layout: true,
        m: 8192,
        n: 7168,
        k: 7168,
        num_groups: 256,
        expected_m: 8192,
        expected_num_groups: 1,
        a_bits: 4,
        b_bits: 4,
        major_a_mn: false,
        major_b_mn: false,
        cd_dtype: 0,
        with_accumulation: false,
        num_sms: 148,
        tc_util: 100,
        k_grouped: false,
    };
    set_mk_alignment_for_contiguous_layout(128);
    assert_eq!(best_config(&desc).layout.block_m, 128);
    // SM100's theoretical alignment (UMMA_N=256): block M follows the knob
    set_mk_alignment_for_contiguous_layout(get_theoretical_mk_alignment_for_contiguous_layout(10));
    assert_eq!(get_mk_alignment_for_contiguous_layout(), 256);
    let cfg = best_config(&desc);
    assert_eq!(cfg.layout.block_m, 256);
    assert!(cfg.smem_size <= deepgemm::heuristics_sm100::SM100_SMEM_CAPACITY);
    set_mk_alignment_for_contiguous_layout(128);
}
