//! SM100 quantization / scale-factor layout kernels:
//! * BF16 -> packed e2m1 (FP4) + packed UE8M0 per-32 scale factors (MXFP4),
//! * BF16 -> e4m3 (FP8) + packed UE8M0 per-32/per-128 scale factors,
//! * FP32 (power-of-two) scale factors -> packed UE8M0 1d1d layout,
//!   1-D and 2-D (128x128 tile) sources,
//! * raw FP4 unpack (one e2m1 code per byte, for the mixed fp8 x fp4 MMA).
//!
//! The SF output layout is exactly what the SM100 1d1d kernel's TMA
//! descriptors expect: a `(ceil(k / (4 * gran)) [* num_groups], tma_aligned(mn))`
//! matrix of `u32` words with mn contiguous; word `(col, m)`'s byte `j` is the
//! UE8M0 exponent of the K slice `[(col * 4 + j) * gran, +gran)`.

use super::common::COMMON_HEADER;

pub const SM100_CAST_KERNELS: &str = r#"
// ============================================================================
// deepgemm-rust :: SM100 cast / SF-layout kernels
// ============================================================================

// Eight floats -> 8 packed e2m1 nibbles (4 bytes) using the Blackwell-native
// fused cvt+pack idiom (cvt.rn.satfinite.e2m1x2.f32 into .b8 registers).
DG_INLINE uint32_t dg_f32x8_to_e2m1x8(const float* v) {
    uint32_t bits;
    asm volatile(
        "{\n"
        ".reg .b8 b0, b1, b2, b3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
        "mov.b32 %0, {b0, b1, b2, b3};\n"
        "}\n"
        : "=r"(bits)
        : "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]),
          "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7]));
    return bits;
}

// Two floats -> two e4m3 bytes (one .b16).
DG_INLINE uint32_t dg_f32x2_to_e4m3x2(float lo, float hi) {
    uint16_t r;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(r) : "f"(lo), "f"(hi));
    return static_cast<uint32_t>(r);
}

// FP32 amax -> UE8M0 exponent selecting the power-of-two scale that maps the
// amax into the finite range of the quantized dtype (e4m3 or e2m1).
DG_INLINE uint32_t dg_ue8m0_sf_exp_fp32(bool is_fp8, float amax) {
    const uint32_t mant_bits = 23;
    const uint32_t mant_mask = (1u << mant_bits) - 1;
    const uint32_t quant_max_mantissa = (is_fp8 ? 0x60u : 0x40u) << (mant_bits - 7);
    const uint32_t quant_max_exp = is_fp8 ? 8u : 2u;
    const uint32_t min_sf_exp = is_fp8 ? 105u : 1u;
    const uint32_t bits = amax < 0 ? __float_as_uint(-amax) : __float_as_uint(amax);
    const uint32_t rounded = (bits + mant_mask - quant_max_mantissa) >> mant_bits;
    const uint32_t sf_exp = (rounded < quant_max_exp ? quant_max_exp : rounded) + 127;
    return sf_exp < min_sf_exp ? min_sf_exp : sf_exp;
}
DG_INLINE float dg_sf_inv_f32(uint32_t sf_exp) {
    return __uint_as_float((254u - sf_exp) << 23);
}

// ---------------------------------------------------------------------------
// BF16 -> packed FP4 (e2m1) + packed UE8M0 per-32 SFs (the MXFP4 recipe).
// One thread quantizes one 128-element K slice of one row: 4 SF groups,
// 64 packed bytes, one SF word.
//   src: (rows, ld) bf16, k-major
//   dst: (rows, k/2) e2m1 pairs
//   sf:  (ceil(k/128), sf_stride) u32 words, sf[col * sf_stride + row]
// ---------------------------------------------------------------------------
extern "C" __global__ void deepgemm_cast_bf16_to_fp4_packed_sf(
        const uint16_t* __restrict__ src, uint32_t src_ld,
        uint8_t* __restrict__ dst, uint32_t dst_ld_bytes,
        uint32_t* __restrict__ sf, uint32_t sf_stride,
        uint32_t rows, uint32_t k) {
    const uint32_t col = blockIdx.x;             // 128-element slice index
    const uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows) return;
    if (col * 128 >= k) return;

    float vals[128];
    #pragma unroll 8
    for (uint32_t i = 0; i < 128; ++ i) {
        const uint32_t kk = col * 128 + i;
        vals[i] = kk < k ? __uint_as_float(static_cast<uint32_t>(src[static_cast<size_t>(row) * src_ld + kk]) << 16) : 0.0f;
    }

    uint32_t sf_word = 0;
    #pragma unroll
    for (uint32_t g = 0; g < 4; ++ g) {
        float amax = 0.0f;
        #pragma unroll
        for (uint32_t i = 0; i < 32; ++ i) {
            const float a = vals[g * 32 + i] < 0 ? -vals[g * 32 + i] : vals[g * 32 + i];
            amax = a > amax ? a : amax;
        }
        const uint32_t sf_exp = dg_ue8m0_sf_exp_fp32(false, amax);
        sf_word |= sf_exp << (g * 8);
        const float inv = dg_sf_inv_f32(sf_exp);
        #pragma unroll
        for (uint32_t i = 0; i < 32; ++ i)
            vals[g * 32 + i] *= inv;
    }

    // Store 64 packed bytes (16 words)
    uint32_t* dst_words = reinterpret_cast<uint32_t*>(dst + static_cast<size_t>(row) * dst_ld_bytes + col * 64);
    #pragma unroll
    for (uint32_t w = 0; w < 16; ++ w) {
        float v[8];
        #pragma unroll
        for (uint32_t i = 0; i < 8; ++ i) v[i] = vals[w * 8 + i];
        dst_words[w] = dg_f32x8_to_e2m1x8(v);
    }
    sf[static_cast<size_t>(col) * sf_stride + row] = sf_word;
}

// ---------------------------------------------------------------------------
// BF16 -> FP8 (e4m3) + packed UE8M0 SFs, granularity 32 (MXFP8) or 128
// (DeepSeek recipe). GRAN = 32: one thread = 128 elems / 4 SF groups.
// GRAN = 128: one thread = 512 elems / 4 SF groups.
// ---------------------------------------------------------------------------
template <uint32_t GRAN>
DG_INLINE void cast_bf16_to_fp8_sf(const uint16_t* __restrict__ src, uint32_t src_ld,
                                   uint8_t* __restrict__ dst, uint32_t dst_ld,
                                   uint32_t* __restrict__ sf, uint32_t sf_stride,
                                   uint32_t rows, uint32_t k,
                                   uint32_t row, uint32_t col) {
    constexpr uint32_t kNumElems = 4 * GRAN;
    const uint32_t k_base = col * kNumElems;
    if (k_base >= k) return;

    float vals[kNumElems];
    #pragma unroll 8
    for (uint32_t i = 0; i < kNumElems; ++ i) {
        const uint32_t kk = k_base + i;
        vals[i] = kk < k ? __uint_as_float(static_cast<uint32_t>(src[static_cast<size_t>(row) * src_ld + kk]) << 16) : 0.0f;
    }

    uint32_t sf_word = 0;
    #pragma unroll
    for (uint32_t g = 0; g < 4; ++ g) {
        float amax = 0.0f;
        #pragma unroll
        for (uint32_t i = 0; i < GRAN; ++ i) {
            const float a = vals[g * GRAN + i] < 0 ? -vals[g * GRAN + i] : vals[g * GRAN + i];
            amax = a > amax ? a : amax;
        }
        const uint32_t sf_exp = dg_ue8m0_sf_exp_fp32(true, amax);
        sf_word |= sf_exp << (g * 8);
        const float inv = dg_sf_inv_f32(sf_exp);
        #pragma unroll
        for (uint32_t i = 0; i < GRAN; ++ i)
            vals[g * GRAN + i] *= inv;
    }

    uint16_t* dst_h = reinterpret_cast<uint16_t*>(dst + static_cast<size_t>(row) * dst_ld + k_base);
    #pragma unroll
    for (uint32_t p = 0; p < kNumElems / 2; ++ p)
        dst_h[p] = static_cast<uint16_t>(dg_f32x2_to_e4m3x2(vals[p * 2], vals[p * 2 + 1]));
    sf[static_cast<size_t>(col) * sf_stride + row] = sf_word;
}

extern "C" __global__ void deepgemm_cast_bf16_to_fp8_sf_gran32(
        const uint16_t* __restrict__ src, uint32_t src_ld,
        uint8_t* __restrict__ dst, uint32_t dst_ld,
        uint32_t* __restrict__ sf, uint32_t sf_stride,
        uint32_t rows, uint32_t k) {
    cast_bf16_to_fp8_sf<32>(src, src_ld, dst, dst_ld, sf, sf_stride, rows, k,
                            blockIdx.y * blockDim.y + threadIdx.y, blockIdx.x);
}
extern "C" __global__ void deepgemm_cast_bf16_to_fp8_sf_gran128(
        const uint16_t* __restrict__ src, uint32_t src_ld,
        uint8_t* __restrict__ dst, uint32_t dst_ld,
        uint32_t* __restrict__ sf, uint32_t sf_stride,
        uint32_t rows, uint32_t k) {
    cast_bf16_to_fp8_sf<128>(src, src_ld, dst, dst_ld, sf, sf_stride, rows, k,
                             blockIdx.y * blockDim.y + threadIdx.y, blockIdx.x);
}

// ---------------------------------------------------------------------------
// FP32 (power-of-two) SF -> packed UE8M0, 1-D source:
//   src: (rows, k / gran) f32, k-major, stride src_ld
//   dst: (ceil(k / (4*gran)), sf_stride) u32; byte j of word (col, m) covers
//        K slice [(col*4+j)*gran, +gran)
// ---------------------------------------------------------------------------
extern "C" __global__ void deepgemm_transform_sf1d_packed_ue8m0(
        const float* __restrict__ src, uint32_t src_ld,
        uint32_t* __restrict__ dst, uint32_t dst_stride,
        uint32_t rows, uint32_t k, uint32_t gran) {
    const uint32_t col = blockIdx.x;
    const uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows) return;
    const uint32_t k_cols = (k + gran - 1) / gran;
    if (col * 4 >= k_cols) return;

    uint32_t word = 0;
    #pragma unroll
    for (uint32_t j = 0; j < 4; ++ j) {
        const uint32_t kc = col * 4 + j;
        const uint32_t bits = kc < k_cols
            ? __float_as_uint(src[static_cast<size_t>(row) * src_ld + kc]) : 0x3F800000u;
        word |= ((bits >> 23) & 0xFFu) << (j * 8);   // f32 exponent byte == UE8M0
    }
    dst[static_cast<size_t>(col) * dst_stride + row] = word;
}

// ---------------------------------------------------------------------------
// FP32 (power-of-two) 2-D SF (128 x 128 tiles, the DeepSeek weight recipe)
// -> packed UE8M0 1-D layout (gran 128): every row of a tile shares the SF.
//   src: (ceil(rows/128), k/128) f32, k-major, stride src_ld
// ---------------------------------------------------------------------------
extern "C" __global__ void deepgemm_transform_sf2d_packed_ue8m0(
        const float* __restrict__ src, uint32_t src_ld,
        uint32_t* __restrict__ dst, uint32_t dst_stride,
        uint32_t rows, uint32_t k) {
    const uint32_t col = blockIdx.x;               // word index over K (512 elems)
    const uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows) return;
    const uint32_t k_cols = (k + 127) / 128;
    if (col * 4 >= k_cols) return;

    const uint32_t tile_row = row / 128;
    uint32_t word = 0;
    #pragma unroll
    for (uint32_t j = 0; j < 4; ++ j) {
        const uint32_t kc = col * 4 + j;
        const uint32_t bits = kc < k_cols
            ? __float_as_uint(src[static_cast<size_t>(tile_row) * src_ld + kc]) : 0x3F800000u;
        word |= ((bits >> 23) & 0xFFu) << (j * 8);
    }
    dst[static_cast<size_t>(col) * dst_stride + row] = word;
}



// ---------------------------------------------------------------------------
// FP32 (power-of-two) SF, transposed TMA-aligned input -> packed UE8M0.
//   src: (k / gran, tma_aligned(rows)) f32 (the SM90 `transform_sf` output)
// ---------------------------------------------------------------------------
extern "C" __global__ void deepgemm_transform_sf_t_f32_to_packed_ue8m0(
        const float* __restrict__ src, uint32_t src_stride,
        uint32_t* __restrict__ dst, uint32_t dst_stride,
        uint32_t rows, uint32_t k, uint32_t gran) {
    const uint32_t col = blockIdx.x;
    const uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows) return;
    const uint32_t k_cols = (k + gran - 1) / gran;
    if (col * 4 >= k_cols) return;
    uint32_t word = 0;
    #pragma unroll
    for (uint32_t j = 0; j < 4; ++ j) {
        const uint32_t kc = col * 4 + j;
        const uint32_t bits = kc < k_cols
            ? __float_as_uint(src[static_cast<size_t>(kc) * src_stride + row]) : 0x3F800000u;
        word |= ((bits >> 23) & 0xFFu) << (j * 8);
    }
    dst[static_cast<size_t>(col) * dst_stride + row] = word;
}

// ---------------------------------------------------------------------------
// MoE contiguous -> psum layout: `psum[g]` = one-past-the-last row of group g
// (the `MGroupedContiguousWithPsumLayout` scheduler input).
// ---------------------------------------------------------------------------
extern "C" __global__ void deepgemm_psum_from_m_indices(
        const int* __restrict__ m_indices, uint32_t* __restrict__ psum,
        uint32_t num_groups, uint32_t m) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= m) return;
    const int g = m_indices[i];
    if (g < 0) return;
    atomicMax(&psum[g], static_cast<uint32_t>(i) + 1);
}

// ---------------------------------------------------------------------------
// Raw FP4 unpack: one e2m1 code per byte (low nibble) — the storage format
// the mixed FP8 x FP4 (MXF8F6F4) MMA expects for its FP4 operand.
// ---------------------------------------------------------------------------
extern "C" __global__ void deepgemm_unpack_fp4_raw(
        const uint8_t* __restrict__ src, uint8_t* __restrict__ dst, uint32_t packed_len) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= packed_len) return;
    const uint8_t byte = src[i];
    dst[i * 2] = byte & 0x0F;
    dst[i * 2 + 1] = byte >> 4;
}
"#;

/// Assemble the SM100 cast kernel translation unit.
pub fn build_sm100_cast_source() -> String {
    format!("{COMMON_HEADER}{SM100_CAST_KERNELS}")
}
