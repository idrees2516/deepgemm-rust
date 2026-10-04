//! Layout / utility kernels: SF transform, FP8 transposes, FP4 unpack,
//! grouped-layout construction (for MoE), and simple reductions.
//!
//! All are simple, portable (SM80+) kernels — no TMA/WGMMA needed.

use super::common::COMMON_HEADER;

/// `transpose_fp32`: transform scale factors from `(mn, sf_k)` row-major into
/// the MN-major TMA-aligned layout `(sf_k * num_batches, aligned_mn)` with mn
/// contiguous (upstream `get_mn_major_tma_aligned_tensor`).
///
/// Grid: (ceil(mn / 64) * ceil(sf_k / 128), num_batches); block 512 threads.
const TRANSPOSE_FP32: &str = r#"
#define BLOCK_MN 64
#define BLOCK_SF_K 128
#define NUM_THREADS 512

extern "C" __global__ void deepgemm_transpose_fp32(
        const float* __restrict__ src,   // (num_batches, mn, sf_k), sf_k contiguous
        float* __restrict__ dst,         // (num_batches, sf_k, aligned_mn), mn contiguous
        uint32_t mn, uint32_t sf_k, uint32_t aligned_mn, uint32_t num_batches) {
    const uint32_t tile_mn = blockIdx.x % (mn / BLOCK_MN + (mn % BLOCK_MN ? 1 : 0));
    const uint32_t tile_sf_k = blockIdx.x / (mn / BLOCK_MN + (mn % BLOCK_MN ? 1 : 0));
    const uint32_t batch = blockIdx.y;

    const uint32_t mn_base = tile_mn * BLOCK_MN;
    const uint32_t sf_k_base = tile_sf_k * BLOCK_SF_K;

    __shared__ float tile[BLOCK_MN][BLOCK_SF_K + 1];  // +1 avoids bank conflicts

    // Cooperative staged load: (BLOCK_MN, BLOCK_SF_K) transposed tile
    for (uint32_t i = threadIdx.x; i < BLOCK_MN * BLOCK_SF_K; i += NUM_THREADS) {
        const uint32_t r = i / BLOCK_SF_K, c = i % BLOCK_SF_K;
        const uint32_t g_r = mn_base + r, g_c = sf_k_base + c;
        float v = 0.0f;
        if (g_r < mn && g_c < sf_k)
            v = src[(batch * (uint64_t)mn + g_r) * sf_k + g_c];
        tile[r][c] = v;
    }
    __syncthreads();

    // Store to (sf_k, aligned_mn) layout: dst[batch][g_c][g_r]
    for (uint32_t i = threadIdx.x; i < BLOCK_MN * BLOCK_SF_K; i += NUM_THREADS) {
        const uint32_t c = i / BLOCK_MN, r = i % BLOCK_MN;
        const uint32_t g_r = mn_base + r, g_c = sf_k_base + c;
        if (g_r < mn && g_c < sf_k)
            dst[((uint64_t)batch * sf_k + g_c) * aligned_mn + g_r] = tile[r][c];
    }
}
"#;

/// Transpose an FP8 (rows, cols) row-major matrix into (cols, rows) row-major
/// (used for `nn/tt/tn` layout adapters). Tiles through shared memory.
const TRANSPOSE_FP8: &str = r#"
#define T_TILE 32
#define T_THREADS 256

extern "C" __global__ void deepgemm_transpose_fp8(
        const uint8_t* __restrict__ src,  // (rows, ld_src)
        uint8_t* __restrict__ dst,        // (cols, ld_dst)
        uint32_t rows, uint32_t cols, uint32_t ld_src, uint32_t ld_dst) {
    __shared__ uint8_t tile[T_TILE][T_TILE + 1];
    const uint32_t r0 = blockIdx.y * T_TILE, c0 = blockIdx.x * T_TILE;
    const uint32_t lr = threadIdx.x / T_TILE, lc = threadIdx.x % T_TILE;

    const uint32_t gr = r0 + lr, gc = c0 + lc;
    if (gr < rows && gc < cols)
        tile[lr][lc] = src[(uint64_t)gr * ld_src + gc];
    __syncthreads();
    const uint32_t gr2 = r0 + lc, gc2 = c0 + lr;
    if (gr2 < rows && gc2 < cols)
        dst[(uint64_t)gc2 * ld_dst + gr2] = tile[lc][lr];
}
"#;

/// Unpack packed-FP4 (e2m1, two values per byte, low nibble first) into FP8
/// e4m3 bytes. Used to route FP4 GEMMs through the FP8 kernel on Hopper.
/// FP4 exponent/mantissa (e2m1) -> FP8 (e4m3) is exact for all 16 values.
const UNPACK_FP4: &str = r#"
extern "C" __global__ void deepgemm_unpack_fp4(
        const uint8_t* __restrict__ src,  // packed fp4
        uint8_t* __restrict__ dst,        // fp8 e4m3 out, len = 2 * packed_len
        uint32_t packed_len) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= packed_len) return;
    const uint8_t byte = src[i];

    // e2m1: [s][ee][m] -> e4m3: [s][eeee][mmm]; bias 1 -> bias 7
    uint8_t lo = byte & 0xF, hi = byte >> 4;
    uint8_t out_lo, out_hi;

    // lo nibble
    {
        uint8_t sign = (lo & 0x8) << 4;
        uint8_t exp_m = lo & 0x7;
        uint8_t e = exp_m >> 1, m = exp_m & 1;
        uint8_t f8;
        if (e == 0)
            f8 = m == 0 ? 0x00 : 0x08;          // +-0, +-0.5
        else if (e == 3)
            f8 = m == 0 ? 0x70 : 0x78;          // +-6, +-8 (e4m3 max finite = 448)
        else
            f8 = ((e + 6) << 3) | (m << 2);
        out_lo = sign | f8;
    }
    // hi nibble
    {
        uint8_t sign = (hi & 0x8) << 4;
        uint8_t exp_m = hi & 0x7;
        uint8_t e = exp_m >> 1, m = exp_m & 1;
        uint8_t f8;
        if (e == 0)
            f8 = m == 0 ? 0x00 : 0x08;
        else if (e == 3)
            f8 = m == 0 ? 0x70 : 0x78;
        else
            f8 = ((e + 6) << 3) | (m << 2);
        out_hi = sign | f8;
    }
    dst[2 * (uint64_t)i] = out_lo;
    dst[2 * (uint64_t)i + 1] = out_hi;
}
"#;

/// Build `m_indices` for contiguous grouped GEMM from per-token expert ids.
/// Output: `m_indices[i] = expert of token i` (rows must already be grouped
/// and padded to the 128-token alignment; padding rows get -1).
const BUILD_M_INDICES: &str = r#"
extern "C" __global__ void deepgemm_build_m_indices(
        const int* __restrict__ expert_ids_per_token, // (num_tokens)
        int* __restrict__ m_indices,                  // (padded_m) out
        uint32_t num_tokens, uint32_t padded_m) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= padded_m) return;
    m_indices[i] = i < num_tokens ? expert_ids_per_token[i] : -1;
}
"#;

/// Count tokens per expert (atomic histogram) — MoE dispatch step 1.
const EXPERT_HISTOGRAM: &str = r#"
extern "C" __global__ void deepgemm_expert_histogram(
        const int* __restrict__ expert_ids,  // (num_tokens)
        int* __restrict__ counts,            // (num_experts) zero-initialized
        uint32_t num_tokens) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_tokens) return;
    atomicAdd(&counts[expert_ids[i]], 1);
}
"#;

/// Permute tokens into contiguous grouped layout + compute aligned offsets.
/// Step 2 of MoE dispatch: `output_pos[token]` = destination row.
const PERMUTE_OFFSETS: &str = r#"
extern "C" __global__ void deepgemm_permute_offsets(
        const int* __restrict__ counts,        // (num_experts) raw counts
        int* __restrict__ offsets,             // (num_experts + 1) out: aligned starts
        uint32_t num_experts, uint32_t alignment) {
    // Single block, single thread: num_experts is small (<= 512)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int acc = 0;
        offsets[0] = 0;
        for (uint32_t e = 0; e < num_experts; ++e) {
            const int c = counts[e];
            const int aligned = (c + (int)alignment - 1) / (int)alignment * (int)alignment;
            acc += aligned;
            offsets[e + 1] = acc;
        }
    }
}
"#;

/// Gather token rows (fp8 bytes) into the permuted contiguous buffer.
const GATHER_ROWS_FP8: &str = r#"
extern "C" __global__ void deepgemm_gather_rows_fp8(
        const uint8_t* __restrict__ src,   // (num_tokens, k)
        uint8_t* __restrict__ dst,         // (padded_m, k)
        const int* __restrict__ dst_pos,   // (num_tokens)
        uint32_t num_tokens, uint32_t k) {
    const uint64_t i = (uint64_t)(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)num_tokens * k;
    if (i >= total) return;
    const uint32_t token = i / k, col = i % k;
    const int pos = dst_pos[token];
    if (pos >= 0)
        dst[(uint64_t)pos * k + col] = src[(uint64_t)token * k + col];
}
"#;

/// Scatter-add bf16 rows into an f32 accumulator with weights (MoE combine,
/// step 1/2 — race-free via f32 atomics).
const SCATTER_ROWS_F32_ACC: &str = r#"
extern "C" __global__ void deepgemm_scatter_rows_f32_acc(
        const uint16_t* __restrict__ src,   // (num_src_rows, n) bf16
        float* __restrict__ dst,            // (num_tokens, n) f32, pre-zeroed
        const int* __restrict__ dst_pos,    // (num_src_rows) target token per src row
        const float* __restrict__ weights,  // (num_src_rows)
        uint32_t num_src_rows, uint32_t n) {
    const uint64_t i = (uint64_t)(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)num_src_rows * n;
    if (i >= total) return;
    const uint32_t row = i / n, col = i % n;
    const int pos = dst_pos[row];
    const float w = weights[row];
    const float v = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&src[(uint64_t)row * n + col]));
    atomicAdd(&dst[(uint64_t)pos * n + col], w * v);
}
"#;

/// Convert an f32 buffer to bf16 in place style (dst <- src).
const F32_TO_BF16: &str = r#"
extern "C" __global__ void deepgemm_f32_to_bf16(
        const float* __restrict__ src, uint16_t* __restrict__ dst, uint64_t n) {
    const uint64_t i = (uint64_t)(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2bfloat16_rn(src[i]).x;
}
"#;

/// SiLU-and-mul activation for MoE gate: out[r,c] = silu(gate[r,c]) * up[r,c],
/// where gate_up = (m, 2n) concatenates gate = [:, :n] and up = [:, n:].
const SILU_MUL_BF16: &str = r#"
extern "C" __global__ void deepgemm_silu_mul_bf16(
        const uint16_t* __restrict__ gate_up,  // (m, 2n) bf16
        uint16_t* __restrict__ out,            // (m, n) bf16
        uint32_t n, uint64_t total) {          // total = m * n
    const uint64_t i = (uint64_t)(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const uint64_t row = i / n, col = i % n;
    const uint64_t base = row * (uint64_t)(2 * n);
    const float g = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&gate_up[base + col]));
    const float u = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&gate_up[base + n + col]));
    const float silu = g / (1.0f + __expf(-g));
    *reinterpret_cast<__nv_bfloat16*>(&out[i]) = __float2bfloat16_rn(silu * u);
}
"#;

/// Fill a buffer with zeros (bf16 elements).
const FILL_ZERO_U16: &str = r#"
extern "C" __global__ void deepgemm_fill_zero_u16(uint16_t* dst, uint64_t n) {
    const uint64_t i = (uint64_t)(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = 0;
}
"#;

/// Fill a u32-sized buffer with zeros (i32 / f32 / u32).
const FILL_ZERO_U32: &str = r#"
extern "C" __global__ void deepgemm_fill_zero_u32(uint32_t* dst, uint64_t n) {
    const uint64_t i = (uint64_t)(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = 0;
}
"#;

/// Fill an i32 buffer with an arbitrary value.
const FILL_I32: &str = r#"
extern "C" __global__ void deepgemm_fill_i32(int* dst, uint64_t n, int value) {
    const uint64_t i = (uint64_t)(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = value;
}
"#;

/// Assign each token its destination row inside the contiguous grouped
/// buffer: dst_pos[t] = offsets[e] + atomicAdd(&slots[e], 1).
const ASSIGN_DST_POS: &str = r#"
extern "C" __global__ void deepgemm_assign_dst_pos(
        const int* __restrict__ expert_ids,   // (num_tokens)
        const int* __restrict__ offsets,      // (num_experts + 1) aligned starts
        int* __restrict__ slots,              // (num_experts) zero-initialized
        int* __restrict__ dst_pos,            // (num_tokens) out
        uint32_t num_tokens) {
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_tokens) return;
    const int e = expert_ids[t];
    const int slot = atomicAdd(&slots[e], 1);
    dst_pos[t] = offsets[e] + slot;
}
"#;

/// Build m_indices for the contiguous grouped GEMM from dst_pos:
/// m_indices[row] = expert of the token placed at `row`, or -1 for padding.
const BUILD_M_INDICES_FROM_POS: &str = r#"
extern "C" __global__ void deepgemm_build_m_indices_from_pos(
        const int* __restrict__ dst_pos,   // (num_tokens)
        const int* __restrict__ expert_ids,// (num_tokens)
        int* __restrict__ m_indices,       // (padded_m) out
        uint32_t num_tokens, uint32_t padded_m) {
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_tokens) return;
    const int pos = dst_pos[t];
    if (pos >= 0 && (uint32_t)pos < padded_m)
        m_indices[pos] = expert_ids[t];
}
"#;

/// All layout/utility kernels in one translation unit (compiled once).
pub fn build_layout_kernel_source() -> String {
    // bf16 intrinsics used by scatter/silu kernels: NVRTC provides
    // __nv_bfloat16 via cuda_bf16.h; declare the minimal intrinsics instead
    // to stay header-free.
    let bf16_helpers: &str = r#"
struct __nv_bfloat16 { unsigned short x; };
DG_INLINE float __bfloat162float(const __nv_bfloat16 v) {
    uint32_t u = ((uint32_t)v.x) << 16;
    return __uint_as_float(u);
}
DG_INLINE __nv_bfloat16 __float2bfloat16_rn(float v) {
    unsigned short r;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(v));
    __nv_bfloat16 h; h.x = r; return h;
}
"#;
    format!("{COMMON_HEADER}\n{bf16_helpers}\n{TRANSPOSE_FP32}\n{TRANSPOSE_FP8}\n{UNPACK_FP4}\n{BUILD_M_INDICES}\n{EXPERT_HISTOGRAM}\n{PERMUTE_OFFSETS}\n{GATHER_ROWS_FP8}\n{SCATTER_ROWS_F32_ACC}\n{F32_TO_BF16}\n{SILU_MUL_BF16}\n{FILL_ZERO_U16}\n{FILL_ZERO_U32}\n{FILL_I32}\n{ASSIGN_DST_POS}\n{BUILD_M_INDICES_FROM_POS}\n")
}
