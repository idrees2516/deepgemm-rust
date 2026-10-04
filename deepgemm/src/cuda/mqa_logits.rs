//! MQA logits kernel (DeepEP-style): per-DPU single-query attention logits.
//!
//! `q` is `(num_dpus, H)`; `k` is `(num_dpus, max_tokens_per_dpu, H)`;
//! `out` is `(num_dpus, max_tokens_per_dpu)`. Each DPU's single query is
//! dotted against all of its KV tokens. Bandwidth-bound: vectorized loads +
//! warp/block reduction, zero CPU synchronization (per-dpu valid token counts
//! are read on device).

use super::common::COMMON_HEADER;

const MQA_LOGITS: &str = r#"
#define MQA_THREADS 256
#define MQA_TOKENS_PER_BLOCK 4

struct __nv_bfloat16 { unsigned short x; };
DG_INLINE float bf16_to_f32(unsigned short v) {
    return __uint_as_float(((uint32_t)v) << 16);
}
DG_INLINE unsigned short f32_to_bf16(float v) {
    unsigned short r;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(v));
    return r;
}

extern "C" __global__ void __launch_bounds__(MQA_THREADS) deepgemm_mqa_logits_bf16(
        const unsigned short* __restrict__ q,   // (num_dpus, H)
        const unsigned short* __restrict__ k,   // (num_dpus, max_tokens, H)
        const int* __restrict__ num_valid,      // (num_dpus) or null
        unsigned short* __restrict__ out,       // (num_dpus, max_tokens) bf16
        uint32_t num_dpus, uint32_t max_tokens, uint32_t H) {
    // One block handles MQA_TOKENS_PER_BLOCK tokens of one dpu.
    const uint32_t dpu = blockIdx.y;
    if (dpu >= num_dpus) return;
    const uint32_t token_base = blockIdx.x * MQA_TOKENS_PER_BLOCK;

    const uint32_t valid = num_valid ? (uint32_t)num_valid[dpu] : max_tokens;
    const unsigned short* q_row = q + (uint64_t)dpu * H;
    const unsigned short* k_base = k + (uint64_t)dpu * max_tokens * H;

    // Each thread strides over H; thread-local partial dot products.
    float partial[MQA_TOKENS_PER_BLOCK];
    #pragma unroll
    for (uint32_t t = 0; t < MQA_TOKENS_PER_BLOCK; ++t)
        partial[t] = 0.0f;

    for (uint32_t h = threadIdx.x; h < H; h += MQA_THREADS) {
        const float qv = bf16_to_f32(q_row[h]);
        #pragma unroll
        for (uint32_t t = 0; t < MQA_TOKENS_PER_BLOCK; ++t) {
            const uint32_t token = token_base + t;
            if (token < max_tokens)
                partial[t] += qv * bf16_to_f32(k_base[(uint64_t)token * H + h]);
        }
    }

    // Block reduction (warp shuffle + smem)
    __shared__ float red[MQA_TOKENS_PER_BLOCK][MQA_THREADS / 32];
    #pragma unroll
    for (uint32_t t = 0; t < MQA_TOKENS_PER_BLOCK; ++t) {
        #pragma unroll
        for (uint32_t off = 16; off > 0; off >>= 1)
            partial[t] += __shfl_down_sync(0xffffffff, partial[t], off);
        if (threadIdx.x % 32 == 0)
            red[t][threadIdx.x / 32] = partial[t];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        #pragma unroll
        for (uint32_t t = 0; t < MQA_TOKENS_PER_BLOCK; ++t) {
            float sum = 0.0f;
            #pragma unroll
            for (uint32_t w = 0; w < MQA_THREADS / 32; ++w)
                sum += red[t][w];
            const uint32_t token = token_base + t;
            if (token < max_tokens)
                // Invalid (padding) tokens get -inf bf16 (0xFF80) so softmax ignores them.
                out[(uint64_t)dpu * max_tokens + token] =
                    token < valid ? f32_to_bf16(sum) : (unsigned short)0xFF80;
        }
    }
}
"#;

pub fn build_mqa_logits_source() -> String {
    format!("{COMMON_HEADER}\n{MQA_LOGITS}\n")
}
