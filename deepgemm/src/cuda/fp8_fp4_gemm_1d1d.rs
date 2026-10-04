//! The flagship Blackwell kernel: SM100 (tcgen05) FP8/FP4 GEMM with
//! fine-grained power-of-two (UE8M0) scaling — "1d1d": both A and B carry
//! per-`gran_k`-element scale factors (gran 32 for MXFP4 / MXFP8, or 128 for
//! the DeepSeek-style recipe).
//!
//! Ported 1:1 from upstream DeepGEMM's `sm100_fp8_fp4_gemm_1d1d.cuh`
//! (cutlass/cute primitives replaced by the raw-PTX layer in
//! [`super::sm100_common`]).
//!
//! Covers all GemmTypes (Normal, MGroupedContiguous[+Psum], MGroupedMasked,
//! KGroupedContiguous[+Psum], Batched), swap-AB for skinny M, 2-CTA
//! (cta_group::2) multicast MMAs, TMEM accumulator double-buffering with the
//! SF-column overlap trick, and four epilogue operators (identity, alpha,
//! stochastic-round-to-BF16, quantize-to-FP8 with dynamic output SFs).
//!
//! MMA selection (matches upstream):
//! * FP4xFP4  -> `tcgen05.mma.kind::mxf4`  (packed 4-bit operands, UMMA_K=64)
//! * anything else (FP8xFP8, FP8xFP4) -> `tcgen05.mma.kind::mxf8f6f4`
//!   (byte operands; an FP4 side is stored unpacked, one e2m1 code per byte).

use super::sm100_common::sm100_header;
use super::subst;

/// Per-instantiation configuration (mirrors the upstream template params).
#[derive(Debug, Clone)]
pub struct Fp8Fp4Config {
    pub gemm_type: u32, // 0..=6 (see SM100_SCHED)
    pub major_a: u32,   // 0 = K, 1 = MN
    pub major_b: u32,
    pub gran_k_a: u32, // 32 or 128
    pub gran_k_b: u32,
    pub shape_m: u32, // 0 = runtime
    pub shape_n: u32,
    pub shape_k: u32,
    pub block_m: u32,
    pub block_n: u32,
    pub block_k: u32, // 256 for MXF4, 128 otherwise
    pub num_groups: u32,
    pub swizzle_a: u32, // bytes: 0/32/64/128
    pub swizzle_b: u32,
    pub swizzle_cd: u32,
    pub num_stages: u32,
    pub num_tma_store_stages: u32,     // 1 or 2
    pub num_non_epilogue_threads: u32, // 128
    pub num_epilogue_threads: u32,     // 128
    pub multicast: u32,                // 1 or 2
    pub is_multicast_on_a: bool,
    pub num_sms: u32,
    pub swap_ab: bool,
    pub ensure_zero_padding: bool,
    pub k_alignment: u32,
    pub with_accumulation: bool,
    /// Operand element widths in bits: 8 = e4m3, 4 = e2m1.
    pub a_bits: u32,
    pub b_bits: u32,
    /// 0 = bf16, 1 = fp32, 2 = e4m3 (with dynamic output SFs).
    pub cd_dtype: u32,
    /// 0 = identity, 1 = alpha, 2 = stochastic-round-bf16, 3 = quantize-fp8.
    pub epilogue_op: u32,
}

impl Fp8Fp4Config {
    pub fn is_mxf4(&self) -> bool {
        self.a_bits == 4 && self.b_bits == 4
    }
}

/// The kernel template. `%%VAR%%` placeholders are substituted per config.
pub const FP8FP4_KERNEL: &str = r#"
// ===========================================================================
// deepgemm-rust :: sm100_fp8_fp4_gemm_1d1d (one block configuration)
// Port of upstream DeepGEMM sm100_fp8_fp4_gemm_1d1d.cuh
// ===========================================================================
#define GEMM_TYPE               %%GEMM_TYPE%%
#define MAJOR_A                 %%MAJOR_A%%           // 0 = K, 1 = MN
#define MAJOR_B                 %%MAJOR_B%%
#define GRAN_K_A                %%GRAN_K_A%%          // 32 or 128
#define GRAN_K_B                %%GRAN_K_B%%
#define K_ALIGNMENT             %%K_ALIGNMENT%%
#define SHAPE_M                 %%SHAPE_M%%           // 0 = runtime
#define SHAPE_N                 %%SHAPE_N%%
#define SHAPE_K                 %%SHAPE_K%%
#define BLOCK_M                 %%BLOCK_M%%
#define BLOCK_N                 %%BLOCK_N%%
#define BLOCK_K                 %%BLOCK_K%%           // 256 (MXF4) or 128
#define NUM_GROUPS              %%NUM_GROUPS%%
#define SWIZZLE_A               %%SWIZZLE_A%%         // bytes
#define SWIZZLE_B               %%SWIZZLE_B%%
#define SWIZZLE_CD              %%SWIZZLE_CD%%
#define NUM_STAGES              %%NUM_STAGES%%
#define NUM_TMA_STORE_STAGES    %%NUM_TMA_STORE_STAGES%%
#define NUM_NON_EPILOGUE_THREADS %%NUM_NON_EPILOGUE_THREADS%%
#define NUM_EPILOGUE_THREADS    %%NUM_EPILOGUE_THREADS%%
#define MULTICAST               %%MULTICAST%%         // 1 or 2
#define IS_MULTICAST_ON_A       %%IS_MULTICAST_ON_A%% // 0/1
#define NUM_SMS                 %%NUM_SMS%%
#define SWAP_AB                 %%SWAP_AB%%           // 0/1
#define ENSURE_ZERO_PADDING     %%ENSURE_ZERO_PADDING%%
#define WITH_ACCUMULATION       %%WITH_ACCUMULATION%% // 0/1
#define A_BITS                  %%A_BITS%%            // 8 = e4m3, 4 = e2m1
#define B_BITS                  %%B_BITS%%
#define CD_DTYPE                %%CD_DTYPE%%          // 0 = bf16, 1 = fp32, 2 = e4m3
#define EPILOGUE_OP             %%EPILOGUE_OP%%       // 0/1/2/3
#define NUM_THREADS             (NUM_NON_EPILOGUE_THREADS + NUM_EPILOGUE_THREADS)
#if GEMM_TYPE == 1 || GEMM_TYPE == 5
#define DG_IS_M_GROUPED_CONTIG 1
#else
#define DG_IS_M_GROUPED_CONTIG 0
#endif
#if GEMM_TYPE == 3 || GEMM_TYPE == 6
#define DG_IS_K_GROUPED_CONTIG 1
#else
#define DG_IS_K_GROUPED_CONTIG 0
#endif
#define SFK_SPAN                128u

// ---------- derived constants ----------
#define IS_MXF4                 (A_BITS == 4 && B_BITS == 4)
#define A_PACK                  (A_BITS == 4 && IS_MXF4 ? 2u : 1u)
#define B_PACK                  (B_BITS == 4 && IS_MXF4 ? 2u : 1u)
#define A_ESZ                   1u                    // storage bytes (fp8/fp4-packed/fp4-unpacked)
#define B_ESZ                   1u
#define A_FMT                   (A_BITS == 4 ? DG_UFMT_E2M1 : DG_UFMT_E4M3)
#define B_FMT                   (B_BITS == 4 ? DG_UFMT_E2M1 : DG_UFMT_E4M3)
#define CD_ELEM_SIZE            (CD_DTYPE == 1 ? 4u : (CD_DTYPE == 2 ? 1u : 2u))
#if GEMM_TYPE == DG_GT_BATCHED
#define IS_3D_TMA               true
#else
#define IS_3D_TMA               false
#endif
#define WITH_OUTPUT_SF          (EPILOGUE_OP == 3)
#define WITH_STOCHASTIC         (EPILOGUE_OP == 2)
#define SF_GRAN_N               32u                   // QuantizeToFP8 SF granularity on N

#define LAYOUT_AD_M             128u
#define UMMA_M                  (LAYOUT_AD_M * MULTICAST)
#define UMMA_N                  (SWAP_AB ? BLOCK_M : BLOCK_N)
#define UMMA_K                  (IS_MXF4 ? 64u : 32u)
#define LOAD_BLOCK_M            (BLOCK_M / (IS_MULTICAST_ON_A ? MULTICAST : 1))
#define LOAD_BLOCK_N            (BLOCK_N / (IS_MULTICAST_ON_A ? 1 : MULTICAST))
#define UMMA_A_SIZE_PER_STAGE   (((LOAD_BLOCK_M + LAYOUT_AD_M - 1) / LAYOUT_AD_M * LAYOUT_AD_M) * BLOCK_K * A_ESZ / A_PACK)

#define NUM_UTCCP_ALIGNED_ELEMS 128u
#define SF_BLOCK_M              ((BLOCK_M + NUM_UTCCP_ALIGNED_ELEMS - 1) / NUM_UTCCP_ALIGNED_ELEMS * NUM_UTCCP_ALIGNED_ELEMS)
#define SF_BLOCK_N              ((BLOCK_N + NUM_UTCCP_ALIGNED_ELEMS - 1) / NUM_UTCCP_ALIGNED_ELEMS * NUM_UTCCP_ALIGNED_ELEMS)
#define SF_BLOCK_K              (BLOCK_K / 128)
#define NUM_SFA_STAGES_PER_LOAD (GRAN_K_A == 32 ? 1u : 4u)
#define NUM_SFB_STAGES_PER_LOAD (GRAN_K_B == 32 ? 1u : 4u)

#define NUM_EPILOGUE_STAGES     2u
#define STORE_BLOCK_M           (SWAP_AB ? 16u : (BLOCK_M < LAYOUT_AD_M ? BLOCK_M : LAYOUT_AD_M))
#define STORE_BLOCK_N           (SWAP_AB ? BLOCK_N : (SWIZZLE_CD / CD_ELEM_SIZE))
#define NUM_UMMA_STORE_THREADS  (SWAP_AB ? NUM_EPILOGUE_THREADS : STORE_BLOCK_M)

// Tensor memory layout (columns), with the SF/accumulator overlap trick.
#define NUM_ACCUM_TMEM_COLS     (UMMA_N * NUM_EPILOGUE_STAGES)
#define NUM_SFA_TMEM_COLS       (SF_BLOCK_M * SF_BLOCK_K / 32)
#define NUM_SFB_TMEM_COLS       (SF_BLOCK_N * SF_BLOCK_K / 32)
#define NUM_SF_TMEM_COLS        (NUM_SFA_TMEM_COLS + NUM_SFB_TMEM_COLS)
#define NUM_OVERLAPPED_TMEM_COLS ((NUM_ACCUM_TMEM_COLS + NUM_SF_TMEM_COLS > 512 ? NUM_ACCUM_TMEM_COLS + NUM_SF_TMEM_COLS - 512 : 0))
#define NUM_TMEM_COLS_RAW       (NUM_ACCUM_TMEM_COLS + NUM_SF_TMEM_COLS - NUM_OVERLAPPED_TMEM_COLS)
#define NUM_TMEM_COLS           (NUM_TMEM_COLS_RAW <= 32 ? 32u : (NUM_TMEM_COLS_RAW <= 64 ? 64u : (NUM_TMEM_COLS_RAW <= 128 ? 128u : (NUM_TMEM_COLS_RAW <= 256 ? 256u : 512u))))
#define TMEM_START_COL_SFA      (NUM_ACCUM_TMEM_COLS - NUM_OVERLAPPED_TMEM_COLS)
#define TMEM_START_COL_SFB      (TMEM_START_COL_SFA + NUM_SFA_TMEM_COLS)

#define NUM_1D_BLOCKS_PER_GROUP (dg_num_1d_blocks_per_group())

#define DG_CD_T                 uint8_t
#define DG_ALIGN4(x)            ((x + 3) / 4 * 4)

static_assert(IS_MXF4 ? BLOCK_K == 256 : BLOCK_K == 128, "invalid block K");
static_assert(BLOCK_K % UMMA_K == 0, "block K must be divisible by UMMA K");
static_assert(MULTICAST == 1 || MULTICAST == 2, "only 1/2 multicast");
static_assert(SWAP_AB ? BLOCK_N == LAYOUT_AD_M
                      : (BLOCK_M == 32 || BLOCK_M == 64 || BLOCK_M == LAYOUT_AD_M),
              "invalid block size");
static_assert(!IS_MXF4 || (MAJOR_A == 0 && MAJOR_B == 0), "MXF4 is K-major only");
static_assert(!IS_MXF4 || (GRAN_K_A == 32 && GRAN_K_B == 32), "MXF4 needs gran 32");
static_assert(GRAN_K_A == 32 || GRAN_K_A == 128, "invalid gran A");
static_assert(GRAN_K_B == 32 || GRAN_K_B == 128, "invalid gran B");
static_assert(!DG_IS_K_GROUPED_CONTIG || (GRAN_K_A == GRAN_K_B && (GRAN_K_A == 32 || GRAN_K_A == 128)),
              "k-grouped SF needs matching granularity 32/128");
static_assert(!DG_IS_K_GROUPED_CONTIG || K_ALIGNMENT % BLOCK_K == 0,
              "K alignment must be divisible by block K");
static_assert(NUM_TMA_STORE_STAGES == 1 || NUM_TMA_STORE_STAGES == 2, "invalid store stages");
static_assert(NUM_UMMA_STORE_THREADS % 32 == 0, "invalid store block M");
static_assert(NUM_TMEM_COLS >= 32 && NUM_TMEM_COLS <= 512, "invalid tmem columns");
static_assert(NUM_OVERLAPPED_TMEM_COLS <= UMMA_N, "too many overlapped columns");
static_assert(!SWAP_AB || NUM_OVERLAPPED_TMEM_COLS == 0 || NUM_OVERLAPPED_TMEM_COLS <= STORE_BLOCK_N,
              "non-swap overlapped columns must fit in the first epilogue store");
static_assert(WITH_OUTPUT_SF ? (CD_DTYPE == 2 && GEMM_TYPE == DG_GT_BATCHED && !WITH_ACCUMULATION)
                             : (CD_DTYPE == 0 || CD_DTYPE == 1),
              "invalid C/D dtype");
static_assert(!WITH_STOCHASTIC || (CD_DTYPE == 0 && !WITH_ACCUMULATION),
              "stochastic rounding needs a direct BF16 output");
static_assert(NUM_STAGES <= 32, "too many stages");

// ------------------------------ shared storage ------------------------------
struct SmemStorage {
    // cd tile bytes: STORE_BLOCK_M * STORE_BLOCK_N * cd_elem
    alignas(1024) uint8_t cd[NUM_TMA_STORE_STAGES * DG_ALIGN4(STORE_BLOCK_M * STORE_BLOCK_N * CD_ELEM_SIZE)];
    alignas(1024) uint8_t a[NUM_STAGES * DG_ALIGN4(LOAD_BLOCK_M * BLOCK_K * A_ESZ / A_PACK)];
    alignas(1024) uint8_t b[NUM_STAGES * DG_ALIGN4(LOAD_BLOCK_N * BLOCK_K * B_ESZ / B_PACK)];
    alignas(1024) uint32_t sfa[NUM_STAGES * SF_BLOCK_M * SF_BLOCK_K];
    uint32_t sfb[NUM_STAGES * SF_BLOCK_N * SF_BLOCK_K];
    uint64_t full_barriers[NUM_STAGES];
    uint64_t sf_full_barriers[NUM_STAGES];
    uint64_t empty_barriers[NUM_STAGES];
    uint64_t tmem_full_barriers[NUM_EPILOGUE_STAGES];
    uint64_t tmem_empty_barriers[NUM_EPILOGUE_STAGES];
    uint64_t tmem_overlap_barriers[NUM_EPILOGUE_STAGES];
    uint32_t tmem_ptr;
};
"#;

/// The kernel body (roles: TMA load warp), continued in `_2`/`_3`.
pub const FP8FP4_KERNEL_1: &str = r#"
// ============================== the kernel ==================================
extern "C" __global__ void __launch_bounds__(NUM_THREADS, 1) deepgemm_sm100_fp8_fp4_gemm(
        int* __restrict__ grouped_layout,
        uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
        uint32_t* __restrict__ sfd, uint32_t sfd_stride,
        uint32_t epi_shape_m, uint32_t epi_shape_n, float epi_alpha,
        const __grid_constant__ TmaDescriptor tensor_map_a,
        const __grid_constant__ TmaDescriptor tensor_map_b,
        const __grid_constant__ TmaDescriptor tensor_map_sfa,
        const __grid_constant__ TmaDescriptor tensor_map_sfb,
        const __grid_constant__ TmaDescriptor tensor_map_cd) {
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    SmemStorage& smem = *reinterpret_cast<SmemStorage*>(smem_buffer);

    const bool is_leader_cta = dg_block_rank_in_cluster() == 0;
    const uint32_t warp_idx = dg_warp_idx();
    const uint32_t lane_idx = dg_lane_idx();

    // Overwrite shape constants if the compiler gives them
    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;
    const uint32_t shape_sfa_k = dg_ceil_div(shape_k, GRAN_K_A * 4);
    const uint32_t shape_sfb_k = dg_ceil_div(shape_k, GRAN_K_B * 4);

    // Synchronize the cluster before the 2-CTA TMEM allocation
    if (MULTICAST > 1) dg_cluster_sync_relaxed();

    // Prefetch TMA descriptors
    if (warp_idx == 0) {
        dg_prefetch_tma_descriptor(&tensor_map_a);
        dg_prefetch_tma_descriptor(&tensor_map_b);
        dg_prefetch_tma_descriptor(&tensor_map_sfa);
        dg_prefetch_tma_descriptor(&tensor_map_sfb);
        dg_prefetch_tma_descriptor(&tensor_map_cd);
    }

    // Initialize barriers / allocate tensor memory
    if (warp_idx == 1 && dg_elect_one()) {
        #pragma unroll 1
        for (uint32_t i = 0; i < NUM_STAGES; ++ i) {
            dg_mbarrier_init(&smem.sf_full_barriers[i], 1);
            dg_mbarrier_init(&smem.empty_barriers[i], 1);
            dg_mbarrier_init(&smem.full_barriers[i], MULTICAST * (1 + 32 * SF_BLOCK_K));
        }
        #pragma unroll 1
        for (uint32_t i = 0; i < NUM_EPILOGUE_STAGES; ++ i) {
            dg_mbarrier_init(&smem.tmem_full_barriers[i], 1);
            dg_mbarrier_init(&smem.tmem_empty_barriers[i], MULTICAST * NUM_UMMA_STORE_THREADS);
            dg_mbarrier_init(&smem.tmem_overlap_barriers[i], MULTICAST * NUM_UMMA_STORE_THREADS);
        }
        dg_fence_barrier_init();
    } else if (warp_idx == 2) {
        if (MULTICAST == 1) dg100_tmem_alloc_1sm(NUM_TMEM_COLS, &smem.tmem_ptr);
        else                dg100_tmem_alloc_2sm(NUM_TMEM_COLS, &smem.tmem_ptr);
    }
    if (MULTICAST > 1) dg_cluster_sync_relaxed(); else __syncthreads();

    dg_grid_dependency_sync();

    // Block scheduler
    uint32_t m_block_idx, n_block_idx;
    DgSched scheduler(shape_m, shape_n, shape_k, grouped_layout);

    // Pipeline and TMA phases
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;
        stage_idx = stage_idx == NUM_STAGES - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    // ============================ warp 0: TMA loads ==========================
    if (warp_idx == 0 && dg_elect_one()) {
        const uint32_t tma_bytes_per_stage =
            LOAD_BLOCK_M * BLOCK_K / A_PACK + LOAD_BLOCK_N * BLOCK_K / B_PACK;

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            const uint32_t load_block_m = SWAP_AB
                ? scheduler.get_aligned_effective_m_in_block(m_block_idx) / MULTICAST
                : LOAD_BLOCK_M;

            const uint32_t num_total_k_blocks =
                dg_ceil_div(scheduler.current_shape_k ? scheduler.current_shape_k : 1, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                // Wait for the consumer release
                dg_mbarrier_wait(&smem.empty_barriers[stage_idx], phase ^ 1);

                // Compute offsets (group handling per GemmType, upstream-faithful)
                constexpr bool a_with_group_mn = (GEMM_TYPE == DG_GT_MGROUPED_MASKED);
                uint32_t m_idx = scheduler.get_global_idx<a_with_group_mn, DG_IX_MN>(shape_m, BLOCK_M, m_block_idx);
                constexpr bool b_with_group_mn = (MAJOR_B == 0) &&
                    (DG_IS_M_GROUPED_CONTIG || GEMM_TYPE == DG_GT_MGROUPED_MASKED);
                uint32_t n_idx = scheduler.get_global_idx<b_with_group_mn, DG_IX_MN>(shape_n, BLOCK_N, n_block_idx, m_block_idx);

                constexpr bool a_with_group_k = DG_IS_K_GROUPED_CONTIG || (MAJOR_A == 1);
                constexpr bool b_with_group_k = DG_IS_K_GROUPED_CONTIG || (MAJOR_B == 1);
                uint32_t k_a_idx = scheduler.get_global_idx<a_with_group_k, DG_IX_K>(shape_k, BLOCK_K, k_block_idx, m_block_idx);
                uint32_t k_b_idx = scheduler.get_global_idx<b_with_group_k, DG_IX_K>(shape_k, BLOCK_K, k_block_idx, m_block_idx);

                // 2-CTA offsets
                if (MULTICAST > 1) {
                    m_idx += IS_MULTICAST_ON_A ? (dg_block_rank_in_cluster() * load_block_m) : 0;
                    n_idx += IS_MULTICAST_ON_A ? 0 : (dg_block_rank_in_cluster() * LOAD_BLOCK_N);
                }

                const uint32_t batch_idx = (GEMM_TYPE == DG_GT_BATCHED) ? scheduler.current_group_idx : 0;

                // Issue SFA/SFB TMAs first so the transpose overlaps A/B transfers
                uint32_t sf_arrival_bytes = 0;
                if (k_block_idx % NUM_SFA_STAGES_PER_LOAD == 0) {
                    const uint32_t sfa_m_idx = m_block_idx * BLOCK_M;
                    const uint32_t sfa_k_idx = scheduler.get_global_idx<!DG_IS_M_GROUPED_CONTIG, DG_IX_SFK>(
                        shape_sfa_k, SF_BLOCK_K, k_block_idx / NUM_SFA_STAGES_PER_LOAD);
                    dg100_tma_copy<SF_BLOCK_M, SF_BLOCK_K, 0, 1, 4, false>(
                        &tensor_map_sfa, &smem.sf_full_barriers[stage_idx],
                        &smem.sfa[stage_idx * SF_BLOCK_M * SF_BLOCK_K], 1, sfa_m_idx, sfa_k_idx, 0);
                    sf_arrival_bytes += SF_BLOCK_M * SF_BLOCK_K * 4;
                }
                if (k_block_idx % NUM_SFB_STAGES_PER_LOAD == 0) {
                    const uint32_t sfb_n_idx = n_block_idx * BLOCK_N;
                    const uint32_t sfb_k_idx = scheduler.get_global_idx<true, DG_IX_SFK>(
                        shape_sfb_k, SF_BLOCK_K, k_block_idx / NUM_SFB_STAGES_PER_LOAD, m_block_idx);
                    dg100_tma_copy<SF_BLOCK_N, SF_BLOCK_K, 0, 1, 4, false>(
                        &tensor_map_sfb, &smem.sf_full_barriers[stage_idx],
                        &smem.sfb[stage_idx * SF_BLOCK_N * SF_BLOCK_K], 1, sfb_n_idx, sfb_k_idx, 0);
                    sf_arrival_bytes += SF_BLOCK_N * SF_BLOCK_K * 4;
                }
                dg_mbarrier_arrive_expect_tx(&smem.sf_full_barriers[stage_idx], sf_arrival_bytes);

                // Issue A/B TMAs
#if MAJOR_A == 0
                dg100_tma_copy<BLOCK_K, LOAD_BLOCK_M, SWIZZLE_A, A_PACK, 1, IS_3D_TMA>(
                    &tensor_map_a, &smem.full_barriers[stage_idx],
                    &smem.a[stage_idx * DG_ALIGN4(LOAD_BLOCK_M * BLOCK_K * A_ESZ / A_PACK)],
                    MULTICAST, k_a_idx, m_idx, batch_idx);
#else
                dg100_tma_copy<LOAD_BLOCK_M, BLOCK_K, SWIZZLE_A, A_PACK, 1, IS_3D_TMA>(
                    &tensor_map_a, &smem.full_barriers[stage_idx],
                    &smem.a[stage_idx * DG_ALIGN4(LOAD_BLOCK_M * BLOCK_K * A_ESZ / A_PACK)],
                    MULTICAST, m_idx, k_a_idx, batch_idx);
#endif
#if MAJOR_B == 0
                dg100_tma_copy<BLOCK_K, LOAD_BLOCK_N, SWIZZLE_B, B_PACK, 1, IS_3D_TMA>(
                    &tensor_map_b, &smem.full_barriers[stage_idx],
                    &smem.b[stage_idx * DG_ALIGN4(LOAD_BLOCK_N * BLOCK_K * B_ESZ / B_PACK)],
                    MULTICAST, k_b_idx, n_idx, batch_idx);
#else
                dg100_tma_copy<LOAD_BLOCK_N, BLOCK_K, SWIZZLE_B, B_PACK, 1, IS_3D_TMA>(
                    &tensor_map_b, &smem.full_barriers[stage_idx],
                    &smem.b[stage_idx * DG_ALIGN4(LOAD_BLOCK_N * BLOCK_K * B_ESZ / B_PACK)],
                    MULTICAST, n_idx, k_b_idx, batch_idx);
#endif

                // Arrive at the full barriers (leader expects the pair's bytes)
                if (is_leader_cta) {
                    dg_mbarrier_arrive_expect_tx(&smem.full_barriers[stage_idx],
                                                 tma_bytes_per_stage * MULTICAST);
                } else {
                    dg_mbarrier_arrive_cluster(&smem.full_barriers[stage_idx], 0);
                }
            }
        }
    }
"#;
pub const FP8FP4_KERNEL_2: &str = r#"
    // ======================= warp 1 (leader): MMA issue ======================
    else if (warp_idx == 1 && is_leader_cta) {
        // Instruction descriptor (swap-AB swaps the operand roles)
        uint32_t instr_desc = SWAP_AB
            ? dg100_instr_desc_bs(B_FMT, A_FMT, UMMA_M, UMMA_N, MAJOR_B, MAJOR_A, 1)
            : dg100_instr_desc_bs(A_FMT, B_FMT, UMMA_M, UMMA_N, MAJOR_A, MAJOR_B, 1);

        uint64_t sf_desc = dg100_make_sf_desc(nullptr);

        // Per-stage descriptor bases (lane i holds stage i's low word)
        uint64_t a_desc0 = (MAJOR_A == 0)
            ? dg100_make_umma_desc_k_major<LOAD_BLOCK_M, BLOCK_K, SWIZZLE_A, A_PACK, 1>(&smem.a[0], 0, 0)
            : dg100_make_umma_desc_mn_major<LOAD_BLOCK_M, BLOCK_K, SWIZZLE_A, 1>(&smem.a[0], 0, 0);
        uint64_t b_desc0 = (MAJOR_B == 0)
            ? dg100_make_umma_desc_k_major<LOAD_BLOCK_N, BLOCK_K, SWIZZLE_B, B_PACK, 1>(&smem.b[0], 0, 0)
            : dg100_make_umma_desc_mn_major<LOAD_BLOCK_N, BLOCK_K, SWIZZLE_B, 1>(&smem.b[0], 0, 0);
        const uint32_t a_stage_bytes = DG_ALIGN4(LOAD_BLOCK_M * BLOCK_K * A_ESZ / A_PACK);
        const uint32_t b_stage_bytes = DG_ALIGN4(LOAD_BLOCK_N * BLOCK_K * B_ESZ / B_PACK);
        uint32_t a_desc_lo = lane_idx < NUM_STAGES ? dg100_desc_lo(a_desc0) + lane_idx * (a_stage_bytes / 16) : 0u;
        uint32_t b_desc_lo = lane_idx < NUM_STAGES ? dg100_desc_lo(b_desc0) + lane_idx * (b_stage_bytes / 16) : 0u;

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            // Wait until this accumulator stage is fully reusable (deferred to
            // the first K block so the SF UTCCP can overlap the epilogue drain)
            const uint32_t accum_stage_idx = static_cast<uint32_t>(scheduler.current_iter) % NUM_EPILOGUE_STAGES;
            const uint32_t accum_phase_idx = (static_cast<uint32_t>(scheduler.current_iter) / NUM_EPILOGUE_STAGES) & 1;
            dg_mbarrier_wait(&smem.tmem_empty_barriers[accum_stage_idx], accum_phase_idx ^ 1);
            dg100_after_thread_sync();

            const uint32_t num_total_k_blocks =
                dg_ceil_div(scheduler.current_shape_k ? scheduler.current_shape_k : 1, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                // broadcast this stage's base descriptor from the owner lane
                const uint32_t a_base_lo = __shfl_sync(0xffffffffu, a_desc_lo, static_cast<int>(stage_idx));
                const uint32_t b_base_lo = __shfl_sync(0xffffffffu, b_desc_lo, static_cast<int>(stage_idx));

                // Wait for the A/B TMA and the SF transposes
                dg_mbarrier_wait(&smem.full_barriers[stage_idx], phase);
                dg100_after_thread_sync();

                const uint32_t sfa_stage_in_group_idx = k_block_idx % NUM_SFA_STAGES_PER_LOAD;
                const uint32_t sfb_stage_in_group_idx = k_block_idx % NUM_SFB_STAGES_PER_LOAD;
                if (dg_elect_one()) {
                    // UTCCP: shared -> tensor memory SF copies at certain stages
                    const uint32_t sfa_base = stage_idx * SF_BLOCK_M * SF_BLOCK_K;
                    const uint32_t sfb_base = stage_idx * SF_BLOCK_N * SF_BLOCK_K;
                    if (sfa_stage_in_group_idx == 0) {
                        #pragma unroll 1
                        for (uint32_t i = 0; i < SF_BLOCK_K * SF_BLOCK_M / NUM_UTCCP_ALIGNED_ELEMS; ++ i) {
                            sf_desc = dg100_sf_desc_set_addr(sf_desc, &smem.sfa[sfa_base + i * NUM_UTCCP_ALIGNED_ELEMS]);
                            if (MULTICAST == 1) dg100_utccp_1sm(sf_desc, TMEM_START_COL_SFA + i * 4);
                            else                dg100_utccp_2sm(sf_desc, TMEM_START_COL_SFA + i * 4);
                        }
                    }
                    if (sfb_stage_in_group_idx == 0) {
                        #pragma unroll 1
                        for (uint32_t i = 0; i < SF_BLOCK_K * SF_BLOCK_N / NUM_UTCCP_ALIGNED_ELEMS; ++ i) {
                            sf_desc = dg100_sf_desc_set_addr(sf_desc, &smem.sfb[sfb_base + i * NUM_UTCCP_ALIGNED_ELEMS]);
                            if (MULTICAST == 1) dg100_utccp_1sm(sf_desc, TMEM_START_COL_SFB + i * 4);
                            else                dg100_utccp_2sm(sf_desc, TMEM_START_COL_SFB + i * 4);
                        }
                    }
                }
                __syncwarp();

                // Overlap barrier: wait for the preceding epilogue to release
                // the overlapped TMEM columns (after the SF UTCCP, before the
                // first UMMA of this block)
                if (NUM_OVERLAPPED_TMEM_COLS > 0) {
                    if (k_block_idx == 0 && scheduler.current_iter > 0) {
                        dg100_before_thread_sync();
                        const int32_t preceding_iter_idx = scheduler.current_iter - 1;
                        const uint32_t preceding_stage_idx = static_cast<uint32_t>(preceding_iter_idx) % NUM_EPILOGUE_STAGES;
                        const uint32_t preceding_phase_idx = (static_cast<uint32_t>(preceding_iter_idx) / NUM_EPILOGUE_STAGES) & 1;
                        dg_mbarrier_wait(&smem.tmem_overlap_barriers[preceding_stage_idx], preceding_phase_idx);
                        dg100_after_thread_sync();
                    }
                }

                if (dg_elect_one()) {
                    // Dynamic update of UMMA N based on the effective M (swap-AB)
                    if (SWAP_AB) {
                        const uint32_t umma_n = scheduler.get_aligned_effective_m_in_block(m_block_idx);
                        dg100_instr_desc_set_n(instr_desc, umma_n);
                    }
                    // Issue the UMMAs over this K block
                    #pragma unroll 4
                    for (uint32_t umma_k_idx = 0; umma_k_idx < BLOCK_K / UMMA_K; ++ umma_k_idx) {
                        const uint32_t offset = umma_k_idx * UMMA_K;
                        // Which 128-K SF sub-block this UMMA K step belongs to
                        const uint32_t subblock_idx = offset / NUM_UTCCP_ALIGNED_ELEMS;
                        // SF id (in units of 32 K elements inside the 128-K sub-block)
                        const uint32_t sf_id_in_subblock = (offset % NUM_UTCCP_ALIGNED_ELEMS) / 32;
                        const uint32_t tmem_col_sfa = TMEM_START_COL_SFA + subblock_idx * SF_BLOCK_M / 32;
                        const uint32_t tmem_col_sfb = TMEM_START_COL_SFB + subblock_idx * SF_BLOCK_N / 32;
                        const uint32_t sfa_id = (GRAN_K_A == 32) ? sf_id_in_subblock : sfa_stage_in_group_idx;
                        const uint32_t sfb_id = (GRAN_K_B == 32) ? sf_id_in_subblock : sfb_stage_in_group_idx;
                        const uint64_t runtime_desc = SWAP_AB
                            ? dg100_instr_desc_with_sf_id(instr_desc, sfb_id, sfa_id)
                            : dg100_instr_desc_with_sf_id(instr_desc, sfa_id, sfb_id);

                        const uint32_t a_byte_off = (MAJOR_A == 0)
                            ? (offset * A_ESZ / A_PACK)
                            : (offset * (SWIZZLE_A / A_ESZ));   // MN-major atom stride
                        const uint32_t b_byte_off = (MAJOR_B == 0)
                            ? (offset * B_ESZ / B_PACK)
                            : (offset * (SWIZZLE_B / B_ESZ));
                        const uint32_t a_lo = dg100_advance_desc_lo(a_base_lo, a_byte_off);
                        const uint32_t b_lo = dg100_advance_desc_lo(b_base_lo, b_byte_off);
                        const uint64_t a_desc = dg100_desc_with_lo(a_desc0, a_lo);
                        const uint64_t b_desc = dg100_desc_with_lo(b_desc0, b_lo);

                        const uint32_t accum_col = accum_stage_idx * (UMMA_N - NUM_OVERLAPPED_TMEM_COLS);
                        const uint32_t scale_c = (umma_k_idx > 0 || k_block_idx > 0) ? 1u : 0u;
                        if (SWAP_AB) {
                            if (IS_MXF4) {
                                if (MULTICAST == 1) dg100_mma_mxf4_1sm(b_desc, a_desc, accum_col, scale_c, runtime_desc, tmem_col_sfb, tmem_col_sfa);
                                else                dg100_mma_mxf4_2sm(b_desc, a_desc, accum_col, scale_c, runtime_desc, tmem_col_sfb, tmem_col_sfa);
                            } else {
                                if (MULTICAST == 1) dg100_mma_mxf8f6f4_1sm(b_desc, a_desc, accum_col, scale_c, runtime_desc, tmem_col_sfb, tmem_col_sfa);
                                else                dg100_mma_mxf8f6f4_2sm(b_desc, a_desc, accum_col, scale_c, runtime_desc, tmem_col_sfb, tmem_col_sfa);
                            }
                        } else {
                            if (IS_MXF4) {
                                if (MULTICAST == 1) dg100_mma_mxf4_1sm(a_desc, b_desc, accum_col, scale_c, runtime_desc, tmem_col_sfa, tmem_col_sfb);
                                else                dg100_mma_mxf4_2sm(a_desc, b_desc, accum_col, scale_c, runtime_desc, tmem_col_sfa, tmem_col_sfb);
                            } else {
                                if (MULTICAST == 1) dg100_mma_mxf8f6f4_1sm(a_desc, b_desc, accum_col, scale_c, runtime_desc, tmem_col_sfa, tmem_col_sfb);
                                else                dg100_mma_mxf8f6f4_2sm(a_desc, b_desc, accum_col, scale_c, runtime_desc, tmem_col_sfa, tmem_col_sfb);
                            }
                        }
                    }
                }
                __syncwarp();

                // Commit to the mbarrier: the SMEM stage is free, and (at the
                // last K block) the accumulator is ready for the epilogue.
                {
                    const bool do_tmem_full = (k_block_idx == num_total_k_blocks - 1);
                    if (MULTICAST == 1) {
                        dg100_umma_arrive_1sm(&smem.empty_barriers[stage_idx]);
                        if (do_tmem_full)
                            dg100_umma_arrive_1sm(&smem.tmem_full_barriers[accum_stage_idx]);
                    } else {
                        const uint16_t cta_mask = static_cast<uint16_t>((1 << MULTICAST) - 1);
                        dg100_umma_arrive_2sm(&smem.empty_barriers[stage_idx], cta_mask);
                        if (do_tmem_full)
                            dg100_umma_arrive_2sm(&smem.tmem_full_barriers[accum_stage_idx], cta_mask);
                    }
                }
                __syncwarp();
            }
        }

        // Safely deconstruct barriers: another round of waits for 2-CTA
        const int32_t iter_idx = scheduler.current_iter - 1;
        if (MULTICAST > 1 && iter_idx >= 0) {
            const uint32_t accum_phase_idx = (static_cast<uint32_t>(iter_idx) / NUM_EPILOGUE_STAGES) & 1;
            dg_mbarrier_wait(&smem.tmem_empty_barriers[static_cast<uint32_t>(iter_idx) % NUM_EPILOGUE_STAGES],
                             accum_phase_idx);
        }
    }
    // ================= warp 2/3: UTCCP SF transposers ========================
    else if (warp_idx == 2 || (SF_BLOCK_K == 2 && warp_idx == 3)) {
        const uint32_t sf_k_subblock_idx = warp_idx - 2;
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            const uint32_t num_total_k_blocks =
                dg_ceil_div(scheduler.current_shape_k ? scheduler.current_shape_k : 1, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                // Wait for the SF TMA arrival
                dg_mbarrier_wait(&smem.sf_full_barriers[stage_idx], phase);

                // Transpose for UTCCP at certain stages
                if (k_block_idx % NUM_SFA_STAGES_PER_LOAD == 0) {
                    #pragma unroll 1
                    for (uint32_t i = 0; i < SF_BLOCK_M / NUM_UTCCP_ALIGNED_ELEMS; ++ i)
                        dg_utccp_transpose(&smem.sfa[(stage_idx * SF_BLOCK_M + sf_k_subblock_idx * SF_BLOCK_M +
                                                      i * NUM_UTCCP_ALIGNED_ELEMS)]);
                }
                if (k_block_idx % NUM_SFB_STAGES_PER_LOAD == 0) {
                    #pragma unroll 1
                    for (uint32_t i = 0; i < SF_BLOCK_N / NUM_UTCCP_ALIGNED_ELEMS; ++ i)
                        dg_utccp_transpose(&smem.sfb[(stage_idx * SF_BLOCK_N + sf_k_subblock_idx * SF_BLOCK_N +
                                                      i * NUM_UTCCP_ALIGNED_ELEMS)]);
                }
                dg_tma_store_fence();   // fence.proxy.async.shared::cta
                dg_mbarrier_arrive_cluster(&smem.full_barriers[stage_idx], 0);
            }
        }
    }
"#;
pub const FP8FP4_KERNEL_3: &str = r#"
    // ======================= epilogue warp groups ============================
    else if (warp_idx >= NUM_NON_EPILOGUE_THREADS / 32 &&
             warp_idx < (NUM_NON_EPILOGUE_THREADS + NUM_UMMA_STORE_THREADS) / 32) {
        const uint32_t epilogue_warp_idx = warp_idx - NUM_NON_EPILOGUE_THREADS / 32;

        // The TMEM base must be column 0 (a fresh allocation).
        (void) dg_ld_shared_u32(&smem.tmem_ptr);

        // The store pipeline is shared across scheduled blocks
        uint32_t tma_stage_idx = 0;

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            const uint32_t accum_stage_idx = static_cast<uint32_t>(scheduler.current_iter) % NUM_EPILOGUE_STAGES;
            const uint32_t accum_phase_idx = (static_cast<uint32_t>(scheduler.current_iter) / NUM_EPILOGUE_STAGES) & 1;

            // Wait for the UMMA arrival
            dg_mbarrier_wait(&smem.tmem_full_barriers[accum_stage_idx], accum_phase_idx);
            dg100_after_thread_sync();

            const uint32_t tmem_base_addr = accum_stage_idx * (UMMA_N - NUM_OVERLAPPED_TMEM_COLS);
            const bool reverse_store_order = NUM_OVERLAPPED_TMEM_COLS > 0 && accum_stage_idx == 0;

            constexpr bool cd_with_group_offset = !DG_IS_M_GROUPED_CONTIG && !DG_IS_K_GROUPED_CONTIG;
            const uint32_t base_m_idx = scheduler.get_global_idx<cd_with_group_offset, DG_IX_MN>(shape_m, BLOCK_M, m_block_idx);
            const uint32_t base_n_idx = n_block_idx * BLOCK_N;
            const bool is_empty_group = DG_IS_K_GROUPED_CONTIG && scheduler.current_shape_k == 0;
            const uint32_t batch_idx = (GEMM_TYPE == DG_GT_BATCHED) ? scheduler.current_group_idx : 0;

#if SWAP_AB
            {
                const uint32_t effective_m = scheduler.get_aligned_effective_m_in_block(m_block_idx);
                dg_store_cd_swap_ab(smem, tma_stage_idx, tmem_base_addr,
                                    base_m_idx, base_n_idx, batch_idx, is_empty_group, effective_m,
                                    epilogue_warp_idx, lane_idx,
                                    sfd, sfd_stride, epi_shape_m, epi_shape_n, epi_alpha,
                                    reverse_store_order,
                                    &smem.tmem_overlap_barriers[accum_stage_idx],
                                    &smem.tmem_empty_barriers[accum_stage_idx],
                                    tensor_map_cd);
            }
#else
            {
                dg_store_cd(smem, tma_stage_idx, tmem_base_addr,
                            base_m_idx, base_n_idx, batch_idx, is_empty_group,
                            epilogue_warp_idx, lane_idx,
                            sfd, sfd_stride, epi_shape_m, epi_shape_n, epi_alpha,
                            reverse_store_order,
                            &smem.tmem_overlap_barriers[accum_stage_idx],
                            &smem.tmem_empty_barriers[accum_stage_idx],
                            tensor_map_cd);
            }
#endif
        }
    }

    // Final sync + tensor memory deallocation
    if (MULTICAST > 1) dg_cluster_sync_relaxed(); else __syncthreads();
    if (warp_idx == 0) {
        if (MULTICAST == 1) dg100_tmem_dealloc_1sm(0, NUM_TMEM_COLS);
        else                dg100_tmem_dealloc_2sm(0, NUM_TMEM_COLS);
    }
}
"#;

/// The two store epilogues (normal and swap-AB), ported 1:1 from upstream
/// `sm100_store_cd.cuh` / `sm100_store_cd_swap_ab.cuh`.

/// Assemble the full SM100 FP8/FP4 GEMM translation unit for one config.
pub fn build_fp8_fp4_source(cfg: &Fp8Fp4Config) -> String {
    let vars: Vec<(&str, String)> = vec![
        ("GEMM_TYPE", cfg.gemm_type.to_string()),
        ("MAJOR_A", cfg.major_a.to_string()),
        ("MAJOR_B", cfg.major_b.to_string()),
        ("GRAN_K_A", cfg.gran_k_a.to_string()),
        ("GRAN_K_B", cfg.gran_k_b.to_string()),
        ("K_ALIGNMENT", cfg.k_alignment.to_string()),
        ("SHAPE_M", cfg.shape_m.to_string()),
        ("SHAPE_N", cfg.shape_n.to_string()),
        ("SHAPE_K", cfg.shape_k.to_string()),
        ("BLOCK_M", cfg.block_m.to_string()),
        ("BLOCK_N", cfg.block_n.to_string()),
        ("BLOCK_K", cfg.block_k.to_string()),
        ("NUM_GROUPS", cfg.num_groups.to_string()),
        ("SWIZZLE_A", cfg.swizzle_a.to_string()),
        ("SWIZZLE_B", cfg.swizzle_b.to_string()),
        ("SWIZZLE_CD", cfg.swizzle_cd.to_string()),
        ("NUM_STAGES", cfg.num_stages.to_string()),
        ("NUM_TMA_STORE_STAGES", cfg.num_tma_store_stages.to_string()),
        (
            "NUM_NON_EPILOGUE_THREADS",
            cfg.num_non_epilogue_threads.to_string(),
        ),
        ("NUM_EPILOGUE_THREADS", cfg.num_epilogue_threads.to_string()),
        ("MULTICAST", cfg.multicast.to_string()),
        (
            "IS_MULTICAST_ON_A",
            (cfg.is_multicast_on_a as u32).to_string(),
        ),
        ("NUM_SMS", cfg.num_sms.to_string()),
        ("SWAP_AB", (cfg.swap_ab as u32).to_string()),
        (
            "ENSURE_ZERO_PADDING",
            (cfg.ensure_zero_padding as u32).to_string(),
        ),
        (
            "WITH_ACCUMULATION",
            (cfg.with_accumulation as u32).to_string(),
        ),
        ("A_BITS", cfg.a_bits.to_string()),
        ("B_BITS", cfg.b_bits.to_string()),
        ("CD_DTYPE", cfg.cd_dtype.to_string()),
        ("EPILOGUE_OP", cfg.epilogue_op.to_string()),
    ];
    let vars_ref: Vec<(&str, &str)> = vars.iter().map(|(k, v)| (*k, v.as_str())).collect();
    // Order: common PTX layer -> per-config #defines + shared storage ->
    // shared epilogue operators/store functions -> the kernel body.
    let defines = subst(FP8FP4_KERNEL, &vars_ref);
    let body = subst(
        &format!("{FP8FP4_KERNEL_1}{FP8FP4_KERNEL_2}{FP8FP4_KERNEL_3}"),
        &vars_ref,
    );
    format!(
        "{}{}{}{}{}",
        sm100_header(),
        defines,
        super::sm100_common::sm100_sched(),
        super::sm100_common::SM100_EPILOGUE_FNS,
        body
    )
}
