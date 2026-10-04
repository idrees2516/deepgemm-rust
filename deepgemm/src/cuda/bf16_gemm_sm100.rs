//! SM100 (Blackwell) BF16 GEMM kernel — the tcgen05 (`kind::f16`) port of
//! upstream DeepGEMM's `sm100_bf16_gemm.cuh`.
//!
//! Same warp-specialized persistent architecture as the FP8/FP4 kernel but
//! without the scale-factor pipeline; includes the upstream stage-merging
//! optimization (fewer, deeper K stages to amortize `tcgen05.commit`) and the
//! tensor-core utilization control (idle-spin throttling).

use super::sm100_common::{sm100_header, SM100_EPILOGUE_FNS};
use super::subst;

#[derive(Debug, Clone)]
pub struct Bf16Config {
    pub gemm_type: u32,
    pub major_a: u32,
    pub major_b: u32,
    pub shape_m: u32,
    pub shape_n: u32,
    pub shape_k: u32,
    pub block_m: u32,
    pub block_n: u32,
    pub block_k: u32, // always 64 (pre-merge)
    pub num_groups: u32,
    pub swizzle_a: u32,
    pub swizzle_b: u32,
    pub swizzle_cd: u32,
    pub num_stages: u32, // pre-merge stage count
    pub num_non_epilogue_threads: u32,
    pub num_epilogue_threads: u32,
    pub multicast: u32,
    pub is_multicast_on_a: bool,
    pub num_sms: u32,
    pub k_alignment: u32,
    pub swap_ab: bool,
    pub ensure_zero_padding: bool,
    pub with_accumulation: bool,
    pub cd_dtype: u32,    // 0 = bf16, 1 = fp32
    pub epilogue_op: u32, // 0/1/2/3 (3 = fp8 output w/ SFs)
    pub tc_util: u32,     // percent of tensor-core utilization (100 = off)
}

pub const BF16_KERNEL: &str = r#"
// ===========================================================================
// deepgemm-rust :: sm100_bf16_gemm (one block configuration)
// Port of upstream DeepGEMM sm100_bf16_gemm.cuh
// ===========================================================================
#define GEMM_TYPE               %%GEMM_TYPE%%
#define MAJOR_A                 %%MAJOR_A%%
#define MAJOR_B                 %%MAJOR_B%%
#define SHAPE_M                 %%SHAPE_M%%
#define SHAPE_N                 %%SHAPE_N%%
#define SHAPE_K                 %%SHAPE_K%%
#define BLOCK_M                 %%BLOCK_M%%
#define BLOCK_N                 %%BLOCK_N%%
#define BLOCK_K_                %%BLOCK_K%%
#define NUM_GROUPS              %%NUM_GROUPS%%
#define SWIZZLE_A               %%SWIZZLE_A%%
#define SWIZZLE_B               %%SWIZZLE_B%%
#define SWIZZLE_CD              %%SWIZZLE_CD%%
#define NUM_STAGES_             %%NUM_STAGES_%%
#define NUM_NON_EPILOGUE_THREADS %%NUM_NON_EPILOGUE_THREADS%%
#define NUM_EPILOGUE_THREADS    %%NUM_EPILOGUE_THREADS%%
#define MULTICAST               %%MULTICAST%%
#define IS_MULTICAST_ON_A       %%IS_MULTICAST_ON_A%%
#define NUM_SMS                 %%NUM_SMS%%
#define K_ALIGNMENT             %%K_ALIGNMENT%%
#define SWAP_AB                 %%SWAP_AB%%
#define ENSURE_ZERO_PADDING     %%ENSURE_ZERO_PADDING%%
#define WITH_ACCUMULATION       %%WITH_ACCUMULATION%%
#define CD_DTYPE                %%CD_DTYPE%%
#define EPILOGUE_OP             %%EPILOGUE_OP%%
#define TC_UTIL                 %%TC_UTIL%%
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
#define SFK_SPAN                K_ALIGNMENT
#define NUM_TMA_STORE_STAGES    2u
#define NUM_EPILOGUE_STAGES     2u
#define NUM_OVERLAPPED_TMEM_COLS 0u
#define CD_ELEM_SIZE            (CD_DTYPE == 1 ? 4u : 2u)
#define WITH_OUTPUT_SF          (EPILOGUE_OP == 3)
#define WITH_STOCHASTIC         (EPILOGUE_OP == 2)
#define SF_GRAN_N               32u
#if GEMM_TYPE == DG_GT_BATCHED
#define IS_3D_TMA               true
#else
#define IS_3D_TMA               false
#endif
#define DG_ALIGN4(x)            ((x + 3) / 4 * 4)

// ---------- stage merging (fewer `tcgen05.commit` round-trips) ----------
#if (NUM_STAGES_ >= 8) && (GEMM_TYPE == DG_GT_NORMAL) && (MAJOR_A == 0) && (MAJOR_B == 0)
#define DO_MERGE_STAGES 1
#else
#define DO_MERGE_STAGES 0
#endif
#if DO_MERGE_STAGES
#define NUM_STAGES_PER_MERGE   (NUM_STAGES_ / 8)
#else
#define NUM_STAGES_PER_MERGE   1u
#endif
#define BLOCK_K                (BLOCK_K_ * NUM_STAGES_PER_MERGE)
#define NUM_STAGES             (NUM_STAGES_ / NUM_STAGES_PER_MERGE)
#define BLOCK_ATOM_K           (BLOCK_K / NUM_STAGES_PER_MERGE)

// ---------- derived constants ----------
#define LAYOUT_AD_M             128u
#define UMMA_M                  (LAYOUT_AD_M * MULTICAST)
#define UMMA_N                  (SWAP_AB ? BLOCK_M : BLOCK_N)
#define UMMA_K                  16u
#define LOAD_BLOCK_M            (BLOCK_M / (IS_MULTICAST_ON_A ? MULTICAST : 1))
#define LOAD_BLOCK_N            (BLOCK_N / (IS_MULTICAST_ON_A ? 1 : MULTICAST))
#define STORE_BLOCK_M           (SWAP_AB ? 16u : (BLOCK_M < LAYOUT_AD_M ? BLOCK_M : LAYOUT_AD_M))
#define STORE_BLOCK_N           (SWAP_AB ? BLOCK_N : (SWIZZLE_CD / CD_ELEM_SIZE))
#define NUM_UMMA_STORE_THREADS  (SWAP_AB ? NUM_EPILOGUE_THREADS : STORE_BLOCK_M)
#define NUM_ACCUM_TMEM_COLS     (NUM_EPILOGUE_STAGES * UMMA_N)
#define NUM_TMEM_COLS_RAW       NUM_ACCUM_TMEM_COLS
#define NUM_TMEM_COLS           (NUM_TMEM_COLS_RAW <= 32 ? 32u : (NUM_TMEM_COLS_RAW <= 64 ? 64u : (NUM_TMEM_COLS_RAW <= 128 ? 128u : (NUM_TMEM_COLS_RAW <= 256 ? 256u : 512u))))
#define NUM_1D_BLOCKS_PER_GROUP (dg_num_1d_blocks_per_group())

static_assert(BLOCK_K_ == 64, "invalid block K");
static_assert(BLOCK_K % UMMA_K == 0, "block K must be divisible by UMMA K");
static_assert(!DG_IS_K_GROUPED_CONTIG || K_ALIGNMENT % BLOCK_K == 0, "K alignment must be divisible by block K");
static_assert(MULTICAST == 1 || MULTICAST == 2, "only 1/2 multicast");
static_assert(SWAP_AB ? BLOCK_N == LAYOUT_AD_M
                      : (BLOCK_M == 32 || BLOCK_M == 64 || BLOCK_M == LAYOUT_AD_M),
              "invalid block size");
static_assert(NUM_UMMA_STORE_THREADS % 32 == 0, "invalid store block M");
static_assert(NUM_TMEM_COLS >= 32 && NUM_TMEM_COLS <= 512, "invalid tmem columns");
static_assert(NUM_STAGES <= 32, "too many stages");
static_assert(WITH_OUTPUT_SF ? (CD_DTYPE == 2 && GEMM_TYPE == DG_GT_BATCHED && !WITH_ACCUMULATION)
                             : (CD_DTYPE == 0 || CD_DTYPE == 1),
              "invalid C/D dtype");
static_assert(!WITH_STOCHASTIC || (CD_DTYPE == 0 && !WITH_ACCUMULATION),
              "stochastic rounding needs a direct BF16 output");
static_assert(TC_UTIL > 0 && TC_UTIL <= 100, "invalid tensor-core utilization");

struct SmemStorage {
    alignas(1024) uint8_t cd[NUM_TMA_STORE_STAGES * DG_ALIGN4(STORE_BLOCK_M * STORE_BLOCK_N * CD_ELEM_SIZE)];
    alignas(1024) uint8_t a[NUM_STAGES * DG_ALIGN4(LOAD_BLOCK_M * BLOCK_K * 2)];
    alignas(1024) uint8_t b[NUM_STAGES * DG_ALIGN4(LOAD_BLOCK_N * BLOCK_K * 2)];
    uint64_t full_barriers[NUM_STAGES];
    uint64_t empty_barriers[NUM_STAGES];
    uint64_t tmem_full_barriers[NUM_EPILOGUE_STAGES];
    uint64_t tmem_empty_barriers[NUM_EPILOGUE_STAGES];
    uint64_t tensor_core_full_barrier;
    uint32_t tmem_ptr;
};
"#;

/// The kernel body (TMA warp, MMA warp, epilogue warps).
pub const BF16_KERNEL_BODY: &str = r#"
// ============================== the kernel ==================================
extern "C" __global__ void __launch_bounds__(NUM_THREADS, 1) deepgemm_sm100_bf16_gemm(
        int* __restrict__ grouped_layout,
        uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
        uint32_t* __restrict__ sfd, uint32_t sfd_stride,
        uint32_t epi_shape_m, uint32_t epi_shape_n, float epi_alpha,
        const __grid_constant__ TmaDescriptor tensor_map_a,
        const __grid_constant__ TmaDescriptor tensor_map_b,
        const __grid_constant__ TmaDescriptor tensor_map_cd) {
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    SmemStorage& smem = *reinterpret_cast<SmemStorage*>(smem_buffer);

    const bool is_leader_cta = dg_block_rank_in_cluster() == 0;
    const uint32_t warp_idx = dg_warp_idx();
    const uint32_t lane_idx = dg_lane_idx();

    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;

    if (MULTICAST > 1) dg_cluster_sync_relaxed();

    if (warp_idx == 0) {
        dg_prefetch_tma_descriptor(&tensor_map_a);
        dg_prefetch_tma_descriptor(&tensor_map_b);
        dg_prefetch_tma_descriptor(&tensor_map_cd);
    }

    if (warp_idx == 1 && dg_elect_one()) {
        #pragma unroll 1
        for (uint32_t i = 0; i < NUM_STAGES; ++ i) {
            dg_mbarrier_init(&smem.full_barriers[i], MULTICAST);
            dg_mbarrier_init(&smem.empty_barriers[i], 1);
        }
        #pragma unroll 1
        for (uint32_t i = 0; i < NUM_EPILOGUE_STAGES; ++ i) {
            dg_mbarrier_init(&smem.tmem_full_barriers[i], 1);
            dg_mbarrier_init(&smem.tmem_empty_barriers[i], MULTICAST * NUM_UMMA_STORE_THREADS);
        }
        if (TC_UTIL < 100) dg_mbarrier_init(&smem.tensor_core_full_barrier, 1);
        dg_fence_barrier_init();
    } else if (warp_idx == 2) {
        if (MULTICAST == 1) dg100_tmem_alloc_1sm(NUM_TMEM_COLS, &smem.tmem_ptr);
        else                dg100_tmem_alloc_2sm(NUM_TMEM_COLS, &smem.tmem_ptr);
    }
    if (MULTICAST > 1) dg_cluster_sync_relaxed(); else __syncthreads();

    dg_grid_dependency_sync();

    uint32_t m_block_idx, n_block_idx;
    DgSched scheduler(shape_m, shape_n, shape_k, grouped_layout);

    uint32_t stage_idx = 0, phase = 0, tensor_core_phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;
        stage_idx = stage_idx == NUM_STAGES - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    // ============================ warp 0: TMA loads ==========================
    if (warp_idx == 0 && dg_elect_one()) {
        const uint32_t load_block_m = SWAP_AB
            ? scheduler.get_aligned_effective_m_in_block(m_block_idx) / MULTICAST : LOAD_BLOCK_M;
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            const uint32_t num_total_k_blocks =
                dg_ceil_div(scheduler.current_shape_k ? scheduler.current_shape_k : 1, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                dg_mbarrier_wait(&smem.empty_barriers[stage_idx], phase ^ 1);

                constexpr bool a_with_group_mn = (GEMM_TYPE == DG_GT_MGROUPED_MASKED);
                uint32_t m_idx = scheduler.get_global_idx<a_with_group_mn, DG_IX_MN>(shape_m, BLOCK_M, m_block_idx);
                constexpr bool b_with_group_mn = (MAJOR_B == 0) &&
                    (DG_IS_M_GROUPED_CONTIG || GEMM_TYPE == DG_GT_MGROUPED_MASKED);
                uint32_t n_idx = scheduler.get_global_idx<b_with_group_mn, DG_IX_MN>(shape_n, BLOCK_N, n_block_idx, m_block_idx);
                constexpr bool a_with_group_k = DG_IS_K_GROUPED_CONTIG || (MAJOR_A == 1);
                constexpr bool b_with_group_k = DG_IS_K_GROUPED_CONTIG || (MAJOR_B == 1);
                uint32_t k_a_idx = scheduler.get_global_idx<a_with_group_k, DG_IX_K>(shape_k, BLOCK_K, k_block_idx, m_block_idx);
                uint32_t k_b_idx = scheduler.get_global_idx<b_with_group_k, DG_IX_K>(shape_k, BLOCK_K, k_block_idx, m_block_idx);

                if (MULTICAST > 1) {
                    m_idx += IS_MULTICAST_ON_A ? (dg_block_rank_in_cluster() * load_block_m) : 0;
                    n_idx += IS_MULTICAST_ON_A ? 0 : (dg_block_rank_in_cluster() * LOAD_BLOCK_N);
                }

                const uint32_t batch_idx = (GEMM_TYPE == DG_GT_BATCHED) ? scheduler.current_group_idx : 0;
#if MAJOR_A == 0
                dg100_tma_copy<BLOCK_K, LOAD_BLOCK_M, SWIZZLE_A, 1, 2, IS_3D_TMA>(
                    &tensor_map_a, &smem.full_barriers[stage_idx],
                    &smem.a[stage_idx * DG_ALIGN4(LOAD_BLOCK_M * BLOCK_K * 2)],
                    MULTICAST, k_a_idx, m_idx, batch_idx);
#else
                dg100_tma_copy<LOAD_BLOCK_M, BLOCK_K, SWIZZLE_A, 1, 2, IS_3D_TMA>(
                    &tensor_map_a, &smem.full_barriers[stage_idx],
                    &smem.a[stage_idx * DG_ALIGN4(LOAD_BLOCK_M * BLOCK_K * 2)],
                    MULTICAST, m_idx, k_a_idx, batch_idx);
#endif
#if MAJOR_B == 0
                dg100_tma_copy<BLOCK_K, LOAD_BLOCK_N, SWIZZLE_B, 1, 2, IS_3D_TMA>(
                    &tensor_map_b, &smem.full_barriers[stage_idx],
                    &smem.b[stage_idx * DG_ALIGN4(LOAD_BLOCK_N * BLOCK_K * 2)],
                    MULTICAST, k_b_idx, n_idx, batch_idx);
#else
                dg100_tma_copy<LOAD_BLOCK_N, BLOCK_K, SWIZZLE_B, 1, 2, IS_3D_TMA>(
                    &tensor_map_b, &smem.full_barriers[stage_idx],
                    &smem.b[stage_idx * DG_ALIGN4(LOAD_BLOCK_N * BLOCK_K * 2)],
                    MULTICAST, n_idx, k_b_idx, batch_idx);
#endif

                const uint32_t arrival_bytes =
                    (LOAD_BLOCK_M * BLOCK_K + LOAD_BLOCK_N * BLOCK_K) * 2;
                if (is_leader_cta) {
                    dg_mbarrier_arrive_expect_tx(&smem.full_barriers[stage_idx],
                                                 arrival_bytes * MULTICAST);
                } else {
                    dg_mbarrier_arrive_cluster(&smem.full_barriers[stage_idx], 0);
                }
            }
        }
    }
    // ======================= warp 1 (leader): MMA issue ======================
    else if (warp_idx == 1 && is_leader_cta) {
        uint32_t instr_desc = SWAP_AB
            ? dg100_instr_desc_f16(DG_UFMT_BF16, DG_UFMT_BF16, UMMA_M, UMMA_N, MAJOR_B, MAJOR_A)
            : dg100_instr_desc_f16(DG_UFMT_BF16, DG_UFMT_BF16, UMMA_M, UMMA_N, MAJOR_A, MAJOR_B);

        uint64_t a_desc0 = (MAJOR_A == 0)
            ? dg100_make_umma_desc_k_major<LOAD_BLOCK_M, BLOCK_ATOM_K, SWIZZLE_A, 1, 2>(&smem.a[0], 0, 0)
            : dg100_make_umma_desc_mn_major<LOAD_BLOCK_M, BLOCK_ATOM_K, SWIZZLE_A, 2>(&smem.a[0], 0, 0);
        uint64_t b_desc0 = (MAJOR_B == 0)
            ? dg100_make_umma_desc_k_major<LOAD_BLOCK_N, BLOCK_ATOM_K, SWIZZLE_B, 1, 2>(&smem.b[0], 0, 0)
            : dg100_make_umma_desc_mn_major<LOAD_BLOCK_N, BLOCK_ATOM_K, SWIZZLE_B, 2>(&smem.b[0], 0, 0);
        const uint32_t a_stage_bytes = DG_ALIGN4(LOAD_BLOCK_M * BLOCK_K * 2);
        const uint32_t b_stage_bytes = DG_ALIGN4(LOAD_BLOCK_N * BLOCK_K * 2);
        uint32_t a_desc_lo = lane_idx < NUM_STAGES ? dg100_desc_lo(a_desc0) + lane_idx * (a_stage_bytes / 16) : 0u;
        uint32_t b_desc_lo = lane_idx < NUM_STAGES ? dg100_desc_lo(b_desc0) + lane_idx * (b_stage_bytes / 16) : 0u;

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            const uint32_t accum_stage_idx = static_cast<uint32_t>(scheduler.current_iter) % NUM_EPILOGUE_STAGES;
            const uint32_t accum_phase_idx = (static_cast<uint32_t>(scheduler.current_iter) / NUM_EPILOGUE_STAGES) & 1;
            dg_mbarrier_wait(&smem.tmem_empty_barriers[accum_stage_idx], accum_phase_idx ^ 1);
            dg100_after_thread_sync();

            const uint32_t num_total_k_blocks =
                dg_ceil_div(scheduler.current_shape_k ? scheduler.current_shape_k : 1, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                const uint32_t a_base_lo = __shfl_sync(0xffffffffu, a_desc_lo, static_cast<int>(stage_idx));
                const uint32_t b_base_lo = __shfl_sync(0xffffffffu, b_desc_lo, static_cast<int>(stage_idx));

                dg_mbarrier_wait(&smem.full_barriers[stage_idx], phase);
                dg100_after_thread_sync();

                if (dg_elect_one()) {
                    if (SWAP_AB) {
                        const uint32_t umma_n = scheduler.get_aligned_effective_m_in_block(m_block_idx);
                        dg100_instr_desc_set_n(instr_desc, umma_n);
                    }
                    const uint64_t runtime_desc = dg100_make_runtime_desc(instr_desc);
                    #pragma unroll 4
                    for (uint32_t umma_k_idx = 0; umma_k_idx < BLOCK_K / UMMA_K; ++ umma_k_idx) {
                        const uint32_t atom_k_idx = umma_k_idx * UMMA_K / BLOCK_ATOM_K;
                        const uint32_t inner_k_idx = umma_k_idx * UMMA_K % BLOCK_ATOM_K;
                        // (mn element offset, k element offset) -> bytes
                        const uint32_t a_elems = atom_k_idx * LOAD_BLOCK_M * BLOCK_ATOM_K +
                            (MAJOR_A == 0 ? inner_k_idx : inner_k_idx * (SWIZZLE_A / 2));
                        const uint32_t b_elems = atom_k_idx * LOAD_BLOCK_N * BLOCK_ATOM_K +
                            (MAJOR_B == 0 ? inner_k_idx : inner_k_idx * (SWIZZLE_B / 2));
                        const uint64_t a_desc = dg100_desc_with_lo(
                            a_desc0, dg100_advance_desc_lo(a_base_lo, a_elems * 2));
                        const uint64_t b_desc = dg100_desc_with_lo(
                            b_desc0, dg100_advance_desc_lo(b_base_lo, b_elems * 2));
                        const uint32_t scale_c = (umma_k_idx > 0 || k_block_idx > 0) ? 1u : 0u;
                        if (SWAP_AB) {
                            if (MULTICAST == 1) dg100_mma_f16bf16_1sm(b_desc, a_desc, accum_stage_idx * UMMA_N, scale_c, runtime_desc);
                            else                dg100_mma_f16bf16_2sm(b_desc, a_desc, accum_stage_idx * UMMA_N, scale_c, runtime_desc);
                        } else {
                            if (MULTICAST == 1) dg100_mma_f16bf16_1sm(a_desc, b_desc, accum_stage_idx * UMMA_N, scale_c, runtime_desc);
                            else                dg100_mma_f16bf16_2sm(a_desc, b_desc, accum_stage_idx * UMMA_N, scale_c, runtime_desc);
                        }
                    }
                }
                __syncwarp();

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

                // Let the tensor cores relax to avoid frequency drops
                if (TC_UTIL < 100) {
                    if (MULTICAST == 1) dg100_umma_arrive_1sm(&smem.tensor_core_full_barrier);
                    else                dg100_umma_arrive_2sm(&smem.tensor_core_full_barrier,
                                                                 static_cast<uint16_t>((1 << MULTICAST) - 1));
                    __syncwarp();
                    dg_mbarrier_wait(&smem.tensor_core_full_barrier, tensor_core_phase);
                    tensor_core_phase ^= 1;
                    const uint64_t num_umma_cycles = (2ull * UMMA_M * UMMA_N * BLOCK_K) / 8192ull;
                    const uint64_t num_dummy_cycles = (100ull - TC_UTIL) * num_umma_cycles / TC_UTIL;
                    const uint64_t start_clock = clock64();
                    if (dg_elect_one())
                        while (static_cast<uint64_t>(clock64()) - start_clock < num_dummy_cycles) {}
                    __syncwarp();
                }
            }
        }

        const int32_t iter_idx = scheduler.current_iter - 1;
        if (MULTICAST > 1 && iter_idx >= 0) {
            const uint32_t accum_phase_idx = (static_cast<uint32_t>(iter_idx) / NUM_EPILOGUE_STAGES) & 1;
            dg_mbarrier_wait(&smem.tmem_empty_barriers[static_cast<uint32_t>(iter_idx) % NUM_EPILOGUE_STAGES],
                             accum_phase_idx);
        }
    }
    // ======================= epilogue warp groups ============================
    else if (warp_idx >= NUM_NON_EPILOGUE_THREADS / 32 &&
             warp_idx < (NUM_NON_EPILOGUE_THREADS + NUM_UMMA_STORE_THREADS) / 32) {
        const uint32_t epilogue_warp_idx = warp_idx - NUM_NON_EPILOGUE_THREADS / 32;
        (void) dg_ld_shared_u32(&smem.tmem_ptr);

        uint32_t tma_stage_idx = 0;
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            const uint32_t accum_stage_idx = static_cast<uint32_t>(scheduler.current_iter) % NUM_EPILOGUE_STAGES;
            const uint32_t accum_phase_idx = (static_cast<uint32_t>(scheduler.current_iter) / NUM_EPILOGUE_STAGES) & 1;

            dg_mbarrier_wait(&smem.tmem_full_barriers[accum_stage_idx], accum_phase_idx);
            dg100_after_thread_sync();

            const uint32_t tmem_base_addr = accum_stage_idx * UMMA_N;
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
                                    false,
                                    &smem.tmem_empty_barriers[accum_stage_idx],
                                    &smem.tmem_empty_barriers[accum_stage_idx],
                                    tensor_map_cd);
            }
#else
            {
                dg_store_cd(smem, tma_stage_idx, tmem_base_addr,
                            base_m_idx, base_n_idx, batch_idx, is_empty_group,
                            epilogue_warp_idx, lane_idx,
                            sfd, sfd_stride, epi_shape_m, epi_shape_n, epi_alpha,
                            false,
                            &smem.tmem_empty_barriers[accum_stage_idx],
                            &smem.tmem_empty_barriers[accum_stage_idx],
                            tensor_map_cd);
            }
#endif
        }
    }

    if (MULTICAST > 1) dg_cluster_sync_relaxed(); else __syncthreads();
    if (warp_idx == 0) {
        if (MULTICAST == 1) dg100_tmem_dealloc_1sm(0, NUM_TMEM_COLS);
        else                dg100_tmem_dealloc_2sm(0, NUM_TMEM_COLS);
    }
}
"#;

/// Assemble the SM100 BF16 GEMM translation unit for one config.
pub fn build_bf16_source(cfg: &Bf16Config) -> String {
    let vars: Vec<(&str, String)> = vec![
        ("GEMM_TYPE", cfg.gemm_type.to_string()),
        ("MAJOR_A", cfg.major_a.to_string()),
        ("MAJOR_B", cfg.major_b.to_string()),
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
        ("NUM_STAGES_", cfg.num_stages.to_string()),
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
        ("K_ALIGNMENT", cfg.k_alignment.to_string()),
        ("SWAP_AB", (cfg.swap_ab as u32).to_string()),
        (
            "ENSURE_ZERO_PADDING",
            (cfg.ensure_zero_padding as u32).to_string(),
        ),
        (
            "WITH_ACCUMULATION",
            (cfg.with_accumulation as u32).to_string(),
        ),
        ("CD_DTYPE", cfg.cd_dtype.to_string()),
        ("EPILOGUE_OP", cfg.epilogue_op.to_string()),
        ("TC_UTIL", cfg.tc_util.to_string()),
    ];
    let vars_ref: Vec<(&str, &str)> = vars.iter().map(|(k, v)| (*k, v.as_str())).collect();
    // Order: core PTX -> defines+storage -> scheduler -> epilogues -> body.
    format!(
        "{}{}{}{}{}",
        sm100_header(),
        subst(BF16_KERNEL, &vars_ref),
        super::sm100_common::sm100_sched(),
        SM100_EPILOGUE_FNS,
        subst(BF16_KERNEL_BODY, &vars_ref)
    )
}
