//! SM100 (Blackwell) common device layer: a self-contained CUDA "header"
//! prepended to every SM100 kernel TU.
//!
//! Faithful PTX-level ports of the tcgen05 (5th-generation tensor core)
//! primitives used by upstream DeepGEMM's SM100 kernels:
//! * `tcgen05.mma` — MXF4 (packed FP4, block-scaled, K=64), MXF8F6F4
//!   (block-scaled FP8/FP6/FP4, K=32) and F16BF16 MMAs, 1-CTA and 2-CTA
//!   (`cta_group::2`) forms,
//! * `tcgen05.cp` (UTCCP: shared -> tensor memory scale-factor copies),
//! * `tcgen05.ld` (tensor memory -> registers),
//! * `tcgen05.alloc/dealloc` (tensor memory allocation),
//! * `tcgen05.commit` (UMMA -> mbarrier completion signaling),
//! * UMMA shared-memory descriptors (SM100 version) and block-scaled
//!   instruction descriptors,
//! * 2-CTA TMA loads (`cp.async.bulk.tensor.*.cta_group::2`),
//! * the full SM100 block scheduler (all 7 GemmTypes incl. k-grouped and
//!   psum layouts),
//! * UE8M0 scale-factor math for the FP8/FP4 quantization epilogues.

use super::common::COMMON_HEADER;

/// tcgen05 PTX layer + UMMA descriptors + scheduler + math (part 1: PTX).
pub const SM100_PTX: &str = r#"
// ============================================================================
// deepgemm-rust :: SM100 (Blackwell) tcgen05 device layer, NVRTC-compatible
// ============================================================================
#define DG_CACHE_HINT_EVICT_NORMAL 0x1000000000000000ull
#define DG_PEER_BIT_MASK 0xFEFFFFFFu   // clear the cluster peer-CTA bit

// ------------------------------ tcgen05 MMA --------------------------------
// Block-scaled MXF8F6F4 (FP8 x FP8 or unpacked FP4), UMMA_K = 32.
DG_INLINE void dg100_mma_mxf8f6f4_1sm(uint64_t desc_a, uint64_t desc_b, uint32_t tmem_c,
                                      uint32_t scale_c, uint64_t instr_desc,
                                      uint32_t tmem_sfa, uint32_t tmem_sfb) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%5], [%6], p; \n"
        "}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(static_cast<uint32_t>(instr_desc >> 32)),
           "r"(scale_c), "r"(tmem_sfa), "r"(tmem_sfb));
}
DG_INLINE void dg100_mma_mxf8f6f4_2sm(uint64_t desc_a, uint64_t desc_b, uint32_t tmem_c,
                                      uint32_t scale_c, uint64_t instr_desc,
                                      uint32_t tmem_sfa, uint32_t tmem_sfb) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%5], [%6], p; \n"
        "}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(static_cast<uint32_t>(instr_desc >> 32)),
           "r"(scale_c), "r"(tmem_sfa), "r"(tmem_sfb));
}

// Block-scaled MXF4 (packed FP4 x FP4, K-major only), UMMA_K = 64.
// CUDA >= 12.9 spells the 32-element SF granularity `.block32`; older
// toolchains use the equivalent `.scale_vec::2X`.
DG_INLINE void dg100_mma_mxf4_1sm(uint64_t desc_a, uint64_t desc_b, uint32_t tmem_c,
                                  uint32_t scale_c, uint64_t instr_desc,
                                  uint32_t tmem_sfa, uint32_t tmem_sfb) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
#if defined(__CUDACC_VER_MAJOR__) && ( (__CUDACC_VER_MAJOR__ > 12) || \
    (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9) )
        "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.block32 [%0], %1, %2, %3, [%5], [%6], p; \n"
#else
        "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X [%0], %1, %2, %3, [%5], [%6], p; \n"
#endif
        "}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(static_cast<uint32_t>(instr_desc >> 32)),
           "r"(scale_c), "r"(tmem_sfa), "r"(tmem_sfb));
}
DG_INLINE void dg100_mma_mxf4_2sm(uint64_t desc_a, uint64_t desc_b, uint32_t tmem_c,
                                  uint32_t scale_c, uint64_t instr_desc,
                                  uint32_t tmem_sfa, uint32_t tmem_sfb) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
#if defined(__CUDACC_VER_MAJOR__) && ( (__CUDACC_VER_MAJOR__ > 12) || \
    (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9) )
        "tcgen05.mma.cta_group::2.kind::mxf4.block_scale.block32 [%0], %1, %2, %3, [%5], [%6], p; \n"
#else
        "tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X [%0], %1, %2, %3, [%5], [%6], p; \n"
#endif
        "}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(static_cast<uint32_t>(instr_desc >> 32)),
           "r"(scale_c), "r"(tmem_sfa), "r"(tmem_sfb));
}

// Plain (non-scaled) F16/BF16 MMA, UMMA_K = 16.
DG_INLINE void dg100_mma_f16bf16_1sm(uint64_t desc_a, uint64_t desc_b, uint32_t tmem_c,
                                     uint32_t scale_c, uint64_t instr_desc) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p; \n"
        "}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(static_cast<uint32_t>(instr_desc >> 32)),
           "r"(scale_c));
}
DG_INLINE void dg100_mma_f16bf16_2sm(uint64_t desc_a, uint64_t desc_b, uint32_t tmem_c,
                                     uint32_t scale_c, uint64_t instr_desc) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p; \n"
        "}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(static_cast<uint32_t>(instr_desc >> 32)),
           "r"(scale_c));
}

// --------------------------- tensor memory ---------------------------------
// Allocate `n_cols` tensor memory columns; the base address is written to
// `dst_smem_ptr` (a uint32 in shared memory). Issued by one full warp.
DG_INLINE void dg100_tmem_alloc_1sm(uint32_t n_cols, uint32_t* dst_smem_ptr) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(smem_u32(dst_smem_ptr)), "r"(n_cols));
}
DG_INLINE void dg100_tmem_alloc_2sm(uint32_t n_cols, uint32_t* dst_smem_ptr) {
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(smem_u32(dst_smem_ptr)), "r"(n_cols));
}
DG_INLINE void dg100_tmem_dealloc_1sm(uint32_t tmem_ptr, uint32_t n_cols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(tmem_ptr), "r"(n_cols));
}
DG_INLINE void dg100_tmem_dealloc_2sm(uint32_t tmem_ptr, uint32_t n_cols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
                 :: "r"(tmem_ptr), "r"(n_cols));
}
DG_INLINE void dg100_tmem_relinquish_1sm() {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}
DG_INLINE void dg100_tmem_relinquish_2sm() {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
}

// UTCCP: async shared -> tensor memory copy used for scale factors.
// `sf_desc` is an SM100 shared-memory descriptor (SWIZZLE_NONE, atom 8x128b).
DG_INLINE void dg100_utccp_1sm(uint64_t sf_desc, uint32_t tmem_col) {
    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                 :: "r"(tmem_col), "l"(sf_desc));
}
DG_INLINE void dg100_utccp_2sm(uint64_t sf_desc, uint32_t tmem_col) {
    asm volatile("tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                 :: "r"(tmem_col), "l"(sf_desc));
}

// Tensor memory loads: each warp reads its own 32 datapath lanes.
// 32x32b x N: N consecutive 32-bit columns per lane.
DG_INLINE void dg100_tmem_load_32dp32b_x4(uint32_t taddr, uint32_t& r0, uint32_t& r1,
                                          uint32_t& r2, uint32_t& r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(taddr));
}
DG_INLINE void dg100_tmem_load_32dp32b_x8(uint32_t taddr, uint32_t* r) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]),
                   "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]) : "r"(taddr));
}
DG_INLINE void dg100_tmem_load_32dp32b_x16(uint32_t taddr, uint32_t* r) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]),
                   "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                   "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]),
                   "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]) : "r"(taddr));
}
DG_INLINE void dg100_tmem_load_32dp32b_x32(uint32_t taddr, uint32_t* r) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15,"
                 " %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]),
                   "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                   "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]),
                   "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]),
                   "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]),
                   "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]),
                   "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]),
                   "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31]) : "r"(taddr));
}
DG_INLINE void dg100_tmem_load_32dp32b_x64(uint32_t taddr, uint32_t* r) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x64.b32 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15,"
                 " %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31,"
                 " %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47,"
                 " %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]),
                   "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                   "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]),
                   "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]),
                   "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]),
                   "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]),
                   "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]),
                   "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31]),
                   "=r"(r[32]), "=r"(r[33]), "=r"(r[34]), "=r"(r[35]),
                   "=r"(r[36]), "=r"(r[37]), "=r"(r[38]), "=r"(r[39]),
                   "=r"(r[40]), "=r"(r[41]), "=r"(r[42]), "=r"(r[43]),
                   "=r"(r[44]), "=r"(r[45]), "=r"(r[46]), "=r"(r[47]),
                   "=r"(r[48]), "=r"(r[49]), "=r"(r[50]), "=r"(r[51]),
                   "=r"(r[52]), "=r"(r[53]), "=r"(r[54]), "=r"(r[55]),
                   "=r"(r[56]), "=r"(r[57]), "=r"(r[58]), "=r"(r[59]),
                   "=r"(r[60]), "=r"(r[61]), "=r"(r[62]), "=r"(r[63]) : "r"(taddr));
}
// 16x256b: each lane receives 2 rows x 2 columns (STSM-friendly layout).
DG_INLINE void dg100_tmem_load_16dp256b_x1(uint32_t taddr, uint32_t& r0, uint32_t& r1,
                                           uint32_t& r2, uint32_t& r3) {
    asm volatile("tcgen05.ld.sync.aligned.16x256b.x1.b32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(taddr));
}

// Wait for all outstanding tcgen05.ld of this warp.
DG_INLINE void dg100_tmem_wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

// --------------------- UMMA completion signaling ---------------------------
// tcgen05.commit: make the mbarrier track completion of all prior MMAs of
// this CTA (leader-CTA barrier for the 2-CTA multicast form).
DG_INLINE void dg100_umma_arrive_1sm(void* bar) {
    if (dg_elect_one()) {
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                     :: "r"(smem_u32(bar)) : "memory");
    }
}
DG_INLINE void dg100_umma_arrive_2sm(void* bar, uint16_t cta_mask) {
    if (dg_elect_one()) {
        asm volatile(
            "{\n"
            "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1; \n"
            "}\n"
            :: "r"(smem_u32(bar)), "h"(cta_mask) : "memory");
    }
}

// ------------------------------ tcgen05 fences -----------------------------
DG_INLINE void dg100_before_thread_sync() { asm volatile("tcgen05.fence::before_thread_sync;"); }
DG_INLINE void dg100_after_thread_sync()  { asm volatile("tcgen05.fence::after_thread_sync;"); }

// ------------------------- 2-CTA TMA loads ---------------------------------
// cta_group::2 form: both CTAs issue the copy; transaction bytes update the
// leader CTA's barrier (peer bit cleared).
DG_INLINE void dg100_tma_load_2d_2sm(const TmaDescriptor* desc, void* bar, void* smem_dst,
                                     uint32_t c0, uint32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
        " [%0], [%1, {%3, %4}], [%2], %5;"
        :: "r"(smem_u32(smem_dst)), "l"(reinterpret_cast<uint64_t>(desc)),
           "r"(smem_u32(bar) & DG_PEER_BIT_MASK), "r"(c0), "r"(c1),
           "l"(static_cast<uint64_t>(DG_CACHE_HINT_EVICT_NORMAL))
        : "memory");
}
DG_INLINE void dg100_tma_load_3d_2sm(const TmaDescriptor* desc, void* bar, void* smem_dst,
                                     uint32_t c0, uint32_t c1, uint32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
        " [%0], [%1, {%3, %4, %5}], [%2], %6;"
        :: "r"(smem_u32(smem_dst)), "l"(reinterpret_cast<uint64_t>(desc)),
           "r"(smem_u32(bar) & DG_PEER_BIT_MASK), "r"(c0), "r"(c1), "r"(c2),
           "l"(static_cast<uint64_t>(DG_CACHE_HINT_EVICT_NORMAL))
        : "memory");
}

// TMA reduce-add stores (for the accumulate epilogue): D[gmem] += SMEM tile.
DG_INLINE void dg100_tma_reduce_add_2d(const TmaDescriptor* desc, const void* smem_src,
                                       uint32_t c0, uint32_t c1) {
    asm volatile(
        "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.bulk_group"
        " [%0, {%2, %3}], [%1];"
        :: "l"(reinterpret_cast<uint64_t>(desc)), "r"(smem_u32(smem_src)), "r"(c0), "r"(c1)
        : "memory");
}
DG_INLINE void dg100_tma_reduce_add_3d(const TmaDescriptor* desc, const void* smem_src,
                                       uint32_t c0, uint32_t c1, uint32_t c2) {
    asm volatile(
        "cp.reduce.async.bulk.tensor.3d.global.shared::cta.add.bulk_group"
        " [%0, {%2, %3, %4}], [%1];"
        :: "l"(reinterpret_cast<uint64_t>(desc)), "r"(smem_u32(smem_src)),
           "r"(c0), "r"(c1), "r"(c2)
        : "memory");
}

// -------------------------- misc shared memory -----------------------------
DG_INLINE uint32_t dg_ld_shared_u32(const uint32_t* ptr) { return *ptr; }
DG_INLINE void dg_st_shared_u32(uint32_t* ptr, uint32_t v) { *ptr = v; }

// stmatrix.x4.trans (b16, 8x8 x4) — swap-AB BF16 epilogue.
DG_INLINE void dg_stsm_x4_trans(uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3,
                                void* smem_ptr) {
    asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16.trans [%0], {%1, %2, %3, %4};\n"
                 :: "r"(smem_u32(smem_ptr)), "r"(r0), "r"(r1), "r"(r2), "r"(r3));
}
// stmatrix.x2 m16n8 trans (b8) — swap-AB FP8 epilogue.
DG_INLINE void dg_stsm_u8x8_x2_trans(uint32_t r0, uint32_t r1, void* smem_ptr) {
    asm volatile("stmatrix.sync.aligned.m16n8.x2.trans.shared.b8 [%0], {%1, %2};\n"
                 :: "r"(smem_u32(smem_ptr)), "r"(r0), "r"(r1));
}

// Stochastic-rounding BF16 cast: cvt.rs.bf16x2.f32 (sm_100a target required).
DG_INLINE uint32_t dg_cvt_rs_bf16x2_f32(float lower, float upper, uint32_t random_bits) {
    uint32_t packed;
    asm volatile("cvt.rs.bf16x2.f32 %0, %1, %2, %3;\n"
                 : "=r"(packed) : "f"(upper), "f"(lower), "r"(random_bits));
    return packed;
}

// Two floats -> one packed fp8x2 e4m3 halfword (round-to-nearest, saturating).
// NOTE: the packed destination is 16-bit (two 8-bit fp8 values).
DG_INLINE uint32_t dg_cvt_fp8x2_f32(float lo, float hi) {
    uint16_t r;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(r) : "f"(lo), "f"(hi));
    return static_cast<uint32_t>(r);
}
"#;

/// UMMA shared-memory + instruction descriptors, TMA copy dispatch,
/// the full SM100 scheduler and UE8M0 math (part 2: descriptors & logic).
pub const SM100_DESC: &str = r#"
// ============================================================================
// UMMA shared-memory descriptors (SM100 format)
// ============================================================================
// 64-bit layout (SM100 "version 1" descriptors):
//   [ 0,14) start_address >> 4        [16,30) LBO >> 4       [32,46) SBO >> 4
//   [46,48) version (= 1)             [49,52) base_offset    [52]    lbo_mode
//   [61,64) layout type: 0 = none, 1 = 128B-base32B, 2 = 128B, 4 = 64B, 6 = 32B
#define DG_UL_NONE     0u
#define DG_UL_BASE32B  1u
#define DG_UL_128B     2u
#define DG_UL_64B      4u
#define DG_UL_32B      6u

DG_INLINE uint64_t dg100_make_smem_desc(uint32_t layout, const void* smem_ptr,
                                        uint32_t sbo_bytes, uint32_t lbo_bytes) {
    uint64_t desc = 0;
    desc |= static_cast<uint64_t>((smem_u32(smem_ptr) >> 4) & 0x3FFFu);
    desc |= static_cast<uint64_t>((lbo_bytes >> 4) & 0x3FFFu) << 16;
    desc |= static_cast<uint64_t>((sbo_bytes >> 4) & 0x3FFFu) << 32;
    desc |= static_cast<uint64_t>(1u) << 46;          // version = 1 (SM100)
    // base_offset = 0, lbo_mode = 0 (legacy)
    desc |= static_cast<uint64_t>(layout) << 61;
    return desc;
}

// SF descriptor for UTCCP: K-major SWIZZLE_NONE, atom 8 x 128 bits,
// {SBO, LBO} = byte stride between atoms on {MN, K}; only one atom on K.
DG_INLINE uint64_t dg100_make_sf_desc(const void* smem_ptr) {
    return dg100_make_smem_desc(DG_UL_NONE, smem_ptr, 8 * 16, 0);
}
DG_INLINE uint64_t dg100_sf_desc_set_addr(uint64_t desc, const void* smem_ptr) {
    desc &= ~static_cast<uint64_t>(0x3FFFu);
    desc |= static_cast<uint64_t>((smem_u32(smem_ptr) >> 4) & 0x3FFFu);
    return desc;
}
DG_INLINE uint64_t dg100_desc_with_lo(uint64_t desc, uint32_t lo) {
    return (desc & 0xFFFFFFFF00000000ull) | static_cast<uint64_t>(lo);
}
DG_INLINE uint32_t dg100_desc_lo(uint64_t desc) { return static_cast<uint32_t>(desc); }

// Inner (contiguous-dim) swizzle atom size in *storage* elements.
template <uint32_t BLOCK_INNER, uint32_t SW, uint32_t PACK, uint32_t ESZ>
DG_INLINE uint32_t dg100_inner_atom_storage() {
    return SW == 0 ? BLOCK_INNER / PACK : SW / ESZ;
}

// UMMA descriptor for a K-major operand tile.
// Requires: SW * PACK == BLOCK_K * ESZ (exactly one swizzle atom along K).
template <uint32_t BLOCK_MN, uint32_t BLOCK_K, uint32_t SW, uint32_t PACK, uint32_t ESZ>
DG_INLINE uint64_t dg100_make_umma_desc_k_major(const void* base_smem_ptr,
                                                uint32_t mn_idx, uint32_t k_idx) {
    const uint32_t layout = (SW == 32) ? DG_UL_32B : (SW == 64) ? DG_UL_64B : DG_UL_128B;
    const uint32_t atom_base = 16;                    // (BASE32B unused on this path)
    const uint32_t num_non_contiguous = 128 / atom_base;
    const uint32_t sbo = num_non_contiguous * BLOCK_K * ESZ / PACK;
    const uint32_t lbo = 0;
    const uint32_t byte_off = (mn_idx * BLOCK_K + k_idx) * ESZ / PACK;
    const uint8_t* ptr = reinterpret_cast<const uint8_t*>(base_smem_ptr) + byte_off;
    return dg100_make_smem_desc(SW == 0 ? DG_UL_NONE : layout, ptr, sbo, lbo);
}

// UMMA descriptor for an MN-major operand tile (no sub-byte packing).
template <uint32_t BLOCK_MN, uint32_t BLOCK_K, uint32_t SW, uint32_t ESZ>
DG_INLINE uint64_t dg100_make_umma_desc_mn_major(void* base_smem_ptr,
                                                 uint32_t mn_idx, uint32_t k_idx) {
    const uint32_t block_mn_atom = SW / ESZ;          // SW > 0 on MN-major paths
    const uint32_t num_non_contiguous = 128 / 16;
    uint32_t sbo = num_non_contiguous * block_mn_atom * ESZ;   // stride on K
    uint32_t lbo = BLOCK_K * block_mn_atom * ESZ;              // stride on MN
    if (SW == 16) { uint32_t t = sbo; sbo = lbo; lbo = t; }    // interleave, no swizzle
    const uint32_t layout = SW == 16 ? DG_UL_NONE
                          : SW == 32 ? DG_UL_32B
                          : SW == 64 ? DG_UL_64B : DG_UL_128B;
    uint8_t* ptr = reinterpret_cast<uint8_t*>(base_smem_ptr) +
                   (mn_idx * BLOCK_K + k_idx * block_mn_atom) * ESZ;
    return dg100_make_smem_desc(layout, ptr, sbo, lbo);
}

// Advance the low 32 bits of a descriptor by a {offset, k_idx} byte offset.
// `stride_k` = 1 for K-major, inner atom size for MN-major.
DG_INLINE uint32_t dg100_advance_desc_lo(uint32_t base_lo, uint32_t byte_off) {
    return base_lo + (byte_off >> 4);
}

// ============================================================================
// Block-scaled instruction descriptors (32-bit, embedded into the 64-bit
// runtime descriptor; the high half is passed to the MMA instruction)
// ============================================================================
// UMMA operand formats: 0 = E4M3, 1 = E5M2, 3 = E2M3, 4 = E3M2, 5 = E2M1,
// F32F16 family: 0 = F16, 1 = BF16, 2 = TF32.
#define DG_UFMT_E4M3 0u
#define DG_UFMT_E5M2 1u
#define DG_UFMT_E2M1 5u
#define DG_UFMT_BF16 1u

// InstrDescriptorBlockScaled:
//   [0,2) sparse_id2  [2,3) sparse_flag  [4,6) b_sf_id  [7,10) a_format
//   [10,13) b_format  [13,14) a_negate   [14,15) b_negate  [15] a_major(MN)
//   [16] b_major(MN)  [17,23) n_dim (N>>3)  [23] scale_format (0=E4M3, 1=E8M0)
//   [24,29) m_dim (M>>4)  [29,31) a_sf_id  [31] k_size (0 = dense K32/K64)
DG_INLINE uint32_t dg100_instr_desc_bs(uint32_t a_fmt, uint32_t b_fmt,
                                       uint32_t m_dim, uint32_t n_dim,
                                       uint32_t a_major_mn, uint32_t b_major_mn,
                                       uint32_t sf_ue8m0) {
    uint32_t d = 0;
    d |= (a_fmt & 7u) << 7;
    d |= (b_fmt & 7u) << 10;
    d |= (a_major_mn & 1u) << 15;
    d |= (b_major_mn & 1u) << 16;
    d |= ((n_dim >> 3) & 0x3Fu) << 17;
    d |= (sf_ue8m0 & 1u) << 23;
    d |= ((m_dim >> 4) & 0x1Fu) << 24;
    return d;
}
// Patch the runtime SF ids (which packed SF byte of the TMEM word applies).
DG_INLINE uint64_t dg100_instr_desc_with_sf_id(uint32_t desc32, uint32_t sfa_id, uint32_t sfb_id) {
    desc32 &= ~((3u << 29) | (3u << 4));
    desc32 |= (sfa_id & 3u) << 29;
    desc32 |= (sfb_id & 3u) << 4;
    return static_cast<uint64_t>(desc32) << 32;
}
DG_INLINE void dg100_instr_desc_set_n(uint32_t& desc32, uint32_t n_dim) {
    desc32 &= ~(0x3Fu << 17);
    desc32 |= ((n_dim >> 3) & 0x3Fu) << 17;
}

// Plain InstrDescriptor (F16/BF16 kind):
//   [4,6) c_format (0=F16, 1=F32, 2=S32) ... [30,32) max_shift
DG_INLINE uint32_t dg100_instr_desc_f16(uint32_t a_fmt, uint32_t b_fmt,
                                        uint32_t m_dim, uint32_t n_dim,
                                        uint32_t a_major_mn, uint32_t b_major_mn) {
    uint32_t d = 0;
    d |= (1u) << 4;                                   // c_format = F32
    d |= (a_fmt & 7u) << 7;
    d |= (b_fmt & 7u) << 10;
    d |= (a_major_mn & 1u) << 15;
    d |= (b_major_mn & 1u) << 16;
    d |= ((n_dim >> 3) & 0x3Fu) << 17;
    d |= ((m_dim >> 4) & 0x1Fu) << 24;
    return d;
}
DG_INLINE uint64_t dg100_make_runtime_desc(uint32_t desc32) {
    return static_cast<uint64_t>(desc32) << 32;
}

// ============================================================================
// TMA copy dispatch (upstream `tma::copy_nd`): loads a
// `BLOCK_OUTER x BLOCK_INNER` box of a 2-D or 3-D tensor map at
// (inner_idx, outer_idx[, batch_idx]). With multicast == 2 the cta_group::2
// form is used (both CTAs of the pair issue it; the leader barrier counts
// the bytes of both).
// ============================================================================
template <uint32_t BLOCK_INNER, uint32_t BLOCK_OUTER, uint32_t SW,
          uint32_t PACK, uint32_t ESZ, bool IS3D>
DG_INLINE void dg100_tma_copy(const TmaDescriptor* desc, void* bar, void* smem_ptr,
                              uint32_t num_tma_multicast,
                              uint32_t inner_idx, uint32_t outer_idx, uint32_t batch_idx) {
    constexpr uint32_t ATOM_STORAGE = (SW == 0) ? (BLOCK_INNER / PACK) : (SW / ESZ);
    constexpr uint32_t ATOM_LOGICAL = ATOM_STORAGE * PACK;
    static_assert(BLOCK_INNER % ATOM_LOGICAL == 0, "TMA inner block must contain whole atoms");
    const uint32_t num_atoms = BLOCK_INNER / ATOM_LOGICAL;
    // Storage-stride between atoms: BLOCK_OUTER rows of ATOM_STORAGE elements.
    const uint32_t atom_stride = BLOCK_OUTER * ATOM_STORAGE * ESZ;
    if (num_tma_multicast == 1) {
        char* dst = reinterpret_cast<char*>(smem_ptr);
        if (IS3D) {
            for (uint32_t i = 0; i < num_atoms; ++i)
                dg_tma_load_3d(desc, bar, dst + i * atom_stride,
                               inner_idx + i * ATOM_LOGICAL, outer_idx, batch_idx);
        } else {
            for (uint32_t i = 0; i < num_atoms; ++i)
                dg_tma_load_2d(desc, bar, dst + i * atom_stride,
                               inner_idx + i * ATOM_LOGICAL, outer_idx);
        }
    } else {
        char* dst = reinterpret_cast<char*>(smem_ptr);
        if (IS3D) {
            for (uint32_t i = 0; i < num_atoms; ++i)
                dg100_tma_load_3d_2sm(desc, bar, dst + i * atom_stride,
                                      inner_idx + i * ATOM_LOGICAL, outer_idx, batch_idx);
        } else {
            for (uint32_t i = 0; i < num_atoms; ++i)
                dg100_tma_load_2d_2sm(desc, bar, dst + i * atom_stride,
                                      inner_idx + i * ATOM_LOGICAL, outer_idx);
        }
    }
}

// GemmType enum values (upstream `deep_gemm::GemmType`).
#define DG_GT_NORMAL 0u
#define DG_GT_MGROUPED_CONTIG 1u
#define DG_GT_MGROUPED_MASKED 2u
#define DG_GT_KGROUPED_CONTIG 3u
#define DG_GT_BATCHED 4u
#define DG_GT_MGROUPED_CONTIG_PSUM 5u
#define DG_GT_KGROUPED_CONTIG_PSUM 6u
"#;

/// The full SM100 block scheduler: Normal, MGroupedContiguous(+Psum),
/// MGroupedMasked, KGroupedContiguous(+Psum), Batched — with L2 swizzling.
pub const SM100_SCHED: &str = r#"
// ============================================================================
// SM100 block scheduler (all GemmTypes)
// ============================================================================
// GEMM_TYPE: 0=Normal 1=MGroupedContiguous 2=MGroupedMasked 3=KGroupedContiguous
//            4=Batched 5=MGroupedContiguousWithPsumLayout 6=KGroupedContiguousWithPsumLayout

// Index types for get_global_idx
#define DG_IX_MN 0
#define DG_IX_K 1
#define DG_IX_SFK 2

// Upstream: pick 8 or 16 blocks per L2 group by a usage model (constexpr:
// usable in static_asserts).
DG_INLINE constexpr uint32_t dg_num_1d_blocks_per_group() {
    uint32_t num_best = 0, min_usage = 0xFFFFFFFFu;
    for (uint32_t candidate = 8; candidate <= 16; candidate += 8) {
        const uint32_t usage = IS_MULTICAST_ON_A
            ? candidate * BLOCK_N + dg_ceil_div(NUM_SMS, candidate) * BLOCK_M
            : candidate * BLOCK_M + dg_ceil_div(NUM_SMS, candidate) * BLOCK_N;
        if (usage < min_usage) min_usage = usage, num_best = candidate;
    }
    return num_best;
}

struct DgSched {
    int current_iter = -1;

    uint32_t num_blocks;
    uint32_t num_m_blocks;
    uint32_t num_n_blocks;

    // For SM90 multicast checks (kept for parity; SM100 always multicasts)
    uint32_t num_blocks_in_group;
    bool is_peer_cta_alive = true;

    // Grouped GEMM state
    int* grouped_layout;
    uint32_t current_group_idx = 0;
    uint32_t current_m_cumsum = 0;              // masked
    uint32_t last_psum_m = 0, current_psum_m, current_m_block_cumsum = 0;  // m-grouped psum
    uint32_t current_shape_k, current_k_start = 0, current_sf_k_cumsum = 0;  // k-grouped
    uint32_t current_k_end = 0;                 // k-grouped psum

    // Load the K-group selected by `current_group_idx`.
    DG_INLINE void get_next_k_group() {
#if DG_IS_K_GROUPED_CONTIG && GEMM_TYPE == 6
        // grouped_layout[i] = psum end offset in K elements
        const uint32_t next_k_end = static_cast<uint32_t>(grouped_layout[current_group_idx]);
        current_k_start = (current_k_end + K_ALIGNMENT - 1) / K_ALIGNMENT * K_ALIGNMENT;
        current_shape_k = next_k_end - current_k_start;
        current_k_end = next_k_end;
#else
        current_k_start += current_shape_k;
        current_shape_k = static_cast<uint32_t>(grouped_layout[current_group_idx]);
#endif
    }

    DG_INLINE DgSched(uint32_t shape_m, uint32_t shape_n, uint32_t shape_k, int* layout) {
        num_m_blocks = dg_ceil_div(shape_m, BLOCK_M);
        num_n_blocks = dg_ceil_div(shape_n, BLOCK_N);
        current_shape_k = DG_IS_K_GROUPED_CONTIG ? 0 : shape_k;
        grouped_layout = layout;
#if GEMM_TYPE == DG_GT_NORMAL || GEMM_TYPE == DG_GT_BATCHED
        num_blocks = num_m_blocks * num_n_blocks;
#elif GEMM_TYPE == DG_GT_MGROUPED_CONTIG
        num_blocks = num_m_blocks * num_n_blocks;
#elif GEMM_TYPE == DG_GT_MGROUPED_MASKED
        // num_blocks resolved dynamically per group
#elif GEMM_TYPE == DG_GT_MGROUPED_CONTIG_PSUM
        current_psum_m = static_cast<uint32_t>(grouped_layout[0]);
        num_m_blocks = dg_ceil_div(current_psum_m, BLOCK_M);
#elif DG_IS_K_GROUPED_CONTIG
        num_blocks = num_m_blocks * num_n_blocks;
        get_next_k_group();
#endif
    }

    DG_INLINE void get_swizzled_block_idx(uint32_t block_idx, uint32_t& m_block_idx,
                                          uint32_t& n_block_idx) {
        static_assert(NUM_1D_BLOCKS_PER_GROUP % MULTICAST == 0, "Invalid group size");
        const uint32_t primary_num_blocks = IS_MULTICAST_ON_A ? num_n_blocks : num_m_blocks;
        const uint32_t secondary_num_blocks = IS_MULTICAST_ON_A ? num_m_blocks : num_n_blocks;
        const uint32_t num_blocks_per_group = secondary_num_blocks * NUM_1D_BLOCKS_PER_GROUP;
        const uint32_t group_idx = block_idx / num_blocks_per_group;
        uint32_t first_block_idx = group_idx * NUM_1D_BLOCKS_PER_GROUP;
        uint32_t in_group_idx = block_idx % num_blocks_per_group;
        num_blocks_in_group = NUM_1D_BLOCKS_PER_GROUP;
        if (primary_num_blocks < first_block_idx)
            num_blocks_in_group = 0;
        else if (primary_num_blocks - first_block_idx < num_blocks_in_group)
            num_blocks_in_group = primary_num_blocks - first_block_idx;

        // (SM90-only dynamic multicast disabling is not needed on SM100's 2-CTA.)
        if (IS_MULTICAST_ON_A) {
            m_block_idx = in_group_idx / num_blocks_in_group;
            n_block_idx = first_block_idx + in_group_idx % num_blocks_in_group;
        } else {
            m_block_idx = first_block_idx + in_group_idx % num_blocks_in_group;
            n_block_idx = in_group_idx / num_blocks_in_group;
        }
    }

    // Resolve a block index to a global coordinate, applying group offsets.
    template <bool WITH_GROUP_OFFSET, uint32_t IX_TYPE>
    DG_INLINE uint32_t get_global_idx(uint32_t shape_dim, uint32_t block_size,
                                      uint32_t block_idx, uint32_t m_block_idx = 0) {
#if GEMM_TYPE == DG_GT_NORMAL
        return block_idx * block_size;
#elif GEMM_TYPE == DG_GT_MGROUPED_CONTIG
        const int32_t offset = WITH_GROUP_OFFSET
            ? dg_max_i32(0, grouped_layout[m_block_idx * BLOCK_M]) : 0;
        return static_cast<uint32_t>(offset) * shape_dim + block_idx * block_size;
#elif GEMM_TYPE == DG_GT_MGROUPED_MASKED || GEMM_TYPE == DG_GT_MGROUPED_CONTIG_PSUM
        const uint32_t offset = WITH_GROUP_OFFSET ? current_group_idx : 0;
        return offset * shape_dim + block_idx * block_size;
#elif DG_IS_K_GROUPED_CONTIG
        uint32_t offset = 0;
        if (WITH_GROUP_OFFSET) {
            if (IX_TYPE == DG_IX_MN) offset = current_group_idx * shape_dim;
            else if (IX_TYPE == DG_IX_K) offset = current_k_start;
            else offset = current_sf_k_cumsum;
        }
        return offset + block_idx * block_size;
#elif GEMM_TYPE == DG_GT_BATCHED
        const uint32_t offset = IX_TYPE == DG_IX_SFK ? current_group_idx : 0;
        return offset * shape_dim + block_idx * block_size;
#endif
    }

    // For swap A/B and psum layouts: the aligned effective M of this block.
    DG_INLINE uint32_t get_aligned_effective_m_in_block(uint32_t m_block_idx) const {
        // UMMA_STEP_N = 16
#if GEMM_TYPE == DG_GT_MGROUPED_CONTIG_PSUM && !ENSURE_ZERO_PADDING
        const uint32_t eff = (m_block_idx == last_psum_m / BLOCK_M + num_m_blocks - 1)
                             ? (current_psum_m - m_block_idx * BLOCK_M) : BLOCK_M;
        return (eff + 15) / 16 * 16;
#else
        (void) m_block_idx;
        return BLOCK_M;
#endif
    }

    DG_INLINE bool get_next_block(uint32_t& m_block_idx, uint32_t& n_block_idx) {
        const uint32_t next_block_idx = static_cast<uint32_t>(++ current_iter) * NUM_SMS + blockIdx.x;

#if GEMM_TYPE == DG_GT_MGROUPED_MASKED
        while (true) {
            if (current_group_idx == NUM_GROUPS) return false;
            num_m_blocks = dg_ceil_div(static_cast<uint32_t>(grouped_layout[current_group_idx]), BLOCK_M);
            const uint32_t current_m_block_cumsum = current_m_cumsum + num_m_blocks;
            if (next_block_idx < current_m_block_cumsum * num_n_blocks) break;
            current_group_idx ++, current_m_cumsum = current_m_block_cumsum;
        }
        get_swizzled_block_idx(next_block_idx - current_m_cumsum * num_n_blocks,
                               m_block_idx, n_block_idx);
#elif GEMM_TYPE == DG_GT_MGROUPED_CONTIG_PSUM
        while (true) {
            if (next_block_idx < (current_m_block_cumsum + num_m_blocks) * num_n_blocks) break;
            if (++ current_group_idx == NUM_GROUPS) return false;
            last_psum_m = (current_psum_m + BLOCK_M - 1) / BLOCK_M * BLOCK_M;
            current_psum_m = static_cast<uint32_t>(grouped_layout[current_group_idx]);
            current_m_block_cumsum += num_m_blocks;
            num_m_blocks = dg_ceil_div(current_psum_m - last_psum_m, BLOCK_M);
        }
        get_swizzled_block_idx(next_block_idx - current_m_block_cumsum * num_n_blocks,
                               m_block_idx, n_block_idx);
        m_block_idx += last_psum_m / BLOCK_M;
#elif DG_IS_K_GROUPED_CONTIG
        while (true) {
            if (current_group_idx == NUM_GROUPS) return false;
            if (next_block_idx < (current_group_idx + 1) * num_blocks) break;
            current_group_idx ++;
            if (current_group_idx >= NUM_GROUPS) return false;
            const uint32_t aligned_shape_k = (current_shape_k + K_ALIGNMENT - 1)
                                             / K_ALIGNMENT * K_ALIGNMENT;
            current_sf_k_cumsum += dg_ceil_div(aligned_shape_k, SFK_SPAN);
            get_next_k_group();
        }
        get_swizzled_block_idx(next_block_idx - current_group_idx * num_blocks,
                               m_block_idx, n_block_idx);
#elif GEMM_TYPE == DG_GT_BATCHED
        if (next_block_idx >= num_blocks * NUM_GROUPS) return false;
        current_group_idx = next_block_idx / num_blocks;
        const uint32_t block_idx = next_block_idx - current_group_idx * num_blocks;
        if (IS_MULTICAST_ON_A) {
            m_block_idx = block_idx / num_n_blocks;
            n_block_idx = block_idx % num_n_blocks;
        } else {
            m_block_idx = block_idx % num_m_blocks;
            n_block_idx = block_idx / num_m_blocks;
        }
#else
        if (next_block_idx >= num_blocks) return false;
        is_peer_cta_alive = num_n_blocks % MULTICAST == 0 ||
                            num_m_blocks % MULTICAST == 0 ||
                            (next_block_idx ^ 1) < num_blocks;
        get_swizzled_block_idx(next_block_idx, m_block_idx, n_block_idx);
#endif
        return true;
    }

    // Whether this M block computes valid (non-padding) rows.
    DG_INLINE bool is_computation_valid(uint32_t m_block_idx, uint32_t m_offset) const {
#if GEMM_TYPE == DG_GT_NORMAL || GEMM_TYPE == DG_GT_BATCHED
        return true;
#elif GEMM_TYPE == DG_GT_MGROUPED_CONTIG
        return grouped_layout[m_offset + m_block_idx * BLOCK_M] >= 0;
#elif GEMM_TYPE == DG_GT_MGROUPED_MASKED
        return m_offset + m_block_idx * BLOCK_M < static_cast<uint32_t>(grouped_layout[current_group_idx]);
#elif GEMM_TYPE == DG_GT_MGROUPED_CONTIG_PSUM
        return m_offset + m_block_idx * BLOCK_M < current_psum_m;
#else
        return true;
#endif
    }
};

"#;

/// UE8M0 scale-factor math + bf16 helpers for the SM100 epilogues
/// (part 4: quantization math).
pub const SM100_MATH: &str = r#"
// ============================================================================
// UE8M0 scale-factor math (power-of-two scales) + BF16 helpers
// ============================================================================
// bf16 <-> f32 bit tricks (bf16 = top half of f32).
DG_INLINE float dg_bf16_to_f32(uint16_t v) {
    uint32_t bits = static_cast<uint32_t>(v) << 16;
    return __uint_as_float(bits);
}
DG_INLINE uint32_t dg_bf16x2_bits(uint16_t lo, uint16_t hi) {
    return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}

// |x| of a packed bf16x2 (clear sign bits).
DG_INLINE uint32_t dg_bf16x2_abs(uint32_t v) { return v & 0x7FFF7FFFu; }
// max of two packed bf16x2 (PTX max.bf16x2, sm_80+).
DG_INLINE uint32_t dg_bf16x2_max(uint32_t a, uint32_t b) {
    uint32_t r;
    asm volatile("max.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}
// Tree-reduce the amax of `n` packed bf16x2 values.
template <uint32_t N>
DG_INLINE uint16_t dg_packed_bf16_amax(const uint32_t* values) {
    uint32_t tree[N / 2];
    #pragma unroll
    for (uint32_t i = 0; i < N / 2; ++ i)
        tree[i] = dg_bf16x2_max(dg_bf16x2_abs(values[i]), dg_bf16x2_abs(values[i + N / 2]));
    #pragma unroll
    for (uint32_t stride = N / 4; stride > 0; stride /= 2) {
        #pragma unroll
        for (uint32_t i = 0; i < stride; ++ i)
            tree[i] = dg_bf16x2_max(tree[i], tree[i + stride]);
    }
    // max of the two halves
    const uint16_t x = static_cast<uint16_t>(tree[0] & 0xFFFF);
    const uint16_t y = static_cast<uint16_t>(tree[0] >> 16);
    return (x & 0x7FFF) > (y & 0x7FFF) ? x : y;
}

// Select the UE8M0 exponent mapping `amax` into the finite range of the
// quantized dtype (E4M3 or E2M1); amax is a BF16 bit pattern.
// The integer-add carry performs the exponent ceiling.
template <bool IS_FP8>
DG_INLINE uint32_t dg_ue8m0_sf_exp_bf16(uint16_t amax_bits) {
    constexpr uint32_t kMantissaBits = 7;                   // bf16
    constexpr uint32_t kMantissaMask = (1u << kMantissaBits) - 1;
    constexpr uint32_t kQuantMaxMantissa = (IS_FP8 ? 0x60u : 0x40u) << (kMantissaBits - 7);
    constexpr uint32_t kQuantMaxExponent = IS_FP8 ? 8 : 2;  // 448 or 6
    constexpr uint32_t kMinSFExponent = IS_FP8 ? 105 : 1;   // amax floors
    const uint32_t rounded_exp = (amax_bits + kMantissaMask - kQuantMaxMantissa) >> kMantissaBits;
    const uint32_t sf_exp = (rounded_exp < kQuantMaxExponent ? kQuantMaxExponent : rounded_exp)
                          + 127;
    return sf_exp < kMinSFExponent ? kMinSFExponent : sf_exp;
}

// 2^-sf_exp as an f32 / bf16 bit pattern.
DG_INLINE float dg_ue8m0_sf_inv_f32(uint32_t sf_exp) {
    return __uint_as_float((254u - sf_exp) << 23);
}
DG_INLINE uint16_t dg_ue8m0_sf_inv_bf16(uint32_t sf_exp) {
    return static_cast<uint16_t>((254u - sf_exp) << 7);
}

// Scale two packed bf16x2 pairs by power-of-two f32 sf_invs and pack into
// four E4M3 bytes (matches upstream `scale_bf16x2_into_fp8x4` bit-exactly:
// power-of-two scaling of BF16 is exact in FP32).
DG_INLINE uint32_t dg_scale_bf16x2_into_fp8x4(uint32_t lower, uint32_t upper,
                                              float sf_inv_lo, float sf_inv_hi) {
    const float l0 = dg_bf16_to_f32(static_cast<uint16_t>(lower & 0xFFFF)) * sf_inv_lo;
    const float l1 = dg_bf16_to_f32(static_cast<uint16_t>(lower >> 16)) * sf_inv_lo;
    const float u0 = dg_bf16_to_f32(static_cast<uint16_t>(upper & 0xFFFF)) * sf_inv_hi;
    const float u1 = dg_bf16_to_f32(static_cast<uint16_t>(upper >> 16)) * sf_inv_hi;
    return dg_cvt_fp8x2_f32(l1, l0) | (dg_cvt_fp8x2_f32(u1, u0) << 16);
}

// FP32 amax variant of the UE8M0 exponent selection (host-side quant kernels).
DG_INLINE uint32_t dg_ue8m0_sf_exp_f32(bool is_fp8, float amax) {
    const uint32_t kMantissaBits = 23;
    const uint32_t kMantissaMask = (1u << kMantissaBits) - 1;
    const uint32_t kQuantMaxMantissa = (is_fp8 ? 0x60u : 0x40u) << (kMantissaBits - 7);
    const uint32_t kQuantMaxExponent = is_fp8 ? 8u : 2u;
    const uint32_t kMinSFExponent = is_fp8 ? 105u : 1u;
    const uint32_t amax_bits = __float_as_uint(amax);
    const uint32_t rounded_exp = (amax_bits + kMantissaMask - kQuantMaxMantissa) >> kMantissaBits;
    const uint32_t sf_exp = (rounded_exp < kQuantMaxExponent ? kQuantMaxExponent : rounded_exp) + 127;
    return sf_exp < kMinSFExponent ? kMinSFExponent : sf_exp;
}
"#;

/// PTX + descriptor + math layers (no scheduler: it needs the per-kernel
/// #defines to come first).
pub fn sm100_header() -> String {
    format!("{COMMON_HEADER}{SM100_PTX}{SM100_DESC}{SM100_MATH}")
}

/// The scheduler layer: must be placed AFTER the per-kernel #defines (it is
/// parameterized by BLOCK_M/N, MULTICAST, NUM_SMS, ...).
pub fn sm100_sched() -> String {
    SM100_SCHED.to_string()
}

/// Epilogue operators + helpers + the two store epilogues, shared by the
/// SM100 FP8/FP4 and BF16 GEMM kernels. Requires the enclosing TU to define
/// (via #define): CD_DTYPE, EPILOGUE_OP, BLOCK_M/N, STORE_BLOCK_M/N,
/// SWIZZLE_CD, NUM_TMA_STORE_STAGES, NUM_UMMA_STORE_THREADS,
/// NUM_OVERLAPPED_TMEM_COLS, WITH_ACCUMULATION, IS_3D_TMA,
/// DG_IS_K_GROUPED_CONTIG, DG_ALIGN4, and a preceding `SmemStorage` with a
/// leading `cd` buffer sized `NUM_TMA_STORE_STAGES *
/// DG_ALIGN4(STORE_BLOCK_M * STORE_BLOCK_N * CD_ELEM_SIZE)` bytes.
pub const SM100_EPILOGUE_FNS: &str = r#"
// ---------------------------- epilogue operators ----------------------------
DG_INLINE void epi_apply_values(uint32_t* values, uint32_t n, float alpha) {
#if EPILOGUE_OP == 1
    // ScaleByAlpha
    for (uint32_t i = 0; i < n; i += 2) {
        float2* v = reinterpret_cast<float2*>(values + i);
        *v = __fmul2_rn(*v, make_float2(alpha, alpha));
    }
#else
    (void) values; (void) n; (void) alpha;
#endif
}

// Store one output SF byte (or a whole packed word) to GMEM.
DG_INLINE void epi_store_sf_byte(uint32_t* sfd, uint32_t sfd_stride, uint32_t shape_m,
                                 uint32_t shape_n, uint32_t row_idx, uint32_t group_n_idx,
                                 uint32_t batch_idx, uint32_t sf) {
    if (row_idx >= shape_m || group_n_idx >= shape_n) return;
    const uint32_t sf_idx = (batch_idx * shape_n + group_n_idx) / SF_GRAN_N;
    uint32_t* sf_word_ptr = sfd + (sf_idx / 4) * sfd_stride + row_idx;
    reinterpret_cast<uint8_t*>(sf_word_ptr)[sf_idx % 4] = sf;
}
DG_INLINE void epi_store_sf_word(uint32_t* sfd, uint32_t sfd_stride, uint32_t shape_m,
                                 uint32_t shape_n, uint32_t row_idx, uint32_t group_n_idx,
                                 uint32_t batch_idx, uint32_t sf_word) {
    if (row_idx >= shape_m || group_n_idx >= shape_n) return;
    const uint32_t sf_idx = (batch_idx * shape_n + group_n_idx) / SF_GRAN_N;
    uint32_t* sf_word_ptr = sfd + (sf_idx / 4) * sfd_stride + row_idx;
    *sf_word_ptr = sf_word;
}

// Stochastic-rounding quartet cast (see upstream `StochasticRoundToBF16`).
DG_INLINE uint32_t sr_partial_hash(uint32_t value, uint32_t quartet_offset) {
    return value * (quartet_offset == 0 ? 0x5671d42bu :
                    quartet_offset == 1 ? 0x9995e499u :
                    quartet_offset == 2 ? 0xace1b8a5u : 0xe153538du);
}
DG_INLINE uint32_t sr_select_bits(uint32_t h, uint32_t quartet_offset) {
    h ^= h >> 23;
    h *= 0x7feb352du;
    h ^= h >> 16;
    h *= quartet_offset < 2 ? 0x846ca68bu : 0xd35a2d97u;
    h ^= h >> 11;
    return (h >> (quartet_offset % 2 * 16)) & 0xffffu;
}
DG_INLINE void sr_cast_quartet(uint32_t a, uint32_t b, uint32_t c, uint32_t d,
                               uint32_t quartet_n_idx, uint32_t& packed_ab, uint32_t& packed_cd) {
    const uint32_t h = sr_partial_hash(a, 0) + sr_partial_hash(b, 1) +
                       sr_partial_hash(c, 2) + sr_partial_hash(d, 3) + quartet_n_idx;
    packed_ab = dg_cvt_rs_bf16x2_f32(__uint_as_float(a), __uint_as_float(b),
                                     sr_select_bits(h, 0) | (sr_select_bits(h, 1) << 16));
    packed_cd = dg_cvt_rs_bf16x2_f32(__uint_as_float(c), __uint_as_float(d),
                                     sr_select_bits(h, 2) | (sr_select_bits(h, 3) << 16));
}

// --------------------------------------------------------------------------
// epilogue helpers (defined before the kernel; the store epilogues are
// fully defined after it)
// --------------------------------------------------------------------------
DG_INLINE void dg_st_shared_v4(uint32_t* ptr, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "r"(smem_u32(ptr)), "r"(v0), "r"(v1), "r"(v2), "r"(v3));
}

// In-warp 128 -> 128 SF transpose required by the UTCCP layout.
DG_INLINE void dg_utccp_transpose(uint32_t* smem_ptr) {
    uint32_t values[4];
    #pragma unroll
    for (uint32_t i = 0; i < 4; ++ i)
        values[i] = smem_ptr[i * 32 + dg_lane_idx()];
    __syncwarp();
    dg_st_shared_v4(smem_ptr + dg_lane_idx() * 4, values[0], values[1], values[2], values[3]);
}

DG_INLINE void dg_store_cd(SmemStorage& smem, uint32_t& tma_stage_idx, uint32_t tmem_base_addr,
                            uint32_t base_m_idx, uint32_t base_n_idx, uint32_t batch_idx,
                            bool is_empty_group, uint32_t epilogue_warp_idx, uint32_t lane_idx,
                            uint32_t* sfd, uint32_t sfd_stride, uint32_t epi_shape_m,
                            uint32_t epi_shape_n, float epi_alpha, bool reverse_store_order,
                            void* tmem_overlap_barrier, void* tmem_empty_barrier,
                            const TmaDescriptor& tensor_map_cd);
DG_INLINE void dg_store_cd_swap_ab(SmemStorage& smem, uint32_t& tma_stage_idx, uint32_t tmem_base_addr,
                                    uint32_t base_m_idx, uint32_t base_n_idx, uint32_t batch_idx,
                                    bool is_empty_group, uint32_t effective_m,
                                    uint32_t epilogue_warp_idx, uint32_t lane_idx,
                                    uint32_t* sfd, uint32_t sfd_stride, uint32_t epi_shape_m,
                                    uint32_t epi_shape_n, float epi_alpha, bool reverse_store_order,
                                    void* tmem_overlap_barrier, void* tmem_empty_barrier,
                                    const TmaDescriptor& tensor_map_cd);

// ===========================================================================
// Normal-path store epilogue: BLOCK_M x BLOCK_N tile, TMEM -> registers ->
// swizzled SMEM -> TMA store/reduce-add. (Only compiled when !SWAP_AB.)
// ===========================================================================
#if !SWAP_AB
DG_INLINE void dg_store_cd(SmemStorage& smem, uint32_t& tma_stage_idx, uint32_t tmem_base_addr,
                            uint32_t base_m_idx, uint32_t base_n_idx, uint32_t batch_idx,
                            bool is_empty_group, uint32_t epilogue_warp_idx, uint32_t lane_idx,
                            uint32_t* sfd, uint32_t sfd_stride, uint32_t epi_shape_m,
                            uint32_t epi_shape_n, float epi_alpha, bool reverse_store_order,
                            void* tmem_overlap_barrier, void* tmem_empty_barrier,
                            const TmaDescriptor& tensor_map_cd) {
    constexpr uint32_t kNumBankGroupBytes = 16;
    constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / CD_ELEM_SIZE;
    static_assert(SWIZZLE_CD > 0, "TMA D must be swizzled");
    static_assert(STORE_BLOCK_N % kNumElemsPerBankGroup == 0, "invalid swizzling");
    static_assert(BLOCK_M % STORE_BLOCK_M == 0, "invalid block sizes");
    static_assert(BLOCK_N % STORE_BLOCK_N == 0, "invalid block sizes");
    static_assert(!WITH_OUTPUT_SF || CD_DTYPE == 2, "FP8 output requires an E4M3 D");
    static_assert(!WITH_OUTPUT_SF || STORE_BLOCK_N % SF_GRAN_N == 0, "store must cover SF groups");

    // Iterate over M waves
    constexpr uint32_t kNumMWaves = BLOCK_M / STORE_BLOCK_M;
    #pragma unroll
    for (uint32_t w = 0; w < kNumMWaves; ++ w) {
        constexpr uint32_t kNumStores = BLOCK_N / STORE_BLOCK_N;
        #pragma unroll
        for (uint32_t s = 0; s < kNumStores; ++ s, tma_stage_idx = (tma_stage_idx + 1) % NUM_TMA_STORE_STAGES) {
            const uint32_t store_idx = reverse_store_order ? kNumStores - 1 - s : s;
            uint8_t* smem_base_ptr = &smem.cd[tma_stage_idx * DG_ALIGN4(STORE_BLOCK_M * STORE_BLOCK_N * CD_ELEM_SIZE)];

            // Swizzled SMEM address of the `bank_group_idx`-th bank group in
            // this warp's atom (see upstream `get_swizzled_smem_ptr`).
            auto get_swizzled_smem_ptr = [&](uint32_t bank_group_idx) -> uint32_t* {
                constexpr bool kHasShortcut = (SWIZZLE_CD / kNumBankGroupBytes) == 8;
                const uint32_t shifted_idx = bank_group_idx + lane_idx * (SWIZZLE_CD / kNumBankGroupBytes);
                uint32_t row = kHasShortcut ? (bank_group_idx / 8 + lane_idx) : (shifted_idx / 8);
                uint32_t col = kHasShortcut ? bank_group_idx : (shifted_idx % 8);
                col ^= row % (SWIZZLE_CD / 16);
                return reinterpret_cast<uint32_t*>(smem_base_ptr +
                       epilogue_warp_idx * 32 * SWIZZLE_CD +
                       row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes);
            };

            // Wait for the shared memory to be released
            if (epilogue_warp_idx == 0) dg_tma_store_wait<NUM_TMA_STORE_STAGES - 1>();
            dg_named_barrier_sync(0, NUM_UMMA_STORE_THREADS);

            const uint32_t m_idx = base_m_idx + w * STORE_BLOCK_M;
            const uint32_t n_idx = base_n_idx + store_idx * STORE_BLOCK_N;

            // TMEM loads: one swizzled bank group per lane (or one complete SF
            // group per lane with the FP8 output)
            constexpr uint32_t kNumElemsPerLoad = WITH_OUTPUT_SF ? SF_GRAN_N : kNumElemsPerBankGroup;
            constexpr uint32_t kNumLoads = STORE_BLOCK_N / kNumElemsPerLoad;
            constexpr uint32_t kNumOverlapLoads =
                (NUM_OVERLAPPED_TMEM_COLS + kNumElemsPerLoad - 1) / kNumElemsPerLoad;
            static_assert(kNumOverlapLoads <= kNumLoads, "first store must cover overlapped columns");

            const uint32_t row_idx = m_idx + epilogue_warp_idx * 32 + lane_idx;
            const bool store_whole_sf_word = WITH_OUTPUT_SF && kNumLoads == 4 &&
                                             epi_shape_n % (4 * SF_GRAN_N) == 0;
            uint32_t sf_word = 0;

            #pragma unroll
            for (uint32_t i = 0; i < kNumLoads; ++ i) {
                const uint32_t load_idx = reverse_store_order ? kNumLoads - 1 - i : i;
                const uint32_t tmem_addr = tmem_base_addr +
                                           w * BLOCK_N +
                                           store_idx * STORE_BLOCK_N + load_idx * kNumElemsPerLoad;
                uint32_t values[kNumElemsPerLoad];
#if WITH_OUTPUT_SF
                dg100_tmem_load_32dp32b_x32(tmem_addr, values);
#elif CD_DTYPE == 1
                dg100_tmem_load_32dp32b_x4(tmem_addr, values[0], values[1], values[2], values[3]);
#else
                dg100_tmem_load_32dp32b_x8(tmem_addr, values);
#endif
                dg100_tmem_wait_ld();
                epi_apply_values(values, kNumElemsPerLoad, epi_alpha);
                #pragma unroll
                for (uint32_t value_idx = 0; value_idx < kNumElemsPerLoad; ++ value_idx)
                    values[value_idx] = is_empty_group ? 0u : values[value_idx];

                // Release the overlapped TMEM columns once they are all read
                if (kNumOverlapLoads > 0) {
                    if (w == 0 && s == 0 && i + 1 == kNumOverlapLoads) {
                        dg100_before_thread_sync();
                        dg_mbarrier_arrive_cluster(tmem_overlap_barrier, 0);
                    }
                }
                if (w == kNumMWaves - 1 && s == kNumStores - 1 && i == kNumLoads - 1) {
                    dg100_before_thread_sync();
                    dg_mbarrier_arrive_cluster(tmem_empty_barrier, 0);
                }

                // Store into shared memory
#if WITH_OUTPUT_SF
                {
                    // Round into BF16 pairs first (bitwise-matches a BF16 D
                    // followed by the standalone cast), then amax/scale/cast
                    uint32_t values_bf16x2[kNumElemsPerLoad / 2];
                    #pragma unroll
                    for (uint32_t pair_idx = 0; pair_idx < kNumElemsPerLoad / 2; ++ pair_idx)
                        values_bf16x2[pair_idx] = dg_pack_bf16x2(
                            __uint_as_float(values[pair_idx * 2]), __uint_as_float(values[pair_idx * 2 + 1]));
                    const uint16_t amax = dg_packed_bf16_amax<kNumElemsPerLoad / 2>(values_bf16x2);
                    const uint32_t sf_exp = dg_ue8m0_sf_exp_bf16<true>(amax);
                    const float sf_inv = dg_ue8m0_sf_inv_f32(sf_exp);
                    constexpr uint32_t kNumBankGroupsPerLoad = kNumElemsPerLoad / kNumElemsPerBankGroup;
                    #pragma unroll
                    for (uint32_t bg = 0; bg < kNumBankGroupsPerLoad; ++ bg) {
                        uint32_t* ptr = get_swizzled_smem_ptr(load_idx * kNumBankGroupsPerLoad + bg);
                        const uint32_t p = bg * (kNumElemsPerBankGroup / 2);
                        dg_st_shared_v4(ptr,
                            dg_scale_bf16x2_into_fp8x4(values_bf16x2[p + 0], values_bf16x2[p + 1], sf_inv, sf_inv),
                            dg_scale_bf16x2_into_fp8x4(values_bf16x2[p + 2], values_bf16x2[p + 3], sf_inv, sf_inv),
                            dg_scale_bf16x2_into_fp8x4(values_bf16x2[p + 4], values_bf16x2[p + 5], sf_inv, sf_inv),
                            dg_scale_bf16x2_into_fp8x4(values_bf16x2[p + 6], values_bf16x2[p + 7], sf_inv, sf_inv));
                    }
                    if (store_whole_sf_word)
                        sf_word |= sf_exp << (load_idx * 8);
                    else
                        epi_store_sf_byte(sfd, sfd_stride, epi_shape_m, epi_shape_n,
                                          row_idx, n_idx + load_idx * SF_GRAN_N, batch_idx, sf_exp);
                }
#elif CD_DTYPE == 1
                {   // FP32
                    dg_st_shared_v4(get_swizzled_smem_ptr(load_idx),
                                    values[0], values[1], values[2], values[3]);
                }
#elif WITH_STOCHASTIC
                {
                    static_assert(kNumElemsPerLoad == 8, "stochastic rounding needs 8 BF16 values");
                    const uint32_t quartet_n_idx = (n_idx + load_idx * kNumElemsPerLoad) / 4;
                    uint32_t packed[4];
                    sr_cast_quartet(values[0], values[1], values[2], values[3], quartet_n_idx, packed[0], packed[1]);
                    sr_cast_quartet(values[4], values[5], values[6], values[7], quartet_n_idx + 1, packed[2], packed[3]);
                    dg_st_shared_v4(get_swizzled_smem_ptr(load_idx), packed[0], packed[1], packed[2], packed[3]);
                }
#else
                {   // BF16
                    dg_st_shared_v4(get_swizzled_smem_ptr(load_idx),
                        dg_pack_bf16x2(__uint_as_float(values[0]), __uint_as_float(values[1])),
                        dg_pack_bf16x2(__uint_as_float(values[2]), __uint_as_float(values[3])),
                        dg_pack_bf16x2(__uint_as_float(values[4]), __uint_as_float(values[5])),
                        dg_pack_bf16x2(__uint_as_float(values[6]), __uint_as_float(values[7])));
                }
#endif
            }
#if WITH_OUTPUT_SF
            if (store_whole_sf_word)
                epi_store_sf_word(sfd, sfd_stride, epi_shape_m, epi_shape_n, row_idx, n_idx, batch_idx, sf_word);
#endif

            // Synchronize all threads and issue the TMA
            dg_tma_store_fence();
            dg_named_barrier_sync(0, NUM_UMMA_STORE_THREADS);
            if (epilogue_warp_idx == 0 && dg_elect_one()) {
                if (IS_3D_TMA || DG_IS_K_GROUPED_CONTIG) {
                    if (WITH_ACCUMULATION)
                        dg100_tma_reduce_add_3d(&tensor_map_cd, smem_base_ptr, n_idx, m_idx, batch_idx);
                    else
                        dg_tma_store_3d(&tensor_map_cd, smem_base_ptr, n_idx, m_idx, batch_idx);
                } else {
                    if (WITH_ACCUMULATION)
                        dg100_tma_reduce_add_2d(&tensor_map_cd, smem_base_ptr, n_idx, m_idx);
                    else
                        dg_tma_store_2d(&tensor_map_cd, smem_base_ptr, n_idx, m_idx);
                }
                dg_tma_store_commit_group();
            }
            __syncwarp();
        }
    }
}
#endif  // !SWAP_AB

// ===========================================================================
// Swap-AB store epilogue: output stored transposed; a full warpgroup reads
// all 128 TMEM rows (STORE_BLOCK_N must be 128). (Only compiled when SWAP_AB.)
// ===========================================================================
#if SWAP_AB
DG_INLINE void dg_store_cd_swap_ab(SmemStorage& smem, uint32_t& tma_stage_idx, uint32_t tmem_base_addr,
                                    uint32_t base_m_idx, uint32_t base_n_idx, uint32_t batch_idx,
                                    bool is_empty_group, uint32_t effective_m,
                                    uint32_t epilogue_warp_idx, uint32_t lane_idx,
                                    uint32_t* sfd, uint32_t sfd_stride, uint32_t epi_shape_m,
                                    uint32_t epi_shape_n, float epi_alpha, bool reverse_store_order,
                                    void* tmem_overlap_barrier, void* tmem_empty_barrier,
                                    const TmaDescriptor& tensor_map_cd) {
    constexpr uint32_t kNumBankGroupBytes = 16;
    constexpr uint32_t kNumSwizzleAtomRows = 8;
    constexpr uint32_t STORE_BLOCK_N_ATOM = SWIZZLE_CD / CD_ELEM_SIZE;
    static_assert(SWIZZLE_CD == 128, "TMA D must be 128B swizzled");
    static_assert(STORE_BLOCK_N == 128, "STORE_BLOCK_N must match TMEM rows");
    static_assert(BLOCK_M % STORE_BLOCK_M == 0, "invalid block sizes");
    static_assert(BLOCK_N % STORE_BLOCK_N == 0, "invalid block sizes");
    static_assert(STORE_BLOCK_M % kNumSwizzleAtomRows == 0, "invalid swizzling");
    static_assert(STORE_BLOCK_N % STORE_BLOCK_N_ATOM == 0, "invalid swizzling");

    // Iterate over M blocks (the scheduler aligns the dynamic effective M)
    const uint32_t num_stores = effective_m / STORE_BLOCK_M;
    for (uint32_t s = 0; s < num_stores; ++ s, tma_stage_idx = (tma_stage_idx + 1) % NUM_TMA_STORE_STAGES) {
        const uint32_t store_idx = reverse_store_order ? num_stores - 1 - s : s;
        if (epilogue_warp_idx == 0) dg_tma_store_wait<NUM_TMA_STORE_STAGES - 1>();
        dg_named_barrier_sync(0, NUM_UMMA_STORE_THREADS);

        constexpr uint32_t kNumTmemLoads = STORE_BLOCK_M / kNumSwizzleAtomRows;
        #pragma unroll
        for (uint32_t i = 0; i < kNumTmemLoads; ++ i) {
            const uint32_t load_idx = reverse_store_order ? kNumTmemLoads - 1 - i : i;
            const uint32_t tmem_addr = tmem_base_addr +
                                       store_idx * STORE_BLOCK_M +
                                       load_idx * kNumSwizzleAtomRows;
            uint32_t values[kNumSwizzleAtomRows];

            // Warps cooperatively write one swizzled atom
            constexpr uint32_t kNumWarpsPerAtom = STORE_BLOCK_N_ATOM / 32;
            const uint32_t outer_atom_offset = (epilogue_warp_idx / kNumWarpsPerAtom) * STORE_BLOCK_M * SWIZZLE_CD;
            const uint32_t inner_atom_offset = load_idx * kNumSwizzleAtomRows * SWIZZLE_CD;
            uint8_t* smem_base_ptr = &smem.cd[tma_stage_idx * DG_ALIGN4(STORE_BLOCK_M * STORE_BLOCK_N * CD_ELEM_SIZE)]
                                     + outer_atom_offset + inner_atom_offset;

#if CD_DTYPE == 1
            {   // FP32: plain .32x32b loads (no STSM)
                dg100_tmem_load_32dp32b_x8(tmem_addr, values);
                dg100_tmem_wait_ld();
                epi_apply_values(values, kNumSwizzleAtomRows, epi_alpha);
            }
#else
            {
                // .16x256b loads satisfy the STSM layout: each lane gets
                // 2 rows (cols 2*(lane%4) onward) x 2 cols (dps lane/4, +8)
                dg100_tmem_load_16dp256b_x1(tmem_addr, values[0], values[1], values[2], values[3]);
                dg100_tmem_load_16dp256b_x1(tmem_addr | 0x00100000u, values[4], values[5], values[6], values[7]);
                dg100_tmem_wait_ld();
                epi_apply_values(values, kNumSwizzleAtomRows, epi_alpha);
            }
#endif
            #pragma unroll
            for (uint32_t value_idx = 0; value_idx < kNumSwizzleAtomRows; ++ value_idx)
                values[value_idx] = is_empty_group ? 0u : values[value_idx];

            // Release the overlapped TMEM columns
            if (NUM_OVERLAPPED_TMEM_COLS > 0) {
                constexpr uint32_t kNumOverlapLoads =
                    (NUM_OVERLAPPED_TMEM_COLS + kNumSwizzleAtomRows - 1) / kNumSwizzleAtomRows;
                const uint32_t num_loads_before_arrive =
                    kNumOverlapLoads < num_stores * kNumTmemLoads ? kNumOverlapLoads : num_stores * kNumTmemLoads;
                if (s * kNumTmemLoads + i + 1 == num_loads_before_arrive) {
                    dg100_before_thread_sync();
                    dg_mbarrier_arrive_cluster(tmem_overlap_barrier, 0);
                }
            }
            if (s == num_stores - 1 && i == kNumTmemLoads - 1) {
                dg100_before_thread_sync();
                dg_mbarrier_arrive_cluster(tmem_empty_barrier, 0);
            }

#if WITH_OUTPUT_SF
            {   // BF16 pairs -> per-row-pair amax -> UE8M0 SFs -> FP8 + STSM
                uint32_t values_bf16x2[kNumSwizzleAtomRows / 2];
                #pragma unroll
                for (uint32_t pair_idx = 0; pair_idx < kNumSwizzleAtomRows / 2; ++ pair_idx)
                    values_bf16x2[pair_idx] = dg_pack_bf16x2(
                        __uint_as_float(values[pair_idx * 2]), __uint_as_float(values[pair_idx * 2 + 1]));
                // The 8 lanes sharing `lane_idx % 4` cover the pair's two SF
                // groups: reduce the packed tree, then cross-group partially
                uint32_t amax_pair_bits = dg_bf16x2_max(
                    dg_bf16x2_max(dg_bf16x2_abs(values_bf16x2[0]), dg_bf16x2_abs(values_bf16x2[1])),
                    dg_bf16x2_max(dg_bf16x2_abs(values_bf16x2[2]), dg_bf16x2_abs(values_bf16x2[3])));
                // quad-tree reduce across the 4 lanes of the group (xor 4, 8)
                amax_pair_bits = dg_bf16x2_max(amax_pair_bits,
                    __shfl_xor_sync(0xffffffffu, amax_pair_bits, 4));
                amax_pair_bits = dg_bf16x2_max(amax_pair_bits,
                    __shfl_xor_sync(0xffffffffu, amax_pair_bits, 8));
                const uint32_t sf_exp_lower = dg_ue8m0_sf_exp_bf16<true>(
                    static_cast<uint16_t>(amax_pair_bits & 0xFFFF));
                const uint32_t sf_exp_upper = dg_ue8m0_sf_exp_bf16<true>(
                    static_cast<uint16_t>(amax_pair_bits >> 16));
                const float sf_inv_lo = dg_ue8m0_sf_inv_f32(sf_exp_lower);
                const float sf_inv_hi = dg_ue8m0_sf_inv_f32(sf_exp_upper);

                uint8_t* smem_ptr = smem_base_ptr + (lane_idx % 8) * SWIZZLE_CD +
                        ((epilogue_warp_idx * 2 + lane_idx / 8) ^ (lane_idx % 8)) * kNumBankGroupBytes;
                dg_stsm_u8x8_x2_trans(
                    dg_scale_bf16x2_into_fp8x4(values_bf16x2[0], values_bf16x2[1], sf_inv_lo, sf_inv_lo),
                    dg_scale_bf16x2_into_fp8x4(values_bf16x2[2], values_bf16x2[3], sf_inv_lo, sf_inv_lo),
                    smem_ptr);

                // Lanes 0-3 own the row pair's even row, lanes 4-7 the odd row
                if (lane_idx < kNumSwizzleAtomRows)
                    epi_store_sf_byte(sfd, sfd_stride, epi_shape_m, epi_shape_n,
                        base_m_idx + store_idx * STORE_BLOCK_M + load_idx * kNumSwizzleAtomRows +
                        2 * (lane_idx % 4) + lane_idx / 4,
                        base_n_idx + epilogue_warp_idx * SF_GRAN_N, batch_idx,
                        lane_idx < 4 ? sf_exp_lower : sf_exp_upper);
            }
#elif CD_DTYPE == 1
            {   // FP32
                const uint32_t col = lane_idx / 4;
                #pragma unroll
                for (uint32_t row = 0; row < kNumSwizzleAtomRows; ++ row) {
                    uint32_t* ptr = reinterpret_cast<uint32_t*>(
                        smem_base_ptr + row * (kNumBankGroupBytes * 8)
                        + (col ^ row) * kNumBankGroupBytes + (lane_idx % 4) * 4);
                    dg_st_shared_u32(ptr, values[row]);
                }
            }
#else
            {
                // Destination SMEM address (transposing STSM)
                const uint32_t row = lane_idx % 8;
                const uint32_t col = (epilogue_warp_idx % 2) * 4 + lane_idx / 8;
                uint8_t* smem_ptr = smem_base_ptr + row * (kNumBankGroupBytes * 8)
                                                  + (col ^ row) * kNumBankGroupBytes;
#if WITH_STOCHASTIC
                {
                    static_assert(kNumSwizzleAtomRows == 8, "hash chains need 8 values");
                    const uint32_t quartet_offset = lane_idx / 4 % 4;
                    uint32_t hashes[kNumSwizzleAtomRows], packed[kNumSwizzleAtomRows / 2];
                    #pragma unroll
                    for (uint32_t value_idx = 0; value_idx < kNumSwizzleAtomRows; ++ value_idx)
                        hashes[value_idx] = sr_partial_hash(values[value_idx], quartet_offset);
                    #pragma unroll
                    for (uint32_t xor_mask = 4; xor_mask <= 8; xor_mask <<= 1) {
                        #pragma unroll
                        for (uint32_t value_idx = 0; value_idx < kNumSwizzleAtomRows; ++ value_idx)
                            hashes[value_idx] += __shfl_xor_sync(0xffffffffu, hashes[value_idx], xor_mask);
                    }
                    #pragma unroll
                    for (uint32_t value_idx = 0; value_idx < kNumSwizzleAtomRows; value_idx += 2) {
                        const uint32_t quartet_n_idx = (base_n_idx + epilogue_warp_idx * 32 +
                                                        value_idx * 4 + lane_idx / 16 * 4) / 4;
                        packed[value_idx / 2] = dg_cvt_rs_bf16x2_f32(
                            __uint_as_float(values[value_idx]), __uint_as_float(values[value_idx + 1]),
                            sr_select_bits(hashes[value_idx] + quartet_n_idx, quartet_offset) |
                            (sr_select_bits(hashes[value_idx + 1] + quartet_n_idx, quartet_offset) << 16));
                    }
                    dg_stsm_x4_trans(packed[0], packed[1], packed[2], packed[3], smem_ptr);
                }
#else
                {
                    dg_stsm_x4_trans(
                        dg_pack_bf16x2(__uint_as_float(values[0]), __uint_as_float(values[1])),
                        dg_pack_bf16x2(__uint_as_float(values[2]), __uint_as_float(values[3])),
                        dg_pack_bf16x2(__uint_as_float(values[4]), __uint_as_float(values[5])),
                        dg_pack_bf16x2(__uint_as_float(values[6]), __uint_as_float(values[7])),
                        smem_ptr);
                }
#endif
            }
#endif  // store chain
        }

        // Synchronize all threads and issue the TMA
        dg_tma_store_fence();
        dg_named_barrier_sync(0, NUM_UMMA_STORE_THREADS);
        if (epilogue_warp_idx == 0 && dg_elect_one()) {
            #pragma unroll 1
            for (uint32_t i = 0; i < STORE_BLOCK_N / STORE_BLOCK_N_ATOM; ++ i) {
                uint8_t* smem_ptr = &smem.cd[tma_stage_idx * DG_ALIGN4(STORE_BLOCK_M * STORE_BLOCK_N * CD_ELEM_SIZE)]
                                    + i * STORE_BLOCK_M * STORE_BLOCK_N_ATOM * CD_ELEM_SIZE;
                const uint32_t m_idx = base_m_idx + store_idx * STORE_BLOCK_M;
                const uint32_t n_idx = base_n_idx + i * STORE_BLOCK_N_ATOM;
                if (IS_3D_TMA || DG_IS_K_GROUPED_CONTIG) {
                    if (WITH_ACCUMULATION)
                        dg100_tma_reduce_add_3d(&tensor_map_cd, smem_ptr, n_idx, m_idx, batch_idx);
                    else
                        dg_tma_store_3d(&tensor_map_cd, smem_ptr, n_idx, m_idx, batch_idx);
                } else {
                    if (WITH_ACCUMULATION)
                        dg100_tma_reduce_add_2d(&tensor_map_cd, smem_ptr, n_idx, m_idx);
                    else
                        dg_tma_store_2d(&tensor_map_cd, smem_ptr, n_idx, m_idx);
                }
            }
            dg_tma_store_commit_group();
        }
        __syncwarp();
    }
}
#endif  // SWAP_AB
"#;
