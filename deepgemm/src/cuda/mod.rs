//! CUDA kernel sources. Each module builds a self-contained NVRTC
//! translation unit (no system includes) from a static header plus
//! per-instantiation `#define` parameters.

pub mod bf16_gemm_sm100;
pub mod common;
pub mod fp8_fp4_gemm_1d1d;
pub mod fp8_gemm_1d2d;
pub mod layout;
pub mod mqa_logits;
pub mod sm100_cast;
pub mod sm100_common;

/// Substitute `%%NAME%%` placeholders in a template (avoids `format!` brace escaping).
pub fn subst(template: &str, vars: &[(&str, &str)]) -> String {
    let mut out = template.to_string();
    for (k, v) in vars {
        out = out.replace(&format!("%%{k}%%"), v);
    }
    out
}
