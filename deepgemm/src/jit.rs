//! NVRTC runtime JIT engine with content-addressed in-memory and on-disk caches.
//!
//! Like upstream DeepGEMM, kernels are CUDA C++ templates instantiated for the
//! exact block configuration at runtime — but compiled with NVRTC directly
//! (no torch, no nvcc, no build step). Compiled PTX is cached:
//! * in memory for the process lifetime,
//! * on disk (`~/.cache/deepgemm-rust/<sha256>.ptx`) across processes.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;

use cudarc::driver::safe::{CudaContext, CudaModule};
use cudarc::nvrtc::CompileOptions;
use sha2::{Digest, Sha256};

use crate::device::Arch;
use crate::types::{DgError, DgResult};

pub struct JitEngine {
    arch: Arch,
    extra_opts: Vec<String>,
    /// (source hash) -> module, per device-context.
    modules: Mutex<HashMap<String, Arc<CudaModule>>>,
    disk_cache_dir: PathBuf,
}

const NVRTC_BASE_OPTS: &[&str] = &[
    "--std=c++17",
    "-DNDEBUG",
    "--extra-device-vectorization",
    // The kernel TU is fully self-contained; no system includes.
    "--no-source-include",
];

impl JitEngine {
    pub fn new(arch: &Arch) -> DgResult<Self> {
        let disk_cache_dir = std::env::var_os("DEEPGEMM_CACHE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                std::env::var_os("HOME")
                    .map(|h| PathBuf::from(h).join(".cache").join("deepgemm-rust"))
                    .unwrap_or_else(|| PathBuf::from(".deepgemm-cache"))
            });
        std::fs::create_dir_all(&disk_cache_dir).ok();

        Ok(Self {
            arch: *arch,
            extra_opts: Vec::new(),
            modules: Mutex::new(HashMap::new()),
            disk_cache_dir,
        })
    }

    fn key(&self, source: &str) -> String {
        let mut h = Sha256::new();
        h.update(source.as_bytes());
        h.update(self.arch.nvrtc_arch.as_bytes());
        for o in &self.extra_opts {
            h.update(o.as_bytes());
        }
        hex::encode(h.finalize())
    }

    /// Compile (or fetch from cache) and load a module for the given
    /// self-contained CUDA C++ translation unit.
    pub fn module(&self, ctx: &Arc<CudaContext>, source: &str) -> DgResult<Arc<CudaModule>> {
        let key = self.key(source);
        if let Some(m) = self.modules.lock().unwrap().get(&key) {
            return Ok(m.clone());
        }

        // Disk cache hit?
        let ptx_path = self.disk_cache_dir.join(format!("{key}.ptx"));
        let ptx = if ptx_path.exists() {
            std::fs::read_to_string(&ptx_path).ok()
        } else {
            None
        };

        let ptx = match ptx {
            Some(p) => p,
            None => {
                let opts = CompileOptions {
                    arch: Some(self.arch.nvrtc_arch),
                    options: NVRTC_BASE_OPTS
                        .iter()
                        .map(|s| s.to_string())
                        .chain(self.extra_opts.iter().cloned())
                        .collect(),
                    ..Default::default()
                };
                let compiled = cudarc::nvrtc::compile_ptx_with_opts(source, opts)
                    .map_err(|e| DgError::Nvrtc(format!("{e:?}")))?;
                let ptx = compiled.to_src();
                // Best-effort disk cache write.
                std::fs::write(&ptx_path, &ptx).ok();
                ptx
            }
        };

        let ptx_obj = cudarc::nvrtc::Ptx::from_src(ptx);
        let module: Arc<CudaModule> = ctx
            .load_module(ptx_obj)
            .map_err(|e| DgError::Driver(format!("load module: {e:?}")))?;
        self.modules.lock().unwrap().insert(key, module.clone());
        Ok(module)
    }

    /// Compile-only (used by `cargo test` to validate CUDA sources without a GPU).
    pub fn compile_only(source: &str, nvrtc_arch: &str) -> DgResult<String> {
        let arch: &'static str = match nvrtc_arch {
            a if a.starts_with("sm_9") => "sm_90a",
            a if a.starts_with("sm_10") || a.starts_with("sm_12") => "sm_100a",
            a if a.starts_with("sm_89") => "sm_89",
            _ => "sm_90a",
        };
        let opts = CompileOptions {
            arch: Some(arch),
            options: NVRTC_BASE_OPTS.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        };
        let compiled = cudarc::nvrtc::compile_ptx_with_opts(source, opts)
            .map_err(|e| DgError::Nvrtc(format!("{e:?}")))?;
        Ok(compiled.to_src())
    }
}
