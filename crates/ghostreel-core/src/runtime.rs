//! Everything an indexing run needs from the machine: helper binaries and the resolved AI
//! backends (plan §2a). Resolved once per run so starting/stopping GhostPen or highllama is
//! picked up by the next run without restarting GhostReel.

use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::config::{Backend, Config};
use crate::doctor::locate;
use crate::models::{self, ModelSpec};
use crate::paths::Paths;
use crate::probe::{self, Target};
use crate::stt::Engine;

/// How transcription will run this time.
#[derive(Debug, Clone)]
pub enum SttSetup {
    Ready(Engine),
    /// Local whisper is chosen but the model file must be downloaded first.
    NeedsModel {
        spec: ModelSpec,
        dir: PathBuf,
        asr_bin: PathBuf,
        language: String,
    },
    /// Can't transcribe now (jobs stay pending and are retried by a later run).
    Unavailable(String),
}

impl SttSetup {
    pub fn describe(&self) -> String {
        match self {
            SttSetup::Ready(e) => e.label(),
            SttSetup::NeedsModel { spec, .. } => format!("local whisper (downloading {})", spec.file_name),
            SttSetup::Unavailable(why) => format!("unavailable: {why}"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Runtime {
    pub ffmpeg: PathBuf,
    pub ffprobe: PathBuf,
    pub stt: SttSetup,
    /// Where frames and other derived files live.
    pub data_dir: PathBuf,
    /// Keyframe extraction settings; `None` disables the frames stage (tests).
    pub frames: Option<crate::frames::FrameOptions>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RuntimeSummary {
    pub stt: String,
}

impl Runtime {
    pub fn summary(&self) -> RuntimeSummary {
        RuntimeSummary { stt: self.stt.describe() }
    }
}

/// Find the local transcription helper. Dev builds also look in the sibling `release/` dir,
/// because `scripts/build-asr.sh` builds it in release mode only.
pub fn locate_asr() -> Option<PathBuf> {
    locate("ghostreel-asr").or_else(|| {
        if !cfg!(debug_assertions) {
            return None;
        }
        let exe = std::env::current_exe().ok()?;
        let target = exe.parent()?.parent()?;
        let name = if cfg!(windows) { "ghostreel-asr.exe" } else { "ghostreel-asr" };
        let p = target.join("release").join(name);
        p.is_file().then_some(p)
    })
}

pub async fn resolve(paths: &Paths, config: &Config) -> Result<Runtime, crate::Error> {
    let ffmpeg =
        locate("ffmpeg").ok_or_else(|| crate::Error::Invalid("ffmpeg not found (see `ghostreel doctor`)".into()))?;
    let ffprobe =
        locate("ffprobe").ok_or_else(|| crate::Error::Invalid("ffprobe not found (see `ghostreel doctor`)".into()))?;
    let stt = resolve_stt(paths, config).await;
    Ok(Runtime {
        ffmpeg,
        ffprobe,
        stt,
        data_dir: paths.data_dir.clone(),
        frames: Some(crate::frames::FrameOptions::default()),
    })
}

async fn resolve_stt(paths: &Paths, config: &Config) -> SttSetup {
    let cfg = &config.stt;
    let probe = match cfg.backend {
        Backend::Local => None,
        _ => Some(probe::stt(&probe::probe_client(), &cfg.url).await),
    };
    let resolution = probe::resolve(cfg.backend, probe);
    match resolution.target {
        Target::Server => return SttSetup::Ready(Engine::server(&cfg.url)),
        Target::Unavailable => return SttSetup::Unavailable(resolution.reason),
        Target::Local => {}
    }
    let model = if cfg.model.trim().eq_ignore_ascii_case("auto") {
        auto_whisper_model(&crate::doctor::nvidia_gpus().await).to_string()
    } else {
        cfg.model.clone()
    };
    local_stt(&paths.models_dir(), &config.models.search_paths, &model)
}

/// large-v3-turbo (1.6 GB, ~2 GB VRAM) is far more accurate — names, accents, Spanish — but on CPU
/// or small GPUs `small` (466 MB) keeps transcription fast.
pub fn auto_whisper_model(gpus: &[crate::doctor::Gpu]) -> &'static str {
    if gpus.iter().any(|g| g.vram_total_mib >= 6000) { "large-v3-turbo" } else { "small" }
}

fn local_stt(models_dir: &Path, search_paths: &[PathBuf], model: &str) -> SttSetup {
    let Some(asr_bin) = locate_asr() else {
        return SttSetup::Unavailable("local transcription helper ghostreel-asr not found".into());
    };
    let spec = match models::whisper(model) {
        Ok(s) => s,
        Err(e) => return SttSetup::Unavailable(e.to_string()),
    };
    let language = "auto".to_string();
    match models::find(&models::search_roots(models_dir, search_paths), &spec.file_name) {
        Some(model) => SttSetup::Ready(Engine::Local { asr_bin, model, language }),
        None => SttSetup::NeedsModel { spec, dir: models_dir.to_path_buf(), asr_bin, language },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doctor::Gpu;

    #[test]
    fn whisper_model_follows_vram() {
        let gpu = |mib| Gpu { name: "g".into(), vram_total_mib: mib, vram_used_mib: 0, driver: "d".into() };
        assert_eq!(auto_whisper_model(&[gpu(8192)]), "large-v3-turbo");
        assert_eq!(auto_whisper_model(&[gpu(4096)]), "small");
        assert_eq!(auto_whisper_model(&[]), "small");
    }
}
