//! User configuration (`config.toml`), shared by the desktop app and the CLI.
//!
//! Every AI capability has an independent `backend` switch (plan §2a):
//! `auto` probes the configured server and falls back to running the model locally,
//! `server` requires the server, `local` never touches it.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::Error;

/// Default endpoints of the sibling services on a dev box.
pub const DEFAULT_VISION_URL: &str = "http://127.0.0.1:8089"; // highllama chat/vision
pub const DEFAULT_EMBED_URL: &str = "http://127.0.0.1:8091"; // highllama embeddings server
pub const DEFAULT_STT_URL: &str = "http://127.0.0.1:8771"; // GhostPen transcription server

/// The one embedding model GhostReel uses everywhere (plan D5): vectors from the server and
/// the local runtime must be interchangeable.
pub const EMBED_MODEL: &str = "embeddinggemma-300M-Q8_0";
pub const EMBED_DIM: usize = 768;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    /// Use the server when it is reachable and capable, else run locally.
    #[default]
    Auto,
    /// Always run in-process.
    Local,
    /// Always use the server; fail if it is unavailable.
    Server,
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Backend::Auto => "auto",
            Backend::Local => "local",
            Backend::Server => "server",
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct VisionConfig {
    pub backend: Backend,
    pub url: String,
    /// Model id sent to the server; empty = whatever the server has loaded.
    pub model: String,
    /// Bearer token for non-local servers.
    pub api_key: String,
    /// Local vision model catalog id (pair: model + projector). Default: "bonsai-27b".
    pub local_model: String,
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            backend: Backend::Auto,
            url: DEFAULT_VISION_URL.into(),
            model: String::new(),
            api_key: String::new(),
            local_model: "bonsai-27b".into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct EmbedConfig {
    pub backend: Backend,
    pub url: String,
    pub model: String,
}

impl Default for EmbedConfig {
    fn default() -> Self {
        Self { backend: Backend::Auto, url: DEFAULT_EMBED_URL.into(), model: EMBED_MODEL.into() }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SttConfig {
    pub backend: Backend,
    pub url: String,
    /// Whisper model for the local backend: `auto` (large-v3-turbo on an NVIDIA GPU with ≥ 6 GB,
    /// else `small`), or a name like `large-v3-turbo`, `small`, `base`.
    pub model: String,
}

impl Default for SttConfig {
    fn default() -> Self {
        Self { backend: Backend::Auto, url: DEFAULT_STT_URL.into(), model: "auto".into() }
    }
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelsConfig {
    /// Custom directory for downloaded model files. If unset, defaults to `Paths::models_dir()`.
    pub dir: Option<PathBuf>,
    /// Extra directories searched for model files before downloading
    /// (e.g. `~/.lmstudio/models`), so an existing copy is reused.
    pub search_paths: Vec<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub vision: VisionConfig,
    pub embed: EmbedConfig,
    pub stt: SttConfig,
    pub models: ModelsConfig,
}

impl Config {
    /// Load `path`, or defaults when the file does not exist.
    pub fn load(path: &Path) -> Result<Self, Error> {
        match std::fs::read_to_string(path) {
            Ok(text) => toml::from_str(&text).map_err(|e| Error::Config(format!("{}: {e}", path.display()))),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(Error::Io(path.to_path_buf(), e)),
        }
    }

    pub fn to_toml(&self) -> Result<String, Error> {
        toml::to_string_pretty(self).map_err(|e| Error::Config(e.to_string()))
    }

    pub fn save(&self, path: &Path) -> Result<(), Error> {
        let text = self.to_toml()?;
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| Error::Io(dir.to_path_buf(), e))?;
        }
        std::fs::write(path, text).map_err(|e| Error::Io(path.to_path_buf(), e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_file_gives_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = Config::load(&dir.path().join("nope.toml")).unwrap();
        assert_eq!(cfg, Config::default());
        assert_eq!(cfg.embed.url, DEFAULT_EMBED_URL);
    }

    #[test]
    fn roundtrip_and_partial_files() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sub/config.toml");
        let mut cfg = Config::default();
        cfg.vision.backend = Backend::Server;
        cfg.stt.backend = Backend::Local;
        cfg.save(&path).unwrap();
        assert_eq!(Config::load(&path).unwrap(), cfg);

        // Unspecified sections/keys fall back to defaults.
        std::fs::write(&path, "[vision]\nbackend = \"local\"\n").unwrap();
        let cfg = Config::load(&path).unwrap();
        assert_eq!(cfg.vision.backend, Backend::Local);
        assert_eq!(cfg.vision.url, DEFAULT_VISION_URL);
        assert_eq!(cfg.stt, SttConfig::default());
    }

    #[test]
    fn invalid_backend_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "[vision]\nbackend = \"cloud\"\n").unwrap();
        assert!(matches!(Config::load(&path), Err(Error::Config(_))));
    }

    #[test]
    fn models_dir_config() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "[models]\ndir = \"/custom/models\"\nsearch_paths = [\"/extra/path\"]\n").unwrap();
        let cfg = Config::load(&path).unwrap();
        assert_eq!(cfg.models.dir, Some(PathBuf::from("/custom/models")));
        assert_eq!(cfg.models.search_paths, vec![PathBuf::from("/extra/path")]);
    }

    #[test]
    fn vision_local_model_default_and_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let cfg = Config::default();
        assert_eq!(cfg.vision.local_model, "bonsai-27b");

        std::fs::write(&path, "[vision]\nlocal_model = \"gemma-3-4b-it\"\n").unwrap();
        let loaded = Config::load(&path).unwrap();
        assert_eq!(loaded.vision.local_model, "gemma-3-4b-it");
        assert_eq!(loaded.vision.backend, Backend::Auto);

        let mut saved_cfg = Config::default();
        saved_cfg.vision.local_model = "qwen2.5-vl-7b".into();
        saved_cfg.save(&path).unwrap();
        let roundtrip = Config::load(&path).unwrap();
        assert_eq!(roundtrip.vision.local_model, "qwen2.5-vl-7b");
    }
}
