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

/// Settings for one model-using capability. Frame descriptions (`[vision]`) and the script chat
/// (`[chat_model]`) each have their own: describing a keyframe is a short prompt plus one image run
/// hundreds of times, while a script chat needs room for tool results and a whole draft, so they
/// want different context windows even though both run the same helper binary.
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
    /// Context window for the local helper, in tokens (2048–131072).
    pub ctx_tokens: u32,
    /// KV cache precision for the local helper: `f16`, `q8_0` or `q4_0`. `q4_0` holds roughly 4×
    /// the context of `f16` in the same VRAM (what highllama runs with).
    pub kv_cache: String,
    /// Flash attention for the local helper: `auto`, `on` or `off`.
    pub flash_attn: String,
}

/// Context window of the frame-description helper: a short prompt plus one image.
pub const DESCRIBE_CTX_TOKENS: u32 = 8192;
/// Context window of the script chat: tool results plus a whole draft.
pub const CHAT_CTX_TOKENS: u32 = 32768;
pub const KV_CACHE_KINDS: &[&str] = &["f16", "q8_0", "q4_0"];
pub const FLASH_ATTN_KINDS: &[&str] = &["auto", "on", "off"];

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            backend: Backend::Auto,
            url: DEFAULT_VISION_URL.into(),
            model: String::new(),
            api_key: String::new(),
            local_model: "bonsai-27b".into(),
            ctx_tokens: DESCRIBE_CTX_TOKENS,
            kv_cache: "q4_0".into(),
            flash_attn: "auto".into(),
        }
    }
}

impl VisionConfig {
    pub fn validate(&self, section: &str) -> Result<(), String> {
        if !(2048..=131_072).contains(&self.ctx_tokens) {
            return Err(format!("{section}.ctx_tokens must be between 2048 and 131072, got {}", self.ctx_tokens));
        }
        if !KV_CACHE_KINDS.contains(&self.kv_cache.as_str()) {
            return Err(format!(
                "{section}.kv_cache must be one of {}, got '{}'",
                KV_CACHE_KINDS.join(", "),
                self.kv_cache
            ));
        }
        if !FLASH_ATTN_KINDS.contains(&self.flash_attn.as_str()) {
            return Err(format!(
                "{section}.flash_attn must be one of {}, got '{}'",
                FLASH_ATTN_KINDS.join(", "),
                self.flash_attn
            ));
        }
        Ok(())
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

/// Script chat settings.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ChatConfig {
    /// Editing instructions for the script chat; empty = the built-in default
    /// (`chat::DEFAULT_EDITOR_PROMPT`). `{project}`, `{fps}`, `{width}`, `{height}` are filled in.
    pub system_prompt: String,
}

/// Keyframe extraction settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FramesConfig {
    /// Maximum gap between keyframes in seconds. Default 8 s; valid range 1–60.
    /// Shorter = more detail for search and scripts; longer = faster indexing.
    /// Scene changes always get a frame regardless of this setting.
    pub max_interval_s: f64,
}

impl Default for FramesConfig {
    fn default() -> Self {
        Self { max_interval_s: 8.0 }
    }
}

impl FramesConfig {
    /// Clamp `max_interval_s` to the valid range (1–60 s) without erroring.
    pub fn clamped_interval(&self) -> f64 {
        self.max_interval_s.clamp(1.0, 60.0)
    }

    /// Validate that `max_interval_s` is within the 1–60 range, returning an error message if not.
    pub fn validate(&self) -> Result<(), String> {
        if self.max_interval_s < 1.0 || self.max_interval_s > 60.0 {
            Err(format!("frames.max_interval_s must be between 1 and 60, got {}", self.max_interval_s))
        } else {
            Ok(())
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Frame descriptions (indexing).
    pub vision: VisionConfig,
    /// Script chat. Absent in configs written before the split: it then follows `vision`, with the
    /// chat's bigger context window.
    #[serde(default)]
    pub chat_model: Option<VisionConfig>,
    pub embed: EmbedConfig,
    pub stt: SttConfig,
    pub models: ModelsConfig,
    pub frames: FramesConfig,
    pub chat: ChatConfig,
}

impl Config {
    /// The script chat's model settings: its own `[chat_model]` section, or the frame-description
    /// settings with the chat's larger context window when the file predates the split.
    pub fn chat_model(&self) -> VisionConfig {
        match &self.chat_model {
            Some(c) => c.clone(),
            None => VisionConfig { ctx_tokens: CHAT_CTX_TOKENS, ..self.vision.clone() },
        }
    }

    /// Check every section that has rules. Returns the first problem as a message.
    pub fn validate(&self) -> Result<(), String> {
        self.vision.validate("vision")?;
        self.chat_model().validate("chat_model")?;
        self.frames.validate()
    }

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
    fn describe_and_chat_have_their_own_model_settings() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");

        // Defaults: same model, different windows.
        let cfg = Config::default();
        assert_eq!(cfg.vision.ctx_tokens, DESCRIBE_CTX_TOKENS);
        assert_eq!(cfg.chat_model().ctx_tokens, CHAT_CTX_TOKENS);
        assert_eq!(cfg.chat_model().local_model, cfg.vision.local_model);
        assert_eq!(cfg.vision.kv_cache, "q4_0");

        // A file written before the split: the chat inherits vision's backend and model.
        std::fs::write(
            &path,
            "[vision]\nbackend = \"server\"\nurl = \"http://x:1/v1\"\nlocal_model = \"qwen2.5-vl-3b\"\n",
        )
        .unwrap();
        let old = Config::load(&path).unwrap();
        let chat = old.chat_model();
        assert_eq!(chat.backend, Backend::Server);
        assert_eq!(chat.url, "http://x:1/v1");
        assert_eq!(chat.local_model, "qwen2.5-vl-3b");
        assert_eq!(chat.ctx_tokens, CHAT_CTX_TOKENS, "but with the chat's window");

        // Once set, the two are independent and survive a roundtrip.
        let mut cfg = Config::default();
        cfg.vision.ctx_tokens = 4096;
        cfg.chat_model = Some(VisionConfig { ctx_tokens: 65536, kv_cache: "q8_0".into(), ..VisionConfig::default() });
        cfg.save(&path).unwrap();
        let back = Config::load(&path).unwrap();
        assert_eq!(back, cfg);
        assert_eq!(back.vision.ctx_tokens, 4096);
        assert_eq!(back.chat_model().ctx_tokens, 65536);
        assert_eq!(back.chat_model().kv_cache, "q8_0");

        // Validation.
        assert!(VisionConfig { ctx_tokens: 1024, ..VisionConfig::default() }.validate("vision").is_err());
        assert!(VisionConfig { kv_cache: "q2_k".into(), ..VisionConfig::default() }.validate("vision").is_err());
        assert!(VisionConfig { flash_attn: "maybe".into(), ..VisionConfig::default() }.validate("chat_model").is_err());
        assert!(back.validate().is_ok());
    }

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

    #[test]
    fn frames_config_default_is_8s() {
        let cfg = Config::default();
        assert_eq!(cfg.frames.max_interval_s, 8.0);
        assert_eq!(cfg.frames.clamped_interval(), 8.0);
        assert!(cfg.frames.validate().is_ok());
    }

    #[test]
    fn frames_config_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let mut cfg = Config::default();
        cfg.frames.max_interval_s = 5.0;
        cfg.save(&path).unwrap();
        let loaded = Config::load(&path).unwrap();
        assert_eq!(loaded.frames.max_interval_s, 5.0);

        // Partial file: no [frames] section → default 8 s
        std::fs::write(&path, "[vision]\nbackend = \"local\"\n").unwrap();
        let loaded = Config::load(&path).unwrap();
        assert_eq!(loaded.frames.max_interval_s, 8.0);
    }

    #[test]
    fn frames_config_clamping_and_validation() {
        let mut fc = FramesConfig { max_interval_s: 0.5 };
        assert_eq!(fc.clamped_interval(), 1.0);
        assert!(fc.validate().is_err());

        fc.max_interval_s = 90.0;
        assert_eq!(fc.clamped_interval(), 60.0);
        assert!(fc.validate().is_err());

        fc.max_interval_s = 15.0;
        assert_eq!(fc.clamped_interval(), 15.0);
        assert!(fc.validate().is_ok());
    }
}
