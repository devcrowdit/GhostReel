//! Model files: find an existing copy (GhostReel's models dir, configured search paths, sibling
//! apps like GhostPen/LM Studio) before downloading, and download with resume + progress.

use std::path::{Path, PathBuf};
use std::time::Duration;

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::io::AsyncWriteExt;

use crate::Error;
use crate::config::Config;
use crate::paths::Paths;

/// Effective directory for downloaded models: `config.models.dir` if set, else `paths.models_dir()`.
pub fn effective_models_dir(paths: &Paths, config: &Config) -> PathBuf {
    config.models.dir.clone().unwrap_or_else(|| paths.models_dir())
}

/// Category of model in GhostReel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    Whisper,
    Vision,
    VisionProjector,
    Embedding,
}

impl std::fmt::Display for ModelKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelKind::Whisper => write!(f, "whisper"),
            ModelKind::Vision => write!(f, "vision"),
            ModelKind::VisionProjector => write!(f, "vision_projector"),
            ModelKind::Embedding => write!(f, "embedding"),
        }
    }
}

/// A downloadable model file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSpec {
    pub file_name: String,
    pub url: String,
}

/// An entry in GhostReel's curated model catalog.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogEntry {
    pub id: String,
    pub kind: ModelKind,
    pub file_name: String,
    pub url: String,
    pub size_bytes: u64,
    pub speed: u8,
    pub accuracy: u8,
    pub note: String,
    pub languages: String,
}

impl CatalogEntry {
    pub fn spec(&self) -> ModelSpec {
        ModelSpec { file_name: self.file_name.clone(), url: self.url.clone() }
    }
}

/// Installation status of a catalog model on this computer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelStatus {
    pub entry: CatalogEntry,
    pub installed_path: Option<PathBuf>,
    pub in_own_dir: bool,
    pub partial_bytes: Option<u64>,
}

/// Built-in catalog of recommended models with verified byte sizes.
pub fn catalog() -> Vec<CatalogEntry> {
    let (vision_spec, proj_spec) = bonsai_vision();
    let embed_spec = embeddinggemma();
    vec![
        // Whisper models (HF ggerganov/whisper.cpp)
        CatalogEntry {
            id: "tiny".into(),
            kind: ModelKind::Whisper,
            file_name: "ggml-tiny.bin".into(),
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin".into(),
            size_bytes: 77_691_713,
            speed: 5,
            accuracy: 1,
            note: "fastest, lowest accuracy".into(),
            languages: "multilingual".into(),
        },
        CatalogEntry {
            id: "tiny.en".into(),
            kind: ModelKind::Whisper,
            file_name: "ggml-tiny.en.bin".into(),
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin".into(),
            size_bytes: 77_704_715,
            speed: 5,
            accuracy: 2,
            note: "fastest, English-only".into(),
            languages: "english".into(),
        },
        CatalogEntry {
            id: "base".into(),
            kind: ModelKind::Whisper,
            file_name: "ggml-base.bin".into(),
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin".into(),
            size_bytes: 147_951_465,
            speed: 4,
            accuracy: 2,
            note: "fast, basic accuracy".into(),
            languages: "multilingual".into(),
        },
        CatalogEntry {
            id: "base.en".into(),
            kind: ModelKind::Whisper,
            file_name: "ggml-base.en.bin".into(),
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin".into(),
            size_bytes: 147_964_211,
            speed: 4,
            accuracy: 3,
            note: "fast, English-only".into(),
            languages: "english".into(),
        },
        CatalogEntry {
            id: "small".into(),
            kind: ModelKind::Whisper,
            file_name: "ggml-small.bin".into(),
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin".into(),
            size_bytes: 487_601_967,
            speed: 3,
            accuracy: 4,
            note: "balanced — sweet spot on a GPU".into(),
            languages: "multilingual".into(),
        },
        CatalogEntry {
            id: "small.en".into(),
            kind: ModelKind::Whisper,
            file_name: "ggml-small.en.bin".into(),
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin".into(),
            size_bytes: 487_614_201,
            speed: 3,
            accuracy: 4,
            note: "balanced, English-only".into(),
            languages: "english".into(),
        },
        CatalogEntry {
            id: "medium".into(),
            kind: ModelKind::Whisper,
            file_name: "ggml-medium.bin".into(),
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin".into(),
            size_bytes: 1_533_763_059,
            speed: 2,
            accuracy: 5,
            note: "most accurate classic whisper, heaviest".into(),
            languages: "multilingual".into(),
        },
        CatalogEntry {
            id: "medium.en".into(),
            kind: ModelKind::Whisper,
            file_name: "ggml-medium.en.bin".into(),
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin".into(),
            size_bytes: 1_533_774_781,
            speed: 2,
            accuracy: 5,
            note: "most accurate classic whisper, English-only".into(),
            languages: "english".into(),
        },
        CatalogEntry {
            id: "large-v3-turbo".into(),
            kind: ModelKind::Whisper,
            file_name: "ggml-large-v3-turbo.bin".into(),
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin".into(),
            size_bytes: 1_624_555_275,
            speed: 4,
            accuracy: 5,
            note: "best accuracy, needs ~2 GB VRAM".into(),
            languages: "multilingual".into(),
        },
        CatalogEntry {
            id: "large-v3-turbo-q8_0".into(),
            kind: ModelKind::Whisper,
            file_name: "ggml-large-v3-turbo-q8_0.bin".into(),
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q8_0.bin".into(),
            size_bytes: 874_188_075,
            speed: 4,
            accuracy: 5,
            note: "high accuracy, 8-bit quantized (~874 MB)".into(),
            languages: "multilingual".into(),
        },
        CatalogEntry {
            id: "large-v3-turbo-q5_0".into(),
            kind: ModelKind::Whisper,
            file_name: "ggml-large-v3-turbo-q5_0.bin".into(),
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin".into(),
            size_bytes: 574_041_195,
            speed: 5,
            accuracy: 4,
            note: "fast, 5-bit quantized (~574 MB)".into(),
            languages: "multilingual".into(),
        },
        // Vision: Bonsai-27B at 1-bit + mmproj image projector
        CatalogEntry {
            id: "bonsai-27b".into(),
            kind: ModelKind::Vision,
            file_name: vision_spec.file_name,
            url: vision_spec.url,
            size_bytes: 3_803_452_480,
            speed: 3,
            accuracy: 4,
            note: "1-bit quantized 27B vision model (~3.8 GB)".into(),
            languages: "multilingual".into(),
        },
        CatalogEntry {
            id: "bonsai-27b-mmproj".into(),
            kind: ModelKind::VisionProjector,
            file_name: proj_spec.file_name,
            url: proj_spec.url,
            size_bytes: 629_246_880,
            speed: 4,
            accuracy: 5,
            note: "image projector for Bonsai-27B (~630 MB)".into(),
            languages: "multilingual".into(),
        },
        // Search embeddings: embeddinggemma 300M
        CatalogEntry {
            id: "embeddinggemma-300M-Q8_0".into(),
            kind: ModelKind::Embedding,
            file_name: embed_spec.file_name,
            url: embed_spec.url,
            size_bytes: 333_590_944,
            speed: 5,
            accuracy: 5,
            note: "768-dim text embeddings, runs on CPU (~334 MB)".into(),
            languages: "multilingual".into(),
        },
    ]
}

/// Find a catalog entry by ID, alias, or exact file name.
pub fn find_entry(id: &str) -> Option<CatalogEntry> {
    let trimmed = id.trim();
    let cat = catalog();
    if let Some(e) = cat.iter().find(|e| e.id.eq_ignore_ascii_case(trimmed)) {
        return Some(e.clone());
    }
    // Aliases
    if trimmed.eq_ignore_ascii_case("embeddinggemma") {
        return cat.iter().find(|e| e.id == "embeddinggemma-300M-Q8_0").cloned();
    }
    if trimmed.eq_ignore_ascii_case("bonsai-27b-projector") {
        return cat.iter().find(|e| e.id == "bonsai-27b-mmproj").cloned();
    }
    cat.into_iter().find(|e| e.file_name.eq_ignore_ascii_case(trimmed))
}

/// Query installation status for every catalog entry.
pub fn status(models_dir: &Path, search_paths: &[PathBuf]) -> Vec<ModelStatus> {
    let roots = search_roots(models_dir, search_paths);
    catalog()
        .into_iter()
        .map(|entry| {
            let own = models_dir.join(&entry.file_name);
            if own.is_file() && own.metadata().map(|m| m.len() > 0).unwrap_or(false) {
                return ModelStatus { entry, installed_path: Some(own), in_own_dir: true, partial_bytes: None };
            }
            if let Some(found) = find(&roots, &entry.file_name) {
                let in_own = found.starts_with(models_dir);
                return ModelStatus { entry, installed_path: Some(found), in_own_dir: in_own, partial_bytes: None };
            }
            let part = models_dir.join(format!("{}.part", entry.file_name));
            let partial_bytes = part.metadata().ok().map(|m| m.len()).filter(|&l| l > 0);
            ModelStatus { entry, installed_path: None, in_own_dir: false, partial_bytes }
        })
        .collect()
}

/// Remove a model file (and any `.part` file) from GhostReel's own `models_dir`.
/// Never touches copies found in search_paths or external apps like GhostPen or LM Studio.
pub fn remove(models_dir: &Path, id: &str) -> Result<PathBuf, Error> {
    let file_name = if let Some(e) = find_entry(id) {
        e.file_name
    } else if let Ok(spec) = whisper(id) {
        spec.file_name
    } else {
        return Err(Error::NotFound(format!("unknown model '{id}'")));
    };

    let dest = models_dir.join(&file_name);
    let part = models_dir.join(format!("{file_name}.part"));
    let mut removed = false;

    if dest.is_file() {
        std::fs::remove_file(&dest).map_err(|e| Error::Io(dest.clone(), e))?;
        removed = true;
    }
    if part.is_file() {
        let _ = std::fs::remove_file(&part);
        removed = true;
    }

    if removed {
        Ok(dest)
    } else {
        Err(Error::NotFound(format!("model '{id}' ({file_name}) is not installed in {}", models_dir.display())))
    }
}

/// whisper.cpp ggml model by name (`large-v3-turbo`, `small`, `base.en`, …).
pub fn whisper(model: &str) -> Result<ModelSpec, Error> {
    let name = model.trim();
    let valid = !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '.' || c == '_');
    if !valid || name.contains("..") {
        return Err(Error::Invalid(format!("invalid whisper model name '{model}'")));
    }
    let file_name = format!("ggml-{name}.bin");
    let url = format!("https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{file_name}");
    Ok(ModelSpec { file_name, url })
}

fn hf(repo: &str, file: &str) -> ModelSpec {
    ModelSpec { file_name: file.to_string(), url: format!("https://huggingface.co/{repo}/resolve/main/{file}") }
}

/// Default local vision model: Bonsai-27B at 1-bit (~3.6 GB) + its image projector (S0).
pub fn bonsai_vision() -> (ModelSpec, ModelSpec) {
    (
        hf("prism-ml/Bonsai-27B-gguf", "Bonsai-27B-Q1_0.gguf"),
        hf("prism-ml/Bonsai-27B-gguf", "Bonsai-27B-mmproj-Q8_0.gguf"),
    )
}

/// The one embedding model GhostReel uses everywhere (plan D5).
pub fn embeddinggemma() -> ModelSpec {
    hf("ggml-org/embeddinggemma-300M-GGUF", "embeddinggemma-300M-Q8_0.gguf")
}

/// Directories where other local apps keep compatible model files.
pub fn well_known_dirs() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(d) = dirs::data_dir() {
        out.push(d.join("GhostPen").join("models")); // whisper ggml models
    }
    if let Some(h) = dirs::home_dir() {
        out.push(h.join(".lmstudio").join("models")); // GGUF models
    }
    out
}

/// Look for `file_name` in `roots` (a few levels deep).
pub fn find(roots: &[PathBuf], file_name: &str) -> Option<PathBuf> {
    fn walk(dir: &Path, name: &str, depth: usize) -> Option<PathBuf> {
        let entries = std::fs::read_dir(dir).ok()?;
        let mut subdirs = Vec::new();
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                subdirs.push(p);
            } else if e.file_name().to_string_lossy().eq_ignore_ascii_case(name)
                && e.metadata().map(|m| m.len() > 0).unwrap_or(false)
            {
                return Some(p);
            }
        }
        if depth == 0 {
            return None;
        }
        subdirs.iter().find_map(|d| walk(d, name, depth - 1))
    }
    roots.iter().find_map(|r| walk(r, file_name, 4))
}

/// Search order: GhostReel's own models dir, configured paths, well-known app dirs.
pub fn search_roots(models_dir: &Path, configured: &[PathBuf]) -> Vec<PathBuf> {
    let mut roots = vec![models_dir.to_path_buf()];
    roots.extend(configured.iter().cloned());
    roots.extend(well_known_dirs());
    roots
}

/// Download `spec` into `dir` (resuming a previous `.part`), reporting `(bytes_done, bytes_total)`.
pub async fn download(
    spec: &ModelSpec,
    dir: &Path,
    mut on_progress: impl FnMut(u64, Option<u64>),
) -> Result<PathBuf, Error> {
    tokio::fs::create_dir_all(dir).await.map_err(|e| Error::Io(dir.to_path_buf(), e))?;
    let dest = dir.join(&spec.file_name);
    let part = dir.join(format!("{}.part", spec.file_name));
    let already = tokio::fs::metadata(&part).await.map(|m| m.len()).unwrap_or(0);

    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(15))
        .read_timeout(Duration::from_secs(60))
        .build()
        .map_err(|e| Error::Invalid(e.to_string()))?;
    let mut req = client.get(&spec.url);
    if already > 0 {
        req = req.header(reqwest::header::RANGE, format!("bytes={already}-"));
    }
    let resp = req.send().await.map_err(|e| Error::Download(format!("{}: {e}", spec.url)))?;
    let status = resp.status();
    let resumed = status == reqwest::StatusCode::PARTIAL_CONTENT;
    if !status.is_success() {
        return Err(Error::Download(format!("{}: HTTP {status}", spec.url)));
    }
    let start = if resumed { already } else { 0 };
    let total = resp.content_length().map(|l| l + start);

    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(resumed)
        .truncate(!resumed)
        .open(&part)
        .await
        .map_err(|e| Error::Io(part.clone(), e))?;
    let mut done = start;
    on_progress(done, total);
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| Error::Download(format!("{}: {e}", spec.url)))?;
        file.write_all(&chunk).await.map_err(|e| Error::Io(part.clone(), e))?;
        done += chunk.len() as u64;
        on_progress(done, total);
    }
    file.flush().await.map_err(|e| Error::Io(part.clone(), e))?;
    drop(file);
    if let Some(t) = total
        && done != t
    {
        return Err(Error::Download(format!("{}: incomplete ({done} of {t} bytes)", spec.url)));
    }
    tokio::fs::rename(&part, &dest).await.map_err(|e| Error::Io(dest.clone(), e))?;
    Ok(dest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[test]
    fn whisper_names() {
        let s = whisper("large-v3-turbo").unwrap();
        assert_eq!(s.file_name, "ggml-large-v3-turbo.bin");
        assert!(s.url.ends_with("/ggml-large-v3-turbo.bin"));
        assert!(whisper("../etc/passwd").is_err());
        assert!(whisper("").is_err());
    }

    #[test]
    fn finds_existing_copies() {
        let tmp = tempfile::tempdir().unwrap();
        let nested = tmp.path().join("lm/org/repo");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(nested.join("model.gguf"), b"x").unwrap();
        std::fs::write(tmp.path().join("empty.bin"), b"").unwrap();
        let roots = vec![tmp.path().join("missing"), tmp.path().to_path_buf()];
        assert!(find(&roots, "MODEL.gguf").is_some());
        assert!(find(&roots, "empty.bin").is_none(), "zero-byte files are not models");
    }

    /// Serves `body`, honouring `Range: bytes=N-`.
    async fn serve_file(body: Vec<u8>) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else { return };
                let body = body.clone();
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 4096];
                    let n = sock.read(&mut buf).await.unwrap_or(0);
                    let req = String::from_utf8_lossy(&buf[..n]).to_lowercase();
                    let from = req.lines().find_map(|l| l.strip_prefix("range: bytes=")).and_then(|r| {
                        r.trim_end_matches('-')
                            .trim_end_matches("-\r")
                            .trim()
                            .trim_end_matches('-')
                            .parse::<usize>()
                            .ok()
                    });
                    let (status, slice) = match from {
                        Some(f) => ("206 Partial Content", &body[f..]),
                        None => ("200 OK", &body[..]),
                    };
                    let head =
                        format!("HTTP/1.1 {status}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n", slice.len());
                    let _ = sock.write_all(head.as_bytes()).await;
                    let _ = sock.write_all(slice).await;
                });
            }
        });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn downloads_and_resumes() {
        let body: Vec<u8> = (0..100_000u32).map(|i| (i % 251) as u8).collect();
        let base = serve_file(body.clone()).await;
        let tmp = tempfile::tempdir().unwrap();
        let spec = ModelSpec { file_name: "m.bin".into(), url: format!("{base}/m.bin") };

        // A previous interrupted download left the first 30 000 bytes.
        std::fs::write(tmp.path().join("m.bin.part"), &body[..30_000]).unwrap();
        let mut last = (0, None);
        let path = download(&spec, tmp.path(), |d, t| last = (d, t)).await.unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), body);
        assert_eq!(last, (100_000, Some(100_000)));
        assert!(!tmp.path().join("m.bin.part").exists());
    }

    #[test]
    fn catalog_has_expected_entries() {
        let cat = catalog();
        assert!(cat.len() >= 14);
        assert!(cat.iter().any(|e| e.id == "tiny" && e.kind == ModelKind::Whisper));
        assert!(cat.iter().any(|e| e.id == "large-v3-turbo" && e.size_bytes == 1_624_555_275));
        assert!(cat.iter().any(|e| e.id == "bonsai-27b" && e.kind == ModelKind::Vision));
        assert!(cat.iter().any(|e| e.id == "bonsai-27b-mmproj" && e.kind == ModelKind::VisionProjector));
        assert!(cat.iter().any(|e| e.id == "embeddinggemma-300M-Q8_0" && e.kind == ModelKind::Embedding));
    }

    #[test]
    fn find_entry_aliases() {
        assert_eq!(find_entry("tiny").unwrap().file_name, "ggml-tiny.bin");
        assert_eq!(find_entry("embeddinggemma").unwrap().id, "embeddinggemma-300M-Q8_0");
        assert_eq!(find_entry("bonsai-27b-projector").unwrap().id, "bonsai-27b-mmproj");
        assert_eq!(find_entry("Bonsai-27B-Q1_0.gguf").unwrap().id, "bonsai-27b");
        assert!(find_entry("nonexistent-model-xyz").is_none());
    }

    #[test]
    fn status_and_remove() {
        let dir = tempfile::tempdir().unwrap();
        let models_dir = dir.path().join("models");
        std::fs::create_dir_all(&models_dir).unwrap();

        // Model not installed anywhere (large-v3-turbo-q5_0)
        let st = status(&models_dir, &[]);
        let q5_st = st.iter().find(|s| s.entry.id == "large-v3-turbo-q5_0").unwrap();
        assert!(q5_st.installed_path.is_none());
        assert!(!q5_st.in_own_dir);
        assert!(q5_st.partial_bytes.is_none());

        // Create a fake partial file for q5_0
        std::fs::write(models_dir.join("ggml-large-v3-turbo-q5_0.bin.part"), b"partial data").unwrap();
        let st = status(&models_dir, &[]);
        let q5_st = st.iter().find(|s| s.entry.id == "large-v3-turbo-q5_0").unwrap();
        assert_eq!(q5_st.partial_bytes, Some(12));

        // Create full file
        std::fs::write(models_dir.join("ggml-large-v3-turbo-q5_0.bin"), b"full data").unwrap();
        let st = status(&models_dir, &[]);
        let q5_st = st.iter().find(|s| s.entry.id == "large-v3-turbo-q5_0").unwrap();
        assert_eq!(q5_st.installed_path, Some(models_dir.join("ggml-large-v3-turbo-q5_0.bin")));
        assert!(q5_st.in_own_dir);

        // Remove
        let removed = remove(&models_dir, "large-v3-turbo-q5_0").unwrap();
        assert_eq!(removed, models_dir.join("ggml-large-v3-turbo-q5_0.bin"));
        assert!(!models_dir.join("ggml-large-v3-turbo-q5_0.bin").exists());
        assert!(!models_dir.join("ggml-large-v3-turbo-q5_0.bin.part").exists());

        // Removing again gives error
        assert!(remove(&models_dir, "large-v3-turbo-q5_0").is_err());

        // External model: `remove` on GhostReel's own dir when file only exists externally does nothing and errors
        assert!(!models_dir.join("ggml-tiny.bin").exists());
        let tiny_st = status(&models_dir, &[]).into_iter().find(|s| s.entry.id == "tiny").unwrap();
        if !tiny_st.in_own_dir {
            // Never removes files outside models_dir
            assert!(remove(&models_dir, "tiny").is_err());
        }
    }

    #[test]
    fn effective_models_dir_respects_config() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths { config_file: dir.path().join("config.toml"), data_dir: dir.path().join("data") };
        let mut cfg = Config::default();
        assert_eq!(effective_models_dir(&paths, &cfg), paths.models_dir());

        let custom = dir.path().join("custom_models");
        cfg.models.dir = Some(custom.clone());
        assert_eq!(effective_models_dir(&paths, &cfg), custom);
    }
}
