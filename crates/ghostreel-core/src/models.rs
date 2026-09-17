//! Model files: find an existing copy (GhostReel's models dir, configured search paths, sibling
//! apps like GhostPen/LM Studio) before downloading, and download with resume + progress.

use std::path::{Path, PathBuf};
use std::time::Duration;

use futures_util::StreamExt;
use tokio::io::AsyncWriteExt;

use crate::Error;

/// A downloadable model file.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelSpec {
    pub file_name: String,
    pub url: String,
}

/// whisper.cpp ggml model by name (`large-v3-turbo`, `small`, `base.en`, …).
pub fn whisper(model: &str) -> Result<ModelSpec, Error> {
    let name = model.trim();
    let valid = !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '.');
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
}
