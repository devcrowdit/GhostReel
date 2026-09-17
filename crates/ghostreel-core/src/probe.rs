//! Probing the optional AI servers and resolving each `auto | local | server` switch (plan §2a).
//!
//! A server is only used when it is reachable **and capable**:
//! - vision: an OpenAI-compatible chat server whose model accepts images
//!   (llama.cpp `/props` → `modalities.vision`), e.g. highllama on :8089;
//! - embeddings: returns `embeddinggemma` vectors of [`EMBED_DIM`] (anything else would silently
//!   corrupt search, plan §2c), e.g. highllama's embeddings server on :8091;
//! - speech-to-text: GhostPen's server advertising timestamped segments on `/v1/models`.

use std::time::Duration;

use serde::Serialize;
use serde_json::{Value, json};

use crate::config::{Backend, EMBED_DIM};

/// Outcome of probing one server.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Probe {
    pub url: String,
    pub reachable: bool,
    pub capable: bool,
    /// Model the server reported/used, when known.
    pub model: Option<String>,
    /// Human-readable explanation (shown by `doctor`).
    pub detail: String,
}

impl Probe {
    fn down(url: &str, detail: impl Into<String>) -> Self {
        Self { url: url.into(), reachable: false, capable: false, model: None, detail: detail.into() }
    }
    fn incapable(url: &str, model: Option<String>, detail: impl Into<String>) -> Self {
        Self { url: url.into(), reachable: true, capable: false, model, detail: detail.into() }
    }
    fn ok(url: &str, model: Option<String>, detail: impl Into<String>) -> Self {
        Self { url: url.into(), reachable: true, capable: true, model, detail: detail.into() }
    }
}

/// Where a capability will run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Target {
    Server,
    Local,
    /// `backend = server` but the server can't do the job.
    Unavailable,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Resolution {
    pub backend: Backend,
    pub target: Target,
    pub probe: Option<Probe>,
    pub reason: String,
}

/// Decide where a capability runs. `probe` is `None` when it wasn't needed (`backend = local`).
pub fn resolve(backend: Backend, probe: Option<Probe>) -> Resolution {
    let (target, reason) = match (backend, &probe) {
        (Backend::Local, _) => (Target::Local, "backend = local".to_string()),
        (_, None) => (Target::Local, "server not probed".to_string()),
        (Backend::Auto, Some(p)) if p.capable => (Target::Server, format!("server OK: {}", p.detail)),
        (Backend::Auto, Some(p)) => (Target::Local, format!("server not usable ({}) → local", p.detail)),
        (Backend::Server, Some(p)) if p.capable => (Target::Server, format!("server OK: {}", p.detail)),
        (Backend::Server, Some(p)) => (Target::Unavailable, format!("backend = server but {}", p.detail)),
    };
    Resolution { backend, target, probe, reason }
}

/// Short-timeout client for probes: a missing server must not stall startup.
pub fn probe_client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(1))
        .timeout(Duration::from_secs(5))
        .build()
        .expect("static reqwest client config")
}

fn base(url: &str) -> &str {
    url.trim_end_matches('/')
}

async fn get_json(client: &reqwest::Client, url: &str) -> Result<Value, String> {
    let resp = client.get(url).send().await.map_err(short_err)?;
    let status = resp.status();
    if !status.is_success() {
        return Err(format!("HTTP {status}"));
    }
    resp.json().await.map_err(|e| format!("invalid JSON: {e}"))
}

fn short_err(e: reqwest::Error) -> String {
    if e.is_connect() {
        "connection refused".into()
    } else if e.is_timeout() {
        "timed out".into()
    } else {
        e.to_string()
    }
}

fn first_model_id(models: &Value) -> Option<String> {
    models["data"].as_array()?.first()?["id"].as_str().map(str::to_string)
}

/// Vision/chat server (highllama / llama-server). `model` empty = the server's loaded model.
pub async fn vision(client: &reqwest::Client, url: &str, model: &str) -> Probe {
    let b = base(url);
    let models = match get_json(client, &format!("{b}/v1/models")).await {
        Ok(v) => v,
        Err(e) => return Probe::down(url, e),
    };
    let model_id = if model.is_empty() { first_model_id(&models) } else { Some(model.to_string()) };
    let props_url = match &model_id {
        Some(m) => format!("{b}/props?model={m}"),
        None => format!("{b}/props"),
    };
    match get_json(client, &props_url).await {
        Ok(props) => match props["modalities"]["vision"].as_bool() {
            Some(true) => Probe::ok(url, model_id, "model accepts images"),
            Some(false) => Probe::incapable(url, model_id, "loaded model has no vision (no mmproj)"),
            None => Probe::incapable(url, model_id, "server does not report modalities"),
        },
        // Not llama.cpp (Ollama, LM Studio…): can't confirm images are supported.
        Err(e) => Probe::incapable(url, model_id, format!("cannot confirm vision support (/props: {e})")),
    }
}

/// Embeddings server: must return `model`-compatible vectors of [`EMBED_DIM`].
pub async fn embeddings(client: &reqwest::Client, url: &str, model: &str) -> Probe {
    let resp = client
        .post(format!("{}/v1/embeddings", base(url)))
        .json(&json!({ "input": "ghostreel probe", "model": model }))
        .send()
        .await;
    let resp = match resp {
        Ok(r) => r,
        Err(e) => return Probe::down(url, short_err(e)),
    };
    let status = resp.status();
    if !status.is_success() {
        return Probe::incapable(url, None, format!("/v1/embeddings HTTP {status}"));
    }
    let body: Value = match resp.json().await {
        Ok(v) => v,
        Err(e) => return Probe::incapable(url, None, format!("invalid JSON: {e}")),
    };
    let served = body["model"].as_str().map(str::to_string);
    let dim = body["data"][0]["embedding"].as_array().map(Vec::len).unwrap_or(0);
    if dim != EMBED_DIM {
        return Probe::incapable(
            url,
            served.clone(),
            format!("{dim}-dim vectors from {} (need {EMBED_DIM}-dim {model})", served.as_deref().unwrap_or("?")),
        );
    }
    let family = model.split('-').next().unwrap_or(model).to_lowercase();
    match &served {
        Some(s) if !s.to_lowercase().contains(&family) => {
            Probe::incapable(url, served.clone(), format!("served by {s}, not {model}"))
        }
        _ => Probe::ok(url, served, format!("{EMBED_DIM}-dim vectors")),
    }
}

/// GhostPen transcription server: needs timestamped segments.
pub async fn stt(client: &reqwest::Client, url: &str) -> Probe {
    let b = base(url);
    match get_json(client, &format!("{b}/v1/models")).await {
        Ok(models) => {
            let model = first_model_id(&models);
            if models["data"][0]["capabilities"]["segments"].as_bool() == Some(true) {
                Probe::ok(url, model, "timestamped segments")
            } else {
                Probe::incapable(url, model, "no timestamped segments (GhostPen too old)")
            }
        }
        Err(e) => match client.get(format!("{b}/health")).send().await {
            Ok(r) if r.status().is_success() => {
                Probe::incapable(url, None, format!("server up but /v1/models {e} (GhostPen too old for timestamps)"))
            }
            _ => Probe::down(url, e),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// Minimal HTTP server: `routes` maps "METHOD /path" (query ignored) to (status, body).
    async fn serve(routes: Vec<(&'static str, u16, String)>) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else { return };
                let routes = routes.clone();
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 8192];
                    let n = sock.read(&mut buf).await.unwrap_or(0);
                    let req = String::from_utf8_lossy(&buf[..n]);
                    let mut parts = req.split_whitespace();
                    let (method, target) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""));
                    let key = format!("{method} {}", target.split('?').next().unwrap_or(""));
                    let (status, body) = routes
                        .iter()
                        .find(|(k, _, _)| *k == key)
                        .map(|(_, s, b)| (*s, b.clone()))
                        .unwrap_or((404, "{}".into()));
                    let resp = format!(
                        "HTTP/1.1 {status} X\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = sock.write_all(resp.as_bytes()).await;
                });
            }
        });
        format!("http://{addr}")
    }

    fn dead_url() -> String {
        // Bind then drop: nothing listens on this port afterwards.
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        format!("http://{}", l.local_addr().unwrap())
    }

    const MODELS: &str = r#"{"data":[{"id":"Bonsai-27B-Q1_0"}]}"#;

    #[tokio::test]
    async fn vision_ok_and_without_mmproj() {
        let c = probe_client();
        let url = serve(vec![
            ("GET /v1/models", 200, MODELS.into()),
            ("GET /props", 200, r#"{"modalities":{"vision":true}}"#.into()),
        ])
        .await;
        let p = vision(&c, &url, "").await;
        assert!(p.capable, "{p:?}");
        assert_eq!(p.model.as_deref(), Some("Bonsai-27B-Q1_0"));

        let url = serve(vec![
            ("GET /v1/models", 200, MODELS.into()),
            ("GET /props", 200, r#"{"modalities":{"vision":false}}"#.into()),
        ])
        .await;
        let p = vision(&c, &url, "").await;
        assert!(p.reachable && !p.capable);

        let p = vision(&c, &dead_url(), "").await;
        assert!(!p.reachable);
    }

    #[tokio::test]
    async fn embeddings_rejects_chat_model_vectors() {
        let c = probe_client();
        let vec768 = format!("[{}]", vec!["0.1"; 768].join(","));
        let good = format!(r#"{{"model":"embeddinggemma-300M-Q8_0","data":[{{"embedding":{vec768}}}]}}"#);
        let url = serve(vec![("POST /v1/embeddings", 200, good)]).await;
        let p = embeddings(&c, &url, "embeddinggemma-300M-Q8_0").await;
        assert!(p.capable, "{p:?}");

        // The pre-fix highllama: Bonsai CLS-pooled, 5120-dim.
        let vec5120 = format!("[{}]", vec!["0.1"; 5120].join(","));
        let bad = format!(r#"{{"model":"Bonsai-27B-Q1_0","data":[{{"embedding":{vec5120}}}]}}"#);
        let url = serve(vec![("POST /v1/embeddings", 200, bad)]).await;
        let p = embeddings(&c, &url, "embeddinggemma-300M-Q8_0").await;
        assert!(p.reachable && !p.capable);
        assert!(p.detail.contains("5120"), "{}", p.detail);

        let url = serve(vec![("POST /v1/embeddings", 501, "{}".into())]).await;
        assert!(!embeddings(&c, &url, "embeddinggemma-300M-Q8_0").await.capable);
    }

    #[tokio::test]
    async fn stt_needs_segments() {
        let c = probe_client();
        let url = serve(vec![(
            "GET /v1/models",
            200,
            r#"{"data":[{"id":"small","capabilities":{"segments":true}}]}"#.into(),
        )])
        .await;
        let p = stt(&c, &url).await;
        assert!(p.capable);
        assert_eq!(p.model.as_deref(), Some("small"));

        // Old GhostPen: /health only.
        let url = serve(vec![("GET /health", 200, "ok".into())]).await;
        let p = stt(&c, &url).await;
        assert!(p.reachable && !p.capable, "{p:?}");
    }

    #[test]
    fn resolve_rules() {
        let up = Probe::ok("u", None, "fine");
        let down = Probe::down("u", "connection refused");
        assert_eq!(resolve(Backend::Auto, Some(up.clone())).target, Target::Server);
        assert_eq!(resolve(Backend::Auto, Some(down.clone())).target, Target::Local);
        assert_eq!(resolve(Backend::Server, Some(down)).target, Target::Unavailable);
        assert_eq!(resolve(Backend::Local, Some(up)).target, Target::Local);
        assert_eq!(resolve(Backend::Local, None).target, Target::Local);
    }
}
