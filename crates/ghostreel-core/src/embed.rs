//! Text embeddings with embeddinggemma (plan D5, §2c): highllama's embeddings server or the local
//! `ghostreel-llm` helper. Both must produce the same 768-dim vectors, so indexes stay portable.

use std::time::Duration;

use serde_json::{Value, json};

use crate::Error;
use crate::config::EMBED_DIM;
use crate::vision::{LocalLlm, LocalModels};

/// embeddinggemma's retrieval prompt formats.
pub fn doc_text(text: &str) -> String {
    format!("title: none | text: {text}")
}

pub fn query_text(query: &str) -> String {
    format!("task: search result | query: {query}")
}

pub enum Embedder {
    Server { url: String, model: String, client: reqwest::Client },
    Local(Box<LocalLlm>),
}

impl Embedder {
    pub fn server(url: &str, model: &str) -> Self {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(120))
            .build()
            .expect("static reqwest client config");
        Embedder::Server { url: url.trim_end_matches('/').to_string(), model: model.to_string(), client }
    }

    pub async fn local(models: &LocalModels) -> Result<Self, Error> {
        let llm = LocalLlm::start(models).await?;
        if llm.embed_dim != Some(EMBED_DIM) {
            return Err(Error::Embed(format!(
                "local embedding model has dim {:?}, expected {EMBED_DIM}",
                llm.embed_dim
            )));
        }
        Ok(Embedder::Local(Box::new(llm)))
    }

    pub fn label(&self) -> String {
        match self {
            Embedder::Server { url, model, .. } => format!("{model} @ {url}"),
            Embedder::Local(_) => "local embeddinggemma".into(),
        }
    }

    /// Embed already-prefixed texts (see [`doc_text`], [`query_text`]). L2-normalized.
    pub async fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, Error> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let mut out = match self {
            Embedder::Server { url, model, client } => {
                let resp = client
                    .post(format!("{url}/v1/embeddings"))
                    .json(&json!({ "input": texts, "model": model }))
                    .send()
                    .await
                    .map_err(|e| Error::Embed(format!("embeddings server at {url}: {e}")))?;
                let status = resp.status();
                if !status.is_success() {
                    return Err(Error::Embed(format!("embeddings server at {url}: HTTP {status}")));
                }
                let v: Value = resp.json().await.map_err(|e| Error::Embed(format!("embeddings server: {e}")))?;
                let mut data: Vec<(usize, Vec<f32>)> = v["data"]
                    .as_array()
                    .ok_or_else(|| Error::Embed("embeddings server: no data".into()))?
                    .iter()
                    .enumerate()
                    .map(|(i, d)| {
                        let idx = d["index"].as_u64().map(|x| x as usize).unwrap_or(i);
                        let vec = d["embedding"]
                            .as_array()
                            .map(|a| a.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect());
                        (idx, vec.unwrap_or_default())
                    })
                    .collect();
                data.sort_by_key(|(i, _)| *i);
                data.into_iter().map(|(_, v)| v).collect::<Vec<_>>()
            }
            Embedder::Local(llm) => llm.embed(texts).await.map_err(|e| Error::Embed(e.to_string()))?,
        };
        if out.len() != texts.len() {
            return Err(Error::Embed(format!("expected {} embeddings, got {}", texts.len(), out.len())));
        }
        for v in &mut out {
            if v.len() != EMBED_DIM {
                return Err(Error::Embed(format!("embedding has {} dimensions, expected {EMBED_DIM}", v.len())));
            }
            normalize(v);
        }
        Ok(out)
    }
}

pub fn normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 1e-12 {
        v.iter_mut().for_each(|x| *x /= n);
    }
}

pub fn to_blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb).max(1e-12)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefixes_and_math() {
        assert_eq!(doc_text("a"), "title: none | text: a");
        assert_eq!(query_text("b"), "task: search result | query: b");
        let mut v = vec![3.0, 4.0];
        normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((cosine(&[1.0, 0.0], &[0.0, 1.0])).abs() < 1e-6);
        assert_eq!(to_blob(&[1.0]).len(), 4);
    }
}
