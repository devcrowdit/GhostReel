//! `ghostreel-llm` — GhostReel's local model helper (llama.cpp via llama-cpp-2).
//!
//! Loads models once, then serves JSON-lines requests on stdin until EOF:
//!
//! ```text
//! → {"id":1,"cmd":"describe","image":"/f.jpg","prompt":"…","schema":{…},"max_tokens":1024}
//! ← {"id":1,"ok":true,"content":"{…json…}","prompt_tokens":1003,"gen_tokens":231,"secs":5.1}
//! → {"id":2,"cmd":"embed","texts":["…","…"]}
//! ← {"id":2,"ok":true,"embeddings":[[…],[…]],"secs":0.02}
//! ← {"id":3,"ok":false,"error":"…"}
//! ```
//!
//! Arguments: `--model <gguf> --mmproj <gguf>` (vision), `--embed-model <gguf>` (embeddings),
//! `--ctx 8192`, `--ngl 999`, `--cpu`. The first stdout line is `{"ready":true,…}` once models load.
//!
//! Sampling with a JSON-schema grammar: sample from the unconstrained distribution, check the token
//! against the grammar, and only if rejected apply the grammar to the full vocabulary and sample
//! again. That keeps generation at full speed (grammar over 248 k tokens each step is ~2.5× slower)
//! without the abort a pre-filtered candidate set can trigger when no candidate is grammatical.

use std::io::{BufRead, Write};
use std::num::NonZeroU32;
use std::process::ExitCode;
use std::time::Instant;

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::{KvCacheType, LlamaContextParams};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::mtmd::{MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::data::LlamaTokenData;
use llama_cpp_2::token::data_array::LlamaTokenDataArray;
use serde::Deserialize;
use serde_json::{Value, json};

struct Args {
    model: Option<String>,
    mmproj: Option<String>,
    embed_model: Option<String>,
    ctx: u32,
    ngl: u32,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args { model: None, mmproj: None, embed_model: None, ctx: 8192, ngl: 999 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut val = || it.next().ok_or(format!("{arg} needs a value"));
        match arg.as_str() {
            "--model" => a.model = Some(val()?),
            "--mmproj" => a.mmproj = Some(val()?),
            "--embed-model" => a.embed_model = Some(val()?),
            "--ctx" => a.ctx = val()?.parse().map_err(|_| "--ctx needs a number")?,
            "--ngl" => a.ngl = val()?.parse().map_err(|_| "--ngl needs a number")?,
            "--cpu" => a.ngl = 0,
            "--version" => {
                println!("ghostreel-llm {}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            "-h" | "--help" => {
                println!(
                    "usage: ghostreel-llm [--model m.gguf --mmproj p.gguf] [--embed-model e.gguf] [--ctx N] [--ngl N|--cpu]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if a.model.is_some() != a.mmproj.is_some() {
        return Err("--model and --mmproj go together".into());
    }
    if a.model.is_none() && a.embed_model.is_none() {
        return Err("nothing to load: pass --model/--mmproj and/or --embed-model".into());
    }
    Ok(a)
}

#[derive(Deserialize)]
struct Request {
    id: Value,
    cmd: String,
    #[serde(default)]
    image: Option<String>,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    schema: Option<Value>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    texts: Option<Vec<String>>,
}

fn reply(v: Value) {
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{v}");
    let _ = out.flush();
}

struct Vision<'a> {
    model: &'a LlamaModel,
    mtmd: MtmdContext,
    ctx: LlamaContext<'a>,
    n_batch: u32,
}

impl Vision<'_> {
    fn sample(
        &mut self,
        prompt_tokens: usize,
        n_past: i32,
        schema: Option<&Value>,
        max_tokens: usize,
        t0: Instant,
    ) -> Result<Value, String> {
        let mut grammar = match schema {
            Some(s) if !s.is_null() => {
                let g = llama_cpp_2::json_schema_to_grammar(&s.to_string()).map_err(|e| format!("schema: {e:?}"))?;
                Some(LlamaSampler::grammar(self.model, &g, "root").map_err(|e| format!("grammar: {e:?}"))?)
            }
            _ => None,
        };
        let mut chain = LlamaSampler::chain_simple([
            LlamaSampler::penalties(self.model.n_vocab(), 64, 1.1, 0.0, 0.0),
            LlamaSampler::top_k(40),
            LlamaSampler::temp(0.2),
            LlamaSampler::dist(42),
        ]);

        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut out = String::new();
        let mut batch = LlamaBatch::new(1, 1);
        let mut gen_tokens = 0usize;
        for pos in (n_past..).take(max_tokens) {
            let mut cur = self.ctx.token_data_array();
            cur.apply_sampler(&chain);
            let mut tok = cur.selected_token().ok_or("sampler selected no token")?;
            if let Some(g) = &grammar {
                let mut single = LlamaTokenDataArray::new(vec![LlamaTokenData::new(tok, 1.0, 0.0)], false);
                single.apply_sampler(g);
                if !single.data[0].logit().is_finite() {
                    // Rejected: constrain the full vocabulary, then sample.
                    let mut full = self.ctx.token_data_array();
                    full.apply_sampler(g);
                    full.apply_sampler(&chain);
                    tok = full.selected_token().ok_or("no grammatical token")?;
                }
            }
            if let Some(g) = &mut grammar {
                g.accept(tok);
            }
            chain.accept(tok);
            if self.model.is_eog_token(tok) {
                break;
            }
            out.push_str(&self.model.token_to_piece(tok, &mut decoder, false, None).map_err(|e| e.to_string())?);
            gen_tokens += 1;
            batch.clear();
            batch.add(tok, pos, &[0], true).map_err(|e| e.to_string())?;
            self.ctx.decode(&mut batch).map_err(|e| format!("decode: {e}"))?;
        }
        Ok(json!({
            "content": out,
            "prompt_tokens": prompt_tokens,
            "gen_tokens": gen_tokens,
            "truncated": gen_tokens >= max_tokens,
            "secs": t0.elapsed().as_secs_f64(),
        }))
    }

    fn describe(
        &mut self,
        image: &str,
        prompt: &str,
        schema: Option<&Value>,
        max_tokens: usize,
    ) -> Result<Value, String> {
        let t0 = Instant::now();
        self.ctx.clear_kv_cache();
        let marker = llama_cpp_2::mtmd::mtmd_default_marker();
        // ChatML with thinking disabled (Qwen-style template used by Bonsai; same as the server's
        // jinja output with enable_thinking=false).
        let text =
            format!("<|im_start|>user\n{marker}{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
        let bitmap = MtmdBitmap::from_file(&self.mtmd, image, false).map_err(|e| format!("image {image}: {e:?}"))?;
        let chunks = self
            .mtmd
            .tokenize(MtmdInputText { text, add_special: true, parse_special: true }, &[&bitmap])
            .map_err(|e| format!("tokenize: {e:?}"))?;
        let prompt_tokens = chunks.total_tokens();
        let n_past = chunks
            .eval_chunks(&self.mtmd, &self.ctx, 0, 0, self.n_batch as i32, true)
            .map_err(|e| format!("prompt eval: {e:?}"))?;

        self.sample(prompt_tokens, n_past, schema, max_tokens, t0)
    }

    fn complete(&mut self, prompt: &str, schema: Option<&Value>, max_tokens: usize) -> Result<Value, String> {
        let t0 = Instant::now();
        self.ctx.clear_kv_cache();
        let text = if prompt.starts_with("<|im_start|>") {
            if prompt.ends_with("<think>\n\n</think>\n\n") {
                prompt.to_string()
            } else {
                format!("{prompt}\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
            }
        } else {
            format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        };
        let chunks = self
            .mtmd
            .tokenize(MtmdInputText { text, add_special: true, parse_special: true }, &[])
            .map_err(|e| format!("tokenize: {e:?}"))?;
        let prompt_tokens = chunks.total_tokens();
        let n_past = chunks
            .eval_chunks(&self.mtmd, &self.ctx, 0, 0, self.n_batch as i32, true)
            .map_err(|e| format!("prompt eval: {e:?}"))?;

        self.sample(prompt_tokens, n_past, schema, max_tokens, t0)
    }
}

struct Embedder<'a> {
    model: &'a LlamaModel,
    ctx: LlamaContext<'a>,
    n_ctx: usize,
}

impl Embedder<'_> {
    fn embed(&mut self, texts: &[String]) -> Result<Value, String> {
        let t0 = Instant::now();
        let mut vectors = Vec::with_capacity(texts.len());
        for text in texts {
            let mut tokens = self
                .model
                .str_to_token(text, llama_cpp_2::model::AddBos::Always)
                .map_err(|e| format!("tokenize: {e}"))?;
            tokens.truncate(self.n_ctx);
            self.ctx.clear_kv_cache();
            let mut batch = LlamaBatch::new(tokens.len().max(1), 1);
            for (i, t) in tokens.iter().enumerate() {
                batch.add(*t, i as i32, &[0], true).map_err(|e| e.to_string())?;
            }
            if self.ctx.decode(&mut batch).is_err() {
                self.ctx.encode(&mut batch).map_err(|e| format!("encode: {e}"))?;
            }
            let v = self.ctx.embeddings_seq_ith(0).map_err(|e| format!("embeddings: {e}"))?;
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            vectors.push(v.iter().map(|x| x / norm).collect::<Vec<f32>>());
        }
        Ok(json!({ "embeddings": vectors, "secs": t0.elapsed().as_secs_f64() }))
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let backend = LlamaBackend::init().map_err(|e| e.to_string())?;
    let t0 = Instant::now();

    let vision_model = match &args.model {
        Some(path) => Some(
            LlamaModel::load_from_file(&backend, path, &LlamaModelParams::default().with_n_gpu_layers(args.ngl))
                .map_err(|e| format!("loading {path}: {e}"))?,
        ),
        None => None,
    };
    let mut vision = match (&vision_model, &args.mmproj) {
        (Some(model), Some(mmproj)) => {
            let mtmd = MtmdContext::init_from_file(
                mmproj,
                model,
                &MtmdContextParams { use_gpu: args.ngl > 0, print_timings: false, ..Default::default() },
            )
            .map_err(|e| format!("loading {mmproj}: {e:?}"))?;
            let n_batch = 512;
            let ctx = model
                .new_context(
                    &backend,
                    LlamaContextParams::default()
                        .with_n_ctx(NonZeroU32::new(args.ctx))
                        .with_n_batch(n_batch)
                        .with_n_ubatch(n_batch)
                        .with_flash_attention_policy(llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_ENABLED)
                        .with_type_k(KvCacheType::Q8_0)
                        .with_type_v(KvCacheType::Q8_0),
                )
                .map_err(|e| format!("vision context: {e}"))?;
            Some(Vision { model, mtmd, ctx, n_batch })
        }
        _ => None,
    };

    let embed_model = match &args.embed_model {
        Some(path) => Some(
            // Small model: fine on CPU, keeps VRAM for the vision model.
            LlamaModel::load_from_file(&backend, path, &LlamaModelParams::default().with_n_gpu_layers(0))
                .map_err(|e| format!("loading {path}: {e}"))?,
        ),
        None => None,
    };
    let mut embedder = match &embed_model {
        Some(model) => {
            let n_ctx = 2048u32;
            let ctx = model
                .new_context(
                    &backend,
                    LlamaContextParams::default()
                        .with_n_ctx(NonZeroU32::new(n_ctx))
                        .with_n_batch(n_ctx)
                        .with_n_ubatch(n_ctx)
                        .with_embeddings(true),
                )
                .map_err(|e| format!("embedding context: {e}"))?;
            Some(Embedder { model, ctx, n_ctx: n_ctx as usize })
        }
        None => None,
    };

    let dim = embed_model.as_ref().map(|m| m.n_embd());
    reply(
        json!({ "ready": true, "vision": vision.is_some(), "embed_dim": dim, "load_secs": t0.elapsed().as_secs_f64() }),
    );

    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let line = line.map_err(|e| e.to_string())?;
        if line.trim().is_empty() {
            continue;
        }
        let req: Request = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                reply(json!({ "id": null, "ok": false, "error": format!("bad request: {e}") }));
                continue;
            }
        };
        let result = match req.cmd.as_str() {
            "describe" => match (&mut vision, &req.image) {
                (Some(v), Some(image)) => v.describe(
                    image,
                    req.prompt.as_deref().unwrap_or("Describe this image."),
                    req.schema.as_ref(),
                    req.max_tokens.unwrap_or(1024),
                ),
                (None, _) => Err("vision model not loaded".into()),
                (_, None) => Err("describe needs image".into()),
            },
            "complete" => match (&mut vision, &req.prompt) {
                (Some(v), Some(prompt)) => v.complete(prompt, req.schema.as_ref(), req.max_tokens.unwrap_or(2048)),
                (None, _) => Err("vision/llm model not loaded".into()),
                (_, None) => Err("complete needs prompt".into()),
            },
            "embed" => match (&mut embedder, &req.texts) {
                (Some(e), Some(texts)) => e.embed(texts),
                (None, _) => Err("embedding model not loaded".into()),
                (_, None) => Err("embed needs texts".into()),
            },
            other => Err(format!("unknown cmd {other}")),
        };
        match result {
            Ok(mut v) => {
                v["id"] = req.id;
                v["ok"] = json!(true);
                reply(v);
            }
            Err(e) => reply(json!({ "id": req.id, "ok": false, "error": e })),
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("ghostreel-llm: {e}");
            ExitCode::FAILURE
        }
    }
}
