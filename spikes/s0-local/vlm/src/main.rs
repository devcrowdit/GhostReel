//! S0 spike — see .agents/plan.md §9/§11.
//!   s0-local vlm <model.gguf> <mmproj.gguf> <image.jpg> [n_gpu_layers]
use anyhow::{anyhow, Context, Result};
use std::{num::NonZeroU32, process::Command, time::Instant};

use llama_cpp_2::context::params::{KvCacheType, LlamaContextParams};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::mtmd::{MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText};
use llama_cpp_2::sampling::LlamaSampler;

const SCHEMA: &str = r#"{"type":"object","properties":{
 "description":{"type":"string","maxLength":600},
 "visible_text":{"type":"array","items":{"type":"string","maxLength":120},"maxItems":10},
 "objects":{"type":"array","items":{"type":"string","maxLength":40},"maxItems":10},
 "setting":{"type":"string","maxLength":120},
 "shot":{"type":"string","maxLength":40},
 "tags":{"type":"array","items":{"type":"string","maxLength":30},"maxItems":10}},
 "required":["description","visible_text","objects","setting","shot","tags"],
 "additionalProperties":false}"#;

const PROMPT: &str = "Describe this video frame for a video search index. Keep visible_text to the most \
important distinct text (max 10 short items, no repeats). Reply with ONLY JSON: {\"description\": str, \
\"visible_text\": [str], \"objects\": [str], \"setting\": str, \"shot\": str, \"tags\": [str]}";

fn vram(tag: &str) {
    let pid = std::process::id().to_string();
    let out = Command::new("nvidia-smi")
        .args(["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"])
        .output();
    let mine = out
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| {
            s.lines()
                .find(|l| l.split(',').next().map(str::trim) == Some(pid.as_str()))
                .map(|l| l.split(',').nth(1).unwrap_or("?").trim().to_string())
        })
        .unwrap_or_else(|| "n/a".into());
    eprintln!("[vram] {tag}: {mine} MiB (this process)");
}

fn vlm(backend: &LlamaBackend, model_path: &str, mmproj: &str, image: &str, ngl: u32) -> Result<()> {
    let t0 = Instant::now();
    let model = LlamaModel::load_from_file(
        backend,
        model_path,
        &LlamaModelParams::default().with_n_gpu_layers(ngl),
    )
    .context("load model")?;
    let mtmd = MtmdContext::init_from_file(
        mmproj,
        &model,
        &MtmdContextParams { use_gpu: true, print_timings: true, ..Default::default() },
    )
    .map_err(|e| anyhow!("mtmd init: {e:?}"))?;
    let n_batch: u32 = std::env::var("S0_UBATCH").ok().and_then(|v| v.parse().ok()).unwrap_or(512);
    let mut ctx = model
        .new_context(
            backend,
            LlamaContextParams::default()
                .with_n_ctx(NonZeroU32::new(8192))
                .with_n_batch(n_batch)
                .with_n_ubatch(n_batch)
                .with_flash_attention_policy(llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_ENABLED)
                .with_type_k(KvCacheType::Q8_0)
                .with_type_v(KvCacheType::Q8_0),
        )
        .context("new context")?;
    eprintln!("[time] load model+mmproj+ctx: {:.1}s", t0.elapsed().as_secs_f32());
    vram("after load");

    // Qwen-style template with thinking disabled (matches the model's jinja when enable_thinking=false).
    let marker = llama_cpp_2::mtmd::mtmd_default_marker();
    let text = format!(
        "<|im_start|>user\n{marker}{PROMPT}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
    let t1 = Instant::now();
    let bitmap = MtmdBitmap::from_file(&mtmd, image, false).map_err(|e| anyhow!("bitmap: {e:?}"))?;
    let chunks = mtmd
        .tokenize(MtmdInputText { text, add_special: true, parse_special: true }, &[&bitmap])
        .map_err(|e| anyhow!("tokenize: {e:?}"))?;
    let n_prompt = chunks.total_tokens();
    let n_past = chunks
        .eval_chunks(&mtmd, &ctx, 0, 0, n_batch as i32, true)
        .map_err(|e| anyhow!("eval: {e:?}"))?;
    let t_prompt = t1.elapsed().as_secs_f32();
    vram("after image+prompt eval");

    let grammar = llama_cpp_2::json_schema_to_grammar(SCHEMA).map_err(|e| anyhow!("schema: {e:?}"))?;
    // S0_SAMPLER: full = grammar over whole vocab (slow); topk = top_k(40) before grammar;
    // none = no grammar (speed baseline).
    let mode = std::env::var("S0_SAMPLER").unwrap_or_else(|_| "topk".into());
    let mut chain = Vec::new();
    if mode == "topk" {
        chain.push(LlamaSampler::top_k(40));
    }
    if mode != "none" {
        chain.push(LlamaSampler::grammar(&model, &grammar, "root").map_err(|e| anyhow!("grammar: {e:?}"))?);
    }
    chain.push(LlamaSampler::penalties(model.n_vocab(), 64, 1.1, 0.0, 0.0));
    chain.push(LlamaSampler::temp(0.2));
    chain.push(LlamaSampler::dist(42));
    eprintln!("[sampler] {mode}");
    let mut sampler = LlamaSampler::chain_simple(chain);

    let t2 = Instant::now();
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut out = String::new();
    let mut batch = LlamaBatch::new(1, 1);
    let mut pos = n_past;
    let mut n_gen = 0;
    for _ in 0..1200 {
        let tok = sampler.sample(&ctx, -1);
        if model.is_eog_token(tok) {
            break;
        }
        out.push_str(&model.token_to_piece(tok, &mut decoder, false, None)?);
        n_gen += 1;
        batch.clear();
        batch.add(tok, pos, &[0], true)?;
        pos += 1;
        ctx.decode(&mut batch)?;
    }
    let t_gen = t2.elapsed().as_secs_f32();
    vram("after generation");
    println!("{out}");
    let valid = serde_json::from_str::<serde_json::Value>(out.trim()).is_ok();
    eprintln!(
        "[time] prompt {n_prompt} tok in {t_prompt:.1}s | gen {n_gen} tok in {t_gen:.1}s ({:.1} tok/s) | frame total {:.1}s | valid_json={valid}",
        n_gen as f32 / t_gen,
        t_prompt + t_gen
    );
    Ok(())
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        return Err(anyhow!("usage: s0-vlm <model.gguf> <mmproj.gguf> <image> [n_gpu_layers]"));
    }
    let ngl = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(999u32);
    let backend = LlamaBackend::init()?;
    vlm(&backend, &a[1], &a[2], &a[3], ngl)
}
