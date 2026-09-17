//! S0 spike — whisper-rs in its own binary (D10: can't share a binary with llama-cpp-2).
//!   s0-asr <ggml-whisper.bin> <audio-file>
use anyhow::{anyhow, Context, Result};
use std::{process::Command, time::Instant};

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

fn asr(model: &str, audio: &str) -> Result<()> {
    use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};
    // 16 kHz mono f32 via ffmpeg, exactly as the real pipeline will do.
    let pcm = Command::new("ffmpeg")
        .args(["-v", "error", "-i", audio, "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"])
        .output()
        .context("run ffmpeg")?
        .stdout;
    let samples: Vec<f32> = pcm.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    let t0 = Instant::now();
    let ctx = WhisperContext::new_with_params(model, WhisperContextParameters::default())
        .map_err(|e| anyhow!("whisper load: {e}"))?;
    let mut state = ctx.create_state().map_err(|e| anyhow!("state: {e}"))?;
    vram("whisper loaded");
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_language(Some("auto"));
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_special(false);
    state.full(params, &samples).map_err(|e| anyhow!("full: {e}"))?;
    for i in 0..state.full_n_segments() {
        if let Some(seg) = state.get_segment(i) {
            // timestamps are in centiseconds
            println!(
                "[{:>7.2} → {:>7.2}] {}",
                seg.start_timestamp() as f32 / 100.0,
                seg.end_timestamp() as f32 / 100.0,
                seg.to_str_lossy().unwrap_or_default().trim()
            );
        }
    }
    eprintln!(
        "[time] whisper: {:.1}s audio in {:.1}s",
        samples.len() as f32 / 16000.0,
        t0.elapsed().as_secs_f32()
    );
    Ok(())
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 {
        return Err(anyhow!("usage: s0-asr <ggml-whisper.bin> <audio-file>"));
    }
    asr(&a[1], &a[2])
}
