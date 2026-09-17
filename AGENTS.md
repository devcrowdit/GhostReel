# AGENTS.md — GhostReel

Instructions for AI coding agents in this repo (symlinked as `CLAUDE.md`).

## What this is

GhostReel (formerly "HighVid") indexes folders of video so you can search inside them: whisper
transcripts, keyframes described by a local vision model, embeddings, hybrid keyword + vector
search that jumps to the timestamp. Tauri v2 desktop app + `ghostreel` CLI sharing
`ghostreel-core`. Sibling of `~/Code/ghostpen`.

Source of truth: [`.agents/plan.md`](.agents/plan.md) (decisions D1–D12, runtime profiles §2a,
S0 findings §11). Progress: [`.agents/TODO.md`](.agents/TODO.md).

## Layout

| path | what |
|---|---|
| `crates/ghostreel-core` | config, paths, SQLite+FTS5+sqlite-vec DB, server probing, doctor |
| `crates/ghostreel-cli` | `ghostreel` binary (`doctor`, `config`) |
| `src-tauri` | desktop app (`ghostreel` app binary, lib `ghostreel_lib`) |
| `src/` | React + TS frontend (Vite) |
| `spikes/s0-local` | throwaway S0 spike (separate workspace, CUDA builds) |

## Build & run

```bash
cargo test --workspace            # core tests (no GPU, no servers needed)
cargo run -p ghostreel-cli -- doctor
npm install && npx tauri dev      # desktop app
npx tauri build --no-bundle       # release app binary (embedded frontend)
```
`GHOSTREEL_CONFIG=<file>` / `GHOSTREEL_DATA=<dir>` override config/data locations (tests, demos).

## Critical rules

1. **Two first-class runtime profiles** (plan §2a): *standalone* (models in-process, Windows +
   Linux, nothing else installed) and *shared servers* (highllama vision `:8089`, highllama
   embeddings `:8091`, GhostPen STT `:8771`). Each capability has `auto|local|server`. Never load
   a local model while the matching server is usable — the dev box must not hold models twice.
2. **Only use a server that is reachable *and capable*** (`probe.rs`): vision needs
   `modalities.vision`; embeddings must be 768-dim `embeddinggemma-300M-Q8_0` (a chat model's
   CLS vectors silently ruin search); STT needs `capabilities.segments` (timestamps).
3. **One embedding model everywhere** (`embeddinggemma-300M-Q8_0`, 768-dim) so indexes are
   portable between profiles. The DB records `embed_model`/`embed_dim`.
4. **whisper-rs and llama-cpp-2 can't share a binary** (duplicate ggml symbols, S0) →
   local transcription will run in a separate `ghostreel-asr` helper.
5. **VRAM budget is the RTX 3070 8 GB.** On the 4070 dev box keep highllama at `KVTYPE=q4_0`;
   GhostPen's whisper segfaults when VRAM runs out.
6. CUDA builds: `CMAKE_BUILD_PARALLEL_LEVEL=4` — full parallel llama.cpp+whisper.cpp CUDA builds OOM.
7. Wayland: keep `apply_wayland_webkit_workaround()` (WebKit DMABUF "Error 71").
8. Migrations in `db.rs` are append-only.
9. Don't create branches in `~/Code/ghostpen` or `~/Code/highllama`; work on main there.
