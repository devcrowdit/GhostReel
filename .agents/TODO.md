# GhostReel TODO

## Done
- [x] S0 spike: server mode (highllama) + local mode (llama-cpp-2 CUDA + mtmd, whisper-rs CUDA) — plan §11
- [x] GhostPen STT: `verbose_json` segments, srt/vtt, `/v1/models` capabilities (ghostpen ab0c1c4)
- [x] highllama: embeddinggemma server on :8091, chat server no longer CLS-pools (highllama 9418ac4)
- [x] localagent: embeddings → :8091, proxy relays /v1/embeddings locally (highllama 9abc7a2)
- [x] M0: workspace (core / cli / tauri), config + paths, SQLite v1 schema with FTS5 + sqlite-vec,
      server probes with capability checks, `ghostreel doctor` (CLI + app status screen), icon

- [x] M1: schema v2 (`projects`, `project_folders`, `video_files` per folder), project/folder
      management (shared + overlap-safe folders), discovery + content hash (BLAKE3 size+edges,
      moves/renames/copies don't re-index), ffprobe metadata (rotation, VFR heuristic, audio),
      persisted resumable jobs with retries, indexer lock, watcher with settle delay,
      CLI `project|folder|index [--watch]|status`, app project sidebar + Library view with live
      indexing progress

## Next — M2 (transcripts)
- [ ] `ghostreel-asr` helper binary (whisper-rs CUDA, JSON-lines segments on stdout)
- [ ] GhostPen client: chunked audio (~5 min Opus), `verbose_json`, offset + seam de-dup
- [ ] `transcribe` stage after `probe` (skip videos without audio), `auto|local|server` resolution
- [ ] whisper model download (`large-v3-turbo`) reusing GhostPen's model files when present
- [ ] app: transcript panel per video

## Later milestones
See plan §9: M2 transcripts (ghostreel-asr + GhostPen client), M3 frames, M4 model manager +
vision, M5 chunk/embed/search, M6 UI, M7 packaging (NSIS, AppImage/deb/rpm/CLI tarball) + CI,
**M8 (last): per-project script chat → timeline preview → OpenTimelineIO → FCP XML for Premiere Pro** (plan §4a).

## Open
- [ ] S0 on Windows (3070): MSVC CUDA build, bundled cuBLAS, CPU fallback, VRAM with desktop
- [ ] Linux packaging checks: `$ORIGIN/lib` CUDA libs, glibc 2.35 container, no /opt/cuda
- [ ] Embedding fingerprint: confirm highllama llama.cpp and llama-cpp-2 vectors agree (≥ 0.999)
