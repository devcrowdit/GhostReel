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

- [x] M2: transcripts — `ghostreel-asr` helper (whisper-rs CUDA, JSON lines, `--cpu`), GhostPen client
      (verbose_json, 5-min chunks cut at silences with 2 s pre-roll, seam de-dup), local runner with
      GPU→CPU retry on CUDA OOM, repeat/non-speech cleanup, whisper model auto-pick (turbo on ≥6 GB
      NVIDIA, else small) + resumable download, `transcribe` stage (skips no-audio videos, postpones
      when unavailable), CLI `transcript [--srt|--json]`, app transcript column + panel with filter
- [x] Indexing progress bar + ETA (learned per-stage rates)

## Next — M3 (frames)
- [ ] keyframe sampling: scene change + max interval, 1280 px long side, JPEG thumbs in `<data>/thumbs/<hash>/`
- [ ] perceptual-hash dedupe (dHash), cap frames per minute
- [ ] `frames` stage after probe; frames table rows; app: frame strip per video

## Open / to review with Sergio
- [ ] Visual check of M2 UI (monitors were asleep during the night run; app built fine, logic tested)
- [ ] Whisper `small` (GhostPen) vs `large-v3-turbo` (local) quality — turbo clearly better on names/Spanish
- [ ] Desktop crash notifications during the night were from intentional asr OOM tests (GPU busy)

## Later milestones
See plan §9: M2 transcripts (ghostreel-asr + GhostPen client), M3 frames, M4 model manager +
vision, M5 chunk/embed/search, M6 UI, M7 packaging (NSIS, AppImage/deb/rpm/CLI tarball) + CI,
**M8 (last): per-project script chat → timeline preview → OpenTimelineIO → FCP XML for Premiere Pro** (plan §4a).

## Open
- [ ] S0 on Windows (3070): MSVC CUDA build, bundled cuBLAS, CPU fallback, VRAM with desktop
- [ ] Linux packaging checks: `$ORIGIN/lib` CUDA libs, glibc 2.35 container, no /opt/cuda
- [ ] Embedding fingerprint: confirm highllama llama.cpp and llama-cpp-2 vectors agree (≥ 0.999)
