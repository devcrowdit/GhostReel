# GhostReel TODO

## Done
- [x] S0 spike: server mode (highllama) + local mode (llama-cpp-2 CUDA + mtmd, whisper-rs CUDA) — plan §11
- [x] GhostPen STT: `verbose_json` segments, srt/vtt, `/v1/models` capabilities (ghostpen ab0c1c4)
- [x] highllama: embeddinggemma server on :8091, chat server no longer CLS-pools (highllama 9418ac4)
- [x] localagent: embeddings → :8091, proxy relays /v1/embeddings locally (highllama 9abc7a2)
- [x] M0: workspace (core / cli / tauri), config + paths, SQLite v1 schema with FTS5 + sqlite-vec,
      server probes with capability checks, `ghostreel doctor` (CLI + app status screen), icon

## Next — M1 (projects + library)
- [ ] schema v2: `projects`, `project_folders`; `ghostreel project create|list|show|remove`
- [ ] `ghostreel folder add --project|list|remove`
- [ ] discovery walk + extension filter, content hash (blake3 of size + first/last 4 MB)
- [ ] ffprobe metadata → `videos` (flag variable-frame-rate footage for the M8 export)
- [ ] persisted `jobs` state machine + `index` / `status` commands, resumable
- [ ] `notify` watcher (`index --watch`)
- [ ] app: project switcher + Library view (folders, per-video status)

## Later milestones
See plan §9: M2 transcripts (ghostreel-asr + GhostPen client), M3 frames, M4 model manager +
vision, M5 chunk/embed/search, M6 UI, M7 packaging (NSIS, AppImage/deb/rpm/CLI tarball) + CI,
**M8 (last): per-project script chat → OpenTimelineIO → FCP XML for Premiere Pro** (plan §4a).

## Open
- [ ] S0 on Windows (3070): MSVC CUDA build, bundled cuBLAS, CPU fallback, VRAM with desktop
- [ ] Linux packaging checks: `$ORIGIN/lib` CUDA libs, glibc 2.35 container, no /opt/cuda
- [ ] Embedding fingerprint: confirm highllama llama.cpp and llama-cpp-2 vectors agree (≥ 0.999)
