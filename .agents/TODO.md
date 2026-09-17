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

- [x] M3: keyframes — low-res scene detection pass + 20 s max gap fillers, 1280 px JPEGs, dHash dedupe
      with 60 s coverage floor, `frames` stage, `ghostreel frames <id>`, app frame strip (asset protocol)

- [x] M4: frame descriptions — `ghostreel-llm` helper (llama-cpp-2 + mtmd, JSON lines, grammar-safe sampling:
      sample free → check grammar → constrain only on rejection), vision runtime (highllama server with
      JSON schema + thinking off, or local helper with Bonsai download), `describe` stage with ±15 s speech
      context, per-frame error records, postpone on transport errors; app frame detail
- [x] M5 (backend + CLI): chunks (transcript 30 s windows, moments, on-screen text), embeddinggemma via
      highllama :8091 or local helper (CPU), `embed` stage (re-queued when transcript/descriptions change;
      keyword-only chunks when no embedder), hybrid search (FTS5 w/ stopwords + sqlite-vec kNN, RRF,
      project scope, moments ≤ 60 s), `ghostreel search`
- [x] M8a: schema v3 (`chat_sessions`, `chat_messages`, `scripts`, `exports`), `script` module (schema v1,
      validation against project footage, transcript segment boundary snapping, versioned drafts), `otio`
      module (OpenTimelineIO .otio timeline builder with V1/A1/V2 tracks, markers, rational timebases),
      `ghostreel-otio` sidecar (OpenTimelineIO + otio-fcp-adapter FCP7 XML converter and validator), `export`
      module, and CLI `ghostreel script import|list|show|export`

## Next
- [ ] M5 app: search box + results (thumbnail, file, time, snippet) → open video panel
- [ ] M6: player (asset protocol for project folders), seek to hit/segment/frame, transcript sync
- [x] Standalone run verified: local asr+llm, 1m14s for 8.5 min footage, peak VRAM 5.5 GB, embeddings cosine 0.9997 vs highllama

## Open / to review with Sergio
- [ ] Visual check of M2 UI (monitors were asleep during the night run; app built fine, logic tested)
- [ ] Whisper `small` (GhostPen) vs `large-v3-turbo` (local) quality — turbo clearly better on names/Spanish
- [ ] Desktop crash notifications during the night were from intentional asr OOM tests (GPU busy)

## Later milestones
See plan §9: M2 transcripts (ghostreel-asr + GhostPen client), M3 frames, M4 model manager +
vision, M5 chunk/embed/search, M6 UI, M7 packaging (NSIS, AppImage/deb/rpm/CLI tarball) + CI,
**M8 (last): per-project script chat → timeline preview → OpenTimelineIO → FCP XML for Premiere Pro** (plan §4a).

## M7 packaging
- [x] `scripts/fetch-sidecars.mjs` + `scripts/sidecars.json`: pinned BtbN LGPL ffmpeg n8.1.2 (sha256-verified)
- [x] `scripts/stage-helpers.mjs`: helpers → `src-tauri/binaries/<name>-<triple>`, CUDA libs (ldd) → `src-tauri/lib/`,
      patchelf fallback, generates `src-tauri/tauri.bundle.json` overlay (externalBin + resources)
- [x] Helper RUNPATH `$ORIGIN/lib:$ORIGIN/../lib/GhostReel/lib` via `build.rs` (asr, llm)
- [x] tauri.conf.json: deb/rpm/appimage sections, NSIS `currentUser` + embedded WebView2 bootstrapper
- [x] `npm run bundle:linux|bundle:windows|bundle:cli`, `scripts/package-cli.sh` (CLI tarball)
- [x] CI: `.github/workflows/check.yml` (fmt, clippy, core tests, frontend, scripts), `release.yml` (linux-x64, windows-x64)
- [x] Local AppImage build with sidecars + CUDA libs (Linux)
- [ ] First CI run of `release.yml` (never executed); Windows job entirely untested
- [ ] Windows: CUDA DLLs via `resources {"lib/": "./"}` next to exe — verify in the NSIS install
- [ ] deb/rpm install to /usr (Tauri default), not /opt/ghostreel as plan §8 says; `/usr/bin/ghostreel` CLI symlink missing
- [ ] CLI inside AppImage (`GhostReel.AppImage cli …` argv dispatch) not done
- [ ] Fresh-machine smoke test (no CUDA toolkit): `ghostreel doctor` CPU fallback, wizard → search
- [ ] Code signing (Windows)

## Open
- [ ] S0 on Windows (3070): MSVC CUDA build, bundled cuBLAS, CPU fallback, VRAM with desktop
- [ ] Linux packaging checks: glibc 2.35 build (CI runner), no /opt/cuda on target machine
- [ ] Embedding fingerprint: confirm highllama llama.cpp and llama-cpp-2 vectors agree (≥ 0.999)
