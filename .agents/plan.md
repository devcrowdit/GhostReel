# GhostReel — Implementation Plan (v0 draft, 2026-09-16)

Local-first video asset search. Point GhostReel at folders of video; it transcribes speech
(whisper), samples keyframes, has a local vision model describe each frame (incl. on-screen
text), embeds everything, and gives you hybrid semantic + keyword search that jumps to the
exact timestamp. Work is organised in **projects** (each with its own video folders), and the
final feature is a per-project **script chat** that assembles an edit from the indexed footage and
exports it through **OpenTimelineIO** as FCP XML for Premiere Pro. Ships as a **Tauri v2 desktop app** and a **headless CLI** sharing one core
crate. Targets Linux, macOS, Windows.

**Hard requirement: fully standalone on Windows AND Linux.** Installs and runs on a normal
Windows or Linux PC with nothing else installed — no highllama, Ollama, Python, ffmpeg or CUDA toolkit. Models run
**in-process** via Rust bindings (`llama-cpp-2`, `whisper-rs`), accelerated with **CUDA**
(both target machines have NVIDIA GPUs), with automatic CPU fallback. The installers/packages ship all
binaries + CUDA runtime libraries; models are downloaded once on first run (then works offline).

Named as a sibling of GhostPen (formerly "HighVid"). Sibling project conventions come from `~/Code/ghostpen` (Tauri v2 + React/TS/Vite, whisper-rs
captions module, GPU feature flags, `scripts/tauri.mjs`).

---

## 1. Key decisions

| # | Decision | Choice | Why |
|---|---|---|---|
| D1 | How to run the vision LLM | **`llama-cpp-2` 0.1.156+** (utilityai) in-process, features `cuda` + `mtmd` (+ `dynamic-backends`, see D10). Default model **Bonsai-27B-Q1_0** (~3.7 GB) + its `mmproj` vision projector. | The llama.cpp equivalent of `whisper-rs`: same engine as highllama. **Verified** in the crate's vendored llama.cpp: `GGML_TYPE_Q1_0 = 41`, CUDA Q1_0 kernels (`mmq-instance-q1_0.cu`) and `tools/mtmd` (image input) are all present. No server, no ports. |
| D2 | Speech-to-text | **`whisper-rs` 0.16** with `cuda` feature; port GhostPen's `captions/model.rs` (model download). Default `large-v3-turbo` (GPU) / `base` (CPU fallback). | Proven in GhostPen; segment timestamps built-in. |
| D3 | Video decode / frames / audio | **ffmpeg + ffprobe** static binaries bundled (Tauri `externalBin`), invoked via `tokio::process`. | `ffmpeg-next` linking on Windows is painful; the CLI gives scene detection for free. |
| D4 | Index storage | **SQLite** (`rusqlite`, `bundled`) + **FTS5** (keyword) + **`sqlite-vec`** (vectors). One `ghostreel.db` file. | Single-file, zero-server, transactional with the job queue. LanceDB: heavier (Arrow, protoc), no win at this scale. |
| D5 | Embeddings | **One model everywhere: `embeddinggemma-300M-Q8_0.gguf`** (ggml-org/embeddinggemma-300M-GGUF, 319 MB, **768-dim**, GGUF-native mean pooling, L2-normalized) — the same file highllama already has. Standalone: in-process via `llama-cpp-2` (CPU is fast enough: 4 texts ≈ 0.1 s, 0 VRAM). Shared-servers: highllama's dedicated embeddings server **`http://127.0.0.1:8091/v1/embeddings`** (embeddinggemma on CPU, started by `highllama start`). Uses embeddinggemma prompt prefixes: docs `title: none \| text: …`, queries `task: search result \| query: …`. See §2c. | Same model + pooling + prefixes on both paths → vectors are interchangeable, so an index built on your box via highllama stays valid on a standalone install and vice versa. |
| D6 | Search | **Hybrid**: FTS5 BM25 + vector KNN, fused with **Reciprocal Rank Fusion**, grouped by video → moments. Optional LLM rerank / "ask your library". | Keyword catches exact on-screen text / names; vectors catch paraphrase. |
| D7 | Repo layout | **Cargo workspace**: `crates/ghostreel-core` (lib), `crates/ghostreel-cli` (bin), `src-tauri` (app), `crates/ghostreel-asr` (transcription worker, see D10). | Avoids GhostPen's feature-gated-bin trick; CLI has zero webview deps. |
| D8 | Frontend | React + TypeScript + Vite + Tailwind (same as GhostPen). | Consistency, reuse. |
| D9 | Models & hardware | **First-run wizard** reads GPU name/VRAM (from ggml's CUDA device list) and downloads models from Hugging Face with progress/resume/sha256 into `%LOCALAPPDATA%\GhostReel\models`: Bonsai-27B-Q1_0 + mmproj (~4 GB), embeddinggemma (~0.3 GB), whisper large-v3-turbo (~1.6 GB). Target GPUs: **RTX 4070 12 GB** (dev) and **RTX 3070 8 GB** (her PC) — design budget is the 3070. If VRAM < ~6 GB free, offload some layers to CPU (`n_gpu_layers`). | Models too big for the installer; one default that fits both machines. |
| D10 | ggml link conflict — **CONFIRMED in S0** | `whisper-rs` and `llama-cpp-2` each statically vendor ggml → `ld.lld: error: duplicate symbol: gguf_type_size(gguf_type)`. **Decision:** transcription runs in **`ghostreel-asr(.exe)`**, a small Rust helper binary (whisper-rs only, ~56 MB with CUDA) that the app/CLI spawns per video and reads JSON-lines segments from stdout. Main binary links only `llama-cpp-2`. | A shared-ggml single binary would need patching whisper-rs-sys to build against llama.cpp's newer ggml — fragile across upgrades. A process boundary also frees whisper's VRAM for sure before the VLM loads. |
| D12 | Transcription backends | Mirror of D11 for speech: **`LocalWhisper`** (spawns bundled `ghostreel-asr`, whisper-rs CUDA) or **`GhostPenServer`** (`POST http://127.0.0.1:8771/v1/audio/transcriptions`, GhostPen's resident whisper). `stt.backend = auto | local | server`; `auto` probes `GET /health` + a capability check (see §2b). | On the dev box GhostPen already holds a whisper model in VRAM (~0.6 GB); GhostReel must not load a second copy. |
| D13 | Projects | **One database, many projects.** `projects` own folders (`project_folders`); a video is indexed once (by content hash) and can belong to several projects via its folders. Search, chat and exports are scoped to a project. | Re-using an indexed clip in a second project costs nothing; per-project DB files would duplicate transcripts, frames and vectors. |
| D14 | Timeline export | GhostReel builds an **OpenTimelineIO** timeline (native `.otio` JSON written from Rust) and converts it with the official **`otio-fcp-adapter`** (`fcp_xml` = Final Cut Pro 7 XML / xmeml, which Premiere Pro imports via *File → Import*) running in a bundled **`ghostreel-otio` sidecar** (Python + `opentimelineio` + `otio-fcp-adapter`, frozen per OS with PyInstaller). A native Rust xmeml writer is the fallback if the sidecar proves problematic. | There is no usable Rust OTIO (crates.io `opentimelineio` is a 2020 placeholder) and OTIO adapters are Python. The sidecar keeps the standalone promise (no Python on the user's PC). Note: Premiere imports **FCP7 XML**, not FCP X `.fcpxml`. |

### Rust options for running local models in-process

| Crate | What | Fit for Bonsai-27B-Q1_0 (vision) |
|---|---|---|
| **`llama-cpp-2`** ✅ | Bindings to llama.cpp: `cuda`, `vulkan`, `metal`, `mtmd`, `dynamic-backends`, `system-ggml` features | **Chosen.** Loads Q1_0 on CUDA, supports image input via mtmd. |
| `mistral.rs` | Rust inference engine, GGUF + vision models | Won't load llama.cpp's new Q1_0 quant. |
| `candle` / `kalosm` | HF Rust ML framework / wrapper | No Q1_0; OK for small models (CLIP). |
| `fastembed-rs` / `ort` | ONNX Runtime | Embeddings/CLIP only. |

→ **D11 — two first-class AI backends** behind one `AiBackend` trait (`describe_frame`,
`embed`, `complete`, `capabilities`):

| Backend | When | Notes |
|---|---|---|
| `LlamaCppLocal` | Default on a standalone PC (her 3070) | Loads models in-process on CUDA as above. |
| `OpenAiServer` | A compatible server is already running (**highllama** on `:8089`, also llama-server / Ollama / LM Studio) | Frames sent as base64 `image_url` to `/v1/chat/completions`. **No local VLM is loaded**, so the model isn't in VRAM twice. |

Selection (setting `ai.backend = auto | local | server`, default `auto`):
1. `auto` probes the configured URL (default `http://127.0.0.1:8089`) with a 1 s timeout:
   `GET /v1/models` + `GET /props` → needs `modalities.vision = true`.
2. Reachable + vision → `OpenAiServer`; otherwise → `LlamaCppLocal` (downloading models if
   missing). The choice + model id are shown in the UI status bar and `ghostreel doctor`.
3. The probe is repeated when a job batch starts, so starting/stopping highllama is picked
   up without restarting GhostReel; a server that disappears mid-run fails the current frame
   (retried later), it doesn't silently load a local copy while the server still holds VRAM.

Server-mode details:
- Bonsai is a thinking model: send `chat_template_kwargs: {"enable_thinking": false}`, and
  if `content` still comes back empty with `reasoning_content`, retry with `max_tokens ≥ 4000`.
- Optional API key (Bearer) for non-local servers; connect timeout 5 s, request timeout
  configurable (default 180 s per frame).
- **Embeddings**: `embed.backend = auto | local | server`; server (highllama) is used only if it
  returns embeddinggemma vectors matching the fingerprint (§2c), else local CPU.
- **Whisper**: `LocalWhisper` or `GhostPenServer` (D12, §2b).

---

## 2. Architecture

```
                 ┌──────────── ghostreel-core (lib) ─────────────┐
 Tauri app ──┐   │ library   folders, watcher (notify), hashing │
             ├──►│ pipeline  job queue + stage workers          │──► ffmpeg/ffprobe (bundled)
 CLI ────────┘   │ transcribe whisper-rs (or ghostreel-asr.exe)   │
                 │ frames    scene detect, pHash dedupe, thumbs │
                 │ models    GPU detect, HF download, verify    │
                 │ vision    llama-cpp-2 + mtmd  (CUDA)         │
                 │ embed     llama-cpp-2 embeddings (CUDA/CPU)  │
                 │ index     SQLite + FTS5 + sqlite-vec         │
                 │ search    hybrid + RRF + grouping            │
                 │ config    shared settings (TOML)             │
                 └──────────────────────────────────────────────┘
```

- Config: `dirs::config_dir()/ghostreel/config.toml` (shared by app & CLI).
- Data: `dirs::data_local_dir()/ghostreel/{ghostreel.db, thumbs/, models/}` (on Windows
  `%LOCALAPPDATA%`, not roaming — models are GBs).
- Model lifecycle: load VLM on first describe, **unload when the queue drains / before whisper
  runs**, so whisper + VLM never compete for VRAM. Embedding model is small and stays loaded
  (needed for search queries anyway).
- App and CLI can run simultaneously: SQLite WAL + a lock row in `workers` so only one
  process runs the pipeline at a time (the other just reads/searches, or enqueues).

## 2a. Runtime profiles

Two independent switches — **inference** (vision LLM) and **transcription** — each `auto | local | server`:

| Profile | Vision LLM | Whisper | Embeddings | Where |
|---|---|---|---|---|
| **Standalone** | `LlamaCppLocal` (Bonsai in-process, CUDA) | `LocalWhisper` (`ghostreel-asr`, CUDA) | local embeddinggemma | Her Windows PC — nothing else installed |
| **Shared servers** | `OpenAiServer` → highllama `:8089` | `GhostPenServer` → GhostPen `:8771` | highllama embeddings `:8091` (if fingerprint matches, §2c), else local on **CPU** (0 VRAM) | Your Linux box — zero duplicated models in VRAM |
| Mixed | any | any | — | e.g. highllama up but GhostPen closed → local whisper only for that run |

Rules:
- `auto` = probe server (1 s timeout) → use it if healthy **and capable**, else local.
- Never load a local model while the matching server is reachable (would double VRAM).
- If a server disappears mid-run the job step fails and is retried later; it does not silently
  fall back to local while the server may still hold VRAM (configurable).
- Local vision model and local whisper are never resident at the same time (§2).
- `ghostreel doctor` and the app status bar show the resolved profile, e.g.
  `vision: highllama Bonsai-27B-Q1_0 @127.0.0.1:8089 · stt: GhostPen @127.0.0.1:8771 · embed: local CPU`.
- Config (`config.toml`):
  ```toml
  [vision]  backend = "auto"  url = "http://127.0.0.1:8089"  # api_key optional
  [stt]     backend = "auto"  url = "http://127.0.0.1:8771"  model = "large-v3-turbo" # local only
  [embed]   backend = "auto"  url = "http://127.0.0.1:8091"  model = "embeddinggemma-300M-Q8_0"  # fingerprint-checked
  ```
- CLI overrides: `ghostreel index --vision server|local --stt server|local`.

## 2b. GhostPen transcription server — required change

GhostPen's server today (`src-tauri/src/captions/server.rs`) returns **only**
`{"text": "..."}` (or raw text) — segment **timestamps are discarded** (`transcribe.rs`
concatenates segments). GhostReel needs per-segment times to jump to the moment in the video.

**Implemented in GhostPen (2026-09-16, uncommitted on main, installed via `scripts/install-local.sh`):**
- `Transcriber::transcribe_segments()` / `ModelPool::transcribe_segments()` → `Transcript { segments: [Segment { start, end, text }], language }`; `transcribe()` = joined text (unchanged behaviour for dictation/captions).
- `response_format=verbose_json` → `{task, language, duration, text, segments: [{id, start, end, text}]}` (seconds); `srt`/`vtt` → real subtitle files; `json`/`text` unchanged.
- `GET /v1/models` → `{"data":[{"id":"<whisper model>","capabilities":{"response_formats":[…],"segments":true}}]}` — GhostReel's capability probe.
- Verified live: 24 s two-part clip → 4 segments with correct times in all formats; GhostPen at 626 MiB VRAM alongside highllama.
- ⚠️ Found: whisper.cpp **segfaults** GhostPen (in `create_state` → `ggml_gallocr_alloc_graph`) when VRAM is exhausted instead of returning an error. GhostReel must keep total VRAM in check on the dev box (highllama `KVTYPE=q4_0`) and treat a vanished GhostPen as "server down → retry later".

GhostReel client behaviour with GhostPen:
- Extract audio with ffmpeg to **16 kHz mono FLAC/Opus** (small upload), split into
  **~5 min chunks with 2 s overlap**, send sequentially, offset + de-duplicate segments at the
  seams. Chunking keeps each request short so GhostPen's pool mutex isn't held for minutes —
  live dictation/captions stay responsive while GhostReel indexes.
- Capability probe: `GET /v1/models` → `capabilities.segments == true`. If the server answers but lacks it (old GhostPen), `auto` falls
  back to local whisper and `doctor` says "GhostPen too old for timestamps".
- Degraded option (off by default): old GhostPen + 30 s chunks → coarse 30 s timestamps.

## 2c. Embeddings — local vs highllama must produce the same vectors

Measured 2026-09-16 (4 test sentences, cosine similarity):

| Endpoint | Model answering | dim | unbox≈unbox (should be high) | unbox≈cat (should be low) |
|---|---|---|---|---|
| highllama `:8089` as currently started (`--embeddings --pooling cls` on the chat process) | **Bonsai-27B** (the `model` field is ignored) | 5120 | 0.521 | **1.000** ❌ unusable |
| `llama-server` + embeddinggemma-300M-Q8_0, native pooling, CPU (`-ngl 0`) | embeddinggemma | 768 | **0.734** | **0.337** ✅ |

**highllama change (2026-09-16, uncommitted on main):** the chat server on `:8089` no longer
CLS-pools Bonsai; embeddinggemma runs as its own llama-server on **`:8091`** (CPU, native
pooling), started/stopped with `highllama start|stop`, checked by `highllama embeddings status`
(reports model + dim). Router mode (one port for both) was tried and rejected: llama.cpp's
router returns HTTP 400 for missing/unknown `model` names, breaking crush/pi/opencode/GhostPen.

GhostReel safeguards (because a wrong embedding endpoint silently ruins search):
1. DB `meta` stores `embed_model = embeddinggemma-300M-Q8_0`, `embed_dim = 768`,
   `embed_prefix_version`, and a **fingerprint**: the vector of a fixed probe sentence.
2. Before using a server for embeddings, GhostReel embeds the probe sentence there and requires
   `dim == 768` **and** cosine ≥ 0.99 against the stored/bundled fingerprint. Otherwise →
   `auto` uses local embeddinggemma (CPU) and `doctor` prints why
   (e.g. "server returned 5120-dim vectors from Bonsai-27B-Q1_0").
3. The bundled fingerprint is generated by `LlamaCppLocal` at build time; S0 follow-up: confirm
   highllama (llama.cpp build d1d3c33) and llama-cpp-2 0.1.156 agree to ≥ 0.999.
4. Standalone and shared-servers profiles default to the same model file; the model manager
   reuses an existing copy (e.g. `~/.lmstudio/models/ggml-org/embeddinggemma-300M-GGUF/`) instead
   of downloading it again when `models.search_paths` points there.

## 3. Ingestion pipeline

Each video walks a **persisted, resumable** state machine (`jobs` table); crash/quit resumes
at the last completed stage. Stages:

1. **discover** — walk folders + `notify` watcher; filter by extension; identity =
   `blake3(size + first 4 MB + last 4 MB)` so renames/moves don't re-index.
2. **probe** — `ffprobe -show_format -show_streams -of json` → duration, codecs, fps,
   resolution, creation time, has_audio.
3. **audio** — `ffmpeg -i in -vn -ac 1 -ar 16000 -f f32le -` streamed into whisper
   (no temp WAV). Long files chunked (e.g. 10 min, 1 s overlap), timestamps offset.
4. **transcribe** — whisper segments (start, end, text, avg logprob); language auto-detect
   stored. Skip if no audio stream.
5. **frames** — scene-change sampling + floor/ceiling:
   `select='gt(scene,0.30)+isnan(prev_selected_t)+gte(t-prev_selected_t,20)'`,
   `showinfo` for pts, scale to ≤768 px long side, JPEG. Drop near-duplicates via
   perceptual hash (`image_hasher`, Hamming ≤ 6). Always keep at least 1 frame / 20 s and
   max N frames / video (configurable). Store thumbnails.
6. **describe** — per kept frame, VLM call with the image (base64 data URL) **plus the
   transcript text around that timestamp (±15 s)** as context. Ask for strict JSON:
   ```json
   {"description": "...", "visible_text": ["..."], "objects": ["..."],
    "people": "count/roles, no identification", "setting": "...",
    "shot": "screen recording|talking head|b-roll|slide|...", "tags": ["..."]}
   ```
   Bonsai is a thinking model: send `chat_template_kwargs: {"enable_thinking": false}`
   (or `max_tokens ≥ 4000` if thinking kept). Parse leniently; retry once on bad JSON.
   One frame at a time (GPU semaphore); batching several frames per context is an M4 optimization.
7. **chunk + embed** — build retrieval documents (§4), embed in batches, write FTS + vectors.
8. **summarize** (optional) — per-video summary + chapters from transcript + frame
   descriptions (text-only LLM call), also embedded.

GPU-bound stages (transcribe, describe) are serialized by a GPU semaphore; CPU stages
(probe, frames) run in parallel. Progress events → Tauri `emit` / CLI progress bars
(`indicatif`).

## 4. Data model (SQLite)

```sql
projects(id, name, description, fps_num, fps_den, width, height, created_at)   -- sequence defaults
project_folders(project_id, folder_id)                -- a folder can serve several projects
folders(id, path, recursive, enabled, added_at)
videos(id, content_hash UNIQUE, path, folder_id, size, mtime, duration_s, width, height,
       fps, vcodec, acodec, has_audio, language, summary, status, error, indexed_at)
jobs(video_id, stage, state, attempts, last_error, updated_at)
transcript_segments(id, video_id, start_s, end_s, text, confidence)
frames(id, video_id, t_s, thumb_path, phash, description_json, visible_text)
chunks(id, video_id, kind,           -- 'moment' | 'transcript' | 'frame' | 'summary'
       start_s, end_s, text, frame_id NULL)
chunks_fts USING fts5(text, content='chunks', content_rowid='id', tokenize='unicode61')
chunks_vec USING vec0(embedding float[DIM])  -- rowid = chunks.id
meta(key, value)                     -- schema_version, embed_model, embed_dim, ...
-- script chat (M8)
chat_sessions(id, project_id, title, created_at, updated_at)
chat_messages(id, session_id, role, content, tool_calls_json, created_at)
scripts(id, project_id, session_id, title, version, script_json, created_at)  -- versioned drafts
exports(id, script_id, format, path, created_at)      -- 'otio' | 'fcp_xml' | 'preview_mp4'
```
Schema v1 (M0) has no projects; **v2 (M1)** adds `projects` + `project_folders` (migrations are
append-only). Video/chunk queries join through `project_folders → folders → videos` for scoping.

**Chunk kinds** (the important retrieval unit is the *moment*):
- `moment` — frame description + visible text + transcript window around that frame.
- `transcript` — whisper segments merged into ~30 s windows, 5 s overlap.
- `frame` — visible text only (exact OCR-ish matches).
- `summary` — whole-video summary / chapters.

Migrations: `rusqlite_migration` or hand-rolled `schema_version`.

## 4a. Script chat & timeline export (final feature, M8)

Per project, the user chats with the LLM to write a video; GhostReel grounds every claim in the
project's indexed footage, produces an editable **script with real clips**, and exports it as an
**OpenTimelineIO** timeline → **FCP XML** for Premiere Pro.

### Flow
1. **Brief** — user describes the video ("90 s product teaser for the CM5 board, energetic,
   16:9, end on the logo"); optional target duration, tone, audience, must-use clips.
2. **Research (tool use)** — the model calls GhostReel tools, scoped to the project:
   | tool | returns |
   |---|---|
   | `search_moments(query, kind?, limit)` | hybrid-search hits: video, start/end, snippet, frame description, thumbnail id |
   | `get_transcript(video_id, start_s, end_s)` | timestamped segments |
   | `get_video(video_id)` | duration, fps, resolution, summary, chapters |
   | `list_videos()` | project inventory with summaries |
   Tool calls use OpenAI `tools` on the server backend (llama.cpp `--jinja`); on the local
   backend each agent step is a **JSON-schema-constrained action** (`{"tool":…,"args":…}` or
   `{"final": script}`) — same grammar trick proven in S0, no tool-call parser needed.
3. **Draft** — model returns a `Script` (JSON schema below); the app renders it next to the chat.
4. **Iterate** — chat edits ("shorter intro", "swap clip 3 for something outdoors") produce a new
   script **version**; the user can also edit directly (reorder beats, trim in/out on a mini
   timeline with the player, replace a clip from search results, edit narration).
5. **Validate** — before saving/export: every clip references an existing video in the project,
   `0 ≤ in < out ≤ duration`, in/out snapped to transcript-segment or frame boundaries (no cut
   mid-word), total duration vs target reported.
6. **Preview** — watch the timeline inside GhostReel before exporting (see *Timeline preview*).
7. **Export** — `.otio` + FCP7 `.xml` (+ optional narration `.txt`/`.srt`); open in Premiere via
   *File → Import*.

### Script schema (v1)
```json
{
  "title": "CM5 teaser", "target_duration_s": 90, "fps": 25, "width": 1920, "height": 1080,
  "beats": [
    {
      "id": "b1", "purpose": "hook",
      "narration": "What if your next board updated itself?",   // voice-over text, optional
      "on_screen_text": "Meet CM5",                              // title/lower-third, optional
      "clips": [
        { "video_id": 42, "in_s": 12.4, "out_s": 16.0, "audio": "source|mute",
          "why": "close-up of the board unboxing" }
      ],
      "notes": "fast cut"
    }
  ]
}
```

### Script → OTIO mapping
| Script | OTIO |
|---|---|
| script | `Timeline(name=title, global_start_time=0 @ fps)` |
| clips in beat order | `Track V1 (Video)`: `Clip(media_reference=ExternalReference(target_url=file:///abs/path, available_range=0..duration @ source fps), source_range=in..out)` |
| clip audio (`audio=source`) | `Track A1 (Audio)` mirroring V1 clips (muted clips → `Gap`) |
| `on_screen_text` | `Track V2`: `Gap`-backed clips with `metadata.ghostreel.title` + a **marker** (Premiere shows markers; real titles are added by the editor) |
| `narration` | **markers** on the V1 clip spanning the beat (`name` = beat id/purpose, `comment` = narration) + optional `narration.srt` sidecar file |
| `why`/`notes` | clip `metadata.ghostreel` + marker comments |
Times are `RationalTime` at the **source clip rate** for `source_range` and the **sequence rate**
(project fps) for the timeline; 29.97/23.976 handled as rational (30000/1001). Paths are absolute
`file://` URLs; Windows paths converted properly (`file:///C:/…`).

### Timeline preview
The preview is rendered **from the same OTIO timeline that gets exported** (Script → OTIO →
preview), so what you watch is exactly what Premiere will import.

- **Segment proxies (cache):** for every V1 clip, ffmpeg cuts `in..out` from the source into a
  small proxy (540p, constant frame rate = sequence fps, 48 kHz stereo AAC, keyframe at start).
  NVENC (`h264_nvenc`) when available, else `libx264 -preset veryfast`. Cached in
  `<data>/proxies/<content_hash>_<in>_<out>_<fps>.mp4`, so reordering or trimming one beat only
  re-cuts the changed clips.
- **Instant in-app playback:** a timeline player in the app plays the proxies back-to-back
  (preloading the next `<video>` element to avoid gaps), with a playhead over the mini timeline,
  the current beat highlighted, `on_screen_text` drawn as an HTML overlay and `narration` shown as
  captions. Clicking a clip seeks there; editing the script re-plays from the edited beat.
- **Rendered preview file:** "Render preview" concatenates the cached proxies (`concat` demuxer,
  stream copy — seconds, no re-encode) into `preview.mp4`, optionally burning in titles
  (`drawtext`) and narration subtitles (`subtitles=narration.srt`). Shareable, and a fallback when
  the webview can't play a codec (Linux WebKitGTK without `gst-libav` → proxies re-encoded to
  VP9/WebM, which base GStreamer plays).
- `ghostreel script preview <script-id> [-o preview.mp4] [--burn-titles] [--burn-narration]`.
- Gaps (`Gap` items, muted clips) render as black/silence of the right length so timing matches
  the export.

### `ghostreel-otio` sidecar (D14)
- Tiny Python CLI: `ghostreel-otio convert in.otio out.xml --adapter fcp_xml` and
  `ghostreel-otio validate in.xml` (reads it back to catch adapter errors).
- Pinned `opentimelineio` + `otio-fcp-adapter`; frozen with PyInstaller in CI for windows-x64 and
  linux-x64 (~30–40 MB); shipped next to `ffmpeg` in the installers. Dev box may use a venv.
- Rust writes `.otio` JSON via serde (OTIO schema `Timeline.1`, `Stack.1`, `Track.1`, `Clip.2`,
  `ExternalReference.1`, `Gap.1`, `Marker.2`, `TimeRange.1`, `RationalTime.1`) — covered by golden
  tests and by round-tripping through `otio.adapters.read_from_file` in CI.

### Acceptance
- Golden tests: Script fixtures → `.otio` JSON snapshots; `.otio` → `fcp_xml` → read back → same
  clip count, ranges and paths.
- **Manual acceptance on Premiere Pro (Windows PC):** import the XML → sequence at project fps
  and resolution, clips link to media without relinking, in/out points match, markers carry the
  narration. Also verify import in DaVinci Resolve (`.otio` native) as a bonus.

### Risks
- **Local-model tool use quality** (Bonsai 27B at 1-bit): mitigate with constrained actions,
  small tool results (top-k snippets, not full transcripts), and a "grounded only" rule —
  the validator rejects clips that weren't returned by a tool in the session.
- **Context size**: long chats + tool results; keep the script as the state and summarise old
  turns; ~16–32 k context on the 3070 budget (KV q4_0).
- **FCP XML quirks** in Premiere (frame-rate mismatches, variable-frame-rate phone footage,
  audio channel layout): detect VFR in ffprobe (M1) and warn; test with real footage early.

## 5. Search

1. Embed query → `chunks_vec` KNN top 100.
2. FTS5 `MATCH` top 100 (query sanitized; prefix + phrase support).
3. RRF (`k = 60`), optional filters (folder, duration, date, kind, has_text).
4. Group hits by video; within a video, merge hits closer than 10 s into one result
   (timestamp range, best thumbnail, snippet with highlights).
5. Optional: LLM rerank top 20; "Ask" mode synthesizes an answer citing `video@mm:ss`.

## 6. Desktop app (Tauri v2)

- **Library**: folders, per-video status/progress, errors, re-index button.
- **Search**: query bar, filter chips, result grid (thumbnail, filename, `mm:ss`, snippet).
- **Player**: HTML5 `<video>` via asset protocol (`convertFileSrc`) seeking to `t`;
  side panel with synced transcript + frame timeline. Codec caveat: WebKitGTK (Linux) needs
  GStreamer plugins, WebView2/WKWebView lack some codecs → "Open in system player at t"
  fallback (`mpv --start=`, `vlc --start-time=`, else plain open).
- **First-run wizard**: hardware check → recommended model tier → download with progress
  (pause/resume) → pick video folders → start indexing. Non-technical wording; no ports,
  endpoints or model files visible unless "Advanced" is opened.
- **Settings**: AI backend (Auto / This computer / Server URL + "Test"), whisper model, GPU on/off, frame density (Fewer/Normal/More),
  "index only while idle / plugged in"; *Advanced*: external OpenAI-compatible endpoint.
- Player fallback on Windows: WebView2 plays H.264/AAC MP4 + WebM; MKV/AVI/HEVC/ProRes open in
  the default Windows player (`opener`), or generate a small H.264 preview proxy with ffmpeg.
- Tray icon + background indexing; `tauri-plugin-single-instance`.
- Capabilities kept minimal (fs scope limited to configured folders + data dir).

## 7. CLI (`ghostreel`)

```
ghostreel project create <name> [--fps 25] | list | show <name> | remove <name>
ghostreel folder add <path> --project <name> [--no-recursive] | list [--project] | remove <path>
ghostreel index [--watch] [--only <stage>] [--force]      # run pipeline
ghostreel status [--json]
ghostreel search "<query>" --project <name> [--limit 20] [--kind moment] [--json]
ghostreel show <video> [--transcript] [--frames] [--json]
ghostreel ask "<question>"                                 # RAG answer w/ citations
ghostreel script chat --project <name> [--session <id>]         # interactive script chat (M8)
ghostreel script show <script-id> [--json]
ghostreel script preview <script-id> [-o preview.mp4] [--burn-titles] [--burn-narration]
ghostreel script export <script-id> --format fcp_xml|otio -o edit.xml   # Premiere: File → Import
ghostreel reindex --embeddings                             # after embed-model change
ghostreel doctor                                           # ffmpeg, whisper model, AI backend chosen, GPU
ghostreel index --backend server --server http://127.0.0.1:8089   # per-run override
```
`clap` derive; `--json` everywhere for scripting. Later: `ghostreel mcp` (stdio MCP server
exposing `search_videos`, `get_transcript`, `get_frame`) so Claude Code / other agents can
query the library.

## 8. Build & packaging (Windows + Linux, CUDA)

Both OSes are **release targets with identical capability** (standalone *and* shared-servers
profiles). macOS (Metal) later.

### Common
- Cargo features: `cuda` (release default) → `llama-cpp-2/cuda`, `whisper-rs/cuda`; CPU-only
  build kept compiling in CI.
- `CMAKE_CUDA_ARCHITECTURES=86;89` (RTX 30xx + 40xx — her 3070, your 4070); add `75;120` if
  ever distributed wider. `CMAKE_BUILD_PARALLEL_LEVEL=4` (full parallel CUDA build OOMs, S0).
- **CPU fallback without crashing**: `dynamic-backends` → ggml loads the CUDA backend library
  at runtime; no driver/GPU → logs and uses the CPU backend. Statically linked CUDA would fail
  to *start* without the libs — verify on both OSes.
- Shipped artifacts per OS: `ghostreel` app, `ghostreel` CLI, `ghostreel-asr` helper, `ffmpeg`/`ffprobe`
  (static, pinned + checksummed by `scripts/fetch-ffmpeg.mjs`), CUDA runtime libs (cudart,
  cuBLAS, cuBLASLt — NVIDIA EULA redistributables, ~500–600 MB). Only requirement on the host:
  an NVIDIA driver new enough for the bundled CUDA major version.
- `sqlite-vec` static, `rusqlite` `bundled`. Models downloaded on first run (~6 GB).

### Windows x64
- MSVC + CUDA Toolkit on the CI runner (`Jimver/cuda-toolkit`).
- CUDA DLLs next to the exe (`cudart64_*.dll`, `cublas64_*.dll`, `cublasLt64_*.dll`).
- Tauri **NSIS** per-user installer (no admin), WebView2 bootstrapper embedded. Code-sign if
  possible (else SmartScreen warning).

### Linux x64
- Build inside an **old-glibc container** (Ubuntu 22.04 / glibc 2.35) with the CUDA toolkit,
  so the binaries run on Arch, Ubuntu 22.04+, Debian 12, Fedora.
- CUDA `.so` files in `lib/` next to the binaries, found via **`RPATH=$ORIGIN/lib`** (set in
  build.rs / linker args) — no `LD_LIBRARY_PATH`, no system CUDA toolkit needed. The driver's
  `libcuda.so` / `libnvidia-*` always come from the host (never bundle them).
- Packages:
  | format | contents | notes |
  |---|---|---|
  | **AppImage** (primary "just download and run") | app + CLI + asr + ffmpeg + CUDA libs | `NO_STRIP=true` as in GhostPen; WebKitGTK bundled by Tauri. CLI reachable via `GhostReel.AppImage cli …` (argv dispatch) or extracted symlink. |
  | **.deb / .rpm** | installs to `/opt/ghostreel/` (bin + lib), symlinks `/usr/bin/ghostreel` | depends only on WebKitGTK 4.1 + GTK from the distro. |
  | **`ghostreel-cli-linux-x64.tar.gz`** | CLI + asr + ffmpeg + CUDA libs, **no GUI deps** | headless servers / SSH boxes; unpack anywhere and run. |
  | AUR `ghostreel-bin` (later) | repackages the tarball/deb | for your Arch box. |
- Data dirs follow XDG (`~/.config/ghostreel`, `~/.local/share/ghostreel/{models,thumbs,ghostreel.db}`).
- Video player: WebKitGTK plays via GStreamer — H.264 needs `gst-libav`/`gst-plugins-bad` on
  the host, which may be missing → same fallback as Windows: open in system player at the
  timestamp (`xdg-open`, or `mpv --start=` / `vlc --start-time=` if present), or play a small
  ffmpeg-generated preview proxy (VP9/WebM plays with base GStreamer plugins).
- Wayland + X11 both supported by Tauri; tray icon via `libayatana-appindicator` (optional).

### CI
- GitHub Actions matrix: **windows-x64-cuda** (NSIS), **linux-x64-cuda** in Ubuntu 22.04
  container (AppImage, deb, rpm, cli tarball), CPU-only check build on both.
- Release smoke test per OS: fresh VM/container without CUDA toolkit → `ghostreel doctor`
  (CPU fallback path) must pass; real GPU test on your box (Linux) and hers (Windows).

### Licensing
llama.cpp/whisper.cpp MIT; ffmpeg LGPL build; CUDA redistributables per NVIDIA EULA; model
licenses (Bonsai, embeddinggemma = Gemma terms) shown in the download step.

## 9. Milestones

| M | Scope | Done when |
|---|---|---|
| S0 | **Spike (Linux first, then Windows PC)**: describe one JPEG via `OpenAiServer` against running highllama (baseline quality/speed, no extra VRAM); then with highllama stopped, tiny Rust bin with `llama-cpp-2` `cuda`+`mtmd` loads Bonsai-27B-Q1_0 + mmproj, describes a JPEG; same binary also links `whisper-rs` `cuda` and transcribes a WAV (tests D10); build on Windows with CUDA, run on her PC (3070 8 GB); measure s/frame and peak VRAM on both GPUs; test missing-GPU fallback | Go/no-go on single binary vs `ghostreel-asr.exe`; perf numbers |
| M0 | Workspace scaffold (core/cli/tauri[/asr]), config, SQLite + migrations, `doctor` (GPU, VRAM, driver, models, ffmpeg) | `ghostreel doctor` green on Windows + Linux |
| M1 | **Projects** (schema v2) + folders per project, discovery, hashing, ffprobe (incl. VFR detection), jobs table, `index`/`status` | Two projects sharing a folder list the same videos once; indexing resumable |
| M2 | Audio → transcripts: `ghostreel-asr` helper (port GhostPen captions) **and** `GhostPenServer` client (chunked, verbose_json); GhostPen PR adding `verbose_json` segments | Same test video gives equivalent timestamped transcripts via both backends |
| M3 | Frame sampling + pHash dedupe + thumbnails | Sensible frame count per video |
| M4 | Model manager (HF download/verify) + VLM frame descriptions via `LlamaCppLocal` **and** `OpenAiServer` (highllama), `auto` probing, load/unload lifecycle | JSON descriptions stored, retry/lenient parse |
| M5 | Chunking, embeddings, FTS5 + sqlite-vec, hybrid search in CLI | `ghostreel search` returns right moments on eval queries |
| M6 | Tauri UI: library, search, player w/ seek, settings, progress | Usable end-to-end in the app |
| M7 | First-run wizard, watcher, summaries, `ask`, packaging: NSIS (Windows) + AppImage/deb/rpm/CLI tarball (Linux) + CI | Fresh Windows PC **and** fresh Linux install (no CUDA toolkit, no highllama/GhostPen): install → wizard → search works |
| M8 | **Script chat + OTIO/FCP XML export** (§4a): project-scoped tools, constrained agent loop (local) / OpenAI tools (server), versioned scripts, script editor + mini timeline, Script→`.otio` writer, **timeline preview** (cached segment proxies, in-app playback, rendered `preview.mp4`), `ghostreel-otio` sidecar (`fcp_xml`), packaging of the sidecar | Chat produces a grounded 60–90 s script from real project footage; in-app preview plays the cut with titles/narration overlays and `preview.mp4` renders in seconds from cache; exported XML imports into **Premiere Pro** on the Windows PC with correct clips, in/out and markers |
| Backlog | MCP server, CLIP image similarity, video-clip input to the VLM, macOS Metal, voice-over TTS track | — |

Keep a small **eval set** (10 videos, ~30 queries with expected video+timestamp) from M5 to
tune sampling threshold, chunk sizes, and prompts.

## 10. Risks / open questions

- **Throughput**: 27B VLM per frame is the bottleneck. 1 h video at 1 frame / 20 s ≈ 180
  describes. Mitigate: aggressive dedupe, short JSON output with thinking disabled, batch
  multiple sequences in one context, re-describe on demand.
- **VRAM budget (RTX 3070, 8 GB)** — measured file sizes from the local copy:
  | item | VRAM |
  |---|---|
  | `Bonsai-27B-Q1_0.gguf` | 3.63 GB |
  | `Bonsai-27B-mmproj-Q8_0.gguf` | 0.60 GB |
  | KV cache @ 8k ctx + compute buffers | ~0.5–1 GB (measure in S0) |
  | embeddinggemma-300M (CPU by default) | 0 GB |
  | Windows desktop / browser | ~0.5–1 GB |
  | **total** | **~6–7 GB → fits 8 GB** |
  **Measured reference (4070, highllama, 2026-09-16):** `llama-server` with Bonsai-27B-Q1_0 +
  mmproj Q8_0, `-ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -c 122880` = **7.5 GB** VRAM total. At 8k
  context the KV cache is a small fraction of that → expect ~5 GB on the 3070. Use flash
  attention + quantized KV (`q8_0` default, `q4_0` if tight) in `LlamaCppLocal` too.
  whisper large-v3-turbo (~1.6 GB) is only loaded while the VLM is unloaded. Fallbacks if
  tight: KV cache `q8_0`, smaller context, or offload a few layers to CPU. The 4070 (12 GB)
  has comfortable headroom. Decode speed should be similar on both (memory bandwidth
  448 vs 504 GB/s); ~66 t/s measured on the 4070 via highllama.
- **D10 ggml conflict** (whisper-rs + llama-cpp-2 in one binary) — decided in S0.
- **Thinking model**: Bonsai reasons in `<think>` by default; apply the chat template with
  thinking disabled, or budget ≥ 4000 tokens (slow). Test in S0.
- **`llama-cpp-2` API churn**: 0.1.x, frequent releases tracking llama.cpp; pin exact version,
  wrap behind `AiBackend`.
- **Antivirus / SmartScreen**: unsigned exe downloading GBs may be flagged; code signing
  strongly recommended.
- **Privacy**: all local; never identify real people in descriptions.
- **Open**: Product name `ghostreel`? Video types (screen
  recordings, camera footage, marketing)? Single library or several? User tags in v1?

## 11. S0 findings

### 2026-09-16 — server mode (highllama, RTX 4070, Bonsai-27B-Q1_0 + mmproj Q8_0)
Test frame: 2560×1440 H.264 screen recording, frame at t=10 s.

| Variant | Time | Tokens (prompt/gen) | Result |
|---|---|---|---|
| 768 px, thinking off, no schema, max_tokens 1500 | 22.3 s | 398 / 1500 (cut off) | **Degenerate repetition loop** in `visible_text` ("If the Nth attempt fails…" ×60). Terminal text too small at 768 px. |
| **1280 px, thinking off, `response_format: json_schema` with `maxItems`/`maxLength`, repeat_penalty 1.1** | **4.6 s** | 1003 / 223 | **Accurate, valid JSON**, key terminal text read correctly (1 name typo). ✅ |
| 1280 px, thinking on, no schema, max_tokens 5000 | 35.1 s | – / 2295 (7.7k chars reasoning) | Slightly richer tags, but some hallucinated items. 7.6× slower. |

Decisions:
- Frames sent at **1280 px long side** (screen recordings need it; ~1000 prompt tokens).
- **Always constrain output with a JSON schema** (llama.cpp grammar) incl. `maxItems`/`maxLength`
  — prevents loops and guarantees parseable output. Same schema → GBNF in `LlamaCppLocal`.
- **Thinking off** by default; optional "deep describe" setting.
- Throughput estimate: ~5 s/frame → 1 h video @ 180 frames ≈ 15 min on the 4070.

### 2026-09-16 — local mode (in-process, RTX 4070, highllama stopped)
Spike: `spikes/s0-local` (workspace: `vlm` = llama-cpp-2 0.1.156 `cuda`+`mtmd`, `asr` = whisper-rs 0.16 `cuda`).
Built on Arch, CUDA 13 toolkit (`/opt/cuda`), `CMAKE_CUDA_ARCHITECTURES=89`.

- **Build**: whisper-rs-sys CUDA ≈ 6 min, llama-cpp-sys-2 CUDA ≈ similar. Building both at full
  parallelism OOM-killed the build → use `CMAKE_BUILD_PARALLEL_LEVEL=4`, `-j 4`.
  Binary sizes: `s0-vlm` 645 MB (CUDA kernels, unstripped), `s0-asr` 56 MB.
- **Single binary: impossible as-is** (D10, duplicate ggml symbols).
- **Whisper (`ggml-small`, CUDA)**: 12.2 s clip transcribed in **0.8 s**, 938 MiB VRAM, segment
  timestamps OK. (espeak TTS clip; "GhostReel"→"HiveVid", "Pantavisor"→"Antiviser" — expected for
  small model + robot voice; retest with large-v3-turbo on real audio.)
- **Bonsai-27B-Q1_0 + mmproj Q8_0 via mtmd (CUDA)**:
  | item | value |
  |---|---|
  | model buffer | 3446 MiB |
  | KV cache (8k ctx, q8_0) | 272 MiB |
  | compute buffer (ubatch 2048 → **512**) | 2052 MiB → **513 MiB** |
  | CLIP/mmproj on CUDA0 | yes |
  | **process VRAM total** (ubatch 512) | **5.4–5.5 GB** → fits the 3070 8 GB ✅ |
  | load (warm page cache) | 0.9 s (6.4 s cold) |
  | image encode | 315 ms |
  | prompt (1003 tok incl. image) | 1.1 s |
- **Sampler matters** (same frame, JSON-schema grammar):
  | sampler chain | gen speed | frame total | valid JSON |
  |---|---|---|---|
  | grammar over full 248k vocab | 27.7 tok/s | 12.7 s | ✅ |
  | **top_k(40) → grammar → penalties → temp 0.2 → dist** | **68.8 tok/s** | **5.1 s** | ✅ |
  | no grammar | 58.3 tok/s | 5.2 s | ❌ (invalid JSON) |
  → Local mode matches highllama speed (5.1 s vs 4.6 s per frame).
- **Bugs found**: `LlamaSampler::sample()` already accepts the token — calling `accept()` again
  corrupts grammar state (`GGML_ASSERT(!stacks.empty())` abort). top_k-before-grammar can in
  theory leave zero valid candidates → production sampler must do llama.cpp's
  "sample, check with grammar, resample over full vocab if rejected" instead of plain top_k.
- **Still open for S0 on Linux packaging**: `$ORIGIN/lib` RPATH with bundled CUDA libs,
  glibc-2.35 container build, run on a machine without `/opt/cuda`.
- **Still open for S0 on Windows (her 3070)**: CUDA build with MSVC, shipping cuBLAS DLLs,
  `dynamic-backends` CPU fallback when no NVIDIA driver, real VRAM with the Windows desktop,
  whisper large-v3-turbo.

### 2026-09-17 — standalone run (no servers), RTX 4070, local helpers only
Test library: 6 videos, 8 m 34 s. `ghostreel index` end to end: **1 m 14 s**, 0 failures.
| stage | backend | notes |
|---|---|---|
| transcribe | local whisper large-v3-turbo (`ghostreel-asr`, CUDA) | peak **1994 MiB** VRAM |
| frames | ffmpeg | 14 frames |
| describe | local Bonsai-27B-Q1_0 + mmproj (`ghostreel-llm`, CUDA) | peak **5526 MiB** VRAM, ~4 s/frame |
| embed | local embeddinggemma (`ghostreel-llm`, CPU) | load 0.34 s |
- Helpers never overlap on the GPU (asr exits before llm starts) → max 5.5 GB, fits the 3070.
- **Local vs highllama embeddings: cosine 0.999748** → indexes are portable between profiles (§2c).
- Search results equivalent to the server profile ("wifi password" → Chapter 6, "fractal" → scenes).
