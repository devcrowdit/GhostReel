<div align="center">

<img src="assets/icon.svg" alt="GhostReel" width="120" height="120" />

# GhostReel

**Search inside your videos — and cut them into a story — locally.**

</div>

GhostReel watches folders of video footage and makes everything in them searchable: it
**transcribes the speech** with Whisper, **describes keyframes** with a local vision model, and
embeds both so you can search by what was *said*, *shown* or *written on screen* and jump
straight to that moment. Organise footage in **projects**, then **chat with the model to script a
video** from your clips, preview the cut, and hand it to **Premiere Pro** as an FCP XML timeline
(via OpenTimelineIO) — or export an MP4 to send as a quick demo.

It runs **fully standalone** — models run on your GPU (CUDA) inside the app, nothing else to
install — on **Windows and Linux**. If you already run model servers, it can use them instead so
the same GPU never holds a model twice:
[highllama](https://github.com/highercomve/highllama) for vision and embeddings and
[GhostPen](https://github.com/highercomve/ghostpen) for transcription.

Built with **Tauri v2** (Rust backend + React/TypeScript frontend) and a `ghostreel` CLI sharing
the same core.

> Design docs: [`.agents/plan.md`](./.agents/plan.md) (decisions, runtime profiles, milestones),
> [`.agents/TODO.md`](./.agents/TODO.md) (build status).

---

## Features

- **Folder indexing:** add folders (whole tree, recursively) to a project; GhostReel finds
  every video and identifies it by content hash (moved or duplicated files are indexed once).
  Indexing is a resumable queue with progress and ETA; `ghostreel index --watch` keeps
  indexing new footage as it arrives.
- **Transcripts with timestamps:** Whisper (tiny → large-v3-turbo), language detection,
  click any line to jump there.
- **Keyframes described by a vision model:** scene changes plus a frame at least every *N*
  seconds (configurable), each described (subject, setting, shot, on-screen text). Scene
  detection decodes on the GPU.
- **Hybrid search:** keywords (SQLite FTS5) + meaning (embeddinggemma vectors, sqlite-vec),
  merged into *moments* you can play from the right second.
- **Script chat:** ask for "a 60 second promo about the neighbourhood using the interviews" —
  the model explores the footage with tools (search, transcripts, keyframes), drafts beats with
  clips and voice-over, and revises from your feedback. Every clip is checked against the
  footage; talking clips keep whole sentences with a breath before the cut; the editor
  instructions are editable in **Settings**.
- **Reference edits:** mark a finished video (e.g. an edit someone already made by hand) as a
  reference — the chat studies its length, pacing and structure but never uses it as footage.
  Or just remove a video from a project's library.
- **Timeline preview & export:** render a low-res preview of any script version, burn titles
  and narration, export **MP4**, **FCP 7 XML** (Premiere Pro) or **.otio**.
- **Model manager:** download and choose models per capability (speech, vision & chat,
  embeddings) and where each runs: *Auto*, *This computer* or *Server*.
- **CLI for everything:** index, search, transcripts, scripts, exports and models from a
  terminal (`ghostreel --help`).

---

## Install

Download the latest build from [Releases](https://github.com/highercomve/ghostreel/releases):

| Platform | File | Notes |
|---|---|---|
| **Windows 10/11** | `GhostReel_x.y.z_x64-setup.exe` | NVIDIA driver 570+ for GPU (CUDA 12.8 runtime is bundled) |
| **Linux** | `.AppImage` or `.deb` | NVIDIA driver 570+ for GPU |
| **CLI only** | `ghostreel-cli-*.zip` / `.tar.gz` | same helpers and ffmpeg, no desktop app |

ffmpeg/ffprobe are bundled. On first start open **Models** and download what you need:

| Capability | Default model | Size | Alternatives |
|---|---|---:|---|
| Speech | Whisper `large-v3-turbo` | 1.6 GB | `small` (488 MB), quantized turbo q8_0 / q5_0 (874 / 574 MB), `.en` variants |
| Frame descriptions & script chat | `Bonsai-27B` Q1_0 + projector | 3.8 GB | Qwen2.5-VL 7B (4.7 GB), Gemma 3 4B (2.5 GB), Qwen2.5-VL 3B (1.9 GB) |
| Search embeddings | `embeddinggemma-300M` Q8_0 | 334 MB | fixed, so indexes stay compatible between computers |

GhostReel also picks up models you already have (GhostPen's whisper models, LM Studio) instead
of downloading them again. The whole setup fits an **8 GB GPU** (RTX 3070).

---

## Two ways to run the models

| | Standalone (default on a fresh machine) | Shared servers |
|---|---|---|
| Speech | `ghostreel-asr` helper (whisper.cpp, CUDA) | GhostPen STT server `:8771` |
| Vision & chat | `ghostreel-llm` helper (llama.cpp, CUDA) | highllama `:8089` (OpenAI-compatible) |
| Embeddings | `ghostreel-llm` helper | highllama embeddings `:8091` |

Each capability is set to **Auto**, **This computer** or **Server** on the **Models** page (or
`ghostreel config set vision.backend server`). *Auto* uses a server only when it is reachable
**and capable** (vision support, timestamped transcripts, the exact embedding model) and falls
back to the local helper otherwise. `ghostreel doctor` shows what would be used and why.

Any OpenAI-compatible vision server works for frame descriptions and chat (URL, model and API
key are configurable).

---

## Usage

1. **New project** → name, frame rate and resolution (used for timeline exports).
2. **Add folder** → pick the footage folder; **Index now**. Progress shows in the project and on
   the **Activity** page. Stop any time; indexing again resumes where it left off.
3. **Search** the project: *"people talking about the park"*, *"for sale sign"*, on-screen text.
   Click a hit to play from that moment; click a video to see its keyframes and transcript.
4. **Scripts** tab → describe the video you want. Revise in the chat ("slower, let people finish
   talking") or edit beats and clips by hand; each change is a new version.
5. **Render preview** to watch the cut, then **Export for Premiere (FCP XML)**, **Export .otio**
   or **Export MP4**.

In Premiere Pro: *File → Import* the `.xml`; the sequence links to your original files.

### CLI

```bash
ghostreel doctor                                   # GPU, ffmpeg, models, servers
ghostreel project create "Greet Mag" --fps 30 --width 3840 --height 2160
ghostreel folder add ~/Videos/shoot -p "Greet Mag"
ghostreel index -p "Greet Mag"                     # --watch to keep indexing new files
ghostreel search -p "Greet Mag" "neighbours talking about the community"
ghostreel script chat --project "Greet Mag" "a 60 second promo using the interviews"
ghostreel script export <script-id> --format fcp_xml -o promo.xml
ghostreel models list | download <id> | use <id>
ghostreel config set frames.max_interval_s 5 && ghostreel index -p "Greet Mag" --redo frames
```

---

## Configuration & privacy

Everything lives in one folder: **`~/.ghostreel/`** (`%USERPROFILE%\.ghostreel\` on Windows).

| Path | What |
|---|---|
| `config.toml` | backends, models, frame interval, chat instructions |
| `ghostreel.db` | projects, transcripts, frame descriptions, vectors, scripts (SQLite) |
| `frames/` | keyframe JPEGs (thumbnails, descriptions) |
| `proxies/`, `previews/` | preview renders (safe to delete; regenerated) |
| `models/` | downloaded models |

`GHOSTREEL_HOME` moves the whole folder. Your videos are only read, never modified or copied.

- In **standalone** mode nothing leaves your machine.
- With **servers**, frames, transcripts and prompts go to the URLs you configured — point them at
  machines you trust. API keys are stored in plaintext in `config.toml`.

---

## Build from source

Requirements: **Rust** (stable), **Node.js 22**, the [Tauri prerequisites](https://v2.tauri.app/start/prerequisites/)
and **ffmpeg** on `PATH` for development. GPU helpers need the **CUDA Toolkit 12.x** (and MSVC on
Windows).

```bash
npm install
cargo test --workspace                 # core tests (no GPU, no servers)
cargo run -p ghostreel-cli -- doctor
npx tauri dev                          # desktop app with hot reload
```

Release bundles (see `AGENTS.md` → Packaging):

```bash
scripts/build-helpers.sh               # ghostreel-asr + ghostreel-llm (CUDA if the toolkit is present)
npm run bundle:linux                   # AppImage + deb (+ CLI tarball)
npm run bundle:windows                 # NSIS installer (on Windows)
```

CI builds both platforms on every `v*` tag (`.github/workflows/release.yml`).

---

## Project layout

```
crates/ghostreel-core   config, DB (SQLite + FTS5 + sqlite-vec), probing, indexing, frames,
                        search, models, script chat, preview, OTIO export
crates/ghostreel-cli    the `ghostreel` binary
crates/ghostreel-asr    whisper helper process (CUDA)
crates/ghostreel-llm    vision / chat / embeddings helper process (CUDA)
src-tauri/              desktop app: commands, task queue, local media server
src/                    React + TypeScript frontend
tools/ghostreel-otio    OpenTimelineIO → FCP XML sidecar
scripts/                helper builds, sidecar fetch/staging, packaging
.agents/                plan and TODO
```

whisper.cpp and llama.cpp run in separate helper processes because their ggml builds can't
share one binary.

## License

MIT
