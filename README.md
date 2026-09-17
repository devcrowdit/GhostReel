# GhostReel

Search inside your videos — locally. GhostReel watches folders of video, transcribes the speech,
describes keyframes with a local vision model, and lets you search by what was said, shown or
written on screen, jumping straight to the moment.

Runs fully standalone (models in-process, CUDA) on Windows and Linux, or reuses servers you
already run: highllama for vision + embeddings, GhostPen for transcription.

Status: early development (M0). See [`.agents/plan.md`](.agents/plan.md).

```bash
cargo run -p ghostreel-cli -- doctor   # what GhostReel would use on this machine
npx tauri dev                          # desktop app
```
