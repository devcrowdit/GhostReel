# Changelog

All notable changes to GhostReel. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [SemVer](https://semver.org/) while the project is 0.x (minor = features, patch = fixes).

## [Unreleased]

## [0.1.6] — 2026-09-18

The standalone script chat went from producing empty, silent scripts to complete, narrated,
interview-led cuts that land on the target length; the index now measures how steady the camera
is and the editor cuts around the shaky stretches; camera originals play in the app through a proxy.

### Added
- **Camera steadiness, measured at index time.** Each video is analysed from its own pictures — a
  patch grid with contrast selection, a robust similarity fit (translation, rotation, scale) with
  a median consensus so a person moving in the shot is not mistaken for the camera, and the camera
  path split by frequency at 30 fps. Two quantities per 4 s window: *tremor* (fast movement the
  hand adds) and *sway* (movement within a second that is undone — the slow back-and-forth of an
  unstabilised walking shot). Decoding runs on the GPU when it can (8× faster), CPU otherwise.
  Validated against ffmpeg's vid.stab on eleven clips (agreement within ~0.1) and against the
  editor's own eye on four labelled stretches.
- **Camera style per clip** — `static`, `tripod`, `stabilised` or `handheld` — read from the
  tremor floor and how much the camera moves on purpose. Shown in the Library's new **Camera**
  column and reported to the script editor by `list_videos`, `search_moments` and `get_video`.
- **Shaky stretches as timestamps**, not a verdict on the file: `get_video` lists `shaky_at`
  ranges, search hits carry a `shaky` flag, and the video panel draws a clickable shake bar under
  the player. A stretch is shaky when tremor or sway is over its line, and — for handheld footage
  — when it is twice the clip's own ordinary level, so a handheld clip keeps its usual stretches
  and loses only the worse ones.
- **The script editor never cuts from a shaky stretch.** A clip that lands on one is moved to the
  nearest steady stretch of the same shot, or dropped with the reason when the shot has none.
- **Thinking, configurable per capability.** The local model can reason before it answers; the
  grammar that constrains its JSON now binds only after `</think>`, and the thought is capped at
  half the output budget so it cannot starve the answer. On for script drafts, off for per-keyframe
  descriptions; `vision.think` / `chat_model.think` in config, CLI and the Models page.
- **`[script]` settings** — every number the script generator uses (clip lengths, target
  tolerances, narration pace, research budget, shake limits), settable with `ghostreel config set`
  and over MCP.
- **MCP**: `get_settings`, `set_settings` (same keys as the CLI) and `preview_script`, so a remote
  GhostReel can be configured, driven and checked from another machine.
- **codex** as a fourth coding-agent CLI backend (`codex exec --json`, images attached with `-i`).
- **Remove a chat session** from the Scripts panel (its scripts are kept).
- **Playback proxy** for camera originals: a 4K 10-bit 4:2:2 file sat black in the player for a
  long time; the app now builds a 720p copy on first open (NVENC when available) and keeps it.
  Keyframes, previews and exports still use the original.
- Rule 6a for the editor: when people talk on camera, build the story from what they say — cut to
  whole sentences on their own audio; voice-over is for the scenery between.

### Changed
- **Research budget**: a server or coding-agent CLI brain gets 60 tool rounds and full-size tool
  results (it was 8 rounds and 1500 characters, sized for a small local model); a local model gets
  10. `chat_model.max_tool_rounds` overrides either.
- **Grounding**: a clip the model never opened is no longer thrown away. If the range is inside the
  video and has indexed keyframes or speech, it is kept and reported as checked; only ranges with
  nothing indexed are dropped. With enough unopened picks a whole script used to collapse to
  "unable to assemble a script".
- **Fitting to target** is one pass that measures once: pictures give way first, speech only if
  that is not enough, and pictures are held longer (never past the voice-over) when the cut is
  short. The final pass before saving works to 2 % of the target. Three independent cutters used to
  compound, and a cut at 58.6 s once came out at 29.5 s.
- **Speech-aware audio**: `get_video` and `list_videos` report whether anyone speaks; a clip with
  no transcript in range cannot carry source audio, and a beat playing someone's own audio cannot
  also carry narration. The editor is told to prefer a mounted or stabilised take when two clips
  cover the same moment.
- The draft prompt lists the footage the model is allowed to cut; it is asked to come in ~10 %
  long, since trimming is safe and growing is not.
- Beats left silent get their narration written one at a time: a small model does that reliably,
  and reliably leaves `narration` empty when it is one field of a nested draft.
- Tool result and round limits, and every other pipeline constant, moved to `[script]`.

### Fixed
- **Models page: "CLI agent" could not be selected** — the desktop app rejected `backend = "cli"`
  outright, so the control snapped back. Selecting it also fills in the default tool; the picker
  used to show `claude` while the config held an empty string.
- **CLI agents looked "not installed" in the app.** A desktop launch inherits the session's PATH
  (`/usr/local/bin:/usr/bin`), not the shell's; `claude`, `agy`, `opencode` and `codex` in
  `~/.local/bin` or `~/.bun/bin` are now found.
- **Standalone script chat lost its draft to a 2048-token cap.** Every local completion asked for
  at most 2048 tokens; a real script runs past that, the helper stopped mid-object, and the one
  place that swallowed the parse error reported "unable to assemble a script" with no reason. The
  draft now gets a budget out of the configured context, and a cut-off answer is an error that says
  so.
- **Silent scripts.** `get_video` never said whether anyone speaks, so the model marked scenery
  `source` and then obeyed the rule that silences narration where people talk. Interviews were
  invisible in `list_videos`, so talking heads were cut as b-roll and muted mid-sentence under a
  voice-over.
- `codex` treats a non-TTY stdin as extra prompt input and hangs; every CLI tool now runs with stdin
  closed.
- Tests: fake CLI scripts used `echo` with `\n`, which dash (Ubuntu's `/bin/sh`) expands; CI failed
  while the Arch dev box passed. They use `printf` now.
- Repo references point at `highercomve/GhostReel` after the transfer; the updater endpoint follows
  the redirect either way.

### Notes for the standalone profile
- Measured on the reference footage (96 clips, 88.7 GB): a full index peaks at 6.5 GB of VRAM and a
  script turn at 6.8 GB with `chat_model.ctx_tokens = 16384`, `kv_cache = q4_0` — inside the 8 GB
  budget.
- A rebuilt `ghostreel-llm` must be built with `scripts/build-helpers.sh` (CUDA); a plain
  `cargo build` silently produces a CPU-only helper.
- Moving the repository leaves stale absolute paths in `target/`; clean `target/*/build/` for
  `tauri` and `llama-cpp-sys-2` after a move.

## [0.1.5] and earlier

See the git history: `git log v0.1.5`.

[Unreleased]: https://github.com/highercomve/GhostReel/compare/v0.1.6...HEAD
[0.1.6]: https://github.com/highercomve/GhostReel/compare/v0.1.5...v0.1.6
