//! `ghostreel` — headless GhostReel CLI.

use std::io::{IsTerminal, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::str::FromStr;
use std::time::Duration;

use anyhow::{Context, bail};
use clap::{Parser, Subcommand};
use ghostreel_core::config::Config;
use ghostreel_core::db::Db;
use ghostreel_core::doctor::{self, Report};
use ghostreel_core::export::ExportFormat;
use ghostreel_core::index::{self, Event, IndexLock};
use ghostreel_core::paths::Paths;
use ghostreel_core::probe::{Resolution, Target};
use ghostreel_core::progress::{Progress, eta_text};
use ghostreel_core::projects::NewProject;
use ghostreel_core::runtime;
use ghostreel_core::script::{Audio, IssueSeverity, Script};
use ghostreel_core::watch::FolderWatcher;

#[derive(Parser)]
#[command(name = "ghostreel", version, about = "Search inside your videos — locally")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Check ffmpeg, GPU, database, models and the AI backends GhostReel would use.
    Doctor {
        /// Machine-readable output.
        #[arg(long)]
        json: bool,
    },
    /// Show or create the configuration.
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },
    /// Manage projects.
    Project {
        #[command(subcommand)]
        action: ProjectAction,
    },
    /// Manage the video folders a project watches.
    Folder {
        #[command(subcommand)]
        action: FolderAction,
    },
    /// Scan folders and run pending indexing jobs (resumable).
    Index {
        /// Only this project's folders (default: all projects).
        #[arg(long, short)]
        project: Option<String>,
        /// Keep running and re-index when files change.
        #[arg(long)]
        watch: bool,
        /// Also retry jobs that already failed the maximum number of times.
        #[arg(long)]
        retry_failed: bool,
        #[arg(long)]
        json: bool,
    },
    /// Print a video's transcript.
    Transcript {
        /// Video id (see `ghostreel status --videos`).
        video_id: i64,
        /// SubRip subtitles instead of plain text.
        #[arg(long)]
        srt: bool,
        #[arg(long)]
        json: bool,
    },
    /// Search a project's videos by meaning and keywords.
    Search {
        query: String,
        #[arg(long, short)]
        project: Option<String>,
        #[arg(long, default_value_t = 10)]
        limit: usize,
        /// Only these chunk kinds: moment, transcript, frame.
        #[arg(long, value_delimiter = ',')]
        kind: Option<Vec<String>>,
        /// Keywords only (don't load an embedding model).
        #[arg(long)]
        keywords: bool,
        #[arg(long)]
        json: bool,
    },
    /// List a video's keyframes (JPEG paths).
    Frames {
        video_id: i64,
        #[arg(long)]
        json: bool,
    },
    /// Show indexing progress.
    Status {
        #[arg(long, short)]
        project: Option<String>,
        /// List every video.
        #[arg(long)]
        videos: bool,
        #[arg(long)]
        json: bool,
    },
    /// Manage and export video editing scripts.
    Script {
        #[command(subcommand)]
        action: ScriptAction,
    },
}

#[derive(Subcommand)]
enum ScriptAction {
    /// Import a script JSON file into a project.
    Import {
        file: PathBuf,
        #[arg(long, short)]
        project: String,
        /// Save even if validation reported errors.
        #[arg(long)]
        force: bool,
    },
    /// List scripts in a project.
    List {
        #[arg(long, short)]
        project: String,
        #[arg(long)]
        json: bool,
    },
    /// Display a script draft.
    Show {
        id: i64,
        #[arg(long)]
        json: bool,
    },
    /// Export a script as an OpenTimelineIO (.otio) or Final Cut Pro 7 XML (.xml) timeline.
    Export {
        id: i64,
        #[arg(long, default_value = "fcp_xml")]
        format: String,
        #[arg(long, short = 'o', alias = "output")]
        out: PathBuf,
    },
    /// Render a preview MP4 of a script timeline.
    Preview {
        id: i64,
        #[arg(long, short = 'o', alias = "output")]
        out: Option<PathBuf>,
        #[arg(long)]
        burn_titles: bool,
        #[arg(long)]
        burn_narration: bool,
    },
}

#[derive(Subcommand)]
enum ProjectAction {
    /// Create a project (sequence settings are used by timeline exports).
    Create {
        name: String,
        #[arg(long, default_value = "")]
        description: String,
        /// Frame rate, e.g. 25, 30, 29.97, 23.976 or 30000/1001.
        #[arg(long, default_value = "25")]
        fps: String,
        #[arg(long, default_value_t = 1920)]
        width: i64,
        #[arg(long, default_value_t = 1080)]
        height: i64,
    },
    List {
        #[arg(long)]
        json: bool,
    },
    Show {
        name: String,
        #[arg(long)]
        json: bool,
    },
    /// Delete a project (indexed video data is kept for reuse).
    Remove { name: String },
}

#[derive(Subcommand)]
enum FolderAction {
    Add {
        path: PathBuf,
        #[arg(long, short)]
        project: String,
        /// Only files directly in the folder.
        #[arg(long)]
        no_recursive: bool,
    },
    List {
        #[arg(long, short)]
        project: Option<String>,
    },
    Remove {
        path: PathBuf,
        #[arg(long, short)]
        project: String,
    },
}

#[derive(Subcommand)]
enum ConfigAction {
    /// Print the config file path.
    Path,
    /// Print the effective configuration (defaults applied).
    Show,
    /// Write the default configuration if no config file exists yet.
    Init,
}

#[tokio::main]
async fn main() -> ExitCode {
    match run(Cli::parse()).await {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

async fn run(cli: Cli) -> anyhow::Result<ExitCode> {
    let paths = Paths::resolve()?;
    match cli.command {
        Command::Doctor { json } => {
            let report = doctor::run(&paths).await;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                print_report(&report);
            }
            Ok(if report.blockers().is_empty() { ExitCode::SUCCESS } else { ExitCode::from(2) })
        }
        Command::Project { action } => project_cmd(&paths, action),
        Command::Folder { action } => folder_cmd(&paths, action),
        Command::Index { project, watch, retry_failed, json } => {
            index_cmd(&paths, project.as_deref(), watch, retry_failed, json).await
        }
        Command::Status { project, videos, json } => status_cmd(&paths, project.as_deref(), videos, json),
        Command::Transcript { video_id, srt, json } => transcript_cmd(&paths, video_id, srt, json),
        Command::Search { query, project, limit, kind, keywords, json } => {
            search_cmd(&paths, &query, project.as_deref(), limit, kind, keywords, json).await
        }
        Command::Frames { video_id, json } => {
            let rows = index::frames(&open_db(&paths)?, &paths.data_dir, video_id)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&rows)?);
            } else if rows.is_empty() {
                println!("no frames for video #{video_id} yet");
            } else {
                for f in rows {
                    println!("[{}] {}", human_duration(f.t_s), f.path.display());
                    if let Some(d) =
                        f.description.as_deref().and_then(|j| serde_json::from_str::<serde_json::Value>(j).ok())
                    {
                        if let Some(err) = d["error"].as_str() {
                            println!("      ✗ {err}");
                        } else {
                            println!("      {}", d["description"].as_str().unwrap_or_default());
                            if let Some(t) = f.visible_text.as_deref().filter(|t| !t.is_empty()) {
                                println!("      on screen: {}", t.replace('\n', " · "));
                            }
                        }
                    }
                }
            }
            Ok(ExitCode::SUCCESS)
        }
        Command::Script { action } => script_cmd(&paths, action).await,
        Command::Config { action } => {
            match action {
                ConfigAction::Path => println!("{}", paths.config_file.display()),
                ConfigAction::Show => {
                    let cfg = Config::load(&paths.config_file)?;
                    print!("{}", cfg.to_toml()?);
                }
                ConfigAction::Init => {
                    if paths.config_file.exists() {
                        println!("exists: {}", paths.config_file.display());
                    } else {
                        Config::default().save(&paths.config_file)?;
                        println!("created: {}", paths.config_file.display());
                    }
                }
            }
            Ok(ExitCode::SUCCESS)
        }
    }
}

/// "25", "29.97", "23.976", "30000/1001" → (num, den).
fn parse_fps(s: &str) -> anyhow::Result<(i64, i64)> {
    if let Some((n, d)) = s.split_once('/') {
        return Ok((n.trim().parse()?, d.trim().parse()?));
    }
    Ok(match s.trim() {
        "23.976" | "23.98" => (24000, 1001),
        "29.97" => (30000, 1001),
        "59.94" => (60000, 1001),
        other => {
            let f: f64 = other.parse().with_context(|| format!("invalid fps '{other}'"))?;
            if f.fract() != 0.0 {
                bail!("use a fraction for non-integer fps, e.g. 30000/1001");
            }
            (f as i64, 1)
        }
    })
}

fn open_db(paths: &Paths) -> anyhow::Result<Db> {
    Ok(Db::open(&paths.db_file())?)
}

fn project_id(db: &Db, name: Option<&str>) -> anyhow::Result<Option<i64>> {
    Ok(match name {
        Some(n) => Some(db.require_project(n)?.id),
        None => None,
    })
}

fn fps_label(num: i64, den: i64) -> String {
    if den == 1 { num.to_string() } else { format!("{:.3} ({num}/{den})", num as f64 / den as f64) }
}

fn project_cmd(paths: &Paths, action: ProjectAction) -> anyhow::Result<ExitCode> {
    let mut db = open_db(paths)?;
    match action {
        ProjectAction::Create { name, description, fps, width, height } => {
            let (fps_num, fps_den) = parse_fps(&fps)?;
            let p = db.create_project(&NewProject { name, description, fps_num, fps_den, width, height })?;
            println!(
                "created project '{}' ({}x{} @ {} fps)",
                p.name,
                p.width,
                p.height,
                fps_label(p.fps_num, p.fps_den)
            );
        }
        ProjectAction::List { json } => {
            let projects = db.projects()?;
            if json {
                println!("{}", serde_json::to_string_pretty(&projects)?);
            } else if projects.is_empty() {
                println!("no projects — create one with: ghostreel project create <name>");
            } else {
                for p in projects {
                    let st = index::status(&db, Some(p.id))?;
                    println!(
                        "{:<24} {:>3} folders {:>5} videos  {}x{} @ {}",
                        p.name,
                        st.folders,
                        st.videos,
                        p.width,
                        p.height,
                        fps_label(p.fps_num, p.fps_den)
                    );
                }
            }
        }
        ProjectAction::Show { name, json } => {
            let p = db.require_project(&name)?;
            let folders = db.folders(Some(p.id))?;
            let st = index::status(&db, Some(p.id))?;
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(
                        &serde_json::json!({ "project": p, "folders": folders, "status": st })
                    )?
                );
            } else {
                println!("{} — {}x{} @ {} fps", p.name, p.width, p.height, fps_label(p.fps_num, p.fps_den));
                if !p.description.is_empty() {
                    println!("  {}", p.description);
                }
                for f in folders {
                    println!("  folder {}{}", f.path.display(), if f.recursive { "" } else { " (not recursive)" });
                }
                print_status(&st);
            }
        }
        ProjectAction::Remove { name } => {
            let p = db.require_project(&name)?;
            db.remove_project(p.id)?;
            println!("removed project '{}'", p.name);
        }
    }
    Ok(ExitCode::SUCCESS)
}

fn folder_cmd(paths: &Paths, action: FolderAction) -> anyhow::Result<ExitCode> {
    let mut db = open_db(paths)?;
    match action {
        FolderAction::Add { path, project, no_recursive } => {
            let p = db.require_project(&project)?;
            let f = db.add_folder(p.id, &path, !no_recursive)?;
            println!("'{}' now watches {} — run: ghostreel index -p \"{}\"", p.name, f.path.display(), p.name);
        }
        FolderAction::List { project } => {
            let pid = project_id(&db, project.as_deref())?;
            for f in db.folders(pid)? {
                let exists = if f.path.is_dir() { "" } else { "  (missing)" };
                println!("{}{}", f.path.display(), exists);
            }
        }
        FolderAction::Remove { path, project } => {
            let p = db.require_project(&project)?;
            db.remove_folder(p.id, &path)?;
            println!("'{}' no longer watches {}", p.name, path.display());
        }
    }
    Ok(ExitCode::SUCCESS)
}

async fn script_cmd(paths: &Paths, action: ScriptAction) -> anyhow::Result<ExitCode> {
    let db = open_db(paths)?;
    match action {
        ScriptAction::Import { file, project, force } => {
            let p = db.require_project(&project)?;
            let content = std::fs::read_to_string(&file)
                .with_context(|| format!("cannot read script file '{}'", file.display()))?;
            let mut script = Script::parse_for_project(&content, &p).with_context(|| "failed to parse script JSON")?;
            let snapped = ghostreel_core::script::snap_to_segments(&db, &mut script)?;
            if snapped > 0 {
                println!("snapped {snapped} clip boundary(ies) to speech segments");
            }
            let issues = ghostreel_core::script::validate(&db, p.id, &script)?;
            for issue in &issues {
                let tag = match issue.severity {
                    IssueSeverity::Error => "error",
                    IssueSeverity::Warning => "warning",
                    IssueSeverity::Info => "info",
                };
                if let Some(beat) = &issue.beat_id {
                    if let Some(clip) = issue.clip_index {
                        println!("  [{tag}] beat {beat} clip {}: {}", clip + 1, issue.message);
                    } else {
                        println!("  [{tag}] beat {beat}: {}", issue.message);
                    }
                } else {
                    println!("  [{tag}] {}", issue.message);
                }
            }
            let has_errors = issues.iter().any(|i| i.severity == IssueSeverity::Error);
            if has_errors && !force {
                bail!("script validation failed with errors; use --force to save anyway");
            }
            let id = ghostreel_core::script::save_version(&db, p.id, &script, None)?;
            let stored = ghostreel_core::script::load(&db, id)?;
            println!("saved script #{} \"{}\" v{}", stored.id, stored.title, stored.version);
        }
        ScriptAction::List { project, json } => {
            let p = db.require_project(&project)?;
            let list = ghostreel_core::script::list(&db, p.id)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&list)?);
            } else if list.is_empty() {
                println!("no scripts in project '{}'", p.name);
            } else {
                for s in list {
                    println!(
                        "#{:<4} {:<24} v{:<3} {:>2} beats {:>2} clips {:>6.1}s",
                        s.id, s.title, s.version, s.beats, s.clips, s.duration_s
                    );
                }
            }
        }
        ScriptAction::Show { id, json } => {
            let stored = ghostreel_core::script::load(&db, id)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&stored)?);
            } else {
                println!(
                    "script #{} \"{}\" v{} ({} beats, {:.1}s)",
                    stored.id,
                    stored.title,
                    stored.version,
                    stored.script.beats.len(),
                    stored.script.total_duration_s()
                );
                for beat in &stored.script.beats {
                    println!("\nbeat [{}] {}:", beat.id, beat.purpose);
                    if let Some(narr) = &beat.narration {
                        println!("  narration: \"{narr}\"");
                    }
                    if let Some(text) = &beat.on_screen_text {
                        println!("  on-screen: \"{text}\"");
                    }
                    if let Some(notes) = &beat.notes {
                        println!("  notes: {notes}");
                    }
                    for clip in &beat.clips {
                        let dur = (clip.out_s - clip.in_s).max(0.0);
                        let audio = match clip.audio {
                            Audio::Source => "source",
                            Audio::Mute => "mute",
                        };
                        let why = clip.why.as_deref().map(|w| format!(" {w}")).unwrap_or_default();
                        println!(
                            "    #{} {:.2}-{:.2} ({:.2}s) {}{}",
                            clip.video_id, clip.in_s, clip.out_s, dur, audio, why
                        );
                    }
                }
            }
        }
        ScriptAction::Export { id, format, out } => {
            let export_fmt = ExportFormat::from_str(&format)?;
            let res = ghostreel_core::export::export_script(&db, &paths.data_dir, id, export_fmt, &out)?;
            println!("{}", res.path.display());
            if ghostreel_core::export::locate_sidecar().is_some() {
                match ghostreel_core::export::validate_export(&res.path) {
                    Ok(v) => println!("{}", serde_json::to_string(&v)?),
                    Err(e) => eprintln!("warning: validation failed: {e}"),
                }
            }
        }
        ScriptAction::Preview { id, out, burn_titles, burn_narration } => {
            let ffmpeg = doctor::locate("ffmpeg").context("ffmpeg not found")?;
            let opts = ghostreel_core::preview::PreviewOptions { burn_titles, burn_narration, out, cancel: None };
            let t0 = std::time::Instant::now();
            let is_tty = std::io::stderr().is_terminal();
            let res = ghostreel_core::preview::render_preview(&db, &paths.data_dir, &ffmpeg, id, &opts, |pct, msg| {
                if is_tty {
                    eprint!("\r\x1b[2K[{:>3.0}%] {}", pct * 100.0, msg);
                    let _ = std::io::stderr().flush();
                } else {
                    eprintln!("[{:>3.0}%] {}", pct * 100.0, msg);
                }
            })?;
            if is_tty {
                eprint!("\r\x1b[2K");
            }
            println!(
                "preview: {} ({:.1} s, {} segments, {} built / {} cached, {}) in {:.1} s",
                res.path.display(),
                res.duration_s,
                res.segments,
                res.proxies_built,
                res.proxies_cached,
                res.encoder,
                t0.elapsed().as_secs_f64()
            );
        }
    }
    Ok(ExitCode::SUCCESS)
}

async fn index_cmd(
    paths: &Paths,
    project: Option<&str>,
    watch: bool,
    retry_failed: bool,
    json: bool,
) -> anyhow::Result<ExitCode> {
    let mut db = open_db(paths)?;
    let pid = project_id(&db, project)?;
    let config = Config::load(&paths.config_file)?;
    let opts = index::Options { project_id: pid, retry_failed, settle_secs: if watch { 10 } else { 0 }, cancel: None };

    // On a terminal: one live progress line (per-video successes are implied by it).
    let tty = !json && std::io::stderr().is_terminal();
    let mut bar_visible = false;
    let mut print = |e: Event| {
        if json {
            if let Ok(line) = serde_json::to_string(&e) {
                println!("{line}");
            }
            return;
        }
        if bar_visible && !matches!(e, Event::Progress(_) | Event::JobDone { .. } | Event::JobStarted { .. }) {
            eprint!("\r\x1b[2K");
            bar_visible = false;
        }
        match e {
            Event::ScanFolder { path } => println!("scan  {}", path.display()),
            Event::FolderMissing { path } => println!("  ! folder not available (kept index): {}", path.display()),
            Event::Scanned { new, changed, unchanged, removed } => {
                println!("  {new} new, {changed} changed, {unchanged} unchanged, {removed} removed")
            }
            Event::JobStarted { .. } => {}
            Event::StageBackend { stage, backend } => println!("{stage}: {backend}"),
            Event::StageUnavailable { stage, reason } => println!("  ! {stage} postponed: {reason}"),
            Event::DownloadingModel { file } => println!("downloading {file}…"),
            Event::JobDone { video_id, stage } => {
                if !tty {
                    println!("  ✓ {stage} #{video_id}")
                }
            }
            Event::JobFailed { video_id, stage, error } => println!("  ✗ {stage} #{video_id}: {error}"),
            Event::Progress(p) => {
                if tty {
                    eprint!("\r\x1b[2K{}", progress_line(&p));
                    let _ = std::io::stderr().flush();
                    bar_visible = true;
                }
            }
        }
    };

    let mut failed_total = 0;
    loop {
        // Hold the indexer lock only while a run is active, so a long `--watch` doesn't block
        // other projects from indexing while it sits idle.
        let lock = match IndexLock::acquire(&paths.data_dir) {
            Ok(l) => l,
            Err(e @ ghostreel_core::Error::Busy(_)) if watch => {
                if !json {
                    println!("{e}; retrying in 15 s");
                }
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_secs(15)) => continue,
                    _ = shutdown_signal() => break,
                }
            }
            Err(e) => return Err(e.into()),
        };
        // Re-resolved every run: GhostPen/highllama may have started or stopped meanwhile.
        let rt = runtime::resolve(paths, &config).await?;
        let s = index::run(&mut db, &rt, &opts, &mut print).await?;
        drop(lock);
        if tty {
            eprint!("\r\x1b[2K");
        }
        failed_total += s.jobs_failed;
        if !json {
            println!(
                "done: {} jobs ok, {} failed{}",
                s.jobs_done,
                s.jobs_failed,
                if s.unsettled > 0 { format!(", {} still copying", s.unsettled) } else { String::new() }
            );
        }
        if !watch {
            break;
        }
        let folders: Vec<(PathBuf, bool)> = db.folders(pid)?.into_iter().map(|f| (f.path, f.recursive)).collect();
        let mut watcher = FolderWatcher::new(&folders)?;
        if !json {
            println!("watching {} folder(s) — Ctrl+C to stop", folders.len());
        }
        let wait_settle = s.unsettled > 0;
        tokio::select! {
            changed = watcher.changed(Duration::from_secs(3)) => if !changed { break },
            _ = tokio::time::sleep(Duration::from_secs(12)), if wait_settle => {},
            _ = shutdown_signal() => break,
        }
    }
    Ok(if failed_total == 0 { ExitCode::SUCCESS } else { ExitCode::from(3) })
}

/// `[██████░░░░░░]  48%  probe 12/25  · about 2 min left · clip.mp4`
fn progress_line(p: &Progress) -> String {
    const WIDTH: usize = 24;
    let filled = ((p.fraction * WIDTH as f64).round() as usize).min(WIDTH);
    let eta = match p.eta_secs {
        Some(s) if p.fraction < 1.0 => format!(" · {} left", eta_text(s)),
        _ => String::new(),
    };
    let current = p
        .current
        .as_ref()
        .and_then(|c| c.file_name())
        .map(|n| format!(" · {}", n.to_string_lossy()))
        .unwrap_or_default();
    let phase = match p.phase.as_str() {
        "hash" => "reading files",
        "probe" => "video details",
        "download" => "downloading model",
        "transcribe_server" | "transcribe_local" => "transcribing",
        "frames" => "keyframes",
        "download_vision" => "downloading vision model",
        "describe_server" | "describe_local" => "describing frames",
        "download_embed" => "downloading embedding model",
        "embed" => "search index",
        other => other,
    };
    format!(
        "[{}{}] {:>3.0}%  {phase} {}/{}{eta}{current}",
        "█".repeat(filled),
        "░".repeat(WIDTH - filled),
        p.fraction * 100.0,
        p.phase_done,
        p.phase_total
    )
}

/// Ctrl+C, or SIGTERM on Unix (systemd / `kill`).
async fn shutdown_signal() {
    #[cfg(unix)]
    {
        let mut term = match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(s) => s,
            Err(_) => return std::future::pending().await,
        };
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {},
            _ = term.recv() => {},
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

fn human_size(bytes: i64) -> String {
    let b = bytes as f64;
    if b >= 1e9 { format!("{:.1} GB", b / 1e9) } else { format!("{:.0} MB", b / 1e6) }
}

fn human_duration(s: f64) -> String {
    let s = s.round() as i64;
    if s >= 3600 { format!("{}h{:02}m", s / 3600, s % 3600 / 60) } else { format!("{}m{:02}s", s / 60, s % 60) }
}

fn print_status(st: &index::Status) {
    println!(
        "  {} videos, {}, {} of footage{}",
        st.videos,
        human_size(st.total_size),
        human_duration(st.total_duration_s),
        if st.vfr_videos > 0 { format!(", {} variable-frame-rate", st.vfr_videos) } else { String::new() }
    );
    for c in &st.stages {
        let skipped = if c.skipped > 0 { format!(", {} skipped", c.skipped) } else { String::new() };
        println!(
            "  {:<10} {} done, {} pending, {} running, {} failed{skipped}",
            c.stage, c.done, c.pending, c.running, c.failed
        );
    }
}

#[allow(clippy::too_many_arguments)]
async fn search_cmd(
    paths: &Paths,
    query: &str,
    project: Option<&str>,
    limit: usize,
    kinds: Option<Vec<String>>,
    keywords_only: bool,
    json: bool,
) -> anyhow::Result<ExitCode> {
    let db = open_db(paths)?;
    let pid = project_id(&db, project)?;
    let mut embedder = None;
    if !keywords_only {
        let config = Config::load(&paths.config_file)?;
        let setup = runtime::resolve_embed(paths, &config).await;
        match runtime::start_embedder(&setup, |_, _| {}).await {
            Ok(e) => embedder = Some(e),
            Err(why) => eprintln!("(meaning search unavailable: {why}; using keywords only)"),
        }
    }
    let opts = ghostreel_core::search::SearchOptions { project_id: pid, limit, kinds };
    let hits = ghostreel_core::search::search(&db, &paths.data_dir, query, embedder.as_mut(), &opts).await?;
    if json {
        println!("{}", serde_json::to_string_pretty(&hits)?);
        return Ok(ExitCode::SUCCESS);
    }
    if hits.is_empty() {
        println!("no matches");
    }
    for (i, h) in hits.iter().enumerate() {
        let name = h.path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        println!(
            "{:>2}. {name} @ {}–{}  [{}; {}]",
            i + 1,
            human_duration(h.start_s),
            human_duration(h.end_s),
            h.kinds.join("+"),
            h.matched_by.join("+")
        );
        println!("    {}", h.snippet.replace('\n', " · "));
    }
    Ok(ExitCode::SUCCESS)
}

fn srt_time(s: f64) -> String {
    let ms = (s.max(0.0) * 1000.0).round() as u64;
    format!("{:02}:{:02}:{:02},{:03}", ms / 3_600_000, ms / 60_000 % 60, ms / 1000 % 60, ms % 1000)
}

fn transcript_cmd(paths: &Paths, video_id: i64, srt: bool, json: bool) -> anyhow::Result<ExitCode> {
    let db = open_db(paths)?;
    let segments = index::transcript(&db, video_id)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&segments)?);
    } else if srt {
        for (i, s) in segments.iter().enumerate() {
            println!("{}\n{} --> {}\n{}\n", i + 1, srt_time(s.start), srt_time(s.end), s.text);
        }
    } else if segments.is_empty() {
        println!("no transcript for video #{video_id} (not transcribed yet, or no speech)");
    } else {
        for s in segments {
            println!("[{}] {}", human_duration(s.start), s.text);
        }
    }
    Ok(ExitCode::SUCCESS)
}

fn status_cmd(paths: &Paths, project: Option<&str>, videos: bool, json: bool) -> anyhow::Result<ExitCode> {
    let db = open_db(paths)?;
    let pid = project_id(&db, project)?;
    let st = index::status(&db, pid)?;
    let rows = if videos { Some(index::videos(&db, pid)?) } else { None };
    if json {
        println!("{}", serde_json::to_string_pretty(&serde_json::json!({ "status": st, "videos": rows }))?);
        return Ok(ExitCode::SUCCESS);
    }
    println!("{}", project.map(|p| format!("Project {p}")).unwrap_or_else(|| "All projects".into()));
    print_status(&st);
    for v in rows.unwrap_or_default() {
        let dims = match (v.width, v.height) {
            (Some(w), Some(h)) => format!("{w}x{h}"),
            _ => "-".into(),
        };
        let fps = v.fps.map(|f| format!("{f:.2}fps")).unwrap_or_default();
        let dur = v.duration_s.map(human_duration).unwrap_or_else(|| "-".into());
        let flags = format!(
            "{}{}{}{}{}",
            if v.vfr { " VFR" } else { "" },
            if v.has_audio == Some(false) { " no-audio" } else { "" },
            if v.segments > 0 {
                format!(" 📝{}{}", v.segments, v.language.as_deref().map(|l| format!(" {l}")).unwrap_or_default())
            } else {
                String::new()
            },
            if v.frames > 0 { format!(" 🖼{}", v.frames) } else { String::new() },
            if v.copies > 1 { format!(" ×{}", v.copies) } else { String::new() }
        );
        println!("  #{:<4} {:<8} {:>7} {:>9} {:<9}{} {}", v.id, v.status, dur, dims, fps, flags, v.path.display());
        if let Some(e) = v.error {
            println!("        {e}");
        }
    }
    Ok(ExitCode::SUCCESS)
}

fn mark(ok: bool) -> &'static str {
    if ok { "✓" } else { "✗" }
}

fn print_backend(name: &str, r: &Resolution) {
    let target = match r.target {
        Target::Server => "server",
        Target::Local => "local",
        Target::Unavailable => "UNAVAILABLE",
    };
    let where_ = r
        .probe
        .as_ref()
        .filter(|_| r.target == Target::Server)
        .map(|p| format!(" {} @ {}", p.model.as_deref().unwrap_or("?"), p.url))
        .unwrap_or_default();
    println!(
        "  {} {name:<11} {target}{where_}  [backend = {}] {}",
        mark(r.target != Target::Unavailable),
        r.backend,
        r.reason
    );
}

fn print_report(r: &Report) {
    println!("GhostReel {}", r.version);
    println!(
        "  config  {}{}",
        r.config_file.display(),
        r.config_error.as_ref().map(|e| format!("  ✗ {e}")).unwrap_or_default()
    );
    println!("  data    {}", r.data_dir.display());

    println!("\nDatabase");
    match r.db.ok {
        true => println!(
            "  ✓ {} (schema v{}, sqlite-vec {})",
            r.db.path.display(),
            r.db.schema_version.unwrap_or(0),
            r.db.sqlite_vec.as_deref().unwrap_or("?")
        ),
        false => println!("  ✗ {}: {}", r.db.path.display(), r.db.error.as_deref().unwrap_or("?")),
    }

    println!("\nTools");
    for t in [&r.ffmpeg, &r.ffprobe] {
        match &t.path {
            Some(p) => println!("  ✓ {:<8} {} ({})", t.name, t.version.as_deref().unwrap_or("?"), p.display()),
            None => println!("  ✗ {:<8} not found", t.name),
        }
    }

    println!("\nGPU");
    if r.gpu.is_empty() {
        println!("  - no NVIDIA GPU detected (local models would run on CPU)");
    }
    for g in &r.gpu {
        println!("  ✓ {} — {} / {} MiB used, driver {}", g.name, g.vram_used_mib, g.vram_total_mib, g.driver);
    }

    println!("\nAI backends");
    print_backend("vision", &r.vision);
    print_backend("embeddings", &r.embeddings);
    print_backend("stt", &r.stt);

    println!("\nLocal model files");
    for m in &r.models {
        match &m.found {
            Some(p) => println!("  ✓ {:<25} {}", m.role, p.display()),
            None => println!("  - {:<25} {} (not downloaded)", m.role, m.pattern),
        }
    }

    let blockers = r.blockers();
    println!();
    if blockers.is_empty() {
        println!("Ready.");
    } else {
        println!("Blockers:");
        for b in blockers {
            println!("  ✗ {b}");
        }
    }
}
