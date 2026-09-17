//! Indexing: scan watched folders, then run the persisted, resumable job pipeline (plan §3).
//!
//! Every video walks the stages in [`STAGES`] order. Job rows live in the database, so a crash
//! or quit resumes where it stopped; `running` rows found at startup are reset to `pending`.
//! Stages so far: `probe` (M1), `transcribe` (M2); frames / describe / embed follow.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rusqlite::{OptionalExtension, params};
use serde::Serialize;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

use crate::Error;
use crate::db::Db;
use crate::media::{self, MediaInfo};
use crate::progress::{Progress, Tracker};
use crate::projects::now;
use crate::runtime::{Runtime, SttSetup};
use crate::stt::{self, Engine};

/// Pipeline stages in execution order.
pub const STAGES: &[&str] = &["probe", "transcribe"];

/// A failed job is retried on later runs until it has failed this many times.
pub const MAX_ATTEMPTS: i64 = 3;

const PARALLEL_HASH: usize = 4;
const PARALLEL_PROBE: usize = 4;

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Event {
    ScanFolder { path: PathBuf },
    FolderMissing { path: PathBuf },
    Scanned { new: usize, changed: usize, unchanged: usize, removed: usize },
    JobStarted { video_id: i64, stage: String, path: PathBuf },
    JobDone { video_id: i64, stage: String },
    JobFailed { video_id: i64, stage: String, error: String },
    /// Which backend a stage uses in this run (e.g. "GhostPen @ http://127.0.0.1:8771").
    StageBackend { stage: String, backend: String },
    /// A stage can't run now; its jobs stay pending for a later run.
    StageUnavailable { stage: String, reason: String },
    DownloadingModel { file: String },
    /// Overall progress and time remaining (throttled to ~4 per second).
    Progress(Progress),
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Summary {
    pub new: usize,
    pub changed: usize,
    pub unchanged: usize,
    pub removed: usize,
    pub jobs_done: usize,
    pub jobs_failed: usize,
    /// Files skipped because they were modified less than `settle_secs` ago (still copying).
    pub unsettled: usize,
}

#[derive(Debug, Clone, Default)]
pub struct Options {
    /// Limit scanning and jobs to one project's folders.
    pub project_id: Option<i64>,
    /// Retry jobs that exhausted their attempts.
    pub retry_failed: bool,
    /// Skip files modified within this many seconds (a copy in progress would hash garbage).
    /// The watcher uses ~10 s and re-runs later; one-shot `index` uses 0.
    pub settle_secs: i64,
}

/// Exclusive indexer lock (`<data>/indexer.lock`): the app and the CLI may both be open, but only
/// one of them runs the pipeline. Released when dropped.
pub struct IndexLock {
    _file: File,
}

impl IndexLock {
    pub fn acquire(data_dir: &Path) -> Result<Self, Error> {
        std::fs::create_dir_all(data_dir).map_err(|e| Error::Io(data_dir.to_path_buf(), e))?;
        let path = data_dir.join("indexer.lock");
        let file = File::options()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&path)
            .map_err(|e| Error::Io(path.clone(), e))?;
        match file.try_lock() {
            Ok(()) => Ok(Self { _file: file }),
            Err(std::fs::TryLockError::WouldBlock) => Err(Error::Busy(path.display().to_string())),
            Err(std::fs::TryLockError::Error(e)) => Err(Error::Io(path, e)),
        }
    }
}

/// Default speeds for a machine that has never indexed (units per second); replaced by measured
/// rates after the first runs.
///
/// `download` is bytes/s; `transcribe_*` are seconds of audio per second.
const PHASES: &[(&str, f64)] = &[
    ("hash", 40.0),
    ("probe", 15.0),
    ("download", 20_000_000.0),
    ("transcribe_server", 40.0),
    ("transcribe_local", 20.0),
];

/// Assumed length of a video whose duration isn't known yet (for the first estimate only).
const UNKNOWN_DURATION_S: f64 = 120.0;

/// Scan + run all pending jobs. Callers hold an [`IndexLock`].
///
/// Emits [`Event::Progress`] (throttled) with overall completion and time remaining.
pub async fn run(db: &mut Db, rt: &Runtime, opts: &Options, mut on_event: impl FnMut(Event)) -> Result<Summary, Error> {
    let mut summary = Summary::default();
    let mut tracker = Tracker::new(PHASES);
    tracker.load_rates(db)?;

    // 1. Walk every folder first, so the run's totals are known before the slow work starts.
    let mut pending = Vec::new();
    for folder in db.folders(opts.project_id)?.into_iter().filter(|f| f.enabled) {
        on_event(Event::ScanFolder { path: folder.path.clone() });
        if !folder.path.is_dir() {
            // Unplugged drive / network share: keep the index, don't treat files as deleted.
            on_event(Event::FolderMissing { path: folder.path.clone() });
            continue;
        }
        let w = walk_folder(db, folder.id, &folder.path, folder.recursive, opts.settle_secs).await?;
        summary.unchanged += w.unchanged;
        summary.unsettled += w.unsettled;
        pending.push(w);
    }

    reset_interrupted(db)?;
    ensure_jobs(db)?;
    let to_hash: Vec<ToHash> = pending.iter_mut().flat_map(|w| std::mem::take(&mut w.to_hash)).collect();
    let hash_work = to_hash.iter().filter(|f| f.known_hash.is_none()).count() as u64;
    let already_queued = claimable_jobs(db, "probe", opts)?.len() as u64;
    tracker.set_total("hash", hash_work);
    // Upper bound until hashing tells us which files are genuinely new content.
    tracker.set_total("probe", already_queued + to_hash.len() as u64);
    let stt_phase = stt_phase(&rt.stt);
    if let Some(phase) = stt_phase {
        let (known, unknown) = transcribe_work(db, opts)?;
        tracker.set_total(phase, (known + (unknown as f64 + to_hash.len() as f64) * UNKNOWN_DURATION_S) as u64);
    }

    // 2. Hash new/changed files (all folders together).
    tracker.start("hash");
    on_event(Event::Progress(tracker.snapshot(None)));
    let (new, changed) = hash_and_store(db, to_hash, &mut tracker, &mut on_event).await?;
    summary.new = new;
    summary.changed = changed;
    for w in &pending {
        summary.removed += db.conn.execute(
            "DELETE FROM video_files WHERE folder_id = ?1 AND last_seen < ?2",
            params![w.folder_id, w.seq],
        )?;
    }
    on_event(Event::Scanned {
        new: summary.new,
        changed: summary.changed,
        unchanged: summary.unchanged,
        removed: summary.removed,
    });

    // 3. Jobs, stage by stage.
    let (done, failed) = run_probe_jobs(db, &rt.ffprobe, opts, &mut tracker, &mut on_event).await?;
    summary.jobs_done += done;
    summary.jobs_failed += failed;

    let (done, failed) = run_transcribe_jobs(db, rt, opts, &mut tracker, &mut on_event).await?;
    summary.jobs_done += done;
    summary.jobs_failed += failed;

    tracker.finish();
    tracker.save_rates(db)?;
    on_event(Event::Progress(tracker.snapshot(None)));
    Ok(summary)
}

struct ToHash {
    folder_id: i64,
    seq: i64,
    path: PathBuf,
    size: i64,
    mtime: i64,
    existed: bool,
    /// Identity already known through an overlapping folder (no hashing needed).
    known_hash: Option<String>,
}

struct Walked {
    folder_id: i64,
    seq: i64,
    unchanged: usize,
    unsettled: usize,
    to_hash: Vec<ToHash>,
}

fn next_scan_seq(db: &Db) -> Result<i64, Error> {
    let cur: Option<String> =
        db.conn.query_row("SELECT value FROM meta WHERE key = 'scan_seq'", [], |r| r.get(0)).optional()?;
    let next = cur.and_then(|v| v.parse::<i64>().ok()).unwrap_or(0) + 1;
    db.conn.execute(
        "INSERT INTO meta(key, value) VALUES ('scan_seq', ?1) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        [next.to_string()],
    )?;
    Ok(next)
}

fn mtime_secs(m: &std::fs::Metadata) -> i64 {
    m.modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// List a folder's video files; mark unchanged/unsettled ones as seen and return the rest.
async fn walk_folder(
    db: &mut Db,
    folder_id: i64,
    root: &Path,
    recursive: bool,
    settle_secs: i64,
) -> Result<Walked, Error> {
    let seq = next_scan_seq(db)?;
    let mut w = Walked { folder_id, seq, unchanged: 0, unsettled: 0, to_hash: Vec::new() };

    // Walk (blocking IO) off the async threads.
    let root_owned = root.to_path_buf();
    let files: Vec<(PathBuf, i64, i64)> = tokio::task::spawn_blocking(move || {
        let walker = walkdir::WalkDir::new(&root_owned).max_depth(if recursive { usize::MAX } else { 1 });
        walker
            .into_iter()
            .filter_entry(|e| e.depth() == 0 || !e.file_name().to_string_lossy().starts_with('.'))
            .filter_map(Result::ok)
            .filter(|e| e.file_type().is_file() && media::is_video_path(e.path()))
            .filter_map(|e| {
                let m = e.metadata().ok()?;
                Some((e.into_path(), m.len() as i64, mtime_secs(&m)))
            })
            .collect()
    })
    .await
    .map_err(|e| Error::Invalid(format!("scan task failed: {e}")))?;

    let known: HashMap<String, (i64, i64)> = {
        let mut st = db.conn.prepare("SELECT path, size, mtime FROM video_files WHERE folder_id = ?1")?;
        st.query_map([folder_id], |r| Ok((r.get::<_, String>(0)?, (r.get::<_, i64>(1)?, r.get::<_, i64>(2)?))))?
            .collect::<Result<_, _>>()?
    };

    let settled_before = now() - settle_secs;
    let tx = db.conn.transaction()?;
    for (path, size, mtime) in files {
        let key = path.to_string_lossy().to_string();
        match known.get(&key) {
            Some(&(s, m)) if s == size && m == mtime => {
                tx.execute(
                    "UPDATE video_files SET last_seen = ?1 WHERE folder_id = ?2 AND path = ?3",
                    params![seq, folder_id, key],
                )?;
                w.unchanged += 1;
            }
            _ if settle_secs > 0 && mtime > settled_before => {
                // Still being written: keep any existing row alive, look again later.
                tx.execute(
                    "UPDATE video_files SET last_seen = ?1 WHERE folder_id = ?2 AND path = ?3",
                    params![seq, folder_id, key],
                )?;
                w.unsettled += 1;
            }
            prev => {
                // Same file already hashed through an overlapping folder: reuse its identity.
                let known_hash: Option<String> = tx
                    .query_row(
                        "SELECT v.content_hash FROM video_files vf JOIN videos v ON v.id = vf.video_id
                          WHERE vf.path = ?1 AND vf.size = ?2 AND vf.mtime = ?3 LIMIT 1",
                        params![key, size, mtime],
                        |r| r.get(0),
                    )
                    .optional()?;
                w.to_hash.push(ToHash { folder_id, seq, path, size, mtime, existed: prev.is_some(), known_hash });
            }
        }
    }
    tx.commit()?;
    Ok(w)
}

/// Hash files concurrently and record videos, file locations and their pending jobs.
async fn hash_and_store(
    db: &mut Db,
    files: Vec<ToHash>,
    tracker: &mut Tracker,
    on_event: &mut impl FnMut(Event),
) -> Result<(usize, usize), Error> {
    let sem = Arc::new(Semaphore::new(PARALLEL_HASH));
    let mut set = JoinSet::new();
    for mut f in files {
        let sem = sem.clone();
        set.spawn(async move {
            if let Some(h) = f.known_hash.take() {
                return (f, false, Ok(h));
            }
            let _permit = sem.acquire_owned().await;
            let p = f.path.clone();
            let hash = tokio::task::spawn_blocking(move || media::content_hash(&p))
                .await
                .unwrap_or_else(|e| Err(Error::Invalid(format!("hash task failed: {e}"))));
            (f, true, hash)
        });
    }
    let (mut new, mut changed) = (0, 0);
    while let Some(joined) = set.join_next().await {
        let Ok((f, hashed, hash)) = joined else { continue };
        if hashed {
            tracker.advance("hash", 1);
            if tracker.should_emit() {
                on_event(Event::Progress(tracker.snapshot(Some(f.path.clone()))));
            }
        }
        // A file that vanished or can't be read mid-scan is skipped; next scan retries it.
        let Ok(hash) = hash else { continue };
        let tx = db.conn.transaction()?;
        tx.execute("INSERT OR IGNORE INTO videos(content_hash, size) VALUES (?1, ?2)", params![hash, f.size])?;
        let video_id: i64 = tx.query_row("SELECT id FROM videos WHERE content_hash = ?1", [&hash], |r| r.get(0))?;
        tx.execute(
            "INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(folder_id, path) DO UPDATE SET video_id = excluded.video_id,
                 size = excluded.size, mtime = excluded.mtime, last_seen = excluded.last_seen",
            params![video_id, f.folder_id, f.path.to_string_lossy(), f.size, f.mtime, f.seq],
        )?;
        for stage in STAGES {
            tx.execute(
                "INSERT OR IGNORE INTO jobs(video_id, stage, state, updated_at) VALUES (?1, ?2, 'pending', ?3)",
                params![video_id, stage, now()],
            )?;
        }
        tx.commit()?;
        if f.existed {
            changed += 1
        } else {
            new += 1
        }
    }
    Ok((new, changed))
}

/// Videos indexed before a stage existed get its job rows now.
fn ensure_jobs(db: &Db) -> Result<(), Error> {
    for stage in STAGES {
        db.conn.execute(
            "INSERT OR IGNORE INTO jobs(video_id, stage, state, updated_at) SELECT id, ?1, 'pending', ?2 FROM videos",
            params![stage, now()],
        )?;
    }
    Ok(())
}

fn reset_interrupted(db: &Db) -> Result<(), Error> {
    db.conn.execute("UPDATE jobs SET state = 'pending', updated_at = ?1 WHERE state = 'running'", [now()])?;
    Ok(())
}

/// Pending (or retryable) jobs for `stage` in scope, with one existing file path per video.
fn claimable_jobs(db: &Db, stage: &str, opts: &Options) -> Result<Vec<(i64, PathBuf)>, Error> {
    let max_attempts = if opts.retry_failed { i64::MAX } else { MAX_ATTEMPTS };
    let sql = "
        SELECT j.video_id, (SELECT vf.path FROM video_files vf
                             JOIN project_folders pf ON pf.folder_id = vf.folder_id
                            WHERE vf.video_id = j.video_id AND (?3 IS NULL OR pf.project_id = ?3)
                            ORDER BY vf.id LIMIT 1) AS path
          FROM jobs j
         WHERE j.stage = ?1
           AND (j.state = 'pending' OR (j.state = 'failed' AND j.attempts < ?2))
           AND (?1 = 'probe' OR EXISTS (SELECT 1 FROM jobs p WHERE p.video_id = j.video_id
                                          AND p.stage = 'probe' AND p.state = 'done'))
         ORDER BY j.video_id";
    let mut st = db.conn.prepare(sql)?;
    let rows = st.query_map(params![stage, max_attempts, opts.project_id], |r| {
        Ok((r.get::<_, i64>(0)?, r.get::<_, Option<String>>(1)?))
    })?;
    Ok(rows
        .filter_map(Result::ok)
        .filter_map(|(id, path)| path.map(|p| (id, PathBuf::from(p))))
        .collect())
}

fn set_job(db: &Db, video_id: i64, stage: &str, state: &str, error: Option<&str>) -> Result<(), Error> {
    let bump = if state == "failed" { 1 } else { 0 };
    db.conn.execute(
        "UPDATE jobs SET state = ?1, last_error = ?2, attempts = attempts + ?3, updated_at = ?4
          WHERE video_id = ?5 AND stage = ?6",
        params![state, error, bump, now(), video_id, stage],
    )?;
    Ok(())
}

fn store_media(db: &Db, video_id: i64, m: &MediaInfo) -> Result<(), Error> {
    db.conn.execute(
        "UPDATE videos SET duration_s = ?1, width = ?2, height = ?3, rotation = ?4, fps = ?5, avg_fps = ?6,
                vfr = ?7, vcodec = ?8, acodec = ?9, has_audio = ?10, created_time = ?11,
                status = 'probed', error = NULL
          WHERE id = ?12",
        params![
            m.duration_s, m.width, m.height, m.rotation, m.fps, m.avg_fps, m.vfr, m.vcodec, m.acodec,
            m.has_audio, m.created_time, video_id
        ],
    )?;
    Ok(())
}

async fn run_probe_jobs(
    db: &mut Db,
    ffprobe_bin: &Path,
    opts: &Options,
    tracker: &mut Tracker,
    on_event: &mut impl FnMut(Event),
) -> Result<(usize, usize), Error> {
    const STAGE: &str = "probe";
    let jobs = claimable_jobs(db, STAGE, opts)?;
    tracker.set_total(STAGE, jobs.len() as u64);
    tracker.start(STAGE);
    on_event(Event::Progress(tracker.snapshot(None)));
    let sem = Arc::new(Semaphore::new(PARALLEL_PROBE));
    let mut set = JoinSet::new();
    for (video_id, path) in jobs {
        set_job(db, video_id, STAGE, "running", None)?;
        on_event(Event::JobStarted { video_id, stage: STAGE.into(), path: path.clone() });
        let (sem, bin) = (sem.clone(), ffprobe_bin.to_path_buf());
        set.spawn(async move {
            let _permit = sem.acquire_owned().await;
            let result = media::ffprobe(&bin, &path).await;
            (video_id, path, result)
        });
    }
    let (mut done, mut failed) = (0, 0);
    while let Some(joined) = set.join_next().await {
        let Ok((video_id, path, result)) = joined else { continue };
        match result {
            Ok(info) => {
                store_media(db, video_id, &info)?;
                set_job(db, video_id, STAGE, "done", None)?;
                on_event(Event::JobDone { video_id, stage: STAGE.into() });
                done += 1;
            }
            Err(e) => {
                let msg = e.to_string();
                set_job(db, video_id, STAGE, "failed", Some(&msg))?;
                db.conn.execute("UPDATE videos SET status = 'error', error = ?1 WHERE id = ?2", params![msg, video_id])?;
                on_event(Event::JobFailed { video_id, stage: STAGE.into(), error: msg });
                failed += 1;
            }
        }
        tracker.advance(STAGE, 1);
        if tracker.should_emit() {
            on_event(Event::Progress(tracker.snapshot(Some(path))));
        }
    }
    Ok((done, failed))
}

// ---- transcribe ---------------------------------------------------------------------------

fn stt_phase(setup: &SttSetup) -> Option<&'static str> {
    match setup {
        SttSetup::Ready(Engine::Server { .. }) => Some("transcribe_server"),
        SttSetup::Ready(Engine::Local { .. }) | SttSetup::NeedsModel { .. } => Some("transcribe_local"),
        SttSetup::Unavailable(_) => None,
    }
}

/// (seconds of audio known to need transcription, videos in the queue whose duration is unknown)
fn transcribe_work(db: &Db, opts: &Options) -> Result<(f64, i64), Error> {
    let max_attempts = if opts.retry_failed { i64::MAX } else { MAX_ATTEMPTS };
    Ok(db.conn.query_row(
        "SELECT COALESCE(SUM(v.duration_s), 0), COALESCE(SUM(v.duration_s IS NULL), 0)
           FROM jobs j JOIN videos v ON v.id = j.video_id
          WHERE j.stage = 'transcribe' AND COALESCE(v.has_audio, 1) = 1
            AND (j.state = 'pending' OR (j.state = 'failed' AND j.attempts < ?1))
            AND EXISTS (SELECT 1 FROM video_files vf JOIN project_folders pf ON pf.folder_id = vf.folder_id
                         WHERE vf.video_id = j.video_id AND (?2 IS NULL OR pf.project_id = ?2))",
        params![max_attempts, opts.project_id],
        |r| Ok((r.get(0)?, r.get(1)?)),
    )?)
}

fn store_transcript(db: &mut Db, video_id: i64, t: &stt::Transcript) -> Result<(), Error> {
    let tx = db.conn.transaction()?;
    tx.execute("DELETE FROM transcript_segments WHERE video_id = ?1", [video_id])?;
    for s in &t.segments {
        tx.execute(
            "INSERT INTO transcript_segments(video_id, start_s, end_s, text) VALUES (?1, ?2, ?3, ?4)",
            params![video_id, s.start, s.end, s.text],
        )?;
    }
    tx.execute("UPDATE videos SET language = ?1 WHERE id = ?2", params![t.language, video_id])?;
    tx.execute(
        "UPDATE jobs SET state = 'done', last_error = NULL, updated_at = ?1 WHERE video_id = ?2 AND stage = 'transcribe'",
        params![now(), video_id],
    )?;
    tx.commit()?;
    Ok(())
}

async fn run_transcribe_jobs(
    db: &mut Db,
    rt: &Runtime,
    opts: &Options,
    tracker: &mut Tracker,
    on_event: &mut impl FnMut(Event),
) -> Result<(usize, usize), Error> {
    const STAGE: &str = "transcribe";
    // Videos without an audio track have nothing to transcribe.
    db.conn.execute(
        "UPDATE jobs SET state = 'skipped', updated_at = ?1
          WHERE stage = 'transcribe' AND state IN ('pending', 'failed')
            AND video_id IN (SELECT id FROM videos WHERE has_audio = 0)
            AND EXISTS (SELECT 1 FROM jobs p WHERE p.video_id = jobs.video_id AND p.stage = 'probe' AND p.state = 'done')",
        [now()],
    )?;
    let jobs = claimable_jobs(db, STAGE, opts)?;
    let Some(phase) = stt_phase(&rt.stt) else {
        if !jobs.is_empty()
            && let SttSetup::Unavailable(reason) = &rt.stt {
                on_event(Event::StageUnavailable { stage: STAGE.into(), reason: reason.clone() });
            }
        return Ok((0, 0));
    };
    if jobs.is_empty() {
        tracker.set_total(phase, 0);
        return Ok((0, 0));
    }

    let durations: HashMap<i64, f64> = {
        let mut st = db.conn.prepare("SELECT id, COALESCE(duration_s, 0) FROM videos")?;
        st.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?.collect::<Result<_, _>>()?
    };
    let total: f64 = jobs.iter().map(|(id, _)| durations.get(id).copied().unwrap_or(0.0)).sum();

    // Local model missing: download it first (resumable), then run locally.
    let engine = match &rt.stt {
        SttSetup::Ready(e) => e.clone(),
        SttSetup::NeedsModel { spec, dir, asr_bin, language } => {
            on_event(Event::DownloadingModel { file: spec.file_name.clone() });
            tracker.start("download");
            let mut last_done = 0u64;
            let result = crate::models::download(spec, dir, |done, len| {
                if let Some(len) = len {
                    tracker.set_total("download", len);
                }
                tracker.advance("download", done.saturating_sub(last_done));
                last_done = done;
                if tracker.should_emit() {
                    on_event(Event::Progress(tracker.snapshot(None)));
                }
            })
            .await;
            match result {
                Ok(model) => Engine::Local { asr_bin: asr_bin.clone(), model, language: language.clone() },
                Err(e) => {
                    on_event(Event::StageUnavailable { stage: STAGE.into(), reason: e.to_string() });
                    return Ok((0, 0));
                }
            }
        }
        SttSetup::Unavailable(_) => unreachable!("handled above"),
    };
    on_event(Event::StageBackend { stage: STAGE.into(), backend: engine.label() });

    tracker.set_total(phase, total.ceil() as u64);
    tracker.start(phase);
    on_event(Event::Progress(tracker.snapshot(None)));

    let (mut done, mut failed) = (0, 0);
    for (video_id, path) in jobs {
        let duration = durations.get(&video_id).copied().unwrap_or(0.0);
        set_job(db, video_id, STAGE, "running", None)?;
        on_event(Event::JobStarted { video_id, stage: STAGE.into(), path: path.clone() });
        let mut credited = 0.0f64;
        let result = stt::transcribe(&engine, &rt.ffmpeg, &path, duration, |secs| {
            let secs = secs.min(duration);
            if secs > credited {
                tracker.advance(phase, (secs - credited).round() as u64);
                credited = secs.round();
            }
            if tracker.should_emit() {
                on_event(Event::Progress(tracker.snapshot(Some(path.clone()))));
            }
        })
        .await;
        match result {
            Ok(t) => {
                store_transcript(db, video_id, &t)?;
                on_event(Event::JobDone { video_id, stage: STAGE.into() });
                done += 1;
            }
            Err(e) => {
                let msg = e.to_string();
                set_job(db, video_id, STAGE, "failed", Some(&msg))?;
                on_event(Event::JobFailed { video_id, stage: STAGE.into(), error: msg });
                failed += 1;
                // A server that went away mid-run: stop hammering it, retry next run.
                if matches!(engine, Engine::Server { .. }) && e.to_string().contains("GhostPen at") {
                    let remaining = (duration - credited).max(0.0);
                    tracker.advance(phase, remaining as u64);
                    on_event(Event::StageUnavailable { stage: STAGE.into(), reason: e.to_string() });
                    break;
                }
            }
        }
        if duration > credited {
            tracker.advance(phase, (duration - credited).round() as u64);
        }
        on_event(Event::Progress(tracker.snapshot(Some(path))));
    }
    Ok((done, failed))
}

#[derive(Debug, Clone, Serialize)]
pub struct TranscriptSegment {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

pub fn transcript(db: &Db, video_id: i64) -> Result<Vec<TranscriptSegment>, Error> {
    let mut st =
        db.conn.prepare("SELECT start_s, end_s, text FROM transcript_segments WHERE video_id = ?1 ORDER BY start_s")?;
    let rows = st.query_map([video_id], |r| Ok(TranscriptSegment { start: r.get(0)?, end: r.get(1)?, text: r.get(2)? }))?;
    Ok(rows.collect::<Result<_, _>>()?)
}

// ---- status -------------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize)]
pub struct StageCounts {
    pub stage: String,
    pub pending: i64,
    pub running: i64,
    pub done: i64,
    pub failed: i64,
    pub skipped: i64,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Status {
    pub folders: i64,
    pub videos: i64,
    pub total_size: i64,
    pub total_duration_s: f64,
    pub vfr_videos: i64,
    pub stages: Vec<StageCounts>,
}

#[derive(Debug, Clone, Serialize)]
pub struct VideoRow {
    pub id: i64,
    pub path: PathBuf,
    pub copies: i64,
    pub size: i64,
    pub duration_s: Option<f64>,
    pub width: Option<i64>,
    pub height: Option<i64>,
    pub fps: Option<f64>,
    pub vfr: bool,
    pub vcodec: Option<String>,
    pub has_audio: Option<bool>,
    pub status: String,
    pub error: Option<String>,
    pub language: Option<String>,
    /// Transcript segments stored (0 before transcription).
    pub segments: i64,
    /// State of the transcribe job (`pending`, `done`, `failed`, `skipped`, …).
    pub transcribe: Option<String>,
}

/// Videos visible to a project (or all), one row per content with its first path.
const SCOPE: &str = "
    SELECT vf.video_id, MIN(vf.path) AS path, COUNT(DISTINCT vf.path) AS copies
      FROM video_files vf JOIN project_folders pf ON pf.folder_id = vf.folder_id
     WHERE (?1 IS NULL OR pf.project_id = ?1)
     GROUP BY vf.video_id";

pub fn status(db: &Db, project_id: Option<i64>) -> Result<Status, Error> {
    let mut s = Status { folders: db.folders(project_id)?.len() as i64, ..Default::default() };
    (s.videos, s.total_size, s.total_duration_s, s.vfr_videos) = db.conn.query_row(
        &format!(
            "SELECT COUNT(*), COALESCE(SUM(v.size), 0), COALESCE(SUM(v.duration_s), 0), COALESCE(SUM(v.vfr), 0)
               FROM ({SCOPE}) sc JOIN videos v ON v.id = sc.video_id"
        ),
        [project_id],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
    )?;
    for stage in STAGES {
        let mut c = StageCounts { stage: stage.to_string(), ..Default::default() };
        let mut st = db.conn.prepare(&format!(
            "SELECT j.state, COUNT(*) FROM ({SCOPE}) sc JOIN jobs j ON j.video_id = sc.video_id
              WHERE j.stage = ?2 GROUP BY j.state"
        ))?;
        for row in st.query_map(params![project_id, stage], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))? {
            let (state, n) = row?;
            match state.as_str() {
                "pending" => c.pending = n,
                "running" => c.running = n,
                "done" => c.done = n,
                "failed" => c.failed = n,
                "skipped" => c.skipped = n,
                _ => {}
            }
        }
        s.stages.push(c);
    }
    Ok(s)
}

pub fn videos(db: &Db, project_id: Option<i64>) -> Result<Vec<VideoRow>, Error> {
    let mut st = db.conn.prepare(&format!(
        "SELECT v.id, sc.path, sc.copies, v.size, v.duration_s, v.width, v.height, v.fps, COALESCE(v.vfr, 0),
                v.vcodec, v.has_audio, v.status, v.error, v.language,
                (SELECT COUNT(*) FROM transcript_segments t WHERE t.video_id = v.id),
                (SELECT j.state FROM jobs j WHERE j.video_id = v.id AND j.stage = 'transcribe')
           FROM ({SCOPE}) sc JOIN videos v ON v.id = sc.video_id
          ORDER BY sc.path"
    ))?;
    let rows = st.query_map([project_id], |r| {
        Ok(VideoRow {
            id: r.get(0)?,
            path: PathBuf::from(r.get::<_, String>(1)?),
            copies: r.get(2)?,
            size: r.get(3)?,
            duration_s: r.get(4)?,
            width: r.get(5)?,
            height: r.get(6)?,
            fps: r.get(7)?,
            vfr: r.get(8)?,
            vcodec: r.get(9)?,
            has_audio: r.get(10)?,
            status: r.get(11)?,
            error: r.get(12)?,
            language: r.get(13)?,
            segments: r.get(14)?,
            transcribe: r.get(15)?,
        })
    })?;
    Ok(rows.collect::<Result<_, _>>()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projects::NewProject;

    /// A fake ffprobe: prints fixed JSON, or fails for files whose name contains "broken".
    fn fake_ffprobe(dir: &Path) -> PathBuf {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let p = dir.join("ffprobe");
            std::fs::write(
                &p,
                r#"#!/bin/sh
for a in "$@"; do last="$a"; done
case "$last" in *broken*) echo "Invalid data found when processing input" >&2; exit 1;; esac
echo '{"streams":[{"codec_type":"video","codec_name":"h264","width":1920,"height":1080,"r_frame_rate":"25/1","avg_frame_rate":"25/1"},{"codec_type":"audio","codec_name":"aac"}],"format":{"duration":"10.0"}}'
"#,
            )
            .unwrap();
            std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
            p
        }
        #[cfg(not(unix))]
        {
            let _ = dir;
            unimplemented!("fake ffprobe script is unix-only")
        }
    }

    fn rt(ffprobe: &Path) -> Runtime {
        Runtime {
            ffmpeg: "ffmpeg".into(),
            ffprobe: ffprobe.to_path_buf(),
            stt: SttSetup::Unavailable("not configured in this test".into()),
        }
    }

    fn write(path: &Path, bytes: &[u8]) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, bytes).unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn scan_probe_rescan_move_delete() {
        let tmp = tempfile::tempdir().unwrap();
        let media = tmp.path().join("media");
        write(&media.join("a.mp4"), b"video-a");
        write(&media.join("sub/b.MOV"), b"video-b");
        write(&media.join("sub/copy-of-a.mp4"), b"video-a");
        write(&media.join("broken.mkv"), b"junk");
        write(&media.join("notes.txt"), b"not a video");
        write(&media.join(".hidden/c.mp4"), b"video-c");
        let bin = fake_ffprobe(tmp.path());

        let mut db = Db::open_in_memory().unwrap();
        let p = db.create_project(&NewProject::named("P")).unwrap();
        db.add_folder(p.id, &media, true).unwrap();
        let opts = Options { project_id: Some(p.id), ..Default::default() };

        let mut events = Vec::new();
        let s = run(&mut db, &rt(&bin), &opts, |e| events.push(e)).await.unwrap();
        assert_eq!((s.new, s.unchanged, s.removed), (4, 0, 0), "4 files, hidden dir and .txt skipped");
        assert_eq!((s.jobs_done, s.jobs_failed), (2, 1), "a/copy-of-a share content; broken fails");

        let st = status(&db, Some(p.id)).unwrap();
        assert_eq!(st.videos, 3);
        assert_eq!((st.stages[0].done, st.stages[0].failed), (2, 1));
        let rows = videos(&db, Some(p.id)).unwrap();
        let a = rows.iter().find(|v| v.path.ends_with("a.mp4")).unwrap();
        assert_eq!((a.copies, a.duration_s, a.width), (2, Some(10.0), Some(1920)));

        // Unchanged rescan does nothing; the failed job is retried (attempt 2).
        let s = run(&mut db, &rt(&bin), &opts, |_| {}).await.unwrap();
        assert_eq!((s.new, s.unchanged, s.jobs_done, s.jobs_failed), (0, 4, 0, 1));

        // Move a file: same content → no new video, no new jobs.
        std::fs::rename(media.join("sub/b.MOV"), media.join("b-moved.mov")).unwrap();
        let s = run(&mut db, &rt(&bin), &opts, |_| {}).await.unwrap();
        assert_eq!((s.new, s.removed, s.jobs_done), (1, 1, 0));
        assert_eq!(status(&db, Some(p.id)).unwrap().videos, 3);

        // Third failure exhausts retries; --retry-failed forces it.
        run(&mut db, &rt(&bin), &opts, |_| {}).await.unwrap();
        let s = run(&mut db, &rt(&bin), &opts, |_| {}).await.unwrap();
        assert_eq!(s.jobs_failed, 0, "gave up after {MAX_ATTEMPTS} attempts");
        let retry = Options { retry_failed: true, ..opts.clone() };
        assert_eq!(run(&mut db, &rt(&bin), &retry, |_| {}).await.unwrap().jobs_failed, 1);

        // Delete: the video disappears from the project.
        std::fs::remove_file(media.join("b-moved.mov")).unwrap();
        run(&mut db, &rt(&bin), &opts, |_| {}).await.unwrap();
        assert_eq!(status(&db, Some(p.id)).unwrap().videos, 2);
        assert!(events.iter().any(|e| matches!(e, Event::JobFailed { .. })));
        let last = events.iter().rev().find_map(|e| match e {
            Event::Progress(p) => Some(p.clone()),
            _ => None,
        });
        let last = last.expect("progress events emitted");
        assert_eq!(last.fraction, 1.0);
        assert_eq!(last.phase_done, last.phase_total);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn projects_share_indexed_videos_and_missing_folders_keep_index() {
        let tmp = tempfile::tempdir().unwrap();
        let media = tmp.path().join("media");
        write(&media.join("a.mp4"), b"video-a");
        let bin = fake_ffprobe(tmp.path());
        let mut db = Db::open_in_memory().unwrap();
        let p1 = db.create_project(&NewProject::named("One")).unwrap();
        let p2 = db.create_project(&NewProject::named("Two")).unwrap();
        db.add_folder(p1.id, &media, true).unwrap();
        db.add_folder(p2.id, &media, true).unwrap();

        let s = run(&mut db, &rt(&bin), &Options { project_id: Some(p1.id), ..Default::default() }, |_| {}).await.unwrap();
        assert_eq!(s.jobs_done, 1);
        // Project Two sees the already-probed video without any work.
        let s = run(&mut db, &rt(&bin), &Options { project_id: Some(p2.id), ..Default::default() }, |_| {}).await.unwrap();
        assert_eq!((s.new, s.jobs_done), (0, 0));
        assert_eq!(status(&db, Some(p2.id)).unwrap().stages[0].done, 1);

        // Unmounted drive: folder missing → files are NOT dropped.
        std::fs::rename(&media, tmp.path().join("unplugged")).unwrap();
        let mut missing = false;
        run(&mut db, &rt(&bin), &Options::default(), |e| missing |= matches!(e, Event::FolderMissing { .. })).await.unwrap();
        assert!(missing);
        assert_eq!(status(&db, None).unwrap().videos, 1);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn overlapping_folders_of_different_projects_both_see_the_file() {
        let tmp = tempfile::tempdir().unwrap();
        let lib = tmp.path().join("lib");
        write(&lib.join("screen/demo.mp4"), b"video-demo");
        write(&lib.join("other.mp4"), b"video-other");
        let bin = fake_ffprobe(tmp.path());
        let mut db = Db::open_in_memory().unwrap();
        let teaser = db.create_project(&NewProject::named("Teaser")).unwrap();
        let docs = db.create_project(&NewProject::named("Docs")).unwrap();
        db.add_folder(teaser.id, &lib, true).unwrap();
        db.add_folder(docs.id, &lib.join("screen"), true).unwrap();

        for _ in 0..3 {
            run(&mut db, &rt(&bin), &Options::default(), |_| {}).await.unwrap();
        }
        assert_eq!(status(&db, Some(teaser.id)).unwrap().videos, 2);
        assert_eq!(status(&db, Some(docs.id)).unwrap().videos, 1);
        let s = run(&mut db, &rt(&bin), &Options::default(), |_| {}).await.unwrap();
        assert_eq!((s.new, s.changed, s.removed, s.unchanged), (0, 0, 0, 3), "stable across rescans");
        let rows = videos(&db, None).unwrap();
        assert!(rows.iter().all(|v| v.copies == 1), "one file seen through two folders is not a copy");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn fresh_files_wait_to_settle() {
        let tmp = tempfile::tempdir().unwrap();
        let media = tmp.path().join("media");
        write(&media.join("copying.mp4"), b"partial");
        let bin = fake_ffprobe(tmp.path());
        let mut db = Db::open_in_memory().unwrap();
        let p = db.create_project(&NewProject::named("P")).unwrap();
        db.add_folder(p.id, &media, true).unwrap();
        let watch = Options { project_id: Some(p.id), settle_secs: 3600, ..Default::default() };
        let s = run(&mut db, &rt(&bin), &watch, |_| {}).await.unwrap();
        assert_eq!((s.new, s.unsettled), (0, 1));
        let s = run(&mut db, &rt(&bin), &Options { project_id: Some(p.id), ..Default::default() }, |_| {}).await.unwrap();
        assert_eq!((s.new, s.unsettled), (1, 0));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn transcribe_stage_end_to_end() {
        let have = |b: &str| std::process::Command::new(b).arg("-version").output().map(|o| o.status.success()).unwrap_or(false);
        if !have("ffmpeg") || !have("ffprobe") {
            eprintln!("skipping: ffmpeg/ffprobe not installed");
            return;
        }
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempfile::tempdir().unwrap();
        let media = tmp.path().join("media");
        std::fs::create_dir_all(&media).unwrap();
        let make = |args: &[&str], out: &str| {
            let ok = std::process::Command::new("ffmpeg")
                .args(["-v", "error"])
                .args(args)
                .arg(media.join(out))
                .status()
                .unwrap()
                .success();
            assert!(ok, "ffmpeg {out}");
        };
        make(&["-f", "lavfi", "-i", "testsrc=size=160x120:rate=10", "-f", "lavfi", "-i", "sine=d=3", "-t", "3", "-c:v", "mpeg4", "-c:a", "aac", "-shortest"], "talk.mp4");
        make(&["-f", "lavfi", "-i", "testsrc=size=160x120:rate=10", "-t", "2", "-c:v", "mpeg4"], "silent.mp4");

        let asr = tmp.path().join("asr");
        std::fs::write(
            &asr,
            "#!/bin/sh\ncat > /dev/null\necho '{\"type\":\"loaded\"}'\necho '{\"type\":\"progress\",\"percent\":100}'\necho '{\"type\":\"segment\",\"start\":0.5,\"end\":2.5,\"text\":\"hello world\",\"no_speech\":0.1}'\necho '{\"type\":\"done\",\"language\":\"en\",\"audio_s\":3}'\n",
        )
        .unwrap();
        std::fs::set_permissions(&asr, std::fs::Permissions::from_mode(0o755)).unwrap();

        let mut db = Db::open_in_memory().unwrap();
        let p = db.create_project(&NewProject::named("P")).unwrap();
        db.add_folder(p.id, &media, true).unwrap();
        let opts = Options { project_id: Some(p.id), ..Default::default() };

        // 1. Transcription unavailable: probe runs, transcribe stays pending (not failed).
        let unavailable = Runtime { ffmpeg: "ffmpeg".into(), ffprobe: "ffprobe".into(), stt: SttSetup::Unavailable("GhostPen down".into()) };
        let mut events = Vec::new();
        let s = run(&mut db, &unavailable, &opts, |e| events.push(e)).await.unwrap();
        assert_eq!((s.jobs_done, s.jobs_failed), (2, 0));
        assert!(events.iter().any(|e| matches!(e, Event::StageUnavailable { .. })));
        let st = status(&db, Some(p.id)).unwrap();
        let tr = st.stages.iter().find(|c| c.stage == "transcribe").unwrap();
        assert_eq!((tr.pending, tr.skipped, tr.failed), (1, 1, 0), "silent video skipped, talk waits");

        // 2. Local engine available: the waiting video gets its transcript.
        let model = tmp.path().join("ggml-test.bin");
        std::fs::write(&model, b"m").unwrap();
        let local = Runtime {
            stt: SttSetup::Ready(Engine::Local { asr_bin: asr, model, language: "auto".into() }),
            ..unavailable
        };
        let mut events = Vec::new();
        let s = run(&mut db, &local, &opts, |e| events.push(e)).await.unwrap();
        assert_eq!((s.jobs_done, s.jobs_failed), (1, 0));
        let rows = videos(&db, Some(p.id)).unwrap();
        let talk = rows.iter().find(|v| v.path.ends_with("talk.mp4")).unwrap();
        assert_eq!((talk.segments, talk.language.as_deref(), talk.transcribe.as_deref()), (1, Some("en"), Some("done")));
        let silent = rows.iter().find(|v| v.path.ends_with("silent.mp4")).unwrap();
        assert_eq!(silent.transcribe.as_deref(), Some("skipped"));
        assert_eq!(transcript(&db, talk.id).unwrap()[0].text, "hello world");
        let last = events.iter().rev().find_map(|e| if let Event::Progress(p) = e { Some(p.clone()) } else { None }).unwrap();
        assert_eq!(last.fraction, 1.0);
        assert!(events.iter().any(|e| matches!(e, Event::StageBackend { .. })));

        // 3. Nothing left: another run does no work.
        let s = run(&mut db, &local, &opts, |_| {}).await.unwrap();
        assert_eq!(s.jobs_done, 0);
    }

    #[test]
    fn lock_is_exclusive() {
        let tmp = tempfile::tempdir().unwrap();
        let first = IndexLock::acquire(tmp.path()).unwrap();
        assert!(matches!(IndexLock::acquire(tmp.path()), Err(Error::Busy(_))));
        drop(first);
        assert!(IndexLock::acquire(tmp.path()).is_ok());
    }
}
