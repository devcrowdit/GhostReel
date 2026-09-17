//! The app's background work queue: everything slow (indexing now; transcripts, previews and exports
//! later) runs one task at a time, in order, with progress the UI can show and cancel.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use ghostreel_core::config::Config;
use ghostreel_core::db::Db;
use ghostreel_core::index::{self, IndexLock};
use ghostreel_core::paths::Paths;
use ghostreel_core::progress::Progress;
use ghostreel_core::runtime;
use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager};
use tokio::sync::{Mutex, Notify};

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TaskKind {
    Index { project_id: i64 },
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TaskState {
    Queued,
    Running,
    Done,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize)]
pub struct Task {
    pub id: u64,
    pub kind: TaskKind,
    pub label: String,
    pub state: TaskState,
    pub progress: Option<Progress>,
    /// Latest human-readable status line ("Transcription: GhostPen @ …").
    pub note: Option<String>,
    pub summary: Option<index::Summary>,
    pub error: Option<String>,
    pub created_at: i64,
    pub finished_at: Option<i64>,
    #[serde(skip)]
    cancel: Arc<AtomicBool>,
}

#[derive(Default)]
pub struct Queue {
    tasks: Mutex<VecDeque<Task>>,
    next_id: std::sync::atomic::AtomicU64,
    wake: Notify,
}

/// Finished tasks kept for the Activity list.
const HISTORY: usize = 30;

impl Queue {
    pub async fn snapshot(&self) -> Vec<Task> {
        self.tasks.lock().await.iter().cloned().collect()
    }

    /// Add a task unless an identical one is already waiting. Returns the task id.
    pub async fn enqueue(&self, app: &AppHandle, kind: TaskKind, label: String) -> u64 {
        let mut tasks = self.tasks.lock().await;
        if let Some(t) = tasks.iter().find(|t| t.kind == kind && t.state == TaskState::Queued) {
            return t.id;
        }
        let id = self.next_id.fetch_add(1, Ordering::SeqCst) + 1;
        tasks.push_back(Task {
            id,
            kind,
            label,
            state: TaskState::Queued,
            progress: None,
            note: None,
            summary: None,
            error: None,
            created_at: ghostreel_core::projects::now(),
            finished_at: None,
            cancel: Arc::new(AtomicBool::new(false)),
        });
        drop(tasks);
        self.wake.notify_one();
        self.emit(app).await;
        id
    }

    pub async fn cancel(&self, app: &AppHandle, id: u64) -> bool {
        let mut tasks = self.tasks.lock().await;
        let Some(t) = tasks.iter_mut().find(|t| t.id == id) else { return false };
        match t.state {
            TaskState::Queued => {
                t.state = TaskState::Cancelled;
                t.finished_at = Some(ghostreel_core::projects::now());
            }
            TaskState::Running => {
                t.cancel.store(true, Ordering::SeqCst);
                t.note = Some("Stopping after the current item…".into());
            }
            _ => return false,
        }
        drop(tasks);
        self.emit(app).await;
        true
    }

    pub async fn clear_finished(&self, app: &AppHandle) {
        self.tasks.lock().await.retain(|t| matches!(t.state, TaskState::Queued | TaskState::Running));
        self.emit(app).await;
    }

    async fn emit(&self, app: &AppHandle) {
        let _ = app.emit("queue", self.snapshot().await);
    }

    async fn update(&self, app: &AppHandle, id: u64, f: impl FnOnce(&mut Task)) {
        {
            let mut tasks = self.tasks.lock().await;
            if let Some(t) = tasks.iter_mut().find(|t| t.id == id) {
                f(t);
            }
            // Trim history.
            let finished = tasks.iter().filter(|t| !matches!(t.state, TaskState::Queued | TaskState::Running)).count();
            if finished > HISTORY {
                let mut drop_n = finished - HISTORY;
                tasks.retain(|t| {
                    if drop_n > 0 && !matches!(t.state, TaskState::Queued | TaskState::Running) {
                        drop_n -= 1;
                        false
                    } else {
                        true
                    }
                });
            }
        }
        self.emit(app).await;
    }

    async fn next_queued(&self) -> Option<(u64, TaskKind, Arc<AtomicBool>)> {
        let mut tasks = self.tasks.lock().await;
        let t = tasks.iter_mut().find(|t| t.state == TaskState::Queued)?;
        t.state = TaskState::Running;
        Some((t.id, t.kind.clone(), t.cancel.clone()))
    }
}

/// The single worker: runs queued tasks forever, in order.
pub async fn worker(app: AppHandle) {
    loop {
        let queue = app.state::<Queue>();
        let Some((id, kind, cancel)) = queue.next_queued().await else {
            queue.wake.notified().await;
            continue;
        };
        queue.emit(&app).await;
        let result = match kind {
            TaskKind::Index { project_id } => run_index(&app, id, project_id, cancel.clone()).await,
        };
        let queue = app.state::<Queue>();
        queue
            .update(&app, id, |t| {
                t.finished_at = Some(ghostreel_core::projects::now());
                match result {
                    Ok(summary) => {
                        t.state = if summary.cancelled { TaskState::Cancelled } else { TaskState::Done };
                        t.summary = Some(summary);
                    }
                    Err(e) => {
                        t.state = TaskState::Failed;
                        t.error = Some(e);
                    }
                }
                if let Some(p) = &mut t.progress
                    && t.state == TaskState::Done
                {
                    p.fraction = 1.0;
                    p.eta_secs = Some(0.0);
                }
            })
            .await;
        let _ = app.emit("task-finished", id);
    }
}

async fn run_index(
    app: &AppHandle,
    task_id: u64,
    project_id: i64,
    cancel: Arc<AtomicBool>,
) -> Result<index::Summary, String> {
    let p = Paths::resolve().map_err(|e| e.to_string())?;
    let config = Config::load(&p.config_file).map_err(|e| e.to_string())?;
    // The CLI may be indexing: wait for it instead of failing.
    let lock = loop {
        match IndexLock::acquire(&p.data_dir) {
            Ok(l) => break l,
            Err(ghostreel_core::Error::Busy(_)) => {
                app.state::<Queue>()
                    .update(app, task_id, |t| {
                        t.note = Some("Waiting for another GhostReel process to finish indexing…".into())
                    })
                    .await;
                if cancel.load(Ordering::SeqCst) {
                    return Ok(index::Summary { cancelled: true, ..Default::default() });
                }
                tokio::time::sleep(std::time::Duration::from_secs(3)).await;
            }
            Err(e) => return Err(e.to_string()),
        }
    };
    let mut db = Db::open(&p.db_file()).map_err(|e| e.to_string())?;
    let rt = runtime::resolve(&p, &config).await.map_err(|e| e.to_string())?;
    let opts = index::Options { project_id: Some(project_id), cancel: Some(cancel), ..Default::default() };

    // Events arrive synchronously from the indexer; forward them through a channel so queue updates
    // (async) don't block it.
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<index::Event>();
    let forward = {
        let app = app.clone();
        tauri::async_runtime::spawn(async move {
            while let Some(e) = rx.recv().await {
                let _ = app.emit("index-event", (task_id, &e));
                let queue = app.state::<Queue>();
                match e {
                    index::Event::Progress(p) => queue.update(&app, task_id, |t| t.progress = Some(p)).await,
                    index::Event::StageBackend { stage, backend } => {
                        queue
                            .update(&app, task_id, |t| t.note = Some(format!("{}: {backend}", stage_name(&stage))))
                            .await
                    }
                    index::Event::StageUnavailable { stage, reason } => {
                        queue
                            .update(&app, task_id, |t| {
                                t.note = Some(format!("{} postponed: {reason}", stage_name(&stage)))
                            })
                            .await
                    }
                    index::Event::DownloadingModel { file } => {
                        queue.update(&app, task_id, |t| t.note = Some(format!("Downloading {file} (once)…"))).await
                    }
                    _ => {}
                }
            }
        })
    };
    let result = index::run(&mut db, &rt, &opts, |e| {
        let _ = tx.send(e);
    })
    .await;
    drop(tx);
    let _ = forward.await;
    drop(lock);
    result.map_err(|e| e.to_string())
}

fn stage_name(stage: &str) -> &str {
    match stage {
        "transcribe" => "Transcription",
        "describe" => "Frame descriptions",
        "embed" => "Search index",
        "frames" => "Keyframes",
        other => other,
    }
}
