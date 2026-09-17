//! The app's background work queue: everything slow (indexing now; transcripts, previews and exports
//! later) runs one task at a time, in order, with progress the UI can show and cancel.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use ghostreel_core::config::Config;
use ghostreel_core::db::Db;
use ghostreel_core::index::{self, IndexLock};
use ghostreel_core::paths::Paths;
use ghostreel_core::progress::Progress;
use ghostreel_core::runtime;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager};
use tokio::sync::{Mutex, Notify, oneshot};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatTurnView {
    pub session_id: i64,
    pub reply: String,
    pub script_id: Option<i64>,
    pub issues: Vec<ghostreel_core::script::Issue>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TaskKind {
    Index { project_id: i64 },
    RenderPreview { script_id: i64, burn_titles: bool, burn_narration: bool },
    Export { script_id: i64, format: String, path: String },
    Chat { project_id: i64, session_id: i64 },
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
    pub output: Option<String>,
    pub error: Option<String>,
    pub created_at: i64,
    pub finished_at: Option<i64>,
    #[serde(skip)]
    cancel: Arc<AtomicBool>,
}

type ChatSender = oneshot::Sender<Result<ChatTurnView, String>>;
type ChatRequest = (String, ChatSender);

#[derive(Default)]
pub struct Queue {
    tasks: Mutex<VecDeque<Task>>,
    next_id: std::sync::atomic::AtomicU64,
    wake: Notify,
    chat_requests: Mutex<HashMap<u64, ChatRequest>>,
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
        if !matches!(kind, TaskKind::Chat { .. })
            && let Some(t) = tasks.iter().find(|t| t.kind == kind && t.state == TaskState::Queued)
        {
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
            output: None,
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

    pub async fn enqueue_chat(
        &self,
        app: &AppHandle,
        project_id: i64,
        session_id: i64,
        message: String,
        label: String,
        tx: oneshot::Sender<Result<ChatTurnView, String>>,
    ) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst) + 1;
        self.chat_requests.lock().await.insert(id, (message, tx));
        let mut tasks = self.tasks.lock().await;
        tasks.push_back(Task {
            id,
            kind: TaskKind::Chat { project_id, session_id },
            label,
            state: TaskState::Queued,
            progress: None,
            note: None,
            summary: None,
            output: None,
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
                if let Some((_, tx)) = self.chat_requests.lock().await.remove(&id) {
                    let _ = tx.send(Err("task cancelled".into()));
                }
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

enum TaskOutcome {
    Index(index::Summary),
    Preview { path: String },
    Export { path: String },
    Chat,
    Cancelled,
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
            TaskKind::Index { project_id } => {
                run_index(&app, id, project_id, cancel.clone()).await.map(TaskOutcome::Index)
            }
            TaskKind::RenderPreview { script_id, burn_titles, burn_narration } => {
                run_preview(&app, id, script_id, burn_titles, burn_narration, cancel.clone()).await
            }
            TaskKind::Export { script_id, format, path } => {
                run_export(&app, id, script_id, &format, &path).await.map(|p| TaskOutcome::Export { path: p })
            }
            TaskKind::Chat { project_id, session_id } => {
                let req = queue.chat_requests.lock().await.remove(&id);
                match req {
                    Some((message, tx)) => {
                        let res = run_chat(&app, id, project_id, session_id, message, cancel.clone()).await;
                        let outcome = match &res {
                            Ok(_) => Ok(TaskOutcome::Chat),
                            Err(e) => Err(e.clone()),
                        };
                        let _ = tx.send(res);
                        outcome
                    }
                    None => Err("chat request missing".into()),
                }
            }
        };
        let queue = app.state::<Queue>();
        queue
            .update(&app, id, |t| {
                t.finished_at = Some(ghostreel_core::projects::now());
                match result {
                    Ok(TaskOutcome::Index(summary)) => {
                        t.state = if summary.cancelled { TaskState::Cancelled } else { TaskState::Done };
                        t.summary = Some(summary);
                    }
                    Ok(TaskOutcome::Preview { path }) => {
                        t.state = TaskState::Done;
                        t.output = Some(path);
                    }
                    Ok(TaskOutcome::Export { path }) => {
                        t.state = TaskState::Done;
                        t.output = Some(path);
                    }
                    Ok(TaskOutcome::Chat) => {
                        t.state = TaskState::Done;
                    }
                    Ok(TaskOutcome::Cancelled) => {
                        t.state = TaskState::Cancelled;
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

async fn run_preview(
    app: &AppHandle,
    task_id: u64,
    script_id: i64,
    burn_titles: bool,
    burn_narration: bool,
    cancel: Arc<AtomicBool>,
) -> Result<TaskOutcome, String> {
    let p = Paths::resolve().map_err(|e| e.to_string())?;
    let db = Db::open(&p.db_file()).map_err(|e| e.to_string())?;
    let ffmpeg = ghostreel_core::doctor::locate("ffmpeg").ok_or_else(|| "ffmpeg not found".to_string())?;
    let opts = ghostreel_core::preview::PreviewOptions { burn_titles, burn_narration, out: None, cancel: Some(cancel) };

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<(f64, String)>();
    let app_clone = app.clone();
    let forward = tauri::async_runtime::spawn(async move {
        let t0 = std::time::Instant::now();
        while let Some((frac, note)) = rx.recv().await {
            let queue = app_clone.state::<Queue>();
            queue
                .update(&app_clone, task_id, |t| {
                    t.note = Some(note);
                    t.progress = Some(Progress {
                        phase: "preview".to_string(),
                        phase_done: (frac * 100.0).round() as u64,
                        phase_total: 100,
                        fraction: frac,
                        eta_secs: None,
                        elapsed_secs: t0.elapsed().as_secs_f64(),
                        current: None,
                    });
                })
                .await;
        }
    });

    let res = tokio::task::spawn_blocking(move || {
        ghostreel_core::preview::render_preview(&db, &p.data_dir, &ffmpeg, script_id, &opts, |frac, msg| {
            let _ = tx.send((frac, msg.to_string()));
        })
    })
    .await
    .map_err(|e| e.to_string())?;

    // The sender was moved into the blocking closure and is gone now: drain the remaining updates
    // before the worker writes the final state, so a late progress event can't overwrite it.
    let _ = forward.await;

    match res {
        Ok(preview_res) => Ok(TaskOutcome::Preview { path: preview_res.path.to_string_lossy().to_string() }),
        Err(ghostreel_core::Error::Preview(s)) if s == "cancelled" => Ok(TaskOutcome::Cancelled),
        Err(e) => Err(e.to_string()),
    }
}

async fn run_export(
    app: &AppHandle,
    task_id: u64,
    script_id: i64,
    format_str: &str,
    out_path_str: &str,
) -> Result<String, String> {
    use std::path::PathBuf;
    use std::str::FromStr;

    let p = Paths::resolve().map_err(|e| e.to_string())?;
    let db = Db::open(&p.db_file()).map_err(|e| e.to_string())?;
    let format = ghostreel_core::export::ExportFormat::from_str(format_str).map_err(|e| e.to_string())?;
    let out_path = PathBuf::from(out_path_str);

    app.state::<Queue>()
        .update(app, task_id, |t| {
            t.note = Some(format!("Exporting to {format}…"));
        })
        .await;

    let res = tokio::task::spawn_blocking(move || {
        ghostreel_core::export::export_script(&db, &p.data_dir, script_id, format, &out_path)
    })
    .await
    .map_err(|e| e.to_string())?
    .map_err(|e| e.to_string())?;

    Ok(res.path.to_string_lossy().to_string())
}

async fn run_chat(
    app: &AppHandle,
    task_id: u64,
    project_id: i64,
    session_id: i64,
    message: String,
    cancel: Arc<AtomicBool>,
) -> Result<ChatTurnView, String> {
    let p = Paths::resolve().map_err(|e| e.to_string())?;
    let config = Config::load(&p.config_file).map_err(|e| e.to_string())?;
    let rt = runtime::resolve(&p, &config).await.map_err(|e| e.to_string())?;
    let backend = ghostreel_core::chat::ChatBackend::from_runtime(&rt).await.map_err(|e| e.to_string())?;
    let setup = runtime::resolve_embed(&p, &config).await;
    let embedder = runtime::start_embedder(&setup, |_, _| {}).await.ok();
    let db = Db::open(&p.db_file()).map_err(|e| e.to_string())?;

    let mut ctx = ghostreel_core::chat::ChatContext { db, data_dir: p.data_dir.clone(), backend, embedder };

    let app_handle = app.clone();
    let mut on_event = move |event: ghostreel_core::chat::ChatEvent| {
        let note = match &event {
            ghostreel_core::chat::ChatEvent::ToolStarted { tool, args } => {
                let s = serde_json::to_string(args).unwrap_or_default();
                // Char-based: args often hold non-ASCII queries ("configuración").
                let snippet =
                    if s.chars().count() > 40 { format!("{}…", s.chars().take(40).collect::<String>()) } else { s };
                format!("{tool}: {snippet}")
            }
            ghostreel_core::chat::ChatEvent::ToolFinished { tool, summary } => {
                format!("{tool}: {summary}")
            }
            ghostreel_core::chat::ChatEvent::Drafting => "Drafting script".to_string(),
            ghostreel_core::chat::ChatEvent::Validating => "Validating script".to_string(),
        };

        let _ = app_handle.emit(
            "chat-progress",
            serde_json::json!({
                "session_id": session_id,
                "event": event,
            }),
        );

        let app_clone = app_handle.clone();
        tauri::async_runtime::spawn(async move {
            let queue = app_clone.state::<Queue>();
            queue.update(&app_clone, task_id, |t| t.note = Some(note)).await;
        });
    };

    if cancel.load(Ordering::SeqCst) {
        return Err("task cancelled".into());
    }

    let turn_res = ghostreel_core::chat::run_turn(&mut ctx, project_id, Some(session_id), &message, &mut on_event)
        .await
        .map_err(|e| e.to_string())?;

    Ok(ChatTurnView {
        session_id: turn_res.session_id,
        reply: turn_res.reply,
        script_id: turn_res.script_id,
        issues: turn_res.issues,
    })
}
