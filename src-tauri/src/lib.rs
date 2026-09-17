//! GhostReel desktop app. UI logic lives in the React frontend; everything else is
//! `ghostreel-core`, shared with the CLI.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use ghostreel_core::db::Db;
use ghostreel_core::doctor::{self, Report};
use ghostreel_core::config::Config;
use ghostreel_core::index::{self, IndexLock, Status, TranscriptSegment, VideoRow};
use ghostreel_core::runtime;
use ghostreel_core::paths::Paths;
use ghostreel_core::projects::{Folder, NewProject, Project};
use serde::Serialize;
use tauri::{AppHandle, Emitter, State};

type CmdResult<T> = Result<T, String>;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

fn paths() -> CmdResult<Paths> {
    Paths::resolve().map_err(err)
}

fn open_db() -> CmdResult<Db> {
    Db::open(&paths()?.db_file()).map_err(err)
}

/// Whether this app instance is currently indexing (the CLI may hold the lock too).
#[derive(Default)]
struct Indexing(AtomicBool);

/// Doctor report + the blockers list the UI shows at the top.
#[derive(Serialize)]
struct DoctorView {
    report: Report,
    blockers: Vec<String>,
}

#[tauri::command]
async fn doctor() -> CmdResult<DoctorView> {
    let report = doctor::run(&paths()?).await;
    let blockers = report.blockers();
    Ok(DoctorView { report, blockers })
}

#[derive(Serialize)]
struct ProjectSummary {
    project: Project,
    status: Status,
}

#[tauri::command]
fn list_projects() -> CmdResult<Vec<ProjectSummary>> {
    let db = open_db()?;
    db.projects()
        .map_err(err)?
        .into_iter()
        .map(|project| Ok(ProjectSummary { status: index::status(&db, Some(project.id)).map_err(err)?, project }))
        .collect()
}

#[tauri::command]
fn create_project(name: String, fps_num: i64, fps_den: i64, width: i64, height: i64) -> CmdResult<Project> {
    open_db()?
        .create_project(&NewProject { name, description: String::new(), fps_num, fps_den, width, height })
        .map_err(err)
}

#[tauri::command]
fn remove_project(project_id: i64) -> CmdResult<()> {
    open_db()?.remove_project(project_id).map_err(err)
}

#[derive(Serialize)]
struct ProjectView {
    project: Project,
    folders: Vec<FolderView>,
    status: Status,
    videos: Vec<VideoRow>,
}

#[derive(Serialize)]
struct FolderView {
    #[serde(flatten)]
    folder: Folder,
    available: bool,
}

#[tauri::command]
fn project_view(project_id: i64) -> CmdResult<ProjectView> {
    let db = open_db()?;
    let folders = db
        .folders(Some(project_id))
        .map_err(err)?
        .into_iter()
        .map(|f| FolderView { available: f.path.is_dir(), folder: f })
        .collect();
    Ok(ProjectView {
        project: db.project(project_id).map_err(err)?,
        folders,
        status: index::status(&db, Some(project_id)).map_err(err)?,
        videos: index::videos(&db, Some(project_id)).map_err(err)?,
    })
}

#[tauri::command]
fn add_folder(project_id: i64, path: PathBuf, recursive: bool) -> CmdResult<Folder> {
    open_db()?.add_folder(project_id, &path, recursive).map_err(err)
}

#[tauri::command]
fn remove_folder(project_id: i64, path: PathBuf) -> CmdResult<()> {
    open_db()?.remove_folder(project_id, &path).map_err(err)
}

#[derive(Clone, Serialize)]
struct IndexFinished {
    project_id: i64,
    summary: Option<index::Summary>,
    error: Option<String>,
}

/// Start indexing a project in the background. Progress arrives as `index-event`
/// (core `index::Event`), completion as `index-finished`.
#[tauri::command]
fn start_index(app: AppHandle, state: State<'_, Indexing>, project_id: i64) -> CmdResult<()> {
    if state.0.swap(true, Ordering::SeqCst) {
        return Err("indexing is already running".into());
    }
    let started = (|| -> CmdResult<(Db, Paths, Config, IndexLock)> {
        let p = paths()?;
        let config = Config::load(&p.config_file).map_err(err)?;
        let lock = IndexLock::acquire(&p.data_dir).map_err(err)?;
        Ok((Db::open(&p.db_file()).map_err(err)?, p, config, lock))
    })();
    let (mut db, p, config, lock) = match started {
        Ok(v) => v,
        Err(e) => {
            state.0.store(false, Ordering::SeqCst);
            return Err(e);
        }
    };
    tauri::async_runtime::spawn(async move {
        let opts = index::Options { project_id: Some(project_id), ..Default::default() };
        let events = app.clone();
        let result = match runtime::resolve(&p, &config).await {
            Ok(rt) => {
                index::run(&mut db, &rt, &opts, move |e| {
                    let _ = events.emit("index-event", e);
                })
                .await
            }
            Err(e) => Err(e),
        };
        drop(lock);
        let finished = match result {
            Ok(summary) => IndexFinished { project_id, summary: Some(summary), error: None },
            Err(e) => IndexFinished { project_id, summary: None, error: Some(e.to_string()) },
        };
        let _ = app.emit("index-finished", finished);
        use tauri::Manager;
        app.state::<Indexing>().0.store(false, Ordering::SeqCst);
    });
    Ok(())
}

#[tauri::command]
fn video_transcript(video_id: i64) -> CmdResult<Vec<TranscriptSegment>> {
    index::transcript(&open_db()?, video_id).map_err(err)
}

#[tauri::command]
fn is_indexing(state: State<'_, Indexing>) -> bool {
    state.0.load(Ordering::SeqCst)
}

/// WebKitGTK's DMABUF renderer dies with "Error 71 (Protocol error) dispatching to Wayland
/// display" on wlroots compositors (Hyprland, Sway) — same workaround as GhostPen. Only on
/// Wayland, and only if the user hasn't chosen a value themselves.
#[cfg(target_os = "linux")]
fn apply_wayland_webkit_workaround() {
    let on_wayland = std::env::var_os("WAYLAND_DISPLAY").is_some();
    if on_wayland && std::env::var_os("WEBKIT_DISABLE_DMABUF_RENDERER").is_none() {
        // SAFETY: called first thing in `run`, before any other thread exists.
        unsafe { std::env::set_var("WEBKIT_DISABLE_DMABUF_RENDERER", "1") };
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    #[cfg(target_os = "linux")]
    apply_wayland_webkit_workaround();

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(Indexing::default())
        .invoke_handler(tauri::generate_handler![
            doctor,
            list_projects,
            create_project,
            remove_project,
            project_view,
            add_folder,
            remove_folder,
            start_index,
            is_indexing,
            video_transcript,
        ])
        .run(tauri::generate_context!())
        .expect("error while running GhostReel");
}
