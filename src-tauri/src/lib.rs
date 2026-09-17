//! GhostReel desktop app. UI logic lives in the React frontend; everything else is
//! `ghostreel-core`, shared with the CLI.

mod media;
mod queue;

use std::path::PathBuf;

use ghostreel_core::config::Config;
use ghostreel_core::db::Db;
use ghostreel_core::doctor::{self, Report};
use ghostreel_core::index::{self, FrameRow, Status, TranscriptSegment, VideoRow};
use ghostreel_core::paths::Paths;
use ghostreel_core::projects::{Folder, NewProject, Project};
use ghostreel_core::runtime;
use serde::Serialize;
use tauri::{AppHandle, State};

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
fn add_folder(app: AppHandle, project_id: i64, path: PathBuf, recursive: bool) -> CmdResult<Folder> {
    let folder = open_db()?.add_folder(project_id, &path, recursive).map_err(err)?;
    allow_media_dir(&app, &folder.path);
    Ok(folder)
}

/// Let the webview load videos from a watched folder (player).
fn allow_media_dir(app: &AppHandle, dir: &std::path::Path) {
    use tauri::Manager;
    let _ = app.asset_protocol_scope().allow_directory(dir, true);
}

/// Cached embedder for search (a local helper takes ~1 s to start; reuse it across queries).
#[derive(Default)]
struct SearchState(tokio::sync::Mutex<Option<ghostreel_core::embed::Embedder>>);

#[tauri::command]
async fn search(
    state: State<'_, SearchState>,
    project_id: i64,
    query: String,
    limit: Option<usize>,
) -> CmdResult<SearchView> {
    let p = paths()?;
    let mut cached = state.0.lock().await;
    let mut note = None;
    if cached.is_none() {
        let config = Config::load(&p.config_file).map_err(err)?;
        let setup = runtime::resolve_embed(&p, &config).await;
        match runtime::start_embedder(&setup, |_, _| {}).await {
            Ok(e) => *cached = Some(e),
            Err(why) => note = Some(format!("Keyword search only ({why})")),
        }
    }
    let opts =
        ghostreel_core::search::SearchOptions { project_id: Some(project_id), limit: limit.unwrap_or(30), kinds: None };
    let vector = match cached.as_mut() {
        Some(e) => match ghostreel_core::search::query_vector(e, &query).await {
            Ok(v) => Some(v),
            Err(e) => {
                // A dead server/helper: drop the cache and fall back to keywords for this query.
                *cached = None;
                note = Some(format!("Keyword search only ({e})"));
                None
            }
        },
        None => None,
    };
    drop(cached);
    let db = Db::open(&p.db_file()).map_err(err)?;
    let hits =
        ghostreel_core::search::search_with_vector(&db, &p.data_dir, &query, vector.as_deref(), &opts).map_err(err)?;
    Ok(SearchView { hits, note })
}

/// Base URL of the local media server (append `?path=<url-encoded absolute path>`).
#[tauri::command]
async fn media_base(server: State<'_, tokio::sync::OnceCell<media::MediaServer>>) -> CmdResult<String> {
    let s = server.get_or_try_init(media::start).await?;
    Ok(s.base.clone())
}

#[derive(Serialize)]
struct SearchView {
    hits: Vec<ghostreel_core::search::Hit>,
    note: Option<String>,
}

/// Open a video in the system player at `t` seconds (mpv/VLC when installed, else the default app).
#[tauri::command]
fn open_external(app: AppHandle, path: PathBuf, t: Option<f64>) -> CmdResult<()> {
    let t = t.unwrap_or(0.0).max(0.0);
    if let Some(mpv) = doctor::locate("mpv") {
        std::process::Command::new(mpv).arg(format!("--start={t:.1}")).arg(&path).spawn().map_err(err)?;
        return Ok(());
    }
    if let Some(vlc) = doctor::locate("vlc") {
        std::process::Command::new(vlc).arg(format!("--start-time={t:.1}")).arg(&path).spawn().map_err(err)?;
        return Ok(());
    }
    use tauri_plugin_opener::OpenerExt;
    app.opener().open_path(path.to_string_lossy(), None::<&str>).map_err(err)
}

#[tauri::command]
fn remove_folder(project_id: i64, path: PathBuf) -> CmdResult<()> {
    open_db()?.remove_folder(project_id, &path).map_err(err)
}

#[tauri::command]
async fn enqueue_index(app: AppHandle, queue: State<'_, queue::Queue>, project_id: i64) -> CmdResult<u64> {
    let name = open_db()?.project(project_id).map_err(err)?.name;
    Ok(queue.enqueue(&app, queue::TaskKind::Index { project_id }, format!("Index “{name}”")).await)
}

#[tauri::command]
async fn enqueue_preview(
    app: AppHandle,
    queue: State<'_, queue::Queue>,
    script_id: i64,
    burn_titles: bool,
    burn_narration: bool,
) -> CmdResult<u64> {
    let db = open_db()?;
    let stored = ghostreel_core::script::load(&db, script_id).map_err(err)?;
    let label = format!("Preview “{}” v{}", stored.title, stored.version);
    Ok(queue.enqueue(&app, queue::TaskKind::RenderPreview { script_id, burn_titles, burn_narration }, label).await)
}

#[tauri::command]
async fn enqueue_export(
    app: AppHandle,
    queue: State<'_, queue::Queue>,
    script_id: i64,
    format: String,
    path: String,
) -> CmdResult<u64> {
    let db = open_db()?;
    let stored = ghostreel_core::script::load(&db, script_id).map_err(err)?;
    let label = format!("Export “{}” → {}", stored.title, format);
    Ok(queue.enqueue(&app, queue::TaskKind::Export { script_id, format, path }, label).await)
}

#[tauri::command]
fn preview_plan(script_id: i64) -> CmdResult<Vec<ghostreel_core::preview::PlannedSegment>> {
    let db = open_db()?;
    ghostreel_core::preview::preview_plan(&db, script_id).map_err(err)
}

#[tauri::command]
async fn queue_list(queue: State<'_, queue::Queue>) -> CmdResult<Vec<queue::Task>> {
    Ok(queue.snapshot().await)
}

#[tauri::command]
async fn cancel_task(app: AppHandle, queue: State<'_, queue::Queue>, id: u64) -> CmdResult<bool> {
    Ok(queue.cancel(&app, id).await)
}

#[tauri::command]
async fn clear_finished_tasks(app: AppHandle, queue: State<'_, queue::Queue>) -> CmdResult<()> {
    queue.clear_finished(&app).await;
    Ok(())
}

#[derive(Serialize)]
struct ModelsStatusView {
    dir: PathBuf,
    models: Vec<ghostreel_core::models::ModelStatus>,
    current_whisper_model: String,
}

#[tauri::command]
fn models_status() -> CmdResult<ModelsStatusView> {
    let p = paths()?;
    let config = Config::load(&p.config_file).unwrap_or_default();
    let dir = ghostreel_core::models::effective_models_dir(&p, &config);
    let models = ghostreel_core::models::status(&dir, &config.models.search_paths);
    let current_whisper_model = config.stt.model;
    Ok(ModelsStatusView { dir, models, current_whisper_model })
}

#[tauri::command]
async fn enqueue_model_download(app: AppHandle, queue: State<'_, queue::Queue>, model_id: String) -> CmdResult<u64> {
    let file_name = if let Some(e) = ghostreel_core::models::find_entry(&model_id) {
        e.file_name
    } else if let Ok(s) = ghostreel_core::models::whisper(&model_id) {
        s.file_name
    } else {
        return Err(format!("unknown model '{model_id}'"));
    };
    let label = format!("Download {file_name}");
    Ok(queue.enqueue(&app, queue::TaskKind::DownloadModel { model_id }, label).await)
}

#[tauri::command]
fn remove_model(model_id: String) -> CmdResult<()> {
    let p = paths()?;
    let config = Config::load(&p.config_file).unwrap_or_default();
    let dir = ghostreel_core::models::effective_models_dir(&p, &config);
    ghostreel_core::models::remove(&dir, &model_id).map_err(err)?;
    Ok(())
}

#[tauri::command]
fn set_whisper_model(model_id: String) -> CmdResult<()> {
    let p = paths()?;
    let mut config = Config::load(&p.config_file).unwrap_or_default();
    config.stt.model = model_id;
    config.save(&p.config_file).map_err(err)?;
    Ok(())
}

#[tauri::command]
fn open_models_dir(app: AppHandle) -> CmdResult<()> {
    let p = paths()?;
    let config = Config::load(&p.config_file).unwrap_or_default();
    let dir = ghostreel_core::models::effective_models_dir(&p, &config);
    let _ = std::fs::create_dir_all(&dir);
    use tauri_plugin_opener::OpenerExt;
    app.opener().open_path(dir.to_string_lossy(), None::<&str>).map_err(err)
}

#[tauri::command]
fn video_transcript(video_id: i64) -> CmdResult<Vec<TranscriptSegment>> {
    index::transcript(&open_db()?, video_id).map_err(err)
}

#[tauri::command]
fn video_frames(video_id: i64) -> CmdResult<Vec<FrameRow>> {
    let p = paths()?;
    index::frames(&Db::open(&p.db_file()).map_err(err)?, &p.data_dir, video_id).map_err(err)
}

#[derive(Serialize)]
struct ScriptView {
    stored: ghostreel_core::script::StoredScript,
    issues: Vec<ghostreel_core::script::Issue>,
}

#[derive(Serialize)]
struct SaveScriptView {
    script_id: i64,
    issues: Vec<ghostreel_core::script::Issue>,
}

#[tauri::command]
async fn chat_turn(
    app: AppHandle,
    queue: State<'_, queue::Queue>,
    project_id: i64,
    session_id: Option<i64>,
    message: String,
) -> CmdResult<queue::ChatTurnView> {
    let session_id = match session_id {
        Some(id) => id,
        None => {
            let title: String = message.chars().take(60).collect();
            let db = open_db()?;
            ghostreel_core::chat::create_session(&db, project_id, &title).map_err(err)?
        }
    };
    let label = format!("Script chat: {}", message.chars().take(40).collect::<String>());
    let (tx, rx) = tokio::sync::oneshot::channel();
    queue.enqueue_chat(&app, project_id, session_id, message, label, tx).await;
    match rx.await {
        Ok(res) => res,
        Err(_) => Err("chat task cancelled or failed".into()),
    }
}

#[tauri::command]
fn chat_sessions(project_id: i64) -> CmdResult<Vec<ghostreel_core::chat::ChatSession>> {
    let db = open_db()?;
    ghostreel_core::chat::sessions(&db, project_id).map_err(err)
}

#[tauri::command]
fn chat_messages(session_id: i64) -> CmdResult<Vec<ghostreel_core::chat::ChatMessage>> {
    let db = open_db()?;
    ghostreel_core::chat::messages(&db, session_id).map_err(err)
}

#[tauri::command]
fn list_scripts(project_id: i64) -> CmdResult<Vec<ghostreel_core::script::ScriptSummary>> {
    let db = open_db()?;
    ghostreel_core::script::list(&db, project_id).map_err(err)
}

#[tauri::command]
fn get_script(script_id: i64) -> CmdResult<ScriptView> {
    let db = open_db()?;
    let stored = ghostreel_core::script::load(&db, script_id).map_err(err)?;
    let issues = ghostreel_core::script::validate(&db, stored.project_id, &stored.script).map_err(err)?;
    Ok(ScriptView { stored, issues })
}

#[tauri::command]
fn save_script(
    project_id: i64,
    mut script: ghostreel_core::script::Script,
    session_id: Option<i64>,
) -> CmdResult<SaveScriptView> {
    let db = open_db()?;
    let _ = ghostreel_core::script::snap_to_segments(&db, &mut script).map_err(err)?;
    let issues = ghostreel_core::script::validate(&db, project_id, &script).map_err(err)?;
    let script_id = ghostreel_core::script::save_version(&db, project_id, &script, session_id).map_err(err)?;
    Ok(SaveScriptView { script_id, issues })
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
        .plugin(tauri_plugin_opener::init())
        .manage(SearchState::default())
        .manage(tokio::sync::OnceCell::<media::MediaServer>::new())
        .manage(queue::Queue::default())
        .setup(|app| {
            // Frames/thumbnails are served from the data dir via the asset protocol; scope it
            // at runtime because GHOSTREEL_DATA can move it.
            use tauri::Manager;
            if let Ok(p) = Paths::resolve() {
                let _ = std::fs::create_dir_all(&p.data_dir);
                app.asset_protocol_scope().allow_directory(&p.data_dir, true)?;
                let previews_dir = p.data_dir.join("previews");
                let _ = std::fs::create_dir_all(&previews_dir);
                let _ = app.asset_protocol_scope().allow_directory(&previews_dir, true);
                let proxies_dir = p.data_dir.join("proxies");
                let _ = std::fs::create_dir_all(&proxies_dir);
                let _ = app.asset_protocol_scope().allow_directory(&proxies_dir, true);
                // Watched folders, so the player can load the videos.
                if let Ok(db) = Db::open(&p.db_file()) {
                    for f in db.folders(None).unwrap_or_default() {
                        let _ = app.asset_protocol_scope().allow_directory(&f.path, true);
                    }
                }
            }
            tauri::async_runtime::spawn(queue::worker(app.handle().clone()));
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            doctor,
            list_projects,
            create_project,
            remove_project,
            project_view,
            add_folder,
            remove_folder,
            enqueue_index,
            enqueue_preview,
            enqueue_export,
            preview_plan,
            queue_list,
            cancel_task,
            clear_finished_tasks,
            video_transcript,
            video_frames,
            search,
            open_external,
            media_base,
            chat_turn,
            chat_sessions,
            chat_messages,
            list_scripts,
            get_script,
            save_script,
            models_status,
            enqueue_model_download,
            remove_model,
            set_whisper_model,
            open_models_dir,
        ])
        .run(tauri::generate_context!())
        .expect("error while running GhostReel");
}
