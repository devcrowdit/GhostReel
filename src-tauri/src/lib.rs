//! GhostReel desktop app. UI logic lives in the React frontend; everything else is
//! `ghostreel-core`, shared with the CLI.

use ghostreel_core::doctor::{self, Report};
use ghostreel_core::paths::Paths;

/// Doctor report + the blockers list the UI shows at the top.
#[derive(serde::Serialize)]
struct DoctorView {
    report: Report,
    blockers: Vec<String>,
}

#[tauri::command]
async fn doctor() -> Result<DoctorView, String> {
    let paths = Paths::resolve().map_err(|e| e.to_string())?;
    let report = doctor::run(&paths).await;
    let blockers = report.blockers();
    Ok(DoctorView { report, blockers })
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
        .invoke_handler(tauri::generate_handler![doctor])
        .run(tauri::generate_context!())
        .expect("error while running GhostReel");
}
