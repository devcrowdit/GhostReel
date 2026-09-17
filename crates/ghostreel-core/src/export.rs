//! Script timeline export and sidecar management (plan §4a, D14).

use std::path::{Path, PathBuf};
use std::str::FromStr;

use rusqlite::params;
use serde::{Deserialize, Serialize};

use crate::Error;
use crate::db::Db;
use crate::otio;
use crate::script;

/// Supported timeline export formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExportFormat {
    Otio,
    FcpXml,
}

impl ExportFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Otio => "otio",
            Self::FcpXml => "fcp_xml",
        }
    }

    pub fn extension(&self) -> &'static str {
        match self {
            Self::Otio => "otio",
            Self::FcpXml => "xml",
        }
    }
}

impl FromStr for ExportFormat {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "otio" | ".otio" | "otio_json" => Ok(Self::Otio),
            "fcp_xml" | "fcpxml" | "fcp" | "xml" | ".xml" => Ok(Self::FcpXml),
            other => Err(Error::Invalid(format!("unknown export format '{other}' (expected 'otio' or 'fcp_xml')"))),
        }
    }
}

impl std::fmt::Display for ExportFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Command invocation details for the ghostreel-otio sidecar.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SidecarCmd {
    pub program: PathBuf,
    pub prefix_args: Vec<String>,
}

fn exe_name(name: &str) -> String {
    if cfg!(windows) { format!("{name}.exe") } else { name.to_string() }
}

/// Locate the ghostreel-otio sidecar executable or python script fallback.
///
/// Order:
/// 1. env GHOSTREEL_OTIO (binary path)
/// 2. ghostreel-otio(.exe) next to current executable
/// 3. PATH lookup
/// 4. Dev fallback: env GHOSTREEL_OTIO_PY (python interpreter) + script
///    (env GHOSTREEL_OTIO_SCRIPT, else tools/ghostreel-otio/ghostreel_otio.py walking up,
///    then from env!("CARGO_MANIFEST_DIR")/../..).
pub fn locate_sidecar() -> Option<SidecarCmd> {
    // 1. env GHOSTREEL_OTIO
    if let Some(p) = std::env::var_os("GHOSTREEL_OTIO") {
        let path = PathBuf::from(p);
        if path.is_file() {
            return Some(SidecarCmd { program: path, prefix_args: vec![] });
        }
    }

    // 2. Next to current executable
    let file = exe_name("ghostreel-otio");
    if let Some(dir) = std::env::current_exe().ok().and_then(|e| e.parent().map(Path::to_path_buf)) {
        let path = dir.join(&file);
        if path.is_file() {
            return Some(SidecarCmd { program: path, prefix_args: vec![] });
        }
    }

    // 3. PATH lookup
    if let Some(paths) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&paths) {
            let path = dir.join(&file);
            if path.is_file() {
                return Some(SidecarCmd { program: path, prefix_args: vec![] });
            }
        }
    }

    // 4. Dev fallback: env GHOSTREEL_OTIO_PY + script
    if let Some(py) = std::env::var_os("GHOSTREEL_OTIO_PY") {
        let py_path = PathBuf::from(py);
        if py_path.is_file() {
            return locate_python_script().map(|script| SidecarCmd {
                program: py_path,
                prefix_args: vec![script.to_string_lossy().into_owned()],
            });
        }
    }

    None
}

fn locate_python_script() -> Option<PathBuf> {
    if let Some(s) = std::env::var_os("GHOSTREEL_OTIO_SCRIPT") {
        let path = PathBuf::from(s);
        if path.is_file() {
            return Some(path);
        }
    }

    // Walk up from current dir looking for tools/ghostreel-otio/ghostreel_otio.py
    if let Ok(mut dir) = std::env::current_dir() {
        loop {
            let candidate = dir.join("tools/ghostreel-otio/ghostreel_otio.py");
            if candidate.is_file() {
                return Some(candidate);
            }
            if !dir.pop() {
                break;
            }
        }
    }

    // Fallback relative to compile-time manifest dir
    let candidate = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/ghostreel-otio/ghostreel_otio.py");
    if candidate.is_file() {
        return Some(candidate);
    }

    None
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExportResult {
    pub export_id: i64,
    pub path: PathBuf,
    pub format: ExportFormat,
}

/// Export a script to an OpenTimelineIO or Final Cut Pro 7 XML timeline file.
pub fn export_script(
    db: &Db,
    data_dir: &Path,
    script_id: i64,
    format: ExportFormat,
    out_path: &Path,
) -> Result<ExportResult, Error> {
    let stored = script::load(db, script_id)?;
    let timeline = otio::build_timeline(db, stored.project_id, &stored.script)?;

    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| Error::Io(parent.to_path_buf(), e))?;
    }

    match format {
        ExportFormat::Otio => {
            let json_str = serde_json::to_string_pretty(&timeline)
                .map_err(|e| Error::Export(format!("failed to serialize timeline JSON: {e}")))?;
            std::fs::write(out_path, json_str).map_err(|e| Error::Io(out_path.to_path_buf(), e))?;
        }
        ExportFormat::FcpXml => {
            let exports_dir = data_dir.join("exports");
            std::fs::create_dir_all(&exports_dir).map_err(|e| Error::Io(exports_dir.clone(), e))?;

            let tmp_otio = exports_dir.join(format!("script-{}-v{}.otio", stored.id, stored.version));
            let json_str = serde_json::to_string_pretty(&timeline)
                .map_err(|e| Error::Export(format!("failed to serialize timeline JSON: {e}")))?;
            std::fs::write(&tmp_otio, json_str).map_err(|e| Error::Io(tmp_otio.clone(), e))?;

            let sidecar = locate_sidecar().ok_or_else(|| Error::Export("ghostreel-otio sidecar not found".into()))?;

            let mut cmd = crate::proc::std_command(&sidecar.program);
            cmd.args(&sidecar.prefix_args);
            cmd.arg("convert").arg(&tmp_otio).arg(out_path).arg("--adapter").arg("fcp_xml");

            let output = cmd.output().map_err(|e| Error::Export(format!("failed to spawn sidecar: {e}")))?;
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                let stdout = String::from_utf8_lossy(&output.stdout);
                return Err(Error::Export(format!("fcp_xml conversion failed: {stderr} {stdout}").trim().into()));
            }
        }
    }

    let now = crate::projects::now();
    db.conn.execute(
        "INSERT INTO exports(script_id, format, path, created_at) VALUES (?1, ?2, ?3, ?4)",
        params![stored.id, format.as_str(), out_path.to_string_lossy(), now],
    )?;
    let export_id = db.conn.last_insert_rowid();

    Ok(ExportResult { export_id, path: out_path.to_path_buf(), format })
}

/// Validate an exported timeline file using the sidecar.
pub fn validate_export(path: &Path) -> Result<serde_json::Value, Error> {
    let sidecar = locate_sidecar().ok_or_else(|| Error::Export("ghostreel-otio sidecar not found".into()))?;

    let mut cmd = crate::proc::std_command(&sidecar.program);
    cmd.args(&sidecar.prefix_args);
    cmd.arg("validate").arg(path);

    let output = cmd.output().map_err(|e| Error::Export(format!("failed to execute sidecar validate: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(Error::Export(format!("timeline validation failed: {stderr} {stdout}").trim().into()));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let val: serde_json::Value = serde_json::from_str(stdout.trim())
        .map_err(|e| Error::Export(format!("invalid sidecar JSON response: {e} ({stdout})")))?;

    Ok(val)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projects::NewProject;

    #[test]
    fn format_parsing_and_extension() {
        assert_eq!(ExportFormat::from_str("otio").unwrap(), ExportFormat::Otio);
        assert_eq!(ExportFormat::from_str(".otio").unwrap(), ExportFormat::Otio);
        assert_eq!(ExportFormat::from_str("otio_json").unwrap(), ExportFormat::Otio);
        assert_eq!(ExportFormat::from_str("fcp_xml").unwrap(), ExportFormat::FcpXml);
        assert_eq!(ExportFormat::from_str("xml").unwrap(), ExportFormat::FcpXml);
        assert_eq!(ExportFormat::from_str(".xml").unwrap(), ExportFormat::FcpXml);
        assert!(ExportFormat::from_str("unknown").is_err());

        assert_eq!(ExportFormat::Otio.as_str(), "otio");
        assert_eq!(ExportFormat::FcpXml.as_str(), "fcp_xml");
        assert_eq!(ExportFormat::Otio.extension(), "otio");
        assert_eq!(ExportFormat::FcpXml.extension(), "xml");
    }

    #[test]
    fn sidecar_export_fcp_xml_integration() {
        if std::env::var_os("GHOSTREEL_OTIO_PY").is_none() {
            eprintln!("skipping export integration test: GHOSTREEL_OTIO_PY not set");
            return;
        }

        let mut db = Db::open_in_memory().unwrap();
        let project = db.create_project(&NewProject::named("TestExport")).unwrap();
        let temp = tempfile::tempdir().unwrap();
        let folder = db.add_folder(project.id, temp.path(), true).unwrap();

        let vid_path = temp.path().join("v1.mp4");
        std::fs::write(&vid_path, b"test").unwrap();

        db.conn
            .execute(
                "INSERT INTO videos(id, content_hash, size, duration_s, fps, has_audio)
                 VALUES (1, 'hash1', 100, 10.0, 25.0, 1)",
                [],
            )
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen)
                 VALUES (1, ?1, ?2, 100, 0, 0)",
                params![folder.id, vid_path.to_str().unwrap()],
            )
            .unwrap();

        let script = script::Script {
            title: "Integration Test".into(),
            target_duration_s: Some(5.0),
            fps: Some(script::Fps::new(25, 1)),
            width: Some(1920),
            height: Some(1080),
            beats: vec![script::Beat {
                id: "b1".into(),
                purpose: "intro".into(),
                narration: Some("Hello".into()),
                on_screen_text: Some("Title".into()),
                clips: vec![script::ScriptClip {
                    video_id: 1,
                    in_s: 0.0,
                    out_s: 4.0,
                    audio: script::Audio::Source,
                    why: None,
                }],
                notes: None,
            }],
        };

        let script_id = script::save_version(&db, project.id, &script, None).unwrap();

        let out_xml = temp.path().join("out.xml");
        let res = export_script(&db, temp.path(), script_id, ExportFormat::FcpXml, &out_xml).unwrap();
        assert_eq!(res.format, ExportFormat::FcpXml);
        assert!(out_xml.is_file());

        let validation = validate_export(&out_xml).unwrap();
        assert_eq!(validation["clips"].as_i64(), Some(2)); // 1 V1 + 1 A1
        assert_eq!(validation["duration_s"].as_f64(), Some(4.0));
    }
}
