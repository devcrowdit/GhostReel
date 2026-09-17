//! Video file facts: content hashing and ffprobe metadata.

use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::time::Duration;

use serde::Serialize;
use serde_json::Value;

use crate::Error;

/// File extensions treated as video.
pub const VIDEO_EXTENSIONS: &[&str] =
    &["mp4", "m4v", "mov", "mkv", "webm", "avi", "wmv", "flv", "mts", "m2ts", "ts", "mxf", "mpg", "mpeg", "3gp"];

pub fn is_video_path(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| VIDEO_EXTENSIONS.iter().any(|v| v.eq_ignore_ascii_case(e)))
}

/// Folders editing apps fill with their own renders and caches (not footage): Premiere Pro preview
/// renders and auto-saves, Adobe media cache, DaVinci Resolve and Final Cut Pro caches and proxies.
const IGNORED_DIRS: &[&str] = &[
    "adobe premiere pro preview files",
    "adobe premiere pro video previews",
    "adobe premiere pro audio previews",
    "adobe premiere pro auto-save",
    "adobe after effects auto-save",
    "media cache files",
    "media cache",
    "peak files",
    "cacheclip",
    "proxymedia",
    "optimizedmedia",
    "render files",
    "transcoded media",
    "proxy media",
    "analysis files",
];

/// Whether a folder holds app-generated renders or caches that shouldn't be listed or indexed.
/// Hidden folders count too. Premiere names its render folders `<sequence>.PRV`.
pub fn is_ignored_dir(name: &str) -> bool {
    let lower = name.to_lowercase();
    lower.starts_with('.') || lower.ends_with(".prv") || IGNORED_DIRS.contains(&lower.as_str())
}

const EDGE: u64 = 4 * 1024 * 1024;

/// Identity of a video's content, cheap even for huge files: BLAKE3 over the size plus the
/// first and last 4 MiB (whole file when ≤ 8 MiB). Renames and moves keep the same hash, so a
/// file is never re-indexed just because it moved.
pub fn content_hash(path: &Path) -> Result<String, Error> {
    let io = |e| Error::Io(path.to_path_buf(), e);
    let mut f = std::fs::File::open(path).map_err(io)?;
    let size = f.metadata().map_err(io)?.len();
    let mut h = blake3::Hasher::new();
    h.update(&size.to_le_bytes());
    let mut buf = vec![0u8; EDGE as usize];
    if size <= 2 * EDGE {
        let mut all = Vec::with_capacity(size as usize);
        f.read_to_end(&mut all).map_err(io)?;
        h.update(&all);
    } else {
        f.read_exact(&mut buf).map_err(io)?;
        h.update(&buf);
        f.seek(SeekFrom::Start(size - EDGE)).map_err(io)?;
        f.read_exact(&mut buf).map_err(io)?;
        h.update(&buf);
    }
    Ok(format!("b3e:{}", h.finalize().to_hex()))
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct MediaInfo {
    pub duration_s: Option<f64>,
    /// Display size (rotation applied).
    pub width: Option<i64>,
    pub height: Option<i64>,
    pub rotation: i64,
    /// Nominal rate (`r_frame_rate`).
    pub fps: Option<f64>,
    /// Measured average rate (`avg_frame_rate`).
    pub avg_fps: Option<f64>,
    /// Probably variable frame rate (phone footage) — timeline exports need a warning.
    /// Header heuristic only (nominal vs average rate differ > 1 %); slightly-VFR files can slip
    /// through, so the M8 export re-checks packet timestamps before building a timeline.
    pub vfr: bool,
    pub vcodec: Option<String>,
    pub acodec: Option<String>,
    pub has_audio: bool,
    pub created_time: Option<String>,
}

fn parse_rate(s: Option<&str>) -> Option<f64> {
    let s = s?;
    let (n, d) = match s.split_once('/') {
        Some((n, d)) => (n.parse::<f64>().ok()?, d.parse::<f64>().ok()?),
        None => (s.parse::<f64>().ok()?, 1.0),
    };
    (n > 0.0 && d > 0.0).then_some(n / d)
}

fn num(v: &Value) -> Option<f64> {
    v.as_f64().or_else(|| v.as_str()?.parse().ok())
}

/// Parse `ffprobe -print_format json -show_format -show_streams` output.
pub fn parse_ffprobe(json: &str) -> Result<MediaInfo, Error> {
    let v: Value = serde_json::from_str(json).map_err(|e| Error::Probe(format!("invalid JSON: {e}")))?;
    let streams = v["streams"].as_array().cloned().unwrap_or_default();
    // Cover art is a "video" stream with attached_pic; skip it.
    let video = streams
        .iter()
        .find(|s| s["codec_type"] == "video" && s["disposition"]["attached_pic"].as_i64() != Some(1))
        .ok_or_else(|| Error::Probe("no video stream".into()))?;
    let audio = streams.iter().find(|s| s["codec_type"] == "audio");

    let rotation = video["side_data_list"]
        .as_array()
        .and_then(|l| l.iter().find_map(|sd| num(&sd["rotation"])))
        .or_else(|| num(&video["tags"]["rotate"]))
        .map(|r| (r.round() as i64).rem_euclid(360))
        .unwrap_or(0);
    let (mut width, mut height) = (video["width"].as_i64(), video["height"].as_i64());
    if rotation == 90 || rotation == 270 {
        std::mem::swap(&mut width, &mut height);
    }

    let fps = parse_rate(video["r_frame_rate"].as_str());
    let avg_fps = parse_rate(video["avg_frame_rate"].as_str());
    let vfr = match (fps, avg_fps) {
        (Some(r), Some(a)) => ((r - a) / r).abs() > 0.01,
        _ => false,
    };

    let duration_s = num(&v["format"]["duration"]).or_else(|| num(&video["duration"]));
    let created_time = v["format"]["tags"]["creation_time"]
        .as_str()
        .or_else(|| video["tags"]["creation_time"].as_str())
        .map(str::to_string);

    Ok(MediaInfo {
        duration_s,
        width,
        height,
        rotation,
        fps,
        avg_fps,
        vfr,
        vcodec: video["codec_name"].as_str().map(str::to_string),
        acodec: audio.and_then(|a| a["codec_name"].as_str()).map(str::to_string),
        has_audio: audio.is_some(),
        created_time,
    })
}

/// Run ffprobe on `file` (bounded: a corrupt file must not hang indexing).
pub async fn ffprobe(ffprobe_bin: &Path, file: &Path) -> Result<MediaInfo, Error> {
    let fut = tokio::process::Command::new(ffprobe_bin)
        .args(["-v", "error", "-print_format", "json", "-show_format", "-show_streams"])
        .arg(file)
        .kill_on_drop(true)
        .output();
    let out = tokio::time::timeout(Duration::from_secs(60), fut)
        .await
        .map_err(|_| Error::Probe("timed out after 60 s".into()))?
        .map_err(|e| Error::Probe(format!("cannot run {}: {e}", ffprobe_bin.display())))?;
    if !out.status.success() {
        let msg = String::from_utf8_lossy(&out.stderr);
        return Err(Error::Probe(msg.lines().next().unwrap_or("failed").trim().to_string()));
    }
    parse_ffprobe(&String::from_utf8_lossy(&out.stdout))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extensions() {
        assert!(is_video_path(Path::new("/a/B.MOV")));
        assert!(is_video_path(Path::new("clip.mkv")));
        assert!(!is_video_path(Path::new("notes.txt")));
        assert!(!is_video_path(Path::new("mp4")));
    }

    #[test]
    fn hash_ignores_name_and_sees_edges() {
        let dir = tempfile::tempdir().unwrap();
        let big = vec![7u8; (9 * 1024 * 1024) as usize];
        let a = dir.path().join("a.mp4");
        let b = dir.path().join("renamed.mp4");
        std::fs::write(&a, &big).unwrap();
        std::fs::write(&b, &big).unwrap();
        assert_eq!(content_hash(&a).unwrap(), content_hash(&b).unwrap());

        let mut tail = big.clone();
        *tail.last_mut().unwrap() = 8;
        std::fs::write(&b, &tail).unwrap();
        assert_ne!(content_hash(&a).unwrap(), content_hash(&b).unwrap());

        std::fs::write(&b, b"small").unwrap();
        assert!(content_hash(&b).unwrap().starts_with("b3e:"));
    }

    #[test]
    fn parses_rotated_phone_vfr() {
        let json = r#"{
          "streams": [
            {"codec_type":"video","codec_name":"hevc","width":1920,"height":1080,
             "r_frame_rate":"30/1","avg_frame_rate":"2500/87",
             "side_data_list":[{"rotation":-90}]},
            {"codec_type":"audio","codec_name":"aac"}
          ],
          "format": {"duration":"12.500000","tags":{"creation_time":"2026-06-15T19:52:21Z"}}
        }"#;
        let m = parse_ffprobe(json).unwrap();
        assert_eq!((m.width, m.height, m.rotation), (Some(1080), Some(1920), 270));
        assert!(m.vfr);
        assert_eq!(m.duration_s, Some(12.5));
        assert!(m.has_audio);
        assert_eq!(m.vcodec.as_deref(), Some("hevc"));
    }

    #[test]
    fn constant_rate_screen_recording_without_audio() {
        let json = r#"{"streams":[{"codec_type":"video","codec_name":"h264","width":2560,"height":1440,
          "r_frame_rate":"30000/1001","avg_frame_rate":"30000/1001"}],"format":{"duration":"59.3"}}"#;
        let m = parse_ffprobe(json).unwrap();
        assert!(!m.vfr && !m.has_audio);
        assert!((m.fps.unwrap() - 29.97).abs() < 0.01);
        assert!(parse_ffprobe(r#"{"streams":[{"codec_type":"audio"}],"format":{}}"#).is_err());
    }
}
