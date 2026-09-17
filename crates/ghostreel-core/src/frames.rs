//! Keyframes (plan §3 step 5): pick moments worth describing, save them as JPEGs, drop duplicates.
//!
//! 1. One fast low-resolution decode finds scene changes (`select=gt(scene,…)`).
//! 2. Gaps longer than `max_interval_s` are filled so static footage (talking heads, screen
//!    recordings) still gets a frame every so often; cuts closer than `min_interval_s` are thinned.
//! 3. Each chosen moment is extracted at full quality (long side 1280 px — enough for a vision model
//!    to read on-screen text, see S0 findings).
//! 4. A 64-bit difference hash drops frames that look the same as the previous kept frame.

use std::path::{Path, PathBuf};
use std::process::Stdio;

use serde::Serialize;
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::Error;

#[derive(Debug, Clone)]
pub struct FrameOptions {
    pub scene_threshold: f64,
    pub max_interval_s: f64,
    pub min_interval_s: f64,
    pub long_side: u32,
    /// Hamming distance at or below which two frames count as duplicates.
    pub dup_distance: u32,
    /// Keep a "duplicate" anyway once this long has passed since the last kept frame: the hash is
    /// coarse, and screen recordings change text without changing layout.
    pub max_dup_gap_s: f64,
    pub max_frames: usize,
}

impl Default for FrameOptions {
    fn default() -> Self {
        Self {
            scene_threshold: 0.3,
            max_interval_s: 20.0,
            min_interval_s: 2.0,
            long_side: 1280,
            dup_distance: 5,
            max_dup_gap_s: 60.0,
            max_frames: 600,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Frame {
    pub t_s: f64,
    pub path: PathBuf,
    pub dhash: u64,
}

/// Scene-change timestamps from one low-resolution decode. `on_progress` gets seconds decoded.
pub async fn scene_times(
    ffmpeg: &Path,
    video: &Path,
    threshold: f64,
    mut on_progress: impl FnMut(f64),
) -> Result<Vec<f64>, Error> {
    let filter = format!("fps=4,scale=256:-2,select='gt(scene\\,{threshold})',showinfo");
    let mut child = tokio::process::Command::new(ffmpeg)
        .args(["-nostdin", "-hide_banner", "-nostats", "-i"])
        .arg(video)
        .args(["-an", "-sn", "-dn", "-vf", &filter, "-f", "null", "-", "-progress", "pipe:1"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| Error::Frames(format!("cannot run {}: {e}", ffmpeg.display())))?;

    let stderr = child.stderr.take().ok_or_else(|| Error::Frames("ffmpeg stderr unavailable".into()))?;
    let showinfo = tokio::spawn(async move {
        let mut times = Vec::new();
        let mut last_error = String::new();
        let mut lines = BufReader::new(stderr).lines();
        while let Ok(Some(l)) = lines.next_line().await {
            if l.contains("Parsed_showinfo") {
                if let Some(t) = l.split("pts_time:").nth(1).and_then(|v| v.split_whitespace().next())
                    && let Ok(t) = t.parse::<f64>()
                {
                    times.push(t);
                }
            } else if !l.trim().is_empty() {
                last_error = l;
            }
        }
        (times, last_error)
    });

    let stdout = child.stdout.take().ok_or_else(|| Error::Frames("ffmpeg stdout unavailable".into()))?;
    let mut lines = BufReader::new(stdout).lines();
    while let Ok(Some(l)) = lines.next_line().await {
        if let Some(us) = l.strip_prefix("out_time_us=").and_then(|v| v.trim().parse::<i64>().ok()) {
            on_progress(us.max(0) as f64 / 1e6);
        }
    }
    let status = child.wait().await.map_err(|e| Error::Frames(e.to_string()))?;
    let (times, last_error) = showinfo.await.unwrap_or_default();
    if !status.success() {
        return Err(Error::Frames(format!("scene detection failed: {last_error}")));
    }
    Ok(times)
}

/// Final sampling plan: an early frame, scene cuts (thinned to `min_interval_s`), and fillers so no
/// gap exceeds `max_interval_s`. Sorted, within `[0, duration)`, at most `max_frames`.
pub fn plan_times(duration_s: f64, scenes: &[f64], opts: &FrameOptions) -> Vec<f64> {
    if duration_s <= 0.0 {
        return vec![0.0];
    }
    // Frame 0 is often black/fade-in: start a little in.
    let first = (duration_s * 0.05).min(1.0);
    let last_ok = (duration_s - 0.05).max(0.0);
    let mut picks = vec![first];
    let mut sorted: Vec<f64> = scenes.iter().copied().filter(|t| *t > first && *t < last_ok).collect();
    sorted.sort_by(f64::total_cmp);
    for t in sorted {
        // Land slightly after the cut so the new shot is fully on screen.
        let t = (t + 0.2).min(last_ok);
        if t - picks.last().unwrap() >= opts.min_interval_s {
            picks.push(t);
        }
    }
    // Fill long gaps (including up to the end).
    let mut filled = Vec::with_capacity(picks.len());
    let bounds: Vec<f64> = picks.iter().copied().chain(std::iter::once(duration_s)).collect();
    for w in bounds.windows(2) {
        let (a, b) = (w[0], w[1]);
        filled.push(a);
        let gap = b - a;
        if gap > opts.max_interval_s {
            let n = (gap / opts.max_interval_s).ceil() as usize;
            for k in 1..n {
                let t = a + gap * k as f64 / n as f64;
                if t < last_ok {
                    filled.push(t);
                }
            }
        }
    }
    if filled.len() > opts.max_frames {
        // Keep an even spread rather than only the beginning.
        let step = filled.len() as f64 / opts.max_frames as f64;
        filled = (0..opts.max_frames).map(|i| filled[(i as f64 * step) as usize]).collect();
    }
    filled
}

/// Extract the frame at `t_s` as a JPEG (long side ≤ `long_side`).
pub async fn extract(ffmpeg: &Path, video: &Path, t_s: f64, out: &Path, long_side: u32) -> Result<(), Error> {
    if let Some(dir) = out.parent() {
        tokio::fs::create_dir_all(dir).await.map_err(|e| Error::Io(dir.to_path_buf(), e))?;
    }
    let scale = format!("scale='if(gt(iw,ih),min({long_side},iw),-2)':'if(gt(iw,ih),-2,min({long_side},ih))'");
    let output = tokio::process::Command::new(ffmpeg)
        .args(["-nostdin", "-v", "error", "-y", "-ss", &format!("{t_s:.3}"), "-i"])
        .arg(video)
        .args(["-frames:v", "1", "-an", "-vf", &scale, "-q:v", "3"])
        .arg(out)
        .kill_on_drop(true)
        .output()
        .await
        .map_err(|e| Error::Frames(format!("cannot run {}: {e}", ffmpeg.display())))?;
    if !output.status.success() || !out.is_file() {
        let msg = String::from_utf8_lossy(&output.stderr);
        return Err(Error::Frames(format!("extract at {t_s:.1}s: {}", msg.lines().next().unwrap_or("no frame"))));
    }
    Ok(())
}

/// 64-bit difference hash: grayscale 9×8, one bit per horizontal neighbour comparison.
pub fn dhash(path: &Path) -> Result<u64, Error> {
    let img = image::open(path).map_err(|e| Error::Frames(format!("{}: {e}", path.display())))?;
    let small = img.grayscale().resize_exact(9, 8, image::imageops::FilterType::Triangle).to_luma8();
    let mut hash = 0u64;
    for y in 0..8 {
        for x in 0..8 {
            let bit = small.get_pixel(x, y)[0] > small.get_pixel(x + 1, y)[0];
            hash = (hash << 1) | bit as u64;
        }
    }
    Ok(hash)
}

pub fn hamming(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

/// Full pipeline for one video. Frames land in `dir` as `t<milliseconds>.jpg`; duplicates are
/// deleted. `on_progress` gets a 0–1 fraction (decode ≈ 70 %, extraction ≈ 30 %).
pub async fn extract_keyframes(
    ffmpeg: &Path,
    video: &Path,
    duration_s: f64,
    dir: &Path,
    opts: &FrameOptions,
    mut on_progress: impl FnMut(f64),
) -> Result<Vec<Frame>, Error> {
    let scenes = scene_times(ffmpeg, video, opts.scene_threshold, |secs| {
        if duration_s > 0.0 {
            on_progress(0.7 * (secs / duration_s).min(1.0));
        }
    })
    .await?;
    let times = plan_times(duration_s, &scenes, opts);

    // Extract in parallel (ffmpeg seeks are cheap), then dedupe in time order.
    let sem = std::sync::Arc::new(tokio::sync::Semaphore::new(4));
    let mut set = tokio::task::JoinSet::new();
    for (i, t) in times.iter().copied().enumerate() {
        let (sem, ffmpeg, video) = (sem.clone(), ffmpeg.to_path_buf(), video.to_path_buf());
        let out = dir.join(format!("t{:09}.jpg", (t * 1000.0).round() as u64));
        let long_side = opts.long_side;
        set.spawn(async move {
            let _permit = sem.acquire_owned().await;
            let r = extract(&ffmpeg, &video, t, &out, long_side).await;
            let h = match r {
                Ok(()) => {
                    let p = out.clone();
                    tokio::task::spawn_blocking(move || dhash(&p))
                        .await
                        .unwrap_or_else(|e| Err(Error::Frames(e.to_string())))
                }
                Err(e) => Err(e),
            };
            (i, t, out, h)
        });
    }
    let mut extracted = Vec::with_capacity(times.len());
    let mut done = 0usize;
    let mut first_error = None;
    while let Some(j) = set.join_next().await {
        let Ok((i, t, out, h)) = j else { continue };
        done += 1;
        on_progress(0.7 + 0.3 * done as f64 / times.len().max(1) as f64);
        match h {
            Ok(h) => extracted.push((i, Frame { t_s: t, path: out, dhash: h })),
            Err(e) => {
                // Seeking past the last decodable frame fails on some files; skip that moment.
                let _ = tokio::fs::remove_file(&out).await;
                first_error.get_or_insert(e);
            }
        }
    }
    extracted.sort_by_key(|(i, _)| *i);
    if extracted.is_empty() {
        return Err(first_error.unwrap_or_else(|| Error::Frames("no frames extracted".into())));
    }

    let mut kept: Vec<Frame> = Vec::with_capacity(extracted.len());
    for (_, f) in extracted {
        let dup = kept
            .last()
            .is_some_and(|k| hamming(k.dhash, f.dhash) <= opts.dup_distance && f.t_s - k.t_s < opts.max_dup_gap_s);
        if dup {
            let _ = tokio::fs::remove_file(&f.path).await;
        } else {
            kept.push(f);
        }
    }
    Ok(kept)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts() -> FrameOptions {
        FrameOptions::default()
    }

    #[test]
    fn plan_fills_gaps_and_thins_cuts() {
        // 69 s video, cuts at 8, 8.5 (flash), 53, 63.
        let t = plan_times(69.0, &[8.0, 8.5, 53.0, 63.0], &opts());
        assert_eq!(t[0], 1.0);
        assert!(t.contains(&8.2) && !t.contains(&8.7), "cut 0.5 s after another is thinned: {t:?}");
        assert!(t.windows(2).all(|w| w[1] - w[0] <= 20.0 + 1e-9), "no gap over 20 s: {t:?}");
        assert!(t.windows(2).all(|w| w[1] > w[0]));
        assert!(*t.last().unwrap() < 69.0);
        // Static 5-minute screen recording: a frame at least every 20 s.
        let t = plan_times(300.0, &[], &opts());
        assert!(t.len() >= 15, "{}", t.len());
        // Tiny clip.
        assert_eq!(plan_times(0.8, &[], &opts()), vec![0.04000000000000001]);
    }

    #[test]
    fn plan_caps_frame_count_evenly() {
        let o = FrameOptions { max_frames: 10, ..opts() };
        let t = plan_times(3600.0, &[], &o);
        assert_eq!(t.len(), 10);
        assert!(*t.last().unwrap() > 3000.0, "spread across the video, not just the start");
    }

    fn have_ffmpeg() -> bool {
        std::process::Command::new("ffmpeg").arg("-version").output().map(|o| o.status.success()).unwrap_or(false)
    }

    #[tokio::test]
    async fn keyframes_of_a_video_with_scene_cuts() {
        if !have_ffmpeg() {
            eprintln!("skipping: no ffmpeg");
            return;
        }
        let tmp = tempfile::tempdir().unwrap();
        let video = tmp.path().join("scenes.mp4");
        // 8 s bars, 30 s static red, 6 s testsrc (moving), 6 s blue.
        let ok = std::process::Command::new("ffmpeg")
            .args(["-v", "error", "-f", "lavfi", "-i", "smptebars=size=640x360:rate=10:d=8"])
            .args(["-f", "lavfi", "-i", "color=c=red:size=640x360:rate=10:d=30"])
            .args(["-f", "lavfi", "-i", "testsrc=size=640x360:rate=10:d=6"])
            .args(["-f", "lavfi", "-i", "color=c=blue:size=640x360:rate=10:d=6"])
            .args(["-filter_complex", "[0:v][1:v][2:v][3:v]concat=n=4:v=1:a=0,format=yuv420p", "-c:v", "mpeg4"])
            .arg(&video)
            .status()
            .unwrap()
            .success();
        assert!(ok);
        let dir = tmp.path().join("frames");
        let mut progress = Vec::new();
        let frames =
            extract_keyframes(Path::new("ffmpeg"), &video, 50.0, &dir, &opts(), |p| progress.push(p)).await.unwrap();

        let times: Vec<f64> = frames.iter().map(|f| f.t_s).collect();
        assert!(frames.iter().all(|f| f.path.is_file()));
        // The static red section yields filler frames that are all duplicates of the first red one.
        let red = times.iter().filter(|t| **t > 8.0 && **t < 38.0).count();
        assert_eq!(red, 1, "static red section collapses to one frame: {times:?}");
        assert!(times.iter().any(|t| (38.0..44.0).contains(t)), "testsrc section present: {times:?}");
        assert!(times.iter().any(|t| *t >= 44.0), "blue section present: {times:?}");
        let files = std::fs::read_dir(&dir).unwrap().count();
        assert_eq!(files, frames.len(), "duplicate JPEGs deleted");
        let img = image::image_dimensions(&frames[0].path).unwrap();
        assert_eq!(img, (640, 360), "never upscaled");
        assert!((progress.last().unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn dhash_distinguishes_images() {
        let tmp = tempfile::tempdir().unwrap();
        let mk = |name: &str, f: &dyn Fn(u32, u32) -> u8| {
            let img = image::GrayImage::from_fn(64, 64, |x, y| image::Luma([f(x, y)]));
            let p = tmp.path().join(name);
            img.save(&p).unwrap();
            p
        };
        let a = dhash(&mk("a.jpg", &|x, _| (x * 4) as u8)).unwrap();
        let a2 = dhash(&mk("a2.jpg", &|x, _| (x * 4).saturating_add(3) as u8)).unwrap();
        let b = dhash(&mk("b.jpg", &|x, _| (255 - x * 4) as u8)).unwrap();
        assert!(hamming(a, a2) <= 5);
        assert!(hamming(a, b) > 20);
    }
}
