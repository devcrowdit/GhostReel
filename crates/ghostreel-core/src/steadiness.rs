//! How steady the camera is, measured from the pictures themselves.
//!
//! A shaky shot looks bad in a cut no matter what is in it, and no frame description says so: the
//! model sees "a wide view of a valley" whether the operator was on a tripod or walking. So it is
//! measured at index time and reported with the footage, as the stretches to cut around.
//!
//! The method follows what stabilisers do (vid.stab, the OpenCV pipeline) rather than a single
//! whole-frame match:
//!
//! 1. A grid of small patches is matched between consecutive frames, and patches with too little
//!    contrast (sky, a wall) are discarded — they cannot be matched and would only add noise.
//! 2. The patch motions are fitted to a *similarity* transform: translation, rotation and scale.
//!    A translation-only fit reads a push-in or a tilt as instability; this one does not. The fit
//!    is robust: patches that disagree with the consensus (someone walking through) are dropped.
//! 3. The transforms are accumulated into the camera's path, which is then split by frequency.
//!    Intentional movement — a pan, a walk, a push — is slow; shake is fast. The path minus its
//!    smoothed self is the shake, and its RMS, as a share of the frame width, is the score.
//!
//! Frames come from ffmpeg as small greyscale images, so this needs no GPL filters
//! (`vidstabdetect` is GPL; GhostReel ships an LGPL ffmpeg) and costs a decode plus arithmetic.

use std::path::Path;

use tokio::io::AsyncReadExt;

use crate::Error;

/// Width of the analysed frame, in pixels. Enough for a grid of patches; small enough that a
/// frame pair costs well under a millisecond.
pub const ANALYSIS_W: usize = 160;
/// Height of the analysed frame.
pub const ANALYSIS_H: usize = 90;
/// Frames per second sampled from the video.
pub const ANALYSIS_FPS: u32 = 10;
/// Side of one measurement patch, in analysis pixels.
const PATCH: usize = 16;
/// Largest shift searched for a patch between two frames, in analysis pixels.
const SEARCH_RADIUS: i32 = 10;
/// Patches whose intensity spread is below this cannot be matched and are skipped. Low: a plain
/// wall behind an interviewee has little contrast and is exactly what proves the camera is still.
/// Throwing it away leaves the person, and the fit then tracks their gestures as camera motion.
const MIN_CONTRAST: f64 = 2.0;
/// A patch whose motion differs from the frame's median by more than this is something moving
/// in the shot, not the camera, and is left out of the fit.
const CONSENSUS_PX: f64 = 2.0;
/// Fewest agreeing patches for a frame pair to count as measured.
const MIN_INLIERS: usize = 4;
/// A frame-to-frame move larger than this share of the frame width is a cut, not a camera move:
/// nothing handheld jumps a quarter of the frame in a tenth of a second. The path restarts there.
const CUT_FRACTION: f64 = 0.25;

/// How steady one stretch of a video is.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Window {
    pub start_s: f64,
    pub end_s: f64,
    /// RMS of the camera's high-frequency movement, as a percentage of the frame width per
    /// frame. A tripod sits near 0; handheld footage runs well above 0.5.
    pub jerk: f64,
}

/// The camera's movement from one frame to the next.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Motion {
    pub dx: f64,
    pub dy: f64,
    /// Rotation, radians.
    pub rot: f64,
    /// Scale factor; 1.0 is none.
    pub scale: f64,
}

fn contrast(frame: &[u8], x0: usize, y0: usize) -> f64 {
    let mut sum = 0.0;
    let mut sq = 0.0;
    for y in y0..y0 + PATCH {
        for x in x0..x0 + PATCH {
            let v = frame[y * ANALYSIS_W + x] as f64;
            sum += v;
            sq += v * v;
        }
    }
    let n = (PATCH * PATCH) as f64;
    let mean = sum / n;
    (sq / n - mean * mean).max(0.0).sqrt()
}

/// Where the patch at (x0, y0) in `a` went in `b`, or `None` when the match is unreliable.
fn match_patch(a: &[u8], b: &[u8], x0: usize, y0: usize) -> Option<(f64, f64)> {
    let (mut best, mut bdx, mut bdy) = (f64::MAX, 0i32, 0i32);
    for dy in -SEARCH_RADIUS..=SEARCH_RADIUS {
        for dx in -SEARCH_RADIUS..=SEARCH_RADIUS {
            let sx = x0 as i32 + dx;
            let sy = y0 as i32 + dy;
            if sx < 0 || sy < 0 || sx as usize + PATCH > ANALYSIS_W || sy as usize + PATCH > ANALYSIS_H {
                continue;
            }
            let mut cost = 0u64;
            for y in 0..PATCH {
                let ar = (y0 + y) * ANALYSIS_W + x0;
                let br = (sy as usize + y) * ANALYSIS_W + sx as usize;
                for x in 0..PATCH {
                    cost += (a[ar + x] as i64).abs_diff(b[br + x] as i64);
                }
            }
            let cost = cost as f64;
            if cost < best {
                best = cost;
                bdx = dx;
                bdy = dy;
            }
        }
    }
    // A match pinned to the search edge means the motion outran it: the number is a floor.
    if bdx.abs() == SEARCH_RADIUS || bdy.abs() == SEARCH_RADIUS {
        return None;
    }
    Some((bdx as f64, bdy as f64))
}

/// Least-squares similarity fit through a set of (from, to) point pairs, about the frame centre.
fn fit_similarity(pairs: &[((f64, f64), (f64, f64))]) -> Option<Motion> {
    if pairs.len() < 2 {
        return None;
    }
    let cx = ANALYSIS_W as f64 / 2.0;
    let cy = ANALYSIS_H as f64 / 2.0;
    // x' = a·x − b·y + tx ; y' = b·x + a·y + ty, with x, y taken from the centre.
    let (mut sxx, mut sxu, mut syv, mut sxv, mut syu, mut su, mut sv, mut sx, mut sy) =
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let n = pairs.len() as f64;
    for ((x, y), (u, v)) in pairs {
        let (x, y, u, v) = (x - cx, y - cy, u - cx, v - cy);
        sxx += x * x + y * y;
        sxu += x * u;
        syv += y * v;
        sxv += x * v;
        syu += y * u;
        su += u;
        sv += v;
        sx += x;
        sy += y;
    }
    // Centre the coordinates so translation separates cleanly from the rotation/scale part.
    let (mx, my, mu, mv) = (sx / n, sy / n, su / n, sv / n);
    let sxx_c = sxx - n * (mx * mx + my * my);
    if sxx_c.abs() < 1e-9 {
        return None;
    }
    let a = ((sxu + syv) - n * (mx * mu + my * mv)) / sxx_c;
    let b = ((sxv - syu) - n * (mx * mv - my * mu)) / sxx_c;
    let tx = mu - (a * mx - b * my);
    let ty = mv - (b * mx + a * my);
    let scale = (a * a + b * b).sqrt();
    if !(0.5..=2.0).contains(&scale) {
        return None;
    }
    Some(Motion { dx: tx, dy: ty, rot: b.atan2(a), scale })
}

fn apply(m: &Motion, x: f64, y: f64) -> (f64, f64) {
    let cx = ANALYSIS_W as f64 / 2.0;
    let cy = ANALYSIS_H as f64 / 2.0;
    let (x, y) = (x - cx, y - cy);
    let (c, s) = (m.rot.cos() * m.scale, m.rot.sin() * m.scale);
    (c * x - s * y + m.dx + cx, s * x + c * y + m.dy + cy)
}

/// The camera's movement between two frames, or `None` when too little could be matched.
pub fn motion_between(a: &[u8], b: &[u8]) -> Option<Motion> {
    let mut pairs: Vec<((f64, f64), (f64, f64))> = Vec::new();
    let mut y0 = PATCH / 2;
    while y0 + PATCH + PATCH / 2 <= ANALYSIS_H {
        let mut x0 = PATCH / 2;
        while x0 + PATCH + PATCH / 2 <= ANALYSIS_W {
            if contrast(a, x0, y0) >= MIN_CONTRAST
                && let Some((dx, dy)) = match_patch(a, b, x0, y0)
            {
                let (px, py) = ((x0 + PATCH / 2) as f64, (y0 + PATCH / 2) as f64);
                pairs.push(((px, py), (px + dx, py + dy)));
            }
            x0 += PATCH;
        }
        y0 += PATCH;
    }
    // The camera moves every patch the same way; a person moves only theirs. Take the median
    // motion as the camera's and drop what disagrees with it before fitting anything — a
    // least-squares fit on the raw set would be pulled toward whoever fills the most patches.
    if pairs.len() < MIN_INLIERS {
        return None;
    }
    let median_of = |vals: &mut Vec<f64>| {
        vals.sort_by(|a, b| a.total_cmp(b));
        vals[vals.len() / 2]
    };
    let mdx = median_of(&mut pairs.iter().map(|(f, t)| t.0 - f.0).collect());
    let mdy = median_of(&mut pairs.iter().map(|(f, t)| t.1 - f.1).collect());
    pairs.retain(|(f, t)| ((t.0 - f.0) - mdx).abs() <= CONSENSUS_PX && ((t.1 - f.1) - mdy).abs() <= CONSENSUS_PX);
    if pairs.len() < MIN_INLIERS {
        return None;
    }
    // Then fit, throw out what disagrees with the fit, fit again.
    let mut fit = fit_similarity(&pairs)?;
    for _ in 0..2 {
        let mut residuals: Vec<f64> = pairs
            .iter()
            .map(|(from, to)| {
                let (px, py) = apply(&fit, from.0, from.1);
                ((px - to.0).powi(2) + (py - to.1).powi(2)).sqrt()
            })
            .collect();
        let mut sorted = residuals.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let median = sorted[sorted.len() / 2];
        let cutoff = (median * 2.5).max(1.0);
        let kept: Vec<_> =
            pairs.iter().zip(residuals.drain(..)).filter(|(_, r)| *r <= cutoff).map(|(p, _)| *p).collect();
        if kept.len() < MIN_INLIERS {
            return None;
        }
        pairs = kept;
        fit = fit_similarity(&pairs)?;
    }
    Some(fit)
}

/// Shake per frame from a run of frame-to-frame motions: the second difference of the camera
/// path — how much its velocity changes — as a percentage of the frame width, with rotation
/// counted by how far it moves the frame edge.
///
/// Not a smoothed-path residual. That scores the *start* of a pan: the path turns a corner, the
/// smoothing rounds it, and the gap between them reads as shake on a tripod. Velocity change is a
/// single spike at that corner and zero along the pan, while under shake it fires every frame.
/// The window statistic is the median, which ignores the spike and keeps the tremor.
pub fn shake_per_frame(motions: &[Option<Motion>]) -> Vec<f64> {
    let cut = CUT_FRACTION * ANALYSIS_W as f64;
    let half_w = ANALYSIS_W as f64 / 2.0;
    // Velocity per frame; an unmeasured step or a cut contributes no movement.
    let v: Vec<(f64, f64, f64)> = motions
        .iter()
        .map(|m| match m {
            Some(m) if m.dx.abs() <= cut && m.dy.abs() <= cut => (m.dx, m.dy, m.rot * half_w),
            _ => (0.0, 0.0, 0.0),
        })
        .collect();
    let mut out = Vec::with_capacity(v.len());
    for i in 0..v.len() {
        let prev = if i == 0 { v[0] } else { v[i - 1] };
        let (ax, ay, ar) = (v[i].0 - prev.0, v[i].1 - prev.1, v[i].2 - prev.2);
        out.push((ax * ax + ay * ay + ar * ar).sqrt() / ANALYSIS_W as f64 * 100.0);
    }
    out
}

/// Middle value: what most frames in a window do, not what the worst one did.
fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn rms(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    (values.iter().map(|v| v * v).sum::<f64>() / values.len() as f64).sqrt()
}

/// Decode the video as tiny greyscale frames and measure the camera's move between each pair.
async fn decode_motions(ffmpeg: &Path, video: &Path, hwaccel: Option<&str>) -> Result<Vec<Option<Motion>>, Error> {
    let mut cmd = crate::proc::command(ffmpeg);
    cmd.args(["-v", "error"]);
    if let Some(h) = hwaccel {
        cmd.args(["-hwaccel", h]);
    }
    let mut child = cmd
        .arg("-i")
        .arg(video)
        .args([
            "-vf",
            &format!("fps={ANALYSIS_FPS},scale={ANALYSIS_W}:{ANALYSIS_H},format=gray"),
            "-f",
            "rawvideo",
            "-",
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| Error::Vision(format!("ffmpeg: {e}")))?;
    let mut stdout = child.stdout.take().ok_or_else(|| Error::Vision("ffmpeg: no stdout".into()))?;

    let size = ANALYSIS_W * ANALYSIS_H;
    let mut prev: Option<Vec<u8>> = None;
    let mut cur = vec![0u8; size];
    let mut motions: Vec<Option<Motion>> = Vec::new();
    loop {
        match stdout.read_exact(&mut cur).await {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(Error::Vision(format!("ffmpeg read: {e}"))),
        }
        if let Some(p) = &prev {
            motions.push(motion_between(p, &cur));
        }
        prev = Some(std::mem::replace(&mut cur, vec![0u8; size]));
    }
    let _ = child.wait().await;
    Ok(motions)
}

/// Measure a whole video: one decode at low resolution, the camera path over all of it, and the
/// shake bucketed into windows of `window_s`. `stride_s` is accepted for configuration
/// compatibility; coverage is contiguous, which the timestamps the editor gets rely on.
pub async fn measure(
    ffmpeg: &Path,
    video: &Path,
    duration_s: f64,
    window_s: f64,
    _stride_s: f64,
) -> Result<Vec<Window>, Error> {
    if duration_s <= 0.0 || window_s <= 0.0 {
        return Ok(Vec::new());
    }
    // Decoding 4K is the whole cost; the GPU does it eight times faster than the CPU when it can.
    // Try it first and fall back — a machine without a usable decoder just takes longer.
    let mut motions = Vec::new();
    for accel in [Some("cuda"), Some("vaapi"), None] {
        motions = decode_motions(ffmpeg, video, accel).await.unwrap_or_default();
        if !motions.is_empty() {
            break;
        }
    }

    let shake = shake_per_frame(&motions);
    let per_window = (window_s * ANALYSIS_FPS as f64).round().max(1.0) as usize;
    let mut out = Vec::new();
    for (i, chunk) in shake.chunks(per_window).enumerate() {
        let start_s = i as f64 * window_s;
        out.push(Window { start_s, end_s: (start_s + window_s).min(duration_s), jerk: median(chunk) });
    }
    Ok(out)
}

/// The worst shake over the windows a clip touches, or `None` when nothing was measured there.
pub fn jerk_in_range(windows: &[Window], in_s: f64, out_s: f64) -> Option<f64> {
    windows.iter().filter(|w| w.end_s > in_s && w.start_s < out_s).map(|w| w.jerk).max_by(|a, b| a.total_cmp(b))
}

/// The stretches inside `in_s..out_s` where the camera is shakier than `max_jerk`, merged where
/// they run together.
///
/// A whole-video verdict is the wrong shape: a ten-minute clip is rarely shaky throughout, and
/// condemning all of it throws away the steady minutes. The editor picks ranges, so it is told
/// which ranges to avoid.
pub fn shaky_spans(windows: &[Window], in_s: f64, out_s: f64, max_jerk: f64) -> Vec<(f64, f64)> {
    if max_jerk <= 0.0 {
        return Vec::new();
    }
    let mut spans: Vec<(f64, f64)> = Vec::new();
    for w in windows.iter().filter(|w| w.end_s > in_s && w.start_s < out_s && w.jerk > max_jerk) {
        let (start, end) = (w.start_s.max(in_s), w.end_s.min(out_s));
        match spans.last_mut() {
            // Touching or overlapping windows describe one shaky stretch, not several.
            Some(last) if start <= last.1 + 0.05 => last.1 = last.1.max(end),
            _ => spans.push((start, end)),
        }
    }
    spans
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A textured scene, sampled through a similarity transform so tests can move a camera.
    fn render(m: &Motion) -> Vec<u8> {
        let cx = ANALYSIS_W as f64 / 2.0;
        let cy = ANALYSIS_H as f64 / 2.0;
        let (c, s) = (m.rot.cos() / m.scale, m.rot.sin() / m.scale);
        let mut out = Vec::with_capacity(ANALYSIS_W * ANALYSIS_H);
        for y in 0..ANALYSIS_H {
            for x in 0..ANALYSIS_W {
                // Inverse map the output pixel back into the scene.
                let (dx, dy) = (x as f64 - cx - m.dx, y as f64 - cy - m.dy);
                let sx = c * dx + s * dy + cx;
                let sy = -s * dx + c * dy + cy;
                // The scene: overlapping blobs and stripes, so every patch has contrast.
                let v = 128.0
                    + 60.0 * ((sx * 0.35).sin() * (sy * 0.27).cos())
                    + 40.0 * (((sx * 0.9 + sy * 0.6) * 0.5).sin())
                    + 20.0 * ((sx * 1.7).cos());
                out.push(v.clamp(0.0, 255.0) as u8);
            }
        }
        out
    }

    #[test]
    fn recovers_translation_rotation_and_scale() {
        let a = render(&Motion { dx: 0.0, dy: 0.0, rot: 0.0, scale: 1.0 });
        let truth = Motion { dx: 3.0, dy: -2.0, rot: 0.02, scale: 1.03 };
        let b = render(&truth);
        let m = motion_between(&a, &b).expect("a textured scene must be measurable");
        assert!((m.dx - truth.dx).abs() < 1.0, "dx {m:?}");
        assert!((m.dy - truth.dy).abs() < 1.0, "dy {m:?}");
        assert!((m.rot - truth.rot).abs() < 0.01, "rot {m:?}");
        assert!((m.scale - truth.scale).abs() < 0.02, "scale {m:?}");
    }

    #[test]
    fn a_locked_off_shot_scores_zero() {
        let still = render(&Motion { scale: 1.0, ..Default::default() });
        let motions: Vec<Option<Motion>> = (0..20).map(|_| motion_between(&still, &still)).collect();
        assert!(median(&shake_per_frame(&motions)) < 0.05);
    }

    /// The distinction the whole measure rests on: a smooth pan and a push-in move a great deal
    /// and are not shake; the same amount of movement reversing every frame is.
    #[test]
    fn smooth_motion_is_not_shake_but_jitter_is() {
        let frames_of = |f: &dyn Fn(usize) -> Motion| -> Vec<Vec<u8>> { (0..30).map(|i| render(&f(i))).collect() };
        let motions_of = |frames: &[Vec<u8>]| -> Vec<Option<Motion>> {
            frames.windows(2).map(|p| motion_between(&p[0], &p[1])).collect()
        };

        let pan = frames_of(&|i| Motion { dx: 1.5 * i as f64, dy: 0.0, rot: 0.0, scale: 1.0 });
        let push = frames_of(&|i| Motion { dx: 0.0, dy: 0.0, rot: 0.0, scale: 1.0 + 0.004 * i as f64 });
        let jitter = frames_of(&|i| Motion {
            dx: if i % 2 == 0 { 0.0 } else { 1.5 },
            dy: if i % 3 == 0 { 1.0 } else { 0.0 },
            rot: 0.0,
            scale: 1.0,
        });

        let pan_s = median(&shake_per_frame(&motions_of(&pan)));
        let push_s = median(&shake_per_frame(&motions_of(&push)));
        let jitter_s = median(&shake_per_frame(&motions_of(&jitter)));
        assert!(pan_s < jitter_s / 3.0, "pan {pan_s} vs jitter {jitter_s}");
        assert!(push_s < jitter_s / 3.0, "push-in {push_s} vs jitter {jitter_s}");
    }

    #[test]
    fn shaky_stretches_are_reported_with_their_timestamps() {
        let w = |start_s: f64, end_s: f64, jerk: f64| Window { start_s, end_s, jerk };
        let windows = [w(0.0, 4.0, 1.2), w(4.0, 8.0, 1.4), w(8.0, 12.0, 0.1), w(12.0, 16.0, 0.9)];
        assert_eq!(shaky_spans(&windows, 0.0, 20.0, 0.5), vec![(0.0, 8.0), (12.0, 16.0)]);
        assert_eq!(shaky_spans(&windows, 5.0, 10.0, 0.5), vec![(5.0, 8.0)]);
        assert!(shaky_spans(&windows, 8.0, 12.0, 0.5).is_empty());
        assert!(shaky_spans(&windows, 0.0, 20.0, 0.0).is_empty());
    }

    #[test]
    fn range_lookup_takes_the_worst_window_it_touches() {
        let w = |start_s: f64, end_s: f64, jerk: f64| Window { start_s, end_s, jerk };
        let windows = [w(0.0, 4.0, 0.2), w(15.0, 19.0, 1.8), w(30.0, 34.0, 0.3)];
        assert_eq!(jerk_in_range(&windows, 0.0, 5.0), Some(0.2));
        assert_eq!(jerk_in_range(&windows, 3.0, 18.0), Some(1.8));
        assert_eq!(jerk_in_range(&windows, 40.0, 50.0), None);
    }
}
