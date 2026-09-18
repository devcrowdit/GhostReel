//! Per-frame camera motion for the first seconds of a file, to see what the estimator sees.
use std::process::Stdio;
#[tokio::main]
async fn main() {
    let path = std::env::args().nth(1).expect("video path");
    let secs = std::env::args().nth(2).and_then(|s| s.parse::<f64>().ok()).unwrap_or(3.0);
    let start = std::env::args().nth(3).and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);
    let (w, h) = (ghostreel_core::steadiness::ANALYSIS_W, ghostreel_core::steadiness::ANALYSIS_H);
    let out = tokio::process::Command::new("/usr/bin/ffmpeg")
        .args([
            "-v",
            "error",
            "-ss",
            &start.to_string(),
            "-t",
            &secs.to_string(),
            "-i",
            &path,
            "-vf",
            &format!("fps={},scale={w}:{h},format=gray", ghostreel_core::steadiness::ANALYSIS_FPS),
            "-f",
            "rawvideo",
            "-",
        ])
        .stderr(Stdio::null())
        .output()
        .await
        .unwrap();
    let frames: Vec<&[u8]> = out.stdout.chunks_exact(w * h).collect();
    let motions: Vec<Option<ghostreel_core::steadiness::Motion>> =
        frames.windows(2).map(|p| ghostreel_core::steadiness::motion_between(p[0], p[1])).collect();
    for (i, m) in motions.iter().enumerate() {
        match m {
            Some(m) => println!("{i:3} dx={:+.2} dy={:+.2} rot={:+.4} scale={:.3}", m.dx, m.dy, m.rot, m.scale),
            None => println!("{i:3} unmeasured"),
        }
    }
    let shake = ghostreel_core::steadiness::shake_per_frame(&motions);
    let rms = (shake.iter().map(|v| v * v).sum::<f64>() / shake.len().max(1) as f64).sqrt();
    println!("frames={} rms_shake={rms:.3}", frames.len());
}
