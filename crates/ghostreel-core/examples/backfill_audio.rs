//! Measure the audio track and off-mic segments for videos indexed before that existed.
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let paths = ghostreel_core::paths::Paths::resolve()?;
    let db = ghostreel_core::db::Db::open(&paths.db_file())?;
    let ffmpeg = ghostreel_core::doctor::locate("ffmpeg").ok_or("no ffmpeg")?;
    let ffprobe = ghostreel_core::doctor::locate("ffprobe").ok_or("no ffprobe")?;
    let only: Vec<i64> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();

    let mut st = db.conn.prepare(
        "SELECT v.id, v.duration_s, vf.path FROM videos v JOIN video_files vf ON vf.video_id = v.id
         WHERE v.duration_s IS NOT NULL GROUP BY v.id ORDER BY v.id",
    )?;
    let rows: Vec<(i64, f64, String)> = st
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
        .filter_map(Result::ok)
        .filter(|(id, _, _)| only.is_empty() || only.contains(id))
        .collect();

    for (id, duration, path) in rows {
        let p = std::path::Path::new(&path);
        let tracks = ghostreel_core::audio::track_count(&ffprobe, p).await.unwrap_or(0);
        if tracks == 0 {
            continue;
        }
        let track = ghostreel_core::audio::pick_track(&ffmpeg, p, tracks, duration).await;
        db.set_audio_track(id, track)?;
        let mut off = 0usize;
        let mut total = 0usize;
        if let Some(track) = track {
            let mut q = db
                .conn
                .prepare("SELECT start_s, end_s FROM transcript_segments WHERE video_id = ?1 ORDER BY start_s")?;
            let spans: Vec<(f64, f64)> =
                q.query_map([id], |r| Ok((r.get(0)?, r.get(1)?)))?.filter_map(Result::ok).collect();
            if !spans.is_empty() {
                let levels = ghostreel_core::audio::segment_levels(&ffmpeg, p, track, &spans).await;
                let flags = ghostreel_core::audio::off_mic_flags(&levels, ghostreel_core::audio::OFF_MIC_MARGIN_DB);
                off = flags.iter().filter(|f| **f == Some(true)).count();
                total = flags.len();
                let rows: Vec<(f64, Option<bool>)> = spans.iter().map(|(a, _)| *a).zip(flags).collect();
                db.set_off_mic(id, &rows)?;
            }
        }
        let name = p.file_name().unwrap().to_string_lossy();
        println!("#{id:<4} {name:32} tracks={tracks} using={track:?} off_mic={off}/{total}");
    }
    Ok(())
}
