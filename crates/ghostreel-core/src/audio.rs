//! Which audio track to use, and who is close to the microphone.
//!
//! Two facts about a video's sound that nothing else in the index knows, and that a script is
//! poor without:
//!
//! - A field recording often carries several tracks — a camera mic, one or two lavs, and tracks
//!   left silent. Taking the first one is a guess: it can be the roomy camera mic, or nothing at
//!   all. The one with usable speech is measured instead.
//! - In an interview the subject is on a lav and the interviewer is off-mic across the room.
//!   Their speech is measurably quieter — 12 dB apart on this project's footage — so the slate
//!   and the questions can be told from the answers without reading a word.

use std::path::Path;

use crate::Error;

/// Below this mean level a track is silence rather than a quiet recording.
const SILENT_DBFS: f64 = -70.0;
/// A segment this far under the video's own speech level is someone away from the microphone.
pub const OFF_MIC_MARGIN_DB: f64 = 8.0;

/// Mean level of a stretch of one audio track, in dBFS. `None` when ffmpeg reported nothing.
pub async fn mean_level(ffmpeg: &Path, video: &Path, stream: u32, start_s: f64, seconds: f64) -> Option<f64> {
    let out = crate::proc::command(ffmpeg)
        // No -v error here: volumedetect reports its measurement at info level, and quieting
        // ffmpeg hides the very number being asked for.
        .args(["-nostats", "-hide_banner", "-ss"])
        .arg(format!("{start_s}"))
        .arg("-t")
        .arg(format!("{seconds}"))
        .arg("-i")
        .arg(video)
        .args(["-map", &format!("0:a:{stream}"), "-filter:a", "volumedetect", "-f", "null", "-"])
        .kill_on_drop(true)
        .output()
        .await
        .ok()?;
    // volumedetect reports on stderr: "[Parsed_volumedetect_0 @ …] mean_volume: -23.0 dB"
    let text = String::from_utf8_lossy(&out.stderr);
    let line = text.lines().find(|l| l.contains("mean_volume:"))?;
    line.split("mean_volume:").nth(1)?.trim().trim_end_matches(" dB").trim().parse().ok()
}

/// The audio track to use: the loudest one that is not silence. `None` when the video has no
/// usable audio at all.
///
/// Sampled from the middle of the video, where someone is most likely to be talking; a track that
/// happens to be quiet at the top of the file is not thereby the wrong one.
pub async fn pick_track(ffmpeg: &Path, video: &Path, tracks: u32, duration_s: f64) -> Option<u32> {
    if tracks == 0 {
        return None;
    }
    let seconds = 20.0_f64.min(duration_s.max(1.0));
    let start = ((duration_s - seconds) / 2.0).max(0.0);
    let mut best: Option<(u32, f64)> = None;
    for stream in 0..tracks {
        let Some(level) = mean_level(ffmpeg, video, stream, start, seconds).await else { continue };
        if level <= SILENT_DBFS {
            continue;
        }
        if best.is_none_or(|(_, b)| level > b) {
            best = Some((stream, level));
        }
    }
    best.map(|(s, _)| s)
}

/// Mean level of each transcript segment, in the order given.
pub async fn segment_levels(ffmpeg: &Path, video: &Path, stream: u32, segments: &[(f64, f64)]) -> Vec<Option<f64>> {
    let mut out = Vec::with_capacity(segments.len());
    for (start_s, end_s) in segments {
        let seconds = end_s - start_s;
        // Too short to measure: a word or two gives a level dominated by its attack.
        out.push(if seconds >= 0.4 { mean_level(ffmpeg, video, stream, *start_s, seconds).await } else { None });
    }
    out
}

/// Whether each segment is someone away from the microphone, judged against the video's own
/// speech. Returns `None` for a segment where nothing could be measured.
///
/// The reference is the median of the measured levels, not the loudest: an interview is mostly
/// the subject, so the middle of the distribution is the person on the lav.
pub fn off_mic_flags(levels: &[Option<f64>], margin_db: f64) -> Vec<Option<bool>> {
    let mut measured: Vec<f64> = levels.iter().flatten().copied().collect();
    if measured.len() < 3 {
        return levels.iter().map(|_| None).collect();
    }
    measured.sort_by(f64::total_cmp);
    let reference = measured[measured.len() / 2];
    levels.iter().map(|l| l.map(|v| v < reference - margin_db)).collect()
}

/// How many audio tracks a video has.
pub async fn track_count(ffprobe: &Path, video: &Path) -> Result<u32, Error> {
    let out = crate::proc::command(ffprobe)
        .args(["-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0"])
        .arg(video)
        .kill_on_drop(true)
        .output()
        .await
        .map_err(|e| Error::Vision(format!("ffprobe: {e}")))?;
    Ok(String::from_utf8_lossy(&out.stdout).lines().filter(|l| !l.trim().is_empty()).count() as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The interviewer's levels from this project's footage, against the subject's.
    #[test]
    fn the_quiet_speaker_is_the_one_off_mic() {
        let levels = vec![
            Some(-33.6), // interviewer
            Some(-36.1), // interviewer
            Some(-20.1), // subject
            Some(-24.3),
            Some(-24.2),
            Some(-24.2),
            Some(-36.2), // interviewer
            Some(-20.1),
            Some(-23.6),
        ];
        let flags = off_mic_flags(&levels, OFF_MIC_MARGIN_DB);
        assert_eq!(
            flags,
            vec![
                Some(true),
                Some(true),
                Some(false),
                Some(false),
                Some(false),
                Some(false),
                Some(true),
                Some(false),
                Some(false)
            ]
        );
    }

    /// One person talking throughout: nobody is off-mic, however their level varies.
    #[test]
    fn a_single_speaker_has_nobody_off_mic() {
        let levels = vec![Some(-22.0), Some(-24.0), Some(-21.0), Some(-25.0), Some(-23.0)];
        assert!(off_mic_flags(&levels, OFF_MIC_MARGIN_DB).iter().all(|f| f == &Some(false)));
    }

    /// Too little measured to have an opinion: say so rather than guess.
    #[test]
    fn too_few_measurements_decide_nothing() {
        assert_eq!(off_mic_flags(&[Some(-20.0), None], OFF_MIC_MARGIN_DB), vec![None, None]);
    }
}
