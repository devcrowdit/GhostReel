//! Retrieval documents per video (plan §4): what search matches against.
//!
//! - `moment`: a described frame + what is said around it — the main unit ("person unboxing a board").
//! - `transcript`: speech in ~30 s windows (5 s overlap) — finds things only said, never shown.
//! - `frame`: on-screen text alone — exact matches for titles, code, product names.

use serde::Serialize;

use crate::vision::FrameDescription;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Chunk {
    pub kind: &'static str,
    pub start_s: f64,
    pub end_s: f64,
    pub text: String,
    pub frame_id: Option<i64>,
}

pub struct Seg<'a> {
    pub start: f64,
    pub end: f64,
    pub text: &'a str,
}

pub struct DescribedFrame<'a> {
    pub id: i64,
    pub t_s: f64,
    pub description: &'a FrameDescription,
}

const WINDOW_S: f64 = 30.0;
const OVERLAP_S: f64 = 5.0;
const MOMENT_SPEECH_S: f64 = 15.0;

/// Transcript windows of ~30 s that overlap by ~5 s, cut on segment boundaries.
pub fn transcript_windows(segs: &[Seg]) -> Vec<Chunk> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < segs.len() {
        let start = segs[i].start;
        let mut j = i;
        let mut text = String::new();
        while j < segs.len() && (j == i || segs[j].end - start <= WINDOW_S) {
            if !text.is_empty() {
                text.push(' ');
            }
            text.push_str(segs[j].text.trim());
            j += 1;
        }
        let end = segs[j - 1].end;
        out.push(Chunk { kind: "transcript", start_s: start, end_s: end, text, frame_id: None });
        if j >= segs.len() {
            break;
        }
        // Next window starts at the first segment that begins within the overlap.
        let next = (i + 1..j).find(|&k| segs[k].start >= end - OVERLAP_S).unwrap_or(j);
        i = next.max(i + 1);
    }
    out
}

fn speech_between(segs: &[Seg], from: f64, to: f64) -> String {
    segs.iter().filter(|s| s.end >= from && s.start <= to).map(|s| s.text.trim()).collect::<Vec<_>>().join(" ")
}

/// Moments and on-screen-text chunks for described frames. A moment spans from its frame to the next
/// frame (at most 30 s), which is where its picture stays relevant.
pub fn frame_chunks(frames: &[DescribedFrame], segs: &[Seg], duration_s: f64) -> Vec<Chunk> {
    let mut out = Vec::new();
    for (i, f) in frames.iter().enumerate() {
        let next = frames.get(i + 1).map(|n| n.t_s).unwrap_or(duration_s.max(f.t_s));
        let end = next.min(f.t_s + WINDOW_S).max(f.t_s);
        let mut text = f.description.search_text();
        let speech = speech_between(segs, f.t_s - MOMENT_SPEECH_S, f.t_s + MOMENT_SPEECH_S);
        if !speech.is_empty() {
            text.push_str(&format!("\nSpeech: {speech}"));
        }
        out.push(Chunk { kind: "moment", start_s: f.t_s, end_s: end, text, frame_id: Some(f.id) });
        if !f.description.visible_text.is_empty() {
            out.push(Chunk {
                kind: "frame",
                start_s: f.t_s,
                end_s: end,
                text: f.description.visible_text.join("\n"),
                frame_id: Some(f.id),
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(start: f64, end: f64, text: &str) -> Seg<'_> {
        Seg { start, end, text }
    }

    #[test]
    fn windows_overlap_and_cover_everything() {
        let texts: Vec<String> = (0..20).map(|i| format!("s{i}")).collect();
        let segs: Vec<Seg> = (0..20).map(|i| seg(i as f64 * 6.0, i as f64 * 6.0 + 5.0, &texts[i])).collect();
        let w = transcript_windows(&segs);
        assert!(w.len() >= 4, "{}", w.len());
        assert!(w.iter().all(|c| c.end_s - c.start_s <= 30.0 + 1e-9));
        // Every segment appears in some window; consecutive windows share text (overlap).
        for t in &texts {
            assert!(w.iter().any(|c| c.text.split(' ').any(|x| x == t)), "{t} missing");
        }
        assert!(w.windows(2).all(|p| p[1].start_s < p[0].end_s), "windows overlap");
        assert_eq!(w.last().unwrap().end_s, 19.0 * 6.0 + 5.0);
    }

    #[test]
    fn long_single_segment_and_empty() {
        assert!(transcript_windows(&[]).is_empty());
        let w = transcript_windows(&[seg(0.0, 90.0, "one very long segment")]);
        assert_eq!(w.len(), 1);
    }

    #[test]
    fn moments_include_speech_and_screen_text() {
        let d1 = FrameDescription {
            description: "Hands unbox a small green board".into(),
            visible_text: vec!["CM5".into()],
            ..Default::default()
        };
        let d2 = FrameDescription { description: "A laptop terminal".into(), ..Default::default() };
        let frames = [
            DescribedFrame { id: 1, t_s: 2.0, description: &d1 },
            DescribedFrame { id: 2, t_s: 80.0, description: &d2 },
        ];
        let segs = [seg(0.0, 5.0, "we unbox the compute module"), seg(70.0, 75.0, "now flash it")];
        let c = frame_chunks(&frames, &segs, 100.0);
        assert_eq!(c.iter().filter(|x| x.kind == "moment").count(), 2);
        assert_eq!(c.iter().filter(|x| x.kind == "frame").count(), 1, "only frames with text");
        let m1 = &c[0];
        assert!(m1.text.contains("unbox a small green board") && m1.text.contains("Speech: we unbox"));
        assert_eq!((m1.start_s, m1.end_s), (2.0, 32.0), "capped at 30 s");
        let m2 = c.iter().find(|x| x.frame_id == Some(2)).unwrap();
        assert!(m2.text.contains("now flash it"));
        assert_eq!(m2.end_s, 100.0);
    }
}
