//! Project-scoped script chat agent (plan §4a, M8c).
//!
//! Grounded script writing through iterative tool calling over local footage
//! and transcripts. Supports server backends (OpenAI-compatible) and local
//! `ghostreel-llm` helper backends.

use std::path::{Path, PathBuf};
use std::time::Duration;

use rusqlite::params;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::Error;
use crate::db::Db;
use crate::embed::Embedder;
use crate::projects::{Project, now};
use crate::script::{Issue, IssueSeverity, Script, ScriptClip, save_version, snap_to_segments, validate};
use crate::search::{SearchOptions, query_vector, search_with_vector};

/// Events emitted during agent execution.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ChatEvent {
    ToolStarted { tool: String, args: Value },
    ToolFinished { tool: String, summary: String },
    Drafting,
    Validating,
}

/// Record of an executed tool call in a chat turn.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCallRecord {
    pub tool: String,
    pub args: Value,
    pub summary: String,
}

/// A stored chat session.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatSession {
    pub id: i64,
    pub project_id: i64,
    pub title: String,
    pub created_at: i64,
    pub updated_at: i64,
}

/// A message in a chat session.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatMessage {
    pub id: i64,
    pub session_id: i64,
    pub role: String,
    pub content: String,
    pub tool_calls: Option<Vec<ToolCallRecord>>,
    pub script_id: Option<i64>,
    pub created_at: i64,
}

/// Result of a single agent turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnResult {
    pub session_id: i64,
    pub reply: String,
    pub script_id: Option<i64>,
    pub script: Option<Script>,
    pub issues: Vec<Issue>,
    pub tool_calls: Vec<ToolCallRecord>,
}

/// Backends for script chat.
pub enum ChatBackend {
    Server {
        url: String,
        model: String,
        api_key: String,
    },
    Local {
        helper: Box<crate::vision::LocalLlm>,
        /// The helper's context window: tool results are trimmed to a fraction of it.
        ctx_tokens: u32,
    },
}

impl ChatBackend {
    pub async fn from_runtime(rt: &crate::runtime::Runtime) -> Result<Self, Error> {
        Self::from_vision_setup(&rt.vision).await
    }

    /// Chat uses the vision model (Bonsai): the vision server, or the local helper with the model
    /// and mmproj (downloaded once when missing, like the describe stage).
    pub async fn from_vision_setup(setup: &crate::runtime::VisionSetup) -> Result<Self, Error> {
        match setup {
            crate::runtime::VisionSetup::Server(s) => {
                Ok(ChatBackend::Server { url: s.url.clone(), model: s.model.clone(), api_key: s.api_key.clone() })
            }
            crate::runtime::VisionSetup::Local { helper, models_dir, model, mmproj, found, runtime } => {
                let mut paths = Vec::with_capacity(2);
                for spec in [model, mmproj] {
                    if let Some(p) = found.iter().find(|p| p.file_name().is_some_and(|n| n == spec.file_name.as_str()))
                    {
                        paths.push(p.clone());
                    } else {
                        paths.push(crate::models::download(spec, models_dir, |_, _| {}).await?);
                    }
                }
                let vision = Some((paths[0].clone(), paths[1].clone()));
                let models = crate::vision::LocalModels {
                    helper: helper.clone(),
                    vision,
                    embed: None,
                    cpu: false,
                    runtime: runtime.clone(),
                };
                let llm = crate::vision::LocalLlm::start(&models).await?;
                Ok(ChatBackend::Local { helper: Box::new(llm), ctx_tokens: runtime.ctx_tokens })
            }
            crate::runtime::VisionSetup::Unavailable(why) => {
                Err(Error::Vision(format!("vision/chat model unavailable: {why}")))
            }
        }
    }
}

/// Context owning all resources needed for a chat turn.
pub struct ChatContext {
    pub db: Db,
    pub data_dir: PathBuf,
    pub backend: ChatBackend,
    pub embedder: Option<Embedder>,
    /// Custom editing instructions (Settings); `None`/empty = [`DEFAULT_EDITOR_PROMPT`].
    pub system_prompt: Option<String>,
}

/// Slack around grounded ranges (search moments are approximate).
const GROUNDING_SLACK_S: f64 = 5.0;
/// Clips longer than this are pacing mistakes (the model pasted a whole tool range). Generous, so
/// people talking can stay on screen for whole sentences.
const MAX_CLIP_S: f64 = 30.0;
/// Over-long clips are trimmed to this length (keeping their start).
const TRIMMED_CLIP_S: f64 = 20.0;
/// Shorter clips flash by before viewers can see or read them.
const MIN_CLIP_S: f64 = 3.0;
/// Total duration further than this fraction from the target triggers one redraft.
const TARGET_TOLERANCE: f64 = TARGET_OVERSHOOT - 1.0;

/// Pacing problems the model can fix in a redraft: clips too long or too short, and (when
/// `enforce_target`) a total far from the target.
pub fn pacing_issues(script: &Script, enforce_target: bool) -> Vec<Issue> {
    let mut issues = Vec::new();
    for beat in &script.beats {
        for (i, c) in beat.clips.iter().enumerate() {
            let len = c.out_s - c.in_s;
            if len > MAX_CLIP_S {
                issues.push(Issue {
                    severity: IssueSeverity::Warning,
                    beat_id: Some(beat.id.clone()),
                    clip_index: Some(i),
                    message: format!(
                        "clip is {len:.1} s long (video #{} {:.1}–{:.1}); use a shorter excerpt",
                        c.video_id, c.in_s, c.out_s
                    ),
                });
            } else if len < MIN_CLIP_S {
                issues.push(Issue {
                    severity: IssueSeverity::Warning,
                    beat_id: Some(beat.id.clone()),
                    clip_index: Some(i),
                    message: format!(
                        "clip is only {len:.1} s (video #{} {:.1}–{:.1}); hold each shot at least {MIN_CLIP_S:.0} s so viewers can see and read it",
                        c.video_id, c.in_s, c.out_s
                    ),
                });
            }
        }
    }
    if let Some(target) = script.target_duration_s.filter(|t| *t > 0.0 && enforce_target) {
        let total = script.total_duration_s();
        if (total - target).abs() > target * TARGET_TOLERANCE {
            issues.push(Issue {
                severity: IssueSeverity::Warning,
                beat_id: None,
                clip_index: None,
                message: format!("total clip duration is {total:.1} s but the target is {target:.0} s"),
            });
        }
    }
    issues
}

/// Drop ungrounded/foreign clips and trim over-long ones; with `enforce_target`, also squeeze the
/// total toward the target. Revisions don't enforce it: the user's feedback ("slower", "longer")
/// must be able to change the length. Returns the issues describing the changes.
fn enforce_grounding_and_pacing(
    db: &Db,
    project_id: i64,
    s: &mut Script,
    grounding: &Grounding,
    enforce_target: bool,
) -> Vec<Issue> {
    let mut issues = Vec::new();
    for beat in &mut s.beats {
        let mut kept = Vec::with_capacity(beat.clips.len());
        for c in beat.clips.drain(..) {
            if !is_video_in_project(db, project_id, c.video_id) || !grounding.is_grounded(c.video_id, c.in_s, c.out_s) {
                issues.push(Issue {
                    severity: IssueSeverity::Warning,
                    beat_id: Some(beat.id.clone()),
                    clip_index: None,
                    message: format!(
                        "dropped clip not grounded in tool results: video #{} {:.1}–{:.1} s",
                        c.video_id, c.in_s, c.out_s
                    ),
                });
                continue;
            }
            kept.push(c);
        }
        for (i, c) in kept.iter_mut().enumerate() {
            if c.out_s - c.in_s > MAX_CLIP_S {
                issues.push(Issue {
                    severity: IssueSeverity::Info,
                    beat_id: Some(beat.id.clone()),
                    clip_index: Some(i),
                    message: format!(
                        "trimmed {:.1} s clip to {TRIMMED_CLIP_S:.0} s (video #{} from {:.1} s)",
                        c.out_s - c.in_s,
                        c.video_id,
                        c.in_s
                    ),
                });
                c.out_s = c.in_s + TRIMMED_CLIP_S;
            }
        }
        beat.clips = kept;
    }
    s.beats.retain(|b| !b.clips.is_empty());
    clamp_to_duration(db, s);
    let repeats = drop_repeated_footage(s);
    if repeats > 0 {
        issues.push(Issue {
            severity: IssueSeverity::Warning,
            beat_id: None,
            clip_index: None,
            message: format!("dropped {repeats} clip(s) that repeated footage already used earlier"),
        });
    }
    let merged = merge_contiguous_clips(s);
    if merged > 0 {
        issues.push(Issue {
            severity: IssueSeverity::Info,
            beat_id: None,
            clip_index: None,
            message: format!("joined {merged} back-to-back cut(s) of the same shot into continuous clips"),
        });
    }
    // Pad before trimming, so the target is met with the people's pauses already in.
    pad_speech(db, s);
    // Padding can grow two clips of the same video into each other: check again.
    let overlapped = drop_repeated_footage(s);
    if overlapped > 0 {
        issues.push(Issue {
            severity: IssueSeverity::Info,
            beat_id: None,
            clip_index: None,
            message: format!("dropped {overlapped} clip(s) that overlapped another once padded"),
        });
    }
    let before = s.total_duration_s();
    let speaking = |c: &ScriptClip| clip_has_speech(db, c.video_id, c.in_s, c.out_s);
    if enforce_target && fit_speech_to_target(db, s) | trim_to_target_with(s, speaking) {
        issues.push(Issue {
            severity: IssueSeverity::Info,
            beat_id: None,
            clip_index: None,
            message: format!("clips shortened proportionally: {before:.1} s → {:.1} s", s.total_duration_s()),
        });
    }
    issues
}

/// Grounding tracker recording footage ranges inspected by tools.
#[derive(Debug, Clone, Default)]
pub struct Grounding {
    pub ranges: Vec<(i64, f64, f64)>,
}

impl Grounding {
    pub fn add(&mut self, video_id: i64, start_s: f64, end_s: f64) {
        self.ranges.push((video_id, start_s, end_s));
    }

    /// Check if a clip lies inside a grounded range for that video (±5 s slack at both ends).
    /// Containment, not overlap: a 45 s clip touching a 5 s search hit is not grounded.
    pub fn is_grounded(&self, video_id: i64, in_s: f64, out_s: f64) -> bool {
        self.ranges
            .iter()
            .any(|&(vid, s, e)| vid == video_id && in_s >= s - GROUNDING_SLACK_S && out_s <= e + GROUNDING_SLACK_S)
    }

    /// Record grounding ranges from a tool invocation.
    pub fn record_tool_call(&mut self, tool: &str, args: &Value, db: &Db) {
        match tool {
            "get_transcript" => {
                if let (Some(vid), Some(s), Some(e)) = (
                    args.get("video_id").and_then(|v| v.as_i64()),
                    args.get("start_s").and_then(|v| v.as_f64()),
                    args.get("end_s").and_then(|v| v.as_f64()),
                ) {
                    self.add(vid, s, e);
                }
            }
            "get_video" => {
                if let Some(vid) = args.get("video_id").and_then(|v| v.as_i64()) {
                    let dur: Option<f64> =
                        db.conn.query_row("SELECT duration_s FROM videos WHERE id = ?1", [vid], |r| r.get(0)).ok();
                    self.add(vid, 0.0, dur.unwrap_or(0.0));
                }
            }
            _ => {}
        }
    }
}

/// Local constrained action representation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum LocalAction {
    Tool { tool: String, args: Value },
    Final { script: Script },
}

/// JSON Schema for Script v1, friendly to llama.cpp grammar.
pub fn script_json_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "title": { "type": "string" },
            "target_duration_s": { "type": "number" },
            "beats": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "string" },
                        "purpose": { "type": "string" },
                        "narration": { "type": "string" },
                        "on_screen_text": { "type": "string" },
                        "clips": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "video_id": { "type": "integer" },
                                    "in_s": { "type": "number" },
                                    "out_s": { "type": "number" },
                                    "audio": { "type": "string", "enum": ["source", "mute"] },
                                    "why": { "type": "string" }
                                },
                                "required": ["video_id", "in_s", "out_s", "audio"],
                                "additionalProperties": false
                            }
                        },
                        "notes": { "type": "string" }
                    },
                    "required": ["id", "purpose", "narration", "clips"],
                    "additionalProperties": false
                }
            }
        },
        "required": ["title", "beats"],
        "additionalProperties": false
    })
}

/// OpenAI tools schema definition for the 4 tools.
pub fn tools_definition() -> Value {
    json!([
        {
            "type": "function",
            "function": {
                "name": "search_moments",
                "description": "Search indexed footage by keyword or meaning in the project",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": { "type": "string", "description": "Search query terms" },
                        "kind": { "type": "string", "enum": ["moment", "transcript", "frame"], "description": "Chunk kind" },
                        "limit": { "type": "integer", "description": "Max results (1-8, default 6)" }
                    },
                    "required": ["query"],
                    "additionalProperties": false
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_transcript",
                "description": "Get timestamped transcript segments for a video within a time range",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "video_id": { "type": "integer", "description": "Video ID" },
                        "start_s": { "type": "number", "description": "Start timestamp in seconds" },
                        "end_s": { "type": "number", "description": "End timestamp in seconds" }
                    },
                    "required": ["video_id", "start_s", "end_s"],
                    "additionalProperties": false
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_video",
                "description": "Get video metadata and keyframe descriptions (what is visible at each time). Pass start_s/end_s to see every keyframe in a range before choosing a clip.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "video_id": { "type": "integer", "description": "Video ID" },
                        "start_s": { "type": "number", "description": "Optional range start in seconds" },
                        "end_s": { "type": "number", "description": "Optional range end in seconds" }
                    },
                    "required": ["video_id"],
                    "additionalProperties": false
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_videos",
                "description": "List all videos belonging to the project with summaries",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }
            }
        }
    ])
}

/// Local LLM action schema (oneOf tool action or final script action).
pub fn local_action_schema() -> Value {
    json!({
        "type": "object",
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    "action": { "type": "string", "enum": ["tool"] },
                    "tool": { "type": "string", "enum": ["search_moments", "get_transcript", "get_video", "list_videos"] },
                    "args": { "type": "object" }
                },
                "required": ["action", "tool", "args"],
                "additionalProperties": false
            },
            {
                "type": "object",
                "properties": {
                    "action": { "type": "string", "enum": ["final"] },
                    "script": script_json_schema()
                },
                "required": ["action", "script"],
                "additionalProperties": false
            }
        ]
    })
}

/// Local LLM final action schema only.
pub fn local_final_action_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "action": { "type": "string", "enum": ["final"] },
            "script": script_json_schema()
        },
        "required": ["action", "script"],
        "additionalProperties": false
    })
}

/// The project's reference edits (finished videos made by a person) as a study guide for the model:
/// length, keyframe timeline and transcript. Empty when there are none.
pub fn reference_edits_text(db: &Db, project_id: i64, max_chars: usize) -> String {
    let Ok(refs) = db.excluded_videos(project_id) else { return String::new() };
    let refs: Vec<_> = refs.into_iter().filter(|(_, role, _)| role == "reference").collect();
    if refs.is_empty() {
        return String::new();
    }
    let mut out = String::from(
        "\nREFERENCE EDITS\nFinished videos a person edited for this project. They are NOT footage: never put their \
         video ids in a script. Study them and match their quality: total length, how long shots are held (keyframe \
         times show where the picture changes), how interviews and scenery alternate, and how the story opens and ends.\n",
    );
    let per_ref = max_chars / refs.len();
    for (video_id, _, path) in refs {
        let name = Path::new(&path).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or(path);
        let duration = db
            .conn
            .query_row("SELECT duration_s FROM videos WHERE id = ?1", [video_id], |r| r.get::<_, Option<f64>>(0))
            .ok()
            .flatten()
            .unwrap_or(0.0);
        let mut block = format!("\n\"{name}\" ({duration:.0} s)\nPicture:\n");
        if let Ok(mut st) = db.conn.prepare("SELECT t_s, description_json FROM frames WHERE video_id = ?1 ORDER BY t_s")
        {
            let rows = st.query_map([video_id], |r| Ok((r.get::<_, f64>(0)?, r.get::<_, Option<String>>(1)?)));
            for (t, d) in rows.into_iter().flatten().flatten() {
                let desc: String = d
                    .and_then(|j| serde_json::from_str::<Value>(&j).ok())
                    .and_then(|v| v.get("description").and_then(|x| x.as_str()).map(str::to_string))
                    .unwrap_or_default()
                    .chars()
                    .take(110)
                    .collect();
                block.push_str(&format!("  {t:.1}s {desc}\n"));
            }
        }
        block.push_str("Speech:\n");
        if let Ok(mut st) =
            db.conn.prepare("SELECT start_s, end_s, text FROM transcript_segments WHERE video_id = ?1 ORDER BY start_s")
        {
            let rows =
                st.query_map([video_id], |r| Ok((r.get::<_, f64>(0)?, r.get::<_, f64>(1)?, r.get::<_, String>(2)?)));
            for (a, b, t) in rows.into_iter().flatten().flatten() {
                block.push_str(&format!("  {a:.1}-{b:.1}s {}\n", t.trim()));
            }
        }
        if block.len() > per_ref {
            let mut cut = per_ref;
            while !block.is_char_boundary(cut) {
                cut -= 1;
            }
            block.truncate(cut);
            block.push_str("…\n");
        }
        out.push_str(&block);
    }
    out
}

/// After this many searches in a row come back empty, the model is told to look at the footage
/// directly instead: a weak local model otherwise repeats the same query until it runs out of turns.
const EMPTY_SEARCHES_BEFORE_HINT: usize = 2;

const EMPTY_SEARCH_HINT: &str = "Those searches found nothing — the words you are searching for are not in this \
footage. Stop searching: call list_videos, then get_video and get_transcript on the videos that look useful, and \
build the script from what they actually show.";

/// A `search_moments` result with no hits.
fn is_empty_search(tool: &str, summary: &str) -> bool {
    tool == "search_moments" && (summary.starts_with("0 hits") || summary == "no matches")
}

/// Check whether a video belongs to a project.
pub fn is_video_in_project(db: &Db, project_id: i64, video_id: i64) -> bool {
    let res: Result<i64, _> = db.conn.query_row(
        "SELECT 1 FROM video_files vf
         JOIN folders f ON f.id = vf.folder_id
         JOIN project_folders pf ON pf.folder_id = f.id
         WHERE pf.project_id = ?1 AND vf.video_id = ?2
           AND NOT EXISTS (SELECT 1 FROM project_exclusions x WHERE x.project_id = pf.project_id AND x.video_id = vf.video_id)
         LIMIT 1",
        params![project_id, video_id],
        |r| r.get(0),
    );
    res.is_ok()
}

/// Truncate a JSON array of items until its serialized string length <= max_len.
fn truncate_json_list<T: Serialize>(items: &mut Vec<T>, max_len: usize) -> String {
    while !items.is_empty() {
        if let Ok(s) = serde_json::to_string(items)
            && s.len() <= max_len
        {
            return s;
        }
        items.pop();
    }
    serde_json::to_string(items).unwrap_or_else(|_| "[]".into())
}

/// Dispatch a tool call using DB and optional precomputed query vector. Returns (json_result, summary).
pub fn dispatch_tool(
    db: &Db,
    data_dir: &Path,
    project_id: i64,
    tool: &str,
    args: &Value,
    grounding: &mut Grounding,
    vector: Option<&[f32]>,
) -> (String, String) {
    dispatch_tool_limited(db, data_dir, project_id, tool, args, grounding, vector, LOCAL_TOOL_RESULT_CHARS)
}

/// Tool results for the local helper (8k context) stay small.
pub const LOCAL_TOOL_RESULT_CHARS: usize = 1500;
/// Servers have large contexts (highllama: 120k); richer results make much better edits.
pub const SERVER_TOOL_RESULT_CHARS: usize = 8000;

/// [`dispatch_tool`] with an explicit cap on the JSON result length.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_tool_limited(
    db: &Db,
    data_dir: &Path,
    project_id: i64,
    tool: &str,
    args: &Value,
    grounding: &mut Grounding,
    vector: Option<&[f32]>,
    max_chars: usize,
) -> (String, String) {
    match tool {
        "search_moments" => {
            let query = match args.get("query").and_then(|v| v.as_str()) {
                Some(q) if !q.trim().is_empty() => q,
                _ => return (json!({"error": "missing or invalid query"}).to_string(), "error: missing query".into()),
            };
            let kind = args.get("kind").and_then(|v| v.as_str()).map(|k| vec![k.to_string()]);
            let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(6).clamp(1, 8) as usize;
            let opts = SearchOptions { project_id: Some(project_id), limit, kinds: kind };

            let hits = match search_with_vector(db, data_dir, query, vector, &opts) {
                Ok(h) => h,
                Err(e) => {
                    return (
                        json!({"error": format!("search failed: {e}")}).to_string(),
                        "error: search failed".into(),
                    );
                }
            };

            #[derive(Serialize)]
            struct CompactHit {
                video_id: i64,
                file: String,
                start_s: f64,
                end_s: f64,
                snippet: String,
            }

            let mut compact = Vec::new();
            for hit in &hits {
                grounding.add(hit.video_id, hit.start_s, hit.end_s);
                let file = hit.path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
                let snippet: String = hit.snippet.chars().take(200).collect();
                compact.push(CompactHit {
                    video_id: hit.video_id,
                    file,
                    start_s: (hit.start_s * 100.0).round() / 100.0,
                    end_s: (hit.end_s * 100.0).round() / 100.0,
                    snippet,
                });
            }

            let summary = format!("{} hits", hits.len());
            let json_str = truncate_json_list(&mut compact, max_chars);
            (json_str, summary)
        }
        "get_transcript" => {
            let video_id = match args.get("video_id").and_then(|v| v.as_i64()) {
                Some(id) => id,
                None => return (json!({"error": "missing video_id"}).to_string(), "error: missing video_id".into()),
            };
            if !is_video_in_project(db, project_id, video_id) {
                return (
                    json!({"error": format!("video #{video_id} not in project")}).to_string(),
                    format!("error: video #{video_id} not in project"),
                );
            }
            let start_s = args.get("start_s").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let end_s = args.get("end_s").and_then(|v| v.as_f64()).unwrap_or(f64::MAX);

            grounding.add(video_id, start_s, end_s);

            let mut st = match db.conn.prepare(
                "SELECT start_s, end_s, text FROM transcript_segments
                 WHERE video_id = ?1 AND end_s >= ?2 AND start_s <= ?3
                 ORDER BY start_s",
            ) {
                Ok(s) => s,
                Err(e) => return (json!({"error": e.to_string()}).to_string(), "error: db prepare".into()),
            };

            #[derive(Serialize)]
            struct CompactSeg {
                start_s: f64,
                end_s: f64,
                text: String,
            }

            let rows = match st.query_map(params![video_id, start_s, end_s], |r| {
                Ok(CompactSeg {
                    start_s: (r.get::<_, f64>(0)? * 100.0).round() / 100.0,
                    end_s: (r.get::<_, f64>(1)? * 100.0).round() / 100.0,
                    text: r.get(2)?,
                })
            }) {
                Ok(rows) => rows,
                Err(e) => return (json!({"error": e.to_string()}).to_string(), "error: db query".into()),
            };

            let mut segs: Vec<CompactSeg> = rows.filter_map(Result::ok).collect();
            let summary = format!("{} segments", segs.len());
            let json_str = truncate_json_list(&mut segs, max_chars);
            (json_str, summary)
        }
        "get_video" => {
            let video_id = match args.get("video_id").and_then(|v| v.as_i64()) {
                Some(id) => id,
                None => return (json!({"error": "missing video_id"}).to_string(), "error: missing video_id".into()),
            };
            if !is_video_in_project(db, project_id, video_id) {
                return (
                    json!({"error": format!("video #{video_id} not in project")}).to_string(),
                    format!("error: video #{video_id} not in project"),
                );
            }

            struct VideoMeta {
                duration_s: Option<f64>,
                fps: Option<f64>,
                width: Option<i64>,
                height: Option<i64>,
                language: Option<String>,
            }

            let vid_meta: Option<VideoMeta> = db
                .conn
                .query_row(
                    "SELECT duration_s, fps, width, height, language FROM videos WHERE id = ?1",
                    [video_id],
                    |r| {
                        Ok(VideoMeta {
                            duration_s: r.get(0)?,
                            fps: r.get(1)?,
                            width: r.get(2)?,
                            height: r.get(3)?,
                            language: r.get(4)?,
                        })
                    },
                )
                .ok();

            let (duration_s, fps, width, height, language) = match vid_meta {
                Some(m) => (m.duration_s, m.fps, m.width, m.height, m.language),
                None => {
                    return (
                        json!({"error": format!("video #{video_id} not found")}).to_string(),
                        "error: not found".into(),
                    );
                }
            };

            grounding.add(video_id, 0.0, duration_s.unwrap_or(0.0));

            let file_path: Option<String> = db
                .conn
                .query_row("SELECT path FROM video_files WHERE video_id = ?1 LIMIT 1", [video_id], |r| r.get(0))
                .ok();
            let file = file_path
                .as_ref()
                .map(|p| Path::new(p).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default())
                .unwrap_or_default();

            #[derive(Serialize)]
            struct FrameSummary {
                t_s: f64,
                summary: String,
            }

            let mut all_frames: Vec<(f64, Option<String>)> = Vec::new();
            if let Ok(mut st) =
                db.conn.prepare("SELECT t_s, description_json FROM frames WHERE video_id = ?1 ORDER BY t_s")
                && let Ok(rows) =
                    st.query_map([video_id], |r| Ok((r.get::<_, f64>(0)?, r.get::<_, Option<String>>(1)?)))
            {
                for r in rows.flatten() {
                    all_frames.push(r);
                }
            }

            let range_start = args.get("start_s").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let range_end = args.get("end_s").and_then(|v| v.as_f64()).unwrap_or(f64::MAX);
            all_frames.retain(|(t, _)| *t >= range_start - 0.5 && *t <= range_end + 0.5);
            let max_sampled = if max_chars > LOCAL_TOOL_RESULT_CHARS { 40 } else { 12 };
            let sampled_raw = if all_frames.len() <= max_sampled {
                all_frames
            } else {
                let count = all_frames.len();
                let mut chosen = Vec::with_capacity(max_sampled);
                for i in 0..max_sampled {
                    let idx = (i * (count - 1)) / (max_sampled - 1);
                    chosen.push(all_frames[idx].clone());
                }
                chosen
            };

            let mut frames = Vec::new();
            for (t_s, desc_json) in sampled_raw {
                let summary_text = if let Some(j) = desc_json {
                    if let Ok(v) = serde_json::from_str::<Value>(&j) {
                        {
                            let desc_chars = if max_chars > LOCAL_TOOL_RESULT_CHARS { 300 } else { 120 };
                            let mut text: String = v
                                .get("description")
                                .and_then(|d| d.as_str())
                                .unwrap_or("")
                                .chars()
                                .take(desc_chars)
                                .collect();
                            if let Some(vt) = v.get("visible_text").and_then(|t| t.as_array()).filter(|a| !a.is_empty())
                            {
                                let vt: Vec<&str> = vt.iter().filter_map(|x| x.as_str()).collect();
                                text.push_str(&format!(" [text: {}]", vt.join(" | ")));
                            }
                            text
                        }
                    } else {
                        String::new()
                    }
                } else {
                    String::new()
                };
                frames.push(FrameSummary { t_s: (t_s * 100.0).round() / 100.0, summary: summary_text });
            }

            let summary = format!("{}s, {} frames", duration_s.unwrap_or(0.0) as i64, frames.len());

            let mut obj = json!({
                "video_id": video_id,
                "file": file,
                "duration_s": duration_s,
                "fps": fps,
                "width": width,
                "height": height,
                "language": language,
                "frames": frames,
            });

            // Drop frames until the JSON fits.
            while obj.to_string().len() > max_chars && !frames.is_empty() {
                frames.pop();
                obj["frames"] = json!(frames);
            }

            (obj.to_string(), summary)
        }
        "list_videos" => {
            let mut st = match db.conn.prepare(
                "SELECT DISTINCT v.id, v.duration_s, v.language, v.summary
                 FROM videos v
                 JOIN video_files vf ON vf.video_id = v.id
                 JOIN folders f ON f.id = vf.folder_id
                 JOIN project_folders pf ON pf.folder_id = f.id
                 WHERE pf.project_id = ?1
                   AND NOT EXISTS (SELECT 1 FROM project_exclusions x WHERE x.project_id = pf.project_id AND x.video_id = vf.video_id)
                 ORDER BY v.id",
            ) {
                Ok(s) => s,
                Err(e) => return (json!({"error": e.to_string()}).to_string(), "error: db prepare".into()),
            };

            #[derive(Serialize)]
            struct VideoItem {
                video_id: i64,
                file: String,
                duration_s: Option<f64>,
                language: Option<String>,
                summary: String,
            }

            let rows = match st.query_map([project_id], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, Option<f64>>(1)?,
                    r.get::<_, Option<String>>(2)?,
                    r.get::<_, Option<String>>(3)?,
                ))
            }) {
                Ok(rows) => rows,
                Err(e) => return (json!({"error": e.to_string()}).to_string(), "error: db query".into()),
            };

            let mut videos = Vec::new();
            for r in rows.flatten() {
                let (vid, duration_s, language, mut sum) = r;
                let file_path: Option<String> = db
                    .conn
                    .query_row("SELECT path FROM video_files WHERE video_id = ?1 LIMIT 1", [vid], |row| row.get(0))
                    .ok();
                let file = file_path
                    .as_ref()
                    .map(|p| Path::new(p).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default())
                    .unwrap_or_default();

                if sum.as_ref().is_none_or(|s| s.trim().is_empty()) {
                    let frame_desc: Option<String> = db
                        .conn
                        .query_row(
                            "SELECT description_json FROM frames WHERE video_id = ?1 AND description_json IS NOT NULL ORDER BY t_s LIMIT 1",
                            [vid],
                            |row| row.get(0),
                        )
                        .ok();
                    if let Some(fd) = frame_desc
                        && let Ok(v) = serde_json::from_str::<Value>(&fd)
                    {
                        sum = v.get("description").and_then(|d| d.as_str()).map(|s| s.to_string());
                    }
                }

                if sum.as_ref().is_none_or(|s| s.trim().is_empty()) {
                    sum = db
                        .conn
                        .query_row(
                            "SELECT text FROM transcript_segments WHERE video_id = ?1 ORDER BY start_s LIMIT 1",
                            [vid],
                            |row| row.get(0),
                        )
                        .ok();
                }

                let summary_text: String = sum.unwrap_or_default().chars().take(150).collect();
                videos.push(VideoItem { video_id: vid, file, duration_s, language, summary: summary_text });
            }

            let summary = format!("{} videos", videos.len());
            let json_str = truncate_json_list(&mut videos, max_chars);
            (json_str, summary)
        }
        unknown => {
            (json!({"error": format!("unknown tool: {unknown}")}).to_string(), format!("error: unknown tool {unknown}"))
        }
    }
}

/// Check grounding and project constraints for a script. Returns error issues for ungrounded clips.
pub fn check_grounding(db: &Db, project_id: i64, script: &Script, grounding: &Grounding) -> Vec<Issue> {
    let mut issues = Vec::new();
    for beat in &script.beats {
        for (clip_idx, clip) in beat.clips.iter().enumerate() {
            if !is_video_in_project(db, project_id, clip.video_id) {
                issues.push(Issue {
                    severity: IssueSeverity::Error,
                    beat_id: Some(beat.id.clone()),
                    clip_index: Some(clip_idx),
                    message: format!("video #{} does not belong to project", clip.video_id),
                });
            } else if !grounding.is_grounded(clip.video_id, clip.in_s, clip.out_s) {
                issues.push(Issue {
                    severity: IssueSeverity::Error,
                    beat_id: Some(beat.id.clone()),
                    clip_index: Some(clip_idx),
                    message: format!(
                        "clip not grounded in tool results: video #{} [{:.1}s–{:.1}s]",
                        clip.video_id, clip.in_s, clip.out_s
                    ),
                });
            }
        }
    }
    issues
}

/// A script is "off target" when its clips add up to more than this factor of the target duration.
const TARGET_OVERSHOOT: f64 = 1.25;
/// Shortest clip automatic trimming leaves.
const MIN_TRIMMED_CLIP_S: f64 = 4.0;

/// Last resort when the model ignores the target: shorten every clip proportionally (keeping its
/// in point, never below 4 s). Returns whether anything changed.
pub fn trim_to_target(script: &mut Script) -> bool {
    trim_to_target_with(script, |_| false)
}

/// [`trim_to_target`] that leaves `keep` clips (people speaking) whole and shortens the others.
fn trim_to_target_with(script: &mut Script, keep: impl Fn(&ScriptClip) -> bool) -> bool {
    let Some(target) = script.target_duration_s.filter(|t| *t > 0.0) else { return false };
    let total = script.total_duration_s();
    if total <= target * TARGET_OVERSHOOT {
        return false;
    }
    let kept: f64 = script.beats.iter().flat_map(|b| &b.clips).filter(|c| keep(c)).map(|c| c.out_s - c.in_s).sum();
    let flexible = total - kept;
    if flexible <= 0.0 {
        return false;
    }
    let factor = ((target - kept) / flexible).clamp(0.0, 1.0);
    let mut changed = false;
    for clip in script.beats.iter_mut().flat_map(|b| b.clips.iter_mut()) {
        if keep(clip) {
            continue;
        }
        let len = clip.out_s - clip.in_s;
        let new_len = (len * factor).max(MIN_TRIMMED_CLIP_S).min(len);
        if new_len < len {
            clip.out_s = clip.in_s + new_len;
            changed = true;
        }
    }
    changed
}

fn video_duration(db: &Db, video_id: i64) -> Option<f64> {
    db.conn
        .query_row("SELECT duration_s FROM videos WHERE id = ?1", [video_id], |r| r.get::<_, Option<f64>>(0))
        .ok()
        .flatten()
}

/// Keep clips inside their video; drop the ones that start past its end.
fn clamp_to_duration(db: &Db, script: &mut Script) {
    for beat in &mut script.beats {
        beat.clips.retain_mut(|c| {
            if let Some(d) = video_duration(db, c.video_id) {
                c.out_s = c.out_s.min(d);
                c.in_s = c.in_s.max(0.0);
            }
            c.out_s - c.in_s >= 1.0
        });
    }
    script.beats.retain(|b| !b.clips.is_empty());
}

/// A video length stated in the user's message: "60 second", "90s", "1.5 minutes", "2 minutos".
pub fn requested_duration_s(message: &str) -> Option<f64> {
    let lower = message.to_lowercase();
    let tokens: Vec<&str> =
        lower.split(|c: char| c.is_whitespace() || c == '-' || c == ',').filter(|t| !t.is_empty()).collect();
    for (i, tok) in tokens.iter().enumerate() {
        // "90s" / "2min" glued forms, or a number followed by a unit word.
        let split = tok.find(|c: char| !(c.is_ascii_digit() || c == '.')).unwrap_or(tok.len());
        let (num, glued) = tok.split_at(split);
        let Ok(n) = num.parse::<f64>() else { continue };
        let unit = if glued.is_empty() { tokens.get(i + 1).copied().unwrap_or("") } else { glued };
        let unit = unit.trim_matches(|c: char| !c.is_alphabetic());
        let secs = if ["s", "sec", "secs", "second", "seconds", "seg", "segs", "segundo", "segundos"].contains(&unit) {
            n
        } else if ["m", "min", "mins", "minute", "minutes", "minuto", "minutos"].contains(&unit) {
            n * 60.0
        } else {
            continue;
        };
        if (5.0..=3600.0).contains(&secs) {
            return Some(secs);
        }
    }
    None
}

/// A later clip overlapping footage an earlier clip already showed (by more than 1 s) is dropped.
fn drop_repeated_footage(script: &mut Script) -> usize {
    let mut used: Vec<(i64, f64, f64)> = Vec::new();
    let mut dropped = 0;
    for beat in &mut script.beats {
        beat.clips.retain(|c| {
            let repeat = used.iter().any(|&(v, a, b)| v == c.video_id && c.out_s.min(b) - c.in_s.max(a) > 1.0);
            if repeat {
                dropped += 1;
            } else {
                used.push((c.video_id, c.in_s, c.out_s));
            }
            !repeat
        });
    }
    script.beats.retain(|b| !b.clips.is_empty());
    dropped
}

/// When the speaking clips alone overrun the target, shorten each proportionally, cutting after a
/// whole sentence (plus the tail) rather than mid-word. Returns whether anything changed.
fn fit_speech_to_target(db: &Db, script: &mut Script) -> bool {
    let Some(target) = script.target_duration_s.filter(|t| *t > 0.0) else { return false };
    let speaking = |c: &ScriptClip| clip_has_speech(db, c.video_id, c.in_s, c.out_s);
    let spoken: f64 =
        script.beats.iter().flat_map(|b| &b.clips).filter(|c| speaking(c)).map(|c| c.out_s - c.in_s).sum();
    // Leave room for the other shots: speech may use up to 70 % of the target.
    let budget = target * 0.7;
    if spoken <= budget * TARGET_OVERSHOOT {
        return false;
    }
    let factor = budget / spoken;
    let mut changed = false;
    for c in script.beats.iter_mut().flat_map(|b| b.clips.iter_mut()) {
        if !speaking(c) {
            continue;
        }
        let max_out = c.in_s + (c.out_s - c.in_s) * factor;
        let ends: Vec<f64> = db
            .conn
            .prepare_cached(
                "SELECT end_s FROM transcript_segments WHERE video_id = ?1 AND start_s < ?3 AND end_s > ?2 ORDER BY start_s",
            )
            .and_then(|mut st| {
                st.query_map(params![c.video_id, c.in_s, c.out_s], |r| r.get::<_, f64>(0))
                    .map(|rows| rows.flatten().collect())
            })
            .unwrap_or_default();
        // Last sentence end that fits with its tail; at least the first sentence.
        let cut = ends.iter().copied().rfind(|e| e + SPEECH_TAIL_S <= max_out).or(ends.first().copied());
        if let Some(e) = cut {
            let out = (e + SPEECH_TAIL_S).min(c.out_s);
            if out < c.out_s - 0.05 {
                c.out_s = out;
                changed = true;
            }
        }
    }
    changed
}

/// Back-to-back clips of the same video in one beat (`0-6`, `6-16`, `16-23`) are jump cuts inside a
/// single continuous take: join them. Returns how many cuts were removed.
fn merge_contiguous_clips(script: &mut Script) -> usize {
    let mut merged = 0;
    for beat in &mut script.beats {
        let mut out: Vec<ScriptClip> = Vec::with_capacity(beat.clips.len());
        for c in beat.clips.drain(..) {
            if let Some(prev) = out.last_mut()
                && prev.video_id == c.video_id
                && (c.in_s - prev.out_s).abs() <= 0.5
                && c.out_s - prev.in_s <= MAX_CLIP_S
            {
                prev.out_s = prev.out_s.max(c.out_s);
                if c.audio == crate::script::Audio::Source {
                    prev.audio = crate::script::Audio::Source;
                }
                merged += 1;
                continue;
            }
            out.push(c);
        }
        beat.clips = out;
    }
    merged
}

/// Default editing instructions. Users can replace them in Settings (`[chat] system_prompt`);
/// `{project}`, `{fps}`, `{width}` and `{height}` are filled in. The tool list and the clip-range
/// rule are always appended, since scripts can't be built without them.
pub const DEFAULT_EDITOR_PROMPT: &str = "You are a senior documentary and promo video editor working on project \"{project}\" \
({fps} fps, {width}x{height}). You cut real footage into a watchable, well-paced story and write the voice-over for it.

HOW TO EDIT
1. Understand the material first: list the videos, look at the keyframes of the promising ones, and read the transcripts of the videos where people talk.
2. Build a story: a hook, 3-6 beats that each make one point, and a clear ending. Every beat has a purpose.
3. Choose only strong shots: a clear subject (people, a landmark, a building, a sign, activity, a striking view). Skip footage whose keyframes describe black or blank frames, blur, transitions, the ground or sky only, empty hillsides with nothing to see, or the same view as the previous clip.
4. Pacing - slower is better than frantic:
   - hold every shot at least 4 s so viewers can see it and read any text; views and b-roll 5-10 s;
   - when someone speaks, keep the clip from just before their first word to the end of their sentences (use the transcript timestamps; up to ~25 s) with audio \"source\", and never cut the moment they stop: hold 1-2 s of the person on screen after the last word;
   - prefer fewer, longer clips over many quick cuts; never jump between unrelated shots every 2 s.
5. Audio: use \"source\" when a person is speaking in the clip; use \"mute\" for scenery and b-roll so wind and handling noise don't play under the voice-over.
6. Narration (voice-over) must cover the beat: about 2.5 spoken words per second of the beat's clips (a 20 s beat needs ~50 words). Set narration to \"\" for beats where people speak on camera (never copy their words into the narration). Every beat has its own narration; never repeat text from another beat, and never reuse the same footage twice. Write natural, specific sentences about what is on screen and why it matters; no filler. on_screen_text is short (a title or a name).
7. Length: the clips add up to the length the user asked for; set target_duration_s to it. If the user gave no length, choose what the material supports (usually 60-180 s).
8. The user's feedback overrides these defaults. When revising, change what they asked for (slower, longer, different shots, more narration) and keep what they didn't mention; never return the previous draft unchanged.
9. Always reply in the user's language.
";

/// Construct the system prompt for the editor agent: the editing instructions (`custom` or the
/// default), the fixed tool contract, and the current draft.
pub fn build_system_prompt(project: &Project, latest_script_json: Option<&str>, custom: Option<&str>) -> String {
    let template = custom.map(str::trim).filter(|c| !c.is_empty()).unwrap_or(DEFAULT_EDITOR_PROMPT);
    let fps = if project.fps_den == 1 {
        project.fps_num.to_string()
    } else {
        format!("{:.3}", project.fps_num as f64 / project.fps_den as f64)
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string()
    };
    let mut prompt = template
        .replace("{project}", &project.name)
        .replace("{fps}", &fps)
        .replace("{width}", &project.width.to_string())
        .replace("{height}", &project.height.to_string());
    prompt.push_str(
        "\n\nTOOLS (use them generously before writing; your clips can only come from what they return)\n\
         - list_videos(): every video with a short summary.\n\
         - search_moments(query, kind, limit): find moments by meaning or keyword (speech, on-screen text, visuals).\n\
         - get_video(video_id, start_s, end_s): keyframe descriptions - what is actually visible and when.\n\
         - get_transcript(video_id, start_s, end_s): what people say, with timestamps.\n\n\
         CLIP RANGES: in_s/out_s must lie inside ranges returned by the tools; never use a video_id or range you \
         have not inspected. Search results are search windows, not clips: cut a sub-range out of them.\n",
    );

    if let Some(draft) = latest_script_json {
        prompt.push_str(&format!(
            "\nCurrent script draft from this session:\n{}\n\
             When the user asks for changes, revise this draft accordingly.\n",
            draft
        ));
    }

    prompt
}

/// Spoken voice-over rate used to check that narration fills its beat.
const NARRATION_WORDS_PER_S: f64 = 2.5;
/// Narration covering less than this share of its beat triggers a redraft.
const MIN_NARRATION_COVERAGE: f64 = 0.6;

/// Editorial problems the model can fix in a redraft: narration too short for its beat, and clips
/// over footage the tools know nothing about (no speech and no described keyframe nearby).
pub fn content_issues(db: &Db, script: &Script) -> Vec<Issue> {
    let mut issues = Vec::new();
    let mut seen_narration: Vec<String> = Vec::new();
    for beat in &script.beats {
        let Some(n) = beat.narration.as_deref().map(str::trim).filter(|n| !n.is_empty()) else { continue };
        let key = n.to_lowercase();
        if seen_narration.contains(&key) {
            issues.push(Issue {
                severity: IssueSeverity::Warning,
                beat_id: Some(beat.id.clone()),
                clip_index: None,
                message:
                    "narration repeats an earlier beat word for word; write new narration for what this beat shows"
                        .into(),
            });
        } else {
            seen_narration.push(key);
        }
    }
    for beat in &script.beats {
        let beat_s: f64 = beat.clips.iter().map(|c| c.out_s - c.in_s).sum();
        let words = beat.narration.as_deref().map(|n| n.split_whitespace().count()).unwrap_or(0);
        let has_speech = beat.clips.iter().any(|c| clip_has_speech(db, c.video_id, c.in_s, c.out_s));
        if words > 0 || !has_speech {
            let spoken_s = words as f64 / NARRATION_WORDS_PER_S;
            if beat_s > 0.0 && spoken_s < beat_s * MIN_NARRATION_COVERAGE {
                issues.push(Issue {
                    severity: IssueSeverity::Warning,
                    beat_id: Some(beat.id.clone()),
                    clip_index: None,
                    message: format!(
                        "narration is {words} words (~{spoken_s:.0} s spoken) but the beat runs {beat_s:.0} s; \
                         write about {:.0} words or shorten the beat",
                        beat_s * NARRATION_WORDS_PER_S
                    ),
                });
            }
        }
        for (i, c) in beat.clips.iter().enumerate() {
            if !clip_has_speech(db, c.video_id, c.in_s, c.out_s)
                && !clip_has_described_frame(db, c.video_id, c.in_s, c.out_s)
            {
                issues.push(Issue {
                    severity: IssueSeverity::Warning,
                    beat_id: Some(beat.id.clone()),
                    clip_index: Some(i),
                    message: format!(
                        "video #{} {:.1}–{:.1} s has no speech and no described keyframe; pick a range you have seen described",
                        c.video_id, c.in_s, c.out_s
                    ),
                });
            }
        }
    }
    issues
}

fn clip_has_speech(db: &Db, video_id: i64, in_s: f64, out_s: f64) -> bool {
    db.conn
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM transcript_segments WHERE video_id = ?1 AND end_s > ?2 AND start_s < ?3)",
            params![video_id, in_s, out_s],
            |r| r.get::<_, bool>(0),
        )
        .unwrap_or(false)
}

/// A described keyframe inside the clip, or shortly before it (the view it continues from).
fn clip_has_described_frame(db: &Db, video_id: i64, in_s: f64, out_s: f64) -> bool {
    db.conn
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM frames WHERE video_id = ?1 AND description_json IS NOT NULL
               AND t_s >= ?2 - 10.0 AND t_s <= ?3)",
            params![video_id, in_s, out_s],
            |r| r.get::<_, bool>(0),
        )
        .unwrap_or(false)
}

/// Silence kept after the last words of a speaking clip, so the cut doesn't clip the person off.
const SPEECH_TAIL_S: f64 = 1.5;
/// Breath kept before the first words.
const SPEECH_LEAD_S: f64 = 0.5;
/// Never extend a clip further than this to finish a sentence.
const MAX_SPEECH_EXTEND_S: f64 = 12.0;

/// Let speaking clips breathe: finish the sentence the clip is in, then hold ~1.5 s of the person
/// before cutting (without running into their next sentence), and start slightly before the first
/// words. Runs after [`snap_to_segments`], which lands cuts exactly on segment boundaries.
fn pad_speech(db: &Db, script: &mut Script) -> usize {
    let mut changed = 0;
    for c in script.beats.iter_mut().flat_map(|b| b.clips.iter_mut()) {
        let Ok(mut st) = db
            .conn
            .prepare_cached("SELECT start_s, end_s FROM transcript_segments WHERE video_id = ?1 ORDER BY start_s")
        else {
            continue;
        };
        let segs: Vec<(f64, f64)> = st
            .query_map([c.video_id], |r| Ok((r.get::<_, f64>(0)?, r.get::<_, f64>(1)?)))
            .map(|rows| rows.flatten().collect())
            .unwrap_or_default();
        let overlapping: Vec<usize> =
            (0..segs.len()).filter(|&i| segs[i].1 > c.in_s + 0.05 && segs[i].0 < c.out_s - 0.05).collect();
        let (Some(&first), Some(&last)) = (overlapping.first(), overlapping.last()) else { continue };
        let duration = video_duration(db, c.video_id).unwrap_or(f64::MAX);
        let (old_in, old_out) = (c.in_s, c.out_s);

        let speech_end = segs[last].1;
        let mut out = speech_end + SPEECH_TAIL_S;
        if let Some(next) = segs.get(last + 1) {
            out = out.min((next.0 - 0.3).max(speech_end));
        }
        let out = out.min(old_out + MAX_SPEECH_EXTEND_S).min(duration);
        if out > c.out_s {
            c.out_s = out;
        }

        let speech_start = segs[first].0;
        let mut lead = (speech_start - SPEECH_LEAD_S).max(0.0);
        if first > 0 {
            lead = lead.max(segs[first - 1].1);
        }
        // Also pulls a clip that starts mid-sentence back to the start of that sentence.
        if c.in_s > lead {
            c.in_s = lead.max(old_in - MAX_SPEECH_EXTEND_S);
        }
        if (c.in_s, c.out_s) != (old_in, old_out) {
            changed += 1;
        }
    }
    changed
}

/// Mute clips nobody speaks in when their beat has narration: ambient wind and handling noise
/// shouldn't play under the voice-over.
fn mute_silent_clips(db: &Db, script: &mut Script) -> usize {
    let mut muted = 0;
    for beat in &mut script.beats {
        if beat.narration.as_deref().is_none_or(|n| n.trim().is_empty()) {
            continue;
        }
        for c in &mut beat.clips {
            if c.audio == crate::script::Audio::Source && !clip_has_speech(db, c.video_id, c.in_s, c.out_s) {
                c.audio = crate::script::Audio::Mute;
                muted += 1;
            }
        }
    }
    muted
}

/// Create a new chat session in the database.
pub fn create_session(db: &Db, project_id: i64, title: &str) -> Result<i64, Error> {
    let t = now();
    db.conn.execute(
        "INSERT INTO chat_sessions(project_id, title, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4)",
        params![project_id, title, t, t],
    )?;
    Ok(db.conn.last_insert_rowid())
}

/// List all chat sessions for a project ordered by most recently updated.
pub fn sessions(db: &Db, project_id: i64) -> Result<Vec<ChatSession>, Error> {
    let mut st = db.conn.prepare(
        "SELECT id, project_id, title, created_at, updated_at
         FROM chat_sessions WHERE project_id = ?1 ORDER BY updated_at DESC",
    )?;
    let rows = st.query_map([project_id], |r| {
        Ok(ChatSession {
            id: r.get(0)?,
            project_id: r.get(1)?,
            title: r.get(2)?,
            created_at: r.get(3)?,
            updated_at: r.get(4)?,
        })
    })?;
    Ok(rows.filter_map(Result::ok).collect())
}

/// Summary of a chat session including message count.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SessionSummary {
    pub id: i64,
    pub project_id: i64,
    pub title: String,
    pub message_count: usize,
    pub updated_at: i64,
}

/// List all chat sessions for a project with their message count.
pub fn sessions_with_counts(db: &Db, project_id: i64) -> Result<Vec<SessionSummary>, Error> {
    let mut st = db.conn.prepare(
        "SELECT cs.id, cs.project_id, cs.title, cs.updated_at, COUNT(cm.id)
         FROM chat_sessions cs
         LEFT JOIN chat_messages cm ON cm.session_id = cs.id
         WHERE cs.project_id = ?1
         GROUP BY cs.id
         ORDER BY cs.updated_at DESC",
    )?;
    let rows = st.query_map([project_id], |r| {
        Ok(SessionSummary {
            id: r.get(0)?,
            project_id: r.get(1)?,
            title: r.get(2)?,
            updated_at: r.get(3)?,
            message_count: r.get::<_, i64>(4)? as usize,
        })
    })?;
    Ok(rows.filter_map(Result::ok).collect())
}

/// Retrieve all messages for a session.
pub fn messages(db: &Db, session_id: i64) -> Result<Vec<ChatMessage>, Error> {
    let mut st = db.conn.prepare(
        "SELECT id, session_id, role, content, tool_calls_json, created_at
         FROM chat_messages WHERE session_id = ?1 ORDER BY id ASC",
    )?;
    let rows = st.query_map([session_id], |r| {
        let id: i64 = r.get(0)?;
        let sid: i64 = r.get(1)?;
        let role: String = r.get(2)?;
        let content: String = r.get(3)?;
        let tool_calls_json: Option<String> = r.get(4)?;
        let created_at: i64 = r.get(5)?;

        let mut tool_calls = None;
        let mut script_id = None;

        if let Some(json_str) = tool_calls_json
            && let Ok(val) = serde_json::from_str::<Value>(&json_str)
        {
            if let Some(arr) = val.as_array() {
                tool_calls = serde_json::from_value::<Vec<ToolCallRecord>>(Value::Array(arr.clone())).ok();
            } else if let Some(obj) = val.as_object() {
                script_id = obj.get("script_id").and_then(|v| v.as_i64());
                if let Some(tc) = obj.get("tool_calls") {
                    tool_calls = serde_json::from_value::<Vec<ToolCallRecord>>(tc.clone()).ok();
                }
            }
        }

        Ok(ChatMessage { id, session_id: sid, role, content, tool_calls, script_id, created_at })
    })?;

    Ok(rows.filter_map(Result::ok).collect())
}

/// Run one turn of the script chat agent.
pub async fn run_turn(
    ctx: &mut ChatContext,
    project_id: i64,
    session_id: Option<i64>,
    message: &str,
    on_event: &mut (dyn FnMut(ChatEvent) + Send),
) -> Result<TurnResult, Error> {
    let project = ctx.db.project(project_id)?;

    let (session_id, is_new_session) = match session_id {
        Some(id) => (id, false),
        None => {
            let title: String = message.chars().take(60).collect();
            let sid = create_session(&ctx.db, project_id, &title)?;
            (sid, true)
        }
    };

    let mut grounding = Grounding::default();
    let mut prior_messages = Vec::new();
    let mut latest_script_json = None;

    if !is_new_session {
        prior_messages = messages(&ctx.db, session_id)?;
        for m in &prior_messages {
            if let Some(calls) = &m.tool_calls {
                for c in calls {
                    grounding.record_tool_call(&c.tool, &c.args, &ctx.db);
                }
            }
        }
        let stored_json: Option<String> = ctx
            .db
            .conn
            .query_row(
                "SELECT script_json FROM scripts WHERE session_id = ?1 ORDER BY version DESC LIMIT 1",
                [session_id],
                |r| r.get(0),
            )
            .ok();
        if let Some(ref json_str) = stored_json
            && let Ok(prev_script) = serde_json::from_str::<Script>(json_str)
        {
            for beat in &prev_script.beats {
                for clip in &beat.clips {
                    grounding.add(clip.video_id, clip.in_s, clip.out_s);
                }
            }
        }
        latest_script_json = stored_json;
    }

    let mut sys_prompt = build_system_prompt(&project, latest_script_json.as_deref(), ctx.system_prompt.as_deref());
    let reference_chars = match ctx.backend {
        ChatBackend::Server { .. } => 6000,
        // Roughly an eighth of the window (~4 chars per token), so results leave room for the draft.
        ChatBackend::Local { ctx_tokens, .. } => ((ctx_tokens as usize) * 4 / 8).clamp(1500, 12000),
    };
    sys_prompt.push_str(&reference_edits_text(&ctx.db, project_id, reference_chars));
    // A length the user states ("60 second promo", "2 minutos") wins over whatever the model sets.
    let requested_s = requested_duration_s(message);
    // Only a first draft (or an explicit length) is squeezed to its target; revisions follow feedback.
    let enforce_target = latest_script_json.is_none() || requested_s.is_some();

    let mut tool_records = Vec::new();
    let mut raw_reply = String::new();
    let mut parsed_script: Option<Script> = None;
    let mut pre_issues: Vec<Issue> = Vec::new();

    match &mut ctx.backend {
        ChatBackend::Server { url, model, api_key } => {
            let client = reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(5))
                .timeout(Duration::from_secs(300))
                .build()
                .map_err(|e| Error::Invalid(format!("reqwest client: {e}")))?;

            let mut req_messages = Vec::new();
            req_messages.push(json!({ "role": "system", "content": sys_prompt }));
            for pm in &prior_messages {
                if pm.role == "user" || pm.role == "assistant" {
                    req_messages.push(json!({ "role": &pm.role, "content": &pm.content }));
                }
            }
            req_messages.push(json!({ "role": "user", "content": message }));

            let mut tool_rounds = 0;
            let mut empty_searches = 0usize;
            let mut hinted = false;
            let mut last_assistant_text = String::new();

            while tool_rounds < 16 {
                tool_rounds += 1;
                let mut body = json!({
                    "messages": req_messages,
                    "tools": tools_definition(),
                    "tool_choice": "auto",
                    "temperature": 0.3,
                    "chat_template_kwargs": { "enable_thinking": false },
                });
                if !model.is_empty() {
                    body["model"] = json!(model);
                }

                let mut req = client.post(format!("{}/v1/chat/completions", url.trim_end_matches('/'))).json(&body);
                if !api_key.is_empty() {
                    req = req.bearer_auth(api_key.as_str());
                }

                let resp = req.send().await.map_err(|e| Error::Vision(format!("server at {url}: {e}")))?;
                if !resp.status().is_success() {
                    let err_text = resp.text().await.unwrap_or_default();
                    return Err(Error::Vision(format!("server HTTP error: {err_text}")));
                }

                let v: Value = resp.json().await.map_err(|e| Error::Vision(format!("bad response JSON: {e}")))?;
                let choice = v.get("choices").and_then(|c| c.get(0)).and_then(|c| c.get("message"));
                let msg_obj = match choice {
                    Some(m) => m,
                    None => break,
                };

                if let Some(txt) = msg_obj.get("content").and_then(|c| c.as_str())
                    && !txt.trim().is_empty()
                {
                    last_assistant_text = txt.to_string();
                }

                let tool_calls = msg_obj.get("tool_calls").and_then(|t| t.as_array());
                if tool_calls.is_none() || tool_calls.unwrap().is_empty() {
                    // Assistant finished tool calling
                    break;
                }

                req_messages.push(msg_obj.clone());

                if empty_searches >= EMPTY_SEARCHES_BEFORE_HINT && !hinted {
                    hinted = true;
                    req_messages.push(json!({ "role": "user", "content": EMPTY_SEARCH_HINT }));
                }

                for tc in tool_calls.unwrap() {
                    let call_id = tc.get("id").and_then(|v| v.as_str()).unwrap_or("");
                    let func = tc.get("function").cloned().unwrap_or(Value::Null);
                    let tool_name = func.get("name").and_then(|n| n.as_str()).unwrap_or("").to_string();
                    let tool_args: Value =
                        func.get("arguments")
                            .and_then(|a| {
                                if let Some(s) = a.as_str() { serde_json::from_str(s).ok() } else { Some(a.clone()) }
                            })
                            .unwrap_or(json!({}));

                    on_event(ChatEvent::ToolStarted { tool: tool_name.clone(), args: tool_args.clone() });

                    let vector = if tool_name == "search_moments" {
                        if let (Some(e), Some(q)) =
                            (ctx.embedder.as_mut(), tool_args.get("query").and_then(|v| v.as_str()))
                        {
                            query_vector(e, q).await.ok()
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    let (result_str, summary) = dispatch_tool_limited(
                        &ctx.db,
                        &ctx.data_dir,
                        project_id,
                        &tool_name,
                        &tool_args,
                        &mut grounding,
                        vector.as_deref(),
                        SERVER_TOOL_RESULT_CHARS,
                    );

                    on_event(ChatEvent::ToolFinished { tool: tool_name.clone(), summary: summary.clone() });

                    let empty = is_empty_search(&tool_name, &summary);
                    let was_search = tool_name.starts_with("search");
                    tool_records.push(ToolCallRecord { tool: tool_name, args: tool_args, summary });

                    if empty {
                        empty_searches += 1;
                    } else if !was_search {
                        empty_searches = 0;
                    }

                    req_messages.push(json!({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": result_str,
                    }));
                }
            }

            // Drafting final script call
            on_event(ChatEvent::Drafting);

            req_messages.push(json!({
                "role": "user",
                "content": format!(
                    "Now produce the final complete Script JSON using only the footage returned by the tools. \
                     Follow the pacing rules, and apply this request from the user: {message}"
                )
            }));

            let mut final_body = json!({
                "messages": req_messages,
                "temperature": 0.3,
                "chat_template_kwargs": { "enable_thinking": false },
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "script",
                        "strict": true,
                        "schema": script_json_schema()
                    }
                }
            });
            if !model.is_empty() {
                final_body["model"] = json!(model);
            }

            let mut req = client.post(format!("{}/v1/chat/completions", url.trim_end_matches('/'))).json(&final_body);
            if !api_key.is_empty() {
                req = req.bearer_auth(api_key.as_str());
            }
            let resp = req.send().await.map_err(|e| Error::Vision(format!("server at {url}: {e}")))?;
            if !resp.status().is_success() {
                let err_text = resp.text().await.unwrap_or_default();
                return Err(Error::Vision(format!("server HTTP error (final script): {err_text}")));
            }
            let v: Value = resp.json().await.map_err(|e| Error::Vision(format!("bad final JSON: {e}")))?;
            let content = v["choices"][0]["message"]["content"].as_str().unwrap_or_default();

            let mut script_res = Script::parse_for_project(content, &project);
            if let (Ok(s), Some(t)) = (&mut script_res, requested_s) {
                s.target_duration_s = Some(t);
            }

            // Grounding + pacing check, redraft once on the server backend
            on_event(ChatEvent::Validating);

            let redraft_reasons = match &script_res {
                Ok(s) => {
                    let mut i = check_grounding(&ctx.db, project_id, s, &grounding);
                    i.extend(pacing_issues(s, enforce_target));
                    i.extend(content_issues(&ctx.db, s));
                    i
                }
                Err(_) => Vec::new(),
            };

            if !redraft_reasons.is_empty() && script_res.is_ok() {
                // Retry once
                on_event(ChatEvent::Drafting);
                let issue_text: Vec<String> = redraft_reasons.iter().map(|i| i.message.clone()).collect();
                req_messages.push(json!({ "role": "assistant", "content": content }));
                req_messages.push(json!({
                    "role": "user",
                    "content": format!(
                        "Fix these problems and return the complete corrected Script JSON:\n{}\n\
                         Every clip must lie inside a range returned by the tools and follow the pacing rules. \
                         Keep applying the user's request: {message}",
                        issue_text.join("\n")
                    )
                }));
                final_body["messages"] = json!(req_messages);

                let mut retry_req =
                    client.post(format!("{}/v1/chat/completions", url.trim_end_matches('/'))).json(&final_body);
                if !api_key.is_empty() {
                    retry_req = retry_req.bearer_auth(api_key.as_str());
                }
                if let Ok(retry_resp) = retry_req.send().await
                    && let Ok(rv) = retry_resp.json::<Value>().await
                {
                    let retry_content = rv["choices"][0]["message"]["content"].as_str().unwrap_or_default();
                    if let Ok(rescript) = Script::parse_for_project(retry_content, &project) {
                        script_res = Ok(rescript);
                    }
                }
            }

            if let Ok(mut s) = script_res {
                if requested_s.is_some() {
                    s.target_duration_s = requested_s;
                }
                pre_issues = enforce_grounding_and_pacing(&ctx.db, project_id, &mut s, &grounding, enforce_target);
                parsed_script = Some(s);
            }

            raw_reply = last_assistant_text;
        }
        ChatBackend::Local { helper, .. } => {
            let mut transcript = format!("<|im_start|>system\n{sys_prompt}<|im_end|>\n");
            for pm in &prior_messages {
                if pm.role == "user" || pm.role == "assistant" {
                    transcript.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", pm.role, pm.content));
                }
            }
            transcript.push_str(&format!("<|im_start|>user\n{message}<|im_end|>\n"));

            let mut tool_rounds = 0;
            let mut empty_searches = 0usize;
            let mut hinted = false;
            while tool_rounds < 8 {
                tool_rounds += 1;
                let out_str = helper.complete(&transcript, Some(local_action_schema())).await?;
                let action: Result<LocalAction, _> = serde_json::from_str(&out_str);
                match action {
                    Ok(LocalAction::Tool { tool, args }) => {
                        on_event(ChatEvent::ToolStarted { tool: tool.clone(), args: args.clone() });
                        let vector = if tool == "search_moments" {
                            if let (Some(e), Some(q)) =
                                (ctx.embedder.as_mut(), args.get("query").and_then(|v| v.as_str()))
                            {
                                query_vector(e, q).await.ok()
                            } else {
                                None
                            }
                        } else {
                            None
                        };

                        let (res, summary) = dispatch_tool(
                            &ctx.db,
                            &ctx.data_dir,
                            project_id,
                            &tool,
                            &args,
                            &mut grounding,
                            vector.as_deref(),
                        );
                        on_event(ChatEvent::ToolFinished { tool: tool.clone(), summary: summary.clone() });
                        let empty = is_empty_search(&tool, &summary);
                        tool_records.push(ToolCallRecord { tool: tool.clone(), args: args.clone(), summary });

                        if empty {
                            empty_searches += 1;
                        } else if !tool.starts_with("search") {
                            empty_searches = 0;
                        }
                        let hint = if empty_searches >= EMPTY_SEARCHES_BEFORE_HINT && !hinted {
                            hinted = true;
                            format!("\n{EMPTY_SEARCH_HINT}")
                        } else {
                            String::new()
                        };
                        transcript.push_str(&format!(
                            "<|im_start|>assistant\n{}\n<|im_end|>\n<|im_start|>user\nTool result for {tool}:\n{res}{hint}\n<|im_end|>\n",
                            out_str
                        ));
                    }
                    Ok(LocalAction::Final { script }) => {
                        let mut s = script;
                        s.fill_from_project(&project);
                        parsed_script = Some(s);
                        break;
                    }
                    Err(e) => {
                        return Err(Error::Invalid(format!("local helper output did not match action schema: {e}")));
                    }
                }
            }

            if parsed_script.is_none() {
                on_event(ChatEvent::Drafting);
                transcript.push_str("<|im_start|>user\nProduce the final script action.<|im_end|>\n");
                let final_str = helper.complete(&transcript, Some(local_final_action_schema())).await?;
                if let Ok(LocalAction::Final { mut script }) = serde_json::from_str::<LocalAction>(&final_str) {
                    script.fill_from_project(&project);
                    parsed_script = Some(script);
                }
            }

            on_event(ChatEvent::Validating);
            if let Some(s) = &mut parsed_script {
                if requested_s.is_some() {
                    s.target_duration_s = requested_s;
                }
                pre_issues = enforce_grounding_and_pacing(&ctx.db, project_id, s, &grounding, enforce_target);
            }
        }
    }

    let mut script_id = None;
    let mut issues = pre_issues;

    if let Some(mut s) = parsed_script {
        if s.clip_count() > 0 && !s.beats.is_empty() {
            let _ = snap_to_segments(&ctx.db, &mut s)?;
            pad_speech(&ctx.db, &mut s);
            clamp_to_duration(&ctx.db, &mut s);
            let muted = mute_silent_clips(&ctx.db, &mut s);
            if muted > 0 {
                issues.push(Issue {
                    severity: IssueSeverity::Info,
                    beat_id: None,
                    clip_index: None,
                    message: format!("muted {muted} clip(s) without speech under the narration"),
                });
            }
            issues.extend(content_issues(&ctx.db, &s));
            issues.extend(validate(&ctx.db, project_id, &s)?);
            let sid = save_version(&ctx.db, project_id, &s, Some(session_id))?;
            script_id = Some(sid);
            parsed_script = Some(s);
        } else {
            parsed_script = None;
        }
    }

    let reply = if !raw_reply.trim().is_empty() {
        raw_reply
    } else if let Some(s) = &parsed_script {
        format!(
            "Drafted '{}': {} beats, {} clips, {:.1} s",
            s.title,
            s.beats.len(),
            s.clip_count(),
            s.total_duration_s()
        )
    } else {
        "I was unable to assemble a script from the available footage.".to_string()
    };

    // Persist messages in database
    let now_ts = now();

    // 1. User message
    ctx.db.conn.execute(
        "INSERT INTO chat_messages(session_id, role, content, tool_calls_json, created_at)
         VALUES (?1, 'user', ?2, NULL, ?3)",
        params![session_id, message, now_ts],
    )?;

    // 2. Tool summary row if any tools were called
    if !tool_records.is_empty() {
        let tool_summary = format!(
            "Used {} tools: {}",
            tool_records.len(),
            tool_records.iter().map(|t| t.tool.as_str()).collect::<Vec<_>>().join(", ")
        );
        let tool_json = serde_json::to_string(&tool_records).ok();
        ctx.db.conn.execute(
            "INSERT INTO chat_messages(session_id, role, content, tool_calls_json, created_at)
             VALUES (?1, 'tool', ?2, ?3, ?4)",
            params![session_id, tool_summary, tool_json, now_ts],
        )?;
    }

    // 3. Assistant message
    let assistant_tc_json = script_id.map(|sid| json!({ "script_id": sid }).to_string());
    ctx.db.conn.execute(
        "INSERT INTO chat_messages(session_id, role, content, tool_calls_json, created_at)
         VALUES (?1, 'assistant', ?2, ?3, ?4)",
        params![session_id, reply, assistant_tc_json, now_ts],
    )?;

    // Update chat_sessions updated_at
    ctx.db.conn.execute("UPDATE chat_sessions SET updated_at = ?1 WHERE id = ?2", params![now_ts, session_id])?;

    Ok(TurnResult { session_id, reply, script_id, script: parsed_script, issues, tool_calls: tool_records })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projects::NewProject;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[test]
    fn local_action_parsing() {
        let tool_json = r#"{"action":"tool","tool":"search_moments","args":{"query":"unbox","limit":5}}"#;
        let action: LocalAction = serde_json::from_str(tool_json).unwrap();
        match action {
            LocalAction::Tool { tool, args } => {
                assert_eq!(tool, "search_moments");
                assert_eq!(args["query"], "unbox");
                assert_eq!(args["limit"], 5);
            }
            _ => panic!("expected Tool action"),
        }

        let final_json = r#"{
            "action": "final",
            "script": {
                "title": "Teaser",
                "beats": [
                    {
                        "id": "b1",
                        "purpose": "hook",
                        "clips": [
                            { "video_id": 1, "in_s": 0.0, "out_s": 3.5 }
                        ]
                    }
                ]
            }
        }"#;
        let action: LocalAction = serde_json::from_str(final_json).unwrap();
        match action {
            LocalAction::Final { script } => {
                assert_eq!(script.title, "Teaser");
                assert_eq!(script.beats.len(), 1);
                assert_eq!(script.beats[0].clips[0].video_id, 1);
            }
            _ => panic!("expected Final action"),
        }
    }

    #[test]
    fn grounding_checks() {
        let mut g = Grounding::default();
        g.add(1, 10.0, 20.0);

        // Within range
        assert!(g.is_grounded(1, 12.0, 18.0));
        // Overlap within 5s slack (e.g. 5.0 to 12.0 overlaps 10.0..20.0)
        assert!(g.is_grounded(1, 6.0, 11.0));
        // Completely outside slack
        assert!(!g.is_grounded(1, 0.0, 4.0));
        // Different video
        assert!(!g.is_grounded(2, 12.0, 18.0));
        // Overlapping but far outside: a whole-window clip touching the hit is not grounded
        assert!(!g.is_grounded(1, 0.0, 45.0));
    }

    #[test]
    fn pacing_redraft_and_trim() {
        use crate::script::{Audio, Beat, ScriptClip};
        let clip = |in_s: f64, out_s: f64| ScriptClip { video_id: 1, in_s, out_s, audio: Audio::Source, why: None };
        let mut s = Script {
            title: "t".into(),
            target_duration_s: Some(20.0),
            fps: None,
            width: None,
            height: None,
            beats: vec![Beat {
                id: "b1".into(),
                purpose: "p".into(),
                narration: None,
                on_screen_text: None,
                clips: vec![clip(0.0, 45.0), clip(90.0, 144.0), clip(200.0, 206.0)],
                notes: None,
            }],
        };
        let issues = pacing_issues(&s, true);
        // two over-long clips + total far from target
        assert_eq!(issues.len(), 3);
        assert_eq!(pacing_issues(&s, false).len(), 2, "revisions don't enforce the target");

        let mut db = Db::open_in_memory().unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let p = db.create_project(&NewProject::named("P")).unwrap();
        let folder = db.add_folder(p.id, tmp.path(), true).unwrap();
        db.conn
            .execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (1, 'h', 1, 400.0)", [])
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen) VALUES (1, ?1, 'a.mp4', 1, 0, 0)",
                [folder.id],
            )
            .unwrap();
        let mut g = Grounding::default();
        g.add(1, 0.0, 400.0);
        let mut revision = s.clone();
        enforce_grounding_and_pacing(&db, p.id, &mut revision, &g, false);
        assert!(revision.total_duration_s() > 20.0 * TARGET_OVERSHOOT, "revision keeps its length");
        let applied = enforce_grounding_and_pacing(&db, p.id, &mut s, &g, true);
        assert!(!applied.is_empty());
        assert!(s.beats[0].clips.iter().all(|c| c.out_s - c.in_s <= TRIMMED_CLIP_S + 1e-9));
        assert!(s.total_duration_s() <= 20.0 * TARGET_OVERSHOOT);
    }

    #[test]
    fn removed_and_reference_videos_leave_the_library() {
        let mut db = Db::open_in_memory().unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let p = db.create_project(&NewProject::named("P")).unwrap();
        let folder = db.add_folder(p.id, tmp.path(), true).unwrap();
        for (id, name) in [(1, "a.mp4"), (2, "wife-edit.mp4"), (3, "c.mp4")] {
            db.conn
                .execute(
                    "INSERT INTO videos(id, content_hash, size, duration_s) VALUES (?1, ?2, 1, 30.0)",
                    params![id, name],
                )
                .unwrap();
            db.conn
                .execute(
                    "INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen) VALUES (?1, ?2, ?3, 1, 0, 0)",
                    params![id, folder.id, name],
                )
                .unwrap();
        }
        db.conn
            .execute(
                "INSERT INTO transcript_segments(video_id, start_s, end_s, text) VALUES (2, 0.0, 4.0, 'we love it')",
                [],
            )
            .unwrap();
        db.exclude_video(p.id, 2, "reference").unwrap();
        db.exclude_video(p.id, 3, "removed").unwrap();
        assert!(db.exclude_video(p.id, 1, "bogus").is_err());

        assert!(is_video_in_project(&db, p.id, 1));
        assert!(!is_video_in_project(&db, p.id, 2), "a reference edit is never footage");
        assert!(!is_video_in_project(&db, p.id, 3));
        let st = crate::index::status(&db, Some(p.id)).unwrap();
        assert_eq!(st.videos, 1);

        let text = reference_edits_text(&db, p.id, 6000);
        assert!(text.contains("wife-edit.mp4") && text.contains("we love it"), "{text}");
        assert!(!text.contains("c.mp4"));

        db.include_video(p.id, 2).unwrap();
        assert!(is_video_in_project(&db, p.id, 2));
        assert!(reference_edits_text(&db, p.id, 6000).is_empty());
    }

    #[test]
    fn empty_searches_are_recognised() {
        assert!(is_empty_search("search_moments", "0 hits"));
        assert!(is_empty_search("search_moments", "no matches"));
        assert!(!is_empty_search("search_moments", "8 hits"));
        assert!(!is_empty_search("get_transcript", "0 segments"), "only searches count");
        assert!(EMPTY_SEARCH_HINT.contains("list_videos"));
    }

    #[test]
    fn requested_duration_from_message() {
        assert_eq!(requested_duration_s("Create a 60 second promo video"), Some(60.0));
        assert_eq!(requested_duration_s("make it 90s long"), Some(90.0));
        assert_eq!(requested_duration_s("un video de 2 minutos"), Some(120.0));
        assert_eq!(requested_duration_s("about 1.5 minutes, please"), Some(90.0));
        assert_eq!(requested_duration_s("a 60-second teaser"), Some(60.0));
        assert_eq!(requested_duration_s("show the 3 houses"), None);
        assert_eq!(requested_duration_s("leave people talking longer"), None);
    }

    #[test]
    fn content_checks_narration_and_undescribed_footage() {
        use crate::script::{Audio, Beat, ScriptClip};
        let mut db = Db::open_in_memory().unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let p = db.create_project(&NewProject::named("P")).unwrap();
        let folder = db.add_folder(p.id, tmp.path(), true).unwrap();
        db.conn
            .execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (1, 'h', 1, 400.0)", [])
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen) VALUES (1, ?1, 'a.mp4', 1, 0, 0)",
                [folder.id],
            )
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO transcript_segments(video_id, start_s, end_s, text) VALUES (1, 100.0, 110.0, 'hi')",
                [],
            )
            .unwrap();
        db.conn.execute("INSERT INTO frames(video_id, t_s, description_json) VALUES (1, 20.0, '{}')", []).unwrap();
        let clip = |in_s: f64, out_s: f64| ScriptClip { video_id: 1, in_s, out_s, audio: Audio::Source, why: None };
        let beat = |id: &str, narration: Option<&str>, clips| Beat {
            id: id.into(),
            purpose: "p".into(),
            narration: narration.map(Into::into),
            on_screen_text: None,
            clips,
            notes: None,
        };
        let mut s = Script {
            title: "t".into(),
            target_duration_s: None,
            fps: None,
            width: None,
            height: None,
            beats: vec![
                // 20 s of scenery with 5 words of narration: too short.
                beat("views", Some("A view of the hills"), vec![clip(22.0, 42.0)]),
                // someone talking, no narration needed
                beat("talk", None, vec![clip(100.0, 110.0)]),
                // nothing known about 300-306 s
                beat(
                    "dead",
                    Some("one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen"),
                    vec![clip(300.0, 306.0)],
                ),
            ],
        };
        let issues = content_issues(&db, &s);
        let msgs: Vec<_> = issues.iter().map(|i| (i.beat_id.clone().unwrap(), i.clip_index)).collect();
        assert_eq!(msgs, vec![("views".to_string(), None), ("dead".to_string(), Some(0))], "{issues:?}");

        // clip cut right at the end of the words: gains the 1.5 s tail and a short lead-in
        db.conn
            .execute(
                "INSERT INTO transcript_segments(video_id, start_s, end_s, text) VALUES (1, 111.0, 115.0, 'next')",
                [],
            )
            .unwrap();
        let mut talk = s.clone();
        talk.beats[1].clips[0] = clip(100.0, 110.0);
        pad_speech(&db, &mut talk);
        let c = &talk.beats[1].clips[0];
        assert!((c.in_s - 99.5).abs() < 1e-9, "{c:?}");
        assert!((c.out_s - 110.7).abs() < 1e-9, "stops before the next sentence: {c:?}");
        db.conn.execute("DELETE FROM transcript_segments WHERE text = 'next'", []).unwrap();
        let mut talk = s.clone();
        talk.beats[1].clips[0] = clip(100.0, 105.0);
        pad_speech(&db, &mut talk);
        assert!((talk.beats[1].clips[0].out_s - 111.5).abs() < 1e-9, "finishes the sentence, then holds");

        assert_eq!(mute_silent_clips(&db, &mut s), 2);
        assert_eq!(s.beats[1].clips[0].audio, Audio::Source, "speech without narration keeps its audio");
    }

    #[tokio::test]
    async fn tool_dispatch_in_memory() {
        let tmp = tempfile::tempdir().unwrap();
        let mut db = Db::open_in_memory().unwrap();
        let p = db.create_project(&NewProject::named("TestProj")).unwrap();
        let folder = db.add_folder(p.id, tmp.path(), true).unwrap();
        let file_path = tmp.path().join("clip.mp4");
        std::fs::write(&file_path, b"data").unwrap();

        let c = &db.conn;
        c.execute(
            "INSERT INTO videos(id, content_hash, size, duration_s, fps, width, height, language, summary)
             VALUES (10, 'hash10', 1000, 30.0, 25.0, 1920, 1080, 'en', 'CM5 unboxing overview')",
            [],
        )
        .unwrap();

        c.execute(
            "INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen)
             VALUES (10, ?1, ?2, 1000, 0, 0)",
            params![folder.id, file_path.to_str().unwrap()],
        )
        .unwrap();

        c.execute(
            "INSERT INTO transcript_segments(video_id, start_s, end_s, text)
             VALUES (10, 0.0, 5.0, 'welcome to cm5 unboxing'),
                    (10, 5.0, 12.0, 'here is the compute module board')",
            [],
        )
        .unwrap();

        c.execute(
            "INSERT INTO frames(video_id, t_s, description_json)
             VALUES (10, 2.0, '{\"description\": \"Hands holding a green board.\"}'),
                    (10, 8.0, '{\"description\": \"Close-up of the chip.\"}')",
            [],
        )
        .unwrap();

        c.execute(
            "INSERT INTO chunks(video_id, kind, start_s, end_s, text)
             VALUES (10, 'moment', 0.0, 5.0, 'Hands holding a green board. welcome to cm5 unboxing')",
            [],
        )
        .unwrap();

        let mut grounding = Grounding::default();

        // 1. list_videos
        let (res, sum) = dispatch_tool(&db, tmp.path(), p.id, "list_videos", &json!({}), &mut grounding, None);
        assert!(res.len() <= 1500);
        assert!(res.contains("clip.mp4"));
        assert_eq!(sum, "1 videos");

        // 2. get_video
        let (res, sum) =
            dispatch_tool(&db, tmp.path(), p.id, "get_video", &json!({"video_id": 10}), &mut grounding, None);
        assert!(res.len() <= 1500);
        assert!(res.contains("Hands holding a green board"));
        assert!(sum.contains("30s"));
        assert!(grounding.is_grounded(10, 0.0, 30.0));

        // 3. get_transcript
        let (res, sum) = dispatch_tool(
            &db,
            tmp.path(),
            p.id,
            "get_transcript",
            &json!({"video_id": 10, "start_s": 0.0, "end_s": 6.0}),
            &mut grounding,
            None,
        );
        assert!(res.len() <= 1500);
        assert!(res.contains("welcome to cm5 unboxing"));
        assert_eq!(sum, "2 segments");

        // 4. search_moments (keyword only)
        let (res, sum) =
            dispatch_tool(&db, tmp.path(), p.id, "search_moments", &json!({"query": "unboxing"}), &mut grounding, None);
        assert!(res.len() <= 1500);
        assert!(res.contains("unboxing"));
        assert_eq!(sum, "1 hits");

        // 5. Foreign video rejected
        let (res, sum) =
            dispatch_tool(&db, tmp.path(), p.id, "get_video", &json!({"video_id": 999}), &mut grounding, None);
        assert!(res.contains("error"));
        assert!(sum.contains("error"));
    }

    #[tokio::test]
    async fn server_loop_against_fake_server() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let server_url = format!("http://127.0.0.1:{port}");

        // Spawn minimal HTTP server handling 3 sequential calls
        tokio::spawn(async move {
            for step in 1..=3 {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut buf = vec![0u8; 8192];
                let mut total_read = 0;
                while total_read < buf.len() {
                    let n = socket.read(&mut buf[total_read..]).await.unwrap();
                    if n == 0 {
                        break;
                    }
                    total_read += n;
                    let s = String::from_utf8_lossy(&buf[..total_read]);
                    if let Some(header_end) = s.find("\r\n\r\n") {
                        if let Some(cl_idx) = s.to_lowercase().find("content-length:") {
                            let rest = &s[cl_idx + 15..];
                            let end_line = rest.find("\r\n").unwrap();
                            let cl: usize = rest[..end_line].trim().parse().unwrap();
                            if total_read >= header_end + 4 + cl {
                                break;
                            }
                        } else {
                            break;
                        }
                    }
                }

                let req_text = String::from_utf8_lossy(&buf[..total_read]);

                let resp_body = match step {
                    1 => {
                        assert!(req_text.contains("tools"));
                        json!({
                            "choices": [{
                                "message": {
                                    "role": "assistant",
                                    "content": null,
                                    "tool_calls": [{
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "search_moments",
                                            "arguments": "{\"query\":\"unboxing\"}"
                                        }
                                    }]
                                }
                            }]
                        })
                    }
                    2 => {
                        json!({
                            "choices": [{
                                "message": {
                                    "role": "assistant",
                                    "content": "Found the unboxing footage. Now drafting your script.",
                                    "tool_calls": null
                                }
                            }]
                        })
                    }
                    3 => {
                        assert!(req_text.contains("response_format"));
                        json!({
                            "choices": [{
                                "message": {
                                    "role": "assistant",
                                    "content": json!({
                                        "title": "CM5 Fast Teaser",
                                        "beats": [
                                            {
                                                "id": "b1",
                                                "purpose": "hook",
                                                "narration": "Meet the new compute module.",
                                                "clips": [
                                                    { "video_id": 1, "in_s": 1.0, "out_s": 4.0, "audio": "source" }
                                                ]
                                            }
                                        ]
                                    }).to_string()
                                }
                            }]
                        })
                    }
                    _ => unreachable!(),
                };

                let resp_bytes = resp_body.to_string();
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    resp_bytes.len(),
                    resp_bytes
                );
                socket.write_all(resp.as_bytes()).await.unwrap();
                let _ = socket.shutdown().await;
            }
        });

        // Set up project and database
        let tmp = tempfile::tempdir().unwrap();
        let mut db = Db::open_in_memory().unwrap();
        let p = db.create_project(&NewProject::named("FakeServerProj")).unwrap();
        let folder = db.add_folder(p.id, tmp.path(), true).unwrap();
        let file_path = tmp.path().join("clip.mp4");
        std::fs::write(&file_path, b"test").unwrap();

        let c = &db.conn;
        c.execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (1, 'hash1', 500, 10.0)", []).unwrap();
        c.execute(
            "INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen)
             VALUES (1, ?1, ?2, 500, 0, 0)",
            params![folder.id, file_path.to_str().unwrap()],
        )
        .unwrap();
        c.execute(
            "INSERT INTO chunks(video_id, kind, start_s, end_s, text)
             VALUES (1, 'moment', 0.0, 5.0, 'unboxing the board')",
            [],
        )
        .unwrap();

        let mut events = Vec::new();
        let mut ctx = ChatContext {
            db,
            data_dir: tmp.path().to_path_buf(),
            backend: ChatBackend::Server { url: server_url, model: "test-model".into(), api_key: String::new() },
            embedder: None,
            system_prompt: None,
        };

        let res =
            run_turn(&mut ctx, p.id, None, "Create a 5s teaser about unboxing", &mut |e| events.push(e)).await.unwrap();

        assert!(res.script_id.is_some());
        let script = res.script.unwrap();
        assert_eq!(script.title, "CM5 Fast Teaser");
        assert_eq!(script.beats.len(), 1);
        assert_eq!(res.tool_calls.len(), 1);
        assert_eq!(res.tool_calls[0].tool, "search_moments");

        // Verify events
        assert!(events.iter().any(|e| matches!(e, ChatEvent::ToolStarted { tool, .. } if tool == "search_moments")));
        assert!(events.iter().any(|e| matches!(e, ChatEvent::ToolFinished { tool, .. } if tool == "search_moments")));
        assert!(events.iter().any(|e| matches!(e, ChatEvent::Drafting)));
        assert!(events.iter().any(|e| matches!(e, ChatEvent::Validating)));

        // Verify messages in DB
        let msgs = messages(&ctx.db, res.session_id).unwrap();
        assert_eq!(msgs.len(), 3); // user, tool, assistant
        assert_eq!(msgs[0].role, "user");
        assert_eq!(msgs[1].role, "tool");
        assert!(msgs[1].tool_calls.is_some());
        assert_eq!(msgs[2].role, "assistant");
        assert_eq!(msgs[2].script_id, res.script_id);
    }

    #[tokio::test]
    async fn grounding_rejection_drops_ungrounded_clips() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let server_url = format!("http://127.0.0.1:{port}");

        // Server does 1 round: returns no tool calls, then returns script with ungrounded clip range (100.0..105.0)
        // on retry returns the same ungrounded script
        tokio::spawn(async move {
            for step in 1..=3 {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut buf = vec![0u8; 8192];
                let n = socket.read(&mut buf).await.unwrap();
                let _req_text = String::from_utf8_lossy(&buf[..n]);

                let resp_body = match step {
                    1 => {
                        json!({
                            "choices": [{
                                "message": {
                                    "role": "assistant",
                                    "content": "No tools needed, drafting.",
                                    "tool_calls": null
                                }
                            }]
                        })
                    }
                    2 | 3 => {
                        json!({
                            "choices": [{
                                "message": {
                                    "role": "assistant",
                                    "content": json!({
                                        "title": "Ungrounded Script",
                                        "beats": [
                                            {
                                                "id": "b1",
                                                "purpose": "hook",
                                                "clips": [
                                                    { "video_id": 1, "in_s": 100.0, "out_s": 105.0 }
                                                ]
                                            }
                                        ]
                                    }).to_string()
                                }
                            }]
                        })
                    }
                    _ => unreachable!(),
                };

                let resp_bytes = resp_body.to_string();
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    resp_bytes.len(),
                    resp_bytes
                );
                socket.write_all(resp.as_bytes()).await.unwrap();
                let _ = socket.shutdown().await;
            }
        });

        let tmp = tempfile::tempdir().unwrap();
        let mut db = Db::open_in_memory().unwrap();
        let p = db.create_project(&NewProject::named("UngroundedProj")).unwrap();
        let folder = db.add_folder(p.id, tmp.path(), true).unwrap();
        let file_path = tmp.path().join("clip.mp4");
        std::fs::write(&file_path, b"test").unwrap();

        let c = &db.conn;
        c.execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (1, 'hash1', 500, 200.0)", [])
            .unwrap();
        c.execute(
            "INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen)
             VALUES (1, ?1, ?2, 500, 0, 0)",
            params![folder.id, file_path.to_str().unwrap()],
        )
        .unwrap();

        let mut ctx = ChatContext {
            db,
            data_dir: tmp.path().to_path_buf(),
            backend: ChatBackend::Server { url: server_url, model: "test-model".into(), api_key: String::new() },
            embedder: None,
            system_prompt: None,
        };

        let res = run_turn(&mut ctx, p.id, None, "Make video", &mut |_| {}).await.unwrap();

        // The clip was not grounded in tool results -> dropped -> no beats left -> script_id None
        assert_eq!(res.script_id, None);
        assert!(res.script.is_none());
    }

    #[test]
    fn run_turn_future_is_send() {
        fn assert_send<T: Send>(_: T) {}
        let _ = |ctx: &mut ChatContext, mut cb: Box<dyn FnMut(ChatEvent) + Send>| {
            assert_send(run_turn(ctx, 1, None, "test", &mut *cb));
        };
    }
}
