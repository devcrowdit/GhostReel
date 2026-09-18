//! MCP server (`ghostreel mcp`): the same operations the CLI does, as tools an agent can call
//! over stdio — projects and folders, indexing, search, transcripts and keyframes, scripts
//! (draft with the model, save, export).
//!
//! Protocol: JSON-RPC 2.0, one message per line on stdin/stdout (MCP stdio transport). Logs go
//! to stderr so they never corrupt the stream.

use std::io::Write as _;
use std::path::PathBuf;
use std::str::FromStr as _;

use anyhow::Context;
use ghostreel_core::config::Config;
use ghostreel_core::db::Db;
use ghostreel_core::paths::Paths;
use ghostreel_core::projects::NewProject;
use ghostreel_core::{doctor, index, runtime, script};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, BufReader};

const PROTOCOL_VERSION: &str = "2024-11-05";

pub async fn serve(paths: Paths) -> anyhow::Result<()> {
    // Handlers hold a Db across awaits (search, chat), which isn't Send: keep them on this thread.
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async move {
            let mut lines = BufReader::new(tokio::io::stdin()).lines();
            eprintln!("ghostreel mcp: ready (data: {})", paths.data_dir.display());
            while let Some(line) = lines.next_line().await? {
                if line.trim().is_empty() {
                    continue;
                }
                let req: Value = match serde_json::from_str(&line) {
                    Ok(v) => v,
                    Err(e) => {
                        send(&error_response(Value::Null, -32700, &format!("invalid JSON: {e}")));
                        continue;
                    }
                };
                let id = req.get("id").cloned().unwrap_or(Value::Null);
                let method = req.get("method").and_then(|m| m.as_str()).unwrap_or("").to_string();
                let params = req.get("params").cloned().unwrap_or(json!({}));

                // Notifications (no id) expect no response.
                if id.is_null() && method.starts_with("notifications/") {
                    continue;
                }

                match method.as_str() {
                    "initialize" => send(&json!({
                        "jsonrpc": "2.0", "id": id,
                        "result": {
                            "protocolVersion": PROTOCOL_VERSION,
                            "capabilities": { "tools": {} },
                            "serverInfo": { "name": "ghostreel", "version": env!("CARGO_PKG_VERSION") },
                            "instructions": INSTRUCTIONS,
                        }
                    })),
                    "ping" => send(&json!({ "jsonrpc": "2.0", "id": id, "result": {} })),
                    "tools/list" => send(&json!({ "jsonrpc": "2.0", "id": id, "result": { "tools": tools() } })),
                    "tools/call" => {
                        let name = params.get("name").and_then(|n| n.as_str()).unwrap_or("").to_string();
                        let args = params.get("arguments").cloned().unwrap_or(json!({}));
                        let result = call_tool(&paths, &name, &args).await;
                        send(&match result {
                            Ok(v) => json!({ "jsonrpc": "2.0", "id": id, "result": tool_result(&v, false) }),
                            // Tool errors belong in the result, so the agent can read and fix them.
                            Err(e) => json!({
                                "jsonrpc": "2.0", "id": id,
                                "result": tool_result(&json!({ "error": e.to_string() }), true)
                            }),
                        });
                    }
                    _ => send(&error_response(id, -32601, &format!("unknown method '{method}'"))),
                }
            }
            Ok(())
        })
        .await
}

const INSTRUCTIONS: &str = "GhostReel indexes folders of video (speech transcripts, keyframe \
descriptions, embeddings) and turns clips into an edit. Typical flow: create_project → add_folder \
→ index (repeat until it reports no pending work) → search / get_transcript / get_video to learn \
the footage → draft_script (the model picks the clips) or save_script (your own JSON) → \
export_script. Clip in_s/out_s must lie inside real footage; save_script reports issues instead \
of silently fixing them.";

fn send(msg: &Value) {
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{msg}");
    let _ = out.flush();
}

fn error_response(id: Value, code: i64, message: &str) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
}

fn tool_result(value: &Value, is_error: bool) -> Value {
    let text = serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string());
    json!({ "content": [{ "type": "text", "text": text }], "isError": is_error })
}

fn obj(props: Value, required: &[&str]) -> Value {
    json!({ "type": "object", "properties": props, "required": required, "additionalProperties": false })
}

fn tools() -> Vec<Value> {
    let project = json!({ "type": "string", "description": "Project name" });
    vec![
        json!({
            "name": "doctor",
            "description": "What GhostReel would use on this machine: ffmpeg, GPU, database, models, AI backends.",
            "inputSchema": obj(json!({}), &[]),
        }),
        json!({
            "name": "list_projects",
            "description": "Every project with its sequence settings and indexing progress.",
            "inputSchema": obj(json!({}), &[]),
        }),
        json!({
            "name": "create_project",
            "description": "Create a project. fps/width/height are the timeline settings used by exports.",
            "inputSchema": obj(json!({
                "name": project,
                "description": { "type": "string" },
                "fps": { "type": "number", "description": "e.g. 25, 30, 29.97 (default 25)" },
                "width": { "type": "integer", "description": "default 1920" },
                "height": { "type": "integer", "description": "default 1080" },
            }), &["name"]),
        }),
        json!({
            "name": "add_folder",
            "description": "Watch a folder of footage for a project (whole tree by default). Editor render/cache folders are skipped.",
            "inputSchema": obj(json!({
                "project": project,
                "path": { "type": "string", "description": "Absolute path of the folder" },
                "recursive": { "type": "boolean", "description": "default true" },
            }), &["project", "path"]),
        }),
        json!({
            "name": "index",
            "description": "Run pending indexing work (scan, transcribe, keyframes, descriptions, embeddings) for up to max_seconds, then report progress. Resumable: call again while anything is pending.",
            "inputSchema": obj(json!({
                "project": project,
                "max_seconds": { "type": "integer", "description": "Stop after this long (default 900)" },
                "redo": { "type": "string", "description": "Reset this stage and later ones first: frames | transcribe | describe | embed" },
            }), &["project"]),
        }),
        json!({
            "name": "index_status",
            "description": "Indexing progress per stage, and per-video state.",
            "inputSchema": obj(json!({ "project": project, "videos": { "type": "boolean", "description": "Include the video list" } }), &["project"]),
        }),
        json!({
            "name": "search",
            "description": "Search a project's footage by meaning and keywords (speech, on-screen text, what is visible). Returns moments with video_id and timestamps to cut from.",
            "inputSchema": obj(json!({
                "project": project,
                "query": { "type": "string" },
                "limit": { "type": "integer", "description": "default 10" },
                "kinds": { "type": "array", "items": { "type": "string", "enum": ["moment", "transcript", "frame"] } },
                "keywords_only": { "type": "boolean", "description": "Skip the embedding model" },
            }), &["project", "query"]),
        }),
        json!({
            "name": "get_transcript",
            "description": "Timestamped speech segments of a video, optionally limited to a range.",
            "inputSchema": obj(json!({
                "video_id": { "type": "integer" },
                "start_s": { "type": "number" },
                "end_s": { "type": "number" },
            }), &["video_id"]),
        }),
        json!({
            "name": "get_video",
            "description": "A video's metadata and its keyframes with descriptions and on-screen text (what is visible when).",
            "inputSchema": obj(json!({
                "video_id": { "type": "integer" },
                "limit": { "type": "integer", "description": "Max keyframes to return (default 40)" },
            }), &["video_id"]),
        }),
        json!({
            "name": "list_scripts",
            "description": "Script versions of a project (newest first).",
            "inputSchema": obj(json!({ "project": project }), &["project"]),
        }),
        json!({
            "name": "get_script",
            "description": "One script version: beats, clips, narration, on-screen text.",
            "inputSchema": obj(json!({ "script_id": { "type": "integer" } }), &["script_id"]),
        }),
        json!({
            "name": "draft_script",
            "description": "Ask GhostReel's editing model to draft or revise a script from the project's footage (it searches and reads transcripts itself). Slow: minutes. Returns the saved version and its issues.",
            "inputSchema": obj(json!({
                "project": project,
                "message": { "type": "string", "description": "e.g. 'a 60 second promo using the interviews, slower cuts'" },
                "session_id": { "type": "integer", "description": "Continue an earlier chat session (revisions)" },
            }), &["project", "message"]),
        }),
        json!({
            "name": "save_script",
            "description": "Save your own script as a new version. Clips must lie inside real footage; returns the issues found (errors mean the clip won't work).",
            "inputSchema": obj(json!({
                "project": project,
                "script": { "type": "object", "description": "{title, target_duration_s?, beats:[{id, purpose, narration?, on_screen_text?, clips:[{video_id, in_s, out_s, audio?}]}]}" },
            }), &["project", "script"]),
        }),
        json!({
            "name": "export_script",
            "description": "Export a script version as an editable timeline: fcp_xml for Premiere Pro, or otio.",
            "inputSchema": obj(json!({
                "script_id": { "type": "integer" },
                "format": { "type": "string", "enum": ["fcp_xml", "otio"], "description": "default fcp_xml" },
                "out": { "type": "string", "description": "Output file path" },
            }), &["script_id", "out"]),
        }),
        json!({
            "name": "delete_project",
            "description": "Delete a project. With purge=true also delete its keyframes, preview renders and the transcripts/descriptions/vectors of footage no other project uses. Video files are never touched.",
            "inputSchema": obj(json!({
                "project": project,
                "purge": { "type": "boolean", "description": "default false: keep the index for reuse" },
            }), &["project"]),
        }),
        json!({
            "name": "set_video_role",
            "description": "Take a video out of a project's library: 'removed' (ignored everywhere) or 'reference' (a finished edit the script chat studies but never cuts from). 'footage' puts it back.",
            "inputSchema": obj(json!({
                "project": project,
                "video_id": { "type": "integer" },
                "role": { "type": "string", "enum": ["footage", "removed", "reference"] },
            }), &["project", "video_id", "role"]),
        }),
    ]
}

fn open_db(paths: &Paths) -> anyhow::Result<Db> {
    Db::open(&paths.db_file()).with_context(|| format!("cannot open {}", paths.db_file().display()))
}

fn arg_str(args: &Value, key: &str) -> anyhow::Result<String> {
    args.get(key)
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .with_context(|| format!("missing string argument '{key}'"))
}

fn arg_i64(args: &Value, key: &str) -> anyhow::Result<i64> {
    args.get(key).and_then(|v| v.as_i64()).with_context(|| format!("missing integer argument '{key}'"))
}

fn project(db: &Db, args: &Value) -> anyhow::Result<ghostreel_core::projects::Project> {
    let name = arg_str(args, "project")?;
    Ok(db.require_project(&name)?)
}

async fn call_tool(paths: &Paths, name: &str, args: &Value) -> anyhow::Result<Value> {
    match name {
        "doctor" => Ok(serde_json::to_value(doctor::run(paths).await)?),
        "list_projects" => {
            let db = open_db(paths)?;
            let mut out = Vec::new();
            for p in db.projects()? {
                let st = index::status(&db, Some(p.id))?;
                out.push(json!({ "project": p, "status": st }));
            }
            Ok(json!(out))
        }
        "create_project" => {
            let db = open_db(paths)?;
            let (fps_num, fps_den) = match args.get("fps").and_then(|v| v.as_f64()) {
                Some(f) => {
                    let fps = script::Fps::from_f64(f);
                    (fps.num, fps.den)
                }
                None => (25, 1),
            };
            let p = NewProject {
                name: arg_str(args, "name")?,
                description: args.get("description").and_then(|v| v.as_str()).unwrap_or_default().to_string(),
                fps_num,
                fps_den,
                width: args.get("width").and_then(|v| v.as_i64()).unwrap_or(1920),
                height: args.get("height").and_then(|v| v.as_i64()).unwrap_or(1080),
            };
            Ok(serde_json::to_value(db.create_project(&p)?)?)
        }
        "add_folder" => {
            let mut db = open_db(paths)?;
            let p = project(&db, args)?;
            let path = PathBuf::from(arg_str(args, "path")?);
            let recursive = args.get("recursive").and_then(|v| v.as_bool()).unwrap_or(true);
            let folder = db.add_folder(p.id, &path, recursive)?;
            Ok(json!({ "folder": folder, "next": "call index to scan it" }))
        }
        "index" => {
            let mut db = open_db(paths)?;
            let p = project(&db, args)?;
            let config = Config::load(&paths.config_file).unwrap_or_default();
            if let Some(stage) = args.get("redo").and_then(|v| v.as_str()) {
                anyhow::ensure!(
                    index::STAGES.contains(&stage),
                    "unknown stage '{stage}'; valid: {}",
                    index::STAGES.join(", ")
                );
                index::reset_stages(&db, &paths.data_dir, Some(p.id), stage)?;
            }
            let max_seconds = args.get("max_seconds").and_then(|v| v.as_u64()).unwrap_or(900);
            let rt = runtime::resolve(paths, &config).await?;
            let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let deadline = cancel.clone();
            let ticker = tokio::task::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_secs(max_seconds)).await;
                deadline.store(true, std::sync::atomic::Ordering::Relaxed);
            });
            let opts = index::Options { project_id: Some(p.id), cancel: Some(cancel), ..Default::default() };
            let summary = index::run(&mut db, &rt, &opts, |_| {}).await?;
            ticker.abort();
            let status = index::status(&db, Some(p.id))?;
            let pending: i64 = status.stages.iter().map(|s| s.pending).sum();
            Ok(json!({
                "summary": summary,
                "status": status,
                "pending_jobs": pending,
                "next": if pending > 0 { "call index again to continue" } else { "indexing complete" },
            }))
        }
        "index_status" => {
            let db = open_db(paths)?;
            let p = project(&db, args)?;
            let status = index::status(&db, Some(p.id))?;
            let videos = if args.get("videos").and_then(|v| v.as_bool()).unwrap_or(false) {
                serde_json::to_value(index::videos(&db, Some(p.id))?)?
            } else {
                Value::Null
            };
            Ok(json!({ "status": status, "videos": videos }))
        }
        "search" => {
            let db = open_db(paths)?;
            let p = project(&db, args)?;
            let query = arg_str(args, "query")?;
            let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
            let kinds = args
                .get("kinds")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|k| k.as_str().map(str::to_string)).collect::<Vec<_>>());
            let mut embedder = None;
            if !args.get("keywords_only").and_then(|v| v.as_bool()).unwrap_or(false) {
                let config = Config::load(&paths.config_file).unwrap_or_default();
                let setup = runtime::resolve_embed(paths, &config).await;
                match runtime::start_embedder(&setup, |_, _| {}).await {
                    Ok(e) => embedder = Some(e),
                    Err(why) => eprintln!("mcp: meaning search unavailable ({why}); keywords only"),
                }
            }
            let opts = ghostreel_core::search::SearchOptions { project_id: Some(p.id), limit, kinds };
            let hits = ghostreel_core::search::search(&db, &paths.data_dir, &query, embedder.as_mut(), &opts).await?;
            Ok(serde_json::to_value(hits)?)
        }
        "get_transcript" => {
            let db = open_db(paths)?;
            let video_id = arg_i64(args, "video_id")?;
            let start = args.get("start_s").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let end = args.get("end_s").and_then(|v| v.as_f64()).unwrap_or(f64::MAX);
            let segs = index::transcript(&db, video_id)?
                .into_iter()
                .filter(|s| s.end > start && s.start < end)
                .collect::<Vec<_>>();
            Ok(serde_json::to_value(segs)?)
        }
        "get_video" => {
            let db = open_db(paths)?;
            let video_id = arg_i64(args, "video_id")?;
            let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(40) as usize;
            let frames = index::frames(&db, &paths.data_dir, video_id)?;
            let step = frames.len().div_ceil(limit.max(1)).max(1);
            let sampled: Vec<Value> = frames
                .iter()
                .step_by(step)
                .map(|f| {
                    let desc = f
                        .description
                        .as_deref()
                        .and_then(|j| serde_json::from_str::<Value>(j).ok())
                        .and_then(|v| v.get("description").and_then(|d| d.as_str()).map(str::to_string));
                    json!({ "t_s": f.t_s, "description": desc, "visible_text": f.visible_text })
                })
                .collect();
            let row = index::videos(&db, None)?.into_iter().find(|v| v.id == video_id);
            Ok(json!({ "video": row, "keyframes": sampled, "keyframes_total": frames.len() }))
        }
        "list_scripts" => {
            let db = open_db(paths)?;
            let p = project(&db, args)?;
            Ok(serde_json::to_value(script::list(&db, p.id)?)?)
        }
        "get_script" => {
            let db = open_db(paths)?;
            Ok(serde_json::to_value(script::load(&db, arg_i64(args, "script_id")?)?)?)
        }
        "draft_script" => {
            let db = open_db(paths)?;
            let p = project(&db, args)?;
            let message = arg_str(args, "message")?;
            let session_id = args.get("session_id").and_then(|v| v.as_i64());
            let config = Config::load(&paths.config_file).unwrap_or_default();
            let vision = runtime::resolve_chat(paths, &config).await;
            let backend = ghostreel_core::chat::ChatBackend::from_vision_setup(&vision).await?;
            let embed_setup = runtime::resolve_embed(paths, &config).await;
            let embedder = runtime::start_embedder(&embed_setup, |_, _| {}).await.ok();
            let mut ctx = ghostreel_core::chat::ChatContext {
                db,
                data_dir: paths.data_dir.clone(),
                backend,
                embedder,
                system_prompt: Some(config.chat.system_prompt.clone()),
            };
            let turn = ghostreel_core::chat::run_turn(&mut ctx, p.id, session_id, &message, &mut |_| {}).await?;
            Ok(serde_json::to_value(turn)?)
        }
        "save_script" => {
            let db = open_db(paths)?;
            let p = project(&db, args)?;
            let raw = args.get("script").context("missing 'script' object")?;
            let mut s: script::Script =
                serde_json::from_value(raw.clone()).context("script JSON does not match the schema")?;
            s.fill_from_project(&p);
            let issues = script::validate(&db, p.id, &s)?;
            let errors = issues.iter().filter(|i| i.severity == script::IssueSeverity::Error).count();
            let script_id = script::save_version(&db, p.id, &s, None)?;
            Ok(json!({
                "script_id": script_id,
                "issues": issues,
                "errors": errors,
                "total_duration_s": s.total_duration_s(),
                "clips": s.clip_count(),
            }))
        }
        "export_script" => {
            let db = open_db(paths)?;
            let script_id = arg_i64(args, "script_id")?;
            let format = args.get("format").and_then(|v| v.as_str()).unwrap_or("fcp_xml");
            let fmt = ghostreel_core::export::ExportFormat::from_str(format)?;
            let out = PathBuf::from(arg_str(args, "out")?);
            let res = ghostreel_core::export::export_script(&db, &paths.data_dir, script_id, fmt, &out)?;
            Ok(json!({ "path": res.path, "format": format }))
        }
        "delete_project" => {
            let mut db = open_db(paths)?;
            let p = project(&db, args)?;
            let purge = args.get("purge").and_then(|v| v.as_bool()).unwrap_or(false);
            let stats = if purge { db.purge_project_data(&paths.data_dir, p.id)? } else { Default::default() };
            db.remove_project(p.id)?;
            Ok(json!({ "removed": p.name, "purged": stats }))
        }
        "set_video_role" => {
            let db = open_db(paths)?;
            let p = project(&db, args)?;
            let video_id = arg_i64(args, "video_id")?;
            match arg_str(args, "role")?.as_str() {
                "footage" => db.include_video(p.id, video_id)?,
                role => db.exclude_video(p.id, video_id, role)?,
            }
            Ok(json!({ "video_id": video_id, "project": p.name }))
        }
        other => anyhow::bail!("unknown tool '{other}'"),
    }
}
