//! Delegate vision / chat to an installed coding-agent CLI (claude, agy, opencode).
//!
//! Each tool has a different invocation shape:
//! - **claude**: `claude -p "<prompt>" --output-format json --allowedTools Read [--model <m>]`
//!   stdout is JSON; answer is `result` (a string).
//! - **agy**: `agy --dangerously-skip-permissions --add-dir <dir> --output-format json [--model <m>] -p "<prompt>"`
//!   stdout is JSON; answer is `response` (a string).
//! - **opencode**: `opencode run [-m <provider/model>] "<prompt>"`
//!   plain stdout; ignore `[opencode-*]` plugin log lines; answer is the first `{...}` object.

use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::Error;
use crate::config::CliAgentConfig;

/// Resolves a binary path for a named CLI tool.
/// Checks `command` field first, then `GHOSTREEL_<TOOL>` env var, then PATH.
pub fn find_binary(cfg: &CliAgentConfig) -> Option<PathBuf> {
    // Explicit command wins.
    if !cfg.command.is_empty() {
        let p = PathBuf::from(&cfg.command);
        if p.is_file() {
            return Some(p);
        }
        return None;
    }
    // Honour the GHOSTREEL_<TOOL> env override (same pattern as doctor::locate).
    let tool = if cfg.tool.is_empty() { return None } else { &cfg.tool };
    if let Some(p) = std::env::var_os(format!("GHOSTREEL_{}", tool.to_uppercase())) {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Some(p);
        }
    }
    // Fall back to PATH.
    let exe = if cfg!(windows) { format!("{tool}.exe") } else { tool.to_string() };
    std::env::split_paths(&std::env::var_os("PATH")?).map(|d| d.join(&exe)).find(|p| p.is_file())
}

pub struct CliAgent {
    pub cfg: CliAgentConfig,
}

impl CliAgent {
    pub fn new(cfg: CliAgentConfig) -> Self {
        Self { cfg }
    }

    /// Resolve binary or return `None`.
    pub fn available(&self) -> Option<PathBuf> {
        find_binary(&self.cfg)
    }

    /// Describe one generated test image, so the user can check the CLI works (and see what it
    /// costs) before starting an index run that calls it once per keyframe.
    pub async fn self_test(&self, data_dir: &Path) -> Result<String, Error> {
        let dir = data_dir.join("tmp");
        std::fs::create_dir_all(&dir).map_err(|e| Error::Io(dir.clone(), e))?;
        let path = dir.join("cli-agent-test.jpg");
        // A red square with a green stripe: enough for the answer to show the model really looked.
        let mut img = image::RgbImage::from_pixel(320, 180, image::Rgb([200, 30, 30]));
        for y in 70..110 {
            for x in 0..320 {
                img.put_pixel(x, y, image::Rgb([30, 180, 60]));
            }
        }
        img.save(&path).map_err(|e| Error::Vision(format!("writing the test image: {e}")))?;
        let started = std::time::Instant::now();
        let json = self.describe(&path, &serde_json::to_string(&crate::vision::schema()).unwrap_or_default()).await?;
        let _ = std::fs::remove_file(&path);
        Ok(format!("{} answered in {:.1} s:\n{json}", self.cfg.tool, started.elapsed().as_secs_f64()))
    }

    /// Build argv for a describe call (image path + text prompt).
    fn build_argv_describe(&self, bin: &Path, image: &Path, prompt: &str) -> Vec<std::ffi::OsString> {
        let mut args: Vec<std::ffi::OsString> = Vec::new();
        match self.cfg.tool.as_str() {
            "claude" => {
                args.push(bin.as_os_str().to_owned());
                args.push("-p".into());
                let full_prompt =
                    format!("Read the image file {} and reply with ONLY compact JSON: {}", image.display(), prompt);
                args.push(full_prompt.into());
                args.push("--output-format".into());
                args.push("json".into());
                args.push("--allowedTools".into());
                args.push("Read".into());
                if !self.cfg.model.is_empty() {
                    args.push("--model".into());
                    args.push(self.cfg.model.clone().into());
                }
            }
            "agy" => {
                args.push(bin.as_os_str().to_owned());
                args.push("--dangerously-skip-permissions".into());
                // Grant access to the directory containing the frame.
                if let Some(dir) = image.parent() {
                    args.push("--add-dir".into());
                    args.push(dir.as_os_str().to_owned());
                }
                args.push("--output-format".into());
                args.push("json".into());
                if !self.cfg.model.is_empty() {
                    args.push("--model".into());
                    args.push(self.cfg.model.clone().into());
                }
                args.push("-p".into());
                let full_prompt =
                    format!("Read the image file {} and reply with ONLY compact JSON: {}", image.display(), prompt);
                args.push(full_prompt.into());
            }
            "opencode" => {
                args.push(bin.as_os_str().to_owned());
                args.push("run".into());
                if !self.cfg.model.is_empty() {
                    args.push("-m".into());
                    args.push(self.cfg.model.clone().into());
                }
                let full_prompt =
                    format!("Read the image file {} and reply with ONLY compact JSON: {}", image.display(), prompt);
                args.push(full_prompt.into());
            }
            other => {
                // Unknown tool — return an empty argv; the caller will error.
                tracing_warn(other);
            }
        }
        for extra in &self.cfg.extra_args {
            args.push(extra.clone().into());
        }
        args
    }

    /// Build argv for a text-only complete call.
    fn build_argv_complete(&self, bin: &Path, prompt: &str) -> Vec<std::ffi::OsString> {
        let mut args: Vec<std::ffi::OsString> = Vec::new();
        match self.cfg.tool.as_str() {
            "claude" => {
                args.push(bin.as_os_str().to_owned());
                args.push("-p".into());
                args.push(prompt.into());
                args.push("--output-format".into());
                args.push("json".into());
                args.push("--allowedTools".into());
                args.push("none".into());
                if !self.cfg.model.is_empty() {
                    args.push("--model".into());
                    args.push(self.cfg.model.clone().into());
                }
            }
            "agy" => {
                args.push(bin.as_os_str().to_owned());
                args.push("--dangerously-skip-permissions".into());
                args.push("--output-format".into());
                args.push("json".into());
                if !self.cfg.model.is_empty() {
                    args.push("--model".into());
                    args.push(self.cfg.model.clone().into());
                }
                args.push("-p".into());
                args.push(prompt.into());
            }
            "opencode" => {
                args.push(bin.as_os_str().to_owned());
                args.push("run".into());
                if !self.cfg.model.is_empty() {
                    args.push("-m".into());
                    args.push(self.cfg.model.clone().into());
                }
                args.push(prompt.into());
            }
            other => {
                tracing_warn(other);
            }
        }
        for extra in &self.cfg.extra_args {
            args.push(extra.clone().into());
        }
        args
    }

    /// Run a CLI and return stdout.
    async fn run_cli(&self, bin: &Path, args: &[std::ffi::OsString]) -> Result<String, Error> {
        if args.is_empty() {
            return Err(Error::Vision(format!("unknown CLI tool '{}'", self.cfg.tool)));
        }
        // args[0] is the binary itself; pass args[1..] to the command.
        let mut cmd = crate::proc::command(bin);
        for a in args.iter().skip(1) {
            cmd.arg(a);
        }
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());
        // ETXTBSY: the binary was written moments ago (a fresh install, or a test fixture) and the
        // kernel still holds it open. One short retry is enough.
        let child = match cmd.spawn() {
            Err(e) if e.raw_os_error() == Some(26) => {
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                cmd.spawn()
            }
            other => other,
        }
        .map_err(|e| Error::Vision(format!("cannot run {}: {e}", bin.display())))?;
        let output = tokio::time::timeout(Duration::from_secs(self.cfg.timeout_secs), child.wait_with_output())
            .await
            .map_err(|_| Error::Vision(format!("{} timed out after {}s", self.cfg.tool, self.cfg.timeout_secs)))?
            .map_err(|e| Error::Vision(format!("{} I/O error: {e}", self.cfg.tool)))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let preview: String = stderr.chars().take(300).collect();
            return Err(Error::Vision(format!("{} exited {:?}: {preview}", self.cfg.tool, output.status.code())));
        }
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    }

    /// Extract the model's text answer from the tool's stdout.
    fn extract_text(tool: &str, raw: &str) -> Result<String, Error> {
        match tool {
            "claude" => {
                // stdout is JSON: {"result": "<answer>", ...}
                let v: serde_json::Value = serde_json::from_str(raw)
                    .map_err(|e| Error::Vision(format!("claude: bad JSON: {e}: {}", preview(raw))))?;
                let text = v["result"]
                    .as_str()
                    .ok_or_else(|| Error::Vision(format!("claude: no `result` field: {}", preview(raw))))?;
                Ok(text.to_string())
            }
            "agy" => {
                // stdout is JSON: {"response": "<answer>", ...}
                let v: serde_json::Value = serde_json::from_str(raw)
                    .map_err(|e| Error::Vision(format!("agy: bad JSON: {e}: {}", preview(raw))))?;
                let text = v["response"]
                    .as_str()
                    .ok_or_else(|| Error::Vision(format!("agy: no `response` field: {}", preview(raw))))?;
                Ok(text.to_string())
            }
            "opencode" => {
                // Plain stdout; plugin log lines start with `[opencode-…]`. Skip them.
                let filtered: Vec<&str> = raw
                    .lines()
                    .filter(|l| {
                        let t = l.trim();
                        !t.starts_with("[opencode-") && !t.is_empty()
                    })
                    .collect();
                Ok(filtered.join("\n"))
            }
            other => Err(Error::Vision(format!("unknown CLI tool '{other}'"))),
        }
    }

    /// Pull the first `{...}` JSON object out of `text`, stripping any ``` fences or prose.
    pub fn extract_json_object(text: &str) -> Result<String, Error> {
        // Strip ```json ... ``` or ``` ... ``` fences.
        let stripped = strip_code_fences(text);
        let haystack = stripped.as_deref().unwrap_or(text);
        let start = haystack.find('{');
        let end = haystack.rfind('}');
        match (start, end) {
            (Some(s), Some(e)) if e > s => Ok(haystack[s..=e].to_string()),
            _ => Err(Error::Vision(format!("no JSON object in CLI output: {}", preview(text)))),
        }
    }

    /// Describe a frame image: returns the extracted JSON string (matching `vision::parse_description`).
    pub async fn describe(&self, image: &Path, schema_hint: &str) -> Result<String, Error> {
        let bin =
            self.available().ok_or_else(|| Error::Vision(format!("CLI tool '{}' not found on PATH", self.cfg.tool)))?;
        let args = self.build_argv_describe(&bin, image, schema_hint);
        let raw = self.run_cli(&bin, &args).await?;
        let text = Self::extract_text(&self.cfg.tool, &raw)?;
        Self::extract_json_object(&text)
    }

    /// Send a plain text prompt and return the first JSON object in the response.
    pub async fn complete(&self, prompt: &str) -> Result<String, Error> {
        let bin =
            self.available().ok_or_else(|| Error::Vision(format!("CLI tool '{}' not found on PATH", self.cfg.tool)))?;
        let args = self.build_argv_complete(&bin, prompt);
        let raw = self.run_cli(&bin, &args).await?;
        let text = Self::extract_text(&self.cfg.tool, &raw)?;
        Self::extract_json_object(&text)
    }
}

fn tracing_warn(tool: &str) {
    // Avoid a hard dependency on `tracing`; just write to stderr in debug builds.
    #[cfg(debug_assertions)]
    eprintln!("cliagent: unknown tool '{tool}'");
    let _ = tool;
}

fn preview(s: &str) -> String {
    let p: String = s.chars().take(200).collect();
    if s.chars().count() > 200 { format!("{p}…") } else { p }
}

/// Strip leading ``` fences and return the inner text if any fence was found.
fn strip_code_fences(s: &str) -> Option<String> {
    let trimmed = s.trim();
    let after_open = trimmed.strip_prefix("```json").or_else(|| trimmed.strip_prefix("```"))?;
    let inner = match after_open.rfind("```") {
        Some(pos) => &after_open[..pos],
        None => after_open,
    };
    Some(inner.trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CliAgentConfig;

    #[cfg(unix)]
    fn write_fake_cli(dir: &Path, name: &str, script: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let p = dir.join(name);
        // Write, close, then chmod: exec'ing a file whose handle is still open fails with ETXTBSY.
        {
            use std::io::Write as _;
            let mut f = std::fs::File::create(&p).unwrap();
            f.write_all(format!("#!/bin/sh\n{script}").as_bytes()).unwrap();
            f.sync_all().unwrap();
        }
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
        p
    }

    fn cfg(tool: &str, command: &Path) -> CliAgentConfig {
        CliAgentConfig {
            tool: tool.to_string(),
            command: command.to_str().unwrap_or("").to_string(),
            timeout_secs: 30,
            concurrency: 1,
            ..Default::default()
        }
    }

    // ---- extract_text --------------------------------------------------------

    #[test]
    fn extract_text_claude_shape() {
        let raw = r#"{"result": "{\"description\":\"a cat\"}", "cost_usd": 0.04}"#;
        let text = CliAgent::extract_text("claude", raw).unwrap();
        assert_eq!(text, r#"{"description":"a cat"}"#);
    }

    #[test]
    fn extract_text_agy_shape() {
        let raw = r#"{"response": "{\"description\":\"a dog\"}", "duration_ms": 1200}"#;
        let text = CliAgent::extract_text("agy", raw).unwrap();
        assert_eq!(text, r#"{"description":"a dog"}"#);
    }

    #[test]
    fn extract_text_opencode_strips_plugin_lines() {
        let raw = "[opencode-lmstudio] connecting…\n{\"description\":\"a bird\"}\n";
        let text = CliAgent::extract_text("opencode", raw).unwrap();
        assert!(text.contains(r#"{"description":"a bird"}"#), "got: {text}");
    }

    // ---- extract_json_object -------------------------------------------------

    #[test]
    fn extract_json_object_from_plain_json() {
        let s = r#"{"description":"test","visible_text":[]}"#;
        assert_eq!(CliAgent::extract_json_object(s).unwrap(), s);
    }

    #[test]
    fn extract_json_object_from_fenced() {
        let s = "```json\n{\"description\":\"test\"}\n```";
        assert_eq!(CliAgent::extract_json_object(s).unwrap(), r#"{"description":"test"}"#);
    }

    #[test]
    fn extract_json_object_strips_prose() {
        let s = "Here you go: {\"description\":\"hi\"} thanks";
        assert_eq!(CliAgent::extract_json_object(s).unwrap(), r#"{"description":"hi"}"#);
    }

    #[test]
    fn extract_json_object_no_json_is_err() {
        assert!(CliAgent::extract_json_object("no json here").is_err());
    }

    // ---- integration: fake CLI scripts (unix only) ---------------------------

    #[cfg(unix)]
    #[tokio::test]
    async fn fake_claude_describe() {
        let tmp = tempfile::tempdir().unwrap();
        let img = tmp.path().join("frame.jpg");
        std::fs::write(&img, b"fake").unwrap();

        // Fake claude: echo the JSON response format
        let answer = r#"{\"description\":\"two boards\",\"visible_text\":[],\"objects\":[\"board\"],\"setting\":\"desk\",\"shot\":\"close-up\",\"tags\":[]}"#;
        let script = format!(r#"echo '{{"result": "{answer}", "cost_usd": 0.04}}'"#);
        let bin = write_fake_cli(tmp.path(), "claude", &script);

        let agent = CliAgent::new(cfg("claude", &bin));
        let json = agent.describe(&img, "schema").await.unwrap();
        assert!(json.contains("two boards"), "got: {json}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn fake_agy_describe() {
        let tmp = tempfile::tempdir().unwrap();
        let img = tmp.path().join("frame.jpg");
        std::fs::write(&img, b"fake").unwrap();

        let answer = r#"{\"description\":\"a mountain\",\"visible_text\":[],\"objects\":[\"peak\"],\"setting\":\"outdoors\",\"shot\":\"wide\",\"tags\":[]}"#;
        let script = format!(r#"echo '{{"response": "{answer}", "duration_ms": 1200}}'"#);
        let bin = write_fake_cli(tmp.path(), "agy", &script);

        let agent = CliAgent::new(cfg("agy", &bin));
        let json = agent.describe(&img, "schema").await.unwrap();
        assert!(json.contains("a mountain"), "got: {json}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn fake_opencode_describe_with_noise() {
        let tmp = tempfile::tempdir().unwrap();
        let img = tmp.path().join("frame.jpg");
        std::fs::write(&img, b"fake").unwrap();

        // opencode emits plugin noise then the JSON
        let script = r#"printf '[opencode-lmstudio] loading...\n{"description":"a desk","visible_text":[],"objects":["laptop"],"setting":"office","shot":"medium","tags":[]}\n'"#;
        let bin = write_fake_cli(tmp.path(), "opencode", script);

        let agent = CliAgent::new(cfg("opencode", &bin));
        let json = agent.describe(&img, "schema").await.unwrap();
        assert!(json.contains("a desk"), "got: {json}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn fenced_json_response_is_handled() {
        let tmp = tempfile::tempdir().unwrap();
        let img = tmp.path().join("frame.jpg");
        std::fs::write(&img, b"fake").unwrap();

        // claude returns fenced JSON in the result field
        let result_val = r#"```json\n{\"description\":\"a monitor\",\"visible_text\":[],\"objects\":[],\"setting\":\"office\",\"shot\":\"close-up\",\"tags\":[]}\n```"#;
        let script = format!(r#"echo '{{"result": "{result_val}"}}'  "#);
        let bin = write_fake_cli(tmp.path(), "claude", &script);

        let agent = CliAgent::new(cfg("claude", &bin));
        let json = agent.describe(&img, "schema").await.unwrap();
        assert!(json.contains("a monitor"), "got: {json}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn non_zero_exit_is_error() {
        let tmp = tempfile::tempdir().unwrap();
        let img = tmp.path().join("frame.jpg");
        std::fs::write(&img, b"fake").unwrap();

        let bin = write_fake_cli(tmp.path(), "claude", "exit 1");
        let agent = CliAgent::new(cfg("claude", &bin));
        let err = agent.describe(&img, "schema").await.unwrap_err();
        assert!(err.to_string().contains("exited"), "got: {err}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn timeout_is_error() {
        let tmp = tempfile::tempdir().unwrap();
        let img = tmp.path().join("frame.jpg");
        std::fs::write(&img, b"fake").unwrap();

        let bin = write_fake_cli(tmp.path(), "claude", "sleep 60");
        let short_cfg = CliAgentConfig {
            tool: "claude".to_string(),
            command: bin.to_str().unwrap_or("").to_string(),
            timeout_secs: 1,
            concurrency: 1,
            ..Default::default()
        };
        let agent = CliAgent::new(short_cfg);
        let err = agent.describe(&img, "schema").await.unwrap_err();
        assert!(err.to_string().contains("timed out"), "got: {err}");
    }

    #[test]
    fn find_binary_uses_command_field() {
        let tmp = tempfile::tempdir().unwrap();
        // A non-existent path: should return None.
        let missing_cfg = CliAgentConfig {
            tool: "claude".into(),
            command: tmp.path().join("nonexistent").to_str().unwrap_or("").into(),
            ..Default::default()
        };
        assert!(find_binary(&missing_cfg).is_none());

        // An existing file: should return Some.
        let real = tmp.path().join("mybin");
        std::fs::write(&real, b"fake").unwrap();
        let real_cfg =
            CliAgentConfig { tool: "claude".into(), command: real.to_str().unwrap_or("").into(), ..Default::default() };
        assert_eq!(find_binary(&real_cfg), Some(real));
    }
}
