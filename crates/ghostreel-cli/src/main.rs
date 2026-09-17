//! `ghostreel` — headless GhostReel CLI.

use std::process::ExitCode;

use clap::{Parser, Subcommand};
use ghostreel_core::config::Config;
use ghostreel_core::doctor::{self, Report};
use ghostreel_core::paths::Paths;
use ghostreel_core::probe::{Resolution, Target};

#[derive(Parser)]
#[command(name = "ghostreel", version, about = "Search inside your videos — locally")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Check ffmpeg, GPU, database, models and the AI backends GhostReel would use.
    Doctor {
        /// Machine-readable output.
        #[arg(long)]
        json: bool,
    },
    /// Show or create the configuration.
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },
}

#[derive(Subcommand)]
enum ConfigAction {
    /// Print the config file path.
    Path,
    /// Print the effective configuration (defaults applied).
    Show,
    /// Write the default configuration if no config file exists yet.
    Init,
}

#[tokio::main]
async fn main() -> ExitCode {
    match run(Cli::parse()).await {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

async fn run(cli: Cli) -> anyhow::Result<ExitCode> {
    let paths = Paths::resolve()?;
    match cli.command {
        Command::Doctor { json } => {
            let report = doctor::run(&paths).await;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                print_report(&report);
            }
            Ok(if report.blockers().is_empty() { ExitCode::SUCCESS } else { ExitCode::from(2) })
        }
        Command::Config { action } => {
            match action {
                ConfigAction::Path => println!("{}", paths.config_file.display()),
                ConfigAction::Show => {
                    let cfg = Config::load(&paths.config_file)?;
                    print!("{}", cfg.to_toml()?);
                }
                ConfigAction::Init => {
                    if paths.config_file.exists() {
                        println!("exists: {}", paths.config_file.display());
                    } else {
                        Config::default().save(&paths.config_file)?;
                        println!("created: {}", paths.config_file.display());
                    }
                }
            }
            Ok(ExitCode::SUCCESS)
        }
    }
}

fn mark(ok: bool) -> &'static str {
    if ok { "✓" } else { "✗" }
}

fn print_backend(name: &str, r: &Resolution) {
    let target = match r.target {
        Target::Server => "server",
        Target::Local => "local",
        Target::Unavailable => "UNAVAILABLE",
    };
    let where_ = r
        .probe
        .as_ref()
        .filter(|_| r.target == Target::Server)
        .map(|p| format!(" {} @ {}", p.model.as_deref().unwrap_or("?"), p.url))
        .unwrap_or_default();
    println!(
        "  {} {name:<11} {target}{where_}  [backend = {}] {}",
        mark(r.target != Target::Unavailable),
        r.backend,
        r.reason
    );
}

fn print_report(r: &Report) {
    println!("GhostReel {}", r.version);
    println!("  config  {}{}", r.config_file.display(), r.config_error.as_ref().map(|e| format!("  ✗ {e}")).unwrap_or_default());
    println!("  data    {}", r.data_dir.display());

    println!("\nDatabase");
    match r.db.ok {
        true => println!(
            "  ✓ {} (schema v{}, sqlite-vec {})",
            r.db.path.display(),
            r.db.schema_version.unwrap_or(0),
            r.db.sqlite_vec.as_deref().unwrap_or("?")
        ),
        false => println!("  ✗ {}: {}", r.db.path.display(), r.db.error.as_deref().unwrap_or("?")),
    }

    println!("\nTools");
    for t in [&r.ffmpeg, &r.ffprobe] {
        match &t.path {
            Some(p) => println!("  ✓ {:<8} {} ({})", t.name, t.version.as_deref().unwrap_or("?"), p.display()),
            None => println!("  ✗ {:<8} not found", t.name),
        }
    }

    println!("\nGPU");
    if r.gpu.is_empty() {
        println!("  - no NVIDIA GPU detected (local models would run on CPU)");
    }
    for g in &r.gpu {
        println!(
            "  ✓ {} — {} / {} MiB used, driver {}",
            g.name, g.vram_used_mib, g.vram_total_mib, g.driver
        );
    }

    println!("\nAI backends");
    print_backend("vision", &r.vision);
    print_backend("embeddings", &r.embeddings);
    print_backend("stt", &r.stt);

    println!("\nLocal model files");
    for m in &r.models {
        match &m.found {
            Some(p) => println!("  ✓ {:<25} {}", m.role, p.display()),
            None => println!("  - {:<25} {} (not downloaded)", m.role, m.pattern),
        }
    }

    let blockers = r.blockers();
    println!();
    if blockers.is_empty() {
        println!("Ready.");
    } else {
        println!("Blockers:");
        for b in blockers {
            println!("  ✗ {b}");
        }
    }
}
