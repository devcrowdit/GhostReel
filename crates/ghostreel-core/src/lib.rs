//! GhostReel core — shared by the desktop app and the `ghostreel` CLI.
//!
//! See `.agents/plan.md` for the architecture. M0 provides configuration, paths, the index
//! database, AI server probing and the doctor report.

pub mod config;
pub mod db;
pub mod doctor;
pub mod index;
pub mod media;
pub mod paths;
pub mod probe;
pub mod progress;
pub mod projects;
pub mod watch;

use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}: {1}")]
    Io(PathBuf, #[source] std::io::Error),
    #[error("invalid config: {0}")]
    Config(String),
    #[error("database: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("database schema v{found} is newer than this GhostReel supports (v{supported}); update GhostReel")]
    SchemaTooNew { found: u32, supported: u32 },
    #[error("database migration failed: {0}")]
    Migration(String),
    #[error("{0}")]
    Invalid(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("ffprobe: {0}")]
    Probe(String),
    #[error("another GhostReel process is already indexing ({0})")]
    Busy(String),
    #[error("cannot determine the user's {0} directory")]
    NoHomeDir(&'static str),
}
