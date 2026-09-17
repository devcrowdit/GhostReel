//! GhostReel core — shared by the desktop app and the `ghostreel` CLI.
//!
//! See `.agents/plan.md` for the architecture. M0 provides configuration, paths, the index
//! database, AI server probing and the doctor report.

pub mod config;
pub mod db;
pub mod doctor;
pub mod paths;
pub mod probe;

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
    #[error("cannot determine the user's {0} directory")]
    NoHomeDir(&'static str),
}
