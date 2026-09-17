//! Where GhostReel keeps its files.
//!
//! - config: `<config_dir>/ghostreel/config.toml` (`~/.config` on Linux, `%APPDATA%` on Windows)
//! - data:   `<data_local_dir>/ghostreel/` (`~/.local/share`, `%LOCALAPPDATA%` — models are GBs,
//!   so never the roaming profile), holding `ghostreel.db`, `thumbs/`, `models/`.
//!
//! `GHOSTREEL_CONFIG` / `GHOSTREEL_DATA` override both (tests, portable installs).

use std::path::PathBuf;

use serde::Serialize;

use crate::Error;

#[derive(Debug, Clone, Serialize)]
pub struct Paths {
    pub config_file: PathBuf,
    pub data_dir: PathBuf,
}

impl Paths {
    pub fn resolve() -> Result<Self, Error> {
        let config_file = match std::env::var_os("GHOSTREEL_CONFIG") {
            Some(p) => PathBuf::from(p),
            None => dirs::config_dir()
                .ok_or(Error::NoHomeDir("config"))?
                .join("ghostreel")
                .join("config.toml"),
        };
        let data_dir = match std::env::var_os("GHOSTREEL_DATA") {
            Some(p) => PathBuf::from(p),
            None => dirs::data_local_dir()
                .ok_or(Error::NoHomeDir("data"))?
                .join("ghostreel"),
        };
        Ok(Self { config_file, data_dir })
    }

    pub fn db_file(&self) -> PathBuf {
        self.data_dir.join("ghostreel.db")
    }

    pub fn models_dir(&self) -> PathBuf {
        self.data_dir.join("models")
    }

    pub fn thumbs_dir(&self) -> PathBuf {
        self.data_dir.join("thumbs")
    }
}
