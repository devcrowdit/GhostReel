//! Filesystem watching for `index --watch` and the app: turns bursts of notify events under the
//! watched folders into one "something changed" signal after a quiet period.

use std::path::PathBuf;
use std::time::Duration;

use notify::{RecursiveMode, Watcher};
use tokio::sync::mpsc;

use crate::Error;

pub struct FolderWatcher {
    _watcher: notify::RecommendedWatcher,
    rx: mpsc::UnboundedReceiver<()>,
}

impl FolderWatcher {
    /// Watch `folders` (recursive flag per folder). Missing folders are skipped.
    pub fn new(folders: &[(PathBuf, bool)]) -> Result<Self, Error> {
        let (tx, rx) = mpsc::unbounded_channel();
        let mut watcher = notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
            if let Ok(ev) = res {
                // Access events (reads) never change the library.
                if !matches!(ev.kind, notify::EventKind::Access(_)) {
                    let _ = tx.send(());
                }
            }
        })
        .map_err(|e| Error::Invalid(format!("cannot start file watcher: {e}")))?;
        for (path, recursive) in folders {
            if path.is_dir() {
                let mode = if *recursive { RecursiveMode::Recursive } else { RecursiveMode::NonRecursive };
                watcher
                    .watch(path, mode)
                    .map_err(|e| Error::Invalid(format!("cannot watch {}: {e}", path.display())))?;
            }
        }
        Ok(Self { _watcher: watcher, rx })
    }

    /// Wait for a change, then until no further change arrives for `quiet`.
    /// Returns `false` if the watcher stopped.
    pub async fn changed(&mut self, quiet: Duration) -> bool {
        if self.rx.recv().await.is_none() {
            return false;
        }
        loop {
            match tokio::time::timeout(quiet, self.rx.recv()).await {
                Ok(Some(())) => continue,
                Ok(None) => return false,
                Err(_) => return true,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn debounces_a_burst_into_one_signal() {
        let tmp = tempfile::tempdir().unwrap();
        let mut w = FolderWatcher::new(&[(tmp.path().to_path_buf(), true)]).unwrap();
        let dir = tmp.path().to_path_buf();
        tokio::spawn(async move {
            for i in 0..5 {
                std::fs::write(dir.join(format!("f{i}.mp4")), b"x").unwrap();
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        });
        let got = tokio::time::timeout(Duration::from_secs(10), w.changed(Duration::from_millis(300))).await;
        assert_eq!(got.ok(), Some(true));
        // The burst was absorbed: nothing further pending.
        let again = tokio::time::timeout(Duration::from_millis(500), w.changed(Duration::from_millis(100))).await;
        assert!(again.is_err());
    }
}
