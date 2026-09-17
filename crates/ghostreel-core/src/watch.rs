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
        let roots: Vec<PathBuf> = folders.iter().map(|(p, _)| p.clone()).collect();
        let mut watcher = notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
            if let Ok(ev) = res {
                // Access events (reads) never change the library, and neither do Premiere renders,
                // caches or hidden files, which editing apps rewrite constantly.
                if !matches!(ev.kind, notify::EventKind::Access(_))
                    && !(!ev.paths.is_empty() && ev.paths.iter().all(|p| in_ignored_dir(p, &roots)))
                {
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

/// Whether `path` lies in a folder the scan skips (below the watched root it belongs to).
fn in_ignored_dir(path: &std::path::Path, roots: &[PathBuf]) -> bool {
    let Some(rel) = roots.iter().find_map(|r| path.strip_prefix(r).ok()) else { return false };
    let mut parts: Vec<_> = rel.components().map(|c| c.as_os_str().to_string_lossy().to_string()).collect();
    let file = parts.pop();
    parts.iter().any(|d| crate::media::is_ignored_dir(d)) || file.is_some_and(|f| f.starts_with('.'))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ignores_editor_renders_below_the_root_only() {
        let roots = vec![PathBuf::from("/home/me/.shoots/Greet")];
        let r = |p: &str| in_ignored_dir(std::path::Path::new(p), &roots);
        assert!(!r("/home/me/.shoots/Greet/DJI_0065.MP4"), "a hidden folder above the root doesn't count");
        assert!(r("/home/me/.shoots/Greet/Adobe Premiere Pro Video Previews/Seq.PRV/a.mov"));
        assert!(r("/home/me/.shoots/Greet/edit/Media Cache Files/x.mpeg"));
        assert!(r("/home/me/.shoots/Greet/.DS_Store"));
        assert!(!r("/home/me/.shoots/Greet/Premiere exports/final.mp4"));
    }

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
