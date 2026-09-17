//! Projects and their watched folders (plan D13).
//!
//! A folder row is shared: several projects can watch the same directory, and its videos are
//! indexed once. Removing a folder from its last project deletes the folder (and its file
//! locations); the videos' index data stays, so re-adding the footage later is instant.

use std::path::{Path, PathBuf};

use rusqlite::{OptionalExtension, params};
use serde::Serialize;

use crate::Error;
use crate::db::Db;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Project {
    pub id: i64,
    pub name: String,
    pub description: String,
    pub fps_num: i64,
    pub fps_den: i64,
    pub width: i64,
    pub height: i64,
    pub created_at: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Folder {
    pub id: i64,
    pub path: PathBuf,
    pub recursive: bool,
    pub enabled: bool,
}

/// Sequence settings for a new project (used by the M8 timeline export).
#[derive(Debug, Clone)]
pub struct NewProject {
    pub name: String,
    pub description: String,
    pub fps_num: i64,
    pub fps_den: i64,
    pub width: i64,
    pub height: i64,
}

impl NewProject {
    pub fn named(name: impl Into<String>) -> Self {
        Self { name: name.into(), description: String::new(), fps_num: 25, fps_den: 1, width: 1920, height: 1080 }
    }
}

pub fn now() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
}

fn row_to_project(r: &rusqlite::Row) -> rusqlite::Result<Project> {
    Ok(Project {
        id: r.get(0)?,
        name: r.get(1)?,
        description: r.get(2)?,
        fps_num: r.get(3)?,
        fps_den: r.get(4)?,
        width: r.get(5)?,
        height: r.get(6)?,
        created_at: r.get(7)?,
    })
}

const PROJECT_COLS: &str = "id, name, description, fps_num, fps_den, width, height, created_at";

impl Db {
    pub fn create_project(&self, p: &NewProject) -> Result<Project, Error> {
        let name = p.name.trim();
        if name.is_empty() {
            return Err(Error::Invalid("project name is empty".into()));
        }
        if p.fps_num <= 0 || p.fps_den <= 0 || p.width <= 0 || p.height <= 0 {
            return Err(Error::Invalid("fps and resolution must be positive".into()));
        }
        if self.project_by_name(name)?.is_some() {
            return Err(Error::Invalid(format!("project '{name}' already exists")));
        }
        self.conn.execute(
            "INSERT INTO projects(name, description, fps_num, fps_den, width, height, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![name, p.description, p.fps_num, p.fps_den, p.width, p.height, now()],
        )?;
        self.project(self.conn.last_insert_rowid())
    }

    pub fn projects(&self) -> Result<Vec<Project>, Error> {
        let mut st = self.conn.prepare(&format!("SELECT {PROJECT_COLS} FROM projects ORDER BY name COLLATE NOCASE"))?;
        let rows = st.query_map([], row_to_project)?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    pub fn project(&self, id: i64) -> Result<Project, Error> {
        self.conn
            .query_row(&format!("SELECT {PROJECT_COLS} FROM projects WHERE id = ?1"), [id], row_to_project)
            .optional()?
            .ok_or_else(|| Error::NotFound(format!("project #{id}")))
    }

    pub fn project_by_name(&self, name: &str) -> Result<Option<Project>, Error> {
        Ok(self
            .conn
            .query_row(
                &format!("SELECT {PROJECT_COLS} FROM projects WHERE name = ?1 COLLATE NOCASE"),
                [name.trim()],
                row_to_project,
            )
            .optional()?)
    }

    /// Look a project up by name, failing with a helpful error.
    pub fn require_project(&self, name: &str) -> Result<Project, Error> {
        self.project_by_name(name)?.ok_or_else(|| Error::NotFound(format!("project '{name}'")))
    }

    /// Rename a project. Names stay unique (case-insensitive); changing only the case is allowed.
    pub fn rename_project(&mut self, id: i64, name: &str) -> Result<Project, Error> {
        let name = name.trim();
        if name.is_empty() {
            return Err(Error::Invalid("project name is empty".into()));
        }
        if self.project_by_name(name)?.is_some_and(|p| p.id != id) {
            return Err(Error::Invalid(format!("project '{name}' already exists")));
        }
        if self.conn.execute("UPDATE projects SET name = ?1 WHERE id = ?2", params![name, id])? == 0 {
            return Err(Error::NotFound(format!("project #{id}")));
        }
        self.project(id)
    }

    /// Delete a project; folders no other project uses are removed with it.
    pub fn remove_project(&mut self, id: i64) -> Result<(), Error> {
        let tx = self.conn.transaction()?;
        let n = tx.execute("DELETE FROM projects WHERE id = ?1", [id])?;
        if n == 0 {
            return Err(Error::NotFound(format!("project #{id}")));
        }
        tx.execute("DELETE FROM folders WHERE id NOT IN (SELECT folder_id FROM project_folders)", [])?;
        tx.commit()?;
        Ok(())
    }

    /// Watch `path` for `project`. Returns the (possibly shared) folder.
    pub fn add_folder(&mut self, project_id: i64, path: &Path, recursive: bool) -> Result<Folder, Error> {
        let canonical = normalize_dir(path)?;
        let path_str = canonical.to_string_lossy().to_string();
        let tx = self.conn.transaction()?;
        // Refuse nesting within the same project: files would belong to two folders.
        let mut st = tx.prepare(
            "SELECT f.path FROM folders f JOIN project_folders pf ON pf.folder_id = f.id WHERE pf.project_id = ?1",
        )?;
        let existing: Vec<String> = st.query_map([project_id], |r| r.get(0))?.collect::<Result<_, _>>()?;
        drop(st);
        for other in &existing {
            if other == &path_str {
                continue;
            }
            let (a, b) = (Path::new(other), canonical.as_path());
            if b.starts_with(a) || a.starts_with(b) {
                return Err(Error::Invalid(format!("{} overlaps folder {other} already in this project", b.display())));
            }
        }
        tx.execute(
            "INSERT INTO folders(path, recursive, enabled, added_at) VALUES (?1, ?2, 1, ?3)
             ON CONFLICT(path) DO UPDATE SET recursive = excluded.recursive, enabled = 1",
            params![path_str, recursive, now()],
        )?;
        let folder_id: i64 = tx.query_row("SELECT id FROM folders WHERE path = ?1", [&path_str], |r| r.get(0))?;
        let n = tx.execute(
            "INSERT OR IGNORE INTO project_folders(project_id, folder_id) VALUES (?1, ?2)",
            params![project_id, folder_id],
        )?;
        if n == 0 && !existing.contains(&path_str) {
            return Err(Error::NotFound(format!("project #{project_id}")));
        }
        tx.commit()?;
        Ok(Folder { id: folder_id, path: canonical, recursive, enabled: true })
    }

    pub fn folders(&self, project_id: Option<i64>) -> Result<Vec<Folder>, Error> {
        let map = |r: &rusqlite::Row| -> rusqlite::Result<Folder> {
            Ok(Folder {
                id: r.get(0)?,
                path: PathBuf::from(r.get::<_, String>(1)?),
                recursive: r.get(2)?,
                enabled: r.get(3)?,
            })
        };
        Ok(match project_id {
            Some(pid) => {
                let mut st = self.conn.prepare(
                    "SELECT f.id, f.path, f.recursive, f.enabled FROM folders f
                     JOIN project_folders pf ON pf.folder_id = f.id WHERE pf.project_id = ?1 ORDER BY f.path",
                )?;
                st.query_map([pid], map)?.collect::<Result<_, _>>()?
            }
            None => {
                let mut st = self.conn.prepare("SELECT id, path, recursive, enabled FROM folders ORDER BY path")?;
                st.query_map([], map)?.collect::<Result<_, _>>()?
            }
        })
    }

    /// Stop watching `path` for `project`; drop the folder if no project uses it any more.
    pub fn remove_folder(&mut self, project_id: i64, path: &Path) -> Result<(), Error> {
        let path_str = normalize_dir(path).unwrap_or_else(|_| path.to_path_buf()).to_string_lossy().to_string();
        let tx = self.conn.transaction()?;
        let n = tx.execute(
            "DELETE FROM project_folders WHERE project_id = ?1
               AND folder_id = (SELECT id FROM folders WHERE path = ?2)",
            params![project_id, path_str],
        )?;
        if n == 0 {
            return Err(Error::NotFound(format!("folder {path_str} in this project")));
        }
        tx.execute("DELETE FROM folders WHERE id NOT IN (SELECT folder_id FROM project_folders)", [])?;
        tx.commit()?;
        Ok(())
    }
}

/// Absolute, symlink-resolved directory path (UNC prefix stripped on Windows so paths stay
/// readable and usable as `file://` URLs later).
pub fn normalize_dir(path: &Path) -> Result<PathBuf, Error> {
    let canonical = std::fs::canonicalize(path).map_err(|e| Error::Io(path.to_path_buf(), e))?;
    if !canonical.is_dir() {
        return Err(Error::Invalid(format!("{} is not a directory", canonical.display())));
    }
    #[cfg(windows)]
    {
        let s = canonical.to_string_lossy();
        if let Some(rest) = s.strip_prefix(r"\\?\") {
            if !rest.starts_with("UNC\\") {
                return Ok(PathBuf::from(rest));
            }
        }
    }
    Ok(canonical)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_crud_and_unique_names() {
        let mut db = Db::open_in_memory().unwrap();
        let p = db.create_project(&NewProject::named("Teaser")).unwrap();
        assert_eq!((p.fps_num, p.width), (25, 1920));
        assert!(db.create_project(&NewProject::named("teaser")).is_err(), "names are case-insensitive");
        assert!(db.create_project(&NewProject::named("  ")).is_err());
        assert_eq!(db.require_project("TEASER").unwrap().id, p.id);
        let other = db.create_project(&NewProject::named("Other")).unwrap();
        assert!(db.rename_project(other.id, "teaser").is_err(), "rename can't take another project's name");
        assert!(db.rename_project(other.id, " ").is_err());
        assert_eq!(db.rename_project(p.id, " TEASER cut ").unwrap().name, "TEASER cut");
        assert_eq!(db.rename_project(p.id, "teaser CUT").unwrap().name, "teaser CUT", "case-only change");
        db.remove_project(other.id).unwrap();
        db.remove_project(p.id).unwrap();
        assert!(db.projects().unwrap().is_empty());
    }

    #[test]
    fn folders_are_shared_and_cleaned_up() {
        let tmp = tempfile::tempdir().unwrap();
        let media = tmp.path().join("media");
        std::fs::create_dir_all(media.join("sub")).unwrap();
        let mut db = Db::open_in_memory().unwrap();
        let a = db.create_project(&NewProject::named("A")).unwrap();
        let b = db.create_project(&NewProject::named("B")).unwrap();

        let fa = db.add_folder(a.id, &media, true).unwrap();
        let fb = db.add_folder(b.id, &media, true).unwrap();
        assert_eq!(fa.id, fb.id, "same directory → one shared folder row");
        assert!(db.add_folder(a.id, &media, true).is_ok(), "re-adding is idempotent");
        assert!(db.add_folder(a.id, &media.join("sub"), true).is_err(), "nested folder rejected");
        assert!(db.add_folder(a.id, &tmp.path().join("missing"), true).is_err());

        db.remove_folder(a.id, &media).unwrap();
        assert_eq!(db.folders(None).unwrap().len(), 1, "still used by B");
        db.remove_project(b.id).unwrap();
        assert!(db.folders(None).unwrap().is_empty(), "orphan folder removed with last project");
    }
}
