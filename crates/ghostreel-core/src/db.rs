//! The index database: SQLite + FTS5 (keyword) + sqlite-vec (vectors), one file (plan D4, §4).

use std::path::Path;
use std::sync::Once;

use rusqlite::{Connection, OptionalExtension, params};

use crate::Error;
use crate::config::{EMBED_DIM, EMBED_MODEL};

/// Ordered migrations; index + 1 is the schema version it produces. Append only.
const MIGRATIONS: &[&str] = &[
    // v1 — initial schema (plan §4)
    r#"
    CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

    CREATE TABLE folders (
        id INTEGER PRIMARY KEY,
        path TEXT NOT NULL UNIQUE,
        recursive INTEGER NOT NULL DEFAULT 1,
        enabled INTEGER NOT NULL DEFAULT 1,
        added_at INTEGER NOT NULL
    );

    CREATE TABLE videos (
        id INTEGER PRIMARY KEY,
        content_hash TEXT NOT NULL UNIQUE,
        path TEXT NOT NULL,
        folder_id INTEGER REFERENCES folders(id) ON DELETE SET NULL,
        size INTEGER NOT NULL,
        mtime INTEGER NOT NULL,
        duration_s REAL,
        width INTEGER,
        height INTEGER,
        fps REAL,
        vcodec TEXT,
        acodec TEXT,
        has_audio INTEGER,
        language TEXT,
        summary TEXT,
        status TEXT NOT NULL DEFAULT 'new',
        error TEXT,
        indexed_at INTEGER
    );
    CREATE INDEX videos_path ON videos(path);

    CREATE TABLE jobs (
        video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
        stage TEXT NOT NULL,
        state TEXT NOT NULL,
        attempts INTEGER NOT NULL DEFAULT 0,
        last_error TEXT,
        updated_at INTEGER NOT NULL,
        PRIMARY KEY (video_id, stage)
    );

    CREATE TABLE transcript_segments (
        id INTEGER PRIMARY KEY,
        video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
        start_s REAL NOT NULL,
        end_s REAL NOT NULL,
        text TEXT NOT NULL,
        confidence REAL
    );
    CREATE INDEX transcript_segments_video ON transcript_segments(video_id, start_s);

    CREATE TABLE frames (
        id INTEGER PRIMARY KEY,
        video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
        t_s REAL NOT NULL,
        thumb_path TEXT,
        phash INTEGER,
        description_json TEXT,
        visible_text TEXT
    );
    CREATE INDEX frames_video ON frames(video_id, t_s);

    CREATE TABLE chunks (
        id INTEGER PRIMARY KEY,
        video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
        kind TEXT NOT NULL CHECK (kind IN ('moment', 'transcript', 'frame', 'summary')),
        start_s REAL,
        end_s REAL,
        text TEXT NOT NULL,
        frame_id INTEGER REFERENCES frames(id) ON DELETE SET NULL
    );
    CREATE INDEX chunks_video ON chunks(video_id);

    CREATE VIRTUAL TABLE chunks_fts USING fts5(
        text, content='chunks', content_rowid='id', tokenize='unicode61'
    );
    CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
        INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
    END;
    CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
    END;
    CREATE TRIGGER chunks_au AFTER UPDATE ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
        INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
    END;

    CREATE VIRTUAL TABLE chunks_vec USING vec0(embedding float[768]);
    "#,
];

static REGISTER_VEC: Once = Once::new();

/// Make sqlite-vec available to every connection opened afterwards.
fn register_sqlite_vec() {
    REGISTER_VEC.call_once(|| {
        // SAFETY: sqlite3_vec_init has the sqlite3 extension-entry signature expected by
        // sqlite3_auto_extension; registering it once, before any connection, is the
        // documented way to load a statically linked extension.
        unsafe {
            #[allow(clippy::missing_transmute_annotations)]
            rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
                sqlite_vec::sqlite3_vec_init as *const (),
            )));
        }
    });
}

pub struct Db {
    pub conn: Connection,
}

impl Db {
    /// Open (creating if needed) and migrate the database at `path`.
    pub fn open(path: &Path) -> Result<Self, Error> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| Error::Io(dir.to_path_buf(), e))?;
        }
        register_sqlite_vec();
        let conn = Connection::open(path)?;
        Self::init(conn)
    }

    pub fn open_in_memory() -> Result<Self, Error> {
        register_sqlite_vec();
        Self::init(Connection::open_in_memory()?)
    }

    fn init(conn: Connection) -> Result<Self, Error> {
        // WAL lets the app and the CLI read while one of them indexes.
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        let mut db = Self { conn };
        db.migrate()?;
        Ok(db)
    }

    pub fn schema_version(&self) -> Result<u32, Error> {
        Ok(self.conn.pragma_query_value(None, "user_version", |r| r.get(0))?)
    }

    fn migrate(&mut self) -> Result<(), Error> {
        let current = self.schema_version()? as usize;
        if current > MIGRATIONS.len() {
            return Err(Error::SchemaTooNew { found: current as u32, supported: MIGRATIONS.len() as u32 });
        }
        for (i, sql) in MIGRATIONS.iter().enumerate().skip(current) {
            let tx = self.conn.transaction()?;
            tx.execute_batch(sql)?;
            tx.pragma_update(None, "user_version", (i + 1) as u32)?;
            if i == 0 {
                tx.execute(
                    "INSERT INTO meta(key, value) VALUES ('embed_model', ?1), ('embed_dim', ?2)",
                    params![EMBED_MODEL, EMBED_DIM.to_string()],
                )?;
            }
            tx.commit()?;
        }
        Ok(())
    }

    pub fn meta(&self, key: &str) -> Result<Option<String>, Error> {
        Ok(self
            .conn
            .query_row("SELECT value FROM meta WHERE key = ?1", [key], |r| r.get(0))
            .optional()?)
    }

    /// sqlite-vec version string, proving the extension is loaded.
    pub fn vec_version(&self) -> Result<String, Error> {
        Ok(self.conn.query_row("SELECT vec_version()", [], |r| r.get(0))?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migrates_and_records_embedding_model() {
        let db = Db::open_in_memory().unwrap();
        assert_eq!(db.schema_version().unwrap(), MIGRATIONS.len() as u32);
        assert_eq!(db.meta("embed_model").unwrap().as_deref(), Some(EMBED_MODEL));
        assert_eq!(db.meta("embed_dim").unwrap().as_deref(), Some("768"));
        assert!(db.vec_version().unwrap().starts_with('v'));
    }

    #[test]
    fn reopen_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ghostreel.db");
        drop(Db::open(&path).unwrap());
        let db = Db::open(&path).unwrap();
        assert_eq!(db.schema_version().unwrap(), MIGRATIONS.len() as u32);
    }

    #[test]
    fn fts_and_vector_search_work_together() {
        let db = Db::open_in_memory().unwrap();
        let c = &db.conn;
        c.execute(
            "INSERT INTO videos(content_hash, path, size, mtime) VALUES ('h', '/v.mp4', 1, 0)",
            [],
        )
        .unwrap();
        for (text, axis) in [("person unboxing a raspberry pi", 0usize), ("cat sleeping on a sofa", 1)] {
            c.execute(
                "INSERT INTO chunks(video_id, kind, start_s, end_s, text) VALUES (1, 'moment', 0, 5, ?1)",
                [text],
            )
            .unwrap();
            let id = c.last_insert_rowid();
            let mut v = vec![0f32; EMBED_DIM];
            v[axis] = 1.0;
            let blob: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
            c.execute("INSERT INTO chunks_vec(rowid, embedding) VALUES (?1, ?2)", params![id, blob])
                .unwrap();
        }

        let fts: i64 = c
            .query_row("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'unboxing'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(fts, 1);

        let mut q = vec![0f32; EMBED_DIM];
        q[1] = 1.0;
        let blob: Vec<u8> = q.iter().flat_map(|f| f.to_le_bytes()).collect();
        let nearest: i64 = c
            .query_row(
                "SELECT rowid FROM chunks_vec WHERE embedding MATCH ?1 ORDER BY distance LIMIT 1",
                [blob],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(nearest, 2);

        // Deleting a chunk keeps FTS in sync.
        c.execute("DELETE FROM chunks WHERE id = 1", []).unwrap();
        let n: i64 = c
            .query_row("SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'unboxing'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 0);
    }
}
