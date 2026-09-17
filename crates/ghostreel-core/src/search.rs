//! Hybrid search (plan D6, §5): FTS5 keyword ranking + sqlite-vec nearest neighbours, fused with
//! Reciprocal Rank Fusion, scoped to a project and grouped into moments.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use rusqlite::params;
use serde::Serialize;

use crate::Error;
use crate::db::Db;
use crate::embed::{Embedder, query_text, to_blob};

const CANDIDATES: usize = 200;
const RRF_K: f64 = 60.0;
/// Hits of the same video closer than this merge into one moment.
const MERGE_GAP_S: f64 = 10.0;

#[derive(Debug, Clone, Default)]
pub struct SearchOptions {
    pub project_id: Option<i64>,
    pub limit: usize,
    /// Restrict to chunk kinds (`moment`, `transcript`, `frame`).
    pub kinds: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Hit {
    pub video_id: i64,
    pub path: PathBuf,
    pub start_s: f64,
    pub end_s: f64,
    pub score: f64,
    /// Kinds of the chunks that matched (`moment`, `transcript`, `frame`).
    pub kinds: Vec<String>,
    /// How it matched: `keyword`, `meaning`, or both.
    pub matched_by: Vec<String>,
    pub snippet: String,
    /// Frame image for the moment (closest keyframe).
    pub frame: Option<PathBuf>,
}

/// Words that match almost every chunk and only add noise to keyword ranking (English + Spanish).
const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "how", "i", "in", "is", "it", "its",
    "of", "on", "or", "that", "the", "this", "to", "was", "we", "what", "when", "where", "which", "who", "with", "you",
    "de", "del", "el", "en", "es", "la", "las", "lo", "los", "un", "una", "y", "o", "que", "con", "por", "para", "se",
    "al", "como",
];
/// A moment never grows beyond this while merging overlapping hits.
const MAX_MOMENT_S: f64 = 60.0;

/// FTS5 query from free text: meaningful words (no stopwords), prefix-matched when long enough
/// ("unbox" → "unboxing"), any word may match; BM25 ranks chunks with more/rarer matches higher.
/// Returns `None` when there is nothing searchable.
pub fn fts_query(q: &str) -> Option<String> {
    let words: Vec<String> =
        q.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).map(|w| w.to_lowercase()).collect();
    let meaningful: Vec<&String> = words.iter().filter(|w| !STOPWORDS.contains(&w.as_str())).collect();
    // A query made only of stopwords ("the who") still searches them.
    let chosen: Vec<&String> = if meaningful.is_empty() { words.iter().collect() } else { meaningful };
    let terms: Vec<String> = chosen
        .into_iter()
        .map(|w| if w.chars().count() >= 4 { format!("\"{w}\"*") } else { format!("\"{w}\"") })
        .collect();
    (!terms.is_empty()).then(|| terms.join(" OR "))
}

struct ChunkInfo {
    video_id: i64,
    kind: String,
    start_s: f64,
    end_s: f64,
    text: String,
    frame_id: Option<i64>,
}

/// Search with an optional embedder (convenience for single-threaded callers like the CLI).
pub async fn search(
    db: &Db,
    data_dir: &Path,
    query: &str,
    embedder: Option<&mut Embedder>,
    opts: &SearchOptions,
) -> Result<Vec<Hit>, Error> {
    let vector = match embedder {
        Some(e) => Some(query_vector(e, query).await?),
        None => None,
    };
    search_with_vector(db, data_dir, query, vector.as_deref(), opts)
}

/// Embed a search query (embeddinggemma query prompt).
pub async fn query_vector(embedder: &mut Embedder, query: &str) -> Result<Vec<f32>, Error> {
    Ok(embedder.embed(&[query_text(query)]).await?.remove(0))
}

/// Rank and group results; `vector` is the query embedding (keyword-only when `None`). Synchronous so
/// callers can embed first and keep the database handle off async boundaries.
pub fn search_with_vector(
    db: &Db,
    data_dir: &Path,
    query: &str,
    vector: Option<&[f32]>,
    opts: &SearchOptions,
) -> Result<Vec<Hit>, Error> {
    let limit = if opts.limit == 0 { 20 } else { opts.limit };
    let allowed: HashSet<i64> = {
        let mut st = db.conn.prepare(
            "SELECT DISTINCT vf.video_id FROM video_files vf JOIN project_folders pf ON pf.folder_id = vf.folder_id
              WHERE (?1 IS NULL OR pf.project_id = ?1)
                AND NOT EXISTS (SELECT 1 FROM project_exclusions x WHERE x.project_id = pf.project_id AND x.video_id = vf.video_id)",
        )?;
        st.query_map([opts.project_id], |r| r.get(0))?.collect::<Result<_, _>>()?
    };

    // Ranked candidate chunk ids from each retriever.
    let mut keyword: Vec<(i64, String)> = Vec::new();
    if let Some(fq) = fts_query(query) {
        let mut st = db.conn.prepare(
            "SELECT rowid, snippet(chunks_fts, 0, '[', ']', '…', 14) FROM chunks_fts WHERE chunks_fts MATCH ?1
              ORDER BY bm25(chunks_fts) LIMIT ?2",
        )?;
        keyword =
            st.query_map(params![fq, CANDIDATES as i64], |r| Ok((r.get(0)?, r.get(1)?)))?.collect::<Result<_, _>>()?;
    }
    let mut semantic: Vec<i64> = Vec::new();
    if let Some(v) = vector {
        let mut st =
            db.conn.prepare("SELECT rowid FROM chunks_vec WHERE embedding MATCH ?1 AND k = ?2 ORDER BY distance")?;
        semantic = st.query_map(params![to_blob(v), CANDIDATES as i64], |r| r.get(0))?.collect::<Result<_, _>>()?;
    }

    let mut ids: Vec<i64> = keyword.iter().map(|(id, _)| *id).chain(semantic.iter().copied()).collect();
    ids.sort_unstable();
    ids.dedup();
    let info: HashMap<i64, ChunkInfo> = {
        let mut map = HashMap::new();
        let mut st =
            db.conn.prepare("SELECT video_id, kind, start_s, end_s, text, frame_id FROM chunks WHERE id = ?1")?;
        for id in &ids {
            if let Ok(c) = st.query_row([id], |r| {
                Ok(ChunkInfo {
                    video_id: r.get(0)?,
                    kind: r.get(1)?,
                    start_s: r.get::<_, Option<f64>>(2)?.unwrap_or(0.0),
                    end_s: r.get::<_, Option<f64>>(3)?.unwrap_or(0.0),
                    text: r.get(4)?,
                    frame_id: r.get(5)?,
                })
            }) {
                map.insert(*id, c);
            }
        }
        map
    };
    let keep = |id: &i64| {
        info.get(id).is_some_and(|c| {
            allowed.contains(&c.video_id) && opts.kinds.as_ref().is_none_or(|k| k.iter().any(|x| x == &c.kind))
        })
    };

    // RRF over the filtered rankings.
    let mut scores: HashMap<i64, (f64, Vec<&'static str>)> = HashMap::new();
    let snippets: HashMap<i64, String> = keyword.iter().cloned().collect();
    for (rank, (id, _)) in keyword.iter().filter(|(id, _)| keep(id)).enumerate() {
        let e = scores.entry(*id).or_default();
        e.0 += 1.0 / (RRF_K + rank as f64 + 1.0);
        e.1.push("keyword");
    }
    for (rank, id) in semantic.iter().filter(|id| keep(id)).enumerate() {
        let e = scores.entry(*id).or_default();
        e.0 += 1.0 / (RRF_K + rank as f64 + 1.0);
        e.1.push("meaning");
    }
    let mut ranked: Vec<(i64, f64, Vec<&'static str>)> = scores.into_iter().map(|(id, (s, m))| (id, s, m)).collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));

    // Group into moments.
    let mut hits: Vec<(Hit, Option<i64>)> = Vec::new();
    for (id, score, matched) in ranked {
        let c = &info[&id];
        let snippet = snippets.get(&id).cloned().unwrap_or_else(|| short(&c.text));
        if let Some((h, _)) = hits.iter_mut().find(|(h, _)| {
            h.video_id == c.video_id && c.start_s <= h.end_s + MERGE_GAP_S && c.end_s + MERGE_GAP_S >= h.start_s
        }) {
            h.score += score * 0.5; // corroborating evidence, diminished
            // Grow the moment only while it stays short enough to be "a moment".
            let (start, end) = (h.start_s.min(c.start_s), h.end_s.max(c.end_s));
            if end - start <= MAX_MOMENT_S {
                h.start_s = start;
                h.end_s = end;
            }
            for m in matched {
                if !h.matched_by.iter().any(|x| x == m) {
                    h.matched_by.push(m.to_string());
                }
            }
            if !h.kinds.contains(&c.kind) {
                h.kinds.push(c.kind.clone());
            }
            continue;
        }
        if hits.len() >= limit * 3 {
            continue;
        }
        hits.push((
            Hit {
                video_id: c.video_id,
                path: PathBuf::new(),
                start_s: c.start_s,
                end_s: c.end_s,
                score,
                kinds: vec![c.kind.clone()],
                matched_by: matched.iter().map(|m| m.to_string()).collect(),
                snippet,
                frame: None,
            },
            c.frame_id,
        ));
    }
    hits.sort_by(|a, b| b.0.score.total_cmp(&a.0.score));
    hits.truncate(limit);

    // Paths and frames.
    let mut out = Vec::with_capacity(hits.len());
    for (mut h, frame_id) in hits {
        h.path = db
            .conn
            .query_row(
                "SELECT vf.path FROM video_files vf JOIN project_folders pf ON pf.folder_id = vf.folder_id
                  WHERE vf.video_id = ?1 AND (?2 IS NULL OR pf.project_id = ?2) ORDER BY vf.id LIMIT 1",
                params![h.video_id, opts.project_id],
                |r| r.get::<_, String>(0),
            )
            .map(PathBuf::from)
            .unwrap_or_default();
        let rel: Option<String> = match frame_id {
            Some(fid) => db.conn.query_row("SELECT thumb_path FROM frames WHERE id = ?1", [fid], |r| r.get(0)).ok(),
            None => db
                .conn
                .query_row(
                    "SELECT thumb_path FROM frames WHERE video_id = ?1 ORDER BY ABS(t_s - ?2) LIMIT 1",
                    params![h.video_id, h.start_s],
                    |r| r.get(0),
                )
                .ok(),
        };
        h.frame = rel.map(|r| data_dir.join(r));
        out.push(h);
    }
    Ok(out)
}

fn short(text: &str) -> String {
    let flat = text.replace('\n', " · ");
    let s: String = flat.chars().take(220).collect();
    if flat.chars().count() > 220 { format!("{s}…") } else { s }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projects::NewProject;

    #[test]
    fn fts_query_building() {
        assert_eq!(fts_query("CM5 unbox!").as_deref(), Some("\"cm5\" OR \"unbox\"*"));
        assert_eq!(fts_query("  ?!  "), None);
        assert_eq!(
            fts_query("the temperature graph in the browser").as_deref(),
            Some("\"temperature\"* OR \"graph\"* OR \"browser\"*")
        );
        assert_eq!(fts_query("the who").as_deref(), Some("\"the\" OR \"who\""), "all-stopword queries still search");
        assert_eq!(fts_query("la red inalámbrica").as_deref(), Some("\"red\" OR \"inalámbrica\"*"));
    }

    #[tokio::test]
    async fn keyword_search_scoped_grouped() {
        let tmp = tempfile::tempdir().unwrap();
        let (a_dir, b_dir) = (tmp.path().join("a"), tmp.path().join("b"));
        std::fs::create_dir_all(&a_dir).unwrap();
        std::fs::create_dir_all(&b_dir).unwrap();
        let mut db = Db::open_in_memory().unwrap();
        let pa = db.create_project(&NewProject::named("A")).unwrap();
        let pb = db.create_project(&NewProject::named("B")).unwrap();
        let fa = db.add_folder(pa.id, &a_dir, true).unwrap();
        let fb = db.add_folder(pb.id, &b_dir, true).unwrap();
        let c = &db.conn;
        c.execute_batch(&format!(
            "INSERT INTO videos(id, content_hash, size) VALUES (1, 'h1', 1), (2, 'h2', 1);
             INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen)
                 VALUES (1, {fa}, '{a}/one.mp4', 1, 0, 0), (2, {fb}, '{b}/two.mp4', 1, 0, 0);
             INSERT INTO chunks(video_id, kind, start_s, end_s, text) VALUES
                 (1, 'transcript', 0, 30, 'we unbox the compute module'),
                 (1, 'moment', 5, 20, 'Hands unbox a green board. On screen: CM5'),
                 (1, 'transcript', 200, 230, 'flash the image'),
                 (2, 'transcript', 0, 30, 'unbox a phone');",
            fa = fa.id,
            fb = fb.id,
            a = a_dir.display(),
            b = b_dir.display()
        ))
        .unwrap();

        let opts = SearchOptions { project_id: Some(pa.id), limit: 10, kinds: None };
        let hits = search(&db, tmp.path(), "unbox", None, &opts).await.unwrap();
        assert_eq!(hits.len(), 1, "two overlapping chunks of video 1 merge; video 2 is another project");
        assert_eq!((hits[0].start_s, hits[0].end_s), (0.0, 30.0));
        assert!(hits[0].kinds.contains(&"moment".to_string()) && hits[0].kinds.contains(&"transcript".to_string()));
        assert!(hits[0].snippet.contains("[unbox]"));
        assert!(hits[0].path.ends_with("one.mp4"));

        let all =
            search(&db, tmp.path(), "unbox", None, &SearchOptions { limit: 10, ..Default::default() }).await.unwrap();
        assert_eq!(all.len(), 2);
        let kinds = SearchOptions { project_id: Some(pa.id), limit: 10, kinds: Some(vec!["transcript".into()]) };
        let hits = search(&db, tmp.path(), "flash", None, &kinds).await.unwrap();
        assert_eq!(hits[0].start_s, 200.0);
        assert!(search(&db, tmp.path(), "zebra", None, &opts).await.unwrap().is_empty());
    }
}
