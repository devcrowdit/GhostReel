import { useCallback, useEffect, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { ask, open } from "@tauri-apps/plugin-dialog";
import {
  addFolder,
  clock,
  enqueueIndex,
  fileName,
  fileUrl,
  fpsLabel,
  humanDuration,
  humanSize,
  projectView,
  removeFolder,
  removeProject,
  search,
  type Hit,
  type ProjectView,
  type VideoRow,
} from "./api";
import TaskCard from "./TaskCard";
import { useQueue } from "./useQueue";
import VideoPanel from "./VideoPanel";
import ScriptsPanel from "./ScriptsPanel";

function SpeechCell({ v }: { v: VideoRow }) {
  if (v.segments > 0) return <span className="good-text">{v.language ? v.language.toUpperCase() : "✓"}</span>;
  switch (v.transcribe) {
    case "skipped":
      return <span className="muted">no audio</span>;
    case "failed":
      return <span className="bad-text">failed</span>;
    case "done":
      return <span className="muted">no speech</span>;
    default:
      return <span className="muted">waiting</span>;
  }
}

/** "…the [compute] module…" → highlighted spans. */
function Snippet({ text }: { text: string }) {
  const parts = text.split(/(\[[^\]]+\])/g);
  return (
    <span>
      {parts.map((p, i) =>
        p.startsWith("[") && p.endsWith("]") ? <mark key={i}>{p.slice(1, -1)}</mark> : <span key={i}>{p}</span>,
      )}
    </span>
  );
}

export default function ProjectPage({ projectId, onChanged }: { projectId: number; onChanged: () => void }) {
  const [tab, setTab] = useState<"library" | "scripts">("library");
  const [view, setView] = useState<ProjectView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [seek, setSeek] = useState<{ t: number; nonce: number } | null>(null);
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<Hit[] | null>(null);
  const [searchNote, setSearchNote] = useState<string | null>(null);
  const [searching, setSearching] = useState(false);
  const panelRef = useRef<HTMLDivElement>(null);
  const tasks = useQueue();

  const refresh = useCallback(async () => {
    try {
      setView(await projectView(projectId));
      setError(null);
    } catch (e) {
      setError(String(e));
    }
  }, [projectId]);

  useEffect(() => {
    setView(null);
    setSelected(null);
    setHits(null);
    setQuery("");
    refresh();
  }, [refresh]);

  // Keep the table fresh while this project's task runs, and once it finishes.
  const myTasks = tasks.filter((t) => t.kind.type === "index" && t.kind.project_id === projectId);
  const running = myTasks.find((t) => t.state === "running");
  const queued = myTasks.find((t) => t.state === "queued");
  const lastFinished = [...myTasks].reverse().find((t) => t.state !== "running" && t.state !== "queued");
  useEffect(() => {
    if (!running) return;
    const id = setInterval(refresh, 2500);
    return () => clearInterval(id);
  }, [running?.id, refresh]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    const un = listen<number>("task-finished", () => {
      refresh();
      onChanged();
    });
    return () => {
      un.then((f) => f());
    };
  }, [refresh, onChanged]);

  const run = async (fn: () => Promise<unknown>) => {
    try {
      await fn();
      await refresh();
      onChanged();
    } catch (e) {
      setError(String(e));
    }
  };

  const onAddFolder = async () => {
    const dir = await open({ directory: true, multiple: false, title: "Add a video folder" });
    if (typeof dir === "string") run(() => addFolder(projectId, dir));
  };

  const onDelete = async () => {
    if (!view) return;
    const ok = await ask(`Delete project "${view.project.name}"? Your video files are not touched.`, {
      title: "Delete project",
      kind: "warning",
    });
    if (ok) run(() => removeProject(projectId));
  };

  const onSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    const q = query.trim();
    if (!q) {
      setHits(null);
      return;
    }
    setSearching(true);
    try {
      const r = await search(projectId, q);
      setHits(r.hits);
      setSearchNote(r.note);
    } catch (err) {
      setError(String(err));
    } finally {
      setSearching(false);
    }
  };

  const openAt = (videoId: number, t: number) => {
    setSelected(videoId);
    setSeek({ t, nonce: Date.now() });
    setTimeout(() => panelRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 50);
  };

  if (!view) return <main>{error ? <div className="banner bad">{error}</div> : <p className="muted">Loading…</p>}</main>;

  const { project: p, status: st } = view;
  const stage = (name: string) => st.stages.find((s) => s.stage === name);
  const selectedVideo = view.videos.find((v) => v.id === selected) ?? null;

  return (
    <main className={tab === "scripts" ? "wide" : ""}>
      <header>
        <div>
          <h1>{p.name}</h1>
          <p className="muted">
            {p.width}×{p.height} · {fpsLabel(p.fps_num, p.fps_den)} fps · {st.videos} videos ·{" "}
            {humanDuration(st.total_duration_s)} · {humanSize(st.total_size)}
          </p>
        </div>
        <button onClick={() => run(() => enqueueIndex(projectId))} disabled={!!queued || view.folders.length === 0}>
          {running ? "Index again" : queued ? "Queued…" : "Index now"}
        </button>
      </header>

      <div className="tab-nav">
        <button
          type="button"
          className={`tab-btn ${tab === "library" ? "active" : ""}`}
          onClick={() => setTab("library")}
        >
          Library
        </button>
        <button
          type="button"
          className={`tab-btn ${tab === "scripts" ? "active" : ""}`}
          onClick={() => setTab("scripts")}
        >
          Scripts
        </button>
      </div>

      {tab === "scripts" ? (
        <ScriptsPanel projectId={projectId} />
      ) : (
        <>
          {error && <div className="banner bad">{error}</div>}
          {running && <TaskCard task={running} compact />}
          {!running && queued && <TaskCard task={queued} compact />}
          {!running && !queued && lastFinished && lastFinished.state !== "done" && <TaskCard task={lastFinished} compact />}

      <form className="search" onSubmit={onSearch}>
        <input
          placeholder="Search this project: “unboxing the board”, “wifi password”, on-screen text…"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            if (!e.target.value.trim()) setHits(null);
          }}
        />
        <button type="submit" disabled={searching || !query.trim()}>
          {searching ? "Searching…" : "Search"}
        </button>
      </form>
      {searchNote && <div className="muted small">{searchNote}</div>}
      {hits && (
        <section className="hits">
          {hits.length === 0 && <div className="muted">No matches.</div>}
          {hits.map((h, i) => (
            <div key={i} className="card hit" onClick={() => openAt(h.video_id, h.start_s)}>
              {h.frame ? <img src={fileUrl(h.frame)} alt="" loading="lazy" /> : <div className="noframe" />}
              <div className="hit-body">
                <div className="hit-head">
                  <span className="label">{fileName(h.path)}</span>
                  <span className="time">
                    {clock(h.start_s)}–{clock(h.end_s)}
                  </span>
                </div>
                <div className="small">
                  <Snippet text={h.snippet} />
                </div>
                <div>
                  {h.kinds.map((k) => (
                    <span key={k} className="tag">
                      {k === "moment" ? "picture" : k === "frame" ? "on screen" : "speech"}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </section>
      )}

      <div ref={panelRef}>
        {selectedVideo && <VideoPanel video={selectedVideo} seek={seek} onClose={() => setSelected(null)} />}
      </div>

      <h2>Folders</h2>
      <section className="card list">
        {view.folders.length === 0 && <div className="muted">Add the folders that hold this project's footage.</div>}
        {view.folders.map((f) => (
          <div key={f.id} className="row folder">
            <span className={f.available ? "" : "bad-text"}>{f.available ? "📁" : "⚠"}</span>
            <span className="path">
              {f.path}
              {!f.available && <span className="muted small"> — not available (drive unplugged?)</span>}
            </span>
            <button className="ghost small" onClick={() => run(() => removeFolder(projectId, f.path))}>
              Remove
            </button>
          </div>
        ))}
        <div>
          <button className="ghost" onClick={onAddFolder}>
            + Add folder
          </button>
        </div>
      </section>

      <h2>
        Videos
        <span className="muted small normal">
          {" "}
          · {stage("probe")?.done ?? 0} ready · {stage("transcribe")?.done ?? 0} transcribed ·{" "}
          {stage("describe")?.done ?? 0} described · {stage("embed")?.done ?? 0} searchable
          {st.vfr_videos ? ` · ${st.vfr_videos} variable frame rate` : ""}
        </span>
      </h2>
      <section className="card">
        {view.videos.length === 0 ? (
          <div className="muted">No videos yet — add a folder and press “Index now”.</div>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Name</th>
                <th>Length</th>
                <th>Size</th>
                <th>Format</th>
                <th>Speech</th>
                <th>Frames</th>
                <th>State</th>
              </tr>
            </thead>
            <tbody>
              {view.videos.map((v) => (
                <tr
                  key={v.id}
                  title={v.path}
                  className={`clickable ${v.id === selected ? "selected" : ""}`}
                  onClick={() => (v.id === selected ? setSelected(null) : openAt(v.id, 0))}
                >
                  <td className="name">
                    {fileName(v.path)}
                    {v.copies > 1 && <span className="tag">×{v.copies}</span>}
                    {v.vfr && (
                      <span className="tag warn" title="Variable frame rate (phone footage) — may drift in Premiere">
                        VFR
                      </span>
                    )}
                    {v.has_audio === false && <span className="tag">no audio</span>}
                  </td>
                  <td>{v.duration_s != null ? humanDuration(v.duration_s) : "–"}</td>
                  <td>{humanSize(v.size)}</td>
                  <td className="muted">
                    {v.width && v.height ? `${v.width}×${v.height}` : "–"}
                    {v.fps ? ` · ${v.fps.toFixed(v.fps % 1 ? 2 : 0)} fps` : ""}
                    {v.vcodec ? ` · ${v.vcodec}` : ""}
                  </td>
                  <td>
                    <SpeechCell v={v} />
                  </td>
                  <td className="muted">{v.frames || "–"}</td>
                  <td>
                    {v.status === "error" ? (
                      <span className="bad-text" title={v.error ?? ""}>
                        error
                      </span>
                    ) : v.status === "probed" ? (
                      <span className="good-text">ready</span>
                    ) : (
                      <span className="muted">waiting</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      <p className="danger-zone">
        <button className="ghost small danger" onClick={onDelete}>
          Delete project
        </button>
      </p>
        </>
      )}
    </main>
  );
}
