import { useCallback, useEffect, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { ask, open } from "@tauri-apps/plugin-dialog";
import {
  addFolder,
  fpsLabel,
  humanDuration,
  humanSize,
  isIndexing,
  projectView,
  removeFolder,
  removeProject,
  startIndex,
  type IndexEvent,
  type IndexFinished,
  type ProjectView,
} from "./api";

const fileName = (p: string) => p.split(/[\\/]/).pop() ?? p;

export default function ProjectPage({ projectId, onChanged }: { projectId: number; onChanged: () => void }) {
  const [view, setView] = useState<ProjectView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [indexing, setIndexing] = useState(false);
  const [progress, setProgress] = useState<string>("");

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
    setProgress("");
    refresh();
    isIndexing().then(setIndexing);
  }, [refresh]);

  useEffect(() => {
    let done = 0;
    let failed = 0;
    const unlisten = [
      listen<IndexEvent>("index-event", ({ payload: e }) => {
        switch (e.event) {
          case "scan_folder":
            setProgress(`Scanning ${e.path}…`);
            break;
          case "folder_missing":
            setProgress(`Folder not available (index kept): ${e.path}`);
            break;
          case "scanned":
            setProgress(`Found ${e.new} new, ${e.changed} changed, ${e.removed} removed — reading video details…`);
            break;
          case "job_done":
            done += 1;
            setProgress(`Read details of ${done} video${done === 1 ? "" : "s"}${failed ? `, ${failed} failed` : ""}…`);
            break;
          case "job_failed":
            failed += 1;
            break;
        }
      }),
      listen<IndexFinished>("index-finished", ({ payload }) => {
        setIndexing(false);
        done = 0;
        failed = 0;
        if (payload.error) setProgress(`Indexing failed: ${payload.error}`);
        else if (payload.summary) {
          const s = payload.summary;
          setProgress(
            `Done: ${s.new} new, ${s.changed} changed, ${s.removed} removed · ${s.jobs_done} processed` +
              (s.jobs_failed ? ` · ${s.jobs_failed} failed` : ""),
          );
        }
        refresh();
        onChanged();
      }),
    ];
    return () => {
      unlisten.forEach((u) => u.then((f) => f()));
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

  const onIndex = async () => {
    setError(null);
    setProgress("Starting…");
    try {
      await startIndex(projectId);
      setIndexing(true);
    } catch (e) {
      setProgress("");
      setError(String(e));
    }
  };

  const onDelete = async () => {
    if (!view) return;
    const ok = await ask(`Delete project "${view.project.name}"? Your video files are not touched.`, {
      title: "Delete project",
      kind: "warning",
    });
    if (ok) run(() => removeProject(projectId));
  };

  if (!view) return <main>{error ? <div className="banner bad">{error}</div> : <p className="muted">Loading…</p>}</main>;

  const { project: p, status: st } = view;
  const probe = st.stages.find((s) => s.stage === "probe");

  return (
    <main>
      <header>
        <div>
          <h1>{p.name}</h1>
          <p className="muted">
            {p.width}×{p.height} · {fpsLabel(p.fps_num, p.fps_den)} fps · {st.videos} videos ·{" "}
            {humanDuration(st.total_duration_s)} · {humanSize(st.total_size)}
          </p>
        </div>
        <button onClick={onIndex} disabled={indexing || view.folders.length === 0}>
          {indexing ? "Indexing…" : "Index now"}
        </button>
      </header>

      {error && <div className="banner bad">{error}</div>}
      {progress && <div className="banner info">{progress}</div>}

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
        {probe && (
          <span className="muted small normal">
            {" "}
            · {probe.done} ready{probe.pending ? ` · ${probe.pending} waiting` : ""}
            {probe.failed ? ` · ${probe.failed} failed` : ""}
            {st.vfr_videos ? ` · ${st.vfr_videos} variable frame rate` : ""}
          </span>
        )}
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
                <th>State</th>
              </tr>
            </thead>
            <tbody>
              {view.videos.map((v) => (
                <tr key={v.id} title={v.path}>
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
    </main>
  );
}
