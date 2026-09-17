import { useCallback, useEffect, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { ask, open } from "@tauri-apps/plugin-dialog";
import {
  addFolder,
  etaText,
  fpsLabel,
  humanDuration,
  humanSize,
  isIndexing,
  PHASE_LABELS,
  projectView,
  removeFolder,
  removeProject,
  startIndex,
  fileUrl,
  videoFrames,
  videoTranscript,
  type FrameRow,
  type IndexEvent,
  type TranscriptSegment,
  type VideoRow,
  type IndexFinished,
  type Progress,
  type ProjectView,
} from "./api";

const fileName = (p: string) => p.split(/[\\/]/).pop() ?? p;

const clock = (s: number) => {
  const t = Math.floor(s);
  const h = Math.floor(t / 3600);
  const m = Math.floor((t % 3600) / 60);
  const sec = String(t % 60).padStart(2, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${sec}` : `${m}:${sec}`;
};

function TranscriptCell({ v }: { v: VideoRow }) {
  if (v.segments > 0) return <span className="good-text">{v.language ? v.language.toUpperCase() : "✓"}</span>;
  switch (v.transcribe) {
    case "skipped":
      return <span className="muted">no audio</span>;
    case "failed":
      return <span className="bad-text">failed</span>;
    case "running":
      return <span>…</span>;
    case "done":
      return <span className="muted">no speech</span>;
    default:
      return <span className="muted">waiting</span>;
  }
}

function TranscriptPanel({ video, onClose }: { video: VideoRow; onClose: () => void }) {
  const [segments, setSegments] = useState<TranscriptSegment[] | null>(null);
  const [frames, setFrames] = useState<FrameRow[]>([]);
  const [activeFrame, setActiveFrame] = useState<number | null>(null);
  const [filter, setFilter] = useState("");
  useEffect(() => {
    setSegments(null);
    videoTranscript(video.id).then(setSegments).catch(() => setSegments([]));
  }, [video.id, video.segments]);
  useEffect(() => {
    videoFrames(video.id).then(setFrames).catch(() => setFrames([]));
  }, [video.id, video.frames]);
  const q = filter.trim().toLowerCase();
  const shown = (segments ?? []).filter((s) => !q || s.text.toLowerCase().includes(q));
  return (
    <section className="card transcript">
      <div className="card-head">
        <span className="label">
          {fileName(video.path)}
          {video.language && <span className="tag">{video.language.toUpperCase()}</span>}
        </span>
        <button className="ghost small" onClick={onClose}>
          Close
        </button>
      </div>
      {frames.length > 0 && (
        <div className="frame-strip">
          {frames.map((f) => (
            <figure
              key={f.id}
              className={f.id === activeFrame ? "active" : ""}
              onClick={() => setActiveFrame(f.id === activeFrame ? null : f.id)}
            >
              <img src={fileUrl(f.path)} alt="" loading="lazy" />
              <figcaption>
                {clock(f.t_s)}
                {f.description && !f.description.startsWith('{"error"') ? " ✓" : ""}
              </figcaption>
            </figure>
          ))}
        </div>
      )}
      {(() => {
        const f = frames.find((x) => x.id === activeFrame);
        if (!f) return null;
        let d: { description?: string; objects?: string[]; setting?: string; shot?: string; tags?: string[]; error?: string } =
          {};
        try {
          d = f.description ? JSON.parse(f.description) : {};
        } catch {
          d = {};
        }
        return (
          <div className="frame-detail">
            <div className="muted small">
              {clock(f.t_s)}
              {d.shot ? ` · ${d.shot}` : ""}
              {d.setting ? ` · ${d.setting}` : ""}
            </div>
            {d.error ? (
              <div className="bad-text small">{d.error}</div>
            ) : d.description ? (
              <>
                <div>{d.description}</div>
                {f.visible_text && (
                  <div className="small">
                    <span className="muted">On screen: </span>
                    {f.visible_text.split("\n").join(" · ")}
                  </div>
                )}
                {(d.tags?.length ?? 0) > 0 && (
                  <div>
                    {d.tags!.map((t) => (
                      <span key={t} className="tag">
                        {t}
                      </span>
                    ))}
                  </div>
                )}
              </>
            ) : (
              <div className="muted small">Not described yet — press “Index now”.</div>
            )}
          </div>
        );
      })()}
      {segments && segments.length > 0 && (
        <input placeholder="Find in transcript…" value={filter} onChange={(e) => setFilter(e.target.value)} />
      )}
      {segments === null ? (
        <div className="muted">Loading…</div>
      ) : segments.length === 0 ? (
        <div className="muted">
          {video.transcribe === "skipped" ? "This video has no audio." : "No transcript yet — press “Index now”."}
        </div>
      ) : (
        <div className="segments">
          {shown.map((s, i) => (
            <div key={i} className="segment">
              <span className="time">{clock(s.start)}</span>
              <span>{s.text}</span>
            </div>
          ))}
          {shown.length === 0 && <div className="muted">No match.</div>}
        </div>
      )}
    </section>
  );
}

export default function ProjectPage({ projectId, onChanged }: { projectId: number; onChanged: () => void }) {
  const [view, setView] = useState<ProjectView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [indexing, setIndexing] = useState(false);
  const [progress, setProgress] = useState<string>("");
  const [bar, setBar] = useState<Progress | null>(null);
  const [notice, setNotice] = useState<string>("");
  const [selected, setSelected] = useState<number | null>(null);

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
    setBar(null);
    setNotice("");
    setSelected(null);
    refresh();
    isIndexing().then(setIndexing);
  }, [refresh]);

  useEffect(() => {
    let done = 0;
    let failed = 0;
    // Refresh the table while indexing so videos appear as they are processed (at most every 2 s).
    let lastRefresh = 0;
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
          case "stage_backend":
            setNotice(`${e.stage === "transcribe" ? "Transcription" : e.stage}: ${e.backend}`);
            break;
          case "stage_unavailable":
            setNotice(`${e.stage === "transcribe" ? "Transcription" : e.stage} postponed: ${e.reason}`);
            break;
          case "downloading_model":
            setNotice(`Downloading ${e.file} (once)…`);
            break;
          case "progress": {
            const { event: _event, ...p } = e;
            setBar(p);
            if (Date.now() - lastRefresh > 2000) {
              lastRefresh = Date.now();
              refresh();
            }
            break;
          }
        }
      }),
      listen<IndexFinished>("index-finished", ({ payload }) => {
        setIndexing(false);
        setBar(null);
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
  const transcribe = st.stages.find((s) => s.stage === "transcribe");
  const selectedVideo = view.videos.find((v) => v.id === selected) ?? null;

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
      {indexing && bar ? (
        <div className="card progress">
          <div className="progress-head">
            <span className="label">
              {PHASE_LABELS[bar.phase] ?? bar.phase}
              {bar.phase_total > 0 && (
                <span className="muted">
                  {" "}
                  · {bar.phase_done} of {bar.phase_total}
                </span>
              )}
            </span>
            <span className="muted">
              {Math.round(bar.fraction * 100)}%
              {bar.eta_secs != null && bar.fraction < 1 && ` · ${etaText(bar.eta_secs)} left`}
            </span>
          </div>
          <div className="meter big">
            <div style={{ width: `${Math.max(2, bar.fraction * 100)}%` }} />
          </div>
          <div className="muted small path">{bar.current ? fileName(bar.current) : progress}</div>
          {notice && <div className="muted small">{notice}</div>}
        </div>
      ) : (
        (progress || notice) && (
          <div className="banner info">
            {progress}
            {notice && <div className="muted small">{notice}</div>}
          </div>
        )
      )}

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
            {transcribe && transcribe.done ? ` · ${transcribe.done} transcribed` : ""}
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
                <th>Speech</th>
                <th>State</th>
              </tr>
            </thead>
            <tbody>
              {view.videos.map((v) => (
                <tr
                  key={v.id}
                  title={v.path}
                  className={`clickable ${v.id === selected ? "selected" : ""}`}
                  onClick={() => setSelected(v.id === selected ? null : v.id)}
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
                    <TranscriptCell v={v} />
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

      {selectedVideo && <TranscriptPanel video={selectedVideo} onClose={() => setSelected(null)} />}

      <p className="danger-zone">
        <button className="ghost small danger" onClick={onDelete}>
          Delete project
        </button>
      </p>
    </main>
  );
}
