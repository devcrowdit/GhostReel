import { cancelTask, etaText, PHASE_LABELS, type Task } from "./api";

const STATE_LABEL: Record<string, string> = {
  queued: "Waiting",
  running: "Running",
  done: "Done",
  failed: "Failed",
  cancelled: "Cancelled",
};

function taskLabel(task: Task): string {
  if (task.label) return task.label;
  switch (task.kind.type) {
    case "index":
      return "Indexing";
    case "chat":
      return "Script chat";
    case "render_preview":
      return `Rendering preview (script #${task.kind.script_id})`;
    case "export":
      return `Exporting (${task.kind.format})`;
  }
}

export default function TaskCard({ task, compact = false }: { task: Task; compact?: boolean }) {
  const p = task.progress;
  const running = task.state === "running";
  const pct = Math.round((p?.fraction ?? 0) * 100);
  const s = task.summary;
  return (
    <div className={`card task ${task.state}`}>
      <div className="progress-head">
        <span className="label">{taskLabel(task)}</span>
        <span className="muted small">
          {running && p ? (
            <>
              {pct}%{p.eta_secs != null && p.fraction < 1 ? ` · ${etaText(p.eta_secs)} left` : ""}
            </>
          ) : (
            STATE_LABEL[task.state]
          )}
        </span>
      </div>
      {running && (
        <>
          <div className="meter big">
            <div style={{ width: `${Math.max(2, pct)}%` }} />
          </div>
          {p && (
            <div className="muted small">
              {PHASE_LABELS[p.phase] ?? p.phase}
              {p.phase_total > 0 ? ` · ${p.phase_done} of ${p.phase_total}` : ""}
              {p.current ? ` · ${p.current.split(/[\\/]/).pop()}` : ""}
            </div>
          )}
        </>
      )}
      {task.note && (running || !compact) && <div className="muted small">{task.note}</div>}
      {task.state === "done" && s && !compact && (
        <div className="muted small">
          {s.new} new · {s.changed} changed · {s.removed} removed · {s.jobs_done} steps done
          {s.jobs_failed ? ` · ${s.jobs_failed} failed` : ""}
        </div>
      )}
      {task.state === "done" && task.output && (
        <div className="muted small path">
          Output: {task.output}
        </div>
      )}
      {task.error && <div className="bad-text small">{task.error}</div>}
      {(task.state === "queued" || running) && (
        <div>
          <button className="ghost small" onClick={() => cancelTask(task.id)}>
            {running ? "Stop" : "Remove from queue"}
          </button>
        </div>
      )}
    </div>
  );
}
