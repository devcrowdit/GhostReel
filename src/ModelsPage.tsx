import { useCallback, useEffect, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import {
  cancelTask,
  enqueueModelDownload,
  humanSize,
  modelsStatus,
  openModelsDir,
  removeModel,
  scoreMeter,
  setWhisperModel,
  type ModelStatus,
  type ModelsStatusView,
} from "./api";
import { useQueue } from "./useQueue";

export default function ModelsPage() {
  const [data, setData] = useState<ModelsStatusView | null>(null);
  const [whisperModel, setLocalWhisperModel] = useState<string>("auto");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const tasks = useQueue();

  const refresh = useCallback(async () => {
    try {
      const res = await modelsStatus();
      setData(res);
      if (res.current_whisper_model) {
        setLocalWhisperModel(res.current_whisper_model);
      }
    } catch (e) {
      setError(String(e));
    }
  }, []);

  useEffect(() => {
    refresh();
    const un = listen("task-finished", () => {
      refresh();
    });
    return () => {
      un.then((f) => f());
    };
  }, [refresh]);

  const handleDownload = async (modelId: string) => {
    try {
      setError(null);
      await enqueueModelDownload(modelId);
    } catch (e) {
      setError(String(e));
    }
  };

  const handleRemove = async (modelId: string) => {
    try {
      setError(null);
      await removeModel(modelId);
      await refresh();
    } catch (e) {
      setError(String(e));
    }
  };

  const handleSetWhisper = async (modelId: string) => {
    try {
      setLocalWhisperModel(modelId);
      await setWhisperModel(modelId);
      await refresh();
    } catch (e) {
      setError(String(e));
    }
  };

  const handleOpenFolder = async () => {
    try {
      await openModelsDir();
    } catch (e) {
      setError(String(e));
    }
  };

  const whisperModels = data?.models.filter((m) => m.entry.kind === "whisper") ?? [];
  const visionModels =
    data?.models.filter((m) => m.entry.kind === "vision" || m.entry.kind === "vision_projector") ?? [];
  const embeddingModels = data?.models.filter((m) => m.entry.kind === "embedding") ?? [];

  const renderStatus = (m: ModelStatus) => {
    const activeTask = tasks.find(
      (t) =>
        t.kind.type === "download_model" &&
        t.kind.model_id === m.entry.id &&
        (t.state === "running" || t.state === "queued"),
    );

    if (activeTask) {
      if (activeTask.state === "running") {
        const pct = Math.round((activeTask.progress?.fraction ?? 0) * 100);
        return (
          <div className="status-cell">
            <span className="status-badge downloading">Downloading {pct}%</span>
            <div className="meter small-meter">
              <div style={{ width: `${Math.max(4, pct)}%` }} />
            </div>
          </div>
        );
      }
      return <span className="status-badge queued">Queued</span>;
    }

    if (m.installed_path) {
      if (m.in_own_dir) {
        return (
          <span className="status-badge installed" title={m.installed_path}>
            Installed
          </span>
        );
      }
      const lower = m.installed_path.toLowerCase();
      const origin = lower.includes("ghostpen")
        ? "GhostPen"
        : lower.includes("lmstudio")
          ? "LM Studio"
          : "external";
      return (
        <span className="status-badge external" title={m.installed_path}>
          Found in {origin}
        </span>
      );
    }

    if (m.partial_bytes) {
      return (
        <span className="status-badge partial">
          Partial ({humanSize(m.partial_bytes)})
        </span>
      );
    }

    return <span className="status-badge not-installed">Not installed</span>;
  };

  const renderActions = (m: ModelStatus) => {
    const activeTask = tasks.find(
      (t) =>
        t.kind.type === "download_model" &&
        t.kind.model_id === m.entry.id &&
        (t.state === "running" || t.state === "queued"),
    );

    if (activeTask) {
      return (
        <button
          className="ghost small"
          onClick={() => cancelTask(activeTask.id)}
          title="Cancel download (keeps partial file)"
        >
          Cancel
        </button>
      );
    }

    if (m.installed_path && m.in_own_dir) {
      return (
        <button
          className="ghost small danger"
          onClick={() => handleRemove(m.entry.id)}
          title="Delete from GhostReel's models folder"
        >
          Remove
        </button>
      );
    }

    if (m.installed_path && !m.in_own_dir) {
      return <span className="muted small">Reused</span>;
    }

    if (m.partial_bytes) {
      return (
        <div className="button-group">
          <button className="small" onClick={() => handleDownload(m.entry.id)} title="Resume download">
            Resume
          </button>
          <button
            className="ghost small danger"
            onClick={() => handleRemove(m.entry.id)}
            title="Discard partial file"
          >
            Clear
          </button>
        </div>
      );
    }

    return (
      <button className="small" onClick={() => handleDownload(m.entry.id)}>
        Download
      </button>
    );
  };

  return (
    <main>
      <header>
        <div>
          <h1>Models</h1>
          <p className="muted">Download and choose local AI models for speech, vision, and embeddings</p>
        </div>
        <button
          className="ghost"
          onClick={async () => {
            setBusy(true);
            await refresh();
            setBusy(false);
          }}
          disabled={busy}
        >
          {busy ? "Checking…" : "Refresh"}
        </button>
      </header>

      {error && <div className="banner bad">{error}</div>}

      {data && (
        <div className="card models-dir-card">
          <div className="card-head">
            <div>
              <div className="label">Models directory</div>
              <div className="muted small path">{data.dir}</div>
            </div>
            <button className="ghost small" onClick={handleOpenFolder}>
              Open folder
            </button>
          </div>
        </div>
      )}

      <h2>Speech (whisper)</h2>
      <div className="card whisper-config-card">
        <div className="whisper-auto-row">
          <label className="radio-label">
            <input
              type="radio"
              name="whisper-select"
              value="auto"
              checked={whisperModel === "auto"}
              onChange={() => handleSetWhisper("auto")}
            />
            <span className="radio-text">
              <strong>Auto</strong> — large-v3-turbo on ≥6 GB NVIDIA GPU, otherwise small (recommended)
            </span>
          </label>
        </div>
      </div>

      <div className="card table-card">
        <table className="models-table">
          <thead>
            <tr>
              <th className="col-radio">Use</th>
              <th>Model</th>
              <th>Size</th>
              <th>Speed</th>
              <th>Accuracy</th>
              <th>Note</th>
              <th>Status</th>
              <th className="col-action">Action</th>
            </tr>
          </thead>
          <tbody>
            {whisperModels.map((m) => {
              const isSelected = whisperModel === m.entry.id;
              return (
                <tr key={m.entry.id} className={isSelected ? "selected-row" : ""}>
                  <td className="col-radio">
                    <input
                      type="radio"
                      name="whisper-select"
                      value={m.entry.id}
                      checked={isSelected}
                      onChange={() => handleSetWhisper(m.entry.id)}
                      title={`Use ${m.entry.id} for transcription`}
                    />
                  </td>
                  <td>
                    <div className="model-name">
                      <strong>{m.entry.id}</strong>
                      {m.entry.languages === "english" && (
                        <span className="pill en-pill">English-only</span>
                      )}
                    </div>
                  </td>
                  <td className="col-size">{humanSize(m.entry.size_bytes)}</td>
                  <td className="col-meter" title={`Speed ${m.entry.speed}/5`}>
                    <span className="score-meter">{scoreMeter(m.entry.speed)}</span>
                  </td>
                  <td className="col-meter" title={`Accuracy ${m.entry.accuracy}/5`}>
                    <span className="score-meter">{scoreMeter(m.entry.accuracy)}</span>
                  </td>
                  <td className="muted small col-note">{m.entry.note}</td>
                  <td>{renderStatus(m)}</td>
                  <td className="col-action">{renderActions(m)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <h2>Vision</h2>
      <div className="card table-card">
        <table className="models-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Size</th>
              <th>Speed</th>
              <th>Accuracy</th>
              <th>Note</th>
              <th>Status</th>
              <th className="col-action">Action</th>
            </tr>
          </thead>
          <tbody>
            {visionModels.map((m) => (
              <tr key={m.entry.id}>
                <td>
                  <div className="model-name">
                    <strong>{m.entry.id}</strong>
                    <span className="pill">{m.entry.kind === "vision" ? "Vision model" : "Projector"}</span>
                  </div>
                </td>
                <td className="col-size">{humanSize(m.entry.size_bytes)}</td>
                <td className="col-meter" title={`Speed ${m.entry.speed}/5`}>
                  <span className="score-meter">{scoreMeter(m.entry.speed)}</span>
                </td>
                <td className="col-meter" title={`Accuracy ${m.entry.accuracy}/5`}>
                  <span className="score-meter">{scoreMeter(m.entry.accuracy)}</span>
                </td>
                <td className="muted small col-note">{m.entry.note}</td>
                <td>{renderStatus(m)}</td>
                <td className="col-action">{renderActions(m)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h2>Search embeddings</h2>
      <div className="card table-card">
        <table className="models-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Size</th>
              <th>Speed</th>
              <th>Accuracy</th>
              <th>Note</th>
              <th>Status</th>
              <th className="col-action">Action</th>
            </tr>
          </thead>
          <tbody>
            {embeddingModels.map((m) => (
              <tr key={m.entry.id}>
                <td>
                  <div className="model-name">
                    <strong>{m.entry.id}</strong>
                    <span className="pill">Embedding</span>
                  </div>
                </td>
                <td className="col-size">{humanSize(m.entry.size_bytes)}</td>
                <td className="col-meter" title={`Speed ${m.entry.speed}/5`}>
                  <span className="score-meter">{scoreMeter(m.entry.speed)}</span>
                </td>
                <td className="col-meter" title={`Accuracy ${m.entry.accuracy}/5`}>
                  <span className="score-meter">{scoreMeter(m.entry.accuracy)}</span>
                </td>
                <td className="muted small col-note">{m.entry.note}</td>
                <td>{renderStatus(m)}</td>
                <td className="col-action">{renderActions(m)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </main>
  );
}
