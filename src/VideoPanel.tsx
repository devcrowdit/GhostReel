import { useEffect, useMemo, useRef, useState } from "react";
import {
  clock,
  fileName,
  fileUrl,
  mediaUrl,
  openExternal,
  videoFrames,
  videoTranscript,
  type FrameRow,
  type TranscriptSegment,
  type VideoRow,
} from "./api";

interface Description {
  description?: string;
  objects?: string[];
  setting?: string;
  shot?: string;
  tags?: string[];
  error?: string;
}

const parseDescription = (f: FrameRow): Description => {
  try {
    return f.description ? JSON.parse(f.description) : {};
  } catch {
    return {};
  }
};

/** Player + keyframes + transcript for one video; `seek` jumps the player (e.g. from a search hit). */
export default function VideoPanel({
  video,
  seek,
  onClose,
  onExclude,
}: {
  video: VideoRow;
  seek: { t: number; nonce: number } | null;
  onClose: () => void;
  /** Take the video out of the project's library (removed, or kept as a reference edit). */
  onExclude?: (role: "removed" | "reference") => void;
}) {
  const player = useRef<HTMLVideoElement>(null);
  const [segments, setSegments] = useState<TranscriptSegment[] | null>(null);
  const [frames, setFrames] = useState<FrameRow[]>([]);
  const [activeFrame, setActiveFrame] = useState<number | null>(null);
  const [filter, setFilter] = useState("");
  const [now, setNow] = useState(0);
  const [playError, setPlayError] = useState(false);
  const [src, setSrc] = useState<string | null>(null);
  const pendingSeek = useRef<number | null>(null);

  useEffect(() => {
    setSegments(null);
    videoTranscript(video.id).then(setSegments).catch(() => setSegments([]));
  }, [video.id, video.segments]);
  useEffect(() => {
    videoFrames(video.id).then(setFrames).catch(() => setFrames([]));
  }, [video.id, video.frames]);
  useEffect(() => {
    setPlayError(false);
    setSrc(null);
    mediaUrl(video.path).then(setSrc).catch(() => setPlayError(true));
  }, [video.id, video.path]);

  const jump = (t: number) => {
    const v = player.current;
    // Before metadata loads, remember the seek and apply it in onLoadedMetadata.
    if (!v || v.readyState < 1) {
      pendingSeek.current = t;
      return;
    }
    v.currentTime = t;
    v.play().catch(() => {});
  };
  useEffect(() => {
    if (seek) jump(seek.t);
  }, [seek?.nonce]); // eslint-disable-line react-hooks/exhaustive-deps

  const current = useMemo(
    () => (segments ?? []).findIndex((s) => now >= s.start && now < s.end),
    [segments, now],
  );
  const q = filter.trim().toLowerCase();
  const shown = (segments ?? []).map((s, i) => ({ s, i })).filter(({ s }) => !q || s.text.toLowerCase().includes(q));
  const active = frames.find((f) => f.id === activeFrame);
  const d = active ? parseDescription(active) : null;

  return (
    <section className="card video-panel">
      <div className="card-head">
        <span className="label">
          {fileName(video.path)}
          {video.language && <span className="tag">{video.language.toUpperCase()}</span>}
        </span>
        <span className="inline">
          <button className="ghost small" onClick={() => openExternal(video.path, player.current?.currentTime ?? 0)}>
            Open in player
          </button>
          {onExclude && (
            <>
              <button
                className="ghost small"
                title="A finished edit made by a person: the script chat studies it but never uses it as footage"
                onClick={() => onExclude("reference")}
              >
                Use as reference edit
              </button>
              <button
                className="ghost small danger"
                title="Stop using this video in search and scripts (the file is not deleted)"
                onClick={() => onExclude("removed")}
              >
                Remove from library
              </button>
            </>
          )}
          <button className="ghost small" onClick={onClose}>
            Close
          </button>
        </span>
      </div>

      <div className="player-wrap">
        <video
          ref={player}
          src={src ?? undefined}
          controls
          preload="metadata"
          onLoadedMetadata={(e) => {
            if (pendingSeek.current != null) {
              e.currentTarget.currentTime = pendingSeek.current;
              pendingSeek.current = null;
              e.currentTarget.play().catch(() => {});
            }
          }}
          onTimeUpdate={(e) => setNow(e.currentTarget.currentTime)}
          onError={() => setPlayError(true)}
        />
        {playError && (
          <div className="play-error">
            This video format can't play here.{" "}
            <button className="ghost small" onClick={() => openExternal(video.path, now)}>
              Open in system player
            </button>
          </div>
        )}
      </div>

      {frames.length > 0 && (
        <div className="frame-strip">
          {frames.map((f) => (
            <figure
              key={f.id}
              className={f.id === activeFrame ? "active" : ""}
              onClick={() => {
                setActiveFrame(f.id === activeFrame ? null : f.id);
                jump(f.t_s);
              }}
            >
              <img src={fileUrl(f.path)} alt="" loading="lazy" />
              <figcaption>{clock(f.t_s)}</figcaption>
            </figure>
          ))}
        </div>
      )}
      {active && d && (
        <div className="frame-detail">
          <div className="muted small">
            {clock(active.t_s)}
            {d.shot ? ` · ${d.shot}` : ""}
            {d.setting ? ` · ${d.setting}` : ""}
          </div>
          {d.error ? (
            <div className="bad-text small">{d.error}</div>
          ) : d.description ? (
            <>
              <div>{d.description}</div>
              {active.visible_text && (
                <div className="small">
                  <span className="muted">On screen: </span>
                  {active.visible_text.split("\n").join(" · ")}
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
            <div className="muted small">Not described yet.</div>
          )}
        </div>
      )}

      {segments && segments.length > 0 && (
        <input placeholder="Find in transcript…" value={filter} onChange={(e) => setFilter(e.target.value)} />
      )}
      {segments === null ? (
        <div className="muted">Loading…</div>
      ) : segments.length === 0 ? (
        <div className="muted">
          {video.transcribe === "skipped" ? "This video has no audio." : "No transcript yet."}
        </div>
      ) : (
        <div className="segments">
          {shown.map(({ s, i }) => (
            <div key={i} className={`segment ${i === current ? "current" : ""}`} onClick={() => jump(s.start)}>
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
