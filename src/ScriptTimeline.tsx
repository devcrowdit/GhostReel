import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { Script, VideoRow } from "./api";

/** One block on a lane, already positioned in seconds on the cut's timeline. */
type Block = {
  key: string;
  kind: "title" | "picture" | "bed";
  beat: number;
  clip: number | null;
  /** Position in the finished cut. */
  start: number;
  end: number;
  /** Where this block reads from inside its source file. */
  sourceIn: number;
  label: string;
  detail: string;
  muted: boolean;
  inferred: boolean;
};

function fmt(t: number): string {
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  return `${m}:${s.toFixed(1).padStart(4, "0")}`;
}

function name(videos: Record<number, VideoRow>, id: number): string {
  const v = videos[id];
  if (!v) return `#${id}`;
  const base = (v.path ?? "").split(/[\\/]/).pop();
  return base || `#${id}`;
}

/** Lay the script out in seconds: where every picture, title and bed sits in the finished cut. */
function layout(script: Script, videos: Record<number, VideoRow>): { blocks: Block[]; total: number } {
  const blocks: Block[] = [];
  let at = 0;

  script.beats.forEach((beat, b) => {
    const beatStart = at;
    beat.clips.forEach((c, i) => {
      const dur = Math.max(0, c.out_s - c.in_s);
      blocks.push({
        key: `p-${b}-${i}`,
        kind: "picture",
        beat: b,
        clip: i,
        start: at,
        end: at + dur,
        sourceIn: c.in_s,
        label: name(videos, c.video_id),
        detail: `${fmt(c.in_s)}–${fmt(c.out_s)}`,
        muted: c.audio === "mute" || beat.bed != null,
        inferred: false,
      });
      at += dur;
    });
    const beatEnd = at;

    if (beat.on_screen_text?.trim()) {
      blocks.push({
        key: `t-${b}`,
        kind: "title",
        beat: b,
        clip: null,
        start: beatStart,
        end: beatEnd,
        sourceIn: 0,
        label: beat.on_screen_text,
        detail: "",
        muted: false,
        inferred: false,
      });
    }

    if (beat.bed) {
      const dur = Math.max(0, beat.bed.out_s - beat.bed.in_s);
      blocks.push({
        key: `b-${b}`,
        kind: "bed",
        beat: b,
        clip: null,
        start: beatStart,
        end: beatStart + Math.min(dur, beatEnd - beatStart),
        sourceIn: beat.bed.in_s,
        label: name(videos, beat.bed.video_id),
        detail: `${fmt(beat.bed.in_s)}–${fmt(beat.bed.out_s)}`,
        muted: false,
        inferred: beat.bed.inferred === true,
      });
    } else {
      // Where a beat has no bed, its speaking clips are what you hear.
      beat.clips.forEach((c, i) => {
        if (c.audio === "mute") return;
        const before = beat.clips.slice(0, i).reduce((n, x) => n + Math.max(0, x.out_s - x.in_s), 0);
        const dur = Math.max(0, c.out_s - c.in_s);
        blocks.push({
          key: `a-${b}-${i}`,
          kind: "bed",
          beat: b,
          clip: i,
          start: beatStart + before,
          end: beatStart + before + dur,
          sourceIn: c.in_s,
          label: name(videos, c.video_id),
          detail: "own audio",
          muted: false,
          inferred: false,
        });
      });
    }
  });

  return { blocks, total: at };
}

type Props = {
  script: Script;
  videos: Record<number, VideoRow>;
  selected: string | null;
  /** Where the preview player is, in timeline seconds. */
  playhead: number;
  onSelect: (key: string | null, beat: number, clip: number | null) => void;
  onSeek: (seconds: number) => void;
  onTogglePlay: () => void;
  onTrim: (beat: number, clip: number | null, edge: "in" | "out", deltaSeconds: number) => void;
  /** Put an edge exactly here, in the source file's own seconds. */
  onSetEdge: (beat: number, clip: number | null, edge: "in" | "out", sourceSeconds: number) => void;
  /** Pull both edges onto the nearest sentence boundaries. */
  onSnap: (beat: number, clip: number | null) => void;
};

/**
 * The cut on three lanes, the way an NLE shows it: what is written over the picture, the picture
 * itself, and the sound. A list of beats hides the one thing that matters most here — that a voice
 * runs on while the pictures change — because a bed has no place in a list.
 *
 * Editing is by keyboard as much as by mouse, because the decisions here are fractions of a
 * second: where a sentence ends is a 0.1 s judgement and a drag cannot hit it. The playhead is the
 * preview player's, so setting an edge from what you just heard is one key.
 */
export default function ScriptTimeline({
  script,
  videos,
  selected,
  playhead,
  onSelect,
  onSeek,
  onTogglePlay,
  onTrim,
  onSetEdge,
  onSnap,
}: Props) {
  const { blocks, total } = useMemo(() => layout(script, videos), [script, videos]);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(900);
  const [zoomed, setZoomed] = useState(false);
  const drag = useRef<{ beat: number; clip: number | null; edge: "in" | "out"; x: number } | null>(null);

  const sel = blocks.find((b) => b.key === selected) ?? null;

  // Zoom shows the selected beat alone, with a little air around it.
  const view = useMemo(() => {
    if (!zoomed || !sel) return { start: 0, end: Math.max(total, 0.001) };
    const beatBlocks = blocks.filter((b) => b.beat === sel.beat);
    const start = Math.min(...beatBlocks.map((b) => b.start));
    const end = Math.max(...beatBlocks.map((b) => b.end));
    const pad = Math.max((end - start) * 0.08, 0.3);
    return { start: Math.max(0, start - pad), end: Math.min(total, end + pad) };
  }, [zoomed, sel, blocks, total]);

  const span = Math.max(view.end - view.start, 0.001);
  const pct = useCallback((v: number) => `${((v - view.start) / span) * 100}%`, [view.start, span]);

  useLayoutEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.observe(el);
    setWidth(el.clientWidth);
    return () => ro.disconnect();
  }, []);

  // Dragging an edge trims by however many seconds the pointer covered.
  useEffect(() => {
    const perPx = span / Math.max(width, 1);
    const move = (e: PointerEvent) => {
      const d = drag.current;
      if (!d) return;
      const delta = (e.clientX - d.x) * perPx;
      if (Math.abs(delta) < 0.02) return;
      drag.current = { ...d, x: e.clientX };
      onTrim(d.beat, d.clip, d.edge, delta);
    };
    const up = () => {
      drag.current = null;
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    return () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
  }, [span, width, onTrim]);

  const keys = (e: React.KeyboardEvent) => {
    // A step you can feel: a second by default, a nudge with Alt, a stride with Ctrl.
    const step = e.altKey ? 0.2 : e.ctrlKey || e.metaKey ? 5 : 1;
    const edge: "in" | "out" = e.shiftKey ? "in" : "out";
    switch (e.key) {
      case "ArrowRight":
      case "ArrowLeft": {
        const dir = e.key === "ArrowRight" ? 1 : -1;
        if (!sel) {
          onSeek(Math.min(Math.max(playhead + dir * step, 0), total));
        } else {
          onTrim(sel.beat, sel.clip, edge, dir * step);
        }
        break;
      }
      case "i":
      case "o": {
        // Put this edge where the playhead is, read back into the source file's own clock.
        if (!sel || playhead < sel.start || playhead > sel.end) return;
        onSetEdge(sel.beat, sel.clip, e.key === "i" ? "in" : "out", sel.sourceIn + (playhead - sel.start));
        break;
      }
      case "s":
        if (sel) onSnap(sel.beat, sel.clip);
        break;
      case "z":
        setZoomed((v) => !v);
        break;
      case " ":
        onTogglePlay();
        break;
      case "Escape":
        onSelect(null, -1, null);
        setZoomed(false);
        break;
      default:
        return;
    }
    e.preventDefault();
  };

  if (total <= 0) return null;

  const lanes: { kind: Block["kind"]; label: string }[] = [
    { kind: "title", label: "Titles" },
    { kind: "picture", label: "Picture" },
    { kind: "bed", label: "Sound" },
  ];

  // A tick roughly every 100 px, on a round number of seconds.
  const step = [0.5, 1, 2, 5, 10, 15, 30, 60].find((s) => (s / span) * width >= 80) ?? 60;
  const ticks: number[] = [];
  for (let t = Math.ceil(view.start / step) * step; t <= view.end; t += step) ticks.push(t);

  const seekFromEvent = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    onSeek(view.start + ((e.clientX - rect.left) / Math.max(rect.width, 1)) * span);
  };

  return (
    <div className="script-timeline" ref={wrapRef} tabIndex={0} onKeyDown={keys}>
      <div className="stl-ruler">
        {ticks.map((t) => (
          <span key={t} className="stl-tick" style={{ left: pct(t) }}>
            {fmt(t)}
          </span>
        ))}
      </div>

      {lanes.map((lane) => (
        <div key={lane.kind} className={`stl-lane stl-${lane.kind}`}>
          <div className="stl-lane-name small muted">{lane.label}</div>
          <div className="stl-track" onClick={seekFromEvent}>
            {blocks
              .filter((b) => b.kind === lane.kind && b.end > view.start && b.start < view.end)
              .map((b) => (
                <div
                  key={b.key}
                  className={[
                    "stl-block",
                    `stl-${b.kind}`,
                    b.muted ? "muted" : "",
                    b.inferred ? "inferred" : "",
                    selected === b.key ? "selected" : "",
                  ]
                    .filter(Boolean)
                    .join(" ")}
                  style={{ left: pct(b.start), width: `${((b.end - b.start) / span) * 100}%` }}
                  onClick={(e) => {
                    e.stopPropagation();
                    onSelect(b.key, b.beat, b.clip);
                    wrapRef.current?.focus();
                  }}
                  title={`${b.label}${b.detail ? ` · ${b.detail}` : ""}${
                    b.inferred ? " · voice carried under the pictures" : ""
                  }${b.muted ? " · silent" : ""}`}
                >
                  {b.kind !== "title" && (
                    <span
                      className="stl-handle in"
                      onPointerDown={(e) => {
                        e.stopPropagation();
                        drag.current = { beat: b.beat, clip: b.clip, edge: "in", x: e.clientX };
                      }}
                    />
                  )}
                  <span className="stl-label">{b.label}</span>
                  {b.detail && <span className="stl-detail">{b.detail}</span>}
                  {b.kind !== "title" && (
                    <span
                      className="stl-handle out"
                      onPointerDown={(e) => {
                        e.stopPropagation();
                        drag.current = { beat: b.beat, clip: b.clip, edge: "out", x: e.clientX };
                      }}
                    />
                  )}
                </div>
              ))}
            {playhead >= view.start && playhead <= view.end && (
              <div className="stl-playhead" style={{ left: pct(playhead) }} />
            )}
          </div>
        </div>
      ))}

      <div className="stl-keys small muted">
        <span><kbd>click</kbd> seek</span>
        <span><kbd>←</kbd><kbd>→</kbd> trim out</span>
        <span><kbd>shift</kbd> trim in</span>
        <span><kbd>alt</kbd> 0.2 s</span>
        <span><kbd>ctrl</kbd> 5 s</span>
        <span><kbd>i</kbd><kbd>o</kbd> edge at playhead</span>
        <span><kbd>s</kbd> snap to sentence</span>
        <span><kbd>z</kbd> zoom{zoomed ? " out" : " to beat"}</span>
        <span><kbd>space</kbd> play</span>
      </div>
    </div>
  );
}
