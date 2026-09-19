import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { Script, VideoRow } from "./api";

/** One block on a lane, already positioned in seconds on the cut's timeline. */
type Block = {
  key: string;
  kind: "title" | "picture" | "bed";
  beat: number;
  clip: number | null;
  start: number;
  end: number;
  label: string;
  /** Sub-label: the source range, or nothing for a title. */
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
  onSelect: (key: string | null, beat: number, clip: number | null) => void;
  onTrim: (beat: number, clip: number | null, edge: "in" | "out", deltaSeconds: number) => void;
};

/**
 * The cut on three lanes, the way an NLE shows it: what is written over the picture, the picture
 * itself, and the sound. A list of beats hides the one thing that matters most here — that a voice
 * runs on while the pictures change — because a bed has no place in a list.
 */
export default function ScriptTimeline({ script, videos, selected, onSelect, onTrim }: Props) {
  const { blocks, total } = layout(script, videos);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(900);
  const drag = useRef<{ key: string; beat: number; clip: number | null; edge: "in" | "out"; x: number } | null>(null);

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
    if (total <= 0) return;
    const perPx = total / Math.max(width, 1);
    const move = (e: PointerEvent) => {
      const d = drag.current;
      if (!d) return;
      const delta = (e.clientX - d.x) * perPx;
      if (Math.abs(delta) < 0.04) return;
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
  }, [total, width, onTrim]);

  if (total <= 0) return null;

  const pct = (v: number) => `${(v / total) * 100}%`;
  const lanes: { kind: Block["kind"]; label: string }[] = [
    { kind: "title", label: "Titles" },
    { kind: "picture", label: "Picture" },
    { kind: "bed", label: "Sound" },
  ];

  // A tick roughly every 100 px, on a round number of seconds.
  const step = [1, 2, 5, 10, 15, 30, 60].find((s) => (s / total) * width >= 80) ?? 60;
  const ticks: number[] = [];
  for (let t = 0; t <= total; t += step) ticks.push(t);

  return (
    <div className="script-timeline" ref={wrapRef}>
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
          <div className="stl-track">
            {blocks
              .filter((b) => b.kind === lane.kind)
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
                  style={{ left: pct(b.start), width: pct(Math.max(b.end - b.start, total / 400)) }}
                  onClick={() => onSelect(b.key, b.beat, b.clip)}
                  title={`${b.label}${b.detail ? ` · ${b.detail}` : ""}${
                    b.inferred ? " · voice carried under the pictures" : ""
                  }${b.muted ? " · silent" : ""}`}
                >
                  {b.kind !== "title" && (
                    <span
                      className="stl-handle in"
                      onPointerDown={(e) => {
                        e.stopPropagation();
                        drag.current = { key: b.key, beat: b.beat, clip: b.clip, edge: "in", x: e.clientX };
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
                        drag.current = { key: b.key, beat: b.beat, clip: b.clip, edge: "out", x: e.clientX };
                      }}
                    />
                  )}
                </div>
              ))}
          </div>
        </div>
      ))}

      <div className="stl-legend small muted">
        Drag a block's edge to trim it. A striped bar is a voice carried under the pictures — drag
        it to change how long it runs.
      </div>
    </div>
  );
}
