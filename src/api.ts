import { convertFileSrc, invoke } from "@tauri-apps/api/core";

export const fileUrl = (path: string) => convertFileSrc(path);

// Mirrors ghostreel-core's doctor::Report (serde field names).
export type Target = "server" | "local" | "unavailable";
export type Backend = "auto" | "local" | "server";

export interface Probe {
  url: string;
  reachable: boolean;
  capable: boolean;
  model: string | null;
  detail: string;
}

export interface Resolution {
  backend: Backend;
  target: Target;
  probe: Probe | null;
  reason: string;
}

export interface Tool {
  name: string;
  path: string | null;
  version: string | null;
}

export interface Gpu {
  name: string;
  vram_total_mib: number;
  vram_used_mib: number;
  driver: string;
}

export interface ModelFile {
  role: string;
  pattern: string;
  found: string | null;
}

export interface Report {
  version: string;
  config_file: string;
  config_error: string | null;
  data_dir: string;
  db: {
    path: string;
    ok: boolean;
    schema_version: number | null;
    sqlite_vec: string | null;
    error: string | null;
  };
  ffmpeg: Tool;
  ffprobe: Tool;
  gpu: Gpu[];
  vision: Resolution;
  embeddings: Resolution;
  stt: Resolution;
  models: ModelFile[];
}

export interface DoctorView {
  report: Report;
  blockers: string[];
}

export const doctor = () => invoke<DoctorView>("doctor");

// ---- projects & library ------------------------------------------------------------------

export interface Project {
  id: number;
  name: string;
  description: string;
  fps_num: number;
  fps_den: number;
  width: number;
  height: number;
  created_at: number;
}

export interface StageCounts {
  stage: string;
  pending: number;
  running: number;
  done: number;
  failed: number;
  skipped: number;
}

export interface Status {
  folders: number;
  videos: number;
  total_size: number;
  total_duration_s: number;
  vfr_videos: number;
  stages: StageCounts[];
}

export interface ProjectSummary {
  project: Project;
  status: Status;
}

export interface FolderView {
  id: number;
  path: string;
  recursive: boolean;
  enabled: boolean;
  available: boolean;
}

export interface VideoRow {
  id: number;
  path: string;
  copies: number;
  size: number;
  duration_s: number | null;
  width: number | null;
  height: number | null;
  fps: number | null;
  vfr: boolean;
  vcodec: string | null;
  has_audio: boolean | null;
  status: string;
  error: string | null;
  language: string | null;
  segments: number;
  transcribe: string | null;
  frames: number;
}

export interface FrameRow {
  id: number;
  t_s: number;
  path: string;
  description: string | null;
  visible_text: string | null;
}

export const videoFrames = (videoId: number) => invoke<FrameRow[]>("video_frames", { videoId });

export interface TranscriptSegment {
  start: number;
  end: number;
  text: string;
}

export const videoTranscript = (videoId: number) => invoke<TranscriptSegment[]>("video_transcript", { videoId });

export interface ProjectView {
  project: Project;
  folders: FolderView[];
  status: Status;
  videos: VideoRow[];
}

export type IndexEvent =
  | { event: "scan_folder"; path: string }
  | { event: "folder_missing"; path: string }
  | { event: "scanned"; new: number; changed: number; unchanged: number; removed: number }
  | { event: "job_started"; video_id: number; stage: string; path: string }
  | { event: "job_done"; video_id: number; stage: string }
  | { event: "job_failed"; video_id: number; stage: string; error: string }
  | { event: "stage_backend"; stage: string; backend: string }
  | { event: "stage_unavailable"; stage: string; reason: string }
  | { event: "downloading_model"; file: string }
  | ({ event: "progress" } & Progress);

export interface Progress {
  phase: string;
  phase_done: number;
  phase_total: number;
  fraction: number;
  eta_secs: number | null;
  elapsed_secs: number;
  current: string | null;
}

export const PHASE_LABELS: Record<string, string> = {
  hash: "Reading new files",
  probe: "Reading video details",
  download: "Downloading speech model",
  transcribe_server: "Transcribing (GhostPen)",
  transcribe_local: "Transcribing",
  frames: "Picking keyframes",
  download_vision: "Downloading vision model",
  describe_server: "Describing frames",
  describe_local: "Describing frames",
};

/** "a few seconds", "42 s", "about 2 min", "about 1 h 20 min" — same wording as the CLI. */
export const etaText = (secs: number) => {
  const s = Math.max(0, Math.round(secs));
  if (s < 10) return "a few seconds";
  if (s < 60) return `${s} s`;
  if (s < 3600) return `about ${Math.ceil(s / 60)} min`;
  return `about ${Math.floor(s / 3600)} h ${String(Math.floor((s % 3600) / 60)).padStart(2, "0")} min`;
};

export interface IndexSummary {
  new: number;
  changed: number;
  unchanged: number;
  removed: number;
  jobs_done: number;
  jobs_failed: number;
  unsettled: number;
  cancelled: boolean;
}

export type TaskState = "queued" | "running" | "done" | "failed" | "cancelled";

export interface Task {
  id: number;
  kind: { type: "index"; project_id: number };
  label: string;
  state: TaskState;
  progress: Progress | null;
  note: string | null;
  summary: IndexSummary | null;
  error: string | null;
  created_at: number;
  finished_at: number | null;
}

export const enqueueIndex = (projectId: number) => invoke<number>("enqueue_index", { projectId });
export const queueList = () => invoke<Task[]>("queue_list");
export const cancelTask = (id: number) => invoke<boolean>("cancel_task", { id });
export const clearFinishedTasks = () => invoke<void>("clear_finished_tasks");

export interface Hit {
  video_id: number;
  path: string;
  start_s: number;
  end_s: number;
  score: number;
  kinds: string[];
  matched_by: string[];
  snippet: string;
  frame: string | null;
}

export const search = (projectId: number, query: string, limit = 30) =>
  invoke<{ hits: Hit[]; note: string | null }>("search", { projectId, query, limit });
export const openExternal = (path: string, t: number) => invoke<void>("open_external", { path, t });

export const clock = (s: number) => {
  const t = Math.max(0, Math.floor(s));
  const h = Math.floor(t / 3600);
  const m = Math.floor((t % 3600) / 60);
  const sec = String(t % 60).padStart(2, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${sec}` : `${m}:${sec}`;
};

export const fileName = (p: string) => p.split(/[\\/]/).pop() ?? p;

export const listProjects = () => invoke<ProjectSummary[]>("list_projects");
export const createProject = (name: string, fpsNum: number, fpsDen: number, width: number, height: number) =>
  invoke<Project>("create_project", { name, fpsNum, fpsDen, width, height });
export const removeProject = (projectId: number) => invoke<void>("remove_project", { projectId });
export const projectView = (projectId: number) => invoke<ProjectView>("project_view", { projectId });
export const addFolder = (projectId: number, path: string, recursive = true) =>
  invoke<void>("add_folder", { projectId, path, recursive });
export const removeFolder = (projectId: number, path: string) => invoke<void>("remove_folder", { projectId, path });


export const FPS_PRESETS: { label: string; num: number; den: number }[] = [
  { label: "23.976", num: 24000, den: 1001 },
  { label: "24", num: 24, den: 1 },
  { label: "25", num: 25, den: 1 },
  { label: "29.97", num: 30000, den: 1001 },
  { label: "30", num: 30, den: 1 },
  { label: "50", num: 50, den: 1 },
  { label: "59.94", num: 60000, den: 1001 },
  { label: "60", num: 60, den: 1 },
];

export const fpsLabel = (num: number, den: number) =>
  FPS_PRESETS.find((p) => p.num === num && p.den === den)?.label ?? (num / den).toFixed(3);

export const humanSize = (bytes: number) =>
  bytes >= 1e9 ? `${(bytes / 1e9).toFixed(1)} GB` : `${Math.round(bytes / 1e6)} MB`;

export const humanDuration = (s: number) => {
  const t = Math.round(s);
  const h = Math.floor(t / 3600);
  const m = Math.floor((t % 3600) / 60);
  const sec = t % 60;
  return h > 0 ? `${h}h ${String(m).padStart(2, "0")}m` : `${m}:${String(sec).padStart(2, "0")}`;
};
