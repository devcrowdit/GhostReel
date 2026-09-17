import { invoke } from "@tauri-apps/api/core";

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
