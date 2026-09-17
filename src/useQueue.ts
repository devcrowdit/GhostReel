import { useEffect, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { queueList, type Task } from "./api";

/** Live view of the background task queue (updates on every `queue` event). */
export function useQueue(): Task[] {
  const [tasks, setTasks] = useState<Task[]>([]);
  useEffect(() => {
    queueList().then(setTasks).catch(() => {});
    const un = listen<Task[]>("queue", ({ payload }) => setTasks(payload));
    return () => {
      un.then((f) => f());
    };
  }, []);
  return tasks;
}

export const isActive = (t: Task) => t.state === "queued" || t.state === "running";
