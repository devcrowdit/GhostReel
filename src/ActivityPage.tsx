import { clearFinishedTasks } from "./api";
import TaskCard from "./TaskCard";
import { isActive, useQueue } from "./useQueue";

export default function ActivityPage() {
  const tasks = useQueue();
  const active = tasks.filter(isActive);
  const finished = tasks.filter((t) => !isActive(t)).reverse();
  return (
    <main>
      <header>
        <div>
          <h1>Activity</h1>
          <p className="muted">Background work runs one task at a time, in order.</p>
        </div>
        {finished.length > 0 && (
          <button className="ghost" onClick={() => clearFinishedTasks()}>
            Clear finished
          </button>
        )}
      </header>
      <h2>Now and next</h2>
      <section className="stack">
        {active.length === 0 ? <div className="muted">Nothing running.</div> : active.map((t) => <TaskCard key={t.id} task={t} />)}
      </section>
      {finished.length > 0 && (
        <>
          <h2>Finished</h2>
          <section className="stack">
            {finished.map((t) => (
              <TaskCard key={t.id} task={t} />
            ))}
          </section>
        </>
      )}
    </main>
  );
}
