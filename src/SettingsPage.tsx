import { useEffect, useRef, useState } from "react";
import { getChatSettings, setChatSystemPrompt, type ChatSettings } from "./api";

export default function SettingsPage() {
  const [settings, setSettings] = useState<ChatSettings | null>(null);
  const [text, setText] = useState("");
  const [saved, setSaved] = useState<"idle" | "saving" | "saved">("idle");
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    getChatSettings()
      .then((s) => {
        setSettings(s);
        setText(s.system_prompt.trim() ? s.system_prompt : s.default_system_prompt);
      })
      .catch((e) => setError(String(e)));
  }, []);

  const save = (value: string) => {
    setSaved("saving");
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(async () => {
      try {
        setSettings(await setChatSystemPrompt(value));
        setSaved("saved");
        setError(null);
      } catch (e) {
        setError(String(e));
        setSaved("idle");
      }
    }, 800);
  };

  if (!settings) return <main>{error ? <div className="banner bad">{error}</div> : <p className="muted">Loading…</p>}</main>;

  const isDefault = text.trim() === settings.default_system_prompt.trim() || !text.trim();

  return (
    <main>
      <header>
        <div>
          <h1>Settings</h1>
          <p className="muted">Saved in ~/.ghostreel/config.toml</p>
        </div>
      </header>

      <h2>Script chat · editor instructions</h2>
      <section className="card">
        <p className="muted small">
          The system prompt that tells the model how to edit: story, shot choice, pacing, narration and audio. It applies
          to the next chat message. <code>{"{project}"}</code>, <code>{"{fps}"}</code>, <code>{"{width}"}</code> and{" "}
          <code>{"{height}"}</code> are filled in. The footage tools and the rule that clips must come from indexed
          footage are always added, so scripts keep working whatever you write here.
        </p>
        {error && <div className="banner bad">{error}</div>}
        <textarea
          className="prompt-editor"
          value={text}
          spellCheck={false}
          onChange={(e) => {
            setText(e.target.value);
            save(e.target.value);
          }}
        />
        <div className="row prompt-actions">
          <span className="muted small">
            {isDefault ? "Using the default instructions" : "Using your instructions"}
            {saved === "saving" ? " · saving…" : saved === "saved" ? " · saved" : ""}
          </span>
          <button
            className="ghost small"
            disabled={isDefault}
            onClick={() => {
              setText(settings.default_system_prompt);
              save("");
            }}
          >
            Reset to default
          </button>
        </div>
      </section>
    </main>
  );
}
