"use client";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";

const STAGES = ["queued", "DEFINE", "FEASIBILITY", "EXPERIMENT", "ECONOMICS", "DELIVERY", "POLICY", "REPORT"];

export default function NewRunPage() {
  const router = useRouter();
  const [examples, setExamples] = useState<Record<string, any>>({});
  const [llm, setLlm] = useState(false);
  const [mode, setMode] = useState<"example" | "text" | "spec">("example");
  const [example, setExample] = useState("brake");
  const [text, setText] = useState("Should Fuse flag vehicles likely to need brake service in the next 7 days?");
  const [spec, setSpec] = useState("");
  const [useLlm, setUseLlm] = useState(false);
  const [progress, setProgress] = useState<{ stage: string; ticket: string } | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api.examples().then((e) => { setExamples(e); setSpec(JSON.stringify(e["brake"], null, 2)); });
    api.health().then((h) => { setLlm(h.llm_available); setUseLlm(h.llm_available); }).catch(() => {});
  }, []);

  async function submit() {
    setErr(null);
    try {
      const body = mode === "example" ? { example, use_llm: useLlm } : mode === "text" ? { text, use_llm: useLlm } : { spec: JSON.parse(spec), use_llm: useLlm };
      const { ticket } = await api.create(body);
      setProgress({ stage: "queued", ticket });
      for (;;) {
        await new Promise((r) => setTimeout(r, 2500));
        const t = await api.ticket(ticket);
        if (t.status === "done" && t.run_id) { router.push(`/runs/${t.run_id}`); return; }
        if (t.status === "failed") { setErr(t.error ?? "The study failed. The API log has the traceback."); setProgress(null); return; }
        if (t.run_id) {
          try {
            const r = await api.run(t.run_id);
            const running = r.steps.find((s) => s.status === "running")?.name ?? r.steps.filter((s) => s.status === "done").slice(-1)[0]?.name ?? "queued";
            setProgress({ stage: running, ticket });
          } catch { /* keep the last stage */ }
        }
      }
    } catch (e) { setErr(String(e)); setProgress(null); }
  }

  const stageIndex = progress ? STAGES.indexOf(progress.stage) : -1;

  return (
    <div className="mx-auto max-w-2xl space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">New feasibility study</h1>
        <p className="mt-1 max-w-[62ch] text-sm text-ink-700">Pick an example capability, describe one in plain language, or paste a structured spec. The engine measures the event rate, trains and ablates, prices the signal set and computes the verdict. It never invents the business numbers.</p>
      </div>

      <div className="panel space-y-5 p-5">
        <div className="field">
          <label>How to specify the study</label>
          <div className="flex flex-wrap gap-2" role="tablist">
            {(["example", "text", "spec"] as const).map((m) => (
              <button key={m} role="tab" aria-selected={mode === m} onClick={() => setMode(m)} className={`btn ${mode === m ? "btn-primary" : ""}`}>
                {m === "example" ? "Example capability" : m === "text" ? "Plain language" : "Structured spec"}
              </button>
            ))}
          </div>
        </div>

        {mode === "example" && (
          <div className="field">
            <label htmlFor="example">Capability</label>
            <select id="example" className="input" value={example} onChange={(e) => { setExample(e.target.value); setSpec(JSON.stringify(examples[e.target.value], null, 2)); }}>
              {Object.keys(examples).map((k) => <option key={k} value={k}>{examples[k].capability_name}: {examples[k].target_event} within {examples[k].horizon_days} days</option>)}
            </select>
            <details className="text-xs text-ink-500"><summary className="cursor-pointer">The spec this example sends</summary><pre className="mono mt-2 max-h-64 overflow-auto rounded-md border border-ink-300 bg-ink-100 p-3 text-ink-700">{spec}</pre></details>
          </div>
        )}
        {mode === "text" && (
          <div className="field">
            <label htmlFor="text">The question</label>
            <textarea id="text" className="input" rows={4} value={text} onChange={(e) => setText(e.target.value)} disabled={!llm} />
            <p className="text-xs text-ink-500">{llm ? "The language model turns this into a spec: target, horizon, consumer. It cannot set a coverage or latency constraint you did not state, and the value assumptions are defaulted and labelled as such." : "Plain-language input needs a language-model provider on the API. Use an example or a structured spec."}</p>
          </div>
        )}
        {mode === "spec" && (
          <div className="field">
            <label htmlFor="spec">ProblemSpec (JSON)</label>
            <textarea id="spec" className="input mono" rows={18} value={spec} onChange={(e) => setSpec(e.target.value)} />
          </div>
        )}

        <label className="flex items-start gap-2 text-sm text-ink-700">
          <input type="checkbox" className="mt-1" disabled={!llm} checked={useLlm} onChange={(e) => setUseLlm(e.target.checked)} />
          <span>Add a narrative to the brief. It is written from the evidence; a sentence with a number and no citation is removed.{!llm && <span className="text-ink-500"> (needs a language-model provider)</span>}</span>
        </label>

        <div className="flex flex-wrap items-center gap-3">
          <button disabled={!!progress || (mode === "text" && !llm)} onClick={submit} className="btn btn-primary">{progress ? "Running" : "Run study"}</button>
          {progress && (
            <ol className="flex flex-wrap gap-1 text-xs" aria-live="polite">
              {STAGES.slice(1).map((s, i) => {
                const idx = i + 1;
                const state = idx < stageIndex ? "done" : idx === stageIndex ? "running" : "pending";
                return (
                  <li key={s} className={`rounded-md border px-2 py-0.5 ${state === "done" ? "border-ink-300 bg-white text-ink-700" : state === "running" ? "border-verdict-pilot text-verdict-pilot" : "border-ink-200 text-ink-400"}`}>
                    {s.charAt(0) + s.slice(1).toLowerCase()}
                  </li>
                );
              })}
            </ol>
          )}
        </div>
        {progress && <p className="text-xs text-ink-500">A study on the demo fleet takes about three minutes. You will be taken to the brief when it finishes.</p>}
        {err && <p className="text-sm text-verdict-no">{err}</p>}
      </div>
    </div>
  );
}
