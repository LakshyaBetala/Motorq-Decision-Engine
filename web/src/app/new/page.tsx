"use client";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";

export default function NewRunPage() {
  const router = useRouter();
  const [examples, setExamples] = useState<Record<string, any>>({});
  const [llm, setLlm] = useState(false);
  const [mode, setMode] = useState<"example" | "text" | "spec">("example");
  const [example, setExample] = useState("brake");
  const [text, setText] = useState("Should Fuse flag vehicles likely to need brake service in the next 7 days?");
  const [spec, setSpec] = useState("");
  const [useLlm, setUseLlm] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api.examples().then((e) => { setExamples(e); setSpec(JSON.stringify(e["brake"], null, 2)); });
    api.health().then((h) => { setLlm(h.llm_available); setUseLlm(h.llm_available); }).catch(() => {});
  }, []);

  async function submit() {
    setErr(null);
    setBusy("Queued…");
    try {
      const body = mode === "example" ? { example, use_llm: useLlm } : mode === "text" ? { text, use_llm: useLlm } : { spec: JSON.parse(spec), use_llm: useLlm };
      const { ticket } = await api.create(body);
      for (;;) {
        await new Promise((r) => setTimeout(r, 3000));
        const t = await api.ticket(ticket);
        if (t.status === "done" && t.run_id) { router.push(`/runs/${t.run_id}`); return; }
        if (t.status === "failed") { setErr(t.error ?? "failed"); setBusy(null); return; }
        setBusy("Running the default plan — quality, leakage, feature analysis, ablation, robustness, economics, policy. A few minutes.");
      }
    } catch (e) { setErr(String(e)); setBusy(null); }
  }

  return (
    <div className="mx-auto max-w-3xl space-y-5">
      <h1 className="text-xl font-semibold tracking-tight">New feasibility study</h1>
      <div className="card p-5 space-y-4">
        <div className="flex gap-2 text-sm">
          {(["example", "text", "spec"] as const).map((m) => (
            <button key={m} onClick={() => setMode(m)} className={`rounded px-3 py-1 ${mode === m ? "bg-ink-900 text-white" : "bg-ink-100 text-ink-700"}`}>
              {m === "example" ? "Example capability" : m === "text" ? "Free text (LLM)" : "Structured spec"}
            </button>
          ))}
        </div>
        {mode === "example" && (
          <div className="space-y-2">
            <select className="w-full rounded border border-ink-300 px-3 py-2 text-sm" value={example} onChange={(e) => { setExample(e.target.value); setSpec(JSON.stringify(examples[e.target.value], null, 2)); }}>
              {Object.keys(examples).map((k) => <option key={k} value={k}>{examples[k].capability_name} — {examples[k].target_event}, {examples[k].horizon_days}d</option>)}
            </select>
            <pre className="mono max-h-64 overflow-auto rounded bg-ink-100 p-3">{spec}</pre>
          </div>
        )}
        {mode === "text" && (
          <div className="space-y-2">
            <textarea className="w-full rounded border border-ink-300 px-3 py-2 text-sm" rows={4} value={text} onChange={(e) => setText(e.target.value)} />
            {!llm && <p className="text-xs text-amber-700">The LLM interface is not configured on the API (ANTHROPIC_API_KEY). Use an example or a structured spec.</p>}
          </div>
        )}
        {mode === "spec" && <textarea className="mono w-full rounded border border-ink-300 p-3" rows={18} value={spec} onChange={(e) => setSpec(e.target.value)} />}
        <label className="flex items-center gap-2 text-sm text-ink-700">
          <input type="checkbox" disabled={!llm} checked={useLlm} onChange={(e) => setUseLlm(e.target.checked)} />
          Add an LLM narrative to the brief (every numeric sentence must cite evidence or it is dropped)
        </label>
        <div className="flex items-center gap-3">
          <button disabled={!!busy || (mode === "text" && !llm)} onClick={submit} className="rounded bg-ink-900 px-4 py-2 text-sm text-white disabled:opacity-40">Run study</button>
          {busy && <span className="text-xs text-ink-500">{busy}</span>}
        </div>
        {err && <p className="text-sm text-red-700">{err}</p>}
      </div>
      <p className="text-xs text-ink-500">Value assumptions (value-bearing fraction, preventable fraction, $/event, fleet) are human inputs. The engine measures the event rate, trains and ablates, prices the signal set, and computes the verdict; it never invents the business numbers.</p>
    </div>
  );
}
