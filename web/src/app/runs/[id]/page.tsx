"use client";
import { useEffect, useMemo, useState } from "react";
import { useParams } from "next/navigation";
import { api, Brief, Evidence, RunDetail } from "@/lib/api";
import { DecisionBadge } from "@/components/Decision";
import { AblationTrace, CrossOem, Importance, Tornado } from "@/components/Charts";
import { AblationTable, CoverageHeatmap, OperatingCurve, WhatIf } from "@/components/Panels";
import Link from "next/link";

function Cite({ ids, onOpen }: { ids: string[]; onOpen: (id: string) => void }) {
  if (!ids?.length) return null;
  return (
    <span className="ml-1 inline-flex flex-wrap gap-1 align-middle">
      {ids.map((id) => (
        <button key={id} className="cite" onClick={() => onOpen(id)} title="open evidence">{id}</button>
      ))}
    </span>
  );
}

function Lines({ lines, onOpen }: { lines: { text: string; evidence_ids: string[] }[]; onOpen: (id: string) => void }) {
  return (
    <ul className="space-y-1.5 text-sm leading-relaxed">
      {lines.map((l, i) => (
        <li key={i} className="pl-3 -indent-3">
          <span className="text-ink-300">• </span>
          {l.text}
          <Cite ids={l.evidence_ids} onOpen={onOpen} />
        </li>
      ))}
    </ul>
  );
}

export default function RunPage() {
  const { id } = useParams<{ id: string }>();
  const [run, setRun] = useState<RunDetail | null>(null);
  const [brief, setBrief] = useState<Brief | null>(null);
  const [md, setMd] = useState("");
  const [evidence, setEvidence] = useState<Record<string, Evidence>>({});
  const [open, setOpen] = useState<string | null>(null);
  const [q, setQ] = useState("Why was GPS excluded from the sufficient set?");
  const [answers, setAnswers] = useState<{ role: string; content: string; evidence_ids: string[] }[]>([]);
  const [asking, setAsking] = useState(false);
  const [llm, setLlm] = useState(false);

  useEffect(() => {
    let alive = true;
    let es: EventSource | null = null;
    const load = async () => {
      const r = await api.run(id);
      if (!alive) return;
      setRun(r);
      if (r.status === "done") {
        const b = await api.brief(id);
        setBrief(b.json);
        setMd(b.markdown);
        const ev = await api.evidence(id);
        setEvidence(Object.fromEntries(ev.map((e) => [e.evidence_id, e])));
        setAnswers(await api.messages(id));
      }
    };
    load().catch(console.error);
    api.health().then((h) => setLlm(h.llm_available)).catch(() => {});
    // live step progress until the run finishes (server-sent events through the proxy)
    try {
      es = new EventSource(`/api/runs/${id}/events`);
      es.onmessage = (m) => {
        const d = JSON.parse(m.data);
        setRun((prev) => (prev ? { ...prev, status: d.status, steps: d.steps } : prev));
        if (d.status === "done" || d.status === "failed") { es?.close(); load().catch(console.error); }
      };
      es.onerror = () => es?.close();
    } catch { /* EventSource unavailable: the initial load still renders finished runs */ }
    return () => { alive = false; es?.close(); };
  }, [id]);

  const byTool = useMemo(() => {
    const m: Record<string, Evidence> = {};
    for (const e of Object.values(evidence)) m[`${e.step}:${e.tool}`] = e;
    const pick = (tool: string) => Object.values(evidence).find((e) => e.tool === tool);
    const byName = (name: string) => Object.values(evidence).find((e) => (e as any).name === name);
    return {
      fa: pick("feature_analysis"),
      ab: byName("ablation") ?? pick("ablation"),
      abd: byName("ablation_daily_cadence"),
      xo: pick("cross_oem_validation"),
      tor: pick("tornado"),
      roi: pick("roi_distribution"),
      cov: byName("coverage_sufficient"),
      op: pick("choose_operating_point"),
      cmp: pick("model_comparison"),
      cd: byName("cost_daily_cadence"),
      cs: byName("cost_sufficient"),
    };
  }, [evidence]);

  async function ask() {
    setAsking(true);
    try {
      const a = await api.ask(id, q);
      setAnswers((s) => [...s, { role: "user", content: q, evidence_ids: [] }, { role: "assistant", content: a.answer, evidence_ids: a.evidence_ids }]);
    } catch (e) { setAnswers((s) => [...s, { role: "assistant", content: String(e), evidence_ids: [] }]); }
    setAsking(false);
  }

  if (!run) return <p className="text-sm text-ink-500">Loading…</p>;
  const v = run.verdict;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">{String(run.spec.capability_name)}</h1>
          <p className="mono text-ink-500">target {String(run.spec.target_event)} · horizon {String(run.spec.horizon_days)}d · run {run.run_id} · dataset {run.dataset_hash} · {run.llm_used ? "LLM narrative" : "headless"}</p>
        </div>
        <div className="flex items-center gap-2">
          {run.status === "done" && (
            <>
              <a className="rounded border border-ink-300 px-2 py-1 text-xs text-ink-700 hover:bg-ink-100" href={`/api/runs/${id}/brief.md`} target="_blank" rel="noreferrer">Export .md</a>
              <button className="rounded border border-ink-300 px-2 py-1 text-xs text-ink-700 hover:bg-ink-100" onClick={async () => { const r = await api.replay(id); alert(`Replay queued (ticket ${r.ticket}). It appears in the runs list as kind=replay with an evidence-by-evidence diff in its messages.`); }}>Replay</button>
            </>
          )}
          <DecisionBadge decision={v?.decision ?? null} size="lg" />
        </div>
      </div>
      {run.derived_from && <p className="text-xs text-ink-500">{run.kind} of <Link className="underline" href={`/runs/${run.derived_from}`}>{run.derived_from}</Link> — harness evidence is the parent's; economics and verdict recomputed.</p>}

      {/* timeline */}
      <div className="card p-4">
        <ol className="flex flex-wrap gap-2 text-xs">
          {run.steps.map((s) => (
            <li key={s.name} className={`rounded border px-2 py-1 ${s.status === "done" ? "border-green-200 bg-green-50 text-green-800" : s.status === "running" ? "border-amber-200 bg-amber-50 text-amber-800" : "border-ink-300 bg-ink-100 text-ink-700"}`}>
              {s.name}{s.note ? <span className="text-ink-500"> · {s.note}</span> : null}
            </li>
          ))}
        </ol>
        {run.error && <p className="mt-2 text-sm text-red-700">{run.error}</p>}
      </div>

      {v && (
        <div className="grid gap-4 md:grid-cols-2">
          <div className="card p-4">
            <h2 className="mb-2 text-sm font-semibold">Hard gates <span className="font-normal text-ink-500">— any failure → not feasible</span></h2>
            <ul className="space-y-1 text-sm">
              {v.gates.map((g) => (
                <li key={g.name} className="flex items-start gap-2">
                  <span className={`mt-[3px] inline-block h-2.5 w-2.5 rounded-full ${g.passed ? "bg-green-600" : "bg-red-600"}`} />
                  <span><span className="font-medium">{g.name}</span> <span className="text-ink-500">{g.value != null ? `${Number(g.value).toFixed(3)} vs ${g.threshold != null ? Number(g.threshold).toFixed(3) : "—"}` : ""}{g.note ? ` · ${g.note}` : ""}</span><Cite ids={g.evidence_ids} onOpen={setOpen} /></span>
                </li>
              ))}
            </ul>
          </div>
          <div className="card p-4">
            <h2 className="mb-2 text-sm font-semibold">Uncertainty flags <span className="font-normal text-ink-500">— any trip → pilot</span></h2>
            <ul className="space-y-1 text-sm">
              {v.flags.map((f) => (
                <li key={f.name} className="flex items-start gap-2">
                  <span className={`mt-[3px] inline-block h-2.5 w-2.5 rounded-full ${f.tripped ? "bg-amber-500" : "bg-ink-300"}`} />
                  <span><span className="font-medium">{f.name}</span> <span className="text-ink-500">{f.value != null ? `${Number(f.value).toFixed(3)} vs ${f.threshold != null ? Number(f.threshold).toFixed(3) : "—"}` : ""}{f.note ? ` · ${f.note}` : ""}</span><Cite ids={f.evidence_ids} onOpen={setOpen} /></span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {(byTool.fa || byTool.ab || byTool.xo || byTool.tor) && (
        <div className="grid gap-4 md:grid-cols-2">
          {byTool.fa && <div className="card p-4"><h2 className="mb-1 text-sm font-semibold">Permutation importance <span className="font-normal text-ink-500">AUC drop, 95% CI</span><Cite ids={[byTool.fa.evidence_id]} onOpen={setOpen} /></h2><Importance ranking={byTool.fa.outputs.ranking} /></div>}
          {byTool.ab && <div className="card p-4"><h2 className="mb-1 text-sm font-semibold">Ablation trace <span className="font-normal text-ink-500">AUC as signals are removed</span><Cite ids={[byTool.ab.evidence_id]} onOpen={setOpen} /></h2><AblationTrace trace={byTool.ab.outputs.trace} /><p className="mt-1 text-xs text-ink-500">Sufficient set: {byTool.ab.outputs.sufficient_set.join(", ")}{byTool.ab.outputs.underpowered ? " · underpowered" : ""}</p></div>}
          {byTool.xo && <div className="card p-4"><h2 className="mb-1 text-sm font-semibold">Leave-one-OEM-out <span className="font-normal text-ink-500">AUC on the held-out OEM</span><Cite ids={[byTool.xo.evidence_id]} onOpen={setOpen} /></h2><CrossOem per_oem={byTool.xo.outputs.per_oem} mean={byTool.xo.outputs.mean_auc} /></div>}
          {byTool.tor && <div className="card p-4"><h2 className="mb-1 text-sm font-semibold">Sensitivity <span className="font-normal text-ink-500">median ROI, input low → high</span><Cite ids={[byTool.tor.evidence_id]} onOpen={setOpen} /></h2><Tornado ranked={byTool.tor.outputs.ranked} base={byTool.tor.outputs.base_median_roi} />{byTool.roi && <p className="mt-1 text-xs text-ink-500">P(ROI &gt; 0) = {byTool.roi.outputs.p_roi_positive?.toFixed(2)} · ROI p5 {byTool.roi.outputs.roi.p5.toFixed(2)} · p50 {byTool.roi.outputs.roi.p50.toFixed(2)} · p95 {byTool.roi.outputs.roi.p95.toFixed(2)}</p>}</div>}
        </div>
      )}

      {(byTool.cov || byTool.op || byTool.ab) && (
        <div className="grid gap-4 md:grid-cols-2">
          {byTool.cov && <div className="card p-4"><h2 className="mb-2 text-sm font-semibold">OEM coverage of the sufficient set <Cite ids={[byTool.cov.evidence_id]} onOpen={setOpen} /></h2><CoverageHeatmap cov={byTool.cov.outputs} /></div>}
          {byTool.op && byTool.cmp && (
            <div className="card p-4">
              <h2 className="mb-1 text-sm font-semibold">Operating point <span className="font-normal text-ink-500">events caught vs alert burden</span><Cite ids={[byTool.op.evidence_id]} onOpen={setOpen} /></h2>
              <OperatingCurve points={byTool.cmp.outputs.sets.sufficient.models.lightgbm.operating_points} chosen={byTool.op.outputs.chosen} />
              <p className="mt-1 text-xs text-ink-500">Chosen: alert top {(byTool.op.outputs.chosen.alert_rate * 100).toFixed(2)}% → {(byTool.op.outputs.chosen.recall * 100).toFixed(0)}% of events caught{byTool.op.outputs.chosen.median_lead_days != null ? `, median lead ${byTool.op.outputs.chosen.median_lead_days.toFixed(1)} d` : ""}{byTool.op.outputs.chosen.false_alerts_per_100_vehicle_months != null ? `, ${byTool.op.outputs.chosen.false_alerts_per_100_vehicle_months.toFixed(1)} false alerts / 100 vehicle-months` : ""}.</p>
            </div>
          )}
          {byTool.ab && <div className="card p-4 md:col-span-2"><h2 className="mb-2 text-sm font-semibold">Ablation steps <Cite ids={[byTool.ab.evidence_id]} onOpen={setOpen} /></h2><AblationTable ab={byTool.ab.outputs} />{byTool.abd && byTool.cd && byTool.cs && <p className="mt-2 text-xs text-ink-700">Daily-cadence-only alternative: {byTool.abd.outputs.sufficient_set.length} signals, AUC {byTool.abd.outputs.sufficient_auc.point.toFixed(3)} vs {byTool.ab.outputs.sufficient_auc.point.toFixed(3)}; marginal cost ${Math.round(byTool.cd.outputs.monthly.marginal_total).toLocaleString()}/mo vs ${Math.round(byTool.cs.outputs.monthly.marginal_total).toLocaleString()}/mo <Cite ids={[byTool.abd.evidence_id, byTool.cd.evidence_id]} onOpen={setOpen} /></p>}</div>}
        </div>
      )}

      {run.status === "done" && run.spec && (run.spec as any).value && (
        <div className="card p-4">
          <h2 className="mb-1 text-sm font-semibold">What if the value assumptions were different?</h2>
          <p className="mb-3 text-xs text-ink-500">These are the human inputs. Change them and recompute the economics and verdict without re-running the experiments.</p>
          <WhatIf runId={id} value={(run.spec as any).value} onDone={(rid) => { window.location.href = `/runs/${rid}`; }} />
        </div>
      )}

      {brief && (
        <div className="card p-5 space-y-5">
          {brief.sections.map((s) => (
            <section key={s.title}>
              <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-ink-500">{s.title}</h2>
              <Lines lines={s.lines} onOpen={setOpen} />
            </section>
          ))}
          {brief.narrative?.length > 0 && (
            <section>
              <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-ink-500">Narrative (LLM, cited)</h2>
              <p className="text-sm leading-relaxed">{brief.narrative.map((l, i) => <span key={i}>{l.text}<Cite ids={l.evidence_ids} onOpen={setOpen} /> </span>)}</p>
            </section>
          )}
          <details className="text-xs text-ink-500"><summary className="cursor-pointer">Markdown</summary><pre className="mono mt-2 whitespace-pre-wrap rounded bg-ink-100 p-3">{md}</pre></details>
        </div>
      )}

      {run.status === "done" && (
        <div className="card p-4 space-y-3">
          <h2 className="text-sm font-semibold">Ask the evidence</h2>
          {answers.map((m, i) => (
            <div key={i} className={`text-sm ${m.role === "user" ? "text-ink-700" : ""}`}>
              <span className="mono mr-2 text-ink-500">{m.role}</span>{m.content}<Cite ids={m.evidence_ids} onOpen={setOpen} />
            </div>
          ))}
          <div className="flex gap-2">
            <input className="flex-1 rounded border border-ink-300 px-3 py-2 text-sm" value={q} onChange={(e) => setQ(e.target.value)} disabled={!llm} />
            <button onClick={ask} disabled={!llm || asking} className="rounded bg-ink-900 px-3 py-2 text-sm text-white disabled:opacity-40">{asking ? "…" : "Ask"}</button>
          </div>
          {!llm && <p className="text-xs text-ink-500">Q&amp;A needs the LLM interface on the API (ANTHROPIC_API_KEY). The brief above is complete without it.</p>}
        </div>
      )}

      {open && evidence[open] && (
        <div className="fixed inset-0 z-20 flex items-center justify-center bg-ink-900/40 p-6" onClick={() => setOpen(null)}>
          <div className="card max-h-[85vh] w-full max-w-3xl overflow-auto p-5" onClick={(e) => e.stopPropagation()}>
            <div className="mb-2 flex items-baseline justify-between">
              <h3 className="text-sm font-semibold">{evidence[open].tool} <span className="mono font-normal text-ink-500">{open}</span></h3>
              <button className="text-xs text-ink-500" onClick={() => setOpen(null)}>close</button>
            </div>
            <p className="mono mb-3 text-ink-500">step {evidence[open].step} · seed {evidence[open].seed} · dataset {evidence[open].dataset_hash} · inputs {evidence[open].inputs_hash.slice(0, 12)} · {evidence[open].duration_ms.toFixed(0)} ms</p>
            <pre className="mono whitespace-pre-wrap rounded bg-ink-100 p-3">{JSON.stringify(evidence[open].outputs, null, 2)}</pre>
          </div>
        </div>
      )}
    </div>
  );
}
