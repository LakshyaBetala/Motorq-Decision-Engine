"use client";
import { useEffect, useMemo, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { api, Brief, Evidence, RunDetail } from "@/lib/api";
import { DecisionBadge, decisionTone } from "@/components/Decision";
import { AblationTrace, CrossOem, CrossOemCaption, Importance, Tornado } from "@/components/Charts";
import { AblationTable, CoverageHeatmap, OperatingCurve, WhatIf } from "@/components/Panels";
import { DataQuality, LearningCurve, PolicyMargins, RedundancyGroups, RuntimeCompute, SeedStability, TuningHeadroom } from "@/components/Evidence";
import { Cite, Drawer, SectionHead, Skeleton } from "@/components/Ui";

const STAGES = ["DEFINE", "FEASIBILITY", "EXPERIMENT", "ECONOMICS", "DELIVERY", "POLICY", "REPORT"];
const STAGE_HINT: Record<string, string> = {
  DEFINE: "the question and the human inputs",
  FEASIBILITY: "do we have the data",
  EXPERIMENT: "can a model predict it, with how little",
  ECONOMICS: "what it costs and what it is worth",
  DELIVERY: "how it would ship",
  POLICY: "the verdict",
  REPORT: "the cited brief",
};

function Lines({ lines, onOpen, show }: { lines: { text: string; evidence_ids: string[] }[]; onOpen: (id: string) => void; show?: number }) {
  // long sections show their first lines and fold the rest; the export has every line
  const [all, setAll] = useState(false);
  const cut = show != null && !all && lines.length > show + 1;
  const visible = cut ? lines.slice(0, show) : lines;
  return (
    <div className="max-w-[78ch]">
      <ul className="space-y-2 text-sm leading-relaxed text-ink-900">
        {visible.map((l, i) => (
          <li key={i} className="border-l-2 border-ink-200 pl-3">
            {l.text}
            <Cite ids={l.evidence_ids} onOpen={onOpen} />
          </li>
        ))}
      </ul>
      {cut && (
        <button type="button" onClick={() => setAll(true)} className="mt-2 text-xs text-accent hover:underline">
          Show all {lines.length} cited lines
        </button>
      )}
    </div>
  );
}

function Ledger({ items, kind, onOpen }: { items: { name: string; ok: boolean; value: number | null; threshold: number | null; note: string; evidence_ids: string[] }[]; kind: "gate" | "flag"; onOpen: (id: string) => void }) {
  return (
    <ul className="divide-y divide-ink-200">
      {items.map((g) => (
        <li key={g.name} className="grid grid-cols-[10px_minmax(0,1fr)_auto] items-baseline gap-x-3 py-1.5 text-sm">
          <span className={`mt-[6px] inline-block h-2 w-2 rounded-full ${kind === "gate" ? (g.ok ? "bg-verdict-build" : "bg-verdict-no") : g.ok ? "bg-ink-300" : "bg-verdict-pilot"}`} aria-label={kind === "gate" ? (g.ok ? "pass" : "fail") : g.ok ? "clear" : "tripped"} />
          <span className="min-w-0">
            <span className="font-medium text-ink-950">{g.name.replaceAll("_", " ")}</span>
            {g.note && <span className="text-ink-500"> — {g.note}</span>}
            <Cite ids={g.evidence_ids} onOpen={onOpen} />
          </span>
          <span className="num text-right text-ink-700">
            {g.value != null ? Number(g.value).toFixed(3) : ""}
            {g.threshold != null && <span className="text-ink-400"> / {Number(g.threshold).toFixed(3)}</span>}
          </span>
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
  const [replayed, setReplayed] = useState<string | null>(null);

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
    // live stage progress until the run finishes (server-sent events through the proxy)
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
      contract: byName("data_contract"),
      quality: byName("quality_sufficient"),
      red: byName("redundancy"),
      lc: byName("learning_curve"),
      rt: byName("runtime"),
      comp: byName("compute"),
      th: byName("tuning_headroom"),
      ss: byName("seed_stability"),
      ps: byName("policy_sensitivity"),
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

  if (!run) return <Skeleton lines={6} className="max-w-3xl" />;
  const v = run.verdict;
  const tone = decisionTone(v?.decision ?? null);
  const gatesFailed = v ? v.gates.filter((g) => !g.passed).length : 0;
  const flagsTripped = v ? v.flags.filter((f) => f.tripped).length : 0;
  const stepOf = (name: string) => run.steps.find((s) => s.name === name);
  const sections = brief ? Object.fromEntries(brief.sections.map((s) => [s.title, s])) : {};
  const chosen = byTool.op?.outputs.chosen;
  const suff: string[] = byTool.ab?.outputs.sufficient_set ?? [];

  return (
    <div className="grid gap-8 lg:grid-cols-[168px_minmax(0,1fr)]">
      {/* stage rail: the pipeline is a real sequence, so it is numbered */}
      <aside className="hidden lg:block">
        <ol className="sticky top-20 space-y-1 text-[13px]">
          {STAGES.map((name, i) => {
            const s = stepOf(name);
            const status = s?.status ?? "pending";
            return (
              <li key={name}>
                <a href={`#${name.toLowerCase()}`} className="group flex items-start gap-2 rounded-md px-2 py-1 hover:bg-white">
                  <span className={`num mt-[2px] w-4 text-right text-[11px] ${status === "done" ? "text-ink-500" : status === "running" ? "text-verdict-pilot" : "text-ink-300"}`}>{i + 1}</span>
                  <span className="min-w-0">
                    <span className={`block ${status === "done" ? "text-ink-950" : status === "running" ? "text-verdict-pilot" : "text-ink-400"}`}>
                      {name.charAt(0) + name.slice(1).toLowerCase()}
                      {status === "running" && <span className="live ml-1.5 inline-block h-1.5 w-1.5 rounded-full bg-verdict-pilot align-middle" />}
                    </span>
                    <span className="block text-[11px] leading-tight text-ink-400">{STAGE_HINT[name]}</span>
                  </span>
                </a>
              </li>
            );
          })}
        </ol>
      </aside>

      <div className="min-w-0 space-y-10">
        {/* verdict strip */}
        <header className={`sticky top-12 z-20 -mx-6 border-b border-ink-300 bg-ink-100/95 px-6 py-3 backdrop-blur lg:mx-0 lg:rounded-md lg:border lg:px-4`}>
          <div className="flex flex-wrap items-center justify-between gap-x-6 gap-y-2">
            <div className="flex items-center gap-4">
              <DecisionBadge decision={v?.decision ?? null} size="lg" />
              <div className="min-w-0">
                <h1 className="truncate text-lg font-semibold tracking-tight text-ink-950">{String(run.spec.capability_name)}</h1>
                <p className="mono text-ink-500">
                  {String(run.spec.target_event)} within {String(run.spec.horizon_days)} d
                  {run.kind !== "study" && <> — {run.kind} of <Link className="underline" href={`/runs/${run.derived_from}`}>{run.derived_from}</Link></>}
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              {v && (
                <p className="mr-2 hidden text-sm text-ink-700 md:block">
                  <span className={`num font-semibold ${gatesFailed ? "text-verdict-no" : "text-verdict-build"}`}>{v.gates.length - gatesFailed}/{v.gates.length}</span> gates
                  <span className="mx-2 text-ink-300">|</span>
                  <span className={`num font-semibold ${flagsTripped ? "text-verdict-pilot" : "text-ink-700"}`}>{flagsTripped}</span> {flagsTripped === 1 ? "flag" : "flags"}
                </p>
              )}
              {run.status === "done" && (
                <>
                  <a className="btn py-1" href={`/api/runs/${id}/brief.md`} target="_blank" rel="noreferrer">Export brief</a>
                  <button className="btn py-1" disabled={!!replayed} onClick={async () => { const r = await api.replay(id); setReplayed(r.ticket); }}>{replayed ? "Replay queued" : "Replay"}</button>
                </>
              )}
            </div>
          </div>
          {replayed && <p className="mt-2 text-xs text-ink-700">Replay queued (ticket {replayed}). It recomputes every stage with the fit cache bypassed and appears in the study list with an evidence-by-evidence diff.</p>}
          {run.error && <p className="mt-2 text-sm text-verdict-no">{run.error}</p>}
        </header>

        {run.status !== "done" && !run.error && (
          <section>
            <p className="text-sm text-ink-700">Running. Stages light up on the left as they finish; the page fills in when the verdict is ready.</p>
            <Skeleton lines={4} className="mt-4 max-w-2xl" />
          </section>
        )}

        {/* 1 DEFINE */}
        {brief && (
          <section id="define">
            <SectionHead title="Definition" hint="the question and the human inputs" />
            <Lines lines={sections["Specification"]?.lines ?? []} onOpen={setOpen} />
          </section>
        )}

        {/* 2 FEASIBILITY */}
        {brief && (
          <section id="feasibility" className="space-y-5">
            <SectionHead title="Data" hint="do we have the data, and can we trust it" />
            {(byTool.contract || byTool.quality) && <DataQuality contract={byTool.contract?.outputs} quality={byTool.quality?.outputs} sufficient={suff} />}
            {byTool.cov && (
              <div>
                <h3 className="mb-2 text-sm font-medium text-ink-950">OEM coverage of the sufficient set <Cite ids={[byTool.cov.evidence_id]} onOpen={setOpen} /></h3>
                <CoverageHeatmap cov={byTool.cov.outputs} />
              </div>
            )}
            <Lines lines={sections["Data feasibility"]?.lines ?? []} onOpen={setOpen} show={5} />
          </section>
        )}

        {/* 3 EXPERIMENT */}
        {brief && (
          <section id="experiment" className="space-y-6">
            <SectionHead title="Model" hint="can it be predicted, and with how few signals" />
            <div className="grid gap-6 md:grid-cols-2">
              {byTool.fa && <div className="panel p-4"><h3 className="mb-1 text-sm font-medium">Permutation importance <span className="font-normal text-ink-500">AUC drop when a signal is shuffled</span><Cite ids={[byTool.fa.evidence_id]} onOpen={setOpen} /></h3><Importance ranking={byTool.fa.outputs.ranking} /></div>}
              {byTool.ab && <div className="panel p-4"><h3 className="mb-1 text-sm font-medium">Ablation <span className="font-normal text-ink-500">AUC as signals are removed</span><Cite ids={[byTool.ab.evidence_id]} onOpen={setOpen} /></h3><AblationTrace trace={byTool.ab.outputs.trace} /><p className="mono mt-1 text-ink-700">sufficient set: {suff.join(", ")}{byTool.ab.outputs.underpowered ? " (underpowered)" : ""}</p></div>}
              {byTool.lc && <div className="panel p-4"><h3 className="mb-1 text-sm font-medium">Learning curve <span className="font-normal text-ink-500">was this much data needed</span><Cite ids={[byTool.lc.evidence_id]} onOpen={setOpen} /></h3><LearningCurve lc={byTool.lc.outputs} /></div>}
              {byTool.red && <div className="panel p-4"><h3 className="mb-1 text-sm font-medium">Signal redundancy <span className="font-normal text-ink-500">which signals say the same thing</span><Cite ids={[byTool.red.evidence_id]} onOpen={setOpen} /></h3><RedundancyGroups red={byTool.red.outputs} sufficient={suff} /></div>}
              {byTool.xo && <div className="panel p-4"><h3 className="mb-1 text-sm font-medium">Leave-one-OEM-out <span className="font-normal text-ink-500">AUC on the held-out OEM</span><Cite ids={[byTool.xo.evidence_id]} onOpen={setOpen} /></h3><CrossOem per_oem={byTool.xo.outputs.per_oem} mean={byTool.xo.outputs.mean_auc} /><CrossOemCaption mean={byTool.xo.outputs.mean_auc} /></div>}
              {byTool.th && <div className="panel p-4"><h3 className="mb-1 text-sm font-medium">Tuning headroom <span className="font-normal text-ink-500">how loose is the lower bound</span><Cite ids={[byTool.th.evidence_id]} onOpen={setOpen} /></h3><TuningHeadroom th={byTool.th.outputs} /></div>}
              {byTool.ss && <div className="panel p-4"><h3 className="mb-1 text-sm font-medium">Seed stability <span className="font-normal text-ink-500">would another split tell a different story</span><Cite ids={[byTool.ss.evidence_id]} onOpen={setOpen} /></h3><SeedStability ss={byTool.ss.outputs} /></div>}
              {byTool.op && byTool.cmp && chosen && (
                <div className="panel p-4">
                  <h3 className="mb-1 text-sm font-medium">Operating point <span className="font-normal text-ink-500">events caught vs alert burden</span><Cite ids={[byTool.op.evidence_id]} onOpen={setOpen} /></h3>
                  <OperatingCurve points={byTool.cmp.outputs.sets.sufficient.models.lightgbm.operating_points} chosen={chosen} />
                  <p className="mt-1 text-xs text-ink-700">
                    Alert on the top <span className="num">{(chosen.alert_rate * 100).toFixed(2)}%</span> of vehicle-days: <span className="num">{(chosen.recall * 100).toFixed(0)}%</span> of events caught
                    {chosen.median_lead_days != null && <>, median lead <span className="num">{chosen.median_lead_days.toFixed(1)}</span> d</>}
                    {chosen.false_alerts_per_100_vehicle_months != null && <>, <span className="num">{chosen.false_alerts_per_100_vehicle_months.toFixed(1)}</span> false alerts per 100 vehicle-months</>}.
                    {chosen.recall_lo != null && <> Recall interval <span className="num">{(chosen.recall_lo * 100).toFixed(0)}–{(chosen.recall_hi * 100).toFixed(0)}%</span> ({chosen.recall_ci_basis === "bootstrap" ? "vehicle-cluster bootstrap of event recall" : "proxy from the AUC interval"}); this interval feeds the ROI distribution.</>}
                  </p>
                </div>
              )}
            </div>
            {byTool.ab && (
              <div>
                <h3 className="mb-2 text-sm font-medium">Ablation steps <Cite ids={[byTool.ab.evidence_id]} onOpen={setOpen} /></h3>
                <AblationTable ab={byTool.ab.outputs} />
                {byTool.abd && byTool.cd && byTool.cs && (
                  <p className="mt-2 max-w-[78ch] text-sm text-ink-700">
                    Daily-cadence-only alternative: {byTool.abd.outputs.sufficient_set.length} signals, AUC <span className="num">{byTool.abd.outputs.sufficient_auc.point.toFixed(3)}</span> vs <span className="num">{byTool.ab.outputs.sufficient_auc.point.toFixed(3)}</span>; marginal cost <span className="num">${Math.round(byTool.cd.outputs.monthly.marginal_total).toLocaleString()}</span>/mo vs <span className="num">${Math.round(byTool.cs.outputs.monthly.marginal_total).toLocaleString()}</span>/mo
                    <Cite ids={[byTool.abd.evidence_id, byTool.cd.evidence_id]} onOpen={setOpen} />
                  </p>
                )}
              </div>
            )}
            <Lines lines={sections["Model"]?.lines ?? []} onOpen={setOpen} show={4} />
            <Lines lines={sections["Robustness"]?.lines ?? []} onOpen={setOpen} show={4} />
          </section>
        )}

        {/* 4 ECONOMICS */}
        {brief && (
          <section id="economics" className="space-y-5">
            <SectionHead title="Economics" hint="what it costs to run and what it is worth" />
            {byTool.tor && (
              <div className="panel max-w-2xl p-4">
                <h3 className="mb-1 text-sm font-medium">Sensitivity <span className="font-normal text-ink-500">median ROI as each input goes low to high</span><Cite ids={[byTool.tor.evidence_id]} onOpen={setOpen} /></h3>
                <Tornado ranked={byTool.tor.outputs.ranked} base={byTool.tor.outputs.base_median_roi} />
                {byTool.roi && (
                  <p className="mt-1 text-xs text-ink-700">
                    P(ROI &gt; 0) = <span className="num">{byTool.roi.outputs.p_roi_positive?.toFixed(2)}</span>; ROI 5th / 50th / 95th percentile <span className="num">{byTool.roi.outputs.roi.p5.toFixed(2)} / {byTool.roi.outputs.roi.p50.toFixed(2)} / {byTool.roi.outputs.roi.p95.toFixed(2)}</span>
                    <Cite ids={[byTool.roi.evidence_id]} onOpen={setOpen} />
                  </p>
                )}
              </div>
            )}
            <Lines lines={sections["Economics"]?.lines ?? []} onOpen={setOpen} show={5} />
            {run.status === "done" && (run.spec as any).value && (
              <div className="panel max-w-2xl p-4">
                <h3 className="text-sm font-medium">Change the value assumptions</h3>
                <p className="mb-3 mt-1 text-xs text-ink-700">These are the human inputs. Recompute the economics and the verdict without re-running the experiments; the result is recorded as a derived study.</p>
                <WhatIf runId={id} value={(run.spec as any).value} onDone={(rid) => { window.location.href = `/runs/${rid}`; }} />
              </div>
            )}
          </section>
        )}

        {/* 5 DELIVERY */}
        {brief && sections["Delivery fit"] && (
          <section id="delivery">
            <SectionHead title="Delivery" hint="how it would ship" />
            <Lines lines={sections["Delivery fit"].lines} onOpen={setOpen} />
          </section>
        )}

        {/* 6 POLICY */}
        {v && (
          <section id="policy" className="space-y-5">
            <SectionHead title="Verdict" hint={`policy ${v.policy_version}; a pure function of the evidence above`} />
            <div className="grid gap-6 md:grid-cols-2">
              <div>
                <h3 className="mb-1 text-sm font-medium">Gates <span className="font-normal text-ink-500">any failure means not feasible</span></h3>
                <Ledger kind="gate" items={v.gates.map((g) => ({ ...g, ok: g.passed }))} onOpen={setOpen} />
              </div>
              <div>
                <h3 className="mb-1 text-sm font-medium">Flags <span className="font-normal text-ink-500">any trip means pilot, not build-ready</span></h3>
                <Ledger kind="flag" items={v.flags.map((f) => ({ ...f, ok: !f.tripped }))} onOpen={setOpen} />
              </div>
            </div>
            {byTool.ps && (
              <div className="panel max-w-2xl p-4">
                <h3 className="mb-1 text-sm font-medium">How close is the verdict to a different one <Cite ids={[byTool.ps.evidence_id]} onOpen={setOpen} /></h3>
                <PolicyMargins ps={byTool.ps.outputs} />
              </div>
            )}
            <p className={`text-sm ${tone.text}`}>
              {v.decision === "BUILD_READY" && "The evidence supports building. Whether to build is still a human call."}
              {v.decision === "PILOT" && "The evidence supports a pilot: every gate passes, and the flags say what a pilot must resolve."}
              {v.decision === "NOT_FEASIBLE" && "The evidence does not support building as specified. The failed gate says what would have to change."}
            </p>
          </section>
        )}

        {/* 7 REPORT */}
        {brief && (
          <section id="report" className="space-y-5">
            <SectionHead title="Brief" hint="the cited document, as exported" />
            {brief.narrative?.length > 0 && (
              <div className="max-w-[78ch]">
                <h3 className="mb-1 text-sm font-medium">Narrative <span className="font-normal text-ink-500">written by the language model from the evidence; uncited numbers are removed</span></h3>
                <p className="text-sm leading-relaxed text-ink-900">{brief.narrative.map((l, i) => <span key={i}>{l.text}<Cite ids={l.evidence_ids} onOpen={setOpen} /> </span>)}</p>
              </div>
            )}
            {(byTool.rt || byTool.comp) && (
              <div>
                <h3 className="mb-1 text-sm font-medium">Runtime <span className="font-normal text-ink-500">what the numbers depend on besides data, question and seed</span><Cite ids={[byTool.rt?.evidence_id, byTool.comp?.evidence_id].filter(Boolean) as string[]} onOpen={setOpen} /></h3>
                <RuntimeCompute rt={byTool.rt?.outputs} comp={byTool.comp?.outputs} />
              </div>
            )}
            <details className="text-xs text-ink-500">
              <summary className="cursor-pointer">Markdown source</summary>
              <pre className="mono mt-2 whitespace-pre-wrap rounded-md border border-ink-300 bg-white p-3 text-ink-700">{md}</pre>
            </details>
            {run.status === "done" && (
              <div className="max-w-2xl space-y-3">
                <h3 className="text-sm font-medium">Ask the evidence</h3>
                {answers.map((m, i) => (
                  <div key={i} className={`text-sm ${m.role === "user" ? "text-ink-500" : "text-ink-900"}`}>
                    {m.content}
                    <Cite ids={m.evidence_ids} onOpen={setOpen} />
                  </div>
                ))}
                <div className="flex gap-2">
                  <input className="input" value={q} onChange={(e) => setQ(e.target.value)} disabled={!llm} onKeyDown={(e) => { if (e.key === "Enter" && llm && !asking) ask(); }} aria-label="Question about this study" />
                  <button onClick={ask} disabled={!llm || asking} className="btn btn-primary">{asking ? "Reading evidence" : "Ask"}</button>
                </div>
                <p className="text-xs text-ink-500">
                  {llm
                    ? "The model reads only this study's stored evidence and cites what it uses. It cannot change a number."
                    : "Needs a language-model provider on the API (ANTHROPIC_API_KEY, or MDE_LLM_PROVIDER=bedrock or gemini). The brief above is complete without it."}
                </p>
              </div>
            )}
          </section>
        )}
      </div>

      <Drawer
        open={!!open && !!evidence[open]}
        onClose={() => setOpen(null)}
        title={open && evidence[open] ? `${evidence[open].tool}` : ""}
        meta={open && evidence[open] ? `${open} — stage ${evidence[open].step}, seed ${evidence[open].seed}, dataset ${evidence[open].dataset_hash}, inputs ${evidence[open].inputs_hash.slice(0, 12)}, ${evidence[open].duration_ms.toFixed(0)} ms` : undefined}
      >
        {open && evidence[open] && (
          <>
            <p className="mb-3 text-xs text-ink-700">This is the stored tool call the citation points to: its inputs are hashed, its outputs are below, verbatim.</p>
            <pre className="mono whitespace-pre-wrap rounded-md border border-ink-300 bg-ink-100 p-3 text-ink-900">{JSON.stringify(evidence[open].outputs, null, 2)}</pre>
          </>
        )}
      </Drawer>
    </div>
  );
}
