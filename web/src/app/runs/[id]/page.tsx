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

// The page reads top-down like the brief: verdict, one-paragraph summary, then each stage with a
// one-line takeaway and its chart. Every cited line and every table is there, one click down.

const STAGES = ["DEFINE", "FEASIBILITY", "EXPERIMENT", "ECONOMICS", "DELIVERY", "POLICY", "REPORT"];
const STAGE_LABEL: Record<string, [string, string]> = {
  DEFINE: ["Question", "what was asked"],
  FEASIBILITY: ["Data", "do we have it"],
  EXPERIMENT: ["Model", "can it be predicted"],
  ECONOMICS: ["Economics", "is it worth it"],
  DELIVERY: ["Delivery", "how it ships"],
  POLICY: ["Verdict", "the decision"],
  REPORT: ["Evidence", "every number's source"],
};

// plain-language meaning of each gate and flag; the raw value and threshold sit beside it
const GATE_TEXT: Record<string, string> = {
  data: "Enough of the fleet emits every needed signal",
  model: "The model beats the best single signal",
  economics: "A positive return is more likely than not",
  cost: "Run cost is under the requested ceiling",
  delivery: "A delivery pattern fits the horizon",
};
const FLAG_TEXT: Record<string, string> = {
  cross_oem_variance: "Performance differs across OEMs",
  temporal_degradation: "Performance decays over time",
  roi_spans_negative: "The return could be negative in a plausible case",
  value_unvalidated: "Value assumptions have not been validated by a pilot",
  short_history: "A needed signal has too little history",
  ablation_underpowered: "A signal removal rests on a point estimate",
  suspicious_signals: "A needed signal is unusually strong; confirm it exists before the event, not because of it",
  cost_placeholders: "Run cost uses placeholder prices",
  alert_burden: "Too many false alerts at the chosen operating point",
  data_still_improving: "More data would still raise the accuracy",
  seed_sensitive: "The result depends on how vehicles were split",
};

const pct = (x: number | null | undefined, d = 0) => (x == null ? "—" : `${(x * 100).toFixed(d)}%`);
const usd = (x: number | null | undefined) => (x == null ? "—" : `$${Math.round(x).toLocaleString()}`);
const f3 = (x: number | null | undefined) => (x == null ? "—" : Number(x).toFixed(3));

function Takeaway({ children }: { children: React.ReactNode }) {
  return <p className="max-w-[78ch] text-[15px] leading-relaxed text-ink-950">{children}</p>;
}

/** A closed disclosure: the full cited lines or a table, one click down from the takeaway. */
function More({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <details className="group mt-3">
      <summary className="cursor-pointer select-none text-xs text-accent hover:underline">{label}</summary>
      <div className="mt-3">{children}</div>
    </details>
  );
}

function Lines({ lines, onOpen }: { lines: { text: string; evidence_ids: string[] }[]; onOpen: (id: string) => void }) {
  return (
    <ul className="max-w-[78ch] space-y-2 text-sm leading-relaxed text-ink-900">
      {lines.map((l, i) => (
        <li key={i} className="border-l-2 border-ink-200 pl-3">
          {l.text}
          <Cite ids={l.evidence_ids} onOpen={onOpen} />
        </li>
      ))}
    </ul>
  );
}

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="min-w-0">
      <dt className="text-xs text-ink-500">{label}</dt>
      <dd className="num mt-0.5 text-xl font-semibold tracking-tight text-ink-950">{value}</dd>
      {sub && <dd className="mt-0.5 text-xs text-ink-500">{sub}</dd>}
    </div>
  );
}

type Check = { name: string; status: "ok" | "flag" | "info"; result: string; cite?: string; chart: React.ReactNode };

function Checklist({ checks, onOpen }: { checks: Check[]; onOpen: (id: string) => void }) {
  const [openIdx, setOpenIdx] = useState<number | null>(null);
  return (
    <ul className="divide-y divide-ink-200 border-y border-ink-200">
      {checks.map((c, i) => (
        <li key={c.name}>
          <button type="button" onClick={() => setOpenIdx(openIdx === i ? null : i)} className="grid w-full grid-cols-[10px_9rem_minmax(0,1fr)_auto] items-baseline gap-x-3 py-2.5 text-left text-sm hover:bg-white">
            <span className={`mt-[6px] inline-block h-2 w-2 rounded-full ${c.status === "ok" ? "bg-verdict-build" : c.status === "flag" ? "bg-verdict-pilot" : "bg-ink-400"}`} aria-label={c.status} />
            <span className="font-medium text-ink-950">{c.name}</span>
            <span className="min-w-0 text-ink-700">{c.result}</span>
            <span className="text-xs text-ink-400">{openIdx === i ? "hide" : "chart"}</span>
          </button>
          {openIdx === i && (
            <div className="panel mb-3 p-4">
              {c.chart}
              {c.cite && <p className="mt-2 text-xs text-ink-500">Stored computation <Cite ids={[c.cite]} onOpen={onOpen} full /></p>}
            </div>
          )}
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
        setAnswers((await api.messages(id)).filter((m) => m.role !== "system"));
      }
    };
    load().catch(console.error);
    api.health().then((h) => setLlm(h.llm_available)).catch(() => {});
    try {
      es = new EventSource(`/api/runs/${id}/events`);
      es.onmessage = (m) => {
        const d = JSON.parse(m.data);
        setRun((prev) => (prev ? { ...prev, status: d.status, steps: d.steps } : prev));
        if (d.status === "done" || d.status === "failed") { es?.close(); load().catch(console.error); }
      };
      es.onerror = () => es?.close();
    } catch { /* EventSource unavailable: finished runs still render */ }
    return () => { alive = false; es?.close(); };
  }, [id]);

  const E = useMemo(() => {
    const byName = (name: string) => Object.values(evidence).find((e) => (e as any).name === name);
    const byTool = (tool: string) => Object.values(evidence).find((e) => e.tool === tool);
    return {
      fa: byTool("feature_analysis"),
      ab: byName("ablation") ?? byTool("ablation"),
      abd: byName("ablation_daily_cadence"),
      xo: byTool("cross_oem_validation"),
      tv: byName("temporal"),
      tor: byTool("tornado"),
      roi: byTool("roi_distribution"),
      cov: byName("coverage_sufficient"),
      op: byTool("choose_operating_point"),
      cmp: byTool("model_comparison"),
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
      usable: byName("usable"),
      rate: byName("event_rate"),
      leak: byName("leakage"),
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
  const stepOf = (name: string) => run.steps.find((s) => s.name === name);
  const sections = brief ? Object.fromEntries(brief.sections.map((s) => [s.title, s])) : {};
  const spec = run.spec as any;

  // ---- the numbers the takeaways are written from
  const ab = E.ab?.outputs, cov = E.cov?.outputs, roi = E.roi?.outputs, chosen = E.op?.outputs?.chosen, xo = E.xo?.outputs, tv = E.tv?.outputs;
  const suff: string[] = ab?.sufficient_set ?? [];
  const nUsable: number | undefined = E.usable?.outputs?.n_usable;
  const dropped: Record<string, string> = E.usable?.outputs?.dropped ?? {};
  const fullAuc = ab?.full_auc?.point, suffAuc = ab?.sufficient_auc?.point;
  const baseline = E.cmp?.outputs?.sets?.sufficient?.baselines;
  const missingOems: string[] = cov?.oems_missing_any_signal ?? [];
  const lacking = xo ? Object.entries(xo.per_oem ?? {}).filter(([, o]: any) => o.diagnosis === "oem_lacks_signal").map(([k]) => k) : [];
  const gatesFailed = v ? v.gates.filter((g) => !g.passed) : [];
  const flagsTripped = v ? v.flags.filter((f) => f.tripped) : [];
  const flagsClear = v ? v.flags.filter((f) => !f.tripped) : [];
  const oem = (s: string) => s.replace("oem_", "").toUpperCase();

  const checks: Check[] = [];
  if (xo && E.xo) checks.push({
    name: "Across OEMs",
    status: v?.flags.find((f) => f.name === "cross_oem_variance")?.tripped ? "flag" : "ok",
    result: `Held-out AUC ${f3(xo.mean_auc)} on average; ${lacking.length ? `${lacking.map(oem).join(", ")} lack the signal (the model itself transfers)` : "no OEM stands out"}.`,
    cite: E.xo.evidence_id,
    chart: <><CrossOem per_oem={xo.per_oem} mean={xo.mean_auc} /><CrossOemCaption mean={xo.mean_auc} /></>,
  });
  if (tv && E.tv && tv.forward_auc) checks.push({
    name: "Over time",
    status: v?.flags.find((f) => f.name === "temporal_degradation")?.tripped ? "flag" : "ok",
    result: `Trained on the first 70% of days and tested on the rest: AUC ${f3(tv.forward_auc.point)} vs ${f3(tv.cv_auc.point)} in cross-validation; ${tv.degradation_upper > 0.03 ? "the signal decays" : "no decay"}.`,
    cite: E.tv.evidence_id,
    chart: <p className="text-sm text-ink-700">Forward windows: {(tv.forward_windows ?? []).map((w: any) => `${w.start} to ${w.end}: AUC ${w.auc}`).join("; ")}.</p>,
  });
  if (E.lc) checks.push({
    name: "More data",
    status: E.lc.outputs.still_improving ? "flag" : "ok",
    result: E.lc.outputs.still_improving ? `Accuracy still rising from half the fleet to all of it (+${f3(E.lc.outputs.auc_gain_half_to_full)}); the reported AUC is a lower bound.` : `Flat from half the fleet to all of it (${E.lc.outputs.auc_gain_half_to_full >= 0 ? "+" : ""}${f3(E.lc.outputs.auc_gain_half_to_full)}); more vehicles would not change the answer.`,
    cite: E.lc.evidence_id,
    chart: <LearningCurve lc={E.lc.outputs} />,
  });
  if (E.ss) checks.push({
    name: "Different split",
    status: E.ss.outputs.seed_sensitive ? "flag" : "ok",
    result: `Three fold assignments give AUC ${E.ss.outputs.per_seed.map((r: any) => r.auc.toFixed(3)).join(", ")}; spread ${E.ss.outputs.auc_spread.toFixed(4)}${E.ss.outputs.seed_sensitive ? ", wider than tolerance" : ", inside tolerance"}.`,
    cite: E.ss.evidence_id,
    chart: <SeedStability ss={E.ss.outputs} />,
  });
  if (E.th && E.th.outputs.best_variant) checks.push({
    name: "Model settings",
    status: E.th.outputs.loose_lower_bound ? "info" : "ok",
    result: `A fixed grid of three alternative configurations changes AUC by ${E.th.outputs.headroom >= 0 ? "+" : ""}${Number(E.th.outputs.headroom).toFixed(4)} at best; ${E.th.outputs.loose_lower_bound ? "the reported AUC is a loose lower bound" : "the default is not leaving signal on the table"}.`,
    cite: E.th.evidence_id,
    chart: <TuningHeadroom th={E.th.outputs} />,
  });
  if (E.red) checks.push({
    name: "Redundant signals",
    status: "info",
    result: `${E.red.outputs.n_signals} screened candidates carry ${E.red.outputs.n_independent_groups} independent pieces of information; ${E.red.outputs.n_pairs_redundant} pairs say the same thing.`,
    cite: E.red.evidence_id,
    chart: <RedundancyGroups red={E.red.outputs} sufficient={suff} />,
  });

  return (
    <div className="grid gap-8 lg:grid-cols-[168px_minmax(0,1fr)]">
      {/* stage rail */}
      <aside className="hidden lg:block">
        <ol className="sticky top-20 space-y-1 text-[13px]">
          {STAGES.map((name, i) => {
            const status = stepOf(name)?.status ?? "pending";
            const [label, hint] = STAGE_LABEL[name];
            return (
              <li key={name}>
                <a href={`#${name.toLowerCase()}`} className="flex items-start gap-2 rounded-md px-2 py-1 hover:bg-white">
                  <span className={`num mt-[2px] w-4 text-right text-[11px] ${status === "done" ? "text-ink-500" : status === "running" ? "text-verdict-pilot" : "text-ink-300"}`}>{i + 1}</span>
                  <span className="min-w-0">
                    <span className={`block ${status === "done" ? "text-ink-950" : status === "running" ? "text-verdict-pilot" : "text-ink-400"}`}>
                      {label}
                      {status === "running" && <span className="live ml-1.5 inline-block h-1.5 w-1.5 rounded-full bg-verdict-pilot align-middle" />}
                    </span>
                    <span className="block text-[11px] leading-tight text-ink-400">{hint}</span>
                  </span>
                </a>
              </li>
            );
          })}
        </ol>
      </aside>

      <div className="min-w-0 space-y-12">
        {/* verdict strip */}
        <header className="sticky top-12 z-20 -mx-6 border-b border-ink-300 bg-ink-100/95 px-6 py-3 backdrop-blur lg:mx-0 lg:rounded-md lg:border lg:px-4">
          <div className="flex flex-wrap items-center justify-between gap-x-6 gap-y-2">
            <div className="flex items-center gap-4">
              <DecisionBadge decision={v?.decision ?? null} size="lg" />
              <div className="min-w-0">
                <h1 className="truncate text-lg font-semibold tracking-tight text-ink-950">{String(spec.capability_name)}</h1>
                <p className="text-sm text-ink-500">
                  Predict <span className="mono">{String(spec.target_event)}</span> within {String(spec.horizon_days)} days
                  {run.kind !== "study" && <> — {run.kind} of <Link className="underline" href={`/runs/${run.derived_from}`}>{run.derived_from}</Link></>}
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              {v && (
                <p className="mr-2 hidden text-sm text-ink-700 md:block">
                  <span className={`num font-semibold ${gatesFailed.length ? "text-verdict-no" : "text-verdict-build"}`}>{v.gates.length - gatesFailed.length}/{v.gates.length}</span> gates
                  <span className="mx-2 text-ink-300">|</span>
                  <span className={`num font-semibold ${flagsTripped.length ? "text-verdict-pilot" : "text-ink-700"}`}>{flagsTripped.length}</span> {flagsTripped.length === 1 ? "flag" : "flags"}
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
          {replayed && <p className="mt-2 text-xs text-ink-700">Replay queued. It recomputes every stage from scratch and appears in the study list with a record-by-record comparison.</p>}
          {run.error && <p className="mt-2 text-sm text-verdict-no">{run.error}</p>}
        </header>

        {run.status !== "done" && !run.error && (
          <section>
            <p className="text-sm text-ink-700">Running. Stages light up on the left as they finish; the page fills in when the verdict is ready.</p>
            <Skeleton lines={4} className="mt-4 max-w-2xl" />
          </section>
        )}

        {/* summary */}
        {v && ab && (
          <section className="space-y-5">
            <Takeaway>
              {v.decision === "PILOT" && <><span className={`font-semibold ${tone.text}`}>Pilot.</span> Every gate passes; {flagsTripped.length} {flagsTripped.length === 1 ? "flag says" : "flags say"} what a pilot has to resolve. </>}
              {v.decision === "BUILD_READY" && <><span className={`font-semibold ${tone.text}`}>Build-ready.</span> Every gate passes and no flag is tripped. </>}
              {v.decision === "NOT_FEASIBLE" && <><span className={`font-semibold ${tone.text}`}>Not feasible as specified.</span> {gatesFailed.map((g) => `The ${g.name} gate fails (${f3(g.value)} against ${f3(g.threshold)})`).join("; ")}. </>}
              {suff.length > 0 && nUsable != null && <>{suff.length} of {nUsable} usable signals predict the event as well as all of them (AUC {f3(suffAuc)} vs {f3(fullAuc)}). </>}
              {cov && <>{pct(cov.fleet_share_full_set, 0)} of the fleet emits all {suff.length}{missingOems.length ? `; ${missingOems.map(oem).join(", ")} do not` : ""}. </>}
              {roi && <>Under the stated value assumptions the probability of a positive return is {Number(roi.p_roi_positive).toFixed(2)}{E.tor && <>, and the result hinges on <span className="mono">{E.tor.outputs.dominant_input}</span></>}.</>}
            </Takeaway>
            <dl className="grid grid-cols-2 gap-x-6 gap-y-4 border-y border-ink-300 py-4 sm:grid-cols-3 lg:grid-cols-6">
              <Stat label="Signals needed" value={`${suff.length}`} sub={nUsable != null ? `of ${nUsable} usable` : undefined} />
              <Stat label="Accuracy (AUC)" value={f3(suffAuc)} sub={baseline?.best_single_signal_auc != null ? `one signal alone ${f3(baseline.best_single_signal_auc)}` : undefined} />
              <Stat label="Fleet coverage" value={pct(cov?.fleet_share_full_set, 0)} sub={`gate ${pct(spec.constraints?.min_oem_coverage ?? 0.6, 0)}`} />
              <Stat label="Events caught" value={pct(chosen?.recall, 0)} sub={chosen ? `at ${pct(chosen.alert_rate, 1)} alert rate` : undefined} />
              <Stat label="P(return > 0)" value={roi ? Number(roi.p_roi_positive).toFixed(2) : "—"} sub="gate 0.50" />
              <Stat label="Net value / yr" value={usd(roi?.net_value_year?.p50)} sub={roi ? `5th pct ${usd(roi.net_value_year.p5)}` : undefined} />
            </dl>
          </section>
        )}

        {/* 1 QUESTION */}
        {brief && (
          <section id="define">
            <SectionHead title="Question" hint="what was asked, and the human inputs" />
            <Takeaway>
              Can <span className="mono">{String(spec.target_event)}</span> be predicted {spec.horizon_days} days ahead for every vehicle, delivered {String(spec.delivery_mode).replace("_", " ")} to <span className="mono">{String(spec.consumer)}</span>, and is it worth building?
            </Takeaway>
            {spec.value && (
              <table className="mt-3 text-[12px]">
                <thead className="text-left text-ink-500"><tr><th className="pr-4 font-medium">value assumption (human input)</th><th className="pr-3 text-right font-medium">low</th><th className="pr-3 text-right font-medium">base</th><th className="text-right font-medium">high</th></tr></thead>
                <tbody className="num">
                  {([["value-bearing share of events", "value_bearing_fraction"], ["preventable share", "preventable_fraction"], ["$ per avoided event", "usd_per_avoided_event"], ["fleet size", "fleet_size"]] as const).map(([label, k]) => (
                    <tr key={k} className="border-t border-ink-300/50"><td className="py-1 pr-4 font-sans text-ink-700">{label}</td><td className="pr-3 text-right">{spec.value[k].low.toLocaleString()}</td><td className="pr-3 text-right">{spec.value[k].base.toLocaleString()}</td><td className="text-right">{spec.value[k].high.toLocaleString()}</td></tr>
                  ))}
                </tbody>
              </table>
            )}
            {spec.value?.source_note && <p className="mt-2 max-w-[78ch] text-xs text-ink-500">Source of the ranges: {spec.value.source_note}. They are placeholders until a pilot validates them.</p>}
            <More label="Specification lines and runtime">
              <Lines lines={sections["Specification"]?.lines ?? []} onOpen={setOpen} />
              {(E.rt || E.comp) && <div className="mt-4"><RuntimeCompute rt={E.rt?.outputs} comp={E.comp?.outputs} /></div>}
            </More>
          </section>
        )}

        {/* 2 DATA */}
        {brief && (
          <section id="feasibility">
            <SectionHead title="Data" hint="do we have it, and which OEMs emit it" />
            <Takeaway>
              {E.contract && <>{E.contract.outputs.stats.n_vehicles.toLocaleString()} vehicles across {E.contract.outputs.stats.n_oems} OEMs over {E.contract.outputs.stats.n_days} days passed the data contract. </>}
              {E.rate && <>The event happens {Number(E.rate.outputs.rate_per_vehicle_year).toFixed(2)} times per vehicle-year (measured, not assumed). </>}
              {nUsable != null && <>{nUsable} signals are usable{Object.keys(dropped).length ? <>; {Object.keys(dropped).length} were dropped as leakage: <span className="mono">{Object.keys(dropped).join(", ")}</span></> : ""}. </>}
              {cov && <>The needed signals are emitted by {pct(cov.fleet_share_full_set, 1)} of the fleet{missingOems.length ? `; OEMs ${missingOems.map(oem).join(", ")} lack at least one` : ""}.</>}
            </Takeaway>
            {cov && (
              <div className="mt-4">
                <h3 className="mb-2 text-sm font-medium text-ink-950">Which OEMs emit the needed signals <Cite ids={[E.cov!.evidence_id]} onOpen={setOpen} /></h3>
                <CoverageHeatmap cov={cov} />
                <p className="mt-1 text-xs text-ink-500">Green: the OEM's vehicles emit the signal. The bottom-right cell is the coverage gate.</p>
              </div>
            )}
            <More label={`Quality per signal and ${(sections["Data feasibility"]?.lines ?? []).length} cited lines`}>
              {(E.contract || E.quality) && <DataQuality contract={E.contract?.outputs} quality={E.quality?.outputs} sufficient={suff} />}
              <div className="mt-4"><Lines lines={sections["Data feasibility"]?.lines ?? []} onOpen={setOpen} /></div>
            </More>
          </section>
        )}

        {/* 3 MODEL */}
        {brief && ab && (
          <section id="experiment">
            <SectionHead title="Model" hint="can it be predicted, and with how few signals" />
            <Takeaway>
              A gradient-boosted model on all {nUsable} signals reaches AUC {f3(fullAuc)}. Removing signals one at a time, and keeping a removal only when a paired bootstrap says accuracy did not drop by more than 0.005, leaves <span className="font-semibold">{suff.length}</span>: <span className="mono">{suff.join(", ")}</span> at AUC {f3(suffAuc)}.
              {baseline?.best_single_signal && <> The best single signal alone (<span className="mono">{baseline.best_single_signal}</span>) reaches {f3(baseline.best_single_signal_auc)}, so the model adds {f3((suffAuc ?? 0) - baseline.best_single_signal_auc)}.</>}
            </Takeaway>
            <div className="mt-4 grid gap-6 md:grid-cols-2">
              {E.fa && <div className="panel p-4"><h3 className="mb-1 text-sm font-medium">Which signals matter <span className="font-normal text-ink-500">accuracy lost when one is shuffled</span><Cite ids={[E.fa.evidence_id]} onOpen={setOpen} /></h3><Importance ranking={E.fa.outputs.ranking} /></div>}
              <div className="panel p-4"><h3 className="mb-1 text-sm font-medium">Removing signals one by one <span className="font-normal text-ink-500">accuracy after each removal</span><Cite ids={[E.ab!.evidence_id]} onOpen={setOpen} /></h3><AblationTrace trace={ab.trace} /><p className="mt-1 text-xs text-ink-500">The line stays flat until a needed signal is removed; those are kept. {ab.underpowered ? "Some removals rest on point estimates (underpowered)." : ""}</p></div>
            </div>
            {E.abd && E.cd && E.cs && (
              <p className="mt-3 max-w-[78ch] text-sm text-ink-700">
                With daily-cadence signals only: {E.abd.outputs.sufficient_set.length} signals at AUC <span className="num">{f3(E.abd.outputs.sufficient_auc.point)}</span> for <span className="num">{usd(E.cd.outputs.monthly.marginal_total)}</span>/mo, against <span className="num">{f3(suffAuc)}</span> for <span className="num">{usd(E.cs.outputs.monthly.marginal_total)}</span>/mo with realtime polling.
                <Cite ids={[E.abd.evidence_id, E.cd.evidence_id]} onOpen={setOpen} />
              </p>
            )}
            <More label={`Every removal step and ${(sections["Model"]?.lines ?? []).length} cited lines`}>
              <AblationTable ab={ab} />
              <div className="mt-4"><Lines lines={sections["Model"]?.lines ?? []} onOpen={setOpen} /></div>
            </More>
          </section>
        )}

        {/* 3b ROBUSTNESS */}
        {brief && checks.length > 0 && (
          <section id="robustness">
            <SectionHead title="Does it hold up" hint="six checks; open one for its chart" />
            <Checklist checks={checks} onOpen={setOpen} />
            <More label={`${(sections["Robustness"]?.lines ?? []).length} cited lines`}>
              <Lines lines={sections["Robustness"]?.lines ?? []} onOpen={setOpen} />
            </More>
          </section>
        )}

        {/* 4 ECONOMICS */}
        {brief && (
          <section id="economics">
            <SectionHead title="Economics" hint="what it costs to run and what it is worth" />
            <Takeaway>
              {chosen && <>Alerting on the top {pct(chosen.alert_rate, 1)} of vehicle-days catches {pct(chosen.recall, 0)} of events{chosen.median_lead_days != null && <> with a median lead of {chosen.median_lead_days.toFixed(0)} days</>}{chosen.false_alerts_per_100_vehicle_months != null && <> and {chosen.false_alerts_per_100_vehicle_months.toFixed(1)} false alerts per 100 vehicle-months</>}; this point maximises median net value. </>}
              {E.cs && <>Running the {suff.length} signals costs {usd(E.cs.outputs.monthly.marginal_total)}/mo, most of it {E.cs.outputs.monthly.build_amortized > E.cs.outputs.monthly.oem_api_calls ? "amortised build" : "OEM API calls"}; infrastructure is {usd(E.cs.outputs.monthly.infra_total)}. </>}
              {roi && <>Over 10,000 draws of the value ranges, the return is positive in {pct(roi.p_roi_positive, 0)} of cases; median net value {usd(roi.net_value_year.p50)} a year, 5th percentile {usd(roi.net_value_year.p5)}.</>}
            </Takeaway>
            <div className="mt-4 grid gap-6 md:grid-cols-2">
              {E.op && E.cmp && chosen && (
                <div className="panel p-4">
                  <h3 className="mb-1 text-sm font-medium">Events caught vs alerts raised <span className="font-normal text-ink-500">the chosen point in orange</span><Cite ids={[E.op.evidence_id]} onOpen={setOpen} /></h3>
                  <OperatingCurve points={E.cmp.outputs.sets.sufficient.models.lightgbm.operating_points} chosen={chosen} />
                  {chosen.recall_lo != null && <p className="mt-1 text-xs text-ink-500">Recall interval {pct(chosen.recall_lo, 0)} to {pct(chosen.recall_hi, 0)} ({chosen.recall_ci_basis === "bootstrap" ? "bootstrap over vehicles" : "proxy from the AUC interval"}); this interval feeds the return calculation.</p>}
                </div>
              )}
              {E.tor && (
                <div className="panel p-4">
                  <h3 className="mb-1 text-sm font-medium">What the return hinges on <span className="font-normal text-ink-500">median ROI as each input goes low to high</span><Cite ids={[E.tor.evidence_id]} onOpen={setOpen} /></h3>
                  <Tornado ranked={E.tor.outputs.ranked} base={E.tor.outputs.base_median_roi} />
                  <p className="mt-1 text-xs text-ink-500">The longest bar is the input a pilot most needs to pin down.</p>
                </div>
              )}
            </div>
            <More label={`Cost breakdown and ${(sections["Economics"]?.lines ?? []).length} cited lines`}>
              <Lines lines={sections["Economics"]?.lines ?? []} onOpen={setOpen} />
            </More>
            {run.status === "done" && spec.value && (
              <More label="Change the value assumptions and recompute">
                <div className="panel max-w-2xl p-4">
                  <p className="mb-3 text-xs text-ink-700">These are the human inputs. The economics and the verdict are recomputed without re-running the experiments; the result is a derived study.</p>
                  <WhatIf runId={id} value={spec.value} onDone={(rid) => { window.location.href = `/runs/${rid}`; }} />
                </div>
              </More>
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

        {/* 6 VERDICT */}
        {v && (
          <section id="policy" className="space-y-6">
            <SectionHead title="Verdict" hint={`a fixed rule (policy ${v.policy_version}) applied to the evidence above`} />
            <div className="grid gap-8 md:grid-cols-2">
              <div>
                <h3 className="mb-2 text-sm font-medium text-ink-950">Gates <span className="font-normal text-ink-500">all must pass</span></h3>
                <ul className="divide-y divide-ink-200 border-y border-ink-200">
                  {v.gates.map((g) => (
                    <li key={g.name} className="grid grid-cols-[10px_minmax(0,1fr)_auto] items-baseline gap-x-3 py-2 text-sm">
                      <span className={`mt-[6px] inline-block h-2 w-2 rounded-full ${g.passed ? "bg-verdict-build" : "bg-verdict-no"}`} />
                      <span className="text-ink-950">{GATE_TEXT[g.name] ?? g.name}<Cite ids={g.evidence_ids} onOpen={setOpen} /></span>
                      <span className="num text-right text-ink-700">{g.value != null ? f3(g.value) : g.passed ? "yes" : "no"}{g.threshold != null && <span className="text-ink-400"> / {f3(g.threshold)}</span>}</span>
                    </li>
                  ))}
                </ul>
              </div>
              <div>
                <h3 className="mb-2 text-sm font-medium text-ink-950">{flagsTripped.length ? "What a pilot must resolve" : "Flags"} <span className="font-normal text-ink-500">{flagsTripped.length} of {v.flags.length} tripped</span></h3>
                <ul className="divide-y divide-ink-200 border-y border-ink-200">
                  {flagsTripped.map((f) => (
                    <li key={f.name} className="grid grid-cols-[10px_minmax(0,1fr)_auto] items-baseline gap-x-3 py-2 text-sm">
                      <span className="mt-[6px] inline-block h-2 w-2 rounded-full bg-verdict-pilot" />
                      <span className="min-w-0 text-ink-950">{FLAG_TEXT[f.name] ?? f.name}{f.note && f.name !== "value_unvalidated" && <span className="block text-xs text-ink-500">{f.note}</span>}<Cite ids={f.evidence_ids} onOpen={setOpen} /></span>
                      <span className="num text-right text-ink-700">{f.value != null && f.threshold != null && f.threshold !== 0 ? <>{f3(f.value)}<span className="text-ink-400"> / {f3(f.threshold)}</span></> : ""}</span>
                    </li>
                  ))}
                </ul>
                {flagsClear.length > 0 && <p className="mt-2 text-xs text-ink-500">Clear: {flagsClear.map((f) => (FLAG_TEXT[f.name] ?? f.name).toLowerCase()).join("; ")}.</p>}
              </div>
            </div>
            {E.ps && (
              <div className="panel max-w-2xl p-4">
                <h3 className="mb-1 text-sm font-medium">Would the verdict change if the thresholds moved? <Cite ids={[E.ps.evidence_id]} onOpen={setOpen} /></h3>
                <PolicyMargins ps={E.ps.outputs} />
              </div>
            )}
            <p className={`text-sm ${tone.text}`}>
              {v.decision === "BUILD_READY" && "The evidence supports building. Whether to build is still a human call."}
              {v.decision === "PILOT" && "The evidence supports a pilot: every gate passes, and the flags above are what the pilot has to answer."}
              {v.decision === "NOT_FEASIBLE" && "The evidence does not support building as specified. The failed gate says what would have to change."}
            </p>
          </section>
        )}

        {/* 7 EVIDENCE */}
        {brief && (
          <section id="report" className="space-y-5">
            <SectionHead title="Evidence" hint="every number above links to the computation that produced it" />
            {brief.narrative?.length > 0 && (
              <div className="max-w-[78ch]">
                <h3 className="mb-1 text-sm font-medium">Narrative <span className="font-normal text-ink-500">written by the language model from the evidence; sentences with uncited numbers were removed</span></h3>
                <p className="text-sm leading-relaxed text-ink-900">{brief.narrative.map((l, i) => <span key={i}>{l.text.replace(/\s*\[ev_[0-9a-f]{12}\]/g, "")}<Cite ids={l.evidence_ids} onOpen={setOpen} /> </span>)}</p>
              </div>
            )}
            <p className="max-w-[78ch] text-sm text-ink-700">
              Click any <span className="cite">source</span> chip to open the stored tool call: its inputs are hashed, its outputs are shown verbatim. Replay recomputes every record from scratch and compares them. The brief exports as Markdown with every citation.
            </p>
            <More label="Markdown source of the brief">
              <pre className="mono whitespace-pre-wrap rounded-md border border-ink-300 bg-white p-3 text-ink-700">{md}</pre>
            </More>
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
                  {llm ? "The model reads only this study's stored evidence and cites what it uses. It cannot change a number." : "Needs a language-model provider on the API. The study is complete without it."}
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
