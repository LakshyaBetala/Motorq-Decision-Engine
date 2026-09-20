"use client";
import { Area, CartesianGrid, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

// ---------------------------------------------------------------- learning curve
export function LearningCurve({ lc }: { lc: any }) {
  const data = lc.points
    .filter((p: any) => p.auc)
    .map((p: any) => ({ pct: Math.round(p.fraction * 100), vehicles: p.n_vehicles, pos: p.n_pos, auc: p.auc.point, lo: p.auc.lo, hi: p.auc.hi }));
  if (!data.length) return <p className="text-xs text-ink-500">Not resolvable on this fleet (too few positives in the subsets).</p>;
  const ys = data.flatMap((d: any) => [d.lo, d.hi]);
  const dom: [number, number] = [Math.max(0.5, Math.floor(Math.min(...ys) * 20) / 20), Math.min(1, Math.ceil(Math.max(...ys) * 20) / 20)];
  const gain = lc.auc_gain_half_to_full;
  const note =
    gain == null
      ? "Gain from half to all vehicles not resolvable."
      : lc.still_improving
        ? `Still rising: +${gain.toFixed(3)} AUC from half to all vehicles. The reported AUC is a lower bound of what more data would give.`
        : `Flat (${gain >= 0 ? "+" : ""}${gain.toFixed(3)} from half to all vehicles). Data-saturated on this fleet; the remaining uncertainty is about value, not signal.`;
  return (
    <div>
      <ResponsiveContainer width="100%" height={200}>
        <ComposedChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
          <CartesianGrid stroke="#e8ebef" vertical={false} />
          <XAxis dataKey="pct" tick={{ fontSize: 11 }} tickFormatter={(v) => `${v}%`} label={{ value: "share of vehicles used", position: "insideBottom", offset: -4, fontSize: 11 }} />
          <YAxis domain={dom} tick={{ fontSize: 11 }} width={40} tickFormatter={(v) => Number(v).toFixed(2)} />
          <Tooltip
            contentStyle={{ fontSize: 12 }}
            formatter={(v: number, n) => [typeof v === "number" ? v.toFixed(3) : v, n]}
            labelFormatter={(l, p) => `${l}% of vehicles: ${p?.[0]?.payload?.vehicles} vehicles, ${p?.[0]?.payload?.pos} positives`}
          />
          <Area type="monotone" dataKey="hi" stroke="none" fill="#cbd5e1" fillOpacity={0.5} isAnimationActive={false} name="95% hi" />
          <Area type="monotone" dataKey="lo" stroke="none" fill="#ffffff" fillOpacity={1} isAnimationActive={false} name="95% lo" />
          <Line type="monotone" dataKey="auc" name="AUC" stroke="#14161a" dot={{ r: 3 }} isAnimationActive={false} />
        </ComposedChart>
      </ResponsiveContainer>
      <p className="mt-1 text-xs text-ink-500">{note}</p>
    </div>
  );
}

// ---------------------------------------------------------------- redundancy groups
export function RedundancyGroups({ red, sufficient }: { red: any; sufficient: string[] }) {
  const suff = new Set(sufficient);
  const clustered = new Set<string>(red.clusters.flat());
  const independent: string[] = (red.signals ?? []).filter((s: string) => !clustered.has(s));
  const chip = (s: string) => (
    <span key={s} className={`mono mb-1 mr-1 inline-block rounded border px-1.5 py-[1px] text-[11px] ${suff.has(s) ? "border-ink-900 bg-ink-900 text-white" : "border-ink-300 text-ink-700"}`}>
      {s}
    </span>
  );
  return (
    <div className="text-sm">
      <p className="mb-2 text-xs text-ink-500">
        {red.n_signals} screened candidates carry <b>{red.n_independent_groups}</b> independent pieces of information (Spearman |ρ| ≥ {red.threshold}, {red.n_pairs_redundant} redundant pairs). Dark chips are in the sufficient set; at most one per group is needed.
      </p>
      {red.clusters.map((g: string[], i: number) => (
        <div key={i} className="mb-1 flex flex-wrap items-center">
          <span className="mr-2 text-[11px] text-ink-500">group {i + 1}</span>
          {g.map(chip)}
        </div>
      ))}
      {independent.length > 0 && (
        <div className="mt-2 flex flex-wrap items-center">
          <span className="mr-2 text-[11px] text-ink-500">independent</span>
          {independent.map(chip)}
        </div>
      )}
      {red.pairs?.length > 0 && (
        <details className="mt-2 text-xs text-ink-500">
          <summary className="cursor-pointer">strongest pairs</summary>
          <ul className="mt-1">
            {red.pairs.slice(0, 10).map((p: any) => (
              <li key={`${p.a}${p.b}`} className="mono">
                {p.a} ~ {p.b}: {p.spearman >= 0 ? "+" : ""}
                {p.spearman.toFixed(3)}
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- data contract & quality
export function DataQuality({ contract, quality, sufficient }: { contract: any; quality: any; sufficient: string[] }) {
  const st = contract?.stats ?? {};
  const flagsOf = (s: string): string[] => quality?.flags?.[s] ?? [];
  const rows = (sufficient ?? []).filter((s) => quality?.per_signal?.[s]);
  const badge = (f: string) => (
    <span key={f} className={`mr-1 rounded px-1 text-[10px] ${f === "frozen" || f === "implausible" ? "bg-red-50 text-verdict-no" : "bg-amber-50 text-verdict-pilot"}`}>
      {f}
    </span>
  );
  return (
    <div className="text-sm">
      {contract && (
        <p className="mb-2 text-xs text-ink-700">
          <span className={`mr-2 rounded px-1.5 py-[1px] text-[11px] font-medium ${contract.ok ? "bg-green-50 text-verdict-build" : "bg-red-50 text-verdict-no"}`}>contract {contract.ok ? "OK" : "FAILED"}</span>
          <span className="num">{st.n_vehicles?.toLocaleString()}</span> vehicles across <span className="num">{st.n_oems}</span> OEMs, <span className="num">{st.n_days}</span> days ({st.date_range?.[0]} to {st.date_range?.[1]}), vehicle-day grid <span className="num">{st.grid_completeness != null ? `${(st.grid_completeness * 100).toFixed(1)}%` : "—"}</span> complete, <span className="num">{st.n_signals_catalogued}</span> catalogued signals
          {contract.warnings?.length > 0 && <span className="block text-verdict-pilot">{contract.warnings.join("; ")}</span>}
          {contract.errors?.length > 0 && <span className="block text-verdict-no">{contract.errors.join("; ")}</span>}
        </p>
      )}
      {rows.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-[12px]">
            <thead className="text-left text-ink-500">
              <tr><th>sufficient-set signal</th><th>non-null</th><th>gap d</th><th>drift PSI</th><th>frozen</th><th>implausible</th><th>history</th><th>flags</th></tr>
            </thead>
            <tbody>
              {rows.map((s) => {
                const q = quality.per_signal[s];
                return (
                  <tr key={s} className="border-t border-ink-300/50">
                    <td className="mono">{s}</td>
                    <td>{(q.nonnull_rate * 100).toFixed(0)}%</td>
                    <td>{q.median_gap_days ?? "—"}</td>
                    <td>{q.psi_first_last_quarter?.toFixed(2)}</td>
                    <td>{q.frozen_share != null ? `${(q.frozen_share * 100).toFixed(1)}%` : "—"}</td>
                    <td>{q.implausible_share != null ? `${(q.implausible_share * 100).toFixed(2)}%` : "—"}</td>
                    <td>{q.history_months?.toFixed(1)} mo</td>
                    <td>{flagsOf(s).map(badge)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p className="mt-1 text-xs text-ink-500">frozen = share of vehicles with an identical value on ≥14 consecutive active days (plateaus at a physical bound excluded); implausible = readings outside the unit&apos;s physical range. Reported, not dropped; they trip the suspicious_signals flag.</p>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- runtime & compute
export function RuntimeCompute({ rt, comp }: { rt: any; comp: any }) {
  const libs = rt?.libraries ?? {};
  const total = comp ? comp.cache_hits + comp.cache_misses : 0;
  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
      {rt && (
        <>
          <dt className="text-ink-500">python</dt><dd className="mono">{rt.python} on {rt.platform}</dd>
          <dt className="text-ink-500">lightgbm</dt><dd className="mono">{libs.lightgbm}, deterministic={String(rt.lgbm_deterministic)}, {rt.lgbm_histogram} histograms, parameter hash {rt.lgbm_params_hash}</dd>
          <dt className="text-ink-500">numpy / sklearn</dt><dd className="mono">{libs.numpy} / {libs["scikit-learn"]}</dd>
          <dt className="text-ink-500">code</dt><dd className="mono">{rt.code_version}</dd>
          <dt className="text-ink-500">parallelism</dt><dd className="mono">{rt.fold_workers} fold workers × {rt.lgbm_threads} threads (does not change results)</dd>
        </>
      )}
      {comp && (
        <>
          <dt className="text-ink-500">fit cache</dt><dd className="mono">{comp.cache_hits} of {total} out-of-fold fit sets served from cache{comp.cache_enabled ? "" : " (cache bypassed: replay)"}</dd>
        </>
      )}
    </dl>
  );
}

// ---------------------------------------------------------------- tuning headroom
export function TuningHeadroom({ th }: { th: any }) {
  const rows = Object.entries(th.variants ?? {}) as [string, any][];
  if (!rows.length) return <p className="text-xs text-ink-500">No grid was run.</p>;
  const fmt = (x: number) => `${x >= 0 ? "+" : ""}${x.toFixed(3)}`;
  return (
    <div className="text-sm">
      <table className="w-full text-[12px]">
        <thead className="text-left text-ink-500">
          <tr><th className="font-medium">configuration</th><th className="text-right font-medium">AUC</th><th className="text-right font-medium">vs default</th><th className="text-right font-medium">95% CI</th></tr>
        </thead>
        <tbody>
          <tr className="border-t border-ink-300/50">
            <td className="mono">default <span className="text-ink-500">(used for the verdict)</span></td>
            <td className="num text-right">{th.default_auc.point.toFixed(3)}</td>
            <td className="num text-right text-ink-400">—</td>
            <td className="num text-right text-ink-400">—</td>
          </tr>
          {rows.map(([name, r]) => (
            <tr key={name} className={`border-t border-ink-300/50 ${name === th.best_variant ? "text-ink-950" : "text-ink-700"}`}>
              <td className="mono">{name}{name === th.best_variant && <span className="ml-1 text-ink-500">best</span>}</td>
              <td className="num text-right">{r.auc.point.toFixed(3)}</td>
              <td className="num text-right">{fmt(r.delta_vs_default.point)}</td>
              <td className="num text-right text-ink-500">{fmt(r.delta_vs_default.lo)} to {fmt(r.delta_vs_default.hi)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-2 text-xs text-ink-500">
        {th.loose_lower_bound
          ? `A fixed grid moves the AUC by ${fmt(th.headroom)} with the whole interval above ${th.headroom_threshold}: the reported AUC is a loose lower bound.`
          : `Nothing in the grid beats the default beyond noise (best ${fmt(th.headroom)}, interval reaching ${fmt(th.headroom_lo)}): the fixed configuration is not leaving signal on the table.`}{" "}
        Best-of-grid is picked on the evaluation folds, so it is an optimistic bound. The verdict always uses the default so studies stay comparable.
      </p>
    </div>
  );
}

// ---------------------------------------------------------------- seed stability
export function SeedStability({ ss }: { ss: any }) {
  const rows: { seed: number; auc: number }[] = ss.per_seed ?? [];
  if (!rows.length) return null;
  const aucs = rows.map((r) => r.auc);
  const lo = Math.min(...aucs), hi = Math.max(...aucs);
  const pad = Math.max(ss.spread_threshold, (hi - lo) * 0.5, 0.002);
  const a0 = lo - pad, a1 = hi + pad;
  const x = (v: number) => ((v - a0) / (a1 - a0)) * 100;
  return (
    <div className="text-sm">
      <div className="relative mt-3 h-8">
        <div className="absolute left-0 right-0 top-1/2 h-px bg-ink-300" />
        {/* tolerance band around the mean */}
        <div className="absolute top-[9px] h-[14px] rounded-sm bg-ink-200/70" style={{ left: `${x(ss.auc_mean - ss.spread_threshold / 2)}%`, width: `${x(ss.auc_mean + ss.spread_threshold / 2) - x(ss.auc_mean - ss.spread_threshold / 2)}%` }} title={`tolerance ${ss.spread_threshold}`} />
        {rows.map((r) => (
          <span key={r.seed} className={`absolute top-1/2 h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white ${ss.seed_sensitive ? "bg-verdict-pilot" : "bg-ink-900"}`} style={{ left: `${x(r.auc)}%` }} title={`seed ${r.seed}: ${r.auc.toFixed(4)}`} />
        ))}
      </div>
      <div className="num flex justify-between text-[11px] text-ink-500"><span>{a0.toFixed(3)}</span><span>{a1.toFixed(3)}</span></div>
      <p className="mt-2 text-xs text-ink-500">
        Sufficient-set AUC under {rows.length} fold assignments: {aucs.map((a) => a.toFixed(4)).join(", ")}; spread <span className="num">{ss.auc_spread.toFixed(4)}</span> against a tolerance of {ss.spread_threshold} (the shaded band).{" "}
        {ss.seed_sensitive
          ? "Wider than the ablation tolerance: read the sufficient set as one of several equivalent choices."
          : "The answer does not depend on how the vehicles were split."}
      </p>
    </div>
  );
}

// ---------------------------------------------------------------- policy sensitivity
export function PolicyMargins({ ps }: { ps: any }) {
  type Row = { name: string; ok: boolean; margin_steps: number | null; margin: number | null; kind: "gate" | "flag" };
  const rows: Row[] = [
    ...(ps.gates ?? []).map((g: any) => ({ name: g.name, ok: g.passed, margin_steps: g.margin_steps, margin: g.margin, kind: "gate" as const })),
    ...(ps.flags ?? []).map((f: any) => ({ name: f.name, ok: !f.tripped, margin_steps: f.margin_steps, margin: f.margin, kind: "flag" as const })),
  ].filter((r) => r.margin_steps != null);
  const cap = 6; // bars saturate at six steps; the number still says the exact figure
  const w = (m: number) => `${(Math.min(Math.abs(m), cap) / cap) * 50}%`;
  const against = ps.one_step_against, favour = ps.one_step_in_favour;
  return (
    <div className="text-sm">
      <ul className="space-y-1.5">
        {rows.map((r) => {
          const m = r.margin_steps as number;
          const color = m >= 0 ? (r.kind === "gate" ? "bg-verdict-build" : "bg-ink-400") : r.kind === "gate" ? "bg-verdict-no" : "bg-verdict-pilot";
          return (
            <li key={`${r.kind}-${r.name}`} className="grid grid-cols-[minmax(0,10rem)_1fr_3.5rem] items-center gap-x-3 text-[12px]">
              <span className="truncate text-ink-900">{r.name.replaceAll("_", " ")}<span className="ml-1 text-ink-400">{r.kind}</span></span>
              <span className="relative block h-3">
                <span className="absolute inset-y-0 left-1/2 w-px bg-ink-900" />
                <span className={`absolute inset-y-[3px] rounded-sm ${color}`} style={m >= 0 ? { left: "50%", width: w(m) } : { right: "50%", width: w(m) }} />
              </span>
              <span className={`num text-right ${m >= 0 ? "text-ink-700" : r.kind === "gate" ? "text-verdict-no" : "text-verdict-pilot"}`}>{m >= 0 ? "+" : ""}{m.toFixed(1)}</span>
            </li>
          );
        })}
      </ul>
      <p className="mt-2 text-xs text-ink-500">
        Distance from each threshold in policy steps (one step is the change to a threshold a reviewer would consider; the sizes are in policy.yaml). Left of the line is the failing or tripped side.
        {ps.binding_gate && <> The binding gate is <span className="mono">{ps.binding_gate}</span>.</>}
      </p>
      <p className="mt-1 text-xs text-ink-700">
        Every threshold one step against the capability: <span className="font-medium">{against.decision.replaceAll("_", " ")}</span>{against.changed.length > 0 && <span className="text-ink-500"> ({against.changed.join(", ")} change)</span>}. One step in its favour: <span className="font-medium">{favour.decision.replaceAll("_", " ")}</span>{favour.changed.length > 0 && <span className="text-ink-500"> ({favour.changed.join(", ")} change)</span>}.{" "}
        {ps.robust ? "The verdict does not hinge on any single threshold choice." : "The verdict is within one step of a threshold; read the margins before acting on it."}
      </p>
    </div>
  );
}
