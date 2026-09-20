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
