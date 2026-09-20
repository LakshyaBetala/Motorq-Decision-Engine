"use client";
import { useState } from "react";
import { CartesianGrid, Line, LineChart, ReferenceDot, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "@/lib/api";

// ---------------------------------------------------------------- coverage heatmap
export function CoverageHeatmap({ cov }: { cov: any }) {
  const signals: string[] = cov.signals;
  const oems = Object.keys(cov.fleet_share_full_set_by_oem).sort();
  const cell = (s: string, o: string) => cov.per_signal[s]?.[o]?.coverage ?? 0;
  const color = (v: number) => (v === 0 ? "#fee2e2" : v < 0.5 ? "#fde68a" : v < 0.9 ? "#bbf7d0" : "#22c55e");
  return (
    <div className="overflow-x-auto">
      <table className="text-[11px]">
        <thead>
          <tr>
            <th className="pr-2 text-left font-medium text-ink-500">signal \ OEM</th>
            {oems.map((o) => <th key={o} className="px-1 font-medium text-ink-500">{o.replace("oem_", "").toUpperCase()}</th>)}
            <th className="pl-2 font-medium text-ink-500">fleet</th>
          </tr>
        </thead>
        <tbody>
          {signals.map((s) => (
            <tr key={s}>
              <td className="mono pr-2 text-ink-700">{s}</td>
              {oems.map((o) => {
                const v = cell(s, o);
                return <td key={o} className="px-1 py-[2px]"><div title={`${s} on ${o}: ${(v * 100).toFixed(0)}%`} className="h-4 w-8 rounded-sm" style={{ background: color(v) }} /></td>;
              })}
              <td className="pl-2 text-ink-500">{(Object.values(cov.per_signal[s] ?? {}) as any[]).reduce((a, c) => a + c.vehicles_covered, 0)}</td>
            </tr>
          ))}
          <tr>
            <td className="pr-2 pt-1 font-medium text-ink-700">whole set</td>
            {oems.map((o) => <td key={o} className="px-1 pt-1 text-center text-ink-700">{(cov.fleet_share_full_set_by_oem[o] * 100).toFixed(0)}%</td>)}
            <td className="pl-2 pt-1 font-medium">{(cov.fleet_share_full_set * 100).toFixed(0)}%</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------- ablation table
export function AblationTable({ ab }: { ab: any }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-[12px]">
        <thead className="text-left text-ink-500"><tr><th>step</th><th>removed</th><th>signals</th><th>AUC</th><th>Δ vs full [95% CI]</th><th>P(non-inferior)</th><th>result</th></tr></thead>
        <tbody>
          {ab.screened_out?.length > 0 && <tr className="border-t border-ink-300/50"><td>screen</td><td className="mono text-ink-500" colSpan={6}>{ab.screened_out.length} signals removed by the importance screen: {ab.screened_out.join(", ")}</td></tr>}
          {ab.trace.map((t: any) => (
            <tr key={t.step} className={`border-t border-ink-300/50 ${t.accepted ? "" : "bg-red-50"}`}>
              <td>{t.step}</td>
              <td className="mono">{t.removed ?? "—"}</td>
              <td>{t.n_signals}</td>
              <td>{t.auc.point.toFixed(4)}</td>
              <td>{t.delta_vs_full ? `${t.delta_vs_full.point >= 0 ? "+" : ""}${t.delta_vs_full.point.toFixed(4)} [${t.delta_vs_full.lo.toFixed(4)}, ${t.delta_vs_full.hi.toFixed(4)}]` : "—"}</td>
              <td>{t.p_noninferior != null ? t.p_noninferior.toFixed(2) : "—"}{t.underpowered ? " (underpowered)" : ""}</td>
              <td>{t.removed == null ? "full set" : t.accepted ? "removed" : "kept: needed"}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-1 text-xs text-ink-500">{ab.cost_aware ? "Cost-aware order (least importance per dollar removed first). " : ""}Tolerance {ab.tolerance} AUC at {(ab.noninferiority_confidence * 100).toFixed(0)}% bootstrap confidence. Underpowered: the bootstrap could not resolve half the tolerance on an accepted removal.</p>
    </div>
  );
}

// ---------------------------------------------------------------- operating point curve
export function OperatingCurve({ points, chosen }: { points: any[]; chosen: any }) {
  const data = points.map((p) => ({ alert: p.alert_rate * 100, event_recall: p.event_recall ?? p.recall, precision: p.precision, fa: p.false_alerts_per_100_vehicle_months }));
  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
        <CartesianGrid stroke="#e8ebef" vertical={false} />
        <XAxis dataKey="alert" tick={{ fontSize: 11 }} tickFormatter={(v) => `${v}%`} label={{ value: "vehicle-days alerted", position: "insideBottom", offset: -4, fontSize: 11 }} />
        <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} width={36} />
        <Tooltip formatter={(v: number, n) => [typeof v === "number" ? v.toFixed(3) : v, n]} labelFormatter={(l, p) => `alert ${l}%: ${p?.[0]?.payload?.fa?.toFixed(1)} false alerts per 100 vehicle-months`} contentStyle={{ fontSize: 12 }} />
        <Line type="monotone" dataKey="event_recall" name="events caught" stroke="#14161a" dot={{ r: 2 }} isAnimationActive={false} />
        <Line type="monotone" dataKey="precision" name="precision" stroke="#94a3b8" dot={{ r: 2 }} isAnimationActive={false} />
        {chosen && <ReferenceDot x={chosen.alert_rate * 100} y={chosen.recall} r={6} fill="#b45309" stroke="none" />}
      </LineChart>
    </ResponsiveContainer>
  );
}

// ---------------------------------------------------------------- what-if panel
type Rng = { low: number; base: number; high: number };
const FIELDS: { key: string; label: string; step: number; max: number; money?: boolean }[] = [
  { key: "value_bearing_fraction", label: "value-bearing fraction of events", step: 0.01, max: 1 },
  { key: "preventable_fraction", label: "preventable fraction", step: 0.01, max: 1 },
  { key: "usd_per_avoided_event", label: "$ per avoided event", step: 100, max: 50000, money: true },
  { key: "fleet_size", label: "fleet size", step: 100, max: 200000 },
];

export function WhatIf({ runId, value, onDone }: { runId: string; value: Record<string, Rng>; onDone: (id: string) => void }) {
  const [v, setV] = useState<Record<string, Rng>>(JSON.parse(JSON.stringify(value)));
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const set = (k: string, which: keyof Rng, x: number) => setV((s) => ({ ...s, [k]: { ...s[k], [which]: x } }));
  async function run() {
    setBusy(true); setErr(null);
    try {
      const payload = { value: { ...v, source_note: "what-if from dashboard" } };
      const r = await api.whatif(runId, payload);
      onDone(r.run_id);
    } catch (e) { setErr(String(e)); }
    setBusy(false);
  }
  return (
    <div className="space-y-3">
      {FIELDS.map((f) => (
        <div key={f.key} className="grid grid-cols-[1fr_auto_auto_auto] items-center gap-2 text-xs">
          <span className="text-ink-700">{f.label}</span>
          {(["low", "base", "high"] as const).map((w) => (
            <label key={w} className="flex items-center gap-1 text-ink-500">{w}
              <input type="number" step={f.step} min={0} max={f.max} className="input num w-24 px-1 py-[2px] text-right" value={v[f.key]?.[w] ?? 0} onChange={(e) => set(f.key, w, Number(e.target.value))} />
            </label>
          ))}
        </div>
      ))}
      <div className="flex items-center gap-3">
        <button onClick={run} disabled={busy} className="btn btn-primary">{busy ? "Recomputing" : "Recompute economics and verdict"}</button>
        <span className="text-xs text-ink-500">Takes seconds. Reuses this study&apos;s evidence and records a derived study.</span>
      </div>
      {err && <p className="text-xs text-verdict-no">{err}</p>}
    </div>
  );
}
