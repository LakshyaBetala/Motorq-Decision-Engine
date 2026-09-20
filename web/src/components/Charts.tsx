"use client";
import { Bar, BarChart, CartesianGrid, ErrorBar, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

// the page palette: ink for data, the verdict colours for good / bad
const C = { primary: "#14161a", muted: "#8a929d", good: "#1f7a4d", warn: "#b3600e", bad: "#b42328" };

export function AblationTrace({ trace }: { trace: any[] }) {
  const data = trace.map((t) => ({ step: t.step, label: t.removed ?? "full", auc: t.auc.point, err: [t.auc.point - t.auc.lo, t.auc.hi - t.auc.point], accepted: t.accepted }));
  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
        <CartesianGrid stroke="#e8ebef" vertical={false} />
        <XAxis dataKey="step" tick={{ fontSize: 11 }} label={{ value: "signals removed", position: "insideBottom", offset: -4, fontSize: 11 }} />
        <YAxis domain={["auto", "auto"]} tick={{ fontSize: 11 }} tickFormatter={(v) => v.toFixed(3)} width={48} />
        <Tooltip formatter={(v: number) => v.toFixed(4)} labelFormatter={(l, p) => `step ${l}: removed ${p?.[0]?.payload?.label}${p?.[0]?.payload?.accepted ? "" : " (rejected)"}`} contentStyle={{ fontSize: 12 }} />
        <Line type="monotone" dataKey="auc" stroke={C.primary} dot={{ r: 3 }} isAnimationActive={false}>
          <ErrorBar dataKey="err" width={3} stroke={C.muted} />
        </Line>
      </LineChart>
    </ResponsiveContainer>
  );
}

export function CrossOem({ per_oem, mean }: { per_oem: Record<string, any>; mean: number | null }) {
  const data = Object.entries(per_oem)
    .filter(([, v]) => v.auc)
    .map(([k, v]) => ({ oem: k, auc: v.auc.point, err: [v.auc.point - v.auc.lo, v.auc.hi - v.auc.point], missing: v.feature_missing_share }));
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
        <CartesianGrid stroke="#e8ebef" vertical={false} />
        <XAxis dataKey="oem" tick={{ fontSize: 11 }} interval={0} tickFormatter={(v) => String(v).replace("oem_", "").toUpperCase()} />
        <YAxis domain={[0.5, 1]} ticks={[0.5, 0.6, 0.7, 0.8, 0.9, 1]} tick={{ fontSize: 11 }} width={40} tickFormatter={(v) => Number(v).toFixed(1)} />
        <Tooltip formatter={(v: number, n, p) => [`${v.toFixed(3)} (missing features ${(p.payload.missing * 100).toFixed(0)}%)`, "AUC held out"]} contentStyle={{ fontSize: 12 }} />
        {mean != null && <ReferenceLine y={mean} stroke={C.muted} strokeDasharray="4 4" />}
        <Bar dataKey="auc" fill={C.primary} isAnimationActive={false}>
          <ErrorBar dataKey="err" width={3} stroke={C.muted} />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export function CrossOemCaption({ mean }: { mean: number | null }) {
  if (mean == null) return null;
  return <p className="mt-1 text-xs text-ink-500">Dashed line: mean held-out AUC <span className="num">{mean.toFixed(3)}</span>. Whiskers are 95% vehicle-bootstrap intervals.</p>;
}

export function Tornado({ ranked, base }: { ranked: any[]; base: number }) {
  const data = ranked.map((r) => ({ input: r.input, low: Math.min(r.roi_at_low, r.roi_at_high) - base, high: Math.max(r.roi_at_low, r.roi_at_high) - base }));
  return (
    <ResponsiveContainer width="100%" height={40 + 26 * data.length}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 16, left: 8, bottom: 4 }} stackOffset="sign">
        <CartesianGrid stroke="#e8ebef" horizontal={false} />
        <XAxis type="number" tick={{ fontSize: 11 }} tickFormatter={(v) => (v + base).toFixed(1)} />
        <YAxis type="category" dataKey="input" tick={{ fontSize: 11 }} width={150} />
        <Tooltip formatter={(v: number) => (v + base).toFixed(2)} contentStyle={{ fontSize: 12 }} />
        <ReferenceLine x={0} stroke={C.primary} />
        <Bar dataKey="low" stackId="a" fill={C.bad} isAnimationActive={false} />
        <Bar dataKey="high" stackId="a" fill={C.good} isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function Importance({ ranking }: { ranking: any[] }) {
  const data = ranking.slice(0, 12).map((r) => ({ signal: r.signal, imp: r.perm_importance, err: [r.perm_importance - r.perm_ci_lo, r.perm_ci_hi - r.perm_importance] }));
  return (
    <ResponsiveContainer width="100%" height={40 + 22 * data.length}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 16, left: 8, bottom: 4 }}>
        <CartesianGrid stroke="#e8ebef" horizontal={false} />
        <XAxis type="number" tick={{ fontSize: 11 }} tickFormatter={(v) => v.toFixed(3)} />
        <YAxis type="category" dataKey="signal" tick={{ fontSize: 11 }} width={170} />
        <Tooltip formatter={(v: number) => v.toFixed(4)} contentStyle={{ fontSize: 12 }} />
        <Bar dataKey="imp" fill={C.primary} isAnimationActive={false}>
          <ErrorBar dataKey="err" width={3} stroke={C.muted} direction="x" />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
