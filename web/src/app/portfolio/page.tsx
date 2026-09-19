"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { api, Portfolio } from "@/lib/api";
import { DecisionBadge } from "@/components/Decision";

const usd = (x: number | null) => (x == null ? "—" : `$${Math.round(x).toLocaleString()}`);

export default function PortfolioPage() {
  const [p, setP] = useState<Portfolio | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { api.portfolio().then(setP).catch((e) => setErr(String(e))); }, []);
  if (err) return <p className="text-sm text-red-700">{err}</p>;
  if (!p) return <p className="text-sm text-ink-500">Loading…</p>;
  const c = p.cogs;
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Portfolio</h1>
        <p className="text-xs text-ink-500">Latest study per capability. Sorted by decision, then P(ROI &gt; 0). What-if variants are on each run&apos;s page.</p>
      </div>
      <div className="card overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="bg-ink-100 text-left text-xs uppercase tracking-wide text-ink-500">
            <tr><th className="px-3 py-2">Capability</th><th className="px-3 py-2">Decision</th><th className="px-3 py-2">P(ROI&gt;0)</th><th className="px-3 py-2">Net value p50 / yr</th><th className="px-3 py-2">Coverage</th><th className="px-3 py-2">Signals</th><th className="px-3 py-2">Run cost / mo</th><th className="px-3 py-2">Hinges on</th><th className="px-3 py-2">Blocking</th></tr>
          </thead>
          <tbody>
            {p.capabilities.length === 0 && <tr><td className="px-3 py-6 text-ink-500" colSpan={9}>No finished studies yet.</td></tr>}
            {p.capabilities.map((r) => (
              <tr key={r.run_id} className="border-t border-ink-300/50 hover:bg-ink-100/60">
                <td className="px-3 py-2 font-medium"><Link href={`/runs/${r.run_id}`}>{r.capability_name}</Link><div className="mono text-ink-500">{r.target_event} · {r.horizon_days}d</div></td>
                <td className="px-3 py-2"><DecisionBadge decision={r.decision} /></td>
                <td className="px-3 py-2">{r.p_roi_positive == null ? "—" : r.p_roi_positive.toFixed(2)}</td>
                <td className="px-3 py-2">{usd(r.net_value_p50)}</td>
                <td className="px-3 py-2">{r.coverage == null ? "—" : `${(r.coverage * 100).toFixed(0)}%`}</td>
                <td className="px-3 py-2" title={r.sufficient_set.join(", ")}>{r.n_sufficient_signals}</td>
                <td className="px-3 py-2">{usd(r.run_cost_marginal_month)}</td>
                <td className="px-3 py-2 mono text-ink-700">{r.dominant_input ?? "—"}</td>
                <td className="px-3 py-2 text-xs text-ink-500">{[...r.gates_failed.map((g) => `gate:${g}`), ...r.flags_tripped].join(", ") || "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {c && (
        <div className="card p-5 space-y-3">
          <h2 className="text-sm font-semibold">Signal economics across the portfolio</h2>
          <p className="text-sm">{c.n_required_by_viable_capabilities} of {c.n_catalog} catalogued signals are required by a viable (build-ready or pilot) capability; <span className="font-medium">{c.unused_signals.length}</span> are required by none. {c.cadence_lever}.</p>
          {c.realtime_signals_required.length > 0 && <p className="mono text-ink-700">realtime required: {c.realtime_signals_required.join(", ")}</p>}
          <div className="grid gap-3 md:grid-cols-2">
            {Object.entries(c.unused_by_category).map(([cat, sigs]) => (
              <div key={cat} className="rounded border border-ink-300/60 p-3">
                <div className="mb-1 text-xs font-medium uppercase tracking-wide text-ink-500">{cat} · {sigs.length} unused</div>
                <div className="mono text-ink-700">{sigs.join(", ")}</div>
              </div>
            ))}
          </div>
          <p className="text-xs text-ink-500">"Unused" means no viable capability's sufficient signal set includes it. It is an input to a conversation with data integrations about package scope and polling cadence — not an instruction to stop collecting.</p>
        </div>
      )}
    </div>
  );
}
