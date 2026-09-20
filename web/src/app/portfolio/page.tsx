"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { api, Portfolio } from "@/lib/api";
import { DecisionBadge } from "@/components/Decision";
import { ApiError, Empty, Skeleton } from "@/components/Ui";

const usd = (x: number | null) => (x == null ? "—" : `$${Math.round(x).toLocaleString()}`);
const CATEGORY: Record<string, string> = {
  location_trips: "Location and trips",
  health: "Vehicle health",
  driver_behavior: "Driver behaviour",
  fuel_energy: "Fuel and energy",
  context: "Context",
  derived: "Derived",
};

export default function PortfolioPage() {
  const [p, setP] = useState<Portfolio | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { api.portfolio().then(setP).catch((e) => setErr(String(e))); }, []);
  if (err) return <ApiError error={err} />;
  if (!p) return <Skeleton lines={5} className="max-w-3xl" />;
  const c = p.cogs;
  return (
    <div className="space-y-10">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Portfolio</h1>
        <p className="mt-1 max-w-[62ch] text-sm text-ink-700">The latest study for each capability, ordered by verdict and then by the probability that it pays for itself. What-if variants live on each study&apos;s page.</p>
      </div>

      {p.capabilities.length === 0 ? (
        <Empty title="No finished studies" body="The portfolio fills in as studies complete. Each capability appears once, with its latest verdict." cta="New study" href="/new" />
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="text-left text-xs text-ink-500">
              <tr className="border-b border-ink-300">
                <th className="py-2 pr-3 font-medium">Capability</th>
                <th className="py-2 pr-3 font-medium">Verdict</th>
                <th className="py-2 pr-3 text-right font-medium">P(ROI &gt; 0)</th>
                <th className="py-2 pr-3 text-right font-medium">Net value / yr, median</th>
                <th className="py-2 pr-3 text-right font-medium">Coverage</th>
                <th className="py-2 pr-3 text-right font-medium">Signals</th>
                <th className="py-2 pr-3 text-right font-medium">Run cost / mo</th>
                <th className="py-2 pr-3 font-medium">Verdict hinges on</th>
                <th className="py-2 font-medium">What blocks build-ready</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-ink-200">
              {p.capabilities.map((r) => (
                <tr key={r.run_id} className="align-top hover:bg-white">
                  <td className="py-2.5 pr-3"><Link href={`/runs/${r.run_id}`} className="font-medium text-ink-950 hover:underline">{r.capability_name}</Link><div className="mono text-ink-500">{r.target_event} within {r.horizon_days} d</div></td>
                  <td className="py-2.5 pr-3"><DecisionBadge decision={r.decision} /></td>
                  <td className="num py-2.5 pr-3 text-right">{r.p_roi_positive == null ? "—" : r.p_roi_positive.toFixed(2)}</td>
                  <td className="num py-2.5 pr-3 text-right">{usd(r.net_value_p50)}</td>
                  <td className="num py-2.5 pr-3 text-right">{r.coverage == null ? "—" : `${(r.coverage * 100).toFixed(0)}%`}</td>
                  <td className="num py-2.5 pr-3 text-right" title={r.sufficient_set.join(", ")}>{r.n_sufficient_signals}</td>
                  <td className="num py-2.5 pr-3 text-right">{usd(r.run_cost_marginal_month)}</td>
                  <td className="mono py-2.5 pr-3 text-ink-700">{r.dominant_input ?? "—"}</td>
                  <td className="py-2.5 text-xs text-ink-700">{[...r.gates_failed.map((g) => `${g} gate`), ...r.flags_tripped.map((f) => f.replaceAll("_", " "))].join(", ") || "nothing"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {c && (
        <section className="space-y-4">
          <h2 className="text-[15px] font-semibold tracking-tight">Signals no viable capability needs</h2>
          <p className="max-w-[72ch] text-sm text-ink-900">
            <span className="num font-semibold">{c.n_required_by_viable_capabilities}</span> of <span className="num">{c.n_catalog}</span> catalogued signals are in the sufficient set of a build-ready or pilot capability; <span className="num font-semibold">{c.unused_signals.length}</span> are in none. {c.cadence_lever}.
          </p>
          {c.realtime_signals_required.length > 0 && <p className="mono text-ink-700">realtime cadence required by: {c.realtime_signals_required.join(", ")}</p>}
          <dl className="grid gap-x-8 gap-y-4 md:grid-cols-2">
            {Object.entries(c.unused_by_category).map(([cat, sigs]) => (
              <div key={cat} className="border-t border-ink-300 pt-2">
                <dt className="text-sm font-medium text-ink-950">{CATEGORY[cat] ?? cat} <span className="num font-normal text-ink-500">{sigs.length}</span></dt>
                <dd className="mono mt-1 text-ink-700">{sigs.join(", ")}</dd>
              </div>
            ))}
          </dl>
          <p className="max-w-[72ch] text-xs text-ink-500">A signal in this list is an input to a conversation with data integrations about package scope and polling cadence, not an instruction to stop collecting it: a future capability may need it, and the portfolio is recomputed whenever a study finishes.</p>
        </section>
      )}
    </div>
  );
}
