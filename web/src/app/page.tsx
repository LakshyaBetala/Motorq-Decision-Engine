"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { api, RunSummary } from "@/lib/api";
import { DecisionBadge } from "@/components/Decision";

export default function RunsPage() {
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    api.runs().then(setRuns).catch((e) => setErr(String(e)));
    const t = setInterval(() => api.runs().then(setRuns).catch(() => {}), 5000);
    return () => clearInterval(t);
  }, []);
  if (err) return <p className="text-sm text-red-700">API unreachable: {err}. Start it with <code className="mono">mde serve</code>.</p>;
  if (!runs) return <p className="text-sm text-ink-500">Loading…</p>;
  return (
    <div className="space-y-4">
      <div className="flex items-baseline justify-between">
        <h1 className="text-xl font-semibold tracking-tight">Feasibility studies</h1>
        <p className="text-xs text-ink-500">Every number in every brief resolves to a stored tool call.</p>
      </div>
      <div className="card overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-ink-100 text-left text-xs uppercase tracking-wide text-ink-500">
            <tr><th className="px-4 py-2">Capability</th><th className="px-4 py-2">Target</th><th className="px-4 py-2">Status</th><th className="px-4 py-2">Decision</th><th className="px-4 py-2">Started</th><th className="px-4 py-2">Run</th></tr>
          </thead>
          <tbody>
            {runs.length === 0 && <tr><td className="px-4 py-6 text-ink-500" colSpan={6}>No runs yet. <Link className="underline" href="/new">Start one.</Link></td></tr>}
            {runs.map((r) => (
              <tr key={r.run_id} className="border-t border-ink-300/50 hover:bg-ink-100/60">
                <td className="px-4 py-2 font-medium"><Link href={`/runs/${r.run_id}`}>{r.capability_name ?? "—"}</Link></td>
                <td className="px-4 py-2 mono">{r.target_event}</td>
                <td className="px-4 py-2 text-ink-700">{r.status}</td>
                <td className="px-4 py-2"><DecisionBadge decision={r.decision} /></td>
                <td className="px-4 py-2 text-ink-500">{r.created_at.slice(0, 19).replace("T", " ")}</td>
                <td className="px-4 py-2 mono text-ink-500">{r.run_id}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
