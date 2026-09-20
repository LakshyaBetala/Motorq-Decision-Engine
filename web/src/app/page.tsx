"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { api, RunSummary } from "@/lib/api";
import { DecisionBar, decisionTone } from "@/components/Decision";
import { ApiError, Empty, Skeleton } from "@/components/Ui";

const KIND: Record<string, string> = { study: "", whatif: "what-if", replay: "replay" };

function when(iso: string) {
  const d = new Date(iso);
  return `${d.toLocaleDateString(undefined, { month: "short", day: "numeric" })} ${d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" })}`;
}

export default function RunsPage() {
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    api.runs().then(setRuns).catch((e) => setErr(String(e)));
    const t = setInterval(() => api.runs().then(setRuns).catch(() => {}), 5000);
    return () => clearInterval(t);
  }, []);

  if (err) return <ApiError error={err} />;

  const done = runs?.filter((r) => r.status === "done") ?? [];
  const tally = { BUILD_READY: 0, PILOT: 0, NOT_FEASIBLE: 0 } as Record<string, number>;
  for (const r of done) if (r.decision) tally[r.decision] = (tally[r.decision] ?? 0) + 1;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Feasibility studies</h1>
          <p className="mt-1 max-w-[62ch] text-sm text-ink-700">
            Each study asks whether a capability can be predicted from the fleet data and whether it is worth building. The verdict is computed by a fixed policy; every number in it links to the calculation that produced it.
          </p>
        </div>
        {runs && runs.length > 0 && (
          <dl className="flex gap-5 text-sm">
            {(["BUILD_READY", "PILOT", "NOT_FEASIBLE"] as const).map((k) => (
              <div key={k} className="flex items-baseline gap-1.5">
                <dd className={`num text-lg font-semibold ${decisionTone(k).text}`}>{tally[k]}</dd>
                <dt className="text-ink-500">{decisionTone(k).label.toLowerCase()}</dt>
              </div>
            ))}
          </dl>
        )}
      </div>

      {!runs && <Skeleton lines={5} className="max-w-3xl" />}

      {runs && runs.length === 0 && (
        <Empty title="No studies yet" body="Start with one of the three example capabilities (brake service, theft risk, battery degradation) or describe your own. A study on the demo fleet takes a few minutes." cta="New study" href="/new" />
      )}

      {runs && runs.length > 0 && (
        <ul className="divide-y divide-ink-300 border-y border-ink-300">
          {runs.map((r) => {
            const running = r.status === "running" || r.status === "queued";
            return (
              <li key={r.run_id}>
                <Link href={`/runs/${r.run_id}`} className="grid grid-cols-[4px_1fr_auto] items-stretch gap-4 py-3 hover:bg-white sm:grid-cols-[4px_minmax(0,2fr)_minmax(0,1.4fr)_auto_auto]">
                  <DecisionBar decision={r.decision} />
                  <div className="min-w-0">
                    <p className="truncate text-sm font-medium text-ink-950">{r.capability_name ?? "untitled"}</p>
                    <p className="mono mt-0.5 text-ink-500">
                      {r.target_event}
                      {(r as any).horizon_days ? ` within ${(r as any).horizon_days} d` : ""}
                      {KIND[(r as any).kind] ? ` (${KIND[(r as any).kind]})` : ""}
                    </p>
                  </div>
                  <p className="hidden self-center text-sm sm:block">
                    {running ? (
                      <span className="inline-flex items-center gap-2 text-ink-700"><span className="live inline-block h-2 w-2 rounded-full bg-verdict-pilot" />running</span>
                    ) : r.status === "failed" ? (
                      <span className="text-verdict-no">failed</span>
                    ) : (
                      <span className={decisionTone(r.decision).text}>{decisionTone(r.decision).label}</span>
                    )}
                  </p>
                  <p className="hidden self-center text-sm text-ink-500 sm:block">{when(r.created_at)}</p>
                  <p className="mono self-center text-ink-400">{r.run_id}</p>
                </Link>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
