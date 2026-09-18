const STYLE: Record<string, string> = {
  BUILD_READY: "bg-green-50 text-verdict-build border-green-200",
  PILOT: "bg-amber-50 text-verdict-pilot border-amber-200",
  NOT_FEASIBLE: "bg-red-50 text-verdict-no border-red-200",
};
const LABEL: Record<string, string> = { BUILD_READY: "Build-ready", PILOT: "Pilot", NOT_FEASIBLE: "Not feasible" };

export function DecisionBadge({ decision, size = "sm" }: { decision: string | null; size?: "sm" | "lg" }) {
  if (!decision) return <span className="text-xs text-ink-500">—</span>;
  const cls = STYLE[decision] ?? "bg-ink-100 text-ink-700 border-ink-300";
  const sz = size === "lg" ? "px-3 py-1 text-base font-semibold" : "px-2 py-[2px] text-xs font-medium";
  return <span className={`inline-block rounded border ${cls} ${sz}`}>{LABEL[decision] ?? decision}</span>;
}
