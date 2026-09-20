const TONE: Record<string, { text: string; bar: string; soft: string; label: string }> = {
  BUILD_READY: { text: "text-verdict-build", bar: "bg-verdict-build", soft: "bg-green-50 border-green-200", label: "Build-ready" },
  PILOT: { text: "text-verdict-pilot", bar: "bg-verdict-pilot", soft: "bg-amber-50 border-amber-200", label: "Pilot" },
  NOT_FEASIBLE: { text: "text-verdict-no", bar: "bg-verdict-no", soft: "bg-red-50 border-red-200", label: "Not feasible" },
};

export function decisionTone(decision: string | null) {
  return (decision && TONE[decision]) || { text: "text-ink-500", bar: "bg-ink-300", soft: "bg-ink-100 border-ink-300", label: decision ?? "—" };
}

export function DecisionBadge({ decision, size = "sm" }: { decision: string | null; size?: "sm" | "lg" }) {
  const t = decisionTone(decision);
  if (!decision) return <span className="text-xs text-ink-400">pending</span>;
  const sz = size === "lg" ? "px-3 py-1 text-base font-semibold" : "px-2 py-[2px] text-xs font-medium";
  return <span className={`inline-block rounded-md border ${t.soft} ${t.text} ${sz}`}>{t.label}</span>;
}

/** A vertical mark in the verdict colour: used as the left edge of a study row. */
export function DecisionBar({ decision }: { decision: string | null }) {
  return <span aria-hidden className={`block h-full w-1 rounded-full ${decisionTone(decision).bar}`} />;
}
