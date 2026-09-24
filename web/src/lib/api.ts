// Thin client over the FastAPI service. All calls go through the Next.js rewrite (/api -> MDE_API_URL).

export type RunSummary = {
  run_id: string;
  capability_name: string | null;
  target_event: string | null;
  status: string;
  decision: string | null;
  created_at: string;
  dataset_hash: string;
  kind: "study" | "whatif" | "replay";
  derived_from: string | null;
  horizon_days: number | null;
};

export type Gate = { name: string; passed: boolean; value: number | null; threshold: number | null; evidence_ids: string[]; note: string };
export type Flag = { name: string; tripped: boolean; value: number | null; threshold: number | null; evidence_ids: string[]; note: string };
export type Verdict = { decision: "BUILD_READY" | "PILOT" | "NOT_FEASIBLE"; gates: Gate[]; flags: Flag[]; policy_version: string };

export type RunDetail = {
  run_id: string;
  spec: Record<string, unknown>;
  dataset_hash: string;
  status: string;
  created_at: string;
  finished_at: string | null;
  verdict: Verdict | null;
  error: string | null;
  llm_used: boolean;
  request_text: string | null;
  derived_from: string | null;
  kind: string;
  steps: { name: string; status: string; started_at: string; ended_at: string | null; note: string | null }[];
  evidence: { evidence_id: string; step: string; tool: string; name?: string; duration_ms: number; created_at: string }[];
};

export type BriefLine = { text: string; evidence_ids: string[] };
export type Brief = {
  run_id: string;
  decision: string;
  capability_name: string;
  target_event: string;
  horizon_days: number;
  policy_version: string;
  sections: { title: string; lines: BriefLine[] }[];
  narrative: BriefLine[];
  generated_at: string;
  llm_used: boolean;
};

export type Portfolio = {
  n_capabilities: number;
  capabilities: {
    capability_name: string; run_id: string; target_event: string; horizon_days: number | null; decision: string | null;
    gates_failed: string[]; flags_tripped: string[]; p_roi_positive: number | null; net_value_p50: number | null;
    coverage: number | null; n_sufficient_signals: number; sufficient_set: string[]; run_cost_marginal_month: number | null;
    dominant_input: string | null; created_at: string; dataset_hash: string;
  }[];
  cogs: {
    n_catalog: number; n_required_by_viable_capabilities: number; required: string[]; required_by: Record<string, string[]>;
    unused_signals: string[]; unused_by_category: Record<string, string[]>; realtime_signals_required: string[]; cadence_lever: string;
  } | null;
};

export type Evidence = {
  evidence_id: string;
  run_id: string;
  step: string;
  tool: string;
  name?: string;
  inputs: Record<string, unknown>;
  inputs_hash: string;
  dataset_hash: string;
  seed: number;
  outputs: Record<string, any>;
  duration_ms: number;
  created_at: string;
};

async function j<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, { ...init, headers: { "content-type": "application/json", ...(init?.headers ?? {}) }, cache: "no-store" });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

export const api = {
  health: () => j<{ ok: boolean; dataset_hash: string; llm_available: boolean }>("/api/health"),
  runs: () => j<RunSummary[]>("/api/runs"),
  run: (id: string) => j<RunDetail>(`/api/runs/${id}`),
  brief: (id: string) => j<{ run_id: string; decision: string; markdown: string; json: Brief }>(`/api/runs/${id}/brief`),
  evidence: (id: string) => j<Evidence[]>(`/api/runs/${id}/evidence`),
  evidenceOne: (id: string, eid: string) => j<Evidence>(`/api/runs/${id}/evidence/${eid}`),
  examples: () => j<Record<string, Record<string, unknown>>>("/api/examples"),
  create: (body: { example?: string; text?: string; spec?: unknown; use_llm?: boolean }) =>
    j<{ ticket: string; spec: unknown; notes: string }>("/api/runs", { method: "POST", body: JSON.stringify(body) }),
  ticket: (t: string) => j<{ ticket: string; status: string; run_id?: string; decision?: string; error?: string }>(`/api/tickets/${t}`),
  ask: (id: string, question: string) => j<{ answer: string; evidence_ids: string[] }>(`/api/runs/${id}/ask`, { method: "POST", body: JSON.stringify({ question }) }),
  whatif: (id: string, body: { value?: unknown; price_overrides?: unknown; constraints?: unknown }) =>
    j<{ run_id: string; decision: string; derived_from: string }>(`/api/runs/${id}/whatif`, { method: "POST", body: JSON.stringify(body) }),
  replay: (id: string) => j<{ ticket: string; replay_of: string }>(`/api/runs/${id}/replay`, { method: "POST" }),
  portfolio: () => j<Portfolio>("/api/portfolio"),
  messages: (id: string) => j<{ role: string; content: string; evidence_ids: string[]; created_at: string }[]>(`/api/runs/${id}/messages`),
};
