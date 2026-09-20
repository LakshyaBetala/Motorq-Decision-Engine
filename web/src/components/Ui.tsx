"use client";
import Link from "next/link";
import { useEffect } from "react";

/** Citation chips: every number on the page resolves to one of these. */
/** A citation is a link into the ledger. It reads "source" so the prose stays legible; the
 *  evidence id itself is in the tooltip and in the drawer it opens. */
export function Cite({ ids, onOpen, full = false }: { ids: string[]; onOpen: (id: string) => void; full?: boolean }) {
  const list = Array.from(new Set((ids ?? []).filter(Boolean)));
  if (!list.length) return null;
  if (full) {
    return (
      <span className="ml-1 inline-flex flex-wrap gap-1 align-middle">
        {list.map((id) => (
          <button key={id} className="cite" onClick={() => onOpen(id)} title="Open the stored tool call">{id}</button>
        ))}
      </span>
    );
  }
  return (
    <span className="ml-1 inline-flex flex-wrap gap-1 align-middle">
      {list.map((id, i) => (
        <button key={id} className="cite" onClick={() => onOpen(id)} title={`Open the stored tool call ${id}`}>
          {list.length === 1 ? "source" : `source ${i + 1}`}
        </button>
      ))}
    </span>
  );
}

export function Skeleton({ lines = 3, className = "" }: { lines?: number; className?: string }) {
  return (
    <div className={`space-y-2 ${className}`} aria-busy="true" aria-label="Loading">
      {Array.from({ length: lines }).map((_, i) => (
        <div key={i} className="skeleton h-4" style={{ width: `${88 - (i % 3) * 14}%` }} />
      ))}
    </div>
  );
}

export function ApiError({ error }: { error: string }) {
  return (
    <div className="panel max-w-xl p-5 text-sm">
      <p className="font-medium text-ink-950">The API is not reachable.</p>
      <p className="mt-1 text-ink-700">
        Start it with <code className="mono rounded bg-ink-100 px-1">uv run mde serve</code> (port 8000), or point the dashboard at it with <code className="mono rounded bg-ink-100 px-1">MDE_API_URL</code>.
      </p>
      <p className="mono mt-3 text-ink-500">{error}</p>
    </div>
  );
}

export function Empty({ title, body, cta, href }: { title: string; body: string; cta?: string; href?: string }) {
  return (
    <div className="panel flex flex-col items-start gap-3 p-6">
      <p className="text-sm font-medium text-ink-950">{title}</p>
      <p className="max-w-[60ch] text-sm text-ink-700">{body}</p>
      {cta && href && <Link href={href} className="btn btn-primary">{cta}</Link>}
    </div>
  );
}

/** Section heading: the title carries the information; a quiet one-line description may follow. */
export function SectionHead({ id, title, hint, children }: { id?: string; title: string; hint?: string; children?: React.ReactNode }) {
  return (
    <div id={id} className="mb-3 flex flex-wrap items-baseline justify-between gap-2 scroll-mt-24">
      <h2 className="text-[15px] font-semibold tracking-tight text-ink-950">
        {title}
        {hint && <span className="ml-2 font-normal text-ink-500">{hint}</span>}
      </h2>
      {children}
    </div>
  );
}

/** The evidence drawer: opens from the right, where the citations point. Escape closes it. */
export function Drawer({ open, onClose, title, meta, children }: { open: boolean; onClose: () => void; title: string; meta?: string; children: React.ReactNode }) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);
  return (
    <>
      <div className="scrim fixed inset-0 z-40 bg-ink-950/30" data-open={open} onClick={onClose} aria-hidden />
      <aside className="drawer fixed inset-y-0 right-0 z-50 flex w-full max-w-2xl flex-col border-l border-ink-300 bg-white shadow-drawer" data-open={open} role="dialog" aria-modal="true" aria-label={title}>
        <div className="flex items-start justify-between gap-4 border-b border-ink-300 px-5 py-3">
          <div>
            <h3 className="text-sm font-semibold text-ink-950">{title}</h3>
            {meta && <p className="mono mt-0.5 text-ink-500">{meta}</p>}
          </div>
          <button className="btn py-1" onClick={onClose}>Close</button>
        </div>
        <div className="min-h-0 flex-1 overflow-auto px-5 py-4">{children}</div>
      </aside>
    </>
  );
}
