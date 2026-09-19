import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = { title: "Motorq Decision Engine", description: "Capability feasibility & signal economics" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header className="border-b border-ink-300/70 bg-white">
          <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-3">
            <Link href="/" className="flex items-baseline gap-2">
              <span className="text-sm font-semibold tracking-tight">Motorq Decision Engine</span>
              <span className="text-xs text-ink-500">capability feasibility · signal economics</span>
            </Link>
            <nav className="flex gap-4 text-sm">
              <Link href="/" className="text-ink-700 hover:text-ink-900">Runs</Link>
              <Link href="/portfolio" className="text-ink-700 hover:text-ink-900">Portfolio</Link>
              <Link href="/new" className="rounded bg-ink-900 px-3 py-1 text-white hover:bg-ink-700">New study</Link>
            </nav>
          </div>
        </header>
        <main className="mx-auto max-w-6xl px-6 py-6">{children}</main>
      </body>
    </html>
  );
}
