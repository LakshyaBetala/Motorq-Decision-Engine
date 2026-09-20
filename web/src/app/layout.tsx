import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const sans = Geist({ subsets: ["latin"], variable: "--font-sans" });
const mono = Geist_Mono({ subsets: ["latin"], variable: "--font-mono" });

export const metadata: Metadata = {
  title: "Motorq Decision Engine",
  description: "Feasibility and signal economics for connected-vehicle capabilities, with every number traceable to the computation that produced it.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable}`}>
      <body>
        <header className="sticky top-0 z-30 border-b border-ink-300 bg-white/95 backdrop-blur">
          <div className="mx-auto flex h-12 max-w-6xl items-center justify-between px-6">
            <Link href="/" className="text-sm font-semibold tracking-tight text-ink-950">
              Motorq Decision Engine
            </Link>
            <nav className="flex items-center gap-1 text-sm">
              <Link href="/" className="rounded-md px-2.5 py-1 text-ink-700 hover:bg-ink-100 hover:text-ink-950">Studies</Link>
              <Link href="/portfolio" className="rounded-md px-2.5 py-1 text-ink-700 hover:bg-ink-100 hover:text-ink-950">Portfolio</Link>
              <Link href="/new" className="btn btn-primary ml-2 py-1">New study</Link>
            </nav>
          </div>
        </header>
        <main className="mx-auto max-w-6xl px-6 pb-16 pt-8">{children}</main>
      </body>
    </html>
  );
}
