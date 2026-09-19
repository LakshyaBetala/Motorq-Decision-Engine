// Runtime proxy to the engine API. Resolved per request so MDE_API_URL works in Docker and at
// runtime, unlike next.config rewrites which are fixed at build time.
import { NextRequest } from "next/server";

export const dynamic = "force-dynamic";

const target = () => (process.env.MDE_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");

async function proxy(req: NextRequest, path: string[]) {
  const url = `${target()}/${path.join("/")}${req.nextUrl.search}`;
  const init: RequestInit = { method: req.method, headers: { "content-type": req.headers.get("content-type") ?? "application/json" }, cache: "no-store" };
  if (req.method !== "GET" && req.method !== "HEAD") init.body = await req.text();
  try {
    const r = await fetch(url, init);
    const ct = r.headers.get("content-type") ?? "application/json";
    if (ct.startsWith("text/event-stream")) return new Response(r.body, { status: r.status, headers: { "content-type": ct, "cache-control": "no-cache" } });
    return new Response(await r.text(), { status: r.status, headers: { "content-type": ct } });
  } catch (e) {
    return Response.json({ detail: `engine API unreachable at ${target()}: ${String(e)}` }, { status: 502 });
  }
}

export async function GET(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  return proxy(req, (await ctx.params).path);
}
export async function POST(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  return proxy(req, (await ctx.params).path);
}
