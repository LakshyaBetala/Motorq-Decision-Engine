import type { NextConfig } from "next";

// API calls go through src/app/api/[...path]/route.ts, a runtime proxy that reads MDE_API_URL
// per request (build-time rewrites would freeze the target).
const nextConfig: NextConfig = { output: "standalone" };

export default nextConfig;
