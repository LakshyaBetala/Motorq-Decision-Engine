import type { Config } from "tailwindcss";

export default {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["var(--font-sans)", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      colors: {
        // graphite neutrals with a slight cool cast: an engineering ledger, not a marketing page
        ink: { 950: "#14161a", 900: "#1c1f24", 700: "#3f4652", 500: "#5b6470", 400: "#8a929d", 300: "#d9dde3", 200: "#e8ebef", 100: "#f6f7f8" },
        // the one interaction accent, reserved for citations and links
        accent: { DEFAULT: "#3157b5", soft: "#e9eef9" },
        verdict: { build: "#1f7a4d", pilot: "#b3600e", no: "#b42328" },
      },
      transitionTimingFunction: { out: "cubic-bezier(0.23, 1, 0.32, 1)" },
      boxShadow: { drawer: "-24px 0 48px -24px rgba(20, 22, 26, 0.25)" },
    },
  },
  plugins: [],
} satisfies Config;
