#!/usr/bin/env bash
# Monthly portfolio / COGS report. Reads the ledger; posts a summary to MDE_WEBHOOK_URL if set.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run mde portfolio
if [ -n "${MDE_WEBHOOK_URL:-}" ]; then
  uv run python scripts/portfolio_post.py
fi
