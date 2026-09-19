"""Post the portfolio summary to MDE_WEBHOOK_URL (used by portfolio_job.sh)."""

from __future__ import annotations

import json
import os
import urllib.request

from motorq_de.agent.service import Service


def main() -> None:
    p = Service().portfolio()
    c = p["cogs"] or {}
    lines = [f"*Portfolio* - {p['n_capabilities']} capabilities"]
    for r in p["capabilities"]:
        lines.append(
            f"- {r['capability_name']}: {r['decision']} (P(ROI>0) {r['p_roi_positive']}, "
            f"{r['n_sufficient_signals']} signals)"
        )
    if c:
        lines.append(
            f"{c['n_required_by_viable_capabilities']}/{c['n_catalog']} signals required; "
            f"{len(c['unused_signals'])} unused. {c['cadence_lever']}."
        )
    body = json.dumps({"text": "\n".join(lines)}).encode()
    req = urllib.request.Request(
        os.environ["MDE_WEBHOOK_URL"], data=body, headers={"content-type": "application/json"}
    )
    urllib.request.urlopen(req, timeout=10)  # noqa: S310


if __name__ == "__main__":
    main()
