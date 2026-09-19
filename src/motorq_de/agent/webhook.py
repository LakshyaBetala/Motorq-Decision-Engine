"""Post a brief summary to a Slack-compatible incoming webhook when a study finishes.

Enabled by MDE_WEBHOOK_URL. Failures never fail the run - they are logged and swallowed.
Only aggregates leave the process: decision, gates, flags, the cited headline numbers and a
link to the dashboard (MDE_DASHBOARD_URL). No VINs, no raw evidence.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import Any

log = logging.getLogger(__name__)


def summary_payload(res: Any, dataset_hash: str) -> dict[str, Any]:
    v = res.verdict
    failed = [g.name for g in v.gates if not g.passed]
    tripped = [f.name for f in v.flags if f.tripped]
    roi = res.evidence["roi"].outputs if "roi" in res.evidence else {}
    ab = res.evidence["ablation"].outputs if "ablation" in res.evidence else {}
    dash = os.environ.get("MDE_DASHBOARD_URL", "").rstrip("/")
    link = f"{dash}/runs/{res.run_id}" if dash else res.run_id
    lines = [
        f"*{res.brief.capability_name}* -> *{v.decision}* (policy v{v.policy_version})",
        f"target {res.brief.target_event}, horizon {res.brief.horizon_days}d, dataset {dataset_hash}",
        f"gates failed: {', '.join(failed) or 'none'}; flags: {', '.join(tripped) or 'none'}",
    ]
    if roi:
        lines.append(
            f"P(ROI>0) {roi.get('p_roi_positive')}; net value p50 ${roi['net_value_year']['p50']:,.0f}/yr [{res.evidence['roi'].evidence_id}]"
        )
    if ab:
        lines.append(
            f"sufficient set ({len(ab['sufficient_set'])}): {', '.join(ab['sufficient_set'][:10])} [{res.evidence['ablation'].evidence_id}]"
        )
    lines.append(link)
    return {"text": "\n".join(lines)}


def notify_run(res: Any, dataset_hash: str) -> bool:
    url = os.environ.get("MDE_WEBHOOK_URL")
    if not url:
        return False
    try:
        body = json.dumps(summary_payload(res, dataset_hash)).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:  # noqa: S310
            return 200 <= r.status < 300
    except Exception as exc:  # never fail a run because a webhook is down
        log.warning("webhook post failed: %s", exc)
        return False
