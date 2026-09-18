"""Canonical hashing for determinism and provenance.

Every tool output is rounded and canonically serialised before hashing so that
`same inputs + same dataset + same seed -> same hash` is a testable statement.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

FLOAT_DP = 6


def round_floats(obj: Any) -> Any:
    """Recursively round floats to FLOAT_DP and convert numpy scalars to Python."""
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        return round(obj, FLOAT_DP)
    if isinstance(obj, int):
        return obj
    if hasattr(obj, "item") and callable(obj.item):  # numpy scalar
        return round_floats(obj.item())
    if isinstance(obj, dict):
        return {str(k): round_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round_floats(v) for v in obj]
    if hasattr(obj, "tolist") and callable(obj.tolist):  # numpy array
        return round_floats(obj.tolist())
    return obj


def canonical_json(obj: Any) -> str:
    return json.dumps(round_floats(obj), sort_keys=True, separators=(",", ":"), default=str)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_inputs(inputs: dict[str, Any]) -> str:
    return sha256_hex(canonical_json(inputs))


def evidence_id(run_id: str, tool: str, inputs_hash: str) -> str:
    return "ev_" + sha256_hex(f"{run_id}|{tool}|{inputs_hash}")[:12]
