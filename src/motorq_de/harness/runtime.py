"""Runtime fingerprint: what a study's numbers depend on besides data, spec and seed.

LightGBM guarantees identical results for the same data, parameters and seed *within* a build
(its `deterministic` mode also removes thread-count effects), but not across library versions,
compilers or platforms. So every run records this fingerprint as evidence; a replay whose
numbers differ is only a defect if the fingerprints match.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

from motorq_de.harness.models import LGBM_PARAMS, cpu_budget, cv_parallelism

LIBRARIES = ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "joblib")

# Fields that change the arithmetic. Everything else (thread counts, hostnames) is context.
NUMERIC_IDENTITY = ("python", "platform", "libraries", "lgbm_params_hash", "code_version")


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def code_version() -> str:
    """Installed package version, plus the git commit when running from a checkout."""
    v = _version("motorq-de") or "unknown"
    try:
        root = Path(__file__).resolve().parents[3]
        sha = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout.strip()
        return f"{v}+{sha}" if sha else v
    except (OSError, subprocess.SubprocessError):
        return v


def fingerprint(n_folds: int = 5) -> dict[str, Any]:
    workers, threads = cv_parallelism(n_folds)
    params = json.dumps(LGBM_PARAMS, sort_keys=True, default=str)
    return {
        "python": platform.python_version(),
        "platform": f"{platform.system()}-{platform.machine()}",
        "libraries": {lib: _version(lib) for lib in LIBRARIES},
        "lgbm_params_hash": hashlib.sha256(params.encode()).hexdigest()[:12],
        "lgbm_deterministic": bool(LGBM_PARAMS.get("deterministic")),
        "lgbm_histogram": "row_wise" if LGBM_PARAMS.get("force_row_wise") else "default",
        "code_version": code_version(),
        "cpu_budget": cpu_budget(),
        "fold_workers": workers,
        "lgbm_threads": threads,
    }


def numeric_identity(fp: dict[str, Any]) -> dict[str, Any]:
    return {k: fp.get(k) for k in NUMERIC_IDENTITY}


def same_numeric_environment(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return numeric_identity(a) == numeric_identity(b)
