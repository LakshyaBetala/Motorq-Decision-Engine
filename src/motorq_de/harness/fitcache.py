"""Out-of-fold prediction cache: never fit the same model on the same rows twice.

A study fits LightGBM well over a hundred times, and several of those fits are duplicates:
model comparison refits the sufficient set ablation just fitted, the paired comparison refits
both sets again, and a re-run of a study on unchanged data refits everything. The cache keys an
out-of-fold prediction vector by everything that determines it - the row identity of the
matrix (dataset, label definition, sampling), the ordered signal list, the model, the fold
count, the seed and the numeric environment - so a hit is bit-identical to a fresh fit.

Replays disable it: their purpose is to recompute.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from motorq_de.harness import runtime
from motorq_de.hashing import hash_inputs


@dataclass
class FitCache:
    directory: Path = field(
        default_factory=lambda: Path(os.environ.get("MDE_CACHE_DIR", "data/cache"))
    )
    enabled: bool = True
    hits: int = 0
    misses: int = 0
    _mem: dict[str, np.ndarray] = field(default_factory=dict)

    @staticmethod
    def key(row_key: str, signals: list[str], model: str, n_splits: int, seed: int) -> str:
        return hash_inputs(
            {
                "rows": row_key,
                "signals": list(signals),  # ordered: column order affects column subsampling
                "model": model,
                "n_splits": n_splits,
                "seed": seed,
                "env": runtime.numeric_identity(runtime.fingerprint()),
            }
        )

    def _path(self, key: str) -> Path:
        return self.directory / "oof" / f"{key}.npy"

    def get(self, key: str) -> np.ndarray | None:
        if not self.enabled:
            return None
        if key in self._mem:
            self.hits += 1
            return self._mem[key]
        p = self._path(key)
        if p.exists():
            arr = np.load(p)
            self._mem[key] = arr
            self.hits += 1
            return arr
        self.misses += 1
        return None

    def put(self, key: str, arr: np.ndarray, meta: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        self._mem[key] = arr
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, arr)
        if meta:
            p.with_suffix(".json").write_text(json.dumps(meta, sort_keys=True, default=str))

    def stats(self) -> dict[str, int]:
        return {"cache_hits": self.hits, "cache_misses": self.misses}
