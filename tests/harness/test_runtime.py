"""The runtime fingerprint separates what changes arithmetic from what only changes speed."""

from __future__ import annotations

import pytest

from motorq_de.harness import runtime


def test_fingerprint_records_versions_and_determinism_settings():
    fp = runtime.fingerprint()
    assert fp["lgbm_deterministic"] is True and fp["lgbm_histogram"] == "row_wise"
    assert fp["libraries"]["lightgbm"] and fp["libraries"]["numpy"]
    assert fp["python"].count(".") == 2
    assert fp["fold_workers"] >= 1 and fp["lgbm_threads"] >= 1


def test_thread_and_worker_counts_are_not_part_of_numeric_identity(monkeypatch: pytest.MonkeyPatch):
    a = runtime.fingerprint()
    monkeypatch.setenv("MDE_CPUS", "2")
    monkeypatch.setenv("MDE_FOLD_JOBS", "1")
    b = runtime.fingerprint()
    assert (a["fold_workers"], a["lgbm_threads"]) != (b["fold_workers"], b["lgbm_threads"])
    assert runtime.same_numeric_environment(a, b)
    c = dict(a, libraries=dict(a["libraries"], lightgbm="0.0.0"))
    assert not runtime.same_numeric_environment(a, c)
