"""Event-level metrics: what a fleet manager experiences, computed from vehicle-day scores."""

from __future__ import annotations

import numpy as np

from motorq_de.harness.stats import event_level_metrics


def _toy():
    # two vehicles; vehicle 0 has one event on day 10 (positives days 3..9 with days_to 7..1),
    # vehicle 1 has an event on day 20 (positives 13..19) and some negatives.
    rows = []
    for v, ev_day in ((0, 10), (1, 20)):
        for d in range(0, 25):
            dt = ev_day - d
            y = 1 if 0 < dt <= 7 else 0
            rows.append((v, d, y, float(dt) if dt > 0 else np.nan))
    v = np.array([r[0] for r in rows])
    dates = (
        np.datetime64("2024-01-01") + np.array([r[1] for r in rows]).astype("timedelta64[D]")
    ).astype("datetime64[D]")
    y = np.array([r[2] for r in rows], dtype=np.int8)
    days_to = np.array([r[3] for r in rows])
    return v, dates, y, days_to


def test_event_recall_and_lead_time():
    v, dates, y, days_to = _toy()
    # score: high only on vehicle 0's positive days -> vehicle 0's event caught, vehicle 1's missed
    s = np.where((v == 0) & (y == 1), 0.9, 0.1).astype(float)
    w = np.ones_like(s)
    m = event_level_metrics(y, s, w, v, dates, days_to, horizon_days=7, alert_rate=7 / 50)
    assert m["n_events"] == 2 and m["n_events_caught"] == 1
    assert m["event_recall"] == 0.5
    assert m["median_lead_days"] == 7.0  # first flag is the earliest positive day
    assert m["false_alerts_per_100_vehicle_months"] == 0.0


def test_false_alert_episodes_scale_with_flagged_negatives():
    v, dates, y, days_to = _toy()
    rng = np.random.default_rng(0)
    s = rng.random(len(y))
    w = np.ones_like(s)
    a = event_level_metrics(y, s, w, v, dates, days_to, 7, alert_rate=0.1)
    b = event_level_metrics(y, s, w, v, dates, days_to, 7, alert_rate=0.4)
    assert b["false_alerts_per_100_vehicle_months"] > a["false_alerts_per_100_vehicle_months"]
    assert 0 <= a["event_recall"] <= 1


def test_weights_correct_for_negative_downsampling():
    v, dates, y, days_to = _toy()
    s = np.where(y == 1, 0.9, 0.5).astype(float)
    # weight negatives 3x (as if 1/3 of negatives were sampled): flagged negatives count 3x
    w1 = np.ones_like(s)
    w3 = np.where(y == 0, 3.0, 1.0)
    a = event_level_metrics(y, s, w1, v, dates, days_to, 7, alert_rate=0.5)
    b = event_level_metrics(y, s, w3, v, dates, days_to, 7, alert_rate=0.5)
    assert a["event_recall"] == b["event_recall"] == 1.0
    assert b["false_alerts_per_100_vehicle_months"] > 0


def test_event_recall_ci_brackets_the_point_and_is_deterministic():
    from motorq_de.harness.stats import event_recall_ci

    # 40 vehicles, one event each; the model catches exactly 30 of them
    rows = []
    for v in range(40):
        for d in range(0, 15):
            dt = 10 - d
            y = 1 if 0 < dt <= 7 else 0
            rows.append((v, d, y, float(dt) if dt > 0 else np.nan))
    v = np.array([r[0] for r in rows])
    dates = np.array([np.datetime64("2025-01-01") + np.timedelta64(r[1], "D") for r in rows])
    y = np.array([r[2] for r in rows])
    days_to = np.array([r[3] for r in rows])
    s = np.where((y == 1) & (v < 30), 0.9, 0.1).astype(float)
    w = np.ones_like(s)
    ci = event_recall_ci(y, s, w, v, dates, days_to, alert_rate=(30 * 7) / len(y), B=200, seed=3)
    assert ci.point == 0.75
    assert ci.lo < 0.75 < ci.hi
    assert 0.55 < ci.lo and ci.hi < 0.95  # 40 clusters: a real, finite interval
    again = event_recall_ci(y, s, w, v, dates, days_to, alert_rate=(30 * 7) / len(y), B=200, seed=3)
    assert (again.lo, again.hi) == (ci.lo, ci.hi)


def test_event_recall_ci_is_nan_without_events():
    from motorq_de.harness.stats import event_recall_ci

    y = np.zeros(10, dtype=int)
    s = np.linspace(0, 1, 10)
    ci = event_recall_ci(
        y,
        s,
        np.ones(10),
        np.arange(10),
        np.arange(10).astype("datetime64[D]"),
        np.full(10, np.nan),
        0.2,
    )
    assert np.isnan(ci.point)
