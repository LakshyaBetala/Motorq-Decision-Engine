"""SyntheticSource must label correctly and satisfy the DataSource protocol."""

from __future__ import annotations

import pandas as pd

from motorq_de.data.source import DataSource
from motorq_de.schemas import ProblemSpec


def test_satisfies_protocol(source):
    assert isinstance(source, DataSource)


def test_labels_match_events_exactly(source, brake_spec: ProblemSpec):
    lf = source.training_frame(brake_spec, ["odometer_delta_mi"])
    df = lf.frame[lf.frame.y >= 0]
    ev = source.events()
    ev = ev[ev.event_type == brake_spec.target_event]
    # pick a handful of events and check every day in (t-h, t) is labelled positive
    for _, row in ev.head(15).iterrows():
        window = df[
            (df.vehicle_id == row.vehicle_id)
            & (df.date < row.date)
            & (df.date >= row.date - pd.Timedelta(days=brake_spec.horizon_days))
        ]
        assert (window.y == 1).all(), (row.vehicle_id, row.date)
        # and the event day itself is not labelled by its own event (exact match excluded)
        same = df[(df.vehicle_id == row.vehicle_id) & (df.date == row.date)]
        if len(same):
            nxt = ev[
                (ev.vehicle_id == row.vehicle_id)
                & (ev.date > row.date)
                & (ev.date <= row.date + pd.Timedelta(days=brake_spec.horizon_days))
            ]
            assert int(same.y.iloc[0]) == int(len(nxt) > 0)


def test_last_horizon_days_marked_unknowable(source, brake_spec):
    lf = source.training_frame(brake_spec, ["odometer_delta_mi"])
    full = source.date_range()
    cutoff = pd.Timestamp(full.end) - pd.Timedelta(days=brake_spec.horizon_days)
    tail = lf.frame[lf.frame.date > cutoff]
    assert (tail.y == -1).all()
    head = lf.frame[lf.frame.date <= cutoff]
    assert (head.y >= 0).mean() > 0.95  # only theft blackouts are unknowable before the cutoff


def test_signal_metadata_and_catalog(source):
    sigs = source.list_signals()
    assert len(sigs) >= 85
    m = source.signal_metadata("brake_pad_wear_pct")
    assert m.category == "health" and m.declared_frequency == "daily"
