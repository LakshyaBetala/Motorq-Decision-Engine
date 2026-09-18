"""Volume and run-cost model for a signal set.

cost = unit price (price_sheet.yaml, sourced) x volume (derived from the signal set,
fleet size and each signal's cadence). Two views are always produced:

    marginal   infra only + OEM packages Motorq does not already buy
    attributed marginal + the OEM package cost of every package the signal set touches

Everything is returned with the intermediate volumes so the arithmetic is auditable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from motorq_de.schemas import SignalMeta

HERE = Path(__file__).parent
DAYS_PER_MONTH = 30.44
FEATURES_PER_SIGNAL = 4  # raw, m7, m30, d30 (see harness.frames)
BYTES_PER_FEATURE = 4


def load_price_sheet(path: Path | None = None) -> dict[str, Any]:
    with open(path or HERE / "price_sheet.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def signal_volume(meta: SignalMeta, fleet_size: float, prices: dict[str, Any]) -> dict[str, float]:
    tel = prices["telemetry"]
    samples = tel["samples_per_day_by_frequency"][meta.declared_frequency]
    records_day = fleet_size * samples
    bytes_day = records_day * tel["bytes_per_record"]
    return {"records_per_day": records_day, "bytes_per_day": bytes_day}


def run_cost(
    signals: list[SignalMeta],
    fleet_size: float,
    delivery_mode: str,
    prices: dict[str, Any] | None = None,
) -> dict[str, Any]:
    p = prices or load_price_sheet()
    sf, kf, oem, inf, eng = (
        p["snowflake"],
        p["kafka"],
        p["oem_data"],
        p["inference"],
        p["engineering"],
    )

    # ---------------------------------------------------------------- volumes
    per_signal = {s.signal_id: signal_volume(s, fleet_size, p) for s in signals}
    records_day = sum(v["records_per_day"] for v in per_signal.values())
    bytes_day = sum(v["bytes_per_day"] for v in per_signal.values())
    gb_month = bytes_day * DAYS_PER_MONTH / 1e9
    raw_stored_tb = (
        bytes_day
        * DAYS_PER_MONTH
        * p["telemetry"]["raw_retention_months"]
        / sf["compression_ratio"]
        / 1e12
    )
    feature_cells_day = fleet_size * len(signals) * FEATURES_PER_SIGNAL
    feature_store_tb = (
        feature_cells_day * BYTES_PER_FEATURE * DAYS_PER_MONTH * 3 / 1e12
    )  # 3 months of features
    predictions_month = fleet_size * DAYS_PER_MONTH * (24 if delivery_mode == "batch_hourly" else 1)

    # ---------------------------------------------------------------- infra costs / month
    ingest_credits = gb_month * (
        sf["snowpipe_streaming_credits_per_gb"]
        if delivery_mode == "streaming"
        else sf["snowpipe_credits_per_gb"]
    )
    ingest_usd = ingest_credits * sf["credit_usd"]
    storage_usd = (raw_stored_tb + feature_store_tb) * sf["storage_per_tb_month"]
    job_minutes_day = feature_cells_day / sf["feature_job_cells_per_minute"]
    compute_usd = (
        job_minutes_day
        / 60
        * sf["feature_job_credits_per_hour"]
        * sf["credit_usd"]
        * DAYS_PER_MONTH
    )
    kafka_usd = (
        gb_month * (kf["per_gb_in"] + kf["per_gb_out"])
        + (bytes_day * kf["retention_days"] / 1e9) * kf["storage_per_gb_month"]
    )
    if delivery_mode not in ("streaming",):
        kafka_usd *= (
            0.5  # batch delivery still rides Kafka in but not the streaming egress. modeled.
        )
    inference_usd = predictions_month / 1e6 * inf["usd_per_million_predictions"]

    # ---------------------------------------------------------------- OEM packages
    packages = sorted({s.category for s in signals})
    add_on = oem["package_add_on_per_vehicle_month"]
    attributed_oem = fleet_size * (
        oem["base_per_vehicle_month"] + sum(add_on.get(c, 0.0) for c in packages)
    )
    incremental_pkgs = [c for c in packages if c not in set(oem["already_purchased_packages"])]
    marginal_oem = fleet_size * sum(add_on.get(c, 0.0) for c in incremental_pkgs)
    # polling: calls/vehicle/day = highest cadence any signal in the set requires
    cadence = p["telemetry"]["samples_per_day_by_frequency"]
    max_calls_day = max((cadence[s.declared_frequency] for s in signals), default=0.0)
    api_calls_month = fleet_size * max_calls_day * DAYS_PER_MONTH
    api_usd = (
        api_calls_month / 1000 * oem["api_call_usd_per_1k"]
        if oem.get("billing_model") == "pull"
        else 0.0
    )

    # ---------------------------------------------------------------- build (one-off, amortised)
    build_usd = eng["build_hours_by_pattern"][delivery_mode] * eng["usd_per_hour"]
    build_month = build_usd / eng["amortization_months"]

    infra_month = ingest_usd + storage_usd + compute_usd + kafka_usd + inference_usd
    return {
        "n_signals": len(signals),
        "signals": [s.signal_id for s in signals],
        "fleet_size": fleet_size,
        "delivery_mode": delivery_mode,
        "volumes": {
            "records_per_day": records_day,
            "gb_per_month": gb_month,
            "raw_stored_tb": raw_stored_tb,
            "feature_store_tb": feature_store_tb,
            "feature_cells_per_day": feature_cells_day,
            "predictions_per_month": predictions_month,
            "per_signal": per_signal,
        },
        "lever_note": (
            "Infra (ingest/storage/compute) is priced per GB and is small at list prices; the material "
            "levers are OEM package scope, polling cadence on pull-based APIs, and build effort."
        ),
        "monthly": {
            "ingest": ingest_usd,
            "storage": storage_usd,
            "compute": compute_usd,
            "kafka": kafka_usd,
            "inference": inference_usd,
            "infra_total": infra_month,
            "oem_api_calls": api_usd,
            "oem_marginal": marginal_oem,
            "oem_attributed": attributed_oem,
            "build_amortized": build_month,
            "marginal_total": infra_month + api_usd + marginal_oem + build_month,
            "attributed_total": infra_month + api_usd + attributed_oem + build_month,
        },
        "polling": {
            "max_calls_per_vehicle_day": max_calls_day,
            "api_calls_per_month": api_calls_month,
            "highest_cadence_signals": [
                s.signal_id for s in signals if cadence[s.declared_frequency] == max_calls_day
            ],
        },
        "build_one_off_usd": build_usd,
        "oem_packages_touched": packages,
        "oem_packages_incremental": incremental_pkgs,
        "placeholders": _placeholders(p),
        "price_sheet_as_of": p["as_of"],
    }


def _placeholders(p: dict[str, Any]) -> list[str]:
    out = []
    for section, body in p.items():
        if isinstance(body, dict):
            if body.get("placeholder"):
                out.append(section)
            for k, v in body.items():
                if k.endswith("_placeholder") and v:
                    out.append(f"{section}.{k.removesuffix('_placeholder')}")
    return sorted(out)


def compare_costs(full: dict[str, Any], reduced: dict[str, Any]) -> dict[str, Any]:
    fm, rm = full["monthly"], reduced["monthly"]

    def pct(a: float, b: float) -> float | None:
        return None if a == 0 else (b - a) / a

    return {
        "full_signals": full["n_signals"],
        "reduced_signals": reduced["n_signals"],
        "infra_full": fm["infra_total"],
        "infra_reduced": rm["infra_total"],
        "infra_change_pct": pct(fm["infra_total"], rm["infra_total"]),
        "marginal_full": fm["marginal_total"],
        "marginal_reduced": rm["marginal_total"],
        "marginal_change_pct": pct(fm["marginal_total"], rm["marginal_total"]),
        "attributed_full": fm["attributed_total"],
        "attributed_reduced": rm["attributed_total"],
        "attributed_change_pct": pct(fm["attributed_total"], rm["attributed_total"]),
        "gb_per_month_full": full["volumes"]["gb_per_month"],
        "gb_per_month_reduced": reduced["volumes"]["gb_per_month"],
    }
