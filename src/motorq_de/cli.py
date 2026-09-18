"""`mde` — command line for the Decision Engine."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich import print as rprint
from rich.table import Table

app = typer.Typer(no_args_is_help=True, help="Motorq Decision Engine")
world_app = typer.Typer(help="Synthetic world")
app.add_typer(world_app, name="world")

DATA_ROOT = Path("data/synthetic")


@world_app.command("generate")
def world_generate(
    profile: str = typer.Option("default", help="small | default | full"),
    seed: int = typer.Option(42),
    out: Path = typer.Option(DATA_ROOT),
):
    from motorq_de.world.generator import WorldGenerator

    g = WorldGenerator(profile=profile, seed=seed)
    rprint(f"[bold]dataset_hash[/] {g.dataset_hash}  profile={profile} seed={seed}")
    path = g.generate(out)
    truth = json.loads((path / "truth.json").read_text(encoding="utf-8"))
    t = Table(title="achieved rates")
    t.add_column("metric"), t.add_column("value")
    for k, v in truth["achieved_rates"].items():
        t.add_row(k, "-" if v is None else f"{v:.4f}")
    rprint(t)
    rprint(f"[green]written[/] {path}")


@world_app.command("inspect")
def world_inspect(dataset: Path | None = typer.Option(None, help="dataset dir; default latest")):
    from motorq_de.data.synthetic import SyntheticSource, latest_dataset

    src = SyntheticSource(dataset or latest_dataset(DATA_ROOT))
    veh = src.vehicles()
    rprint(f"[bold]{src.dataset_hash}[/]  vehicles={len(veh)}  range={src.date_range()}")
    t = Table(title="fleet by OEM")
    (
        t.add_column("oem"),
        t.add_column("n"),
        t.add_column("ev share"),
        t.add_column("brake_pad_wear coverage"),
    )
    cov = src.coverage_by_oem("brake_pad_wear_pct")
    for oem, g in veh.groupby("oem"):
        c = cov[oem]
        t.add_row(oem, str(len(g)), f"{(g.powertrain == 'ev').mean():.2f}", f"{c.coverage:.2f}")
    rprint(t)
    ev = src.events()
    rprint(ev.event_type.value_counts().to_dict())


if __name__ == "__main__":
    app()


run_app = typer.Typer(help="Decision runs")
app.add_typer(run_app, name="run")

EXAMPLES = {
    "brake": {
        "capability_name": "brake_service_7d",
        "target_event": "brake_service_event",
        "horizon_days": 7,
        "value": {
            # share of brake services that would have been unplanned downtime (value-bearing);
            # $/event from public fleet downtime studies ($3.5k-6.5k/incident). See README sources.
            "value_bearing_fraction": {"low": 0.05, "base": 0.10, "high": 0.20},
            "preventable_fraction": {"low": 0.3, "base": 0.4, "high": 0.5},
            "usd_per_avoided_event": {"low": 1500, "base": 3500, "high": 6500},
            "fleet_size": {"low": 5000, "base": 10000, "high": 20000},
            "source_note": "downtime $448-760/day and $3.5k-6.5k/incident (oxmaint, fleetrabbit); predictive maintenance cuts unplanned downtime 30-50%",
        },
    },
    "theft": {
        "capability_name": "theft_risk_30d",
        "target_event": "theft_event",
        "horizon_days": 30,
        "value": {
            "value_bearing_fraction": {"low": 0.6, "base": 0.8, "high": 1.0},
            "preventable_fraction": {"low": 0.1, "base": 0.2, "high": 0.3},
            "usd_per_avoided_event": {"low": 8000, "base": 15000, "high": 30000},
            "fleet_size": {"low": 5000, "base": 10000, "high": 20000},
            "source_note": "NICB 2024: 250/100k residents; ~85% recovered; loss per unrecovered/damaged vehicle is a human estimate",
        },
    },
    "battery": {
        "capability_name": "battery_service_14d",
        "target_event": "battery_degradation_event",
        "horizon_days": 14,
        "value": {
            "value_bearing_fraction": {"low": 0.3, "base": 0.5, "high": 0.8},
            "preventable_fraction": {"low": 0.2, "base": 0.3, "high": 0.5},
            "usd_per_avoided_event": {"low": 2000, "base": 5000, "high": 12000},
            "fleet_size": {"low": 500, "base": 1000, "high": 3000},
            "source_note": "Geotab 2026 degradation rates; $/event is a human estimate of an unplanned EV service",
        },
    },
}


@run_app.command("headless")
def run_headless(
    example: str = typer.Option("brake", help="brake | theft | battery, or path to a spec JSON"),
    dataset: Path | None = typer.Option(None, help="dataset dir; default latest"),
    out: Path | None = typer.Option(None, help="write brief markdown here"),
):
    """Run the default plan with no LLM and print the cited brief."""
    from motorq_de.agent.runner import Runner
    from motorq_de.data.synthetic import SyntheticSource, latest_dataset
    from motorq_de.ledger.store import Ledger
    from motorq_de.schemas import ProblemSpec

    if example in EXAMPLES:
        spec = ProblemSpec.model_validate(EXAMPLES[example])
    else:
        spec = ProblemSpec.model_validate_json(Path(example).read_text(encoding="utf-8"))
    src = SyntheticSource(dataset or latest_dataset(DATA_ROOT))
    rprint(f"[bold]dataset[/] {src.dataset_hash}  [bold]spec[/] {spec.capability_name}")
    runner = Runner(src, Ledger(), progress=lambda m: rprint(f"  [dim]{m}[/]"))
    res = runner.run(spec)
    rprint(f"\n[bold green]{res.verdict.decision}[/]  run {res.run_id}  replans={res.replans}\n")
    if out:
        out.write_text(res.brief_md, encoding="utf-8")
        rprint(f"[green]brief written[/] {out}")
    else:
        print(res.brief_md)


@run_app.command("list")
def run_list():
    from motorq_de.ledger.store import Ledger

    t = Table(title="runs")
    for c in ("run_id", "capability", "target", "status", "decision", "created"):
        t.add_column(c)
    for r in Ledger().list_runs():
        t.add_row(
            r["run_id"],
            str(r["capability_name"]),
            str(r["target_event"]),
            r["status"],
            str(r["decision"]),
            r["created_at"][:19],
        )
    rprint(t)


@run_app.command("evidence")
def run_evidence(evidence_id: str):
    from motorq_de.ledger.store import Ledger

    e = Ledger().get_evidence(evidence_id)
    if not e:
        rprint("[red]not found[/]")
        raise typer.Exit(1)
    print(json.dumps(e, indent=2, default=str))
