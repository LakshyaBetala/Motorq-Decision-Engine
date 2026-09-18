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
