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

from motorq_de.agent.service import EXAMPLE_SPECS as EXAMPLES  # noqa: E402


@run_app.command("headless")
def run_headless(
    example: str = typer.Option("brake", help="brake | theft | battery, or path to a spec JSON"),
    dataset: Path | None = typer.Option(None, help="synthetic dataset dir; default latest"),
    snowflake: Path | None = typer.Option(
        None, help="Snowflake adapter YAML (uses SNOWFLAKE_* env credentials)"
    ),
    out: Path | None = typer.Option(None, help="write brief markdown here"),
):
    """Run the default plan with no LLM and print the cited brief."""
    from motorq_de.agent.service import Service, snowflake_source
    from motorq_de.schemas import ProblemSpec

    if example in EXAMPLES:
        spec = ProblemSpec.model_validate(EXAMPLES[example])
    else:
        spec = ProblemSpec.model_validate_json(Path(example).read_text(encoding="utf-8"))
    svc = Service(source=snowflake_source(snowflake) if snowflake else None, dataset=dataset)
    rprint(f"[bold]dataset[/] {svc.source.dataset_hash}  [bold]spec[/] {spec.capability_name}")
    res = svc.run(spec, progress=lambda m: rprint(f"  [dim]{m}[/]"))
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


@run_app.command("ask")
def run_ask(
    text: str = typer.Argument(..., help="free-text capability request"),
    dataset: Path | None = typer.Option(None),
    out: Path | None = typer.Option(None),
):
    """Parse a free-text request with the LLM, run the pipeline, add a cited narrative."""
    from motorq_de.agent.service import Service

    svc = Service(dataset=dataset)
    if not svc.llm_available():
        rprint(
            "[red]LLM interface unavailable[/] - set ANTHROPIC_API_KEY (or use `mde run headless`)"
        )
        raise typer.Exit(2)
    spec, notes = svc.spec_from_text(text)
    rprint(f"[bold]spec[/] {spec.model_dump_json(indent=1)}\n[dim]{notes}[/]")
    res = svc.run(
        spec, request_text=text, use_llm=True, progress=lambda m: rprint(f"  [dim]{m}[/]")
    )
    rprint(f"\n[bold green]{res.verdict.decision}[/]  run {res.run_id}\n")
    if out:
        out.write_text(res.brief_md, encoding="utf-8")
        rprint(f"[green]brief written[/] {out}")
    else:
        print(res.brief_md)


@app.command("ask")
def ask(run_id: str, question: str):
    """Ask a question about a finished run; the answer cites evidence ids."""
    from motorq_de.agent.service import Service

    svc = Service()
    if not svc.llm_available():
        rprint("[red]LLM interface unavailable[/] - set ANTHROPIC_API_KEY")
        raise typer.Exit(2)
    text, cited = svc.ask(run_id, question)
    print(text)
    rprint(f"[dim]cites: {', '.join(cited) or 'none'}[/]")


@app.command("serve")
def serve(host: str = "127.0.0.1", port: int = 8000):
    """Start the HTTP API."""
    import uvicorn

    uvicorn.run("motorq_de.api:app", host=host, port=port, reload=False)


@run_app.command("replay")
def run_replay(run_id: str, dataset: Path | None = typer.Option(None)):
    """Re-run a study from its stored spec and diff every evidence record."""
    from motorq_de.agent.service import Service

    svc = Service(dataset=dataset)
    res = svc.replay(run_id, progress=lambda m: rprint(f"  [dim]{m}[/]"))
    d = res.diff or {}
    rprint(
        f"\n[bold]{'IDENTICAL' if d.get('identical') else 'DIFFERS'}[/]  replay {res.run_id} of {run_id}"
    )
    t = Table(title="evidence diff")
    t.add_column("name"), t.add_column("status"), t.add_column("first differences")
    for r in d.get("records", []):
        t.add_row(r["name"], r["status"], json.dumps(r.get("paths", []))[:120])
    rprint(t)
    raise typer.Exit(0 if d.get("identical") else 1)


@run_app.command("whatif")
def run_whatif(
    run_id: str,
    value_json: str = typer.Option(None, help="ValueAssumptions JSON"),
    prices_json: str = typer.Option(None, help="price sheet overrides JSON"),
    out: Path | None = typer.Option(None),
):
    """Re-run economics + policy on a finished study with new human inputs (seconds)."""
    from motorq_de.agent.service import Service

    svc = Service()
    res = svc.whatif(
        run_id,
        json.loads(value_json) if value_json else None,
        json.loads(prices_json) if prices_json else None,
    )
    rprint(f"[bold green]{res.verdict.decision}[/]  what-if {res.run_id} of {run_id}")
    if out:
        out.write_text(res.brief_md, encoding="utf-8")
    else:
        print(res.brief_md)


@app.command("portfolio")
def portfolio_cmd(json_out: bool = typer.Option(False, "--json")):
    """All capabilities side by side, plus the signals no viable capability needs."""
    from motorq_de.agent.service import Service

    p = Service().portfolio()
    if json_out:
        print(json.dumps(p, indent=2, default=str))
        return
    t = Table(title="capabilities (latest study each)")
    for c in (
        "capability",
        "target",
        "decision",
        "P(ROI>0)",
        "net p50/yr",
        "coverage",
        "signals",
        "cost/mo",
        "dominant input",
    ):
        t.add_column(c)
    for r in p["capabilities"]:
        t.add_row(
            r["capability_name"],
            str(r["target_event"]),
            str(r["decision"]),
            "-" if r["p_roi_positive"] is None else f"{r['p_roi_positive']:.2f}",
            "-" if r["net_value_p50"] is None else f"${r['net_value_p50']:,.0f}",
            "-" if r["coverage"] is None else f"{r['coverage']:.2f}",
            str(r["n_sufficient_signals"]),
            "-"
            if r["run_cost_marginal_month"] is None
            else f"${r['run_cost_marginal_month']:,.0f}",
            str(r["dominant_input"]),
        )
    rprint(t)
    c = p["cogs"]
    if c:
        rprint(
            f"[bold]COGS[/] {c['n_required_by_viable_capabilities']} of {c['n_catalog']} catalog signals are required by a viable capability; {len(c['unused_signals'])} are not. {c['cadence_lever']}."
        )
        for cat, sigs in c["unused_by_category"].items():
            rprint(f"  [dim]{cat}[/]: {', '.join(sigs)}")
