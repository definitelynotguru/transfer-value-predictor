"""tvp: ingest | feasibility | build-features | train | evaluate | predict | pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    help="Methodology study: how wrong is a linear fee model for PL-active players' "
    "reported transfer fees?",
    no_args_is_help=True,
    add_completion=False,
)

ConfigOpt = Annotated[Path, typer.Option("--config", help="Path to config.yaml")]
InputDirOpt = Annotated[
    Path | None,
    typer.Option("--input-dir", help="Manually downloaded snapshot directory to import"),
]


def _cfg(path: Path):
    from transfer_value.config import load_config

    return load_config(path)


def _fail(exc: Exception) -> None:
    typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
    raise typer.Exit(1)


@app.command()
def ingest(config: ConfigOpt = Path("config.yaml"), input_dir: InputDirOpt = None) -> None:
    """Verify raw hashes, normalize tables, write typed interim Parquet."""
    from transfer_value.ingest import run_ingest

    try:
        m = run_ingest(_cfg(config), input_dir)
    except Exception as exc:
        _fail(exc)
    typer.echo(f"ingest ok: {m['row_counts']}")


@app.command()
def feasibility(config: ConfigOpt = Path("config.yaml"), input_dir: InputDirOpt = None) -> None:
    """Schema capability report + exact count funnel; nonzero if any gate fails."""
    from transfer_value.feasibility import run_feasibility
    from transfer_value.ingest import import_manual_snapshot

    try:
        cfg = _cfg(config)
        if input_dir is not None:
            import_manual_snapshot(cfg, input_dir)
        r = run_feasibility(cfg)
    except Exception as exc:
        _fail(exc)
    for name, g in r.get("gates", {}).items():
        extra = g.get("actual", g.get("error", ""))
        typer.echo(f"  {'PASS' if g['passed'] else 'FAIL'} {name} {extra}")
    if not r["passed"]:
        _fail(
            RuntimeError(
                f"feasibility failed ({r.get('failure', 'gates')}); see "
                "data/processed/capability_report.json"
            )
        )
    typer.echo("feasibility ok")


@app.command("build-features")
def build_features(config: ConfigOpt = Path("config.yaml")) -> None:
    """Labels, as-of features, funnel, and label stats."""
    from transfer_value.dataset import run_build_features

    try:
        m = run_build_features(_cfg(config))
    except Exception as exc:
        _fail(exc)
    typer.echo(f"build-features ok: {m['rows']} rows, run {m['run_id']}")


@app.command()
def train(config: ConfigOpt = Path("config.yaml")) -> None:
    """Chronological CV selection; refit each family's finalist on the train window."""
    from transfer_value.model import run_train

    try:
        cv = run_train(_cfg(config))
    except Exception as exc:
        _fail(exc)
    typer.echo(f"train ok: selected {cv['selected']}")


@app.command()
def evaluate(config: ConfigOpt = Path("config.yaml")) -> None:
    """Score models and baselines once on the time holdout."""
    from transfer_value.evaluate import run_evaluate

    try:
        r = run_evaluate(_cfg(config))
    except Exception as exc:
        _fail(exc)
    for m in r["methods"].values():
        star = "*" if m["selected_by_cv"] else " "
        typer.echo(
            f" {star} {m['label']:<30} MAE €{m['mae_eur'] / 1e6:6.2f}m  log-MAE {m['log_mae']:.3f}"
        )
    typer.echo("evaluate ok (* = selected by CV)")


@app.command()
def predict(
    player: Annotated[str | None, typer.Option("--player", help="Exact player name")] = None,
    player_id: Annotated[str | None, typer.Option("--player-id")] = None,
    date: Annotated[str | None, typer.Option("--date", help="As-of ISO date")] = None,
    config: ConfigOpt = Path("config.yaml"),
) -> None:
    """Hypothetical reported-fee estimate for one player (demo only)."""
    from transfer_value.predict import run_predict

    try:
        r = run_predict(_cfg(config), player, player_id, date)
    except Exception as exc:
        _fail(exc)
    typer.echo(
        f"{r['player']} (id {r['player_id']}) as of {r['as_of']}\n"
        f"  position {r['position']} ({r['position_source']}), age {r['age']}\n"
        f"  lookback seasons {r['lookback_seasons']}: {r['lookback_minutes']} min, "
        f"{r['goals']} G, {r['assists']} A\n"
        f"  hypothetical reported fee: €{r['predicted_reported_fee_eur'] / 1e6:.1f}m\n"
        + (
            f"  {r['interval_level']:.0%} training-set conformal interval: "
            f"€{r['interval_lower_eur'] / 1e6:.1f}m to €{r['interval_upper_eur'] / 1e6:.1f}m\n"
            if r.get("interval_level") is not None
            else f"  interval: unavailable ({r['interval_status']})\n"
        )
        + f"  model {r['model']}, run {r['run_id']}\n  {r['note']}"
    )


@app.command()
def followup(
    config: Annotated[Path, typer.Option("--config", help="Path to followup.yaml")] = Path(
        "followup.yaml"
    ),
) -> None:
    """Post-holdout follow-up (exposure, time terms, market-value comparator). Not a headline."""
    from transfer_value.followup import load_followup_config, run_followup

    try:
        r = run_followup(load_followup_config(config))
    except Exception as exc:
        _fail(exc)
    for name, v in r["variants"].items():
        star = "*" if v["selected_by_cv"] else " "
        typer.echo(
            f" {star} {name:<24} CV log-MAE {v['cv_mean_log_mae']:.4f}  "
            f"test MAE €{v['test']['mae_eur'] / 1e6:6.2f}m"
        )
    for name, v in r["context_round"]["variants"].items():
        star = "*" if v["selected_by_cv"] else " "
        typer.echo(
            f" {star} {name:<24} CV log-MAE {v['cv_mean_log_mae']:.4f}  "
            f"test MAE €{v['test']['mae_eur'] / 1e6:6.2f}m"
        )
    mv = r["market_value_comparator"]
    typer.echo(
        f"market value matched {mv['matched_rows']}/{mv['test_rows']} test rows: MAE "
        f"€{mv['methods']['market_value']['mae_eur'] / 1e6:.2f}m vs headline "
        f"€{mv['methods']['headline']['mae_eur'] / 1e6:.2f}m"
    )
    conf = r["conformal"]
    for key in ("headline", "followup", "enriched"):
        for lv, s in conf[key]["levels"].items():
            typer.echo(
                f"conformal {key:<9} {float(lv):.0%}: ×/÷{s['factor']:.1f}, "
                f"test coverage {s['test']['coverage']:.0%}"
            )
    typer.echo("followup ok (* = selected by CV; not a headline result)")


@app.command()
def pipeline(config: ConfigOpt = Path("config.yaml")) -> None:
    """feasibility → ingest → build-features → train → evaluate. Never downloads."""
    feasibility(config, None)
    ingest(config, None)
    build_features(config)
    train(config)
    evaluate(config)
    typer.echo("pipeline ok")


if __name__ == "__main__":
    app()
