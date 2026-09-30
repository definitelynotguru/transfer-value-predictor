"""Publish one completed run to docs/results/ and render README numeric blocks from it.

Never fits models or recomputes metrics. Two steps:
1. If local artifacts exist, verify they form one complete run (matching run IDs and artifact
   hashes) and copy the publication set into docs/results/.
2. Render every <!-- BEGIN:name --> ... <!-- END:name --> block in README.md from docs/results/.

--check writes nothing and exits 1 if README or docs/results/ is stale. It works from a clean
checkout (no artifacts) by checking README against the committed docs/results/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
from pathlib import Path

from transfer_value.charts import CHARTS

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
DOCS = ROOT / "docs" / "results"
ART = ROOT / "artifacts"
PROCESSED = ROOT / "data" / "processed"

PUBLISH = {
    "metrics.json": ART / "metrics.json",
    "manifest.json": ART / "manifest.json",
    "cv_results.json": ART / "cv_results.json",
    "coefficients.csv": ART / "coefficients.csv",
    "worst_misses.json": ART / "worst_misses.json",
    "funnel.json": PROCESSED / "funnel.json",
    "label_stats.json": PROCESSED / "label_stats.json",
    "figures/predicted_vs_actual.png": ART / "figures" / "predicted_vs_actual.png",
    "figures/residuals_vs_predicted.png": ART / "figures" / "residuals_vs_predicted.png",
    "figures/residual_distribution.png": ART / "figures" / "residual_distribution.png",
    "followup/followup.json": ART / "followup" / "followup.json",
    "charts/manifest.json": ART / "charts" / "manifest.json",
    **{f"charts/{name}": ART / "charts" / name for name in CHARTS},
}


class ReportError(RuntimeError):
    pass


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def collect_artifacts() -> dict[str, bytes] | None:
    if not (ART / "manifest.json").exists():
        return None
    manifest = json.loads((ART / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ReportError("artifacts/manifest.json is not a completed run")
    for rel, digest in manifest["artifact_sha256"].items():
        p = ART / rel
        if not p.exists() or sha(p.read_bytes()) != digest:
            raise ReportError(f"artifacts/{rel} does not match the run manifest")
    rid = manifest["run_id"]
    for name in ("metrics.json", "cv_results.json", "funnel.json", "label_stats.json"):
        other = json.loads(PUBLISH[name].read_text()).get("run_id")
        if other != rid:
            raise ReportError(f"{name} run_id {other} != manifest run_id {rid}")
    fup = json.loads(PUBLISH["followup/followup.json"].read_text())
    if fup["headline_run_id"] != rid:
        raise ReportError(f"follow-up was built on run {fup['headline_run_id']}, not {rid}")
    if fup["headline_metrics_sha256"] != manifest["artifact_sha256"]["metrics.json"]:
        raise ReportError("follow-up was built against different headline metrics; rerun it")
    charts = json.loads(PUBLISH["charts/manifest.json"].read_text())
    if charts["headline_run_id"] != rid:
        raise ReportError(f"charts were drawn from run {charts['headline_run_id']}, not {rid}")
    if charts["followup_sha256"] != sha(PUBLISH["followup/followup.json"].read_bytes()):
        raise ReportError("charts were drawn from a different follow-up; rerun tvp charts")
    for name in CHARTS:
        if sha(PUBLISH[f"charts/{name}"].read_bytes()) != charts["files"].get(name):
            raise ReportError(f"artifacts/charts/{name} does not match the charts manifest")
    return {rel: src.read_bytes() for rel, src in PUBLISH.items()}


def m(eur: float, dp: int = 2) -> str:
    sign = "−" if eur < 0 else ""
    return f"{sign}€{abs(eur) / 1e6:.{dp}f}m"


def table(headers: list[str], rows: list[list], align: str | None = None) -> str:
    align = align or "l" + "r" * (len(headers) - 1)
    sep = ["---:" if a == "r" else "---" for a in align]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def render(docs: dict[str, bytes]) -> dict[str, str]:
    j = {k: json.loads(v) for k, v in docs.items() if k.endswith(".json")}
    met, man, fun, lab, cv = (
        j["metrics.json"],
        j["manifest.json"],
        j["funnel.json"],
        j["label_stats.json"],
        j["cv_results.json"],
    )
    split, sw = met["split"], met["study_window"]
    blocks = {}

    order = [
        "train_median",
        "train_mean",
        "train_median_by_position",
        "linear",
        "ridge",
        "elastic_net",
    ]
    rows = []
    for k in order:
        r = met["methods"][k]
        label = f"**{r['label']}**" if r["selected_by_cv"] else r["label"]
        rows.append(
            [
                label,
                "yes" if r["selected_by_cv"] else "",
                m(r["mae_eur"]),
                m(r["median_ae_eur"]),
                m(r["rmse_eur"]),
                f"{r['log_mae']:.3f}",
                r["rows"],
            ]
        )
    blocks["results"] = (
        table(
            [
                "Method",
                "Selected by CV",
                "Test MAE",
                "Median AE",
                "RMSE",
                "Test log-MAE",
                "Test rows",
            ],
            rows,
            "llrrrrr",
        )
        + f"\n\nSplit: train `transfer_date < {split['test_start']}` "
        f"({split['train']['rows']} transfers, cycles {split['train']['cycles'][0]}–"
        f"{split['train']['cycles'][-1]}); test `>= {split['test_start']}` "
        f"({split['test']['rows']} transfers, {split['test']['unique_players']} players, "
        f"{split['test']['date_min']} to {split['test']['date_max']}, cycles "
        f"{', '.join(map(str, split['test']['cycles']))}). "
        f"Predictions clamped at zero: "
        f"{sum(v['clamped_at_zero'] for v in met['methods'].values())}."
    )

    b = met["bootstrap"]
    labels = {k: met["methods"][k]["label"] for k in met["methods"]}
    rows = [
        [
            labels[k],
            m(c["observed_delta_mae_eur"]),
            f"{m(c['ci_low'])} to {m(c['ci_high'])}",
            c["verdict"],
        ]
        for k, c in b["comparisons"].items()
    ]
    blocks["bootstrap"] = (
        table(
            ["Baseline", "ΔMAE (model − baseline)", f"{int(b['confidence'] * 100)}% CI", "Reading"],
            rows,
            "lrll",
        )
        + f"\n\n{b['replicates']:,} paired replicates, seed {b['seed']}, resampling unit: "
        f"{b['resampling_unit']} ({b['unique_test_players']} unique test players; "
        f"{split['test_players_with_repeat_transfers']} with more than one test transfer). "
        "Negative favors the model. The interval covers holdout sampling noise only; it does "
        "not include retraining or future drift."
    )

    rows = []
    for fam in ("linear", "ridge", "elastic_net"):
        c = met["cv_finalists"][fam]
        params = ", ".join(f"{k}={v}" for k, v in c["params"].items()) or "none"
        rows.append(
            [
                labels[fam],
                params,
                f"{c['mean_log_mae']:.4f}",
                f"{c['std_log_mae']:.4f}",
                " / ".join(f"{x:.3f}" for x in c["fold_log_mae"]),
                "yes" if fam == met["selected_model"]["family"] else "",
            ]
        )
    folds = ", ".join(
        f"cycle {f['validation_cycle']} ({f['train_rows']} train / "
        f"{f['validation_rows']} validation)"
        for f in cv["folds"]
    )
    blocks["cv"] = (
        table(
            ["Family", "Best params", "CV log-MAE", "Fold SD", "Per fold", "Selected"],
            rows,
            "llrrll",
        )
        + f"\n\nFolds: {folds}. Tie-break: {'; '.join(cv['tie_break'])}."
    )

    d = met["residual_diagnostics"]
    drift = d["train_in_sample_mean_log_residual_by_cycle"]
    pos_rows = [
        [p, v["rows"], m(v["mae_eur"]), m(v["median_residual_eur"])]
        for p, v in d["test_by_position"].items()
    ]
    blocks["diagnostics"] = (
        f"- {d['test_share_underpredicted']:.0%} of test transfers were under-predicted. "
        f"Mean residual {m(d['test_mean_residual_eur'], 1)}, median "
        f"{m(d['test_median_residual_eur'], 1)}; mean log residual "
        f"{d['test_mean_log_residual']:+.3f} (predictions about "
        f"{2.718281828 ** d['test_mean_log_residual']:.2f}× too low on the log scale).\n"
        "- Mean in-sample log residual of the selected model by training cycle: "
        + ", ".join(f"{c}: {v:+.2f}" for c, v in drift.items())
        + ". Test cycles: "
        + ", ".join(f"{c}: {v:+.2f}" for c, v in d["test_mean_log_residual_by_cycle"].items())
        + ".\n- Median reported fee by cycle: "
        + ", ".join(f"{c}: {m(v, 1)}" for c, v in d["median_fee_by_cycle_eur"].items())
        + ".\n\n"
        + table(["Position", "Test rows", "MAE", "Median residual"], pos_rows)
    )

    misses = j["worst_misses.json"]
    rows = []
    for i, r in enumerate(misses, 1):
        rows.append(
            [
                i,
                r["name"],
                f"{r['from_club_name']} → {r['to_club_name']}",
                r["transfer_date"][:10],
                m(r["fee_eur"], 1),
                m(r["prediction_eur"], 1),
                ("+" if r["residual_eur"] >= 0 else "") + m(r["residual_eur"], 1),
                f"{int(r['age'])}",
                f"{r['position']}",
                f"{int(r['minutes'])} min, {int(r['goals'])} G, {int(r['assists'])} A "
                f"({r['lookback_season_ids']})",
            ]
        )
    blocks["misses"] = table(
        [
            "#",
            "Player",
            "Move",
            "Date",
            "Reported",
            "Predicted",
            "Residual",
            "Age",
            "Pos",
            "Lookback",
        ],
        rows,
        "rlllrrrrll",
    )

    coefs = list(csv.DictReader(io.StringIO(docs["coefficients.csv"].decode())))
    sel = [c for c in coefs if c["selected"] == "True"]
    rows = []
    for c in sel:
        sd = ""
        if c.get("train_sd"):
            x = float(c["train_sd"])
            sd = f"{x:,.0f}" if x >= 100 else f"{x:.3g}"
        is_icpt = c["kind"] == "intercept"
        rows.append(
            [
                f"`{c['term']}`",
                c["kind"],
                f"{float(c['coef_log1p']):+.3f}",
                "" if is_icpt else f"×{float(c['multiplier_on_1p_fee']):.3f}",
                sd,
            ]
        )
    blocks["coefficients"] = table(
        ["Term", "Kind", "Coefficient (log1p fee)", "exp(coef)", "1 training SD ="],
        rows,
        "llrrr",
    )

    rows = []
    for s in fun["steps"]:
        note = s.get("reason") or s.get("note") or ""
        rows.append(
            [
                s["step"].replace("_", " "),
                f"{s['rows']:,}",
                f"{s['dropped']:,}" if "dropped" in s else "",
                note,
            ]
        )
    rt = lab["round_trip_diagnostic"]
    blocks["funnel"] = (
        table(["Step", "Rows", "Dropped", "Reason / note"], rows, "lrrl")
        + "\n\nPosition source in the final cohort: "
        + ", ".join(f"{k} {v}" for k, v in fun["position_source_counts"].items())
        + f" ({met['position_proxy_pct_total']}% proxy overall; train "
        f"{split['train']['position_proxy_pct']}%, test {split['test']['position_proxy_pct']}%)."
        f" Rows per cycle: "
        + ", ".join(f"{k}: {v}" for k, v in fun["by_cycle"].items())
        + f". Possible loans or buy-backs (a paid move followed within {rt['window_days']} days "
        f"by a zero/unknown-fee move back to the seller): {rt['final_cohort_flagged']} of "
        f"{fun['final_rows']} final rows ({rt['final_cohort_flagged_test']} in test); kept, "
        "disclosed here."
    )

    blocks.update(render_followup(j["followup/followup.json"]))

    env = man["environment"]
    deps = ", ".join(f"{k} {v}" for k, v in env["dependencies"].items())
    src = table(
        ["File", "SHA-256"],
        [[f"`{k}`", f"`{v}`"] for k, v in sorted(man["source_hashes"].items())],
        "ll",
    )
    blocks["reproduce"] = (
        f"- Run ID `{man['run_id']}`; code revision `{man['code_revision']['commit']}`"
        f"{' (dirty)' if man['code_revision']['dirty'] else ''}\n"
        f"- Python {env['python']}; {deps}\n"
        f"- `uv.lock` sha256 `{env['lockfile_sha256']}`\n"
        f"- Config sha256 `{man['config_hash']}`; seed {man['seed']}; study window "
        f"{sw['transfer_start']} to {sw['transfer_end_exclusive']} (exclusive), T "
        f"{sw['test_start']}\n\n" + src
    )
    return blocks


def _ci(c: dict) -> str:
    return f"{m(c['ci_low'])} to {m(c['ci_high'])}"


def render_followup(f: dict) -> dict[str, str]:
    blocks = {}
    rows = []
    for name in f["variant_order"]:
        v = f["variants"][name]
        sel = v["selected_by_cv"]
        rows.append(
            [
                f"**{name}**" if sel else name,
                v["model"],
                f"{v['cv_mean_log_mae']:.4f}",
                " / ".join(f"{x:.3f}" for x in v["cv_fold_log_mae"]),
                "yes" if sel else "",
                m(v["test"]["mae_eur"]),
                f"{v['test']['log_mae']:.3f}",
                f"{v['test_mean_log_residual']:+.3f}",
            ]
        )
    c = f["followup_vs_headline"]["comparisons"]["headline"]
    b = f["followup_vs_headline"]
    coef = {r["term"]: r for r in f["selected_coefficients"]}
    time_terms = [t for t in ("cycle_trend", "league_price_level") if t in coef]
    time_note = "".join(
        f" `{t}` coefficient {coef[t]['coef_log1p']:+.3f} per training SD "
        f"(SD {coef[t]['train_sd']:.3g})."
        for t in time_terms
    )
    levels = f["price_level"]["by_cycle_median_level_eur"]
    blocks["followup"] = (
        table(
            [
                "Variant",
                "Best model",
                "CV log-MAE",
                "Per fold",
                "Selected by CV",
                "Test MAE",
                "Test log-MAE",
                "Mean test log residual",
            ],
            rows,
            "llrllrrr",
        )
        + f"\n\nSelected by CV: **{f['selected_variant']}** ({f['selected_model']}).{time_note} "
        f"Paired bootstrap, follow-up minus headline test MAE: {m(c['observed_delta_mae_eur'])} "
        f"(95% CI {_ci(c)}; {b['replicates']:,} replicates, player clusters). "
        "Test columns for non-selected variants are descriptive only.\n\n"
        "Median league price level by cycle: "
        + ", ".join(f"{k}: {m(v, 1)}" for k, v in levels.items())
        + "."
    )

    w = f["window_diagnostic"]
    rows = []
    for name, label in (
        ("off_season", "Off-season (no season in progress)"),
        ("in_season", "In-season"),
    ):
        t, tr = w["test"][name], w["train_in_sample"][name]
        rows.append(
            [
                label,
                f"{tr['rows']} / {t['rows']}",
                f"{t['mean_season_equivalents']:.2f}",
                f"{tr['mean_raw_minutes']:,.0f}",
                f"{tr['mean_log_residual__headline']:+.3f}",
                f"{tr['mean_log_residual__followup']:+.3f}",
                f"{t['mean_log_residual__headline']:+.3f}",
                f"{t['mean_log_residual__followup']:+.3f}",
            ]
        )
    blocks["window"] = table(
        [
            "Window",
            "Train / test rows",
            "Test lookback seasons",
            "Train raw minutes",
            "Train residual (headline)",
            "Train residual (follow-up)",
            "Test residual (headline)",
            "Test residual (follow-up)",
        ],
        rows,
        "lrrrrrrr",
    ) + (
        "\n\nResiduals are mean `log1p(fee) − prediction`; train columns are in-sample. "
        "Lookback seasons count an ongoing season by the share of its games played before D."
    )

    mv = f["market_value_comparator"]
    ctx = f["context_round"]
    labels = {
        "train_median": "Train median fee",
        "headline": "Headline model (LinearRegression)",
        "followup": f"Follow-up model ({f['selected_variant']})",
        "enriched": f"Enriched follow-up ({ctx['selected_variant']})",
        "market_value": "Transfermarkt market value (comparator)",
    }
    rows = [
        [
            labels[k],
            m(mv["methods"][k]["mae_eur"]),
            m(mv["methods"][k]["median_ae_eur"]),
            f"{mv['methods'][k]['log_mae']:.3f}",
        ]
        for k in ("train_median", "headline", "followup", "enriched", "market_value")
    ]
    boots = "; ".join(
        f"{labels[ref].split(' (')[0].lower()} minus market value "
        f"{m(bb['comparisons']['market_value']['observed_delta_mae_eur'])} "
        f"(95% CI {_ci(bb['comparisons']['market_value'])})"
        for ref, bb in mv["bootstrap"].items()
    )
    blocks["market_value"] = (
        table(["Method", "Test MAE", "Median AE", "Test log-MAE"], rows, "lrrr")
        + f"\n\nMatched test rows: {mv['matched_rows']} of {mv['test_rows']} (rule: {mv['rule']}; "
        f"median valuation age {mv['median_valuation_age_days']:.0f} days). "
        f"Reported fee above market value in {mv['share_fee_above_market_value']:.0%} of rows. "
        f"Paired bootstrap: {boots}."
    )
    blocks.update(render_context(ctx, f["variants"][f["selected_variant"]]))
    blocks.update(render_conformal(f["conformal"], f["selected_variant"], ctx["selected_variant"]))
    blocks["boosting"] = render_boosting(f["boosting"], f["variants"]["headline"])
    return blocks


def _pct(x: float) -> str:
    return f"{x:.0%}"


def render_context(c: dict, first_round: dict) -> dict[str, str]:
    rows = []
    for name in c["variant_order"]:
        v = c["variants"][name]
        sel = v["selected_by_cv"]
        rows.append(
            [
                f"**{name}**" if sel else name,
                v["model"],
                f"{v['cv_mean_log_mae']:.4f}",
                " / ".join(f"{x:.3f}" for x in v["cv_fold_log_mae"]),
                "yes" if sel else "",
                m(v["test"]["mae_eur"]),
                f"{v['test']['log_mae']:.3f}",
                f"{v['test_mean_log_residual']:+.3f}",
            ]
        )
    b = c["vs_headline_and_followup"]["comparisons"]
    abl = "; ".join(
        f"without {g.replace('_', ' ')} {a['cv_mean_log_mae']:.4f}"
        for g, a in c["ablation_cv"].items()
    )
    coef = {r["term"]: r for r in c["selected_coefficients"]}
    coefs = ", ".join(
        f"`{t}` {coef[t]['coef_log1p']:+.3f}"
        for t in ("team_points_per_game", "europe_minutes", "other_league_minutes")
        if t in coef
    )
    miss_rows = [
        [
            r["name"],
            m(r["fee_eur"], 1),
            m(r["headline_eur"], 1),
            m(r["followup_eur"], 1),
            m(r["enriched_eur"], 1),
            f"{r['europe_minutes']:,.0f}",
            f"{r['other_league_minutes']:,.0f} ({r['other_league_goals']:.0f} G)",
            f"{r['team_points_per_game']:.2f}",
        ]
        for r in c["headline_worst_misses"]
    ]
    sel = c["selected_variant"]
    text = (
        table(
            [
                "Variant",
                "Best model",
                "CV log-MAE",
                "Per fold",
                "Selected by CV",
                "Test MAE",
                "Test log-MAE",
                "Mean test log residual",
            ],
            rows,
            "llrllrrr",
        )
        + f"\n\nSelected by CV: **{sel}** ({c['selected_model']}), CV log-MAE "
        f"{c['variants'][sel]['cv_mean_log_mae']:.4f} against "
        f"{first_round['cv_mean_log_mae']:.4f} for the first-round follow-up. Coefficients per "
        f"training SD: {coefs}. CV with one context group removed from the selected variant: "
        f"{abl}. Paired bootstrap on test MAE, enriched minus headline "
        f"{m(b['headline']['observed_delta_mae_eur'])} (95% CI {_ci(b['headline'])}); enriched "
        f"minus first-round follow-up {m(b['followup']['observed_delta_mae_eur'])} (95% CI "
        f"{_ci(b['followup'])}). Rows with any European minutes: train "
        f"{_pct(c['share_with_europe_minutes']['train'])}, test "
        f"{_pct(c['share_with_europe_minutes']['test'])}; with other-league minutes: train "
        f"{_pct(c['share_with_other_league_minutes']['train'])}, test "
        f"{_pct(c['share_with_other_league_minutes']['test'])}.\n\n"
        "The headline's five worst misses under each model:\n\n"
        + table(
            [
                "Player",
                "Reported",
                "Headline",
                "Follow-up",
                "Enriched",
                "European min",
                "Other-league min",
                "Team pts/game",
            ],
            miss_rows,
            "lrrrrrrr",
        )
    )
    return {"context": text}


def render_conformal(c: dict, followup_name: str, enriched_name: str) -> dict[str, str]:
    labels = {
        "headline": "**Headline (LinearRegression)**",
        "followup": f"Follow-up ({followup_name})",
        "enriched": f"Enriched follow-up ({enriched_name})",
    }
    rows = []
    for key in ("headline", "followup", "enriched"):
        for lv, s in c[key]["levels"].items():
            t = s["test"]
            rows.append(
                [
                    labels[key],
                    _pct(float(lv)),
                    s["calibration_rows"],
                    f"×/÷ {s['factor']:.2f}",
                    _pct(t["coverage"]),
                    _pct(t["share_above_upper"]),
                    _pct(t["share_below_lower"]),
                    f"{m(s['median_lower_eur'], 1)} to {m(s['median_upper_eur'], 1)}",
                ]
            )
    lv = f"{c['predict_level']:.2f}"
    h = c["headline"]["levels"][lv]

    def groups(d: dict) -> str:
        return ", ".join(f"{k} {_pct(v['coverage'])}" for k, v in d.items())

    misses = "; ".join(
        f"{r['name']} {m(r['fee_eur'], 0)} in {m(r['lower_eur'], 1)} to {m(r['upper_eur'], 0)} "
        f"({'covered' if r['covered'] else 'missed'})"
        for r in c["headline_worst_misses"]
    )
    text = (
        table(
            [
                "Model",
                "Nominal",
                "Calibration rows",
                "Interval (on 1 + fee)",
                "Test coverage",
                "Above upper",
                "Below lower",
                "Median test interval",
            ],
            rows,
            "lrrrrrrl",
        )
        + f"\n\nHeadline at {_pct(c['predict_level'])}, test coverage by cycle: "
        f"{groups(h['test_by_transfer_cycle'])}; by position: {groups(h['test_by_position'])}; "
        f"by window: {groups(h['test_by_window'])}. Mean calibration residual (log scale): "
        + ", ".join(
            f"{k} {c[k]['calibration_mean_residual']:+.3f}"
            for k in ("headline", "followup", "enriched")
        )
        + f". Headline worst five at {_pct(c['predict_level'])}: {misses}."
    )
    return {"conformal": text}


def render_boosting(b: dict, head: dict) -> str:
    g, gm, hc = b["gbm"], b["gbm_monotone"], b["headline_compression"]

    def reach(x: dict) -> str:
        return (
            f"{m(x['prediction_for_largest_fee_eur'], 1)} for the "
            f"{m(x['largest_fee_eur'], 0)} transfer, top-decile log residual "
            f"{x['top_decile_mean_log_residual']:+.2f}, slope {x['calibration_slope']:.2f}"
        )

    return (
        "**Gradient boosting did not fix the compression.** Same headline inputs, same folds, "
        f"a fixed {len(b['grid'])}-point grid (`HistGradientBoostingRegressor`, learning rate "
        f"{b['learning_rate']}). CV log-MAE: unconstrained {g['cv_mean_log_mae']:.4f}, "
        f"monotonic {gm['cv_mean_log_mae']:.4f} (more goals, assists, minutes and per-90 output "
        f"can only raise the prediction, more age can only lower it), against "
        f"{head['cv_mean_log_mae']:.4f} for the headline linear model, so CV would never have "
        f"picked either. On the test set (descriptive only): test MAE {m(g['test']['mae_eur'])} "
        f"and {m(gm['test']['mae_eur'])} against {m(head['test']['mae_eur'])}. The top of the "
        f"market gets no closer: linear {reach(hc)}; boosting {reach(g['compression'])}; "
        f"monotonic {reach(gm['compression'])}. Trees cannot predict above the leaf averages "
        "they were trained on, and a slope of actual on predicted log fee near 1 says the "
        "linear predictions are not too tightly bunched for their inputs. The elite misses are "
        "a missing-information problem, not a functional-form one."
    )


def apply_blocks(text: str, blocks: dict[str, str]) -> str:
    for name, body in blocks.items():
        pat = re.compile(rf"(<!-- BEGIN:{name} -->\n)(?:.*?\n)?(<!-- END:{name} -->)", re.S)
        if not pat.search(text):
            raise ReportError(f"README has no block {name}")
        text = pat.sub(lambda mt, b=body: mt.group(1) + b + "\n" + mt.group(2), text)
    return text


def read_docs() -> dict[str, bytes]:
    missing = [rel for rel in PUBLISH if not (DOCS / rel).exists()]
    if missing:
        raise ReportError(f"docs/results missing {missing}; run without --check first")
    return {rel: (DOCS / rel).read_bytes() for rel in PUBLISH}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml", help="accepted for CLI symmetry")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    try:
        fresh = collect_artifacts()
        stale = []
        if fresh is not None:
            for rel, data in fresh.items():
                p = DOCS / rel
                if not p.exists() or p.read_bytes() != data:
                    stale.append(f"docs/results/{rel}")
                    if not args.check:
                        p.parent.mkdir(parents=True, exist_ok=True)
                        p.write_bytes(data)
        docs = fresh if (fresh is not None and not args.check) else read_docs()
        text = README.read_text()
        new = apply_blocks(text, render(docs))
        if new != text:
            stale.append("README.md")
            if not args.check:
                README.write_text(new)
    except ReportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.check:
        if stale:
            print("stale: " + ", ".join(stale), file=sys.stderr)
            return 1
        print("report up to date")
    else:
        print("updated: " + (", ".join(stale) if stale else "nothing (already current)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
