"""Report charts built from the verified headline run and the follow-up outputs.

Reads artifacts only; fits nothing. The headline run's own evaluation figures stay as they
were published, so its manifest is never rewritten. Output is deterministic PNG.
"""

from __future__ import annotations

import csv
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

from transfer_value.io import read_json, read_parquet, sha256_file, write_json  # noqa: E402

INK = "#1f2933"
MUTED = "#6b7785"
FAINT = "#c9d0d8"
GRID = "#eceff3"
BLUE = "#2f6db5"
RED = "#d1495b"
GREEN = "#2e9c6a"
GOLD = "#e3a33b"
GREY = "#9aa5b1"
DARK_GREEN = "#1d6b48"
NAVY = "#1b3f6b"
POSITION_COLORS = {"GK": "#8d99a6", "DF": BLUE, "MF": GREEN, "FW": RED}
POSITION_NAMES = {"GK": "Goalkeeper", "DF": "Defender", "MF": "Midfielder", "FW": "Forward"}

CHARTS = (
    "01_predicted_vs_reported.png",
    "02_residual_structure.png",
    "03_fee_drift.png",
    "04_coefficients.png",
    "05_model_comparison.png",
    "06_interval_coverage.png",
    "07_biggest_fees_intervals.png",
    "08_worst_misses_by_model.png",
)

FRIENDLY = {
    "goals": "Goals",
    "assists": "Assists",
    "minutes": "Minutes",
    "appearances": "Appearances",
    "goals_per90": "Goals per 90",
    "assists_per90": "Assists per 90",
    "age": "Age",
    "goals_per_season": "Goals per season",
    "assists_per_season": "Assists per season",
    "minutes_share": "Share of available minutes",
    "appearance_share": "Share of available matches",
    "league_price_level": "League price level",
    "europe_minutes": "European cup minutes",
    "other_league_minutes": "Other-league minutes",
    "other_league_goals": "Other-league goals",
    "team_points_per_game": "Team points per game",
}
FOLLOWUP_ONLY_INPUTS = {
    "league_price_level",
    "europe_minutes",
    "other_league_minutes",
    "other_league_goals",
    "team_points_per_game",
    "goals_per_season",
    "assists_per_season",
    "minutes_share",
    "appearance_share",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": FAINT,
            "axes.labelcolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlecolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK,
            "ytick.labelcolor": INK,
            "legend.frameon": False,
            "legend.fontsize": 9,
        }
    )


def _eur(v: float, _=None) -> str:
    if v >= 1e6:
        x = v / 1e6
        return f"€{x:.0f}m" if x >= 10 or x == int(x) else f"€{x:.1f}m"
    if v >= 1e3:
        return f"€{v / 1e3:.0f}k"
    return f"€{v:.0f}"


EUR = FuncFormatter(_eur)
LOG_TICKS = [3e5, 1e6, 2e6, 5e6, 1e7, 2e7, 5e7, 1e8, 2e8, 5e8]


def _log_eur_axis(axis, lo: float, hi: float) -> None:
    axis.set_major_locator(FixedLocator([t for t in LOG_TICKS if lo <= t <= hi]))
    axis.set_minor_locator(NullLocator())
    axis.set_major_formatter(EUR)


def _figure(
    w: float,
    h: float,
    title: str,
    subtitle: str,
    footer: str,
    top: float = 0.80,
    left: float = 0.08,
    bottom: float = 0.13,
):
    fig = plt.figure(figsize=(w, h))
    fig.text(0.04, 0.965, title, fontsize=15, fontweight="bold", color=INK, va="top")
    # Character budgets per inch of figure width, measured for DejaVu Sans at these sizes.
    wrapped = "\n".join(textwrap.wrap(subtitle, width=int(w * 12.8)))
    fig.text(0.04, 0.905, wrapped, fontsize=10, color=MUTED, va="top", linespacing=1.45)
    foot = "\n".join(textwrap.wrap(footer, width=int(w * 16)))
    fig.text(0.04, 0.015, foot, fontsize=7.5, color=MUTED, va="bottom", linespacing=1.4)
    fig.subplots_adjust(left=left, right=0.97, top=top, bottom=bottom)
    return fig


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={"Software": None})
    plt.close(fig)


def _ratio_ticks(ax, lim: float) -> None:
    """Label a log1p-residual axis as the multiple of the prediction."""
    cands = [1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 16]
    ticks = [math.log(c) for c in cands if abs(math.log(c)) <= lim]
    ax.set_yticks(ticks)
    ax.set_yticklabels(
        [
            "same"
            if abs(t) < 1e-9
            else (f"×{math.exp(t):.0f}" if t > 0 else f"÷{math.exp(-t):.0f}")
            for t in ticks
        ]
    )


def load_inputs(art: Path, followup_dir: Path) -> dict:
    manifest = read_json(art / "manifest.json")
    for rel in ("metrics.json", "test_predictions.parquet", "coefficients.csv"):
        if sha256_file(art / rel) != manifest["artifact_sha256"][rel]:
            raise RuntimeError(f"artifacts/{rel} does not match the headline manifest")
    fu = read_json(followup_dir / "followup.json")
    if fu["headline_run_id"] != manifest["run_id"]:
        raise RuntimeError("follow-up was built on a different headline run; rerun tvp followup")
    if fu["headline_metrics_sha256"] != manifest["artifact_sha256"]["metrics.json"]:
        raise RuntimeError("follow-up was built against different headline metrics")
    with (art / "coefficients.csv").open() as fh:
        coefs = list(csv.DictReader(fh))
    return {
        "manifest": manifest,
        "metrics": read_json(art / "metrics.json"),
        "pred": read_parquet(art / "test_predictions.parquet"),
        "coefs": coefs,
        "fu": fu,
        "fu_pred": read_parquet(followup_dir / "test_predictions.parquet"),
        "followup_sha256": sha256_file(followup_dir / "followup.json"),
    }


def _footer(d: dict, extra: str = "") -> str:
    base = (
        "Data: Transfermarkt via transfermarkt-datasets. Test set: PL-active players' paid "
        f"permanent transfers from {d['metrics']['split']['test_start']} on. "
        f"Run {d['manifest']['run_id']}."
    )
    return f"{base} {extra}".strip()


def chart_predicted_vs_reported(d: dict, path: Path) -> None:
    p, conf = d["pred"], d["fu"]["conformal"]["headline"]["levels"]["0.80"]
    q, f = conf["q_log"], conf["factor"]
    x, y = p["prediction_eur"].to_numpy(), p["fee_eur"].to_numpy()
    r = np.log1p(y) - p["prediction_log"].to_numpy()
    fig = _figure(
        10.5,
        8.2,
        "The headline model ranks transfers sensibly, but aims low",
        "Each dot is one held-out transfer. Dots above the dashed line were under-predicted. "
        f"The blue band is the model's 80% conformal interval (prediction ×/÷ {f:.1f}), "
        "calibrated on training-window errors only. The five biggest euro misses are numbered.",
        _footer(d, "Log scales on both axes."),
        top=0.83,
    )
    ax = fig.add_subplot(111)
    lo = max(2e5, min(x.min(), y.min()) * 0.7)
    hi = max(x.max(), y.max()) * 1.6
    grid = np.geomspace(lo, hi, 200)
    if math.isfinite(f):
        ax.fill_between(
            grid,
            np.maximum((1 + grid) / f - 1, lo),
            (1 + grid) * f - 1,
            color=BLUE,
            alpha=0.09,
            lw=0,
            label=f"80% interval (×/÷ {f:.1f})",
        )
    ax.plot(grid, grid, color=INK, lw=1, ls=(0, (4, 3)), label="Prediction = reported fee")
    for pos in ("GK", "DF", "MF", "FW"):
        g = p["position"].to_numpy() == pos
        ax.scatter(
            x[g],
            y[g],
            s=34,
            color=POSITION_COLORS[pos],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.6,
            label=f"{POSITION_NAMES[pos]} ({g.sum()})",
            zorder=3,
        )
    worst = p.sort_values(["abs_error_eur", "transfer_id"], ascending=[False, True]).head(5)
    worst = worst.assign(rank=range(1, len(worst) + 1))
    # Worst misses cluster together; alternating sides by predicted fee keeps numbers apart.
    for i, (_, row) in enumerate(worst.sort_values("prediction_eur").iterrows()):
        right = i % 2 == 1
        ax.annotate(
            str(row["rank"]),
            (row["prediction_eur"], row["fee_eur"]),
            xytext=(7 if right else -7, 0),
            textcoords="offset points",
            ha="left" if right else "right",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=INK,
        )
    key = "Worst misses (reported vs predicted)\n" + "\n".join(
        f"{r['rank']}  {r['name']}: {_eur(r['fee_eur'])} vs {_eur(r['prediction_eur'])}"
        for _, r in worst.iterrows()
    )
    ax.text(
        0.02,
        0.98,
        key,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color=INK,
        linespacing=1.5,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    _log_eur_axis(ax.xaxis, lo, hi)
    _log_eur_axis(ax.yaxis, lo, hi)
    ax.set_xlabel("Predicted fee")
    ax.set_ylabel("Reported fee")
    stats = (
        f"Under-predicted: {np.mean(r > 0):.0%} of {len(r)}\n"
        f"Inside the 80% band: {np.mean(np.abs(r) <= q):.0%}\n"
        f"Above the band: {np.mean(r > q):.0%}   below: {np.mean(r < -q):.0%}"
    )
    ax.text(
        0.98,
        0.04,
        stats,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color=INK,
        linespacing=1.5,
        bbox={"boxstyle": "round,pad=0.6", "fc": "white", "ec": FAINT},
    )
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.16))
    _save(fig, path)


def _bins(order: np.ndarray, n_bins: int) -> list[np.ndarray]:
    return [b for b in np.array_split(order, min(n_bins, len(order))) if len(b)]


def chart_residual_structure(d: dict, path: Path) -> None:
    p = d["pred"]
    q = d["fu"]["conformal"]["headline"]["levels"]["0.80"]["q_log"]
    pred = p["prediction_eur"].to_numpy()
    r = np.log1p(p["fee_eur"].to_numpy()) - p["prediction_log"].to_numpy()
    fig = _figure(
        12,
        7.2,
        "Misses are multiplicative, lean high, and shrink as prices rise",
        "Residual = reported fee relative to the prediction, on a log scale (×2 means the fee "
        "was twice the prediction). Left: every price band is under-predicted on average, but "
        "the cheap end is far noisier. Right: that is why one interval width over-covers "
        "expensive players and under-covers cheap ones.",
        _footer(d, "Bands are equal-count groups of test transfers ordered by predicted fee."),
        top=0.79,
        bottom=0.2,
    )
    gs = fig.add_gridspec(1, 3, width_ratios=[2.2, 0.55, 1.35], wspace=0.08)
    ax = fig.add_subplot(gs[0])
    lim = max(2.2, float(np.abs(r).max()) * 1.08)
    band = min(q, lim)
    ax.axhspan(-band, band, color=BLUE, alpha=0.07, lw=0)
    ax.axhline(0, color=INK, lw=0.9, ls=(0, (4, 3)))
    ax.scatter(pred, r, s=20, color=GREY, alpha=0.65, edgecolor="none", zorder=2)
    order = np.argsort(pred)
    bins = _bins(order, 6)
    for b in bins:
        lo_x, hi_x = pred[b].min(), pred[b].max()
        mid = math.exp((math.log(lo_x) + math.log(hi_x)) / 2)
        q1, med, q3 = np.percentile(r[b], [25, 50, 75])
        ax.plot([mid, mid], [q1, q3], color=RED, lw=3, alpha=0.55, solid_capstyle="round")
        ax.scatter([mid], [med], s=46, color=RED, zorder=4, edgecolor="white", linewidth=0.8)
    ax.set_xscale("log")
    _log_eur_axis(ax.xaxis, pred.min() * 0.8, pred.max() * 1.15)
    ax.set_xlim(pred.min() * 0.8, pred.max() * 1.15)
    ax.set_ylim(-lim, lim)
    _ratio_ticks(ax, lim)
    ax.set_xlabel("Predicted fee")
    ax.set_ylabel("Reported fee relative to prediction")
    ax.set_title("Residual by predicted fee")
    ax.legend(
        handles=[
            Line2D([], [], marker="o", ls="", color=GREY, label="One transfer"),
            Line2D([], [], marker="o", color=RED, lw=3, alpha=0.8, label="Band median, middle 50%"),
            Patch(color=BLUE, alpha=0.15, label=f"80% interval (×/÷ {math.exp(q):.1f})"),
        ],
        loc="lower right",
    )

    axh = fig.add_subplot(gs[1], sharey=ax)
    axh.axhspan(-band, band, color=BLUE, alpha=0.07, lw=0)
    axh.hist(
        r, bins=np.linspace(-lim, lim, 30), orientation="horizontal", color=GREY, edgecolor="white"
    )
    axh.axhline(float(np.mean(r)), color=RED, lw=1.5)
    axh.axhline(0, color=INK, lw=0.9, ls=(0, (4, 3)))
    axh.text(
        axh.get_xlim()[1] * 0.97,
        -lim * 0.93,
        f"red line: average fee\n×{math.exp(float(np.mean(r))):.2f} the prediction",
        ha="right",
        va="bottom",
        fontsize=8,
        color=RED,
    )
    axh.tick_params(labelleft=False)
    axh.set_xlabel("Transfers")
    axh.grid(axis="y", visible=False)
    axh.set_title("Distribution")

    axc = fig.add_subplot(gs[2])
    labels, cover, above, below = [], [], [], []
    for b in _bins(order, 4):
        labels.append(f"Predicted {_eur(pred[b].min())} to {_eur(pred[b].max())} ({len(b)})")
        cover.append(np.mean(np.abs(r[b]) <= q))
        above.append(np.mean(r[b] > q))
        below.append(np.mean(r[b] < -q))
    yy = np.arange(len(labels))[::-1] * 1.3
    _coverage_bars(axc, yy, cover, below, above, height=0.6)
    for yv, lab in zip(yy, labels, strict=True):
        axc.text(0, yv + 0.36, lab, fontsize=8.5, color=INK, va="bottom")
    axc.axvline(0.8, color=INK, lw=1, ls=(0, (2, 2)))
    axc.text(0.8, yy[0] + 0.85, "promised 80%", ha="center", fontsize=8, color=INK)
    axc.set_yticks([])
    axc.set_ylim(-0.5, yy[0] + 1.1)
    axc.set_xlim(0, 1)
    axc.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    axc.grid(axis="y", visible=False)
    axc.spines["left"].set_visible(False)
    axc.set_title("80% interval coverage by price band")
    axc.legend(handles=_coverage_legend(), loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    _save(fig, path)


INSIDE = "#dce6f2"


def _coverage_bars(ax, y, cover, below, above, height: float) -> None:
    """Inside starts at zero so it reads directly against the promised-coverage line."""
    cover, below, above = map(np.asarray, (cover, below, above))
    ax.barh(y, cover, color=INSIDE, height=height)
    ax.barh(y, below, left=cover, color=BLUE, height=height)
    ax.barh(y, above, left=cover + below, color=RED, height=height)
    for yv, c in zip(y, cover, strict=True):
        ax.text(
            c / 2,
            yv,
            f"{c:.0%} inside",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=INK,
        )


def _coverage_legend() -> list:
    return [
        Patch(color=INSIDE, label="Inside"),
        Patch(color=BLUE, label="Fee below interval"),
        Patch(color=RED, label="Fee above interval"),
    ]


def chart_fee_drift(d: dict, path: Path) -> None:
    diag = d["metrics"]["residual_diagnostics"]
    med = {int(k): v for k, v in diag["median_fee_by_cycle_eur"].items()}
    level = {int(k): v for k, v in d["fu"]["price_level"]["by_cycle_median_level_eur"].items()}
    tr = {int(k): v for k, v in diag["train_in_sample_mean_log_residual_by_cycle"].items()}
    te = {int(k): v for k, v in diag["test_mean_log_residual_by_cycle"].items()}
    cycles = sorted(med)
    test_cycles = sorted(te)
    fig = _figure(
        11,
        7.4,
        "Fees inflate over time, and a model without a time term drifts low",
        "Top: the median reported fee in the study cohort per transfer cycle, and the trailing "
        "league price level the follow-up uses (median of every earlier paid move touching a PL "
        "club). Bottom: the headline model's average miss per cycle. It predicts an "
        "average-era fee, so early cycles come out too high and recent ones too low.",
        _footer(d, "A cycle runs from the day after one PL season ends to the day after the next."),
        top=0.77,
    )
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.32)
    ax = fig.add_subplot(gs[0])
    colors = [RED if c in test_cycles else BLUE for c in cycles]
    ax.bar(cycles, [med[c] for c in cycles], color=colors, alpha=0.8, width=0.7)
    lv = [c for c in cycles if c in level]
    ax.plot(lv, [level[c] for c in lv], color=INK, marker="o", ms=4, lw=1.5)
    ax.yaxis.set_major_formatter(EUR)
    ax.set_xticks(cycles)
    ax.set_title("Median reported fee per cycle")
    ax.legend(
        handles=[
            Patch(color=BLUE, alpha=0.8, label="Training cycles"),
            Patch(color=RED, alpha=0.8, label="Test cycles"),
            Line2D(
                [], [], color=INK, marker="o", ms=4, label="League price level (follow-up input)"
            ),
        ],
        loc="upper left",
    )
    ax.grid(axis="x", visible=False)

    ax2 = fig.add_subplot(gs[1], sharex=ax)
    ax2.axhline(0, color=INK, lw=0.9, ls=(0, (4, 3)))
    ax2.axvspan(min(test_cycles) - 0.5, max(test_cycles) + 0.5, color=RED, alpha=0.06, lw=0)
    tc = sorted(tr)
    ax2.plot(tc, [tr[c] for c in tc], color=BLUE, marker="o", lw=2, label="Training (in-sample)")
    ax2.plot(
        test_cycles,
        [te[c] for c in test_cycles],
        color=RED,
        marker="o",
        lw=2,
        label="Test (held out)",
    )
    for c, v in list(tr.items()) + list(te.items()):
        ax2.text(
            c,
            v + (0.07 if v >= 0 else -0.07),
            f"{v:+.2f}",
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=8,
            color=INK,
        )
    lim = max(0.9, max(abs(v) for v in [*tr.values(), *te.values()]) * 1.3)
    ax2.set_ylim(-lim, lim)
    _ratio_ticks(ax2, lim)
    ax2.set_ylabel("Reported vs predicted")
    ax2.set_title("Headline model's mean miss per cycle (log scale)")
    ax2.legend(loc="upper left")
    ax2.grid(axis="x", visible=False)
    _save(fig, path)


def chart_coefficients(d: dict, path: Path) -> None:
    head = [c for c in d["coefs"] if c["selected"] == "True" and c["kind"] == "num"]
    enr = [c for c in d["fu"]["context_round"]["selected_coefficients"] if c["kind"] == "num"]
    fig = _figure(
        12,
        7.4,
        "What the models lean on",
        "Change in the predicted fee for a one-standard-deviation increase in each input, "
        "holding the others fixed. ×2 doubles the prediction. Inputs are correlated (minutes "
        "and appearances especially), so read the pattern, not single signs. Inputs marked "
        "(new) are ones the headline does not have. Position terms are left out: with full one-hot "
        "encoding only their differences mean anything.",
        _footer(d, "Enriched model: CV-selected in the second follow-up round (post-holdout)."),
        top=0.79,
        left=0.15,
    )
    gs = fig.add_gridspec(1, 2, wspace=0.62)
    panels = [
        (head, f"Headline ({d['metrics']['selected_model']['name']})"),
        (enr, f"Enriched follow-up ({d['fu']['context_round']['selected_model']})"),
    ]
    lim = max(abs(float(c["coef_log1p"])) for rows, _ in panels for c in rows) * 1.25
    for i, (rows, title) in enumerate(panels):
        ax = fig.add_subplot(gs[i])
        rows = sorted(rows, key=lambda c: float(c["coef_log1p"]))
        vals = [float(c["coef_log1p"]) for c in rows]
        names = [
            FRIENDLY.get(c["term"], c["term"])
            + ("  (new)" if c["term"] in FOLLOWUP_ONLY_INPUTS else "")
            for c in rows
        ]
        colors = [GREEN if v > 0 else RED for v in vals]
        ax.barh(range(len(rows)), vals, color=colors, alpha=0.85, height=0.66)
        for j, v in enumerate(vals):
            ax.text(
                v + (0.02 if v >= 0 else -0.02) * lim / 0.5,
                j,
                f"×{math.exp(v):.2f}",
                va="center",
                ha="left" if v >= 0 else "right",
                fontsize=8.5,
                color=INK,
            )
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(names)
        ax.axvline(0, color=INK, lw=0.9)
        ax.set_xlim(-lim, lim)
        ax.set_xlabel("Coefficient on log1p(fee) per training SD")
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    _save(fig, path)


def _variant_label(name: str) -> str:
    return name.replace("+", ", ").replace("_", " ")


def _model_labels(fu: dict) -> dict[str, str]:
    return {
        "headline": "Headline linear",
        "followup": f"Follow-up: {_variant_label(fu['selected_variant'])}",
        "enriched": f"Enriched: {_variant_label(fu['context_round']['selected_variant'])}",
    }


def chart_model_comparison(d: dict, path: Path) -> None:
    met, fu = d["metrics"], d["fu"]
    ctx, b, mv = fu["context_round"], fu["boosting"], fu["market_value_comparator"]["methods"]
    head = fu["variants"]["headline"]
    first = fu["variants"][fu["selected_variant"]]
    enr = ctx["variants"][ctx["selected_variant"]]
    names = _model_labels(fu)
    n_test = met["split"]["test"]["rows"]
    rows = [
        ("Train median fee", None, met["methods"]["train_median"]["mae_eur"], GREY, ""),
        (
            "Train median by position",
            None,
            met["methods"]["train_median_by_position"]["mae_eur"],
            GREY,
            "",
        ),
        (
            "Headline linear (pre-registered)",
            head["cv_mean_log_mae"],
            head["test"]["mae_eur"],
            BLUE,
            "",
        ),
        ("Gradient boosting", b["gbm"]["cv_mean_log_mae"], b["gbm"]["test"]["mae_eur"], GOLD, ""),
        (
            "Gradient boosting, monotonic",
            b["gbm_monotone"]["cv_mean_log_mae"],
            b["gbm_monotone"]["test"]["mae_eur"],
            GOLD,
            "",
        ),
        (names["followup"], first["cv_mean_log_mae"], first["test"]["mae_eur"], GREEN, ""),
        (names["enriched"], enr["cv_mean_log_mae"], enr["test"]["mae_eur"], DARK_GREEN, ""),
        ("Transfermarkt market value", None, mv["market_value"]["mae_eur"], INK, "//"),
    ]
    fig = _figure(
        12,
        6.8,
        "What each change bought",
        "Left: what model selection saw, the average cross-validated error on the log scale "
        "inside the training window (lower is better; the axis starts at 0.70). Right: mean "
        f"absolute euro error on the {n_test} held-out transfers. Only the headline row was "
        "pre-registered. Boosting, the follow-ups and the market-value comparator were all "
        "added after the test set was scored.",
        _footer(
            d,
            "Market value is never a model input; it is scored as if it were a prediction. "
            "Baselines and market value have no CV score because nothing was selected for them.",
        ),
        top=0.78,
        left=0.25,
    )
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.35], wspace=0.05)
    y = np.arange(len(rows))[::-1]
    ax = fig.add_subplot(gs[0])
    for yy, (_, cv, _, col, _) in zip(y, rows, strict=True):
        if cv is None:
            ax.text(0.703, yy, "no CV score", va="center", fontsize=8, color=MUTED)
            continue
        ax.barh(yy, cv, color=col, alpha=0.85, height=0.62)
        ax.text(cv + 0.003, yy, f"{cv:.3f}", va="center", fontsize=9, color=INK)
    ax.axvline(head["cv_mean_log_mae"], color=BLUE, lw=1, ls=(0, (3, 3)))
    ax.set_xlim(0.70, 0.87)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_title("CV log-MAE (training window)")
    ax.grid(axis="y", visible=False)
    ax2 = fig.add_subplot(gs[1], sharey=ax)
    for yy, (_, _, mae, col, hatch) in zip(y, rows, strict=True):
        ax2.barh(
            yy,
            mae,
            color=col if not hatch else "white",
            edgecolor=col,
            hatch=hatch,
            alpha=0.85,
            height=0.62,
            lw=1.2,
        )
        ax2.text(mae + 2e5, yy, f"€{mae / 1e6:.2f}m", va="center", fontsize=9, color=INK)
    ax2.axvline(head["test"]["mae_eur"], color=BLUE, lw=1, ls=(0, (3, 3)))
    ax2.text(
        head["test"]["mae_eur"],
        y[0] + 0.55,
        "headline",
        ha="center",
        fontsize=8,
        color=BLUE,
    )
    ax2.xaxis.set_major_formatter(EUR)
    ax2.set_xlim(0, max(r[2] for r in rows) * 1.15)
    ax2.set_ylim(-0.6, len(rows) - 0.1)
    ax2.tick_params(labelleft=False)
    ax2.set_title("Test MAE (euros)")
    ax2.grid(axis="y", visible=False)
    _save(fig, path)


def chart_interval_coverage(d: dict, path: Path) -> None:
    conf, fu = d["fu"]["conformal"], d["fu"]
    names = _model_labels(fu)
    rows = []
    for key in ("headline", "followup", "enriched"):
        for lv, s in conf[key]["levels"].items():
            rows.append((key, float(lv), s))
    head80 = conf["headline"]["levels"][f"{conf['predict_level']:.2f}"]["test"]
    fig = _figure(
        12,
        6.8,
        "Every interval covers at least what it promises",
        f"Each bar splits the {d['metrics']['split']['test']['rows']} test transfers into those "
        "whose reported fee fell inside the interval, below it, or above it. The dotted tick is "
        "the promised coverage. At 80% the headline misses high "
        f"({head80['share_above_upper']:.0%}) about twice as often as low "
        f"({head80['share_below_lower']:.0%}), a leftover of fee inflation. The price-level "
        "term evens that out, and the context inputs make the interval narrower.",
        _footer(
            d,
            "Widths come from training-window CV errors only, fixed before any test row is "
            "seen. The median interval is for the median test prediction.",
        ),
        top=0.78,
        left=0.24,
        bottom=0.17,
    )
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1], wspace=0.06)
    ax = fig.add_subplot(gs[0])
    per_model = len(conf["headline"]["levels"])
    y = -np.array([i + (i // per_model) * 0.6 for i in range(len(rows))])
    tests = [s["test"] for _, _, s in rows]
    _coverage_bars(
        ax,
        y,
        [t["coverage"] for t in tests],
        [t["share_below_lower"] for t in tests],
        [t["share_above_upper"] for t in tests],
        height=0.66,
    )
    for yy, (_, lv, s) in zip(y, rows, strict=True):
        t = s["test"]
        ax.plot([lv, lv], [yy - 0.42, yy + 0.42], color=INK, lw=1.4, ls=(0, (1.5, 1.5)))
        ax.text(
            1.015,
            yy,
            f"{t['share_below_lower']:.0%} below\n{t['share_above_upper']:.0%} above",
            va="center",
            fontsize=8,
            color=MUTED,
            linespacing=1.3,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{names[k]}\n{lv:.0%} interval" for k, lv, _ in rows], fontsize=8.8, linespacing=1.3
    )
    ax.set_xlim(0, 1.13)
    ax.xaxis.set_major_locator(FixedLocator([0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", visible=False)
    ax.set_title("Where the reported fee landed")
    ax.legend(
        handles=[
            *_coverage_legend(),
            Line2D([], [], color=INK, ls=(0, (1.5, 1.5)), label="Promised coverage"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.07),
        ncol=4,
    )
    axw = fig.add_subplot(gs[1], sharey=ax)
    finite = [s["factor"] for _, _, s in rows if math.isfinite(s["factor"])]
    for yy, (_, lv, s) in zip(y, rows, strict=True):
        if not math.isfinite(s["factor"]):
            axw.text(1.1, yy, "unbounded (too few\ncalibration rows)", va="center", fontsize=8)
            continue
        axw.barh(yy, s["factor"], color=BLUE if lv < 0.85 else NAVY, alpha=0.85, height=0.66)
        axw.text(
            s["factor"] + 0.1,
            yy,
            f"×/÷ {s['factor']:.2f}\nmedian {_eur(s['median_lower_eur'])} to "
            f"{_eur(s['median_upper_eur'])}",
            va="center",
            fontsize=8,
            color=INK,
            linespacing=1.3,
        )
    axw.set_xlim(1, max(finite, default=2) * 1.9)
    axw.xaxis.set_major_locator(FixedLocator([1, 2, 4, 6]))
    axw.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"×{v:.0f}"))
    axw.tick_params(labelleft=False)
    axw.set_title("Width (prediction ×/÷)")
    axw.grid(axis="y", visible=False)
    _save(fig, path)


def chart_biggest_fees(d: dict, path: Path, n: int = 25) -> None:
    p = d["pred"].merge(
        d["fu_pred"][["transfer_id", "lower_eur__headline__0.80", "upper_eur__headline__0.80"]],
        on="transfer_id",
    )
    top = p.sort_values(["fee_eur", "transfer_id"], ascending=[False, True]).head(n)[::-1]
    lo = top["lower_eur__headline__0.80"].to_numpy()
    hi = top["upper_eur__headline__0.80"].to_numpy()
    fee, pred = top["fee_eur"].to_numpy(), top["prediction_eur"].to_numpy()
    covered = (fee >= lo) & (fee <= hi)
    n_above = int((fee > pred).sum())
    above_pred = "All of these fees" if n_above == len(top) else f"{n_above} of these fees"
    fig = _figure(
        11,
        10,
        f"The {len(top)} biggest test fees against the headline model's 80% interval",
        "Each bar is the 80% conformal interval around the headline prediction (hollow dot). "
        "The diamond is the reported fee: blue when inside, red when outside. "
        f"{above_pred} sit above the prediction, but the interval is wide enough to reach "
        f"{covered.sum()} of {len(top)}.",
        _footer(d, "Log scale. Interval = prediction ×/÷ the same factor for every transfer."),
        top=0.83,
        left=0.35,
        bottom=0.08,
    )
    ax = fig.add_subplot(111)
    yy = np.arange(len(top))
    xlo = min(pred.min(), lo[lo > 0].min(initial=np.inf)) * 0.85
    xhi = max(fee.max(), hi[np.isfinite(hi)].max(initial=0)) * 1.15
    ax.hlines(yy, np.clip(lo, xlo, xhi), np.clip(hi, xlo, xhi), color=BLUE, alpha=0.28, lw=6)
    ax.plot(
        np.column_stack([pred, fee]).T,
        np.column_stack([yy, yy]).T,
        color=MUTED,
        lw=0.8,
        ls=(0, (2, 2)),
        zorder=2,
    )
    ax.scatter(pred, yy, s=42, facecolor="white", edgecolor=BLUE, lw=1.5, zorder=3)
    ax.scatter(fee[covered], yy[covered], marker="D", s=46, color=BLUE, zorder=4)
    ax.scatter(fee[~covered], yy[~covered], marker="D", s=52, color=RED, zorder=4)
    ax.set_yticks(yy)
    ax.set_yticklabels(
        [
            f"{r['name']} → {r['to_club_name']}  {_eur(r['fee_eur'])}  ({r['transfer_date']:%b %Y})"
            for _, r in top.iterrows()
        ],
        fontsize=8.5,
    )
    for tick, ok in zip(ax.get_yticklabels(), covered, strict=True):
        if not ok:
            tick.set_color(RED)
    ax.set_xscale("log")
    ax.set_xlim(xlo, xhi)
    _log_eur_axis(ax.xaxis, xlo, xhi)
    ax.set_ylim(-0.7, len(top) - 0.3)
    ax.grid(axis="y", visible=False)
    ax.legend(
        handles=[
            Line2D([], [], color=BLUE, alpha=0.28, lw=6, label="80% interval"),
            Line2D([], [], marker="o", ls="", mfc="white", mec=BLUE, label="Headline prediction"),
            Line2D([], [], marker="D", ls="", color=BLUE, label="Reported fee, inside"),
            Line2D([], [], marker="D", ls="", color=RED, label="Reported fee, outside"),
        ],
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        ncol=4,
    )
    _save(fig, path)


def chart_worst_misses(d: dict, path: Path) -> None:
    ctx = d["fu"]["context_round"]
    names = _model_labels(d["fu"])
    rows = ctx["headline_worst_misses"][::-1]
    fig = _figure(
        11,
        7,
        "Richer inputs help when the missing information was about football",
        "The headline model's five biggest euro misses, re-predicted by each follow-up model. "
        "The context inputs move Diaby (a Bundesliga season the headline ignores) and Díaz "
        "(a strong team and European minutes) most of the way. Durán's Saudi move and Neto's "
        "injury-shortened seasons stay badly under-predicted: nothing in the data describes "
        "the buyer or the injuries.",
        _footer(
            d,
            "Follow-up models were designed after the holdout was scored. Context inputs are "
            "measured over the same lookback window as the headline inputs.",
        ),
        top=0.73,
        left=0.16,
    )
    ax = fig.add_subplot(111)
    xlo = min(r[k] for r in rows for k in ("headline_eur", "followup_eur", "enriched_eur")) * 0.8
    series = [
        ("headline_eur", names["headline"], BLUE),
        ("followup_eur", names["followup"], GOLD),
        ("enriched_eur", names["enriched"], GREEN),
    ]
    for i, r in enumerate(rows):
        vals = [r[k] for k, *_ in series] + [r["fee_eur"]]
        ax.plot([min(vals), max(vals)], [i, i], color=FAINT, lw=2, zorder=1)
        ax.annotate(
            "",
            xy=(r["enriched_eur"], i),
            xytext=(r["headline_eur"], i),
            arrowprops={"arrowstyle": "-|>", "color": GREEN, "lw": 1.6, "shrinkB": 5},
            zorder=2,
        )
        for k, _, col in series:
            ax.scatter([r[k]], [i], color=col, s=64, zorder=3, edgecolor="white")
        ax.scatter([r["fee_eur"]], [i], color=INK, marker="D", s=70, zorder=4)
        ax.text(
            r["fee_eur"] * 1.07,
            i,
            _eur(r["fee_eur"]),
            va="center",
            fontsize=9,
            color=INK,
            fontweight="bold",
        )
        ax.text(
            xlo * 1.04,
            i - 0.3,
            f"European cups {r['europe_minutes']:,.0f} min · other leagues "
            f"{r['other_league_minutes']:,.0f} min · team {r['team_points_per_game']:.2f} pts/game",
            va="center",
            fontsize=7.8,
            color=MUTED,
        )
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["name"] for r in rows], fontsize=9.5, fontweight="bold")
    ax.set_xscale("log")
    xhi = max(r["fee_eur"] for r in rows) * 1.35
    ax.set_xlim(xlo, xhi)
    _log_eur_axis(ax.xaxis, xlo, xhi)
    ax.set_ylim(-0.7, len(rows) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.legend(
        handles=[Line2D([], [], marker="o", ls="", color=col, label=lab) for _, lab, col in series]
        + [Line2D([], [], marker="D", ls="", color=INK, label="Reported fee")],
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        ncol=4,
        fontsize=8.5,
    )
    _save(fig, path)


def run_charts(fcfg, out: Path | None = None) -> dict:
    d = load_inputs(fcfg.headline.artifact_dir, fcfg.output_dir)
    out = out or fcfg.output_dir.parent / "charts"
    _style()
    builders = (
        chart_predicted_vs_reported,
        chart_residual_structure,
        chart_fee_drift,
        chart_coefficients,
        chart_model_comparison,
        chart_interval_coverage,
        chart_biggest_fees,
        chart_worst_misses,
    )
    for name, build in zip(CHARTS, builders, strict=True):
        build(d, out / name)
    manifest = {
        "headline_run_id": d["manifest"]["run_id"],
        "followup_sha256": d["followup_sha256"],
        "files": {name: sha256_file(out / name) for name in CHARTS},
    }
    write_json(manifest, out / "manifest.json")
    return {**manifest, "output_dir": str(out)}


__all__ = ["CHARTS", "run_charts"]
