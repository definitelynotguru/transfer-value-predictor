"""Deterministic evaluation figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

_EUR_M = FuncFormatter(lambda v, _: f"€{v / 1e6:g}m")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 110, "font.size": 10})


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, metadata={"Software": None})
    plt.close(fig)


def predicted_vs_actual(df: pd.DataFrame, model: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    colors = {"GK": "#7f7f7f", "DF": "#1f77b4", "MF": "#2ca02c", "FW": "#d62728"}
    for pos, g in df.groupby("position", sort=True):
        ax.scatter(
            g["prediction_eur"],
            g["fee_eur"],
            s=18,
            alpha=0.7,
            label=pos,
            color=colors.get(pos, "k"),
        )
    lo = max(1e5, min(df["fee_eur"].min(), df["prediction_eur"].min()) * 0.8)
    hi = max(df["fee_eur"].max(), df["prediction_eur"].max()) * 1.2
    ax.plot([lo, hi], [lo, hi], color="black", lw=1, ls="--", label="y = x")
    ax.set(
        xscale="log",
        yscale="log",
        xlim=(lo, hi),
        ylim=(lo, hi),
        xlabel="Predicted fee (EUR, log scale)",
        ylabel="Reported fee (EUR, log scale)",
        title=f"Predicted vs reported fee, test set (n={len(df)})\n{model}",
    )
    ax.xaxis.set_major_formatter(_EUR_M)
    ax.yaxis.set_major_formatter(_EUR_M)
    ax.legend(title="Position", loc="upper left")
    ax.grid(alpha=0.3, which="both")
    _save(fig, path)


def residuals_vs_predicted(df: pd.DataFrame, model: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.scatter(df["prediction_eur"], df["residual_eur"], s=18, alpha=0.7, color="#1f77b4")
    ax.axhline(0, color="black", lw=1, ls="--")
    ax.set(
        xlabel="Predicted fee (EUR)",
        ylabel="Residual = reported − predicted (EUR)",
        title=f"Residuals vs predicted, test set (n={len(df)})\n{model}",
    )
    ax.xaxis.set_major_formatter(_EUR_M)
    ax.yaxis.set_major_formatter(_EUR_M)
    ax.grid(alpha=0.3)
    _save(fig, path)


def residual_distribution(df: pd.DataFrame, model: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    r = df["residual_eur"].to_numpy() / 1e6
    bins = np.linspace(r.min(), r.max(), 40)
    ax.hist(r, bins=bins, color="#1f77b4", alpha=0.8, edgecolor="white")
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set(
        xlabel="Residual = reported − predicted (EUR millions)",
        ylabel="Transfers",
        title=f"Residual distribution, test set (n={len(df)})\n{model}",
    )
    ax.grid(alpha=0.3)
    _save(fig, path)
