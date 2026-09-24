"""Assemble canonical tables into the supervised transfer table plus the exact count funnel.

Feasibility, build-features, and predict all call these functions, so the published funnel is
computed by the same code that produces the training rows.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from transfer_value import clean
from transfer_value.config import Config
from transfer_value.features import FEATURE_REASONS, build_features
from transfer_value.io import read_parquet
from transfer_value.labels import LABEL_REASONS, apply_labels, label_exclusion_reason
from transfer_value.seasons import cycle_of, season_table, transfer_cycles

INTERIM_TABLES = ("transfers", "players", "games", "appearances", "lineups", "seasons")


class StudyWindowError(ValueError):
    pass


def canonical_tables(cfg: Config, raw_dir: Path) -> tuple[dict[str, pd.DataFrame], dict]:
    ids = cfg.competition_ids
    comps = clean.read_raw(raw_dir, "competitions")
    clean.verify_competitions(comps, ids, cfg.raw["competition"]["code"])
    games = clean.clean_games(clean.read_raw(raw_dir, "games"), ids)
    transfers, t_stats = clean.clean_transfers(clean.read_raw(raw_dir, "transfers"))
    players = clean.clean_players(clean.read_raw(raw_dir, "players"))
    appearances, a_stats = clean.clean_appearances(clean.read_raw(raw_dir, "appearances"), games)
    lineups, l_stats = clean.clean_lineups(clean.read_raw(raw_dir, "game_lineups"), games)
    seasons = season_table(games, int(cfg.raw["competition"]["games_per_season"]))
    tables = {
        "transfers": transfers,
        "players": players,
        "games": games,
        "appearances": appearances,
        "lineups": lineups,
        "seasons": seasons,
    }
    stats = {"transfers": t_stats, "appearances": a_stats, "lineups": l_stats}
    return tables, stats


def load_interim(cfg: Config) -> dict[str, pd.DataFrame]:
    return {name: read_parquet(cfg.interim_dir / f"{name}.parquet") for name in INTERIM_TABLES}


def verify_study_window(cfg: Config, seasons: pd.DataFrame) -> dict:
    """Pinned dates must equal the values derived from source season metadata."""
    cycles = transfer_cycles(seasons).set_index("season_id")
    ids = [int(s) for s in cfg.study["season_ids"]]
    missing = [s for s in ids if s not in cycles.index]
    if missing:
        raise StudyWindowError(f"study seasons without cycle metadata: {missing}")
    if len(ids) < 2:
        raise StudyWindowError("need at least two study seasons")
    incomplete = seasons[seasons["season_id"].isin(ids) & ~seasons["complete"]]
    if not incomplete.empty:
        raise StudyWindowError(f"incomplete study seasons: {incomplete['season_id'].tolist()}")
    derived = {
        "transfer_start": cycles.loc[ids[0], "cycle_start"],
        "transfer_end_exclusive": cycles.loc[ids[-1], "cycle_end_exclusive"],
        "test_start": cycles.loc[ids[-2], "cycle_start"],
    }
    pinned = {
        "transfer_start": pd.Timestamp(cfg.transfer_start),
        "transfer_end_exclusive": pd.Timestamp(cfg.transfer_end_exclusive),
        "test_start": pd.Timestamp(cfg.test_start),
    }
    diffs = {k: (pinned[k].date(), derived[k].date()) for k in pinned if pinned[k] != derived[k]}
    if diffs:
        raise StudyWindowError(
            f"pinned dates differ from season metadata (pinned, derived): {diffs}"
        )
    return {
        "study_seasons": ids,
        "test_seasons": ids[-2:],
        **{k: v.date().isoformat() for k, v in derived.items()},
        "rationale": (
            "Cycle s runs from the day after season s-1's last match to the day after season s's "
            "last match. T is the start of the earlier of the last two study cycles."
        ),
    }


def assemble(tables: dict[str, pd.DataFrame], cfg: Config) -> dict:
    start = pd.Timestamp(cfg.transfer_start)
    end = pd.Timestamp(cfg.transfer_end_exclusive)
    t_start = pd.Timestamp(cfg.test_start)
    transfers = tables["transfers"]

    labeled, label_stats = apply_labels(transfers, start, end)
    features, candidates = build_features(
        labeled,
        tables["appearances"],
        tables["lineups"],
        tables["players"],
        tables["seasons"],
        [int(s) for s in cfg.study["source_seasons"]],
        cfg.min_minutes,
        bool(cfg.study["include_cards"]),
    )
    cycles = transfer_cycles(tables["seasons"])
    features["transfer_cycle"] = cycle_of(features["transfer_date"], cycles)
    if features["transfer_cycle"].isna().any():
        raise StudyWindowError("feature rows outside any transfer cycle")
    features["partition"] = (features["transfer_date"] >= t_start).map(
        {True: "test", False: "train"}
    )

    funnel = build_funnel(transfers, candidates, features, start, end)
    rt = label_stats["round_trip_diagnostic"]
    flagged = set(rt.pop("flagged_transfer_ids"))
    in_cohort = features["transfer_id"].isin(flagged)
    rt["final_cohort_flagged"] = int(in_cohort.sum())
    rt["final_cohort_flagged_test"] = int((in_cohort & (features["partition"] == "test")).sum())
    rt["final_cohort_flagged_transfer_ids"] = sorted(features.loc[in_cohort, "transfer_id"])
    return {
        "features": features,
        "candidates": candidates,
        "label_stats": label_stats,
        "funnel": funnel,
    }


def run_build_features(cfg: Config) -> dict:
    from transfer_value.io import (
        environment_info,
        frame_fingerprint,
        read_json,
        run_id,
        write_json,
        write_parquet,
    )

    cfg.require_pinned()
    imeta = read_json(cfg.interim_dir / "manifest.json")
    if imeta["competition_ids"] != cfg.competition_ids:
        raise StudyWindowError("interim tables were built for other competitions; rerun ingest")
    pinned = cfg.raw["source"].get("sha256") or {}
    for name, digest in imeta["source_hashes"].items():
        if pinned.get(name) and pinned[name] != digest:
            raise StudyWindowError(f"interim built from different bytes for {name}; rerun ingest")
    tables = load_interim(cfg)
    window = verify_study_window(cfg, tables["seasons"])
    data = assemble(tables, cfg)
    feats, funnel = data["features"], data["funnel"]
    if funnel["final_rows"] < cfg.split["min_total_rows"]:
        raise StudyWindowError(f"only {funnel['final_rows']} supervised rows")
    if funnel["test_rows"] < cfg.split["min_test_rows"]:
        raise StudyWindowError(f"only {funnel['test_rows']} test rows")

    env = environment_info(cfg.root)
    rid = run_id(imeta["source_hashes"], cfg.hash(), env["lockfile_sha256"])
    funnel.update(
        run_id=rid, source_hashes=imeta["source_hashes"], config_hash=cfg.hash(), gates_passed=True
    )
    label_stats = {**data["label_stats"], "run_id": rid}
    write_parquet(feats, cfg.processed_dir / "features.parquet")
    write_json(label_stats, cfg.processed_dir / "label_stats.json")
    write_json(funnel, cfg.processed_dir / "funnel.json")
    manifest = {
        "stage": "build-features",
        "run_id": rid,
        "source_hashes": imeta["source_hashes"],
        "config_hash": cfg.hash(),
        "features_fingerprint": frame_fingerprint(feats),
        "rows": len(feats),
        "study_window": window,
        "environment": env,
    }
    write_json(manifest, cfg.processed_dir / "features_manifest.json")
    return manifest


def build_funnel(
    transfers: pd.DataFrame,
    candidates: pd.DataFrame,
    features: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict:
    """Sequential funnel: each dropped row has exactly one first-failure reason."""
    label_reason = label_exclusion_reason(transfers, start, end)
    steps = [{"step": "raw_transfer_rows_after_exact_dedup", "rows": len(transfers)}]
    remaining = len(transfers)
    label_names = {
        "missing_player_id": "valid_player_ids",
        "missing_transfer_date": "valid_transfer_dates",
        "outside_study_window": "in_study_window",
        "undisclosed_fee": "fee_field_present",
        "malformed_fee": "fee_well_formed",
        "zero_fee_free_or_loan": "positive_reported_fee",
    }
    for reason in LABEL_REASONS:
        dropped = int((label_reason == reason).sum())
        remaining -= dropped
        steps.append(
            {"step": label_names[reason], "rows": remaining, "dropped": dropped, "reason": reason}
        )
    note = "Implied by a positive fee under the source encoding (loans are recorded as 0)."
    for step in ("classified_permanent", "classified_non_loan", "unique_transfer_identity"):
        steps.append(
            {
                "step": step,
                "rows": remaining,
                "dropped": 0,
                "note": note
                if step != "unique_transfer_identity"
                else "Natural key unique; conflicts fail ingest.",
            }
        )
    feature_names = {
        "no_lookback_coverage": "lookback_coverage_available",
        "no_league_appearances_in_lookback": "league_active_candidates",
        "insufficient_lookback_minutes": "sufficient_lookback_minutes",
        "invalid_per90": "valid_per90_features",
        "missing_age": "usable_age",
        "missing_position": "usable_position",
    }
    fr = candidates["feature_exclusion_reason"]
    for reason in FEATURE_REASONS:
        dropped = int((fr == reason).sum())
        remaining -= dropped
        steps.append(
            {"step": feature_names[reason], "rows": remaining, "dropped": dropped, "reason": reason}
        )
    if remaining != len(features):
        raise AssertionError(f"funnel does not reconcile: {remaining} != {len(features)}")
    steps.append({"step": "final_supervised_rows", "rows": len(features)})
    n_train = int((features["partition"] == "train").sum())
    n_test = int((features["partition"] == "test").sum())
    if n_train + n_test != len(features):
        raise AssertionError("train + test != final rows")
    return {
        "steps": steps,
        "final_rows": len(features),
        "train_rows": n_train,
        "test_rows": n_test,
        "position_source_counts": {
            str(k): int(v) for k, v in features["position_source"].value_counts().items()
        },
        "by_cycle": {str(k): int(v) for k, v in features.groupby("transfer_cycle").size().items()},
    }
