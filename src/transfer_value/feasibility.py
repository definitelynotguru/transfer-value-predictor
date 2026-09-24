"""Schema capability report plus the exact count funnel and data gates.

Runs on raw files with the same normalization, label, season, feature, and split code as the
production pipeline, so its counts are the real ones, not an approximation.
"""

from __future__ import annotations

import pandas as pd

from transfer_value.config import RAW_FILES, SCHEMA, Config
from transfer_value.dataset import assemble, canonical_tables, verify_study_window
from transfer_value.ingest import verified_source_hashes
from transfer_value.io import write_json
from transfer_value.split import SplitError, chronological_folds, holdout


def schema_probe(cfg: Config) -> dict:
    out = {}
    for table, fname in RAW_FILES.items():
        path = cfg.raw_dir / fname
        header = list(pd.read_csv(path, nrows=0).columns) if path.exists() else []
        out[table] = {
            "file": fname,
            "exists": path.exists(),
            "missing_required_columns": [c for c in SCHEMA[table] if c not in header],
        }
    return out


def capability_report(cfg: Config, tables: dict[str, pd.DataFrame], stats: dict) -> dict:
    t, a, p, s = tables["transfers"], tables["appearances"], tables["players"], tables["seasons"]
    fee_classes = t["fee_class"].value_counts().to_dict()
    lineup_cov = float(a.merge(tables["lineups"], how="left")["lineup_position"].notna().mean())
    seasons = [
        {
            "season_id": int(r.season_id),
            "first_match": r.first_match.date().isoformat(),
            "last_match": r.last_match.date().isoformat(),
            "n_games": int(r.n_games),
            "complete": bool(r.complete),
        }
        for r in s.itertuples()
    ]
    caps = {
        "player_identity": bool(t["player_id"].notna().all() and p["player_id"].is_unique),
        "transfer_date": bool(t["transfer_date"].notna().mean() > 0.99),
        "transfer_fee": bool(fee_classes.get("paid", 0) > 0),
        "paid_vs_free_classification": True,
        "permanent_vs_loan_classification": True,
        "appearance_date": bool(a["match_date"].notna().all()),
        "player_to_game_joins": bool(a["season_id"].notna().all()),
        "competition_filtering": bool(len(a) > 0),
        "stable_transfer_identity": bool(t["transfer_id"].is_unique),
        "season_identity_and_completion": bool(s["complete"].any()),
        "birth_date_coverage": bool(p["date_of_birth"].notna().mean() > 0.95),
    }
    return {
        "capabilities": caps,
        "all_required_capabilities": all(caps.values()),
        "selected_columns": SCHEMA,
        "competition": {
            "ids": cfg.competition_ids,
            "code": cfg.raw["competition"]["code"],
            "verified_against": "competitions.csv (competition_code, type=domestic_league)",
        },
        "transfer_type_interpretation": (
            "No transfer-type or loan column exists. Upstream (dcaribou/transfermarkt-datasets "
            "dbt base_transfers.sql) parses the Transfermarkt fee string: '-', '?', '' -> null; "
            "'free transfer' -> 0; strings starting with '€' -> amount; everything else "
            "('loan transfer', 'End of loan', 'Loan fee: €X', ...) -> 0. A positive fee is "
            "therefore a paid permanent transfer. Spot checks: Lukaku Chelsea->Inter 2022 loan, "
            "Cancelo Man City->Bayern 2023 loan, Sancho Man Utd->Chelsea 2024 loan all have fee 0."
        ),
        "fee_parsing": {
            "source_type": "numeric EUR (already parsed upstream)",
            "classes": {str(k): int(v) for k, v in fee_classes.items()},
            "currency": cfg.raw["fees"]["currency"],
        },
        "transfer_identity": (
            "No source transfer ID. transfer_id = sha1(player_id|transfer_date|from_club_id|"
            "to_club_id)[:16]; the natural key is unique after exact-duplicate removal."
        ),
        "seasons": seasons,
        "historical_position": {
            "available": True,
            "source": "game_lineups.position for the same league appearances",
            "appearance_coverage": round(lineup_cov, 4),
            "fallback": "players.position (current) as a flagged retrospective proxy",
            "historical_player_record": "not available in this snapshot",
        },
        "age": "floor years from players.date_of_birth; no birth-year fallback column exists",
        "cleaning_stats": stats,
    }


def run_feasibility(cfg: Config) -> dict:
    probe = schema_probe(cfg)
    broken = {k: v for k, v in probe.items() if not v["exists"] or v["missing_required_columns"]}
    if broken:
        report = {"schema_probe": probe, "passed": False, "failure": "schema"}
        write_json(report, cfg.processed_dir / "capability_report.json")
        return report

    hashes = verified_source_hashes(cfg)
    tables, stats = canonical_tables(cfg, cfg.raw_dir)
    report = capability_report(cfg, tables, stats)
    report["schema_probe"] = probe
    report["source_hashes"] = hashes
    report["config_hash"] = cfg.hash()
    gates: dict[str, dict] = {}

    if not report["all_required_capabilities"]:
        report.update(passed=False, failure="capabilities", gates=gates)
        write_json(report, cfg.processed_dir / "capability_report.json")
        return report

    cfg.require_pinned()
    report["study_window"] = verify_study_window(cfg, tables["seasons"])
    data = assemble(tables, cfg)
    funnel = data["funnel"]
    feats = data["features"]

    gates["min_total_rows"] = {
        "required": cfg.split["min_total_rows"],
        "actual": funnel["final_rows"],
        "passed": funnel["final_rows"] >= cfg.split["min_total_rows"],
    }
    gates["min_test_rows"] = {
        "required": cfg.split["min_test_rows"],
        "actual": funnel["test_rows"],
        "passed": funnel["test_rows"] >= cfg.split["min_test_rows"],
    }
    try:
        train, _ = holdout(
            feats,
            pd.Timestamp(cfg.test_start),
            cfg.split["min_total_rows"],
            cfg.split["min_test_rows"],
        )
        folds = chronological_folds(
            train, cfg.cv["n_folds"], cfg.cv["min_train_rows"], cfg.cv["min_validation_rows"]
        )
        gates["chronological_folds"] = {
            "passed": True,
            "folds": [
                {
                    "validation_cycle": f.validation_cycle,
                    "train": len(f.train_idx),
                    "validation": len(f.val_idx),
                }
                for f in folds
            ],
        }
    except SplitError as exc:
        gates["chronological_folds"] = {"passed": False, "error": str(exc)}

    funnel["source_hashes"] = hashes
    funnel["config_hash"] = cfg.hash()
    funnel["gates_passed"] = all(g["passed"] for g in gates.values())
    report["gates"] = gates
    report["passed"] = funnel["gates_passed"]
    write_json(funnel, cfg.processed_dir / "funnel.json")
    write_json(report, cfg.processed_dir / "capability_report.json")
    return report
