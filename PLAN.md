# Transfer Value Predictor: implementation plan

Status: planning only. Reviewed against [SPEC.md](SPEC.md), accepted v0.5.1, on 2026-09-22. The workspace contains the two planning documents; no source snapshot, package, tests, or research results exist yet. Phase headings describe work to implement, not completed work.

## Improvements from the spec review

| Gap in the previous plan | Resolution | Spec basis |
|--------------------------|------------|------------|
| Four pasted drafts, duplicate phases, and an unclosed code fence | Keep one sequence of Phases 0 through 10 and format code, configuration, and paths explicitly | §8, §13 |
| A full feasibility funnel depended on feature code scheduled later | Start with a bounded schema probe, then build the minimum shared eligibility helpers for the exact count gate before full model development | §5.6, §13 |
| Hashes were described as a way to recover source bytes | Separate snapshot retrieval from hash verification and document immutable retrieval when available | §4.1, §5.1 |
| The holdout window and earlier feature-history coverage were underspecified | Pin the study window and metadata-derived `T`; retain earlier seasons needed for lookback | §5.6, §7.1 |
| Conflicting artifact names and ignored README figures | Define one artifact contract and commit a small `docs/results/` publication set | G5, G7, G8, §10 |
| Only the selected model had a clear final-fit path | Refit each family finalist on train, report all three, and keep the CV-selected headline model fixed | G2, §6.2 |
| Bootstrap trigger and residual sign were vague | Compute paired intervals for every core run; define cluster multiplicity and `actual − predicted` residuals | §4.3, §10.1 |
| Coefficient extraction lacked interpretation rules | Record fitted feature names/scales, explain log-target units and position contrasts, and avoid causal claims | G8, §10 |
| Phase 10 had no content | Add the ordered README contract, evidence inputs, worked leakage example, reproduction steps, and acceptance checks | §10, §15, §17 |

## Summary

Build `transfer_value` as a reproducible methodology study for paid permanent transfers involving players with sufficient recent
Premier League activity.

The finished project should let a reader:

1. Acquire a specific free Transfermarkt snapshot.
2. Verify the snapshot supports the study before building against it.
3. Verify the acquired source bytes using SHA-256 hashes. Retrieval requires the matching snapshot.
4. Construct one leakage-controlled row per qualifying transfer.
5. Train LinearRegression, Ridge, and ElasticNet models on `log1p(fee_eur)`.
6. Select models using expanding chronological validation.
7. Evaluate once on a future time holdout.
8. Compare against simple baselines.
9. Inspect coefficients, plots, uncertainty, and the five worst misses.
10. Reproduce the result from the documented configuration and lockfile.

The primary implementation should stay focused on the v_ship study. Mode B, ablations, the market-value comparator, Streamlit, and
Big-5 support should have explicit extension plans but must not complicate the core data path.

The canonical study path, after feasibility has pinned the source configuration, will be:

```bash
uv sync --locked --extra dev
uv run --locked python scripts/fetch_data.py --config config.yaml
uv run --locked tvp feasibility --config config.yaml
uv run --locked tvp pipeline --config config.yaml
```

`tvp pipeline` will run the stages after data acquisition and configuration gates pass.

Phase 0 creates only the package shell needed to inspect real source data. Phase 10 defines the publication commands and the demoted prediction example.

## Decisions fixed before implementation

### Toolchain

- Use Python 3.11 for broad wheel availability and stable support.
- Use uv as the canonical environment and dependency manager.
- Commit uv.lock.
- Keep pip install -e ".[dev]" documented as a fallback, but do not maintain two independently resolved dependency sets.
- Pin direct dependency versions in pyproject.toml; let uv.lock pin the full graph.
- Add ruff for formatting and linting. Do not add a larger static-analysis stack unless implementation reveals a concrete need.

Runtime dependencies are pandas, numpy, scikit-learn, matplotlib, joblib, typer, pyyaml, and pyarrow. Define pytest and ruff in the `dev` optional dependency extra so the uv path and `pip install -e ".[dev]"` read the same declarations. Pin versions compatible with Python 3.11 during implementation. Do not add notebooks, services, or optional demo dependencies to the required workflow.

Implementation references checked for this review: [uv lockfile checks](https://docs.astral.sh/uv/concepts/projects/sync/), [scikit-learn preprocessing leakage](https://scikit-learn.org/stable/common_pitfalls.html), and [linear coefficient interpretation](https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html). Select and lock compatible package versions during implementation rather than treating a documentation URL as a version pin.

### Data acquisition

Use the [community Kaggle dataset](https://www.kaggle.com/datasets/davidcariboo/player-scores) or the files documented by [dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets). Attribute both Transfermarkt and the dataset maintainers.

- Prefer scripted downloads from the public upstream URLs documented by dcaribou/transfermarkt-datasets.
- Support a manually downloaded Kaggle snapshot through --input-dir.
- Do not require Kaggle credentials in the core workflow.
- Keep download logic separate from ingestion. Tests should be able to run entirely against local fixtures.
- Write one source manifest containing:
    - source URL,
    - local filename,
    - download timestamp,
    - byte size,
    - SHA-256,
    - source snapshot or commit reference when available,
    - schema probe result.

- Never silently replace an existing file. A changed file must produce a new hash and a visible source-manifest update.
- Raw source files remain ignored by Git.

Implement downloads in `scripts/fetch_data.py`. Hash a temporary file before replacing the destination. If a configured hash does not match, fail without replacing the existing snapshot. Explicit source refreshes update the manifest and require feasibility to run again. Manual imports use the same validation path. Record the actual acquisition date when known; do not present an ingest timestamp as a historical download date.

### Study population

The core population is paid permanent transfers where the player has sufficient recent Premier League feature history, regardless of destination club. Fees are positive, finite, nominal EUR. Null, undisclosed, free, loan, loan-with-option, and swap-without-fee rows are excluded with counts. No fee imputation is allowed.

The implementation must never describe this as "all Premier League transfers" or "transfers into the Premier League."

### Time semantics

- Every feature row is built relative to the recorded transfer_date.
- Appearance rows require match_date < transfer_date.
- The two latest fully completed source Premier League competition seasons before the transfer are included.
- If the transfer falls inside an ongoing competition season, include that season's pre-transfer matches.
- A summer-gap transfer usually has no ongoing season and therefore uses only the two latest completed seasons.
- Season identity comes from source competition-season metadata, not an inferred July-to-June calendar bucket.
- Performance features use date-only `match_date < transfer_date` comparisons. Historical position is preferred; current position is a retrospective proxy explicitly exempt from the strict as-of claim.
- All rows sharing a transfer date remain in the same chronological validation block.
- Repeated appearances by the same player across different transfer dates are allowed. The model is estimating a future transfer for
  a player who may have appeared previously.

### Training semantics

- Fit preprocessing inside every validation fold through a single scikit-learn Pipeline.
- Use ColumnTransformer for numeric and categorical preprocessing.
- Use dense encoded output because the feature set is small and coefficient inspection is a required deliverable.
- Train directly on y_log = log1p(fee_eur).
- Convert predictions back with expm1.
- Clamp negative euro predictions to zero and record how often clamping occurred.
- Select by mean validation MAE in log space.
- Report euro-space MAE only on the untouched time test set.
- Never use the test set for feature decisions, model selection, baseline fitting, or threshold selection.

## Implementation phases

### Phase 0: repository and execution contract

Create only the shell needed for feasibility, keeping the existing planning documents:

```text
README.md
SPEC.md
PLAN.md
pyproject.toml
uv.lock
config.yaml
.gitignore
src/transfer_value/
tests/
scripts/
data/raw/.gitkeep
data/interim/.gitkeep
data/processed/.gitkeep
artifacts/.gitkeep
```

Configure the package entry point:

```toml
[project.scripts]
tvp = "transfer_value.cli:app"
```

Define development commands:

```bash
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked pytest
uv run --locked tvp --help
```

Do not make the pipeline download data implicitly. A clean checkout cannot produce real metrics without a source snapshot. The
README should state that source acquisition is the one manual or network-dependent step, after which the pipeline runs locally.

Expose `tvp --help` and `tvp feasibility --help`, load the bootstrap input paths, and return clear errors for missing config or files. Leave source IDs and `T` unresolved until inspected. Add the remaining modules as their phases need them; do not create stub production logic that returns fixture data. Use the same offline checks in CI.

### Phase 1: source acquisition and feasibility

The feasibility command must answer whether the chosen snapshot can support the study before model development.

There are two passes. First inspect headers and bounded samples to verify schema and source semantics. Then count exact eligible rows using the same normalization, labels, season, feature, and split helpers used by the production pipeline. Implement the minimum portions of the Phase 2 through 6 contracts needed for this pass here, after the schema is verified; those phases then finish the same helpers and their tests. Do not write an independent approximation of the cohort.

A sample-only report must identify its incomplete gates. `tvp feasibility` returns success only after the exact funnel, production row thresholds, and configured chronological folds pass. A source that fails schema capabilities stops immediately; the remaining package and model work wait for a passing gate.

#### Source inspection

The required source tables are appearances, transfers, players, games, and competitions from the community Transfermarkt dataset. Resolve their actual filenames and compression from the upstream manifest; do not assume example column names are authoritative.

For every required table:

- Confirm the file exists.
- Read headers without loading the complete dataset.
- Record row counts and inferred dtypes.
- Check required identifiers.
- Check null rates for fields needed by the label and cohort.
- Sample distinct values for transfer-type and loan-related fields.
- Detect whether fees are numeric or encoded as strings.
- Detect whether the source has a stable transfer identifier.
- Detect whether competition-season identifiers and season completion dates exist.
- Detect whether historical position is available.
- Detect whether appearances can be joined to games and competitions.

Distinguish sampled diagnostics from full counts. Once the schema is usable, scan the required data for exact totals, exclusions, and date coverage. Save one machine-readable `data/processed/capability_report.json` and print a concise explanation in the CLI.

#### Capability gates

Feasibility must fail with a clear report if the source cannot support:

- player identity,
- transfer date,
- transfer fee,
- paid-versus-free classification,
- permanent-versus-loan classification,
- appearance date,
- player-to-game joins,
- competition filtering,
- stable transfer identity without unresolved collisions,
- source season identity and usable completion/coverage evidence,
- birth-date or birth-year coverage and appearance statistics needed for the cohort.

Historical position is not a hard failure because the spec allows a flagged retrospective proxy. The report must state whether that
proxy will be used.

#### Count funnel

Write data/processed/funnel.json with counts and drop reasons:

```text
raw transfer rows
valid player IDs
valid transfer dates
fee field present
positive reported fee
classified permanent transfer
classified non-loan transfer
resolved unique transfer identities
Premier League-active candidates
rows with sufficient lookback minutes
rows with valid per-90 features
rows with usable age
rows with usable position
final supervised rows
rows in train
rows in test
```

Every filter must have a named count. Give each rejected row one first-failure reason so sequential exclusions reconcile; record overlapping diagnostics separately. Train and test are two branches of the final cohort and must sum to it. Record the report's source/config hashes and whether every gate passed.

The feasibility command should also write a human-readable capability report explaining:

- which source columns were selected,
- how transfer types were interpreted,
- how fees were parsed,
- whether transfer identity is stable,
- which competition IDs represent the Premier League,
- which seasons are available,
- whether historical position exists,
- what fallback policies are active.

Only after this report passes should the project proceed to full pipeline integration and model construction. Production defaults remain at least 200 supervised rows and 40 test rows, plus the configured fold minimums. If coverage is too small, revise the source/study window from coverage evidence before any model scoring and rerun feasibility. Do not silently lower the gates or choose a split based on test performance.

### Phase 2: canonical configuration

Create a typed configuration loader around config.yaml.

The configuration should contain:

```yaml
data:
  raw_dir: data/raw
  interim_dir: data/interim
  processed_dir: data/processed
  artifact_dir: artifacts
  competition_name: Premier League
  competition_ids: null
  source_seasons: null
  schema_mapping: {}
  transfer_type_mapping: {}
  fee_currency: null

study:
  season_ids: null
  transfer_start: null
  transfer_end_exclusive: null
  min_minutes: 90
  include_cards: false
  include_cups: false
  position_policy: historical_then_proxy
  prediction_as_of: null

split:
  test_start: null
  min_total_rows: 200
  min_test_rows: 40

cv:
  n_folds: 3
  min_train_rows: 80
  min_validation_rows: 20

runtime:
  random_state: 42
  bootstrap_replicates: 5000
  bootstrap_seed: 42
  bootstrap_confidence: 0.95
```

Rules:

- competition_ids, source_seasons, and split.test_start remain unset until feasibility resolves them.
- Training and evaluation fail if required values are unset.
- Fixture tests use a separate fixture configuration with intentionally lower row gates.
- Production gates must never be weakened to make fixture tests pass.
- Configuration is copied into every run artifact.
- Store a canonical hash of the effective configuration beside metrics.

`data.source_seasons` includes the earlier seasons required for feature lookback. `study.season_ids` identifies the transfer study window used to choose the last two holdout seasons. Pin the transfer-date interval separately so later rows in an updated snapshot cannot silently extend the holdout. Store URLs and per-file hashes in `data/raw/SOURCE.md`; never infer a verified snapshot from a matching filename alone.

Validate dates and mappings at the command boundary. Core `min_minutes` is at least 90. Production `cv.n_folds` is an explicit integer from three to five; never silently reduce it. Fixture-only configs may use smaller counts. No training or evaluation may start until Q1, Q2, the study window, and `T` are resolved.

### Phase 3: normalized data layer

Create private normalized domain representations behind the ingestion boundary. Do not expose raw CSV column names throughout the
package.

The normalized tables should include the following concepts:

#### Transfer

- transfer_id
- player_id
- transfer_date
- from_club_id
- to_club_id
- fee_eur
- raw_fee
- transfer_type
- is_loan
- label_exclusion_reason
- source-row provenance

#### Appearance

- appearance_id
- game_id
- player_id
- match_date
- competition_id
- competition_season_id
- minutes
- goals
- assists
- optional cards
- historical position if available

#### Player

- player_id
- name
- date of birth or birth year
- current position
- historical position fields when available

#### Competition season

- competition ID
- season ID
- season name
- first-match date when available
- last-match or completion date when available
- completion status and the source evidence for it

Normalize:

- IDs to one stable internal type.
- Dates to date-like values with one documented convention.
- Missing numeric statistics to the source-defined missing value policy.
- Fees to positive numeric EUR values only after parsing and validation.
- Transfer-type values to a small internal classification enum.

Preserve raw values for diagnostics. Never overwrite the raw fee or raw transfer-type value with a parsed value.

Use strings for canonical IDs and date-only semantics for matches and transfers. Reject or count invalid/missing dates without timezone shifts. Required performance statistics must be finite and non-negative. Missing minutes, goals, or assists are not automatically zero; resolve source null semantics during feasibility and report exclusions when required values are unavailable.

#### Transfer identity and duplicates

Use a stable source transfer ID if one exists.

If no stable ID exists:

1. Preserve a fingerprint of the full source row and the original row number for audit.
2. Build a deterministic candidate identity from normalized player ID, date, clubs, fee, transfer type, and the source-row fingerprint. Do not use row position as identity.
3. Count and remove exact duplicate source records only under the documented deduplication policy.
4. Check collisions in the natural player/date/club key as well. Adding a hash must not hide conflicting descriptions of the same apparent event.
5. Keep genuinely distinct repeated transfers separate when source evidence distinguishes them; reject unresolved identity conflicts.
6. Record duplicate, ambiguity, and conflict counts in the funnel.

The generated ID must remain stable across runs on the same snapshot.

#### Fee parsing

Implement a dedicated parser that:

- accepts the exact encodings found during feasibility,
- converts supported EUR representations to numeric values,
- rejects unknown currency symbols or ambiguous strings,
- distinguishes null, undisclosed, free, and malformed values,
- never converts missing or undisclosed values to zero,
- records parse failures by raw value.

No model stage should have to parse fees.

Transfer type and fee availability are separate facts. A positive fee does not prove that a row is permanent. Unrecognized type values require an explicit mapping or exclusion policy before feasibility can pass. Keep the raw value and the exclusion reason. Do not interpret optional add-ons as guaranteed fees.

### Phase 4: ingestion, cleaning, and labels

tvp ingest should:

1. Load the configured raw files.
2. Validate columns and uniqueness constraints.
3. Normalize identifiers, dates, fees, and enums.
4. Join only through IDs and validate expected join cardinalities.
5. Verify the pinned Premier League competition IDs against metadata; do not silently replace them.
6. Write typed interim Parquet files.
7. Write or update SOURCE.md.
8. Write an ingestion manifest containing source hashes, schema mapping, row counts, and package version.
9. Use temporary files and atomic renames so an interrupted run cannot leave a half-written artifact.

The label builder should apply the predicates in a visible, ordered sequence:

```text
player_id exists
transfer_date exists
fee parses successfully
fee_eur is finite and > 0
transfer is not free
transfer is not a loan
transfer type is classified as permanent
```

Cohort eligibility is applied after the lookback feature computation because sufficient recent PL minutes cannot be known from the
transfer table alone.

Write label_stats.json with:

- paid rows,
- free rows,
- loan rows,
- undisclosed rows,
- malformed fee rows,
- unsupported transfer-type rows,
- duplicate rows,
- rows rejected for missing identity or date.

Assert unique canonical `player_id` and `game_id`. Count exact duplicate appearance records and reject conflicting appearance IDs so joins cannot multiply minutes or goals. Define `clean.py` as the normalization owner and `labels.py` as the single label-inclusion owner; feasibility calls these same functions. `seasons.py` owns season resolution, and `features.py` owns aggregation.

### Phase 5: leakage-safe feature builder

Implement the feature builder as pure functions over normalized tables and configuration.

For each qualifying transfer:

1. Resolve the ongoing source PL competition season at the transfer date, if any; a summer gap has none.
2. Select the two latest fully completed PL seasons before the transfer.
3. Add pre-transfer appearances from an ongoing season when applicable.
4. Filter every appearance with match_date < transfer_date.
5. Filter to Premier League competition IDs.
6. Aggregate:
    - goals,
    - assists,
    - minutes,
    - appearances with minutes greater than zero,
    - optional cards.

7. Compute goals per 90 and assists per 90 only when total minutes meet min_minutes.
8. Drop the row if either required per-90 value is unavailable.
9. Compute floor age at the transfer date from birth date. Otherwise use transfer year minus birth year and mark `age_is_approximate`; drop and count rows with neither.
10. Resolve position:
    - lookback appearance position first,
    - historical position as-of the transfer date second,
    - current player position as a retrospective proxy last.

11. Mark whether the position is historical or proxy.
12. Emit source coverage fields for diagnostics.

Map positions explicitly to `{GK, DF, MF, FW}`. Use the mode from eligible lookback appearances with a fixed category-order tie-breaker, then a historical player record dated before the transfer, then the current proxy. Drop rows with no resolved position. Record `position_source` as `historical_appearance`, `historical_player_record`, or `current_proxy` and report proxy counts for the full cohort, train, and test.

Resolve season completion from the competition's metadata or a verified full match schedule, not the player's final appearance. A truncated snapshot's last observed match does not establish that a season has completed. Retain enough earlier source seasons for the two-season lookback; exclude and count transfers whose required source coverage is unavailable. If a named calendar-window fallback is ever used under the spec, report it as a distinct configured policy, never as completed source seasons. Do not silently switch to rolling 365-day windows.

Every feature row should carry:

```text
transfer_id
player_id
transfer_date
feature_cutoff_date
lookback_season_ids
lookback_match_count
lookback_minutes
position_source
position_is_proxy
fee_eur
fee_log1p
```

The final feature table must have exactly one row per qualifying transfer ID.

Also retain the model feature values, first/last contributing match dates, age provenance, and source-row counts. Sort by player ID, match date, game ID, and appearance ID before aggregation. Keep audit fields and labels separate from the explicit model-input allowlist; selecting every numeric column would leak IDs, dates, or fee-derived fields.

Do not impute missing performance to zero. A player with no qualifying lookback is an excluded cohort member, not a zero-performance
player.

Add explicit tests for:

- appearance on the transfer date being excluded,
- appearance after the transfer date being excluded,
- July transfer in the gap between seasons,
- ongoing season transfer,
- extended 2019/20 season,
- insufficient minutes,
- duplicate appearance records,
- repeated players with different transfer dates,
- current-position proxy flagging.

### Phase 6: split and validation system

Create one split module that owns every temporal partition.

#### Final holdout

```text
train: transfer_date < T
test:  transfer_date >= T
```

Choose `T` from source metadata at the start of the earlier of the last two study seasons, as required by `SPEC.md` §7.1. Record the exact source boundary and rationale; do not hard-code July 1 or confuse first/last observed matches with a documented season boundary. Pin the study window and `T` from coverage before any model selection or test scoring.

Validate:

- both partitions are non-empty,
- no transfer ID appears in both,
- all test dates are greater than or equal to T,
- all train dates are less than T,
- T is recorded in metrics and README output.

Require at least 200 final supervised transfers and 40 test transfers under the production defaults. Persist season IDs, train/test date ranges, transfer counts, unique-player counts, repeated test-player counts, and player overlap across train and test. Repeated players are allowed; repeated transfer identities are not.

#### Expanding chronological folds

Use a custom date-block splitter or a carefully configured TimeSeriesSplit wrapper.

The splitter must:

- sort by transfer date,
- treat all rows with the same date as one block,
- create expanding training windows,
- validate only on future date blocks,
- enforce minimum train and validation sizes,
- produce exactly the configured number of folds, three by default and no more than five,
- fail clearly when the train window is too small,
- never use future rows to preprocess an earlier validation fold.

Do not use GroupKFold for primary model selection.

Retain grouped validation only as an optional sensitivity analysis of player overlap. Since GroupKFold may train on later dates, it is not an isolated estimate of the effect of seeing a player before and does not replace chronological validation.

Persist fold boundaries, counts, and transfer IDs in `artifacts/cv_results.json`, with the boundary/count summary in `artifacts/metrics.json` so a reviewer can verify the split without rerunning the code.

### Phase 7: modeling and selection

Build a model factory that produces the same preprocessing pipeline for all estimators.

Numeric inputs:

```text
goals
assists
minutes
appearances
goals_per90
assists_per90
age
optional cards
```

Categorical input:

```text
position
```

Pipeline:

```text
ColumnTransformer
  numeric: StandardScaler
  categorical: OneHotEncoder(handle_unknown="ignore", sparse_output=False)
estimator
```

Models:

```text
LinearRegression
Ridge(alpha in {0.1, 1, 10, 100})
ElasticNet(alpha in {0.1, 1, 10}, l1_ratio in {0.15, 0.5, 0.85})
```

Use the effective scikit-learn version recorded by uv.lock. Keep compatibility shims out of the core unless the selected version
requires them.

Model selection procedure:

1. Sort the training table by transfer date and deterministic transfer ID.
2. Generate expanding date-block folds.
3. For each model family and hyperparameter combination:
    - clone a fresh pipeline,
    - fit it on each fold's training portion,
    - transform the target to log space,
    - compute validation MAE in log space,
    - record fold metrics.

4. Select the lowest mean log-MAE, averaging the fold MAEs with equal fold weights.
5. Use fixed deterministic tie-breaking:
    - lower mean log-MAE,
    - lower fold standard deviation,
    - fixed estimator order,
    - fixed parameter order.

6. Refit the best candidate from each family on all training rows. Mark the overall CV-selected model before opening the holdout.
7. Never inspect test metrics until selection is complete.
8. Persist the fitted pipeline with its feature schema and target metadata.

The model artifact must include:

- fitted pipeline,
- selected estimator,
- selected hyperparameters,
- feature column list,
- categorical vocabulary,
- target transform,
- training date range,
- source manifest hash,
- configuration hash,
- package version,
- random seed.

Use an explicit numeric/categorical allowlist and `ColumnTransformer` remainder dropping. The categorical encoder keeps all categories with `drop=None`. Record convergence failures and nonfinite predictions; do not silently score a failed fit. Pin solver settings and use seed 42 where the estimator accepts randomness.

Write `artifacts/models/linear.joblib`, `ridge.joblib`, and `elastic_net.joblib`, with the selected pipeline at `artifacts/model.joblib`. Store all fold scores, candidate means/standard deviations, parameters, and tie-break results in `artifacts/cv_results.json`. Write the selected model's fitted feature names, coefficients, intercept, training scales, and category metadata to `artifacts/coefficients.csv` and the manifest for Phase 10. Training writes selection results, not final holdout metrics.

### Phase 8: baselines and evaluation

Implement baselines as train-only predictors:

1. global train median fee,
2. global train mean fee,
3. position-specific train median with global-median fallback,
4. optional contemporaneous Transfermarkt market value comparator.

The comparator must never enter the Mode A feature matrix. It is a benchmark with its own missingness and timestamp rules.

Evaluate the three family finalists and the three required baselines on the same test rows. Keep the headline model selected by CV even if another family has a lower test error. Count unseen-position fallback uses. Optional comparators with missing values get a separate matched-cohort table.

Required metrics:

- euro MAE,
- median absolute error,
- euro RMSE,
- log-space MAE,
- optional euro R²,
- row count,
- date range,
- number of predictions clamped at zero.

Use `prediction_eur = maximum(expm1(prediction_log), 0)` for every euro metric and demo. Preserve raw log predictions for log-MAE, and define residuals as `fee_eur − prediction_eur`. Reject nonfinite outputs rather than hiding overflow as clipping. Score baseline log-MAE by applying `log1p` to their non-negative euro predictions. No MAPE is used.

Report:

- raw predictions,
- expm1 predictions,
- clamped predictions,
- residuals,
- transfer IDs,
- player IDs and names,
- position source,
- lookback minutes,
- transfer date,
- actual fee,
- predicted fee,
- absolute error.

#### Uncertainty

Compute the paired intervals for every core run, which covers the spec's small-test requirement without an arbitrary "near minimum" threshold:

- resample the same test units for the model and every baseline,
- compute the model-minus-baseline MAE difference within each replicate,
- use 5,000 seeded bootstrap replicates,
- resample by player cluster when a player appears more than once,
- otherwise resample transfer rows,
- report 95% percentile confidence intervals,
- describe the interval as uncertainty in the observed test-set difference, not proof of general superiority.

The evaluation output must state when a baseline comparison is too uncertain to call a win.

When clustering, sample unique test player IDs with replacement and include every transfer for each sampled player, preserving a player's multiplicity if drawn repeatedly. Compute transfer-weighted MAE on the resulting rows. Use identical sampled units for the model and baselines in each replicate. An `isin` filter would erase repeated draws and is incorrect. Keep the fitted models fixed; these intervals describe the observed holdout difference and do not include retraining or future time-drift uncertainty.

#### Worst misses

Generate `artifacts/worst_misses.json` and `artifacts/worst_misses.md` for the selected model's five largest absolute euro errors, with transfer ID breaking ties. Save the underlying test rows in `artifacts/test_predictions.parquet`.

Each case should include:

- player,
- transfer date,
- destination and source when available,
- reported fee,
- predicted fee,
- signed residual,
- position and whether it was a proxy,
- lookback minutes,
- key feature values,
- a short factual description.

Do not claim that a residual proves negotiation, hype, club wealth, or contract length caused it. Those remain hypotheses.

#### Plots

Generate deterministic figures:

- predicted versus actual fee with identity line,
- residual versus predicted fee,
- residual distribution,
- optional log-space predicted versus actual plot.

Use fixed figure dimensions, labels, units, and seed-independent ordering. Save the input metric summary beside each figure.

Save figures as `artifacts/figures/predicted_vs_actual.png`, `residuals_vs_predicted.png`, and `residual_distribution.png`. Include the identity/zero reference lines, clear EUR units, and the full outlier range. Optional zooms must accompany the full plot.

#### Artifact and CLI contract

Every command accepts `--config`; ingestion and feasibility also accept `--input-dir`. Missing prerequisites, unresolved source mappings, source/config/schema mismatches, or failed data gates return nonzero with a specific error. The public commands are `ingest`, `feasibility`, `build-features`, `train`, `evaluate`, `predict`, and `pipeline`.

`tvp pipeline` runs feasibility, ingestion, feature construction, training, and evaluation after explicit data acquisition. It stops on the first failure and never downloads data or prompts for a prediction. Standalone stages must enforce the same prerequisites, including feature/source identity at training and evaluation.

| Producer | Required output |
|----------|-----------------|
| Acquisition/ingest | `data/raw/SOURCE.md` with exact URLs, snapshot references, dates, byte sizes, hashes, and attribution |
| Feasibility | `data/processed/capability_report.json`, `data/processed/funnel.json` with gate status |
| Ingest | Typed canonical tables in `data/interim/`, plus `data/interim/manifest.json` |
| Build-features | `data/processed/features.parquet`, `label_stats.json`, final `funnel.json`, and a manifest |
| Train | Model files, `artifacts/cv_results.json`, `artifacts/coefficients.csv`, and training metadata |
| Evaluate | `artifacts/metrics.json`, `test_predictions.parquet`, worst-miss JSON/Markdown, figures, and completed `artifacts/manifest.json` |
| Report builder | README numeric blocks and the selected publication files in `docs/results/` |

The run manifest records schema version, run ID, code revision when available, Python/dependency versions, effective config and hash, lockfile hash, source hashes, seed, model identity, feature schema, target/retransformation policy, split, row counts, and completion status. Attach the same run ID to derived outputs and record artifact hashes.

Write outputs to temporary files or a staging directory before promoting a completed set. Keep the previous valid set on failure. Downstream commands must reject mixed or incomplete artifacts. Repeated execution with unchanged inputs must reproduce the substantive outputs; wall-clock timestamps are not part of result identity.

### Phase 9: prediction CLI

tvp predict must use the persisted model and feature builder rather than reimplementing feature logic.

Behavior:

- --player resolves exact normalized names first.
- Ambiguous names produce a nonzero exit and list candidate IDs.
- --date is optional syntactically but must resolve to either:
    - the configured prediction_as_of, or
    - an explicit error if no configured date exists.

- Never default to the machine's current date.
- Print:
    - player,
    - as-of date,
    - position and position source,
    - lookback seasons,
    - lookback minutes,
    - predicted reported fee,
    - model name and artifact version.

- Label predictions as hypothetical estimates, not observed fees.
- Refuse prediction if the requested date predates the available feature data or cannot produce the required lookback.
- Keep prediction output separate from test evaluation output.

Provide `--player-id` to disambiguate names. Name resolution is only a CLI convenience; feature joins still use IDs. Reuse the saved model's preprocessing, feature schema, and target transform, and verify source/config compatibility before predicting. Reject dates beyond verified source coverage. A forecast-style demo must also use a model trained only on transfers earlier than its requested as-of date.

### Phase 10: README and research presentation

Finish the research narrative in the order required by `SPEC.md` §10. This phase depends on a verified real-data run from Phases 1 through 8 and a working prediction command from Phase 9. A written plan or a fixture run does not count as completed research.

The README should answer the study question with measured errors, then explain the method and its limits. A model that loses to a baseline is a valid result. Do not change the split, cohort, or features after seeing the holdout to manufacture a better headline.

#### Deliverables and evidence

Use one evaluated run for every published table, figure, coefficient, and case study. Keep a `run_id` in the manifest and derived summaries so results from different runs cannot be mixed.

| Deliverable | Authoritative input | Published location |
|------------|---------------------|--------------------|
| Baseline and model metrics, split, proxy coverage, uncertainty | `artifacts/metrics.json` | README tables and `docs/results/metrics.json` |
| Source and run identity | `data/raw/SOURCE.md`, `artifacts/manifest.json` | Source attribution and `docs/results/manifest.json` |
| Cohort funnel and exclusion counts | `data/processed/funnel.json`, `data/processed/label_stats.json` | README summary and `docs/results/funnel.json` |
| Prediction and residual figures | `artifacts/figures/` from the saved test predictions | `docs/results/figures/` |
| Five worst misses | `artifacts/worst_misses.json`, `artifacts/worst_misses.md` | Five README writeups |
| Coefficients and training scales | `artifacts/coefficients.csv` | README explanation and `docs/results/coefficients.csv` |
| Model-selection details | `artifacts/cv_results.json` | `docs/results/cv_results.json`, linked from the method |

Keep raw tables, full prediction tables, interim data, and model binaries ignored. Commit the small publication outputs in `docs/results/`, the README, and `data/raw/SOURCE.md`. Ignore rules must allow the parent directory of `SOURCE.md`; ignoring all of `data/` would also hide that file. README images must render from a clean checkout before anyone downloads data or runs the model.

Add a small `scripts/build_report.py` that reads the completed run and formats the numeric README blocks and selected publication files. It must not fit models or recompute metrics. Keep human-written case narratives outside generated blocks. Reject incomplete runs or mismatched run identities instead of combining stale artifacts. A `--check` mode should detect stale generated content without changing files.

#### README order and content

Keep the following order. The opening should fit the research question, the result table, and a link to the five worst misses into a short first screen.

1. **Title, subtitle, and question.** Use "Transfer Value Predictor" with "A methodology study of PL-active players' reported transfer fees." Ask how well recent PL performance, age, and position explain reported fees on a later time holdout. Prefer "reported fee" throughout.
2. **Scope.** Explain that each row is a paid permanent transfer of a player with sufficient recent PL activity, regardless of destination. Distinguish reported fees from Transfermarkt market values. State that newcomers without qualifying PL history are outside the cohort.
3. **Results first.** Show the three train-only baselines beside LinearRegression, Ridge, and ElasticNet. Mark the model selected by chronological CV before evaluating the test set. Print the exact `T`, study seasons, `n_train`, and `n_test` next to the table.
4. **Plots.** Embed predicted versus actual fees with the identity line, residuals versus predictions, and the residual distribution. Include a short reading of each plot, units, sample size, model, and residual sign. Show the full fee range, including elite outliers.
5. **Five worst misses.** Include five individually written cases from the selected model's largest absolute euro errors. Keep the artifact's deterministic ordering and link to the supporting run.
6. **Method.** Explain the cohort, features, season lookback, strict appearance cutoff, position exception, target transform, models, chronological folds, and final split. State that log-MAE selects the model while euro-MAE evaluates it.
7. **Leakage checklist and example.** Include the checks and worked row below, with links to the protecting tests. A checkbox is checked only after that behavior has been verified.
8. **Coefficients in plain English.** Show the selected model's coefficient table with the units and limitations described below.
9. **Data provenance and filters.** Attribute Transfermarkt and the dataset maintainers. Link the Kaggle dataset, upstream repository, and committed source manifest. Include the funnel, exclusion reasons, date coverage, selected competition-season IDs, and position-proxy percentage.
10. **Quickstart.** Give one canonical $0 path for environment setup, explicit snapshot acquisition, feasibility, and the local pipeline. Explain the manual import fallback in a short note.
11. **Prediction CLI.** Place the optional prediction example after results, failures, and the study workflow. Use a real resolvable player and a fixed supported date from the verified snapshot. Label the output a hypothetical reported-fee estimate.
12. **Optional analyses, if shipped.** Keep ablations, the market-value comparator, and Mode B separate and identify their cohort and target. Omit this section if none shipped.
13. **What I would not claim.** Include the explicit claim limits below.
14. **Limitations.** Include every limitation in `SPEC.md` §10.1 and the observed coverage limits from this run.
15. **Reproduce.** Record the seed, Python and dependency versions, lockfile and config hashes, source hashes, code revision, run ID, and exact commands. Treat this as the audit record for the quickstart; avoid another competing setup path.

#### Results table and uncertainty

Use this column contract. Populate it only from the real evaluation artifact; do not publish placeholder numbers or fixture metrics as findings.

| Method | Selected by CV | Test MAE (€) | Median AE (€) | RMSE (€) | Test log-MAE | Test rows |
|--------|----------------|--------------|---------------|----------|--------------|-----------|

The six required rows are train median, train mean, train median by position, LinearRegression, Ridge, and ElasticNet. Every row uses the same test transfers. Put CV log-MAE and selected hyperparameters in a separate table so validation and holdout metrics cannot be confused.

For the selected model, show a second table with `ΔMAE = model MAE − baseline MAE` and its 95% paired bootstrap interval for each required baseline. Negative values favor the model. Report 5,000 replicates, seed 42, the resampling unit, and unique test-player count. Repeated players require player-cluster resampling. If an interval crosses zero, describe the observed difference and the uncertainty without claiming a clear improvement.

Do not use MAPE, "X% accurate," or R² as the headline. Report how many predictions were clamped at zero. State that direct `expm1` retransformation does not correct retransformation bias or guarantee an expected euro fee.

#### Five worst-miss writeups

Each case must contain:

- Player name and ID, transfer ID, recorded transfer date, and clubs if the source supplies them.
- Reported fee, predicted fee, absolute error, and signed residual using `actual − predicted`. A positive residual means the model underpredicted.
- Lookback seasons, minutes, age, position, position source, and the relevant performance aggregates.
- One short paragraph describing what the row and model actually show.
- Any possible explanation clearly labeled as a hypothesis. Contract length, negotiation, buying-club wealth, and hype are not observed causes in the core feature set.

Select by descending absolute error, with transfer ID as the tie-breaker. Do not replace inconvenient cases with recognizable players. Five transfers may include a repeated player; do not silently change the ranking to obtain five different names.

#### Leakage checklist and worked example

The shipped README must let a reader verify that:

- [ ] Every contributing appearance satisfies `match_date < transfer_date`, including exclusion of same-day matches.
- [ ] Source competition-season metadata defines the two completed seasons and any ongoing-season prefix.
- [ ] The current-position fallback is explicitly exempt from the strict as-of claim, with total, train, and test proxy percentages.
- [ ] No destination-club strength, fee-derived fields, or market value enters the model inputs.
- [ ] Training joins use IDs, and transfer identities do not overlap across train and test.
- [ ] Scalers and encoders fit only on each chronological fold's training rows.
- [ ] Baselines use training labels only, including the position-median fallback.
- [ ] Model and hyperparameter selection use only the training window; headline numbers use `transfer_date >= T`.

Use a clearly hypothetical example, mirrored in `tests/test_leakage.py`:

| Field | Allowed construction | Rejected construction |
|-------|----------------------|-----------------------|
| Player and recorded transfer date | Alex Example, 2023-07-15 | Same player and date |
| Completed-season lookback | Eligible PL matches in 2021/22 and 2022/23 | An arbitrary July-to-June bucket described as a completed season |
| Transfer-day and later appearances | Excluded | Goals on 2023-07-15 or in August 2023 enter aggregates |
| Market value dated 2023-08-01 | Excluded from Mode A inputs | Used to predict the July fee |
| Current player position | Allowed only with a retrospective-proxy flag | Described as known before the deal without historical evidence |

The test must show that adding same-day or later appearances leaves the performance features unchanged. A separate feature-schema assertion must reject market value and fee-derived inputs. The example illustrates the policy and is not a measured research result.

#### Coefficient interpretation

Extract feature names and coefficients from the fitted preprocessing pipeline, never from an independently rebuilt column list. Store the intercept, numeric training means and scales, category vocabulary, and estimator parameters with the coefficient output.

For a nonconstant numeric feature, its coefficient describes the change in predicted `log1p(fee_eur)` for a one-training-standard-deviation increase, holding the other encoded inputs fixed. Before output clipping, exponentiating that coefficient gives a multiplier on `1 + predicted fee`, not an additive euro amount. A constant feature has no meaningful one-standard-deviation interpretation and should be labeled accordingly.

The planned encoder keeps all position categories. With an intercept, unregularized position coefficients are not individually identifiable. Explain contrasts between observed categories using their coefficient difference; do not invent a dropped reference category or call one coefficient an absolute position premium. Record the encoding policy, including how unseen categories are handled.

Goals, assists, minutes, appearances, and per-90 rates are related. Coefficient signs and magnitudes describe the fitted model conditional on its other inputs; they are not causal effects or a reliable ranking of football skills. Do not multiply already standardized coefficients by the training standard deviation a second time.

#### Claim limits and limitations

The "What I would not claim" section must state that this study does not establish:

- A player's true value or a recommended transfer price.
- Coverage of all transfers into the Premier League or players without qualifying PL history.
- Knowledge available before negotiations or agreement, since the cutoff is the recorded transfer date.
- Strictly historical position information for rows using the current-position proxy.
- A causal explanation for individual residuals or coefficient signs.
- General superiority from a small holdout or a noisy baseline difference.

The limitations section must cover reported or estimated fees and possible add-ons, exclusion of undisclosed/free/loan transfers, selection bias toward known-fee PL-active players, missing source history, retrospective position, nominal EUR without inflation adjustment, and errors on elite outliers. Treat negotiation, contract length, buying-club wealth, and hype as hypotheses. Transfermarkt market value is a comparator if included; its relative performance does not prove or disprove leakage.

#### Reproduction and presentation checks

Document this path after the source mapping, exact URLs, hashes, season window, and `T` have been pinned:

```bash
uv sync --locked --extra dev
uv run --locked python scripts/fetch_data.py --config config.yaml
uv run --locked tvp feasibility --config config.yaml
uv run --locked tvp pipeline --config config.yaml
uv run --locked python scripts/build_report.py --config config.yaml
uv run --locked python scripts/build_report.py --config config.yaml --check
```

These are planned commands, not commands that currently run in this workspace. The README must verify them against the implemented CLI before publication. The acquisition helper is the explicit network step. A manual snapshot imported through `--input-dir` must receive the same hash and schema validation. A mutable upstream URL plus a hash does not guarantee that the matching old bytes remain downloadable; document an immutable snapshot location when available and record any retrieval limitation.

Explain `pip install -e ".[dev]"` only as a compatibility fallback. It does not consume `uv.lock` or establish the same transitive dependency resolution. Keep `uv sync --locked` canonical because it fails if the project requires a lockfile update; `--frozen` skips that freshness check.

For an optional research talk, reuse the same evidence in this sequence: question and cohort, baseline results, temporal method and leakage example, plots, five misses, coefficient interpretation, limitations, reproduction. A slide deck, PDF export, or demo application is an optional presentation format and does not block the README deliverable.

#### Phase 10 acceptance checks

- [ ] The README follows all 15 sections of the content contract, omitting only unshipped optional analyses.
- [ ] The opening contains the research question, baseline/model table, and a link to the five worst misses.
- [ ] Every published number and figure comes from one identified real-data run; no fixture result or placeholder appears as a finding.
- [ ] All three baselines and all three models use the same holdout; the selected model is identified by training-window CV.
- [ ] Exact `T`, study-season IDs, train/test counts, proxy percentages, and bootstrap details are visible.
- [ ] Five factual case writeups, the leakage example, coefficient units, claim limits, and all required limitations are present.
- [ ] README images and relative links work from a clean checkout without generated local artifacts.
- [ ] The documented acquisition and reproduction path has succeeded in a fresh environment against the pinned snapshot.
- [ ] Regenerating the report and running its `--check` mode confirms that published tables and figures match their artifacts.
- [ ] Source attribution, per-file SHA-256 hashes, effective config, lockfile identity, and code revision are recorded.
- [ ] Offline tests, lint, formatting, and CLI smoke checks pass. A baseline loss does not block completion.

## Verification throughout implementation

Write the relevant behavior tests alongside each phase. Use a separate fixture config with smaller row gates, and never download data in unit tests.

| Area | Required evidence |
|------|-------------------|
| Ingest and labels | Positive permanent fee included; free, loan, loan-with-option, undisclosed, malformed, and noncash rows excluded with counts; missing required columns fail |
| Identity and joins | Unique canonical players/games; duplicate appearances cannot multiply aggregates; exact transfer duplicates counted; conflicting transfer identities rejected |
| Features and seasons | Per-90 arithmetic, zero and insufficient minutes, orphan exclusion, exact age and birth-year fallback, historical/proxy positions, summer gaps, extended 2019/20, and same-day/post-transfer exclusion |
| Split and selection | Boundary at `T`, no transfer overlap, whole date blocks, expanding folds, configured minimums, fold-local preprocessing, and deterministic candidate selection |
| Evaluation | Hand-checkable metrics and baselines, unseen-position fallback, residual sign, clipping, deterministic five-miss ranking, and paired bootstrap with repeated-player multiplicity preserved |
| CLI and reporting | Offline fixture pipeline, missing-prerequisite failures, prediction ambiguity and date errors, artifact compatibility, and numeric report agreement with saved results |
| Reproduction | Two runs with the same source bytes, config, lockfile, and seed reproduce row identities, split, selected parameters, and metrics within a recorded numeric tolerance |

Compare meaningful outputs rather than joblib or figure byte identity across platforms. Timestamps are audit metadata and are excluded from deterministic result comparisons. Rerun the fresh-environment, real-data workflow for final acceptance; fixtures prove behavior, not real source feasibility.

## Optional extensions after the core study

Keep `SPEC.md` O1 through O5 outside the core acceptance checklist:

- O1: Performance-only and age/position ablations, then an optional club-tier feature with documented pre-transfer availability. Use the same cohort for feature comparisons and disclose any coverage change. Position-proxy sensitivity is optional too.
- O2: A separate market-value target with timestamp rules and separate artifacts. Add `--target fee|mv` only when Mode B is implemented.
- O3: An as-of Transfermarkt market-value comparator. Report its coverage and score both it and the fee model on the matched subset; preserve the core full-holdout table.
- O4: A local Streamlit demo after the CLI and research report work.
- O5: A Big-5 configuration with verified competition/season mappings and separate results.

GroupKFold remains an optional training-window sensitivity analysis. It cannot establish chronological forecasting performance because it may train on later transfers to validate earlier ones. Do not use any optional result to reselect the headline model on the same test set.

## Open questions to resolve during feasibility

| Question | Required resolution |
|----------|---------------------|
| Q1: Which source IDs identify Premier League matches? | Verify against competition metadata and pin in `config.yaml` before label/cohort construction |
| Q2: Which fields and values classify transfers and fees? | Pin the column mapping, transfer-type mapping, fee encoding, and EUR evidence before building labels |
| Q3: What are the study window and `T`? | Use source season metadata, retain earlier lookback seasons, and pin dates before training or test scoring |
| Q4: Can the snapshot be identified and reacquired? | Record URLs, snapshot reference, acquisition dates, byte sizes, and per-file SHA-256; verify any immutable retrieval path |
| Q7: Is historical position available? | Record provenance and the fallback policy; publish the measured proxy proportion |
| Do the exact eligible cohort and chronological folds meet production gates? | Persist the full funnel and fold sizes; stop before modeling if they do not |

Q5 and Q6 concern optional Mode B and club-tier definitions. Resolve them only if those extensions are pursued. No source values, dates, or results are assumed by this plan.
