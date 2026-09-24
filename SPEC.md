# Spec: Transfer Value Predictor (PL-active players)

**Subtitle:** Methodology study — how wrong is a linear fee model for **PL-active players’ realized fees** under a leakage-safe temporal split?

**Status:** Accepted (v0.5.1) — summer-gap / as-of proxy / bootstrap / stale scrub  
**Package / CLI name:** `transfer_value` / `tvp`  
**Owner:** Parth Dubey (aevum .)  
**Constraint:** $0 total (no paid APIs, no cloud ML credits, no paid proxies)  
**Intent:** Full build from day one. Portfolio methodology project — not a valuation product.  
**Build call:** ~2–3 weeks if optionals stay optional; worth building as rigor showcase.

**Study population (LOCKED):**  
Paid **permanent** transfers of players with **sufficient recent Premier League appearances** in the feature lookback, **regardless of destination club**. This is **not** “all transfers into the PL” (that cohort would exclude newcomers with no PL history and needs a separate Mode if ever pursued).

**North-star framing:**  
> Among players with recent PL minutes, what share of their **reported** transfer fees can performance + age + position explain under a leakage-safe temporal split — and where does the model fail?

Not: “AI that prices players.” Prefer: “how wrong is a linear fee model, and why.”

---

## 0. Changelog

| Ver | Date (IST) | Change |
|-----|------------|--------|
| 0.1 | 2026-09-19 | Brief initial draft from locked stack |
| 0.2 | 2026-09-19 | Merge Head SWE architecture + Stress Test portfolio stress |
| 0.3 | 2026-09-19 | Head SWE R2: single lookback default; July-1 season bounds; assists_per90 parity; pin Q1/Q2/T before train |
| 0.4 | 2026-09-19 | Stress Test R2: soften beat-baselines gate; README hero = question/failures; limitations negotiation noise; optionals stay optional |
| 0.5 | 2026-09-19 | Owner review: PL-active cohort; expanding chronological folds; source season IDs; feasibility funnel first; soften ceiling/causal claims; hashes + lockfile; summary wording |
| 0.5.1 | 2026-09-19 | Summer-gap lookback rule + fixtures; position proxy exempt from strictly-as-of + % proxy; paired/cluster bootstrap; scrub oracle-ceiling + date-only SOURCE |

### Decisions locked in v0.2 (were open or wrong in v0.1)

| Topic | v0.1 | v0.2 (locked) |
|-------|------|----------------|
| Row grain | player-season (+ optional fee join) | **One row per paid permanent transfer** |
| Feature cutoff | end of season S before fee window | **As-of transfer date:** only appearances with `date < transfer_date` (prior completed seasons preferred; same-season only if match date &lt; transfer date) |
| Primary split | season holdout on player-season | **Time split on `transfer_date < T` / `>= T`** |
| Package path | `src/tvp/` | **`src/transfer_value/`** per Head SWE |
| Hero metric | mixed | **Test MAE (€ after expm1)** + log-MAE for model pick; **never** MAPE; R² not hero |
| Framing | educational predictor | **Methodology / failure analysis** showcase |

### Decisions locked in v0.5

| Topic | Before | v0.5 (locked) |
|-------|--------|----------------|
| Cohort | Ambiguous (PL appearances only) | **PL-active players** with enough lookback PL minutes; destination unrestricted |
| Model-selection CV | GroupKFold primary | **Expanding chronological folds** inside train; GroupKFold optional sensitivity |
| Season definition | July–June called “completed PL seasons” | **Source competition-season IDs** (+ completion/last-match dates); July bins only if renamed as calendar windows |
| Position | Prefer `players.position` | Prefer **historical** (lookback mode); current record = retrospective proxy if needed |
| Implementation order | Scaffold first | **Feasibility count funnel first** |
| TM market value | “Oracle ceiling” | **Comparator only** — beating it ≠ proof of leakage |
| Residual causes | Negotiation/etc. as expected drivers | **Hypotheses**, not established causes |
| Reproducibility | Date stamp OK | **sha256 of downloaded bytes** + dependency **lockfile** required |

### Decisions locked in v0.5.1

| Topic | Lock |
|-------|------|
| Summer-gap lookback | Two latest **completed** source PL seasons before `D`, plus ongoing season matches with `match_date < D` |
| Position vs as-of | Performance features strictly as-of `D`; position proxy **explicitly exempt**; report % proxy rows |
| Uncertainty | Paired bootstrap ΔMAE on same test indices; **player-cluster** resampling when players repeat |
| Stale scrub | No “oracle ceiling”; SOURCE requires sha256 (no date-only provenance) |

---

## 1. One-sentence summary

Build a reproducible $0 pipeline that joins Transfermarkt community CSVs into leakage-safe transfer-level features for **PL-active players**, trains Linear / Ridge / ElasticNet on `log1p(fee_eur)`, compares them to naive baselines on a time-held-out test set, and documents where linear performance stats fail.

---

## 2. Problem statement

### 2.1 Portfolio gap this fills

Author’s public work (Gungnir, ModelRouter, Lunex, Recall, Factory Auto-BYOK) signals systems / agents / full-stack. This project fills the **classical supervised ML hygiene** gap: wrangling, labels, leakage, baselines, temporal evaluation, honest residuals.

### 2.2 Question answered

Given pre-transfer Premier League performance and demographics for **players with sufficient recent PL appearances**, how well (and how poorly) do regularized linear models predict their **reported** transfer fees (any destination)?

**Claim discipline:** Features use information available before the **recorded** `transfer_date`. That is not a claim that the model predicts fees before negotiation concludes (agreement can precede the recorded transfer date).

### 2.3 What has to be true for this to be worth building (Stress Test)

1. Reader learns ML hygiene, not that fees are “predictable.”  
2. Label = realized / reported fee ≠ Transfermarkt market value (unless Mode B is explicitly separate).  
3. Free / loan / undisclosed handled explicitly — never as €0.  
4. One README command path, $0, seeded, reproducible.  
5. Scope finishable polished in ~2–3 weeks beside other repos.  
6. Spec optimizes for **rigor a skeptic can’t poke in 2 minutes**, not model novelty.

---

## 3. Goals and non-goals

### 3.1 Must ship (G)

| ID | Goal |
|----|------|
| G1 | CLI path: `ingest → build-features → train → evaluate → predict` (plus `pipeline`) |
| G2 | Three models under identical preprocessing: LinearRegression, Ridge, ElasticNet |
| G3 | Target `y = log1p(fee_eur)`; reports use `expm1` |
| G4 | Time-based test split with hard-coded cut `T` |
| G5 | Naive baselines in README **beside** model MAE (compare; publish miss as finding if not beaten) |
| G6 | Leakage checklist + automated `test_leakage.py` + one worked example row in Spec/README |
| G7 | Predicted-vs-actual + residuals plots; **5 worst-miss case studies** in README |
| G8 | Coefficient table with plain-English interpretation |
| G9 | `pytest` green on fixtures; **no network in unit tests** |
| G10 | Schema assertions at ingest (fail loud) |
| G11 | Determinism: `random_state`, sorted joins before groupby |
| G12 | Honest limitations + “What I would not claim” section |
| G13 | Feasibility count funnel before full scaffold; `SOURCE.md` with **sha256** of each downloaded file |
| G14 | Dependency lockfile (`uv.lock` or fully pinned `requirements.txt`) committed |

### 3.2 Non-goals (N)

| ID | Non-goal |
|----|----------|
| N1 | Paid football-data.org Deep Data / any paid API |
| N2 | StatsBomb / openfootball / football.db in core path |
| N3 | Live Transfermarkt scraping in CI or as required path |
| N4 | Neural nets, GBMs, AutoML |
| N5 | Betting, investment, or “player valuation product” framing |
| N6 | SaaS deploy, auth, multi-user hosting |
| N7 | Streamlit as substitute for CLI/tests (optional demo only) |
| N8 | Agent wrappers around the model for résumé spice |
| N9 | Multi-league kitchen sink (default = PL only; Big-5 only if config flag and time allows) |
| N10 | Claiming high R² or “X% accurate” |

### 3.3 Optional adds that raise signal (Stress Test) — ship if time

**Not in v_ship Acceptance / critical path.** Do not block “done” on these. If pulled into musts, expect 4+ weeks — don’t.

| ID | Add |
|----|-----|
| O1 | Ablation table: performance-only vs +age/position vs +prior-season club tier proxy |
| O2 | Mode B section: predict TM market value **separately** to show label literacy (never as free feature for fee model) |
| O3 | TM market value as **comparator baseline** (not a feature, not a mathematical ceiling): MAE if you scored contemporaneous value |
| O4 | Local Streamlit demo after CLI is solid |
| O5 | Big-5 league flag (default remains PL-only) |

---

## 4. Success criteria

### 4.1 Engineering done

1. Clean clone → `uv sync` / `pip install -e ".[dev]"` with **lockfile** → `tvp pipeline` (or documented steps) produces `artifacts/metrics.json` + figures with pinned seed.  
2. No paid keys required.  
3. Raw dumps **not** committed; download/ingest writes `data/raw/SOURCE.md` with URL, date, and **sha256** per file (hashes identify bytes even if upstream is mutable / not archived).  
4. `pytest` green offline.  
5. `tvp --help` works.  
6. Count funnel artifact `data/processed/funnel.json` from feasibility step.

### 4.2 Methodology done (hiring reader)

1. Temporal split + leakage note survive a 2-minute skeptic skim.  
2. Baselines present; linear is compared honestly to naive baselines and optional TM-value **comparator** (not framed as crushing an “oracle ceiling”).  
3. Plots + coefficients explained in plain English.  
4. Failure analysis (5 misses) present.  
5. README quality matches author’s best agent repos.

### 4.3 Acceptance metrics (portfolio-honest)

On **test** set (`transfer_date >= T`), after `expm1`:

| Metric | Role |
|--------|------|
| MAE (€) | **Headline** model quality |
| Median AE (€) | Robustness to outliers |
| RMSE (€) | Report, secondary |
| MAE(log1p) | Model selection among Linear/Ridge/ElasticNet |
| R² (€) | Optional only; never hero |

**Baseline comparison (success target, not a ship blocker):**

Attempt to beat on test MAE (€):

- (a) train-set median fee  
- (b) train-set mean fee  
- (c) position-median fee (computed on train only)  

If the model **does not** beat them, that is still an **Acceptable shipping outcome** — publish the gap as a finding with diagnosis (sparse labels, negotiation noise, selection bias, etc.). Do not metric-hack (change split, impute fees, add leaky features) to force a win.

Optional comparator (not required to beat; **not** a mathematical ceiling): contemporaneous TM market value MAE — shows label gap / another benchmark. Beating it does **not** automatically imply leakage.

**Uncertainty:** If `n_test` is near the minimum (~40), report a confidence interval on the **model − baseline MAE difference** using a **paired bootstrap**: resample the **same** test transfer indices for model and baselines each replicate, then take percentile CI on ΔMAE. When the same `player_id` appears more than once in the test set, use **player-cluster** (block) resampling so correlated transfers are not treated as independent. Do not over-claim a “win” from a noisy Δ.

**Test set size:** after first real ingest, set minimum `N_test` in config. If below gate, widen window or document limit — **do not fake**. Placeholder until EDA: refuse train if labeled `n < 200` total or `n_test < 40`.

**Forbidden primary metrics:** MAPE.

---

## 5. Data

### 5.1 Primary source

| Field | Value |
|-------|-------|
| Dataset | [davidcariboo/player-scores](https://www.kaggle.com/datasets/davidcariboo/player-scores) (Kaggle) |
| Upstream / mirrors | [dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets); public R2 gzip CSVs / DuckDB per upstream README |
| Files required | `appearances.csv`, `transfers.csv`, `players.csv`, `games.csv`, `competitions.csv` |
| Optional | `player_valuations.csv`, `clubs.csv` (ablation / Mode B / club tier) |
| Redistribution | **Do not commit raw dumps.** Attribute Transfermarkt + dataset maintainers in README and `data/raw/SOURCE.md` |

Pin exact download URL(s), snapshot date, and **sha256 of each downloaded file** in `SOURCE.md` at first successful ingest (required; hashes identify bytes even when upstream is mutable). Date-only provenance is **not** accepted.

### 5.2 Rejected for core

| Source | Why |
|--------|-----|
| football-data.org Free | No scorers/squads/minutes without paid Deep Data |
| StatsBomb Open Data | Sparse PL open seasons; event grain; no fees; different IDs |
| openfootball / football.db | Fixtures grain; no fees |

### 5.3 Column dictionary (assert at ingest)

> Final column names must be verified against the downloaded snapshot; ingest **fails** if required columns missing. Names below match common transfermarkt-datasets schema — adjust `config.SCHEMA` once and keep assertions as source of truth.

**`appearances` (required):**  
`appearance_id`, `game_id`, `player_id`, `player_club_id`, `date` (or join via `games.date`), `goals`, `assists`, `minutes_played`, `yellow_cards`, `red_cards`

**`transfers` (required):**  
`player_id`, `transfer_date` (or equivalent), `from_club_id`, `to_club_id`, `transfer_fee` / `fee` (EUR numeric), transfer type / loan indicators as present in snapshot (`transfer_type`, `is_loan`, etc. — **document exact encoding after first ingest**)

**`players` (required):**  
`player_id`, `name`, `date_of_birth` / `birth_year`, `position`, uniqueness on `player_id`

**`games` / `competitions`:**  
IDs needed to filter Premier League domestic league matches only.

### 5.4 Premier League filter (appearances / cohort eligibility)

- Resolve `competition_id` for **Premier League** from `competitions` at runtime; pin ID(s) in `config.yaml` after first download.  
- **Appearances for features and cohort:** only matches in that competition (default **exclude** domestic cups / UCL unless `include_cups: true`).  
- Filter appearances by **competition**, not by club alone (avoids cup noise).

### 5.4.1 Study cohort (LOCKED — critical)

**In scope:** paid permanent transfers where the player has **sufficient recent PL appearances** in the lookback window (same `min_minutes` / per90 drop rules as features — default: enough minutes that per90 fields are non-null, i.e. `minutes >= 90` in lookback unless config raises the bar).

**Destination:** **unrestricted** (PL → PL, PL → abroad, etc. all allowed if PL-activity gate passes).

**Out of scope unless a future separate Mode:** “transfers into the PL” defined by destination club. That Mode would need destination eligibility and must **acknowledge missing newcomers** with no PL history.

**Title/question language:** say **PL-active players’ realized fees**, not “Premier League transfers,” unless destination eligibility is added.

### 5.5 Label rules (Mode A — fee) — SQL-equivalent

Include transfer row if **all** hold:

```text
fee_eur IS NOT NULL
AND fee_eur > 0
AND NOT is_loan
AND transfer_type NOT IN (free, loan, loan_with_option, swap_without_fee)  -- exact set pinned post-ingest
AND player_id IS NOT NULL
AND transfer_date IS NOT NULL
AND player has sufficient PL lookback appearances (cohort gate; see §5.4.1 / §5.6)
```

**Never:**

- Treat null / undisclosed as 0  
- Treat free as 0 for training  
- Impute fees  
- Require `to_club` or `from_club` to be PL (destination unrestricted)

Excluded cohorts may be counted in `label_stats.json` / `funnel.json` for README honesty (selection bias).

**Currency:** EUR only; assert. **Inflation:** nominal EUR (no deflator in v_ship); state that in README.

**Grain:** **one supervised row per qualifying transfer** (same player may appear multiple times). Split must not duplicate the same transfer across train/test. Prefer a stable `transfer_id` if the snapshot provides one; else composite key `(player_id, transfer_date, from_club_id, to_club_id)`.

### 5.6 Feature rules (as-of / lookback)

For each labeled transfer at date `D` for `player_id=P`:

1. **Lookback window (LOCKED DEFAULT — only one; summer-gap safe):** aggregate appearances for P with `match_date < D` from:
   - the **two latest source PL competition-seasons that are fully completed before `D`**, plus
   - if a PL competition-season is **ongoing** at `D`, that season’s matches with `match_date < D`.
   Do **not** resolve “the season containing D” via July–June bins (ambiguous in the summer gap between seasons). A July transfer typically has **no** ongoing season — only the two latest completed seasons before `D`.
   **Season identity / completion:** use dataset competition-season IDs and completion (or last match date). Counterexample fixture target: extended **2019/20** (ran into July 2020). If a calendar window is ever used as fallback, name it a **calendar window**, never a completed season. No rolling-365d alternative.
2. Aggregates: `goals`, `assists`, `minutes`, `appearances` (games with minutes &gt; 0), optional cards.  
3. `goals_per90 = 90 * goals / minutes` and `assists_per90 = 90 * assists / minutes` if `minutes >= 90` else **null** for that per90 field; if either per90 is null under the default policy, **drop the row** (do not zero-fill). Config may raise `min_minutes` above 90.  
4. `age`: floor age in years at `D` from DOB when present; else `transfer_year - birth_year`.  
5. `position` (historical first; **as-of exemption**):
   - **Preferred (strictly as-of):** mode of position from lookback PL appearances if the schema provides it; else any historical position field known as-of `D`.  
   - **Fallback (retrospective proxy, NOT strictly as-of):** current `players.position` mapped to `{GK, DF, MF, FW}`. May reflect a post-transfer position change. **Explicitly exempt** from the “features strictly as-of `transfer_date`” claim. Report **% of supervised rows using proxy** in README / metrics. Optional sensitivity: drop proxy rows or drop position.  
   - If still missing → **drop**.  
6. **Forbidden features:** destination club strength, post-`D` appearances, contemporaneous or post-deal market value, fee-related fields, anything known only after the deal.  
7. **Date semantics:** `D` = recorded `transfer_date` in the dataset. Model estimates use pre-`D` info for performance features; they do **not** claim pre-agreement / pre-negotiation foresight.

### 5.7 Join orphans

If no qualifying appearances in lookback → **drop** transfer from supervised set; count in stats. **Do not impute zeros** (zeros lie).

### 5.8 Joins

- Join **only** on `player_id` / game/competition IDs.  
- **Never** join on player name for training.  
- CLI `predict --player` may resolve names for UX (exact match first; optional rapidfuzz later).

### 5.9 Worked leakage example (required in README)

| Field | Good row | Bad (leaky) row |
|-------|----------|-----------------|
| Player | Example: hypothetical “Alex Example” | same |
| Transfer date | 2023-07-15 | 2023-07-15 |
| Features include | PL goals Aug 2022–May 2023 | Goals from Aug 2023 at new club |
| Market value on 2023-08-01 | **Not used as feature** | Used as feature while predicting July fee |
| Outcome | Allowed | **Rejected by feature cutoff + checklist** |

Automated test mirrors the bad row and asserts goals ignore post-transfer appearances.

### 5.10 Mode B (optional, separate)

Predict `log1p(market_value_eur)` with valuation timestamp ≤ as-of rule. Same leakage discipline. Metrics **never mixed** into Mode A tables. Purpose: label literacy + comparator discussion (not a ceiling).

---

## 6. Modeling

### 6.1 Pipeline

`sklearn.pipeline.Pipeline` + `ColumnTransformer`:

- Numeric (`StandardScaler`): goals, assists, minutes, appearances, goals_per90, assists_per90, age [, cards]  
- Categorical (`OneHotEncoder(handle_unknown="ignore")`): position  

Estimator ∈ {LinearRegression, Ridge, ElasticNet}.

### 6.2 Grid (train window only)

| Model | Grid |
|-------|------|
| LinearRegression | — |
| Ridge | `alpha ∈ {0.1, 1, 10, 100}` |
| ElasticNet | `alpha ∈ {0.1, 1, 10}`, `l1_ratio ∈ {0.15, 0.5, 0.85}` |

**Objectives (state explicitly):**

- **Model selection:** mean CV **MAE on log1p(fee)** inside the train window.  
- **Headline evaluation:** **MAE on euro fee** after `expm1` on the time holdout.  
These are different objectives; do not pretend log-MAE selection optimizes euro-MAE.

**CV inside train (LOCKED):** **expanding chronological folds** on `transfer_date` (e.g. fold k trains on the earliest portion of train, validates on the next time block; never train on later transfers to validate earlier ones).  

**Optional sensitivity only:** `GroupKFold` by `player_id` within train — report if useful; **not** the selection criterion.

Final claimed numbers = **time test** (`transfer_date >= T`) metrics only.

### 6.3 Baselines (required)

Computed on train labels only; scored on test:

1. Predict train median fee (€)  
2. Predict train mean fee (€)  
3. Predict train median fee **by position**  
4. Optional comparator: TM market value at/near transfer (**comparator**, not ceiling)

### 6.4 Ablations (O1, if time)

1. Performance only (G/A/minutes/per90)  
2. + age + position  
3. + prior-season club tier proxy (e.g. club PL table finish or club median fee — **must still be pre-`D`**)

---

## 7. Split strategy

### 7.1 Primary (headline)

```text
train: transfer_date <  T
test:  transfer_date >= T
```

- **`T` default:** ISO cut on `transfer_date` aligned to the **start of the last two source PL competition-seasons** in the configured window (often near 1 July, but pinned from season metadata — not assumed equal to “season completed”). Exact `T` in `config.yaml` after feasibility.  
- Spec/README must print `T`, competition-season list, and `n_train` / `n_test`.

### 7.2 Model selection inside train

Expanding chronological folds (§6.2).  

### 7.3 Optional sensitivity

`GroupKFold` by `player_id` within train — stability check only; not selection; not headline.

### 7.4 Rejected as sole strategy

Random ShuffleSplit; player-only split without time; GroupKFold as the only selection mechanism (can train on later transfers to score earlier ones).

### 7.5 Determinism

`random_state: 42` everywhere; sort by `(player_id, transfer_date, …)` before groupby/aggregations.

---

## 8. Architecture

### 8.1 Repo layout (Head SWE — required)

```text
transfer-value-predictor/          # repo root (folder name OK; package is transfer_value)
  README.md
  Spec.md
  pyproject.toml
  config.yaml
  .gitignore                       # data/** raw dumps, *.joblib, .venv, __pycache__, .streamlit
  data/
    raw/                           # gitignored; SOURCE.md committed
    interim/                       # gitignored typed parquet
    processed/                     # gitignored matrices + feature docs
  src/transfer_value/
    __init__.py
    cli.py                         # typer: ingest | feasibility | build-features | train | evaluate | predict | pipeline
    config.py                      # loads config.yaml; schema constants
    ingest.py                      # load + schema asserts → interim; sha256 SOURCE.md
    feasibility.py                 # count funnel + schema capability report
    clean.py                       # ids, PL filter, fee/loan/free hygiene
    features.py                    # as-of aggregates + per-90 + age/position
    labels.py                      # paid permanent only; log1p helper
    split.py                       # time/player split — single source of truth
    model.py                       # Pipeline + grid
    evaluate.py                    # metrics € + log; baselines; plots; worst misses
    predict.py
    plots.py
    io.py                          # parquet/joblib
  tests/
    fixtures/                      # 3–5 players; known joins/fees; leak cases
    test_clean.py
    test_features.py
    test_labels.py
    test_split.py
    test_leakage.py
    test_cli_smoke.py
  notebooks/                       # optional EDA only
  artifacts/                       # gitignored: models/, figures/
```

No `src/utils` dumping ground.

### 8.2 CLI matrix

| Command | Exit 0 when | Nonzero when |
|---------|-------------|--------------|
| `tvp ingest` | interim parquet written; SOURCE.md with sha256 stamped | missing files/columns |
| `tvp feasibility` | funnel.json + schema capability report; exit 0 if gates pass | cannot classify loan/fee/identity or n too small |
| `tvp build-features` | processed feature/label matrix written; label_stats.json | empty after filters below gate |
| `tvp train` | model.joblib + metrics partial | n too small; schema drift |
| `tvp evaluate` | metrics.json + figures + worst_misses.md fragment | missing artifacts |
| `tvp predict --player NAME [--date ISO]` | printed prediction | unresolved player |
| `tvp pipeline` | all stages OK | any stage fail |

### 8.3 Dependencies

**Required:** pandas, numpy, scikit-learn, matplotlib, joblib, typer, pyyaml, pyarrow, pytest  

**Optional:** duckdb, seaborn, streamlit, rapidfuzz, requests/httpx  

**Pin** versions in `pyproject.toml` **and** commit a lockfile (`uv.lock` preferred, or fully pinned `requirements.txt`).

---

## 9. Testing plan

### 9.1 Fixtures must cover

- Paid permanent with fee  
- Loan (excluded)  
- Free / null fee (excluded)  
- minutes = 0 / minutes &lt; 90 → null per90 → drop  
- Appearance **after** transfer date (must not enter features)  
- Orphan transfer (no prior appearances → dropped)  
- **Summer-gap July transfer:** lookback uses two latest completed seasons before `D` only (no bogus “season containing D”)  
- **Extended 2019/20 season:** completion/last-match after typical July boundary; lookback/completion logic still correct  
- Position proxy row vs historical position row (proxy flagged; not claimed as strictly as-of)  

### 9.2 Tests

| File | Asserts |
|------|---------|
| `test_clean.py` | filters, null fee ≠ 0 |
| `test_labels.py` | cardinality / inclusion predicates |
| `test_features.py` | per-90 math; as-of cutoff |
| `test_split.py` | boundary `T`; no transfer id in both sides |
| `test_leakage.py` | post-transfer goals ignored |
| `test_cli_smoke.py` | typer invoke on fixtures / micro parquet, offline |

**No network in unit tests.** Manual ingest from Kaggle documented separately.

---

## 10. README content contract (order)

**Hero (above the fold):** north-star question + baselines table + pointer to worst misses.  
**Forbidden as hero:** “€XXm predicted fee” flex screenshots or lead demos.  
`tvp predict` stays in the CLI for demo; in README it is **last** among usage sections (results/failures first).

1. **H1 + subtitle** (“methodology study”) + lead question  
2. What this is / is not  
3. **Results first:** baselines vs model MAE table (no “X% accurate”)  
4. Plots (pred vs actual with y=x; residuals)  
5. **Five worst misses** with short narrative  
6. Method: features, as-of rule, log target, models, **split `T`**  
7. **Leakage checklist** (skeptic-scannable)  
8. Coefficients in plain English  
9. Data provenance + filters + attribution  
10. Quickstart (one path, $0)  
11. Predict CLI (demoted; optional demo only)  
12. Ablations / Mode B if shipped (clearly optional)  
13. **What I would not claim**  
14. Limitations (see §10.1)  
15. Reproduce (seed, versions, config)  

Tone: calm, precise. Prefer “reported fee” over “true value.”

### 10.1 Limitations bullets (required)

- **Hypotheses for large residuals** (not established causes): negotiation, contract length, buying-club wealth, hype — intentionally out of the feature set. Residuals alone do not prove which factor dominated.  
- Selection bias: only known-fee permanent transfers of **PL-active** players (excludes many newcomers / non-PL paths).  
- Fees are reported/estimated (Transfermarkt), not ledger truth; recorded `transfer_date` may follow agreement date.  
- Position may be a retrospective proxy when historical position is unavailable.  
- Nominal EUR (no inflation adjustment).  
- Linear models will miss elite outliers; that is a finding, not a bug to hide.  
- TM market value is a **comparator**, not a ceiling; beating or losing to it is not automatic proof of leakage or purity.

---

## 11. Assumption killers (monitor in build)

| Killer | Response in Spec/build |
|--------|------------------------|
| Label noise (add-ons, undisclosed) | Exclude undisclosed; discuss add-ons in limitations |
| Leakage | As-of rules + tests + checklist |
| Wrong target (MV as fee) | Mode A vs B separated |
| Shuffle split | Forbidden as headline |
| “Linear can’t predict fees” | Expected for outliers; frame as failure analysis |
| Duplicate IDs / position drift | Assert unique player_id; position policy |
| Selection bias (known-fee only) | label_stats + README |

---

## 12. Ethics / compliance

- Attribute Transfermarkt + dataset authors.  
- No full raw dump in git.  
- No betting / investment advice.  
- Future StatsBomb notebook (if any) isolated + license note.

---

## 13. Implementation order

0. **Feasibility first (before trusting fixtures/schema assumptions):** download snapshot → schema probe (permanent vs loan, fee field, transfer identity) → **count funnel**  
   `raw transfers → paid permanent → cohort eligible (PL-active) → sufficient appearances → train/test`  
   Write `funnel.json` + capability notes. **Stop** if the snapshot cannot support the Spec.  
1. Minimal package scaffold + lockfile only as needed to run feasibility  
2. Fixture CSVs aligned to **verified** schema + clean/label/feature/split/leakage tests  
3. `ingest` + schema asserts + SOURCE.md (**sha256**)  
4. Cohort filter + label_stats / funnel  
5. Feature as-of builder (source season IDs; historical position preference)  
6. Expanding chrono CV train + baselines + evaluate plots + worst misses + MAE Δ uncertainty if small n_test  
7. Predict CLI (demoted in README)  
8. README to contract  
9. Optional ablations / Mode B / Streamlit  

---

## 14. Open questions (resolve at first ingest / EDA)

**Gate:** `tvp train` and `tvp evaluate` **must not run** until Q1, Q2, and Q3 (`T`) are pinned in `config.yaml`. Ingest/build-features may run earlier; schema asserts fail loud if competition/type columns are unresolved when labels are built.

| # | Question | Owner | Gate |
|---|----------|-------|------|
| Q1 | Exact PL `competition_id`(s) in snapshot | Head SWE at ingest | **Before train** |
| Q2 | Exact transfer type column + enum values | Head SWE at ingest | **Before train** |
| Q3 | Final `T` ISO date from source season metadata for holdout | Brief + Head SWE | **Before train** |
| Q4 | Confirm sha256 written for every downloaded file | Head SWE | **Required at ingest** (v0.5) |
| Q7 | Historical position field availability in snapshot | Head SWE at feasibility | Document proxy if absent |
| Q5 | Ship Mode B in same CLI (`--target fee\|mv`) or separate script? | Default: `--target` flag if O2 done | Optional |
| Q6 | Club tier proxy definition for O1 | Only if ablation pursued | Optional |

---

## 15. Acceptance checklist

- [ ] $0 required deps only  
- [ ] Spine = transfermarkt community tables only  
- [ ] One row per paid permanent transfer; free/loan/undisclosed excluded with counts  
- [ ] Performance features strictly as-of `transfer_date`; position proxy **explicitly exempt** + % proxy reported  
- [ ] No name joins in training  
- [ ] Cohort = PL-active sufficient lookback minutes; destination unrestricted; titled accordingly  
- [ ] Time split with documented `T`; expanding chrono CV for selection; GroupKFold optional only  
- [ ] log-MAE selects; euro-MAE evaluates (stated)  
- [ ] Source competition-season IDs for lookback (no fake “completed season” July bins)  
- [ ] Baselines in README beside model MAE (miss = valid finding)  
- [ ] Leakage section + worked example + `test_leakage.py`  
- [ ] Five worst-miss writeups; residual causes as hypotheses  
- [ ] “What I would not claim”  
- [ ] Feasibility funnel before full build; sha256 SOURCE.md; dependency lockfile  
- [ ] pytest offline green; CLI help works  
- [x] Spec status → **Accepted** (v0.5.1)  
- [ ] Q1/Q2/`T` pinned in `config.yaml` before first `train`/`evaluate`  
- [x] Stress Test R2 + owner v0.5 / v0.5.1 locks  
- [ ] Paired player-cluster bootstrap for MAE Δ when n_test small  

---

## 16. Collaboration log

| Date (IST) | Who | Note |
|------------|-----|------|
| 2026-09-19 | Brief | v0.1 draft |
| 2026-09-19 | Head SWE | Repo layout, pipeline stages, failure modes, time split, metrics, tests, vetoes |
| 2026-09-19 | Product Idea Stress Test | Portfolio framing, assumption killers, README contract, scope adds, rigor bottleneck |
| 2026-09-19 | Brief | v0.2 merge; re-review requested |
| 2026-09-19 | Head SWE | R2: Accepted for implement; lookback single default; July-1 bounds; assists_per90 parity; pin Q1/Q2/T before train |
| 2026-09-19 | Brief | v0.3 tweaks applied |
| 2026-09-19 | Product Idea Stress Test | R2: Accept after soften §4.3; README hero; limitations; keep O2/O4/Big-5 optional |
| 2026-09-19 | Brief | v0.4 Accepted |
| 2026-09-19 | Owner review | Cohort, chrono CV, season IDs, funnel-first, soften ceiling/causal claims, hashes+lockfile |
| 2026-09-19 | Brief | v0.5 applied |
| 2026-09-19 | Owner review | Summer-gap lookback; as-of vs position proxy; paired/cluster bootstrap; stale scrub |
| 2026-09-19 | Brief | v0.5.1 applied to project Spec.md |

---

## 17. Appendix — Leakage checklist (ship in README)

- [ ] No post-transfer appearances in features  
- [ ] No destination-club features  
- [ ] No fee or fee-derived fields as inputs  
- [ ] No market value as feature in Mode A  
- [ ] Train metrics ≠ reported test metrics  
- [ ] Baselines fit on train only  
- [ ] Split defined by `transfer_date` vs `T`  

---

**End of Spec v0.5.1 — Accepted**
