# CHARTS.md — Benchmark Explorer: Chart Reference

This document describes, per Research Question, what the CSV contains, how each
chart is constructed, and what it is meant to answer.

---

## General processing rules (apply to every RQ)

- **Always aggregate seeds first.** We never show individual seed results.
  All charts operate on grouped/median aggregations across `seed`.
- **General → specific.** Start by looking at the overall picture across all
  models and all datasets, then look for outliers or dataset-/model-specific
  patterns that confirm or break the general trend.
- **We always compare libraries.** Color dimension = library (or library ×
  approximator combination). The comparison is always cross-library.
- **runtime and n_model_evals are equivalent.** The same comparison can (and
  should) be made with each metric separately.
- **Prefer relative metrics over absolute:**
  - `relative_mae` instead of `mean_abs_diff`
  - `relative_additivity_gap` instead of `additivity_gap`

---

## CSV schema (shared across all RQs)

| Column | Type | Description |
|---|---|---|
| `dataset` | str | Dataset name (ames_housing, adult_census, gisette) |
| `model` | str | Model family (random_forest, decision_tree, …) |
| `n_features` | int | Number of features used in the run |
| `n_samples` | int | Number of samples |
| `backend` | str | Backend / library string used by the runner |
| `library` | str | Human-readable library label |
| `computation_type` | str | `"approximation"` rows only are kept |
| `approximator` | str | Approximator algorithm (kernel, permutation, …) |
| `budget` | int | Model-evaluation budget |
| `n_eval` | int | Number of evaluation samples |
| `runtime_s` | float | Wall-clock time in seconds |
| `n_model_evals` | int | Number of model forward passes |
| `mean_abs_diff` | float | Absolute MAE vs. exact Shapley values |
| `relative_mae` | float | MAE normalised by the scale of the exact values |
| `sign_agreement` | float | Fraction of features where the sign matches exact values |
| `mean_sample_rho` | float | Spearman ρ between approximate and exact rankings (0–1) |
| `reference_backend` | str | Which backend produced the exact-value ground truth |
| `is_failure` | bool | Derived: run timed out or produced degenerate output |
| `method` | str | Derived: `library/approximator` combined label |

Quality ground truth is extracted from the `pairwise_metrics` JSON column in
new-format CSVs and backfilled into the flat columns above.

---

## RQ1 — Dimensionality

**File:** `results_config-dimensionality.csv`

**Research question:** As a user with a dataset that has many features, I want
to find the fastest model-agnostic library for high-dimensional datasets?

### Benchmark config

```yaml
seeds:        [0..9]          # 10 seeds → always aggregate
n_background: 100
n_eval:       10
budget:       512
imputer:      marginal
libraries:    [shap, shapiq, lightshap, dalex]
approximators:[kernel, permutation]   # not every lib×approx combo exists
models:       random_forest / decision_tree / linear_regularized / gradient_boosting
datasets:
  ames_housing:   n_features: [4, 16, 64]
  adult_census:   n_features: [4, 7, 14]
  gisette:        n_features: [4, 32, 256]
```

Scope: model-agnostic methods only (captum excluded). No interaction terms.

### Charts

#### 1. Runtime vs Number of Features
`fn: fig_runtime_vs_features`

- **X-axis:** `n_features` (log scale)
- **Y-axis:** `runtime_s` median per (library, approximator, n_features) group, log scale
- **Color:** library × approximator combination
- **Facets:** dataset (rows) and model (columns) — general view first, then per-dataset/model
- **What it answers:** Which library scales best as the number of features grows?
  A flat slope = good scalability. A steep slope = computational bottleneck.
- **Processing:** Group by [library, approximator, n_features, dataset, model],
  aggregate seeds → median runtime. Plot as line with markers.

#### 2. Spearman ρ vs Number of Features
`fn: fig_rho_vs_features`

- **X-axis:** `n_features`
- **Y-axis:** `mean_sample_rho` median
- **Color:** library × approximator
- **Reference line:** ρ = 0.9 (good-quality threshold)
- **What it answers:** Does approximation quality hold up as dimensionality grows?
  A declining line = the method loses accuracy for high-dimensional data.
- **Processing:** Same grouping as chart 1 but aggregating `mean_sample_rho`.

#### 3. Failure Rate Heatmap
`fn: fig_failure_heatmap_by_features`

- **Rows:** library × approximator combination
- **Columns:** `n_features` values in the data
- **Cell value:** `is_failure` mean (fraction of failed runs)
- **Color scale:** white → red
- **What it answers:** At which feature count does a method start to fail
  (timeout or degenerate output)? The first red cell in a row is the break point.
- **Processing:** Group by [method, n_features], compute `is_failure.mean()`.
  Pivot to wide format for the heatmap.

**Side panel (not in _CHARTS):** `fig_speed_ranking_at_nfeatures` — bar chart
comparing libraries at the minimum and maximum selected `n_features`.

---

## RQ2 — Approximation Accuracy

**File:** `results_config-accuracy.csv`

**Research question:** As a user using Shapley approximations, I want to know
how good the values actually are so that I can trust the explanations without
wasting too much computing time needed for exact values.

### Benchmark config

```yaml
seeds:        [0..9]
n_background: [50, 200]    # two settings — does n_background change quality?
n_eval:       10
imputer:      marginal
libraries:    [shap, shapiq, lightshap, dalex]
approximators:[kernel, permutation]
# Usable budget × n_background combinations:
#   50  × 64    (low background, low budget)
#   50  × 512   (low background, high budget)
#   200 × 512   (high background, high budget)
# WARNING: 200 × 64 can produce garbage — exclude or flag
models:       random_forest / decision_tree / linear_regularized / gradient_boosting
datasets:
  ames_housing:  n_features: [4, 12]
  adult_census:  n_features: [4, 8]
  gisette:       n_features: [4, 14]
```

Feature counts are intentionally low so exact Shapley values are computable
(used as the ground truth reference).

### Metrics of interest (to analyse in priority order)

1. `relative_mae` — primary accuracy metric
2. `relative_additivity_gap` — consistency check
3. `sign_agreement` — flag if 0 (completely wrong signs)
4. `mean_sample_rho` — rank-order quality (Spearman ρ)

All metrics should be cross-compared with `runtime_s` and `n_model_evals`.

### Charts

#### 1. Speed–Accuracy Pareto Frontier
`fn: fig_pareto(_agg(df))`

- **X-axis:** median `runtime_s`
- **Y-axis:** median `mean_sample_rho`
- **Color:** library; gray = dominated methods
- **What it answers:** Which methods are Pareto-optimal — no other method is
  both faster AND more accurate?
- **Processing:** Aggregate all runs per (library, approximator) → one point
  each. Mark Pareto front.

#### 2. Spearman ρ vs Budget
`fn: fig_budget_rho`

- **X-axis:** `budget` (model evaluation budget)
- **Y-axis:** median `mean_sample_rho`
- **Color:** library × approximator
- **Reference line:** ρ = 0.9
- **What it answers:** How much budget is needed before rank-order agreement
  plateaus? A flat line at the left = already converged at low budget.

#### 3. Runtime vs Budget
`fn: fig_runtime_vs_budget`

- **X-axis:** `budget`
- **Y-axis:** median `runtime_s`
- **Color:** library × approximator
- **What it answers:** What is the wall-clock cost of increasing budget per
  library?

#### 4. ρ Convergence Curve
`fn: fig_metric_vs_budget(df, "mean_sample_rho")`

- **X-axis:** `budget`
- **Y-axis:** `mean_sample_rho` (with spread band)
- **Color:** library × approximator
- **What it answers:** How quickly does quality converge with more computation?

#### 5. Spearman ρ Distribution
`fn: fig_distribution`

- **Chart type:** box + strip plot
- **X-axis:** library × approximator
- **Y-axis:** `mean_sample_rho` per run
- **Reference line:** ρ = 0.9
- **What it answers:** How consistent is each method? Wide spread = unreliable.

#### 6. Method Ranking by Spearman ρ
`fn: fig_leaderboard_bars(compute_leaderboard(df))`

- **Chart type:** horizontal bar chart
- **X-axis:** median `mean_sample_rho`
- **Labels:** failure rate annotated on the right (red if > 10 %)
- **What it answers:** Overall winner ranking across all selected runs.

---

## RQ3 — Neural Networks

**File:** `results_config-neural-networks-RQ3.csv`

**Research question:** As a user with a neural network, I want to know which
library is the fastest?

### Benchmark config

```yaml
seeds:        [0..9]
n_background: 200
n_eval:       100
budget:       512
imputer:      marginal
device:       cuda   # GPU run; a CPU variant exists as RQ5

libraries and their supported approximators:
  captum       → gradient_shap, deep_lift_shap   (gradient-based, NN-specific)
  shap_nn      → gradient, deep                  (gradient-based, NN-specific)
  lightshap    → kernel, permutation             (model-agnostic)
  dalex        → permutation                     (model-agnostic)
  shapiq_proxy → proxy                           (model-agnostic, ProxySHAP)
  shapiq       → kernel, permutation             (model-agnostic)

models (fixed architecture, varied type):
  mlp:         hidden_sizes [128, 64], 50 epochs
  transformer: d_model 64, 4 heads, 2 layers, 50 epochs
  cnn_1d:      64 filters, kernel 3, 50 epochs

datasets (fixed feature count):
  ames_housing:  n_features: 79
  adult_census:  n_features: 14
  gisette:       n_features: 256
```

**Note:** True Shapley values computed by shapiq (used as reference).
Feature count is fixed — the sweep is over model architecture type.

### Charts

#### 1. Runtime Ranking — Fastest to Slowest
`fn: fig_runtime_ranking`

- **Chart type:** horizontal bar chart sorted by median runtime
- **X-axis:** median `runtime_s`
- **Color:** library
- **What it answers:** Which library/approximator combination is fastest for
  neural networks? Short bar = fastest.

#### 2. Runtime Distribution per Library
`fn: fig_runtime_boxplots`

- **Chart type:** box plot (log scale)
- **X-axis:** library × approximator
- **Y-axis:** `runtime_s` (log)
- **What it answers:** How consistent is each library's speed? Wide boxes or
  high outliers = inconsistent performance.

#### 3. Speed vs Accuracy Scatter
`fn: fig_rho_vs_runtime`

- **X-axis:** `runtime_s` per run
- **Y-axis:** `mean_sample_rho` per run
- **Color:** library
- **What it answers:** Does being fast come at the cost of accuracy? Ideal =
  top-left (fast AND accurate).

#### 4. Pareto Frontier — Speed vs Accuracy
`fn: fig_pareto(_agg(df))`

- Same construction as RQ2 Pareto chart.
- **What it answers:** Which NN libraries sit on the Pareto front?

---

## RQ4 — Tree Models

**File:** `results_config-tree.csv`

**Research question:** As a user with deep/complex tree models, I want to know
which library handles extreme tree depths efficiently without hitting
computational bottlenecks or breaking points?

### Benchmark config

```yaml
seeds:        [0..9]
n_background: 100
n_eval:       10
backend_timeout_s: 600
imputer:      marginal

# Model-agnostic approximation is DISABLED for this RQ.
# Only tree-native exact backends are used.

tree_libraries:   [shap_tree, shapiq_tree, woodelf, fasttreeshap]
tree_modes:       [path_dependent, interventional]

# Pairwise (order-2) interactions, path-dependent only:
interaction_libraries:    [shapiq_tree, woodelf]
interaction_max_features: 32   # cap because of quadratic blowup

models:
  xgboost:       max_depth: [4, 8, 12, 15, 20, 50, 80]
  lightgbm:      max_depth: [4, 8, 12, 15, 20, 50, 80]
  random_forest: max_depth: [4, 8, 12, 15, 20, 50, 80]

datasets:
  ames_housing:  n_features: [4, 16, 64]
  adult_census:  n_features: [4, 7, 14]
  gisette:       n_features: [4, 32, 256, 512]
```

The complexity axis auto-detects from the CSV: `tree_depth` → `max_depth` →
`n_estimators` → `model`. Charts are built dynamically via `_build_charts(comp_col)`.

### Tab 1 — Path-dependent
Backends: `shap_tree`, `shapiq_tree`, `woodelf`, `fasttreeshap` (all
path-dependent mode).

#### 1. Failure Rate Heatmap
`fn: fig_failure_vs_complexity`

- **Rows:** library
- **Columns:** complexity value (e.g. max_depth)
- **Cell value:** fraction of failed/timed-out runs
- **What it answers:** At which depth does each library start to fail? The first
  red cell is the breaking point.

#### 2. Runtime vs Complexity
`fn: fig_runtime_vs_complexity`

- **X-axis:** complexity axis (max_depth)
- **Y-axis:** median `runtime_s` with seed spread band
- **Color:** library; rows = dataset, columns = model family
- **What it answers:** Which library scales best with tree depth? Steep slope =
  computational bottleneck. Flat = scales gracefully.
- **Note:** Same chart should also be produced with `n_model_evals` on Y.

#### 3. Spearman ρ vs Complexity
`fn: fig_rho_vs_complexity`

- **X-axis:** complexity axis
- **Y-axis:** median `mean_sample_rho`
- **Color:** library
- **What it answers:** Does approximation quality hold up at extreme depths?
  (Less relevant for exact methods but included as a correctness sanity check.)

#### 4. Method Quality Ranking
`fn: fig_leaderboard_bars(compute_leaderboard(df))`

- Median `mean_sample_rho` aggregated across the current filter selection.
- **What it answers:** Which tree library delivers the best quality overall?

### Tab 2 — Interventional
Backends: `woodelf_interventional`, `shapiq_tree_interventional`
(ignore `shap_true_value`).

Charts mirror Tab 1 charts 2 and 3 (Runtime vs depth, Runtime vs n_features).
The question is whether interventional computation costs more than path-dependent.

### Tab 3 — Interactions (order-2 pairwise)
Backends: `shapiq_tree`, `woodelf`, `fasttreeshap` — interaction order = 2.
Feature cap at `interaction_max_features = 32` (quadratic blowup).

#### Planned charts (not yet implemented):
1. **Cross-library agreement heatmap** — cells = mean `mean_abs_diff` (or
   `sign_agreement`) between each pair of libraries. One heatmap per model family.
   Answers: do the exact interaction methods agree with each other?
2. **Runtime vs n_features (log-log)** — x = n_features, y = runtime_s, color =
   library, rows = dataset. The quadratic-blowup chart.
3. **Runtime vs max_depth** — x = max_depth, y = runtime_s, color = library,
   rows = dataset, model family as filter.

---

## RQ5 — GPU vs CPU (planned)

Libraries: Captum, Woodelf (neural networks on CPU).
Same benchmark config as RQ3 but `device: cpu`.

The comparison is direct: same model, same data, runtime on CPU vs GPU.
Relevant metric: `runtime_s` and `n_model_evals`.

Not yet implemented in the explorer.
