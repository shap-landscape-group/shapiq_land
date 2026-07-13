# Figure traceability: US 2 — Approximation accuracy & budget efficiency

**CSV:** `results/rq2_accuracy.csv` (2016 approximation rows) · **Code:** `pages/rq2_accuracy.py` → `shared/charts.py`
**Swept:** budget {64, 512} · n_background {50, 200} · seeds {0,1,2} · 3 datasets · 4 models · 4 libraries · 2 approximators · M = 4–14
**Legend:** 🟢 match · 🟠 close/partial · 🔴 different/missing

---

## Figure 7: Absolute Error Decay Profiles

| | By report we could show… | By code we currently show… |
|--|--------------------------|----------------------------|
| **Plot** | 🔴 Grouped bar chart, 100% percentile intervals `errorbar=("pi",100)` | Point + line plot, no intervals |
| **X-axis** | 🟢 budget N ∈ {64, 512} | budget, linear |
| **Y-axis** | 🟢 `relative_mae` vs exact oracle | median `relative_mae`, linear |
| **Lines / layout** | 🔴 Bars = lib×approx; facet cols=`n_background`, rows=`dataset` | Lines = method; n_bg as line style (solid/circle=50, dashed/diamond=200); no facets |
| **Read** | Interaction effect: n_bg 200 + budget 64 = error penalty | Same interaction visible via dashed-vs-solid gap; UI warns for n_bg=200 × budget=64 |
| **Steps** | 1. CSV: `budget`, `relative_mae` vs `shap_true_value`, `library`, `approximator`, `n_background`, `dataset`<br>2. Filter approximation rows<br>3. Facet cols=`n_background`, rows=`dataset`<br>4. Bar per lib×approx at each budget, pi-100 interval over seeds/models<br>5. Map x=`budget`, y=`relative_mae` | 1. Extract `relative_mae` from `pairwise_metrics["shap_true_value"]`<br>2. Exclude failures (`relative_mae > 1.0` or NaN)<br>3. Click metric toggle → `relative_mae`; n_bg/budget filters<br>4. Group `method × budget × n_background` → median over seeds/models/datasets<br>5. `fig_budget_quality_lines(df, "relative_mae")` |
| **Verdict** | **Report** — facets isolate the n_bg interaction; bars + pi-100 show spread | Code shows same effect compactly but hides dataset differences and all variance |

| Aspect | Report | Implementation |
|--------|--------|----------------|
| 🔴 Plot type | Grouped bar chart + pi-100 error bars | Point + line plot, no error bars |
| 🟢 X-axis | budget N ∈ {64, 512} | budget, linear |
| 🟢 Y-axis | relative_mae | median(relative_mae), linear |
| 🟢 Lines/Color | Library × Approximator | method (library × approximator) |
| 🟠 Faceting | cols=n_background, rows=dataset | None — n_bg encoded as line style + UI filters |
| 🟠 Seed agg | pi-100 intervals over seeds | Median only |
| 🟢 Source | relative_mae vs exact ground truth | `pairwise_metrics["shap_true_value"]["relative_mae"]` |
| 🟠 Intended | Show n_bg × budget interaction effect | Interaction visible but variance/dataset split lost |

**Function:** `fig_budget_quality_lines(df, "relative_mae")` · **UI:** metric toggle (default ρ — must click), n_bg/budget/dataset/model filters, warning banner for n_bg=200 × budget=64

---

## Figure 8: Feature Importance Ordering Stability

| | By report we could show… | By code we currently show… |
|--|--------------------------|----------------------------|
| **Plot** | 🟢 Grouped point + line plot | Line + markers per method × n_bg |
| **X-axis** | 🟢 budget N | budget, linear |
| **Y-axis** | 🟢 `mean_sample_rho`, focus band [0.8, 1.0] | median ρ, auto-zoomed to data + ρ=0.9 line |
| **Lines / layout** | 🟠 By method | By method × n_background (style-split) |
| **Read** | Is budget 64 enough to trust the ranking, or is 512 needed? | Same question — this is the page's default chart |
| **Steps** | 1. CSV: `budget`, `mean_sample_rho` vs oracle<br>2. Group method × budget<br>3. Map x=`budget`, y=ρ, zoom [0.8, 1.0] | 1. Extract ρ from `pairwise_metrics["shap_true_value"]`<br>2. Exclude failures<br>3. Group `method × budget × n_background` → median<br>4. Auto y-zoom [min−0.04, max+0.03]; ρ=0.9 reference<br>5. `fig_budget_quality_lines(df, "mean_sample_rho")` |
| **Verdict** | **Match** — closest report↔code pair in US2 | Extras in code: `fig_distribution` box+strip shows per-run spread the report omits |

| Aspect | Report | Implementation |
|--------|--------|----------------|
| 🟢 Plot type | Grouped point and line plot | Line + markers |
| 🟢 X-axis | budget N ∈ {64, 512} | budget, linear |
| 🟢 Y-axis | mean_sample_rho, focus [0.8, 1.0] | median ρ, auto-zoom to data band |
| 🟠 Lines/Color | By method | By method × n_background (solid/dashed) |
| 🟢 Reference | Convergence band | Horizontal line at ρ = 0.9 |
| 🟠 Seed agg | Not specified | Median over seeds/models/datasets |
| 🟠 Intended | Find budget where ranking stabilizes | Same, plus n_bg split the report doesn't ask for |

**Function:** `fig_budget_quality_lines(df, "mean_sample_rho")` (default metric) · **UI:** n_bg/budget/dataset/model/approx filters · **Extra:** `fig_distribution(df)` box+strip per method

---

## Figure 9: Axiomatic Integrity Check

| | By report we could show… | By code we currently show… |
|--|--------------------------|----------------------------|
| **Plot** | 🔴 Box-and-whisker plot | Not implemented on the RQ2 page |
| **X-axis** | 🔴 `library` | — |
| **Y-axis** | 🔴 `relative_additivity_gap`, linear | — |
| **Lines / layout** | 🔴 One box per library; approx frameworks near 1e-16; `shap_true_value` anomaly 4–22% at n_bg=200 | — |
| **Read** | Efficiency axiom holds at machine epsilon for approximators; exact backend breaks at large n_bg | Cannot be shown: `load_data()` keeps only approximation rows — the anomalous `true_value` rows are dropped before any chart |
| **Steps** | 1. CSV: `relative_additivity_gap`, `library`, `n_background`, `backend`<br>2. Include BOTH approximation and `shap_true_value` rows<br>3. Box plot x=`library`, y=gap<br>4. Highlight true_value @ n_bg=200 anomaly | 1. Column exists in CSV (range 0 → 0.81; anomaly rows confirmed present in raw data)<br>2. `load_data()` filters `computation_type == "approximation"` → anomaly rows excluded<br>3. No box-plot builder for this metric; not in any UI toggle |
| **Verdict** | **Report only** — and the code pipeline actively hides the headline anomaly | **Missing**; needs loader change + new figure to verify the 4–22% claim |

| Aspect | Report | Implementation |
|--------|--------|----------------|
| 🔴 Plot type | Box-and-whisker | None |
| 🔴 X-axis | library | — |
| 🔴 Y-axis | relative_additivity_gap, linear | — (column exists in CSV, unused on page) |
| 🔴 Data scope | Approximation + exact backend rows | Only approximation rows survive `load_data()` |
| 🔴 Intended | Verify axiom + expose shap_true_value anomaly | Not verifiable in current app |

**Function:** none · **UI:** none — `relative_additivity_gap` absent from all RQ2 toggles

---

## Figure 10: Practitioner's Pareto Frontier

| | By report we could show… | By code we currently show… |
|--|--------------------------|----------------------------|
| **Plot** | 🟠 Pareto scatter with explicit frontier | Scatter, no frontier line drawn |
| **X-axis** | 🟢 `runtime_s` | median `runtime_s` (or `n_model_evals` via toggle), linear |
| **Y-axis** | 🟠 accuracy = 1 − `relative_mae` | median ρ (default) or `relative_mae` (log) — never 1−MAE |
| **Lines / layout** | 🟠 Marker = (N × n_background) config | Circle=n_bg 50, diamond=n_bg 200; red border = n_bg 200 × budget ≤ 64; color = library |
| **Read** | Upper-left = Pareto-optimal picks for deployment | Same trade-off readable, but optimal set not marked |
| **Steps** | 1. CSV: `runtime_s`, `relative_mae`, `budget`, `n_background`<br>2. Compute y = 1 − `relative_mae`<br>3. One point per lib×approx×budget×n_bg<br>4. Mark Pareto frontier (min runtime, max accuracy) | 1. Extract quality from `pairwise_metrics["shap_true_value"]`<br>2. Click x/y toggles (y default = ρ)<br>3. Group `method × budget × n_background` → median over seeds/models/datasets<br>4. `fig_quality_vs_cost_rq2(df, x, y)` — no `pareto_mark` applied (helper exists in `data.py` but unused here) |
| **Verdict** | **Report** for the deliverable — frontier is the point of the figure | Code has all ingredients (`pareto_mark` exists); one wiring step from matching |

| Aspect | Report | Implementation |
|--------|--------|----------------|
| 🟠 Plot type | Pareto scatter + frontier boundary | Scatter without frontier |
| 🟢 X-axis | runtime_s | median runtime_s, linear (toggle: n_model_evals, log) |
| 🟠 Y-axis | 1 − relative_mae | mean_sample_rho or relative_mae — no 1−MAE option |
| 🟠 Markers | Unique per (N × n_background) | Shape = n_bg, color = library; budget only in hover/border |
| 🟢 Warning | — | Red border flags n_bg=200 × budget ≤ 64 |
| 🟠 Intended | Pareto-optimal picks for engineers | Trade-off visible; optimal set unmarked |

**Function:** `fig_quality_vs_cost_rq2(df, x, y)` · **UI:** x toggle (runtime/evals), y toggle (ρ/MAE/sign), n_bg/budget filters · **Note:** `pareto_mark()` + `fig_pareto()` exist in shared code but are not used on this page

---

## Code-only extras (no Overleaf counterpart)

- **Cross-library agreement heatmap** — `fig_pairwise_heatmap_rq2(df, metric)`: row = source library, col = reference/other library, cell = mean metric from `pairwise_metrics` JSON.
- **Spearman ρ distribution** — `fig_distribution(df)`: box + strip per method, one point per run.
- **Method leaderboard** — `fig_leaderboard_bars(compute_leaderboard(df))`: median ρ ranking with failure labels.
