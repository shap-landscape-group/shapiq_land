"""
RQ2 results converter.

Research question:
    "As a user using Shapley approximations, I want to know how good the
    values actually are so that I can trust the explanations without
    wasting too much computing time needed for exact values."

Input file:
    RQ1+RQ2/results_config-accuracy.csv  (5,000 rows)
        5 datasets x 4 models x 10 seeds x 25 method configurations,
        all at n_features = 12 (dimensionality held constant so the
        comparison isolates accuracy, not scaling).
        4 exact/reference backends:  shap_true_value, shapiq_true_value,
                                     lightshap_exact, dalex_true_value
        21 approximations: {shap, shapiq, lightshap} x {kernel,
        permutation} x budgets {128, 512, 2048} + dalex/permutation
        x the same 3 budgets.

Reference definition (explicit - do not change silently):
    All error metrics are computed against `shap_true_value`, matched
    WITHIN the same dataset x model x seed cell via the pairwise_metrics
    JSON (key "shap_true_value").
    Why this backend: the reference-agreement check below shows
    lightshap_exact matches it to ~1e-15 relative MAE and
    shapiq_true_value to ~1e-6 - numerically the same oracle - while
    dalex_true_value deviates by a median 3.2% relative MAE (max 12.6%).
    dalex_true_value is therefore NOT interchangeable with the other
    three and is excluded as a reference; the deviation itself is
    reported in rq2_reference_agreement.csv.

Purpose:
    Convert raw accuracy benchmark runs into validated, seed-aggregated
    tables for the RQ2 page: convergence over budget, runtime-vs-error
    trade-off, seed stability, and the reference cross-check.

Important limitations:
    * n_features is fixed at 12; conclusions do not transfer to high
      dimensions (that is RQ1's job).
    * dalex approximation error is measured against shap_true_value, not
      against dalex's own reference - the ~3% reference disagreement is
      part of dalex's measured error budget.

Aggregation principles:
    * Seeds (10) are the only dimension aggregated. dataset, model,
      library, approximator and budget always survive into the
      aggregated tables; pooling beyond that is a page-level display
      decision.
    * Median + q25/q75 throughout, consistent with RQ1.

Outputs (results/converted/):
    rq2_accuracy_by_seed.csv         one row per approximation run with
                                     error metrics vs shap_true_value
    rq2_convergence_aggregated.csv   seeds aggregated per dataset x model
                                     x method x budget
    rq2_runtime_accuracy.csv         runtime-vs-error pairs (medians) for
                                     the Pareto view, incl. reference
                                     runtimes as separate rows
    rq2_reference_agreement.csv      pairwise agreement between the four
                                     exact/reference backends

Run:
    python results_converters/rq2_results_converter.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_ACCURACY = os.path.join(_HERE, "results", "RQ1+RQ2", "results_config-accuracy.csv")
OUT_DIR = os.path.join(_HERE, "results", "converted")

# The oracle every approximation is scored against (see module docstring).
REFERENCE_BACKEND = "shap_true_value"

REFERENCE_BACKENDS = ["shap_true_value", "shapiq_true_value",
                      "lightshap_exact", "dalex_true_value"]

EXPECTED_DATASETS = {"ames_housing", "bike", "covertype",
                     "diabetes_130", "gisette"}
EXPECTED_MODELS = {"decision_tree", "gradient_boosting",
                   "linear_regularized", "random_forest"}
EXPECTED_SEEDS = set(range(10))
EXPECTED_BUDGETS = {128.0, 512.0, 2048.0}
EXPECTED_METHODS_PER_CELL = 25   # 21 approximations + 4 references

REQUIRED_COLUMNS = [
    "dataset", "model", "n_features", "seed", "library", "backend",
    "computation_type", "approximator", "budget", "runtime_s",
    "n_model_evals", "pairwise_metrics",
]

# Error metrics copied per run from the pairwise entry vs the reference.
ERROR_METRICS = ["relative_mae", "mean_abs_diff",
                 "sign_agreement", "mean_sample_rho"]

# Columns read directly from the raw row (not from pairwise JSON).
DIRECT_METRICS = ["relative_additivity_gap"]


# ─────────────────────────────────────────────────────────────────────────────
#  Loading and validation
# ─────────────────────────────────────────────────────────────────────────────

def load_accuracy() -> pd.DataFrame:
    df = pd.read_csv(RAW_ACCURACY)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{RAW_ACCURACY}: missing required columns {missing}")

    # n_features must be constant (12): the accuracy design controls
    # dimensionality. A second value would mean scaling effects leak in.
    if df["n_features"].nunique() != 1:
        raise ValueError(
            f"accuracy file has multiple n_features values "
            f"{sorted(df['n_features'].unique())}; expected exactly one"
        )

    for col, expected in [("dataset", EXPECTED_DATASETS),
                          ("model", EXPECTED_MODELS),
                          ("seed", EXPECTED_SEEDS)]:
        actual = set(df[col].dropna().unique())
        if actual != expected:
            print(f"  [design warning] {col}: expected {sorted(expected)}, "
                  f"got {sorted(actual)}")

    budgets = set(df.loc[df["computation_type"] == "approximation", "budget"]
                  .dropna().unique())
    if budgets != EXPECTED_BUDGETS:
        print(f"  [design warning] budgets: expected {EXPECTED_BUDGETS}, got {budgets}")

    cell_sizes = df.groupby(["dataset", "model", "seed"]).size()
    bad = cell_sizes[cell_sizes != EXPECTED_METHODS_PER_CELL]
    if not bad.empty:
        print(f"  [design warning] {len(bad)} cells deviate from "
              f"{EXPECTED_METHODS_PER_CELL} methods/cell")

    dupes = df.duplicated(
        subset=["dataset", "model", "seed", "backend",
                "approximator", "budget"]).sum()
    if dupes:
        print(f"  [design warning] {dupes} duplicated run rows")

    ref_missing = (
        set(REFERENCE_BACKENDS) - set(df["backend"].dropna().unique()))
    if ref_missing:
        raise ValueError(f"reference backends missing from file: {ref_missing}")

    return df


def _parse_pairwise(cell: str) -> dict:
    """Parse the pairwise_metrics JSON cell (contains bare NaN tokens)."""
    if not isinstance(cell, str) or not cell.strip():
        return {}
    try:
        return json.loads(cell.replace(": NaN", ": null").replace(":NaN", ":null"))
    except (ValueError, TypeError):
        return {}


# ─────────────────────────────────────────────────────────────────────────────
#  Reference cross-check
# ─────────────────────────────────────────────────────────────────────────────

def build_reference_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise agreement between the four exact/reference backends.

    Row selection: only reference rows (computation_type == true_value /
    exact-style backends). For each ordered pair (source backend row,
    target backend key in its pairwise JSON) we aggregate the agreement
    metrics over all 200 dataset x model x seed cells.

    This table is what justifies (and lets the reader verify) the choice
    of shap_true_value as the single oracle: three backends agree at
    numerical-noise level, dalex_true_value does not.
    """
    refs = df[df["backend"].isin(REFERENCE_BACKENDS)]

    records = []
    for _, row in refs.iterrows():
        pm = _parse_pairwise(row["pairwise_metrics"])
        for target in REFERENCE_BACKENDS:
            entry = pm.get(target)
            if not isinstance(entry, dict) or row["backend"] == target:
                continue
            records.append({
                "source_backend": row["backend"],
                "target_backend": target,
                "dataset": row["dataset"],
                "model": row["model"],
                "seed": row["seed"],
                **{m: entry.get(m) for m in ERROR_METRICS},
            })

    pairs = pd.DataFrame(records)
    # Aggregate over all cells: median for the central tendency, max for
    # the worst case (the dalex deviation is best characterised by both).
    return (
        pairs.groupby(["source_backend", "target_backend"])
        .agg(
            cell_count=("relative_mae", "size"),
            relative_mae_median=("relative_mae", "median"),
            relative_mae_max=("relative_mae", "max"),
            rho_median=("mean_sample_rho", "median"),
            rho_min=("mean_sample_rho", "min"),
            sign_agreement_median=("sign_agreement", "median"),
        )
        .reset_index()
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Per-run error extraction
# ─────────────────────────────────────────────────────────────────────────────

def build_by_seed(df: pd.DataFrame) -> pd.DataFrame:
    """One row per approximation run with its error vs the reference.

    Row selection: computation_type == "approximation" only (4,200 rows).
    Matching: the pairwise_metrics JSON in each run row already contains
    the comparison against every method in the SAME dataset x model x seed
    cell, so looking up key "shap_true_value" IS the within-cell match -
    no join is needed and no cross-seed leakage is possible.
    """
    apx = df[df["computation_type"] == "approximation"].copy()
    apx["method"] = apx["library"] + " / " + apx["approximator"]

    extracted = {m: [] for m in ERROR_METRICS}
    for cell_json in apx["pairwise_metrics"]:
        pm = _parse_pairwise(cell_json)
        ref = pm.get(REFERENCE_BACKEND)
        ref = ref if isinstance(ref, dict) else {}
        for m in ERROR_METRICS:
            extracted[m].append(ref.get(m, np.nan))
    for m in ERROR_METRICS:
        apx[m] = extracted[m]

    n_missing = apx["relative_mae"].isna().sum()
    if n_missing:
        print(f"  [warning] {n_missing} approximation runs lack a "
              f"{REFERENCE_BACKEND} comparison")

    for col in DIRECT_METRICS:
        if col in apx.columns:
            apx[col] = pd.to_numeric(apx[col], errors="coerce")

    keep = ["dataset", "model", "n_features", "seed", "library",
            "approximator", "budget", "method", "runtime_s",
            "n_model_evals"] + ERROR_METRICS + DIRECT_METRICS
    return apx[[c for c in keep if c in apx.columns]]


# ─────────────────────────────────────────────────────────────────────────────
#  Seed aggregation
# ─────────────────────────────────────────────────────────────────────────────

GROUP_KEYS = ["dataset", "model", "library", "approximator", "budget", "method"]


def aggregate_convergence(by_seed: pd.DataFrame) -> pd.DataFrame:
    """Seeds aggregated; everything else preserved.

    Unit before aggregation:  one approximation run for one seed
    Unit after aggregation:   one method configuration within one
                              dataset x model cell at one budget

    q25/q75 columns carry the seed spread so the page can draw
    uncertainty bands instead of implying false precision. The IQR of
    relative_mae across seeds is itself the "stability across seeds"
    metric requested by the research question.
    """
    def q25(s): return s.quantile(0.25)
    def q75(s): return s.quantile(0.75)

    return (
        by_seed.groupby(GROUP_KEYS)
        .agg(
            seed_count=("seed", "nunique"),
            runtime_median=("runtime_s", "median"),
            runtime_q25=("runtime_s", q25),
            runtime_q75=("runtime_s", q75),
            relative_mae_median=("relative_mae", "median"),
            relative_mae_q25=("relative_mae", q25),
            relative_mae_q75=("relative_mae", q75),
            rho_median=("mean_sample_rho", "median"),
            rho_q25=("mean_sample_rho", q25),
            rho_q75=("mean_sample_rho", q75),
            sign_agreement_median=("sign_agreement", "median"),
            sign_agreement_q25=("sign_agreement", q25),
            sign_agreement_q75=("sign_agreement", q75),
            rel_additivity_gap_median=("relative_additivity_gap", "median"),
            rel_additivity_gap_q25=("relative_additivity_gap", q25),
            rel_additivity_gap_q75=("relative_additivity_gap", q75),
        )
        .reset_index()
    )


def build_runtime_accuracy(df: pd.DataFrame,
                           convergence: pd.DataFrame) -> pd.DataFrame:
    """Runtime-vs-error table for the Pareto trade-off figure.

    Approximation rows come from the convergence table (medians over
    seeds). Reference backends are appended as separate rows with
    relative_mae = 0 BY DEFINITION (each is exact relative to itself;
    shap_true_value is the scoring oracle) so the page can show what
    "paying for exactness" costs in runtime. is_reference marks them so
    they are never styled as ordinary methods.
    """
    apx = convergence.copy()
    apx["is_reference"] = False
    apx["backend"] = apx["library"] + "_approx"

    refs = df[df["backend"].isin(REFERENCE_BACKENDS)].copy()
    ref_rows = (
        refs.groupby(["dataset", "model", "library", "backend"])
        .agg(
            seed_count=("seed", "nunique"),
            runtime_median=("runtime_s", "median"),
            runtime_q25=("runtime_s", lambda s: s.quantile(0.25)),
            runtime_q75=("runtime_s", lambda s: s.quantile(0.75)),
        )
        .reset_index()
    )
    ref_rows["approximator"] = "exact"
    ref_rows["budget"] = np.nan
    ref_rows["method"] = ref_rows["backend"]
    # relative_mae = 0 only for backends numerically equal to the oracle;
    # dalex_true_value gets its measured median deviation instead, so the
    # Pareto view does not overstate its exactness.
    dalex_dev = 0.032  # median relative_mae vs shap_true_value, see rq2_reference_agreement.csv
    ref_rows["relative_mae_median"] = np.where(
        ref_rows["backend"] == "dalex_true_value", dalex_dev, 0.0)
    ref_rows["rho_median"] = np.nan
    ref_rows["is_reference"] = True

    combined = pd.concat([apx, ref_rows], ignore_index=True)
    keep = ["dataset", "model", "library", "backend", "approximator",
            "budget", "method", "is_reference", "seed_count",
            "runtime_median", "runtime_q25", "runtime_q75",
            "relative_mae_median", "rho_median"]
    return combined[[c for c in keep if c in combined.columns]]


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print("RQ2 converter: loading accuracy file …")
    df = load_accuracy()

    ref_agreement = build_reference_agreement(df)
    by_seed = build_by_seed(df)
    convergence = aggregate_convergence(by_seed)
    runtime_accuracy = build_runtime_accuracy(df, convergence)

    outputs = {
        "rq2_accuracy_by_seed.csv": by_seed,
        "rq2_convergence_aggregated.csv": convergence,
        "rq2_runtime_accuracy.csv": runtime_accuracy,
        "rq2_reference_agreement.csv": ref_agreement,
    }
    for name, table in outputs.items():
        path = os.path.join(OUT_DIR, name)
        table.to_csv(path, index=False)
        print(f"  wrote {name}: {len(table)} rows")

    # Print the oracle-choice evidence so every conversion re-states it.
    vs_shap = ref_agreement[
        ref_agreement["source_backend"] == REFERENCE_BACKEND]
    print("\n  reference agreement vs shap_true_value (median relative MAE):")
    for _, r in vs_shap.iterrows():
        print(f"    {r['target_backend']}: {r['relative_mae_median']:.2e}")


if __name__ == "__main__":
    sys.exit(main())
