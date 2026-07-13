"""
RQ1 results converter.

Research question:
    "As a user with a dataset that has many features, I want to find the
    fastest model-agnostic library for high-dimensional datasets."

Input files:
    RQ1+RQ2/results_config-dimensionality.csv          (6,720 rows)
        4 datasets x 3 feature counts x 4 models x 10 seeds x 14 method
        configurations (library x approximator x budget {512, 1024}).
        All rows are approximations - there is NO exact reference here.
    RQ1+RQ2/results_config-dimensionality-extreme.csv  (28 rows)
        Stress test: gisette only, n_features=1000, seed 0 only,
        budget 2048 only, 7 method configurations x 4 models.

Purpose:
    Convert raw benchmark runs into validated and aggregated result tables
    used by pages/rq1_dimensionality.py. All row selection, seed
    aggregation and derived-metric logic lives here so the page code only
    filters and draws.

Important limitations (encoded in the outputs, respected by the page):
    * Both input files contain ONLY approximate methods. Pairwise metrics
      here measure cross-method AGREEMENT, not accuracy against ground
      truth. No output column is named "accuracy".
    * The extreme file has 1 dataset / 1 seed / 1 budget / fewer methods.
      It is written to a SEPARATE table and must never be plotted as a
      point on the standard scaling curves.
    * 120 rows sit at the ~600 s wall-clock cap (all lightshap). They are
      kept but flagged `hit_time_cap`, because their runtime is a lower
      bound, not a measurement.
    * 40 shap/permutation rows report n_model_evals = 0. Shapley values
      and runtimes for these rows look valid, so this is treated as an
      instrumentation gap: the eval count is set to NaN (flag
      `evals_missing`), not dropped.

Aggregation principles:
    * Seeds are the ONLY dimension aggregated by default. The experiment
      structure dataset x model x n_features x library x approximator x
      budget is always preserved in the aggregated table; the page decides
      what to facet or filter.
    * Median + quartiles (q25/q75) are used instead of mean +- std because
      the runtime distribution contains cap-limited values (600 s) that
      would drag a mean upward.
    * Time-capped runs are EXCLUDED from runtime medians (their true
      runtime is unknown) but COUNTED in `time_cap_count` so feasibility
      charts can show how often a method hits the cap.

Outputs (results/converted/):
    rq1_scaling_by_seed.csv       one row per run, validated + flagged
    rq1_scaling_aggregated.csv    one row per method config per cell,
                                  seeds aggregated
    rq1_feasibility.csv           time-cap / eval-gap counts per method
                                  config x dataset x n_features
    rq1_extreme_stress_test.csv   the 1000-feature stress test, per run
                                  (single seed -> nothing to aggregate)

Run:
    python results_converters/rq1_results_converter.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(_HERE, "RQ1+RQ2")
OUT_DIR = os.path.join(_HERE, "results", "converted")

RAW_STANDARD = os.path.join(RAW_DIR, "results_config-dimensionality.csv")
RAW_EXTREME  = os.path.join(RAW_DIR, "results_config-dimensionality-extreme.csv")

# Runs at or above this runtime are treated as having hit the benchmark's
# 10-minute execution cap. The raw data clusters at 600.0 +- 0.02 s, so a
# threshold slightly below catches the whole cluster without touching real
# sub-cap measurements (next-slowest run is well under 595 s).
TIME_CAP_SECONDS = 595.0

# Expected experimental design, used for validation. If the benchmark is
# re-run with a different grid these lists must be updated - the converter
# fails loudly rather than silently accepting a different design.
EXPECTED_DATASETS = {"ames_housing", "covertype", "diabetes_130", "gisette"}
EXPECTED_MODELS = {"decision_tree", "gradient_boosting",
                   "linear_regularized", "random_forest"}
EXPECTED_SEEDS = set(range(10))
EXPECTED_BUDGETS = {512, 1024}
EXPECTED_LIBRARIES = {"shap", "shapiq", "lightshap", "dalex"}
# 14 = 3 libraries x 2 approximators x 2 budgets + dalex (permutation only) x 2 budgets
EXPECTED_METHODS_PER_CELL = 14

REQUIRED_COLUMNS = [
    "dataset", "model", "n_features", "seed", "library", "backend",
    "approximator", "budget", "runtime_s", "n_model_evals",
    "computation_type", "pairwise_metrics",
]


# ─────────────────────────────────────────────────────────────────────────────
#  Loading and validation
# ─────────────────────────────────────────────────────────────────────────────

def _validate_columns(df: pd.DataFrame, path: str) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")


def _validate_design(df: pd.DataFrame) -> list[str]:
    """Compare the standard file against the documented experimental grid.

    Returns human-readable warnings instead of raising for coverage gaps,
    because a partially re-run benchmark should still convert - but the
    console output must say what is missing.
    """
    warnings = []

    for col, expected in [("dataset", EXPECTED_DATASETS),
                          ("model", EXPECTED_MODELS),
                          ("seed", EXPECTED_SEEDS),
                          ("budget", EXPECTED_BUDGETS),
                          ("library", EXPECTED_LIBRARIES)]:
        actual = set(df[col].dropna().unique())
        if actual - expected:
            warnings.append(f"unexpected {col} values: {actual - expected}")
        if expected - actual:
            warnings.append(f"missing {col} values: {expected - actual}")

    # Every dataset x model x n_features x seed cell must contain exactly
    # the 14 method configurations - fewer means lost runs, more means
    # duplicates. Both would silently bias any aggregation.
    cell_sizes = df.groupby(["dataset", "model", "n_features", "seed"]).size()
    bad = cell_sizes[cell_sizes != EXPECTED_METHODS_PER_CELL]
    if not bad.empty:
        warnings.append(
            f"{len(bad)} cells deviate from {EXPECTED_METHODS_PER_CELL} "
            f"methods/cell (min={cell_sizes.min()}, max={cell_sizes.max()})"
        )

    dupes = df.duplicated(
        subset=["dataset", "model", "n_features", "seed",
                "library", "approximator", "budget"]).sum()
    if dupes:
        warnings.append(f"{dupes} duplicated run rows")

    return warnings


def load_standard() -> pd.DataFrame:
    df = pd.read_csv(RAW_STANDARD)
    _validate_columns(df, RAW_STANDARD)

    if (df["computation_type"] != "approximation").any():
        raise ValueError(
            "RQ1 standard file unexpectedly contains non-approximation rows; "
            "the RQ1 pipeline assumes agreement-only metrics."
        )

    for w in _validate_design(df):
        print(f"  [design warning] {w}")
    return df


def load_extreme() -> pd.DataFrame:
    df = pd.read_csv(RAW_EXTREME)
    _validate_columns(df, RAW_EXTREME)
    # The stress test is deliberately narrow; only sanity-check its shape.
    if df["n_features"].nunique() != 1 or df["seed"].nunique() != 1:
        raise ValueError("extreme file is expected to be single-feature-count, single-seed")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Derived columns and flags
# ─────────────────────────────────────────────────────────────────────────────

def add_run_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Attach validity flags and the canonical method label to each run.

    method: "library / approximator" - budget stays a separate column
    because RQ1 treats budget as an experimental axis, not part of the
    method's identity.
    """
    df = df.copy()
    df["method"] = df["library"] + " / " + df["approximator"]

    # Runs at the wall-clock cap: runtime is a lower bound. They stay in
    # the by-seed table (feasibility analysis needs them) but are excluded
    # from runtime aggregation below.
    df["hit_time_cap"] = df["runtime_s"] >= TIME_CAP_SECONDS

    # n_model_evals == 0 is physically impossible for a completed run
    # (every method must query the model at least once per explained
    # sample). Affects 40 shap/permutation rows; their Shapley output and
    # runtime look normal, so we treat the counter as missing rather than
    # discarding the run.
    df["evals_missing"] = df["n_model_evals"] <= 0
    df.loc[df["evals_missing"], "n_model_evals"] = np.nan

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Pairwise agreement extraction
# ─────────────────────────────────────────────────────────────────────────────

def _parse_pairwise(cell: str) -> dict:
    """Parse the pairwise_metrics JSON cell (contains bare NaN tokens)."""
    if not isinstance(cell, str) or not cell.strip():
        return {}
    try:
        return json.loads(cell.replace(": NaN", ": null").replace(":NaN", ":null"))
    except (ValueError, TypeError):
        return {}


def add_cross_method_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise each run's agreement with the OTHER 13 methods in its cell.

    The pairwise_metrics JSON keys look like "shap_approx|kernel|512".
    Since RQ1 has no exact reference, the only defensible scalar per run is
    consensus-style: the mean of mean_sample_rho against all other methods
    in the same dataset x model x n_features x seed cell (self-comparison
    excluded). High consensus does NOT prove correctness - all methods
    could agree and be wrong - so the column is named cross-method
    agreement, never accuracy.
    """
    df = df.copy()
    self_keys = (
        df["backend"] + "|" + df["approximator"] + "|" +
        df["budget"].astype(int).astype(str)
    )

    agreements = []
    for cell_json, self_key in zip(df["pairwise_metrics"], self_keys):
        pm = _parse_pairwise(cell_json)
        rhos = [
            v.get("mean_sample_rho")
            for k, v in pm.items()
            if k != self_key and isinstance(v, dict)
            and v.get("mean_sample_rho") is not None
        ]
        agreements.append(float(np.mean(rhos)) if rhos else np.nan)

    df["cross_method_rho_mean"] = agreements
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Seed aggregation
# ─────────────────────────────────────────────────────────────────────────────

# The grouping preserves the full experiment structure; ONLY the 10 seeds
# are collapsed. Anything coarser (e.g. pooling datasets) is a display
# decision that belongs to the page, where it is stated explicitly.
GROUP_KEYS = ["dataset", "model", "n_features",
              "library", "approximator", "budget", "method"]


def aggregate_seeds(by_seed: pd.DataFrame) -> pd.DataFrame:
    """One row per method configuration per experiment cell.

    Unit before aggregation:  one benchmark execution for one seed
    Unit after aggregation:   one method configuration within one
                              dataset / model / feature-count / budget cell

    Median + q25/q75 instead of mean +- std: the runtime column contains
    cap-limited values and right-skewed outliers that would distort a mean.

    Time-capped runs are excluded from the runtime statistics (their true
    runtime is unknown - 600 s is a floor), but they are counted in
    time_cap_count so the page can flag partially-capped configurations.
    """
    uncapped = by_seed[~by_seed["hit_time_cap"]]

    runtime_stats = uncapped.groupby(GROUP_KEYS).agg(
        runtime_median=("runtime_s", "median"),
        runtime_q25=("runtime_s", lambda s: s.quantile(0.25)),
        runtime_q75=("runtime_s", lambda s: s.quantile(0.75)),
        evals_median=("n_model_evals", "median"),
        cross_method_rho_median=("cross_method_rho_mean", "median"),
    )

    counts = by_seed.groupby(GROUP_KEYS).agg(
        seed_count=("seed", "nunique"),
        run_count=("seed", "size"),
        time_cap_count=("hit_time_cap", "sum"),
        evals_missing_count=("evals_missing", "sum"),
    )

    agg = counts.join(runtime_stats).reset_index()
    agg["successful_run_count"] = agg["run_count"] - agg["time_cap_count"]
    return agg


def build_feasibility_table(by_seed: pd.DataFrame) -> pd.DataFrame:
    """Fraction of runs hitting the time cap per method x dataset x n_features.

    Budgets and models are pooled ON PURPOSE here: the feasibility question
    is "can this method finish on this dataset at this dimensionality at
    all?", and a cap in any model/budget combination is a deployment risk.
    The pooled denominators are kept in run_count so the fraction stays
    interpretable.
    """
    feas = by_seed.groupby(
        ["library", "approximator", "method", "dataset", "n_features"]
    ).agg(
        run_count=("seed", "size"),
        time_cap_count=("hit_time_cap", "sum"),
    ).reset_index()
    feas["time_cap_fraction"] = feas["time_cap_count"] / feas["run_count"]
    return feas


# ─────────────────────────────────────────────────────────────────────────────
#  Extreme stress test (kept fully separate)
# ─────────────────────────────────────────────────────────────────────────────

def build_extreme_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-run table for the 1000-feature stress test.

    Single seed and single budget -> aggregation would be meaningless; the
    page shows raw values and must label them as a one-shot stress test.
    Time-cap flagging still applies (6 of 8 lightshap rows exceed 600 s,
    up to ~3.5 h) but capped rows are NOT excluded here: for a feasibility
    stress test the fact that a run needed >600 s IS the finding.
    """
    df = add_run_flags(df)
    keep = ["dataset", "model", "n_features", "seed", "library",
            "approximator", "budget", "method", "runtime_s",
            "n_model_evals", "hit_time_cap", "evals_missing"]
    return df[keep].sort_values(["model", "library", "approximator"])


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print("RQ1 converter: loading standard dimensionality file …")
    std = load_standard()
    std = add_run_flags(std)
    std = add_cross_method_agreement(std)

    by_seed_cols = ["dataset", "model", "n_features", "seed", "library",
                    "approximator", "budget", "method", "runtime_s",
                    "n_model_evals", "cross_method_rho_mean",
                    "hit_time_cap", "evals_missing"]
    by_seed = std[by_seed_cols]

    agg = aggregate_seeds(by_seed)
    feas = build_feasibility_table(by_seed)

    print("RQ1 converter: loading extreme stress-test file …")
    extreme = build_extreme_table(load_extreme())

    outputs = {
        "rq1_scaling_by_seed.csv": by_seed,
        "rq1_scaling_aggregated.csv": agg,
        "rq1_feasibility.csv": feas,
        "rq1_extreme_stress_test.csv": extreme,
    }
    for name, table in outputs.items():
        path = os.path.join(OUT_DIR, name)
        table.to_csv(path, index=False)
        print(f"  wrote {name}: {len(table)} rows")

    # Console summary of data-quality findings so a re-run surfaces them.
    print(f"  time-capped runs (standard): {int(by_seed['hit_time_cap'].sum())} "
          f"({by_seed[by_seed['hit_time_cap']].groupby('library').size().to_dict()})")
    print(f"  eval-counter gaps (standard): {int(by_seed['evals_missing'].sum())}")
    print(f"  time-capped runs (extreme): {int(extreme['hit_time_cap'].sum())}")


if __name__ == "__main__":
    sys.exit(main())
