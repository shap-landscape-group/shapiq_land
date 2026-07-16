import os
import json
import pandas as pd
import numpy as np

# Canonical seed-level schema shared by every contributor (NN, tree/woodelf, …).
BY_SEED_COLUMNS = [
    "dataset", "model", "library", "approximator", "device", "seed", "n_features",
    "runtime_s", "relative_additivity_gap", "mean_sample_rho", "relative_mae",
    "sign_agreement", "method",
]


def extract_nn_metrics(row):
    metrics_str = row.get("pairwise_metrics")
    if pd.isna(metrics_str):
        return pd.Series([np.nan, np.nan, np.nan], index=["mean_sample_rho", "relative_mae", "sign_agreement"])
    try:
        metrics_json = json.loads(metrics_str)
        approximator = row.get("approximator")
        target_key = None
        for k in metrics_json.keys():
            parts = k.split("|")
            if len(parts) >= 2 and parts[1] == approximator:
                target_key = k
                break

        if target_key and target_key in metrics_json:
            m = metrics_json[target_key]
            return pd.Series([
                m.get("mean_sample_rho"),
                m.get("relative_mae"),
                m.get("sign_agreement")
            ], index=["mean_sample_rho", "relative_mae", "sign_agreement"])
    except Exception:
        pass
    return pd.Series([np.nan, np.nan, np.nan], index=["mean_sample_rho", "relative_mae", "sign_agreement"])

def extract_tree_metrics(row):
    metrics_str = row.get("pairwise_metrics")
    if pd.isna(metrics_str):
        return pd.Series([np.nan, np.nan, np.nan], index=["mean_sample_rho", "relative_mae", "sign_agreement"])
    try:
        metrics_json = json.loads(metrics_str)
        backend = row.get("backend")
        if backend in metrics_json:
            m = metrics_json[backend]
            return pd.Series([
                m.get("mean_sample_rho"),
                m.get("relative_mae"),
                m.get("sign_agreement")
            ], index=["mean_sample_rho", "relative_mae", "sign_agreement"])
    except Exception:
        pass
    return pd.Series([np.nan, np.nan, np.nan], index=["mean_sample_rho", "relative_mae", "sign_agreement"])


def process_nn(nn_path):
    """Neural-network CPU-vs-GPU runs → canonical seed-level rows."""
    print(f"Loading NN data from {nn_path}...")
    df_nn = pd.read_csv(nn_path)

    print("Extracting metrics for NN...")
    metrics_nn = df_nn.apply(extract_nn_metrics, axis=1)
    df_nn = pd.concat([df_nn, metrics_nn], axis=1)

    # Clean devices to be uniform (cpu/gpu)
    df_nn["device"] = df_nn["device"].astype(str).str.lower().replace({"cuda": "gpu"})

    keep = ["dataset", "model", "library", "approximator", "device", "seed", "n_features",
            "runtime_s", "relative_additivity_gap", "mean_sample_rho", "relative_mae", "sign_agreement"]
    df = df_nn[[c for c in keep if c in df_nn.columns]].copy()
    for col in keep:
        if col not in df.columns:
            df[col] = np.nan
    df = df[keep]
    df["method"] = df["library"] + " / " + df["approximator"]
    return df[BY_SEED_COLUMNS]


def parse_woodelf_backend(backend):
    """woodelf backend strings encode device + imputation, e.g. woodelf_gpu_path_dependent."""
    device = "gpu" if "gpu" in backend else "cpu"
    approximator = str(backend).replace("woodelf_gpu_", "").replace("woodelf_", "")
    return device, approximator


def process_woodelf(woodelf_path):
    """woodelf tree explainer CPU-vs-GPU runs → canonical seed-level rows.

    The woodelf benchmark encodes the device (cpu/gpu) and imputation strategy
    (path_dependent/interventional) inside a single ``backend`` column, and
    sweeps ``max_depth`` for each seed.  We collapse the depth sweep to a single
    representative (median) runtime per seed so woodelf lines up with the
    one-row-per-seed structure of every other RQ5 contributor.
    """
    print(f"Loading woodelf data from {woodelf_path}...")
    df = pd.read_csv(woodelf_path)

    print("Extracting metrics for woodelf...")
    metrics = df.apply(extract_tree_metrics, axis=1)
    df = pd.concat([df, metrics], axis=1)

    dev_approx = df["backend"].map(parse_woodelf_backend)
    df["device"] = [d for d, _ in dev_approx]
    df["approximator"] = [a for _, a in dev_approx]

    # Collapse the max_depth sweep within each seed → one representative row.
    keys = ["dataset", "model", "library", "approximator", "device", "seed", "n_features"]
    df = (
        df.groupby(keys, dropna=False)
        .agg(
            runtime_s=("runtime_s", "median"),
            relative_additivity_gap=("relative_additivity_gap", "median"),
            mean_sample_rho=("mean_sample_rho", "median"),
            relative_mae=("relative_mae", "median"),
            sign_agreement=("sign_agreement", "median"),
        )
        .reset_index()
    )
    df["method"] = df["library"] + " / " + df["approximator"]
    return df[BY_SEED_COLUMNS]


def process_woodelf_depth(woodelf_path):
    """woodelf runtime resolved against the swept tree depth (not collapsed).

    ``process_woodelf`` collapses the max_depth sweep so woodelf lines up with the
    per-seed comparison charts.  This keeps every depth so the scaling chart can
    plot runtime against tree depth — the variable the benchmark actually sweeps.
    """
    df = pd.read_csv(woodelf_path)
    dev_approx = df["backend"].map(parse_woodelf_backend)
    df["device"] = [d for d, _ in dev_approx]
    df["approximator"] = [a for _, a in dev_approx]
    df["max_depth"] = df["max_depth_config"]
    cols = ["dataset", "model", "library", "approximator", "device",
            "max_depth", "seed", "n_features", "runtime_s"]
    return df[cols].copy()


def aggregate_by_seed(df_combined):
    """Aggregate the seed-level rows across seeds (median + Q25/Q75 spread)."""
    print("Aggregating across seeds...")
    group_cols = ["dataset", "model", "library", "approximator", "method", "device", "n_features"]
    return (
        df_combined.groupby(group_cols, dropna=False)
        .agg(
            runtime_median=("runtime_s", "median"),
            runtime_q25=("runtime_s", lambda x: x.quantile(0.25)),
            runtime_q75=("runtime_s", lambda x: x.quantile(0.75)),
            gap_mean=("relative_additivity_gap", "mean"),
            gap_q25=("relative_additivity_gap", lambda x: x.quantile(0.25)),
            gap_q75=("relative_additivity_gap", lambda x: x.quantile(0.75)),
            rho_median=("mean_sample_rho", "median"),
            rho_q25=("mean_sample_rho", lambda x: x.quantile(0.25)),
            rho_q75=("mean_sample_rho", lambda x: x.quantile(0.75)),
            relative_mae_median=("relative_mae", "median"),
            relative_mae_q25=("relative_mae", lambda x: x.quantile(0.25)),
            relative_mae_q75=("relative_mae", lambda x: x.quantile(0.75)),
            sign_agreement_median=("sign_agreement", "median"),
            seed_count=("seed", "count"),
        )
        .reset_index()
    )


def main():
    print("Converting RQ5 results...")
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(here)

    # Source files (each contributor is optional — a missing source is preserved
    # from the previously converted output so re-running never drops data).
    nn_path = os.path.join(project_root, "results", "rq5_model_gpu-vs-cpu-shapiq-shap.csv")
    woodelf_path = os.path.join(project_root, "results", "rq5_woodelf.csv")

    out_dir = os.path.join(project_root, "results", "converted")
    os.makedirs(out_dir, exist_ok=True)
    aggregated_out = os.path.join(out_dir, "rq5_gpu_cpu_comparison_aggregated.csv")
    by_seed_out = os.path.join(out_dir, "rq5_gpu_cpu_comparison_by_seed.csv")
    woodelf_depth_out = os.path.join(out_dir, "rq5_woodelf_depth_scaling.csv")

    existing = pd.DataFrame()
    if os.path.exists(by_seed_out):
        existing = pd.read_csv(by_seed_out)

    frames = []

    # ── Neural-network contributor (captum / shapiq / …) ──────────────────────
    if os.path.exists(nn_path):
        frames.append(process_nn(nn_path))
    elif not existing.empty:
        print(f"NN source {os.path.basename(nn_path)} not found — preserving existing non-woodelf rows.")
        frames.append(existing[existing["library"] != "woodelf"][BY_SEED_COLUMNS])

    # ── woodelf tree-explainer contributor ────────────────────────────────────
    if os.path.exists(woodelf_path):
        frames.append(process_woodelf(woodelf_path))
    elif not existing.empty:
        print(f"woodelf source {os.path.basename(woodelf_path)} not found — preserving existing woodelf rows.")
        frames.append(existing[existing["library"] == "woodelf"][BY_SEED_COLUMNS])

    if not frames:
        raise SystemExit("No RQ5 source data found and no existing converted output to preserve.")

    df_combined = pd.concat(frames, ignore_index=True)

    print(f"Saving seed-level runs to {by_seed_out}...")
    df_combined.to_csv(by_seed_out, index=False)

    df_agg = aggregate_by_seed(df_combined)
    print(f"Saving aggregated runs to {aggregated_out}...")
    df_agg.to_csv(aggregated_out, index=False)

    # woodelf depth-resolved runtimes (drives the tree-depth scaling chart).
    if os.path.exists(woodelf_path):
        df_depth = process_woodelf_depth(woodelf_path)
        print(f"Saving woodelf depth scaling to {woodelf_depth_out}...")
        df_depth.to_csv(woodelf_depth_out, index=False)

    libs = ", ".join(sorted(df_combined["library"].dropna().unique()))
    print(f"Done! Contributors: {libs} | {len(df_combined)} seed-level rows, {len(df_agg)} aggregated rows.")

if __name__ == "__main__":
    main()
