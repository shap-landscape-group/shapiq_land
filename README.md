# Shapley Benchmark Explorer

Four focused Dash pages, one per research question.

## Structure

```
benchmark_explorer.py        # main entry point (nav + page container)
shared.py                    # design tokens, data loading, all chart builders
pages/
  home.py                    # overview / landing  →  /
  rq1_dimensionality.py      # RQ1: high vs low dim  →  /rq1
  rq2_accuracy.py            # RQ2: approximation accuracy  →  /rq2
  rq3_neural_networks.py     # RQ3: NN runtime  →  /rq3
  rq4_trees.py               # RQ4: tree depth / bottlenecks  →  /rq4

results.csv                  # shared data for RQ1 + RQ2
results_dimensionality.csv   # optional: dedicated RQ1 data (falls back to results.csv)
results_accuracy.csv         # optional: dedicated RQ2 data (falls back to results.csv)
results_nn.csv               # RQ3 data  (page shows placeholder until this exists)
results_trees.csv            # RQ4 data  (page shows placeholder until this exists)
```

---

## Setup (first time only)

```bash
cd /Users/Lucky/Downloads/shapiq_land

# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Run locally

```bash
cd /Users/Lucky/Downloads/shapiq_land
source .venv/bin/activate
python benchmark_explorer.py
```

Then open **<http://localhost:8050>** in your browser.

To use a different port:

```bash
PORT=8080 python benchmark_explorer.py
```

---

## Run in production (gunicorn)

```bash
cd /Users/Lucky/Downloads/shapiq_land
source .venv/bin/activate
gunicorn benchmark_explorer:server --bind 0.0.0.0:8050
```

---

## Adding new CSV data

| File | Used by | Notes |
|---|---|---|
| `results.csv` | Home, RQ1, RQ2 | Current benchmark runs |
| `results_dimensionality.csv` | RQ1 | Overrides `results.csv` for RQ1 if present |
| `results_accuracy.csv` | RQ2 | Overrides `results.csv` for RQ2 if present |
| `results_nn.csv` | RQ3 | Neural network benchmark runs |
| `results_trees.csv` | RQ4 | Tree-depth sweep runs |

Drop the file in this directory and **refresh the browser** — no code changes needed.

### Minimum CSV columns

All files follow the same schema:

```
dataset, model, n_features, n_samples, backend, library, computation_type,
approximator, budget, n_eval, runtime_s, n_model_evals, mean_abs_diff,
relative_mae, sign_agreement, mean_sample_rho, reference_backend
```

`results_trees.csv` benefits from an extra `tree_depth` (or `max_depth`) column — the
page uses it as the complexity axis. Without it, `model` is used instead.

---

## Deactivate the environment

```bash
deactivate
```
