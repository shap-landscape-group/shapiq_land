# Shapley Benchmark Explorer

Four focused Dash pages, one per research question.

## Structure

```
benchmark_explorer.py        # app entry point — shell layout, sidebar
shared/                      # shared package
  tokens.py                  # design tokens, CSS, chart layout defaults
  data.py                    # CSV loading helpers
  charts.py                  # reusable chart builders
  layout.py                  # reusable layout components
pages/
  home.py                    # overview / landing  →  /
  rq1_dimensionality.py      # RQ1: high vs low dim  →  /rq1
  rq2_accuracy.py            # RQ2: approximation accuracy  →  /rq2
  rq3_neural_networks.py     # RQ3: NN runtime  →  /rq3
  rq4_trees.py               # RQ4: tree depth / bottlenecks  →  /rq4
results/
  rq1_dimensionality.csv     # RQ1 data
  rq2_accuracy.csv           # RQ2 data
  rq3_neural_networks.csv    # RQ3 data
  rq4_trees.csv              # RQ4 data
```

---

## Setup (first time only)

```bash
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
source .venv/bin/activate
python benchmark_explorer.py
```

Then open **<http://localhost:8050>** in your browser.

---

## Run in production (gunicorn)

```bash
source .venv/bin/activate
gunicorn benchmark_explorer:server --bind 0.0.0.0:$PORT
```

---

## Updating CSV data

Each page loads exactly one CSV from the `results/` folder:

| File | Used by |
|---|---|
| `results/rq1_dimensionality.csv` | Home, RQ1 |
| `results/rq2_accuracy.csv` | RQ2 |
| `results/rq3_neural_networks.csv` | RQ3 |
| `results/rq4_trees.csv` | RQ4 |

Replace the file and **refresh the browser** — no code changes needed.

### Minimum CSV columns

```
dataset, model, n_features, n_samples, backend, library, computation_type,
approximator, budget, n_eval, runtime_s, n_model_evals, mean_abs_diff,
relative_mae, sign_agreement, mean_sample_rho, reference_backend
```

`rq4_trees.csv` benefits from an extra `tree_depth` (or `max_depth`) column —
the page uses it as the complexity axis. Without it, `model` is used instead.

---

## Deactivate the environment

```bash
deactivate
```
