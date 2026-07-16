# SHAP Landscape Explorer

Interactive visualiser for the **Toolbox for Trustworthy Machine Learning** practical
([shap-landscape-group](https://github.com/shap-landscape-group)).
It compares Shapley-value libraries across five research questions.

| | |
|---|---|
| **Live** | [web-production-4ae37.up.railway.app](https://web-production-4ae37.up.railway.app/) |
| **Benchmarks** | [Shap-Landscape-Benchmark](https://github.com/shap-landscape-group/Shap-Landscape-Benchmark) |

Raw experiment outputs come from the benchmark repo; converters here turn them into
the CSVs the Dash app plots.

---

## Research questions

| Route | Question |
|-------|----------|
| [`/rq1`](https://web-production-4ae37.up.railway.app/rq1) | Approximation accuracy |
| [`/rq2`](https://web-production-4ae37.up.railway.app/rq2) | Dimensionality / scaling |
| [`/rq3`](https://web-production-4ae37.up.railway.app/rq3) | Neural networks |
| [`/rq4`](https://web-production-4ae37.up.railway.app/rq4) | Tree models |
| [`/rq5`](https://web-production-4ae37.up.railway.app/rq5) | GPU vs CPU |

---

## Structure

```
benchmark_explorer.py     # app shell — sidebar, topbar, page routing
pages/                    # one Dash page per RQ (+ home)
shared/                   # tokens, layout helpers, chart builders, data I/O
results/                  # raw benchmark dumps
results/converted/        # tables the pages load
results_converters/       # raw → converted pipeline (run after refreshing data)
Procfile                  # Railway / gunicorn entry
requirements.txt
```

Page modules keep the original data-pipeline names (`rq1_dimensionality.py`,
`rq2_accuracy.py`, …) while routes follow the RQ numbering above.

---

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python benchmark_explorer.py          # → http://localhost:8050
# or
gunicorn benchmark_explorer:server --bind 0.0.0.0:$PORT
```

---

## Refreshing data

1. Pull new CSVs from [Shap-Landscape-Benchmark](https://github.com/shap-landscape-group/Shap-Landscape-Benchmark) into `results/`.
2. Re-run the matching script in `results_converters/`.
3. Reload the browser — pages read only from `results/converted/`.
