"""
benchmark_explorer.py — Gunicorn entry-point for the SHAP-IQ Benchmark Explorer.

Run locally:   python benchmark_explorer.py
Deploy:        gunicorn benchmark_explorer:server --bind 0.0.0.0:$PORT
"""
import os

import dash
import pandas as pd
from dash import html, dcc, callback, Input, Output, State, page_container, page_registry
import shared as S

_RESULTS = os.path.join(os.path.dirname(__file__), "results")

# ── App ────────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server  # gunicorn target


# ── Sidebar nav page definitions ──────────────────────────────────────────────
_NAV_PAGES = [
    {
        "path":     "/rq1",
        "rq":       "RQ1",
        "title":    "Approximation Accuracy",
        "subtitle": "How good are the approximations — and how much compute do you need to trust them?",
    },
    {
        "path":     "/rq2",
        "rq":       "RQ2",
        "title":    "Dimensionality",
        "subtitle": "Which model-agnostic library is fastest for high-dimensional datasets?",
    },
    {
        "path":     "/rq3",
        "rq":       "RQ3",
        "title":    "Neural Networks",
        "subtitle": "Which library is fastest when your model is a neural network?",
    },
    {
        "path":     "/rq4",
        "rq":       "RQ4",
        "title":    "Tree Models",
        "subtitle": "Which library handles extreme tree depths without hitting bottlenecks or breaking?",
    },
    {
        "path":     "/rq5",
        "rq":       "RQ5",
        "title":    "GPU vs CPU",
        "subtitle": "How much speedup does running neural network explanations on GPU provide compared to CPU?",
    },
]


# ── App layout ─────────────────────────────────────────────────────────────────
app.layout = html.Div(
    [
        dcc.Location(id="url"),
        dcc.Store(id="sidebar-is-open", data=True),
        # Filter stores — always in layout so chart callbacks never race the topbar
        dcc.Store(id="rq2-ds",    data=None),
        dcc.Store(id="rq2-mdl",   data=None),
        dcc.Store(id="rq2-approx", data=None),   # None → all approximators
        dcc.Store(id="rq1-ds",    data=None),
        dcc.Store(id="rq1-mdl",   data=None),
        dcc.Store(id="rq1-approx", data=None),
        dcc.Store(id="rq4-ds",    data=None),
        dcc.Store(id="rq4-lib",   data=None),
        dcc.Store(id="rq4-model", data=None),

        # App shell
        html.Div(
            [
                # Sidebar
                html.Div(
                    [
                        html.Div(
                            [
                                html.A("SHAP-IQ Benchmark", href="/", className="sidebar-brand"),
                                html.Button("◀", id="sidebar-close-btn", n_clicks=0,
                                            className="sidebar-collapse-btn"),
                            ],
                            className="sidebar-top",
                        ),
                        html.Div(
                            [
                                html.Span("Research Questions", className="nav-section-label"),
                                html.Div(id="sidebar-nav"),
                            ],
                            className="sidebar-nav",
                        ),
                        html.Div(
                            html.A("shapiq docs ↗", href="https://shapiq.readthedocs.io/en",
                                   target="_blank", className="sidebar-footer-link"),
                            className="sidebar-footer",
                        ),
                    ],
                    id="sidebar",
                    className="sidebar",
                ),

                # Main area
                html.Div(
                    [
                        html.Div(
                            [
                                html.Button("☰", id="sidebar-open-btn", n_clicks=0,
                                            className="topbar-toggle-btn",
                                            title="Toggle sidebar",
                                            style={"alignSelf": "center", "flexShrink": "0"}),
                                html.Div(id="page-topbar-slot",
                                         style={"flex": "1", "display": "flex",
                                                "alignItems": "flex-start", "flexWrap": "wrap",
                                                "gap": "12px"}),
                            ],
                            className="main-topbar",
                        ),
                        html.Div(
                            html.Div(page_container, className="page-inner"),
                            className="page-scroll",
                        ),
                    ],
                    className="main-area",
                ),
            ],
            className="app-shell",
        ),
    ],
    className="page-wrapper",
)


# ── Index string ───────────────────────────────────────────────────────────────
_head = (
    "<!DOCTYPE html>\n<html>\n<head>\n{%metas%}"
    "<title>SHAP-IQ Benchmark Explorer</title>{%favicon%}{%css%}\n"
    "<link rel=\"preconnect\" href=\"https://fonts.googleapis.com\">\n"
    "<link href=\"https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700"
    "&display=swap\" rel=\"stylesheet\">\n<style>\n"
)
_tail = (
    "\n</style>\n</head>\n<body>\n{%app_entry%}\n"
    "<footer>{%config%}{%scripts%}{%renderer%}</footer>\n</body>\n</html>"
)
app.index_string = _head + S.COMMON_STYLES + _tail


# ── Callbacks: sidebar ────────────────────────────────────────────────────────
@callback(
    Output("sidebar-is-open", "data"),
    Input("sidebar-close-btn", "n_clicks"),
    Input("sidebar-open-btn",  "n_clicks"),
    State("sidebar-is-open",   "data"),
    prevent_initial_call=True,
)
def _toggle_sidebar(_close, _open, is_open):
    return not is_open


@callback(
    Output("sidebar", "className"),
    Input("sidebar-is-open", "data"),
)
def _sync_sidebar(is_open):
    return "sidebar" if is_open else "sidebar sidebar-collapsed"


@callback(
    Output("sidebar-nav", "children"),
    Input("url", "pathname"),
)
def _update_sidebar_nav(pathname):
    return [
        html.A(
            html.Div([
                html.Span(p["rq"],      className="nav-rq-tag"),
                html.Span(p["title"],   className="nav-title"),
                html.Div(p["subtitle"], className="nav-subtitle"),
            ]),
            href=p["path"],
            className="nav-card nav-card-active" if pathname == p["path"] else "nav-card",
        )
        for p in _NAV_PAGES
    ]


# ── Callback: page-specific topbar controls ───────────────────────────────────
@callback(
    Output("page-topbar-slot", "children"),
    Input("url", "pathname"),
)
def _render_page_topbar(pathname):
    """Populate the topbar with page-specific filter controls."""

    def _lbl(text):
        return html.Div(text, style={
            "fontSize": "9px", "fontWeight": "700", "color": S.TEXT2,
            "textTransform": "uppercase", "letterSpacing": "0.06em", "marginBottom": "3px",
        })

    def _src_tag(src):
        return html.Div(
            [html.Span("Source: ", style={"fontWeight": "600"}),
             html.Code(os.path.basename(src) if src else "—",
                       style={"fontFamily": "monospace", "fontSize": "10px",
                              "background": S.BG, "padding": "1px 5px", "borderRadius": "3px"})],
            style={"fontSize": "11px", "color": S.TEXT2, "alignSelf": "center",
                   "borderRight": f"1px solid {S.BORDER}", "paddingRight": "12px",
                   "marginRight": "4px", "whiteSpace": "nowrap"},
        )

    # ── RQ2 — Dimensionality (/rq2) ─────────────────────────────────────
    # Visible controls use -ctl suffix IDs; they sync into dcc.Store nodes
    # (rq2-ds, rq2-mdl, rq2-approx) that live in the main layout and are
    # always present, so the chart callback never races the topbar.
    if pathname == "/rq2":
        _csv = os.path.join(_RESULTS, "converted", "rq1_scaling_aggregated.csv")
        df = pd.read_csv(_csv) if os.path.exists(_csv) else pd.DataFrame(
            columns=["dataset", "model", "approximator"])
        src = _csv if os.path.exists(_csv) else None

        datasets = sorted(df["dataset"].dropna().unique()) if not df.empty else []
        models   = sorted(df["model"].dropna().unique()) if not df.empty else []
        approxs  = sorted(df["approximator"].dropna().unique()) if not df.empty else []
        _multi_dd = {"fontSize": "12px", "minHeight": "28px"}

        return [
            _src_tag(src),
            html.Div([
                _lbl("Dataset"),
                dcc.Dropdown(
                    id="rq2-ds-ctl",
                    options=[{"label": d, "value": d} for d in datasets],
                    value=None, multi=True, placeholder="All datasets",
                    style={**_multi_dd, "width": "180px"},
                ),
            ], style={"marginRight": "4px"}),
            html.Div([
                _lbl("Model"),
                dcc.Dropdown(
                    id="rq2-mdl-ctl",
                    options=[{"label": m, "value": m} for m in models],
                    value=None, multi=True, placeholder="All models",
                    style={**_multi_dd, "width": "180px"},
                ),
            ], style={"marginRight": "4px"}),
            html.Div([
                _lbl("Approximator"),
                dcc.Checklist(
                    id="rq2-approx-ctl",
                    options=[{"label": f" {a}", "value": a} for a in approxs],
                    value=list(approxs),
                    inline=True,
                    inputStyle={"marginRight": "3px"},
                    labelStyle={"marginRight": "8px", "fontSize": "12px", "cursor": "pointer"},
                ),
            ]),
        ]

    # ── RQ1 — Accuracy (/rq1) ─────────────────────────────────────────────
    if pathname == "/rq1":
        _csv = os.path.join(_RESULTS, "converted", "rq2_convergence_aggregated.csv")
        df = pd.read_csv(_csv) if os.path.exists(_csv) else pd.DataFrame(
            columns=["dataset", "model", "approximator"])
        src = _csv if os.path.exists(_csv) else None

        datasets = sorted(df["dataset"].dropna().unique()) if not df.empty else []
        models   = sorted(df["model"].dropna().unique()) if not df.empty else []
        approxs  = sorted(df["approximator"].dropna().unique()) if not df.empty else []
        _multi_dd = {"fontSize": "12px", "minHeight": "28px"}

        return [
            _src_tag(src),
            html.Div([
                _lbl("Dataset"),
                dcc.Dropdown(
                    id="rq1-ds-ctl",
                    options=[{"label": d, "value": d} for d in datasets],
                    value=None, multi=True, placeholder="All datasets",
                    style={**_multi_dd, "width": "180px"},
                ),
            ], style={"marginRight": "4px"}),
            html.Div([
                _lbl("Model"),
                dcc.Dropdown(
                    id="rq1-mdl-ctl",
                    options=[{"label": m, "value": m} for m in models],
                    value=None, multi=True, placeholder="All models",
                    style={**_multi_dd, "width": "180px"},
                ),
            ], style={"marginRight": "4px"}),
            html.Div([
                _lbl("Approximator"),
                dcc.Checklist(
                    id="rq1-approx-ctl",
                    options=[{"label": f" {a}", "value": a} for a in approxs],
                    value=list(approxs),
                    inline=True,
                    inputStyle={"marginRight": "3px"},
                    labelStyle={"marginRight": "8px", "fontSize": "12px", "cursor": "pointer"},
                ),
            ]),
        ]

    # ── RQ3 ───────────────────────────────────────────────────────────────
    if pathname == "/rq3":
        df, src = S.try_load_data(
            os.path.join(_RESULTS, "converted", "rq3_neural_networks_aggregated.csv"),
        )
 
        datasets = [{"label": "All datasets", "value": "__all__"}] + \
                   [{"label": d, "value": d} for d in sorted(df["dataset"].dropna().unique())]
        models   = sorted(df["model"].dropna().unique())   if not df.empty else []
        libs     = sorted(df["library"].dropna().unique()) if not df.empty else []
        _mlbl    = {"mlp": "MLP", "transformer": "Transformer", "cnn_1d": "CNN-1D"}
 
        return [
            _src_tag(src),
            html.Div([
                _lbl("Dataset"),
                dcc.Dropdown(id="rq3-ds", options=datasets, value="__all__",
                             clearable=False,
                             style={"width": "150px", "fontSize": "12px", "minHeight": "28px"}),
            ], style={"marginRight": "4px"}),
            html.Div([
                _lbl("Model"),
                dcc.Checklist(
                    id="rq3-model",
                    options=[{"label": f" {_mlbl.get(m, m)}", "value": m} for m in models],
                    value=list(models),
                    inline=True,
                    inputStyle={"marginRight": "3px"},
                    labelStyle={"marginRight": "8px", "fontSize": "12px", "cursor": "pointer"},
                ),
            ], style={"marginRight": "4px"}),
            html.Div([
                _lbl("Library"),
                dcc.Dropdown(
                    id="rq3-lib",
                    options=[{"label": lib, "value": lib} for lib in libs],
                    value=None,
                    multi=True,
                    clearable=True,
                    placeholder="All libraries",
                    style={"width": "240px", "fontSize": "12px", "minHeight": "28px"},
                ),
            ]),
        ]

    # ── RQ4 — Tree Models (/rq4) ────────────────────────────────────────────
    if pathname == "/rq4":
        _csv = os.path.join(_RESULTS, "converted", "rq4_trees_by_seed.csv")
        df = pd.read_csv(_csv) if os.path.exists(_csv) else pd.DataFrame()
        src = _csv if os.path.exists(_csv) else None

        datasets = sorted(df["dataset"].dropna().unique()) if not df.empty else []
        libs = sorted(df["library"].dropna().unique()) if not df.empty else []
        models = sorted(df["model"].dropna().unique()) if not df.empty and "model" in df.columns else []
        _multi_dd = {"fontSize": "12px", "minHeight": "28px"}

        return [
            _src_tag(src),
            html.Div([
                _lbl("Dataset"),
                dcc.Dropdown(
                    id="rq4-ds-ctl",
                    options=[{"label": d, "value": d} for d in datasets],
                    value=None, multi=True, placeholder="All datasets",
                    style={**_multi_dd, "width": "180px"},
                ),
            ], style={"marginRight": "4px"}),
            html.Div([
                _lbl("Library"),
                dcc.Dropdown(
                    id="rq4-lib-ctl",
                    options=[{"label": lib, "value": lib} for lib in libs],
                    value=None, multi=True, placeholder="All libraries",
                    style={**_multi_dd, "width": "180px"},
                ),
            ], style={"marginRight": "4px"}),
            html.Div([
                _lbl("Model"),
                dcc.Dropdown(
                    id="rq4-model-ctl",
                    options=[{"label": m, "value": m} for m in models],
                    value=None, multi=True, placeholder="All models",
                    style={**_multi_dd, "width": "180px"},
                ),
            ]),
        ]

    # ── RQ5 ───────────────────────────────────────────────────────────────
    if pathname == "/rq5":
        _csv = os.path.join(_RESULTS, "converted", "rq5_gpu_cpu_comparison_aggregated.csv")
        df, src = S.try_load_data(_csv)

        datasets = [{"label": "All datasets", "value": "__all__"}]
        models   = [{"label": "All models",   "value": "__all__"}]
        devices  = [{"label": "All devices",  "value": "__all__"}]

        if not df.empty:
            datasets += [{"label": d, "value": d} for d in sorted(df["dataset"].dropna().unique())]
            models   += [{"label": m, "value": m} for m in sorted(df["model"].dropna().unique())]
            if "device" in df.columns:
                devices += [{"label": dev.upper(), "value": dev} for dev in sorted(df["device"].dropna().unique())]

        return [
            _src_tag(src),
            html.Div([
                _lbl("Dataset"),
                dcc.Dropdown(id="rq5-ds", options=datasets, value="__all__",
                             clearable=False,
                             style={"width": "150px", "fontSize": "12px", "minHeight": "28px"}),
            ], style={"marginRight": "4px"}),
            html.Div([
                _lbl("Model"),
                dcc.Dropdown(id="rq5-mdl", options=models, value="__all__",
                             clearable=False,
                             style={"width": "140px", "fontSize": "12px", "minHeight": "28px"}),
            ], style={"marginRight": "4px"}),
            html.Div([
                _lbl("Device"),
                dcc.Dropdown(id="rq5-device", options=devices, value="__all__",
                             clearable=False,
                             style={"width": "120px", "fontSize": "12px", "minHeight": "28px"}),
            ], style={"marginRight": "4px"}),
        ]

    return []


# ── Topbar → Store sync callbacks ─────────────────────────────────────────────
# The -ctl controls live in the dynamic topbar slot; the stores are always in
# the main layout.  suppress_callback_exceptions handles the case where the
# topbar hasn't rendered the -ctl element yet.
@app.callback(Output("rq2-ds",    "data"), Input("rq2-ds-ctl",    "value"), prevent_initial_call=True)
def _sync_rq1_ds(v):    return v

@app.callback(Output("rq2-mdl",   "data"), Input("rq2-mdl-ctl",   "value"), prevent_initial_call=True)
def _sync_rq1_mdl(v):   return v

@app.callback(Output("rq2-approx","data"), Input("rq2-approx-ctl","value"), prevent_initial_call=True)
def _sync_rq1_approx(v): return v  # None or list — page handles both

@app.callback(Output("rq1-ds",    "data"), Input("rq1-ds-ctl",    "value"), prevent_initial_call=True)
def _sync_rq2_ds(v):    return v

@app.callback(Output("rq1-mdl",   "data"), Input("rq1-mdl-ctl",   "value"), prevent_initial_call=True)
def _sync_rq2_mdl(v):   return v

@app.callback(Output("rq1-approx","data"), Input("rq1-approx-ctl","value"), prevent_initial_call=True)
def _sync_rq2_approx(v): return v

@app.callback(Output("rq4-ds",    "data"), Input("rq4-ds-ctl",    "value"), prevent_initial_call=True)
def _sync_rq4_ds(v):    return v

@app.callback(Output("rq4-lib",   "data"), Input("rq4-lib-ctl",   "value"), prevent_initial_call=True)
def _sync_rq4_lib(v):   return v

@app.callback(Output("rq4-model", "data"), Input("rq4-model-ctl", "value"), prevent_initial_call=True)
def _sync_rq4_model(v): return v


# ── Dev server ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8050)),
        dev_tools_ui=False,
        dev_tools_props_check=False,
    )

