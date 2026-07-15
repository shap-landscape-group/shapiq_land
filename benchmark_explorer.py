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


# ── Advisor panel (content) ────────────────────────────────────────────────────
_T = "transform 0.28s cubic-bezier(0.4, 0, 0.2, 1)"
_PANEL_CLOSED = {
    "transform": "translateX(100%)", "transition": _T,
    "boxShadow": "-6px 0 32px rgba(10,15,40,0.55)",
    "borderLeft": "1px solid rgba(165,184,248,0.18)",
}
_PANEL_OPEN = {
    "transform": "translateX(0)", "transition": _T,
    "boxShadow": "-6px 0 32px rgba(10,15,40,0.55)",
    "borderLeft": "1px solid rgba(165,184,248,0.18)",
}
_OVERLAY_CLOSED = {"opacity": "0", "pointerEvents": "none", "transition": "opacity 0.25s"}
_OVERLAY_OPEN   = {"opacity": "1", "pointerEvents": "all",  "transition": "opacity 0.25s"}
_ADV_LABEL_STYLE = {
    "color": "rgba(255,255,255,0.5)", "fontSize": "10px", "fontWeight": "600",
    "textTransform": "uppercase", "letterSpacing": "0.1em",
    "display": "block", "marginBottom": "4px",
}
_ADV_OPT_STYLE = {
    "color": "rgba(255,255,255,0.82)", "fontSize": "13px",
    "whiteSpace": "normal", "overflow": "visible",
    "display": "flex", "alignItems": "flex-start", "flexWrap": "wrap",
}
_ADV_IN_STYLE  = {"marginRight": "8px", "marginTop": "2px", "flexShrink": "0"}
_ADV_RADIO_STYLE = {"display": "flex", "flexDirection": "column", "gap": "8px", "marginTop": "8px"}

_MODEL_OPTIONS = [
    {"label": "Tree model  (random forest, XGBoost, …)",      "value": "tree"},
    {"label": "Neural network  (PyTorch, CNN, Transformer, …)", "value": "neural_network"},
    {"label": "Other / black-box model",                        "value": "black_box"},
]
_NEED_OPTIONS = [
    {"label": "Speed  — fastest approximations",      "value": "speed"},
    {"label": "Accuracy  — most trustworthy values",  "value": "accuracy"},
    {"label": "Scale  — many features or samples",    "value": "memory"},
]
_DIM_OPTIONS = [
    {"label": "Low-dimensional   (< 20 features)",  "value": "low"},
    {"label": "High-dimensional  (≥ 20 features)",  "value": "high"},
]

_advisor_body = html.Div(
    [
        html.Div(
            [
                html.Div(
                    html.Span("Library Advisor",
                              style={"fontWeight": "700", "fontSize": "15px",
                                     "color": "white", "letterSpacing": "-0.01em"}),
                ),
                html.Button("✕", id="advisor-close-btn", n_clicks=0,
                            style={"background": "none", "border": "none",
                                   "color": "rgba(255,255,255,0.4)", "fontSize": "16px",
                                   "cursor": "pointer", "padding": "4px 8px", "lineHeight": "1",
                                   "borderRadius": "4px"}),
            ],
            style={"display": "flex", "justifyContent": "space-between", "alignItems": "center",
                   "borderBottom": "1px solid rgba(255,255,255,0.1)", "padding": "18px 20px",
                   "position": "sticky", "top": "0", "background": "#2B2B2B", "zIndex": "10"},
        ),
        html.Div(
            [
                html.P("Answer three questions to get a tailored library recommendation "
                       "with links to the relevant benchmark charts.",
                       style={"fontSize": "12px", "color": "rgba(255,255,255,0.5)",
                              "lineHeight": "1.7", "margin": "0 0 22px"}),
                html.Div([
                    html.Span("1 — Model type", style=_ADV_LABEL_STYLE),
                    dcc.RadioItems(id="adv-model-type", options=_MODEL_OPTIONS, value=None,
                                   labelStyle=_ADV_OPT_STYLE, style=_ADV_RADIO_STYLE,
                                   inputStyle=_ADV_IN_STYLE),
                ], style={"marginBottom": "22px"}),
                html.Div([
                    html.Span("2 — What matters most?", style=_ADV_LABEL_STYLE),
                    dcc.RadioItems(id="adv-primary-need", options=_NEED_OPTIONS, value=None,
                                   labelStyle=_ADV_OPT_STYLE, style=_ADV_RADIO_STYLE,
                                   inputStyle=_ADV_IN_STYLE),
                ], style={"marginBottom": "22px"}),
                html.Div([
                    html.Span("3 — Feature count", style=_ADV_LABEL_STYLE),
                    dcc.RadioItems(id="adv-feature-count", options=_DIM_OPTIONS, value=None,
                                   labelStyle=_ADV_OPT_STYLE, style=_ADV_RADIO_STYLE,
                                   inputStyle=_ADV_IN_STYLE),
                ], style={"marginBottom": "24px"}),
                html.Div(id="adv-recommendation"),
            ],
            style={"padding": "20px"},
        ),
    ],
    id="advisor-panel",
    className="advisor-panel",
    style=_PANEL_CLOSED,
)


# ── App layout ─────────────────────────────────────────────────────────────────
app.layout = html.Div(
    [
        dcc.Location(id="url"),
        dcc.Store(id="sidebar-is-open", data=True),
        # Filter stores — always in layout so chart callbacks never race the topbar
        dcc.Store(id="rq2-ds",    data="__all__"),
        dcc.Store(id="rq2-mdl",   data="__all__"),
        dcc.Store(id="rq2-approx", data=None),   # None → all approximators
        dcc.Store(id="rq1-ds",    data="__all__"),
        dcc.Store(id="rq1-mdl",   data="__all__"),
        dcc.Store(id="rq1-approx", data=None),

        # Advisor overlay + panel
        html.Div(id="advisor-overlay", className="advisor-overlay", n_clicks=0,
                 style=_OVERLAY_CLOSED),
        _advisor_body,

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

        # Advisor strip (26px right edge)
        html.Button(
            html.Span("Advisor", className="advisor-strip-label"),
            id="advisor-strip",
            className="advisor-strip",
            n_clicks=0,
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


# ── Callbacks: advisor ────────────────────────────────────────────────────────
@callback(
    Output("advisor-panel",   "style"),
    Output("advisor-overlay", "style"),
    Input("advisor-strip",     "n_clicks"),
    Input("advisor-close-btn", "n_clicks"),
    Input("advisor-overlay",   "n_clicks"),
    State("advisor-panel",    "style"),
    prevent_initial_call=True,
)
def _toggle_advisor(_strip, _close, _overlay, panel_style):
    from dash import ctx
    is_open = (panel_style or {}).get("transform") == "translateX(0)"
    if ctx.triggered_id == "advisor-strip":
        return (_PANEL_CLOSED, _OVERLAY_CLOSED) if is_open else (_PANEL_OPEN, _OVERLAY_OPEN)
    return _PANEL_CLOSED, _OVERLAY_CLOSED


# ── Callback: advisor recommendation ─────────────────────────────────────────
@callback(
    Output("adv-recommendation", "children"),
    Input("adv-model-type",    "value"),
    Input("adv-primary-need",  "value"),
    Input("adv-feature-count", "value"),
)
def _advisor_rec(model_type, primary_need, feature_count):
    answered = sum(v is not None for v in [model_type, primary_need, feature_count])

    if answered == 0:
        return html.P("Select options above to see a recommendation.",
                      style={"fontSize": "12px", "color": "rgba(255,255,255,0.3)",
                             "textAlign": "center", "marginTop": "8px"})

    rec = S.recommend_library(
        model_type    or "black_box",
        primary_need  or "speed",
        feature_count or "low",
    )

    connector_items = [
        html.Li(
            html.A(f"→ {name.replace('_', ' ').title()}", href=rec["rq_page"],
                   style={"color": S.ACCENT, "textDecoration": "none",
                          "fontSize": "12px", "fontWeight": "500"}),
            style={"listStyle": "none", "marginBottom": "5px"},
        )
        for name, section_id in rec["chart_ids"].items()
    ]

    return html.Div([
        html.Div([
            html.Div("Recommended",
                     style={"fontSize": "9px", "fontWeight": "700", "color": S.PINK,
                            "textTransform": "uppercase", "letterSpacing": "0.1em",
                            "marginBottom": "6px"}),
            html.Div(rec["library"],
                     style={"fontSize": "16px", "fontWeight": "700", "color": "white",
                            "lineHeight": "1.3", "letterSpacing": "-0.01em"}),
            html.Div(f"via {rec['approximator']}",
                     style={"fontSize": "11px", "color": "rgba(255,255,255,0.45)",
                            "marginTop": "3px"}),
        ], style={"background": "rgba(255,255,255,0.06)",
                  "border": "1px solid rgba(255,255,255,0.12)",
                  "borderTop": f"3px solid {S.PINK}",
                  "borderRadius": "8px", "padding": "14px 16px", "marginBottom": "14px"}),
        html.P(rec["why"],
               style={"fontSize": "12px", "color": "rgba(255,255,255,0.62)",
                      "lineHeight": "1.75", "margin": "0 0 14px"}),
        html.Div([
            html.Div(f"See in {rec['rq_name']}",
                     style={"fontSize": "9px", "fontWeight": "700",
                            "color": "rgba(255,255,255,0.3)",
                            "textTransform": "uppercase", "letterSpacing": "0.08em",
                            "marginBottom": "8px"}),
            html.Ul(connector_items, style={"margin": "0", "padding": "0"}),
            dcc.Link(f"Also check: {rec.get('secondary_name', '')} →",
                     href=rec.get("secondary_page", "/"),
                     style={"fontSize": "11px", "color": "rgba(255,255,255,0.38)",
                            "display": "block", "marginTop": "10px"},
                     ) if "secondary_page" in rec else None,
        ], style={"borderTop": "1px solid rgba(255,255,255,0.1)", "paddingTop": "12px"}),
        html.Div([
            html.Span(m, style={
                "fontSize": "10px", "fontWeight": "500",
                "color": "rgba(255,255,255,0.45)",
                "background": "rgba(255,255,255,0.07)",
                "borderRadius": "3px", "padding": "2px 7px",
                "marginRight": "4px", "display": "inline-block", "marginTop": "4px",
            }) for m in rec["metrics"]
        ], style={"marginTop": "12px"}),
        html.P("* Partial — refine answers above for a sharper pick." if answered < 3 else "",
               style={"fontSize": "10px", "color": "rgba(255,255,255,0.25)",
                      "marginTop": "10px", "fontStyle": "italic"}),
    ])


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

    # ── RQ1 ───────────────────────────────────────────────────────────────
    # Visible controls use -ctl suffix IDs; they sync into dcc.Store nodes
    # (rq1-ds, rq1-mdl, rq1-approx) that live in the main layout and are
    # always present, so the chart callback never races the topbar.
    if pathname == "/rq2":
        _csv = os.path.join(_RESULTS, "converted", "rq2_convergence_aggregated.csv")
        df = pd.read_csv(_csv) if os.path.exists(_csv) else pd.DataFrame(
            columns=["dataset", "model", "approximator"])
        src = _csv if os.path.exists(_csv) else None

        datasets = [{"label": "All datasets", "value": "__all__"}] + \
                   [{"label": d, "value": d} for d in sorted(df["dataset"].dropna().unique())]
        models   = [{"label": "All models",   "value": "__all__"}] + \
                   [{"label": m, "value": m} for m in sorted(df["model"].dropna().unique())]
        approxs  = sorted(df["approximator"].dropna().unique()) if not df.empty else []

        return [
            _src_tag(src),
            html.Div([
                _lbl("Dataset"),
                dcc.Dropdown(id="rq2-ds-ctl", options=datasets, value="__all__",
                             clearable=False,
                             style={"width": "150px", "fontSize": "12px", "minHeight": "28px"}),
            ], style={"marginRight": "4px"}),
            html.Div([
                _lbl("Model"),
                dcc.Dropdown(id="rq2-mdl-ctl", options=models, value="__all__",
                             clearable=False,
                             style={"width": "140px", "fontSize": "12px", "minHeight": "28px"}),
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

    # ── RQ2 ───────────────────────────────────────────────────────────────
    if pathname == "/rq1":
        _csv = os.path.join(_RESULTS, "converted", "rq1_scaling_aggregated.csv")
        df = pd.read_csv(_csv) if os.path.exists(_csv) else pd.DataFrame(
            columns=["dataset", "model", "approximator"])
        src = _csv if os.path.exists(_csv) else None

        datasets = [{"label": "All datasets", "value": "__all__"}] + \
                   [{"label": d, "value": d} for d in sorted(df["dataset"].dropna().unique())]
        models   = [{"label": "All models",   "value": "__all__"}] + \
                   [{"label": m, "value": m} for m in sorted(df["model"].dropna().unique())]
        approxs  = sorted(df["approximator"].dropna().unique()) if not df.empty else []

        return [
            _src_tag(src),
            html.Div([
                _lbl("Dataset"),
                dcc.Dropdown(id="rq1-ds-ctl", options=datasets, value="__all__",
                             clearable=False,
                             style={"width": "150px", "fontSize": "12px", "minHeight": "28px"}),
            ], style={"marginRight": "4px"}),
            html.Div([
                _lbl("Model"),
                dcc.Dropdown(id="rq1-mdl-ctl", options=models, value="__all__",
                             clearable=False,
                             style={"width": "140px", "fontSize": "12px", "minHeight": "28px"}),
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
def _sync_rq1_ds(v):    return v or "__all__"

@app.callback(Output("rq2-mdl",   "data"), Input("rq2-mdl-ctl",   "value"), prevent_initial_call=True)
def _sync_rq1_mdl(v):   return v or "__all__"

@app.callback(Output("rq2-approx","data"), Input("rq2-approx-ctl","value"), prevent_initial_call=True)
def _sync_rq1_approx(v): return v  # None or list — page handles both

@app.callback(Output("rq1-ds",    "data"), Input("rq1-ds-ctl",    "value"), prevent_initial_call=True)
def _sync_rq2_ds(v):    return v or "__all__"

@app.callback(Output("rq1-mdl",   "data"), Input("rq1-mdl-ctl",   "value"), prevent_initial_call=True)
def _sync_rq2_mdl(v):   return v or "__all__"

@app.callback(Output("rq1-approx","data"), Input("rq1-approx-ctl","value"), prevent_initial_call=True)
def _sync_rq2_approx(v): return v


# ── Dev server ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8050)),
        dev_tools_ui=False,
        dev_tools_props_check=False,
    )

