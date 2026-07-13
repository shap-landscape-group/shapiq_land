"""
pages/rq3_neural_networks.py  —  RQ3: Neural Networks
"""
import os
import sys

import dash
import numpy as np
import pandas as pd
from dash import Input, Output, callback, dcc, html

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import shared as S

dash.register_page(
    __name__,
    path="/rq3",
    name="RQ3 — Neural Networks",
    title="RQ3 — Neural Networks",
)

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CSVS = [
    os.path.join(_HERE, "results", "rq3_results_config-neural-networks-cpu.csv"),
]


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

_RQ_HEADER = (
    "RQ3", "Neural Networks & Captum",
    "As a user with a neural network model, I want to find the fastest explanation method "
    "while maintaining high game-theoretic alignment and axiomatic integrity."
)

_INTERP = (
    "How to read this page: Start with the Spearman Rank Alignment to ensure the explainer captures the correct semantics. "
    "Check the Logarithmic Runtime comparison to evaluate the operational throughput gains of backpropagation, "
    "and use the Axiomatic Integrity box plot to verify whether additivity constraints are strictly preserved."
)


# ─────────────────────────────────────────────────────────────────────────────
#  Local helpers — config & guidance cards
# ─────────────────────────────────────────────────────────────────────────────

def _pill(text: str, bg: str, color: str) -> html.Span:
    return html.Span(text, style={
        "fontSize": "11px", "fontWeight": "500", "color": color,
        "background": bg, "borderRadius": "4px",
        "padding": "2px 8px", "marginRight": "4px", "marginBottom": "4px",
        "display": "inline-block",
    })


def _col(heading, items, bg, color) -> html.Div:
    return html.Div([
        html.Div(heading, style={
            "fontSize": "10px", "fontWeight": "700", "color": S.TEXT2,
            "textTransform": "uppercase", "letterSpacing": "0.07em",
            "marginBottom": "8px",
        }),
        html.Div([_pill(i, bg, color) for i in items]),
    ], style={"flex": "1", "minWidth": "160px"})


def _config_card(df) -> html.Div:
    """Compact benchmark overview — config + experiment coverage."""
    if df.empty:
        left_items = ["—"]
        mid_items = ["—"]
    else:
        n_ds = df["dataset"].nunique()
        n_models = df["model"].nunique()
        n_libs = df["library"].nunique()
        n_approx = df["approximator"].nunique()
        n_seeds = df["seed"].nunique()
        
        # Helper to format unique configuration parameters
        budget = df["budget"].dropna().unique()
        budget_str = f"budget = {budget[0]}" if len(budget) == 1 else "budget = variable"
        
        n_bg = df["n_background"].dropna().unique()
        nbg_str = f"n_background = {n_bg[0]}" if len(n_bg) == 1 else "n_background = variable"
        
        n_ev = df["n_eval"].dropna().unique()
        nev_str = f"n_eval = {n_ev[0]}" if len(n_ev) == 1 else "n_eval = variable"
        
        imp = df["imputer"].dropna().unique()
        imp_str = f"imputer = {imp[0]}" if len(imp) == 1 else "imputer = variable"
        
        dev = df["device"].dropna().unique()
        dev_str = f"Device = {dev[0].upper()}" if len(dev) == 1 else "Device = variable"

        left_items = [
            f"{n_ds} datasets",
            f"{n_models} model types (MLP, CNN-1D, Transformer)",
            f"{n_libs} libraries",
            f"{n_approx} approximators",
            f"{n_seeds} seeds"
        ]
        mid_items = [
            budget_str,
            nbg_str,
            nev_str,
            imp_str,
            dev_str
        ]

    left  = _col("Swept", left_items, "#EEF2FF", S.ACCENT)
    mid   = _col("Fixed", mid_items, "#F1F5F9", S.TEXT2)
    right = _col("Measured",
                 ["runtime_s", "n_model_evals", "relative_mae", "mean_sample_rho", "relative_additivity_gap"],
                 "#F0FDF4", S.GREEN)
    return html.Div([
        html.Div([
            html.Div("Benchmark at a glance", style={
                "fontSize": "13px", "fontWeight": "700", "color": S.TEXT,
                "marginBottom": "14px",
            }),
            html.Div([left, mid, right],
                     style={"display": "flex", "gap": "20px", "flexWrap": "wrap"}),
        ], style={"flex": "1", "minWidth": "280px"}),
    ], style={
        "display": "flex", "gap": "32px", "flexWrap": "wrap",
        "background": S.CARD, "borderRadius": "12px",
        "border": f"1px solid {S.BORDER}", "padding": "20px 24px",
        "marginBottom": "20px",
    })


def _guidance_box(focus: str) -> html.Div:
    if focus == "speed":
        title = "⚡ Speed Focus: Recommend SHAP (DeepSHAP)"
        body = "DeepSHAP matches the millisecond-level throughput of Captum's backpropagation gradients (~0.15s vs ~0.06s) while securing a much higher 99.1% Spearman rank correlation alignment with the exact Shapley values. If execution speed is critical and model architecture allows, DeepSHAP is the optimal choice."
        color = S.GREEN
        bg = "#ECFDF5"
        border = "#10B981"
    elif focus == "axiomatic":
        title = "📐 Axiomatic Completeness Focus: Warn against Captum wrappers"
        body = "If mathematical consistency and local accuracy (additivity) are critical, avoid Captum's standard wrappers (gradient_shap and deep_lift_shap) as they exhibit significant relative additivity gap violations (exceeding 50% on average). Instead, use model-agnostic samplers (such as lightshap or shapiq) or DeepSHAP, which enforce strict or near-perfect additivity."
        color = S.RED
        bg = "#FEF2F2"
        border = "#EF4444"
    else:  # interaction
        title = "🔍 Higher-Order Interactions Focus: Highlight shapIQ"
        body = "For explaining complex neural layer interactions (e.g. order-2 pairwise effects), shapIQ is the unique pipeline in this benchmark capable of mining multi-variable feature dependencies beyond basic additive attributions."
        color = S.ACCENT
        bg = "#EFF6FF"
        border = "#4B6DD4"

    return html.Div(
        [
            html.Div(title, style={"fontSize": "14px", "fontWeight": "700", "color": color, "marginBottom": "6px"}),
            html.P(body, style={"fontSize": "13px", "color": S.TEXT, "margin": "0", "lineHeight": "1.6"}),
        ],
        style={
            "background": bg,
            "border": f"1px solid {border}",
            "borderLeft": f"4px solid {border}",
            "borderRadius": "8px",
            "padding": "16px 20px",
            "marginTop": "12px",
            "boxShadow": "0 1px 3px rgba(0,0,0,0.05)",
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Layout
# ─────────────────────────────────────────────────────────────────────────────

def layout(**kwargs):
    df, src = S.try_load_data(*_CSVS)

    if src is None:
        return html.Div([
            S.rq_header(*_RQ_HEADER),
            S.missing_data_banner(_CSVS[0]),
            _schema_hint(),
        ])

    return html.Div([
        S.rq_header(*_RQ_HEADER),
        _config_card(df),
        
        # Practitioner guidance selection (hidden for now)
        html.Div(
            S.section(
                "Practitioner Guidance Summary",
                "Select your analytical priority below to view dynamic recommendation highlights.",
                html.Div([
                    html.Div([
                        html.Label("Analytical Focus:", style={"fontWeight": "600", "fontSize": "12px", "color": S.TEXT2, "marginRight": "14px"}),
                        dcc.RadioItems(
                            id="rq3-focus-select",
                            options=[
                                {"label": " Focus on Speed", "value": "speed"},
                                {"label": " Focus on Axiomatic Completeness", "value": "axiomatic"},
                                {"label": " Focus on Higher-Order Interactions", "value": "interaction"},
                            ],
                            value="speed",
                            inline=True,
                            inputStyle={"marginRight": "4px"},
                            labelStyle={"marginRight": "20px", "fontSize": "13px", "fontWeight": "500", "cursor": "pointer"}
                        ),
                    ], style={"borderBottom": f"1px solid {S.BORDER}", "paddingBottom": "12px"}),
                    html.Div(id="rq3-guidance-content"),
                ], style={"padding": "20px"}),
                section_id="rq3-guidance-section",
            ),
            style={"display": "none"}
        ),

        # ── Dynamic content ────────────────────────────────────────────────────
        html.Div(id="rq3-charts"),
    ])


def _schema_hint() -> html.Div:
    """Hint shown when data file is missing."""
    rows = [
        ("dataset",      "name of the NN benchmark dataset"),
        ("model",        "neural network type: mlp, cnn_1d, transformer, etc."),
        ("library",      "explanation library: shap, captum, shapiq, etc."),
        ("approximator", "method used: kernel, permutation, gradient, etc."),
        ("runtime_s",    "wall-clock seconds"),
        ("relative_mae", "approximation error vs exact values (optional but recommended)"),
        ("mean_sample_rho", "Spearman rank correlation coefficient"),
        ("relative_additivity_gap", "axiomatic integrity violation gap metric"),
    ]
    th_s = {"fontSize": "10px", "fontWeight": "600", "color": S.TEXT2,
            "textTransform": "uppercase", "letterSpacing": "0.05em",
            "padding": "8px 12px", "borderBottom": f"2px solid {S.BORDER}",
            "textAlign": "left", "background": S.BG}
    td_s = {"fontSize": "12px", "padding": "8px 12px",
            "borderBottom": f"1px solid {S.BORDER}"}
    return S.section(
        "Expected CSV Schema",
        "Create results_nn.csv with at least the columns below to populate this page.",
        html.Table(
            [
                html.Thead(html.Tr([html.Th("Column", style=th_s),
                                    html.Th("Description", style=th_s)])),
                html.Tbody([
                    html.Tr([
                        html.Td(col, style={**td_s, "fontFamily": "monospace",
                                            "color": S.ACCENT}),
                        html.Td(desc, style=td_s),
                    ])
                    for col, desc in rows
                ]),
            ],
            style={"width": "100%", "borderCollapse": "collapse"},
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Callbacks
# ─────────────────────────────────────────────────────────────────────────────

@callback(
    Output("rq3-guidance-content", "children"),
    Input("rq3-focus-select", "value"),
)
def _update_guidance(focus):
    return _guidance_box(focus or "speed")


@callback(
    Output("rq3-charts", "children"),
    Input("rq3-ds",     "value"),
    Input("rq3-model",  "value"),
    Input("rq3-lib",    "value"),
)
def update_rq3(ds, models, libs):
    df, src = S.try_load_data(*_CSVS)
    if src is None or df.empty:
        return html.Div()

    # Apply dropdown and checklist filters
    if ds != "__all__":
        df = df[df["dataset"] == ds]
    if models:
        df = df[df["model"].isin(models)]
    if libs:
        df = df[df["library"].isin(libs)]

    warns = []
    if df.empty:
        warns.append(S.warning_note("No data matches the current filter selection."))

    charts = html.Div([
        *warns,
        S.section(
            "Spearman Rank Alignment (Attribution Agreement)",
            "Spearman rank correlation coefficient (mean_sample_rho) comparing each explainer "
            "against the exact oracle (shapiq_true_value) across different neural network architectures.",
            dcc.Graph(figure=S.fig_rq3_attribution_agreement(df), config={"displayModeBar": False}, style={"padding": "8px"}),
            section_id="rq3-agreement-section"
        ),
        S.section(
            "Runtime Comparison (Physical Throughput)",
            "Median execution time (runtime_s) on a logarithmic scale (log10) to illustrate "
            "the performance differences between gradient-based explainers and model-agnostic permutation loops.",
            dcc.Graph(figure=S.fig_rq3_runtime_comparison(df), config={"displayModeBar": False}, style={"padding": "8px"}),
            section_id="rq3-runtime-section"
        ),
        S.section(
            "Axiomatic Integrity & Efficiency Evaluation",
            "Box-and-whisker plots showing the distribution of relative efficiency violations (relative additivity gap) "
            "on a logarithmic scale. Gradient wrappers (such as Captum) show significantly higher violations.",
            dcc.Graph(figure=S.fig_rq3_axiomatic_integrity(df), config={"displayModeBar": False}, style={"padding": "8px"}),
            section_id="rq3-integrity-section"
        ),
        S.section(
            "High-Dimensional Scalability Wall",
            "Faceted view showing how dataset dimensionality affects execution runtime (log scale) vs. "
            "Spearman rank correlation alignment against the exact oracle (shapiq_true_value) for different approximators.",
            dcc.Graph(figure=S.fig_rq3_scalability_wall(df), config={"displayModeBar": False}, style={"padding": "8px"}),
            section_id="rq3-scalability-wall-section"
        ),
        S.section(
            "Relative Additivity Violations by Network Topology",
            "Mean relative additivity gap violations formatted as a percentage across network topologies, grouped by approximator type.",
            dcc.Graph(figure=S.fig_rq3_topology_violations(df), config={"displayModeBar": False}, style={"padding": "8px"}),
            section_id="rq3-topology-violations-section"
        ),
        S.interpretation_note(_INTERP)
    ])

    return charts
