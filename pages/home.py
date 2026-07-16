"""
pages/home.py — Landing page.

Hero banner with quick links to each research-question page.
"""
import os
import sys

import dash
from dash import html

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import shared as S

dash.register_page(__name__, path="/", name="Home", title="Benchmark Explorer")

_RQ_LINKS = [
    ("/rq1", "RQ1", "Accuracy",
     "How good are Shapley approximations — and what budget do you need to trust them?"),
    ("/rq2", "RQ2", "Dimensionality",
     "Which model-agnostic library scales best as feature count grows?"),
    ("/rq3", "RQ3", "Neural Networks",
     "Runtime and agreement for gradient-based explainers on neural nets."),
    ("/rq4", "RQ4", "Tree Models",
     "Exact tree explainers — depth scaling, agreement and failure modes."),
    ("/rq5", "RQ5", "GPU vs CPU",
     "Hardware speedup for neural-network attribution workloads."),
]

_GRADIENT_BORDER = (
    f"linear-gradient(135deg, {S.ACCENT} 0%, #7C3AED 55%, {S.PINK} 100%)"
)


def layout(**kwargs):
    return html.Div([
        html.Div([
            html.Div("SHAP-IQ",
                     style={"fontSize": "10px", "fontWeight": "700",
                            "color": S.ACCENT,
                            "letterSpacing": "0.16em", "textTransform": "uppercase",
                            "marginBottom": "12px"}),
            html.H1("Shapley Approximation Benchmark",
                    style={"fontSize": "26px", "fontWeight": "700",
                           "color": S.TEXT, "margin": "0 0 10px",
                           "letterSpacing": "-0.02em", "lineHeight": "1.25"}),
            html.P(
                "Interactive results from five benchmark studies comparing Shapley "
                "approximation libraries. Pick a research question to explore charts, "
                "filters and per-section notes.",
                style={"color": S.TEXT2, "fontSize": "13px",
                       "lineHeight": "1.75", "margin": "0 0 24px", "maxWidth": "560px"},
            ),
            html.Div([
                html.A(
                    [
                        html.Span(rq, style={
                            "fontSize": "9px", "fontWeight": "700",
                            "letterSpacing": "0.1em", "textTransform": "uppercase",
                            "color": S.ACCENT, "display": "block",
                            "marginBottom": "4px",
                        }),
                        html.Span(title, style={
                            "fontSize": "14px", "fontWeight": "600",
                            "color": S.TEXT, "display": "block",
                            "marginBottom": "6px", "letterSpacing": "-0.01em",
                        }),
                        html.Span(subtitle, style={
                            "fontSize": "11px", "color": S.TEXT2,
                            "lineHeight": "1.5", "display": "block",
                        }),
                    ],
                    href=href,
                    style={
                        "display": "block", "padding": "14px 16px",
                        "borderRadius": "10px", "textDecoration": "none",
                        "background": S.BG,
                        "border": f"1px solid {S.BORDER}",
                        "transition": "border-color 0.15s, box-shadow 0.15s",
                    },
                )
                for href, rq, title, subtitle in _RQ_LINKS
            ], style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fill, minmax(220px, 1fr))",
                "gap": "10px",
            }),
        ], style={
            "background": (
                f"linear-gradient({S.CARD}, {S.CARD}) padding-box, "
                f"{_GRADIENT_BORDER} border-box"
            ),
            "border": "4px solid transparent",
            "borderRadius": "14px",
            "padding": "36px 32px",
            "boxShadow": "0 1px 4px rgba(26,32,64,0.06)",
        }),
    ])
