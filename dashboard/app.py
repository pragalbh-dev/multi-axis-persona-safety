"""Persona Steering Playground — Stage 1 layout-only skeleton.

Stage 8 fills the callbacks. Importing this module must NOT touch GPUs, load
models, or read parquets at import time. We only declare the layout +
constants the build pipeline needs.

Run locally for layout inspection:
    uv run python -m dashboard.app
"""

from __future__ import annotations

from pathlib import Path

from dash import Dash, dcc, html

# Reflects dashboard/data/schema.md and dashboard/wireframe.md. The set of
# valid models is determined at build time by the parquets in
# `dashboard/data/{model_id}.parquet`. The four core subjects (after the
# Stage 0 → 1 pivot) plus the two Gemma 4 thinking variants:
SUBJECT_MODELS: list[str] = [
    "gemma_2_27b",
    "qwen_3_32b",
    "gemma_4_31b_thinking_off",
    "gemma_4_31b_thinking_on",
]

STEERING_MODES: list[str] = [
    "none",
    "addition",
    "ablation",
    "capping",
    "cap+steer",
]

DEFENSE_CONFIGS: list[str] = [
    "AA cap",
    "+PC2",
    "+PC3",
    "+all-LASSO",
]

DATASETS: list[str] = ["dan", "shah_reconstructed"]


def make_layout() -> html.Div:
    """Build the static layout. No data binding — Stage 8 wires callbacks."""
    return html.Div(
        children=[
            html.H1("Multi-Axis Persona Safety — Steering Playground"),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "260px 1fr 320px", "gap": "16px"},
                children=[
                    _selectors_panel(),
                    _output_panel(),
                    _geometry_panel(),
                ],
            ),
            html.Div(
                "Stage 1 layout skeleton — Stage 8 fills callbacks.",
                style={"opacity": 0.5, "marginTop": "16px"},
            ),
        ]
    )


def _selectors_panel() -> html.Div:
    return html.Div(
        children=[
            html.H3("Selectors"),
            html.Label("Model"),
            dcc.Dropdown(
                id="model-dropdown",
                options=[{"label": m, "value": m} for m in SUBJECT_MODELS],
                value=SUBJECT_MODELS[0],
            ),
            html.Br(),
            html.Label("Steering mode"),
            dcc.Dropdown(
                id="steering-mode",
                options=[{"label": m, "value": m} for m in STEERING_MODES],
                value="none",
            ),
            html.Br(),
            html.Label("λ"),
            dcc.Slider(id="lambda-slider", min=-2.0, max=2.0, step=0.5, value=0.0),
            html.Br(),
            html.Label("Defense"),
            dcc.Checklist(
                id="defense-toggles",
                options=[{"label": d, "value": d} for d in DEFENSE_CONFIGS],
                value=[],
            ),
            html.Br(),
            html.Label("Dataset"),
            dcc.RadioItems(
                id="dataset-radio",
                options=[{"label": d, "value": d} for d in DATASETS],
                value="dan",
            ),
            html.Br(),
            html.Label("Prompt"),
            dcc.Input(id="prompt-search", type="text", placeholder="search prompts…"),
            dcc.Dropdown(id="prompt-dropdown", options=[]),
        ],
    )


def _output_panel() -> html.Div:
    return html.Div(
        children=[
            html.H3("Output"),
            html.Label("Input"),
            dcc.Textarea(id="input-text", disabled=True, style={"width": "100%", "height": 80}),
            html.Label("Response"),
            dcc.Textarea(id="response-text", disabled=True, style={"width": "100%", "height": 200}),
            html.Div(id="harm-badge"),
            html.Div(id="asr-row"),
            html.Div(id="capability-row"),
        ],
    )


def _geometry_panel() -> html.Div:
    return html.Div(
        children=[
            html.H3("Geometry"),
            dcc.Graph(id="pc-scatter"),
            html.Div(id="aa-projection-bar"),
            html.Div(id="capping-markers"),
        ],
    )


# Bottom-of-module Dash instance so `python -m dashboard.app` works without
# Stage 8's wiring being in place.
app = Dash(__name__, title="Persona Steering Playground")
app.layout = make_layout()


if __name__ == "__main__":  # pragma: no cover
    # Stage 8 may swap to gunicorn / HF Spaces serving.
    here = Path(__file__).resolve().parent
    print(f"Layout-only Dash app. Data shards expected at {here / 'data'}/*.parquet")
    app.run(debug=False, port=8050)
