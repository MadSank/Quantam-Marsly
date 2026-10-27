"""Theme definitions, color constants, Plotly templates, and global CSS for the Mars-QCNN application."""

import streamlit as st
import plotly.graph_objects as go
from typing import Optional

# ==============================================================================
# DESIGN SYSTEM COLOR PALETTE CONSTANTS (Mission Console Tokens)
# ==============================================================================

# Model Identity Palette (Muted, Distinct Lightness)
COLOR_CNN = "#6FA3D4"            # Muted Steel Blue (Classical ResNet-18)
COLOR_QCNN = "#A08CCB"           # Muted Violet (Hybrid QCNN Head)
COLOR_HYBRID = "#A08CCB"         # Alias for QCNN
COLOR_MLP = "#8A97A8"            # Cool Slate Grey (Matched MLP Control)
COLOR_BASELINE = "#C9B27C"       # Sand (Majority Baseline)

# Semantic & Scientific Status Palette
COLOR_SAFE = "#5FB08C"           # Muted Emerald Green
COLOR_UNSAFE = "#E0605C"         # Muted Crimson Red / Coral
COLOR_UNSAFE_TEXT = "#EC7B77"    # High-contrast red text for badges
COLOR_AGREE = "#A9B4C2"          # Neutral Slate (Agreement != Correctness)
COLOR_DISAGREE = "#E0B04C"       # Warm Gold (Disagreement warning)
COLOR_WARN = "#E0B04C"           # Warm Gold
COLOR_CORRECT = "#5FB08C"        # Muted Emerald Green
COLOR_INCORRECT = "#E0605C"      # Muted Crimson Red

# Canvas, Surface & Line Palette
COLOR_BG = "#0B0F14"             # Page Canvas
COLOR_BG_CARD = "#121821"        # Panel / Card Surface
COLOR_SURFACE_2 = "#18202B"      # Raised Surface / Table Header
COLOR_SIDEBAR = "#0F141B"        # Sidebar Background
COLOR_BORDER = "#252F3D"         # Panel Hairline Border
COLOR_LINE_STRONG = "#36445A"    # Emphasized Rules & Axis Lines
COLOR_CONTROL_BORDER = "#5B6B82" # Input & Button Control Border
COLOR_TEXT = "#E8ECF1"           # Primary High-Contrast Text
COLOR_MUTED = "#A9B4C2"          # Secondary Text
COLOR_TEXT_3 = "#8392A5"         # Captions & Meta Text
COLOR_GRID = "#1E2733"           # Subtle Chart Gridlines

# Accent Palette (Interface Chrome Only - Never encodes data)
COLOR_RUST = "#D0693F"           # Primary Mars Rust
COLOR_RUST_HI = "#E08A63"        # Highlight / Hover
COLOR_RUST_FILL = "#B4532F"      # Primary Button Fill
COLOR_RUST_FILL_HOVER = "#A94A28"# Primary Button Hover
COLOR_RUST_TINT = "rgba(208, 105, 63, 0.12)" # Active Nav Wash

# Typography Font Stacks
FONT_SANS = '"IBM Plex Sans", "Segoe UI Variable Text", "Segoe UI", system-ui, -apple-system, sans-serif'
FONT_MONO = '"IBM Plex Mono", "Cascadia Mono", Consolas, "SF Mono", monospace'

# ==============================================================================
# GLOBAL STYLESHEET (CSS)
# ==============================================================================

CUSTOM_CSS = f"""
<style>
/* CSS Variables rooted from design tokens */
:root {{
    --mq-bg: {COLOR_BG};
    --mq-surface: {COLOR_BG_CARD};
    --mq-surface-2: {COLOR_SURFACE_2};
    --mq-sidebar: {COLOR_SIDEBAR};
    --mq-line: {COLOR_BORDER};
    --mq-line-strong: {COLOR_LINE_STRONG};
    --mq-control-border: {COLOR_CONTROL_BORDER};
    --mq-text: {COLOR_TEXT};
    --mq-text-2: {COLOR_MUTED};
    --mq-text-3: {COLOR_TEXT_3};
    --mq-grid: {COLOR_GRID};
    --mq-rust: {COLOR_RUST};
    --mq-rust-hi: {COLOR_RUST_HI};
    --mq-rust-fill: {COLOR_RUST_FILL};
    --mq-rust-fill-hover: {COLOR_RUST_FILL_HOVER};
    --mq-rust-tint: {COLOR_RUST_TINT};
    --mq-cnn: {COLOR_CNN};
    --mq-qcnn: {COLOR_QCNN};
    --mq-mlp: {COLOR_MLP};
    --mq-baseline: {COLOR_BASELINE};
    --mq-safe: {COLOR_SAFE};
    --mq-unsafe: {COLOR_UNSAFE};
    --mq-unsafe-text: {COLOR_UNSAFE_TEXT};
    --mq-warn: {COLOR_WARN};
    --mq-font-sans: {FONT_SANS};
    --mq-font-mono: {FONT_MONO};
    --mq-r: 4px;
    --mq-r-sm: 2px;
}}

/* Canvas and Main Container Rules */
.main .block-container {{
    max-width: 1440px !important;
    padding-top: 1.75rem !important;
    padding-bottom: 3rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}}

/* Typography Refinements */
h1, h2, h3, h4, h5, h6 {{
    font-family: var(--mq-font-sans) !important;
    color: var(--mq-text) !important;
    letter-spacing: -0.01em;
}}

h1 {{
    font-size: 1.75rem !important;
    font-weight: 600 !important;
    margin-bottom: 0.25rem !important;
}}

h2 {{
    font-size: 1.25rem !important;
    font-weight: 600 !important;
    margin-top: 1.5rem !important;
    margin-bottom: 0.5rem !important;
}}

h3 {{
    font-size: 1.05rem !important;
    font-weight: 600 !important;
}}

/* Page Header Components */
.mq-eyebrow {{
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: var(--mq-font-mono);
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--mq-text-3);
    margin-bottom: 4px;
}}

.mq-eyebrow-dash {{
    display: inline-block;
    width: 14px;
    height: 2px;
    background: var(--mq-rust);
}}

.mq-eyebrow-step {{
    color: var(--mq-rust-hi);
}}

.mq-header-line {{
    width: 100%;
    height: 1px;
    background: var(--mq-line);
    margin-top: 12px;
    margin-bottom: 18px;
}}

/* Step Progress Indicator (6-segment bar) */
.mq-step-progress {{
    display: inline-flex;
    align-items: center;
    gap: 4px;
}}

.mq-step-segment {{
    width: 22px;
    height: 4px;
    border-radius: 1px;
    background: var(--mq-line-strong);
}}

.mq-step-segment.active {{
    background: var(--mq-rust);
}}

/* Native Metric Container Restyling */
div[data-testid="stMetric"] {{
    background: transparent !important;
    padding: 6px 12px !important;
    border: none !important;
}}

div[data-testid="stMetricLabel"] {{
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
    color: var(--mq-text-3) !important;
    font-weight: 500 !important;
}}

div[data-testid="stMetricValue"] {{
    font-family: var(--mq-font-mono) !important;
    font-size: 1.35rem !important;
    font-weight: 500 !important;
    font-variant-numeric: tabular-nums !important;
    color: var(--mq-text) !important;
}}

/* Card Surfaces & Panels */
.metric-card, .mq-card {{
    background: var(--mq-surface);
    border: 1px solid var(--mq-line);
    border-radius: var(--mq-r);
    padding: 16px;
    margin-bottom: 12px;
}}

/* Status and Model Badges */
.badge, .mq-badge {{
    display: inline-flex;
    align-items: center;
    padding: 2px 6px;
    font-family: var(--mq-font-mono);
    font-size: 0.6875rem;
    font-weight: 600;
    border-radius: var(--mq-r-sm);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    line-height: 1.2;
}}

.badge-safe, .badge-correct, .mq-badge-safe {{
    background-color: rgba(95, 176, 140, 0.14);
    color: var(--mq-safe);
    border: 1px solid rgba(95, 176, 140, 0.45);
}}

.badge-unsafe, .badge-incorrect, .mq-badge-unsafe {{
    background-color: rgba(224, 96, 92, 0.14);
    color: var(--mq-unsafe-text);
    border: 1px solid rgba(224, 96, 92, 0.45);
}}

.badge-disagree, .mq-badge-warn {{
    background-color: rgba(224, 176, 76, 0.14);
    color: var(--mq-warn);
    border: 1px solid rgba(224, 176, 76, 0.45);
}}

.badge-neutral, .mq-badge-neutral {{
    background-color: rgba(169, 180, 194, 0.12);
    color: var(--mq-text-2);
    border: 1px solid rgba(169, 180, 194, 0.35);
}}

.badge-model-cnn, .mq-badge-cnn {{
    background-color: rgba(111, 163, 212, 0.14);
    color: var(--mq-cnn);
    border: 1px solid rgba(111, 163, 212, 0.45);
}}

.badge-model-qcnn, .mq-badge-qcnn {{
    background-color: rgba(160, 140, 203, 0.14);
    color: var(--mq-qcnn);
    border: 1px solid rgba(160, 140, 203, 0.45);
}}

.badge-model-mlp, .mq-badge-mlp {{
    background-color: rgba(138, 151, 168, 0.14);
    color: var(--mq-mlp);
    border: 1px solid rgba(138, 151, 168, 0.45);
}}

/* Model Identity Chips */
.mq-chip {{
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 1px;
    margin-right: 6px;
}}
.mq-chip-cnn {{ background: var(--mq-cnn); }}
.mq-chip-qcnn {{ background: var(--mq-qcnn); }}
.mq-chip-mlp {{ background: var(--mq-mlp); }}
.mq-chip-baseline {{ background: var(--mq-baseline); }}

/* Note Blocks (Replacing Heavy Border Bars with Flat Ruled Panels) */
.mq-note, .callout-box, .callout-warning {{
    background: var(--mq-surface);
    border: 1px solid var(--mq-line);
    border-radius: var(--mq-r);
    padding: 14px 16px;
    margin: 14px 0;
}}

.mq-note-integrity {{
    border-color: rgba(208, 105, 63, 0.45);
    background: var(--mq-surface);
}}

.mq-note-warning, .callout-warning {{
    border-color: rgba(224, 176, 76, 0.40);
    background: rgba(224, 176, 76, 0.05);
}}

.mq-note-label {{
    font-family: var(--mq-font-mono);
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 6px;
}}

/* Key-Value Tables */
.mq-kv-table {{
    width: 100%;
    font-size: 0.84rem;
    border-collapse: collapse;
}}

.mq-kv-table tr {{
    border-bottom: 1px solid var(--mq-line);
}}

.mq-kv-table tr:last-child {{
    border-bottom: none;
}}

.mq-kv-label {{
    color: var(--mq-text-3);
    padding: 5px 0;
    width: 45%;
}}

.mq-kv-value {{
    color: var(--mq-text);
    font-family: var(--mq-font-mono);
    font-variant-numeric: tabular-nums;
    text-align: right;
    padding: 5px 0;
}}

/* Primary Button Styling */
button[data-testid="stBaseButton-primary"],
button[kind="primary"] {{
    background-color: var(--mq-rust-fill) !important;
    border-color: var(--mq-rust-fill) !important;
    color: #FFFFFF !important;
    border-radius: var(--mq-r) !important;
    font-weight: 500 !important;
}}

button[data-testid="stBaseButton-primary"]:hover,
button[kind="primary"]:hover {{
    background-color: var(--mq-rust-fill-hover) !important;
    border-color: var(--mq-rust-fill-hover) !important;
}}

/* Secondary / Default Button Styling */
button[data-testid="stBaseButton-secondary"],
button[kind="secondary"] {{
    background-color: transparent !important;
    border: 1px solid var(--mq-control-border) !important;
    color: var(--mq-text) !important;
    border-radius: var(--mq-r) !important;
}}

button[data-testid="stBaseButton-secondary"]:hover,
button[kind="secondary"]:hover {{
    background-color: var(--mq-surface-2) !important;
    border-color: var(--mq-rust-hi) !important;
}}

/* Tab Styling */
button[role="tab"] {{
    font-family: var(--mq-font-sans) !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    color: var(--mq-text-2) !important;
    padding: 8px 16px !important;
}}

button[role="tab"][aria-selected="true"] {{
    color: var(--mq-text) !important;
}}

/* Sidebar Active Marker */
section[data-testid="stSidebar"] {{
    background-color: var(--mq-sidebar) !important;
}}

div[data-testid="stSidebarNav"] a[aria-current="page"] {{
    background-color: var(--mq-rust-tint) !important;
    border-left: 2px solid var(--mq-rust) !important;
    color: var(--mq-text) !important;
}}

/* Dataframe & Table Styling */
div[data-testid="stDataFrame"] {{
    border: 1px solid var(--mq-line) !important;
    border-radius: var(--mq-r) !important;
    background: var(--mq-surface) !important;
}}

/* Native expander border refinement */
div[data-testid="stExpander"] {{
    border: 1px solid var(--mq-line) !important;
    border-radius: var(--mq-r) !important;
    background: var(--mq-surface) !important;
    margin-bottom: 12px !important;
}}
</style>
"""


def apply_theme():
    """Inject global custom CSS into Streamlit page."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def badge_html(text: str, badge_type: str = "safe") -> str:
    """Generate HTML for a styled status or model badge."""
    clean_type = badge_type.lower().replace("badge-", "")
    return f'<span class="badge badge-{clean_type}">{text}</span>'


# ==============================================================================
# PLOTLY CHART THEME & BUILDER HELPER (Single Mission Console Template)
# ==============================================================================

MARSQCNN_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_SANS, size=12, color=COLOR_MUTED),
        title=dict(
            font=dict(family=FONT_SANS, size=14, color=COLOR_TEXT),
            x=0.0,
            xanchor="left",
            y=0.98,
            yanchor="top",
        ),
        xaxis=dict(
            gridcolor=COLOR_GRID,
            gridwidth=1,
            zerolinecolor=COLOR_LINE_STRONG,
            zerolinewidth=1,
            linecolor=COLOR_LINE_STRONG,
            linewidth=1,
            tickfont=dict(family=FONT_SANS, size=11, color=COLOR_MUTED),
            title=dict(font=dict(family=FONT_SANS, size=12, color=COLOR_MUTED)),
        ),
        yaxis=dict(
            gridcolor=COLOR_GRID,
            gridwidth=1,
            zerolinecolor=COLOR_LINE_STRONG,
            zerolinewidth=1,
            linecolor=COLOR_LINE_STRONG,
            linewidth=1,
            tickfont=dict(family=FONT_SANS, size=11, color=COLOR_MUTED),
            title=dict(font=dict(family=FONT_SANS, size=12, color=COLOR_MUTED)),
        ),
        legend=dict(
            orientation="h",
            x=0.0,
            y=1.12,
            xanchor="left",
            yanchor="bottom",
            font=dict(family=FONT_SANS, size=11, color=COLOR_TEXT),
            bgcolor="rgba(0,0,0,0)",
        ),
        hoverlabel=dict(
            bgcolor=COLOR_SURFACE_2,
            bordercolor=COLOR_LINE_STRONG,
            font=dict(family=FONT_SANS, size=12, color=COLOR_TEXT),
        ),
    )
)


def finalize_figure(
    fig: go.Figure,
    height: int = 360,
    legend_pos: str = "top",
    margin: Optional[dict] = None,
) -> go.Figure:
    """Apply the unified Mars-QCNN Mission Console template, dimensions, and margins to a Plotly figure."""
    if margin is None:
        margin = dict(l=54, r=24, t=64, b=48)
        
    fig.update_layout(
        template=MARSQCNN_TEMPLATE,
        height=height,
        margin=margin,
    )
    
    if legend_pos == "top":
        fig.update_layout(legend=dict(orientation="h", x=0.0, y=1.12, xanchor="left", yanchor="bottom"))
    elif legend_pos == "none":
        fig.update_layout(showlegend=False)
        
    return fig
