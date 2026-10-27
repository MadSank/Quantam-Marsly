"""Plotly chart builders for Mars and Moon comparative analysis (Mission Console styling)."""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Optional

from style.theme import (
    COLOR_CNN,
    COLOR_QCNN,
    COLOR_HYBRID,
    COLOR_MLP,
    COLOR_BASELINE,
    COLOR_SAFE,
    COLOR_UNSAFE,
    COLOR_DISAGREE,
    COLOR_BORDER,
    COLOR_LINE_STRONG,
    COLOR_MUTED,
    COLOR_TEXT,
    FONT_SANS,
    FONT_MONO,
    MARSQCNN_TEMPLATE,
)


def build_mars_regression_chart(aggregate_df: pd.DataFrame, metric_type: str = "RMSE") -> go.Figure:
    """Build grouped bar chart comparing CNN vs QCNN on continuous terrain regression targets."""
    targets = [
        ("mean_slope_5m", "Mean Slope [°]"),
        ("roughness_m", "Roughness [m]"),
        ("hazard_fraction", "Hazard Fraction [fraction]"),
    ]
    x_labels = [label for _, label in targets]
    
    cnn_vals, cnn_sds = [], []
    qcnn_vals, qcnn_sds = [], []
    
    for key, _ in targets:
        metric_name = f"{metric_type}_{key}"
        row_cnn = aggregate_df[(aggregate_df["model"] == "CNN") & (aggregate_df["metric"] == metric_name)]
        row_qcnn = aggregate_df[(aggregate_df["model"] == "CNN_QCNN") & (aggregate_df["metric"] == metric_name)]
        
        cnn_vals.append(row_cnn["mean_site_metric"].values[0] if len(row_cnn) else 0.0)
        cnn_sds.append(row_cnn["sd_across_sites"].values[0] if len(row_cnn) else 0.0)
        
        qcnn_vals.append(row_qcnn["mean_site_metric"].values[0] if len(row_qcnn) else 0.0)
        qcnn_sds.append(row_qcnn["sd_across_sites"].values[0] if len(row_qcnn) else 0.0)
        
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            name="Classical ResNet-18",
            x=x_labels,
            y=cnn_vals,
            error_y=dict(type="data", array=cnn_sds, visible=True, thickness=1.5, color=COLOR_CNN),
            marker_color=COLOR_CNN,
            hovertemplate="<b>Classical ResNet-18</b><br>%{x}: %{y:.3f} (±%{error_y.array:.3f})<extra></extra>",
        )
    )
    
    fig.add_trace(
        go.Bar(
            name="Hybrid QCNN Head",
            x=x_labels,
            y=qcnn_vals,
            error_y=dict(type="data", array=qcnn_sds, visible=True, thickness=1.5, color=COLOR_QCNN),
            marker_color=COLOR_QCNN,
            hovertemplate="<b>Hybrid QCNN Head</b><br>%{x}: %{y:.3f} (±%{error_y.array:.3f})<extra></extra>",
        )
    )
    
    fig.update_layout(
        template=MARSQCNN_TEMPLATE,
        title=dict(
            text=f"<b>Mars Continuous Terrain Regression ({metric_type})</b><br><span style='font-size:0.82rem;color:{COLOR_MUTED};'>Error bars: ± 1 Sample SD across 5 held-out geographic sites. Physical units labeled per target.</span>",
            x=0.01,
            xanchor="left",
        ),
        barmode="group",
        height=380,
        margin=dict(l=60, r=30, t=75, b=60),
        yaxis=dict(title=f"{metric_type} Error (Physical Units: [°], [m], [fraction])"),
        xaxis=dict(tickfont=dict(size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    return fig


def build_mars_safety_chart(aggregate_df: pd.DataFrame) -> go.Figure:
    """Build grouped bar chart comparing CNN vs QCNN on safe-proxy classification metrics."""
    metrics_spec = [
        ("safe_recall", "Safe Recall [%]"),
        ("false_positive_rate", "Hazard FPR [%]"),
        ("predicted_safe_fraction", "Safe Coverage [%]"),
    ]
    
    x_labels = [label for _, label in metrics_spec]
    cnn_vals, cnn_sds = [], []
    qcnn_vals, qcnn_sds = [], []
    
    for key, _ in metrics_spec:
        row_cnn = aggregate_df[(aggregate_df["model"] == "CNN") & (aggregate_df["metric"] == key)]
        row_qcnn = aggregate_df[(aggregate_df["model"] == "CNN_QCNN") & (aggregate_df["metric"] == key)]
        
        cnn_vals.append(row_cnn["mean_site_metric"].values[0] * 100.0 if len(row_cnn) else 0.0)
        cnn_sds.append(row_cnn["sd_across_sites"].values[0] * 100.0 if len(row_cnn) else 0.0)
        
        qcnn_vals.append(row_qcnn["mean_site_metric"].values[0] * 100.0 if len(row_qcnn) else 0.0)
        qcnn_sds.append(row_qcnn["sd_across_sites"].values[0] * 100.0 if len(row_qcnn) else 0.0)
        
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            name="Classical ResNet-18",
            x=x_labels,
            y=cnn_vals,
            error_y=dict(type="data", array=cnn_sds, visible=True, thickness=1.5, color=COLOR_CNN),
            marker_color=COLOR_CNN,
            hovertemplate="<b>Classical ResNet-18</b><br>%{x}: %{y:.2f}% (±%{error_y.array:.2f}%)<extra></extra>",
        )
    )
    
    fig.add_trace(
        go.Bar(
            name="Hybrid QCNN Head",
            x=x_labels,
            y=qcnn_vals,
            error_y=dict(type="data", array=qcnn_sds, visible=True, thickness=1.5, color=COLOR_QCNN),
            marker_color=COLOR_QCNN,
            hovertemplate="<b>Hybrid QCNN Head</b><br>%{x}: %{y:.2f}% (±%{error_y.array:.2f}%)<extra></extra>",
        )
    )
    
    max_val = max(cnn_vals + qcnn_vals) if (cnn_vals and qcnn_vals) else 10.0
    max_sd = max(cnn_sds + qcnn_sds) if (cnn_sds and qcnn_sds) else 5.0
    upper_bound = max(20.0, float(max_val + max_sd + 3.0))
    
    fig.update_layout(
        template=MARSQCNN_TEMPLATE,
        title=dict(
            text=f"<b>Mars Landing-Safety Proxy Rates (%) (Fixed Threshold)</b><br><span style='font-size:0.82rem;color:{COLOR_MUTED};'>Rule: Mean Slope < 8.0° & Hazard Fraction < 0.05. Error bars: ± 1 Sample SD across 5 sites.</span>",
            x=0.01,
            xanchor="left",
        ),
        barmode="group",
        height=380,
        margin=dict(l=60, r=30, t=75, b=60),
        yaxis=dict(title="Rate (%)", range=[0.0, upper_bound]),
        xaxis=dict(tickfont=dict(size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    return fig


def build_moon_comparison_chart(recomputed_df: pd.DataFrame) -> go.Figure:
    """Build grouped bar chart comparing Moon classification metrics across models."""
    models_spec = [
        ("cnn", "Classical CNN", COLOR_CNN, None),
        ("hybrid_qcnn", "Hybrid QCNN", COLOR_HYBRID, None),
        ("frozen_embedding_mlp", "Frozen MLP Control", COLOR_MLP, None),
        ("majority_train_baseline", "Majority Baseline", COLOR_BASELINE, "/"),
    ]
    
    metrics = ["accuracy", "balanced_accuracy", "macro_f1"]
    metric_labels = ["Accuracy", "Balanced Accuracy", "Macro-F1"]
    
    fig = go.Figure()
    
    for mod_key, mod_name, color, pattern in models_spec:
        vals, sds = [], []
        for m in metrics:
            row = recomputed_df[(recomputed_df["model"] == mod_key) & (recomputed_df["metric"] == m)]
            if len(row):
                vals.append(row["mean"].values[0])
                sds.append(row["sample_sd"].values[0])
            else:
                vals.append(0.0)
                sds.append(0.0)
                
        trace_kwargs = dict(
            name=mod_name,
            x=metric_labels,
            y=vals,
            error_y=dict(type="data", array=sds, visible=True, thickness=1.5, color=color),
            marker_color=color,
            hovertemplate=f"<b>{mod_name}</b><br>%{{x}}: %{{y:.3f}} (±%{{error_y.array:.3f}})<extra></extra>",
        )
        if pattern:
            trace_kwargs["marker_pattern_shape"] = pattern
            
        fig.add_trace(go.Bar(**trace_kwargs))
        
    fig.update_layout(
        template=MARSQCNN_TEMPLATE,
        title=dict(
            text=f"<b>Moon Crater Classification Performance (LROCNet)</b><br><span style='font-size:0.82rem;color:{COLOR_MUTED};'>Scores range from 0.0 to 1.0. Error bars represent ± 1 Sample SD across Seeds 42, 43, 44.</span>",
            x=0.01,
            xanchor="left",
        ),
        barmode="group",
        height=380,
        margin=dict(l=60, r=30, t=75, b=60),
        yaxis=dict(title="Score (Range [0.0, 1.0])", range=[0.0, 0.85]),
        xaxis=dict(tickfont=dict(size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    return fig


def build_loss_curve_chart(
    epochs: np.ndarray,
    train_loss: np.ndarray,
    val_loss: Optional[np.ndarray] = None,
    title: str = "Training Loss Progression",
) -> go.Figure:
    """Build line chart for epoch loss progression."""
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=train_loss,
            mode="lines" if len(epochs) > 20 else "lines+markers",
            name="Train Loss",
            line=dict(color=COLOR_CNN, width=2),
        )
    )
    
    if val_loss is not None and len(val_loss) > 0:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=val_loss,
                mode="lines" if len(epochs) > 20 else "lines+markers",
                name="Validation Loss",
                line=dict(color=COLOR_QCNN, width=2, dash="dash"),
            )
        )
        
    fig.update_layout(
        template=MARSQCNN_TEMPLATE,
        title=dict(text=title),
        height=300,
        margin=dict(l=48, r=24, t=48, b=40),
        xaxis=dict(title="Epoch"),
        yaxis=dict(title="Loss"),
    )
    return fig


def build_mars_grid_scatter(
    df_site: pd.DataFrame,
    color_by: str = "Ground Truth Safe Proxy",
    selected_tile_id: Optional[str] = None,
) -> go.Figure:
    """Build 2D scatter plot of intra-site HiRISE local mosaic coordinates (col vs row).
    
    Axes are titled 'Intra-Site Grid Col (px/tile offset)' and 'Intra-Site Grid Row (px/tile offset)',
    with the y-axis inverted so row 0 is at the top.
    """
    fig = go.Figure()
    
    if "row" not in df_site.columns or "col" not in df_site.columns or df_site.empty:
        fig.update_layout(title="No grid coordinates available for this site")
        return fig
        
    if color_by == "Ground Truth Safe Proxy":
        for val, name, color in [(1, "Ground Truth Safe", COLOR_SAFE), (0, "Ground Truth Unsafe", COLOR_UNSAFE)]:
            sub = df_site[df_site["true_safe"] == val]
            if not sub.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sub["col"],
                        y=sub["row"],
                        mode="markers",
                        marker=dict(size=7, color=color, opacity=0.85),
                        name=name,
                        customdata=sub[["tile_id", "true_mean_slope_5m", "true_hazard_fraction"]].values,
                        hovertemplate=(
                            "<b>Tile: %{customdata[0]}</b><br>"
                            "Col: %{x}, Row: %{y}<br>"
                            "Slope: %{customdata[1]:.2f}°<br>"
                            "Hazard: %{customdata[2]:.4f}<extra></extra>"
                        ),
                    )
                )
    elif color_by == "Model Agreement":
        for val, name, color in [(True, "Models Agree", COLOR_CNN), (False, "Models Disagree (CNN != QCNN)", COLOR_DISAGREE)]:
            sub = df_site[df_site["agree_safe"] == val]
            if not sub.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sub["col"],
                        y=sub["row"],
                        mode="markers",
                        marker=dict(size=7, color=color, opacity=0.85),
                        name=name,
                        customdata=sub[["tile_id", "cnn_pred_safe", "qcnn_pred_safe", "true_safe"]].values,
                        hovertemplate=(
                            "<b>Tile: %{customdata[0]}</b><br>"
                            "Col: %{x}, Row: %{y}<br>"
                            "CNN Safe: %{customdata[1]}, QCNN Safe: %{customdata[2]}<br>"
                            "True Safe: %{customdata[3]}<extra></extra>"
                        ),
                    )
                )
    elif color_by == "CNN vs QCNN Absolute Slope Error":
        err_col = df_site.get("cnn_abs_err_slope", (df_site["cnn_pred_mean_slope_5m"] - df_site["true_mean_slope_5m"]).abs())
        fig.add_trace(
            go.Scatter(
                x=df_site["col"],
                y=df_site["row"],
                mode="markers",
                marker=dict(
                    size=7,
                    color=err_col,
                    colorscale="Cividis",
                    showscale=True,
                    colorbar=dict(title="Slope Err (°)", thickness=14, len=0.75),
                    opacity=0.9,
                ),
                name="CNN Slope Error",
                customdata=df_site[["tile_id", "cnn_abs_err_slope", "qcnn_abs_err_slope"]].values if "qcnn_abs_err_slope" in df_site else df_site[["tile_id"]].values,
                hovertemplate=(
                    "<b>Tile: %{customdata[0]}</b><br>"
                    "Col: %{x}, Row: %{y}<br>"
                    "CNN Abs Error: %{customdata[1]:.2f}°<br>"
                    "QCNN Abs Error: %{customdata[2]:.2f}°<extra></extra>"
                ),
            )
        )
        
    # Highlight selected tile if provided
    if selected_tile_id and (df_site["tile_id"] == selected_tile_id).any():
        sel_row = df_site[df_site["tile_id"] == selected_tile_id].iloc[0]
        fig.add_trace(
            go.Scatter(
                x=[sel_row["col"]],
                y=[sel_row["row"]],
                mode="markers",
                marker=dict(
                    symbol="star",
                    size=16,
                    color="#E08A63",
                    line=dict(color="#FFFFFF", width=2),
                ),
                name=f"Selected ({selected_tile_id})",
                hovertemplate=f"<b>⭐ Selected Tile: {selected_tile_id}</b><br>Col: {sel_row['col']}, Row: {sel_row['row']}<extra></extra>",
            )
        )

    fig.update_layout(
        template=MARSQCNN_TEMPLATE,
        height=440,
        margin=dict(l=60, r=40, t=50, b=50),
        xaxis=dict(
            title="Intra-Site Grid Col (px/tile offset)",
        ),
        yaxis=dict(
            title="Intra-Site Grid Row (px/tile offset)",
            autorange="reversed",  # Row 0 at top
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
    )
    return fig
