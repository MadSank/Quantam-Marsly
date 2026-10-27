"""Confusion matrix visualization components using Plotly (Mission Console styling)."""

from typing import List
import numpy as np
import plotly.graph_objects as go
from style.theme import MARSQCNN_TEMPLATE, COLOR_TEXT, COLOR_BG, FONT_SANS


def render_confusion_matrix_heatmap(
    cm: List[List[int]],
    class_names: List[str] = ["ejecta", "oldcrater", "none"],
    title: str = "Confusion Matrix",
    model_name: str = "Model",
) -> go.Figure:
    """Render a styled Plotly heatmap for a 3x3 confusion matrix with row-normalized annotations."""
    cm_arr = np.array(cm, dtype=int)
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_arr, row_sums, out=np.zeros_like(cm_arr, dtype=float), where=row_sums != 0)
    
    # Custom text annotations: count + percentage with contrast-aware font coloring
    text_annotations = []
    for r in range(len(class_names)):
        row_text = []
        for c in range(len(class_names)):
            count = cm_arr[r, c]
            pct = cm_norm[r, c] * 100.0
            row_text.append(f"<b>{count}</b><br>({pct:.1f}%)")
        text_annotations.append(row_text)
        
    # Mission Console Sequential Blue/Steel Palette
    colorscale = [
        [0.0, "#141B25"],
        [0.5, "#2C4A6B"],
        [1.0, "#7FB0DE"],
    ]
        
    fig = go.Figure(
        data=go.Heatmap(
            z=cm_norm,
            x=[f"Pred: {c}" for c in class_names],
            y=[f"True: {c}" for c in class_names],
            text=text_annotations,
            texttemplate="%{text}",
            colorscale=colorscale,
            zmin=0.0,
            zmax=1.0,
            colorbar=dict(title="Row Recall", thickness=12, len=0.85),
            hoverongaps=False,
        )
    )
    
    fig.update_layout(
        template=MARSQCNN_TEMPLATE,
        title=dict(text=title, font=dict(family=FONT_SANS, size=13, color=COLOR_TEXT)),
        height=320,
        margin=dict(l=50, r=30, t=50, b=40),
        xaxis=dict(tickangle=-15),
        yaxis=dict(autorange="reversed"),
    )
    
    return fig
