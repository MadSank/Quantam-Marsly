"""Image card and metadata rendering components for Mars and Moon explorers (Mission Console styling)."""

from typing import Dict, Any
import streamlit as st
import numpy as np
from PIL import Image

from style.theme import badge_html, COLOR_RUST, COLOR_BORDER


def render_mars_patch_card(
    img: np.ndarray,
    metadata: Dict[str, Any],
    selected: bool = False,
    card_key: str = "",
) -> bool:
    """Render a compact card for a Mars 64x64 terrain patch.
    
    Returns True if user clicked inspect/select.
    """
    tile_id = metadata.get("tile_id", "Unknown")
    is_safe = metadata.get("safe_label", 0) == 1
    slope = metadata.get("mean_slope_5m", 0.0)
    hazard = metadata.get("hazard_fraction", 0.0)
    
    border_style = f"border: 1px solid {COLOR_RUST};" if selected else f"border: 1px solid {COLOR_BORDER};"
    
    with st.container():
        st.markdown(
            f"""
            <div style="background: var(--mq-surface); border-radius: var(--mq-r); padding: 8px; {border_style} margin-bottom: 6px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <span style="font-size: 0.72rem; color: var(--mq-text-3); font-family: var(--mq-font-mono);">{tile_id[:16]}...</span>
                    {badge_html("SAFE" if is_safe else "UNSAFE", "safe" if is_safe else "unsafe")}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.image(img, width="stretch", clamp=True)
        st.caption(f"Slope: **{slope:.1f}°** | Haz: **{hazard:.3f}**")
        return st.button("Inspect Patch", key=f"btn_{card_key}_{tile_id}", width="stretch")


def render_moon_image_card(
    img: Image.Image,
    sample_id: str,
    class_label: str,
    split: str,
    selected: bool = False,
    card_key: str = "",
) -> bool:
    """Render a compact card for a Moon 227x227 crater image.
    
    Returns True if user clicked inspect/select.
    """
    filename = sample_id.split("/")[-1]
    border_style = f"border: 1px solid {COLOR_RUST};" if selected else f"border: 1px solid {COLOR_BORDER};"
    
    badge_type = "safe" if class_label == "ejecta" else ("disagree" if class_label == "oldcrater" else "neutral")
    
    with st.container():
        st.markdown(
            f"""
            <div style="background: var(--mq-surface); border-radius: var(--mq-r); padding: 8px; {border_style} margin-bottom: 6px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <span style="font-size: 0.72rem; color: var(--mq-text-3); font-family: var(--mq-font-mono);">{filename[:14]}...</span>
                    {badge_html(class_label.upper(), badge_type)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.image(img, width="stretch")
        st.caption(f"Split: **{split}** | Class: **{class_label}**")
        return st.button("Inspect Image", key=f"btn_{card_key}_{filename}", width="stretch")
