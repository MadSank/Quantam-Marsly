"""Filtering and pagination UI components (Mission Console styling)."""

import streamlit as st
import math


def render_pagination_controls(
    total_items: int,
    items_per_page: int = 12,
    current_page: int = 0,
    key_prefix: str = "page",
) -> int:
    """Render compact pagination controls and return the active page index."""
    total_pages = max(1, math.ceil(total_items / max(1, items_per_page)))
    safe_current_page = max(0, min(current_page, total_pages - 1))
    
    col_info, col_prev, col_num, col_next = st.columns([3, 1, 2, 1])
    
    with col_info:
        start_idx = safe_current_page * items_per_page
        end_idx = min(start_idx + items_per_page, total_items)
        st.caption(f"Showing **{start_idx + 1}–{end_idx}** of **{total_items:,}** items")
        
    with col_prev:
        if st.button("◀ Prev", key=f"{key_prefix}_prev", disabled=(safe_current_page <= 0), width="stretch"):
            safe_current_page = max(0, safe_current_page - 1)
            st.rerun()
            
    with col_num:
        st.markdown(
            f"<div style='text-align: center; padding-top: 6px; font-family: var(--mq-font-mono); font-size: 0.85rem; color: var(--mq-text);'>"
            f"Page <strong>{safe_current_page + 1}</strong> of <strong>{total_pages}</strong></div>",
            unsafe_allow_html=True,
        )
        
    with col_next:
        if st.button("Next ▶", key=f"{key_prefix}_next", disabled=(safe_current_page >= total_pages - 1), width="stretch"):
            safe_current_page = min(total_pages - 1, safe_current_page + 1)
            st.rerun()
            
    return safe_current_page


def render_prev_next_controls(
    current_idx: int,
    total_items: int,
    item_label: str = "Item",
    key_prefix: str = "step",
) -> int:
    """Render compact Prev/Next stepping buttons for direct item-by-item inspection.
    
    Returns the updated item index.
    """
    safe_idx = max(0, min(current_idx, max(0, total_items - 1)))
    col_prev, col_info, col_next = st.columns([1, 2, 1])
    
    with col_prev:
        if st.button(f"◀ Prev {item_label}", key=f"{key_prefix}_prev", disabled=(safe_idx <= 0), width="stretch"):
            safe_idx = max(0, safe_idx - 1)
            st.rerun()
            
    with col_info:
        if total_items > 0:
            st.markdown(
                f"<div style='text-align: center; padding-top: 6px; font-family: var(--mq-font-mono); font-size: 0.85rem; color: var(--mq-text-2);'>"
                f"{item_label} <strong>{safe_idx + 1}</strong> of <strong>{total_items:,}</strong></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='text-align: center; padding-top: 6px; font-family: var(--mq-font-mono); font-size: 0.85rem; color: var(--mq-text-3);'>"
                f"0 {item_label}s</div>",
                unsafe_allow_html=True,
            )
            
    with col_next:
        if st.button(f"Next {item_label} ▶", key=f"{key_prefix}_next", disabled=(safe_idx >= total_items - 1), width="stretch"):
            safe_idx = min(total_items - 1, safe_idx + 1)
            st.rerun()
            
    return safe_idx
