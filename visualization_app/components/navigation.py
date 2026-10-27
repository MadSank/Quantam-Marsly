"""Coursework Demonstration Walkthrough Navigation Components (Mission Console styling)."""

from typing import Optional
import streamlit as st
from components.layout import step_progress


def render_walkthrough_footer(
    next_page_num: Optional[int] = None,
    next_page_title: Optional[str] = None,
    next_page_filename: Optional[str] = None,
    next_page_icon: Optional[str] = "▶",
    highlight_text: Optional[str] = None,
    prev_page_num: Optional[int] = None,
    prev_page_title: Optional[str] = None,
    prev_page_filename: Optional[str] = None,
    prev_page_icon: Optional[str] = "◀",
    **kwargs,
) -> None:
    """Render a compact, robust walkthrough navigation footer at the bottom of pages."""
    # Resolve aliases if provided in kwargs
    curr_num = kwargs.get("current_page_num")
    if next_page_num is None:
        next_page_num = (curr_num + 1) if curr_num is not None else None
    if next_page_title is None:
        next_page_title = kwargs.get("next_page_name")
    if next_page_filename is None and kwargs.get("next_page_path"):
        next_page_filename = kwargs.get("next_page_path").replace("pages/", "")
    if highlight_text is None:
        highlight_text = kwargs.get("next_page_takeaway", "")

    if prev_page_title is None:
        prev_page_title = kwargs.get("prev_page_name")
    if prev_page_filename is None and kwargs.get("prev_page_path"):
        prev_page_filename = kwargs.get("prev_page_path").replace("pages/", "")
    if prev_page_num is None and prev_page_title:
        prev_page_num = (curr_num - 1) if curr_num is not None else max(1, (next_page_num or 2) - 2 if next_page_num else 1)

    st.markdown("---")
    
    # Walkthrough Footer Header with Progress Bar
    current_step_display = curr_num if curr_num is not None else (prev_page_num + 1 if prev_page_num else 1)
    progress_html = step_progress(current_step_display)
    
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 8px;">
            <div style="font-family: var(--mq-font-mono); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--mq-text-3);">
                <span class="mq-eyebrow-dash"></span> COURSEWORK DEMONSTRATION // STEP {current_step_display} OF 6
            </div>
            <div>
                {progress_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    col_prev, col_next = st.columns([1, 1])
    
    if prev_page_filename and prev_page_title:
        with col_prev:
            rendered_prev = False
            try:
                st.page_link(
                    f"pages/{prev_page_filename}",
                    label=f"Step {prev_page_num or ''}: {prev_page_title}",
                    icon=prev_page_icon or "◀",
                    width="stretch",
                )
                rendered_prev = True
            except Exception:
                try:
                    st.page_link(
                        prev_page_filename,
                        label=f"Step {prev_page_num or ''}: {prev_page_title}",
                        icon=prev_page_icon or "◀",
                        width="stretch",
                    )
                    rendered_prev = True
                except Exception:
                    rendered_prev = False
                    
            if not rendered_prev:
                st.markdown(
                    f"""
                    <div style="background: var(--mq-surface); border: 1px solid var(--mq-line); border-radius: var(--mq-r); padding: 10px 14px;">
                        <span style="font-size: 0.72rem; color: var(--mq-text-3); font-family: var(--mq-font-mono);">PREVIOUS STEP:</span><br/>
                        <strong>Step {prev_page_num or ''}: {prev_page_title}</strong>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                
    if next_page_filename and next_page_title:
        with col_next:
            rendered_next = False
            try:
                st.page_link(
                    f"pages/{next_page_filename}",
                    label=f"Step {next_page_num}: {next_page_title} ▶",
                    icon=next_page_icon or "▶",
                    width="stretch",
                )
                rendered_next = True
            except Exception:
                try:
                    st.page_link(
                        next_page_filename,
                        label=f"Step {next_page_num}: {next_page_title} ▶",
                        icon=next_page_icon or "▶",
                        width="stretch",
                    )
                    rendered_next = True
                except Exception:
                    rendered_next = False
                    
            if not rendered_next:
                st.markdown(
                    f"""
                    <div style="background: var(--mq-surface); border: 1px solid var(--mq-line-strong); border-radius: var(--mq-r); padding: 10px 14px;">
                        <span style="font-size: 0.72rem; color: var(--mq-rust-hi); font-family: var(--mq-font-mono);">NEXT DEMONSTRATION STEP:</span><br/>
                        <strong>Step {next_page_num}: {next_page_title}</strong>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    elif not next_page_filename:
        with col_next:
            st.markdown(
                """
                <div style="background: var(--mq-surface); border: 1px solid var(--mq-safe); border-radius: var(--mq-r); padding: 10px 14px;">
                    <span style="font-size: 0.72rem; color: var(--mq-safe); font-family: var(--mq-font-mono);">WALKTHROUGH CONCLUDED:</span><br/>
                    <strong>All 6 Demonstration Steps Complete</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
    if highlight_text:
        st.caption(f"💡 **Presenter Highlight:** {highlight_text}")
