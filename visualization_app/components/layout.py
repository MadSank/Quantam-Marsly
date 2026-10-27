"""Layout and structural UI helpers for the Mars-QCNN Mission Console redesign."""

import html
import textwrap
from typing import Optional, List, Tuple
import streamlit as st


def step_progress(current_step: int) -> str:
    """Generate HTML for a 6-segment horizontal progress bar."""
    segments = []
    for step in range(1, 7):
        active_class = "active" if step <= current_step else ""
        segments.append(f'<span class="mq-step-segment {active_class}"></span>')
    segments_html = "".join(segments)
    return f'<div class="mq-step-progress" aria-label="Step {current_step} of 6">{segments_html}</div>'


def model_chip(model_name: str) -> str:
    """Generate HTML for an 8x8px square model color chip."""
    clean = model_name.lower()
    chip_class = "cnn"
    if "qcnn" in clean or "hybrid" in clean:
        chip_class = "qcnn"
    elif "mlp" in clean:
        chip_class = "mlp"
    elif "baseline" in clean:
        chip_class = "baseline"
    return f'<span class="mq-chip mq-chip-{chip_class}"></span>'


def page_header(
    step: Optional[int],
    title: str,
    subtitle: str,
    meta: Optional[str] = None,
    section_name: Optional[str] = None,
) -> None:
    """Render standardized Mission Console page header with mono eyebrow, native h1 title, and lede."""
    eyebrow_content = []
    eyebrow_content.append('<span class="mq-eyebrow-dash"></span>')
    
    if step is not None:
        eyebrow_content.append(f'<span class="mq-eyebrow-step">STEP {step} OF 6</span>')
        if section_name:
            eyebrow_content.append(f'<span style="color: var(--mq-text-3);">// {html.escape(section_name.upper())}</span>')
        eyebrow_content.append(step_progress(step))
    else:
        eyebrow_content.append('<span class="mq-eyebrow-step">MISSION CONSOLE</span>')
        if section_name:
            eyebrow_content.append(f'<span style="color: var(--mq-text-3);">// {html.escape(section_name.upper())}</span>')
            
    eyebrow_html = f'<div class="mq-eyebrow">{" ".join(eyebrow_content)}</div>'
    st.markdown(eyebrow_html, unsafe_allow_html=True)
    
    # Native h1 title ensures accessibility and AppTest visibility
    st.title(title)
    
    if subtitle:
        st.caption(subtitle)
        
    if meta:
        st.markdown(
            f'<div style="font-family: var(--mq-font-mono); font-size: 0.75rem; color: var(--mq-text-3); margin-top: 2px;">{html.escape(meta)}</div>',
            unsafe_allow_html=True,
        )
        
    st.markdown('<div class="mq-header-line"></div>', unsafe_allow_html=True)


def section_header(title: str, caption: Optional[str] = None) -> None:
    """Render standardized section subheader with consistent spacing and caption."""
    st.subheader(title)
    if caption:
        st.caption(caption)


def note_block(kind: str, label: str, body: str) -> None:
    """Render a structured scientific caveat, disclosure, or limitation note panel."""
    border_class = "mq-note-integrity" if kind in ("integrity", "correction") else ("mq-note-warning" if kind == "warning" else "mq-note")
    label_color = "var(--mq-rust-hi)" if kind in ("integrity", "correction") else ("var(--mq-warn)" if kind == "warning" else "var(--mq-text-2)")
    clean_body = " ".join(line.strip() for line in body.strip().splitlines())
    
    note_html = (
        f'<div class="mq-note {border_class}">'
        f'<div class="mq-note-label" style="color: {label_color};">{html.escape(label)}</div>'
        f'<div style="font-size: 0.88rem; line-height: 1.5; color: var(--mq-text);">{clean_body}</div>'
        f'</div>'
    )
    st.markdown(note_html, unsafe_allow_html=True)


def kv_table(rows: List[Tuple[str, str]]) -> str:
    """Generate HTML for a high-density, two-column key-value tabular display."""
    table_rows = []
    for label, val in rows:
        esc_label = html.escape(str(label))
        esc_val = html.escape(str(val))
        table_rows.append(
            f'<tr><td class="mq-kv-label">{esc_label}</td><td class="mq-kv-value">{esc_val}</td></tr>'
        )
    return f'<table class="mq-kv-table">{"".join(table_rows)}</table>'
