# UI_REDESIGN_PLAN.md

## Mars-QCNN Research Explorer — Visual Redesign Implementation Plan

| | |
| :--- | :--- |
| **Executor** | Gemini Flash, running locally in Antigravity |
| **Application** | `D:\Mars-QCNN\visualization_app` (Streamlit 1.66.0, Plotly 7.1.0, Python 3.11.15 — per audit §1.2) |
| **Primary source of truth** | `UI_AUDIT_HANDOFF.md` (audit dated October 20, 2026) |
| **Author's access** | The planner had **no access** to the repository or the screenshots. Everything below is derived from the audit text. Anything the audit does not establish is labelled. |
| **Scope** | Visual design pass only. No new features, no model or data work, no architecture rewrite. |

---

## 0. How to Read and Execute This Document

### 0.1 Evidence labels

| Label | Meaning |
| :--- | :--- |
| **[AUDIT]** | Stated directly in `UI_AUDIT_HANDOFF.md`. Section numbers refer to the audit. |
| **[VERIFY]** | Not established by the audit. Flash must confirm it in the repository (or the running app) **before** acting on it. If the check fails, follow the stated fallback. |
| **[ASSUMPTION]** | The planner's inference, with a safe default. Flash should confirm, and report if wrong. |
| **[NEW]** | A file or function this plan asks Flash to create. It does not exist today. |

Names that the audit shows truncated (for example `render_parameter_count_...`) must be read in full from the source. Never guess the complete name.

### 0.2 Global execution rules (apply to every phase)

| # | Rule |
| :--- | :--- |
| **R1** | **UI only.** Do not change datasets, data loading, model outputs, metric calculations, timing values, predictions, filters' logic, exports, or any scientific wording. |
| **R2** | **Incremental.** Complete one numbered step, run the tests, look at the result in a browser, checkpoint, then continue. No multi-file rewrites without a passing checkpoint between them. |
| **R3** | **Read before editing.** Open every file fully before changing it. The audit's line ranges (for example `app.py` 50–93) are approximate and will drift. |
| **R4** | **Text is frozen.** Scientific text, disclosures, caveats, captions, units, table headers and chart titles keep their exact wording. Only dead decoration (emoji glyphs) may be removed from titles, and only after rule R5 is satisfied. Markdown bodies must remain byte-identical. |
| **R5** | **Test-coupling rule.** Before changing any title, label, expander, tab, alert type, color, or element order, grep `tests/` for it. Tests fall into two classes. **Class S** (scientific values, disclosures, data shapes, the removed-statistical-section guard) — never edit. **Class C** (cosmetic literals: hex colors, emoji, page-title strings, tab/expander labels, element-type counts) — first change the *implementation* so the assertion still holds; if that is impossible, make the smallest strictly-equivalent edit (for example import the theme constant instead of a hard-coded hex), and list it under "Needs user review" in the final report. Never loosen, delete or skip a test. |
| **R6** | **Selector policy.** CSS may target only: `[data-testid="…"]` hooks, ARIA roles/attributes (`button[role="tab"]`), and Streamlit's documented `st-key-<key>` container classes. Never target `st-emotion-cache-*`/`css-*` classes, positional chains (`div > div > div`, `:nth-child`), or inline-style attributes. Every selector must be confirmed present in the live DOM in Phase 0.7; a selector that does not exist is dropped, not guessed. |
| **R7** | **No new Python dependencies.** The only permitted additions are (a) a dev-only screenshot tool, if one is not already installed, kept out of requirements; and (b) optional self-hosted font files (Phase 4.6). |
| **R8** | **Do not rename or move** files in `pages/`, nor modules imported by tests. Streamlit derives sidebar labels and URLs from page filenames; tests load pages by path. |
| **R9** | **Preserve every widget key and `st.session_state` key** (for example `inspected_mars_row`). New keys may be added; existing ones are never renamed or removed. |
| **R10** | **Report, don't silently resolve.** If the repository contradicts the audit or this plan, stop that step, write the discrepancy into the phase report, choose the more conservative option, and continue only where the contradiction does not apply. |

### 0.3 Constraint on Streamlit HTML

Existing code builds cards with `st.markdown(..., unsafe_allow_html=True)` **[AUDIT §4.1, §4.4]**. Keep that mechanism. Two hazards apply to any new HTML:

- Markdown treats lines indented by four or more spaces as code blocks. Build HTML strings without leading indentation and without blank lines inside the markup.
- Markdown inside an HTML block is not reliably parsed. Therefore, **containers that hold markdown bodies must be native** (`st.container(border=True, key=…)`), not an HTML `<div>` wrapper. HTML strings are reserved for small, text-only pieces (badges, key–value tables, step pills).

---

## 1. Executive Summary and Selected Visual Direction

### 1.1 What the redesign does

The application's *content* is already strong: it has rigorous disclosures, deterministic attribution, and clear Mars/Moon separation **[AUDIT §4]**. Its *presentation* reads as a default dark Streamlit app dressed with Material-500 colors, emoji titles, five separate boxed metrics per page, and a 2,500 px-tall explorer page. The redesign changes presentation only, in this order of leverage:

1. **One token set** (colors, type, spacing, radius) defined once in `style/theme.py` and emitted as CSS variables and Python constants. Most of the visual change comes from here.
2. **One chart template** applied through a single helper, replacing five hand-tuned layouts **[AUDIT §3.3]**.
3. **Six small layout helpers** (page header, note block, section header, key–value table, step progress, model ledger) in a **[NEW]** `components/layout.py`, so every page opens the same way and carries the same caveat styling.
4. **Page-level restructuring only where the audit shows a usability defect**: Mars Test Explorer height, Moon cards at 1366 px, confusion-matrix squeeze, nested tabs, the Methodology wall of expanders, the landing dead end, and the duplicated walkthrough.

### 1.2 Selected visual direction: "Mission Console"

A flat, ruled, instrument-panel look in the spirit of mission-analysis and technical-report software rather than a consumer dashboard.

- **Surfaces:** cool near-black canvas, slightly lighter flat panels, 1 px hairlines. No shadows, gradients, blur, glow, or glass.
- **Accent:** a single Mars-rust accent used only for interface chrome (active navigation, primary action, header tick, focus ring, step progress). **Rust never encodes data.**
- **Data colors:** muted steel-blue (classical CNN), muted violet (hybrid QCNN), cool grey (matched MLP), sand (baseline). Status colors (safe / unsafe / disagree) are muted and always accompanied by a text label.
- **Typography:** a clean technical sans for text, a monospace for numbers, eyebrows and identifiers. Compact scale: page titles 28 px, not Streamlit's 44 px.
- **Signature details (deliberately few):** a small rust dash before the mono "eyebrow" line of each page header; a hairline closing the header; a six-segment step progress indicator reused in the header and the footer; a small series-color chip that precedes each model's name wherever it appears.
- **Radius:** 4 px everywhere, 2 px for badges.

### 1.3 Theme decision: dark-first, single theme

**Decision: ship one dark theme. Do not add light-theme support.**

Reasons: (1) The application is already dark-only and every Plotly figure hard-codes dark backgrounds **[AUDIT §3.1, §3.3]**; (2) the planetary-science, mission-analysis context suits a dark canvas; (3) a second theme would require every chart, HTML fragment and image card to be fully token-driven and would double visual QA across the screenshot matrix, with no benefit for a desktop coursework presentation tool **[AUDIT §10.3]**; (4) a single theme is what makes the "restrained identity" achievable.

One required safeguard: force the dark base in Streamlit config (`base = "dark"`, step 1.1) so that a machine set to light mode cannot produce a mismatched light background behind the dark custom CSS. **[VERIFY]** whether this mismatch exists today (the audit lists `.reportview-container`, an obsolete hook, as the page-background rule — see G2).

### 1.4 What this plan explicitly does not do

- No gradients, glassmorphism, glow, decorative planetary imagery, hero banners, radial gauges, or colored "winner" deltas.
- No sidebar icons, no emoji, no iconography on Methodology.
- No change to the six-page structure, file names, URLs, or the native Streamlit multipage mechanism (no `st.navigation` migration).
- No restoration of the removed paired-site statistical assessment (see §7, check P1).

---

## 2. Audit Findings That Motivate the Redesign

### 2.1 Global design-system issues (fixed once, centrally)

| ID | Finding | Evidence | Where it is fixed |
| :--- | :--- | :--- | :--- |
| **G1** | Styling is scattered: hard-coded inline styles in cards and raw HTML tables; the guide callout is hand-written HTML with an `<ol>`. | §4.1, §4.4, §7 | `style/theme.py` classes; `components/layout.py`; `components/image_cards.py` |
| **G2** | Obsolete or doubtful CSS hooks: `.reportview-container` (a legacy class name) is the page-background rule; `.stMetric` is a class rather than a `data-testid`. They may be doing nothing. | §3.2 items 1–2 | **[VERIFY]** in Phase 0.7; replace with `config.toml` + `data-testid` hooks |
| **G3** | An emoji prefixes every page title (`🪐 📊 🖼️ 🔴 🌕 📈 🔬`). | §2 | `components/layout.py` `page_header`, each page |
| **G4** | No type hierarchy: native title and metric sizes are oversized for a dense scientific tool (the audit describes "tighter padding" and tall pages). | §4, screenshots §9 | Typography tokens, `config.toml` heading sizes |
| **G5** | Generic palette: `#4FC3F7`, `#AB47BC`, `#66BB6A`, `#EF5350`, `#FFA726`, `#FFB74D` are saturated Material-500 values — the default "AI dashboard" signature. Also a **semantic collision**: `COLOR_BASELINE #FFB74D` and `COLOR_DISAGREE #FFA726` are nearly the same amber. And `COLOR_AGREE` is green, implying that agreement is "good" even when both models are wrong. | §3.1 | Token table §3.2 |
| **G6** | Chart theming is copy-pasted per builder (`paper_bgcolor`, `plot_bgcolor`, fonts, grids, margins). | §3.3 | One Plotly template + one `finalize_figure` helper in `style/theme.py` |
| **G7** | Metrics are rendered as separate boxed cards (5 per explorer page) and notes use a 4 px left-border bar — both strong "template" signals; vertical rhythm is loose. | §3.2 items 2, 5, 6; §4.4 | Compact metric strip; note-block spec |
| **G8** | Navigation lacks step context: pages never say "step N of 6", and the walkthrough is shown twice (landing and Overview). | §4.1, §4.2, §7 | `page_header` eyebrow; `render_walkthrough_footer` |
| **G9** | Streamlit deprecation warnings (`use_container_width`). | §7 | Phase 1.5 (isolated, mechanical) |
| **G10** | Tables: raw `st.dataframe` with unstyled columns (7-column logits table) and raw HTML `<table>` in cards. | §4.2, §4.4, §4.5 | `column_config` presentation; `.mq-kv` class |
| **G11** | Nested tabs (Model Comparison Tab 3) and an "expander wall" (Methodology). | §4.6, §4.7 | Page specs 4.5, 4.7 |
| **G12** | Mermaid diagrams can render with default light boxes against the dark surface. | §4.2 | **[VERIFY]** the rendering mechanism, then theme variables (page spec 4.2) |

### 2.2 Page-specific issues (fixed per page)

| Page | Source file | Page-specific defects (audit) |
| :--- | :--- | :--- |
| Landing | `app.py` | Dead end (no call to action); crowded HTML `<ol>`; empty lower half at 1920×1080 (§4.1) |
| 1 Overview | `pages/1_Overview.py` | Walkthrough expanded by default and duplicated; plain headline tables; Mermaid dissonance (§4.2) |
| 2 Training Data | `pages/2_Training_Data_Explorer.py` | 12 stacked buttons add noise at 1366; inspection state vs pagination mismatch; dual (sidebar + split) filters confusing (§4.3) |
| 3 Mars Test | `pages/3_Mars_Test_Explorer.py` | Page > 2,500 px tall; raw HTML tables; the grid-position scatter hidden in a closed expander (§4.4) |
| 4 Moon Test | `pages/4_Moon_Test_Explorer.py` | 3-column cards cramped at 1366; 7-column raw logits table; `st.success` banner pushes controls down (§4.5) |
| 5 Model Comparison | `pages/5_Model_Comparison.py` | Three squeezed confusion matrices; Mars tab has no summary in the main view; nested tabs (§4.6) |
| 6 Methodology | `pages/6_Methodology.py` | Five fully expanded expanders; no structure for scanning (§4.7) |

> **Behavioral defects are reported, not fixed.** The page-2 "inspection state vs pagination" mismatch (§4.3) is a logic issue. Flash must **not** change session-state behavior. Make the state *visible* (the detail pane header names the inspected patch), record the observation in the report, and leave the logic alone.

### 2.3 Disposition of the audit's own recommendations (§7)

The audit's §7 suggestions partly conflict with the requested direction. Dispositions:

| Audit suggestion | Disposition | Reason |
| :--- | :--- | :--- |
| Replace `use_container_width=True` with `width="stretch"` | **Adopt** (isolated step 1.5) | Removes deprecation warnings; mechanical. Verify each call's signature (they differ per element). |
| CSS utility classes and CSS variables | **Adopt**, namespaced `mq-` | Core of the token system. |
| Landing "visually stunning hero" with action cards | **Adapt** → a 6-row route list with native page links and one primary button | Hero banners and cards are exactly the template look to avoid. |
| Overview "subtle glassmorphic styling" | **Reject** | Explicitly excluded. |
| Overview "colored difference pills" | **Reject** | Green/red deltas would imply a CNN-vs-QCNN win/loss. The Mars paired differences cross zero (§6), so the UI must not visually overstate any difference. Numeric columns stay neutral. |
| Training Explorer clickable cards | **Adapt** → keep the native button, switch to the compact `tertiary` type; selected tile gets a rust outline | Native Streamlit cannot make images clickable; avoids fragile JS. |
| Mars Explorer 2-row restructure; promote scatter out of expander | **Adopt** as three tabs under the inspection panel | Cuts height; scatter becomes discoverable. |
| Moon "radial meters" | **Adapt** → one "model ledger" with linear confidence bars | Radial meters are decorative; linear bars align and fit at 1366. |
| Confusion matrix model selector | **Adapt** → segmented control with default **All three** plus single-model views | Side-by-side comparison is the point; the selector adds a large single view. |
| Methodology → tabs with iconography and spec tables | **Adapt** → tabs, no icons, **no restructuring of text** | Test `test_methodology_file_content` and rule R4 require literal text preservation. |
| Footer "gradient buttons" | **Reject gradients**; **adopt** step progress | Gradients excluded. |

### 2.4 Audit discrepancies and unknowns Flash must resolve in Phase 0

| # | Item | Why it matters |
| :--- | :--- | :--- |
| D1 | The audit's test run shows `rootdir: D:\Mars-QCNN` and paths `tests\test_m1…`, but the §1.3 tree places `tests/` under `visualization_app/`. | The test command is `.\.venv\Scripts\python.exe -m pytest tests/ -v` **[AUDIT §8]**. Determine the correct working directory, run it once, and record that exact invocation. Do not invent a different command. |
| D2 | Is the repository under git? The audit never mentions version control. | Determines the checkpoint/rollback mechanism (§6, Phase 0.2). |
| D3 | Does a `.streamlit/config.toml` already exist, and from which directory is `streamlit run` launched (audit screenshots used `localhost:8503`)? | The config file must sit where Streamlit actually reads it. |
| D4 | Which code actually renders the Training Explorer tiles: inline code in page 2 (lines 103–134, 220–248) or `render_mars_patch_card` / `render_moon_image_card`? The audit's component map lists those card builders as "reusable" without naming call sites. | Decides whether to restyle in place or in the component. Do not refactor to force reuse. |
| D5 | Is the mapping "page N = walkthrough step N" correct, and where do the six step titles live (`app.py` lines 50–93, `components/navigation.py`)? | Drives the `STEP N OF 6` eyebrow and progress indicator. |
| D6 | How does `1_Overview.py` render Mermaid (component with a script? plain code block?), and does it use a CDN? | Decides whether theming is possible without adding a network dependency. |
| D7 | What do the five metric labels on pages 3 and 4 say, and do they truncate at 1366? | Informs metric-strip CSS. |
| D8 | Do `st.set_page_config` calls exist per page, with which `layout`, `page_title`, `page_icon`, `initial_sidebar_state`? | Layout width, favicon, sidebar state. |
| D9 | The exact full names of the truncated functions in `components/metric_tables.py` and `components/confusion_matrix.py`. | Avoid invented names. |

---

## 3. Design-System Specification

### 3.1 Principles

1. **Hierarchy before decoration.** Every page: eyebrow → title → one-line purpose → primary content. Secondary material is quieter, never louder.
2. **One of everything.** One note style, one badge style, one panel style, one chart template, one table style.
3. **Flat and ruled.** Separation comes from spacing and 1 px lines, not shadows or nested boxes. Maximum one level of bordered container; never a bordered container inside a bordered container.
4. **Color has jobs.** Rust = interface chrome; blue/violet/grey/sand = model identity; green/gold/red = status; neutrals = everything else. A color never serves two jobs.
5. **Caveats stay adjacent to the claim.** A limitation is placed directly above or beside the result it qualifies, in the same visual weight as the result, never visually demoted.
6. **Native first, CSS second.** Use Streamlit's own theming and parameters wherever they exist; use CSS only for what they cannot express, through the stable hooks allowed by rule R6.

### 3.2 Design tokens

Single source of truth: Python constants in `style/theme.py`. The `:root` CSS block is **generated from those constants** (a small helper that returns the variable block, concatenated with the static CSS, avoiding f-string brace escaping), so each hex value is typed once. **Keep all existing constant names** (`COLOR_CNN`, `COLOR_QCNN`, `COLOR_HYBRID`, `COLOR_MLP`, `COLOR_BASELINE`, `COLOR_SAFE`, `COLOR_UNSAFE`, `COLOR_AGREE`, `COLOR_DISAGREE`, `COLOR_CORRECT`, `COLOR_INCORRECT`, `COLOR_BG_CARD`, `COLOR_BORDER`, `COLOR_TEXT`, `COLOR_MUTED`) and change their values, so no import breaks; add new names alongside.

Contrast ratios below were computed with the WCAG 2.x formula.

**Surfaces, lines and text**

| CSS variable | Hex | Role | Python constant | Contrast / note |
| :--- | :--- | :--- | :--- | :--- |
| `--mq-bg` | `#0B0F14` | Page canvas | `COLOR_BG` **[NEW]** | — |
| `--mq-surface` | `#121821` | Panels, notes, table bodies | `COLOR_BG_CARD` (value changes) | — |
| `--mq-surface-2` | `#18202B` | Inputs, raised/hover, table headers | `COLOR_SURFACE_2` **[NEW]** | — |
| `--mq-sidebar` | `#0F141B` | Sidebar background | `COLOR_SIDEBAR` **[NEW]** | text on it 15.6:1 |
| `--mq-line` | `#252F3D` | Hairlines, dividers, panel borders (decorative) | `COLOR_BORDER` (value changes) | decorative only |
| `--mq-line-strong` | `#36445A` | Emphasized rules, chart axis lines | `COLOR_LINE_STRONG` **[NEW]** | decorative |
| `--mq-control-border` | `#5B6B82` | Borders of inputs/buttons (must identify the control) | `COLOR_CONTROL_BORDER` **[NEW]** | 3.28:1 on surface, 3.02:1 on surface-2 (meets 3:1) |
| `--mq-text` | `#E8ECF1` | Primary text | `COLOR_TEXT` (value changes) | 16.2:1 on bg, 13.8:1 on surface-2 |
| `--mq-text-2` | `#A9B4C2` | Secondary text, chart ticks | `COLOR_MUTED` (value changes) | 9.2:1 on bg, 7.8:1 on surface-2 |
| `--mq-text-3` | `#8392A5` | Captions, metric labels, eyebrow (non-accent) | `COLOR_TEXT_3` **[NEW]** | 6.1:1 on bg, 5.2:1 on surface-2 |
| `--mq-grid` | `#1E2733` | Chart gridlines | `COLOR_GRID` **[NEW]** | subtle by design |

**Accent (interface chrome only)**

| CSS variable | Hex | Role | Contrast / note |
| :--- | :--- | :--- | :--- |
| `--mq-rust` | `#D0693F` | Indicators, active-nav marker, header tick, text accent | 5.3:1 on bg, 4.5:1 on surface-2 |
| `--mq-rust-hi` | `#E08A63` | Hover, focus ring, eyebrow accent text, links | 7.3:1 on bg |
| `--mq-rust-fill` | `#B4532F` | Primary button fill (white text) | white on it 5.0:1 |
| `--mq-rust-fill-hover` | `#A94A28` | Primary button hover | white on it 5.7:1 |
| `--mq-rust-tint` | `rgba(208,105,63,0.12)` | Active-nav background, selected-tile wash | rust-hi text on it 5.9:1 |

**Data identity (charts, chips, card markers)**

| Python constant | Hex | Replaces | Note |
| :--- | :--- | :--- | :--- |
| `COLOR_CNN` | `#6FA3D4` | `#4FC3F7` | Muted steel blue; 7.2:1 on bg |
| `COLOR_QCNN` = `COLOR_HYBRID` | `#A08CCB` | `#AB47BC` | Muted violet; retained hue family so the audience's learned CNN=blue / QCNN=violet mapping survives; 6.5:1 on bg |
| `COLOR_MLP` | `#8A97A8` | `#78909C` | Cool grey |
| `COLOR_BASELINE` | `#C9B27C` | `#FFB74D` | Sand. Always drawn **dashed or patterned**. Distinct from the disagree gold (below). |

**Status (always paired with a text label)**

| Python constant | Hex | Replaces | Note |
| :--- | :--- | :--- | :--- |
| `COLOR_SAFE`, `COLOR_CORRECT` | `#5FB08C` | `#66BB6A` | Muted green; 7.4:1 on bg |
| `COLOR_UNSAFE`, `COLOR_INCORRECT` | `#E0605C` | `#EF5350` | Fills and chart marks; 5.5:1 on bg |
| `COLOR_UNSAFE_TEXT` **[NEW]** | `#EC7B77` | — | Text and badge text on the red tint (5.6:1; the fill color alone gives 4.3:1) |
| `COLOR_DISAGREE`, `COLOR_WARN` **[NEW]** | `#E0B04C` | `#FFA726` | Gold; 9.6:1 on bg |
| `COLOR_AGREE` | `#A9B4C2` | `#66BB6A` | **Neutral** — agreement is not correctness. Badge uses the neutral style. **[VERIFY]** where `COLOR_AGREE` is used before changing. |

> Colorblind safety: green/red status is never conveyed by color alone — every use carries a word (`SAFE`, `UNSAFE`, `CORRECT`, `INCORRECT`, `DISAGREE`). Blue vs violet series differ in lightness, and every chart direct-labels values and orders series identically (CNN, Hybrid, MLP, baseline).

Palette rules: rust is never a chart series or status; no more than the four identity hues appear in one chart; baseline is always dashed/patterned; no color outside this table is introduced without adding it here.

### 3.3 Typography

**Font stacks** (a single stack serves both the system-font stage and the optional self-hosted stage in Phase 4.6; no external CDN is ever used — the tool may run offline):

```
--mq-font-sans: "IBM Plex Sans", "Segoe UI Variable Text", "Segoe UI", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
--mq-font-mono: "IBM Plex Mono", "Cascadia Mono", Consolas, "SF Mono", Menlo, monospace;
```

If IBM Plex is not installed or self-hosted, the stack falls through to Segoe UI on the Windows target. Plotly figures use the same stacks (Python constants `FONT_SANS`, `FONT_MONO` **[NEW]**).

**Scale** (do **not** change the root font size; body text stays at Streamlit's default):

| Role | Size / line | Weight | Other |
| :--- | :--- | :--- | :--- |
| Page title (`st.title`) | 1.75rem / 1.2 | 600 | letter-spacing −0.01em |
| Section heading (`st.header`/`h2`) | 1.25rem / 1.3 | 600 | margin-top 32 px, bottom 8 px |
| Subsection (`st.subheader`/`h3`) | 1rem / 1.35 | 600 | |
| Lede (page subtitle) | 0.9375rem / 1.5 | 400 | `--mq-text-2`, max-width 78ch |
| Body | Streamlit default | 400 | prose blocks capped at 78ch where long (Methodology, notes) |
| Table / dataframe text | 0.875rem | 400 | numbers right-aligned, `tabular-nums` |
| Caption (`st.caption`) | 0.8125rem / 1.45 | 400 | `--mq-text-3` |
| Eyebrow | 0.6875rem | 500 | **mono**, uppercase, letter-spacing 0.08em, `--mq-rust-hi` for the step label, `--mq-text-3` for the rest |
| Metric label | 0.75rem | 500 | uppercase, letter-spacing 0.04em, `--mq-text-3`, **wraps** instead of truncating |
| Metric value | 1.5rem | 500 | **mono**, `tabular-nums` (native default is far larger) |
| Badge / chip | 0.6875rem | 600 | uppercase, letter-spacing 0.04em |

All numeric displays (metrics, table cells, logits, parameter counts) use `font-variant-numeric: tabular-nums` so columns align.

Native route first: set heading sizes/weights and fonts through `config.toml` theme keys (Phase 1.1, **[VERIFY]** each key against the installed 1.66.0); use CSS only for the roles config cannot express (eyebrow, metric value font, prose measure).

### 3.4 Spacing, widths and rhythm

4 px base. Tokens: `--mq-s1: 4px`, `--mq-s2: 8px`, `--mq-s3: 12px`, `--mq-s4: 16px`, `--mq-s5: 24px`, `--mq-s6: 32px`, `--mq-s7: 48px`.

| Item | Spec |
| :--- | :--- |
| Layout | `layout="wide"` retained **[VERIFY, D8]** |
| Content max-width | 1440 px, left-aligned within the main area (prevents charts and cards stretching absurdly at 1920) |
| Main-area padding | top 2rem (down from Streamlit's default, which is roughly 6rem), sides 2rem |
| Gap between sections | 24 px (`--mq-s5`); heading margin-top 32 px |
| Panel padding | 16 px; note-block padding 12 px × 16 px |
| Control-row gaps | 8 px |
| Page header | eyebrow, title, lede, optional meta line; 16 px below, closed by a 1 px `--mq-line` rule |
| Metric strip | one bordered container, height ≤ 72 px |
| Column floor | at 1366 px with the sidebar open, usable content ≈ 966 px. Therefore: 2-up columns ≥ 470 px each, 3-up ≥ 310 px, **never more than 3 content columns**; a 5-up *metric* strip (≈ 190 px each) is the only 5-up allowed |
| Sidebar width | optional: 17rem when viewport ≤ 1440 px (Phase 4.1, with fallback) |

### 3.5 Shape, borders and depth

- Radius: `--mq-r: 4px` for panels, notes, inputs, buttons, images; `--mq-r-sm: 2px` for badges and chips.
- Borders: 1 px only. No 2–4 px side bars on notes or cards.
- Shadows: none. Gradients: none. `backdrop-filter`: none.
- Dividers: `1px solid var(--mq-line)`.
- Focus: `outline: 2px solid var(--mq-rust-hi); outline-offset: 1px` on all interactive elements.
- Motion: 120 ms color/background transitions only, disabled under `prefers-reduced-motion: reduce`.
- Native `st.container(border=True)` border color is set through the theme key `borderColor` where it exists, not by CSS.

### 3.6 Component specifications

Each component lists whether Streamlit supports it natively or CSS is required, and the fragility of the CSS.

**Page header — `page_header(step, title, subtitle, meta=None)` [NEW, `components/layout.py`]**
- Eyebrow line (HTML, text only): leading 14 × 2 px rust dash, then `STEP 3 OF 6` in rust-hi followed by a muted section keyword, all mono caps. The landing page uses `OVERVIEW OF THE EXPLORER` style text with no step number **[VERIFY wording against existing copy]**.
- Title: **keeps the native `st.title(...)` call**, so the element is still a real `h1` (accessibility) and remains visible to AppTest. Emoji glyph removed from the string only after R5 passes. Style via config/CSS.
- Lede: `st.markdown`/`st.caption` using the existing descriptive text.
- Optional meta line: mono, text-3, e.g. dataset counts that are already shown on the page.
- Closing hairline. Fragility: low (own `mq-` classes).

**Sidebar and navigation (native file-based multipage retained)**
- Native `[data-testid="stSidebarNav"]` links restyled: text 0.875rem `--mq-text-2`; hover → `--mq-surface-2`; **active** → `--mq-text`, `--mq-rust-tint` background, 2 px rust marker at the left edge of the link (an *indicator*, not a callout bar). No icons (file-based pages cannot carry icons without renaming files — forbidden by R8).
- Sidebar background `--mq-sidebar`, set natively via `[theme.sidebar]` where supported, otherwise CSS on `[data-testid="stSidebar"]`.
- Sidebar workspace path on the landing page: mono, 0.75rem, text-3.
- Step context lives in the page header eyebrow and the footer progress, not in the sidebar.
- Fragility: medium (nav `data-testid` names can change between versions); every rule here is optional and dropped if Phase 0.7 does not find the hook.

**Buttons**

| Kind | Streamlit | Style |
| :--- | :--- | :--- |
| Primary (`type="primary"`) | native | CSS override of fill to `--mq-rust-fill`, white text, hover `--mq-rust-fill-hover` (native primary fill would be `#D0693F` with white text at only 3.9:1). Hook: `[data-testid="stBaseButton-primary"]` **[VERIFY name]**. |
| Secondary (default) | native | transparent, 1 px `--mq-control-border`, text `--mq-text`, hover `--mq-surface-2` |
| Tertiary (`type="tertiary"`) | native | text-only `--mq-text-2`, hover `--mq-rust-hi` |
| Disabled | native | text-3 on surface-2 |
| Height | | ≥ 32 px (hit target); radius 4 px |

**Inputs, selects, radios**: native widgets, themed through `config.toml` (background `--mq-surface-2`, border via control-border where the theme exposes it). Horizontal `st.radio` preset bars **stay `st.radio`** (tests and keys); no CSS pill-skinning. Hooks used only for the focus ring.

**Tabs**: native `st.tabs`; tab buttons targeted by `button[role="tab"]` (ARIA, stable). Unselected `--mq-text-2`, selected `--mq-text` with the native primary-color underline (rust). Font 0.9375rem, weight 500; tab bar gets a 1 px `--mq-line` bottom rule. Nested-tab variant (smaller, 0.8125rem) exists only as a fallback (page spec 4.5).

**Expanders**: native `st.expander`; border 1 px `--mq-line`, radius 4, no fill; summary 0.9375rem / 600. Default `expanded` values are changed only where a page spec says so.

**Metrics**: keep every `st.metric(...)` call (AppTest sees them). Place five metrics inside **one** `st.container(border=True)` row instead of five bordered cards; neutralize per-metric box (transparent background, no border). Value mono 1.5rem; label 0.75rem caps, wrapping. Hooks: `[data-testid="stMetric"]`, `stMetricLabel`, `stMetricValue` **[VERIFY]**. Delta arrows (if any are used) neutral colored.

**Notes — `note_block(kind, label, body, *, key)` [NEW]**
- Implementation: `st.container(border=True, key=f"mq-note-{kind}-{n}")` containing one text-only HTML label line and then `st.markdown(body)`. CSS targets `[class*="st-key-mq-note-"]` and the kind suffix (the `st-key-` class is Streamlit's documented container-styling hook).
- Look: surface background, 1 px border, 4 px radius, padding 12/16 px, label chip (mono caps 0.6875rem) top-left; **no side bars**.
- Kinds: `integrity` (label color rust-hi, border rust at 45 % alpha), `correction` (rust-hi), `warning` (gold label, gold border at 40 %), `limitation` (gold label, neutral border), `disclosure` (text-2 label, neutral border), `info` (text-2).
- Replaces: `.callout-box` and `.callout-warning` HTML on `app.py`, `1_Overview.py`, `5_Model_Comparison.py`, `6_Methodology.py`. Legacy class names stay defined (restyled) until Phase 5 grep proves them unused.
- Native `st.info/success/warning/error` stay as elements (R5); restyled globally and compactly via theme keys `blueColor/greenColor/orangeColor/redColor` **[VERIFY]**, plus a small padding rule on `[data-testid="stAlert"]` if needed.

**Badges — `badge_html(...)` in `style/theme.py` (restyled, signature unchanged)**
- 2 px radius, 0.6875rem / 600 caps, padding 2px 6px, 1 px border at ~45 % status alpha, background status tint at 14 % alpha, text in the status text variant (`COLOR_UNSAFE_TEXT` for red).
- Variants (legacy class names retained): `.badge-safe`, `.badge-correct`, `.badge-unsafe`, `.badge-incorrect`, `.badge-disagree`, `.badge-model-cnn`, `.badge-model-qcnn`, `.badge-model-mlp`; add `.badge-neutral` (agree).
- Always text; never icon-only.

**Series chip**: 8 × 8 px square in the series color preceding a model's name in cards, ledger rows and chart-adjacent labels (`.mq-chip--cnn` etc.). It is the same color as the corresponding chart series.

**Key–value table — `kv_table(rows)` [NEW]**: replaces raw HTML `<table style="…">` in prediction cards. Two columns: label (text-3, 0.8125rem) / value (text, mono tabular-nums). 1 px `--mq-line` row separators, 6 px vertical padding. Escape all dynamic strings with `html.escape`. Units remain in the label or value exactly as now.

**Tables (`st.dataframe`)**: keep `st.dataframe` (sorting, search and download stay). Presentation only: `hide_index=True` where an index is meaningless, `column_config` for alignment and column widths, `width="stretch"`. **Never change displayed precision or any cell value.** For the Moon logits table, `st.column_config.ProgressColumn` may be applied to probability columns only if the displayed digits can match the current display; otherwise skip (R4). No colored deltas or best-value highlighting (see §2.3).

**Download controls**: native `st.download_button`, secondary style, label text unchanged, right-aligned in the control row. Data and file names unchanged.

**Captions and explanatory notes**: `st.caption` (13 px, text-3). A caption that qualifies a chart sits directly under the chart, 8 px gap.

**Section header — `section_header(title, caption=None)` [NEW]**: wraps `st.subheader` plus an optional caption; used so heading spacing is uniform.

**Step progress — `step_progress(current)` [NEW]**: six 22 × 4 px segments (filled rust up to the current step, `--mq-line-strong` beyond) with an `aria-label="Step 3 of 6"`. Text-only HTML. Reused in the footer; the header eyebrow carries the words.

### 3.7 Chart specification

One Plotly template plus one helper, both in `style/theme.py` (the builders in `components/charts.py` and `components/confusion_matrix.py` already import theme constants **[VERIFY]**):

- `MARSQCNN_TEMPLATE` — a `plotly.graph_objects.layout.Template`.
- `finalize_figure(fig, *, height, legend="top")` — applies the template, fixed height, legend placement, margins, hover style. Each builder calls it as its last step. Per-builder code retains only what is specific to that chart.

**Hard rules for charts**
1. Styling attributes only. **Do not change** any trace data, trace names, axis titles, figure title text, hover text, error-bar arrays, category order, or axis inversions (the mosaic scatter's inverted row axis stays). The audit's test `test_mars_chart_layout_and_units` and `test_chart_builders_return_valid_figures` constrain these (R5).
2. Units appear in every axis title or subplot title exactly as today (°, m, unitless fraction, ms, s).
3. Error bars represent **sample standard deviation**; wherever they appear the existing "±1 SD" wording is preserved **[AUDIT §6]**.
4. Bar charts start at zero. For 0–1 metrics use a fixed range ≈ [0, 1.05]. **[VERIFY]** the current ranges; if a current range is not zero-based, set zero-based and show value labels with the existing precision so small differences are not visually inflated.

**Template values**

| Property | Value |
| :--- | :--- |
| `paper_bgcolor`, `plot_bgcolor` | `rgba(0,0,0,0)` — the chart sits on the page canvas or the container surface |
| Font | `FONT_SANS`, 12 px, color `--mq-text-2`; axis titles 12 px `--mq-text-2`; chart title 14 px `--mq-text`, **left-aligned** (`x=0, xanchor="left"`) |
| Gridlines | `#1E2733`, 1 px, horizontal only for bar/line charts |
| Axis lines / zerolines | `#36445A`, 1 px; ticks outside, 4 px |
| Legend | horizontal, top-left above the plot (`orientation="h", x=0, y=1.1`), no border, series order CNN → Hybrid → MLP → baseline |
| Margins | `t=56, r=24, b=48, l=56` (adjusted only where a title or rotated tick forces it) |
| Hover label | `bgcolor #18202B`, `bordercolor #36445A`, `font` 12 px `--mq-text`; existing `hovertemplate` strings kept |
| Bars | `marker_line_width=0`, `bargap≈0.35`, `bargroupgap≈0.08`; value labels outside bars with the **existing** numeric precision; baseline series dashed/patterned (`marker_pattern_shape`) |
| Lines (loss curves) | model hue; **train solid, validation dashed**; line width 2; no markers unless ≤ 20 epochs |
| Fixed heights | single bar chart 360 px; multi-subplot regression chart 380 px; softmax chart 300 px; heatmap 340 px; scatter 420 px; loss curve 320 px |
| Interaction | `config={"displaylogo": False, "displayModeBar": "hover"}` **[VERIFY** `st.plotly_chart(config=…)` in 1.66**]** |
| Streamlit theme coupling | pass `theme=None` to every `st.plotly_chart` so the figure renders exactly as specified rather than being re-themed by Streamlit **[VERIFY** current values per call; D-decision: use `None` everywhere**]** |

**Heatmaps (confusion matrices)**
- One shared single-hue sequential scale across all three models, anchored to the same `zmin/zmax`, so cell intensities are comparable: `[[0.0,"#141B25"],[0.5,"#2C4A6B"],[1.0,"#7FB0DE"]]`.
- Annotation text color switches by normalized cell value: `#0B0F14` when the value is above ≈ 0.65, otherwise `#E8ECF1`; Flash must compute the WCAG ratio for each cell color against its text color and keep it ≥ 4.5:1 (dark text on `#7FB0DE` is 8.4:1; light text on `#2C4A6B` is 7.7:1).
- Tick labels: class names only (`ejecta`, `oldcrater`, `none`) at 11 px, `tickangle=0`; axis titles "Predicted class" / "True class" carry the meaning previously repeated as `Pred:` prefixes. **[VERIFY]** the existing tick text and that no test asserts it (R5).
- Preserve the existing `Count (Pct%)` cell content; set it on two lines (`count<br>(pct%)`) to fit narrow cells.

**Continuous-valued scatter (`build_mars_grid_scatter`)**: categorical coloring (`color_by` is safe/unsafe or similar) uses the status colors with text-bearing legend entries; continuous coloring uses Plotly's perceptually uniform, CVD-safe `Cividis` (never a rainbow ramp). The selected patch is marked with a 2 px `#E8ECF1` ring, not rust (rust never encodes data).

### 3.8 What Streamlit supports natively versus what needs CSS

| Need | Native mechanism | CSS needed? | Fragility |
| :--- | :--- | :--- | :--- |
| Dark base, background, text, accent | `config.toml` `[theme]` (`base`, `primaryColor`, `backgroundColor`, `secondaryBackgroundColor`, `textColor`) | No | Low |
| Fonts, heading sizes/weights, radius, border color, link color, sidebar section, alert colors | Newer theme keys (`font`, `headingFontSizes`, `headingFontWeights`, `baseRadius`, `borderColor`, `linkColor`, `[theme.sidebar]`, `blueColor`…) **[VERIFY each key against 1.66.0; drop any rejected key]** | No | Low–medium |
| Hide Deploy button / chrome | `[client] toolbarMode = "minimal"` **[VERIFY]** | No | Low |
| Bordered panel | `st.container(border=True)` | No | Low |
| Style a specific container | `st.container(key=…)` → `st-key-…` class | Yes, via that class | Low (documented) |
| Compact primary action, navigation | `st.button(type="primary")`, `st.page_link`, `st.switch_page` (reuse the mechanism `render_walkthrough_footer` already uses) | Primary fill override | Medium |
| Metrics layout | `st.metric` in columns/container | Small `data-testid` rules | Medium |
| Table presentation | `st.dataframe` + `column_config` | No | Low |
| Page header, notes, kv table, step progress, badges | Helpers in `components/layout.py` | Own `mq-` classes | Low |
| Sidebar nav active state | none | `data-testid` rules | Medium (optional) |
| Sidebar width at ≤ 1440 px | none | Media query on `stSidebar[aria-expanded]` **[VERIFY]** | Medium (optional; drop if collapse breaks) |

### 3.9 CSS authoring rules

1. All CSS lives in `style/theme.py` and is injected only by `apply_theme()`; pages call it exactly as they do today.
2. New classes are prefixed `mq-`. Legacy class names remain defined (restyled) until Phase 5.
3. Delete the dead `.reportview-container` rule once Phase 0.7 and Phase 1 confirm it has no effect.
4. Every color in CSS is a `var(--mq-…)`. A grep for raw hex in CSS (outside the `:root` block) must return nothing.
5. Maximum specificity: one class plus one attribute. No `!important` except for the primary button fill and the optional sidebar width, each marked with a comment.
6. Every rule is commented with the audit issue it addresses (G1…G12) so it can be removed cleanly.

### 3.10 Responsive and accessibility requirements

**Responsive.** The supported range is 1366×768 to 1920×1080 **[AUDIT §10.3]**; mobile and sub-1200 px are out of scope, but the app must not produce horizontal page overflow at 1280 px. Streamlit columns do not collapse at these widths, so layout must be chosen to work at 1366 (column floors in §3.4). The sidebar may be collapsed by the user at 1366 px; the layout must also look correct then.

**Accessibility.**
- Body and secondary text ≥ 4.5:1 against its background (all tokens above comply); interactive-control boundaries ≥ 3:1 (`--mq-control-border`).
- Status is never conveyed by color alone (text labels in every badge, legend entry, table cell).
- Visible keyboard focus on every control; tab order follows Streamlit's native order (do not reorder DOM).
- Native buttons/tabs/expanders retained (keyboard operable); custom HTML limited to non-interactive text.
- `prefers-reduced-motion` honored.
- Minimum text size on the page: 12 px (chart ticks), 11 px only for uppercase mono eyebrows/badges.
- Each page has exactly one `h1`.

---

