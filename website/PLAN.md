# Website plan: Mars | Moon exploration site for the CNN vs hybrid CNN–QCNN study

## Context

The study's results currently live only in the Streamlit app (`visualization_app/`), which needs the gitignored `runs/` and `data/raw/` artifacts on disk. The goal is a standalone, deployable Vite + React + TS site in `website/`. It presents the same audited results as an exploration experience: choose a world, pick a site, pan the strip, see per-tile predictions, watch the comparison animate. `visualization_app/` and `src/` stay untouched. All numbers come from a one-off export into small committed JSON and image files.

## How the work proceeds

- Build one step at a time. After each step: run the tests, commit to `website-redesign` with a clear message, push `website-redesign` to `origin`, then stop for review.
- **Merging:** merge `website-redesign` into `main` with a normal merge commit. Don't squash or rebase: the `/lab` permalinks point at commit SHAs on this branch and must stay reachable from `main`.
- After steps 5, 6, 8 and 9, also take built-in-browser screenshots at 1920×1080 and 1366×768.

## Findings

### Sites, strips and per-tile predictions

**Mars: five real HiRISE sites.** Each site is one ortho strip (JP2) plus a DTM, and each site is one leave-one-site-out fold.

| site_id | Display name | Ortho product | Ortho size (w×h px) | Tiles | Fold |
|---|---|---|---|---|---|
| site_01 | Northern Mid-Latitude Crater | ESP_062806_2150_RED_C_01 | 6109×7735 | 2,195 | 0 |
| site_02 | Kaiser Crater | ESP_039245_1325_RED_D_01 | 2682×3835 | 1,885 | 1 |
| site_03 | Layered Yardangs | ESP_023051_1865_RED_C_01 | — | 2,484 | 2 |
| site_04 | Coprates Chasma | ESP_026417_1660_IRB_C_01 (band 2 = RED) | — | 979 | 3 |
| site_05 | Huo Hsing Vallis | PSP_003495_2065_RED_C_01 | 8131×27626 | 7,690 | 4 |

Per-tile predictions exist for both models, for every fold and seed:
- Files: `runs/pipeline_repair_audit/retraining_seedset/runs/{model1_resnet18,model2_qcnn}/fold_k_seed_{42,43,44}/test_predictions.parquet`.
- Tile positions are in `runs/controlled_evaluation/processed/metadata.parquet` (`row`, `col`, `native_window_px`), keyed by `tile_id`.
- `src/data/preprocess_real.py` indexes the DTM and ortho arrays with the same `r, c`, so the positions map straight onto the ortho strip.

**Moon: no sites or strips.** The data is LROCNet: 779 independent 227×227 test crops in 3 classes, with predictions from 3 models × 3 seeds in `runs/logs/seed_*/results.json`. The Moon rail is by class: ejecta, oldcrater and none.

### Coordinates

- **Mars:** coordinates come exactly from `crs_wkt` (equirectangular) and `transform` in `metadata.parquet`, using pure numpy. Site centres:

  | Site | Latitude | Longitude (°E) |
  |---|---|---|
  | site_01 | 34.81° N | 100.68 |
  | site_02 | 47.01° S | 18.74 |
  | site_03 | 6.32° N | 34.69 |
  | site_04 | 13.86° S | 300.97 |
  | site_05 | 26.44° N | 66.60 |

- **Moon:** no coordinates. `provenance_manifest.csv` leaves site, observation and source IDs blank by design, and the filename prefixes are unverified. The Moon globe will have **no pins**.

### Inputs for the `/lab` page

**Checkpoints:** one representative checkpoint per model is loaded read-only on CPU.
- Mars: `runs/pipeline_repair_audit/retraining_seedset/runs/{model1_resnet18,model2_qcnn}/fold_0_seed_42/best_checkpoint.pt`. Fold 0 means test site site_01. Every fold and seed has a `best_checkpoint.pt` and a `last_checkpoint.pt`.
- Moon: `data/raw/moon/lrocnet/experiment_lunar_cnn_qcnn/runs/checkpoints/{cnn,hybrid,frozen_mlp}_seed_42.pt`.

**Training curves:** these exist, so the Lab training-curves section is built.
- Mars: all 30 runs (2 models × 5 folds × 3 seeds) have `epoch_log.csv` with `epoch, train_loss, val_loss, lr, epoch_wall_time_sec`. Fold 0 seed 42 has 8 CNN epochs and 4 QCNN epochs.
- Moon: each seed has `runs/logs/seed_*/history.csv` with `model, epoch, train_loss, val_macro_f1, seconds` for cnn (6 epochs), qcnn (7) and frozen_mlp (5).
- Wall-time columns are **not** exported to the Lab page, because timings aren't comparable.

**Code permalinks:**
- GitHub remote: `MadSank/Quantam-Marsly`.
- Mars code (`src/`) is tracked, so it can be permalinked.
- The Moon experiment code is project code, not code shipped with the LROCNet dataset. It is tracked in `experiments/moon/` (moved from gitignored `data/raw/moon/lrocnet/` in commit `8602eb8`; run artifacts stay in `data/raw`).
  - `/lab` permalinks the Moon code exactly like the Mars code. The Moon hybrid's circuit is the shared `src/quantum/circuits.py::BatchedQCNNHead`.
  - **Code history note:** the current `experiments/moon/train.py` is not the exact source that produced the reported Moon numbers; test access was moved later. `/lab` shows the "Code history" section of `experiments/moon/README.md` next to every Moon excerpt, linked by permalink. That section records what changed, the evidence, and the two re-checks that would settle it.
- Permalinks use the commit SHA recorded at export time. They resolve on GitHub only after that commit is pushed; at the time of writing, origin only has the initial commit.

## Pages

1. `/`: split landing, "choose a world", with Mars and Moon halves and a link to About.
2. `/mars`: site rail, strip panel and results panel. `?site=&model=&seed=` is held in the URL so views are shareable.
3. `/moon`: class rail, crop-mosaic panel and results panel. `?class=&model=&seed=`.
4. `/about`: the neutral shared page, covering method, data sources with licences and citations, and every required caveat.
5. `/lab`, "Under the hood": a neutral-theme page for a reviewer to inspect how the project works. It has six sections:
   1. **Pipeline diagram.** HiRISE strip + DTM → tiling → slope / roughness / hazard targets → safe-proxy label → CNN / hybrid → leave-one-site-out evaluation. Each stage links to its code permalink.
   2. **CNN internals (Mars ResNet-18).**
      - The conv1 filter grid: 64 filters.
      - Feature maps at each ResNet stage for one real test tile from fold 0 (site_01), with the tile named.
   3. **Hybrid/QCNN internals.**
      - A circuit diagram generated from the actual PennyLane code (`qml.draw_mpl` → SVG).
      - The trained rotation parameters per layer, shown as a heatmap.
      - A plain-language explanation of the classical→qubit projection: frozen 512-d embedding → 8-d bottleneck → angle encoding.
      - The same three items for the Moon hybrid head.
   4. **Parameter breakdown.**
      - Total vs trainable parameters per model: Mars CNN 11,242,563 vs hybrid head 4,152; Moon CNN 59,347 during training (195 after freeze), hybrid head 568, MLP head 547.
      - An explanation of the frozen backbone.
   5. **Training curves per fold and seed.**
      - Mars: train and validation loss per epoch.
      - Moon: train loss and validation macro-F1 per epoch.
      - A fold/seed selector; both models are shown in fixed order using the neutral-shell model colours.
      - No wall-time is shown.
   6. **Code walkthrough.**
      - Curated excerpts: model definitions, circuit and batched evaluation, training loop, train-fold-only scaler, and leave-one-site-out split.
      - Syntax-highlighted at build time with Shiki.
      - Each excerpt has a short plain-language note and a GitHub permalink to the exact lines at the exported commit.

   The page makes no performance claims beyond the existing caveats; it links to About for those.
6. *(Phase 2)* A globe mode inside each world, opened by a toggle in the image panel.

Routing uses `react-router` with hash routing so any static host works.

## Component tree

```
<App>
  <Routes>
    <Landing> → <WorldDoor world="mars"/> <WorldDoor world="moon"/>
    <WorldLayout world>                 // sets data-world on root → theme tokens; ThemeContext
      <WorldHeader/>                    // world switch, About link, caveat chip
      <Explorer>
        <Rail items onSelect>           // shared; RailItem = thumbnail + label + n
        <ImagePanel>
          <PanZoomViewport>             // shared: wheel/drag/pinch, keyboard +/-/arrows, reset
            Mars:  <StripImage/> (or <DeepZoomImage/> for pyramid sites)
                   + <TileOverlay model seed/>         (canvas; prediction layer)
                   + <TruthLayer visible/>             (canvas; ground truth, toggleable, off by default)
            Moon:  <CropMosaic crops model seed/>
          <ModelToggle cnn|qcnn [|mlp on Moon]/> <SeedSelect/> <TruthToggle/> <OverlayLegend/> <TileInspector/>
        <ResultsPanel>
          <MetricGroup title="Regression error (lower is better)">  <PairedBar/>×n
          <MetricGroup title="Safe-proxy decisions">  recall + FPR + coverage always together
          <SeedBreakdown/>
          <CaveatNote id=…/>
          <AnimatedNumber/> <GrowBar/>  // shared; respect reduced motion
    <AboutPage> <Method/> <DataSources/> <CaveatList all/>
    <LabPage>                           // neutral shell
      <PipelineDiagram/>                // SVG stages, each a link to a code permalink
      <FilterGrid/> <FeatureMapStrip/>  // CNN internals
      <CircuitFigure world/> <RotationHeatmap/> <ProjectionExplainer/>   // Mars + Moon hybrid
      <ParamBreakdown/>
      <TrainingCurves/>                 // fold/seed selector; loss / val metric only
      <CodeExcerpt/>×n                  // pre-highlighted HTML + note + permalink
  <MotionConfig reducedMotion="user">
lib/: typed data loaders + cache, format (units; null → "undefined (no predicted-safe tiles)"), useReducedMotion
```

Motion uses `motion`. Charts are hand-built SVG components following the dataviz skill.

## JSON data schema (`website/public/data/`)

Two rules apply to every file:
- `null` replaces NaN.
- Every file carries `source` paths and `generated_at`.

**`manifest.json`**
- `{ schema_version, generated_at, worlds, files }`

**`mars/sites.json`**
- `[{ site_id, name, fold, ortho_product_id, dtm_product_id, center: {lat, lon_e}, n_tiles, n_true_safe, native_resolution_m, preview: {src, width, height, scale_from_native}, deepzoom?: {dzi, tile_size, levels}, thumb: {src, width, height} }]`

**`mars/headline.json`** holds the 5-site aggregates:
- `{ fixed: { CNN: {metric: {mean, sd, n_sites}}, CNN_QCNN: {...} }, validation_calibrated: {...} }`
- No interval columns and no paired differences are exported.

**`mars/sites/<site_id>.json`**
- `{ site_id, mean: { fixed: {CNN, CNN_QCNN}, validation_calibrated: {...} }, per_seed: [{ seed, model, regime, RMSE_*, MAE_*, safe_recall, false_positive_rate, predicted_safe_fraction, safe_precision, TP, FP, TN, FN, risk_threshold? }] }`
- Sources are `*_site_mean.csv` and `*_per_site_seed.csv`.

**`mars/tiles/<site_id>.json`** is columnar to keep it small:
- `{ tile_id[], row[], col[], size_native, true: {slope[], rough[], hazard[], safe[]}, pred: { cnn: {"42": {slope[], rough[], hazard[], safe[]}, …}, qcnn: {…} } }`
- Values are rounded to 3 decimal places. If it gets too large, fall back to typed-array `.bin` files with a JSON header.

**`moon/summary.json`**
- `{ classes, test_counts, models: {cnn, hybrid_qcnn, frozen_embedding_mlp, majority_train_baseline}: {accuracy, balanced_accuracy, macro_f1: {mean, sd}}, per_seed, parameter_counts: { cnn: {trainable_during_training: 59347, trainable_recorded_post_freeze: 195, note}, hybrid_head: 568, mlp_head: 547 }, timing: { …, comparable: false, note } }`

**`moon/classes/<class>.json`**
- `{ class, crops: [{ sample_id, img, truth, pred: { cnn: {"42": {label, conf}}, hybrid_qcnn, frozen_embedding_mlp } }] }`

**`caveats.json`**
- `[{ id, world: "mars"|"moon"|"both", title, text, source }]`
- The text is quoted verbatim from `FINAL_READINESS_AUDIT.md` or the app's caveat rules.
- The mixed-results caveat uses the audit's §1 wording only:
  - Mars: "Numerically, the QCNN reduces the reported regression errors and false-positive rate, but the CNN has higher safe recall and coverage. These results therefore do not identify a clear overall winner."
  - Moon: "Three seeds on one supplied split are descriptive evidence, not proof of a general advantage."
- The site never mentions intervals or "crossing zero".

**`about.json`**
- `{ method: [...], sources: [{ name, provider, licence, citation, url, used_for }] }`
- Sources to credit:
  - HiRISE ortho images and DTMs: NASA/JPL/University of Arizona.
  - LROCNet: its licence and paper citation.
  - Phase 2 globe textures, when they are added.
- Any licence or citation text that can't be confirmed locally is marked `"TO CONFIRM"`. I'll list those items for you at step 10 rather than guess.

**`lab/` (Lab page; about 3 MB budget, included in the size report)**

**`lab/checkpoints.json`** records which checkpoint fed each figure.
- `[{ model, world, path, fold?, seed }]`
- Mars uses fold 0 / seed 42; Moon uses seed 42.

**`lab/params.json`**
- `{ mars: { cnn: {total, trainable}, hybrid: {total, trainable, frozen_backbone, bottleneck, qcnn_head} }, moon: { cnn: {trainable_during_training, trainable_post_freeze}, hybrid_head, mlp_head } }`
- Values are counted from the loaded modules and cross-checked against `param_counts.json` and the audit.

**CNN internals**
- `lab/mars_cnn_conv1.png`: an 8×8 grid of the 64 conv1 filters, each normalised per filter.
- `lab/mars_cnn_features.json`: `{ tile_id, site_id, stages: [{name, shape, channels_shown, png}] }`.
- One PNG per stage (`lab/mars_cnn_stage_<name>.png`), showing a grid of the first 16 channels, with the input tile PNG alongside.

**Hybrid internals**
- `lab/{mars,moon}_circuit.svg`: drawn by `qml.draw_mpl`, saved as SVG and minified.
- `lab/{mars,moon}_rotations.json`: `{ layers: [{name, shape, values[][]}], value_range }`, read straight from the checkpoint's circuit weights.
- `lab/{mars,moon}_projection.json`: `{ in_dim, out_dim, encoding, notes }`.

**`lab/curves.json`**
- `{ mars: { "<fold>_<seed>": { cnn: {epoch[], train_loss[], val_loss[]}, qcnn: {…} } }, moon: { "<seed>": { cnn: {epoch[], train_loss[], val_macro_f1[]}, qcnn: {…}, frozen_mlp: {…} } } }`
- No wall-time or lr columns.

**`lab/excerpts.json`**
- `{ commit, repo: "MadSank/Quantam-Marsly", excerpts: [{ id, title, path, start, end, note, permalink, code, html }] }`
- `code` is the raw text sliced at export.
- `html` is filled in by Shiki at build time.
- Moon excerpts also carry `history_note: { text, permalink }`. The text is the "Code history" section of `experiments/moon/README.md`, read at export.
- Excerpts are located by a function or class name, not by hard-coded line numbers, so they stay correct when the code moves.

## Images

- **Mars previews:** `img/mars/<id>_preview.webp`, quality about 80, with a per-site 2–98% display stretch labelled "display stretch only". Long edges:
  - site_01: full native resolution, 6109×7735.
  - site_05: 8000 px.
  - site_02, site_03, site_04: about 4000 px.
- **Mars rail thumbnails:** `img/mars/<id>_thumb.webp`, about 320 px.
- **Deep zoom (step 12):** if the budget allows, a tiled pyramid (512 px WebP tiles, DZI-style) for site_01 and site_05 under `img/mars/<id>_dz/`. The previews stay in place as fallback.
- **Moon:** test crops go into one WebP sprite sheet per class at 128 px each, plus a thumbnail per class.
- **Size reporting:** every export script prints per-file sizes and the `public/data` total, broken down into data, images and lab. The total must stay under **25 MB**, with `lab/` at about 3 MB of that. I'll report sizes after steps 2, 3, 11 and 12.
- **Never** copy raw TIFF, JP2 or IMG files.

## Export scripts (`website/scripts/`)

**`export_data.py`** runs on `visualization_app/.venv`.
- Puts `visualization_app` on `sys.path` and reuses `data_loading.paths`.
- Reuses these `mars_data` loaders:
  - `load_mars_metadata`
  - `load_mars_aggregate_metrics`
  - `load_mars_per_site_metrics`
  - `load_mars_site_mean_metrics`
  - `load_mars_validation_calibrated_metrics`
  - `load_mars_predictions`
- Reuses these `moon_data` loaders:
  - `load_moon_results`, which already applies the 59,347 correction
  - `get_moon_predictions_df`
  - `load_moon_recomputed_metrics`
- The `@st.cache_data` wrappers work outside Streamlit, with only a warning.
- Coordinates come from `crs_wkt` + `transform` with numpy. Read `transform` as `row["transform"]`, because a pandas method shadows the attribute name.

**`export_images.py`** also runs on the app venv.
- Its Pillow has JPEG 2000 support and decodes at reduced resolution.
- Site_04 uses band 2, and the stretch uses valid pixels only.
- If decoding site_05 fails, fall back to running this script on root `.venv`, which has rasterio's decimated read.

**`export_internals.py`** runs on the root `.venv`, which has torch and pennylane.
- Read-only: it loads checkpoints with `map_location="cpu"`, uses `torch.no_grad()` and `model.eval()`, and never trains or writes back.
- **Mars:**
  - Builds the models with `src.models.build_model` from that run's `config_snapshot.yaml`, then loads `fold_0_seed_42/best_checkpoint.pt`.
  - The feature-map tile is one real site_01 test tile, chosen deterministically (the first tile in `test_predictions.parquet` order). Its image comes from the processed shard. The training-fold image normalisation is applied exactly as `src/controlled_evaluation.py` does it.
  - The circuit diagram comes from the QNode in `src/quantum/circuits.py`.
- **Moon:**
  - Imports the model classes from the tracked `experiments/moon/train.py`.
  - Loads the `*_seed_42.pt` checkpoints.
- **Curves:** read from `epoch_log.csv` and `history.csv`.
- **Excerpts:**
  - Read from `src/` at `git rev-parse HEAD`.
  - The script refuses to run if any excerpted file has uncommitted changes, so permalinks match the shipped code.
- Matplotlib uses the Agg backend. Figures use the neutral-shell palette, with no green or red.

**`scripts/highlight.mjs`** runs as `prebuild`.
- Runs Shiki over `lab/excerpts.json` and writes the `html` fields.
- No highlighter ships to the browser.
- Writes `img/manifest.json` with the source → derivative mapping.

## Theme tokens

The page root carries `data-world="mars|moon"`, and shared components read only these semantic tokens. Hallmark (step 5) may refine values, fonts and layout. It must keep:
- the token names;
- the model mapping: **CNN = warm hue, hybrid/QCNN = cool hue in both worlds**, with MLP distinct on the Moon;
- green and red reserved for ground truth only.

| Token | Mars (warm) | Moon (cool, high contrast) |
|---|---|---|
| `--bg` | `#1A0F0A` | `#08090B` |
| `--surface` | `#2A1810` | `#121418` |
| `--surface-2` | `#3A2216` | `#1C1F24` |
| `--border` | `#5A3422` | `#2E333B` |
| `--text` | `#F4E6D8` | `#F2F4F7` |
| `--text-muted` | `#C29A7C` | `#9AA3AE` |
| `--accent` | `#D2691E` rust | `#E6E9EE` silver |
| `--accent-2` | `#C8962E` ochre | `#7E8896` slate |
| `--model-cnn` (warm) | `#E8A15A` sand-amber | `#E8C58F` pale amber |
| `--model-qcnn` (cool) | `#7FB2C4` dusty teal | `#8FB7E8` steel blue |
| `--model-mlp` | — | `#B9A6E0` lavender |
| `--truth-safe` / `--truth-unsafe` | `#9BC27A` / `#D9573A` | `#9BC27A` / `#E06A5A` |
| `--focus` | `#FFD08A` | `#FFFFFF` |
| fonts | display: geologic serif; body: humanist sans | display: condensed grotesk; body: same sans; tabular numerals |

Within each world, the two model colours have equal visual weight: same lightness band, different hue, and neither green nor red. The dataviz palette validator checks contrast and colour-blind separation in both worlds.

## Rules for presenting the science

These rules are built into the components:

- **Neutral presentation:**
  - Both models always appear side by side in a fixed order (CNN, then QCNN).
  - There are no winner badges, no "best" highlighting and no sorting by performance.
  - Each metric shows its direction ("lower is better").
- **Safety metrics:**
  - Safe recall, FPR and coverage always appear together.
  - Both decision regimes, fixed and validation-calibrated, are shown.
- **Mixed-results caveat:** shown persistently in the results panel, using the audit's §1 wording from `caveats.json`. No paired-difference table, no interval chart, and no mention of intervals.
- **Moon:**
  - Balanced accuracy and macro-F1 are listed first; accuracy is shown with the majority-baseline line (68.4%).
  - Timings appear only on About, as "separately measured, not comparable".
  - The CNN parameter count is 59,347 trainable during training, with the 195 explained.
- **Other required caveats:**
  - "Safe" is labelled a proxy everywhere it appears.
  - "No quantum advantage claimed; circuits simulated" appears in both worlds' footers and on About.
  - The Mars hybrid uses a frozen backbone, so this is a pipeline comparison, not equal capacity.

## Tests (`website/`)

**Vitest + Testing Library**

**`data.audit.test.ts`** checks the exported JSON against constants transcribed from `FINAL_READINESS_AUDIT.md`:
- Mars RMSE / MAE for all 6 target × model pairs.
- Fixed-threshold recall / FPR / coverage: 5.56 / 5.27 / 5.67 and 2.77 / 2.78 / 3.02.
- Validation-calibrated values: 43.33 / 39.65 / 41.35 and 30.62 / 33.94 / 33.86.
- The Moon table: mean ± SD for all 4 models.
- Counts: 15,233 tiles; 779 Moon test crops; 3,608 / 613 / 779 split sizes.
- Parameter counts: 59,347 / 195; 11,242,563 / 4,152; heads 568 / 547.
- Tolerances match the app's tests.

**`data.integrity.test.ts`**
- Site tile counts sum to 15,233, and site_01 has 2,195.
- CNN and QCNN tile ID lists are identical for each site and seed.
- Spot checks: per-seed metrics recomputed from the tile JSON match `mars/sites/<site_id>.json`.
- Moon confidences are between 0 and 1, and each predicted label matches its argmax.

**`caveats.test.tsx`**
- Every required caveat ID exists.
- Each caveat renders on About and in its world.
- "59,347" is rendered.
- The mixed-results text matches the audit's §1 wording verbatim.

**`removed-section.test.ts`** searches `src/` and `public/data/` and fails on any of these strings:
- `paired_differences`
- `Paired Site Differences`
- `t95_low`, `t95_high`, `t95`
- `t interval`, `t-interval`
- `cross zero`, `crosses zero`, `crossing zero`

**`framing.test.tsx`**
- Both models render in fixed order regardless of their values.
- No element has `data-winner`.

**`overlay.test.ts`**
- The overlay style function never returns a truth colour for prediction layers.
- Disagreement cases produce a hatched or dashed style that differs from agreement cases by more than hue alone.

**`reduced-motion.test.tsx`**
- With reduced motion, `AnimatedNumber` renders the final value immediately.

**`sources.test.ts`** is enabled from step 10.
- Every `about.json` source has a licence and a citation, and no `TO CONFIRM` values remain.

**`lab.test.tsx`** is enabled from step 11.
- **Parameter counts:** the counts rendered on `/lab` and stored in `lab/params.json` equal the audit constants: 11,242,563; 4,152; 59,347; 195; 568; 547.
- **Permalinks:**
  - Every excerpt's permalink matches `https://github.com/MadSank/Quantam-Marsly/blob/<commit>/<path>#L<start>-L<end>`.
  - `<path>` exists in the repo at `<commit>`, checked with `git cat-file -e <commit>:<path>`.
  - `start ≤ end ≤` that file's line count at that commit.
  - The excerpt's `code` equals those lines.
- **No performance claims:**
  - The rendered Lab text and the `lab/*.json` notes contain none of these phrases: `better`, `outperform`, `superior`, `wins`, `winner`, `beats`, `faster`, `speedup`, `speed-up`, `advantage`, `improves`, `more accurate`.
  - Exceptions:
    - Caveat text pulled verbatim from `caveats.json`, allowed through by caveat ID.
    - The exact direction labels `lower is better` and `higher is better`, which every metric must show. Only those exact phrases are allowed; any other use of `better` still fails.
  - The same removed-section deny-list applies.
- `curves.json` has no wall-time or lr fields.
- Every Moon excerpt has a `history_note` whose text equals the README's "Code history" section, and the rendered page shows it next to each Moon excerpt.

**Export tests (pytest, `website/scripts/tests/`)**
- `export_data.py` and `export_images.py` tests run on the app venv and are skipped when `runs/` is absent.
  - They check that exported values equal the source CSV and parquet files before rounding.
- `export_internals.py` tests run on the root venv and are skipped when the checkpoints are absent.
  - Parameter counts match `param_counts.json`.
  - Rotation arrays match the checkpoint tensors.
  - Loading never modifies a checkpoint's mtime or hash.

## Build order

1. **Scaffold.** `website/` with Vite + React + TS, oxlint, Vitest, react-router (hash), `motion`, and the token CSS. `npm run dev`, `build`, `test` and `lint` all work. `website/README.md` covers the export commands.
2. **Export data.** Write `export_data.py` for the Mars and Moon JSON, `caveats.json`, and an `about.json` skeleton, with the pytest parity tests.
3. **Export images.** Write `export_images.py` for the previews (with the resolutions above), thumbnails and Moon sprites. Report sizes.
4. **Audit tests.** Port the audit, integrity, caveat and removed-section tests; they pass against the exported data.
5. **Landing.** The split "choose a world" screen, using the hallmark skill under the constraints above. Screenshots.
6. **Mars explorer, part 1:** rail, strip, and the shared `PanZoomViewport`, with keyboard support. Screenshots.
7. **Mars explorer, part 2:** tile overlay on canvas.
   - Predicted-safe tiles get a fill and outline in the active model's colour; predicted-unsafe tiles stay unfilled.
   - Predicted safe but actually unsafe: model-colour fill plus diagonal hatching.
   - Predicted unsafe but actually safe: dashed outline in the model colour.
   - Green and red are never used for predictions.
   - The ground-truth layer uses `--truth-safe` / `--truth-unsafe`; it's a separate toggle, off by default.
   - Also: CNN ↔ QCNN toggle, seed selector, tile inspector, and a legend explaining every encoding. Add `overlay.test.ts`.
8. **Mars explorer, part 3:** results panel with animated numbers and bars, per-seed breakdown, site-switch transitions, reduced motion, and the dataviz skill for charts. Screenshots.
9. **Moon explorer:** class rail and crop mosaic.
   - Crops are outlined in the model colour, dashed when the prediction is wrong.
   - Ground-truth layer is a toggle.
   - 3-way model toggle including the MLP; results panel.
   - Screenshots.
10. **About page:** method, the full caveat list, and data sources with licence and citation from `about.json`. Enable `sources.test.ts`, and list any `TO CONFIRM` items for you.
11. **Lab page ("Under the hood").**
    - Write `export_internals.py` and its pytest, plus `highlight.mjs` (Shiki, prebuild).
    - Build the six `/lab` sections and enable `lab.test.tsx`.
    - Report the `lab/` size against the ~3 MB budget.
    - Before building, recheck the training-curve inventory listed in Findings and report any gaps.
12. **Polish:**
    - Responsive layout: rail becomes a horizontal scroller on phones; image and results panels stack.
    - Accessibility: focus order, live-region announcements, table fallbacks for charts.
    - Performance: lazy-load data, and lazy-load the `/lab` route.
    - Lighthouse pass.
    - Deep-zoom pyramid for site_01 and site_05 if the budget allows. Report its size.
13. **Phase 2 globe (planned, not built now):**
    - Use `@react-three/fiber` + `drei`, lazy-loaded.
    - **Mars texture:** the USGS Viking MDIM 2.1 or MOLA-colourized equirectangular mosaic, downscaled to 4096×2048, as KTX2 plus a 2k WebP fallback. Downloading it needs your OK.
    - **Moon texture:** check whether `data/raw/moon/mosaic/moon_mosaic_big.tif` is a global equirectangular mosaic; otherwise use the LRO WAC global mosaic.
    - **Pins:** Mars pins come from `sites.json` centres. Clicking one sets `?site=` and the explorer animates to that site. No Moon pins.
    - Reduced motion disables auto-rotation and the camera fly-to.
    - Texture credits are added to `about.json`.

## Critical files

- **New:** `website/` (`PLAN.md`, `package.json`, `vite.config.ts`, `src/**`, `public/data/**`, `scripts/export_data.py`, `scripts/export_images.py`, `scripts/export_internals.py`, `scripts/highlight.mjs`, `scripts/tests/`).
- **Edited:** `CLAUDE.md`, adding one line: "website work follows website/PLAN.md".
- **Read only:**
  - `visualization_app/data_loading/{paths,mars_data,moon_data}.py`
  - `src/models/**`, `src/quantum/circuits.py` (imported by `export_internals.py`, never modified)
  - Mars `best_checkpoint.pt` (fold 0, seed 42) and Moon `*_seed_42.pt` checkpoints
  - `runs/pipeline_repair_audit/retraining_seedset/**`
  - `runs/controlled_evaluation/processed/{metadata.parquet,dataset_metadata.json}`
  - `data/raw/**`
  - `FINAL_READINESS_AUDIT.md`
- **Unchanged:** `visualization_app/`, `src/`. The existing generic `node_modules/` and `dist/` patterns in `.gitignore` already cover `website/`.

## Verification

1. The export scripts run (data and images on the app venv, internals on the root venv); the size report shows under 25 MB in total with `lab/` at about 3 MB, and the export pytest passes.
2. `npm test` passes: audit, integrity, caveat, removed-section, framing, overlay, reduced-motion and lab tests.
3. `npm run build` succeeds from a copy of `website/` in the scratchpad with no `runs/` or `data/raw/`.
4. `npm run dev` in the built-in browser at 1920×1080, 1366×768 and 375×812, in both worlds:
   - The overlay lines up with strip features.
   - Toggling models and seeds gives numbers that match the audit.
   - Reduced motion is respected.
   - The console has no errors.
   - `/lab` shows the circuit SVGs, the filter grid, feature maps and curves, and every permalink opens the right lines on GitHub once the commit is pushed.
5. The app's `pytest` suite still passes, and `git diff --stat` shows only `website/` and `CLAUDE.md`.
