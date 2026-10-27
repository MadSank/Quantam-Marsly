# Mars | Moon Explorer

A standalone website for exploring the CNN vs hybrid CNN–QCNN study. It is built with Vite, React and TypeScript, and follows [PLAN.md](PLAN.md).

The site builds only from the committed files in `public/data/`. It never reads `runs/` or `data/raw/`.

## Develop

```bash
npm install
npm run dev      # http://localhost:5173
npm test         # Vitest (jsdom)
npm run lint     # oxlint
npm run build    # type-check + production build to dist/
```

To run one test file, use `npx vitest run src/test/scaffold.test.tsx`. Add `-t "<name>"` to run a single test by name.

## Refresh the exported data

These scripts are added in build steps 2–3. They need the gitignored experiment artifacts on disk and the Streamlit app's venv. Run them from the repository root:

```bash
visualization_app/.venv/Scripts/python.exe website/scripts/export_data.py
visualization_app/.venv/Scripts/python.exe website/scripts/export_images.py
visualization_app/.venv/Scripts/python.exe -m pytest website/scripts/tests
```

The pytest suite checks the committed JSON against the source CSV, parquet and `results.json` files. It is skipped when those artifacts are absent.

It also checks the exported images:
- No raw rasters are shipped.
- `public/data` stays within the 25 MB budget.
- Image sizes match the JSON.
- Each Mars tile's stated position lines up with the preview: per-tile brightness is correlated against the model's input patches.

`export_images.py` decodes the JP2 strips at reduced resolution (about 2.5 minutes in total). The two scripts can run in either order, because `export_data.py` keeps the image fields that `export_images.py` writes.

`export_data.py` also refuses to write a caveat unless its text is a verbatim quote of `FINAL_READINESS_AUDIT.md` or `visualization_app/README.md`.

## Design checks (Hallmark)

The pages follow Hallmark's Vantage theme on the plan's token names. Each page stylesheet starts with its Hallmark stamp and any waivers (`src/styles/landing.css`).

To lint a page against a running dev server:

```bash
node <hallmark-skill>/scripts/sloplint.mjs website/src/styles/landing.css website/src/pages/Landing.tsx --genre modern-minimal --render --url http://localhost:5173/
```

The render tier needs `puppeteer-core` (pass its folder in `HALLMARK_PUPPETEER`) and a local Chrome. On this Windows machine the installed skill's files have CRLF line endings, which `theme.mjs` cannot parse, so run it from an LF-normalised copy of the skill.

Review screenshots for each step are in `screenshots/`.

## Layout

- `src/styles/tokens.css`: semantic design tokens. The neutral shell is `:root`, and each world overrides them under `[data-world="mars"|"moon"]`. Components read only token names.
- `src/theme/`: `WorldLayout` applies a world's tokens; `useWorld()` reads the current world.
- Routing uses a hash router (`/#/mars`), so the built `dist/` works on any static host.
