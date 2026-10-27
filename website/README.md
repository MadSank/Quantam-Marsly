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
```

## Layout

- `src/styles/tokens.css`: semantic design tokens. The neutral shell is `:root`, and each world overrides them under `[data-world="mars"|"moon"]`. Components read only token names.
- `src/theme/`: `WorldLayout` applies a world's tokens; `useWorld()` reads the current world.
- Routing uses a hash router (`/#/mars`), so the built `dist/` works on any static host.
