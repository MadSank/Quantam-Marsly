// Fill the `html` field of every excerpt in public/data/lab/excerpts.json with Shiki output, and
// render each Moon excerpt's history note (Markdown from experiments/moon/README.md) with marked.
// Runs as `prebuild` (PLAN.md "scripts/highlight.mjs"), so no highlighter or Markdown parser ships
// to the browser. The note's `text` stays as exported; only `history_note.html` is added.
//
// The theme is Shiki's CSS-variables theme: each token gets `color: var(--shiki-token-…)`, and
// src/styles/lab.css maps those names onto the site's tokens. The file is rewritten only when an
// html field changes. img/manifest.json is written by export_images.py, so it is not touched here.
import { readFileSync, writeFileSync } from 'node:fs'
import { marked } from 'marked'
import { createCssVariablesTheme, createHighlighter } from 'shiki'

const path = new URL('../public/data/lab/excerpts.json', import.meta.url)
const file = JSON.parse(readFileSync(path, 'utf8'))

const theme = createCssVariablesTheme({ name: 'site', variablePrefix: '--shiki-', fontStyle: true })
const highlighter = await createHighlighter({ themes: [theme], langs: ['python'] })

let changed = 0
for (const e of file.excerpts) {
  const html = highlighter.codeToHtml(e.code, {
    lang: 'python',
    theme: 'site',
    // Line numbers match the file, so a reader can match the excerpt to the permalink.
    transformers: [
      {
        pre(node) {
          delete node.properties.style
          delete node.properties.tabindex
          node.properties.style = `--line-start:${e.start}`
        },
      },
    ],
  })
  if (html !== e.html) {
    e.html = html
    changed++
  }
  if (e.history_note) {
    // The note opens with its own "## Code history" heading; the page supplies the heading.
    const note = marked.parse(e.history_note.text.replace(/^## Code history\n+/, ''), { async: false })
    if (note !== e.history_note.html) {
      e.history_note.html = note
      changed++
    }
  }
}
highlighter.dispose()

if (changed) writeFileSync(path, JSON.stringify(file, null, 1) + '\n')
console.log(`highlight: ${file.excerpts.length} excerpts, ${changed} updated`)
