import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join, relative, resolve } from 'node:path'

export const WEBSITE_ROOT = resolve(import.meta.dirname, '../..')
export const DATA_ROOT = join(WEBSITE_ROOT, 'public', 'data')

/** Read a committed export file (relative to public/data). */
export function loadData<T>(rel: string): T {
  return JSON.parse(readFileSync(join(DATA_ROOT, rel), 'utf8')) as T
}

/** All files below a directory, as paths relative to the website root. */
export function walk(dir: string): string[] {
  const out: string[] = []
  for (const name of readdirSync(dir)) {
    const full = join(dir, name)
    if (statSync(full).isDirectory()) out.push(...walk(full))
    else out.push(relative(WEBSITE_ROOT, full).replaceAll('\\', '/'))
  }
  return out
}
