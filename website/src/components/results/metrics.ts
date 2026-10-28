// Metric definitions for the Mars results panel (PLAN.md "Rules for presenting the science").
import type { MarsModel, Metric, Regime } from '../../data/types'

export type Direction = 'lower is better' | 'higher is better' | null

export interface MetricDef {
  key: string
  label: string
  direction: Direction
  /** Fractions shown as percentages on a fixed 0–100% scale. */
  percent: boolean
  unit: string
  digits: number
  /** Shown under the label when the metric has no better direction. */
  note?: string
}

/** Fixed display order: CNN first, then the hybrid. Never sorted by value. */
export const MARS_MODELS: ReadonlyArray<{ key: MarsModel; label: string; short: string; data: 'cnn' | 'qcnn' }> = [
  { key: 'CNN', label: 'CNN', short: 'CNN', data: 'cnn' },
  { key: 'CNN_QCNN', label: 'Hybrid CNN–QCNN', short: 'Hybrid', data: 'qcnn' },
]

export const REGRESSION: MetricDef[] = [
  { key: 'RMSE_mean_slope_5m', label: 'Mean slope RMSE', direction: 'lower is better', percent: false, unit: '°', digits: 1 },
  { key: 'RMSE_roughness_m', label: 'Roughness RMSE', direction: 'lower is better', percent: false, unit: ' m', digits: 2 },
  { key: 'RMSE_hazard_fraction', label: 'Hazard fraction RMSE', direction: 'lower is better', percent: false, unit: '', digits: 3 },
]

// Recall, false-positive rate and coverage always appear together.
export const DECISIONS: MetricDef[] = [
  { key: 'safe_recall', label: 'Safe recall', direction: 'higher is better', percent: true, unit: '%', digits: 1 },
  { key: 'false_positive_rate', label: 'False-positive rate', direction: 'lower is better', percent: true, unit: '%', digits: 1 },
  {
    key: 'predicted_safe_fraction',
    label: 'Coverage, share predicted safe',
    direction: null,
    percent: true,
    unit: '%',
    digits: 1,
    note: 'read with recall and false-positive rate',
  },
]

export const REGIMES: ReadonlyArray<{ key: Regime; label: string; note: string }> = [
  { key: 'fixed', label: 'Fixed threshold', note: 'Safe when predicted slope < 8° and hazard < 0.05.' },
  {
    key: 'validation_calibrated',
    label: 'Validation-calibrated',
    note: 'Risk threshold set on the validation site so its false-positive rate is at most 5%, then applied to the test site.',
  },
]

export const UNDEFINED_TEXT = 'undefined (no predicted-safe tiles)'

/** Display value; null (NaN in the source) is spelled out rather than drawn. */
export function formatMetric(def: MetricDef, v: Metric): string {
  if (v === null || !Number.isFinite(v)) return UNDEFINED_TEXT
  return def.percent ? `${(v * 100).toFixed(def.digits)}%` : `${v.toFixed(def.digits)}${def.unit}`
}

/** Shared scale end for one metric's small multiple: 100% for rates, else a rounded-up max. */
export function scaleMax(def: MetricDef, values: Metric[]): number {
  if (def.percent) return 1
  const finite = values.filter((v): v is number => v !== null && Number.isFinite(v))
  const max = Math.max(...finite, 0)
  if (max === 0) return 1
  const pow = 10 ** Math.floor(Math.log10(max))
  return Math.ceil((max * 1.05) / (pow / 2)) * (pow / 2)
}
