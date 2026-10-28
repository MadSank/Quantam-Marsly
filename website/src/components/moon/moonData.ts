// Moon definitions shared by the explorer and the globe entry (PLAN.md steps 9 and 9b).
import type { MoonClass, MoonModel } from '../../data/types'
import type { ChartModel } from '../results/MetricChart'
import type { MetricDef } from '../results/metrics'

/** Fixed display order: CNN, then the hybrid, then the matched MLP control. Never sorted by value. */
export const MOON_MODELS: ReadonlyArray<ChartModel & { key: MoonModel; url: string }> = [
  { key: 'cnn', label: 'CNN', short: 'CNN', data: 'cnn', url: 'cnn' },
  { key: 'hybrid_qcnn', label: 'Hybrid CNN–QCNN', short: 'Hybrid', data: 'qcnn', url: 'qcnn' },
  { key: 'frozen_embedding_mlp', label: 'Frozen-embedding MLP', short: 'MLP', data: 'mlp', url: 'mlp' },
]

export const modelFromUrl = (v: string | null) => MOON_MODELS.find((m) => m.url === v) ?? MOON_MODELS[0]

// Balanced accuracy and macro-F1 lead; accuracy follows with the majority baseline (the test set is 68.4% "none").
export const MOON_METRICS: MetricDef[] = [
  { key: 'balanced_accuracy', label: 'Balanced accuracy', direction: 'higher is better', percent: false, unit: '', digits: 3, max: 1 },
  { key: 'macro_f1', label: 'Macro-F1', direction: 'higher is better', percent: false, unit: '', digits: 3, max: 1 },
  { key: 'accuracy', label: 'Accuracy', direction: 'higher is better', percent: false, unit: '', digits: 3, max: 1 },
]

/** Class names as shown; the data keys stay visible in the URL and the inspector. */
export const CLASS_NAMES: Record<MoonClass, string> = {
  ejecta: 'Ejecta',
  oldcrater: 'Old crater',
  none: 'None',
}

/** Short tags for the "Predicted class" labels on crops. */
export const CLASS_TAGS: Record<MoonClass, string> = {
  ejecta: 'ejecta',
  oldcrater: 'old crater',
  none: 'none',
}

export const MOON_CLASSES: MoonClass[] = ['ejecta', 'oldcrater', 'none']

/** Shown on the Moon globe: why the crops are not placed on it (PLAN.md step 9b). */
export const LOCATION_NOTE = 'LROCNet crops carry no verified source locations, so they are not placed on the globe.'
