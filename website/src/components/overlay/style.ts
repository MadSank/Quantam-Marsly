// Tile overlay encodings (PLAN.md step 7). Prediction layers draw only in the active model's
// colour; the two disagreement cases differ from agreement by pattern (hatching, dashes), not
// by hue. Truth colours belong to the separate ground-truth layer only.

/** The two Mars models, always in this order: CNN first, then the hybrid. */
export const MODELS = [
  { key: 'cnn', label: 'CNN', token: '--model-cnn' },
  { key: 'qcnn', label: 'Hybrid CNN–QCNN', token: '--model-qcnn' },
] as const

export type ModelKey = (typeof MODELS)[number]['key']

export const modelLabel = (key: ModelKey) => MODELS.find((m) => m.key === key)!.label

/** A tile's prediction against the safe-proxy label. */
export type Outcome = 'safe-agree' | 'safe-wrong' | 'unsafe-wrong' | 'unsafe-agree'

export type TileColour = 'model' | 'truth-safe' | 'truth-unsafe'

export interface TileStyle {
  colour: TileColour | null
  fill: boolean
  hatch: boolean
  outline: 'solid' | 'dashed' | 'none'
}

export function outcome(predSafe: number, trueSafe: number): Outcome {
  if (predSafe) return trueSafe ? 'safe-agree' : 'safe-wrong'
  return trueSafe ? 'unsafe-wrong' : 'unsafe-agree'
}

const PREDICTION: Record<Outcome, TileStyle> = {
  // Predicted safe, proxy says safe: model-colour fill and outline.
  'safe-agree': { colour: 'model', fill: true, hatch: false, outline: 'solid' },
  // Predicted safe, proxy says unsafe: the same fill, hatched.
  'safe-wrong': { colour: 'model', fill: true, hatch: true, outline: 'solid' },
  // Predicted unsafe, proxy says safe: dashed outline, no fill.
  'unsafe-wrong': { colour: 'model', fill: false, hatch: false, outline: 'dashed' },
  // Predicted unsafe, proxy says unsafe: left unmarked.
  'unsafe-agree': { colour: null, fill: false, hatch: false, outline: 'none' },
}

export function predictionStyle(o: Outcome): TileStyle {
  return PREDICTION[o]
}

/**
 * The ground-truth layer, drawn only when toggled on: a corner marker in the truth colours.
 * A marker, not an outline, so it never reads as one of the prediction encodings.
 */
export function truthStyle(trueSafe: number): TileStyle & { marker: 'corner' } {
  return { colour: trueSafe ? 'truth-safe' : 'truth-unsafe', fill: false, hatch: false, outline: 'none', marker: 'corner' }
}

/** Legend rows in a fixed order, matching the four outcomes. */
export const OUTCOME_ROWS: Array<{ outcome: Outcome; label: string }> = [
  { outcome: 'safe-agree', label: 'Predicted safe, proxy safe' },
  { outcome: 'safe-wrong', label: 'Predicted safe, proxy unsafe' },
  { outcome: 'unsafe-wrong', label: 'Predicted unsafe, proxy safe' },
  { outcome: 'unsafe-agree', label: 'Predicted unsafe, proxy unsafe' },
]
