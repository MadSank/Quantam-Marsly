// Types for the JSON written by website/scripts/export_data.py and export_images.py.
// Schema: website/PLAN.md "JSON data schema". Missing/undefined metrics are null.

export type Metric = number | null

export interface Stamp {
  source: string[]
  generated_at: string
}

export interface ImageRef {
  src: string
  width: number
  height: number
}

// ---------------------------------------------------------------- Mars

export type MarsModel = 'CNN' | 'CNN_QCNN'
export type Regime = 'fixed' | 'validation_calibrated'

export interface MarsSite {
  site_id: string
  name: string
  fold: number
  ortho_product_id: string
  dtm_product_id: string
  center: { lat: number; lon_e: number }
  /** Convex hull of the tile corners, [lat, lon_e] pairs (globe footprint, step 8b). */
  footprint: Array<[number, number]>
  /** [lat, lon_e] of the native grid's corners, for draping the preview on the globe. */
  image_corners: Record<'top_left' | 'top_right' | 'bottom_left', [number, number]>
  n_tiles: number
  n_true_safe: number
  native_resolution_m: number
  native_height_px: number
  native_width_px: number
  tile_size_native_px: number
  preview: (ImageRef & { scale_from_native: { x: number; y: number } }) | null
  thumb: ImageRef | null
}

export interface MarsSitesFile extends Stamp {
  sites: MarsSite[]
}

export interface AggregateCell {
  mean: Metric
  sd: Metric
  n_sites: number
}

export interface MarsHeadline extends Stamp {
  models: MarsModel[]
  parameter_counts: Record<MarsModel, { total_trainable: number; total_frozen: number; head_trainable: number }>
  fixed: Record<MarsModel, Record<string, AggregateCell>>
  validation_calibrated: Record<MarsModel, Record<string, AggregateCell>>
}

export interface MarsSeedRow {
  regime: Regime
  model: MarsModel
  fold: number
  seed: number
  [metric: string]: Metric | string
}

export interface MarsSiteFile extends Stamp {
  site_id: string
  models: MarsModel[]
  mean: Record<Regime, Record<MarsModel, Record<string, Metric>>>
  per_seed: MarsSeedRow[]
}

export interface TileTargets {
  slope: Metric[]
  rough: Metric[]
  hazard: Metric[]
  safe: number[]
}

export interface MarsTilesFile extends Stamp {
  site_id: string
  fold: number
  seeds: number[]
  size_native: number
  tile_id: string[]
  row: number[]
  col: number[]
  true: TileTargets
  pred: Record<'cnn' | 'qcnn', Record<string, TileTargets>>
}

/** mars/contours/<site_id>.json, from scripts/export_contours.py; page decoration only. */
export interface MarsContoursFile extends Stamp {
  site_id: string
  dtm_product_id: string
  width: number
  height: number
  interval_m: number
  levels: Array<{ elevation_m: number; index: boolean; d: string }>
}

// ---------------------------------------------------------------- Moon

export type MoonModel = 'cnn' | 'hybrid_qcnn' | 'frozen_embedding_mlp'
export type MoonClass = 'ejecta' | 'oldcrater' | 'none'

export type MoonMetricKey = 'accuracy' | 'balanced_accuracy' | 'macro_f1'

export interface MoonSummary extends Stamp {
  /** Run folder the Moon results come from (the deterministic 10-seed run). */
  run: string
  seeds: number[]
  classes: MoonClass[]
  models: MoonModel[]
  split_counts: Record<'train' | 'val' | 'test', Record<MoonClass, number>>
  split_sizes: Record<'train' | 'val' | 'test', number>
  metrics: Record<MoonModel | 'majority_train_baseline', Record<string, { mean: number; sd: number }>>
  /** Per-seed min and max of each metric, per model. */
  ranges: Record<MoonModel | 'majority_train_baseline', Record<MoonMetricKey, { min: number; max: number }>>
  per_seed: Array<Record<string, number | string>>
  parameter_counts: {
    cnn: { trainable_during_training: number; trainable_recorded_post_freeze: number; note: string }
    hybrid_head: number
    mlp_head: number
  }
  timing: { comparable: false; caveat_id: string; milliseconds_per_image: Record<MoonModel, Record<string, number>> }
}

export interface MoonPrediction {
  label: MoonClass
  conf: number
  probs: [number, number, number]
}

export interface MoonCrop {
  sample_id: string
  img: number
  truth: MoonClass
  pred: Record<MoonModel, Record<string, MoonPrediction>>
}

export interface MoonClassFile extends Stamp {
  class: MoonClass
  classes: MoonClass[]
  seeds: number[]
  sprite: (ImageRef & { cell: number; cols: number; rows: number; count: number }) | null
  thumb: ImageRef | null
  crops: MoonCrop[]
}

// ---------------------------------------------------------------- shared

export interface Caveat {
  id: string
  world: 'mars' | 'moon' | 'both'
  title: string
  text: string
  source: string
  quote_parts: number
}

export interface CaveatsFile extends Stamp {
  caveats: Caveat[]
}

// ---------------------------------------------------------------- About

export interface AboutMethod {
  id: string
  world: 'mars' | 'moon' | 'both'
  title: string
  paragraphs: string[]
}

export interface AboutSource {
  name: string
  provider: string
  licence: string
  credit: string
  citation: string
  url: string
  licence_url: string
  used_for: string
}

export interface AboutFile extends Stamp {
  method: AboutMethod[]
  sources: AboutSource[]
}

// ---------------------------------------------------------------- Lab (export_internals.py)

export interface LabParams extends Stamp {
  mars: {
    cnn: { total: number; trainable: number; head: number }
    hybrid: { total: number; trainable: number; frozen_backbone: number; bottleneck: number; qcnn_head: number }
  }
  moon: {
    cnn: { trainable_during_training: number; trainable_post_freeze: number }
    hybrid_head: number
    mlp_head: number
    hybrid_bottleneck: number
    hybrid_qcnn: number
  }
}

export interface LabFeatures extends Stamp {
  tile_id: string
  site_id: string
  fold: number
  seed: number
  input: { png: string; channel: string }
  input_channels: string[]
  stages: Array<{ name: string; shape: number[]; channels_shown: number; png: string }>
  bottleneck_angles: number[]
}

export interface LabRotations {
  world: 'mars' | 'moon'
  source: string
  layers: Array<{ name: string; kind: string; shape: number[]; values: number[][]; labels: string[] }>
  readout: { scale: number[]; bias: number[] }
  value_range: [number, number]
  unit: string
}

export interface LabProjection {
  world: 'mars' | 'moon'
  in_dim: number
  out_dim: number
  encoding: string
  notes: string[]
}

export interface LabCurve {
  epoch: number[]
  train_loss: number[]
  val_loss?: number[]
  val_macro_f1?: number[]
}

export interface LabCurves extends Stamp {
  epoch_base: number
  mars: Record<string, Record<'cnn' | 'qcnn', LabCurve>>
  moon: Record<string, Record<MoonModel, LabCurve>>
}

export interface LabExcerpt {
  id: string
  title: string
  world: 'mars' | 'moon' | 'both'
  stage: string | null
  path: string
  symbol: string
  start: number
  end: number
  note: string
  permalink: string
  code: string
  html: string | null
  history_note?: { text: string; permalink: string; html?: string }
}

export interface LabExcerpts extends Stamp {
  commit: string
  repo: string
  excerpts: LabExcerpt[]
}
