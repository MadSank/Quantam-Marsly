// Exported data vs the headline values in FINAL_READINESS_AUDIT.md (transcribed here).
// Each tolerance is half a unit in the last digit the audit reports.
import { describe, expect, it } from 'vitest'
import type { MarsHeadline, MarsSitesFile, MoonSummary } from '../data/types'
import { loadData } from './data'

const headline = loadData<MarsHeadline>('mars/headline.json')
const moon = loadData<MoonSummary>('moon/summary.json')

function close(actual: number | null, expected: number, tol: number) {
  expect(actual).not.toBeNull()
  expect(Math.abs((actual as number) - expected)).toBeLessThanOrEqual(tol)
}

describe('Mars regression errors (audit §1, 3 dp)', () => {
  const table: Array<[string, number, number, number, number]> = [
    // metric suffix, CNN RMSE, CNN MAE, QCNN RMSE, QCNN MAE
    ['mean_slope_5m', 10.25, 8.712, 8.19, 6.837],
    ['roughness_m', 1.566, 1.172, 1.364, 1.027],
    ['hazard_fraction', 0.453, 0.392, 0.377, 0.314],
  ]
  it.each(table)('%s', (target, cnnRmse, cnnMae, qRmse, qMae) => {
    const f = headline.fixed
    close(f.CNN[`RMSE_${target}`].mean, cnnRmse, 0.0005)
    close(f.CNN[`MAE_${target}`].mean, cnnMae, 0.0005)
    close(f.CNN_QCNN[`RMSE_${target}`].mean, qRmse, 0.0005)
    close(f.CNN_QCNN[`MAE_${target}`].mean, qMae, 0.0005)
  })
})

describe('Mars safe-proxy decisions (audit §1, percent to 2 dp)', () => {
  const pct = (v: number | null) => (v as number) * 100
  const cases: Array<[keyof MarsHeadline & ('fixed' | 'validation_calibrated'), 'CNN' | 'CNN_QCNN', number, number, number]> = [
    ['fixed', 'CNN', 5.56, 5.27, 5.67],
    ['fixed', 'CNN_QCNN', 2.77, 2.78, 3.02],
    ['validation_calibrated', 'CNN', 43.33, 39.65, 41.35],
    ['validation_calibrated', 'CNN_QCNN', 30.62, 33.94, 33.86],
  ]
  it.each(cases)('%s %s recall / FPR / coverage', (regime, model, recall, fpr, coverage) => {
    const m = headline[regime][model]
    close(pct(m.safe_recall.mean), recall, 0.005)
    close(pct(m.false_positive_rate.mean), fpr, 0.005)
    close(pct(m.predicted_safe_fraction.mean), coverage, 0.005)
    expect(m.safe_recall.n_sites).toBe(5)
  })
})

describe('Moon metrics (audit §1, mean ± sample SD, 3 dp)', () => {
  const table: Array<[string, number, number, number, number, number, number]> = [
    ['cnn', 0.53, 0.035, 0.558, 0.053, 0.465, 0.024],
    ['hybrid_qcnn', 0.529, 0.039, 0.608, 0.006, 0.497, 0.026],
    ['frozen_embedding_mlp', 0.451, 0.037, 0.587, 0.021, 0.44, 0.028],
    ['majority_train_baseline', 0.684, 0, 0.333, 0, 0.271, 0],
  ]
  it.each(table)('%s', (model, acc, accSd, bal, balSd, f1, f1Sd) => {
    const m = moon.metrics[model as keyof MoonSummary['metrics']]
    close(m.accuracy.mean, acc, 0.0005)
    close(m.accuracy.sd, accSd, 0.0005)
    close(m.balanced_accuracy.mean, bal, 0.0005)
    close(m.balanced_accuracy.sd, balSd, 0.0005)
    close(m.macro_f1.mean, f1, 0.0005)
    close(m.macro_f1.sd, f1Sd, 0.0005)
  })
})

describe('counts', () => {
  it('Mars: 15,233 patches over five sites', () => {
    const { sites } = loadData<MarsSitesFile>('mars/sites.json')
    expect(sites).toHaveLength(5)
    expect(sites.reduce((n, s) => n + s.n_tiles, 0)).toBe(15233)
  })

  it('Moon: 3,608 / 613 / 779 images', () => {
    expect(moon.split_sizes).toEqual({ train: 3608, val: 613, test: 779 })
  })
})

describe('parameter counts', () => {
  it('Mars: CNN 11,242,563 trainable; hybrid 4,152 trainable', () => {
    expect(headline.parameter_counts.CNN.total_trainable).toBe(11242563)
    expect(headline.parameter_counts.CNN_QCNN.total_trainable).toBe(4152)
  })

  it('Moon: CNN 59,347 during training (195 after freeze); heads 568 / 547', () => {
    const pc = moon.parameter_counts
    expect(pc.cnn.trainable_during_training).toBe(59347)
    expect(pc.cnn.trainable_recorded_post_freeze).toBe(195)
    expect(pc.hybrid_head).toBe(568)
    expect(pc.mlp_head).toBe(547)
  })
})
