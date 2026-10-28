import type { CSSProperties } from 'react'
import type { LabRotations } from '../../data/types'

interface Props {
  rotations: LabRotations
  caption: string
}

/**
 * The trained rotation angles of one QCNN head, one row per circuit stage, one cell per angle.
 * The shade shows the size of the angle on one neutral ramp (zero to the largest magnitude); the
 * signed value is printed in every cell, so the shade is never the only carrier of the number.
 */
export function RotationHeatmap({ rotations, caption }: Props) {
  const limit = Math.max(...rotations.value_range.map(Math.abs))
  return (
    <figure className="heatmap">
      <figcaption>{caption}</figcaption>
      <div className="heatmap-scroll">
      <table className="heatmap-table">
        <tbody>
          {rotations.layers.map((layer) => (
            <tr key={layer.name}>
              <th scope="row">{layer.name}</th>
              {layer.values[0].map((v, i) => {
                const t = Math.min(1, Math.abs(v) / limit)
                return (
                  <td
                    key={i}
                    data-strong={t > 0.54 || undefined}
                    style={{ '--t': t.toFixed(3) } as CSSProperties}
                    title={`${layer.labels[i]} gate ${i + 1}: ${v.toFixed(4)} rad`}
                  >
                    <span className="heatmap-gate">{layer.labels[i]}</span>
                    <span className="heatmap-value">{v.toFixed(2)}</span>
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
      </div>
      <div className="heatmap-legend" aria-hidden="true">
        <span>0 rad</span>
        <span className="heatmap-ramp" />
        <span>{limit.toFixed(2)} rad, either sign</span>
      </div>
    </figure>
  )
}
