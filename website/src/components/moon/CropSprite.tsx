import type { MoonClassFile } from '../../data/types'
import { dataUrl } from '../../lib/dataUrl'

/** One test crop drawn from its class's sprite sheet at `size` CSS pixels. Decorative unless `alt` is given. */
export function CropSprite({ sprite, index, size, alt }: {
  sprite: NonNullable<MoonClassFile['sprite']>
  index: number
  size: number
  alt?: string
}) {
  const col = index % sprite.cols
  const row = Math.floor(index / sprite.cols)
  return (
    <span
      className="crop-sprite"
      role={alt ? 'img' : undefined}
      aria-label={alt}
      aria-hidden={alt ? undefined : true}
      style={{
        width: size,
        height: size,
        backgroundImage: `url("${dataUrl(sprite.src)}")`,
        backgroundSize: `${sprite.cols * size}px ${sprite.rows * size}px`,
        backgroundPosition: `${-col * size}px ${-row * size}px`,
      }}
    />
  )
}
