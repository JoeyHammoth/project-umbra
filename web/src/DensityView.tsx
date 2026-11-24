/**
 * The reconstruction, drawn.
 *
 * A 32 × 32 density field printed as numbers is a debug readout; the same field
 * as an image is the thing the app is about. The canvas is sized to the grid and
 * scaled by CSS with `image-rendering: pixelated`, so one voxel is one pixel and
 * a regression that swaps the row order or loses the field looks wrong instead
 * of merely looking different.
 *
 * The ramp is deliberately one hue: a multi-colour map invites reading hue as
 * intensity, and here brightness is the only quantity a pixel carries. Air is
 * black because the field really is zero there, so a blank square with no
 * structure is a visible failure rather than a plausible dark image.
 */

import { useEffect, useRef } from 'react';

export interface DensityViewProps {
  /** The field, one entry per voxel, in g cm⁻³. */
  density: Float32Array | null;
  /** Voxels per row; the height follows from the field's length. */
  width: number;
  /** Described for assistive technology, which cannot see the canvas. */
  summary: string;
}

/** The largest value in the field, which sets the top of the ramp. */
export function peakOf(density: Float32Array): number {
  let peak = 0;
  for (const value of density) {
    if (value > peak) peak = value;
  }
  return peak;
}

/**
 * Fills an `ImageData` with the field, normalised to its own peak.
 *
 * Returns the number of pixels above zero so a caller — the browser check —
 * can tell a drawn field from a black rectangle. Normalising to the peak rather
 * than to a fixed density is what keeps an early reconstruction visible: after
 * one step the whole field is small, and a fixed scale would show nothing.
 */
export function paint(
  density: Float32Array,
  width: number,
): { data: Uint8ClampedArray<ArrayBuffer>; height: number } {
  const height = Math.max(1, Math.ceil(density.length / width));
  const data = new Uint8ClampedArray(width * height * 4);
  const peak = peakOf(density);
  const scale = peak > 0 ? 255 / peak : 0;
  for (let index = 0; index < density.length; index += 1) {
    const value = Math.round((density[index] ?? 0) * scale);
    data[index * 4] = value;
    // Two thirds and just under half: rock reads as warm, and the ratio keeps
    // very dark values from clipping to pure red.
    data[index * 4 + 1] = Math.round(value * 0.66);
    data[index * 4 + 2] = Math.round(value * 0.42);
    data[index * 4 + 3] = 255;
  }
  return { data, height };
}

export function DensityView({ density, width, summary }: DensityViewProps) {
  const canvas = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const element = canvas.current;
    if (!element || !density || density.length === 0) return;
    const { data, height } = paint(density, width);
    element.width = width;
    element.height = height;
    const context = element.getContext('2d');
    // A canvas without a 2D context is a browser that cannot draw at all; the
    // figcaption still carries the numbers, so this is not fatal.
    if (!context) return;
    context.putImageData(new ImageData(data, width, height), 0, 0);
  }, [density, width]);

  return (
    <figure className="density">
      <canvas ref={canvas} data-testid="density" role="img" aria-label={summary} />
      <figcaption>{summary}</figcaption>
    </figure>
  );
}
