/**
 * The reconstruction, drawn.
 *
 * A 32 × 32 density field printed as numbers is a debug readout; the same field
 * as an image is the thing the app is about. The canvas is sized to the grid and
 * scaled by CSS with `image-rendering: pixelated`, so one voxel is one pixel and
 * a regression that swaps the row order or loses the field looks wrong instead
 * of merely looking different.
 *
 * Air is black because the field really is zero there, so a blank square with no
 * structure is a visible failure rather than a plausible dark image. Everything
 * above zero goes through the heatmap below: monotone in luminance, so brightness
 * still tracks density, but spread across hue, because a 20-step reconstruction
 * spans more than a decade and a single hue rounds every value below a tenth of
 * the peak to the same near-black.
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

type Stop = readonly [t: number, r: number, g: number, b: number];

/**
 * A five-stop heatmap: indigo → purple → crimson → orange → pale yellow.
 *
 * The first stop is deliberately not black. Air is painted black below, and a
 * ramp starting at zero would make the faintest real density indistinguishable
 * from empty space — which is also what lets the browser check count `r > 0` as
 * "this voxel carried a value".
 */
const RAMP: readonly Stop[] = [
  [0, 16, 8, 40],
  [0.25, 87, 16, 110],
  [0.5, 188, 55, 84],
  [0.75, 249, 142, 8],
  [1, 252, 255, 164],
];

/** Colour at fraction `t` of the peak, linearly interpolated between stops. */
export function rampAt(t: number): [number, number, number] {
  const x = t < 0 ? 0 : t > 1 ? 1 : t;
  for (let stop = 1; stop < RAMP.length; stop += 1) {
    const [t1, r1, g1, b1] = RAMP[stop]!;
    if (x > t1) continue;
    const [t0, r0, g0, b0] = RAMP[stop - 1]!;
    const f = t1 === t0 ? 0 : (x - t0) / (t1 - t0);
    return [
      Math.round(r0 + (r1 - r0) * f),
      Math.round(g0 + (g1 - g0) * f),
      Math.round(b0 + (b1 - b0) * f),
    ];
  }
  const [, r, g, b] = RAMP[RAMP.length - 1]!;
  return [r, g, b];
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
  const scale = peak > 0 ? 1 / peak : 0;
  for (let index = 0; index < density.length; index += 1) {
    const value = density[index] ?? 0;
    // Opaque everywhere, black where the field is zero: a hole in the rock is
    // black structure, not a window onto the page behind the canvas.
    data[index * 4 + 3] = 255;
    if (!(value > 0) || scale === 0) continue;
    const [r, g, b] = rampAt(value * scale);
    data[index * 4] = r;
    data[index * 4 + 1] = g;
    data[index * 4 + 2] = b;
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
