/**
 * The reconstruction and the scan, drawn.
 *
 * A 32 × 32 density field printed as numbers is a debug readout; the same field
 * as an image is the thing the app is about. The canvas is sized to the grid and
 * scaled by CSS with `image-rendering: pixelated`, so one voxel is one pixel and
 * a regression that swaps the row order or loses the field looks wrong instead
 * of merely looking different.
 *
 * The two panels answer different questions and are deliberately not on one
 * colour scale. Density is a positive quantity, drawn from indigo to pale yellow
 * across the field's own positive range — windowed, not 0-to-peak, because a
 * reconstruction is mostly rock near 2.65 and a zero-to-peak ramp paints it as a
 * flat slab — and the legend prints both ends of that window, so the contrast on
 * show is stated rather than implied. `Z` is a significance: fixed at 0–5σ rather
 * than normalised to its own peak, because a ramp that stretched a null field to
 * full brightness would make noise look like a detection.
 */

import { useEffect, useRef, useState } from 'react';
import type { PointerEvent as ReactPointerEvent } from 'react';

export interface DensityViewProps {
  /** The field, one entry per voxel, in g cm⁻³. */
  density: Float32Array | null;
  /** The scan's `Z` at every voxel; null before a scan has landed. */
  z?: Float32Array | null;
  /** Voxels per row; the height follows from the field's length. */
  width: number;
  /** Described for assistive technology, which cannot see the canvas. */
  summary: string;
  /** The same for the `Z` panel, whose units are sigma rather than g cm⁻³. */
  scanSummary?: string;
}

/** Full scale of the `Z` panel, in sigma. */
export const Z_CEILING = 5;

/**
 * The window a field is drawn on: its smallest positive value and its peak.
 *
 * Zero is excluded from the low end because air is not part of the rock's range:
 * including it would undo the windowing for any field that has a hole in it.
 */
export function rangeOf(field: Float32Array): { lo: number; hi: number } {
  let lo = Number.POSITIVE_INFINITY;
  let hi = 0;
  for (const value of field) {
    if (!(value > 0)) continue;
    if (value < lo) lo = value;
    if (value > hi) hi = value;
  }
  return hi > 0 ? { lo: Number.isFinite(lo) ? lo : 0, hi } : { lo: 0, hi: 0 };
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
 * A five-stop heatmap for density: indigo → purple → crimson → orange → pale
 * yellow.
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

/**
 * A four-stop map for significance: near-black → deep blue → cyan → white.
 *
 * Hue is not doing arithmetic here; the panel is a lookup of `Z` against a fixed
 * ceiling, so the eye reads "how bright" and the legend supplies the number.
 */
const SIGMA_RAMP: readonly Stop[] = [
  [0, 12, 18, 38],
  [0.35, 33, 78, 138],
  [0.65, 66, 168, 190],
  [1, 236, 246, 240],
];

/** Colour at fraction `t` of `ramp`, linearly interpolated between stops. */
function colourAt(ramp: readonly Stop[], t: number): [number, number, number] {
  const x = t < 0 ? 0 : t > 1 ? 1 : t;
  for (let stop = 1; stop < ramp.length; stop += 1) {
    const [t1, r1, g1, b1] = ramp[stop]!;
    if (x > t1) continue;
    const [t0, r0, g0, b0] = ramp[stop - 1]!;
    const f = t1 === t0 ? 0 : (x - t0) / (t1 - t0);
    return [
      Math.round(r0 + (r1 - r0) * f),
      Math.round(g0 + (g1 - g0) * f),
      Math.round(b0 + (b1 - b0) * f),
    ];
  }
  const [, r, g, b] = ramp[ramp.length - 1]!;
  return [r, g, b];
}

/** Colour at fraction `t` of the density peak. */
export function rampAt(t: number): [number, number, number] {
  return colourAt(RAMP, t);
}

/** The ramp as a CSS gradient, so the legend and the pixels cannot drift apart. */
function legendOf(ramp: readonly Stop[]): string {
  return `linear-gradient(90deg, ${ramp
    .map(([t, r, g, b]) => `rgb(${r} ${g} ${b}) ${Math.round(t * 100)}%`)
    .join(', ')})`;
}

const LEGEND = legendOf(RAMP);
const SIGMA_LEGEND = legendOf(SIGMA_RAMP);

/**
 * Fills an `ImageData` with the density, normalised to its own peak.
 *
 * Returns the height so the caller can size the canvas. Normalising to the peak
 * rather than to a fixed density is what keeps an early reconstruction visible:
 * after one step the whole field is small, and a fixed scale would show nothing.
 */
export function paint(
  density: Float32Array,
  width: number,
): { data: Uint8ClampedArray<ArrayBuffer>; height: number } {
  const height = Math.max(1, Math.ceil(density.length / width));
  const data = new Uint8ClampedArray(width * height * 4);
  const { lo, hi } = rangeOf(density);
  const span = hi - lo;
  for (let index = 0; index < density.length; index += 1) {
    const value = density[index] ?? 0;
    // Opaque everywhere, black where the field is zero: a hole in the rock is
    // black structure, not a window onto the page behind the canvas.
    data[index * 4 + 3] = 255;
    if (!(value > 0) || hi <= 0) continue;
    // A field with no spread is drawn mid-ramp: any single colour would imply a
    // position in a range that does not exist.
    const [r, g, b] = rampAt(span > 0 ? (value - lo) / span : 0.5);
    data[index * 4] = r;
    data[index * 4 + 1] = g;
    data[index * 4 + 2] = b;
  }
  return { data, height };
}

/**
 * Fills an `ImageData` with `Z`, against the fixed ceiling.
 *
 * Nothing is normalised: a null field must stay dark, and a `Z` of 1.66 must look
 * like 1.66 of 5 rather than like the best thing the scan found.
 */
export function paintScan(
  z: Float32Array,
  width: number,
  ceiling = Z_CEILING,
): { data: Uint8ClampedArray<ArrayBuffer>; height: number } {
  const height = Math.max(1, Math.ceil(z.length / width));
  const data = new Uint8ClampedArray(width * height * 4);
  for (let index = 0; index < z.length; index += 1) {
    const value = Math.max(0, z[index] ?? 0) / ceiling;
    const [r, g, b] = colourAt(SIGMA_RAMP, value);
    data[index * 4] = r;
    data[index * 4 + 1] = g;
    data[index * 4 + 2] = b;
    data[index * 4 + 3] = 255;
  }
  return { data, height };
}

/** Draws `painted` into `canvas`, sizing it to the field. */
function draw(
  canvas: HTMLCanvasElement | null,
  painted: { data: Uint8ClampedArray<ArrayBuffer>; height: number },
  width: number,
): void {
  if (!canvas) return;
  const { data, height } = painted;
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d');
  // A canvas without a 2D context is a browser that cannot draw at all; the
  // figcaption still carries the numbers, so this is not fatal.
  if (!context) return;
  context.putImageData(new ImageData(data, width, height), 0, 0);
}

export function DensityView({
  density,
  z = null,
  width,
  summary,
  scanSummary,
}: DensityViewProps) {
  const densityCanvas = useRef<HTMLCanvasElement | null>(null);
  const scanCanvas = useRef<HTMLCanvasElement | null>(null);
  const [hover, setHover] = useState<{
    x: number;
    y: number;
    density: number;
    z: number | null;
  } | null>(null);
  const height = density ? Math.max(1, Math.ceil(density.length / width)) : 1;

  useEffect(() => {
    if (!density || density.length === 0) return;
    draw(densityCanvas.current, paint(density, width), width);
  }, [density, width]);

  useEffect(() => {
    if (!z || z.length === 0) return;
    draw(scanCanvas.current, paintScan(z, width), width);
  }, [z, width]);

  /**
   * Voxel under the pointer, shared by both panels so the readout agrees with
   * whichever image is hovered.
   *
   * The canvas is scaled by CSS, so the pointer has to come back through the
   * element's own box: reading `offsetX` would report CSS pixels, and a voxel
   * would be whichever one happened to sit under them.
   */
  const track = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    if (!density) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const x = Math.floor(((event.clientX - rect.left) / rect.width) * width);
    const y = Math.floor(((event.clientY - rect.top) / rect.height) * height);
    if (x < 0 || y < 0 || x >= width || y >= height) {
      setHover(null);
      return;
    }
    const index = y * width + x;
    setHover({
      x,
      y,
      density: density[index] ?? 0,
      z: z ? (z[index] ?? 0) : null,
    });
  };

  return (
    <div className="views">
      <figure className="density">
        <canvas
          ref={densityCanvas}
          data-testid="density"
          role="img"
          aria-label={summary}
          onPointerMove={track}
          onPointerLeave={() => setHover(null)}
        />
        <div className="density-legend" aria-hidden="true">
          <span>{rangeOf(density ?? new Float32Array()).lo.toFixed(2)}</span>
          <span className="density-ramp" style={{ background: LEGEND }} />
          <span>{rangeOf(density ?? new Float32Array()).hi.toFixed(2)} g cm⁻³</span>
        </div>
        <figcaption>
          {summary}
          {hover ? (
            <span className="density-readout">
              {' '}
              · voxel ({hover.x}, {hover.y}) ρ = {hover.density.toFixed(3)} g cm⁻³
            </span>
          ) : null}
        </figcaption>
      </figure>

      {z ? (
        <figure className="density">
          <canvas
            ref={scanCanvas}
            data-testid="scan"
            role="img"
            aria-label={scanSummary ?? `Z score, ${Z_CEILING} sigma full scale`}
            onPointerMove={track}
            onPointerLeave={() => setHover(null)}
          />
          <div className="density-legend" aria-hidden="true">
            <span>0</span>
            <span className="density-ramp" style={{ background: SIGMA_LEGEND }} />
            <span>{Z_CEILING}σ</span>
          </div>
          <figcaption>
            {scanSummary ?? `Z score, ${Z_CEILING}σ full scale.`}
            {hover?.z !== null && hover !== null ? (
              <span className="density-readout">
                {' '}
                · voxel ({hover.x}, {hover.y}) Z = {hover.z.toFixed(2)}
              </span>
            ) : null}
          </figcaption>
        </figure>
      ) : null}
    </div>
  );
}
