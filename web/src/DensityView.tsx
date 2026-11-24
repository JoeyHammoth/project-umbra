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
 * colour scale. Density is a positive quantity, drawn across a 2–98% window of
 * its own positive values with both ends printed in the legend: windowed rather
 * than 0-to-peak, because a reconstruction is mostly rock near 2.65 and a
 * zero-to-peak ramp paints it as a flat slab; percentile rather than min-to-max,
 * because a single hot voxel would flatten it again from the other side. `Z` is a
 * significance, fixed at 0–5σ rather than normalised to its own peak, because a
 * ramp that stretched a null field to full brightness would make noise look like
 * a detection.
 *
 * The ramps and the window live in `ramp.ts` so the arithmetic that decides what
 * a pixel means is testable without a browser; this file is the wiring.
 */

import { useEffect, useRef, useState } from 'react';
import type { PointerEvent as ReactPointerEvent } from 'react';

import { GlobalPPlot } from './GlobalPPlot';
import {
  DENSITY_LEGEND,
  SIGMA_LEGEND,
  WINDOW_HIGH,
  WINDOW_LOW,
  Z_CEILING,
  rampAt,
  rangeOf,
  sigmaAt,
} from './ramp.ts';

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
  /** Global `p` at 0, 0.05, …, 6σ; null before the calibration lands. */
  globalP?: Float32Array | null;
  /** The scan's peak, marked on that curve. */
  leePeakZ?: number;
  /** Toys behind the curve, stated in its caption. */
  leeToys?: number;
  /** Mean Euler characteristic at the anchor; 0 when no toy crossed it. */
  leeMeanEuler?: number;
  /** Standard error of that mean, drawn as a band around the model. */
  leeEulerStderr?: number;
  /** The Euler anchor, in sigma. */
  leeU0?: number;
  /** The largest maximum in the null, where the measured curve stops. */
  leeMaxZ?: number;
}

/**
 * Fills an `ImageData` with the density, across the field's own window.
 *
 * Air stays black: it is not part of the rock's range, and painting it black also
 * means a hole reads as structure rather than as a window onto the page. Values
 * outside the window clip to its ends, which the legend states.
 */
export function paint(
  density: Float32Array,
  width: number,
  low = WINDOW_LOW,
  high = WINDOW_HIGH,
): { data: Uint8ClampedArray<ArrayBuffer>; height: number } {
  const height = Math.max(1, Math.ceil(density.length / width));
  const data = new Uint8ClampedArray(width * height * 4);
  const { lo, hi } = rangeOf(density, low, high);
  const span = hi - lo;
  for (let index = 0; index < density.length; index += 1) {
    const value = density[index] ?? 0;
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
    const [r, g, b] = sigmaAt(Math.max(0, z[index] ?? 0) / ceiling);
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

/** The empty field the legend reads before a reconstruction arrives. */
const NOTHING = new Float32Array();

export function DensityView({
  density,
  z = null,
  width,
  summary,
  scanSummary,
  globalP = null,
  leePeakZ = 0,
  leeToys = 0,
  leeMeanEuler = 0,
  leeEulerStderr = 0,
  leeU0 = 0,
  leeMaxZ = 0,
}: DensityViewProps) {
  const densityCanvas = useRef<HTMLCanvasElement | null>(null);
  const scanCanvas = useRef<HTMLCanvasElement | null>(null);
  const [hover, setHover] = useState<{
    x: number;
    y: number;
    density: number;
    z: number | null;
  } | null>(null);
  // The window is a control, not a constant. 2–98% is the right default, but
  // the only way to see what it clips is to move it: the sliders are the field's
  // own percentiles, so 0–100% draws the outliers the default hides.
  const [floor, setFloor] = useState(WINDOW_LOW);
  const [ceiling, setCeiling] = useState(WINDOW_HIGH);
  const height = density ? Math.max(1, Math.ceil(density.length / width)) : 1;
  const window = rangeOf(density ?? NOTHING, floor, ceiling);

  useEffect(() => {
    if (!density || density.length === 0) return;
    draw(densityCanvas.current, paint(density, width, floor, ceiling), width);
  }, [density, width, floor, ceiling]);

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
          <span>{window.lo.toFixed(2)}</span>
          <span className="density-ramp" style={{ background: DENSITY_LEGEND }} />
          <span>
            {`${window.hi.toFixed(2)} g cm⁻³ · ${Math.round(floor * 100)}–${Math.round(
              ceiling * 100,
            )}%`}
            {window.below + window.above > 0
              ? ` · ${window.below + window.above} outside`
              : ''}
          </span>
        </div>
        <div className="window">
          <label>
            <span>floor {Math.round(floor * 100)}%</span>
            <input
              type="range"
              min={0}
              max={49}
              step={0.5}
              value={floor * 100}
              onChange={(event) => setFloor(Number(event.target.value) / 100)}
            />
          </label>
          <label>
            <span>ceiling {Math.round(ceiling * 100)}%</span>
            <input
              type="range"
              min={51}
              max={100}
              step={0.5}
              value={ceiling * 100}
              onChange={(event) => setCeiling(Number(event.target.value) / 100)}
            />
          </label>
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
            {hover !== null && hover.z !== null ? (
              <span className="density-readout">
                {' '}
                · voxel ({hover.x}, {hover.y}) Z = {hover.z.toFixed(2)}
              </span>
            ) : null}
          </figcaption>
        </figure>
      ) : null}

      {globalP && globalP.length > 0 ? (
        <GlobalPPlot
          curve={globalP}
          peakZ={leePeakZ}
          toys={leeToys}
          meanEuler={leeMeanEuler}
          meanEulerStderr={leeEulerStderr}
          u0={leeU0}
          maxToyZ={leeMaxZ}
        />
      ) : null}
    </div>
  );
}
