/**
 * Colour scales, and the window each field is drawn on.
 *
 * Plain TypeScript with no React and no canvas, so the arithmetic that decides
 * what a pixel means can be tested under `node --test` alongside the compute
 * layer. The component that paints is then only about wiring.
 */

export type Stop = readonly [t: number, r: number, g: number, b: number];

/**
 * A five-stop heatmap for density: indigo → purple → crimson → orange → pale
 * yellow.
 *
 * The first stop is deliberately not black. Air is painted black by the view, and
 * a ramp starting at zero would make the faintest real density indistinguishable
 * from empty space — which is also what lets the browser check count `r > 0` as
 * "this voxel carried a value".
 */
export const RAMP: readonly Stop[] = [
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
export const SIGMA_RAMP: readonly Stop[] = [
  [0, 12, 18, 38],
  [0.35, 33, 78, 138],
  [0.65, 66, 168, 190],
  [1, 236, 246, 240],
];

/** Full scale of the `Z` panel, in sigma. */
export const Z_CEILING = 5;

/** Where the density window starts and ends, as quantiles of the positive values. */
export const WINDOW_LOW = 0.02;
export const WINDOW_HIGH = 0.98;

/** Colour at fraction `t` of `ramp`, linearly interpolated between stops. */
export function colourAt(ramp: readonly Stop[], t: number): [number, number, number] {
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

/** Colour for `Z` at fraction `t` of the ceiling. */
export function sigmaAt(t: number): [number, number, number] {
  return colourAt(SIGMA_RAMP, t);
}

/** The ramp as a CSS gradient, so the legend and the pixels cannot drift apart. */
export function legendOf(ramp: readonly Stop[]): string {
  return `linear-gradient(90deg, ${ramp
    .map(([t, r, g, b]) => `rgb(${r} ${g} ${b}) ${Math.round(t * 100)}%`)
    .join(', ')})`;
}

export const DENSITY_LEGEND = legendOf(RAMP);
export const SIGMA_LEGEND = legendOf(SIGMA_RAMP);

/** The window a field is drawn on, in the field's own units. */
export interface Window {
  lo: number;
  hi: number;
  /** Positive values strictly below the floor. */
  below: number;
  /** Positive values strictly above the ceiling. */
  above: number;
  /** Whether anything fell outside, which the legend states. */
  clipped: boolean;
}

/** Value at `fraction` through an ascending array. */
function quantile(sorted: readonly number[], fraction: number): number {
  const index = Math.min(sorted.length - 1, Math.max(0, Math.round(fraction * (sorted.length - 1))));
  return sorted[index] ?? 0;
}

/**
 * The window a density field is drawn on: its 2nd to 98th percentile.
 *
 * Percentiles rather than the extremes, because a single hot voxel would
 * otherwise set the top of the ramp and flatten every other voxel into the dark
 * end — the same failure as drawing a zero-to-peak range on a nearly uniform
 * field, just from the other direction. Zero is excluded entirely: air is not
 * part of the rock's range, and including it would drag the floor down for any
 * field with a hole in it.
 *
 * `clipped` says whether anything fell outside, so the legend can print `2–98%`
 * instead of implying it is showing the whole field.
 */
export function rangeOf(
  field: Float32Array,
  low = WINDOW_LOW,
  high = WINDOW_HIGH,
): Window {
  const values: number[] = [];
  for (const value of field) {
    if (value > 0) values.push(value);
  }
  if (values.length === 0) return { lo: 0, hi: 0, below: 0, above: 0, clipped: false };
  values.sort((a, b) => a - b);
  const lo = quantile(values, low);
  const hi = quantile(values, high);
  // The counts, not just a flag: "2–98%" says a window was applied, and these
  // say how much of the field it moved out of view.
  const below = values.filter((value) => value < lo).length;
  const above = values.filter((value) => value > hi).length;
  return { lo, hi, below, above, clipped: below > 0 || above > 0 };
}
