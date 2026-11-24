/**
 * The global-`p` curve as geometry.
 *
 * The look-elsewhere correction is a function, not a number: it says how often a
 * background-only search reaches each threshold. Drawing it needs a log axis,
 * because the interesting end is a thousandth and the top is one, and that axis
 * arithmetic belongs somewhere testable rather than inside an SVG.
 */

/** A 3σ detection's global false-positive rate (spec §7). */
export const NOMINAL_3_SIGMA = 1.35e-3;

/** Lowest global `p` the axis shows; below it the curve is clamped to the floor. */
export const P_FLOOR = 1e-3;

/** The axis runs 0 to 6 sigma, the same ceiling the `Z` panel uses. */
export const Z_AXIS = 6;

/** Where `p` sits between the floor and one: 0 at the floor, 1 at `p = 1`. */
export function logFraction(p: number): number {
  const floor = Math.log10(P_FLOOR);
  const value = Number.isFinite(p) ? Math.min(1, Math.max(P_FLOOR, p)) : P_FLOOR;
  return (Math.log10(value) - floor) / -floor;
}

export interface CurvePoint {
  /** Threshold as a fraction of the axis, 0 to 1. */
  x: number;
  /** Global `p` as a fraction of the log axis, 0 to 1. */
  y: number;
}

/**
 * The curve as points, evenly spaced in `Z`, with `p` on the log axis.
 *
 * A single point has no span, so it is drawn at the left edge; an empty curve is
 * an empty path rather than a fabricated straight line.
 */
export function curvePoints(values: ArrayLike<number>): CurvePoint[] {
  const count = values.length;
  if (count === 0) return [];
  return Array.from({ length: count }, (_, index) => ({
    x: count === 1 ? 0 : index / (count - 1),
    y: logFraction(values[index] ?? 1),
  }));
}

/** Where a `Z` sits on that axis, clamped to it. */
export function zFraction(z: number, stop = Z_AXIS): number {
  if (!Number.isFinite(z) || !(stop > 0)) return 0;
  return Math.min(1, Math.max(0, z / stop));
}
