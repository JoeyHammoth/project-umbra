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

/**
 * Spacing of the model curve: the same 0.05σ grid the empirical curve arrives on,
 * so the two can be read against the same marks.
 */
export const MODEL_STEP = 0.05;

/** One-sided normal tail, the local `p` the Gross–Vitells model starts from. */
export function normalTail(z: number): number {
  if (!Number.isFinite(z)) return z > 0 ? 0 : 1;
  // The series below is good to about 1e-7 but not exact at the origin, and the
  // tail of zero is the one value worth having exactly.
  if (z === 0) return 0.5;
  return 0.5 * erfc(z / Math.SQRT2);
}

/** `erfc` by Abramowitz–Stegun 7.1.26: five terms, about 1e-7, which is finer
 * than the f32 the curve arrives in. */
function erfc(x: number): number {
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  const erf = 1 - poly * Math.exp(-ax * ax);
  return x >= 0 ? 1 - erf : 1 + erf;
}

/**
 * Global `p` from the Euler anchor: the Gross–Vitells upcrossing law the Rust
 * core uses (`umbra_stats::lee::extrapolated_global_p`).
 *
 * `⟨N⟩ = p_local + meanEuler·exp(−(z² − u₀²)/2)` counts the excursion regions,
 * and `1 − e^{−⟨N⟩}` is the chance that some region is above `z`. Neither part
 * is decoration: there is no `(z/u₀)^{d−1}` factor, because these excursions are
 * strips rather than smooth `d`-D blobs and their count falls with the local
 * tail; and the count is turned into a probability, because at the anchor it is
 * eight regions per toy, where the linear form reported `p = 1`.
 *
 * At or below the anchor nothing is extrapolated and the local `p` is returned,
 * which is the one place the two differ.
 */
export function modelGlobalP(z: number, u0: number, meanEuler: number): number {
  const local = normalTail(z);
  if (!(u0 > 0) || !(meanEuler > 0) || !Number.isFinite(z) || z <= u0) {
    return Math.min(1, local);
  }
  const mean = local + meanEuler * Math.exp(-(z * z - u0 * u0) / 2);
  return Math.min(1, Math.max(local, 1 - Math.exp(-mean)));
}

/**
 * The extrapolation as points, from the anchor to the axis ceiling.
 *
 * Empty when there is no anchor to stand on: zero measured Euler characteristic
 * means the toys never formed an excursion there, and drawing a model from that
 * would be drawing the local `p` under a global label.
 *
 * One step above the anchor, not on it: at `u0` the model returns the local `p`,
 * and just above it the excursion term arrives at full size, so including the
 * anchor would draw a vertical jump that no data has. The empirical curve is the
 * one measured at the anchor.
 */
export function extrapolatedPoints(u0: number, meanEuler: number): CurvePoint[] {
  if (!(u0 > 0) || !(u0 < Z_AXIS) || !(meanEuler > 0)) return [];
  return modelCurve(u0, meanEuler);
}

/** The same curve for any `χ ≥ 0`, so a band's lower edge can reach zero. */
function modelCurve(u0: number, meanEuler: number): CurvePoint[] {
  const points: CurvePoint[] = [];
  for (let z = u0 + MODEL_STEP; z <= Z_AXIS + 1e-9; z += MODEL_STEP) {
    points.push({
      x: zFraction(z),
      y: logFraction(modelGlobalP(z, u0, meanEuler)),
    });
  }
  return points;
}

/** The extrapolation with the anchor's own standard error drawn around it. */
export interface EulerBand {
  /** The model at the measured `⟨χ⟩`. */
  centre: CurvePoint[];
  /** The model at `⟨χ⟩ − stderr`; empty when there is no band to draw. */
  low: CurvePoint[];
  /** The model at `⟨χ⟩ + stderr`. */
  high: CurvePoint[];
}

/**
 * The band `⟨χ⟩`'s standard error puts around the extrapolation.
 *
 * `⟨χ⟩` is a mean over toys, so it has a width, and the model inherits it: the
 * whole branch moves by `exp(−Δz²/2)` times the anchor's error. Drawing only the
 * centre line claims a normalisation the campaign did not measure — on 200 toys
 * the error is tens of percent of the anchor, which at 5σ is still a visible
 * band. Empty edges mean there is no band: no anchor, or no measured error.
 */
export function eulerBand(u0: number, meanEuler: number, stderr: number): EulerBand {
  const centre = extrapolatedPoints(u0, meanEuler);
  if (centre.length === 0 || !(stderr > 0) || !(u0 < Z_AXIS)) {
    return { centre, low: [], high: [] };
  }
  return {
    centre,
    low: modelCurve(u0, Math.max(0, meanEuler - stderr)),
    high: modelCurve(u0, meanEuler + stderr),
  };
}

/** Where a `Z` sits on that axis, clamped to it. */
export function zFraction(z: number, stop = Z_AXIS): number {
  if (!Number.isFinite(z) || !(stop > 0)) return 0;
  return Math.min(1, Math.max(0, z / stop));
}
