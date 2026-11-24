/**
 * How precisely a toy count pins a rate.
 *
 * A global `p` from 200 toys is a point estimate with a width, and the page was
 * printing the point: 0.803 from 161 of 200 reads like three decimals of
 * knowledge, when the interval behind it is roughly ±0.055. The figure is a
 * measurement of a rate, so it needs the interval that a rate carries.
 *
 * Wilson rather than the textbook normal interval because the interesting end is
 * small: with no toy at the peak the normal interval collapses to a point at
 * zero, which claims certainty about a rare event from a sample that saw none of
 * it. Wilson keeps a finite ceiling. The same construction as
 * `umbra_stats::lee::rate_estimate`, so the page and the campaign report agree.
 */

/** A binomial rate with its two-sided 95% interval. */
export interface RateInterval {
  /** Exceedances over toys. */
  rate: number;
  low: number;
  high: number;
  /** Half the interval's width, which is what a `±` quotes. */
  half: number;
}

/**
 * Wilson score interval for `exceedances` of `toys` at 95%.
 *
 * Returns `null` for an empty sample, where no interval exists rather than an
 * interval from nothing.
 */
export function wilsonInterval(exceedances: number, toys: number): RateInterval | null {
  if (!(toys > 0)) return null;
  const n = toys;
  const k = Math.min(Math.max(exceedances, 0), n);
  const p = k / n;
  const z = 1.959_963_984_540_054;
  const denominator = 1 + (z * z) / n;
  const centre = (p + (z * z) / (2 * n)) / denominator;
  const half =
    (z / denominator) * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n));
  return {
    rate: p,
    low: Math.max(0, centre - half),
    high: Math.min(1, centre + half),
    half,
  };
}

/**
 * The smallest global `p` a toy sample can express.
 *
 * The estimator is `(k + ½) / (n + 1)`, so a tail no toy reaches still reports
 * a finite bound instead of zero — and that bound is the floor of what this
 * sample can say. At 200 toys it is 0.0025, well above the 3σ rate of 0.00135,
 * which is why the page's curve cannot reach its own 3σ line.
 */
export function emptyTail(toys: number): number {
  if (!(toys > 0)) return 1;
  return 0.5 / (toys + 1);
}
