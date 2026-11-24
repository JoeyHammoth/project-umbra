/**
 * The look-elsewhere curve, drawn.
 *
 * The page prints the global `p` at the peak, but a single number hides the shape
 * of the correction: that a 3σ local peak happens in a background-only search a
 * thousand times more often than a 3σ global one. This plots the toys' null
 * against the threshold, marks the scan's own peak, draws the global-3σ rate as
 * a reference line, and — when the toys formed an excursion at the Euler anchor —
 * draws the Gross–Vitells extrapolation past where the toys reach, dashed,
 * because that part is a model and not a measurement.
 *
 * Three things the figure refuses to overstate. The model is dashed and labelled
 * as a model. It carries `⟨χ⟩`'s standard error as a shaded band, because the
 * anchor is a mean over toys and the branch moves with it. And the largest
 * maximum in the null is marked, so the reader can see exactly where the
 * measured part stops and the extrapolated part begins.
 *
 * The geometry lives in `plot.ts` so the axis arithmetic is tested without a
 * renderer; this file only places points.
 */

import type { CurvePoint } from './plot.ts';
import {
  NOMINAL_3_SIGMA,
  Z_AXIS,
  curvePoints,
  eulerBand,
  logFraction,
  zFraction,
} from './plot.ts';

const WIDTH = 600;
const HEIGHT = 150;
const LEFT = 40;
const RIGHT = 590;
const TOP = 12;
const BOTTOM = 118;

export interface GlobalPPlotProps {
  /** Global `p` at 0, 0.05, …, 6σ. */
  curve: ArrayLike<number> | null;
  /** The scan's peak, which the marker sits on. */
  peakZ: number;
  /** Toys behind the curve, stated in the caption. */
  toys: number;
  /**
   * Mean Euler characteristic at the anchor. Zero — no toy ever crossed `u0` —
   * means there is no measured excursion to extrapolate from, so no dashed curve.
   */
  meanEuler?: number;
  /** Standard error of that mean; 0 draws the centre line without a band. */
  meanEulerStderr?: number;
  /** Euler anchor, in sigma. */
  u0?: number;
  /** The largest maximum in the null: where the measured curve stops. */
  maxToyZ?: number;
}

export function GlobalPPlot({
  curve,
  peakZ,
  toys,
  meanEuler = 0,
  meanEulerStderr = 0,
  u0 = 0,
  maxToyZ = 0,
}: GlobalPPlotProps) {
  const points = curvePoints(curve ?? []);
  if (points.length === 0) return null;
  const x = (fraction: number) => LEFT + fraction * (RIGHT - LEFT);
  const y = (fraction: number) => BOTTOM - fraction * (BOTTOM - TOP);
  const path = (list: CurvePoint[], close = false) => {
    if (list.length === 0) return '';
    const body = list
      .map(
        (point, index) =>
          `${index === 0 ? 'M' : 'L'}${x(point.x).toFixed(1)},${y(point.y).toFixed(1)}`,
      )
      .join(' ');
    return close ? `${body} Z` : body;
  };
  const curvePath = path(points);
  // The model is drawn from its anchor to the edge, and only when the anchor was
  // measured: with no excursion the Gross–Vitells term is zero and the "model"
  // would be the local p wearing a global label.
  const band = eulerBand(u0, meanEuler, meanEulerStderr);
  const modelPath = path(band.centre);
  const bandPath =
    band.low.length > 0 && band.high.length === band.low.length
      ? path([...band.low, ...[...band.high].reverse()], true)
      : '';
  // The curve is sampled on a fixed grid, so the peak is reported at the nearest
  // sample rather than at an interpolated point no toy ever measured.
  const index = Math.min(
    points.length - 1,
    Math.max(0, Math.round((zFraction(peakZ) * (points.length - 1)) as number)),
  );
  const marked = points[index]!;
  const peakP = Number(curve?.[index] ?? 1);
  const threshold = y(logFraction(NOMINAL_3_SIGMA));
  const toyEdge = band.centre.length > 0 && maxToyZ > 0 ? zFraction(maxToyZ) : null;

  return (
    <figure className="density global-p">
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        data-testid="global-p"
        data-model={band.centre.length > 0 ? 'model' : 'none'}
        aria-label={`Global p from ${toys} background-only toys against the scan threshold. The tall line marks this scan's peak at ${peakZ.toFixed(
          2,
        )}σ, where the toys give p = ${peakP.toFixed(
          3,
        )}; the dashed line is the global 3σ rate.${
          band.centre.length > 0
            ? ` The dashed curve is the Gross–Vitells model from the ${u0.toFixed(
                2,
              )}σ anchor, extrapolated beyond where the toys reach.${
                bandPath
                  ? ` The shaded band is the anchor's standard error.`
                  : ''
              }`
            : ''
        }${
          toyEdge !== null
            ? ` The dotted line at ${maxToyZ.toFixed(
                2,
              )}σ is the largest maximum among the toys, where the measured curve stops.`
            : ''
        }`}
      >
        {bandPath ? (
          <path
            className="global-p-band"
            data-testid="global-p-band"
            d={bandPath}
          />
        ) : null}
        {band.centre.length > 0 ? (
          <path
            className="global-p-model"
            data-testid="global-p-model"
            d={modelPath}
          />
        ) : null}
        <line
          className="global-p-threshold"
          x1={LEFT}
          x2={RIGHT}
          y1={threshold}
          y2={threshold}
        />
        <text x={RIGHT} y={threshold - 5} textAnchor="end">
          3σ global
        </text>
        <path className="global-p-curve" d={curvePath} />
        {band.centre.length > 0 ? (
          <text
            className="global-p-model-label"
            x={RIGHT}
            y={threshold + 13}
            textAnchor="end"
          >
            Gross–Vitells model
          </text>
        ) : null}
        {toyEdge !== null ? (
          <g className="global-p-toys-mark">
            <line
              className="global-p-toys"
              data-testid="global-p-toys"
              data-max-z={maxToyZ.toFixed(2)}
              x1={x(toyEdge)}
              x2={x(toyEdge)}
              y1={TOP}
              y2={BOTTOM}
            />
            <text
              className="global-p-toys-label"
              x={x(toyEdge) - 4}
              y={TOP + 10}
              textAnchor="end"
            >
              toys stop {maxToyZ.toFixed(2)}σ
            </text>
          </g>
        ) : null}
        <line
          className="global-p-marker"
          x1={x(zFraction(peakZ))}
          x2={x(zFraction(peakZ))}
          y1={TOP}
          y2={BOTTOM}
        />
        <circle className="global-p-dot" cx={x(marked.x)} cy={y(marked.y)} r={4} />
        <text x={LEFT - 8} y={TOP + 4} textAnchor="end">
          1
        </text>
        <text x={LEFT - 8} y={BOTTOM + 4} textAnchor="end">
          10⁻³
        </text>
        <text x={LEFT} y={HEIGHT - 8}>
          0σ
        </text>
        <text x={RIGHT} y={HEIGHT - 8} textAnchor="end">
          {Z_AXIS}σ
        </text>
      </svg>
      <figcaption>
        Global <em>p</em> from {toys} background-only toys: the chance that a
        search of this grid reaches each threshold when there is nothing to find.
        The vertical line is this scan's peak at {peakZ.toFixed(2)}σ, where it
        reads {peakP.toFixed(3)}; the dashed line is the global 3σ rate.
        {band.centre.length > 0
          ? ` The dashed curve is the Gross–Vitells model from the ${u0.toFixed(
              2,
            )}σ anchor (mean χ = ${meanEuler.toFixed(2)}${
              bandPath ? `, banded by its ±${meanEulerStderr.toFixed(2)} error` : ''
            }): above the anchor it adds the excursion regions a toy's maximum alone does not count, and past the toys' reach it is a model, not a measurement.`
          : ''}
        {toyEdge !== null
          ? ` The dotted line at ${maxToyZ.toFixed(
              2,
            )}σ is where the toys stop — the largest maximum in the null — so everything to its right is extrapolated.`
          : ''}
      </figcaption>
    </figure>
  );
}
