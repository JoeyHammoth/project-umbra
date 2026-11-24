import assert from 'node:assert/strict';
import test from 'node:test';

import { P_FLOOR, curvePoints, logFraction, zFraction } from './plot.ts';

test('the log axis puts p = 1 at the top and the floor at the bottom', () => {
  assert.equal(logFraction(1), 1);
  assert.equal(logFraction(P_FLOOR), 0);
  assert.equal(
    logFraction(P_FLOOR / 1000),
    0,
    'below the floor clamps rather than leaving the axis',
  );
  assert.equal(
    logFraction(Number.NaN),
    0,
    'a missing value is drawn at the floor, not dropped as a gap',
  );
  assert.ok(
    logFraction(0.1) > logFraction(0.01) && logFraction(0.01) > logFraction(0.001),
    'a decade is a fixed step on a log axis',
  );
});

test('the curve spans the axis left to right and falls as the threshold rises', () => {
  const points = curvePoints(new Float32Array([0.9, 0.3, 0.02]));
  assert.deepEqual(
    points.map((point) => point.x),
    [0, 0.5, 1],
  );
  assert.ok(
    points[0]!.y > points[1]!.y && points[1]!.y > points[2]!.y,
    'a smaller p must be drawn lower',
  );
  assert.deepEqual(curvePoints(new Float32Array()), [], 'no curve, no fabricated line');
});

test('a threshold is placed on the axis it belongs to, and clamped to it', () => {
  assert.equal(zFraction(0), 0);
  assert.equal(zFraction(3), 0.5);
  assert.equal(zFraction(6), 1);
  assert.equal(zFraction(9), 1, 'beyond the axis clamps');
  assert.equal(zFraction(-1), 0);
  assert.equal(zFraction(Number.NaN), 0);
});
