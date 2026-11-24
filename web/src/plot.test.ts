import assert from 'node:assert/strict';
import test from 'node:test';

import {
  P_FLOOR,
  Z_AXIS,
  curvePoints,
  extrapolatedPoints,
  logFraction,
  modelGlobalP,
  normalTail,
  zFraction,
} from './plot.ts';

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

test('the local tail is the normal one, to the precision the axis needs', () => {
  assert.ok(Math.abs(normalTail(0) - 0.5) < 1e-12);
  assert.ok(Math.abs(normalTail(1) - 0.158655) < 5e-5);
  assert.ok(
    Math.abs(normalTail(3) - 1.3499e-3) < 1e-5,
    '3 sigma local p is the nominal 1.35e-3',
  );
  // The five-term erfc is good to about 1e-7 absolute, which at the 2.9e-7 tail of
  // 5 sigma is 0.2% relative; the point is that it does not collapse to zero.
  assert.ok(
    Math.abs(normalTail(5) / 2.8665e-7 - 1) < 5e-3,
    'the far tail stays accurate rather than collapsing to zero',
  );
  assert.equal(normalTail(-3), 1 - normalTail(3));
});

test('the model sits above the local p, and above the anchor it is defined', () => {
  // Above u0 the excursion term only adds, so the global p can never be below
  // the local one — that inversion is exactly the bug the anchor guards.
  assert.equal(modelGlobalP(2.0, 2.5, 8.1), normalTail(2.0));
  assert.ok(modelGlobalP(4.0, 2.5, 8.1) > normalTail(4.0));
  assert.ok(modelGlobalP(4.0, 2.5, 8.1) <= 1);
  assert.ok(
    modelGlobalP(4.0, 2.5, 8.1) > modelGlobalP(4.0, 2.5, 0.5),
    'a larger Euler anchor means a heavier tail',
  );
  assert.ok(
    modelGlobalP(5.0, 2.5, 8.1) < modelGlobalP(4.0, 2.5, 8.1),
    'and the tail falls with the threshold, as a tail must',
  );
});

test('the extrapolation is drawn only with an anchor, from that anchor to the edge', () => {
  assert.deepEqual(
    extrapolatedPoints(2.5, 0),
    [],
    'no measured excursion, no model to draw',
  );
  const points = extrapolatedPoints(2.5, 8.1);
  assert.equal(points.length, 70, 'one step above 2.5 sigma to 6, in steps of 0.05');
  assert.ok(
    Math.abs(points[0]!.x - zFraction(2.55)) < 1e-12,
    'the curve starts just above its anchor, where the model is defined',
  );
  assert.ok(Math.abs(points.at(-1)!.x - 1) < 1e-6, 'and reaches the axis ceiling');
  assert.ok(
    points.every((point, index) => index === 0 || point.y <= points[index - 1]!.y),
    'p falls as the threshold rises',
  );
});

test('a threshold is placed on the axis it belongs to, and clamped to it', () => {
  assert.equal(zFraction(0), 0);
  assert.equal(zFraction(3), 0.5);
  assert.equal(zFraction(6), 1);
  assert.equal(zFraction(9), 1, 'beyond the axis clamps');
  assert.equal(zFraction(-1), 0);
  assert.equal(zFraction(Number.NaN), 0);
  assert.equal(zFraction(3, 0), 0, 'a degenerate axis does not divide by zero');
  assert.equal(Z_AXIS, 6);
});
