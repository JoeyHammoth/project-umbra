import assert from 'node:assert/strict';
import test from 'node:test';

import { RAMP, colourAt, rangeOf, rampAt } from './ramp.ts';

test('a single outlier cannot set the top of the window', () => {
  const field = new Float32Array(1000).fill(2.5);
  field[0] = 40;
  const window = rangeOf(field);
  assert.equal(window.lo, 2.5);
  assert.ok(window.hi < 5, `the 98th percentile should stay near the rock, got ${window.hi}`);
  assert.equal(window.clipped, true, 'the outlier is outside the window and says so');
});

test('a flat field has no window to draw, and is not called clipped', () => {
  const window = rangeOf(new Float32Array(64).fill(2.65));
  assert.equal(window.lo, window.hi);
  assert.equal(window.clipped, false);
});

test('air is outside the window rather than at its floor', () => {
  const field = new Float32Array([0, 0, 0, 2, 4, 6, 8]);
  const window = rangeOf(field);
  assert.equal(window.lo, 2, 'zero must not pull the floor down');
  assert.equal(window.hi, 8);
  assert.equal(window.clipped, false, 'nothing inside the positive range was cut');
});

test('an empty field has an empty window instead of dividing by nothing', () => {
  const window = rangeOf(new Float32Array(16));
  assert.deepEqual(window, { lo: 0, hi: 0, clipped: false });
});

test('the density ramp never starts at black, which is how the browser check counts voxels', () => {
  assert.ok(rampAt(0)[0] > 0, 'a voxel at the bottom of the window must still be visible');
  assert.deepEqual(colourAt(RAMP, 1), [252, 255, 164]);
  assert.deepEqual(rampAt(-1), rampAt(0), 'below the window clamps');
  assert.deepEqual(rampAt(2), rampAt(1), 'above the window clamps');
});
