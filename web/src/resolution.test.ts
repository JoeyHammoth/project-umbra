import assert from 'node:assert/strict';
import test from 'node:test';

import { emptyTail, wilsonInterval } from './resolution.ts';

test('the Wilson interval brackets the estimate and narrows with the sample', () => {
  const peak = wilsonInterval(161, 200);
  assert.ok(peak, 'a sample of 200 has an interval');
  assert.equal(peak.rate, 0.805);
  assert.ok(peak.low < peak.rate && peak.rate < peak.high);
  assert.ok(
    Math.abs(peak.half - 0.0547) < 5e-4,
    `161 of 200 is ±0.055, got ±${peak.half.toFixed(4)}`,
  );

  const finer = wilsonInterval(1610, 2000);
  assert.ok(finer, 'ten times the toys still has an interval');
  assert.ok(
    finer.half < peak.half,
    'ten times the toys narrows the interval rather than leaving it put',
  );
});

test('an empty tail keeps a ceiling instead of collapsing to certainty', () => {
  const none = wilsonInterval(0, 200);
  assert.ok(none, 'zero exceedances is still a sample');
  assert.equal(none.low, 0);
  assert.ok(
    none.high > 0.01,
    `no toy at the peak cannot rule out a percent, got ${none.high.toFixed(4)}`,
  );
  assert.equal(wilsonInterval(0, 0), null, 'no toys, no interval');
});

test('the smallest expressible p is the half-count tail of the sample', () => {
  assert.equal(emptyTail(200), 0.5 / 201);
  assert.ok(emptyTail(200) > 1.35e-3, '200 toys cannot reach the 3σ rate');
  assert.ok(emptyTail(1000) < 1.35e-3, 'a thousand can');
  assert.equal(emptyTail(0), 1);
});
