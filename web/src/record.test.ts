import assert from 'node:assert/strict';
import { test } from 'node:test';

import { parseRecordedVerdict, recordedFor } from './record.ts';

const FILE = `# The M3 rate criterion's last recorded verdict.
#
# Current: docs/calibration/lee-2d-1e5-fixed.md — 128 of 100 000 over the 4.87σ
# threshold.
verdict=UNDECIDED toys=20000 dim=3 rate=1.100e-3 low=7.266e-4 high=1.665e-3 threshold=5.05
verdict=UNDECIDED toys=100000 dim=2 rate=1.280e-3 low=1.077e-3 high=1.522e-3 threshold=4.87
`;

test('every verdict line parses, and the comments do not', () => {
  const entries = parseRecordedVerdict(FILE);
  assert.equal(entries.length, 2, 'one entry per line, comments skipped');
  assert.equal(entries[1]!.dim, 2);
  assert.equal(entries[1]!.toys, 100_000);
  assert.ok(Math.abs(entries[1]!.rate - 1.28e-3) < 1e-12);
  assert.equal(entries[1]!.threshold, 4.87);
});

test('a dimension is looked up by its own entry, not the last one written', () => {
  const entries = parseRecordedVerdict(FILE);
  assert.equal(recordedFor(entries, 2)?.toys, 100_000);
  assert.equal(recordedFor(entries, 3)?.toys, 20_000);
  assert.equal(recordedFor(entries, 4), undefined, 'an unrecorded dimension is absent');
});

test('a truncated or unparseable line is skipped rather than rendered as NaN', () => {
  const entries = parseRecordedVerdict(
    'verdict=UNDECIDED toys=1000 dim=2\nverdict=PASS toys=abc dim=2 rate=1e-3\n',
  );
  assert.deepEqual(entries, []);
});

test('the toys a sample still needs are carried through, including none at all', () => {
  const entries = parseRecordedVerdict(
    'verdict=UNDECIDED toys=20000 dim=3 rate=1.1e-3 low=7e-4 high=1.6e-3 needs=11008000\n' +
      'verdict=FAIL toys=1000 dim=2 rate=2e-3 needs=none\n' +
      'verdict=PASS toys=300000 dim=2 rate=1.42e-3 needs=136483\n',
  );
  assert.equal(entries[0]!.needs, 11_008_000, 'a number is a budget, not an absence');
  assert.equal(entries[1]!.needs, null, 'no sample size helps, which is not `0`');
  assert.equal(entries[2]!.needs, 136_483);
});

test('a sample that decides asks for no more toys than it has', () => {
  const [entry] = parseRecordedVerdict('verdict=PASS toys=300000 dim=2 rate=1.42e-3 needs=136483\n');
  assert.ok(entry!.needs! <= entry!.toys, 'the page reads this as "decided"');
});

test('an interval missing its ends is reported at the point, not at infinity', () => {
  const [entry] = parseRecordedVerdict('verdict=PASS toys=10 dim=2 rate=1.35e-3\n');
  assert.equal(entry!.low, 1.35e-3);
  assert.equal(entry!.high, 1.35e-3);
  assert.equal(entry!.threshold, null);
});
