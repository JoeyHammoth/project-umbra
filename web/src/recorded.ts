/**
 * The recorded campaign, imported from the file CI maintains.
 *
 * Kept apart from `record.ts` so the parser can be tested without a bundler:
 * `?raw` is a Vite import, and `node --test` has no Vite.
 */

import recordedText from '../../docs/calibration/recorded-verdict.txt?raw';

import { parseRecordedVerdict, recordedFor } from './record.ts';

/**
 * Every recorded criterion, by search dimension.
 *
 * Shown as a table because one verdict is not the whole answer: the plane is
 * decided at 3 × 10⁵ toys and the volume is decided by its own campaign at a
 * different threshold, and a page that quoted only the plane would leave a reader
 * believing the page's scene had been calibrated to it.
 */
export const recordedVerdicts = parseRecordedVerdict(recordedText).sort((a, b) => a.dim - b.dim);

/** The shipped scene is the 2-D one, so that is the entry the page quotes. */
export const recordedVerdict = recordedFor(recordedVerdicts, 2);
