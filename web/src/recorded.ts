/**
 * The recorded campaign, imported from the file CI maintains.
 *
 * Kept apart from `record.ts` so the parser can be tested without a bundler:
 * `?raw` is a Vite import, and `node --test` has no Vite.
 */

import recordedText from '../../docs/calibration/recorded-verdict.txt?raw';

import { parseRecordedVerdict, recordedFor } from './record.ts';

/** The shipped scene is the 2-D one, so that is the entry the page quotes. */
export const recordedVerdict = recordedFor(parseRecordedVerdict(recordedText), 2);
