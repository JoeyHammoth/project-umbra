/**
 * The campaign's recorded verdict, as data.
 *
 * `docs/calibration/recorded-verdict.txt` is the file CI moves when a campaign
 * changes its answer. Parsing it here rather than copying numbers into the page
 * is the point: the site then quotes the same line the gate reads, and a stale
 * page becomes a bug in one file instead of a difference nobody notices.
 *
 * One entry per search dimension, because a volume and a plane are different
 * search spaces — the criterion is the rate at the *model's own* 3σ threshold,
 * and that threshold is 4.87σ on the plane and 5.05σ on the volume.
 */

export interface RecordedVerdict {
  verdict: string;
  /** Background-only toys behind the number. */
  toys: number;
  /** Search dimension: 2 for a plane, 3 for a volume. */
  dim: number;
  /** Empirical global rate at the model's 3σ threshold. */
  rate: number;
  /** Low end of its 95% Wilson interval. */
  low: number;
  /** High end of the same interval. */
  high: number;
  /** The model's 3σ threshold in sigma; null if the model never reached it. */
  threshold: number | null;
  /**
   * Toys the interval would need before it fits the band; null when no sample
   * size decides it, because the point estimate is itself outside the band.
   */
  needs: number | null;
}

/**
 * Every `verdict=` line in the file, in file order.
 *
 * Lines that are not a verdict — the comments that say what the file is for —
 * and lines missing the count, dimension or rate are skipped rather than turned
 * into `NaN`: a page that renders `NaN` is worse than one that renders nothing.
 */
export function parseRecordedVerdict(text: string): RecordedVerdict[] {
  const entries: RecordedVerdict[] = [];
  for (const line of text.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed.startsWith('verdict=')) continue;
    const fields = new Map<string, string>();
    for (const pair of trimmed.split(/\s+/)) {
      const at = pair.indexOf('=');
      if (at > 0) fields.set(pair.slice(0, at), pair.slice(at + 1));
    }
    const toys = Number(fields.get('toys'));
    const dim = Number(fields.get('dim'));
    const rate = Number(fields.get('rate'));
    const low = Number(fields.get('low'));
    const high = Number(fields.get('high'));
    const threshold = Number(fields.get('threshold'));
    const needs = Number(fields.get('needs'));
    if (!Number.isFinite(toys) || !Number.isFinite(dim) || !Number.isFinite(rate)) {
      continue;
    }
    entries.push({
      verdict: fields.get('verdict') ?? 'UNDECIDED',
      toys,
      dim,
      rate,
      low: Number.isFinite(low) ? low : rate,
      high: Number.isFinite(high) ? high : rate,
      threshold: Number.isFinite(threshold) ? threshold : null,
      // `needs=none` means no sample size helps, which is not the same as a
      // number: rendering `NaN` toys would invent a campaign that cannot work.
      needs: Number.isFinite(needs) ? needs : null,
    });
  }
  return entries;
}

/** The entry for one search dimension, or `undefined` when none is recorded. */
export function recordedFor(
  entries: RecordedVerdict[],
  dim: number,
): RecordedVerdict | undefined {
  return entries.filter((entry) => entry.dim === dim).at(-1);
}
