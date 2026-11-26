#!/usr/bin/env bash
# Whether a campaign's verdict is news against the recorded one.
#
# `docs/calibration/recorded-verdict.txt` holds one `verdict=...` line per search
# dimension, because a volume and a plane are different search spaces: the
# criterion is the rate at the model's own 3σ threshold, and that threshold is
# 4.87σ on the 1000² plane and 5.05σ on the 40³ volume. Comparing a 3-D rate with
# the 2-D record would compare two different questions, so the lookup is keyed by
# dimension and the earlier "not comparable" branch becomes "no record yet".
#
# A dimension with no record can still make news: the first *decisive* verdict is
# the run that should put an entry there, while a first undecided one is not a
# change and must not open an issue every week. A run with fewer toys than the
# record cannot overturn it, however surprising its interval.
#
# Prints `changed=true|false` on stdout and says why on stderr, so a workflow can
# append stdout to `$GITHUB_OUTPUT` and let the reasoning show in the log.
#
# Exits 2 on a malformed invocation or a recorded file with no line for the
# dimension *and* no readable entries at all.
#
# Usage: bash tools/lee-compare.sh <recorded-verdict.txt> <verdict> <toys> <dim>

set -uo pipefail

recorded=${1:-}
verdict=${2:-}
toys=${3:-}
dim=${4:-}
if [ -z "$recorded" ] || [ -z "$verdict" ] || [ -z "$toys" ] || [ -z "$dim" ]; then
    echo "usage: $0 <recorded-verdict.txt> <verdict> <toys> <dim>" >&2
    exit 2
fi
case "$toys" in
    '' | *[!0-9]*)
        echo "toys must be a number, got '$toys'" >&2
        exit 2
        ;;
esac
if [ ! -f "$recorded" ]; then
    echo "no such file: $recorded" >&2
    exit 2
fi

entry=$(grep -E "^verdict=[^ ]+ toys=[0-9]+ dim=${dim}( |$)" "$recorded" | tail -1)
if [ -z "$entry" ]; then
    echo "recorded:  nothing for ${dim}-D" >&2
    if [ "$verdict" = "UNDECIDED" ]; then
        echo "an undecided run with no record to compare against is not a change" >&2
        echo "changed=false"
    else
        echo "this is the first decisive ${dim}-D verdict, so it is the one to record" >&2
        echo "changed=true"
    fi
    exit 0
fi

rec_verdict=$(printf '%s\n' "$entry" | sed -n 's/^verdict=\([^ ]*\).*/\1/p')
rec_toys=$(printf '%s\n' "$entry" | sed -n 's/.* toys=\([0-9]*\).*/\1/p')
echo "recorded:  $rec_verdict at $rec_toys toys, ${dim}-D" >&2
echo "this run:  $verdict at $toys toys, ${dim}-D" >&2
if [ "$toys" -lt "$rec_toys" ]; then
    echo "fewer toys than the recorded run — cannot overturn it" >&2
    echo "changed=false"
elif [ "$verdict" = "$rec_verdict" ]; then
    echo "unchanged" >&2
    echo "changed=false"
else
    echo "the verdict moved" >&2
    echo "changed=true"
fi
