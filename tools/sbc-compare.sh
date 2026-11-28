#!/usr/bin/env bash
# Whether an SBC campaign's verdict is news against the recorded one.
#
# `docs/calibration/sbc-recorded.txt` holds one `verdict=...` line per engine,
# because the engine is what the bars are being judged for: A1's card is the
# data-only Fisher map and A2's is the TV-regularised one, so a verdict from one
# says nothing about the other. Within an engine, a run is comparable only when
# it has at least as many draws as the record.
#
# A verdict of FAIL is recorded, not hidden: the campaign's job is to say what
# the bars do, and the recorded line is where "what they do" lives. A run that
# disagrees with it is the run that should move the line (and the doc it points
# at), which is why the gate reports a *change* rather than failing on FAIL —
# a permanently red weekly job would say nothing after its first week.
#
# Prints `changed=true|false` on stdout and says why on stderr, so a workflow can
# append stdout to `$GITHUB_OUTPUT`.
#
# Exits 2 on a malformed invocation or a missing record file.
#
# Usage: bash tools/sbc-compare.sh <recorded.txt> <verdict> <draws> <engine>

set -uo pipefail

recorded=${1:-}
verdict=${2:-}
draws=${3:-}
engine=${4:-}
if [ -z "$recorded" ] || [ -z "$verdict" ] || [ -z "$draws" ] || [ -z "$engine" ]; then
    echo "usage: $0 <recorded.txt> <verdict> <draws> <engine>" >&2
    exit 2
fi
case "$verdict" in
    PASS | FAIL) ;;
    *)
        echo "verdict must be PASS or FAIL, got '$verdict'" >&2
        exit 2
        ;;
esac
case "$draws" in
    '' | *[!0-9]*)
        echo "draws must be a number, got '$draws'" >&2
        exit 2
        ;;
esac
if [ ! -f "$recorded" ]; then
    echo "no such file: $recorded" >&2
    exit 2
fi

entry=$(grep -E "^verdict=(PASS|FAIL) engine=${engine} draws=[0-9]+" "$recorded" | tail -1)
if [ -z "$entry" ]; then
    echo "recorded:  nothing for engine $engine" >&2
    echo "this is the first recorded $engine run, so it is the one to record" >&2
    echo "changed=true"
    exit 0
fi

rec_verdict=$(printf '%s\n' "$entry" | sed -n 's/^verdict=\([^ ]*\).*/\1/p')
rec_draws=$(printf '%s\n' "$entry" | sed -n 's/.* draws=\([0-9]*\).*/\1/p')
echo "recorded:  $rec_verdict at $rec_draws draws, engine $engine" >&2
echo "this run:  $verdict at $draws draws, engine $engine" >&2
if [ "$draws" -lt "$rec_draws" ]; then
    echo "fewer draws than the recorded run — cannot overturn it" >&2
    echo "changed=false"
elif [ "$verdict" = "$rec_verdict" ]; then
    echo "unchanged" >&2
    echo "changed=false"
else
    echo "the verdict moved" >&2
    echo "changed=true"
fi
