#!/usr/bin/env bash
# How large a confirmation run to dispatch, if any.
#
# The campaign sizes its own follow-up from the report's `needs` field — the toys
# its interval would need before it fits ±20% — rather than from a constant,
# because the two dimensions need very different samples. A 2-D 10⁴ run is about
# 7e4 toys short; a 3-D 20 000-toy run whose point estimate sits 2% above the
# band's floor needs orders of magnitude more to prove itself, and saying that is
# more useful than dispatching a fixed 10⁵ that could not decide it either.
#
# Three reasons not to dispatch: a scheduled run only reports, a confirmation
# never confirms itself, and a sample that already decides asks for nothing. The
# count is capped at what the job's timeout can run: 3e5 toys cost ~230 minutes on
# four CI cores at the measured 46 ms/toy, and the job allows 360.
#
# Prints `toys=<n|none>` and `dispatch=true|false` on stdout, the reasoning on
# stderr.
#
# Exits 2 on a malformed invocation.
#
# Usage: bash tools/lee-confirm.sh <verdict> <toys> <needs> <confirm> <event>

set -uo pipefail

verdict=${1:-}
toys=${2:-}
needs=${3:-}
confirm=${4:-false}
event=${5:-}
if [ -z "$verdict" ] || [ -z "$toys" ] || [ -z "$event" ]; then
    echo "usage: $0 <verdict> <toys> <needs> <confirm> <event>" >&2
    exit 2
fi
case "$toys" in
    '' | *[!0-9]*)
        echo "toys must be a number, got '$toys'" >&2
        exit 2
        ;;
esac

# The most the campaign job can run inside its 360-minute timeout, at the
# measured 46 ms/toy on four cores.
MAX_TOYS=300000

skip() {
    echo "toys=none"
    echo "dispatch=false"
    echo "$1" >&2
}

if [ "$event" = "schedule" ]; then
    skip "a scheduled run only reports"
    exit 0
fi
if [ "$confirm" = "true" ]; then
    skip "a confirmation never confirms itself"
    exit 0
fi
if [ "$verdict" != "UNDECIDED" ]; then
    skip "the verdict is $verdict, so there is nothing to confirm"
    exit 0
fi
case "$needs" in
    '' | none | *[!0-9]*)
        skip "no toy count would decide this run: the point estimate is outside the band"
        exit 0
        ;;
esac
if [ "$needs" -le "$toys" ]; then
    skip "this sample already decides, so a confirmation would only repeat it"
    exit 0
fi

if [ "$needs" -gt "$MAX_TOYS" ]; then
    echo "the interval would need $needs toys; dispatching the $MAX_TOYS cap, which will report by how much it falls short" >&2
    toys=$MAX_TOYS
else
    toys=$needs
fi
echo "toys=$toys"
echo "dispatch=true"
echo "dispatching a $toys-toy confirmation" >&2
