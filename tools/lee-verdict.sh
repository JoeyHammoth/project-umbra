#!/usr/bin/env bash
# The LEE campaign's verdict, read out of its report for CI.
#
# The campaign writes one machine-readable line into the report it produces:
#
#   <!-- lee-verdict verdict=UNDECIDED toys=10000 dim=2 rate=1.400e-3 low=... -->
#
# That line is what the campaign job acts on. The verdict decides whether a
# confirmation run is dispatched, and the toy count decides whether the result is
# comparable with the recorded one at all (more toys are stronger evidence; fewer
# cannot overturn a decision). Reading those out of the markdown prose would mean
# the gate breaks the next time a sentence is reworded.
#
# Prints the fields as space-separated `key=value` pairs on one line, so a
# workflow can split it straight into `$GITHUB_OUTPUT`.
#
# Exits 2 when the report carries no machine line or a field is missing: a
# campaign whose verdict cannot be read has to fail its job, not pass quietly.
#
# Usage: bash tools/lee-verdict.sh <report.md>

set -uo pipefail

report=${1:-}
if [ -z "$report" ]; then
    echo "usage: $0 <report.md>" >&2
    exit 2
fi
if [ ! -f "$report" ]; then
    echo "no such report: $report" >&2
    exit 2
fi

line=$(sed -n 's/.*<!-- lee-verdict \(.*\) -->.*/\1/p' "$report" | tail -1)
if [ -z "$line" ]; then
    echo "no lee-verdict line in $report" >&2
    exit 2
fi

for field in verdict toys dim; do
    case " $line " in
        *" $field="*) ;;
        *)
            echo "lee-verdict line has no $field: $line" >&2
            exit 2
            ;;
    esac
done

printf '%s\n' "$line"
