#!/usr/bin/env bash
# The SBC campaign's verdict, read out of its report for CI.
#
# The campaign writes one machine-readable line into the report it produces:
#
#   <!-- sbc-verdict engine=a2 draws=100 posterior=100 ks_p=0.000e0 min_p=... pass=false -->
#
# That line is what a gate reads. The pooled KS p decides; `draws` and
# `posterior` decide whether the run is comparable with a recorded one at all
# (more draws are stronger evidence, fewer cannot overturn a decision). Reading
# those out of the markdown prose would mean the gate breaks the next time a
# sentence is reworded.
#
# Prints the fields as space-separated `key=value` pairs on one line, so a
# workflow can split it straight into `$GITHUB_OUTPUT`.
#
# Exits 2 when the report carries no machine line or a field is missing: a
# calibration whose verdict cannot be read has to fail its job, not pass quietly.
#
# Usage: bash tools/sbc-verdict.sh <report.md>

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

line=$(sed -n 's/.*<!-- sbc-verdict \(.*\) -->.*/\1/p' "$report" | tail -1)
if [ -z "$line" ]; then
    echo "no sbc-verdict line in $report" >&2
    exit 2
fi

for field in engine draws posterior ks_p pass; do
    case " $line " in
        *" $field="*) ;;
        *)
            echo "sbc-verdict line has no $field: $line" >&2
            exit 2
            ;;
    esac
done

printf '%s\n' "$line"
