#!/usr/bin/env bash
# Versioned asset integrity gate.
#
# Spec §8.5 makes a published version directory immutable: `manifest.json` pins
# each file's path, byte length and SHA-256, and the loader checks schema and
# length on every load. This gate holds the repository to the same contract, so
# a table edited in place, truncated, or joined by a stray file fails CI instead
# of failing in someone's browser.
#
# For every web/public/<asset>/v<N>/ directory:
#   - manifest.json exists and declares schema umbra.assets/1
#   - every file it lists exists, at the recorded byte length and SHA-256
#   - every file present is listed; nothing unlisted sits beside the assets
#   - the manifest is in a form this check can parse in full, and lists at least
#     one file — an unparseable or empty manifest is a failure, not a pass
#
# The gate does not regenerate the tables; that is `umbra-cli gen-flux`, and a
# byte-level drift check against a rerun is a separate question from whether the
# committed files match the committed manifest.
#
# Usage: bash tools/check-assets.sh   (exit 0 = consistent, 1 = problems listed)

set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

failures=0
fail() {
    printf 'FAIL: %s\n' "$*" >&2
    failures=$((failures + 1))
}

shopt -s nullglob
dirs=(web/public/*/v[0-9]*/)
shopt -u nullglob

if [ "${#dirs[@]}" -eq 0 ]; then
    fail "no versioned asset directories (web/public/*/v<N>/) found"
fi

checked_files=0

for dir in "${dirs[@]}"; do
    manifest="${dir}manifest.json"
    if [ ! -f "$manifest" ]; then
        fail "$dir has no manifest.json (spec §8.5)"
        continue
    fi

    grep -qE '"schema"[[:space:]]*:[[:space:]]*"umbra\.assets/1"' "$manifest" ||
        fail "$manifest does not declare schema umbra.assets/1"

    # Work on one line so entries that a formatter split across lines still
    # parse, and so a split object is caught by the count comparison below.
    flat=$(tr '\n' ' ' <"$manifest")
    declared=$(grep -o '"path"' <<<"$flat" | wc -l)
    parsed=$(grep -oE '\{[^{}]*"path"[^{}]*\}' <<<"$flat" | wc -l)

    if [ "$declared" -ne "$parsed" ]; then
        fail "$manifest: $declared file entr(ies) but this check parsed $parsed
      (keep one {\"path\", \"bytes\", \"sha256\"} object per file, in that order)"
    fi
    if [ "$parsed" -eq 0 ]; then
        fail "$manifest lists no files"
    fi

    declare -A listed=()
    while IFS='|' read -r path bytes hash; do
        [ -n "$path" ] && [ -n "$hash" ] || continue
        listed["$path"]=1
        file="${dir}${path}"
        if [ ! -f "$file" ]; then
            fail "$manifest lists '$path', which is missing"
            continue
        fi
        actual_bytes=$(wc -c <"$file" | tr -d '[:space:]')
        if [ "$actual_bytes" != "$bytes" ]; then
            fail "$file is $actual_bytes bytes, manifest says $bytes"
        fi
        actual_hash=$(sha256sum "$file" | cut -d' ' -f1)
        if [ "$actual_hash" != "$hash" ]; then
            fail "$file has sha256 $actual_hash, manifest says $hash"
        fi
        checked_files=$((checked_files + 1))
    done < <(grep -oE '\{[^{}]*"path"[^{}]*\}' <<<"$flat" |
        sed -E 's/.*"path"[[:space:]]*:[[:space:]]*"([^"]*)".*"bytes"[[:space:]]*:[[:space:]]*([0-9]+).*"sha256"[[:space:]]*:[[:space:]]*"([0-9a-f]{64})".*/\1|\2|\3/')

    while IFS= read -r file; do
        relative=${file#"$dir"}
        [ -n "${listed[$relative]:-}" ] ||
            fail "$file is not listed in $manifest; a published version directory is immutable"
    done < <(find "$dir" -type f ! -name manifest.json | sort)
done

if [ "$failures" -gt 0 ]; then
    printf '\n%d asset integrity problem(s). Published versions are immutable (spec §8.5):\n' \
        "$failures" >&2
    printf 'change the file and the manifest in the same commit, or publish a new version directory.\n' >&2
    exit 1
fi

printf 'assets: %d version director(ies), %d file(s) match their manifests\n' \
    "${#dirs[@]}" "$checked_files"
