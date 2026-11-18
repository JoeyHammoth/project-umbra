#!/usr/bin/env bash
# Licence and provenance gate.
#
# Three assertions, each aimed at something a reviewer can forget rather than
# something a compiler catches:
#
#   1. LICENSE exists, and every manifest declares a licence — the root
#      [workspace.package] table, every crate manifest (inheriting or stating
#      its own), and every package.json.
#   2. Every non-Rust file under crates/*/tests/ is registered in
#      tools/fixture-origins.txt, so a test fixture cannot arrive silently.
#   3. Every fixture registered as third-party is named in
#      THIRD-PARTY-NOTICES.md, which is where its licence and notice live.
#
# Scope is deliberate: Rust manifests, npm manifests, and test fixtures. It does
# not audit crates.io or npm dependencies — those carry their own licence fields
# and their versions are pinned in Cargo.lock and web/package-lock.json.
#
# Usage: bash tools/check-licensing.sh   (exit 0 = clean, 1 = problems listed)

set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

failures=0
fail() {
    printf 'FAIL: %s\n' "$*" >&2
    failures=$((failures + 1))
}

# --- 1. Licence declarations ------------------------------------------------

[ -f LICENSE ] || fail "LICENSE is missing"

grep -qE '^license[[:space:]]*=' Cargo.toml ||
    fail "Cargo.toml declares no license in [workspace.package]"

shopt -s nullglob
for manifest in crates/*/Cargo.toml; do
    grep -qE '^license' "$manifest" || fail "$manifest declares no license"
done
shopt -u nullglob

while IFS= read -r manifest; do
    grep -qE '^[[:space:]]*"license"[[:space:]]*:' "$manifest" ||
        fail "$manifest declares no \"license\" field"
done < <(find . \( -name node_modules -o -name target -o -name .git \) -prune -o \
    -name package.json -print | sort)

# --- 2. Fixture registry ----------------------------------------------------

registry=tools/fixture-origins.txt
[ -f "$registry" ] || fail "$registry is missing"

declare -A origin_of=()
if [ -f "$registry" ]; then
    while read -r path origin _note; do
        case "$path" in '' | '#'*) continue ;; esac
        case "$origin" in
            third-party | generated) ;;
            *)
                fail "$registry: '$path' has origin '$origin' (want third-party or generated)"
                continue
                ;;
        esac
        [ -f "$path" ] || {
            fail "$registry lists '$path', which does not exist"
            continue
        }
        origin_of["$path"]="$origin"
        if [ "$origin" = third-party ]; then
            grep -qF -- "$path" THIRD-PARTY-NOTICES.md 2>/dev/null ||
                fail "'$path' is third-party but is not named in THIRD-PARTY-NOTICES.md"
        fi
    done <"$registry"
fi

while IFS= read -r fixture; do
    [ -n "${origin_of[$fixture]:-}" ] ||
        fail "'$fixture' is an unregistered test fixture: add it to $registry as
      'third-party' (plus its notice in THIRD-PARTY-NOTICES.md) or 'generated'"
done < <(find crates -path '*/tests/*' -type f ! -name '*.rs' -print | sort)

# --- Verdict ----------------------------------------------------------------

if [ "$failures" -gt 0 ]; then
    printf '\n%d licensing problem(s). See THIRD-PARTY-NOTICES.md and %s.\n' \
        "$failures" "$registry" >&2
    exit 1
fi

printf 'licensing: LICENSE, %d crate manifests, npm manifests and %d registered fixture(s) accounted for\n' \
    "$(printf '%s\n' crates/*/Cargo.toml | wc -l)" "${#origin_of[@]}"
