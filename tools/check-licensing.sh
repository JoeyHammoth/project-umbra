#!/usr/bin/env bash
# Licence and provenance gate.
#
# Four assertions, each aimed at something a reviewer can forget rather than
# something a compiler catches:
#
#   1. LICENSE exists, and every manifest declares a licence — the root
#      [workspace.package] table, every crate manifest (inheriting or stating
#      its own), and every package.json.
#   2. Every non-Rust file under crates/*/tests/ is registered in
#      tools/fixture-origins.txt, so a test fixture cannot arrive silently.
#   3. Every fixture registered as third-party is named in
#      THIRD-PARTY-NOTICES.md, which is where its licence and notice live.
#   4. Every file under web/public/ is covered by a row of the spec's §8.4 asset
#      inventory, and that row's licence is one this repo may ship. The spec's
#      own risk R14 names "an asset with no entry in §8.4" as the failure.
#
# Scope is deliberate: Rust manifests, npm manifests, test fixtures, and shipped
# web assets. It does not audit crates.io or npm dependencies — those carry
# their own licence fields and their versions are pinned in Cargo.lock and
# web/package-lock.json.
#
# Usage: bash tools/check-licensing.sh   (exit 0 = clean, 1 = problems listed)

set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

failures=0
fail() {
    printf 'FAIL: %s\n' "$*" >&2
    failures=$((failures + 1))
}

trim() {
    local s=$1
    s="${s#"${s%%[![:space:]]*}"}"
    s="${s%"${s##*[![:space:]]}"}"
    printf '%s' "$s"
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

# --- 3. Shipped web assets vs the spec inventory ----------------------------

spec=source/UMBRA_DESIGN_SPEC.md
asset_paths=()
asset_licences=()
asset_names=()

if [ ! -f "$spec" ]; then
    fail "$spec is missing, so the assets under web/public/ have no licence record"
else
    # §8.4 table rows read: | Asset | Path | Size | Licence | Loader |
    while IFS='|' read -r _ asset path size licence loader _; do
        asset=$(trim "$asset")
        path=$(trim "$path")
        path=${path//\`/}
        licence=$(trim "$licence")
        case "$path" in 'Path' | -*) continue ;; esac # header and |---| separator
        case "$licence" in
            generated* | trained* | 'SIL OFL'* | CC0* | 'repo licence'*) ;;
            *)
                fail "$spec §8.4: '$asset' is licensed '$licence', which this repo may not ship
      (allowed: generated | trained in-repo | SIL OFL | CC0 | repo licence)"
                ;;
        esac
        asset_paths+=("$path")
        asset_licences+=("$licence")
        asset_names+=("$asset")
    done < <(awk '/^### 8\.4 /{inside=1;next} inside&&/^### /{exit} inside' "$spec" | grep '^|')

    if [ "${#asset_paths[@]}" -eq 0 ]; then
        fail "$spec §8.4: no asset rows parsed, so nothing was actually checked"
    fi

    for i in "${!asset_paths[@]}"; do
        case "${asset_paths[$i]}" in web/public/*) ;; *) continue ;; esac
        # §8.4 lists the NPE model as optional, so it is legitimately absent
        # until trained; absence there is not a licensing problem.
        case "${asset_names[$i]}" in *optional*) continue ;; esac
        [ -e "${asset_paths[$i]}" ] ||
            fail "$spec §8.4 lists '${asset_paths[$i]}', which does not exist"
    done

    if [ -d web/public ]; then
        while IFS= read -r file; do
            covered=""
            for i in "${!asset_paths[@]}"; do
                p=${asset_paths[$i]}
                case "$p" in
                    */) [[ "$file" == "$p"* ]] && covered=${asset_licences[$i]} ;;
                    *) [[ "$file" == "$p" ]] && covered=${asset_licences[$i]} ;;
                esac
                [ -n "$covered" ] && break
            done
            [ -n "$covered" ] ||
                fail "'$file' has no entry in $spec §8.4: add it to the asset inventory
      with its licence, or it does not belong in the shipped bundle"
        done < <(find web/public -type f -print | sort)
    fi
fi

# --- Verdict ----------------------------------------------------------------

if [ "$failures" -gt 0 ]; then
    printf '\n%d licensing problem(s). See THIRD-PARTY-NOTICES.md, %s and %s §8.4.\n' \
        "$failures" "$registry" "$spec" >&2
    exit 1
fi

printf 'licensing: LICENSE, %d crate manifests, npm manifests, %d registered fixture(s) and %d inventoried asset row(s) accounted for\n' \
    "$(printf '%s\n' crates/*/Cargo.toml | wc -l)" "${#origin_of[@]}" "${#asset_paths[@]}"
