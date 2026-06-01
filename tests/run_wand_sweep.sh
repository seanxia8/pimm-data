#!/bin/bash
# Run the LUCiD test suite against every WAND config in parallel.
#
# Usage:  bash tests/run_wand_sweep.sh [N_PARALLEL]   (default 6)
#
# Each config is an independent pytest process pointed via LUCID_DATA_ROOT
# at /sdf/data/neutrino/cjesus/DORAEMON/WAND/SK_like/config_NNNNNN/. Configs are
# independent (no shared fixture state) so processes run safely concurrent.
# Practical floor on wall time is the longest-single-config time
# (~40 s for config_000003); past ~6 parallel workers FS read bandwidth
# saturates and there's no further speedup on this volume.
set -euo pipefail

cd "$(dirname "$0")/.."        # repo root

WAND_ROOT="${WAND_ROOT:-/sdf/data/neutrino/cjesus/DORAEMON/WAND/SK_like}"
J="${1:-6}"

if [[ ! -d "$WAND_ROOT" ]]; then
  echo "WAND_ROOT not found: $WAND_ROOT" >&2
  exit 1
fi

CONFIGS=()
for d in "$WAND_ROOT"/config_*/; do
  CONFIGS+=("$(basename "$d")")
done
[[ ${#CONFIGS[@]} -gt 0 ]] || { echo "no configs under $WAND_ROOT" >&2; exit 1; }

OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
START=$(date +%s)
echo "Sweeping ${#CONFIGS[@]} configs from $WAND_ROOT  (-j$J)"

printf '%s\n' "${CONFIGS[@]}" | xargs -I{} -P "$J" bash -c '
  cfg="$1"; out="$2"; root="$3"
  start=$(date +%s)
  result=$(LUCID_DATA_ROOT="$root/$cfg" \
    pytest tests/test_lucid.py -q 2>&1 | tail -1)
  end=$(date +%s)
  printf "%-15s %5ds  %s\n" "$cfg" "$((end-start))" "$result" >> "$out/results"
' _ {} "$OUT" "$WAND_ROOT"

WALL=$(( $(date +%s) - START ))
echo
sort "$OUT/results"
echo
echo "Wall: ${WALL}s with -j$J across ${#CONFIGS[@]} configs"

# Exit non-zero if any config reported failures.
if grep -qE 'failed|error' "$OUT/results"; then
  echo "FAIL: one or more configs had failures" >&2
  exit 1
fi
