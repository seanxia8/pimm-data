#!/usr/bin/env bash
# Build a symlink mirror of the WAND dataset that exposes the LUCiD 3D-segment
# modality under the name pimm-data's reader expects: `step/wc_step_*.h5` instead
# of WAND's `edep/wc_edep_*.h5`. sensor/hits/labl are dir-symlinked through.
#
# Rationale: the LUCiD step reader globs `step/{name}_step_*.h5`, but WAND ships
# the 3D segments as `edep/wc_edep_*.h5`. This mirror lets the LUCiD step recipes
# (ssl_step / seg_step / recon_sensor_to_step) load real WAND with no reader
# change. (Alternative: add an `edep` dir-name option to LUCiDStepReader.)
#
# Usage:  scripts/make_wand_step_mirror.sh [SRC] [DST]
set -euo pipefail
SRC="${1:-/sdf/data/neutrino/cjesus/DORAEMON/WAND/SK_like}"
DST="${2:-/sdf/data/neutrino/omara/wand_sk_like_step}"

mkdir -p "$DST"
n=0; c=0
cd "$SRC"
for cfg in config_*; do
  [ -d "$cfg/edep" ] || continue
  c=$((c + 1))
  mkdir -p "$DST/$cfg/step"
  for sub in sensor hits labl; do
    [ -e "$SRC/$cfg/$sub" ] && ln -sfn "$SRC/$cfg/$sub" "$DST/$cfg/$sub"
  done
  for base in $(ls "$cfg/edep" | grep -E '^wc_edep_[0-9]+\.h5$'); do
    idx="${base#wc_edep_}"
    ln -sfn "$SRC/$cfg/edep/$base" "$DST/$cfg/step/wc_step_$idx"
    n=$((n + 1))
  done
done
echo "built $DST : $n step links across $c configs"
