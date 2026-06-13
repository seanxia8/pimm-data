# Challenge campaign — dataset × task → config

The single cross-dataset matrix of the ML **challenges** we care about and the
data-layer recipe each needs. It consolidates the two per-detector task tables
that already existed — LUCiD's `docs/LUCID_DATASET.md` §"Tasks → files" and
pimm's `docs/DETECTOR_DATASET.md` §"Task → config" — and extends them to the
new `OpticalDataset`. It lives here because **pimm-data is the one repo that
owns all three datasets** (`JAXTPCDataset`, `LUCiDDataset`, `OpticalDataset`) and
the modality/transform vocabulary.

**Goal:** drive a campaign that builds the runnable training config for every
row below. The configs themselves live in **`particle-imaging-models/configs/`**
(pimm owns models + training); this doc is the tracker (each row carries its
config path + status).

## How to read a row

Every challenge is `Dataset(modalities=…, labels=…) → transforms → Collect`:

- **modalities** — the tuple passed to the dataset. Vocabulary: `step` (3D truth
  deposits / segments — LUCiD calls these `edep`), `sensor` (raw readout), `hits`
  (per-particle decomposition of `sensor`; the **instance**-bearing modality),
  `labl` (label dimension tables, requested via `labels=`).
  - The LUCiD source doc's columns map as **`inst → hits`, `seg → step`**.
- **labels** — `labels='pdg'|'cluster'|'interaction'|'ancestor'` (JAXTPC) /
  `labels=True` (LUCiD) attaches `segment`/`instance` to `step`/`hits`.
  `RemapSegment(scheme='motif_5cls')` turns raw `pdg` into dense class indices.
- **Collect target** — the flat-prefixed batch the model consumes
  (`step_coord`/`step_segment`, `sensor_*`, …); SSL uses `MultiCrop`.
- **status** — `exists` (config in pimm), `planned` (to build).

---

## JAXTPC (LArTPC; wire/pixel auto-detected)

| Challenge | modalities | labels | key transforms | Collect | config | status |
|---|---|---|---|---|---|---|
| 3D semantic seg (5-class motif) | `('step',)` | `'pdg'` | `RemapSegment(motif_5cls)`, `GridSample`, `RandomRotate/Flip` | `step`: coord/grid_coord/segment, feat=coord+energy | **recipe** `configs/jaxtpc/semseg_5cls.py` (pimm training cfg: `…/semseg-pt-v3m2-jaxtpc-5cls.py`) | recipe ✓ |
| 3D instance seg | `('step',)` | `'cluster'` or `'ancestor'` | `GridSample` | `step`: coord/segment/instance | — | planned |
| 3D self-supervised (SSL) | `('step',)` | — | `MultiCrop` (global/local) | `step`: coord, feat | — | planned |
| Interaction classification/grouping | `('step',)` | `'interaction'` | `GridSample` | `step`: coord/segment | — | planned |
| Raw-readout SSL | `('sensor',)` | — | `MultiCrop` | `sensor`: coord/feat | — | planned |
| Instance seg on hits | `('hits',)` | `'cluster'` | `GridSample` | `hits`: coord/segment/instance | — | planned |
| sensor → step charge/energy recon | `('step','sensor')` | `'pdg'` (opt) | per-modality `Apply` | `sensor` in, `step` target | — | planned |

## LUCiD (Water Cherenkov; PMT)

Direct from `LUCID_DATASET.md` §"Tasks → files" (`inst→hits`, `seg→step`):

| Challenge | modalities | labels | Collect | config | status |
|---|---|---|---|---|---|
| SSL on raw PMT readout | `('sensor',)` | — | `sensor`: coord/feat (`MultiCrop`) | `configs/lucid/pretrain/pretrain-sonata-v1m1-sk-like-mu-e.py` | exists |
| SSL on per-particle decomposition | `('hits',)` | — | `hits`: coord/feat | — | planned |
| SSL on 3D segments | `('step',)` | — | `step`: coord/feat | — | planned |
| Per-segment Cherenkov forward sim | `('step',)` | — | `step`: coord + physics (beta/n_cherenkov) | — | planned |
| sensor → inst denoising/deconv | `('sensor','hits')` | — | `sensor` in, `hits` target | — | planned |
| sensor → seg recon (vertex/energy/dir) | `('sensor','step')` | — | `sensor` in, `step` target | — | planned |
| Per-PMT semantic/instance seg | `('hits',)` | `True` | `hits`: coord/grid_coord/segment/instance, feat=coord+energy+time | **recipe** `configs/lucid/perpmt_seg_hits.py` | recipe ✓ |
| 3D semantic/instance seg on segments | `('step',)` | `True` | `step`: coord/segment/instance | — | planned |
| Event class/regression (E, dir, vertex) | `('sensor','hits')` | `True` | event-level target + `sensor`/`hits` | — | planned |
| Containment-filtered training | (any) | `True` | + `min_segments`/containment filter | — | planned |

## Optical (PMT light; per-chunk waveforms — new)

`OpticalDataset(schema='label'|'east_west')`; each `sensor` row is a waveform
chunk, `instance` = group (interaction for `label`, side for `east_west`), the
packed `adc` is the second row-space. (Targets here are new — not in the source
docs.)

| Challenge | schema | modalities | per-chunk target | Collect | status |
|---|---|---|---|---|---|
| Interaction/operator discrimination | `label` | `('sensor',)` | `instance` (interaction) | `sensor`: pmt_id/t0_ns/length/pe/instance/adc | recipe ✓ `configs/optical/interaction_discrimination.py` |
| Per-channel PE regression | `label` | `('sensor',)` | `pe` | `sensor`: … + pe target | planned |
| Waveform SSL / pretraining | `label`/`east_west` | `('sensor',)` | — | `sensor`: adc (+ wave_offset) | planned |
| Waveform denoising / compression | `east_west` | `('sensor',)` | clean/coeffs | `sensor`: adc | planned |
| Side-aware readout (east/west) | `east_west` | `('sensor',)` | — | `sensor`: … + instance(side) | planned |

---

## Building the configs

These configs are **data-loader specs** — the point is to set up each
challenge's data loading *exactly*, not the model. So the campaign is built in
two layers:

1. **Recipe (pimm-data `configs/<dataset>/<challenge>.py`)** — the data-loading
   half only, in the new flat-prefixed API (`Apply(on=)` + `Collect(modalities=)`).
   It lives here because the new API + `OpticalDataset` exist only in pimm-data
   (not yet in pimm's pinned submodule), so a recipe is **verifiable now**:
   `tests/test_campaign_configs.py` execs each recipe against the synthetic
   fixtures and asserts the challenge's flat keys. `status = recipe ✓`.
2. **Training config (`particle-imaging-models/configs/`)** — lifts a recipe's
   `data`/`transform` block and adds the model/optimizer/hooks half, following
   the existing naming (`<task>-<backbone>-<dataset>-<variant>.py`). Needs pimm's
   `libs/pimm-data` submodule bumped to the redesign SHA first.

Built so far (representative-first, one per dataset): JAXTPC semseg, LUCiD
per-PMT seg on hits, Optical interaction discrimination — all `recipe ✓`.

### Findings / follow-ups

- **LUCiD raw-sensor SSL** is not new-API-clean yet: `AggregateSensorHits` writes
  the aggregated point cloud to the **top level** and drops the sub-dict (legacy
  flatten), but the new `MultiCrop(on='sensor')` + `Collect(modalities=)` flow
  needs it kept **nested** in the `sensor` sub-dict. Add a nested write-back mode
  to `AggregateSensorHits` before building the SSL recipes (`MultiViewGenerator`
  → `MultiCrop` is otherwise a direct swap).
- **`coord` comes out float64** after `NormalizeCoord` (python-float scale). Cast
  to float32 in the recipe if the target model is strict (feat is already
  float32 via `feat_keys`).
- **Optical needs a waveform/sequence model** (PT-v3 is point-cloud only); the
  optical recipes are data-half + placeholder model until one exists.

Update the `config`/`status` cell as each is built so this stays the campaign's
source of truth.
