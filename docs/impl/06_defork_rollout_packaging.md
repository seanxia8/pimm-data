# Part 06 — De-fork rollout, packaging & migration (implementation spec)

**Status:** final implementation spec before coding the de-fork. This is a
**runbook** — file-by-file actions, then ordered steps with per-step gate +
rollback note, then the exact shim / config rewrites / packaging edits / the
pilarnet Rb merge. Every action is grounded in code with `file:line`.

**Source decisions:** `engagement_plan_transform_dataset_placement.md` Part VIII
(D5 de-fork bucket in parallel; D17 de-fork replaces MOST of `pimm/datasets/`;
D18 de-fork boundary — all of `pimm/datasets/` → pimm-data, pimm keeps
`MultiDatasetDataloader` + model/hook registries + hooks + thin `__init__` shim;
D23 namespaced collate is GAIN-not-REPLACE and FUTURE; D32 aggregation deferred;
D33 rollout order build-behind→parity→shim→delete-vendored + the Ra/Rb gates;
**D43 — Phase A (cross-modality joint-index bug fix, A1–A5) lands FIRST as a
standalone PR on `jaxtpc-loader-codec-opt` BEFORE the de-fork, patching the current
`src/pimm_data/jaxtpc.py` which the de-fork KEEPS; the base then factors A2 up;
D42/D44/D46/D47 define that fix**) and
Part IX inventory + risks Ra/Rb/Rf; `implementation_plan_pimm_data_datalayer.md`
§2 (rollout Steps 0–5), §4 (config migration Ra), §5 (packaging + Rb);
`shard_event_filtering_handoff.md` §4/§5 (the desync + Phase A).

**Files this part touches (pimm side, on branch `research`):**
- `pimm/datasets/__init__.py` — REPLACE with the re-export/re-register shim (§5).
- `configs/detector/_base_/jaxtpc_seg.py` + `configs/detector/semseg/semseg-pt-v3m2-jaxtpc-5cls.py` — migrate off `seg/resp/corr` (§6, Ra).
- `scripts/train.sh:228` — snapshot the submodule (§7).
- `environment.yml` — pip-install the submodule editable (§7).
- DELETE (Step 5): `pimm/datasets/{anchors,builder,defaults,detector_transforms,jaxtpc_dataset,lucid_dataset,lucid_event_ssl,pilarnet,transform,utils}.py`, `pimm/datasets/readers/`, and `pimm/datasets/preprocessing/` if data-layer-only.

**Files this part touches (pimm-data side):**
- `src/pimm_data/jaxtpc.py` — **Phase A** (D43): A1 `_read_shard_meta` lru_cache,
  A2 joint index, A3 volume-aware `min_deposits` + raise, A4 length-mismatch +
  `strict_lengths` (§4 Phase A). Lands FIRST, standalone PR, before Step 0.
- `src/pimm_data/readers/jaxtpc_step.py` — Phase A A3: volume-aware `n_actual` sum
  (`jaxtpc_step.py:91-97` is volume-blind today).
- `src/pimm_data/pilarnet.py` — MERGE v2→v3 (§8, Rb).
- `pyproject.toml` — pin `torch` (§7).

> **Constants verified at write time** (do not re-derive — re-check only if the
> trees moved): pimm `research` HEAD `ffb0823` (`git rev-parse HEAD`); pimm-data
> `Collect.__init__(self, keys, offset_keys_dict=None, stream=None, **kwargs)`
> at `src/pimm_data/transform.py:108`; `train.sh:228` is exactly
> `  cp -r scripts tools pimm "$CODE_DIR" 2>/dev/null`; `pimm/datasets/__init__.py:9`
> is exactly `from .lucid_dataset import LUCiDDataset`.

---

## 1. Purpose & scope

Move the entire data layer out of the pimm Pointcept fork and into pimm-data
**without ever breaking running training** — the invariant that gates every step
is **"PILArNet/panda/hmae training works."** The mechanism (D33): build the new
code **behind** the existing vendored datasets in pimm-data, prove **parity** on a
synthetic test matrix (Part 07), drop a thin **re-export/re-register shim** into
`pimm/datasets/__init__.py` so configs resolve pimm-data classes through pimm's
registries, **flip** transforms then PILArNet then JAXTPC/LUCiD configs one group
at a time, and **delete the vendored files last** in a single commit that a single
`git revert` fully restores.

In scope for Part 06:
1. The **file-by-file REPLACE/MERGE/KEEP/DISSOLVE/NEW action table** (§3).
2. The **ordered rollout runbook**, Steps 0–5, with per-step actions, GATE,
   rollback-safety, and the invariant check (§4).
3. The **exact `pimm/datasets/__init__.py` shim** — what it imports, what it
   re-registers, what stays authored in pimm (§5).
4. **Config migration (Ra)** — the concrete `jaxtpc_seg.py` rewrite and the
   `__init__.py:9` stale-import fix, plus the grep proof PILArNet/panda are
   unaffected (§6).
5. **Packaging** — submodule layout, `environment.yml` pip line, `train.sh:228`
   snapshot edit, torch pin, registry de-dup (Rf) (§7).
6. The **pilarnet Rb merge** with both signatures quoted (§8).

Out of scope (owned elsewhere, only sequenced here): the transform merges
(`01_transforms.md`), the base (`02_dataset_base.md`), readers (`03_readers.md`),
label decoration (`04_label_decoration.md`), collate/eval (`05_collate_streams_eval.md`),
the Step-0 fixtures/harness (`07_test_matrix_fixtures.md`). The **namespaced
multi-stream collate (D23) is GAIN-not-REPLACE and FUTURE** — not built here; the
near-term collate is a byte-identical REPLACE (`05_collate_streams_eval.md` §3.1).
**Track B (densify/noise, D1/D33)** is JAXTPC-only and deferred — Part 06 only
records that the de-fork does not block it.

---

## 2. Current state (the two forks; the only live coupling; file:line)

**Two trees with a near-duplicate data layer:**

- **pimm** (`research`, HEAD `ffb0823`): `pimm/datasets/` holds the *live* data
  layer Pointcept training imports. `__init__.py` (13 lines) wires it:
  ```
  pimm/datasets/__init__.py:1  from .defaults import DefaultDataset, ConcatDataset
  pimm/datasets/__init__.py:2  from .builder import build_dataset
  pimm/datasets/__init__.py:3  from .utils import point_collate_fn, collate_fn, inseg_collate_fn
  pimm/datasets/__init__.py:7  from .pilarnet import PILArNetH5Dataset
  pimm/datasets/__init__.py:8  from .jaxtpc_dataset import JAXTPCDataset
  pimm/datasets/__init__.py:9  from .lucid_dataset import LUCiDDataset      ← STALE (Ra)
  pimm/datasets/__init__.py:10 from . import detector_transforms  # register PDGToSemantic
  pimm/datasets/__init__.py:12 from .dataloader import MultiDatasetDataloader
  ```
- **pimm-data** (`/sdf/group/neutrino/omara/pimm-data`): the rebuilt layer —
  `src/pimm_data/{defaults,builder,collate,transform,detector_transforms,anchors,pilarnet,jaxtpc,lucid}.py`,
  `src/pimm_data/readers/` (8 readers), `src/pimm_data/_registry.py` (vendored
  Registry), `src/pimm_data/utils/`, `src/pimm_data/testing.py`. `__init__.py`
  registers everything via import side-effects (`src/pimm_data/__init__.py:16-36`).

**The only live coupling today** is the `lucid_event_ssl.py` bridge — it is the
*sole* place pimm already calls into pimm-data:
```
pimm/datasets/lucid_event_ssl.py:23  def _load_pimm_data_lucid_dataset():
pimm/datasets/lucid_event_ssl.py:25      from pimm_data import LUCiDDataset
pimm/datasets/lucid_event_ssl.py:31  @DATASETS.register_module()
pimm/datasets/lucid_event_ssl.py:32  class LUCiDEventSSLDataset(Dataset):
```
This module is **registered by import side-effect only** and **is not imported by
`pimm/datasets/__init__.py`** (grep confirms `lucid_event_ssl` appears nowhere in
`__init__.py`). It is pulled in transitively by the LUCiD SSL config
`configs/lucid/pretrain/pretrain-sonata-v1m1-sk-like-mu-e.py` (the only config that
references `LUCiDEventSSLDataset`). The `register_module(module=...)` pattern it
already uses for re-registering a pimm-data class (`pimm/datasets/lucid_event_ssl.py:31`
decorating a class that wraps `pimm_data.LUCiDDataset`) is the **proof-of-pattern**
the shim generalizes (Rf).

**Registry boundary (Rf).** pimm and pimm-data each have their **own** `Registry`
object — `pimm/utils/registry.py:59` `class Registry` (mmcv-derived) and
`src/pimm_data/_registry.py:66` `class Registry` (vendored, byte-compatible
`register_module(name=None, force=False, module=None)` at `_registry.py:185`). The
config builder (`tools/train.py` → pimm's `build_dataset`) resolves `type=` strings
against **pimm's** `DATASETS`/`TRANSFORMS`. So pimm-data classes must be
**re-registered into pimm's registries** — not shared by object identity. The
duplicate `DefaultDataset`/`ConcatDataset`/`PDGToSemantic`/`Collect` names across
the two registries are the double-registration hazard the shim must guard (§5,
`force=True` or membership-guard).

**Risk Ra (config breakage on flip).** Only two configs reference old machinery:
- `configs/detector/_base_/jaxtpc_seg.py:72` and `:82` — `modalities=("seg",)`.
  The new `JAXTPCDataset` (pimm-data) takes `modalities` ∈ `{'step','sensor','hits','labl'}`
  (`src/pimm_data/jaxtpc.py:99,202`); `"seg"` is invalid → `ValueError` at
  construction.
- `configs/detector/semseg/semseg-pt-v3m2-jaxtpc-5cls.py:9` — `_base_` includes
  `../_base_/jaxtpc_seg.py`, so it inherits the break.
- `output_mode` has **zero** config hits (grep clean).
- `__init__.py:9` stale `from .lucid_dataset import LUCiDDataset` + the unimported
  `lucid_event_ssl` registration.

**Risk Rb (pilarnet drift).** pimm `pilarnet.py` is **v3** (`revision` includes
`"v3"`, 6-wide `cluster_extra`, `is_primary`, shared-`rotations` overlay);
pimm-data `pilarnet.py` is **v2** (no `v3` branch). §8 merges v3 into pimm-data.

**Risk Rf (registry/logger re-export).** `defaults.py`/`builder.py`/
`detector_transforms.py` are adapter/superset-equivalent, NOT byte-identical: pimm
versions import `from pimm.utils.logger import get_root_logger` and
`from pimm.utils.cache import shared_dict` (`pimm/datasets/defaults.py:18-19`);
pimm-data versions use stdlib `logging` and a vendored cache. The shim re-exports
pimm-data's classes into pimm's registries; it does not try to reconcile the logger.

---

## 3. File-by-file action table (REPLACE/MERGE/KEEP/DISSOLVE/NEW)

Action verbs: **REPLACE** = pimm file deleted, pimm-data file is authoritative,
shim re-registers; **MERGE** = port a delta into pimm-data, then REPLACE;
**KEEP** = stays authored in pimm (data layer does not own it); **DISSOLVE** =
class goes away, behavior absorbed into the base + a config; **NEW** = pimm-data
gains it (built in Parts 01–05, listed here for completeness).

| pimm file (`research`) | Action | pimm-data target | Reason / `file:line` |
|---|---|---|---|
| `pimm/datasets/anchors.py` | **REPLACE (byte-identical)** | `src/pimm_data/anchors.py` | `diff -q` clean (verified `ANCHORS IDENTICAL`). No merge. |
| `pimm/datasets/utils.py` (collate) | **REPLACE (byte-identical)** | `src/pimm_data/collate.py` | `collate_fn`/`point_collate_fn`/`inseg_collate_fn` byte-identical (verified `diff`). D23 namespaced collate is GAIN/FUTURE, NOT here. (`05_collate_streams_eval.md` §3.1) |
| `pimm/datasets/builder.py` | **REPLACE via shim** | `src/pimm_data/builder.py` | Adapter-equivalent: both define `DATASETS` + `build_dataset`. pimm-data: `src/pimm_data/builder.py:5,8`. The shim re-exports `build_dataset` so configs that import it from pimm still resolve. |
| `pimm/datasets/defaults.py` | **REPLACE via shim (Rf)** | `src/pimm_data/defaults.py` | Superset-equivalent; pimm imports `get_root_logger`/`shared_dict` (`pimm/datasets/defaults.py:18-19`), pimm-data uses stdlib. Re-register `DefaultDataset`/`ConcatDataset`. `TestModeMixin` extraction is in 02. |
| `pimm/datasets/detector_transforms.py` | **REPLACE via shim (Rf)** | `src/pimm_data/detector_transforms.py` | pimm has only `PDGToSemantic` (`pimm/.../detector_transforms.py:15`); pimm-data adds **`ApplyToStream`** (`src/.../detector_transforms.py:27`) + **`RemapSegment`** (`:133`) needed by the migrated `jaxtpc_seg.py` (§6). 6 classes vs 1 (verified). |
| `pimm/datasets/transform.py` | **MERGE delta → REPLACE** | `src/pimm_data/transform.py` | Delta owned by 01: `RelativeLogNormalize`, `GridSample` reducers, `LogTransform.clip`, `get_view` guard, **v3 vertex/`is_primary` plumbing**, `MixedScaleGeometryMultiViewGenerator`. pimm has them (`pimm/.../transform.py:31,42,278,1195,1682`); pimm-data does **not** yet (grep empty). **Do NOT overwrite pimm-data's `Collect`** — it is ahead (`stream=` at `src/.../transform.py:108`). |
| `pimm/datasets/pilarnet.py` | **MERGE v2→v3 (Rb) → REPLACE** | `src/pimm_data/pilarnet.py` | §8. pimm is v3, pimm-data v2. |
| `pimm/datasets/jaxtpc_dataset.py` | **REPLACE** (Ra-gated) | `src/pimm_data/jaxtpc.py` | Old uses `seg/resp/corr/labl` modalities (`pimm/.../jaxtpc_dataset.py:5,18,37-40`); new uses `step/sensor/hits/labl` (`src/pimm_data/jaxtpc.py:99`). Migrate `jaxtpc_seg.py` first (§6). |
| `pimm/datasets/lucid_dataset.py` | **REPLACE** | `src/pimm_data/lucid.py` | New nested LUCiD (`src/pimm_data/lucid.py:62`). The stale `__init__.py:9` import is the fix-first item (§6). |
| `pimm/datasets/lucid_event_ssl.py` | **DISSOLVE** → base + config | `src/pimm_data` `MultiModalEventDataset` (02) + LUCiD SSL config | `LUCiDEventSSLDataset` (`pimm/.../lucid_event_ssl.py:32`) folds into the base (holdout `_split_indices:189`, min-points `_event_point_count:254`, sensor aggregation `_aggregate_hits:268`). Config rewrite at Step 4. Registration must move into the shim (it's never imported by `__init__.py` today). |
| `pimm/datasets/readers/` (`jaxtpc_seg_reader.py`, `jaxtpc_resp_reader.py`, `jaxtpc_corr_reader.py`, `jaxtpc_labl_reader.py`, `lucid_seg_reader.py`, `lucid_sensor_reader.py`) | **REPLACE** | `src/pimm_data/readers/` (8 new) | Old `seg/resp/corr` readers superseded by `jaxtpc_{step,sensor,hits,labl}` + `lucid_{step,sensor,hits,labl}` (03). |
| `pimm/datasets/dataloader.py` | **KEEP-IN-PIMM** | — | `MultiDatasetDataloader` (`pimm/.../dataloader.py:23`) imports `pimm.utils.comm` (DDP) and `pimm.utils.env.set_seed` (`:6,9`). Trainer-side; D18. |
| `pimm/utils/registry.py` | **KEEP-IN-PIMM** | — | pimm's authoritative `DATASETS`/`TRANSFORMS` for config lookup (Rf). pimm-data has its own (`_registry.py`). Not shared. |
| `pimm/datasets/__init__.py` | **REPLACE with shim** | — | §5. The one file rewritten (not deleted) on the pimm side. |
| — | **NEW** | `src/pimm_data/multimodal.py` (`MultiModalEventDataset`) | 02. Listed for completeness. |

---

## 4. Ordered rollout runbook (Steps 0–5)

**Invariant, checked at every step: PILArNet/panda/hmae training works.** Each
step is one landable commit (or a small group), independently revertable.

### Phase A — Cross-modality joint-index bug fix (lands FIRST, standalone PR, D43)

**This precedes Step 0.** It is a **bug fix**, not part of the de-fork — independently
valuable and the prerequisite the de-fork's base factors up (D42/D43). It patches the
**current** `src/pimm_data/jaxtpc.py` (the file the de-fork KEEPS — only the pimm-side
vendored copy is deleted at Step 5), on branch `jaxtpc-loader-codec-opt`.

**Actions** (`shard_event_filtering_handoff.md` §5 Phase A):
- **A1.** Module-level `@lru_cache _read_shard_meta(path) -> (n_events, n_volumes,
  present_event_keys, readout_type)` (~20 LOC, no API change) — collapses the ~3×
  redundant per-shard opens (4 readers × ~800 doraemon shards). Highest ROI / lowest
  risk.
- **A2.** Joint event index at the **dataset** level: intersect present `event_*` keys
  across loaded modalities (keyed on `source_event_idx`), hand per-shard index arrays to
  every reader; replace `_n_events=min(...)` (`jaxtpc.py:180`) + the shared `local_idx`
  (`get_data`, `jaxtpc.py:233-269`) with the intersected map. Readers fall back to their
  own `_find_files`/`_build_index` when the kwargs are absent (backward compat).
- **A3.** Make `min_deposits` volume-aware when `volume=N` set (`jaxtpc_step.py:91-97` is
  volume-blind today); **raise** if `min_deposits>0`/`min_segments>0` is passed without
  the source modality loaded.
- **A4.** Length-mismatch handling: warn with concrete per-modality counts instead of the
  silent `min(...)`; `strict_lengths=True` hard-errors.
- **A5.** Cross-modality regression test (`modalities=('step','sensor','hits','labl')`,
  `min_deposits>0`: same physics event across every modality for every idx; plus a
  gap-in-one-modality variant). **Both fail on `master`/HEAD today.** **Folded into the
  Step-0 test matrix** (Part 07 / Part 02 §6.16).

**GATE.** A5 green (both variants); existing `tests/test_jaxtpc_robustness.py` still green.

**Rollback-safety.** Standalone PR on the current `jaxtpc.py`; revert the PR — zero
de-fork coupling (the de-fork has not started). **Not absorbed by Step 1**: the base
factors A2 up into `_build_joint_index` (Part 02 §3.3a) afterward, reusing this logic.

**Invariant check.** Patches `jaxtpc.py` only; PILArNet/panda/hmae do not import it.

### Step 0 — Test matrix (gate, no code move)

**Actions.** Stand up the parity/determinism harness in pimm-data on
`src/pimm_data/testing.py` synthetic fixtures (Part 07; master §6). No pimm change,
no flip. Branch-parity assertions gate on the colleague's `research` branch being
fetchable (HEAD `ffb0823`); placeholder golden arrays where not.

**GATE.** Harness stands up green on fixtures (parity skips cleanly if the branch
is absent).

**Rollback-safety.** Nothing in pimm changed — trivially safe.

**Invariant check.** No pimm import path touched ⇒ PILArNet/panda/hmae untouched.

### Step 1 — Additive build in pimm-data (no pimm change)

**Actions.** Land Parts 01–05 *behind* the existing vendored datasets, in spine
order (01 transforms incl. the Rb-relevant v3 plumbing + prefix-match; 03 readers;
02 base + `TestModeMixin`; 04 label decoration; 05 collate-confirm + eval-rewire +
seams). All additive in `src/pimm_data/`. **Do the Rb pilarnet v2→v3 merge (§8) as
part of this step** so PILArNet is flip-ready before Step 3.

**GATE.** Step-0 suite green for each landed part (each part's §6).

**Rollback-safety.** All changes are inside pimm-data; pimm still imports its own
vendored layer. Revert any pimm-data commit with zero pimm impact.

**Invariant check.** pimm `__init__.py` still imports the vendored
`pilarnet`/`jaxtpc_dataset`/`lucid_dataset` — training is byte-for-byte unchanged.

### Step 2 — Re-export shim in `pimm/datasets/__init__.py`

**Actions.** Replace `pimm/datasets/__init__.py` with the shim (§5): import each
pimm-data class and **re-register** it into pimm's `DATASETS`/`TRANSFORMS` via
`register_module(module=Cls)`, guarding double-registration. Start with the
**byte-identical REPLACE** files (`anchors`, collate, `builder`, `defaults`) +
`PILArNetH5Dataset` so the lowest-risk surface flips first. Keep re-exporting the
trainer names (`point_collate_fn`/`collate_fn`/`inseg_collate_fn`/`DefaultDataset`/
`ConcatDataset`/`build_dataset`/`MultiDatasetDataloader`).

**GATE.** PILArNet 1-step smoke through the shim (a `pretrain`/`semseg` config
builds and runs one training step).

**Rollback-safety.** Single-file change; `git checkout pimm/datasets/__init__.py`
restores the vendored wiring. The vendored `.py` files are still present (not yet
deleted), so reverting the shim is total.

**Invariant check.** The smoke run is the check — PILArNet builds + steps.

### Step 3 — Flip transforms + PILArNet

**Actions.** With the Rb merge (§8) already landed in Step 1, point the shim's
`TRANSFORMS` re-registration at pimm-data's `transform.py` + `detector_transforms.py`
and `DATASETS` at pimm-data's `PILArNetH5Dataset`. PILArNet/panda/hmae/voltmae/
polarmae/lejepa configs are transform-compatible: they consume `segment_motif`/
`segment_pid`/`instance_particle` via `Copy` (e.g.
`configs/panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-fft.py:132`
`Copy(keys_dict={"segment_motif":"segment"})`), all emitted by `pilarnet.py`.

**GATE.** **Identical first-batch tensors vs vendored** — seed `random`/`np.random`/
`torch` identically, build the same config against vendored vs shimmed layer,
`assert_array_equal` per key on batch 0 (master §6 / 07).

**Rollback-safety.** Shim-only change; revert the shim hunk. Vendored files intact.

**Invariant check.** First-batch parity is the invariant, made literal.

### Step 4 — Migrate JAXTPC configs (Ra), flip `JAXTPCDataset`, dissolve `LUCiDEventSSLDataset`

**Actions.**
1. **Fix `__init__.py:9`** (Ra): the shim no longer imports a `lucid_dataset`
   module; it imports `LUCiDDataset` from pimm-data and re-registers, and it
   **registers the `LUCiDEventSSLDataset` successor** (which the old `__init__`
   never imported — §2).
2. **Migrate `configs/detector/_base_/jaxtpc_seg.py`** off `seg` → `step`/`labl`
   (§6) and confirm the `semseg-pt-v3m2-jaxtpc-5cls.py` child still builds (no model
   change; `in_channels=4`).
3. **Flip `JAXTPCDataset`** in the shim to pimm-data's `jaxtpc.py`.
4. **Dissolve `LUCiDEventSSLDataset`**: rewrite
   `configs/lucid/pretrain/pretrain-sonata-v1m1-sk-like-mu-e.py` to use the
   `MultiModalEventDataset` base (02) with a LUCiD config (the base owns the
   holdout/min-points/aggregation `LUCiDEventSSLDataset` did inline).

**GATE.** JAXTPC semseg config (`semseg-pt-v3m2-jaxtpc-5cls.py`) **and** the LUCiD
SSL config each **build and run 1 step**; the eval probe's `event_identity`
disjointness guard passes (05 §3.4).

**Rollback-safety.** Config edits + shim edit. Revert the config commits and the
shim hunk; vendored files still present so JAXTPC/LUCiD fall back.

**Invariant check.** PILArNet/panda/hmae are not touched by the JAXTPC/LUCiD config
edits (they share no `_base_`; §6 grep proof) — they keep stepping.

### Step 5 — Delete vendored files

**Actions.** Delete the vendored data-layer files in **one commit** (the §3
DELETE list). The shim already points everything at pimm-data; this commit removes
the now-dead source so a single `git revert` of *this commit* restores the entire
vendored tree.

**GATE (D33).** All of:
- `grep -rn 'seg\|resp\|corr\|output_mode' configs/` for the **modality** usage is
  clean (only the migrated `step`/`labl` remain; the substring `seg` may legitimately
  appear in `semseg`/`segment` — gate on the *modalities=* and reader usage).
- `jaxtpc_seg.py` migrated (no `modalities=("seg",)`).
- `pimm/datasets/__init__.py` stale `lucid_dataset` import gone.
- Full parity suite green (07).
- ≥1 full PILArNet run has soaked end-to-end.

**Rollback-safety.** **Single `git revert <delete-commit>`** restores every vendored
file; because the shim is a separate earlier commit, reverting the delete alone
brings back the files without un-doing the shim (the shim then harmlessly
re-registers — guarded). This is the "single-revert-safe" property.

**Invariant check.** The soaked PILArNet run is the final invariant evidence.

---

## 5. The `pimm/datasets/__init__.py` shim

The shim **replaces** the 13-line vendored `__init__.py`. It (a) imports the
trainer-facing names from pimm-data and the kept-in-pimm `MultiDatasetDataloader`,
(b) **re-registers** every pimm-data dataset/transform class into **pimm's** own
`DATASETS`/`TRANSFORMS` (Rf — config lookup is against pimm's registries), guarding
double-registration, and (c) re-exports the exact names the trainer/`tools` import.

**Names the trainer needs re-exported** (verified consumers): `point_collate_fn`,
`collate_fn`, `inseg_collate_fn` (used by `pimm/datasets/dataloader.py:7`,
`partial(point_collate_fn, ...)` at `:72`); `DefaultDataset`, `ConcatDataset`
(`ConcatDataset` is imported by `dataloader.py:8`); `build_dataset`;
`MultiDatasetDataloader`; and the dataset classes `PILArNetH5Dataset`,
`JAXTPCDataset`, `LUCiDDataset`, `LUCiDEventSSLDataset`-successor.

**Exact shim:**
```python
# pimm/datasets/__init__.py — de-fork re-export/re-register shim.
#
# The data layer lives in pimm-data (installed as the `libs/pimm-data`
# submodule, editable). pimm keeps only: MultiDatasetDataloader (DDP),
# model/hook registries, hooks/evaluators, and this shim. Config `type=`
# strings resolve against *pimm's* DATASETS/TRANSFORMS (pimm/utils/registry.py),
# so every pimm-data class is RE-REGISTERED here via register_module(module=...)
# — the pattern lucid_event_ssl already used (its old line 31).

# 1. Pull builder + registries from pimm-data; keep pimm's registries authoritative.
from .builder import DATASETS, build_dataset            # pimm's DATASETS (lookup target)
from .transform import TRANSFORMS, Compose              # pimm's TRANSFORMS (lookup target)

# 2. Collate — byte-identical REPLACE; trainer + dataloader import these names.
from pimm_data.collate import collate_fn, point_collate_fn, inseg_collate_fn

# 3. Core dataset bases + detector classes + transforms from pimm-data.
import pimm_data as _pd
from pimm_data import (
    DefaultDataset, ConcatDataset,
    PILArNetH5Dataset, JAXTPCDataset, LUCiDDataset,
)
# anchors (direct-use names some configs/tools reference)
from pimm_data import compute_anchors, ANCHOR_DEFAULT_CFG

# 4. Re-register pimm-data datasets + transforms into PIMM's registries.
#    Guard double-registration: skip if already present, else force-register.
def _reregister(registry, name, cls):
    if name in registry:           # Registry.__contains__ -> get(name) is not None
        return
    registry.register_module(name=name, module=cls)   # _registry-compatible API

for _cls in (DefaultDataset, ConcatDataset,
             PILArNetH5Dataset, JAXTPCDataset, LUCiDDataset):
    _reregister(DATASETS, _cls.__name__, _cls)

# 5. Re-register every pimm-data TRANSFORM (PDGToSemantic, ApplyToStream,
#    RemapSegment, Collect, GridSample, RelativeLogNormalize, …) into pimm's
#    TRANSFORMS. pimm-data's transform module already registered them in ITS
#    registry; copy the class objects across by name.
from pimm_data.transform import TRANSFORMS as _PD_TRANSFORMS
from pimm_data.builder import DATASETS as _PD_DATASETS
for _name, _cls in _PD_TRANSFORMS.module_dict.items():
    _reregister(TRANSFORMS, _name, _cls)
for _name, _cls in _PD_DATASETS.module_dict.items():
    _reregister(DATASETS, _name, _cls)

# 6. DISSOLVED LUCiDEventSSLDataset successor — register so the LUCiD SSL config
#    resolves it (the old __init__ NEVER imported lucid_event_ssl, so this is the
#    fix for the "unimported registration" half of Ra). Until 02 lands the base,
#    this may alias MultiModalEventDataset configured for LUCiD-SSL.
#    NOTE: MultiModalEventDataset is NOT yet exported from pimm_data.__init__
#    (verified: only DATASETS/TRANSFORMS/Compose/build_dataset/DefaultDataset/
#    ConcatDataset/collate fns/dataset classes/compute_anchors are, at
#    src/pimm_data/__init__.py:38-55). Part 02 must add it to __all__; this block
#    lands at Step 4, after 02.
from pimm_data import MultiModalEventDataset            # 02 (add to pimm_data __all__)
_reregister(DATASETS, "LUCiDEventSSLDataset", MultiModalEventDataset)
_reregister(DATASETS, "MultiModalEventDataset", MultiModalEventDataset)

# 7. KEEP-IN-PIMM: DDP dataloader (imports pimm.utils.comm / env).
from .dataloader import MultiDatasetDataloader

__all__ = [
    "DATASETS", "TRANSFORMS", "Compose", "build_dataset",
    "collate_fn", "point_collate_fn", "inseg_collate_fn",
    "DefaultDataset", "ConcatDataset",
    "PILArNetH5Dataset", "JAXTPCDataset", "LUCiDDataset",
    "MultiDatasetDataloader",
    "compute_anchors", "ANCHOR_DEFAULT_CFG",
]
```

**Notes on the shim.**
- Steps 2–4 of §4 land the shim **incrementally**: at Step 2 only blocks 1–4
  (byte-identical surface) flip; at Step 3 block 5 (transforms) + PILArNet flip; at
  Step 4 blocks 5–6 (JAXTPC/LUCiD/dissolved SSL) flip. Reuse the one shim file,
  uncommenting blocks per step, so each step is one diff hunk.
- `builder.py`/`transform.py` referenced at the top (`from .builder import …`) are
  during Step 2 still pimm's vendored modules; once those are deleted (Step 5) the
  shim must import `DATASETS`/`TRANSFORMS`/`Compose`/`build_dataset` from
  `pimm_data` instead. **Resolve at code time** by importing from `pimm_data`
  directly from the start (pimm-data exposes all four:
  `src/pimm_data/__init__.py:17-19`) so the shim is delete-safe without a re-edit.
  This is the recommended form — the snippet's `from .builder import …` is shown to
  match the *transitional* Step-2 layout; switch to `from pimm_data import …`
  before Step 5.
- **Double-registration guard.** `Registry.__contains__` is `get(key) is not None`
  (`pimm/utils/registry.py:116`); `_reregister` uses it so re-importing the shim
  (or a revert that brings vendored files back) never raises
  `KeyError: already registered` (`pimm/utils/registry.py:248`). `force=True` is the
  alternative; membership-skip is preferred so a genuine name clash still surfaces
  if it is a *different* class — at code time, optionally assert
  `registry.get(name) is cls` on the skip branch.

**What stays authored in pimm (NOT in the shim, NOT re-exported from pimm-data):**
`MultiDatasetDataloader` (`pimm/datasets/dataloader.py`), `pimm/utils/registry.py`
(the authoritative registry classes + `DATASETS`/`TRANSFORMS` instances), all
hooks/evaluators (e.g. the eval probe rewired in 05 §3.4), and the model registry.

---

## 6. Config migration (Risk Ra)

### 6.1 `jaxtpc_seg.py` rewrite (`seg` → `step`/`labl`)

The new `JAXTPCDataset` emits a **nested** dict (`src/pimm_data/jaxtpc.py:16-31`);
the 3D-seg task selects the `step` stream, decorates `segment` from `labl`, and
terminates with a stream-scoped `Collect`. Concretely:

- `modalities=("seg",)` → `modalities=("step", "labl")`
  (`configs/detector/_base_/jaxtpc_seg.py:72,82`). `labl` provides the
  `track_*`/`deposit_to_track` chain that `_decorate_step_from_labl`
  (`src/pimm_data/jaxtpc.py:416`) uses to write `step['segment']`.
- Add `label_key='pdg'` so `track_pdg` is the decorated column
  (`src/pimm_data/jaxtpc.py:427` `meta_col = f'track_{self._label_key}'`); the raw
  per-deposit PDG lands in `step['segment']`.
- The per-stream geometric/voxel ops (which hardcode `'coord'`/`'segment'`) must run
  **inside the `step` sub-dict** → wrap them in `ApplyToStream(stream='step', [...])`
  (`src/pimm_data/detector_transforms.py:27`).
- Replace `PDGToSemantic(scheme='motif_5cls')` (the old fallback that synthesized
  labels from `pdg`) with `RemapSegment(scheme='motif_5cls')` operating on the
  decorated `segment` (`src/pimm_data/detector_transforms.py:133`); raw PDG → 5-class.
- Terminal `Collect(stream='step', keys=("coord","grid_coord","segment"),
  feat_keys=("coord","energy"))` lifts the `step` stream to the bare flat dict the
  model sees (`src/pimm_data/transform.py:124-140`); `Copy(segment_motif→segment)`
  is dropped (decoration already produced `segment`).
- **No model change**: `in_channels=4` (`semseg-pt-v3m2-jaxtpc-5cls.py:37`) still
  matches `feat=cat(coord(3), energy(1))`.

**Concrete `transform` block (replaces `jaxtpc_seg.py:19-41`):**
```python
transform = [
    dict(type="ApplyToStream", stream="step", transforms=[
        dict(type="NormalizeCoord", center=_center, scale=_scale),
        dict(type="LogTransform", min_val=0.01, max_val=20.0),
        dict(type="RemapSegment", scheme="motif_5cls"),   # raw track_pdg -> 5cls
        dict(type="GridSample", grid_size=grid_size, hash_type="fnv",
             mode="train", return_grid_coord=True),
        dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
        dict(type="RandomRotate", angle=[-1, 1], axis="x", center=[0, 0, 0], p=0.8),
        dict(type="RandomRotate", angle=[-1, 1], axis="y", center=[0, 0, 0], p=0.8),
        dict(type="RandomFlip", p=0.5),
    ]),
    dict(type="ToTensor"),
    dict(type="Collect", stream="step",
         keys=("coord", "grid_coord", "segment"),
         feat_keys=("coord", "energy")),
]
```
`test_transform` is the same with the four augment ops (`RandomRotate`×3,
`RandomFlip`) dropped — matching the original `jaxtpc_seg.py:43-61` train/test
asymmetry. The `data=` block changes only the two `modalities=("seg",)` →
`("step","labl")` plus `label_key='pdg'`; `min_deposits`, `max_len`, `split`,
`dataset_name`, `num_classes=5`, `names`, `ignore_index` are unchanged
(`jaxtpc_seg.py:63-87`).

**Why `ApplyToStream`+`RemapSegment` must come from pimm-data:** neither exists in
pimm's `detector_transforms.py` (grep: `class ApplyToStream|class RemapSegment` →
no match in `pimm/datasets/`). So this migrated config only works **after the shim
re-registers pimm-data's transforms** (§5 block 5) — i.e. it lands in Step 4, after
the transform flip (Step 3). Sequencing is load-bearing.

### 6.2 The `__init__.py:9` stale import

`pimm/datasets/__init__.py:9` is `from .lucid_dataset import LUCiDDataset`. After
Step 5 there is no `lucid_dataset.py`. Two halves:
1. The import is removed — the shim imports `LUCiDDataset` from `pimm_data` (§5
   block 3).
2. `LUCiDEventSSLDataset` was **registered only by importing `lucid_event_ssl.py`**,
   which `__init__.py` never did (grep: `lucid_event_ssl` absent from `__init__.py`).
   It worked only because the LUCiD SSL config imported the module transitively. The
   shim makes registration explicit (§5 block 6) so the dissolved successor
   resolves without relying on a config-side import.

### 6.3 PILArNet/panda unaffected (grep proof)

```
$ grep -rln 'modalities' configs/ --include=*.py
configs/detector/_base_/jaxtpc_seg.py            # the only file (lines 72, 82)

$ grep -rln 'output_mode' configs/
(no matches)

$ grep -rln 'jaxtpc_seg\|JAXTPCDataset' configs/
configs/detector/semseg/semseg-pt-v3m2-jaxtpc-5cls.py
configs/detector/_base_/jaxtpc_seg.py
```
The ~21 PILArNet configs (`grep -rln PILArNetH5Dataset configs/` → 21 files across
`configs/panda/`, `configs/hmae/`, `configs/voltmae/`, `configs/polarmae/`,
`configs/lejepa/`) use `type="PILArNetH5Dataset"` and consume `segment_motif`/
`segment_pid`/`instance_particle` via `Copy` (e.g.
`configs/panda/panseg/detector-v1m2-pt-v3m2-ft-pid-fft.py:164`
`Copy(keys_dict={"instance_particle":"instance","segment_pid":"segment"})`). They
share **no `_base_`** with `jaxtpc_seg.py` and never set `modalities=`/`output_mode`,
so the Ra migration cannot touch them. They flip cleanly at Step 3 once the Rb
pilarnet merge (§8) preserves those exact emitted keys.

---

## 7. Packaging

### 7.1 Submodule (not loose editable)

Add pimm-data as a git submodule under pimm at `libs/pimm-data`:
```bash
git -C <pimm> submodule add <pimm-data-url> libs/pimm-data
git -C <pimm> submodule update --init --recursive
```
A submodule pins an **exact SHA** in pimm's tree — so a pimm checkout reproduces
the precise data-layer revision, fixing the "editable + uncopied" repro hole the
current loose-editable install has. Other `libs/` entries
(`./libs/pointops` … `environment.yml:53-56`) establish the `libs/` convention.

### 7.2 `environment.yml` pip line

Add an editable install of the submodule to the existing `pip:` block (after
`./libs/pytorch3d_ops`, `environment.yml:56`):
```yaml
  - pip:
    - ...
    - ./libs/pytorch3d_ops
    - -e ./libs/pimm-data          # NEW: editable install of the data-layer submodule
```
`-e` keeps it editable for development while the submodule SHA pins the revision.
pimm-data's deps (`numpy`, `h5py`, `hdf5plugin`, `torch` — `pyproject.toml:6-12`)
are already satisfied by the conda env; the `hdf5plugin` dep is also needed to read
JAXTPC production output (default codec blosc-zstd) and is currently **not** in
`environment.yml` — the pimm-data install pulls it in transitively.

### 7.3 `train.sh:228` snapshot edit

The repro snapshot copies the code tree on a fresh (non-resume) run:
```
scripts/train.sh:228      cp -r scripts tools pimm "$CODE_DIR" 2>/dev/null
```
Extend it to capture the data-layer SHA so the snapshot is self-describing. Minimal,
robust form — copy the submodule **and** record its SHA next to `config.py` in the
experiment dir:
```bash
  cp -r scripts tools pimm "$CODE_DIR" 2>/dev/null
  # de-fork: snapshot the pimm-data submodule + pin its SHA for repro
  cp -r libs/pimm-data "$CODE_DIR/libs/pimm-data" 2>/dev/null
  git -C libs/pimm-data rev-parse HEAD > "$EXP_DIR/pimm_data_sha.txt" 2>/dev/null
```
(Quote the copy target as `"$CODE_DIR/libs/pimm-data"` to survive spaces in
`$EXP_DIR`.) The `cp -r ... 2>/dev/null` keeps it non-fatal if the submodule path
is absent (e.g. NO_COPY dev mode, `train.sh:209-212`, which already bypasses the
copy block entirely). Resume mode (`train.sh:212`) reuses the snapshot, so the SHA
file persists across resumes. This is the repro contract the master plan §5 and
D41 require (per-run record of the data-layer revision).

### 7.4 Torch pin (`pimm-data/pyproject.toml`)

`pyproject.toml:11` currently has bare `"torch"`. Pin it to the env so
`pip install -e` never pulls a different CUDA build over the conda `pytorch=2.5.0`
(`environment.yml:16`):
```toml
dependencies = [
    "numpy",
    "h5py",
    "hdf5plugin",
    "torch>=2.5,<2.6",     # was: "torch" — match pimm env (pytorch=2.5.0+cu124)
]
```
`>=2.5,<2.6` (rather than `==2.5.0`) tolerates patch bumps while preventing a major
swap; the conda env already supplies `2.5.0`, so `-e` is a no-op for torch.

### 7.5 Registry de-dup (Rf)

Do **not** share one `Registry` object across the boundary. pimm keeps its
authoritative `DATASETS`/`TRANSFORMS` (`pimm/utils/registry.py`); pimm-data keeps
its own (`src/pimm_data/_registry.py`). The shim copies **class objects** by name
into pimm's registries (§5). The two registry classes have a compatible
`register_module(name=None, force=False, module=None)` (`pimm/utils/registry.py:262`
≈ `src/pimm_data/_registry.py:185`), so `register_module(module=Cls)` works on
pimm's registry from the shim. The membership guard (`_reregister`) prevents the
`KeyError: ... already registered` (`pimm/utils/registry.py:248`,
`_registry.py:173`) that double-import / revert would otherwise raise — the names
`DefaultDataset`, `ConcatDataset`, `Collect`, `PDGToSemantic` exist in *both*
trees, so without the guard the first re-register after a vendored-file revert
throws.

---

## 8. pilarnet Rb merge (both signatures; v3 `cluster_extra` + `is_primary` + shared `rotations`)

Merge the v3 path from pimm's `pilarnet.py` into pimm-data's. The class is
otherwise structurally identical (same `_build_index`, `get_data` shape, overlay,
`map_instance_ids`). Three concrete deltas.

**Δ1 — `revision` Literal + the v3 `cluster_extra` branch.**

Constructor signature — **pimm-data (v2, target before merge)**:
```
src/pimm_data/pilarnet.py:71      revision: Literal["v1", "v2"] = "v2",
```
**pimm (v3, source)**:
```
pimm/datasets/pilarnet.py:73      revision: Literal["v1", "v2", "v3"] = "v2",
```
Merge: widen the Literal to `["v1", "v2", "v3"]`. Then add the v3 branch in
`get_data` after the v2 branch. **pimm-data v2 branch (current,
`src/pimm_data/pilarnet.py:237-244`)**:
```python
        else:  # v2
            cluster_size, group_id, interaction_id, semantic_id, pid = (
                h5_file["cluster"][file_idx].reshape(-1, 6)[:, [0, 2, -3, -2, -1]].T
            )
            mom, vtx_x, vtx_y, vtx_z = h5_file["cluster_extra"][file_idx].reshape(-1, 5)[:, [1, 2, 3, 4]].T
            pid[pid == -1] = (
                5 if not self.old_pid_mapping else 6
            )
```
**pimm v3 branch to port (`pimm/datasets/pilarnet.py:254-282`)** — note the `elif
self.revision == "v2"` rename and the new `elif self.revision == "v3"` reading a
**6-wide** `cluster_extra` (vs v2's 5-wide) with `is_primary` as column 5:
```python
        elif self.revision == "v2":
            cluster_size, group_id, interaction_id, semantic_id, pid = (
                h5_file["cluster"][file_idx].reshape(-1, 6)[:, [0, 2, -3, -2, -1]].T
            )
            mom, vtx_x, vtx_y, vtx_z = h5_file["cluster_extra"][file_idx].reshape(-1, 5)[:, [1, 2, 3, 4]].T
            pid[pid == -1] = (
                5 if not self.old_pid_mapping else 6
            )
        elif self.revision == "v3":
            cluster_size, group_id, interaction_id, semantic_id, pid = (
                h5_file["cluster"][file_idx].reshape(-1, 6)[:, [0, 2, -3, -2, -1]].T
            )
            n_clusters = cluster_size.shape[0]
            raw_extra = h5_file["cluster_extra"][file_idx]
            cluster_extra = (
                raw_extra.reshape(n_clusters, -1)
                if n_clusters > 0
                else np.empty((0, 6), dtype=np.float32)
            )
            if cluster_extra.shape[1] != 6:
                raise ValueError(
                    f"Expected v3 cluster_extra width 6, got {cluster_extra.shape[1]}"
                )
            mom, vtx_x, vtx_y, vtx_z, is_primary = cluster_extra[:, [1, 2, 3, 4, 5]].T
            pid[pid == -1] = (
                5 if not self.old_pid_mapping else 6
            )
        else:
            raise ValueError(f"Unsupported PILArNet revision: {self.revision}")
```
Port verbatim. Then the v3-only `is_primary` propagation, guarded by
`if self.revision == "v3":`, at three sites already present in pimm:
- repeat-by-cluster_size: `data_is_primary = np.repeat(is_primary, cluster_size)`
  (`pimm/datasets/pilarnet.py:307-308`);
- `remove_low_energy_scatters` slice: `is_primary = is_primary[1:]`
  (`pimm/datasets/pilarnet.py:295-296`);
- energy-threshold mask: `data_is_primary = data_is_primary[threshold_mask]`
  (`pimm/datasets/pilarnet.py:322-323`);
- emit: `data_dict["is_primary"] = data_is_primary.astype(np.int32)[:, None]`
  (`pimm/datasets/pilarnet.py:338-339`).
- overlay concat-key: append `"is_primary"` to `concat_keys` when v3
  (`pimm/datasets/pilarnet.py:482-483`).

**Δ2 — shared `rotations` overlay param.**

`_apply_random_90_rotation` — **pimm-data (current,
`src/pimm_data/pilarnet.py:355-366`)** draws fresh rotations internally:
```python
    def _apply_random_90_rotation(self, coord, center=None):
        if center is None:
            center = np.array([384.0, 384.0, 384.0], dtype=np.float32)
        coord = coord - center
        for axis in ["x", "y", "z"]:
            n_rot = random.randint(0, 3)
            if n_rot > 0:
                rot_mat = self._get_rotation_matrix_90(axis, n_rot)
                coord = coord @ rot_mat.T
        coord = coord + center
        return coord
```
**pimm (target, `pimm/datasets/pilarnet.py:401-414`)** accepts a `rotations` dict so
the *same* draw applies to both `coord` and the v3 `vertex`:
```python
    def _apply_random_90_rotation(self, coord, center=None, rotations=None):
        if center is None:
            center = np.array([384.0, 384.0, 384.0], dtype=np.float32)
        if rotations is None:
            rotations = {axis: random.randint(0, 3) for axis in ("x", "y", "z")}
        coord = coord - center
        for axis in ["x", "y", "z"]:
            n_rot = rotations[axis]
            if n_rot > 0:
                rot_mat = self._get_rotation_matrix_90(axis, n_rot)
                coord = coord @ rot_mat.T
        coord = coord + center
        return coord
```
Default `rotations=None` reproduces the old internal-draw behavior exactly (so v1/v2
overlay is byte-unchanged). In `_apply_overlay`, the caller draws **once** per extra
event and applies it to both `coord` and (v3) the valid-vertex rows — **pimm
(`pimm/datasets/pilarnet.py:523-537`)**:
```python
            if "coord" in extra:
                detector_center = np.array([384.0, 384.0, 384.0], dtype=np.float32)
                rotations = {axis: random.randint(0, 3) for axis in ("x", "y", "z")}
                extra["coord"] = self._apply_random_90_rotation(
                    extra["coord"], center=detector_center, rotations=rotations
                )
                if self.revision == "v3" and "vertex" in extra:
                    valid_vertex = ~(extra["vertex"] == -1).all(axis=1)
                    extra["vertex"][valid_vertex] = self._apply_random_90_rotation(
                        extra["vertex"][valid_vertex],
                        center=detector_center,
                        rotations=rotations,
                    )
```
vs **pimm-data (current, `src/pimm_data/pilarnet.py:474-477`)** which rotates only
`coord` with a fresh internal draw and has no vertex co-rotation. Port the pimm
form.

**Δ3 — docstring.** Update the class docstring to mention v3
(`pimm/datasets/pilarnet.py:47-52` vs `src/pimm_data/pilarnet.py:46-49`) — purely
informational.

**Parity guard (Rb).** After the merge, a `revision="v1"` and `revision="v2"`
`get_data` call must be byte-identical to the pre-merge pimm-data output (the v3
branch and `rotations=None` default make v1/v2 paths untouched). The
identical-first-batch test (Step 3 GATE) covers PILArNet v2; add a v1/v2 golden-array
check in 07. v3 has no config consumer yet (no `revision="v3"` in `configs/`), so it
is dormant until a v3 dataset is wired — consistent with the transform-side v3
plumbing being `_valid_vertex_mask`-guarded and dormant (01 §3.1.5).

---

## 9. Tests / gates

Per-step gates (the GATE lines in §4) plus the cross-cutting checks. All on
synthetic fixtures (07), no GPU/WAND.

- **Phase A cross-modality regression (A5, D42/D43).** `modalities=
  ('step','sensor','hits','labl')` with `min_deposits>0` returns the **same** physics
  event (`source_event_idx`) across every modality for every idx; plus a
  gap-in-one-modality variant. **Both fail on `master`/HEAD today** — they lock the
  joint-index fix. Folded into the Step-0 matrix (Part 07 / Part 02 §6.16). Phase A's
  own GATE before Step 0.
- **Per-step build + 1-step smoke.** Step 2: PILArNet builds + 1 train step through
  the shim. Step 4: JAXTPC semseg config (`semseg-pt-v3m2-jaxtpc-5cls.py`) **and**
  the dissolved LUCiD SSL config each build + 1 step.
- **First-batch tensor parity vs vendored (Step 3).** Seed `random`/`np.random`/
  `torch` identically; build the *same* config against the vendored layer (pre-shim
  checkout) and the shimmed layer; `assert_array_equal` per batch-0 key (`coord`,
  `feat`, `segment`, `offset`, `instance_*`). Gate transform-parity assertions on the
  branch being fetchable (HEAD `ffb0823`); golden arrays otherwise (master §6).
- **pilarnet Rb parity (§8).** v1/v2 `get_data` byte-identical pre/post merge; v3
  `cluster_extra` width-6 assertion fires on a malformed fixture; `is_primary`
  present + repeated-by-cluster_size; shared-`rotations` makes overlaid `coord` and
  `vertex` share orientation.
- **Grep gates (Step 5, D33).** `modalities=("seg"`/`output_mode`/`from .lucid_dataset import`
  all absent; only migrated `("step","labl")` remains. The bare substring `seg`
  survives in `semseg`/`segment` — gate on *modality/reader* usage, not the raw
  substring.
- **Double-registration guard.** Importing the shim twice (and after a
  vendored-file revert) does not raise `KeyError: already registered`
  (`pimm/utils/registry.py:248`).
- **Full-suite green before delete.** Entire Part 07 matrix green + ≥1 soaked full
  PILArNet run before the Step-5 delete commit.

---

## 10. Risks & rollback

| Risk | What breaks | Mitigation / rollback |
|---|---|---|
| **Ra — config breakage on flip** | `JAXTPCDataset(modalities=("seg",))` raises `ValueError` (`src/pimm_data/jaxtpc.py:202`); `__init__.py:9` import dies after Step 5 | Migrate `jaxtpc_seg.py` + child *before* flipping `JAXTPCDataset` (Step 4 ordering); shim fixes the `__init__` import + registers the dissolved SSL successor (§5/§6). PILArNet/panda proven unaffected by grep (§6.3). |
| **Rb — pilarnet drift** | pimm-data v2 ≠ pimm v3; flipping PILArNet to pimm-data loses `is_primary`/v3 vertex | Land the §8 merge in Step 1 (before Step 3 flip). v1/v2 parity guard + width-6 assertion (§9). v3 dormant (no config consumer). |
| **Rf — registry/logger re-export** | `defaults`/`builder`/`detector_transforms` not byte-identical; class-name clash across two registries → `KeyError` | Re-register by class object into pimm's registry with the membership guard (§5/§7.5); never share a registry object. |
| **Transform dependency on shim ordering** | Migrated `jaxtpc_seg.py` uses `ApplyToStream`/`RemapSegment` that exist only in pimm-data | They register only after Step 3's transform flip; the JAXTPC migration is Step 4 (after). Sequencing enforced in §4. |
| **Submodule SHA not snapshotted** | non-reproducible run (editable + uncopied) | `train.sh:228` snapshot edit + `pimm_data_sha.txt` (§7.3); torch pin (§7.4). |
| **Delete commit not revert-safe** | a bad flip can't be undone cleanly | Vendored files survive Steps 2–4; the Step-5 delete is one commit. **`git revert <delete-commit>`** restores the entire vendored tree; the shim's guard makes the restored-files re-register harmless. |
| **Phase A skipped / done after de-fork (D43)** | the de-fork's base inherits the cross-modality desync (handoff §4); labeled JAXTPC tasks join misaligned modalities silently | Land Phase A (A1–A5) **first** as a standalone PR on the current `jaxtpc.py` (§4 Phase A); the base factors A2 up (Part 02 §3.3a). A5 in the Step-0 matrix gates the de-fork. Independent of B/C scope (still owed by user, D48). |

**Single-revert-safe property (the design goal).** Because the shim (Step 2–4) and
the delete (Step 5) are **separate commits**, and the shim's `_reregister` is
membership-guarded:
- Revert **the delete** → vendored files return, shim still points at pimm-data,
  re-registration is skipped (names already present from the restored vendored
  modules' import side-effects). Training falls back to whichever module imports
  first; for a true rollback, also revert the shim.
- Revert **the shim** (any of Steps 2–4) → `pimm/datasets/__init__.py` returns to
  the 13-line vendored wiring; vendored `.py` files (still present pre-Step-5)
  drive training exactly as before. This is the clean per-step rollback.
