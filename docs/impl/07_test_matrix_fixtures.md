# Part 07 — Test matrix & fixtures (implementation spec)

**Status:** final test plan before coding. Consolidates and de-duplicates the
per-part Tests sections (Parts 01–05) into one coherent matrix, fills the gaps,
and specifies the `testing.py` fixture extensions those tests depend on. This is
the executable plan for **Step 0** of the rollout runbook (the parity/determinism
gate that must be green before any pimm-side flip).

**Source:** `implementation_plan_pimm_data_datalayer.md` §6 (test matrix), §2
(rollout Steps 0–5), §9.3 (eval reproducibility, D41); the sibling specs
`01_transforms.md` §6, `02_dataset_base.md` §6, `03_readers.md` §6,
`04_label_decoration.md` §6, `05_collate_streams_eval.md` §6.

**Files (read-only ground truth):**
- `src/pimm_data/testing.py` — synthetic fixture generators
  (`make_jaxtpc_sample` `testing.py:55`, `make_lucid_sample` `testing.py:333`)
  and the documented cross-modality FK invariants (`testing.py:14-27`).
- `tests/conftest.py` — `jaxtpc_data_root`/`lucid_data_root`/`jaxtpc_pixel_data_root`
  session fixtures (`conftest.py:74-100`), the `real_data_only` marker
  (`conftest.py:50-71`), env-var override path (`conftest.py:37-47`).
- `tests/test_transforms.py`, `tests/test_jaxtpc_transforms.py`,
  `tests/test_lucid.py`, `tests/test_cache.py`, `tests/test_jaxtpc*.py`,
  `tests/test_pdg.py` — existing suite.
- `pyproject.toml` — `[tool.pytest.ini_options] testpaths=["tests"]`
  (`pyproject.toml:24-25`); `test` extra = `pytest` (`pyproject.toml:15`).
- Branch parity reference (must be importable for `[parity-vs-branch]`):
  `/sdf/home/o/omara/.claude/jobs/21ffc656/particle-imaging-models/pimm/datasets/transform.py`
  — `RelativeLogNormalize` @ `:278`, `GridSample` @ `:1183`, `Collect` @ `:120`,
  `MixedScaleGeometryMultiViewGenerator` @ `:1682`. **It imports
  `from pimm.utils.registry import Registry` (branch `transform.py:22`)** —
  `pimm` is NOT installed in the pimm-data env (`python3 -c "import pimm"` →
  `ModuleNotFoundError`), so the parity loader must stub that module (§7.3).

**This part writes no source.** It is the spec a coding agent follows to (a) extend
`testing.py`, (b) author `tests/test_*.py`, and (c) wire markers into
`pyproject.toml`/`conftest.py`.

---

## 1. Purpose & scope

Pin down, before any code lands:

1. **`testing.py` fixture extensions** (§3) — a *prerequisite work item*. The
   current fixtures write neither `source_event_idx` nor per-event `n_hits`
   (Parts 02/03 flagged this), and have no `per_interaction` group, no `T_reco`,
   and stamp `format_version=3` instead of `5`. The determinism, min-points,
   readers, and decoration tests **cannot run** until these are stamped. This
   section gives the exact fields, for both detectors, that close the gap.
2. **The consolidated, de-duplicated test matrix** (§4), organized by part
   (transforms / dataset-base / readers / label-decoration / collate-streams-eval /
   de-fork-migration). Each test: a stable id, what-it-guards, setup, action,
   EXPECTED, a tag, and which fixture it uses. Overlaps between the per-part specs
   are merged into one owner test (cross-references noted).
3. **The cross-cutting must-pass gates** (§6) mapping each test group to the
   rollout Step 0–5 it blocks.
4. **Markers / structure / running** (§7) — `slow`, `requires_branch`,
   `requires_config`, `real_data_only`; how `JAX_PLATFORM_NAME=cpu pytest` runs
   them; the branch-importability handling; the golden-array strategy.
5. **Determinism & repro test designs in detail** (§5) — the holdout invariant
   under shard reorder / add-remove / rank / machine simulated with no real data,
   plus the train≡eval transform-equality assertion.

**Locked constraints (honored throughout):** all tests **CPU-only**, **synthetic
fixtures**, **no real WAND**, **no GPU/JAX**. Branch outputs are the reference for
transform parity — seed `random` / `np.random` / `torch` identically. Parity
tests **skip-gracefully** when the branch isn't importable, and **every parity
test is paired with a golden-array regression** so a skip can never mask a
regression (§7.4). Nothing in the pimm-data test path imports `jax` or `torch.cuda`.

---

## 2. Current test infra (testing.py fixtures + invariants; tests/ layout; markers)

### 2.1 `testing.py` — what the generators write today

`make_jaxtpc_sample(outdir, dataset_name='sim', n_events=2, n_files=1,
n_volumes=2, n_deposits=60, n_groups=6, n_tracks=6, n_pixels_per_plane=40,
readout_type='wire', seed=0)` (`testing.py:55-90`) writes
`{outdir}/{step,sensor,hits,labl}/{dataset_name}_{mod}_NNNN.h5`. Per-event it
builds `n_volumes` volumes with consistent FKs (`_build_jaxtpc_event`
`testing.py:93-147`). Wire readout = U/V/Y planes; pixel = single `Pixel` plane
with a `pz` axis (`testing.py:96`).

`make_lucid_sample(outdir, dataset_name='wc', n_events=2, n_files=1,
n_segments=80, n_hits=120, n_hits_entries=200, n_sensors=64, n_tracks=8,
n_particles=3, seed=0)` (`testing.py:333-367`) writes the same four-modality
layout; a single PMT geometry is reused across events (`testing.py:346-347`).

**Documented cross-modality invariants** (the §4 hand-gather references rely on
these — `testing.py:14-27`):
- **JAXTPC:** `hits.deposit_to_group` indexes into `hits.group_to_track`;
  `labl.deposit_to_track[i] == hits.group_to_track[hits.deposit_to_group[i]]`
  (`testing.py:18-21`, enforced at `testing.py:114`); every per-deposit track id
  is in `labl.track_ids`; per-plane CSR entries decode to the declared `n_pixels`
  (`testing.py:22`). Every PDG ∈ `{13,11,211,22,2212}` (`testing.py:50`), every
  group's track ∈ `track_ids` (`testing.py:108-109`), and at least one PDG `> 20`
  (proton 2212) so a test can distinguish raw from remapped (`testing.py:101-104`).
- **LUCiD:** every `step.track_idx` is a valid `per_track` row; every
  `hits.particle_idx` and `per_track.particle_idx` is a valid `per_particle` index
  (`< n_particles`); every `per_track.ancestor` is itself a `track_id`
  (`testing.py:23-26`, built at `testing.py:375-386`). `track_idx`/`particle_idx`
  are **positional** row indices, not Geant4 id values (`testing.py:405-408`).
  `track_cluster`/`track_interaction` carry `>1` unique value (`testing.py:316-318`,
  `interaction = (row % 3) + 1 ∈ {1,2,3}`).

**Gaps to fill (the prerequisite, §3):**
- No `source_event_idx` anywhere — neither a per-file `config/source_event_idx`
  vector nor a per-event `evt.attrs['source_event_idx']`.
- No per-event `n_hits` attr on LUCiD sensor/hits; no `pg.attrs['n_pixels']` on
  JAXTPC sensor planes; no `vol.attrs['n_actual']` on JAXTPC hits volumes (step
  *does* stamp `n_actual` at `testing.py:239`).
- No `per_interaction` group (LUCiD labl writes `per_event`/`per_particle`/
  `per_track` only, `testing.py:532-554`).
- No `T_reco` dataset on LUCiD hits (`testing.py:506-519` writes
  `sensor_idx`/`particle_idx`/`PE`/`T`).
- `format_version` stamped `= 3` (`testing.py:469,496,511,526`); confirmed schema
  is `5`.
- No `per_particle.interaction_idx` (Part 04 one-hop `instance_interaction`).

### 2.2 `tests/` layout, fixtures, markers

`tests/conftest.py`:
- Session fixtures `jaxtpc_data_root` (`conftest.py:74-78`), `jaxtpc_pixel_data_root`
  (`conftest.py:81-93`), `lucid_data_root` (`conftest.py:96-100`), plus
  `jaxtpc_is_synthetic`/`lucid_is_synthetic` (`conftest.py:103-112`). Each falls
  back to the synthesizer unless `*_DATA_ROOT` env vars point at a validated v3/v5
  layout (`_resolve_root` `conftest.py:37-47`).
- One marker registered: `real_data_only` (`conftest.py:50-54`), auto-skipped when
  neither `JAXTPC_DATA_ROOT` nor `LUCID_DATA_ROOT` is set
  (`pytest_collection_modifyitems` `conftest.py:57-71`).

`pyproject.toml`: `testpaths=["tests"]` (`pyproject.toml:25`); no other markers
declared; `test` extra = `pytest`.

Existing tests (all CPU, synthetic, no GPU/JAX): `test_transforms.py` (Compose +
registry + e2e collate, `:48-105`), `test_jaxtpc_transforms.py` (ApplyToStream ×
transform × stream matrix), `test_lucid.py` (modality matrix + labl decoration —
**the hand-gather reference pattern** at `test_lucid.py:104-112` is the model for
the §4 decoration tests), `test_jaxtpc*.py`, `test_cache.py` (the
`pytest.mark.skipif(... reason=...)` pattern at `test_cache.py:24-27` is the model
for `requires_*` markers), `test_pdg.py`.

**No determinism/holdout/`read_meta`/parity tests exist yet** — every test in §4
tagged `[determinism]`/`[migration-smoke]`/`[parity-vs-branch]` is net-new.

---

## 3. Fixture extensions required (exact fields, both detectors — prerequisite)

**Work item P07-FIX, blocks everything in §4 except the pure-array transform
tests (TR-01..TR-13, which build their own in-memory dicts).** Implement in
`src/pimm_data/testing.py`. All additions are **backward-compatible**: existing
tests that don't request the new fields keep passing; new tests opt in via new
kwargs / always-present attrs.

### 3.1 New `make_*_sample` kwargs

Add to **both** generators:

```python
def make_jaxtpc_sample(..., source_event_idx=None, stamp_n_hits=True, ...):
def make_lucid_sample(...,  source_event_idx=None, stamp_n_hits=True,
                      with_per_interaction=False, n_interactions=2,
                      with_t_reco=True, ...):
```

- **`source_event_idx`** — `None` (default) → **omit** all `source_event_idx`
  fields, exercising the D26 fallback path (DS-15, RD-08). A list/array of length
  `n_files * n_events` → stamp it. A sentinel string `'shuffled'` → stamp a
  **deterministic non-positional permutation** of `range(n_files*n_events)`
  (seeded), so a test can assert "identity ≠ positional index" (RD-07) and that
  reorder leaves identity invariant (DS-01). The per-file slice is written both as
  the `config/source_event_idx` vector and the per-event attr (§3.2/§3.3).
- **`stamp_n_hits`** — default `True`: stamp the per-event/per-plane/per-volume
  count attrs (§3.2/§3.3) so cheap-`read_meta` == array (DS-07, RD-01..04).
  `False` → omit them to exercise the "absent → 0" path (RD edge, DS-edge).
- **`with_per_interaction`** (LUCiD) — default `False` (no regression). `True` →
  add the `per_interaction` group with `n_interactions` rows + CSR primaries
  (§3.3) so RD-05 and decoration LD-05 can run.
- **`with_t_reco`** (LUCiD) — default `True` → add a `T_reco` dataset to hits;
  `False` → omit it (RD-06 absent sub-case).

A second LUCiD generator variant for the per-particle one-hop axis: also stamp
**`per_particle.interaction_idx`** unconditionally (it is cheap, harmless to
existing tests, and unblocks LD-04). Default values = `track_particle_idx`-derived
so they are internally consistent: `per_particle_interaction_idx[p] =
interaction[ first track mapped to particle p ]` (any deterministic choice; the
test recomputes from the same rule).

### 3.2 Count + identity attrs (both detectors)

| Detector / modality | field | where | value |
|---|---|---|---|
| **LUCiD sensor** | `evt.attrs['n_hits']` | each `event_NNN` group (`testing.py:498-503`) | `len(sensor_idx)` (== `n_hits` kwarg) |
| LUCiD sensor | `config/source_event_idx` uint32 `(n_events,)` | file `config` group | per-file slice of the resolved `source_event_idx` |
| LUCiD sensor | `evt.attrs['source_event_idx']` | each event group | same value, per-event (vector/attr agreement, RD-07) |
| **LUCiD step** | (already) `evt.attrs['n_segments']` | `testing.py:473` | unchanged — reused as step `n_hits` proxy |
| LUCiD step | `evt.attrs['source_event_idx']` | each event group | attr only (no vector confirmed on step, RD-03) |
| **LUCiD hits** | `evt.attrs['n_particle_hits']` | each event group (`testing.py:513-519`) | `n_hits_entries` |
| LUCiD hits | `evt.attrs['source_event_idx']` | each event group | attr only |
| **LUCiD labl** | `config/source_event_idx` uint32 `(n_events,)` + `evt.attrs['source_event_idx']` | (`testing.py:527-554`) | same values as sensor (cross-modality agreement, DS-09, RD-07) |
| **JAXTPC sensor** | `pg.attrs['n_pixels']` | every plane group (`testing.py:260-275`) | `len(plane['values'])` (the decoded sparse length read_event yields) |
| JAXTPC sensor | `evt.attrs['source_event_idx']` | each event group | resolved value (attr only; no JAXTPC vector, RD-05/§3.8 of Part 03) |
| **JAXTPC step** | (already) `vg.attrs['n_actual']` | `testing.py:239` | unchanged |
| JAXTPC step | `evt.attrs['source_event_idx']` | each event group | resolved value |
| **JAXTPC hits** | `vg.attrs['n_actual']` | every `volume_N` group (`testing.py:290-300`) | `n_deposits` (matches step, the deposit count) |
| JAXTPC hits | `evt.attrs['source_event_idx']` | each event group | resolved value |

**Critical consistency requirement (so cheap == array holds, DS-07/RD-01..04):**
the stamped count must equal what `read_event` actually returns for that event.
- LUCiD sensor `n_hits` = the `PE`/`T` length written (`testing.py:501-503`).
- JAXTPC sensor `pg.attrs['n_pixels']` must equal `len(plane['values'])` (the
  sparse stream the sensor reader decodes, `testing.py:264`) — **not** the CSR
  `total`. Part 03 §3.5 defines JAXTPC sensor `n_hits` = Σ plane `n_pixels`, and
  the reader's `read_event` sensor point count is the decoded sparse length, so
  the fixture must stamp `n_pixels = n_sparse` to keep RD-03 honest. (If the
  intended invariant is instead "Σ over the CSR-decoded hits-pixel count," stamp
  that and assert against the hits reader; pick one and document in the test — the
  recommended choice is sparse-length to match the sensor stream.)

### 3.3 `per_interaction` group + CSR primaries (LUCiD labl, gated by `with_per_interaction`)

Add to `_write_lucid_labl` (after the `per_track` block, `testing.py:546-554`):

```
g['per_interaction']:
  source_type               int16  (I,)
  t0                        f32    (I,)
  vertex_x, vertex_y, vertex_z   f32 (I,)        # three scalars, NOT a (3,) dset
  n_primaries               int32  (I,)
  n_particles               int32  (I,)
  neutrino_pdg              int16  (I,)
  neutrino_energy_MeV       f32    (I,)
  contained                 bool   (I,)
  # CSR primaries (ragged), offsets (I+1,), data (Σ n_primaries,):
  primary_track_ids_data    int32  ;  primary_track_ids_offsets   int32 (I+1,)
  primary_pdgs_data         int32  ;  primary_pdgs_offsets        int32 (I+1,)
  primary_energies_data     f32    ;  primary_energies_offsets    int32 (I+1,)
```

Use a **known ragged layout** the test hard-codes: e.g. `n_interactions=2`,
interaction 0 has 2 primaries, interaction 1 has 1 → `offsets=[0,2,3]`,
`data` length 3 (RD-05 asserts `offsets[-1]==data.size` and
`data[off[0]:off[1]]` equals the known primaries of interaction 0). Vertex values
distinct per interaction so LD-05's `target_vertex` stack `(3,)` is checkable.

### 3.4 `T_reco` dataset (LUCiD hits, gated by `with_t_reco`)

Add to `_write_lucid_hits` (`testing.py:513-519`): `g.create_dataset('T_reco',
data=s['T'] + 0.5)` — same length as `T`, distinct values (so RD-06 can assert
`t_reco != t` element-wise and `t_reco.shape == t.shape` after the `pe_threshold`
mask). The `+0.5` offset is arbitrary but must be deterministic.

### 3.5 `format_version` → 5

Flip the four `cfg.attrs['format_version'] = 3` writes
(`testing.py:469,496,511,526`) to `= 5`, matching the confirmed WAND schema and
the reader docstring fix (Part 03 §3.9). RD-11 (grep guard) and DS smoke depend on
this.

### 3.6 Helper: multi-source fixture builder (for mixture / collision tests)

Add a tiny test helper (in the test module or `conftest.py`, not `testing.py`
proper) that materializes **two source roots** for the mixture/holdout/collision
tests (DS-05, DS-08, DS-11, LD-07):

```python
def two_sources(tmp_path, builder=make_lucid_sample, **kw):
    a = builder(str(tmp_path/'config_1'), source_event_idx='shuffled', **kw)
    b = builder(str(tmp_path/'config_3'), source_event_idx='shuffled', **kw)
    return [a, b]
```

For the **identical-filename collision** test (DS-11) build both with the *same*
`dataset_name='wc'` so both produce `wc_sensor_0000.h5` → the source-prefix
`get_data_name` fix is the only thing distinguishing them.

### 3.7 Determinism corpus (for shard reorder / add-remove, DS-01/DS-02)

Build with `n_files=4, n_events=8` and `source_event_idx='shuffled'` so there are
32 events with stable, non-positional identities spread over 4 shards. The
reorder test renames `*_0000.h5 ↔ *_0003.h5` on disk (across **all four**
modalities) and invalidates the manifest cache; the add/remove test writes a 31-
and a 33-event corpus differing by exactly one identity.

**Cross-modality A5 corpus variants (DS-19, D42/D44).** Two additional shapes off
the same generator: (a) write all four modalities full but give some events a
sub-threshold step deposit count so a `min_deposits>0` build masks a
**non-contiguous** step subset (handoff §4 #1); (b) **delete a middle `event_*`
group from one modality only** (e.g. drop `event_003` from the hits shard, keep it
in step/sensor/labl) so the per-modality present-key sets differ (handoff §4 #2).
Both must keep `source_event_idx` stamped so the test can assert same-identity-
across-modalities per served idx and that the joint index intersects the gap out.
For DS-19(c) reuse `make_jaxtpc_sample(n_volumes=2)` with selected events' deposits
concentrated in volume 1.

---

## 4. Consolidated test matrix

Tags: **[parity-vs-branch]** (assert byte-equal to the colleague's branch, seeded
identically; skips if branch unimportable but is **paired with a [golden]**),
**[new-behavior]** (net-new semantics, no branch reference), **[determinism]**
(reproducibility/holdout/repro), **[migration-smoke]** (config builds + one
`__getitem__`/1-step), **[golden]** (hard-coded expected array/scalar — the
skip-proof anchor).

Each row is one owner test; where a per-part spec had a near-duplicate, it is
merged here and the origin noted. File column = target test module.

### 4.1 Transforms (Part 01) — `tests/test_transforms.py` (+ `test_jaxtpc_transforms.py`)

| id | guards | setup | action | EXPECTED | tag | fixture |
|---|---|---|---|---|---|---|
| TR-01 | `RelativeLogNormalize` negatives + no-NaN | `time=[-240,0,50,8000]` f32, defaults | apply `RelativeLogNormalize()` | `≈[-1.0,-0.1998,-0.1274,1.0]` (atol 1e-3); all finite; f32; in `[-1,1]` | [golden] | none (in-memory) |
| TR-02 | all-equal / single-point | `time=[5,5,5]`, `time=[7.0]` | apply | `[5,5,5]→[-1,-1,-1]`; `[7.0]→[-1.0]`; no NaN | [new-behavior] | none |
| TR-03 | ctor validation | `scale=0`, `max_val=-1`, `out_max<out_min` | construct | each raises `ValueError` | [new-behavior] | none |
| TR-04 | missing-key strict | dict without `time` | apply `RelativeLogNormalize(keys=("time",))` | `ValueError` "Key time not found" | [new-behavior] | none |
| TR-05 | `RelativeLogNormalize` parity | random `time` incl. negatives, seeded | pimm-data vs branch, same ctor | `assert_array_equal` | [parity-vs-branch] + [golden] (TR-01 is the golden anchor) | none |
| TR-06 | `GridSample` `min_keys`/`sum_keys` back-compat byte-equal | synthetic `coord`/`charge`/`t0_us`, seed `random`+`np.random` | pimm-data `GridSample(sum_keys=['charge'],min_keys=['t0_us'])` vs branch | `assert_array_equal` `coord`/`charge`/`t0_us`/`grid_coord` | [parity-vs-branch] | jaxtpc_data_root or in-memory |
| TR-07 | `GridSample` `max` vs hand groupby | 5 pts → 2 known voxels, known `charge` | `reducers={'charge':'max'}` | per-voxel max == manual groupby; dtype preserved | [new-behavior] + [golden] | none |
| TR-08 | `GridSample` `mean` count-divide + int→float | int `count_col` + float `charge`, known voxels | `reducers={'count_col':'mean','charge':'mean'}` | `count_col` dtype f32 == sum/count; `charge` == sum/count in its float dtype | [new-behavior] + [golden] | none |
| TR-09 | `GridSample` `first` determinism across seeds | multi-pt voxels + `plane_id` col | run `reducers={'plane_id':'first'}` under several `np.random` seeds | `first`-reduced `plane_id` identical across seeds (surviving `coord` rows differ) | [determinism] | none |
| TR-10 | shim equivalence + explicit wins | same input two ways; also `reducers={'k':'max'}`+`min_keys=['k']` | run both, same seed | `reducers` path == `sum_keys`/`min_keys` path; conflict → max | [new-behavior] | none |
| TR-11 | unknown op raises | `reducers={'charge':'median'}` | construct | `ValueError` listing allowed ops | [new-behavior] | none |
| TR-12 | `LogTransform.clip` clamps domain | `energy=[-5,0,1e6]` | `clip=True` vs `clip=False` | `clip=True` finite, in `[-1,1]`; `clip=False` non-finite for `-5` and == branch | [new-behavior] + [parity-vs-branch] (clip=True vs branch) | none |
| TR-13 | `LogTransform` default unchanged | in-domain `energy` | `LogTransform()` default | byte-identical to captured pre-merge golden AND branch | [golden] + [parity-vs-branch] | none |
| TR-14 | `get_view` empty raises | `MultiViewGenerator`, `coord=(0,3)` | `get_view` | `ValueError` | [new-behavior] | none |
| TR-15 | `get_view` 1-pt clamp | `coord=(1,3)`, `scale=(0.1,0.4)` | `get_view` | `view['coord'].shape[0]==1` | [new-behavior] | none |
| TR-16 | v3 vertex co-transform (flip/rotate/scale/shift/center/normalize/positive/conditional) | `coord (N,3)` + `vertex (M,3)` one sentinel `(-1,-1,-1)`, seeded | run each geometric transform pimm-data vs branch | `assert_array_equal` `coord` AND `vertex`; sentinel row unchanged | [parity-vs-branch] + [golden] (one hard-coded flip case) | none |
| TR-17 | `PointClip` vertex-blind | `coord` exceeding range + `vertex` | `PointClip` | `coord` clipped, `vertex` bit-identical | [new-behavior] | none |
| TR-18 | geometric transforms no-op vertex when absent | dict `coord` only | run all geometric transforms | identical to captured pre-merge golden (porting vertex hooks changed nothing) | [golden] + [parity-vs-branch] | none |
| TR-19 | `index_operator` prefix-match subsets per-point keys | `coord (10,3)`, `segment_pid`, `instance_particle`, `particle_idx`, `sensor_idx`, no `Update` | `ShufflePoint` then `GridSample(train)` | all four label/FK keys subset+permuted with `coord` (verify via tagged `particle_idx` identity col) | [new-behavior] | none |
| TR-20 | per-event `target_*` NOT subset | `coord (10,3)` + `target_vertex (3,)` + `target_energy` scalar/`(1,)` | `GridSample(train)` reducing N<10 | `target_vertex` still `(3,)`; `target_energy` unchanged | [new-behavior] | none |
| TR-21 | underscore-boundary match | `segment_pid (N,1)` carries; `segmentation_meta (N,2)` does not | `ShufflePoint` | `segment_pid` permuted; `segmentation_meta` left unchanged | [new-behavior] | none |
| TR-22 | no duplicate `index_valid_keys` on chaining | dict with `segment_pid` | `ShufflePoint`→`RandomDropout`→`GridSample` | `index_valid_keys` contains `segment_pid` exactly once | [new-behavior] | none |
| TR-23 | `MixedScaleGeometryMultiViewGenerator` parity | synthetic cloud `coord`/`energy`, seeded | pimm-data vs branch, `fine_center_mode` both `'geometry'`+`'random'` | equal `global_*`/`local_*` + `global_offset`/`local_offset` | [parity-vs-branch] (skip OK; `slow`) | none |
| TR-24 | registered-count bump | — | extend `test_transforms_registered_count` (`test_transforms.py:48-56`) | `RelativeLogNormalize` + `MixedScaleGeometryMultiViewGenerator` registered; floor `>= 50` | [new-behavior] + [golden] | none |
| TR-25 | e2e JAXTPC seg w/ reducers + prefix-match | `ApplyToStream('step',[GridSample(reducers={'charge':'sum'},return_grid_coord),ToTensor])→Collect('step',keys=('coord','grid_coord','segment'),feat_keys=('coord','energy'))` | `collate_fn([ds[0],ds[1]])` | batch has `coord`/`feat`/`segment`/`offset`; `len(segment)==len(coord)` (prefix-match kept `segment_*` aligned through `GridSample`) | [migration-smoke] | jaxtpc_data_root |

De-dup note: Part 01 T6–T13 (GridSample reducers, LogTransform) and the v3 vertex
T16–T18 are all here; the 2-D-coord `RandomRotate` failure already covered by
`test_jaxtpc_transforms.py:230-236` — keep, don't duplicate.

### 4.2 Dataset base (Part 02) — `tests/test_multimodal.py` (NEW)

| id | guards | setup | action | EXPECTED | tag | fixture |
|---|---|---|---|---|---|---|
| DS-01 | holdout invariant under shard reorder | 4-file/8-event corpus, `source_event_idx='shuffled'`, `split='val'`, `holdout={seed:0,fractions:(.8,.1,.1)}` | build `ds_a`; rename `_0000↔_0003` across all mods, drop cache; build `ds_b` | `set(event_identity(i))` identical (§5.1) | [determinism] | §3.7 corpus |
| DS-02 | holdout invariant under add/remove events | 32-event base vs 31+1-swapped corpus | diff `val` identity sets | diff is exactly the removed identity deleted + the added one inserted iff its bucket is `val`; every other assignment unchanged (§5.2) | [determinism] | §3.7 corpus |
| DS-03 | rank-identical index | monkeypatch comm shim `get_world_size()==4`, iterate `get_rank()∈{0,1,2,3}` (rank-0 first seeds cache) | compare `data_list` across 4 builds | byte-identical `data_list` + identity map on all ranks (§5.3) | [determinism] | §3.7 corpus |
| DS-04 | machine/version determinism | fixed `(config_id, source_event_idx)` tuples | `_bucket_u64(0,c,s)` vector | exact match vs hard-coded golden blake2b vector (§5.4) | [golden] + [determinism] | none |
| DS-05 | config-stratification | two sources, 1000 events each, `fractions=(.8,.1,.1)` | per-config count of train/val/test | each config ≈ 80/10/10 (±3σ binomial); global ≈ 80/10/10 | [determinism] | two_sources(n=1000) |
| DS-06 | `n_per_config` mode | `holdout={seed:0,n_per_config:5}`, two configs | build val+test+train | exactly 5 holdout/config; rest train; deterministic across rebuilds (k-smallest-u, stable ties) | [determinism] | two_sources |
| DS-07 | min-points cheap==array + `>=` boundary | known per-event counts; threshold == one event's exact count | (a) `min_points=threshold` cheap path; (b) array-count primary stream `>= threshold` | identical surviving `local_idx` sets; boundary event KEPT (`>=`); `op='>'` drops it (colleague-parity diff) | [new-behavior] + [golden] | jaxtpc/lucid root (stamped) |
| DS-08 | empty source | two sources, one with `min_points` so high all fail | build; inspect `datasets`/`data_list` | `len(datasets)==2`; no `data_list` entry with `source_idx==1`; mapping intact; all-empty → `ValueError` | [new-behavior] | two_sources |
| DS-09 | identity stable across modalities | same sources/seed, `('sensor',)` vs `('step','labl')` | compare `event_identity` over local_idx intersection | identical `(config_id, source_event_idx)` (modality-independent) | [determinism] | jaxtpc/lucid root (stamped) |
| DS-10 | `volume` orthogonality (JAXTPC) | `n_volumes=2`, build `volume=0` and `volume=1`, same seed/split | compare `event_identity` sets + split membership | identical (volume is orthogonal view, not holdout axis) | [new-behavior] | jaxtpc_data_root (stamped) |
| DS-11 | `get_data_name` uniqueness across configs | two sources, **identically-named** shards (`wc_sensor_0000.h5`) | collect `{get_data_name(i)}` | set size == `len(ds)`; names carry `config_0/`…`config_1/` prefix; no-prefix would collide (regression sentinel) | [new-behavior] + [golden] | two_sources(same name) |
| DS-12 | probe contract shape | any 2-source build | assert `data_list` `list[(int,int)]`; `datasets` `list[dict]` with `source_root` realpaths; run probe `_event_keys`/`_dataset_split` | `_event_keys` non-None set of `(source_key,event_idx)`; `_dataset_split` returns split str; disjoint train/val builds → `train_keys & val_keys == set()` | [new-behavior] | two_sources |
| DS-13 | manifest cache invalidation | build once (writes cache); record filename+mtime | (a) rebuild unchanged; (b) `touch` a shard; (c) truncate the `.npz` | (a) no rescan (counter monkeypatch), identical `data_list`; (b)+(c) rescan, identical `data_list`, new cache file | [new-behavior] | jaxtpc/lucid root |
| DS-14 | atomic write safety | monkeypatch `_scan_manifest` to write tmp then raise before `os.replace` | build (fails), inspect cache dir, rebuild clean | no partial `.npz` at final path (only orphan `.tmp.<pid>`); clean rebuild valid | [new-behavior] | jaxtpc/lucid root |
| DS-15 | fallback when `source_event_idx` absent | fixture WITHOUT `source_event_idx` (`source_event_idx=None`) | build; capture logs | exactly one `log.warning`/source about positional fallback; still partitions deterministically for fixed shard set; reordering NOW changes membership (degraded guarantee documented, §5.5) | [determinism] | default fixture |
| DS-16 | split validation + `holdout` alias | `split='bogus'`; `split='holdout'`; `data_root` + `sources` both given; duplicate `config_id`; `fractions` not summing to 1 | construct each | `ValueError` for bogus/both/dup/bad-fractions; `holdout`→`val` with one-time warning | [new-behavior] | default fixture |
| DS-17 | `data_root` back-compat alias | `LUCiDDataset(data_root=root, ...)` (subclass first positional) | build | works unchanged; `sources` normalized to `[data_root]`; `len>0` | [migration-smoke] | lucid_data_root |
| DS-19 | **cross-modality alignment (A5 / joint index, D42/D44/D47)** | (a) `('step','sensor','hits','labl')`, `min_deposits>0`, step mask non-contiguous; (b) gap in one modality only (e.g. hits missing a middle `event_*`); (c) `volume=0, min_deposits>0`, some events' deposits all in volume 1; + `min_deposits>0` w/ `('hits','labl')` (no step) | for every served idx read each modality, compare `source_event_idx`; for (c) compare survivors vs volume-blind sum + `event_identity` vs `volume=1` | (a)+(b) **same** `source_event_idx` across all loaded modalities for every idx (== `event_identity(idx)[1]`); A4 warn (or `ValueError` under `strict_lengths`) reports per-modality counts; (c) volume-0-below-threshold events dropped, `event_identity` unchanged vs `volume=1`, no-step case raises. **All variants FAIL on `master`/HEAD + the pre-joint-index design** (Part 02 §6.16, handoff §5 A5) | [determinism] + [new-behavior] | §3.7 corpus + jaxtpc(n_volumes=2) |

De-dup note: Part 02 6.1–6.15 map 1:1 here (6.4→DS-04 golden, 6.7→DS-07, 6.15→DS-15);
the split-validation edge cases (Part 02 §5) are DS-16; **Part 02 §6.16
(cross-modality A5 / joint-index regression) is DS-19** — it is also Phase A's own
GATE (Part 06 §4 Phase A), folded into this Step-0 matrix per D43. The
`TestModeMixin`/`prepare_test_data` byte-identity (Part 02 §3.0) is covered by DS-18
below (migration).

### 4.3 Readers (Part 03) — `tests/test_readers_meta.py` (NEW) + `test_lucid.py`/`test_jaxtpc.py` extend

| id | guards | setup | action | EXPECTED | tag | fixture |
|---|---|---|---|---|---|---|
| RD-01 | LUCiD sensor `read_meta` n_hits==count | `make_lucid_sample(n_hits=120, stamp_n_hits=True)` | `r.read_meta(0)['n_hits']` vs `len(r.read_event(0)['sensor_idx'])` | equal (120) | [new-behavior] | lucid_data_root (stamped) |
| RD-02 | JAXTPC step `read_meta` n_hits==Σ deposits + 3-tuple locate | `make_jaxtpc_sample(n_volumes=2,n_deposits=60)` | `JAXTPCStepReader.read_meta(0)` | `set(m)=={'source_event_idx','n_hits'}`; `n_hits==120==read_event(0)['coord'].shape[0]` (guards the 3-tuple-unpack copy-paste bug, Part 03 §5.4) | [new-behavior] + [golden] | jaxtpc_data_root (stamped) |
| RD-03 | JAXTPC sensor n_hits==Σ n_pixels (D40 invariant) | wire AND pixel fixtures, `pg.attrs['n_pixels']` stamped | `read_meta(0)['n_hits']` vs `sum(len(v) for k,v in read_event(0).items() if k.endswith('.value'))` | equal for both readouts | [new-behavior] | jaxtpc_data_root + jaxtpc_pixel_data_root |
| RD-04 | JAXTPC hits n_hits==Σ n_actual | `vol.attrs['n_actual']` stamped | `JAXTPCHitsReader.read_meta(0)['n_hits']` | == `n_volumes*n_deposits` | [new-behavior] | jaxtpc_data_root (stamped) |
| RD-05 | `per_interaction` surfaced + CSR (LUCiD labl) | `with_per_interaction=True`, I=2, known ragged (2,1) primaries | `LUCiDLablReader.read_event(0)` | all 10 `labl_interaction_<scalar>` keys, shape `(I,)`, correct dtype; 6 CSR keys; `offsets (I+1,)`, `offsets[-1]==data.size`; `data[off[0]:off[1]]`==known primaries of interaction 0; **no `target_vertex` key** (that is Part 04) | [new-behavior] + [golden] | lucid (with_per_interaction) |
| RD-06 | `T_reco` present + aligned + masked; absent variant | `with_t_reco=True` and a `False` variant; `pe_threshold=p` | `LUCiDHitsReader(pe_threshold=p).read_event(0)` | `'t_reco'` present, `t_reco.shape==t.shape==sensor_idx.shape` after mask, `t_reco != t`; absent variant → key omitted (not synthesized) | [new-behavior] | lucid (both variants) |
| RD-07 | `source_event_idx` vector==attr, != positional | `source_event_idx='shuffled'`, vector + attr both stamped (sensor+labl) | per event, vector-path `read_meta` vs direct attr read | equal per event; equal across sensor vs labl; **≠ positional idx** (proves identity not position) | [new-behavior] + [golden] | lucid (shuffled) |
| RD-08 | `source_event_idx` fallback `None` | variant with neither vector nor attr (`source_event_idx=None`) | `read_meta(0)['source_event_idx']` | `None`; reader does not raise, does not warn (base's job) | [new-behavior] | default fixture |
| RD-09 | `read_meta` attr-only cost guard (all 8 readers) | any fixture; wrap event group so dataset `__getitem__` slicing raises, `.attrs`/`.shape`/iteration allowed | call `read_meta` on each reader | succeeds for all 8 (no array decode); JAXTPC step legacy flat uses `positions.shape` (metadata) only | [new-behavior] | jaxtpc + lucid roots |
| RD-10 | absent count attr → 0 | `stamp_n_hits=False` | `read_meta(0)['n_hits']` | `0` (never raises) | [new-behavior] | unstamped fixture |
| RD-11 | `format_version` docstring/fixture == 5 | — | grep four LUCiD reader modules for `format_version: 3`; check fixture `config.attrs['format_version']` | zero matches; fixtures stamp `5` | [golden] | none |

De-dup note: Part 03 6.1–6.11 → RD-01..RD-11 (6.10→RD-09 cost guard, 6.11→RD-11).
LUCiD labl `contained` dtype guards already in `test_lucid.py:391-426` — keep.

### 4.4 Label decoration (Part 04) — `tests/test_label_decoration.py` (NEW)

Reference values are **hand-computed from the fixture's raw FK arrays**
(`testing.py` invariants), independent of decorator code — the model is the
existing `test_lucid.py:104-112` cross-recompute.

| id | guards | setup | action | EXPECTED | tag | fixture |
|---|---|---|---|---|---|---|
| LD-01 | LUCiD hits == hand positional gather | `LUCiDDataset(modalities=('hits','labl'))` | `sub=get_data(0)['hits']`; recompute `ref_seg=category[particle_idx]`, `ref_inst=particle_idx` | `segment_pid.ravel()==ref_seg`; `instance_particle.ravel()==particle_idx`; int32; seg ∈ `[0,5)`, inst `< n_particles` | [new-behavior] + [golden] | lucid_data_root |
| LD-02 | JAXTPC hits == hand searchsorted gather | `JAXTPCDataset(modalities=('hits','labl'),readout_type='wire')` | per plane entry `k`: `tid=g2t_v{v}[gid_k]`, `row=where(track_ids==tid)`, `ref_seg_k=track_pdg_v{v}[row]` | `segment_pid.ravel()==ref_seg` (concat sorted-plane order); `instance_particle.ravel()==concat(group_id)`; seg ∈ `{13,11,211,22,2212}`, ≥1 `> 20` | [new-behavior] + [golden] | jaxtpc_data_root |
| LD-03 | named keys present (both detectors) | default `label_config`, `('step','labl')` and `('hits','labl')` | inspect `get_data(0)[stream]` keys | `segment_pid`, `instance_particle`, `instance_interaction` present (+ `instance_ancestor` LUCiD); bare `segment`/`instance` only under `label_key=` back-compat; named keys absent on `sensor` | [new-behavior] | both roots |
| LD-04 | `instance_interaction` one-hop | JAXTPC + LUCiD hits+labl; LUCiD `per_particle.interaction_idx` stamped (§3.1) | JAXTPC recompute `track_interaction_v{v}[row(g2t[gid])]`; LUCiD recompute `per_particle.interaction_idx[particle_idx]` (one-hop) == two-hop where they agree | matches hand one-hop gather; JAXTPC ∈ `{1,2,3}` | [new-behavior] + [golden] | both roots |
| LD-05 | per-event `target_*` shape, not per-point | LUCiD `with_per_interaction=True` (or monkeypatched `event_value` → `(3,)`) | `data=get_data(0)`; inspect `data['_targets']` | `target_vertex (3,)`; `target_energy ()`/`(1,)`; `target_contained` bool scalar; **none** inside `data[stream]`; none in stream `index_valid_keys`; after N-changing transform per-point cols shrink but `_targets` unchanged | [new-behavior] + [golden] | lucid (with_per_interaction) |
| LD-06 | fill on unresolved FK | hand-corrupt one event's `deposit_to_track` to a track_id ∉ `track_ids`, and a `particle_idx` to `n_particles+5` | decorate | those rows `segment_pid==-1` (and `instance==-1` for out-of-range positional); no exception; neighbors unaffected | [new-behavior] | both roots |
| LD-07 | multi-config `event_label`/`config_id` stability | two source roots, mixture w/ explicit `{config:label}`, `event_broadcast` | read several events; recompute per-event `event_label` from source | `event_label`/`config_id` per-point `(N,1)`, constant within event, == source label; same `(config_id,source_event_idx)`→same decoration (pure function, no RNG) | [determinism] | two_sources |
| LD-08 | extensibility (add axis w/o decorator edit) | append `dict(out="segment_ancestor",scope="point",fk="particle_idx",source=("particle","ancestor_particle_idx"),fill=-1)`; `segment*` prefix covered by index_operator | decorate then `GridSample`/`SphereCrop` | `segment_ancestor` emitted == `ancestor_particle_idx[particle_idx]`; stays length-aligned w/ `coord` after N-change; no decorator edit | [new-behavior] | lucid_data_root |
| LD-09 | back-compat single-axis `label_key` (JAXTPC) | `JAXTPCDataset(modalities=('step','labl'),label_key='pdg')` back-compat one-spec | compare vs current `_decorate_step_from_labl` (snapshot arrays) | bare `segment` (raw `track_pdg`) + `instance` (raw track_id) byte-identical to today; following `RemapSegment(scheme='motif_5cls')` same remapped classes | [golden] + [migration-smoke] | jaxtpc_data_root |
| LD-10 | cross-stream consistency (step↔hits) | `('step','hits','labl')` | for deposit `i`, step `segment_pid` vs hits `segment_pid` of an entry in same group (via `deposit_to_track[i]==group_to_track[deposit_to_group[i]]`) | equal (the `testing.py:19-21,114` invariant) | [new-behavior] | jaxtpc_data_root |

De-dup note: Part 04 6.1–6.9 → LD-01..LD-09; LD-10 promotes the §4.2 cross-stream
note in Part 04 to a test. The existing `test_lucid.py:79-128` decoration tests
overlap LD-01/LD-03 on the *bare* keys — keep those (they guard the back-compat
bare path), add LD-* for the *named* keys.

### 4.5 Collate / streams / eval (Part 05) — `tests/test_collate_eval.py` (NEW)

| id | guards | setup | action | EXPECTED | tag | fixture |
|---|---|---|---|---|---|---|
| CE-01 | single-stream collate shape + offset | 3 LUCiD `sensor` samples, counts `(n0,n1,n2)`, seeded `GridSample(train)`→`ToTensor`→`Collect(stream='sensor',keys=('coord','energy'),feat_keys=('coord','energy'))` | `collate_fn([s0,s1,s2])` | `coord (n0+n1+n2,3)`; `feat (·,3)`; `offset.tolist()==[n0,n0+n1,n0+n1+n2]` int; `coord[offset[0]:offset[1]]`==sample 1 coord | [new-behavior] + [golden] | lucid_data_root |
| CE-02 | `event_label` per-point recovery by offset | 3 samples, per-point-broadcast `event_label` `[0,1,0]`, collected w/ `event_label` in `keys` | `collate_fn(...)` then mimic `_labels_by_event(batch['event_label'],[0,n0,n0+n1,N],N)` | `tensor([0,1,0])` via `numel()==n_points` branch; == offset-window-first slice; NOT the `numel()==n_events` branch | [new-behavior] + [golden] | in-memory |
| CE-03 | probe disjointness via `event_identity` | stub dataset exposing `event_identity`/`split`/`__len__`; train (`split='train'`) + val (`split='holdout'`) disjoint; `Subset`-wrapped variants | rewired `_validate_heldout_source` (rank-0 path) | no raise; `train_keys & val_keys==set()`; overlapping pair → `RuntimeError` mentioning leakage + formatted `"config_id:src_evt_idx"`; `Subset` → subset still disjoint; `split='all'` val → `RuntimeError` (forbidden-split) | [new-behavior] | stub |
| CE-04 | `_event_keys` no longer reaches internals | dataset w/ `event_identity`+`split` but NO `data_list`/`datasets` | `_event_keys(dataset)` | returns identity set (not `None`); a dataset lacking `event_identity` → `None` (graceful contract preserved) | [new-behavior] | stub |
| CE-05 | train≡eval transform-equality | parse live config `base_event_transform`/`transform`/`val_transform`; deterministic subset = {NormalizeCoord,Update,GridSample,LogTransform,RelativeLogNormalize} | compare registered `type`+params of `val_transform` det ops vs same slice of `transform` | identical type+param dicts for the shared fragment; only diff is appended tail (val: ToTensor/Collect; train: MultiView+jitter); no drifted `grid_size`/`scale`/`max_val` (§5.6) | [determinism] | live config (requires_config) |
| CE-06 | seam: nested output + per-event decoration preserved | `JAXTPCDataset(modalities=('step','hits','labl'))` | inspect `get_data(0)` | nested — `set(keys) ⊇ {'step','hits'}`, each w/ own `coord`; **no bare top-level `coord`/`segment`**; each stream's labels self-contained; `Collect(stream='step')` then `Collect(stream='hits')` on a deepcopy lift independently; `bridges`/`labl` reachable at top level, NOT in single-stream collected output | [new-behavior] | jaxtpc_data_root |
| CE-07 | `index_operator` keeps `event_label` aligned through N-change | stream dict `coord` + `event_label (N,1)` per-point broadcast + `index_valid_keys` incl. `event_label`; deterministic N-reducing `GridSample` | run | `event_label` subset to new N, still constant within event; per-event `target_* (D,)` (D≠N) NOT subset | [new-behavior] | in-memory |
| CE-08 | `_`-prefixed metadata drop at collate | per-sample dicts each w/ `_ragged` (variable length) + `coord`/`offset` | `collate_fn(batch)` | `_ragged` absent from output; `coord`/`offset` present+correct; no `default_collate` ragged error | [new-behavior] | in-memory |
| CE-09 | collate byte-identity guard | `pimm_data.collate` vs pimm `pimm.datasets.utils` | source compare `collate_fn`/`point_collate_fn`/`inseg_collate_fn` | byte-identical source (REPLACE never drifts) | [parity-vs-branch] + [golden] | none (branch file) |

De-dup note: Part 05 T1–T9 → CE-01..CE-09. CE-05 is the train≡eval assertion
(§5.6); CE-03/CE-04 are the eval-hook rewire; CE-09 the collate REPLACE guard.

### 4.6 De-fork / migration (Step 0–5 smoke) — `tests/test_migration_smoke.py` (NEW)

These guard the rollout itself: that existing training keeps working at each step
and the migrated configs build. They are **build + one `__getitem__`/1-step**, no
GPU. Configs live in pimm (`requires_config` — skip if the pimm configs tree isn't
on the path; §7.5).

| id | guards | setup | action | EXPECTED | tag | fixture |
|---|---|---|---|---|---|---|
| MG-01 | `TestModeMixin` extraction byte-identical | `DefaultDataset` (npy path) + `LUCiD`/`JAXTPC` test-mode | run `prepare_test_data` before/after mixin extraction (snapshot) | `fragment_list` byte-identical; npy path unchanged (guarded `segment` pop is strict superset, Part 02 §3.0) | [migration-smoke] + [golden] | synthetic npy + roots |
| MG-02 | PILArNet/panda/hmae build through shim | re-export shim importable; build each `type="PILArNetH5Dataset"` config | `build_dataset(cfg)` + `ds[0]` | builds; one `__getitem__` returns expected keys; no registry double-registration error | [migration-smoke] | requires_config |
| MG-03 | migrated JAXTPC seg config builds + runs 1 step | `jaxtpc_seg.py` migrated (`modalities=("step","labl")`, `ApplyToStream('step')`, `RemapSegment(scheme='motif_5cls')`, terminal `Collect`) | `build_dataset` + `collate_fn([ds[0],ds[1]])` | batch has `coord (·,3)`/`feat`/`segment`/`offset`; `in_channels=4` path unchanged | [migration-smoke] | jaxtpc_data_root |
| MG-04 | LUCiD SSL config (dissolved `LUCiDEventSSLDataset`) builds | LUCiD SSL config → base + LUCiD `label_config` | `build_dataset` + `ds[0]` | builds; nested `sensor` stream; `event_label` per-point present; probe contract surfaces (`event_identity`/`split`) | [migration-smoke] | requires_config |
| MG-05 | `__init__.py` stale-import fix | shim re-registers `LUCiDEventSSLDataset` successor; `from .lucid_dataset import LUCiDDataset` stale line fixed | import `pimm.datasets` (or the shim) | no `ImportError`; `DATASETS`/`TRANSFORMS` resolve pimm-data classes | [migration-smoke] | requires_config |
| MG-06 | registry re-export exports preserved | shim | assert `collate_fn`/`point_collate_fn`/`inseg_collate_fn`/`DefaultDataset`/`ConcatDataset`/`build_dataset`/dataset classes exported | all present (Part 05 §3.1 shim requirement) | [migration-smoke] | requires_config |
| MG-07 | first-batch tensor parity vs vendored (Step 3 gate) | PILArNet config; pimm-data transforms vs vendored | compare first-batch tensors | identical (`coord`/`feat`/`segment`/`offset`) — the Step 3 "identical first-batch tensors vs vendored" gate | [parity-vs-branch] + [golden] | requires_config + requires_branch |
| MG-08 | `seg|resp|corr|output_mode` grep clean (Step 5 gate) | migrated configs tree | grep `configs/` for `seg\|resp\|corr\|output_mode` legacy markers | zero hits (D33 Step-5 gate) | [migration-smoke] + [golden] | requires_config |

De-dup note: the impl-plan §6 "Migration smoke" bullet (PILArNet/panda/hmae +
migrated JAXTPC + LUCiD SSL build + 1 step) is MG-02/MG-03/MG-04; MG-07/MG-08 are
the explicit Step-3/Step-5 gates from §2.

---

## 5. Determinism & repro test designs (detail)

The headline invariant (Part 02 §4 invariant 1): for fixed `(sources, seed,
fractions, modalities, min_points)`, the set of `event_identity(i)` is identical
across process restarts, machines, DDP world size, NumPy/Python versions, shard
file order, and shard add/remove — because identity keys on the **writer-stamped
`source_event_idx`**, not the dense `local_idx`. These tests simulate every axis
of that invariant **without any real data**, using the §3.7 corpus.

### 5.1 Shard reorder (DS-01)

Holdout bucket = `blake2b(struct.pack('<qqq', seed, config_id,
source_event_idx)) / 2**64` (Part 02 §3.5). `source_event_idx` is the
writer-stamped vector, *not* the position. So renaming `wc_sensor_0003.h5` →
`wc_sensor_0000.h5` (and the other 3 modalities) changes the `local_idx → event`
mapping but **not** the `local_idx → source_event_idx` mapping (the vector is
re-read from each renamed file). Procedure:

1. `make_lucid_sample(d, n_files=4, n_events=8, source_event_idx='shuffled')`.
2. Build `ds_a` (`split='val'`), record `S_a = {ds_a.event_identity(i) for i}`.
3. On disk swap shard `_0000` ↔ `_0003` for all of `step/sensor/hits/labl` (rename
   to temp, then cross-rename) so file *order* changes but content is identical.
4. **Delete the manifest cache** (point `$PIMM_DATA_CACHE` at a fresh tmp dir, or
   bump mtime — DS-13 covers invalidation; here just nuke it) so the rebuild
   re-scans.
5. Build `ds_b`, record `S_b`.
6. `assert S_a == S_b`. Also assert `ds_a.split == ds_b.split == 'val'`.

This is a pure-function-of-identity assertion; it does not depend on order. Run it
for both `fractions` and `n_per_config` modes.

### 5.2 Add / remove events (DS-02)

The hash is per-identity, so adding or removing an event must leave every *other*
event's split assignment untouched. Procedure:

1. Build corpus C32 (4×8) and corpus C32' that **removes** identity `r` and
   **adds** a brand-new identity `a` (new `source_event_idx` not in C32). Easiest:
   write C32 with `source_event_idx=[...32 values...]`, write C32' with that list
   minus `r` plus `a` (still 32 events on disk, but the identity *set* differs by
   `{r}` removed, `{a}` added).
2. Build `split='val'` over each → `V`, `V'`.
3. `removed = V - V'`, `added = V' - V`.
4. `assert removed <= {r}` (r leaves val iff it was in val) and
   `assert added <= {a}` (a enters val iff its bucket is val); critically
   `assert (V & V') == (V - {r})` i.e. every surviving event keeps its assignment.

This proves the holdout is **stable under corpus churn** — the property that lets a
dataset grow over time without reshuffling train/val/test.

### 5.3 DDP rank-identity (DS-03)

The manifest is built rank-0-under-barrier and read identically by every rank
(Part 02 §3.6). Simulate DDP **in-process** with a monkeypatched comm shim
(`_serial_comm` is the production fallback; the test injects a fake that reports
`get_world_size()==4`). Procedure:

1. Monkeypatch the lazily-imported comm: `get_rank()` returns a value the test
   controls, `get_world_size()==4`, `synchronize()` = no-op.
2. Build with `get_rank()==0` first (writes the cache), then rebuild three more
   times with `get_rank()∈{1,2,3}` (each reads the same `.npz`).
3. `assert` all four `data_list` lists are byte-identical and all four
   `{event_identity(i)}` maps agree.

No real `torch.distributed`, no multiprocessing — the comm shim is the seam.

### 5.4 Machine / version (DS-04 — the golden anchor)

Because blake2b is deterministic and `struct.pack('<qqq', ...)` fixes byte order,
`_bucket_u64(seed, config_id, source_event_idx)` is identical on every machine and
NumPy/Python version. Hard-code a **golden vector**: e.g.
`_bucket_u64(0, 0, 7) == <0xXXXX value>` for a handful of known tuples, computed
once and pasted in. `assert _bucket_u64(...) == GOLDEN[...]` for each. This is the
skip-proof guard against an accidental change to the seed scheme, the pack format,
the digest size, or the `/2**64` normalization — a regression here silently
reshuffles every holdout, so it must be golden, not derived.

### 5.5 Fallback when `source_event_idx` absent (DS-15)

With `source_event_idx=None` the reader returns `None`, the base warns once per
source and falls back to `(config_id, positional)`. The test asserts:
- exactly **one** `log.warning` per source (caplog),
- the holdout still partitions deterministically for a **fixed** shard set,
- and explicitly that **reordering shards NOW changes membership** (the inverse of
  DS-01) — documenting the degraded guarantee so nobody assumes stability without
  `source_event_idx`.

### 5.6 train≡eval transform-equality (CE-05)

The eval/probe pipeline must apply the *same registered transforms with the same
params* as training's deterministic subset, so the held-out events see identical
preprocessing (D41 / Part 05 §3.5). Enforced by construction via a shared
`base_event_transform` fragment both splices in; the test **asserts the
construction held**:

1. Parse the live config's three lists: `base_event_transform`, `transform`
   (`= base_event_transform + [...]`), `val_transform`
   (`= base_event_transform + [...]`).
2. Define the **deterministic subset** = ops that affect numeric values
   independent of RNG: `NormalizeCoord`, `Update`, `GridSample`, `LogTransform`,
   `RelativeLogNormalize` (NOT MultiView, jitter, flip, rotate — augmentation,
   train-only).
3. Extract `(type, params)` for those ops from both `transform` and
   `val_transform`.
4. `assert` the two lists of `(type, params)` dicts are **equal** — same
   `grid_size`, `scale`, `max_val`, `keys`, etc. The only allowed difference is the
   appended tail (val: `ToTensor`/`Collect`; train: augmentation + `Collect`).
5. Negative control: a deliberately-drifted `val_transform` (e.g. a different
   `grid_size`) must make the assertion **fail** — include it as a sub-test so the
   assertion has teeth.

This is config-introspection only (no dataset build, no GPU), tagged
`requires_config`. It guards the single most insidious eval bug: a val pipeline
that silently re-voxelizes or re-normalizes differently from train.

---

## 6. Gates → rollout steps (which tests must pass before each Step)

Maps the impl-plan §2 runbook (Steps 0–5) to the test groups that block each. A
step does not land until its gate row is green.

| Step | What lands | Gate (must be green) |
|---|---|---|
| **Step 0** — test matrix (no code move) | the §3 fixture extensions + this whole suite scaffolded | **P07-FIX merged**; TR-01..TR-24 (transforms, in-memory + jaxtpc); DS-01..DS-17; RD-01..RD-11; LD-01..LD-10; CE-01..CE-09 all green on synthetic fixtures. Parity tests either pass (branch importable) or skip **with their golden pair green** (§7.4). This is the pure gate — no pimm change. |
| **Step 1** — additive build in pimm-data (transform merges, `index_operator`, base+`TestModeMixin`, readers, `label_config`) | `transform.py`, `multimodal.py`, readers, decorator | Step-0 suite green **end-to-end** (now exercising real code, not just fixtures). Specifically: TR-* (merges), DS-* (base), RD-* (readers), LD-* (decoration), MG-01 (`TestModeMixin` byte-identical). `DefaultDataset`/npy path unchanged (MG-01 golden). |
| **Step 2** — re-export shim in `pimm/datasets/__init__.py` | shim (byte-identical REPLACE files first) | MG-02 (PILArNet 1-step through shim), MG-05 (`__init__` stale-import fix), MG-06 (exports preserved), CE-09 (collate byte-identity). |
| **Step 3** — flip transforms + PILArNet | `segment_motif`/`PDGToSemantic` resolve from pimm-data | **MG-07** (identical first-batch tensors vs vendored) — the named Step-3 gate; plus TR-* parity green (or golden-paired). |
| **Step 4** — migrate JAXTPC configs, flip `JAXTPCDataset`, dissolve `LUCiDEventSSLDataset` | `jaxtpc_seg.py` migrated; LUCiD SSL → base+config | MG-03 (JAXTPC seg builds + 1 step), MG-04 (LUCiD SSL builds + 1 step), CE-05 (train≡eval), CE-03/CE-04 (eval-hook rewire), DS-12 (probe contract). |
| **Step 5** — delete vendored files | last commit | **MG-08** (`seg\|resp\|corr\|output_mode` grep clean), full parity suite green, MG-02..MG-07 green, the whole §4 matrix green. (D33 gate also requires a soaked ≥1 full PILArNet run — out of CPU-CI scope; flag as a manual gate.) |

The single hard ordering constraint baked into the tests: CE-03/CE-04 (probe
rewire) **hard-require** `dataset.event_identity` (Part 02). If the probe rewire is
attempted before the base lands, `_event_keys` returns `None` and the guard raises
"cannot verify disjointness" — CE-04 asserts exactly that graceful-`None` contract,
so landing order is enforced by the test, not just the runbook.

---

## 7. Running (commands, markers, branch-importability, golden strategy)

### 7.1 Markers (declare in `pyproject.toml`)

Add to `[tool.pytest.ini_options]`:

```toml
markers = [
    "slow: kernel/heavy synthetic-data tests (e.g. 1000-event stratification, MixedScale).",
    "requires_branch: needs the colleague's pimm branch transform.py importable (parity).",
    "requires_config: needs the pimm configs/ tree + registries on the path (migration).",
    "real_data_only: skip on the synthetic fixture (already registered in conftest.py:50).",
]
```

- **`slow`** — DS-05 (1000×2 stratification), TR-23 (MixedScale parity). Excluded
  by `-m "not slow"` for the fast inner loop.
- **`requires_branch`** — every `[parity-vs-branch]` test; auto-skips when the
  branch module can't be loaded (§7.3).
- **`requires_config`** — every `[migration-smoke]`/`requires_config` test
  (MG-02/04/05/06/07/08, CE-05); auto-skips when `pimm`'s `configs/`/registries
  aren't importable. Apply the `test_cache.py:24-27` `skipif` pattern.

### 7.2 Commands

```bash
# Fast inner loop — synthetic only, no GPU/JAX, no branch, no pimm configs:
JAX_PLATFORM_NAME=cpu python3 -m pytest tests/ -v -m "not slow and not requires_branch and not requires_config"

# Full synthetic suite incl. parity (branch must be importable):
JAX_PLATFORM_NAME=cpu python3 -m pytest tests/ -v -m "not requires_config"

# Everything incl. migration smoke (pimm configs on the path):
JAX_PLATFORM_NAME=cpu PYTHONPATH=/sdf/home/o/omara/.claude/jobs/21ffc656/particle-imaging-models \
  python3 -m pytest tests/ -v

# A single group:
JAX_PLATFORM_NAME=cpu python3 -m pytest tests/test_multimodal.py -v
```

`JAX_PLATFORM_NAME=cpu` is set defensively (no test imports `jax`, but pimm-data is
installed alongside JAXTPC whose import path could pull JAX transitively in a
migration test — pinning CPU keeps it honest and matches the JAXTPC repo
convention). No test imports `torch.cuda`; `torch` is CPU tensors only.

### 7.3 Branch-importability handling (the parity loader)

The branch `transform.py` does `from pimm.utils.registry import Registry`
(branch `transform.py:22`), and **`pimm` is not installed** in the pimm-data env
(`python3 -c "import pimm"` → `ModuleNotFoundError`). So a plain
`importlib` of the branch file fails on import, not just on absence. The parity
harness must **stub `pimm.utils.registry` before loading the branch module**. Put a
session fixture in `conftest.py`:

```python
@pytest.fixture(scope="session")
def branch_transform_module():
    import importlib.util, sys, types
    path = ("/sdf/home/o/omara/.claude/jobs/21ffc656/"
            "particle-imaging-models/pimm/datasets/transform.py")
    if not os.path.exists(path):
        pytest.skip(f"branch transform.py not found at {path}")
    # Stub pimm.utils.registry with pimm-data's own Registry so the branch's
    # `from pimm.utils.registry import Registry` resolves (the Registry impls are
    # compatible for transform registration).
    if "pimm" not in sys.modules:
        from pimm_data._registry import Registry
        pimm = types.ModuleType("pimm")
        utils = types.ModuleType("pimm.utils")
        reg = types.ModuleType("pimm.utils.registry")
        reg.Registry = Registry
        pimm.utils = utils; utils.registry = reg
        sys.modules.update({"pimm": pimm, "pimm.utils": utils,
                            "pimm.utils.registry": reg})
    try:
        spec = importlib.util.spec_from_file_location("branch_transform", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as e:
        pytest.skip(f"branch transform.py not importable: {e}")
    return mod
```

Every `[parity-vs-branch]` test **requests this fixture** and is decorated
`@pytest.mark.requires_branch`. A missing/unimportable branch → `pytest.skip` with
a clear reason; it never errors the run. (The stub uses pimm-data's own `Registry`
from `_registry`, which the branch only uses for `@register_module()` decoration —
compatible. If the branch class under test pulls a second pimm symbol, stub it the
same way; today only `Registry` is needed for the four parity classes
`RelativeLogNormalize`/`GridSample`/`Collect`/`MixedScale...`.)

For **CE-09** (collate byte-identity) the reference is pimm's
`pimm/datasets/utils.py`, read as a *file* (string compare), not imported — so it
needs no stub, only `os.path.exists` + a `requires_branch` skip.

### 7.4 Golden-array strategy (skip can't mask a regression)

**Rule:** every `[parity-vs-branch]` test is paired with a `[golden]` so that when
the branch is unavailable, the golden still fails on a regression. The pairing:

| parity test | golden pair |
|---|---|
| TR-05 (RelativeLogNormalize parity) | TR-01 (hard-coded `[-1.0,-0.1998,-0.1274,1.0]`) |
| TR-12 (LogTransform clip parity) | TR-12's own finite-range golden + TR-13 |
| TR-13 (LogTransform default parity) | TR-13 (captured pre-merge golden array) |
| TR-16 (vertex parity) | TR-16's hard-coded single flip case + TR-18 |
| TR-18 (vertex-absent no-op parity) | TR-18 (captured pre-merge golden) |
| TR-23 (MixedScale parity) | a small golden of `global_offset`/`local_offset` shapes+a checksum (full-array golden is large; checksum is the skip-proof anchor) |
| CE-09 (collate byte-identity) | CE-09 itself reads the file and string-compares — golden is the pimm source bytes; if the file is gone it skips, but the **import-and-behavior** of pimm-data's own collate is exercised by CE-01/CE-08 |
| MG-07 (first-batch parity vs vendored) | a captured first-batch tensor checksum golden |

How to capture a golden: run the pimm-data implementation **once** during
development, `np.save` / paste the array (or a `hashlib.sha1(arr.tobytes())`
checksum for large arrays) into the test as a literal, and assert against it. A
checksum is acceptable for big arrays; a full literal for ≤~16 values (TR-01).
Goldens are committed; they are the regression floor independent of the branch.

**Why this matters:** a parity test that only runs when the branch is importable
can be silently skipped in CI (branch path absent) and never catch a regression.
The golden pair runs **unconditionally**, so a regression in pimm-data's own
output is caught even with the branch gone. The parity test then adds the stronger
"byte-equal to the colleague" assertion when the branch *is* present.

### 7.5 `requires_config` handling

Migration-smoke tests build pimm configs (`type="PILArNetH5Dataset"`,
`jaxtpc_seg.py`, the LUCiD SSL config). These need pimm's `configs/` tree and its
`DATASETS`/`TRANSFORMS` registries resolvable. Gate with a module-level
`importorskip`/`skipif`:

```python
pytestmark = pytest.mark.requires_config
configs_root = os.environ.get("PIMM_CONFIGS_ROOT")
if configs_root is None or not os.path.isdir(configs_root):
    pytest.skip("pimm configs tree not available (set PIMM_CONFIGS_ROOT)",
                allow_module_level=True)
```

For configs that only need the *registry* (not the file tree), build a minimal
config dict inline and `build_dataset(cfg)` against the synthetic fixture root
(MG-03 does this with `jaxtpc_data_root`) — that path needs no pimm checkout, only
pimm-data, so MG-03 runs in the fast loop while MG-02/04/05/06/07/08 are
`requires_config`.

---

## 8. Risks

- **Skip masking (the headline risk).** A `[parity-vs-branch]` test that skips
  (branch absent) must never be the *only* guard on a behavior. Mitigation: §7.4's
  mandatory golden pairing — every parity test has an unconditional golden anchor.
  Reviewers must reject any new parity test without a golden pair. CE-09's
  file-read skip is backstopped by CE-01/CE-08 exercising pimm-data's own collate.

- **Fixture-vs-real drift.** The synthetic fixtures encode an *assumed* schema
  (attr names, CSR layout, `format_version=5`). If real WAND/JAXTPC files diverge
  (e.g. a writer renames `n_pixels`, or stamps `n_hits` differently), the suite
  stays green while production breaks. Mitigations: (a) the `real_data_only` marker
  + env-var override (`conftest.py:37-71`) lets the *same* tests run against a real
  shard when `JAXTPC_DATA_ROOT`/`LUCID_DATA_ROOT` is set — run this in a nightly,
  not the fast loop; (b) RD-11 grep-guards the `format_version` literal; (c) the
  Part 03 §3 attr names are the contract — any fixture change must mirror a
  confirmed-schema change, not invent one. The DS-07 "cheap == array" and RD-01..04
  invariants would catch a count-attr mismatch *on the fixture*, but only a
  real-data run catches a fixture-vs-real attr-name drift — call that out in the
  nightly job.

- **Branch availability.** The colleague branch lives under
  `~/.claude/jobs/21ffc656/...` (outside the pimm-data tree). It may move, be
  cleaned, or have its `import pimm` surface change. Mitigations: the §7.3 loader
  stubs `pimm.utils.registry` (so a missing *pimm install* doesn't block parity)
  and skips cleanly on a missing *file*; the golden pairs make a skip non-fatal.
  If the branch later needs a second pimm symbol, the stub set must grow — flagged
  in §7.3.

- **`source_event_idx` coverage asymmetry.** WAND has both the per-file vector and
  the per-event attr; JAXTPC has only the per-event attr (no `config/source_event_idx`
  vector), and the JAXTPC `make_labl.py` stand-in stamps neither on labl
  (Part 03 §3.8). The fixtures must mirror this asymmetry (JAXTPC: attr only; LUCiD
  sensor/labl: vector + attr) so RD-07/DS-09 test the *actual* precedence
  (vector→attr→None), not an idealized uniform schema. DS-15/RD-08 explicitly
  exercise the `None`→fallback path so the degraded-determinism case is covered, not
  hidden.

- **Determinism golden brittleness vs value.** DS-04's hard-coded blake2b vector
  will break if anyone changes the seed scheme/pack/digest — which is exactly the
  point (a change there silently reshuffles every holdout). Keep it golden; document
  in-test that an intentional change requires regenerating the vector.

- **`per_interaction` / JAXTPC vertex aspirational.** LD-05's `target_vertex`
  depends on the `per_interaction` scope being surfaced (Part 03) and stamped in the
  fixture (§3.3). For JAXTPC there is no per-interaction vertex table today
  (Part 04 §7) — LD-05 is LUCiD-only until a JAXTPC writer lands one; do not fake a
  JAXTPC vertex. The test stands the scope up via fixture/monkeypatch precisely so
  it doesn't silently pass on an omitted axis.

- **`target_mask` has no producer** (impl §7). Do not author a test that asserts a
  `target_mask` axis — the hmae config references it but `HMAECollate` doesn't emit
  it; that is a drift to flag to the hmae owner, not a behavior to test here.
