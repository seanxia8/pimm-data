# Part 03 — Readers (implementation spec)

**Status:** implementation-ready spec. Single-stream-per-task structure (D35); JAXTPC
in scope alongside LUCiD (D40). Additive only: a new `read_meta(idx)` method on every
reader + a handful of `read_event` surfacing additions. **No `read_event` hot-path
behavior changes** except the explicitly listed extra keys.

**Source decisions:**
- **D10** — surface ALL stored streams/labels now (sensor + `n_hits`; hits
  `particle_idx`/`T_reco`; step `group_id`/`sensor_hits`; labl incl. `per_interaction`).
- **D27** — reader surfacing + cost: add `source_event_idx` + `config_id` + per-event
  `n_hits`; request a per-file `n_hits` **vector** from writers; min-points/index via a
  persisted manifest cache (never array reads in steady state). `read_meta` is the cheap
  selection probe used by that cache build.
- **D40** — JAXTPC is in scope; sensor `n_hits` via Σ`n_pixels`; `track_interaction`
  already surfaced; `source_event_idx` present in files (`save.py`).
- **D42/D46** — the base builds a **joint event index** by intersecting the
  per-modality present-`event_*`-key sets keyed on `source_event_idx` (the desync
  fix, `shard_event_filtering_handoff.md` §4). This part therefore **surfaces each
  reader's present-key set + its `source_event_idx`-per-key** to the base (via the
  already-present `indices`/`cumulative_lengths` + the new `read_meta` /
  `_has_sei_vec`/`_sei_vecs`, §3.0); the cross-reader file opens are deduped by a
  module-level `@lru_cache _read_shard_meta(path)` (D46/A1) shared with the manifest
  scan. The intersection/translation itself lives in the base (Part 02 §3.3a), not
  here.

Implementation-plan anchor: **§3.4** (reader additions; includes the CONFIRMED WAND
schema, one-shard check on `config_000001`). The `config_id` is **not** a reader field —
it is assigned by the base per source (§3.3), so it is out of scope here; this part
delivers `source_event_idx` + `n_hits` only.

**Files (read-only; grounding):**
- LUCiD readers: `src/pimm_data/readers/lucid_sensor.py`, `lucid_step.py`,
  `lucid_hits.py`, `lucid_labl.py`.
- JAXTPC readers: `src/pimm_data/readers/jaxtpc_sensor.py`, `jaxtpc_step.py`,
  `jaxtpc_hits.py`, `jaxtpc_labl.py`.
- JAXTPC writer (attr/dataset names): `JAXTPC/production/save.py` —
  `save_event_sensor` (`save.py:330`), `_save_wire_plane` (`save.py:234`),
  `_save_wire_plane_sparse` (`save.py:264`), `_save_pixel_plane` (`save.py:294`),
  `save_event_step` (`save.py:374`), `save_event_hits` (`save.py:583`); labl writer
  `JAXTPC/production/make_labl.py:159`.
- Synthetic fixtures (need additions for §6): `src/pimm_data/testing.py` —
  `make_jaxtpc_sample` (`testing.py:55`), `make_lucid_sample` (`testing.py:333`) and the
  per-modality `_write_*` writers.
- Reader registry/exports: `src/pimm_data/readers/__init__.py`.

**Locked constraints (do not relitigate):**
1. `read_meta(idx) -> {source_event_idx, n_hits}` reads **only `evt.attrs` (and per-file
   `config/` vectors)** — **never decodes arrays** (no `evt['PE'][:]`, no CSR repeat, no
   cumsum). It is the base's O(1)/event selection probe; the manifest cache (Part 02 §3.4)
   wraps it so steady-state never touches array data.
2. `read_event` (hot path) stays as-is except the listed extra keys; the listed extras are
   **already in the files** (one-shard-confirmed), so surfacing is a read, not a writer ask.
3. Schema names are **hardcoded** at `format_version=5` (confirmed + uniform on
   `config_000001`). **No dynamic schema discovery.** Version-gate on `format_version`
   only if a future file needs it — not needed now.
4. WAND files are **v5**; the four LUCiD reader docstrings stale-say `format_version: 3`.
   **Fix the docstrings** (`lucid_sensor.py:3`, `lucid_step.py:3`, `lucid_hits.py:3`,
   `lucid_labl.py:3`) to v5 as part of this part — a comment-only change, no code impact.
5. Absent `source_event_idx` → D26 fallback (the **base** warns once and uses
   `(config_id, positional)`); the reader returns `None` for the field, it does not invent
   one. Absent `n_hits` source → reader returns `0` (or the cheapest count it can compute
   attr-only); never raise.

---

## 1. Purpose & scope

This part adds, to **all 8 readers**, a uniform cheap-metadata probe and the small set of
"stored-but-dropped" surfacing fixes called out by D10/D27.

**In scope:**
- A new `read_meta(idx) -> {'source_event_idx': int|None, 'n_hits': int}` on every reader.
  Per-reader the `n_hits` source attr/dataset differs (table in §3); all are attr-only.
- A per-file `source_event_idx` fast path: prefer the per-file vector
  `config/source_event_idx` `uint32 (n_events,)` (LUCiD sensor **and** labl) — O(1)/file,
  O(1)/event lookup; fall back to the per-event attr `event_NNN.attrs['source_event_idx']`;
  fall back to `None`.
- `read_event` surfacing additions:
  - **LUCiD hits** `+T_reco` (currently dropped).
  - **LUCiD labl** `+per_interaction` scope (currently dropped) — exact field names + CSR
    handling below.
  - **JAXTPC**: nothing new for `group_id` (hits reader already emits `hits.{plane}.group_id`,
    `jaxtpc_hits.py:275/281`) and nothing for `track_interaction` (labl reader already emits
    `labl_v{N}_track_interaction`, generic loop `jaxtpc_labl.py:139-144`).
- The shared `_locate_event` is **reused verbatim** by `read_meta` (the gotcha: JAXTPC
  `jaxtpc_step._locate_event` returns a **3-tuple** — §5).
- The `format_version: 3 → 5` docstring fix.

**Out of scope (other parts):**
- `config_id` assignment (base, Part 02 §3.3).
- The manifest cache itself (base, Part 02 §3.4) — this part only provides the probe.
- Label decoration / FK→named-key mapping that *consumes* `per_interaction`/`group_id`
  (Part 04). This part surfaces raw fields only.
- The optional writer-side per-file `n_hits` vector (a reversible ask; §7).

---

## 2. Current state (per reader — what's emitted, what's dropped)

No reader has a `read_meta` today (grep `def read_meta` → none). All eight share the same
`_build_index`/`h5py_worker_init`/`_locate_event`/`__len__`/`close` skeleton.

### LUCiD

| reader | file:line | `read_event` emits | dropped / not surfaced |
|---|---|---|---|
| `LUCiDSensorReader` | `lucid_sensor.py:124` | `sensor_idx`, `pmt_pe` (`evt['PE']`), `pmt_t` (`evt['T']`), `pmt_coord` (file-level `config/sensor_positions`) | per-event `evt.attrs['n_hits']`; per-event/per-file `source_event_idx`. `_build_index` reads only `config.attrs['n_events']`/`n_sensors` (`lucid_sensor.py:88-90`). |
| `LUCiDStepReader` | `lucid_step.py:125` | `coord` (start/end midpoint), `energy`, `time`, `track_idx`, `contained`; +physics `direction`/`beta_start`/`n_cherenkov` | `source_event_idx`. `n_segments` IS already read in `_build_index` for `min_segments` (`lucid_step.py:92`) — reused for `read_meta`. |
| `LUCiDHitsReader` | `lucid_hits.py:105` | `sensor_idx`, `particle_idx`, `pe` (`evt['PE']`), `t` (`evt['T']`) | **`T_reco`** (stored, dropped — D10); per-event `evt.attrs['n_particle_hits']`; `source_event_idx`. |
| `LUCiDLablReader` | `lucid_labl.py:211` | `labl_event_{t0,contained}`; `labl_particle_*` (category/contained/genealogy CSR + derived `ancestor_particle_idx`); `labl_track_*` (`track_id`/`pdg`/`parent_id`/`particle_idx`/`ancestor`/`interaction`/`initial_energy`/`n_cherenkov` + derived) | **`per_interaction`** scope entirely (the docstring `lucid_labl.py:15-20` explicitly notes it is "not yet surfaced"); per-file `config/source_event_idx`; per-event `evt.attrs['source_event_idx']`. Has `n_particles`/`n_tracks` attrs but no point count. |

`LUCiDLablReader._build_index` reads only `config.attrs['n_events']` (`lucid_labl.py:131`).
The labl file is not a points stream → its `read_meta` `n_hits` is **n/a** (§3, returns 0;
the base never uses labl for min-points).

### JAXTPC

| reader | file:line | `read_event` emits | dropped / not surfaced |
|---|---|---|---|
| `JAXTPCSensorReader` | `jaxtpc_sensor.py:211` | `sensor.{plane}.{wire,time,value}` (wire) / `{py,pz,time,value}` (pixel), delta-decoded; `readout_type` attr | per-event `evt.attrs['source_event_idx']` (writer stamps it, `save.py:344`); a sensor "n_hits" total — **no single attr**, must Σ plane `n_pixels`. Writer stamps `evt.attrs[f'n_vol{v}']` (`save.py:349`) = per-volume `n_actual` deposit count (NOT sensor pixels) and per-plane `g.attrs['n_pixels']` (`save.py:261/291/327`). |
| `JAXTPCStepReader` | `jaxtpc_step.py:141` | `coord`, `energy`, `volume_id`; +physics `dx`/`theta`/`phi`/`t0_us`/`charge`/`photons` | per-event `evt.attrs['source_event_idx']` (`save.py:391`); a deposit count — Σ per-volume `vg.attrs['n_actual']` (`save.py:400`). The `min_deposits>0` branch already does this Σ (`jaxtpc_step.py:91-97`). |
| `JAXTPCHitsReader` | `jaxtpc_hits.py:220` | `hits.{plane}.{wire/py,pz,time,group_id,charge}` (CSR-decoded); per-volume `group_to_track_v{N}`/`deposit_to_group_v{N}`/`qs_fractions_v{N}` | per-event `evt.attrs['source_event_idx']` (`save.py:625`); a count — Σ per-volume `vol_grp.attrs['n_actual']` (`save.py:635`). **`group_id` already surfaced** (`jaxtpc_hits.py:275/281`). |
| `JAXTPCLablReader` | `jaxtpc_labl.py:111` | per-volume `labl_v{N}_track_ids` + generic `labl_v{N}_{col}` for every other dataset (`track_pdg`/`track_cluster`/`track_interaction`/`track_ancestor`/`deposit_to_track`, `jaxtpc_labl.py:139-144`) | `source_event_idx`. **`track_interaction` already surfaced** (generic loop). The current `make_labl.py` writer (`make_labl.py:194`) stamps only `event_id` on the labl event group, **not** `source_event_idx` — see §5/§7. |

**JAXTPC `_locate_event` shapes:** sensor/hits/labl return a **2-tuple** `(f, event_key)`
(`jaxtpc_sensor.py:103`, `jaxtpc_hits.py:103`, `jaxtpc_labl.py:103`); **step returns a
3-tuple** `(f, event_key, n_volumes)` (`jaxtpc_step.py:131`). This is the §5 gotcha.

---

## 3. Target design (per reader)

### 3.0 Shared `read_meta` contract & helpers

Every `read_meta` follows the identical skeleton — locate the event via the reader's own
`_locate_event` (reused, never reimplemented), then read attrs only:

```python
def read_meta(self, idx):
    if not self._initted:
        self.h5py_worker_init()
    f, event_key = self._locate_event(idx)          # JAXTPC step: 3-tuple — see §5
    evt = f[event_key]
    return {
        'source_event_idx': self._source_event_idx(f, idx, evt),
        'n_hits': <per-reader attr-only count>,
    }
```

Return contract:
- `source_event_idx`: a Python `int`, or `None` when neither the per-file vector nor the
  per-event attr is present (→ base D26 fallback). Cast `np.uint32` → `int`.
- `n_hits`: a non-negative Python `int`. `0` when the source attr is absent (never raise).
  For the labl readers `n_hits` is **n/a** → return `0` (labl is never a min-points modality).

**Shared `source_event_idx` resolver** (one private helper per reader, or a tiny mixin —
implementer's call; the *logic* is fixed). Precedence, cheapest first:

1. **Per-file vector** `config/source_event_idx` `uint32 (n_events,)`, indexed by the
   event's *local* row. Cache the vector per file handle on first touch (it is one small
   1-D array per shard, read once). Confirmed present on LUCiD **sensor** and **labl**.
   The local row = the same `local_idx`/`event_num` the reader already computes inside
   `_locate_event`; recompute it cheaply or thread it out (see note below).
2. **Per-event attr** `evt.attrs['source_event_idx']` (LUCiD sensor; all four JAXTPC via
   `save.py`; LUCiD labl per-event attr per the confirmed schema). Fallback when no vector.
3. `None`.

Note on the local index: `_locate_event` computes `local_idx`/`event_num` internally but
the LUCiD/JAXTPC variants return only `(f, event_key[, n_volumes])`. Two acceptable
implementations (reversible, implementer's choice):
- (a) re-derive `file_idx`/`local_idx` in `read_meta` with the same two `searchsorted`
  lines (3 lines, no array reads), then index `config/source_event_idx[local_idx]`; or
- (b) parse the event number back out of `event_key` (`int(event_key.rsplit('_',1)[1])`)
  and, for the vector path, map it to the local row via `self.indices[file_idx]`.
  Default recommendation: **(a)** — mirrors `_locate_event`, no string parsing, and the
  per-file vector is indexed by local row (matches how `config/n_events` is laid out).

The per-event attr path (2) needs only `evt`, so it works without the local row.

**Exposure to the base for O(1)/file identity (frozen surface).** The base's manifest
scan (Part 02 §3.6) wants to read a whole file's identity in **one vector read** rather
than `read_meta`-per-event when the per-file `config/source_event_idx` vector exists. The
reader exposes that fast path with three attributes set in `_build_index`
(LUCiD sensor + labl, where the vector is confirmed) / left empty elsewhere:

```python
# set during _build_index, inside the per-file `with h5py.File(...)` block:
self._sei_vecs   # list parallel to self.h5_files; element = config/source_event_idx[:]
                 #   as int64 (n_events_in_file,), or None when the file has no vector
self._has_sei_vec  # bool: True iff EVERY shard carried the vector (base may then
                   #   skip the per-event attr walk entirely for this source)

# public accessor (thin wrapper over the §3.0 resolver), for the base + tests:
def source_event_idx(self, idx) -> int | None:   # vector -> attr -> None
    ...
```

- **JAXTPC readers** have no vector: `_sei_vecs` is all-`None`, `_has_sei_vec` is
  `False`, and `source_event_idx(idx)` falls straight to the per-event attr
  (`evt.attrs['source_event_idx']`, `save.py:344/391/625`). No `config` vector read is
  ever attempted (constraint 6).
- **LUCiD sensor + labl** set `_sei_vecs[file_idx] = cfg['source_event_idx'][:].astype(
  np.int64)` when `'source_event_idx' in cfg` (one small 1-D read per shard, at index
  time when the file is already open), `None` otherwise. LUCiD step + hits have no
  confirmed vector → `_sei_vecs` all-`None`, attr path only.
- The base reads `reader._has_sei_vec` to choose the fast path; when `True` it reads
  `config/source_event_idx[:]` once per file; when `False` it calls `read_meta` (or
  `source_event_idx(idx)`) per event during the one-time rank-0 manifest scan. Either way
  `read_meta`'s `source_event_idx` field is the authoritative per-event answer.

All existing reader surfaces (`cumulative_lengths`/`indices`/`h5_files`/`read_event`/
`__len__`/`close`) are unchanged; `read_meta`/`source_event_idx`/`_has_sei_vec`/
`_sei_vecs` are the only additions.

**Present-key set for the joint index (D42/D46).** The base's joint-index step
(Part 02 §3.3a) intersects the `event_*` keys **actually present** across loaded
modalities, keyed on `source_event_idx`. The per-modality present-key set is already
materialized by the gap-tolerant `_build_index` (`indices` = per-file present
event-number arrays; `cumulative_lengths`) from `0757ee0`; combined with the §3.0
`source_event_idx(idx)` resolver this gives the base, per modality, the
`{source_event_idx -> local_idx}` map it intersects. **No new reader method is
required for this** — the base reads `indices`/`cumulative_lengths` + `read_meta`/
`source_event_idx`. The only D46 add is the module-level `@lru_cache`
`_read_shard_meta(path) -> (n_events, n_volumes, present_event_keys, readout_type)`
(handoff A1) the readers call inside `_build_index` so the base's joint-index scan
and the manifest scan (Part 02 §3.4/§3.6) do not re-open each of the ~800 doraemon
shards 3× — it collapses the redundant per-reader opens. It is an internal
memoization, no API change.

### 3.1 `LUCiDSensorReader.read_meta`

- `source_event_idx`: vector `config/source_event_idx` (preferred) → attr
  `evt.attrs['source_event_idx']` → `None`.
- `n_hits`: `int(evt.attrs['n_hits'])` (confirmed per-event scalar attr; `0` if absent).
- `read_event`: **unchanged.**

### 3.2 `LUCiDStepReader.read_meta`

- `source_event_idx`: `evt.attrs['source_event_idx']` → `None`. (No `config` vector
  confirmed on step; do not assume one — attr path only, falling back to `None`.)
- `n_hits`: `int(evt.attrs.get('n_segments', 0))` — the segment count (same attr the
  `min_segments` index filter already reads, `lucid_step.py:92`). This is step's natural
  point count; the base treats "step min-points" as a segment count.
- `read_event`: **unchanged** (`contained`/`direction`/`beta_start`/`n_cherenkov` already
  emitted; `group_id`/`sensor_hits` from D10 are not LUCiD-step fields).

### 3.3 `LUCiDHitsReader.read_meta` + `+T_reco`

- `source_event_idx`: `evt.attrs['source_event_idx']` → `None` (no confirmed hits vector).
- `n_hits`: `int(evt.attrs['n_particle_hits'])` (`0` if absent). This is the per-particle
  entry count `E` (multi-hit per PMT), not the unique-PMT count — that is the correct
  cheap proxy for the hits stream's point count.
- `read_event` **+`T_reco`** (D10): the file stores a reconstructed time alongside `T`.
  After the existing `t = evt['T'][:].astype(np.float32)` (`lucid_hits.py:115`), add:

  ```python
  if 'T_reco' in evt:
      t_reco = evt['T_reco'][:].astype(np.float32)
  ```

  Apply the **same `pe_threshold` mask** (`lucid_hits.py:117-122`) to `t_reco` so it stays
  row-aligned with `sensor_idx`/`particle_idx`/`pe`/`t`, then add `'t_reco': t_reco` to the
  returned dict. **Guard on presence** (`if 'T_reco' in evt`) — older shards may omit it;
  if absent, do not add the key (do not synthesize from `T`). Update the output-dict
  docstring (`lucid_hits.py:10-16`) to list `t_reco (E,) float32`.

### 3.4 `LUCiDLablReader.read_meta` + `+per_interaction`

- `source_event_idx`: vector `config/source_event_idx` (preferred — confirmed present on
  labl) → attr `evt.attrs['source_event_idx']` → `None`.
- `n_hits`: **n/a** → `0` (labl is not a points stream; the base never min-points on labl).
- `read_event` **+`per_interaction`** (D10): currently the reader handles `per_event`,
  `per_particle`, `per_track` only (`lucid_labl.py:220-249`). Add a fourth block that
  surfaces `per_interaction` **as raw fields** (no decoration — that is Part 04). Mirror the
  existing flat-key convention with a `labl_interaction_` prefix.

  **Scalar/vector fields** (one value per interaction `I`; confirmed names):

  | source dataset | emitted key | dtype |
  |---|---|---|
  | `source_type` | `labl_interaction_source_type` | int32 (file uint8) |
  | `t0` | `labl_interaction_t0` | float32 |
  | `vertex_x` | `labl_interaction_vertex_x` | float32 |
  | `vertex_y` | `labl_interaction_vertex_y` | float32 |
  | `vertex_z` | `labl_interaction_vertex_z` | float32 |
  | `n_primaries` | `labl_interaction_n_primaries` | int32 |
  | `n_particles` | `labl_interaction_n_particles` | int32 |
  | `neutrino_pdg` | `labl_interaction_neutrino_pdg` | int32 (file int16) |
  | `neutrino_energy_MeV` | `labl_interaction_neutrino_energy_MeV` | float32 |
  | `contained` | `labl_interaction_contained` | bool |

  **CSR-encoded primaries** (ragged per interaction; surface the raw `data`/`offsets`
  pairs — `offsets` is `(I+1,)`, `data` is `(Σn_primaries_i,)`; the consumer slices
  `data[offsets[i]:offsets[i+1]]`):

  | source dataset | emitted key | dtype |
  |---|---|---|
  | `primary_track_ids_data` | `labl_interaction_primary_track_ids_data` | int32 |
  | `primary_track_ids_offsets` | `labl_interaction_primary_track_ids_offsets` | int32 |
  | `primary_pdgs_data` | `labl_interaction_primary_pdgs_data` | int32 |
  | `primary_pdgs_offsets` | `labl_interaction_primary_pdgs_offsets` | int32 |
  | `primary_energies_data` | `labl_interaction_primary_energies_data` | float32 |
  | `primary_energies_offsets` | `labl_interaction_primary_energies_offsets` | int32 |

  Reuse the existing `_cast` static method (`lucid_labl.py:159`) for dtypes: register the
  new int fields in `_INT_KEYS` (`source_type`, `n_primaries`, `n_particles`, `neutrino_pdg`,
  the CSR `*_data`/`*_offsets` integer arrays) and treat `contained` as bool (already
  special-cased) and `t0`/`vertex_*`/`neutrino_energy_MeV`/`primary_energies_data` as
  float32 (the default branch). Implementation pattern — add tuples and a guarded block:

  ```python
  _INTERACTION_SCALAR_KEYS = (
      'source_type', 't0', 'vertex_x', 'vertex_y', 'vertex_z',
      'n_primaries', 'n_particles', 'neutrino_pdg', 'neutrino_energy_MeV',
      'contained',
  )
  _INTERACTION_CSR_KEYS = (
      'primary_track_ids_data', 'primary_track_ids_offsets',
      'primary_pdgs_data', 'primary_pdgs_offsets',
      'primary_energies_data', 'primary_energies_offsets',
  )
  # add the int members above to _INT_KEYS

  pi = evt['per_interaction'] if 'per_interaction' in evt else None
  if pi is not None:
      for k in _INTERACTION_SCALAR_KEYS + _INTERACTION_CSR_KEYS:
          if k in pi:
              data[f'labl_interaction_{k}'] = self._cast(pi[k][:], k)
  ```

  **Surface CSR raw** — do NOT expand/`np.repeat` the primaries here (the consumer/Part 04
  decides; matches how `per_particle` genealogy CSR is already surfaced raw,
  `lucid_labl.py:230-232`). **`target_vertex` is NOT assembled here** — that stack of
  `vertex_{x,y,z}` is a Part 04 decoration concern (impl-plan §3.4 / §3.5); the reader emits
  the three raw scalars. Update the module docstring (`lucid_labl.py:15-20`) from
  "not yet surfaced — deferred" to a description of the emitted `labl_interaction_*` keys.

  **Also surface `per_particle.interaction_idx`** (the Part 04 one-hop FK). The confirmed
  v5 schema (impl §3.4) adds a per-particle `interaction_idx` column giving
  `particle_idx → interaction_idx` directly — the one-hop source for `instance_interaction`
  with **no `per_track` detour** (Part 04 §3.3/§3.4). The current reader's `_PARTICLE_KEYS`
  (`lucid_labl.py:69-73`) omits it. Add `'interaction_idx'` to `_PARTICLE_KEYS` and to
  `_INT_KEYS` (`lucid_labl.py:80-83`); it then flows through the existing per_particle loop
  (`lucid_labl.py:230-232`) as `labl_particle_interaction_idx (P,) int32`, presence-gated by
  the loop's `if k in pp`. A v3 shard lacking it simply omits the key (Part 04 §5 case 2:
  omit, don't fabricate). This is the only `per_particle` add; it is what lets Part 04's
  LUCiD resolver do `source=("particle","interaction_idx")` gathered positionally by
  `particle_idx`, exactly as the locked facts require.

### 3.5 `JAXTPCSensorReader.read_meta`

- `source_event_idx`: `evt.attrs['source_event_idx']` → `None` (`save.py:344`; no `config`
  vector confirmed on JAXTPC — attr path only).
- `n_hits` = **Σ per-plane `n_pixels`** (D40). The writer stamps `g.attrs['n_pixels']` on
  **every** plane group (wire and pixel — `save.py:261`, `save.py:291`, `save.py:327`), so
  this is attr-only. Walk plane groups exactly the way `read_event` does **but read the attr
  instead of decoding**:

  ```python
  total = 0
  for _plane_label, pg in self._iter_planes(evt):   # reuse jaxtpc_sensor.py:157
      total += int(pg.attrs.get('n_pixels', 0))
  ```

  `_iter_planes` already handles both old (planes under event) and new (planes under
  `volume_N/`) layouts and yields only payload-bearing groups (`jaxtpc_sensor.py:157-175`),
  so it is the correct, no-array iterator. **Do NOT** use `evt.attrs['n_vol{v}']` for
  `n_hits` — that is the per-volume **deposit** count (`save.py:349`), not the sensor pixel
  count. (A future writer-side per-event total would make this O(1); §7.)
- `read_event`: **unchanged.**

### 3.6 `JAXTPCStepReader.read_meta`

- `source_event_idx`: `evt.attrs['source_event_idx']` → `None` (`save.py:391`).
- `n_hits` = **Σ per-volume `n_actual`** (D40): `sum(int(evt[f'volume_{v}'].attrs.get(
  'n_actual', 0)) for v in range(n_volumes) if f'volume_{v}' in evt)`. This is the exact
  Σ the `min_deposits>0` index branch already computes (`jaxtpc_step.py:91-97`); for the
  legacy single-volume flat layout fall back to the same `evt['positions'].shape[0]`
  proxy — but prefer the attr; `n_actual` is the deposit count, and for legacy flat events
  there is no per-event count attr so `positions.shape[0]` is the cheapest available
  (still no decode — `.shape` reads metadata, not data). `n_volumes` comes from the
  3-tuple `_locate_event` (§5).
- `read_event`: **unchanged** (`charge`/`photons`/`volume_id` already emitted; step carries
  no `group_id`/`sensor_hits` — those live in hits).

### 3.7 `JAXTPCHitsReader.read_meta`

- `source_event_idx`: `evt.attrs['source_event_idx']` → `None` (`save.py:625`).
- `n_hits` = **Σ per-volume `n_actual`** (the deposit count, written on each `volume_N`
  group, `save.py:635`):

  ```python
  total = 0
  for vol_key in evt:
      vol = evt[vol_key]
      if isinstance(vol, h5py.Group) and vol_key.startswith('volume_') \
              and 'n_actual' in vol.attrs:
          total += int(vol.attrs['n_actual'])
  ```

  Note: this counts **deposits** (the truth point count), consistent with step, not the
  CSR-expanded per-plane pixel entry count (which would require decoding). The base uses
  one consistent min-points axis per detector; deposit count is the cheap one.
- `read_event`: **unchanged** — `group_id` already surfaced (`jaxtpc_hits.py:275/281`).

### 3.8 `JAXTPCLablReader.read_meta`

- `source_event_idx`: `evt.attrs.get('source_event_idx')` → `None`. **Caveat:** the current
  stand-in `make_labl.py` writer does **not** stamp `source_event_idx` on the labl event
  group (it stamps only `event_id`, `make_labl.py:194-196`), even though it *reads*
  `source_event_idx` from the hits file (`make_labl.py:190`). So today this returns `None`
  for JAXTPC labl and the base falls back. The fix is a one-line writer add (§7); the reader
  side is forward-compatible (reads the attr if present). **The base should source JAXTPC
  identity from the sensor/step/hits reader, not labl** — labl identity is a nice-to-have.
- `n_hits`: **n/a** → `0`.
- `read_event`: **unchanged** — `track_interaction` already surfaced by the generic loop
  (`jaxtpc_labl.py:139-144`).

### 3.9 Docstring fix (all four LUCiD readers)

Change `format_version: 3` → `format_version: 5` in the module docstrings:
`lucid_sensor.py:3`, `lucid_step.py:3`, `lucid_hits.py:3`, `lucid_labl.py:3` (and the
`per_event`/`per_track` "v3 schema" comment at `lucid_labl.py:161`). Comment-only; no code
path depends on the literal. JAXTPC reader docstrings do not mention a version → no change.

---

## 4. Expected behavior (examples)

**LUCiD sensor `read_meta`** (one event, `n_hits` attr = 120, vector present):

```python
r = LUCiDSensorReader(root, dataset_name='wc')
r.read_meta(0)        # {'source_event_idx': 8412, 'n_hits': 120}
# source_event_idx came from config/source_event_idx[local_row]; no array decoded.
len(r.read_event(0)['sensor_idx'])   # == 120  (consistency, but read_event NOT called by read_meta)
```

**LUCiD hits `read_event` with `T_reco`:**

```python
d = LUCiDHitsReader(root).read_event(0)
sorted(d)   # ['particle_idx', 'pe', 'sensor_idx', 't', 't_reco']   (t_reco NEW)
d['t_reco'].shape == d['t'].shape    # row-aligned after pe_threshold mask
```

**LUCiD labl `read_event` with `per_interaction`** (I interactions, CSR primaries):

```python
d = LUCiDLablReader(root).read_event(0)
# scalars (I,)
d['labl_interaction_vertex_x'].shape        # (I,)
d['labl_interaction_neutrino_pdg'].dtype    # int32
# CSR — slice interaction i's primaries:
off = d['labl_interaction_primary_pdgs_offsets']    # (I+1,)
dat = d['labl_interaction_primary_pdgs_data']       # (Σ n_primaries,)
primaries_of_0 = dat[off[0]:off[1]]
```

**JAXTPC sensor `read_meta`** (Σ over plane `n_pixels`, here 2 vols × 3 wire planes):

```python
r = JAXTPCSensorReader(root, dataset_name='sim')
m = r.read_meta(0)                  # {'source_event_idx': 17, 'n_hits': 246}
# n_hits == sum of g.attrs['n_pixels'] over all decoded planes ==
#           sum(len(v) for k,v in r.read_event(0).items() if k.endswith('.value'))
```

**JAXTPC step `read_meta`** (Σ `n_actual`, 3-tuple locate):

```python
m = JAXTPCStepReader(root).read_meta(0)   # {'source_event_idx': 17, 'n_hits': 120}
# n_hits == sum(vg.attrs['n_actual']) == read_event(0)['coord'].shape[0]
```

---

## 5. Edge cases

1. **Missing `source_event_idx`.** Vector absent **and** attr absent → reader returns
   `None`. The **base** (Part 02 §3.3, D26) emits one warning and uses `(config_id,
   positional)` for holdout/identity. The reader must not fabricate a value and must not
   warn per-event (warning is the base's job, once).
2. **Absent `n_hits` source attr.** `evt.attrs.get('n_hits', 0)` / `pg.attrs.get(
   'n_pixels', 0)` / `vg.attrs.get('n_actual', 0)` — all default `0`, never raise. An event
   with no hit attr counts as 0 hits (correctly filtered out by a `>= min_points > 0`).
3. **Empty event.** JAXTPC sensor with zero payload planes → `_iter_planes` yields nothing
   → `n_hits = 0`. JAXTPC step/hits with all volumes `n_actual=0` → `0`. LUCiD hits/sensor
   with the attr present but `=0` → `0`. `read_event` on an empty JAXTPC step returns
   `_empty_dict()` (`jaxtpc_step.py:254`); `read_meta` never calls it, so it just returns
   `{'source_event_idx': ..., 'n_hits': 0}`.
4. **3-tuple `_locate_event` (JAXTPC step).** `jaxtpc_step._locate_event` returns
   `(f, event_key, n_volumes)` (`jaxtpc_step.py:131-139`), unlike every other reader's
   2-tuple. `JAXTPCStepReader.read_meta` **must unpack three values**:
   `f, event_key, n_volumes = self._locate_event(idx)` — and it *uses* `n_volumes` for the
   Σ`n_actual` loop. A copy-paste of the 2-tuple skeleton here is the most likely bug; call
   it out in the test (§6, test 9). The other three JAXTPC readers and all four LUCiD readers
   are 2-tuple.
5. **Per-file vector vs per-event attr disagreement.** They should agree (both written by
   the same writer). The vector is authoritative (O(1)) and preferred; the test (§6, test 7)
   asserts agreement on the fixtures so a future drift is caught.
6. **Index gaps.** JAXTPC `_build_index` indexes only event groups actually present (skips
   capacity-overflow gaps, `jaxtpc_sensor.py:81-84`); `read_meta(idx)` uses the same
   `_locate_event`, so `idx` is always a valid present event — no `KeyError`.
7. **`T_reco` absent on a shard.** Guard `if 'T_reco' in evt`; omit the key rather than
   duplicate `T`. Downstream must treat `t_reco` as optional.
8. **Worker init.** `read_meta` calls `h5py_worker_init()` if `not self._initted` (same
   lazy-open guard as `read_event`), so it is safe to call from a fresh fork or before the
   first `read_event`.

---

## 6. Tests (Step-0 matrix; on `testing.py` synthetic fixtures — no GPU, no WAND)

Numbered; each is `setup → action → EXPECTED`. **Fixture gaps to close first** (the current
`testing.py` writers omit several of the attrs/datasets these tests need — see the note at
the end of this section).

1. **`read_meta` `n_hits` == full read count (LUCiD sensor).**
   Setup: `make_lucid_sample(n_hits=120)` with the `n_hits` attr stamped (fixture add).
   Action: `r.read_meta(0)['n_hits']` vs `len(r.read_event(0)['sensor_idx'])`.
   EXPECTED: equal (120). And `read_meta` triggers **no array read** (assert by patching
   `evt.__getitem__` / spying that only `.attrs` is touched, or simply that it is O(1) and
   correct on a fixture where the attr deliberately disagrees with the array length is NOT
   done — the fixture writes a consistent attr).

2. **`read_meta` `n_hits` == Σ deposits (JAXTPC step).**
   Setup: `make_jaxtpc_sample(n_volumes=2, n_deposits=60)`.
   Action: `JAXTPCStepReader.read_meta(0)['n_hits']` vs `read_event(0)['coord'].shape[0]`.
   EXPECTED: equal (== Σ `n_actual` = 120).

3. **`read_meta` `n_hits` == Σ `n_pixels` (JAXTPC sensor) — the D40 invariant.**
   Setup: `make_jaxtpc_sample(readout_type='wire')` **and** `='pixel'`, with the fixture
   sensor writer stamping `pg.attrs['n_pixels']` (fixture add).
   Action: `JAXTPCSensorReader.read_meta(0)['n_hits']` vs
   `sum(len(v) for k,v in read_event(0).items() if k.endswith('.value'))`.
   EXPECTED: equal, for both readouts.

4. **`read_meta` `n_hits` == Σ deposits (JAXTPC hits).**
   Setup: as test 2 (hits writer stamps `volume_N.attrs['n_actual']` — fixture add).
   Action: `JAXTPCHitsReader.read_meta(0)['n_hits']`.
   EXPECTED: == Σ `n_actual` (== `n_volumes * n_deposits`).

5. **`per_interaction` surfaced with right fields + CSR (LUCiD labl).**
   Setup: `make_lucid_sample` with a `per_interaction` group added (fixture add: I≥2
   interactions, scalars + CSR primaries with a known ragged layout, e.g. interaction 0 has
   2 primaries, interaction 1 has 1).
   Action: `LUCiDLablReader.read_event(0)`.
   EXPECTED: all ten `labl_interaction_<scalar>` keys present with shape `(I,)` and correct
   dtype (`neutrino_pdg`/`source_type` int32, `vertex_*`/`t0`/`neutrino_energy_MeV` float32,
   `contained` bool); all six CSR keys present; `offsets` shape `(I+1,)`,
   `offsets[-1] == data.size`; `data[offsets[0]:offsets[1]]` equals the known primaries of
   interaction 0. Assert **no `target_vertex`** key (that is Part 04).

5b. **`per_particle.interaction_idx` one-hop FK surfaced (LUCiD labl).**
   Setup: `make_lucid_sample` with a `per_particle/interaction_idx` column (fixture add).
   Action: `LUCiDLablReader.read_event(0)`.
   EXPECTED: `labl_particle_interaction_idx` present, shape `(P,)`, int32, each value a
   valid interaction index; absent on a v3 fixture variant (presence-gated, no fabrication).
   This is the column Part 04's LUCiD resolver gathers positionally by `particle_idx` for
   `instance_interaction` (no `per_track` detour).

6. **`T_reco` present + row-aligned + threshold-masked (LUCiD hits).**
   Setup: fixture hits writer adds `T_reco` dataset (same length as `T`).
   Action: `LUCiDHitsReader(root, pe_threshold=p).read_event(0)`.
   EXPECTED: `'t_reco'` present; `t_reco.shape == t.shape == sensor_idx.shape` after the
   `pe_threshold` mask; with a shard where `T_reco` is absent, `'t_reco'` key is **omitted**
   (run the absent-shard sub-case by writing one fixture variant without it).

7. **`source_event_idx` vector vs per-event attr agreement.**
   Setup: fixture LUCiD sensor + labl write BOTH `config/source_event_idx` (vector) and
   `event_NNN.attrs['source_event_idx']` (consistent values, e.g. a shuffled permutation so
   it differs from the positional index — catches off-by-one).
   Action: for each event, `read_meta(idx)['source_event_idx']` (vector path) vs the
   per-event attr read directly.
   EXPECTED: equal for every event; equal across the two modalities (sensor vs labl) for the
   same `idx`; and (importantly) **not** equal to the positional `idx` (proving identity, not
   position, is returned).

8. **`source_event_idx` fallback to `None`.**
   Setup: a fixture variant with neither vector nor attr.
   Action: `read_meta(0)['source_event_idx']`.
   EXPECTED: `None` (reader does not raise, does not warn — base's job).

9. **3-tuple `_locate_event` does not break `read_meta` (JAXTPC step).**
   Setup: `make_jaxtpc_sample`.
   Action: `JAXTPCStepReader.read_meta(0)` returns a 2-key dict (regression guard against a
   2-tuple unpack copy-paste).
   EXPECTED: `set(m) == {'source_event_idx', 'n_hits'}`; `n_hits == 120`. (A `ValueError:
   too many values to unpack` here is the bug this test catches.)

10. **`read_meta` is attr-only (cost guard, all readers).**
    Setup: any fixture.
    Action: wrap the open `h5py.File`'s event group so dataset `__getitem__` (slicing)
    raises, but `.attrs` and `.shape`/group iteration are allowed; call `read_meta`.
    EXPECTED: succeeds for all 8 readers (proves no array decode). (For JAXTPC step legacy
    flat layout, `positions.shape` is metadata-only and allowed; assert it does not slice.)

11. **`format_version` docstring fix (lint/grep guard, optional).**
    Action: grep the four LUCiD reader modules for `format_version: 3`.
    EXPECTED: zero matches (all say `: 5`).

**Fixture additions required in `testing.py`** (call out for the test author — current
writers omit these; see `_write_lucid_sensor` `testing.py:491`, `_write_lucid_hits`
`testing.py:506`, `_write_lucid_labl` `testing.py:522`, `_write_jaxtpc_sensor`
`testing.py:250`, `_write_jaxtpc_hits` `testing.py:278`, and the JAXTPC writers that omit
`source_event_idx`):
- LUCiD sensor: `event.attrs['n_hits']`; `config/source_event_idx` vector +
  `event.attrs['source_event_idx']` (a non-positional permutation).
- LUCiD hits: `event.attrs['n_particle_hits']`; a `T_reco` dataset (and a no-`T_reco`
  variant for the absent sub-case).
- LUCiD labl: a `per_interaction` group (scalars + CSR primaries, known ragged layout);
  a `per_particle/interaction_idx` column (each particle → a valid interaction index, for
  the Part 04 one-hop FK); `config/source_event_idx` vector + per-event attr. (Writer:
  `_write_lucid_labl` `testing.py:537-544` for per_particle; new `per_interaction` group.)
- JAXTPC sensor: `pg.attrs['n_pixels']` on each plane group; `event.attrs['source_event_idx']`.
- JAXTPC hits: `vol_grp.attrs['n_actual']`; `event.attrs['source_event_idx']`.
- JAXTPC step: `event.attrs['source_event_idx']` (`n_actual` already written,
  `testing.py:239`).
- Flip the fixture `format_version` to 5 (`testing.py:469/496/511/526` write `= 3`) so the
  fixtures match the confirmed schema (and test 11 is meaningful).

---

## 7. Reversible defaults & risks

- **Default `n_hits` axis per detector (reversible):** LUCiD sensor → `n_hits`; LUCiD step →
  `n_segments`; LUCiD hits → `n_particle_hits`; JAXTPC sensor → Σ`n_pixels`; JAXTPC
  step/hits → Σ`n_actual` (deposit count). These are the cheap attr-only proxies; the base's
  `min_points` `modality=` picks which one. Changing the JAXTPC hits axis from deposit-count
  to CSR-pixel-count would require decoding → rejected (not attr-only).
- **Writer-side per-event `n_hits` vector ask (D27/D40 — reversible, deferred):** JAXTPC
  sensor `n_hits` is currently a small per-plane attr **walk** (Σ`n_pixels` over plane
  groups). For O(1)/event the JAXTPC writer (`save_event_sensor`, `save.py:330`) could stamp
  a per-event total `evt.attrs['n_hits'] = Σ n_pixels`, and either writer could add a per-file
  `config/n_hits` `uint32 (n_events,)` vector (LUCiD already has the per-event `n_hits` attr
  but **no** `config/n_hits` vector — confirmed). The attr-walk is correct and cheap enough
  for the manifest-cache build (one-time, rank-0); the vector is a pure speedup. **Default:
  do not block on the writer** — ship the attr-walk; file the vector as a follow-up ask.
- **JAXTPC labl `source_event_idx` (risk, one-line writer fix):** `make_labl.py:194` stamps
  only `event_id`, not `source_event_idx`, although it reads `source_event_idx` from hits at
  `make_labl.py:190`. Reader returns `None` for labl identity today. **Mitigation:** source
  JAXTPC identity from sensor/step/hits (which DO stamp it); add
  `labl_evt.attrs['source_event_idx'] = source_event_idx` in `make_labl.py` when the proper
  edepsim-side labl writer lands (it is a stand-in, `make_labl.py:2-4`). Forward-compatible:
  the reader reads the attr if present.
- **CSR raw vs expanded (locked, not reversible):** `per_interaction` primaries are surfaced
  as raw `data`/`offsets` (not `np.repeat`-expanded). Expanding here would (a) bake a join the
  consumer may not want and (b) diverge from how genealogy CSR is already surfaced raw. Part 04
  decides any expansion.
- **`source_event_idx` precedence (reversible default):** vector → attr → `None`. If a future
  file has only the attr (no vector), the attr path covers it; if only the vector, that path
  covers it. The order is a speed preference, not a correctness one (they must agree; test 7).
- **Risk — silent 2-tuple unpack on JAXTPC step:** see §5.4 / §6 test 9; the most likely
  copy-paste bug. Mitigated by an explicit test.

---

## 8. Dependencies on other parts

- **Part 02 — Dataset base** is the sole consumer of `read_meta` + the identity-vector
  exposure: the manifest-cache build (Part 02 §3.6) walks `read_meta(idx)` (or the
  `config/source_event_idx[:]` vector fast path when `reader._has_sei_vec`, §3.0) over
  every event once (rank-0, under DDP barrier) to get `(source_event_idx, n_hits)` for
  min-points + hash-holdout; steady state reads the cache. Frozen surfaces the base reads:
  `read_meta(idx)`, `source_event_idx(idx)`, `reader._has_sei_vec`, `reader._sei_vecs`
  (§3.0), plus the unchanged `cumulative_lengths`/`indices`/`h5_files`/`read_event`/
  `__len__`/`close`. `event_identity(idx)` (base) returns `(config_id, source_event_idx)`
  — `config_id` from the source spec, `source_event_idx` from this part. The base owns the
  **D26 fallback warning** and the `(config_id, positional)` substitution when
  `source_event_idx is None`. The base must source JAXTPC identity from sensor/step/hits,
  not labl (§3.8).
  The base **also** consumes the present-key surface for its **joint event index**
  (Part 02 §3.3a, D42): it intersects the per-modality present `event_*` keys
  (`indices`/`cumulative_lengths` + `source_event_idx`-per-key) so one `local_idx`
  no longer addresses different physics events across modalities. This part provides
  the inputs (present keys + `source_event_idx`, deduped via the `_read_shard_meta`
  lru_cache, §3.0/D46); the intersection logic is the base's. **D44 interaction:**
  the base intercepts `min_deposits`/`min_segments` and no-ops the step/lucid-step
  internal index mask (`jaxtpc_step.py:84-100`, `lucid_step.py:85-94`) so a
  non-contiguous per-reader index can never desync from the other modalities — i.e.
  with the base in place that internal mask path is dead; this part leaves the reader
  code as-is (the no-op is the base passing `min_deposits=0` through, or Phase A
  removing it), but readers must not *rely* on it for correctness.
- **Part 04 — Label decoration** consumes the **newly surfaced** raw fields:
  `per_interaction` (→ `target_vertex` = stack `vertex_{x,y,z}`, `target_energy` ←
  `neutrino_energy_MeV`, `target_contained` ← `contained`); the new LUCiD
  `per_particle.interaction_idx` (→ one-hop `instance_interaction`, gathered positionally
  by `particle_idx`, §3.4); and the already-surfaced JAXTPC `group_id` / labl
  `track_interaction`. This part surfaces raw only; Part 04 maps FK → named schema keys.
  The `labl_interaction_*` / `labl_particle_interaction_idx` naming here is the input
  contract Part 04 reads. Until this part lands those surfacings, Part 04 omits the
  corresponding axes (Part 04 §5 case 2 / §7) — so this part **unblocks** them.
- **Part 01 — Transforms** is independent of `read_meta`; it touches the per-point columns
  `read_event` emits (e.g. `t`/`t_reco` feed `RelativeLogNormalize`), so the new `t_reco`
  key becomes a candidate normalize input but requires no transform change in this part.
- **Fixtures (`testing.py`)** are shared with Parts 01/02/04; the §6 fixture additions
  (`n_hits`/`n_pixels`/`n_actual` attrs, `source_event_idx` vector+attr, `per_interaction`
  group, `T_reco` dataset, `format_version=5`) are net-new to `testing.py` and must land with
  this part's tests.
- **Writer (JAXTPC `save.py` / `make_labl.py`)** — no required change for this part; the
  optional per-event/per-file `n_hits` vector and the `make_labl.py` `source_event_idx` stamp
  are §7 follow-up asks, not blockers.
