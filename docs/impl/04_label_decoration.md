# Part 04 — Label decoration (implementation spec)

**Status:** implementation-ready spec. Single-stream-per-task structure (D35); JAXTPC
in scope alongside LUCiD (D40).

> **⚠ Cross-reference note — this doc's in-text "Part NN" labels predate the final
> filenames (and differ from `02`'s scheme).** Match by **title**, not number.
> Canonical map for *this* doc: "Part 02" (`index_operator` prefix-match) →
> `01_transforms.md`; "Part 03" (readers) → `03_readers.md` ✓; "Part 05" (base) →
> `02_dataset_base.md`; "Part 06" (collate) → `05_collate_streams_eval.md`.
> `00_index.md` §2/§5 is authoritative.

**Source decisions:** D20 (labeling in reader/dataset from `labl`, not a transform),
D22 (generic `segment_*`/`instance_*`/`target_*` schema), D28 (reader emits raw FKs →
dataset decorates via `label_config`; per-event targets are per-event, not per-point;
names reconciled to real configs), D38 (open/extensible decoration framework — axes are
registered `label_config` entries, not hardcoded). Implementation-plan anchor: §3.5.

**Files (read-only; grounding):**
- `src/pimm_data/lucid.py` — `_build_hits` (`lucid.py:235`), `_build_step` (`lucid.py:272`),
  `_lookup_per_particle` (`lucid.py:326`), `_lookup_per_track` (`lucid.py:338`).
- `src/pimm_data/jaxtpc.py` — `_decorate_hits_from_labl` (`jaxtpc.py:457`),
  `_decorate_step_from_labl` (`jaxtpc.py:416`).
- `src/pimm_data/readers/lucid_labl.py` — labl scopes + derived `ancestor_particle_idx`.
- `src/pimm_data/readers/jaxtpc_labl.py` — per-volume `track_ids`/`track_*`/`deposit_to_track`.
- `src/pimm_data/detector_transforms.py` — `PDGToSemantic` (`detector_transforms.py:66`),
  `RemapSegment` (`detector_transforms.py:132`) — *downstream* consumers, not part of this part.
- `src/pimm_data/transform.py` — `index_operator` default `index_valid_keys`
  (`transform.py:41-71`) — carries the named keys (Part 02 / §3.2).
- `src/pimm_data/testing.py` — synthetic FK invariants used by §6 tests.

**Locked constraints (do not relitigate):**
1. Labeling is done in the **reader/dataset from `labl`**, NOT a transform. Readers emit
   raw FKs (`particle_idx`/`track_idx`/`group_id` + the FK chain tables); the dataset
   decorates (D20/D28).
2. The decorator emits **named schema keys** — `segment_pid`, `instance_particle`,
   `instance_interaction`, `instance_ancestor`, `segment_interaction`, `target_vertex`,
   `target_energy`, `target_contained` — matching the real configs (panda/panseg consume
   `segment_pid`/`instance_particle`; see `transform.py:52-53`).
3. `label_config` is an **OPEN list** of axis specs (extensible; D38).
4. Per-event targets (`target_vertex`/`target_energy`/`target_contained`) are **per-event,
   not per-point**. `event_label`/`config_id` are **per-point broadcast**
   (`scope="event_broadcast"`) so the probe slices by offset.
5. Confirmed labl facts: `per_particle.category` (uint8) is the semantic source;
   `per_particle.interaction_idx` gives `instance_interaction` **one-hop** from
   `particle_idx` (no per_track detour); `target_vertex` = stack `vertex_x/y/z` (three
   scalars). LUCiD FK = `particle_idx→per_particle` / `track_idx→per_track`; JAXTPC FK =
   `group_id→group_to_track→track→labl` (hits) / `deposit_to_track→track→labl` (step).

---

## 1. Purpose & scope

This part specifies the **label-decoration framework** that lives in the dataset layer:
a declarative `label_config` (a list of axis specs) plus a **single generic decorator**
`_decorate_from_labl(sub, labl, fk_resolver)` that generalizes the four hand-written
methods that exist today (`lucid._build_hits`/`_build_step` via `_lookup_per_particle`/
`_lookup_per_track`; `jaxtpc._decorate_hits_from_labl`/`_decorate_step_from_labl`).

In scope:
- The `label_config` schema (fields `out`, `scope`, `fk`, `source`, `fill`).
- The generic `_decorate_from_labl` algorithm (gather-with-fill, plus searchsorted on a
  non-positional FK for JAXTPC).
- The per-detector `fk_resolver` — the **only** subclass-specific piece.
- The default `label_config` per detector (labl-field → named-key map) for LUCiD and JAXTPC.
- How per-event `target_*` attach (carried per-event, not broadcast) vs `event_broadcast`
  (materialized per-point).
- Extensibility (a new axis = a new spec + a prefix entry in `index_operator`, Part 02).
- Back-compat for the single-axis `label_key` knob on JAXTPC.

Out of scope (other parts): readers' raw-FK surfacing and `per_interaction` exposure
(Part 03 / impl §3.4); `index_operator` prefix-match (Part 02 / impl §3.2); the
downstream `RemapSegment`/`PDGToSemantic` transforms (these consume the decorated keys,
they do not produce them — `detector_transforms.py:66,132`); collate (Part 06 / impl §3.6).

---

## 2. Current state (the two hardcoded single-axis decorators, plus LUCiD's two)

There are **four** hand-written decoration paths today; all are single-axis (`segment`
and `instance` only) and detector-specific. The framework collapses them to one.

### 2.1 LUCiD — gather-by-positional-index (the FK *is* the row)

`_build_hits` (`lucid.py:235-270`) — for the `hits` stream, `particle_idx` is a direct
positional index into `per_particle`:
- `lucid.py:263` sets `'instance' = particle_idx.astype(np.int32)` (the FK itself is the
  instance label — `source="self"`).
- `lucid.py:268-269` sets `'segment'` via `_lookup_per_particle(particle_idx, category)`,
  where `category = labl['particle']['category']` (`lucid.py:266`).

`_build_step` (`lucid.py:272-289`) — for the `step` stream there is **one extra hop**:
deposits carry `track_idx` (positional into `per_track`), and `per_track.particle_idx`
maps to `per_particle`:
- `lucid.py:281-283` `particle_idx = _lookup_per_track(track_idx, track_particle_idx)`.
- `lucid.py:284` `'instance' = particle_idx`.
- `lucid.py:287-288` `'segment' = _lookup_per_particle(particle_idx, category)`.

`_lookup_per_particle` (`lucid.py:326-336`) and `_lookup_per_track` (`lucid.py:338-346`)
are byte-identical except the argument name — both are **gather-with-fill** on a positional
index with a bounds mask:
```python
n = per_particle_col.shape[0]
valid = (particle_idx >= 0) & (particle_idx < n)
out = np.full(particle_idx.shape, fill, dtype=per_particle_col.dtype)
if valid.any():
    out[valid] = per_particle_col[particle_idx[valid]]
```
(`fill=-1` default, `lucid.py:328`/`341`.) These two methods differ only in which FK column
they gather and are the **gather-with-fill primitive** the generic decorator reuses.

### 2.2 JAXTPC — searchsorted on a non-positional FK (`track_ids` is a value table)

JAXTPC's labl is keyed by **track_id value**, not row position, so a binary search on a
sorted `track_ids` table is required.

`_decorate_step_from_labl` (`jaxtpc.py:416-455`):
- Per-volume loop over `labl_by_volume` (`jaxtpc.py:429`); deposits selected by
  `volume_id == vol_num` (`jaxtpc.py:431`).
- FK: `per_dep_tid = vdata['deposit_to_track']` (`jaxtpc.py:436`) — per-deposit track_id,
  row-aligned to the volume's step rows; length-checked against the masked count
  (`jaxtpc.py:438-441`).
- `instance[mask] = per_dep_tid` (`jaxtpc.py:442`) — instance is the raw track_id.
- `segment`: searchsorted gather (`jaxtpc.py:444-453`): sort `track_ids`, `searchsorted`
  the FK, clip, **verify the match** (`matched = s_tids[pos] == per_dep_tid`), and
  `np.where(matched, s_vals[pos], -1)`. `meta_col = f'track_{self._label_key}'`
  (`jaxtpc.py:427`).

`_decorate_hits_from_labl` (`jaxtpc.py:457-497`) — same searchsorted gather, with **one
extra hop** to turn a `group_id` into a track_id first:
- Per-plane loop (`jaxtpc.py:461`); `gid = cols['group_id']` (`jaxtpc.py:463`); volume
  index parsed from the plane label `volume_{v}_{U|V|Y}` (`jaxtpc.py:465-466`).
- `g2t = hits_flat[f'group_to_track_v{v}']` (`jaxtpc.py:472-473`); `tids =
  np.where(valid, g2t[gid], -1)` with `valid = (gid>=0) & (gid<len(g2t))`
  (`jaxtpc.py:482-483`) — the **`group_id → group_to_track → track_id`** hop.
- Then the identical searchsorted gather on `track_ids`/`track_{label_key}`
  (`jaxtpc.py:485-492`). `instance` is set separately in `_build_hits_cloud`
  (`jaxtpc.py:313` `instance == group_id`), not in this method.

### 2.3 What's wrong with the current state (why the framework)

- **Single-axis only.** Each path hardcodes exactly `segment` + `instance`. Adding
  `instance_interaction` or `target_vertex` means editing the decorator, not a config.
- **Named-key gap.** The configs/evaluators consume `segment_pid`/`instance_particle`
  (`transform.py:52-53`), but the dataset emits bare `segment`/`instance`. Today this is
  bridged downstream (`PDGToSemantic` writes `segment_pid`/`instance_particle`,
  `detector_transforms.py:99,112`), which only works on the PDG-fallback path, not the
  labl path.
- **No per-event targets.** Vertex/energy/contained (per-interaction) have no path at all.
- **Duplicated primitives.** `_lookup_per_particle`/`_lookup_per_track` are one function;
  the two JAXTPC searchsorted gathers are one function. Four near-duplicates.

---

## 3. Target design

### 3.1 `label_config` schema

`label_config` is an **open list of axis specs** (dicts). Each spec declares one output
key. The decorator iterates the list; an unknown `out` name is just another row (no
enum). New axis families (edge/graph for NuGraph, hierarchy) are added by appending specs
(§3.6) — no decorator change.

| field   | type | meaning |
|---|---|---|
| `out`   | str  | The emitted key name. Named-schema keys: `segment_pid`, `segment_interaction`, `instance_particle`, `instance_interaction`, `instance_ancestor`, `target_vertex`, `target_energy`, `target_contained`, `event_label`, `config_id`. |
| `scope` | str  | `"point"` (per-point, length N, subset by N-changing transforms), `"event"` (per-event target — length-1 / `(D,)`, NOT subset), or `"event_broadcast"` (per-event value materialized to a length-N per-point column). |
| `fk`    | str  | Name of the per-point foreign key in the stream sub-dict (e.g. `"particle_idx"`, `"track_idx"`, `"group_id"`, `"deposit_to_track"`). `point`/`event_broadcast` axes that gather from a labl table need it; `scope="event"` and `source="self"` don't. |
| `source`| str \| tuple | Where the value comes from: `"self"` (the FK value itself is the label — LUCiD `instance`), `(table, column)` (gather `labl[table][column]` through `fk`; `table ∈ {"particle","track","interaction"}`, mapped to the detector's labl layout by the resolver), or a literal/host value for `event_broadcast` (mixture label / `config_id`). |
| `fill`  | int  | Sentinel for unresolved FKs and out-of-range gathers. Default `-1` (matches `lucid.py:328`, `jaxtpc.py:425/453`). |

Default per-detector `label_config` lives **in the subclass** (built in `__init__` from
`modalities` + the detector's labl layout) so a config need not spell it out; a config may
override it wholesale via the `label_config=` constructor arg (impl §3.3 constructor).

Reference shape (impl §3.5):
```python
dict(out="segment_pid",          scope="point", fk="particle_idx", source=("particle","category"), fill=-1)
dict(out="instance_particle",    scope="point", fk="particle_idx", source="self")
dict(out="instance_interaction", scope="point", fk="particle_idx", source=("track","interaction"))
dict(out="target_vertex",        scope="event",                    source=("interaction","vertex"))
dict(out="target_energy",        scope="event",                    source=("interaction","neutrino_energy_MeV"))
dict(out="event_label",          scope="event_broadcast",          source="<mixture label>")
```

### 3.2 `_decorate_from_labl(sub, labl, fk_resolver)` — the generic algorithm

One method on the base, replacing all four current paths. `sub` is the stream sub-dict
being decorated (`step`/`hits`); `labl` is the event's nested labl dict; `fk_resolver` is
the detector callback (§3.3). The method iterates `self._label_config_for(stream)`:

```
for spec in label_config_for(stream):
    out, scope, fill = spec["out"], spec["scope"], spec.get("fill", -1)

    if scope == "event":
        # per-event target: read once, attach as length-1 / (D,) — NOT a per-point column.
        val = fk_resolver.event_value(labl, spec["source"])   # e.g. stack vertex_{x,y,z} -> (3,)
        if val is not None:
            event_targets[out] = val
        continue

    if scope == "event_broadcast":
        val = fk_resolver.event_value(labl, spec["source"])   # scalar (mixture label / config_id)
        sub[out] = np.full((N, 1), val, dtype=...)            # per-point, lifted by Collect, sliced by offset
        continue

    # scope == "point"
    fk = fk_resolver.gather_fk(sub, labl, spec["fk"])         # (N,) int FK, -1 where unresolved
    if fk is None:
        continue                                              # FK absent in this event -> axis omitted
    if spec["source"] == "self":
        sub[out] = fk.astype(np.int32)                        # the FK itself is the label
        continue
    table, col = spec["source"]
    column, keyed_by = fk_resolver.resolve_column(labl, table, col)   # (values, key_table | None)
    if column is None:
        continue
    sub[out] = _gather_with_fill(fk, column, keyed_by, fill)
```

Two gather primitives (generalize the existing code):

**`_gather_with_fill(fk, column, keyed_by, fill)`** —
- `keyed_by is None` → **positional gather** (LUCiD): exactly `_lookup_per_particle`
  (`lucid.py:329-336`) — bounds-mask `(fk>=0)&(fk<len(column))`, `np.full(..., fill)`,
  scatter the valid rows.
- `keyed_by is not None` → **searchsorted gather** (JAXTPC): exactly the
  `jaxtpc.py:447-453` block — `order=argsort(keyed_by)`, `searchsorted`, `clip`, verify
  `matched = s_keys[pos]==fk`, `np.where(matched, s_vals[pos], fill)`.

**Per-event targets do not enter `sub`** — they accumulate in an `event_targets` dict the
caller attaches as `_`-prefixed list-collated metadata (impl §3.5: "carried as
`_`-prefixed list-collated metadata, excluded from `index_valid_keys`"). Concretely the
`get_data` builder does e.g. `data['_targets'] = event_targets` (a per-event dict),
distinct from the per-point columns inside `sub`. They are never indexed by N-changing
transforms (the leading dim ≠ `n_points`, the Part 02 exclusion rule).

`fill` semantics match today's: an FK that is `< 0`, out of range, or whose value isn't in
`keyed_by` resolves to `fill` (default `-1`, i.e. `ignore_index`). This is exactly the
behavior of `np.where(matched, ..., -1)` (`jaxtpc.py:453,492`) and the bounds mask
(`lucid.py:331`).

### 3.3 `fk_resolver` per detector (the ONLY subclass-specific piece)

The resolver hides the two structural differences: (a) how a stream's per-point FK is
produced, (b) whether a labl table is positional or value-keyed, and (c) the labl-layout
naming. Three methods:

- **`gather_fk(sub, labl, fk_name) -> (N,) int | None`** — produce the per-point FK column
  named `fk_name`, applying any cross-table hop, or `None` if the chain is unavailable.
- **`resolve_column(labl, table, col) -> (values, keyed_by | None)`** — return the labl
  column and, for value-keyed tables, the parallel key array to searchsorted against
  (`None` ⇒ positional gather).
- **`event_value(labl, source) -> scalar | (D,) | None`** — per-event/per-interaction
  read.

**LUCiD resolver** (positional tables; FK is a row index):
- `gather_fk(sub, labl, "particle_idx")` → for `hits`, `sub['particle_idx']` directly
  (`lucid.py:242,262`); for `step`, the **one extra hop**
  `_lookup_per_track(sub['track_idx'], labl['track']['particle_idx'])` (`lucid.py:281-283`)
  so both streams expose a uniform `particle_idx` FK.
- `resolve_column(labl, "particle", "category")` → `(labl['particle']['category'], None)`
  — **positional**, so `keyed_by=None` (`lucid.py:266`, `category` is uint8 per D28/§5
  facts).
- `resolve_column(labl, "track", "interaction")` → `(labl['track']['interaction'], None)`.
  But note: for the **one-hop `instance_interaction`** that the locked facts mandate, the
  LUCiD default uses `per_particle.interaction_idx` once the reader surfaces it
  (Part 03), gathered positionally by `particle_idx` — i.e. `source=("particle",
  "interaction_idx")`, no per_track detour. (`instance_ancestor` uses the reader-derived
  `per_particle.ancestor_particle_idx`, `lucid_labl.py:46-47,196-209`, also positional by
  `particle_idx`.)
- `event_value(labl, ("interaction","vertex"))` → `np.stack([per_interaction.vertex_x,
  _y, _z])` → `(3,)` (the three scalars per D28/§5; needs the `per_interaction` scope
  surfaced — Part 03 / impl §3.4, currently **not** read, `lucid_labl.py:15-20`).

**JAXTPC resolver** (value-keyed `track_ids` table; FK chains through bridges):
- `gather_fk(sub, labl, fk_name)`:
  - `step`: `deposit_to_track` per volume, masked by `volume_id` and length-checked
    (`jaxtpc.py:431-442`), concatenated to a single `(N,)` FK column. (Multi-volume:
    iterate `labl_by_volume`.)
  - `hits`: the **`group_id → group_to_track → track_id`** hop per plane
    (`jaxtpc.py:482-483`) using `bridges[f'group_to_track_v{v}']`, concatenated across
    planes in the same plane order as `_merge_plane_dotted` (`jaxtpc.py:313,461`).
- `resolve_column(labl, "track", col)` → `(labl[vN][f'track_{col}'], labl[vN]['track_ids'])`
  — **value-keyed**, so `keyed_by = track_ids` (the searchsorted path, `jaxtpc.py:444-453`).
  The labl-layout naming (`track_pdg`/`track_interaction`/`track_cluster`/`track_ancestor`,
  `testing.py:313-321`, `jaxtpc_labl.py:13-17`) is encoded here, per-volume.
- `event_value` → JAXTPC's per-event targets come from `labl` `track_interaction` /
  future per-interaction tables; vertex/energy targets are a future surfacing (the JAXTPC
  labl reader surfaces `track_interaction` today — `jaxtpc.py:78-79`, `testing.py:320` —
  but not a per-interaction vertex table). Flag, don't invent (§7).

Crucially `_decorate_from_labl` itself contains **zero** detector branches — `instance ==
group_id` vs `instance == particle_idx`, positional vs searchsorted, one-hop vs two-hop
all live in the resolver + the spec's `source`/`fk`.

### 3.4 Default `label_config` map (labl-field → named-key)

Reconciled to the real configs and to the Part X label matrix in the engagement doc
(`engagement_plan_…:234-244`).

**LUCiD default** (`hits` and `step`; per-particle semantics):

| `out` | `scope` | `fk` | `source` | labl provenance |
|---|---|---|---|---|
| `segment_pid` | point | `particle_idx` | `("particle","category")` | `per_particle.category` (uint8) `lucid.py:266` |
| `instance_particle` | point | `particle_idx` | `"self"` | the FK itself `lucid.py:263,284` |
| `instance_interaction` | point | `particle_idx` | `("particle","interaction_idx")` | one-hop `per_particle.interaction_idx` (D28; reader surfacing Part 03) |
| `instance_ancestor` | point | `particle_idx` | `("particle","ancestor_particle_idx")` | reader-derived `lucid_labl.py:46-47` |
| `target_vertex` | event | — | `("interaction","vertex")` | stack `vertex_{x,y,z}` → `(3,)` (`per_interaction`, Part 03) |
| `target_energy` | event | — | `("interaction","neutrino_energy_MeV")` | `per_interaction.neutrino_energy_MeV` |
| `target_contained` | event | — | `("interaction","contained")` | `per_interaction.contained` (or `per_event.contained` `lucid.py:78`) |
| `event_label` | event_broadcast | — | mixture label | base materializes (per-point) |
| `config_id` | event_broadcast | — | source `config_id` | base materializes (per-point) |

For the `step` stream the same specs apply; the LUCiD resolver's `gather_fk` performs the
extra `track_idx → per_track.particle_idx` hop (`lucid.py:281-283`) so the `fk` column is a
uniform `particle_idx` for both streams.

**JAXTPC default** (`hits` and `step`; per-track semantics):

| `out` | `scope` | `fk` | `source` | labl provenance |
|---|---|---|---|---|
| `segment_pid` | point | (step) `deposit_to_track` / (hits) `group_id` | `("track","pdg")` | `track_pdg` value-keyed by `track_ids` `jaxtpc.py:427,447-453` |
| `instance_particle` | point | (step) `deposit_to_track` / (hits) `group_id` | `"self"` | step: raw track_id `jaxtpc.py:442`; hits: `instance == group_id` `jaxtpc.py:313` |
| `instance_interaction` | point | (step) `deposit_to_track` / (hits) `group_id` | `("track","interaction")` | `track_interaction` value-keyed `testing.py:320`, `jaxtpc.py:78` |
| `segment_interaction` | point | same | `("track","interaction")` | same column, semantic-axis use (D28: interaction is BOTH) |
| `event_label` | event_broadcast | — | mixture label | base materializes |
| `config_id` | event_broadcast | — | source `config_id` | base materializes |

Note JAXTPC `instance_particle` differs by stream: step uses the raw track_id (the FK),
hits uses `group_id` (the FK is `group_id`, `source="self"`). Both reduce to "the FK is the
label," so the same `source="self"` spec works; the resolver supplies the right FK column.

**Back-compat `label_key` (JAXTPC).** The existing `label_key='pdg'` constructor knob
(`jaxtpc.py:78-81,119`) maps to a one-spec config:
`[dict(out="segment", scope="point", fk=<stream FK>, source=("track", label_key), fill=-1)]`
plus the always-on `instance` spec (`source="self"`). This reproduces today's `segment`
(raw `track_{label_key}`, then `RemapSegment` downstream — `detector_transforms.py:139`)
and `instance` (`jaxtpc.py:313,442`) byte-for-byte. Bare `segment`/`instance` therefore
survive as single-axis aliases; configs that already pass `label_key=` keep working with no
edit (impl §3.5: "Bare `segment`/`instance` remain a back-compat single-axis alias").

### 3.5 Per-event vs per-point handling (precise)

- **`scope="point"`** → written into the stream `sub` dict as an `(N,1)` (or `(N,)`)
  column. Picked up by `index_operator`'s `index_valid_keys` (Part 02 / `transform.py:41-71`
  already lists `segment_pid`/`instance_particle`/`instance_interaction`/
  `segment_interaction`) so N-changing transforms keep it aligned. New axes need their `out`
  name covered by the prefix-match (Part 02 / §3.6).
- **`scope="event"`** (targets) → **not** in `sub`; attached to the event dict as a
  `_`-prefixed metadata entry (e.g. `data['_targets']`), length-1 / `(D,)`. Excluded from
  `index_valid_keys` by the leading-dim≠N rule (Part 02 / impl §3.2:
  "Exclude per-event `target_*` via a leading-dim ≠ `n_points` shape check"). List-collated
  (D24). The regression/classifier head reads `_targets`, not a per-point column.
- **`scope="event_broadcast"`** (`event_label`/`config_id`) → materialized in the
  **primary stream** as an `(N,1)` per-point array (impl §3.5: "per-point arrays inside the
  stream so `Collect(keys=[...,'event_label'])` lifts them and the probe slices by
  offset"). The base owns this materialization (impl §3.3: base owns
  `event_label`+`config_id` materialization); the linear-probe evaluator slices per-event by
  the batch `offset` vector.

This is the one true difference between the two target families: `target_*` stay one row
per event; `event_label`/`config_id` are broadcast to one row per point so the existing
single-stream collate carries them with no schema change (impl §3.6).

---

## 4. Expected behavior (concrete)

Using the synthetic fixtures (`testing.py`) and their guaranteed invariants
(`testing.py:17-26`).

### 4.1 JAXTPC `hits` event (wire readout, default `label_config`)

Given one event's plane CSR (`testing.py:150-226`) with per-plane `group_id` (= `gid`),
`group_to_track` and `track_pdg`/`track_interaction`/`track_ids` per volume:

- `instance_particle` = the per-entry `group_id`, concatenated across planes in sorted
  plane order (`jaxtpc.py:313,378`). Value: exactly the CSR-decoded `group_ids` repeated by
  `group_sizes` (`testing.py:170,167`).
- `segment_pid[k]` = `track_pdg[searchsorted(sort(track_ids), group_to_track[gid_k])]` when
  the resolved track_id is present in `track_ids`, else `-1`. Because every group's
  `group_to_track` is drawn from `track_ids` (`testing.py:108-109`) and every PDG comes
  from `_JAXTPC_PDG_POOL = {13,11,211,22,2212}` (`testing.py:50,105-106`), every entry
  resolves (no `-1` from the gather) and the value is one of those five PDG codes — at least
  one `> 20` (proton 2212) so a test can distinguish raw from remapped
  (`testing.py:101-104` comment).
- `instance_interaction[k]` = `track_interaction[…]` = `(track_row % 3) + 1`
  (`testing.py:317-318`), i.e. in `{1,2,3}`, value-keyed by the resolved track_id. This is
  the **one-hop** result for JAXTPC: `group_id → track → track_interaction` (JAXTPC has no
  per-particle table; the per-track interaction column is the one hop after the bridge).
- Hand-computed reference (the §6 invariant): for entry `k` on plane `p` of volume `v`,
  `tid = group_to_track_v{v}[gid_k]`; `row = index_of(tid in track_ids_v{v})`;
  `segment_pid[k] = track_pdg_v{v}[row]`, `instance_interaction[k] = track_interaction_v{v}[row]`.

### 4.2 JAXTPC `step` event

- `instance_particle` = `deposit_to_track` (raw track_id, masked by `volume_id`,
  `jaxtpc.py:442`).
- `segment_pid[i]` = `track_pdg[row(deposit_to_track[i])]` — and by the fixture invariant
  `deposit_to_track[i] == group_to_track[deposit_to_group[i]]` (`testing.py:19-21`,
  `testing.py:114`), so the step `segment_pid` for deposit `i` equals the hits `segment_pid`
  of any entry in the same group — a cross-stream consistency check.
- `instance_interaction[i]` = `track_interaction[row]` ∈ `{1,2,3}`.

### 4.3 LUCiD `hits` event

- `instance_particle` = `particle_idx` directly (`lucid.py:263`), each a valid index `<
  n_particles` (`testing.py:430-431`).
- `segment_pid[k]` = `category[particle_idx[k]]` (positional gather, `lucid.py:268-269`),
  value in `[0,5)` (`testing.py:389`).
- `instance_interaction[k]` = `per_particle.interaction_idx[particle_idx[k]]` — **one-hop**,
  positional by `particle_idx`, no per_track detour (the locked fact). (Today the fixture
  carries `per_track.interaction` `testing.py:401`; once `per_particle.interaction_idx` is
  surfaced — Part 03 — this becomes the one-hop column. Until then the LUCiD default may
  resolve `instance_interaction` through the `per_track` two-hop and a test asserts the
  values match.)

### 4.4 LUCiD `step` event

- `particle_idx = _lookup_per_track(track_idx, per_track.particle_idx)` (`lucid.py:281-283`);
  `track_idx` is a **positional** row index into `per_track` (`testing.py:405-408`).
- `instance_particle` = that `particle_idx`; `segment_pid` = `category[particle_idx]`
  (`lucid.py:287-288`). Invariant: every `step.track_idx` is a valid `per_track` row and
  every `per_track.particle_idx` is a valid `per_particle` index (`testing.py:23-26`), so no
  `-1` appears from a well-formed fixture.

### 4.5 Per-event `target_vertex` shape

`target_vertex` for one event = `np.stack([vertex_x, vertex_y, vertex_z])` → shape `(3,)`
(three scalars, per the locked fact and impl §3.4). It is attached to `data['_targets']
['target_vertex']`, **not** as a per-point `(N,3)` column. In a collated batch of B events
it is a list of B `(3,)` arrays (D24 list-collated `_`-prefixed metadata), or stacked to
`(B,3)` by the head — never `(ΣN, 3)`. `target_energy` is `()`/`(1,)`; `target_contained`
is `()` bool. (These depend on the `per_interaction` scope being surfaced — Part 03; until
then the framework simply omits the axis, see §5.)

---

## 5. Edge cases

1. **Unresolved FK → `fill`.** An FK `< 0`, out of range, or a value absent from `keyed_by`
   resolves to `fill` (default `-1` = ignore-index). Positional path: the bounds mask
   leaves the row at `fill` (`lucid.py:331-335`). Searchsorted path: `matched` is False so
   `np.where(..., fill)` (`jaxtpc.py:452-453`, `492`). The decorator never raises on an
   unresolved FK.
2. **Missing labl table / column.** If `labl is None`, no axis is decorated (the stream
   keeps only its raw FK columns) — mirrors today's `if labl is not None` guards
   (`lucid.py:265,277`; `jaxtpc.py:281,321`). If the labl is present but a specific
   `source` column is absent, the resolver returns `None` and the decorator **omits that
   one axis** (continue), it does not fill a whole column — mirrors
   `jaxtpc.py:444,478,484`'s `if … in vdata` guards and `lucid.py:267,280,286`'s
   `.get(...) is not None` checks. So `instance_interaction`/`target_*` are silently
   skipped when the labl predates the column, never fabricated.
3. **Sensor stream never decorated.** `sensor` has no per-particle separation; both datasets
   reject `('sensor','labl')` (`lucid.py:170-174`, `jaxtpc.py:213-218`) and the sensor
   builder emits no `segment`/`instance` (`lucid.py:220-233`, `jaxtpc.py:289-299`). The
   decorator is only invoked on `hits`/`step`. `event_broadcast` axes (`event_label`/
   `config_id`) are materialized in the **primary** stream by the base regardless of which
   modality that is, including a sensor-only SSL run — they don't require labl.
4. **CSR per_interaction (JAXTPC + LUCiD).** Per-interaction tables are CSR-encoded
   (LUCiD: `primary_{track_ids,pdgs,energies}_{data,offsets}`, `lucid_labl.py:18-20`,
   `engagement_plan_…:182-186`). A `scope="event"` axis that reads a per-interaction scalar
   (`vertex_x`, `neutrino_energy_MeV`, `contained`) reads the **dense scalar columns**, not
   the CSR. The CSR primaries are surfaced raw (Part 03) for a future per-interaction event
   unit (D36); the decorator does not decode CSR. Multi-interaction (pile-up) events would
   give a vector of vertices — out of scope here (per-event single-vertex assumed; flag
   §7).
5. **Multi-volume JAXTPC FK concat order.** The `step` FK (`deposit_to_track`) is built per
   volume by `volume_id` mask; the `hits` FK is built per plane. The decorated columns must
   concatenate in the **same order** as the stream's coord rows
   (`_merge_plane_dotted` sorted-plane order, `jaxtpc.py:378`; step volume order from the
   reader). The resolver owns this; a mis-order silently mislabels — covered by the
   cross-stream consistency test (§6.6) and the per-entry hand-gather test (§6.2).
6. **Length mismatch (defensive).** `deposit_to_track` length must equal the masked step
   count; today a mismatch logs and skips the volume (`jaxtpc.py:438-441`). The framework
   preserves this: a length mismatch on a `point` axis FK skips that volume's rows (they
   stay `fill`), never crashes.

---

## 6. Tests (Step-0 matrix; on `testing.py` fixtures; CPU, no GPU/WAND)

All use `make_jaxtpc_sample` / `make_lucid_sample` (`testing.py:55,333`) and the FK
invariants documented in the module docstring (`testing.py:17-26`). Reference values are
**hand-computed** from the fixture's raw FK arrays, independent of the decorator code.

**6.1 — Decoration == hand-computed positional gather (LUCiD `hits`).**
Setup: `make_lucid_sample(tmp, n_events=2)`; build `LUCiDDataset(modalities=('hits','labl'))`.
Action: `sub = ds.get_data(0)['hits']`; recompute `ref_seg = category[particle_idx]`,
`ref_inst = particle_idx` from the raw labl arrays.
EXPECTED: `sub['segment_pid'].ravel() == ref_seg`; `sub['instance_particle'].ravel() ==
particle_idx`; dtype int32; all in `[0,5)` for segment, `< n_particles` for instance.

**6.2 — Decoration == hand-computed searchsorted gather (JAXTPC `hits`).**
Setup: `make_jaxtpc_sample(tmp, readout_type='wire', n_events=2)`;
`JAXTPCDataset(modalities=('hits','labl'))`.
Action: for each plane entry `k` recompute `tid = g2t_v{v}[gid_k]`, `row = where(track_ids_v{v}
== tid)`, `ref_seg_k = track_pdg_v{v}[row]`.
EXPECTED: `sub['segment_pid'].ravel() == ref_seg` (concat in sorted-plane order);
`sub['instance_particle'].ravel() == concat(group_id per plane)`; values of `segment_pid`
∈ `{13,11,211,22,2212}` with at least one `> 20`.

**6.3 — Named keys present (both detectors).**
Setup: default `label_config` on each dataset, `modalities=('step','labl')` and
`('hits','labl')`.
Action: inspect `get_data(0)[stream]` keys.
EXPECTED: `segment_pid`, `instance_particle`, `instance_interaction` present (and
`instance_ancestor` for LUCiD); bare `segment`/`instance` present **only** when the
back-compat `label_key=` single-axis config is used; named keys absent on the `sensor`
stream.

**6.4 — `instance_interaction` one-hop correctness.**
Setup: JAXTPC `hits`+`labl`; LUCiD `hits`+`labl`.
Action (JAXTPC): recompute `track_interaction_v{v}[row(g2t[gid])]`. Action (LUCiD):
recompute `per_particle.interaction_idx[particle_idx]` (one-hop) and assert it equals the
two-hop `per_track.interaction` resolved via `particle_idx` for fixtures where they agree.
EXPECTED: `sub['instance_interaction'].ravel()` matches the hand one-hop gather;
JAXTPC values ∈ `{1,2,3}` (`testing.py:317-318`).

**6.5 — Per-event `target_*` length-1 / shape, not per-point.**
Setup: a fixture extended with a `per_interaction` scope (or a monkeypatched resolver
`event_value` returning `(3,)` vertex / scalar energy) so the axis is producible.
Action: `data = ds.get_data(0)`; inspect `data['_targets']`.
EXPECTED: `target_vertex` shape `(3,)`; `target_energy` shape `()`/`(1,)`;
`target_contained` bool scalar; **none** appears as a key inside `data[stream]` (per-point);
none appears in that stream's `index_valid_keys`; after an N-changing transform the
per-point columns shrink but `_targets` is unchanged.

**6.6 — Fill on unresolved FK.**
Setup: hand-corrupt one event's FK (set a `deposit_to_track` entry to a track_id absent
from `track_ids`, and a `particle_idx` to `n_particles+5`).
Action: decorate.
EXPECTED: those rows get `segment_pid == -1` (and `instance == -1` for the out-of-range
positional case); no exception raised; neighboring rows unaffected.

**6.7 — Multi-config label stability (mixture).**
Setup: two source roots (two `make_*_sample` dirs), mixture with explicit
`{config: label}`; build with `event_broadcast` `event_label`/`config_id`.
Action: read several events; recompute per-event `event_label` from the source they came
from.
EXPECTED: `event_label`/`config_id` are per-point `(N,1)` columns, **constant within an
event**, equal to the source's mixture label; the same `(config_id, source_event_idx)`
yields the same decoration across runs (decoration is a pure function of labl + FK, no RNG).

**6.8 — Extensibility: add an axis without touching the decorator.**
Setup: append `dict(out="segment_ancestor", scope="point", fk="particle_idx",
source=("particle","ancestor_particle_idx"), fill=-1)` (LUCiD) to `label_config`; ensure the
`segment*` prefix is covered by `index_operator` (Part 02).
Action: decorate, then run a `GridSample`/`SphereCrop` (N-changing).
EXPECTED: `segment_ancestor` is emitted, equals `ancestor_particle_idx[particle_idx]`
(`lucid_labl.py:46-47,196-209`), and stays length-aligned with `coord` after the N-changing
transform (prefix-match picked it up) — no decorator edit was required.

**6.9 — Back-compat single-axis `label_key` (JAXTPC).**
Setup: `JAXTPCDataset(modalities=('step','labl'), label_key='pdg')` with the back-compat
one-spec config.
Action: compare against the current `_decorate_step_from_labl` output (import the present
method or snapshot its arrays).
EXPECTED: bare `segment` (raw `track_pdg`) and `instance` (raw track_id) byte-identical to
today (`jaxtpc.py:442,453`); a following `RemapSegment(scheme='motif_5cls')`
(`detector_transforms.py:132`) produces the same remapped classes as before.

(LUCiD + JAXTPC are both exercised in 6.1–6.9; each numbered test names its detector.)

---

## 7. Reversible defaults & risks

**Reversible defaults (decide at code time; document in code, D34):**
- The `event_targets` carrier name (`_targets` vs per-axis `_target_vertex`) and whether
  targets are stacked at the head or list-collated — D24 says `_`-prefixed list-collated;
  exact key name is reversible.
- Whether the default `label_config` lives as a subclass class-attr or is assembled in
  `__init__` from `modalities`.
- The spec field encoding (`source="self"` vs a dedicated `kind="fk_value"`).
- `fill` default `-1` per axis (matches today; overridable per spec).

**Risks / flags (do NOT invent producers):**
- **`target_mask` has no producer.** The hmae config references `target_mask` but
  `HMAECollate` does not emit it (engagement `…:208,248`; impl §7 "`target_mask` has no
  producer — hmae config↔`HMAECollate` drift — flag for the hmae owner, don't invent one").
  This part does **not** add a `target_mask` axis; if a config asks for it, surface the
  drift to the hmae owner.
- **`per_interaction` is not surfaced yet.** Both labl readers currently skip it (LUCiD:
  `lucid_labl.py:15-20`; JAXTPC has only per-track `track_interaction`). `target_vertex`/
  `target_energy`/`target_contained` are **producible only after Part 03 surfaces
  `per_interaction`** (impl §3.4). Until then the decorator omits these axes (the
  `resolve_column`/`event_value` returns `None`, §5 case 2) — it must not fabricate vertex
  values. The §6.5 test stands up the scope via fixture/monkeypatch precisely so it doesn't
  silently pass on an omitted axis.
- **Multi-interaction (pile-up) events** would make per-event targets a vector, breaking the
  single-`(3,)`-vertex assumption. Out of scope (per-event single-vertex, D36 default event
  unit); flag if a pile-up task lands.
- **JAXTPC vertex/energy targets** have no labl table today (only `track_interaction`). The
  JAXTPC `target_vertex`/`target_energy` rows in the matrix are aspirational pending a
  writer-side per-interaction table — flag, don't invent.

---

## 8. Dependencies on other parts

- **Readers (Part 03 / impl §3.4)** must surface the raw FKs and chain tables the resolver
  needs: LUCiD `per_particle.interaction_idx` (for one-hop `instance_interaction`) and the
  `per_interaction` scope (`vertex_{x,y,z}`, `neutrino_energy_MeV`, `contained`, CSR
  primaries — currently NOT read, `lucid_labl.py:15-20`); JAXTPC `track_interaction`
  (already surfaced, `jaxtpc.py:78`) and any future per-interaction table. The decorator is
  blocked on these for the corresponding axes (§7).
- **`index_operator` prefix-match (Part 02 / impl §3.2)** must carry the named point keys.
  The default `index_valid_keys` already lists `segment_pid`/`instance_particle`/
  `instance_interaction`/`segment_interaction` (`transform.py:51-54,70`), but a **new** axis
  family needs its `out` prefix (`segment*`/`instance*`/`target*`) added there
  (underscore-boundary match) so N-changing transforms keep it aligned; per-event `target_*`
  must be **excluded** by the leading-dim≠N rule.
- **Collate (Part 06 / impl §3.6)** — single-stream collate carries the per-point
  `event_label`/`config_id` columns with no change; per-event `_`-prefixed `_targets` are
  list-collated (D24). No collate change is needed for the per-point axes.
- **Base `MultiModalEventDataset` (Part 05 / impl §3.3)** owns `event_label`/`config_id`
  materialization (the `event_broadcast` source values) and dispatches `get_data` to the
  subclass builders that call `_decorate_from_labl`. The base also owns the `label_config=`
  constructor arg; subclasses own the `fk_resolver` and default config.
- **Base joint event index (Part 05 / impl §3.3a, D42) — decoration correctness depends on
  it.** Label decoration joins the stream reader (step/hits) and the labl reader at the
  *same* event; that join is only correct if the readers are **aligned**, which the base's
  joint index guarantees (it intersects present `event_*` keys across modalities keyed on
  `source_event_idx`). Without it the base inherits the cross-modality desync (handoff §4):
  under `min_deposits>0` or a partial gap, the stream and labl readers return different
  physics events for the same idx and every FK join here becomes meaningless. The **A5
  cross-modality regression test** (Part 05 / impl §6.16) — same `source_event_idx` across
  `('step','sensor','hits','labl')` for every served idx, plus the gap-in-one-modality
  variant — is the gate that locks this alignment for the labeled tasks this part produces.
- **Downstream transforms (`RemapSegment`/`PDGToSemantic`, `detector_transforms.py`)** are
  *consumers* of the decorated keys, not part of this part. `RemapSegment`
  (`detector_transforms.py:132`) remaps raw `segment`/`segment_pid` to class indices;
  `PDGToSemantic` (`detector_transforms.py:66`) is the **no-labl fallback** that synthesizes
  `segment_pid`/`instance_particle` from PDG (`detector_transforms.py:99,112`) — it must
  stay a no-op when the labl decorator already wrote those keys (its
  `detector_transforms.py:89` `if 'segment' in data_dict` guard generalizes to the named
  keys).
