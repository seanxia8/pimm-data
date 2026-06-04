# Part 05 — Collate, streams, eval & reproducibility (implementation spec)

**Status:** build spec. Part of the pimm-data data-layer implementation series
(`01_transforms.md`, `02_dataset_base.md`, `04_label_decoration.md`, this is `05`).
Near-term structure is **single-stream-per-task (D35)**; multi-stream is
**design-for-extension, NOT built (D39)**.

**Source decisions:** D19, D23, D24 (multi-stream topology — all downgraded to
FUTURE by D35/D39), D35 (single-stream-per-task near-term, collate reverts to
byte-identical REPLACE), D39 (the four seams to lock), D41 (eval reproducibility).
Implementation references: `implementation_plan_pimm_data_datalayer.md` §3.6
(collate), §3.7 (eval-hook rewiring), §9.1 (the four seams), §9.3 (reproducibility).
Reversible-detail latitude from D34.

**Files (read-only ground truth for this spec):**

- pimm-data: `src/pimm_data/collate.py` (the collate functions),
  `src/pimm_data/transform.py` (`Collect`, `index_operator`),
  `src/pimm_data/detector_transforms.py` (`ApplyToStream`),
  `src/pimm_data/jaxtpc.py` / `src/pimm_data/lucid.py` (nested dataset output).
- pimm (`research` branch,
  `/sdf/home/o/omara/.claude/jobs/21ffc656/particle-imaging-models`):
  `pimm/datasets/utils.py` (collate — confirmed byte-identical, below),
  `pimm/datasets/transform.py` (the older `Collect`, behind pimm-data's),
  `pimm/engines/train.py` (loader construction),
  `pimm/engines/hooks/lucid_event_probe.py` (the eval hook to rewire),
  `pimm/datasets/lucid_event_ssl.py` (current `data_list`/`datasets`/`event_label`
  producer — to be dissolved into the base),
  `configs/lucid/pretrain/pretrain-sonata-v1m1-sk-like-mu-e.py` (the live config
  exercising the probe + train/val transforms).

**Files this part edits (at code time, not in this spec):**

- pimm-data: `src/pimm_data/collate.py` — *no change* (verified byte-identical;
  §3.1). Multi-stream collate is a FUTURE additive function, not added now.
- pimm: `pimm/engines/hooks/lucid_event_probe.py` — rewire `_event_keys` onto
  `event_identity(idx)` (§3.4).
- The per-point `event_label`/`config_id` materialization lives in the base
  (`multimodal.py`, Part 02 §"identity"); this part only specifies the **contract**
  the collate + probe rely on.

---

## 1. Purpose & scope

This part pins down four tightly coupled pieces of the single-stream build:

1. **Collate.** Confirm the near-term collate is the *existing* `collate_fn` /
   `point_collate_fn` (byte-identical REPLACE, D35/§3.6 — no behavioral change),
   document its mechanics (the `offset` diff/cumsum rebasing, the `_`-prefixed key
   drop, what reaches the model), and how `event_label` / `config_id` survive
   collation as ordinary per-point columns.

2. **`Collect(stream=)` contract.** How one nested stream
   (`data_dict[stream]`) becomes a flat `coord` / `feat` / `offset` + named-label
   dict, what `keys` / `*_keys` must list, and the `ApplyToStream` wrap for
   per-stream transforms that hardcode `'coord'`/`'segment'`.

3. **Multi-stream extension seams.** Document the four seams D39 locks now so the
   FUTURE namespaced-collate + model-side primary/aux adapter is a bounded
   *addition*, not a rewrite. **Explicitly marked NOT-BUILT.**

4. **Eval-hook rewiring + reproducibility.** Replace
   `lucid_event_probe._event_keys`'s reach into `data_list` / `datasets` /
   `source_root` with the stable `event_identity(idx)` + public `split` API
   (keep `_dataset_split`); state the per-point `event_label` collation
   requirement; and the per-run reproducibility contract (train≡eval transforms,
   per-run holdout+pipeline record, disjointness guard).

**Out of scope (other parts):** the `MultiModalEventDataset` base, holdout hash,
`event_identity`/`split` implementation, and `event_label`/`config_id`
materialization (Part 02); label decoration / `label_config` (Part 04); transform
merges and `index_operator` prefix-match details (Part 01). This part *consumes*
those contracts and states exactly what it needs from each.

---

## 2. Current state (file:line)

### 2.1 Collate — `pimm_data/collate.py` == pimm `datasets/utils.py`

**Byte-identity verified.** `diff
src/pimm_data/collate.py
…/pimm/datasets/utils.py` → identical (no differing bytes). Both files are the
same upstream Pointcept `utils.py` header
(`collate.py:1-12` / `utils.py:1-12`), so the near-term REPLACE is a literal
file copy with zero behavioral delta. Three functions:

**`collate_fn(batch, mix_prob=0)`** (`collate.py:15-50`) — recursive packer:

- `torch.Tensor` leaf → `torch.cat(list(batch))` (`collate.py:23-24`): per-key
  tensors from each sample are concatenated along dim 0 (point axis).
- `str` leaf → `list(batch)` (`collate.py:25-27`): the judgement is *before*
  `Sequence` because `str` is a `Sequence`; so `name` survives as a python list of
  strings, never tensorized.
- `Sequence` (list/tuple) leaf (`collate.py:28-33`): appends a per-sample length
  tensor `torch.tensor([data[0].shape[0]])`, recurses elementwise via `zip(*batch)`,
  then `torch.cumsum(..., dim=0).int()` on the last element → an **offset** vector.
  (This is the list-input path; the dict path below is what the LUCiD/JAXTPC configs
  use.)
- `Mapping` (dict) leaf (`collate.py:34-48`): the hot path. For each `key` in
  `batch[0]`:
  - **`"offset" not in key`** → recurse `collate_fn([d[key] for d in batch])`.
  - **`"offset" in key`** (`collate.py:38-43`) → the **diff/cumsum offset rebasing**:
    ```python
    torch.cumsum(
        collate_fn([d[key].diff(prepend=torch.tensor([0])) for d in batch]),
        dim=0,
    )
    ```
    Each sample's `offset` is a cumulative-sum vector (single-event:
    `tensor([n_points])`; from `Collect`'s `offset_keys_dict`). `.diff(prepend=0)`
    converts cumsum→per-sub-cloud counts, the inner `collate_fn` concatenates those
    counts across samples, and the outer `cumsum` re-accumulates into a **global**
    offset across the batch. This is what lets a global model see a single flat
    point cloud with batch boundaries marked by `offset`.
  - **`_`-prefixed key drop** (`collate.py:45-46`):
    `for key in batch[0] if not key.startswith("_")`. Any key whose name starts
    with `_` is **silently dropped** at collate — the mechanism D24 relies on to
    carry ragged per-event metadata (`labl` / `bridges` / per-event targets) that
    must *not* be tensor-collated.

- else → `default_collate(batch)` (`collate.py:49-50`): scalars/0-d arrays stack
  into a `(B,)` tensor.

**`point_collate_fn(batch, mix_prob=0)`** (`collate.py:53-74`) — wraps `collate_fn`,
then with probability `mix_prob` does the **CutMix-style pair fusion**: if
`"instance"` present, renumber instance ids of odd samples by the running
`num_instance` (`collate.py:59-69`); then halves `offset` by keeping
`offset[1:-1:2]` + the last (`collate.py:70-73`), fusing sample pairs into one
cloud. `mix_prob=0` in the live LUCiD config (`pretrain-…-mu-e.py:17`) → this branch
is a no-op there.

**`inseg_collate_fn(batch, mix_prob=0)`** (`collate.py:81-99`) — instance-seg path:
each batch item is a *list* of per-query dicts; flattens then calls `collate_fn`.

**Where collate is wired in** (`pimm/engines/train.py`): train loader uses
`partial(collate_fn, mix_prob=self.cfg.mix_prob)` (`train.py:331`); val loader uses
bare `collate_fn` (`train.py:354`); `InsegTrainer` uses `inseg_collate_fn`
(`train.py:449`, `:474`). The collate fn is a config/trainer choice passed to
`torch.utils.data.DataLoader(collate_fn=…)` — **the dataset never calls collate**;
it returns one flat dict per `__getitem__` and the loader batches.

### 2.2 `Collect(stream=)` — `pimm_data/transform.py:87-140`

pimm-data's `Collect` is **ahead of** pimm's (`pimm/datasets/transform.py:120-143`)
and §3.1 of the impl plan says *do NOT overwrite it* (merge direction reversed for
this one class). The pimm version (`transform.py:120-143`) has **no `stream=`**, no
tensor autoconvert, no passthrough — it indexes `data_dict[key]` directly and
`.float()`s feat keys. pimm-data's adds three things:

- **`stream=` scoping** (`transform.py:108`, `:124-125`):
  `source = data_dict[self.stream] if self.stream is not None else data_dict`. With
  `stream='step'`, keys are pulled from the nested `data_dict['step']` sub-dict; the
  output stays **bare** (`coord`, `segment`, `feat`, …) so collate / `Point` / model
  see the same flat shape as the old single-cloud path (`transform.py:99` docstring).
- **`_to_tensor` autoconvert** (`transform.py:116-122`, `:129`, `:135`): numpy →
  `torch.from_numpy` so DataLoader workers transfer via fd-sharing (~400 B) instead of
  pickling (`transform.py:101-105` docstring).
- **`name`/`split` passthrough** (`transform.py:136-139`): when `stream` is set, lifts
  top-level `name`/`split` into the output if not already pulled — so per-event
  metadata at the top level survives the stream scoping.

Key construction (`transform.py:124-140`): for each `key` in `keys` →
`data[key] = _to_tensor(source[key])`; for each `offset_keys_dict` entry (default
`dict(offset="coord")`, `transform.py:109-110`) →
`data[key] = torch.tensor([source[value].shape[0]])` (the per-event single-element
offset = point count); for each `**kwargs` entry ending `_keys` (e.g. `feat_keys`),
strip the suffix and `torch.cat([_to_tensor(source[k]).float() … ], dim=1)`
(`transform.py:132-135`).

**Live usage** (`pretrain-…-mu-e.py`): the val `Collect`
(`pretrain-…-mu-e.py:258-262`) is **single-stream, no `stream=`**:
`keys=("coord","energy","time","event_label","grid_size","name")`,
`feat_keys=("coord","energy","time")`. It lists `event_label` directly in `keys`
(`:260`). The train `Collect` (`:225-252`) is the MultiView variant
(`global_*`/`local_*` keys, empty `offset_keys_dict`).

### 2.3 `ApplyToStream` — `pimm_data/detector_transforms.py:26-63`

Dispatches a sub-pipeline into a nested sub-dict
(`detector_transforms.py:53-62`): `if self.stream not in data_dict: return
data_dict` (no-op when the stream is absent, unless `required=True` →
`KeyError`, `:54-60`); else `data_dict[self.stream] = self.inner(data_dict[self.stream])`
(`:61-62`) where `self.inner = Compose(transforms)` (`:51`). This is the wrapper for
transforms that hardcode `'coord'`/`'segment'` (GridSample, RandomRotate) so they
operate on the chosen stream's sub-dict. Registered as a transform
(`detector_transforms.py:26`), config-buildable
(`detector_transforms.py:36-39` docstring example).

### 2.4 Nested dataset output (the seam already in place)

- pimm-data `JAXTPCDataset.get_data` (`jaxtpc.py:233-269`) returns a **nested** dict:
  top-level `name`/`split` (`jaxtpc.py:237-240`) + per-modality sub-dicts
  `data['labl']` (`:252`), `data['hits']` (`:256`), `data['sensor']` (`:262`),
  `data['step']` (`:266`), plus `data['bridges']` (`:258-259`, a per-event join
  artifact). Module docstring confirms the contract (`jaxtpc.py:14-35`): nested, "no
  prefixed aliases — transforms pick a stream explicitly". `LUCiDDataset` mirrors this
  (`lucid.py:12-36` docstring).
- The **old vendored** pimm `JAXTPCDataset` (`pimm/datasets/jaxtpc_dataset.py:188-268`)
  is the *flat/prefixed* legacy (`resp_coord`, `corr_coord`, copies the primary spatial
  source up to bare `coord`/`segment`, `:242-257`) — this is the `seg/resp/corr`
  structure D33/§4 migrates off; it is NOT the seam. The seam is the pimm-data nested
  form above.

### 2.5 The eval hook's internal reach — `lucid_event_probe.py`

`EventLinearProbeEvaluator._event_keys` (`lucid_event_probe.py:160-190`) reaches into
dataset internals to build the per-event identity set used by the train/val
disjointness guard:

```python
161  def _event_keys(cls, dataset):
...
166      while isinstance(dataset, Subset):
167          indices = [int(idx) for idx in dataset.indices]
...
172          dataset = dataset.dataset
173
174      data_list = getattr(dataset, "data_list", None)
175      sources = getattr(dataset, "datasets", None)
176      if data_list is None or sources is None or len(data_list) == 0:
177          return None
...
186      for dataset_idx in indices:
187          source_idx, event_idx = data_list[int(dataset_idx) % base_len]
188          source_key = cls._source_key(sources[int(source_idx)], int(source_idx))
189          keys.add((source_key, int(event_idx)))
190      return keys
```

`_source_key` (`lucid_event_probe.py:150-158`) digs further:
`source.get("source_root") or source.get("data_root")` (`:153`) → `os.path.realpath`
(`:155`), else `source["name"]` (`:157`), else `str(source_idx)`. This couples the
hook to the **exact internal layout** of `lucid_event_ssl.py`'s `datasets` list of
dicts (`lucid_event_ssl.py:169-177`, each carrying `source_root`/`name`/`label`) and
`data_list` of `(source_idx, event_idx)` tuples (`lucid_event_ssl.py:251`). The
identity unit is `(realpath(source_root), event_idx)` — `event_idx` here is the
**positional file index**, NOT a stable `source_event_idx`, so it is only stable while
the file ordering is stable.

`_format_event_key` (`lucid_event_probe.py:192-195`) renders a 2-tuple
`f"{source_key}:{event_idx}"`. `_dataset_split` (`:140-148`) reads
`getattr(dataset, "split", None)` after peeling `Subset` (`_base_dataset`, `:134-138`)
— **already on the public `split` API**, keep as-is. `_validate_heldout_source`
(`:197-246`) calls `_event_keys` on both train and val loaders (`:219-220`) and raises
on overlap (`:229-238`).

The label path: `_labels_by_event` (`lucid_event_probe.py:114-128`) maps the collated
`input_dict[self.label_key]` (default `"event_label"`, `:36`) to one label per event:
- if `labels.numel() == n_events` → return as-is (`:118-119`) — the **current**
  length-1-per-event path;
- elif `labels.numel() == n_points` → slice the first label in each offset window
  `labels[start:end][0]` (`:120-124`) — the **per-point broadcast** path this part
  switches to;
- else → `ValueError` (`:125-128`).

`_process_batch` (`:248-266`) reads `point.feat` + `point["offset"]` (or
`input_dict["offset"]`, `:253`), builds `offsets = [0] + offset.tolist()` (`:254`), and
calls `_labels_by_event(input_dict[self.label_key], offsets, features.shape[0])`
(`:255-257`).

### 2.6 The current `event_label` materialization (to change)

`lucid_event_ssl.py:310` sets `"event_label": np.array([source["label"]],
dtype=np.int64)` — a **length-1 per-event** array; `"config_id":
np.array([source_idx], dtype=np.int64)` (`:311`) likewise. Under the current val
`Collect` (lists `event_label` in `keys`, `pretrain-…-mu-e.py:260`) and `collate_fn`,
these `(1,)` arrays concatenate to a `(n_events,)` tensor → the
`labels.numel() == n_events` branch (`lucid_event_probe.py:118`). This is the path Part
02 changes to **per-point broadcast** (`event_broadcast` scope, §3.5 of impl plan /
Part 04) so the probe uses the offset-slice branch.

### 2.7 What is NOT yet built

- `MultiModalEventDataset` base / `event_identity(idx)` — **no `multimodal.py`
  exists** (`ls` → not found). Part 02 builds it. This part's eval rewrite *depends*
  on `event_identity` + the per-point `event_broadcast` materialization existing.
- Namespaced multi-stream collate — does not exist and is **NOT built** here (FUTURE,
  §3).

---

## 3. Target design

### 3.1 Collate stays the existing single-stream `collate_fn` (REPLACE, no change)

**Decision (D35 / §3.6).** Near-term collate IS the existing
`collate_fn`/`point_collate_fn`/`inseg_collate_fn` — a byte-identical REPLACE of
pimm's `utils.py` into pimm-data's `collate.py` (already done; `diff` clean, §2.1).
**No behavioral change, no new function added now.**

**Why unchanged is correct:**

- Each task selects **one** stream via `Collect(stream=)` → a flat
  `coord`/`feat`/`offset`+labels dict (§3.2). After `Collect`, the per-sample dict is
  exactly the single-cloud shape the existing collate already handles — the nested
  structure is gone by collate time. There is nothing stream-shaped left to pack.
- The diff/cumsum offset rebasing (`collate.py:38-43`) and the `_`-prefixed drop
  (`collate.py:45-46`) are the two mechanisms the single-stream + future-multi-stream
  designs both need; both already exist. Touching collate now would risk the parity
  guarantee (Step 3 gate: "identical first-batch tensors vs vendored", §2 of impl
  plan) for zero functional gain.
- `event_label`/`config_id` reach the batch as **ordinary per-point columns** (§3.5),
  collated by the plain tensor-cat path (`collate.py:23-24`) — no special-casing.

**Shim requirement (§5 of impl plan):** the pimm `__init__.py` re-export shim must keep
exporting `point_collate_fn`/`collate_fn`/`inseg_collate_fn` from pimm-data so
`train.py:331/354/449/474` resolve unchanged.

**The multi-stream collate is a FUTURE additive function** (D23/D24, §8 of impl plan)
— see §3.3. NOT built now.

### 3.2 `Collect(stream=)` contract (single-stream)

The terminal transform of every task pipeline. Given the nested dataset dict, it
emits the flat model-input dict.

**Contract:**

1. **Stream scoping.** `Collect(stream='step', …)` pulls all keys from
   `data_dict['step']` (`transform.py:124-125`). Single-stream-per-task means exactly
   one `Collect(stream=…)` per pipeline; the other streams in the nested dict are
   simply never collected (and, if `_`-prefixed or non-tensor, dropped at collate).
2. **`keys=` must list** every per-point/per-event array the model + downstream
   consumers need, by its **final name**: at minimum `coord`; the task's labels
   (`segment` / `segment_pid` / `instance_particle` / per-event `target_*`); and the
   probe's `event_label` when the probe is active (val pipeline). `name`/`split` are
   auto-passed-through when `stream` is set (`transform.py:136-139`) — do not also list
   them unless a non-stream `Collect` is used (then list `name` explicitly, as the live
   val config does, `pretrain-…-mu-e.py:260`).
3. **`offset` derivation.** `offset_keys_dict` (default `dict(offset="coord")`) sets
   `offset = torch.tensor([source['coord'].shape[0]])` (`transform.py:130-131`) — the
   per-event point count, which collate rebases into the global batch offset
   (`collate.py:38-43`). MultiView pipelines pass `offset_keys_dict=dict()` and supply
   `global_offset`/`local_offset` themselves (`pretrain-…-mu-e.py:241`).
4. **`*_keys` → concatenated feature tensors.** `feat_keys=("coord","energy","time")`
   produces `feat = cat([coord, energy, time], dim=1).float()` (`transform.py:132-135`).
   Every key in a `*_keys` list must be a 2-D `(N, c)` array in the stream so the
   `dim=1` cat is valid; this fixes the model's `in_channels` (e.g.
   `in_channels=5` for `[x,y,z,log_PE,rel_log_time]`, `pretrain-…-mu-e.py:58`).

**Per-stream transforms via `ApplyToStream`.** N-changing or coord-hardcoded transforms
(GridSample, RandomRotate, NormalizeCoord) run **inside**
`ApplyToStream(stream='step', transforms=[…])` (`detector_transforms.py:36-39`) so they
mutate `data_dict['step']` in place before the terminal `Collect(stream='step', …)`.
A pipeline that operates on one stream is: `ApplyToStream(stream=S, [normalize,
gridsample, augment, label-encode]) → Collect(stream=S, keys=…, feat_keys=…)`.

**`index_valid_keys` caveat (from §3.2 of impl plan / Part 01):** `ApplyToStream` does
NOT propagate `index_valid_keys` across the wrapper boundary, so any explicit
`Update(index_valid_keys=…)` a stream needs must itself be **inside** that stream's
`ApplyToStream`. The migrated `jaxtpc_seg` config does exactly this (§4 of impl plan).

**Migrated JAXTPC seg config terminal** (impl plan §4, ground truth for the contract):
`Collect(stream='step', keys=("coord","grid_coord","segment"),
feat_keys=("coord","energy"))`.

### 3.3 The four multi-stream extension seams (NOT BUILT — design-for-extension)

> **NOT BUILT.** This section documents the seams D39/§9.1 lock so the FUTURE
> multi-stream-in-batch path (D23/D24) is a **bounded additive change** — a new
> collate function + a model-side adapter — never a rewrite of the dataset, the
> decoration, or the single-stream collate. Build-now only flips if a concrete
> multi-stream model is committed this cycle (D39).

**Seam 1 — Nested dataset output (never flatten in the dataset).** Already in place
(§2.4): `get_data` returns `{stream: {coord, feat, …}}` (`jaxtpc.py:233-269`,
`lucid.py`). A second stream in a batch is *just present* in the nested dict; no
dataset change is needed to add it. The single-stream path simply collects one of the
streams and ignores the rest. **What to preserve:** never copy a stream's keys up to
bare top-level `coord`/`segment` (the legacy `jaxtpc_dataset.py:242-257` anti-pattern);
keep every stream self-contained under its name.

**Seam 2 — Per-event label decoration (joins resolved before batching).** Each stream
carries its own decorated labels (`segment`, `segment_pid`, `instance_*`,
`event_label`), built in the dataset/reader from `labl` (Part 04 / D24/D28). Any
cross-stream value (e.g. a `hits`↔`step` join, the `bridges` artifact at
`jaxtpc.py:257-259`) is **computed per-event before batching** and carried as
`_`-prefixed list-collated metadata (dropped by `collate.py:45-46`). There is therefore
**never a join-at-collate** — the future namespaced collate only *packs* already-joined
per-point columns. **What to preserve:** keep cross-stream artifacts `_`-prefixed (so
collate drops them) and resolve them in `get_data`, not in a transform or collate.

**Seam 3 — Stream-aware collate *structure*.** The single-stream collate operates on a
*named* stream's flat dict (after `Collect(stream=)`), so the future
`multistream_collate_fn` is a **loop over streams**, each producing its own
`{coord, feat, offset, labels}` with its **own offset** (per-stream cumsum via the same
`collate.py:38-43` mechanism), assembled into `{stream: {…}}`. The existing
`collate_fn` is reusable per-stream inside that loop — the future function calls it once
per stream rather than once total. **What to preserve:** do not bake a single global
`offset` assumption anywhere downstream of collate that a second stream would break;
the model-side adapter (below) is where primary/aux is resolved.

**Seam 4 — Stable `event_identity`.** Future multi-stream batches must align streams
*by event* (stream A's i-th sub-cloud ↔ stream B's i-th sub-cloud). The base's
`event_identity(idx) -> (config_id, source_event_idx)` (Part 02) is the alignment key:
each stream's per-event order is the dataset's `data_list` order, and
`event_identity` is the stable, modality-independent label for that order.
**What to preserve:** `event_identity` must be a pure function of stable file identity
(not positional index), owned by the base, and survive `Subset` wrapping (Part 02).

**Future-add sketch (NOT BUILT):**

- *Collate:* add `multistream_collate_fn(batch)` to `collate.py` that, for each stream
  name present in `batch[0]`, runs the per-stream pack (reusing `collate_fn`'s dict
  path) and returns `{stream: {coord, feat, offset, …}}` with per-stream offsets. The
  single-stream `collate_fn` is untouched and stays the default. (D34: collate-fn
  decomposition is a reversible detail decided at code time.)
- *Model side:* a multi-stream segmentor **adapter** names the *primary* stream and
  flattens it to a `Point` (the model-side primary-stream→`Point` step, D19/D23);
  auxiliary streams are passed as named side-inputs. The dataset and decoration do not
  change.
- *Mix-up:* `point_collate_fn`'s mix branch is **disabled under multi-stream** for the
  first cut (D23; coherent cross-stream mix-up is a research problem) — a reversible
  default (D34).

### 3.4 Eval-hook rewiring (`lucid_event_probe.py`)

**Goal (§3.7 / D41).** Decouple the probe from dataset internals
(`data_list`/`datasets`/`source_root`) onto the public, stable
`event_identity(idx)` + `split` API the base owns (Part 02). Behavior preserved:
train/val disjointness guard, per-event label mapping, `Subset` peeling.

**Replace `_event_keys` (`lucid_event_probe.py:160-190`).** New body:

1. Peel `Subset` to recover the *base-dataset indices* (keep the existing loop at
   `:166-172` that composes nested `Subset.indices`).
2. Resolve the base dataset (reuse `_base_dataset`, `:134-138`).
3. Build the key set from `event_identity` instead of `data_list`/`datasets`:
   ```python
   ident = getattr(dataset, "event_identity", None)
   if ident is None:
       return None                      # keep the graceful-None contract (:176)
   indices = range(len(dataset)) if subset_indices is None else subset_indices
   return {ident(int(i)) for i in indices}
   ```
   `event_identity(i)` returns `(config_id, source_event_idx)` (Part 02) — a stable,
   modality-independent tuple. This **removes the reach into**
   `getattr(dataset,"data_list")` (`:174`), `getattr(dataset,"datasets")` (`:175`),
   and the `source_root`/`data_root`/`name` dig in `_source_key` (`:150-158`).
   `_source_key` is **deleted** (its job — turning a source descriptor into a stable
   key — is now `event_identity`'s `config_id`).

4. **Generalize `_format_event_key`** (`:192-195`) from a fixed 2-tuple to a
   multi-component tuple:
   ```python
   @staticmethod
   def _format_event_key(key):
       return ":".join(str(c) for c in key)
   ```
   (works for the `(config_id, source_event_idx)` 2-tuple and any future N-tuple).

5. **Keep `_dataset_split`** (`:140-148`) unchanged — it already reads the public
   `split` after peeling `Subset` (`_base_dataset`), which is exactly the contract
   Part 02 guarantees survives `Subset`.

6. `_validate_heldout_source` (`:197-246`) is unchanged in structure: it still calls
   `_event_keys(train_dataset)` / `_event_keys(val_dataset)` (`:219-220`), still
   raises on `train_keys & val_keys` overlap (`:229-238`), and the
   forbidden/heldout split check (`:208-215`) keeps using `_dataset_split`. The only
   change is what `_event_keys` returns (now `event_identity` tuples).

**Per-point `event_label` collation requirement (§3.5 / D28).** The base materializes
`event_label`/`config_id` as **per-point broadcast** arrays (`event_broadcast` scope,
shape `(N,1)` per event), replacing the current length-1 form (§2.6,
`lucid_event_ssl.py:310-311`). The val `Collect` must list `event_label` in `keys`
(as the live config already does, `pretrain-…-mu-e.py:260`); after collate it is a
`(total_points,)` tensor, so `_labels_by_event` takes the
`labels.numel() == n_points` branch (`lucid_event_probe.py:120-124`) and slices the
first label per offset window. **No probe code change is needed for the label path** —
the existing offset-slice branch already handles it; the requirement is purely on the
*dataset materialization + the `Collect` key list*. (The `numel()==n_events` branch
stays as a back-compat fallback, `:118-119`.)

**Net probe dependency after rewire:** only (a) `dataset.event_identity(idx)` +
public `split` for the guard, and (b) per-point `event_label` collated +
`point.feat`/`point["offset"]` from the model (`:252-253`). No dataset-internal
attribute access remains.

### 3.5 Reproducibility contract (D41 / §9.3)

The eval/probe path must be a faithful replay of training data conditions. Three
enforced properties:

1. **Per-run record.** Persist, next to the experiment snapshot (alongside
   `config.py` in the `train.sh` code-snapshot, §5 of impl plan), (a) the **holdout
   spec** — `seed`, fractions or `n_per_config`, and the identity scheme (the blake2b
   hash of `(config_id, source_event_idx)`, Part 02) — and (b) the **resolved
   transform pipeline** (config-hashed). This makes the exact heldout set and the exact
   preprocessing reconstructible from the run artifacts. (Mechanism: the submodule/SHA
   snapshot of §5 captures the data-layer code; the holdout+pipeline record captures
   the *parameters*.)

2. **train≡eval transforms (shared base fragment).** The val/probe transform pipeline
   must be the *same registered transforms with the same params* as training's
   deterministic subset — no re-specified or drifted `GridSample`/normalize. Enforce
   **by construction**: a shared `base_event_transform` config fragment that both train
   and val splice in (the live config already does this:
   `base_event_transform` at `pretrain-…-mu-e.py:149-179`; `transform =
   base_event_transform + [...]` at `:181`; `val_transform = base_event_transform +
   [...]` at `:255`). Train appends *augmentation* (MultiView, jitter); val appends
   only `ToTensor`/`Collect`. The shared fragment carries the determinism-sensitive
   ops (NormalizeCoord, GridSample, LogTransform, RelativeLogNormalize) identically.
   The test matrix (§6, T8) asserts the val pipeline's registered transform list equals
   the train pipeline's deterministic subset.

3. **Disjointness guard.** The holdout is selected by the same hash for both splits
   (Part 02), so train/val/test are provably disjoint and stable. The probe verifies
   this at runtime via `event_identity`: `_validate_heldout_source`
   (`lucid_event_probe.py:197-246`) refuses to run on a non-heldout `split`
   (`:208-215`) and raises on any `train_keys & val_keys` overlap (`:229-238`). After
   the §3.4 rewire, the keys are `event_identity` tuples, so the guard is checking the
   *same stable identity* the holdout hash splits on — closing the loop.

---

## 4. Expected behavior

### 4.1 A single-stream batch dict shape

Pipeline: `… ApplyToStream(stream='sensor', [NormalizeCoord, GridSample, …]) →
ToTensor → Collect(stream='sensor', keys=("coord","energy","time","event_label",
"name"), feat_keys=("coord","energy","time"))`, batch size `B`, event `b` having
`n_b` points after GridSample.

Per-sample `Collect` output (b-th):
```
{ 'coord':       torch.float32 (n_b, 3),
  'energy':      torch.float32 (n_b, 1),
  'time':        torch.float32 (n_b, 1),
  'event_label': torch.int64   (n_b, 1)      # per-point broadcast (§3.5)
  'feat':        torch.float32 (n_b, 5)      # cat(coord, energy, time)
  'offset':      torch.int64   (1,) == [n_b]
  'name':        str           (passed through)
  'split':       str           (passed through) }
```

After `collate_fn` over the batch (`N = Σ n_b`):
```
{ 'coord':       (N, 3),
  'energy':      (N, 1),
  'time':        (N, 1),
  'event_label': (N, 1)   # concatenated per-point → sliceable by offset
  'feat':        (N, 5),
  'offset':      (B,)  == cumsum([n_0, n_1, …, n_{B-1}])   # diff/cumsum rebasing
  'name':        [str, …]  length B (str → python list, collate.py:25-27)
  'split':       [str, …]  length B }
```

`offset[b]` is the exclusive end index of event `b`'s points in the flat arrays;
event `b` spans `coord[offset[b-1]:offset[b]]` (with `offset[-1] == 0` implicitly).

### 4.2 `event_label` offset-slice recovery (probe)

In `_process_batch` (`lucid_event_probe.py:248-266`):
`offsets = [0] + point["offset"].tolist()` → `[0, n_0, n_0+n_1, …, N]`.
`_labels_by_event(event_label, offsets, N)` sees `event_label.numel() == N == n_points`
→ the `:120-124` branch: for each `(start, end)` window it takes
`event_label[start:end][0]` — the first point's label, which (per-point broadcast)
equals every point's label in that event → one label per event, in batch order.
`X` (mean-pooled `point.feat` per event, `:264`) and `y` (these per-event labels,
`:265`) line up index-for-index.

### 4.3 Probe disjointness via `event_identity`

`_validate_heldout_source` (`:197-246`): `val_split = _dataset_split(val_dataset)`
must be in `{holdout, val, test}` and not `{train, all}` (`:209-215`). Then
`train_keys = _event_keys(train_dataset)` = `{event_identity(i)}` over train indices;
`val_keys` likewise over val indices. Because both datasets use the *same blake2b
holdout hash* on `(config_id, source_event_idx)` (Part 02), the train and val identity
sets are **disjoint by construction**; `train_keys & val_keys` is empty (`:229`), the
guard logs `overlap=0` (`:240-246`), and the probe proceeds. If a config accidentally
overlaps (e.g. both splits set `split='all'`), the guard catches it and raises with up
to 5 example keys (`:230-238`).

---

## 5. Edge cases

1. **N-changing transforms before collate.** GridSample (train mode),
   RandomDropout, SphereCrop, ShufflePoint change point count. They must run inside
   `ApplyToStream(stream=S, …)` *before* `Collect(stream=S)`, and rely on
   `index_operator` (`transform.py:38-84`) carrying every per-point column (the
   `index_valid_keys` prefix-match, Part 01 §3.2) so `coord`/`energy`/`time`/labels
   stay aligned. `event_label`/`config_id` (per-point) must be in `index_valid_keys`
   (covered by the `event_broadcast` materialization + the prefix-match append) or
   they desync — then the offset-slice in §4.2 would read a stale label. **Per-event
   `target_*` must NOT be subset** — Part 01 excludes them via a leading-dim ≠
   `n_points` shape check; they travel as `_`-prefixed metadata and are dropped at
   collate (`collate.py:45-46`).

2. **Empty stream.** If a stream's sub-dict has zero points (e.g. an event with no
   `hits` after a cut), `Collect(stream=S)` produces `coord` of shape `(0, 3)` and
   `offset == [0]`. `collate_fn` concatenates fine (zero-row tensors cat cleanly); the
   batch offset simply does not advance for that event. The probe's `_process_batch`
   skips empty events (`if end <= start: continue`, `:262-263`), so a zero-point event
   contributes no embedding/label — no crash. `ApplyToStream` with `required=False`
   (default) makes a *missing* stream key a no-op (`detector_transforms.py:54-55`),
   distinct from a *present-but-empty* stream.

3. **`Subset` wrapping.** DDP/probe wrap datasets in `torch.utils.data.Subset`.
   `_event_keys` peels nested `Subset` layers and composes indices
   (`lucid_event_probe.py:166-172`), then maps through `event_identity`.
   `_dataset_split`/`_base_dataset` (`:134-148`) peel `Subset` to read the public
   `split`. The base guarantees `event_identity` + `split` survive `Subset` (Part 02,
   Seam 4). A `Subset` over a base dataset therefore yields a *subset* of the identity
   set, still disjoint from the train identity set.

4. **`_`-prefixed metadata drop.** Ragged per-event artifacts (the `bridges` join at
   `jaxtpc.py:257-259`, per-event `target_*`, raw `labl` tables) that must reach the
   sample but not be tensor-collated are carried `_`-prefixed and dropped at collate
   (`collate.py:45-46`). Note `data['bridges']` (`jaxtpc.py:259`) is **not**
   `_`-prefixed in the current dataset code — if such an artifact ever survives into a
   collected sample it would hit `default_collate` and likely raise on ragged shapes;
   the contract (D24) is that anything ragged carried past `Collect` must be
   `_`-prefixed. In single-stream this is moot because `Collect(stream=S)` only lifts
   the keys it is told to (`transform.py:128-129`) — `bridges`/`labl` live at the top
   level and are simply never collected. The drop matters once a transform copies a
   ragged value into the collected dict.

5. **`name`/`split` as strings, not tensors.** `collate_fn` routes `str` before
   `Sequence` (`collate.py:25-27`), so `name`/`split` survive as python lists of
   length `B`. Listing them in a `*_keys` feat list would crash (`.float()` on a str);
   they belong only in `keys` (or auto-passthrough, `transform.py:136-139`).

6. **`mix_prob > 0` interaction.** `point_collate_fn` mix branch renumbers `instance`
   and halves `offset` (`collate.py:58-73`). It is single-stream-safe but its `instance`
   renumbering assumes one cloud; under any future multi-stream path it must be disabled
   (§3.3, D23). The live LUCiD config uses `mix_prob=0` (`pretrain-…-mu-e.py:17`) so the
   branch is inert.

---

## 6. Tests

On `pimm_data/testing.py` synthetic fixtures (no GPU, no WAND). Fixture FK invariants
documented at `testing.py:23-24` (`make_jaxtpc_sample` `:55`, `make_lucid_sample`
`:333`). Numbered; each: setup / action / EXPECTED.

**T1 — Single-stream collate shape + offset.**
*Setup:* build 3 nested LUCiD `sensor` samples with point counts `(n0,n1,n2)`; run each
through `ApplyToStream(stream='sensor',[GridSample(mode='test'→ no, use a fixed
GridSample(mode='train') with a seed)]) → ToTensor → Collect(stream='sensor',
keys=("coord","energy"), feat_keys=("coord","energy"))`.
*Action:* `collate_fn([s0,s1,s2])`.
*EXPECTED:* `coord.shape == (n0+n1+n2, 3)`; `feat.shape == (·, 3)` (=cat coord+energy);
`offset.tolist() == [n0, n0+n1, n0+n1+n2]` (dtype int); `offset.numel() == 3`. Assert
`coord[offset[0]:offset[1]]` equals sample 1's `coord`.

**T2 — `event_label` per-point recovery by offset.**
*Setup:* 3 samples, per-point-broadcast `event_label` of values `[0,1,0]` (each as an
`(n_b,1)` array of the constant), collected with `event_label` in `keys`.
*Action:* `collate_fn(...)` then mimic `_labels_by_event(batch['event_label'],
[0,n0,n0+n1,n0+n1+n2], N)`.
*EXPECTED:* returns `tensor([0,1,0])` (one per event) via the `numel()==n_points`
branch; assert it equals slicing `event_label[offset_window][0]` for each event;
assert NOT the `numel()==n_events` branch (numel is `N`, not 3).

**T3 — Probe disjointness via `event_identity`.**
*Setup:* a stub dataset exposing `event_identity(i)->(config_id, src_evt_idx)`,
public `split`, and `__len__`, with a train instance (`split='train'`, identities
`{(0,a)…}`) and a val instance (`split='holdout'`, disjoint identities), each
optionally `Subset`-wrapped. Use the rewired `_event_keys`/`_validate_heldout_source`.
*Action:* call `_validate_heldout_source` (rank-0 path) with train/val loaders over
those datasets.
*EXPECTED:* no raise; `train_keys & val_keys == set()`. Then construct an
*overlapping* pair (shared identity) → EXPECTED `RuntimeError` mentioning leakage with
the formatted key `"config_id:src_evt_idx"` (multi-component `_format_event_key`).
Also: `Subset` wrapping yields a subset of identities, still disjoint. Also: a
`split='all'` val → `RuntimeError` from the forbidden-split check (`:208-215`).

**T4 — `_event_keys` no longer reaches dataset internals.**
*Setup:* a dataset exposing `event_identity` + `split` but **deliberately lacking**
`data_list`/`datasets` attributes.
*Action:* `_event_keys(dataset)`.
*EXPECTED:* returns the identity set (not `None`) — proving the rewire reads
`event_identity`, not `data_list`/`datasets`. Conversely a dataset lacking
`event_identity` → returns `None` (graceful-None contract preserved).

**T5 — train≡eval transform-list assertion.**
*Setup:* parse the live config's `base_event_transform`, `transform`, `val_transform`
(`pretrain-…-mu-e.py:149-263`); define the "deterministic subset" as the
non-augmenting ops (NormalizeCoord, Update, GridSample, LogTransform,
RelativeLogNormalize).
*Action:* compare the registered transform `type`+params of `val_transform`'s
deterministic ops against the same slice of `transform`.
*EXPECTED:* identical type+param dicts for the shared fragment (both come from
`base_event_transform`); the only val-vs-train difference is the appended tail
(`ToTensor`/`Collect` for val; MultiView+jitter for train). Assert no drifted
`grid_size`/`scale`/`max_val` between them.

**T6 — Seam test: nested output + per-event decoration preserved (multi-stream stays
additive).**
*Setup:* build a JAXTPC fixture with ≥2 modalities (`modalities=('step','hits','labl')`,
`make_jaxtpc_sample`); call `JAXTPCDataset.get_data(0)`.
*Action:* inspect the returned dict.
*EXPECTED:* it is **nested** — `set(keys) ⊇ {'step','hits'}`, each a sub-dict with its
own `coord`; **no bare top-level `coord`/`segment`** (Seam 1: never flatten). Each
stream's labels are present and self-contained (`step['segment']`, `hits['instance']`,
`hits['segment']` when labl present, Seam 2). Then `Collect(stream='step', keys=…)`
lifts only the `step` stream → flat dict; a *second* `Collect(stream='hits', …)` on a
deepcopy lifts `hits` independently → proves a future per-stream collate loop is a pure
addition (no change to dataset/decoration). Assert cross-stream artifacts
(`bridges`/`labl`) are reachable at top level and are NOT in the single-stream
collected output (never collected).

**T7 — `index_operator` keeps `event_label` aligned through N-change.**
*Setup:* a stream dict with `coord`, `event_label` (per-point broadcast `(N,1)`), and
an `index_valid_keys` including `event_label`; apply a deterministic N-reducing
GridSample.
*EXPECTED:* `event_label` is subset to the new N and still constant within the event
(so the §4.2 offset-slice still returns the right label). A per-event `target_*` of
shape `(D,)` (D≠N) is **not** subset.

**T8 — `_`-prefixed metadata drop at collate.**
*Setup:* per-sample dicts each carrying a `_ragged` key (variable-length per sample)
alongside `coord`/`offset`.
*Action:* `collate_fn(batch)`.
*EXPECTED:* `_ragged` absent from the output dict (`collate.py:45-46`); `coord`/`offset`
present and correct; no `default_collate` ragged-shape error.

**T9 — collate byte-identity guard.**
*Setup:* import `pimm_data.collate` and pimm `pimm.datasets.utils`.
*Action:* `diff`/source compare of `collate_fn`/`point_collate_fn`/`inseg_collate_fn`.
*EXPECTED:* byte-identical source (regression guard so the REPLACE never silently
drifts; mirrors the verified `diff` in §2.1).

---

## 7. Reversible defaults & risks

**Reversible (D34 — decided at code time, recorded in code, not litigated):**

- Collate-fn **decomposition** for the future multi-stream path (a new
  `multistream_collate_fn` vs a `mode=` flag on `collate_fn`). Default: a separate
  function, single-stream `collate_fn` untouched.
- Mix-up **disabled under multi-stream** (the `point_collate_fn` mix branch); default
  off whenever >1 stream is collated.
- `_format_event_key` rendering (`":".join`) — cosmetic; only affects log/error text.
- Whether the per-run repro record is a sidecar JSON vs embedded in the config snapshot
  — default: alongside `config.py` in the `train.sh` snapshot (§5 of impl plan).

**Risks:**

- **`event_label` materialization timing.** The probe's offset-slice recovery (§4.2)
  depends on the base emitting `event_label` **per-point** (Part 02). Until that lands,
  the probe runs on the legacy length-1 form via the `numel()==n_events` fallback
  (`lucid_event_probe.py:118-119`) — keep that branch so the rewire is safe to land
  before/after the base change.
- **`event_identity` availability.** The §3.4 rewire hard-requires
  `dataset.event_identity` (Part 02). Landing the probe rewire before the base would
  make `_event_keys` return `None` → the guard raises "cannot verify disjointness"
  (`:221-227`). Sequence: land Part 02 base first, or keep the `None`→raise contract so
  the failure is loud, not silent.
- **`index_valid_keys` desync of per-point labels.** If `event_label`/`config_id` are
  not in `index_valid_keys` when an N-changing transform runs, the offset-slice reads a
  stale label (silent wrong-label, not a crash). Mitigated by the Part 01 prefix-match
  append + T7.
- **Ragged-key collate crash.** Any ragged value that reaches the collected dict
  *without* a `_` prefix hits `default_collate` and raises. Contract: `_`-prefix all
  ragged per-event metadata (D24); T8 guards the drop.
- **`mix_prob`.** Inert at `0` in the live config; if turned on under any future
  multi-stream path it corrupts `instance`/`offset` (§5.6) — gate it off there.

---

## 8. Dependencies on other parts

- **Part 02 (`MultiModalEventDataset` base) — hard dependency.** Provides
  `event_identity(idx) -> (config_id, source_event_idx)`, public `split` (survives
  `Subset`), `data_list` as `(source_idx, local_idx)` tuples, the `datasets`-equivalent
  source list, and the **per-point `event_label`/`config_id` (`event_broadcast`)
  materialization** that replaces `lucid_event_ssl.py:310-311`. The §3.4 eval rewire and
  the §4.2 label recovery consume these directly. The holdout hash there is what makes
  the §4.3 disjointness guard correct.
- **Part 04 (label decoration / `label_config`) — dependency.** Defines
  `event_label`/`config_id` as `scope="event_broadcast"` (per-point) and the per-event
  `target_*` as `_`-prefixed metadata. §3.2's `Collect(keys=[…,'event_label'])`,
  §4.2's offset-slice, and §5.1's per-event-target exclusion all rely on this.
- **Part 01 (transforms / `index_operator` prefix-match) — dependency.** The
  `index_valid_keys` prefix-match (and per-event-target shape exclusion) is what keeps
  per-point `event_label` aligned through N-changing transforms inside `ApplyToStream`
  (§5.1, T7).
- **Packaging / shim (§5 of impl plan) — dependency.** The pimm `__init__.py`
  re-export shim must keep exporting `collate_fn`/`point_collate_fn`/`inseg_collate_fn`
  + dataset classes so `train.py:331/354/449/474` resolve from pimm-data unchanged; the
  `train.sh` code-snapshot (+ submodule SHA) is the substrate for the §3.5 per-run repro
  record.
- **FUTURE (§8 of impl plan, D23/D24) — provided-for, not depended-on.** The
  multi-stream namespaced collate + model-side primary/aux adapter build on the four
  seams (§3.3); this part guarantees those seams hold but does not build the future
  path.
