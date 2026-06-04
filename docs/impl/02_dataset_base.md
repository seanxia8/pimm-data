# Part 02 — MultiModalEventDataset base (implementation spec)

**Status:** final spec before coding. Implementation-ready.

> **⚠ Cross-reference note — this doc's in-text "Part NN" labels predate the final
> filenames.** Match cross-references by **title**, not number. Canonical map for
> *this* doc: "Part 03" (collate) → `05_collate_streams_eval.md`; "Part 04"
> (readers / `read_meta`) → `03_readers.md`; "Part 05" (label decoration) →
> `04_label_decoration.md`; "Part 06" (probe/eval rewire) → `05_collate_streams_eval.md`
> (§eval). `00_index.md` §2/§5 is authoritative.

**Source decisions:** D6, D7, D8, D9 (placement); D26 (holdout specifics), D27
(reader surfacing + manifest cache), D30 (TestModeMixin), D34 (reversible
defaults), D36 (event unit), D37 (common vs different), D40 (JAXTPC in scope);
**D42 (joint index — the base must intersect per-modality present-key sets, NOT
inherit `_n_events=min(...)`), D43 (Phase A lands first; the base factors A2 up),
D44 (intercept `min_deposits`/`min_segments` at the base; volume-aware min-points
does NOT affect holdout/identity), D46 (`_read_shard_meta` lru_cache under the
manifest build; readers surface per-modality present-key sets), D47 (A4
length-mismatch warn + `strict_lengths`)**. Plan §3.3
(`implementation_plan_pimm_data_datalayer.md`), §3.4, §6, §7, §9.2; the
cross-modality desync bug + Phase A A1–A5 in `shard_event_filtering_handoff.md`
§4/§5.

**Files:**
- NEW `src/pimm_data/multimodal.py` — `MultiModalEventDataset`, `TestModeMixin`.
- EDIT `src/pimm_data/defaults.py` — extract `TestModeMixin` out of
  `DefaultDataset` (currently lines 60–66 init + 150–181 `prepare_test_data`).
- EDIT `src/pimm_data/lucid.py` / `src/pimm_data/jaxtpc.py` — re-parent onto the
  base; delete the 3 duplicated `prepare_test_data` copies
  (`lucid.py:358–390`, `jaxtpc.py:508–540`, `defaults.py:150–181`).
- EDIT `src/pimm_data/readers/*` — add `read_meta(idx)` (Part 04 spec; the base
  *consumes* it — interface frozen here).
- EDIT `src/pimm_data/__init__.py` — export `MultiModalEventDataset`,
  `TestModeMixin`.
- TESTS `tests/` driven by `src/pimm_data/testing.py` fixtures.

This spec covers the **selection layer only**. Label decoration (`label_config`)
is Part 05; reader `read_meta`/surfacing is Part 04; collate is Part 03. Where
this base calls into those, the *interface* is frozen here and the
*implementation* is owned there.

---

## 1. Purpose & scope

`MultiModalEventDataset` is the single owner of **which events exist and which
split they belong to**. It sits between the per-modality readers (Part 04) and
the per-modality builders (the subclass `get_data`). Its responsibilities (D37
COMMON):

1. **Multi-source mixture** — `sources=[{root,label,weight}]`; assign a stable
   integer `config_id` per source; concatenate into one flat index (D9).
2. **Deterministic holdout** — seeded blake2b on `(config_id,
   source_event_idx)`, 3-way, config-stratified by folding `config_id` into the
   digest (D7/D26). Rank- and machine-identical.
3. **Cheap min-points filter** — `n_hits`-attr threshold, inclusive `>=`,
   *filter-then-hash-split* (D8). No array reads in steady state.
4. **Joint event index** (D42) — intersect the `event_*` keys **actually present**
   across the loaded modalities into one canonical per-source map (`source_event_idx`
   is the join key); `_read_event`/`read_meta` translate `local_idx` per modality
   against it. This is the prerequisite that stops the base inheriting the
   cross-modality desync (`shard_event_filtering_handoff.md` §4); the size replaces
   the per-reader `min(...)` and the manifest/identity are built over the intersected
   index. See §3.3a.
5. **Manifest cache** — persist the scanned `(config_id, source_event_idx,
   n_hits)` triples per source so steady-state startup never reopens event
   groups; rank-0 builds under a DDP barrier (D27).
6. **Identity / split API** — `event_identity(idx) -> (config_id,
   source_event_idx)`, public `self.split`, `data_list` as `(source_idx,
   local_idx)` tuples, a `datasets`-equivalent source list — exactly the shape
   the leakage guard in `lucid_event_probe.py` reads.
7. **`get_data` dispatch** — resolve `idx -> (source_idx, local_idx)`, route to
   the subclass readers, materialize `event_label`/`config_id` (per-point
   broadcast for the probe), prefix `get_data_name` with the source (collision
   fix).

What it does **NOT** own (D37 DIFFERENT, stays in `LUCiDDataset`/`JAXTPCDataset`):
readers (PMT vs wire/pixel), geometry resolution, per-modality builders
(`_build_sensor`/`_build_step`/…), FK chains and label decoration
(`particle_idx→category` vs `group_id→track→label`), and detector sub-selectors.
**JAXTPC `volume` is an orthogonal sub-selector applied inside `_build_readers`,
NOT the mixture/holdout axis** (D37/D40). One stored readout (`event_NNN`) = one
sample; `volume` never subdivides the holdout (D36/§9.2).

**Three distinct selection axes — do NOT collapse them (D45).** (1) `sources=`
is the **mixture** axis (D9): a list of physically-distinct populations, each
config-stratified with its own `config_id`/`label`, holdout-bucketed
independently. (2) **multi-run / shard-union** is *one* logical population spread
over several run directories of the **same physics** carrying **one** label, with
the run treated as part of identity — NOT four `sources`. (3) **shard-tag** is an
orthogonal *within-run* sub-selector (like `volume`), not a population at all.
doraemon's four `run_*` directories are multi-run, **not** four sources. The
multi-run + shard-tag sub-selectors are a **Phase-B** addition layered on the joint
index (`shard_event_filtering_handoff.md` B1/B2); this part builds the mixture axis
and the joint index only.

Out of scope for this part: namespaced multi-stream collate (FUTURE, D23/D39),
label decoration internals (Part 05), `read_meta` reader implementation
(Part 04).

---

## 2. Current state (file:line)

**`DefaultDataset` (`src/pimm_data/defaults.py`)** — the base both detector
datasets inherit.
- `__init__(split, data_root, transform, test_mode, test_cfg, cache,
  ignore_index, loop)` at `defaults.py:37–72`.
- The test-mode block to factor out: `defaults.py:60–66` (builds
  `test_voxelize`/`test_crop`/`post_transform`/`aug_transform`).
- `data_list = self.get_data_list()` at `defaults.py:68`; the npy
  `get_data_list` at `defaults.py:74–91`.
- `get_data_name` = `os.path.basename(self.data_list[idx % len])` at
  `defaults.py:136–137` — the collision source (basename only).
- `prepare_test_data` (the fragment/`inverse` TTA path) at `defaults.py:150–181`
  — this is the fragment the seg eval needs (D30).
- `__getitem__` dispatch at `defaults.py:183–187`; `__len__ = len(data_list) *
  loop` at `defaults.py:189–190`.

**`ConcatDataset` (`defaults.py:194–226`)** — the existing `(dataset_idx,
data_idx)` tuple-index pattern (`data_list` zip at `defaults.py:203–212`,
`get_data` at `defaults.py:214–216`). The base's `data_list` shape mirrors this
(`(source_idx, local_idx)`) but selection is internal, not per-source `Subset`.

**`LUCiDDataset` (`src/pimm_data/lucid.py`)**:
- `__init__` at `lucid.py:89–155`; builds 4 readers (`lucid.py:121–142`),
  computes `self._n_events = min(len(r) for r in active_readers)`
  (`lucid.py:149`), then calls `super().__init__` (`lucid.py:151–155`).
- `get_data_list` = `list(range(n))` with `max_len` cap (`lucid.py:182–187`).
- `get_data` (`lucid.py:189–214`) routes to `_build_sensor`/`_build_hits`/
  `_build_step`/`_build_labl`.
- `get_data_name` (`lucid.py:348–356`): searches `_canonical_reader`'s
  `cumulative_lengths` → `f"{fname}_evt{event_num:03d}"` — **no source prefix**
  (collision across configs sharing filenames, e.g. `wc_sensor_0000.h5_evt007`).
- `prepare_test_data` (`lucid.py:358–390`) — near-duplicate of `defaults.py`'s.
- `__del__` closes readers (`lucid.py:392–400`).

**`JAXTPCDataset` (`src/pimm_data/jaxtpc.py`)**:
- `__init__` at `jaxtpc.py:95–195`; the **`volume` sub-selector** lives here:
  it auto-detects `readout_type` (`jaxtpc.py:156–162`) then rewrites
  `sensor_reader.planes`/`hits_reader.planes` to the volume's plane labels
  (`jaxtpc.py:164–173`), and drops non-matching labl volumes in `get_data`
  (`jaxtpc.py:245–250`). This is the orthogonal selector to preserve as-is.
- Empty-data guard raising `ValueError` at `jaxtpc.py:188–195`.
- `get_data_name` (`jaxtpc.py:499–506`) — same no-prefix collision shape as LUCiD.
- `prepare_test_data` (`jaxtpc.py:508–540`) — third near-duplicate copy.

**Readers** expose `cumulative_lengths` (np.cumsum), `indices` (per-file
event-number arrays), `h5_files`, `read_event(idx)`, `__len__`, `close()` — e.g.
`lucid_sensor.py:80–142`, `jaxtpc_sensor.py:71–256`. Index is built from event
groups *actually present* (gap-tolerant; `jaxtpc_sensor.py:82–84`,
`jaxtpc_step.py:107–109`). **No reader surfaces `source_event_idx` or `n_hits`
today** (grep-confirmed); `read_meta` is added in Part 04.

**The probe leakage guard (`pimm/engines/hooks/lucid_event_probe.py`)** — the
contract this base must satisfy:
- `_dataset_split` (`lucid_event_probe.py:141–148`): peels `Subset`
  (`_base_dataset`, lines 135–138), reads `getattr(dataset, "split", None)`,
  returns str or tuple. **`self.split` must survive `Subset` wrapping.**
- `_validate_heldout_source` (`lucid_event_probe.py:197–246`): rejects val
  splits in `{"train","all"}`, requires them ⊆ `{"holdout","val","test"}`.
- `_event_keys` (`lucid_event_probe.py:161–190`): peels nested `Subset`
  (lines 166–172), reads `data_list = getattr(dataset,"data_list")` and
  `sources = getattr(dataset,"datasets")` (lines 174–176); **returns None if
  either is missing** → guard then raises "cannot verify disjointness". For each
  index it unpacks `source_idx, event_idx = data_list[i % base_len]`
  (line 187), maps to a `_source_key` (lines 151–158: `source.get("source_root")`
  or `source.get("data_root")` → realpath, else `source["name"]`, else
  `str(source_idx)`), and forms `(source_key, int(event_idx))`. **`data_list`
  must be a list of `(source_idx, local_idx)` int tuples and `datasets` a list
  of dicts each with `source_root`/`data_root`/`name`.**
- The colleague's reference `lucid_event_ssl.py` already produces exactly this
  shape (`data_list` at `lucid_event_ssl.py:251`, `datasets` at
  `lucid_event_ssl.py:154–178`, `event_label`/`config_id` per-event at
  `lucid_event_ssl.py:310–311`) — but with **per-source
  `np.random.permutation` holdout** (`lucid_event_ssl.py:202–205`) that is
  **not rank/version-stable** and a min-points filter that uses `>` not `>=`
  and reads full arrays (`lucid_event_ssl.py:254–259`). The base **replaces**
  both.

---

## 3. Target design

### 3.0 Class topology (D30)

```
DefaultDataset (existing, defaults.py) ── inherits ──┐
                                                     │ (TestModeMixin extracted from it)
TestModeMixin (new, defaults.py)  ───────────────────┤
                                                     │
MultiModalEventDataset(TestModeMixin, Dataset) ──────┘   ← new, multimodal.py
        ▲                          ▲
        │                          │
   LUCiDDataset             JAXTPCDataset    ← re-parented onto the base
```

`MultiModalEventDataset` does **not** inherit `DefaultDataset` directly — it
inherits a factored **`TestModeMixin`** (D30) plus `torch.utils.data.Dataset`.
`DefaultDataset` keeps its npy `get_data_list`/`get_data` (the PILArNet/npy path
is unaffected) and *also* mixes in `TestModeMixin` so its `prepare_test_data`
keeps working byte-identically. The mixin is the single home for the
fragment/`inverse` TTA path; the 3 duplicated `prepare_test_data` copies
(`defaults.py:150–181`, `lucid.py:358–390`, `jaxtpc.py:508–540`) are deleted.

**`TestModeMixin` (extracted, frozen surface):**

```python
class TestModeMixin:
    """Owns the test-time fragment/inverse TTA path (D30).

    A host class must set: self.transform (Compose), self.test_mode (bool),
    self.test_cfg, and provide self.get_data(idx). build_test_transforms()
    must be called from the host __init__ when test_mode is True.
    """

    def build_test_transforms(self):
        # Factored verbatim from DefaultDataset.__init__:60-66.
        self.test_voxelize = TRANSFORMS.build(self.test_cfg.voxelize)
        self.test_crop = (TRANSFORMS.build(self.test_cfg.crop)
                          if self.test_cfg.crop else None)
        self.post_transform = Compose(self.test_cfg.post_transform)
        self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

    def prepare_test_data(self, idx):
        # The nested-aware variant from lucid.py:358-390 / jaxtpc.py:508-540:
        # pop "name", conditionally pop "segment"/"origin_segment"/"inverse",
        # then aug_transform -> test_voxelize -> test_crop -> post_transform
        # into result_dict["fragment_list"]. (NOT defaults.py's variant, which
        # unconditionally pops "segment" — nested datasets rely on a terminal
        # Collect to lift segment, so the pop is guarded by `if "segment" in".)
        ...
```

Rationale for choosing the nested-aware (`lucid`/`jaxtpc`) body over
`defaults.py`'s: `defaults.py:154` unconditionally does
`segment=data_dict.pop("segment")`, which only works because npy `get_data`
always synthesizes a `segment` key (`defaults.py:121–126`). The nested datasets
produce `segment` only via a terminal `Collect`; their copies guard with
`if "segment" in data_dict` (`lucid.py:367`, `jaxtpc.py:517`). The mixin uses
the guarded form, which is a strict superset (npy path always has `segment`, so
the guard is always true → byte-identical for `DefaultDataset`).

### 3.1 Frozen `__init__` signature

```python
@DATASETS.register_module()
class MultiModalEventDataset(TestModeMixin, Dataset):
    def __init__(
        self,
        sources,                 # str | list[str | dict{root, label?, config_id?, split?, weight?}]
        modalities=('sensor',),
        *,
        split='train',           # holdout role: 'train' | 'val' | 'test' | 'all'
        holdout=None,            # None | {seed:int, fractions:(tr,va,te), strata:'config'}
                                 #      | {seed:int, n_per_config:int}
        min_points=None,         # None | int | {threshold:int, modality:str='sensor', op:str='>='}
        strict_lengths=False,    # D47/A4: True => hard-error on a per-modality present-key
                                 #   mismatch instead of intersecting + warning (§3.3a)
        max_events=-1,           # cap AFTER holdout+min-points, per whole-dataset (-1 = no cap)
        mixture=None,            # None | {weights:'uniform'|list[float], mode:'replicate'|'sampler'}
        label_config=None,       # Part 05; carried through, not interpreted here
        dataset_name='wc',
        transform=None,
        test_mode=False,
        test_cfg=None,
        loop=1,
        ignore_index=-1,
        cache=False,
        data_root=None,          # back-compat alias: data_root=X  ->  sources=[X]
        **reader_kwargs,         # forwarded verbatim to _build_readers (e.g. volume=, label_key=,
                                 #   min_segments=, pe_threshold=, include_physics=, pmt_positions=)
    ):
```

Notes on the signature (frozen):
- **`*` after `modalities`** forces every selection knob to be keyword-only.
  `sources` and `modalities` are the only positionals. This prevents a config
  from accidentally passing `holdout` positionally and prevents future arg
  insertion from shifting meaning.
- **`data_root=` back-compat alias** (plan §3.3): if `data_root is not None` and
  `sources is None`/unset, normalize to `sources=[data_root]`. If both are given,
  raise `ValueError`. Every existing single-root config (`LUCiDDataset(data_root=
  ...)`, `JAXTPCDataset(data_root=...)`) keeps working unchanged because the
  subclasses re-expose `data_root` as their first positional and forward it.
- **`reader_kwargs`** is the seam for DIFFERENT (D37) subclass knobs. The base
  never inspects them — **except `min_deposits`/`min_segments`, which it intercepts
  (D44)**: these are routed through the dataset-level min-points path on the joint
  index (§3.6), and the step/lucid-step reader's internal index mask is no-op'd so
  it can no longer build a non-contiguous per-reader index that desyncs from the
  other modalities (handoff §4 #1). `volume`, `label_key`, `include_physics`,
  `pe_threshold`, `pmt_positions`, `pmt_positions_file`, `label_keys` flow through
  unmodified. Passing `min_deposits>0`/`min_segments>0` **without** the source
  modality loaded raises (D44/A3; see §5).
- **`strict_lengths`** (D47/A4): default `False` → on a per-modality present-key
  count mismatch the base intersects and emits one `log.warning` with the concrete
  per-modality counts (never the old silent `min(...)`); `True` → raise. Distinct
  from the **manifest-as-INPUT** include/exclude contract, which is **Phase B,
  deferred** (`shard_event_filtering_handoff.md` B4; the snapshot half is superseded
  by D41/§3.7) — this part adds only the warn + strict flag.
- `loop`, `ignore_index`, `cache`, `transform`, `test_mode`, `test_cfg` are the
  standard `DefaultDataset` knobs (`defaults.py:37–47`), set on `self` directly
  (the base does not call `DefaultDataset.__init__`; it owns its own
  `data_list`).

### 3.2 `__init__` order of operations (D8: filter-then-hash-split)

```python
def __init__(self, ...):
    super().__init__()  # torch Dataset

    # (1) Normalize sources -> list[ResolvedSource]; assign config_id.
    self._sources = self._normalize_sources(sources, data_root, dataset_name, split)

    # (2) Standard tail state (set before reader build so __del__ is safe).
    self.split = self._normalize_split(split)   # public; survives Subset (probe)
    self.transform = Compose(transform)
    self.test_mode = test_mode
    self.test_cfg = test_cfg if test_mode else None
    self.loop = loop if not test_mode else 1     # mirror defaults.py:54-56
    self.ignore_index = ignore_index
    self.cache = cache
    self._modalities = tuple(modalities)
    self._label_config = label_config
    self._reader_kwargs = reader_kwargs
    self._holdout_cfg = self._normalize_holdout(holdout)
    self._min_points_cfg = self._normalize_min_points(min_points)

    # (3) Per-source readers (SUBCLASS HOOK). Validates modalities, builds the
    #     reader set, applies orthogonal sub-selectors (JAXTPC volume).
    for src in self._sources:
        src.readers = self._build_readers(src.root, self.split, **reader_kwargs)

    # (3a) JOINT EVENT INDEX (D42).  Intersect the present event_* keys across the
    #      loaded modalities into ONE canonical per-source map keyed on
    #      source_event_idx; this is the source of truth every reader is then
    #      translated against. Replaces the per-reader `_n_events=min(...)` that
    #      silently desynced modalities (handoff §4). Also emits the A4
    #      length-mismatch warning / honors strict_lengths (§3.3a, D47).
    for src in self._sources:
        src.joint = self._build_joint_index(src.readers)    # JointIndex (§3.3a)
        src.n_events = src.joint.n_events                    # == |intersection|, NOT min()

    # (4) Manifest cache: load-or-build the (source_event_idx, n_hits) table
    #     per source, built OVER THE JOINT INDEX. Rank-0 build under DDP barrier;
    #     never array reads in steady state. (§3.4)
    self._manifests = self._load_or_build_manifests()       # {config_id: Manifest}

    # (5) min-points filter (cheap n_hits, inclusive >=).  FILTER FIRST.
    #     min_deposits/min_segments are intercepted here too, on the joint index
    #     (D44) — the step/lucid-step reader internal mask is no-op'd so it can't
    #     desync.
    candidates = self._apply_min_points(self._manifests)    # per source: kept local_idx[]

    # (6) hash holdout: assign each surviving (config_id, source_event_idx) a
    #     bucket; keep those whose bucket matches self.split. THEN split.
    candidates = self._apply_holdout(candidates)

    # (7) max_events cap (deterministic head of the post-holdout order).
    candidates = self._apply_max_events(candidates, max_events)

    # (8) mixture: build the flat index, replicating by integer weight (default).
    self.data_list = self._build_index(candidates, mixture)   # [(source_idx, local_idx), ...]
    self.datasets  = [src.descriptor for src in self._sources] # probe `datasets`

    # (9) test-mode transforms + empty guard + log.
    if self.test_mode:
        self.build_test_transforms()
    if len(self.data_list) == 0:
        raise ValueError(...)  # see §5
    log.info("Totally %d x %d samples across %d sources, split=%s",
             len(self.data_list), self.loop, len(self._sources), self.split)
```

**Order is load-bearing** and matches plan §3.3 ("normalize sources →
`_build_readers` → build/load manifest cache → min-points filter (`>=`) → hash
holdout → `max_events` → mixture → standard tail") and D8 ("filter-then-hash-
split"), **with the joint-index step (3a) inserted between `_build_readers` and the
manifest build** (D42): the manifest, min-points filter, holdout, identity and
`data_list` are ALL built over the intersected index, so `local_idx` means the same
physics event in every modality. Filtering before splitting means the *kept* set is what gets
bucketed, so every event that survives min-points lands in exactly one split and
the three splits partition the filtered population (not the raw one). This is the
property the probe's disjointness guard relies on.

### 3.3 `_normalize_sources` and the `ResolvedSource` record

A source is one mixture element (D9: the "config"/"run" axis). Accepted forms:
- `"path"` → one source, `root="path"`, `config_id=0`, `label=0`.
- `["pathA", "pathB"]` → two sources, `config_id` = list position.
- `[{"root":..., "label":..., "config_id":..., "split":..., "weight":...}, ...]`
  — explicit. `config_id` defaults to list position when omitted (explicit
  values must be unique; raise on collision). `label` defaults to `config_id`
  (D9 "explicit labels, replicate default"). `weight` defaults to `1`. `split`
  (per-source reader-discovery subdir) defaults to the top-level `split` arg
  passed verbatim to the reader's file glob — **this is NOT the holdout split**;
  it is the reader's `split=` subdirectory knob (`lucid_sensor.py:69–78`). The
  holdout role lives in `self.split`.

```python
@dataclass
class ResolvedSource:
    config_id: int           # stable, folded into the holdout digest (D26 stratification)
    label: int               # event_label materialized per-point (D9, probe)
    root: str                # source data_root (realpath kept for probe _source_key)
    reader_split: str        # forwarded to readers' file discovery (NOT holdout)
    weight: float
    readers: dict = None     # filled in step (3): {'sensor':..., 'step':..., ...}
    joint: object = None     # filled in step (3a): JointIndex over intersected keys (§3.3a)
    n_events: int = 0        # filled in step (3a): == joint.n_events (NOT min over readers)
    @property
    def descriptor(self) -> dict:
        # The probe reads source_root/data_root/name off this (lucid_event_probe.py:151-158)
        return {"source_root": os.path.realpath(self.root),
                "data_root": self.root,
                "name": os.path.basename(os.path.normpath(self.root)),
                "config_id": self.config_id, "label": self.label}
```

`config_id` is the mixture/stratification axis (D37). It is **assigned by the
base, never a reader field** (plan §3.4: "`config_id` is not a reader field").

### 3.3a Joint event index (D42 — the desync fix)

**This is the step that stops the base inheriting the cross-modality desync.** As
specified before this section was added, the base re-implemented the exact
`jaxtpc.py:180` hazard: `_source_n_events = min(len(r) for r in active_readers)`
and one `local_idx` handed to every reader (`get_data`, `jaxtpc.py:233-269`). With
no joint index that is only correct if every modality's gap-tolerant present-key
list is the *same* contiguous list of physics events — which the per-reader
present-key indexing from `0757ee0` does **not** guarantee. Two failure modes
(handoff §4): (1) `min_deposits>0` masks step to a non-contiguous subset while
sensor/hits/labl keep all events → `local_idx` k addresses different physics events
per modality; (2) a gap in some-but-not-all modalities misaligns them. Single-stream
(D35) dodges it only for a *single-modality unlabeled* run; **label decoration
(Part 05) re-exposes it for every labeled task** (the stream reader and the labl
reader are joined at the same `local_idx`). Holdout/identity (D26/D40) silently
depend on an alignment that was never enforced.

**Fix (normative):** build one canonical index per source by **intersecting the
`event_*` keys actually present across the loaded modalities**, keyed on the
writer-stamped `source_event_idx` (the join key — already surfaced by `read_meta`,
Part 04). `source_event_idx` is what makes the join meaningful: two modalities'
local rows refer to the same physics event iff they carry the same
`source_event_idx`, regardless of gaps or reorder. The dense `local_idx` is then a
per-modality *translation* off this canonical map, never a shared assumption.

```python
@dataclass
class JointIndex:
    # one per source; built in __init__ step (3a), consumed by manifest/min-points/
    # holdout/identity/_read_event. Canonical order = sorted by source_event_idx.
    source_event_idx: np.ndarray            # (n_events,) int64 — the intersection, sorted
    # per modality: canonical position -> that modality's dense local_idx (what its
    # read_event/read_meta take). e.g. local_for['step'][k] is the step reader row
    # for canonical event k.
    local_for: dict                          # {modality: np.ndarray (n_events,) int64}
    has_source_event_idx: bool               # False on any source => D26 positional fallback
    @property
    def n_events(self): return len(self.source_event_idx)

def _build_joint_index(self, readers) -> JointIndex:
    """Intersect present event_* keys across loaded modalities (D42).

    Per modality, read its present (source_event_idx -> local_idx) map cheaply
    from read_meta / the reader's present-key surface (Part 04 / D46). Intersect
    the source_event_idx sets; canonical order = sorted(intersection). For each
    modality record local_for[mod][k] = that modality's local_idx for canonical
    event k. Emit the A4 warning (or raise under strict_lengths) when the present
    sets differ. When source_event_idx is absent on a source (read_meta -> None),
    fall back to POSITIONAL join: intersect on dense local_idx and set
    has_source_event_idx=False (the D26 positional-fallback warning fires once,
    §3.5); modalities are then assumed positionally aligned (the pre-D42 behavior,
    now explicit and warned rather than silent).
    """
    ...
```

- **Source of the per-modality present sets (D46).** Readers surface their present
  `event_*` keys (and `source_event_idx` per key) via `read_meta` / the
  `_has_sei_vec`/`_sei_vecs` vector fast path (Part 04 §3.0). The cross-reader file
  opens this scan triggers are deduped by a module-level `@lru_cache`
  `_read_shard_meta(path) -> (n_events, n_volumes, present_event_keys, readout_type)`
  (handoff A1) **adopted as an impl detail under the manifest-cache build** (§3.4) —
  complementary to D27, not competing. So the joint-index scan and the manifest scan
  share the same one-time, deduped per-shard reads.
- **Size replaces `min(...)`.** `src.n_events = src.joint.n_events ==
  |intersection|`. The old `_source_n_events`/`min(...)` is deleted as the index
  size; it survives only as the input the A4 mismatch warning compares against.
- **A4 length-mismatch (D47).** If the modalities' present-key counts differ, warn
  with the concrete per-modality counts (e.g. `step=200 hits=199 -> joint=199`)
  instead of the silent drop; `strict_lengths=True` raises instead. The intersected
  size is authoritative either way.
- **Everything downstream is built over the joint index.** The manifest (§3.4/§3.6),
  min-points (§3.6), holdout (§3.5), `event_identity` (§3.7), `data_list` and
  `get_data_name` (§3.8) index canonical positions; `_read_event` (§3.8) translates
  each canonical `local_idx` to the per-modality dense row via
  `src.joint.local_for[modality]` before calling that reader's `read_event`. This is
  the single point that guarantees Invariant 9 (§4) — same `source_event_idx` across
  every served modality for every served idx.
- **`volume` interaction (D44).** The joint index is built over whole `event_NNN`
  readouts; `volume` is an orthogonal sub-selector applied *inside* `_build_readers`
  and does not change which events intersect, preserving §9.2 / Invariant 7.

**Phase A vs the base (D43).** This joint-index logic lands FIRST as a standalone
bug-fix PR patching the current `src/pimm_data/jaxtpc.py` (Phase A: A1
meta-cache, A2 joint index, A3 volume-aware + raise, A4 length-mismatch, A5
regression test) — the file the de-fork KEEPS. **The base then factors A2 up** into
`_build_joint_index` here; it is not throwaway work and is not absorbed by Step 1.
See Part 06 §4 (a Phase A step before Step 0) and `00_index.md` §4.

### 3.4 `_build_readers` subclass hook (the DIFFERENT seam, D37)

```python
def _build_readers(self, source_root, split, **reader_kwargs) -> dict:
    """SUBCLASS HOOK. Build and return the per-modality reader set for one source.

    Returns {modality: reader} for modalities in self._modalities. Each reader
    MUST expose: __len__, read_event(local_idx), read_meta(local_idx)
    (Part 04), cumulative_lengths, indices, h5_files, close().

    Subclasses:
      - validate self._modalities (the existing _validate_modalities),
      - construct readers per modality (PMT vs wire/pixel),
      - apply ORTHOGONAL sub-selectors here (JAXTPC `volume`: rewrite
        sensor/hits reader.planes after readout_type detection — jaxtpc.py:164-173),
      - pick the canonical reader for identity ordering.
    Raise NotImplementedError in the base.
    """
    raise NotImplementedError
```

The base's `_source_n_events(readers)` = `min(len(r) for r in readers.values())`
(mirrors `lucid.py:149` / `jaxtpc.py:180`) is **no longer the index size** (D42):
the joint-index step (§3.3a) supersedes it with `|intersection|`. `min(...)`
survives only as the comparison the A4 mismatch warning reports against. The
subclass also stores its
canonical reader (the one whose `(h5_files, indices, cumulative_lengths)` define
`source_event_idx` ordering — `lucid.py:147–148` / `jaxtpc.py:178–179`); the base
reads it via `readers['__canonical__']` (a sentinel key the subclass sets) or a
`_canonical_reader(readers)` method. **Reversible default:** sentinel key
`'__canonical__'` pointing at the same object as one of the modality entries.

**LUCiD** `_build_readers`: ports `lucid.py:121–148` verbatim — 4 readers gated
on `modalities`, canonical = `step or hits or sensor or labl`. No sub-selector.

**JAXTPC** `_build_readers`: ports `jaxtpc.py:132–179` verbatim — including
readout-type detection (`jaxtpc.py:156–162`) and the **`volume` plane rewrite**
(`jaxtpc.py:164–173`). `volume` arrives via `reader_kwargs`. Canonical =
`step or sensor or hits or labl`. **`volume` does not enter `config_id` or the
holdout digest** (D37/D40/§9.2): two `volume=` values of the same files map to
the *same* `(config_id, source_event_idx)` and therefore the *same* split — a
volume sub-selection is a different *view* of the same event, not a different
event.

### 3.5 Holdout algorithm (D26) with code sketch

Identity: `(config_id, source_event_idx)`. `source_event_idx` is the **stable,
writer-stamped** original event index — `config/source_event_idx[local_idx]`
(per-file vector, preferred, O(1)/file — WAND sensor+labl have it; JAXTPC has
the per-event attr `event_NNN.attrs['source_event_idx']`, plan §3.4/§9.2,
`production/save.py:344/391/625`). It is **not** the dense `local_idx` (which
shifts when shards are reordered/added/removed). The manifest (§3.6) caches it.

Bucketing is a **pure function** of `(seed, config_id, source_event_idx)` →
identical on every rank and machine (replaces `lucid_event_ssl.py:202–205`'s
`np.random.permutation`, which depends on NumPy version + per-source ordering).

```python
import hashlib, struct

def _bucket_u64(seed: int, config_id: int, source_event_idx: int) -> float:
    """blake2b(seed, config_id, source_event_idx) -> uniform in [0,1).

    config_id is folded into the digest => config-stratified WITHOUT any
    per-config bookkeeping (D26). Each config independently gets ~the same
    fraction split, because the hash decorrelates across config_id.
    """
    payload = struct.pack('<qqq', int(seed), int(config_id), int(source_event_idx))
    digest = hashlib.blake2b(payload, digest_size=8).digest()  # 8 bytes
    u = int.from_bytes(digest, 'little')                       # uint64
    return u / 2.0**64                                          # [0,1)

# Bucket -> split role, 3-way. fractions=(f_tr, f_va, f_te), sum==1.
def _bucket_to_split(u: float, fractions) -> str:
    f_tr, f_va, f_te = fractions
    if u < f_tr:            return 'train'
    if u < f_tr + f_va:     return 'val'
    return 'test'
# 'all' selects everything regardless of bucket.
```

**`fractions` mode** (default holdout): each surviving candidate's `u` is
computed once; keep it iff `_bucket_to_split(u, fractions) == self.split`
(or `self.split == 'all'`). Stratification is *implicit*: because `config_id`
is in the digest, the `u` values for each config are independent uniforms, so
each config contributes ~`f_va` to `val`, etc. No grouping or per-config
counting required — this is the "fold `config_id` into the digest" trick (D26).

**`n_per_config` mode** (D26: "take the `k` smallest-`u` events per config"):
deterministic exact count per config. For each config, sort surviving
candidates by `u` ascending; the first `k` form the holdout pool; the holdout
pool is split `val`/`test` by a secondary thresholding of the same `u` within
the pool (default `n_per_config` puts the whole pool in `val`+`test` and the
rest in `train`). Concretely:

```python
def _apply_holdout(self, candidates):  # candidates: {config_id: [local_idx,...]}
    cfg = self._holdout_cfg
    if cfg is None:                      # no holdout => everything is 'train'
        return candidates if self.split in ('train','all') else {c: [] for c in candidates}
    seed = cfg['seed']
    out = {}
    if 'n_per_config' in cfg:
        k = cfg['n_per_config']
        for c, locs in candidates.items():
            sei = self._manifests[c].source_event_idx        # vector aligned to locs' positions
            us  = np.array([_bucket_u64(seed, c, int(sei[l])) for l in locs])
            order = np.argsort(us, kind='stable')             # ties -> by position (stable)
            holdout_pos = set(order[:k].tolist())             # k smallest-u
            sel = []
            for rank, l in enumerate(locs):
                in_holdout = rank in holdout_pos
                if self.split == 'all':                       sel.append(l)
                elif self.split == 'train' and not in_holdout: sel.append(l)
                elif self.split in ('val','test') and in_holdout:
                    # split the holdout pool val/test by u parity within pool
                    half = us[rank] < np.median(us[list(holdout_pos)])
                    if (self.split == 'val') == bool(half):   sel.append(l)
            out[c] = sel
    else:                                                     # fractions mode
        fr = cfg['fractions']
        for c, locs in candidates.items():
            sei = self._manifests[c].source_event_idx
            out[c] = [l for l in locs
                      if self.split == 'all'
                      or _bucket_to_split(_bucket_u64(seed, c, int(sei[l])), fr) == self.split]
    return out
```

**Fallback** (D26): when `source_event_idx` is absent for a source (no per-file
vector AND no per-event attr — Part 04 reports this via `read_meta`), fall back
to identity `(config_id, positional)` where `positional = local_idx` (the dense
reader index), and emit **one** `log.warning` per source. This is deterministic
given a fixed shard set but **not** stable under shard add/remove/reorder — the
warning makes that explicit. The hash and split logic are otherwise unchanged
(just substitute `positional` for `source_event_idx`).

**Why blake2b, not Python `hash()` or `np.random`:** `hash()` is salted per
process (`PYTHONHASHSEED`) → not reproducible; `np.random.permutation` depends on
NumPy version and per-source ordering (the colleague's bug). blake2b in the
stdlib is process-, version-, and machine-stable; `struct.pack('<qqq', ...)`
fixes byte order so big/little-endian machines agree (D26 "machine"
determinism). `digest_size=8` is enough entropy for ≤10^8 events (collisions
between distinct identities only matter if they land within `1/2^64` of the same
fraction boundary — negligible, and even then both go to the same split, which
is harmless).

### 3.6 Min-points + manifest cache (D8/D27)

**`n_hits` semantics** (D8 cheap, inclusive `>=`):
- `min_points` normalized: `int` → `{threshold:int, modality:'sensor',
  op:'>='}`; `None` → no filter. `op` reversible default `'>='` (D8; **note the
  intentional parity diff vs the colleague's `>`**, plan §7).
- `n_hits` is read from `read_meta(local_idx)['n_hits']` (Part 04), which reads
  **only `evt.attrs`** — never datasets. Sources (plan §3.4 table):
  LUCiD sensor `evt.attrs['n_hits']`; LUCiD step `evt.attrs['n_segments']`;
  LUCiD hits `evt.attrs['n_particle_hits']`; JAXTPC sensor Σ plane
  `g.attrs['n_pixels']` over plane groups; JAXTPC step/hits Σ vol
  `g.attrs['n_actual']`. `config_id` is not involved.
- **`min_deposits`/`min_segments` are intercepted here (D44).** The legacy
  per-reader knobs route through this dataset-level min-points path on the **joint
  index**: `min_deposits` maps to `min_points={'modality':'step', ...}`,
  `min_segments` to the LUCiD step segment count. The step/lucid-step reader's
  internal index mask (`jaxtpc_step.py:84-100`, `lucid_step.py:85-94`) is **no-op'd**
  so it can no longer build a non-contiguous per-reader index that the other
  modalities never see (handoff §4 #1). Because the filter runs on the joint index,
  every modality drops the *same* canonical events.
- **Volume-aware min-points (D44/A3).** When `volume=N` is set, the JAXTPC
  min-points count is scoped to that volume's deposits (handoff §4 #3:
  `jaxtpc_step.py:91-97` today sums `n_actual` across **all** volumes regardless of
  `self.volume`, so a volume-0 filter can keep an event whose deposits all live in
  volume 1). The volume scoping affects **which events survive min-points only**; it
  does **NOT** enter `config_id`, the holdout digest, or `event_identity`
  (reconciles D37/§9.2 — `volume` stays an orthogonal view, never a holdout axis).
  Passing `min_deposits>0`/`min_segments>0` without the source modality loaded
  raises (A3; §5).
- Filter: keep `local_idx` iff `n_hits[local_idx] >= threshold`. This must
  select the **identical set** as a full array-count of the chosen modality's
  point cloud (test §6.7); the writer-side ask (D27/§9.2) is to stamp a
  per-event `n_hits` so cheap == array exactly.

**Manifest** (D27: "persisted manifest cache (rank-0 build under DDP barrier),
never array reads in steady state"). One manifest per source:

```python
@dataclass
class Manifest:
    config_id: int
    n_events: int
    source_event_idx: np.ndarray   # (n_events,) int64, aligned to local_idx
    n_hits: np.ndarray             # (n_events,) int64, aligned to local_idx
    has_source_event_idx: bool     # False => holdout fallback for this source
```

**Cache key / invalidation.** The on-disk cache filename is a hash of the
material inputs so any change invalidates it without manual cleanup:

```python
key = blake2b('|'.join([
    pimm_data.__version__,            # schema/algorithm version
    'manifest-v1',                    # bump on Manifest layout change
    type(self).__name__,             # LUCiD vs JAXTPC (different n_hits source)
    repr(sorted(self._modalities)),
    str(reader_kwargs.get('volume')), # JAXTPC volume changes n_hits source? -> NO, but keep for safety
    # per-source file identity: path + size + mtime of EACH shard, sorted
    *[f"{p}:{os.path.getsize(p)}:{int(os.path.getmtime(p))}"
      for p in sorted(src.canonical.h5_files)],
]).encode()).hexdigest()
cache_path = os.path.join(cache_dir, f"{key}.npz")
```

Invalidation triggers (any → new key → rebuild): a shard added/removed/renamed,
a shard's size or mtime changes, modalities change, the package version bumps,
the Manifest layout version bumps. **Reversible default** (D34): `cache_dir =
$PIMM_DATA_CACHE or os.path.join(source_root, '.pimm_manifest')`; per-source
`.npz` storing `{source_event_idx, n_hits, has_source_event_idx, n_events}`.

**Rank-0 build under barrier + atomic write** (D27):

```python
def _load_or_build_manifests(self):
    import pimm.utils.comm as comm  # the existing DDP comm (kept in pimm)
    out = {}
    for src in self._sources:
        path = self._manifest_path(src)
        if not os.path.exists(path):
            if comm.get_rank() == 0:
                m = self._scan_manifest(src)        # the ONLY place evt.attrs are walked
                self._atomic_write(path, m)         # write tmp + os.replace (atomic on POSIX)
            if comm.get_world_size() > 1:
                comm.synchronize()                  # barrier: all ranks wait for rank-0 write
        out[src.config_id] = self._read_manifest(path)   # every rank reads the SAME file
    return out
```

- **Rank-identical index** (D27): every rank reads the identical `.npz`, so the
  manifest (and hence the entire `data_list`) is byte-identical across ranks.
  The barrier guarantees the file exists before any non-zero rank reads it.
- **Atomic write**: write to `path + f'.tmp.{os.getpid()}'`, `np.savez`, then
  `os.replace(tmp, path)` — `os.replace` is atomic on POSIX, so a concurrent
  reader never sees a half-written file and a crash mid-write leaves the old
  cache intact.
- **`comm` is imported lazily inside the method** so pimm-data has no hard
  import-time dependency on pimm (D18: `comm` stays in pimm). When pimm is
  absent (unit tests, standalone), fall back to a single-process shim
  (`get_rank()->0`, `get_world_size()->1`, `synchronize()->noop`). **Reversible
  default:** a tiny `_serial_comm` module-level object used when
  `import pimm.utils.comm` raises `ImportError`.
- **Scan cost**: `_scan_manifest` opens each shard once, walks `evt.attrs`
  (and for JAXTPC sensor, plane-group `n_pixels` attrs) for all events,
  reads `config/source_event_idx[:]` if present (one vector read/file). This is
  the only steady-state-avoidable cost; it runs once per (rank-0, cache-miss).

### 3.7 `event_identity(idx)`, public `split`, `data_list`/`datasets` shape

These are the exact surfaces the probe (`lucid_event_probe.py`) and the eval
contract (D41) read. **Frozen.**

```python
def event_identity(self, idx):
    """Modality-independent stable identity (D26).
    Returns (config_id:int, source_event_idx:int). Mirrors the holdout key,
    so two datasets built with the same sources+seed map idx-space to the same
    identities regardless of which modalities are loaded.
    """
    source_idx, local_idx = self.data_list[idx % len(self.data_list)]
    src = self._sources[source_idx]
    m = self._manifests[src.config_id]
    return (src.config_id, int(m.source_event_idx[local_idx]))
```

`local_idx` here is the canonical joint-index position (§3.3a), so
`m.source_event_idx[local_idx]` is the single `source_event_idx` shared by every
loaded modality for that event — identity is computed once, over the intersected
index, and is modality-independent by construction (D42).

- **`self.split`** — set in step (2), a plain str (or tuple) attribute.
  Survives `Subset` because the probe peels `Subset` then reads
  `getattr(base, "split")` (`lucid_event_probe.py:135–148`). It MUST be one of
  `{'train','val','test','all'}` so the guard's
  `val_split_set <= {'holdout','val','test'}` check passes for eval loaders
  (`lucid_event_probe.py:209–215`). (`'holdout'` is accepted by the guard as a
  synonym for `'val'`; we standardize on `'val'`.)
- **`self.data_list`** — `list[tuple[int,int]]` = `(source_idx, local_idx)`.
  `source_idx` indexes `self._sources`/`self.datasets`; `local_idx` is the
  per-source dense reader index (what `read_event`/`read_meta` take). This is
  exactly what `_event_keys` unpacks (`lucid_event_probe.py:187`).
- **`self.datasets`** — `list[dict]`, one per source, each = `src.descriptor`
  (§3.3): carries `source_root` (realpath), `data_root`, `name`, `config_id`,
  `label`. The probe's `_source_key` reads `source_root`→`data_root`→`name`
  (`lucid_event_probe.py:151–158`). Naming `datasets` (not `sources`) is
  required because the guard does `getattr(dataset, "datasets")` literally
  (`lucid_event_probe.py:175`).

**Probe identity vs guard identity.** The guard forms its disjointness key from
`(_source_key, event_idx)` where `event_idx = local_idx` (`lucid_event_probe.py:
187–189`). Within a single run, train and val datasets share the same source set
and the same `local_idx → source_event_idx` manifest, so disjointness in
`(source_key, local_idx)` ⟺ disjointness in `(config_id, source_event_idx)`. The
hash holdout guarantees the latter (every identity lands in exactly one split),
so the guard sees zero overlap. (A follow-up, Part 06, may rewire the probe to
call `event_identity` directly per plan §3.7; this base satisfies *both* the
current `data_list`/`datasets` guard and the future `event_identity` guard.)

### 3.8 `get_data(idx)` + `get_data_name(idx)` (collision fix)

```python
def get_data(self, idx):
    source_idx, local_idx = self.data_list[idx % len(self.data_list)]
    src = self._sources[source_idx]
    data = self._read_event(src, local_idx)           # SUBCLASS: per-modality builders
    n_points = self._primary_n_points(data)           # rows of the primary stream
    # event_broadcast materialization (D9/§3.5): per-point arrays so Collect lifts
    # them and the probe slices by offset (lucid_event_probe.py:115-128).
    data['event_label'] = np.full((n_points,), src.label, dtype=np.int64)
    data['config_id']   = np.full((n_points,), src.config_id, dtype=np.int64)
    data['name']  = self.get_data_name(idx)
    data['split'] = self.split if isinstance(self.split, str) else 'custom'
    return data
```

`_read_event(src, local_idx)` is the SUBCLASS dispatch — it is the existing
`get_data` body of `lucid.py:189–214` / `jaxtpc.py:233–269`, refactored to take
an explicit reader set + local index instead of `self.*_reader` + `idx % len`.
**Here `local_idx` is a canonical joint-index position (§3.3a), not a raw shared
reader index:** `_read_event` translates it per modality via
`src.joint.local_for[modality][local_idx]` before calling that reader's
`read_event` (D42). This replaces the old "same idx to every reader"
(`jaxtpc.py:233-269`) that was the desync's mechanism — each reader is now handed
its own dense row for the *same* `source_event_idx`.
`_primary_n_points` reads the row count of the primary stream (sensor/step/hits
`coord`); for nested output the base picks the first modality in `self._modalities`
that produced a `coord`. (`event_label`/`config_id` placement is reversible per
D34; the per-point-broadcast default is what the probe needs.)

**`get_data_name` — source-prefix collision fix** (plan §3.3):

```python
def get_data_name(self, idx):
    source_idx, local_idx = self.data_list[idx % len(self.data_list)]
    src = self._sources[source_idx]
    canon = self._canonical_reader(src.readers)
    file_idx = int(np.searchsorted(canon.cumulative_lengths, local_idx, side='right'))
    base = int(canon.cumulative_lengths[file_idx - 1]) if file_idx > 0 else 0
    event_num = canon.indices[file_idx][local_idx - base]
    fname = os.path.basename(canon.h5_files[file_idx])
    # SOURCE PREFIX: distinguishes identical filenames across configs.
    return f"config_{src.config_id}/{fname}_evt{event_num:03d}"
```

The body is `lucid.py:348–356` / `jaxtpc.py:499–506` verbatim, **plus the
`config_{config_id}/` prefix**. Today both subclasses return
`f"{fname}_evt{event_num:03d}"` (no prefix), so two sources whose shards are both
named `wc_sensor_0000.h5` collide on `wc_sensor_0000.h5_evt007` — the probe's
disjointness/feature keying and any name-based dedup silently merge them. The
prefix is the fix (plan §3.3: "fixes the cross-config filename collision";
mirrors the colleague's `f'{source["name"]}/{...}'` at `lucid_event_ssl.py:323`,
but uses the stable `config_id` rather than a basename that can itself collide).

### 3.9 Mixture index construction + weighting (D9)

```python
def _build_index(self, candidates, mixture):
    """candidates: {config_id: [local_idx,...]} (post holdout+max_events).
    Returns flat [(source_idx, local_idx), ...] in deterministic order.
    """
    mode = (mixture or {}).get('mode', 'replicate')   # reversible default: replicate (D9/D34)
    weights = self._resolve_weights(mixture, self._sources)  # per-source float
    data_list = []
    for source_idx, src in enumerate(self._sources):
        locs = sorted(candidates.get(src.config_id, []))     # stable order
        if mode == 'replicate':
            reps = self._integer_reps(weights[source_idx])   # e.g. 2.0 -> 2 copies
            for _ in range(reps):
                data_list.extend((source_idx, l) for l in locs)
        else:  # 'sampler' -> emit once; weights consumed by a WeightedRandomSampler
            data_list.extend((source_idx, l) for l in locs)  # weight recorded in self.sample_weights
    return data_list
```

- **`replicate` (default, D9):** a source's weight `w` (relative to the others)
  becomes integer copies of its event list, so a `w=2` source appears twice as
  often per epoch. Weights are normalized so the smallest is `1` copy
  (`_integer_reps` = `round(w / min(w))`). This needs no sampler and is
  rank-deterministic. `weights='uniform'` (default) → all `1` → one copy each.
  Explicit `weights=[...]` length-matches `sources`.
- **`sampler` mode** (reversible, D34): emit each event once, store
  `self.sample_weights` (per `data_list` entry = source weight); a
  `WeightedRandomSampler` (built pimm-side) consumes it. The base only *records*
  the weights; it never owns the sampler (DDP sampler stays in pimm, D18).
- **`volume` is not a mixture dimension** (D37): mixture is purely over sources.

---

## 4. Expected behavior (examples + invariants)

**Example A — single source, default holdout.**
```python
ds_train = LUCiDDataset(data_root='/data/wand/config_1', modalities=('sensor',),
                        split='train',
                        holdout={'seed': 0, 'fractions': (0.8, 0.1, 0.1), 'strata': 'config'},
                        min_points=10, dataset_name='wc')
ds_val   = LUCiDDataset(data_root='/data/wand/config_1', modalities=('sensor',),
                        split='val', holdout={'seed': 0, 'fractions': (0.8, 0.1, 0.1)},
                        min_points=10, dataset_name='wc')
```
- `ds_train` and `ds_val` are **disjoint** in `event_identity` (no identity in
  both); together with a `split='test'` build they **partition** the
  min-points-surviving population.
- `len(ds_train) ≈ 0.8 * (#events with n_hits >= 10)`.

**Example B — two-config mixture, PID probe.**
```python
ds = LUCiDDataset(sources=[{'root':'/data/wand/config_1','label':0},
                           {'root':'/data/wand/config_3','label':1}],
                  modalities=('sensor',), split='val',
                  holdout={'seed':0,'fractions':(0.8,0.1,0.1)}, dataset_name='wc')
```
- `ds.datasets[0]['source_root']` and `[1]['source_root']` are distinct
  realpaths; `ds.data_list[k] == (source_idx, local_idx)`.
- Each event's `data['event_label']` is a per-point `int64` array of the source
  label; the probe slices `[start:end][0]` per event (`lucid_event_probe.py:122`).
- Both configs contribute ~10% of their surviving events to `val` (stratified).

**Invariants (MUST hold):**
1. **Determinism (the headline).** For fixed `(sources, seed, fractions,
   modalities, min_points)`, `set(event_identity(i) for i in range(len(ds)))`
   is identical across: process restarts, machines, DDP world size, NumPy/Python
   versions, shard file order, shard additions/removals (because identity keys on
   `source_event_idx`, not `local_idx`). Only adding/removing *events* (not
   shards) changes membership, and only for the events that newly appear/vanish.
2. **Partition.** `train ⊎ val ⊎ test == all` over the min-points-surviving
   population, for any fixed `(seed, fractions)`; the three are pairwise disjoint
   in `event_identity`.
3. **Stratification.** For each `config_id`, `P(bucket == 'val') ≈ f_va` (within
   sampling noise), independently of other configs.
4. **Filter-then-split.** Changing `min_points` changes the surviving population
   and therefore the per-split membership, but a fixed surviving population always
   splits identically (min-points and holdout commute in the sense that the split
   is a pure function of identity, applied to whatever survives).
5. **Rank-identical.** `ds.data_list` is byte-identical on every DDP rank.
6. **Identity stability across modalities.** `event_identity(i)` for a
   `modalities=('sensor',)` build equals it for a `modalities=('step','labl')`
   build of the same sources/seed (modality-independent, D26) — provided both
   modalities expose the same `source_event_idx` per event (they do; it is
   per-event metadata, plan §3.4).
7. **`volume` orthogonality.** `JAXTPCDataset(..., volume=0)` and `volume=1` of
   the same files yield the same `event_identity` set and the same split
   membership (D37/§9.2).
8. **Name uniqueness.** `get_data_name(i)` is unique across the whole dataset,
   including across sources with identically-named shards (the `config_{id}/`
   prefix).
9. **Cross-modality alignment (D42, the desync fix).** For every served `idx` and
   every loaded modality, the event `_read_event` returns from that modality has the
   **same** `source_event_idx` (== `event_identity(idx)[1]`). Holds under
   `min_deposits>0`/`min_segments>0` (D44) and under a gap present in some-but-not-all
   modalities — because every modality is translated off the one joint index (§3.3a),
   never a shared raw `local_idx`. This is the property the A5 regression test
   (§6.16) locks; it FAILS on the pre-joint-index design.

---

## 5. Edge cases & error handling

- **Empty source / empty result.** If a source discovers zero events, or every
  event fails min-points, or the chosen split is empty: raise a clear
  `ValueError` at the end of `__init__` (mirrors the existing
  `jaxtpc.py:188–195` guard) naming the source, `split`, `min_points`, and
  `holdout`. A single empty source among several is allowed only if the overall
  `data_list` is non-empty; the empty source still appears in `self.datasets`
  (so `source_idx`→descriptor stays aligned) but contributes no `data_list`
  entries. **Test §6.8** covers this.
- **`split` not in `{'train','val','test','all'}`** → `ValueError` (the probe
  guard would otherwise reject it cryptically). `'holdout'` accepted as a
  deprecated alias for `'val'` (the colleague used it,
  `lucid_event_ssl.py:52`) with a one-time warning.
- **Both `data_root` and `sources` given** → `ValueError` (ambiguous).
- **Duplicate explicit `config_id`** across sources → `ValueError` (breaks
  stratification + identity uniqueness).
- **`fractions` not summing to 1.0** (±1e-6) → `ValueError`.
- **`holdout=None` with `split in {'val','test'}`** → empty result → the empty-
  result `ValueError` fires (you cannot ask for a holdout split without a holdout
  spec).
- **`source_event_idx` absent** → fallback to `(config_id, positional)` + one
  `log.warning` per source (§3.5). Not an error; determinism degrades to
  "stable given fixed shard set" only.
- **`min_points` modality not loaded** (e.g. `min_points={'modality':'hits'}`
  but `'hits' not in modalities`) → `ValueError` at normalize time (the manifest
  has no `n_hits` column for an unloaded modality). The chosen modality's reader
  must be among `self._modalities`.
- **`min_deposits>0`/`min_segments>0` without the source modality loaded** (D44/A3,
  handoff §4 #4) → `ValueError`. Today the step reader is only built when
  `'step' in modalities` (`jaxtpc.py:132`), so `modalities=('hits','labl'),
  min_deposits=N` silently no-ops with no warning; the base raises instead (it is a
  user error to filter on a modality that is not loaded). **Test in §6.16.**
- **Per-modality present-key mismatch** (D47/A4) → intersect + one `log.warning`
  with concrete per-modality counts by default; `strict_lengths=True` → `ValueError`
  naming the mismatched modalities and their counts. **Test in §6.16.**
- **Manifest cache unwritable** (read-only `source_root`, no
  `$PIMM_DATA_CACHE`) → fall back to in-memory scan on every rank with a
  `log.warning` (correctness preserved; steady-state cost regresses). Rank-0
  still scans; non-zero ranks also scan (no shared file) — still rank-identical
  because the scan is deterministic, just redundant.
- **Manifest cache stale/corrupt** (`np.load` raises, or `n_events` mismatch vs
  the reader's current `__len__`) → treat as cache-miss, rebuild, overwrite via
  atomic write.
- **`n_hits` attr missing on some events** (`read_meta` returns `None`/`-1`) →
  treat that event's `n_hits` as `0` (fails any positive threshold) and emit one
  aggregated `log.warning` per source counting affected events. Do **not** fall
  back to an array read in the hot path.
- **`idx` beyond `len`** — `get_data`/`get_data_name`/`event_identity` all use
  `idx % len(self.data_list)` (matches `defaults.py:94`), so `loop > 1` wraps
  correctly.
- **`__del__`** closes all readers across all sources (port `lucid.py:392–400`),
  guarded by `try/except` and `getattr(..., None)` so a partially-constructed
  instance (exception during `_build_readers`) cleans up safely.

---

## 6. Tests

All tests use `src/pimm_data/testing.py` fixtures (`make_lucid_sample`,
`make_jaxtpc_sample`) — pure numpy+h5py, no GPU/WAND (plan §6). These fixtures
satisfy the cross-modality FK invariants documented in `testing.py:14–27`.
**Gap to close in Part 04/fixtures:** `testing.py` does NOT currently write
`source_event_idx` or per-event `n_hits` attrs (verified: `_write_lucid_sensor`
at `testing.py:491–504` writes only `sensor_idx/PE/T`; JAXTPC writers stamp
`n_actual`/`n_pixels`-implied but not `source_event_idx`). Tests must either
extend the fixtures to stamp `source_event_idx`/`n_hits` (preferred — add a
`source_event_idx=` kwarg + `n_hits` attr) or exercise the **fallback** path
explicitly. Note this dependency for the fixture owner.

**Test 6.1 — holdout determinism under shard reorder.**
- *Setup:* `make_lucid_sample(dir, n_files=4, n_events=8)` with stamped
  `source_event_idx`. Build `ds_a = LUCiDDataset(..., split='val', holdout=
  {seed:0, fractions:(.8,.1,.1)})`. Rename/reorder shards on disk (swap
  `_0000`↔`_0003`), invalidate cache, build `ds_b` identically.
- *Action:* compare `set(ds_a.event_identity(i))` vs `ds_b`.
- *EXPECTED:* identical identity sets (reorder changes `local_idx` but not
  `source_event_idx`, so membership is invariant).

**Test 6.2 — holdout determinism under add/remove events.**
- *Setup:* base dataset of 32 events; a second with one event removed and one
  added (new `source_event_idx`).
- *Action:* diff the `val` identity sets.
- *EXPECTED:* the diff is exactly `{removed_identity}` deleted and possibly
  `{added_identity}` inserted (iff its bucket is `val`); every other event's
  split assignment is unchanged.

**Test 6.3 — rank-identical index.**
- *Setup:* monkeypatch the comm shim to report `get_world_size()==4` and iterate
  `get_rank() in {0,1,2,3}` building the dataset each time (rank-0 first to seed
  the cache, then the others read it under a stubbed barrier).
- *Action:* compare `data_list` across the four builds.
- *EXPECTED:* byte-identical `data_list` and `event_identity` map on all ranks.

**Test 6.4 — machine/version determinism (golden).**
- *Setup:* fixed fixture; compute `[_bucket_u64(0, c, s) for known (c,s)]`.
- *Action:* assert against a hardcoded golden vector (blake2b is fixed).
- *EXPECTED:* exact match — guards against an accidental hash/seed/pack change.

**Test 6.5 — config-stratification.**
- *Setup:* two sources, 1000 events each, `fractions=(.8,.1,.1)`.
- *Action:* per-config count of `val` vs `test` vs `train`.
- *EXPECTED:* each config independently ≈ 80/10/10 (±3σ binomial); the global
  ratio is also ≈ 80/10/10. Folding `config_id` into the digest yields
  per-config balance without bookkeeping (D26).

**Test 6.6 — `n_per_config` mode.**
- *Setup:* `holdout={seed:0, n_per_config:5}`, two configs.
- *Action:* build `split='val'`+`split='test'`; count holdout per config; build
  `split='train'`.
- *EXPECTED:* exactly 5 holdout events per config (val+test = 5 each), the rest
  in train; deterministic across rebuilds (`k`-smallest-`u`, stable ties).

**Test 6.7 — min-points: cheap == array, and `>=` boundary.**
- *Setup:* fixture with known per-event point counts; pick a `threshold` equal to
  one event's exact count.
- *Action:* (a) build with `min_points=threshold` (cheap `read_meta` path);
  (b) independently array-count each event's primary stream rows and apply
  `>= threshold`.
- *EXPECTED:* identical surviving `local_idx` sets (cheap == array); the
  boundary event (count == threshold) **is kept** (`>=`, not `>`); building with
  `op='>'` drops it (documents the colleague-parity diff, plan §7).

**Test 6.8 — empty source.**
- *Setup:* two sources, one with `min_points` so high all its events fail.
- *Action:* build; inspect `datasets` and `data_list`.
- *EXPECTED:* `len(datasets)==2`; no `data_list` entry has `source_idx==1`;
  `source_idx`→descriptor mapping intact; build succeeds (overall non-empty).
  A second build where *all* sources are emptied raises `ValueError`.

**Test 6.9 — identity stability across modalities.**
- *Setup:* same sources/seed, build once `modalities=('sensor',)`, once
  `('step','labl')`.
- *Action:* compare `event_identity` over the intersection of `local_idx`.
- *EXPECTED:* identical `(config_id, source_event_idx)` for the same underlying
  event (D26 modality-independence). (Requires both modalities to stamp the same
  `source_event_idx` — fixture invariant.)

**Test 6.10 — `volume` orthogonality (JAXTPC).**
- *Setup:* `make_jaxtpc_sample(n_volumes=2)`; build `volume=0` and `volume=1`,
  same seed/fractions/split.
- *Action:* compare `event_identity` sets and split membership.
- *EXPECTED:* identical (volume is an orthogonal view, not the mixture/holdout
  axis, D37/§9.2).

**Test 6.11 — `get_data_name` uniqueness across configs.**
- *Setup:* two sources whose shards are *identically named*
  (`wc_sensor_0000.h5`) — copy the same fixture into two roots.
- *Action:* collect `{get_data_name(i) for i in range(len)}`.
- *EXPECTED:* set size == `len(ds)` (no collision); names carry distinct
  `config_0/`…`config_1/` prefixes. Re-running without the prefix fix (or against
  the old subclass `get_data_name`) collides (regression sentinel).

**Test 6.12 — probe contract shape.**
- *Setup:* any 2-source build.
- *Action:* assert `data_list` is `list[(int,int)]`; `datasets` is `list[dict]`
  with `source_root` realpaths; run the probe's `_event_keys` and
  `_dataset_split` against the dataset.
- *EXPECTED:* `_event_keys` returns a non-None set of `(source_key, event_idx)`;
  `_dataset_split` returns the `split` string; for disjoint train/val builds the
  guard's `train_keys & val_keys == set()`.

**Test 6.13 — manifest cache invalidation.**
- *Setup:* build once (writes cache); record cache filename + mtime.
- *Action:* (a) rebuild unchanged → cache hit (no rescan; assert
  `_scan_manifest` not called, e.g. via a counter monkeypatch). (b) `touch` a
  shard (bump mtime) → rebuild → cache *miss* (new key, rescan). (c) truncate the
  `.npz` → rebuild → treated as miss, rebuilt, valid.
- *EXPECTED:* (a) no rescan, identical `data_list`; (b)+(c) rescan, identical
  `data_list` (content unchanged), new cache file.

**Test 6.14 — atomic write safety.**
- *Setup:* monkeypatch `_scan_manifest` to write the tmp file then raise before
  `os.replace`.
- *Action:* build (expect failure), inspect cache dir, then build again clean.
- *EXPECTED:* no partial `.npz` at the final path (only an orphan `.tmp.<pid>`,
  ignored on next run); the clean rebuild produces a valid cache.

**Test 6.15 — fallback when `source_event_idx` absent.**
- *Setup:* fixture WITHOUT `source_event_idx` (current `testing.py` default).
- *Action:* build; capture logs.
- *EXPECTED:* exactly one `log.warning` per source about positional fallback;
  holdout still partitions deterministically given the fixed shard set;
  reordering shards now *does* change membership (documents the degraded
  guarantee).

**Test 6.16 — cross-modality alignment (A5; the desync regression, D42/D44/D47).**
This is the A5 regression test from `shard_event_filtering_handoff.md` §5, lifted to
the base. **Both variants FAIL on the pre-joint-index design** (and on
`master`/HEAD of `jaxtpc.py`).
- *(a) min_deposits desync (handoff §4 #1).* Build
  `modalities=('step','sensor','hits','labl')` with `min_deposits>0` (or
  `min_segments>0`) on a fixture where the step deposit count masks a non-contiguous
  subset. For **every** served `idx`, read the served event from every modality and
  assert they share the **same** `source_event_idx` (identifying attr) — i.e.
  `event_identity(idx)` agrees with what each modality actually returned. EXPECTED:
  identical `source_event_idx` across all four modalities for every idx; before the
  fix, step returns `valid[k]` while sensor/hits/labl return `present[k]`
  (different physics events, `bridges`/`group_to_track` joins meaningless).
- *(b) gap-in-one-modality (handoff §4 #2).* Fixture where one modality is missing a
  middle `event_*` group the others have (e.g. hits missing `event_131` while step
  has it). Build all-modalities, assert same-`source_event_idx`-across-modalities for
  every served idx. EXPECTED: the joint index intersects out the gapped event so no
  served idx straddles it; before the fix, `_n_events=min(...)` + shared `local_idx`
  misaligns at the gap. Also assert the A4 `log.warning` (or `ValueError` under
  `strict_lengths=True`) reports the concrete per-modality counts.
- *(c) volume-aware min-points (D44/A3).* `make_jaxtpc_sample(n_volumes=2)` with all
  deposits of some events in volume 1; build `volume=0, min_deposits>0`. EXPECTED:
  events whose volume-0 deposit count is below threshold are dropped (not kept on a
  volume-blind all-volume sum); the surviving set's `event_identity` is **unchanged**
  vs `volume=1` of the same files (volume scopes min-points but not holdout/identity,
  Invariant 7). And `min_deposits>0` with `modalities=('hits','labl')` (no step)
  raises (§5).

---

## 7. Reversible defaults & risks

**Reversible defaults (D34 — chosen at code time, recorded here):**
- `min_points` op default `'>='` (D8). **Intentional parity diff** vs the
  colleague's `>` (`lucid_event_ssl.py:232`); a boundary event differs by one.
- `mixture.mode` default `'replicate'`; `weights` default `'uniform'` (D9).
- `event_label`/`config_id` materialized as **per-point** broadcast arrays in
  `get_data` (what the probe needs); per-event layout is a reversible alternative.
- Manifest cache: per-source `.npz` keyed by version+modalities+per-shard
  (path,size,mtime); `cache_dir = $PIMM_DATA_CACHE or <root>/.pimm_manifest`.
- blake2b `digest_size=8`, `struct.pack('<qqq', seed, config_id,
  source_event_idx)` little-endian; 3-way `train/val/test`.
- Canonical-reader sentinel key `'__canonical__'` in the `_build_readers` dict.
- `_serial_comm` shim when `pimm.utils.comm` is unimportable (standalone/tests).
- `n_per_config` val/test split = median-`u` within the holdout pool.

**Risks:**
- **`source_event_idx` coverage.** WAND has both the per-file vector and the
  per-event attr; JAXTPC has only the per-event attr (no `config/source_event_idx`
  vector — `production/save.py` stamps `evt.attrs` only). The base must accept
  *either* (vector preferred, attr fallback, positional last). Until the JAXTPC
  writer adds a vector (D27 ask), JAXTPC manifest build does a per-event attr
  walk — cheap but O(events)/file once.
- **`testing.py` fixtures lack `source_event_idx`/`n_hits` attrs today** — tests
  6.1/6.2/6.6/6.7/6.9 require extending the fixtures (Part 04 / fixture owner) or
  they only exercise the fallback. Flagged in §6.
- **Manifest staleness.** mtime-based invalidation misses a same-size,
  same-mtime content change (rare; e.g. restore-from-backup preserving mtime).
  Acceptable; a `--no-manifest-cache` escape hatch is the mitigation.
- **`n_per_config` val/test boundary** is less natural than `fractions`; if a
  task needs exact per-config val *and* test counts, prefer two `n_per_config`
  fractions or `fractions` mode.
- **Probe still reads `data_list`/`datasets` directly** (not `event_identity`)
  until Part 06 rewires it. This base satisfies both, so the rewire is
  non-blocking.
- **Cross-modality desync was INHERITED, now fixed by the joint index (D42).** The
  pre-§3.3a spec re-implemented `_n_events=min(...)` + one shared `local_idx` → it
  silently inherited the `jaxtpc.py:180`/`:233-269` desync (handoff §4); the §3.3a
  joint index is the fix and is normative. Risk now shifts to the **fallback**: when
  `source_event_idx` is absent on a source the join degrades to a positional
  intersection (D26 fallback) and a gap in some-but-not-all modalities can still
  misalign — surfaced by the one-time warning, not silent. Mitigation: the
  writer-side `source_event_idx` asks (Part 03 §7; `make_labl.py` stamp).
- **Phase A lands first (D43).** The joint index ships as a standalone bug-fix PR on
  `jaxtpc.py` (Phase A A1–A5) **before** the de-fork; the base factors A2 up. If the
  ordering slips (base built before Phase A lands), the base still specifies the
  joint index here, so it does not regress — but the standalone fix is the
  independently-valuable, sooner-shipped form (handoff §5).
- **`strict_lengths` default is permissive (D47/A4).** Default `False` warns and
  intersects; a 1-shard mismatch is recoverable-with-warning, not a hard stop. Set
  `strict_lengths=True` in repro-critical runs. Manifest-as-INPUT include/exclude
  curation is **Phase B, deferred** (not built here).

---

## 8. Dependencies on other parts

- **Part 04 (readers / `read_meta` + surfacing).** Frozen interface consumed
  here: `read_meta(local_idx) -> {'source_event_idx': int|None, 'n_hits':
  int}` reading **only `evt.attrs`** (+ `config/source_event_idx[:]` vector when
  present); readers keep `cumulative_lengths`/`indices`/`h5_files`/`read_event`/
  `__len__`/`close` (already present). JAXTPC writer-side `n_hits`/
  `source_event_idx` vector ask (D27/§9.2) lives there. **The joint index (§3.3a,
  D42) additionally consumes each reader's per-modality present-`event_*`-key set +
  `source_event_idx`-per-key** (the `_has_sei_vec`/`_sei_vecs` surface, Part 04
  §3.0); the cross-reader opens are deduped by the `_read_shard_meta` lru_cache
  (D46) shared with the manifest scan.
- **Part 03 (collate).** Single-stream `collate_fn` (byte-identical REPLACE,
  plan §3.6); `event_label`/`config_id` reach the batch as ordinary per-point
  columns produced here.
- **Part 05 (`label_config` decoration).** `label_config` is carried through
  `__init__` untouched and handed to the subclass builders; the FK resolver is
  the per-subclass piece. This base only materializes `event_label`/`config_id`
  (the `event_broadcast` scope); `segment_*`/`instance_*`/`target_*` are Part 05.
  **Decoration correctness depends on the joint index (D42):** label decoration
  joins the stream reader and the labl reader at the same event, so it is only
  correct if those readers are aligned — which is exactly what §3.3a guarantees
  (without it, decoration re-exposes the desync for every labeled task). The A5
  regression (§6.16) covers the `('step','sensor','hits','labl')` case Part 05
  relies on.
- **Part 06 (eval/probe rewiring).** Will switch `lucid_event_probe.py:161–190`
  to call `event_identity`; this base's `event_identity`/`split`/`data_list`/
  `datasets` are the frozen surfaces it relies on (D41 reproducibility: per-run
  recording of seed+fractions+identity scheme).
- **pimm `comm`** (D18, stays in pimm): imported lazily for the rank-0/barrier
  manifest build; `_serial_comm` fallback when absent.
- **`DefaultDataset`/`TestModeMixin` (`defaults.py`).** The mixin extraction
  (D30) is a precondition; `DefaultDataset` and the npy/PILArNet path must remain
  byte-identical after it (the guarded `prepare_test_data` is a strict superset).
