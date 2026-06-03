# ADR: pimm-data ↔ pimm package boundary

**Status:** Decided (this document is decisions-only — no code changed yet).
**Date:** 2026-06-02.
**Supersedes/consolidates:** the exploratory `engagement_plan_transform_dataset_placement.md`.
**Implementation + corrections:** see `IMPLEMENTATION-boundary-refactor.md` — the file-by-file change
manifest, whose §1 carries audit corrections that supersede a few statements here (test-move scope,
torch-importer count, the torch-free-guard mechanics).
**Context:** Triggered by Sam Young's proposal to keep the data package minimal (H5 readers
+ basic collate + minimal Dataset + maybe transforms) and push the "real" Dataset, transforms,
and collate into `pimm`. This ADR records what we decided after auditing the actual contents
of both packages, and why we diverge from the literal proposal in places.

---

## 1. Foundational decisions (locked)

| Axis | Decision | Why |
|---|---|---|
| **Audience** | `pimm-data` is a **standalone, external-first** package. | People outside the `pimm` model repo should be able to load detector data with lean deps (numpy/h5py/hdf5plugin/torch) and no `pimm` import. Already true today; we commit to keeping it true. |
| **Dataset** | **Keep batteries-included.** `JAXTPCDataset`/`LUCiDDataset`/`PILArNetH5Dataset`/`MultiModalEventDataset` stay in `pimm-data`. | The Dataset's value is the cross-modality joint index, nested-dict assembly, label decoration, and the F1–F17 fork-safety hardening — that is **data-layer** machinery, not model logic. The readers are also public, so a power user who wants their own packing/padding can build on the readers directly without us deleting the hardened path. |
| **Motivation** | **Pre-emptive hygiene**, not an urgent unblock. | No one is currently blocked. We have room to draw boundaries correctly and to do the work as separate, reviewable PRs rather than rushing. |
| **Transform principle** | **By genericness**, refined into tiers (§3). | Empirically, ~84% of transforms are generic IO/geometry/normalization, not pretraining-recipe-specific. The dividing line is "would a stranger loading detector data, with no `pimm` model, plausibly want this?" |
| **Distribution** | **Versioned git tags now; defer PyPI.** `pimm` pins a tag (not a raw SHA); external users `pip install` from a git tag. | External-first-ready without standing up PyPI release engineering before any external user exists. Promote to PyPI when there's real demand. |
| **Dependencies** | **torch stays a normal required dep; no install-time torch-optionality.** Preserve the (already-true) torch-free reader/index/schema layer as the documented "bring-your-own-framework / roll-your-own-Dataset" entry point. | Verified: only `transform.py`/`collate.py`/`anchors.py` import torch, and anchors is moving out — so after the SSL move the torch footprint is just transforms + collate + the Dataset base. The data-access layer already imposes no framework. Framework-neutral/TF is overkill (nobody needs TF; JAX/numpy users consume numpy from the readers). Install-extras machinery = permanent discipline for one saved `pip install torch`; YAGNI. |
| **Schema** | **Document a canonical, stable on-disk schema spec** (consumer-side, in pimm-data; cross-linked from JAXTPC). Optional cheap insurance: a `schema_version` attribute stamped at write time + asserted by readers. | The spec is what lets external users write their own readers/datasets (a persona we explicitly serve) and pins the JAXTPC-writer ↔ pimm-data-reader contract. Treated as stable, so no version-negotiation logic. NOT co-located as JAXTPC code — that would couple pimm-data to a producer package. |
| **Governance** | **Permanent fork; you/Sam decide.** | The OmarAlterkait forks are the home. Optimize for the group's needs; do not constrain design to match youngsm/pimm upstream conventions. |

### Where we diverge from Sam's letter (on the record)

Sam's proposal is directionally right (standalone package; models/packing/recipes in `pimm`)
and its **goal is already largely met** by the de-fork. Three corrections from the audit:

1. **"Packing/Point lock-in" is already solved.** `Point` and all packing
   (serialize/sparsify/octree) live entirely in `pimm`'s model layer
   (`pimm/models/utils/structure.py`); `pimm-data` emits **neutral dicts** and the
   `Point(input_dict)` conversion happens inside `model.forward()`. `collate` is also
   neutral (offset/cumsum concat, no `Point`) and opt-in. So a consumer who wants padding
   instead of offsets just supplies their own collate — the Dataset output is already
   packing-agnostic. Nothing needs to move to fix this.
2. **Transforms are NOT "mostly tied to individual pretraining methods."** ~42/50 are generic;
   only ~5 are genuine SSL recipes. Moving all transforms to `pimm` would strand generic
   preprocessing (NormalizeCoord/GridSample/LogTransform) behind a `pimm` dependency —
   reintroducing the exact lock-in we're trying to avoid.
3. **The Dataset is not thin glue.** "Just interface the readers" understates the joint-index +
   nested-assembly work. A `pimm`-side custom Dataset would either re-implement that
   (duplication of freshly-hardened code) or be a thin subclass of `pimm-data`'s base — which
   is what `JAXTPCDataset(ShardEventDataset)` already is. The "model-ingestible shaping" Sam
   wants in `pimm` already exists as the transform-recipe configs + `Point(input_dict)`.

**Net:** keep the Dataset and collate in `pimm-data`; move only the genuinely recipe-specific
transforms; the rest of Sam's structure is already in place.

---

## 2. Registry mechanics (decided: single shared registry)

**Decision: keep ONE shared `TRANSFORMS` registry owned by `pimm-data`. When the SSL transforms
move to `pimm`, they register into that same registry object via an explicit, eager import at
`pimm` package init (the pattern `detector_transforms.py` already uses internally). Ownership is
enforced by CODE LOCATION (the classes physically live in `pimm`'s source tree), not by registry
identity.**

### Why not a "pure" pimm-data registry + pimm child registry

Verified against the vendored `Registry` (`_registry.py:138-151`) and `Compose`
(`transform.py:2371-2382`):

- `Registry.get()` resolves an **unscoped** key (`"MultiViewGenerator"`) against **only its own
  `_module_dict`**. Parent/child fallback fires **only for scoped keys** (`"pimm.MultiViewGenerator"`).
- `Compose` is hardwired to `pimm-data`'s module-global `TRANSFORMS`.

So a child-registry design would force **either** scoping every SSL transform name in every config
**or** a duplicate `pimm`-side `Compose` bound to the child registry. Both are friction with no
real payoff, because:

- **The "purity" the child design buys is unobservable.** The registry is populated by *imports*.
  An external user who does `import pimm_data` never triggers `pimm`'s registrations, so
  `pimm_data.TRANSFORMS` is **already pure in their process**. The only process where `pimm` names
  appear in the registry is one that imported `pimm` — exactly where you *want* unified resolution.
- **The registry is a runtime lookup table, not the ownership boundary.** Conflating "which object
  holds the name" with "who owns the code" is a category error. Code location is the real boundary
  and it's clean under the shared-registry design.

**Refinement vs. status quo:** make `pimm`'s registration **explicit and eager** (an import in
`pimm`'s package/`datasets` init), not a lazy side-effect that could be missed or tree-shaken, so a
config referencing `MultiViewGenerator` can never hit an unregistered-name error due to import order.

### Considered and rejected
- *Pure pimm-data registry + pimm child w/ fallback* — breaks unscoped config resolution; needs
  config-wide scoping or a duplicate Compose; relies on a scoped-resolution path configs don't use.
- *Fully separate, no fallback* — maximal friction for config authors.

---

## 3. Transform disposition

Tiers under the external-first lens. (Audit: two independent agents — one semantic, one usage-map —
cross-checked; adjudications noted.)

### KEEP in pimm-data

**Core IO / nested-dict infra / label & readout decoding (Tier A):**
`Collect`, `Copy`, `Update`, `ToTensor`, `ApplyToStream`, `Compose`/`index_operator` (helpers),
`GridSample`, `PDGToSemantic`, `RemapSegment`, `AggregateSensorHits`.

> **Adjudication:** the usage-map agent flagged `AggregateSensorHits`, `RelativeLogNormalize`,
> `Update`, and even `PDGToSemantic` as "move/drop — only seen in pretrain configs." **Overridden.**
> That signal is an artifact of LUCiD currently having *only* pretraining configs; these are
> detector-data **decoding** (PMT hit aggregation, PMT-time normalization, PDG→class) and belong in
> the data layer by nature. Lesson: "referenced only in pretrain configs" ≠ "pretraining-specific."

**Generic preprocessing (Tier B1):**
`NormalizeCoord`, `LogTransform`, `RelativeLogNormalize`, `MomentumTransform`, `PositiveShift`,
`CenterShift`, `PointClip`, `SphereCrop`, `LocalCovarianceFeatures`.

**Canonical training augmentation (Tier B2 — kept per the batteries-included decision):**
`RandomRotate`, `RandomRotateTargetAngle`, `RandomFlip`, `RandomScale`, `RandomShift`,
`RandomJitter`, `RandomDropout`, `ShufflePoint`, `ConditionalRandomTransform`,
`EnergeticTranslation`, `EnergyJitter`, `MultiplicativeRandomJitter`, `ElasticDistortion`,
`SetRandomValue`, `HardExampleCrop`.

> **Rationale:** these depend only on `coord`/`energy`, have zero recipe/`pimm` coupling, are tiny,
> and `RandomRotate`/`GridSample` are used in *both* supervised and pretrain configs. Keeping them
> lets `pimm-data` train a model standalone — the point of an external-first batteries package.

### FIX, then keep (placed as augmentation)
- `RandomDrop` — **silent no-op bug**: `data_dict[key][idx][:] = value` writes into a fancy-index
  *copy* (`transform.py:2356`). Repair to assign back, then keep.
- `ClipGaussianJitter` — **+3-mean bug**: `self.mean = np.mean(3)` → scalar `3.0`, not a zero-mean
  vector (`transform.py:753`). Repair to zero-mean, then keep.

### MOVE to pimm

**SSL pretraining recipes (Tier C) — verified zero internal coupling (config-string-only):**
`ContrastiveViewsGenerator`, `MultiViewGenerator`, `MixedScaleGeometryMultiViewGenerator`,
`HierarchicalMaskGenerator`, `ComputeAnchors` — and the backing **`anchors.py`** module.

- Per the anchors decision: move `anchors.py` wholesale and **drop `compute_anchors` /
  `ANCHOR_DEFAULT_CFG` from `pimm-data`'s public `__init__.py`** (breaking-but-intended API change;
  `transform.py` already imports it under `try/except`, so removal degrades gracefully). If a
  reconstruction use-case for standalone endpoint/Bragg/branch mining ever appears, re-extract a
  focused `pimm-data` geometry utility *then* (YAGNI).

**Task-head target generator:**
`InstanceParser` — emits a detection/instance-seg `bbox` target `(center3, size3, θ, class)` tied to
a specific head. Move now; its generic centroid/axis half can be re-extracted as a `pimm-data`
geometry util later if a real need appears (same YAGNI stance as anchors). *Med confidence.*

### DROP
- **Structurally-dead color/RGB (8):** `NormalizeColor`, `ChromaticAutoContrast`,
  `ChromaticTranslation`, `ChromaticJitter`, `RandomColorGrayScale`, `RandomColorJitter`,
  `HueSaturationTranslation`, `RandomColorDrop`. Verified: **no dataset or config ever produces a
  `color` key** — detector data is coord/energy/time, never RGB. These can never fire. This is
  removing *structurally impossible* code, not the "delete merely-unused functions" we avoid.
- **Domain-inappropriate footgun:** `CropBoundary` — hardcodes ScanNet wall/floor class ids `{0,1}`;
  on detector data it would silently drop real particle classes 0/1. Worse than dead. *Drop on
  correctness grounds; override if you have a detector use for it.*

### Boundary note: instance-mix in collate
`point_collate_fn(mix_prob>0)` (`collate.py`) bakes a two-sample instance-mixup (a training
augmentation) into a core IO function. Per batteries-included, **keep collate including the
`mix_prob` branch** (it defaults to 0 and is opt-in), but note it is recipe-flavored logic living in
the data layer — acceptable, documented, not moved.

### Disposition counts
KEEP ≈ 24 (A + B1 + canonical B2) · FIX-and-keep 2 · MOVE 6 (5 SSL + InstanceParser, + `anchors.py`)
· DROP 9 (8 color + CropBoundary).

---

## 4. Scope & naming (recommendation)

**Scope this round = data-loading boundary only** (readers / datasets / transforms / collate /
registry). **Submissions/eval utilities are explicitly out** — confirmed not needed in the near
future, so we do not broaden `pimm-data` into a general detector toolkit now.

**Rename: warranted, but deferred to its own focused PR — do not bundle it with the transform work.**
- *Why warranted:* the name "pimm-data" brands the standalone package as an appendage of `pimm`,
  which mildly contradicts the external-first, "don't lock people into pimm" positioning. The right
  name is domain-oriented (detector/event data), independent of `pimm`.
- *Why deferred & isolated:* a package rename touches the repo name, the import name (`pimm_data` →
  new name), the submodule path + `.gitmodules`, the editable-install line, and docs — broad mechanical
  churn that would re-disturb the just-stabilized de-fork and make the transform-restructure diff
  unreviewable if mixed in. (Note: config `type="ClassName"` strings are class names, not the
  package name, so they are unaffected by a rename.) There are no external dependents yet besides
  `pimm` itself, so the rename cost is low and roughly flat in the near term — no urgency to bundle.
- *Name:* decide when you do the rename PR; happy to brainstorm options then.

---

## 5. Distribution, dependencies & schema (detail)

**Dependency tiering (already substantially true).** Verified torch-import graph: `readers/*`,
`_base.py`, `_shard_meta.py`, `_joint_index.py`, `_label_decorate.py` are **torch-free**; only
`transform.py`, `collate.py`, and `anchors.py` import torch. Once `anchors.py` moves to `pimm`
(§3), pimm-data's torch surface is exactly **transforms + collate + the `ShardEventDataset` base**
(`torch.utils.data.Dataset`). Action: *preserve* this property (a CI/lint guard asserting
`pimm_data.readers` imports with no torch in `sys.modules`) and **promote the reader classes +
the joint-index helper to the top-level public API** as the documented "bring-your-own-framework"
path. Do **not** add a `[torch]` extra / lazy-`__init__` guarding now (YAGNI).

**Schema spec.** Author a canonical, versioned-by-tag doc describing the on-disk HDF5 layout that
JAXTPC `production/save.py` writes and pimm-data reads, covering per file type
(sensor / step / hits / labl): group structure, dataset names + dtypes, attributes, the
delta/CSR encodings, codec expectations (blosc-zstd default → needs `hdf5plugin`), and the 1-based
group-id convention. Lives in `pimm-data/docs/` (consumer side), cross-linked from JAXTPC. Optional:
stamp a `schema_version` attribute at write time and assert it in the readers for loud-fail-on-drift.

## 6. Planned workstreams (NOT executed — for later PRs)

Sequenced so each PR is small and reviewable; do them in `pimm-data` first, then bump the `pimm`
submodule pin.

1. **PR-A (pimm-data): drops.** Delete the 8 color transforms + `CropBoundary`; remove any config
   references. Lowest risk; shrinks surface. Move the two SSL-exercising tests' color/boundary
   assertions if any.
2. **PR-B (pimm-data): fix the two buggy augs** (`RandomDrop`, `ClipGaussianJitter`) + add a
   regression test for each (they currently have none / are no-ops).
3. **PR-C (pimm): receive the SSL transforms + `anchors.py`.** Create `pimm`'s transforms module,
   move the 5 SSL transforms + `InstanceParser` + `anchors.py`, register them eagerly into the
   shared `pimm_data.TRANSFORMS` at `pimm` package init. **Test move (corrected):** only
   `test_transform_v3_vertex.py::test_mixed_scale_multiview_smoke` (`:110–131`) moves — drop the
   import at `:17` in the staying file. `test_transform_merges.py` only *mentions* MultiViewGenerator
   in a docstring and does NOT move. See `IMPLEMENTATION-boundary-refactor.md` §1.1 / PR-C.
4. **PR-D (pimm-data): drop `compute_anchors`/`ANCHOR_DEFAULT_CFG` from public `__init__.py`** once
   PR-C lands (the only consumer is gone). Confirm `transform.py`'s `try/except` import is removed
   with `ComputeAnchors`.
5. **PR-E (pimm): submodule pin bump** to the post-PR-A/B/D `pimm-data` tag; run the LUCiD + JAXTPC
   real-data validation to confirm parity.
6. **PR-F (pimm-data): public surface + schema** *(independent of A–E; can land in parallel).*
   Promote the reader classes + joint-index helper to the top-level `__init__` public API; add the
   CI guard that `pimm_data.readers` imports torch-free; author the on-disk schema spec doc (§5) and
   cross-link it from JAXTPC.
7. **Tag a release** once A–F land; switch `pimm`'s submodule pin to the tag rather than a raw SHA.
8. **(Later, separate) Rename PR.** Out of scope for the above sequence.

**Verification gate for each:** synthetic suite + real-data JAXTPC + real-data LUCiD (cjesus WAND
root) must pass, matching the de-fork validation baseline.

---

## 7. Open / deferred items
- Package **rename** (warranted; separate PR; name TBD).
- **Submissions/eval utilities** — deferred indefinitely (not needed soon).
- Whether `LocalCovarianceFeatures` is worth keeping long-term hinges on whether any model actually
  consumes `local_shape`/`local_eigvals`; kept for now (generic, no harm), revisit if it stays
  unconsumed.
- *(Promoting the reader classes to the public API is now a decision — see §5 / PR-F.)*

### Lower-tier decisions still open (not yet adjudicated)
- **Generic Pointcept datasets** (`DefaultDataset`/`ConcatDataset`, npy-folder loaders): keep under
  external-first, or drop as non-detector cruft? (Prior session kept them — "others will use the
  loaders"; worth re-confirming now that the package is detector-external-first.)
- **Joint-index semantics**: the index currently *intersects* present events across modalities
  (drops events missing from any). Expose *union-with-masking* as a public knob, or keep
  intersection-only?
- **CI/test strategy for moved transforms**: the SSL transforms' tests move to `pimm`, which needs a
  different runtime image (flash_attn/spconv) — confirm those specific transform tests run without
  the heavy model deps, or gate them.
- **Backward-compat window** for the breaking `compute_anchors` public-API removal (any out-of-repo
  importers?).
- **Dataset output dict as a documented contract**: should the nested-dict output keys
  (`step`/`hits`/`sensor`/`labl` → `coord`/`energy`/…) be a documented, tag-versioned interface like
  the on-disk schema?
