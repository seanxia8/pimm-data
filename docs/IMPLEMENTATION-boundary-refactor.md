# Implementation manifest: pimm-data ↔ pimm boundary refactor

**Companion to:** `ADR-pimm-data-package-boundary.md` (the decisions). This document is the
**complete, file-by-file change list** — every edit, with line numbers, ordering, tests, and risks —
produced from a 5-agent audit of both repos. Where the audit corrected or refined the ADR, it is
called out in §1.

**Repos.** Source of truth = standalone `pimm-data` (`/sdf/group/neutrino/omara/pimm-work/pimm-data/`).
Consumer = `pimm` (`/sdf/group/neutrino/omara/pimm-work/particle-imaging-models/`); its vendored
`libs/pimm-data/` is the submodule checkout — never edit it directly, it follows the pin.

**Status:** PLAN ONLY — nothing executed. Two open decisions in §2 gate PR-G / the guard wording.

---

## 1. Corrections & refinements to the ADR (found by the audit)

These supersede the corresponding ADR statements:

1. **Test-move claim was wrong (ADR §3/§6 PR-C).** `tests/test_transform_merges.py` mentions
   `MultiViewGenerator` **only in its module docstring** (`:3`); all 14 tests exercise *kept*
   transforms. **It does NOT move.** In `tests/test_transform_v3_vertex.py`, only **one** of 8 test
   functions uses a moved transform — `test_mixed_scale_multiview_smoke` (`:110–131`). Only that
   function moves; the import at `:17` is dropped from the staying file. 5 of the 6 moved transforms
   + all of `anchors.py` have **no existing tests** (coverage gap — add smoke tests during the move).

2. **Torch-importer count was undercounted (ADR §1 row).** Not just `transform.py`/`collate.py`/
   `anchors.py` — the four Dataset bases also `import torch` (`pilarnet.py:14`, `defaults.py:15`,
   `multimodal.py:32`, `_dataset_base.py:28`, all `from torch.utils.data import Dataset`). **After
   `anchors.py` moves, pimm-data's torch importers are: `transform.py`, `collate.py`, `pilarnet.py`,
   `defaults.py`, `multimodal.py`, `_dataset_base.py`.** (ADR §5 already named the Dataset base; the
   §1 summary row is the inaccurate one.)

3. **The torch-free guard as worded is impossible without restructuring `__init__.py` (ADR §5).**
   Proven 3 ways: `import pimm_data.readers` (or any `pimm_data.*` submodule) first runs the eager
   `pimm_data/__init__.py`, whose line 19 `from .transform import …` pulls torch. So a guard that
   "imports `pimm_data.readers` with no torch in `sys.modules`" cannot pass while `__init__.py` is
   eager — and ADR §5 forbids the lazy-`__init__` change that would make it pass. **Resolution (§2.B):
   the real invariant is "the reader/index/decode *source* imports no torch" — guard that statically
   (AST), which is durable and always valid; the dynamic import-under-ban becomes an `xfail` marker.**
   Also: promoting readers to the public API is an **ergonomics** win (`from pimm_data import
   JAXTPCEdepReader`), **not** a torch-free *install* path — you still need torch installed to
   `import pimm_data`. The ADR §5 "bring-your-own-framework entry point" framing is corrected to
   "framework-neutral reader *code* (returns numpy)", which matches the "torch stays required"
   decision (torch is required; the reader code just carries no torch semantics).

4. **`_base.py` vs `_dataset_base.py` (ADR §5 nit).** The torch-bearing Dataset base is
   `_dataset_base.py`. The torch-**free** files are `readers/_base.py`, `_shard_meta.py`,
   `_joint_index.py`, `_label_decorate.py`. Don't conflate `readers/_base.py` (torch-free) with
   `_dataset_base.py` (torch). ADR §5 cites `_base.py`; it means `_dataset_base.py`.

5. **Version is duplicated** — `pyproject.toml:3` **and** `src/pimm_data/__init__.py:59`
   (`__version__ = "0.1.0"`). A release must bump **both** (no dynamic linkage). No CHANGELOG exists.

6. **"Tag-pinning" a submodule is still SHA-pinning underneath.** The gitlink always records a commit
   SHA; `.gitmodules` never stores a tag. "Pin a tag" = checkout the tag's commit in the submodule,
   stage the gitlink, and *document* the tag. Do **not** add `branch = <tag>` to `.gitmodules`
   (that's for `--remote` branch tracking and reintroduces drift). Verify with
   `git -C libs/pimm-data describe --tags --exact-match HEAD`. First release must create the tag
   **before** repinning (no tags exist yet).

7. **`ConcatDataset` has a live import** (`pimm/datasets/dataloader.py:9`) so a naive drop crashes
   `import pimm.datasets`. But its consumer chain is **already dead**: `MultiDatasetDataloader` ←
   `MultiDatasetTrainer.build_train_loader` (`pimm/engines/train.py:404–416`), which imports
   `from pointcept.datasets import MultiDatasetDataloader` (`train.py:405`) — **`pointcept` does not
   exist in the repo** (stale pre-de-fork name), and no config selects `MultiDatasetTrainer`/
   `ConcatDataset`. Detector multi-source training uses `MultiModalEventDataset`, not `ConcatDataset`.
   So dropping `ConcatDataset` means also removing the dead `MultiDatasetDataloader`/`MultiDatasetTrainer`
   path. See §2.A + PR-G.

8. **Latent schema gaps (not blocking, discovered while writing the spec):** the writer stamps a
   per-event `source_event_idx` **attribute** that no reader consumes, while the reader *prefers* a
   `config/source_event_idx` **dataset** the writer never emits (it falls back to
   `global_event_offset + event_num`, which works). Documented in the schema spec; optional cleanup.

---

## 2. Decisions (resolved)

### 2.A — Generic datasets `DefaultDataset` / `ConcatDataset` → **DROP BOTH** (decided)
Evidence (Agent 5): **`DefaultDataset` is fully dead** in both repos — export-only, nothing
instantiates it, no config references it, `ShardEventDataset` no longer inherits it
(`_dataset_base.py:11–15,36`). **`ConcatDataset`** is imported by `dataloader.py:9` but the whole
`MultiDatasetDataloader`/`MultiDatasetTrainer` path behind it is already broken (stale `pointcept`
import) and superseded by `MultiModalEventDataset`. **Decision: drop both, including the dead
`MultiDatasetDataloader`/`MultiDatasetTrainer` machinery** (PR-G, now ungated).

### 2.B — Torch boundary enforcement → **static import-contract test, no xfail, no `__init__` restructure** (decided; best-practice)
The literal ADR goal ("`pimm_data.readers` imports with no torch") cannot hold while `__init__.py` is
eager, and a torch-free *install/import* is a **deliberate non-goal** ("torch stays a required dep").
So:
- **Enforce the architectural invariant** *"the framework-neutral IO layer
  (`readers/*`, `_joint_index.py`, `_shard_meta.py`, `_label_decorate.py`) must not import torch"* via a
  **static AST test in the existing pytest suite** (parses each IO file, asserts no `torch` import,
  direct or via a torch-bearing sibling). This runs today and catches creep — the genuine value is a
  framework-neutral data path (readers return numpy; JAX/numpy users consume them).
- **Drop the `xfail(strict=True)` dynamic "import under torch-ban" test.** It implies torch-free import
  is a future aspiration; it isn't. Replace with a one-line comment stating the rationale (eager
  `__init__` + torch required by design).
- **Best-practice tool, when CI exists:** `import-linter` (declarative forbidden-import contract) is the
  idiomatic enforcement — adopt it once either repo has a CI/pre-commit runner (neither does today, so
  an unrun contract would be theater; the in-suite AST test is what actually runs now).
- **Documented upgrade path (NOT now):** if a torch-free *install* is ever wanted, the principled move
  is lazy `__init__` via PEP 562 `__getattr__` (scipy/sklearn pattern) + a `[torch]` extra — makes
  `import pimm_data.readers` genuinely torch-free without splitting packages. Correctly deferred under
  the YAGNI call.

---

## 3. Change manifest, PR by PR

Sequencing matches ADR §6; each PR is independently reviewable. PR-A/B/G/F-schema are pimm-data-only;
PR-C touches both; PR-D/E are the cross-repo glue.

### PR-A — Drops (pimm-data) `transform.py` + test
Delete each class **individually** (kept transforms are interleaved — do NOT delete ranges en masse):

| Transform | Delete lines |
|---|---|
| `NormalizeColor` | 292–297 |
| `ChromaticAutoContrast` | 770–788 |
| `ChromaticTranslation` | 791–802 *(incl. trailing blank; leaves `EnergeticTranslation`@803)* |
| `ChromaticJitter` | 816–831 *(incl. trailing blanks; leaves `EnergyJitter`@832)* |
| `RandomColorGrayScale` | 851–880 |
| `RandomColorJitter` | 883–1064 |
| `HueSaturationTranslation` | 1067–1132 |
| `RandomColorDrop` | 1135–1149 |
| `CropBoundary` | 1525–1532 |

**DO NOT TOUCH** the kept transforms sitting between them: `EnergeticTranslation` (803–813),
`EnergyJitter` (832–850).
Also:
- **Remove orphaned `import numbers`** (`transform.py:11`) — only used by dropped `RandomColorJitter`.
- **Keep `import torch`** (`transform.py:18`) — used pervasively by kept transforms.
- **Update the count assertion**: `tests/test_transforms.py:48–51` asserts `len(TRANSFORMS) >= 48`;
  actual after drop = 37 (transform.py) + 4 (detector_transforms) = **41**. Change to `>= 39` and fix
  the stale `# 45 …` comment at `:49`.
- All other references to dropped names are self-internal (color helpers used only by other dropped
  color classes — Agent 1 §4); nothing else in either repo references them; **zero live configs** use
  them (only 4 *commented-out* `RandomDrop` lines in `configs/lejepa/...`, and `RandomDrop` is kept).

### PR-B — Bug fixes (pimm-data) `transform.py` + regression tests
- `ClipGaussianJitter` (`:753`): `self.mean = np.mean(3)` → `self.mean = np.zeros(3)`.
- `RandomDrop` (`:2356`): `data_dict[self.key][idx][:] = self.value` →
  `data_dict[self.key][idx] = self.value` (assign back through the fancy index, not into a copy).
- Add regression tests to `tests/test_transforms.py` (Agent 1 §3 has ready-to-paste tests): assert
  `ClipGaussianJitter(scalar=0)` leaves coords exactly 0 and `mean.shape == (3,)`; assert `RandomDrop`
  actually zeroes the targeted rows and preserves the rest.

### PR-C — Move SSL transforms + `anchors.py` to pimm (both repos)
**pimm-data deletions** (`transform.py`), do in the SAME change to avoid double-registration
(`_registry.py:172` raises on duplicate name):
- `ContrastiveViewsGenerator` 1535–1557; `MultiViewGenerator` 1560–1747;
  `MixedScaleGeometryMultiViewGenerator` 1750–1893 *(subclass of MultiViewGenerator — must move with
  it)*; `ComputeAnchors` 1896–1923; `InstanceParser` 1926–2058; `HierarchicalMaskGenerator`
  2142–2342.
- Delete the `try/except` anchors guard `transform.py:30–35`.
- Move `src/pimm_data/anchors.py` → `pimm/datasets/anchors.py` (self-contained: numpy/torch/scipy
  only; no intra-pimm-data imports).

**pimm additions:**
- New file `pimm/datasets/transforms.py` (mirrors pimm-data's `detector_transforms.py` pattern).
  Header:
  ```python
  from typing import Optional
  import copy
  import numpy as np
  from scipy.spatial import cKDTree
  from pimm_data.transform import TRANSFORMS, Compose   # shared registry + Compose stay in pimm-data
  from .anchors import compute_anchors, ANCHOR_DEFAULT_CFG
  ```
  then paste the 6 classes, each keeping its `@TRANSFORMS.register_module()` decorator. (No
  pimm-data helper is stranded — verified zero references to `index_operator`/rotation helpers/etc.
  inside the moved code. Only `Compose` + `TRANSFORMS` are external, and both stay in pimm-data and
  are imported.)
- **Eager registration** — add to `pimm/datasets/__init__.py` immediately after the
  `from pimm_data import (...)` block (after `:20`), before `MultiDatasetDataloader`:
  ```python
  # Register pimm-owned SSL transforms into the shared pimm_data.TRANSFORMS.
  from . import transforms as _pimm_transforms  # noqa: F401
  ```
  **Order is load-bearing:** must come after the `pimm_data` import (registry must exist) and inside
  `pimm/datasets/__init__.py` (every `build_dataset`/`Compose` consumer routes through it, so
  registration is guaranteed before any config resolves `type="MultiViewGenerator"`). A lazy/in-function
  import or placing it in the empty `pimm/__init__.py` would let configs hit a `KeyError` at
  build time (`_registry.py:54`). This is the ADR §2 "explicit & eager" requirement.

**Tests:**
- Move `test_mixed_scale_multiview_smoke` (`tests/test_transform_v3_vertex.py:110–131`) to a new pimm
  test (`tests/test_ssl_transforms.py`); drop `MixedScaleGeometryMultiViewGenerator` from the import
  at `tests/test_transform_v3_vertex.py:17` in the staying file.
- `test_transform_merges.py` stays (docstring-only mention; optional: scrub `:3`).
- Add a registration-smoke test in pimm asserting all 6 names resolve via
  `pimm.datasets.TRANSFORMS.get(...)` (extends `tests/test_shim.py:24` pattern).
- Add minimal smoke tests for the 5 currently-untested moved transforms while the code is fresh.
- Heavy-dep check passed: none import flash_attn/spconv/torch_scatter — moved tests run in a **light
  CPU image** (torch+numpy+scipy).

### PR-D — Remove `compute_anchors` from pimm-data public API (pimm-data)
After PR-C lands (consumer gone):
- `src/pimm_data/__init__.py`: delete `:36–37` (comment + `from .anchors import compute_anchors,
  ANCHOR_DEFAULT_CFG`) and the `__all__` entries `:55–56`.
- Blast radius: **zero in-repo importers** outside `transform.py`/`__init__.py` (Agent 5 §D). Breaking
  for any out-of-repo `from pimm_data import compute_anchors` — none known. If a compat window is
  wanted, leave a lazy re-export with a `DeprecationWarning` for one tagged release instead of a hard
  delete (optional; ADR §7).

### PR-E — Public surface + torch-free guard (pimm-data)
- Promote readers + joint-index to top-level API. In `src/pimm_data/__init__.py`, after `:37`:
  ```python
  from .readers import (
      JAXTPCEdepReader, JAXTPCSensorReader, JAXTPCHitsReader, JAXTPCLablReader,
      LUCiDEdepReader, LUCiDSensorReader, LUCiDHitsReader, LUCiDLablReader,
  )
  from ._joint_index import build_joint_index
  ```
  and append those 9 names to `__all__` (after `:56`). No name collisions; torch-neutral (the readers/
  joint-index closure imports only stdlib+numpy+h5py).
- Torch-free guard test `tests/test_torch_free_readers.py` — per §2.B: a parametrized **static AST
  check** that none of `_joint_index.py`, `_shard_meta.py`, `_label_decorate.py`, `readers/*.py` import
  torch (directly or via a torch-bearing sibling). Durable, runs in the existing suite. **No `xfail`
  dynamic test** (torch-free *import* is a non-goal); add a one-line comment stating the rationale.
  Migrate to an `import-linter` contract if/when real CI lands.
- *(Optional, separate)* extend the pimm shim (`pimm/datasets/__init__.py`) to re-export readers if
  you want `pimm.datasets.JAXTPCEdepReader` — not required; PR-E is pimm-data-only.

### PR-F — Schema doc only (pimm-data; **no `schema_version`, no JAXTPC change** — decided)
- New doc `docs/SCHEMA-on-disk.md` (full spec drafted by Agent 4: sensor/edep/hits/labl group
  structure, dataset names/dtypes, the delta + CSR encodings, codec/`hdf5plugin`, 1-based group ids,
  local↔global coords, wire/pixel asymmetry). Cross-link from JAXTPC's README/CLAUDE.
- **Decision: skip the `schema_version` attribute entirely.** Treat the schema as stable; if it ever
  changes, update this doc. No `save.py`/`make_labl.py` stamp, no reader assert, no JAXTPC repo change
  this round. (The latent schema gaps in §1.8 are documented in the spec as informational, not fixed.)
- *(Optional)* document the in-memory **output-dict contract** (top-level `edep/hits/sensor/labl`
  sub-dicts → keys) — already in module docstrings `jaxtpc.py:14–35`, `lucid.py:12–26`; copy up to the
  doc if you want it documented alongside the on-disk spec.

### PR-G — Drop generic datasets *(decided — §2.A)*
- `DefaultDataset` (fully dead): delete class `defaults.py:26…`; remove exports `__init__.py:22,46`;
  remove re-export `pimm/datasets/__init__.py:17,32`. (`test_shim.py` does not assert it — safe.)
- `ConcatDataset` + dead loader: remove class `defaults.py:194…` and exports; **first** remove the
  live import + the dead machinery: `pimm/datasets/dataloader.py:9,32` (`ConcatDataset` import/type
  hint) and the `MultiDatasetDataloader`/`MultiDatasetTrainer` path (`train.py:404–416`, stale
  `pointcept` import `:405`). Confirm no config selects them.

### PR-H — Distribution (release + repin) *(after A–G land)*
- Bump version in **both** `pyproject.toml:3` and `src/pimm_data/__init__.py:59`.
- `git tag -a vX.Y.Z && git push origin vX.Y.Z` (tag the pimm-data release commit).
- Repin pimm submodule to the tag: `git -C libs/pimm-data fetch --tags && checkout vX.Y.Z`;
  `git add libs/pimm-data && git commit`. **Do not** add `branch=` to `.gitmodules`. Verify
  `git -C libs/pimm-data describe --tags --exact-match HEAD`.
- Update the README install line (add the `pip install "pimm-data @ git+…@vX.Y.Z"` form) and the
  `environment.yml` comment to name the tag.
- Run the verification gate (below).
- **(Later, separate) Rename PR** — out of scope.

---

## 4. Cross-cutting risks, ordering, verification

- **Ordering:** PR-A, PR-B, PR-E, PR-F-schema, PR-G are independent pimm-data changes and can land in
  any order. PR-C must delete-from-pimm-data and add-to-pimm in one logical change (double-registration
  guard). PR-D after PR-C. PR-H (tag + repin) last.
- **Double-registration** (`_registry.py:172`, `force=False`): the same transform name cannot live in
  both packages at once — PR-C's deletions and additions are atomic.
- **Eager-registration order** (PR-C): the #1 silent-failure risk; mitigated by the explicit import in
  `pimm/datasets/__init__.py` + a registration-smoke test.
- **`ConcatDataset` live import** (PR-G): remove the importer before the class or `import pimm.datasets`
  breaks.
- **No test CI exists in either repo.** pimm-data has zero CI; pimm has only a Docker-build workflow
  (`.github/workflows/docker.yml`, triggers on `v*` tags but runs no pytest). The moved SSL tests have
  no automated home today — running them is currently manual/local. (Worth a follow-up: a minimal
  pytest workflow in each repo.)
- **Verification gate for every PR** (matches the de-fork baseline): pimm-data synthetic suite (`pytest`,
  ~0.5s on synthetic v3 fixtures) + real-data JAXTPC (`JAXTPC_DATA_ROOT=… pytest`) + real-data LUCiD
  (`LUCID_DATA_ROOT=/sdf/data/neutrino/cjesus/DORAEMON/WAND/SK_like/config_… pytest`) + pimm
  `tests/test_shim.py`.

## 5. Lower-tier items (facts gathered; no action unless you want it)
- **Joint-index union knob** (ADR §7): the intersection is `_joint_index.py:78–83` (`common &=`);
  both detector datasets route through `ShardEventDataset._build_joint_index`
  (`_dataset_base.py:104–110`), so a single `join='intersect'|'union'` keyword there + at `:81` covers
  JAXTPC and LUCiD. Union would emit presence masks instead of trimming.
- **Output-dict contract** documented at `jaxtpc.py:14–35` / `lucid.py:12–26` (see PR-F optional).
- **Schema cleanups** discovered (§1.8): the per-event `source_event_idx` attr (written, unread) and
  the `config/source_event_idx` dataset (read-preferred, never written). JAXTPC-side; non-blocking.
- **Stale pimm-data docs**: `docs/impl/02_dataset_base.md`, `docs/DESIGN.md`, `docs/ROADMAP.md` still
  describe the old `DefaultDataset`/`TestModeMixin` inheritance; `_dataset_base.py` is authoritative.
