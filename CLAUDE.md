# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install -e .                          # editable install (extras: [test])
pytest                                    # full suite (uses synthetic fixtures)
pytest tests/test_jaxtpc.py::test_name    # single test

# Run against real production datasets instead of synthetic fixtures:
JAXTPC_DATA_ROOT=/path/to/jaxtpc pytest
LUCID_DATA_ROOT=/path/to/wc    pytest
JAXTPC_PIXEL_DATA_ROOT=/path/to/pixel pytest tests/test_jaxtpc_pixel.py
```

There is no linter/formatter configured and no build step beyond `pip install -e .`.

## Architecture

This package is a thin data layer between sharded HDF5 detector simulation
output and PyTorch training loops. It does **not** own physics models — it
returns nested numpy dicts that transforms reshape into model-ready tensors.

### Three layers

```
readers/ (HDF5 → flat dict)  →  jaxtpc.py / lucid.py (flat → nested)  →  transform.py (nested → model batch)
```

1. **Readers** (`readers/{jaxtpc,lucid}_{step,sensor,hits,labl}.py`) — one
   per `(detector, modality)` pair. Each opens a glob of shard files,
   builds an event index, and exposes `read_event(idx) → flat dict`. They
   share no base class but follow the same protocol.
2. **Datasets** (`jaxtpc.py`, `lucid.py`, `pilarnet.py`) — subclass
   `DefaultDataset` (in `defaults.py`, vendored from Pointcept). They wire
   together per-modality readers, join `labl` dimension tables onto
   `step`/`hits` point clouds, and emit a **nested** dict keyed by modality
   (`{'step': {...}, 'hits': {...}, 'labl': {...}, 'bridges': {...}}`).
   `get_data(idx)` returns raw numpy; `__getitem__` runs the transform
   pipeline.
3. **Transforms** (`transform.py`, `detector_transforms.py`) — Pointcept-
   style registry of point-cloud augmentations. The pipeline ends with
   `Collect`, which scopes to one modality and converts numpy → torch.

### Registry pattern (vendored from mmcv/Pointcept)

`_registry.py` provides `Registry`. Two top-level registries:

- `TRANSFORMS` in `transform.py` — populated via `@TRANSFORMS.register_module()`
- `DATASETS` in `builder.py` — populated via `@DATASETS.register_module()`

Configs are plain dicts with a `type:` key: `dict(type='GridSample', ...)`,
`dict(type='JAXTPCDataset', ...)`. **Import order matters** — `__init__.py`
imports modules in a specific order so the decorator side-effects run
before anything reads the registry. Adding a new transform/dataset means
both decorating it and importing the module from `__init__.py`.

### Nested-dict output and `ApplyToStream` / `Collect`

Datasets emit `{'step': {...}, 'sensor': {...}, 'hits': {...}, 'labl': {...}}`.
There is **no bare `coord` at the top level**. Transforms that hardcode
`'coord'`/`'segment'` must be wrapped:

```python
dict(type='ApplyToStream', stream='step', transforms=[
    dict(type='GridSample', grid_size=0.5, mode='train'),
])
```

`Collect(stream='step', keys=[...], feat_keys=[...])` is the terminal
transform: it pulls a flat dict out of one modality and converts arrays
to torch tensors. **`Collect` must be last** — tensor output is required
for `DataLoader(num_workers > 0)` to use file-descriptor IPC instead of
pickling full arrays.

### Modality semantics and FK chain

| Modality | Role |
|---|---|
| `step`   | Per-Geant4-step truth deposits (3D) |
| `sensor` | Aggregated detector readout (raw per-channel signal) |
| `hits`   | Per-particle decomposition of `sensor` (instance-resolved) |
| `labl`   | Dimension tables (PDG, interaction, ancestry) |

`labl` has no point cloud — it joins onto `step`/`hits` to populate per-
point `segment` and `instance` columns. `('labl',)` alone and
`('sensor', 'labl')` are rejected (no instance separation to join against).

JAXTPC's FK chain (per volume, suffix `_v{N}`):
`step deposit → deposit_to_track → labl.track_ids → track_{label_key}`,
and `hits group → group_to_track → labl.track_ids → track_{label_key}`.
JAXTPC also returns a top-level `bridges` sub-dict containing
`group_to_track_v{N}`, `deposit_to_group_v{N}`, `qs_fractions_v{N}`.

### Detector-specific quirks

- **JAXTPC auto-detects readout type** (`wire` vs `pixel`) from the
  sensor/hits HDF5. Coord dimensionality and the per-volume plane filter
  (`volume=N`) depend on it — see `jaxtpc.py:_coord_keys` and the post-
  detection plane-filter block in `__init__`.
- **`label_key`** (JAXTPC) selects which `labl` column becomes `segment`:
  `'pdg'` (default), `'cluster'`, `'interaction'`, `'ancestor'`. Raw values
  flow through — use `RemapSegment(scheme='motif_5cls')` to convert to
  dense class indices.
- **LUCiD** requires `format_version >= 3` shards. `instance` is
  `particle_idx`; `segment` is `labl.particle.category`.

### Tests

`tests/conftest.py` provides session-scoped fixtures
(`jaxtpc_data_root`, `lucid_data_root`, `jaxtpc_pixel_data_root`) that
build synthetic v3 datasets via `pimm_data.testing` unless the matching
`*_DATA_ROOT` env var points at real shards. Tests that only make sense
on production data should use `@pytest.mark.real_data_only` — they
auto-skip when both env vars are unset (collection-time skip, see
`pytest_collection_modifyitems`).

`pimm_data.testing` regenerates a minimal but cross-modality-consistent
v3 fixture from pure numpy + h5py. Cross-modality invariants (e.g.
`labl.deposit_to_track[i] == hits.group_to_track[hits.deposit_to_group[i]]`)
are encoded there; preserve them when changing reader behaviour.
