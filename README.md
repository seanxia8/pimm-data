# pimm-data

Multimodal detector dataset loaders for particle-imaging ML workflows.

Reads simulation output produced by:

- **JAXTPC** — Liquid Argon TPC simulation (`edep` / `sensor` / `hits` / `labl` HDF5)
- **LUCiD** — Water Cherenkov / PhotonSim simulation (`edep` / `sensor` / `hits` / `labl` HDF5, `format_version >= 3`)

Datasets inherit from `torch.utils.data.Dataset` and return nested
`dict[str, dict[str, np.ndarray]]` samples (one sub-dict per modality).

## Quick start

### Install

```bash
pip install -e /path/to/pimm-data
```

### Load a JAXTPC dataset

```python
from pimm_data import JAXTPCDataset

ds = JAXTPCDataset(data_root="/path/to/jaxtpc_data", split="", modalities=("edep", "labl"))
sample = ds.get_data(0)          # numpy arrays, no transforms
print(sample["edep"].keys())     # coord, energy, volume_id, segment, instance, ...
```

### Load a LUCiD dataset

```python
from pimm_data import LUCiDDataset

ds = LUCiDDataset(data_root="/path/to/wc_data", modalities=("sensor",))
sample = ds.get_data(0)
print(sample["sensor"].keys())   # coord, energy, time, sensor_idx
```

### Training-ready example

```python
from torch.utils.data import DataLoader
from pimm_data import JAXTPCDataset, collate_fn

ds = JAXTPCDataset(
    data_root="/path/to/jaxtpc_data",
    split="",
    modalities=("edep", "labl"),
    label_key="pdg",
    transform=[
        dict(type="ApplyToStream", stream="edep", transforms=[
            dict(type="RemapSegment", scheme="motif_5cls"),
            dict(type="GridSample", grid_size=0.5, mode="train",
                 return_grid_coord=True),
        ]),
        dict(type="Collect", stream="edep",
             keys=("coord", "grid_coord", "segment"),
             feat_keys=("coord", "energy")),
    ],
)
loader = DataLoader(ds, batch_size=4, num_workers=4,
                    collate_fn=collate_fn, pin_memory=True)
batch = next(iter(loader))
```

> **`transform` takes a plain list of dicts**, not a `Compose` object.
> `Compose` wrapping is handled internally.

> **`get_data(idx)`** returns raw numpy arrays with no transforms.
> **`ds[idx]`** applies the full transform pipeline and returns tensors.

> **`split` parameter:** JAXTPC defaults to `split='train'`, looking in
> `data_root/edep/train/`. LUCiD defaults to `split=''`. If your files
> sit directly in `data_root/edep/` with no split subdirectory, pass
> `split=''`.

> **`label_key`** controls which labl column populates `segment`.
> Options: `'pdg'` (default), `'cluster'`, `'interaction'`, `'ancestor'`.
> Use `RemapSegment` to map raw values to task-specific class indices.

---

## Naming conventions

### Modality names (top-level dict keys, directory names, file tags)

| Modality | Meaning | Why this name |
|----------|---------|---------------|
| `edep`   | 3D energy deposits (one row per Geant4 simulation step) | Standard HEP abbreviation. |
| `sensor` | Aggregated detector readout (pixels, wires, or PMTs) | What the detector measures. |
| `hits`   | Per-particle charge attribution at sensor elements | Natural detector term ("PMT hits" in WC, "track hits" in TPC). |
| `labl`   | Metadata dimension tables (PDG, interaction, ancestry) | Short for "labels." |

These appear as directory names (`edep/`), HDF5 file tags
(`sim_edep_0000.h5`), and top-level keys (`data['edep']`).

### ML column names (per-point arrays inside a modality sub-dict)

| Column     | Meaning | Lives inside |
|------------|---------|-------------|
| `segment`  | Semantic segmentation class ID per point | `data['edep']['segment']`, `data['hits']['segment']` |
| `instance` | Instance segmentation ID per point | `data['edep']['instance']`, `data['hits']['instance']` |

`segment` and `instance` are **not** modalities — they are per-point
label columns attached when `labl` is in the modalities tuple.

### Foreign-key arrays

| FK array | Meaning | Scope |
|----------|---------|-------|
| `deposit_to_track` | Edep deposit → track ID | Per volume, in `labl` |
| `deposit_to_group` | Edep deposit → hits group ID | Per volume, in `bridges` |
| `group_to_track`   | Hits group → track ID | Per volume, in `bridges` |

All are suffixed `_v{N}` for multi-volume detectors (e.g. `deposit_to_group_v0`).

### The `bridges` dict

`bridges` appears as a top-level key only when `hits` is loaded
(JAXTPC only). It holds the FK arrays linking edep, hits, and labl:

```python
data['bridges'] = {
    'group_to_track_v0':   (G0,),   # hits group_id → track_id
    'deposit_to_group_v0': (N_v0,), # edep row → hits group_id
    'qs_fractions_v0':     ...,     # charge-sharing fractions
    # ... _v1, _v2, etc.
}
```

---

## Data layout

Each dataset expects sharded HDF5 files organized by modality:

```
data_root / [modality_subdir/] / [split/] / {dataset_name}_{modality}_{shard:04d}.h5
```

**JAXTPC** (`dataset_name='sim'`):

```
data_root/
  edep/     sim_edep_0000.h5    sim_edep_0001.h5    ...
  sensor/   sim_sensor_0000.h5  ...
  hits/     sim_hits_0000.h5    ...
  labl/     sim_labl_0000.h5    ...
```

**LUCiD** (`dataset_name='wc'`):

```
data_root/
  edep/     wc_edep_0000.h5     ...
  sensor/   wc_sensor_0000.h5   ...
  hits/     wc_hits_0000.h5     ...
  labl/     wc_labl_0000.h5     ...
```

**Path resolution rules:**

1. **Modality subdirectory** — `data_root/modality/` is checked first.
   If it doesn't exist, the reader falls back to `data_root` directly.

2. **Split subdirectory** — Readers glob `reader_root/split/{name}_{mod}_*.h5`
   first, falling back to `reader_root/{name}_{mod}_*.h5` if empty.
   `split=''` (LUCiD default) skips the split subdirectory entirely.

3. **Only requested modalities must be present** — Directories for
   modalities not in your `modalities` tuple are never accessed.

Readers are index-synchronized by event ordinal — `edep/` event 0 and
`sensor/` event 0 refer to the same physics event.

---

## Modality combinations

Each modality produces a sub-dict at a top-level key of the same name.

| Modality | Contains | Point-cloud dim |
|---|---|---|
| `edep`   | 3D truth deposits (one row per Geant4 step) | 3D |
| `sensor` | Raw sparse detector response | 2D (wire), 3D (pixel, PMT) |
| `hits`   | Per-particle decomposition of `sensor` | same as `sensor` |
| `labl`   | Dimension tables: per-track metadata | — |

`labl` has no point cloud — it attaches `segment` and `instance`
columns to an instance-bearing modality (`edep` or `hits`). Two
combinations are rejected:

- `('labl',)` — nothing to join against.
- `('sensor', 'labl')` — `sensor` is aggregated, no per-particle info.

---

## Output schema

### Top-level structure

```python
data = ds.get_data(idx)
# {
#     'name': str, 'split': str,
#     'edep':    {...},    # when 'edep' in modalities
#     'sensor':  {...},    # when 'sensor' in modalities
#     'hits':    {...},    # when 'hits' in modalities
#     'labl':    {...},    # when 'labl' in modalities
#     'bridges': {...},    # JAXTPC only, when 'hits' in modalities
# }
```

### JAXTPC sub-dicts

JAXTPC supports two readout types (auto-detected from HDF5):
**wire** (U/V/Y planes, coord is `(M, 2)`) and **pixel** (coord is `(M, 3)`).

```python
data['edep'] = {
    'coord':     (N, 3) float32,
    'energy':    (N, 1) float32,
    'volume_id': (N, 1) int32,
    # include_physics=True (default):
    'dx': (N, 1), 'theta': (N, 1), 'phi': (N, 1),
    'charge': (N, 1), 'photons': (N, 1), 't0_us': (N, 1),
    # present only when 'labl' is also in modalities:
    'instance': (N,) int32,     # = raw Geant4 track_id
    'segment':  (N,) int32,     # = track_{label_key} value
}

data['sensor'] = {
    'coord':        (M, D) float32,    # D=2 wire, D=3 pixel
    'energy':       (M, 1) float32,
    'plane_id':     (M, 1) int32,
    'planes':       [str, ...],        # plane labels in plane_id order
    'readout_type': 'wire' | 'pixel',
    'raw': {plane_label: {...}},       # per-plane arrays before merge
}

data['hits'] = {
    'coord':        (E, D) float32,
    'energy':       (E, 1) float32,
    'plane_id':     (E, 1) int32,
    'instance':     (E,) int32,        # = group_id
    'planes':       [str, ...],
    'readout_type': 'wire' | 'pixel',
    'raw': {plane_label: {...}},
    # present only when 'labl' is also in modalities:
    'segment':      (E,) int32,        # = track_{label_key} via group_to_track
}

data['labl'] = {              # keyed by volume
    'v0': {'track_ids': (T0,), 'deposit_to_track': (N_v0,),
           'track_pdg': (T0,), 'track_cluster': (T0,),
           'track_interaction': (T0,), 'track_ancestor': (T0,)},
    'v1': {...},
}

data['bridges'] = {           # only when 'hits' is loaded
    'group_to_track_v0':   (G0,),
    'deposit_to_group_v0': (N_v0,),
    'qs_fractions_v0':     ...,
}
```

### LUCiD sub-dicts

```python
data['edep'] = {
    'coord':       (N, 3) float32,    # midpoint of Geant4 step
    'energy':      (N, 1) float32,
    'time':        (N, 1) float32,
    'track_idx':   (N,) int32,        # FK → labl.track
    # include_physics=True (default):
    'direction':   (N, 3), 'beta_start': (N, 1), 'n_cherenkov': (N, 1),
    # present when 'labl' in modalities:
    'particle_idx': (N,), 'instance': (N,), 'segment': (N,),
}

data['sensor'] = {
    'coord':      (H, 3) float32,    # PMT positions
    'energy':     (H, 1) float32,    # post-smearing PE
    'time':       (H, 1) float32,
    'sensor_idx': (H,) int32,
}

data['hits'] = {
    'coord':        (E, 3) float32,
    'energy':       (E, 1) float32,
    'time':         (E, 1) float32,
    'sensor_idx':   (E,) int32,
    'particle_idx': (E,) int32,
    'instance':     (E,) int32,       # = particle_idx
    # present when 'labl' in modalities:
    'segment':      (E,) int32,       # = labl.particle.category
}

data['labl'] = {
    'event':    {'t0': (), 'overall_containment': ()},
    'particle': {'category': (P,), 'containment': (P,),
                 'ancestor_particle_idx': (P,), ...},
    'track':    {'track_id': (T,), 'pdg': (T,), 'parent_id': (T,),
                 'particle_idx': (T,), 'ancestor': (T,),
                 'ancestor_particle_idx': (T,), ...},
}
```

### Instance and segment semantics

**Instance** has different physical meaning per dataset:

| Dataset | Modality | `instance` = |
|---------|----------|-------------|
| JAXTPC  | `edep`   | `track_id` (raw Geant4 track ID) |
| JAXTPC  | `hits`   | `group_id` (hits' native grouping) |
| LUCiD   | both     | `particle_idx` (FK into labl.particle) |

If a task needs uniform instance convention, remap in a transform.

**Segment** carries raw label values controlled by `label_key`:

| Dataset | Source | Raw values |
|---------|--------|------------|
| JAXTPC  | `track_{label_key}` | PDG codes (default), cluster IDs, etc. |
| LUCiD   | `labl.particle.category` | Detector-specific category integers |

Use `RemapSegment` to map raw values to dense class indices:

```python
dict(type='RemapSegment', scheme='motif_5cls')
```

---

## Transform pipeline

The dataset returns a nested dict → transforms process it →
`Collect` extracts a flat dict for the model.

```
get_data()  →  ApplyToStream(stream='edep', transforms=[...])  →  Collect(stream='edep', ...)
nested dict       augments edep sub-dict (numpy)                    flat dict (tensors)
```

### ApplyToStream

Dispatches transforms to a specific modality sub-dict. Inner transforms
operate on numpy. If the stream is missing, the transform is a no-op
(pass `required=True` to raise instead).

```python
dict(type='ApplyToStream', stream='edep', transforms=[
    dict(type='NormalizeCoord', center=[0, 0, 0], scale=4000.0),
    dict(type='GridSample', grid_size=0.001, mode='train',
         return_grid_coord=True),
])
```

### Collect

Terminal transform. Extracts keys from a stream, auto-converts numpy
to torch tensors, and builds the `offset` key for batching.

```python
dict(type='Collect', stream='edep',
     keys=('coord', 'grid_coord', 'segment'),
     feat_keys=('coord', 'energy'))
# Output: {coord, grid_coord, segment, feat, offset, name, split}
```

`ToTensor` is **not needed** — `Collect` handles tensor conversion.
This also enables efficient `DataLoader` parallelism (see below).

### Practical configs

**SSL on 3D edep (no labels):**

```python
transform=[
    dict(type='ApplyToStream', stream='edep', transforms=[
        dict(type='NormalizeCoord', center=[0, 0, 0], scale=4000.0),
        dict(type='GridSample', grid_size=0.001, mode='train',
             return_grid_coord=True),
        dict(type='ShufflePoint'),
    ]),
    dict(type='Collect', stream='edep',
         keys=('coord', 'grid_coord'),
         feat_keys=('coord', 'energy')),
]
```

**Supervised segmentation (edep + labl):**

```python
transform=[
    dict(type='ApplyToStream', stream='edep', transforms=[
        dict(type='RemapSegment', scheme='motif_5cls'),
        dict(type='NormalizeCoord', center=[0, 0, 0], scale=4000.0),
        dict(type='GridSample', grid_size=0.001, mode='train',
             return_grid_coord=True),
    ]),
    dict(type='Collect', stream='edep',
         keys=('coord', 'grid_coord', 'segment'),
         feat_keys=('coord', 'energy')),
]
```

**Instance segmentation on 2D hits:**

```python
transform=[
    dict(type='ApplyToStream', stream='hits', transforms=[
        dict(type='RemapSegment', scheme='motif_5cls'),
        dict(type='GridSample', grid_size=1.0, mode='train',
             return_grid_coord=True),
    ]),
    dict(type='Collect', stream='hits',
         keys=('coord', 'grid_coord', 'segment', 'instance'),
         feat_keys=('coord', 'energy')),
]
```

---

## DataLoader and performance

### Standard setup

```python
from torch.utils.data import DataLoader
from pimm_data import JAXTPCDataset, collate_fn

ds = JAXTPCDataset(data_root="...", modalities=("edep", "labl"),
                   label_key="pdg", transform=[...])
loader = DataLoader(ds, batch_size=4, num_workers=4,
                    collate_fn=collate_fn, pin_memory=True)
```

### Why Collect must be the last transform

`Collect` converts numpy arrays to torch tensors. PyTorch's
`ForkingPickler` transfers tensors via file-descriptor sharing (~400
bytes) instead of pickling the full array. Without tensor output,
`num_workers > 0` is **slower** than serial for large events.

### Choosing num_workers

| GPU step time | Recommended workers |
|---|---|
| > 200 ms | 2 |
| 100–200 ms | 4 |
| < 100 ms | 6 |

Per-worker memory is ~240 MB. `prefetch_factor=2` (default) is sufficient.

### Batch output

`collate_fn` concatenates variable-length point clouds and tracks
boundaries with a cumulative `offset` tensor:

```python
batch['coord'].shape    # (total_points, 3)
batch['feat'].shape     # (total_points, 4)
batch['segment'].shape  # (total_points,)
batch['offset']         # tensor([N0, N0+N1, N0+N1+N2, ...])
```

`point_collate_fn` adds mix-up augmentation.
`inseg_collate_fn` flattens instance-segmentation query batches.

### Performance tips

1. **Use `volume=0`** to load a single TPC volume (2–4x faster).
2. **Load only needed modalities** — `hits` is heaviest (~90 MB/event).
3. **End transforms with `Collect`** — enables zero-copy IPC.
4. **`GridSample` early** — downsample before expensive transforms.
5. **`cache=True`** with `SharedArray` installed eliminates HDF5
   decompression after the first epoch.

---

## API reference

### JAXTPCDataset

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data_root` | str | *(required)* | Root directory with modality subdirectories |
| `split` | str | `'train'` | Split subdirectory. Pass `''` for flat layouts. |
| `modalities` | tuple | `('edep',)` | Subset of `'edep'`, `'sensor'`, `'hits'`, `'labl'` |
| `dataset_name` | str | `'sim'` | File prefix (matches `sim_edep_*.h5`) |
| `volume` | int/None | `None` | Single-volume filter. `None` = all volumes. |
| `label_key` | str | `'pdg'` | Labl column for `segment`: `'pdg'`, `'cluster'`, `'interaction'`, `'ancestor'` |
| `min_deposits` | int | `0` | Drop events with fewer deposits |
| `include_physics` | bool | `True` | Load dx, theta, phi, charge, photons, t0_us |
| `label_keys` | list/None | `None` | Whitelist of labl datasets to load (`None` = all) |
| `transform` | list/None | `None` | List of transform dicts (NOT a Compose object) |
| `max_len` | int | `-1` | Cap on event count (`-1` = no cap) |
| `cache` | bool | `False` | Shared-memory caching (requires `SharedArray`) |

### LUCiDDataset

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data_root` | str | *(required)* | Root directory with modality subdirectories |
| `split` | str | `''` | Split subdirectory (default: no split) |
| `modalities` | tuple | `('sensor',)` | Subset of `'edep'`, `'sensor'`, `'hits'`, `'labl'` |
| `dataset_name` | str | `'wc'` | File prefix (matches `wc_sensor_*.h5`) |
| `min_segments` | int | `0` | Drop events with fewer edep segments |
| `include_physics` | bool | `True` | Load direction, beta_start, n_cherenkov |
| `pe_threshold` | float | `0.0` | Drop hits entries with PE ≤ threshold |
| `transform` | list/None | `None` | List of transform dicts |

### Config-driven construction

```python
from pimm_data import build_dataset
ds = build_dataset(dict(type='JAXTPCDataset', data_root='...', modalities=('edep',)))
```

---

## Layout

```
src/pimm_data/
    jaxtpc.py          JAXTPCDataset
    lucid.py           LUCiDDataset
    readers/           Per-modality HDF5 readers
    transform.py       Compose, TRANSFORMS, Collect, ...
    detector_transforms.py  ApplyToStream, PDGToSemantic, RemapSegment
    collate.py         collate_fn, point_collate_fn, inseg_collate_fn
    utils/pdg.py       pdg_to_semantic(pdg, scheme)
```

## Tests

```bash
pytest
```

Tests use synthetic data by default. Point at real datasets with:

```bash
export JAXTPC_DATA_ROOT=/path/to/jaxtpc_dataset
export LUCID_DATA_ROOT=/path/to/wc_dataset
pytest
```
