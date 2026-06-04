# Part 01 — Transforms (implementation spec)

**Status:** final pre-coding spec. Implements the §3.1 transform merges +
§3.2 `index_operator` prefix-match of
`implementation_plan_pimm_data_datalayer.md`. This is the executable spec; the
engagement doc (`engagement_plan_transform_dataset_placement.md`) remains the
decision record.

**Source decisions:** D11 (upstream his transforms), D25 (`index_operator`
prefix-match + per-stream N-changing), D29 (`GridSample` `{key:op}` reducer map +
back-compat shim), D31 (negative-time correctness split), D34 (reversible impl
details deferred with documented defaults), D38 (open/extensible label axes →
prefix families). Locked constraints honored: single-stream near-term (D35);
**do NOT overwrite pimm-data's `Collect`** (it is ahead of the branch); reducer
API is `{key: op}` with `sum_keys`/`min_keys` shim; `RelativeLogNormalize`
subtract-min is a correctness fix; reversible defaults stated below.

**Files touched (read-only ground truth in parens):**
- `src/pimm_data/transform.py` — the merges (current state grounded against
  pimm-data and the branch `pimm/datasets/transform.py`).
- `src/pimm_data/detector_transforms.py` — **untouched** by this part; relevant
  only because `ApplyToStream` (lines 26–63) is the per-stream wrapper through
  which the prefix-match (§3.2) and N-changing transforms run.
- `tests/test_transforms.py`, `tests/test_jaxtpc_transforms.py` — extend.
- `tests/conftest.py`, `src/pimm_data/testing.py` — fixtures reused as-is; one new
  helper proposed (§6).

**Path conventions in this doc.** "pimm-data" = `/sdf/group/neutrino/omara/pimm-data/src/pimm_data/transform.py`.
"branch" = the colleague's reference `/sdf/home/o/omara/.claude/jobs/21ffc656/particle-imaging-models/pimm/datasets/transform.py`.
Line numbers are from the snapshots read for this spec.

---

## 1. Purpose & scope

Bring pimm-data's `transform.py` up to (and slightly past) the colleague's
`research`-branch transform set, so the LUCiD-SSL config and the broad
multi-task pipeline run on pimm-data alone. Concretely:

1. **`RelativeLogNormalize`** — NET-NEW. Per-event relative log normalization
   for PMT hit times (`T`), including the D31 negative-time correctness step.
2. **`GridSample` `{key: op}` reducers** — generalize the branch's
   `sum_keys`/`min_keys` to a `{key: op}` map over `sum/min/max/mean/first`, with
   a back-compat shim. Only `min`/`sum` exist on the branch; `max/mean/first` are
   net-new and need fully-specified per-op semantics (fill values, `mean`
   count-divide, `first` deterministic representative).
3. **`LogTransform.clip`** — NET-NEW `clip` flag (already on the branch) that
   clamps the pre-`log10`/pre-linear domain.
4. **`MultiViewGenerator.get_view` guard** — NET-NEW empty-cloud raise +
   size clamp (already on the branch).
5. **v3 `vertex`/`is_primary` plumbing** — port the branch's vertex co-transform
   helpers and wire them into every geometric transform; add `vertex` to the
   default `index_valid_keys` and the `revision=="v3"` `is_primary` append.
   `PointClip` stays vertex-blind.
6. **`MixedScaleGeometryMultiViewGenerator`** — NET-NEW subclass (already on the
   branch); off the SSL critical path but ported verbatim.
7. **`index_operator` prefix-match (D25)** — carry `segment_*`/`instance_*`/
   `target_*` + `particle_idx`/`sensor_idx`/`plane_id` through N-changing
   transforms by underscore-boundary prefix-match, with a per-event `target_*`
   shape-exclusion. This is net-new **on top of** the branch.

Out of scope (other parts): `MultiModalEventDataset`, readers, `label_config`
decoration, collate, eval-hook rewiring. Out of scope entirely for now:
namespaced multi-stream collate, `AggregateBySensor`, Track B.

---

## 2. Current state (file:line — what exists in pimm-data vs the branch)

| Item | pimm-data | branch | Delta to implement |
|---|---|---|---|
| `RelativeLogNormalize` | absent | class @ `transform.py:277-318` | port verbatim (it is the reference) |
| `index_operator` default list | `transform.py:41-71` (no `vertex`, no `revision`) | `transform.py:66-103` (adds `"vertex"` @ 81; `revision=="v3"`→append `is_primary` @ 98-103) | merge branch list+revision, **then** add prefix-match (§3.2) |
| `_valid_vertex_mask` / `_apply_to_v3_vertex` / `_translate_axis` | absent | `transform.py:31-53` | port verbatim |
| `Collect` | `transform.py:87-140` (ahead: `stream=`, `_to_tensor` autoconvert, name/split passthrough) | `transform.py:119-143` (older, no stream) | **DO NOT TOUCH** — pimm-data wins |
| `LogTransform` | `transform.py:234-265` (no `clip`) | `transform.py:238-274` (`clip=False` @ 240; clamp @ 251-252,260-261) | add `clip` param + clamps |
| `NormalizeCoord` | `transform.py:210-231` (subtracts centroid in each branch; no vertex) | `transform.py:213-235` (computes centroid+scale once, then `_apply_to_v3_vertex`) | refactor to branch form (compute once) + vertex call |
| `PositiveShift` | `transform.py:281-287` | `transform.py:334-341` (+vertex) | add vertex call |
| `CenterShift` | `transform.py:290-313` | `transform.py:344-379` (+vertex via `_translate_axis`) | add vertex calls per-axis |
| `ConditionalRandomTransform` | `transform.py:315-371` | `transform.py:381-443` (+vertex on translation) | add vertex call |
| `RandomShift` | `transform.py:373-384` (`+= [x,y,z]` list) | `transform.py:445-458` (`shift=np.array`; +vertex) | use array shift + vertex call |
| `PointClip` | `transform.py:387-399` | `transform.py:461-473` (**identical**, vertex-blind) | none — keep blind |
| `RandomRotate` | `transform.py:425-459` | `transform.py:499-538` (+vertex; `center=np.asarray`) | `np.asarray(center)` + vertex call |
| `RandomRotateTargetAngle` | `transform.py:462-498` | `transform.py:541-582` (+vertex) | as above |
| `RandomScale` | `transform.py:501-513` | `transform.py:585-598` (+vertex) | add vertex call |
| `RandomFlip` | `transform.py:516-538` | `transform.py:601-641` (+vertex via `np.column_stack`) | add vertex calls per-axis |
| `GridSample` | `transform.py:1079-1228` (`sum_keys` only; var `summed` @ 1130-1143) | `transform.py:1182-1343` (`sum_keys`+`min_keys`; var `reduced`; min fill @ 1249-1255) | generalize to `{key:op}` + shim (§3.2/§3.3.2) |
| `MultiViewGenerator.get_view` | `transform.py:1414-1427` (no guard) | `transform.py:1529-1548` (`max_size<=0` raise @ 1532-1533; `size=max(1,min(...))` @ 1538) | add guard; **only `get_view` changes** |
| `MixedScaleGeometryMultiViewGenerator` | absent | `transform.py:1681-1824` | port verbatim |
| `InstanceParser`, `LocalCovarianceFeatures`, `HierarchicalMaskGenerator`, `HMAECollate`, `ComputeAnchors`, `RandomDrop`, `CropBoundary`, `ContrastiveViewsGenerator`, color/jitter, crops | identical to branch (modulo the vertex hooks above) | identical | none |

**Registry import differs (do not "fix" as part of this part):** pimm-data uses
`from ._registry import Registry` (`transform.py:22`); branch uses
`from pimm.utils.registry import Registry` (`transform.py:22`). Keep pimm-data's
import. `Compose` in pimm-data (`transform.py:2081-2109`) already supports mixed
dict/callable entries and is ahead of the branch — leave it.

**Key correctness asymmetry to preserve.** pimm-data `NormalizeCoord`
(`transform.py:216-231`) subtracts the centroid *inside both the `center is None`
and the explicit-`center` branches* before computing `m` from the already-shifted
coords. The branch (`transform.py:219-235`) computes `centroid` and `scale` first,
divides once, **then** applies `(vertex - centroid)/scale`. These are numerically
identical for `coord` (the branch just hoists). Adopt the branch's single-compute
form so the vertex transform uses the same `centroid`/`scale` the coords used —
this is required for vertex parity, not a behavior change for `coord`.

---

## 3. Target design (per transform)

### 3.1 `RelativeLogNormalize` (NET-NEW)

**Signature** (port the branch verbatim, `transform.py:277-318`):
```python
@TRANSFORMS.register_module()
class RelativeLogNormalize(object):
    def __init__(self, keys=("time",), scale=50.0, max_val=4000.0,
                 out_min=-1.0, out_max=1.0): ...
```

**Constructor invariants** (raise `ValueError`): `scale > 0`, `max_val > 0`,
`out_max > out_min`. `keys` coerced to a tuple if not already. Precompute
`self.denom = np.log1p(self.max_val / self.scale)`.

**Algorithm** (`relative_log_transform(x)`), in exact order:
1. `x = np.asarray(x, dtype=np.float32)` — cast first (input may be `float16`
   like LUCiD `T`/JAXTPC `t0_us`, or `float64`).
2. `x = x - np.min(x)` — **D31 negative-time correctness step.** WAND `T` reaches
   ~−240 ns; without this the subsequent `log1p` would see a negative domain for
   the most-negative-time hits. Per-event (per-array) min, so this is **not
   idempotent** and is **order-sensitive** w.r.t. any prior point reorder/subset
   of the same column — see §5/§7.
3. `x = np.clip(x, 0.0, self.max_val)` — upper-truncate the Michel/decay tail
   (config_1 ~0.54% of hits > 4000 ns). Lower clip at 0 is belt-and-suspenders
   after step 2 (floating-point min can leave a tiny negative).
4. `y = np.log1p(x / self.scale) / self.denom` — maps `[0, max_val]` → `[0, 1]`.
5. `y = self.out_min + y * (self.out_max - self.out_min)` — affine to
   `[out_min, out_max]`.
6. `return np.clip(y, self.out_min, self.out_max).astype(np.float32, copy=False)`.

`__call__`: for each `k in self.keys`, if `k in data_dict` apply the transform
in place; else `raise ValueError(f"Key {k} not found in data_dict")` (matches
`LogTransform`/`RelativeLogNormalize` branch behavior — strict, not silent skip).

**Data shapes.** Operates element-wise; preserves shape `(N,)` or `(N,1)`. Output
dtype `float32`.

### 3.2 `GridSample` `{key: op}` reducers (D29)

**Signature.** Extend the branch ctor (`transform.py:1184-1207`) with one new
optional arg `reducers`:
```python
def __init__(self, grid_size=0.05, hash_type="fnv", mode="train",
             return_inverse=False, return_grid_coord=False,
             return_min_coord=False, return_displacement=False,
             project_displacement=False,
             sum_keys=None, min_keys=None, reducers=None): ...
```

**Back-compat shim (D29; explicit `reducers` wins).** Build a single normalized
`self.reducers: dict[str,str]` in `__init__`:
```python
merged = {}
for k in (sum_keys or []):  merged[k] = "sum"
for k in (min_keys or []):  merged[k] = "min"
if reducers:                merged.update({k: op for k, op in reducers.items()})
ALLOWED = {"sum", "min", "max", "mean", "first"}
bad = {op for op in merged.values() if op not in ALLOWED}
if bad: raise ValueError(f"GridSample: unknown reducer op(s) {sorted(bad)}; "
                         f"allowed {sorted(ALLOWED)}")
self.reducers = merged
```
Keep `self.sum_keys`/`self.min_keys` as attributes too (some configs/tests may
introspect), but the `__call__` path drives off `self.reducers` only. **Parity
requirement (D29 / §6):** with no `reducers=` given, `sum_keys`/`min_keys` must
produce byte-identical output to the branch — so the shim must reproduce the
branch's exact arithmetic for those two ops (below).

**Where reductions run.** In `mode == "train"` only (the test-mode branch
`transform.py:1279-1305` does no aggregation; leave it untouched — reducers are a
train-time voxel-merge concept). The block currently at branch
`transform.py:1235-1258` (the `if self.sum_keys or self.min_keys:` block, var
`reduced`) is replaced by a `{key:op}` loop:

```python
reduced = {}
if self.reducers:
    voxel_of_point = np.empty(len(key), dtype=inverse.dtype)
    voxel_of_point[idx_sort] = inverse        # branch line 1237-1238
    num_voxels = len(count)
    for rk, op in self.reducers.items():
        if rk not in data_dict:
            continue
        vals = data_dict[rk]
        reduced[rk] = self._reduce(op, vals, voxel_of_point, num_voxels,
                                   count, idx_sort, inverse)
data_dict = index_operator(data_dict, idx_unique)   # branch line 1256
for rk, agg in reduced.items():
    data_dict[rk] = agg                              # branch line 1257-1258
```
The order — compute `reduced` from the **pre-subset** arrays, then
`index_operator(idx_unique)`, then overwrite the reduced keys — is exactly the
branch order (`transform.py:1256-1258`) and must be preserved: `index_operator`
first subsets `rk` to one survivor per voxel (`idx_unique`), then we overwrite
with the voxel-aggregate. Net effect: reduced keys end up length `num_voxels`,
same as every other surviving column.

**Per-op implementation** (`_reduce(op, vals, voxel_of_point, num_voxels, count, idx_sort, inverse)`):

- **`sum`** (branch-identical, `transform.py:1240-1245`):
  ```python
  agg = np.zeros((num_voxels,) + vals.shape[1:], dtype=vals.dtype)
  np.add.at(agg, voxel_of_point, vals)
  ```
  Empty voxels cannot occur (every voxel has ≥1 point), so fill is the additive
  identity `0` and is never observed.

- **`min`** (branch-identical, `transform.py:1246-1255`):
  ```python
  fill = np.inf if np.issubdtype(vals.dtype, np.floating) else np.iinfo(vals.dtype).max
  agg = np.full((num_voxels,) + vals.shape[1:], fill, dtype=vals.dtype)
  np.minimum.at(agg, voxel_of_point, vals)
  ```
  Fill is never observed (every voxel non-empty) but is the correct min-identity.

- **`max`** (NET-NEW, mirror of `min`):
  ```python
  fill = -np.inf if np.issubdtype(vals.dtype, np.floating) else np.iinfo(vals.dtype).min
  agg = np.full((num_voxels,) + vals.shape[1:], fill, dtype=vals.dtype)
  np.maximum.at(agg, voxel_of_point, vals)
  ```
  Use `np.iinfo(vals.dtype).min` for signed ints; for **unsigned** ints `.min`
  is `0`, which is the correct max-identity for non-negative data — and again
  unobserved because voxels are non-empty. Keep dtype = input dtype.

- **`mean`** (NET-NEW; sum/count, promote int→float):
  ```python
  out_dtype = vals.dtype if np.issubdtype(vals.dtype, np.floating) else np.float32
  agg = np.zeros((num_voxels,) + vals.shape[1:], dtype=out_dtype)
  np.add.at(agg, voxel_of_point, vals.astype(out_dtype, copy=False))
  counts = count.reshape((num_voxels,) + (1,) * (vals.ndim - 1)).astype(out_dtype)
  agg /= counts            # count >= 1 for every voxel, no div-by-zero
  ```
  `count` is the per-voxel point count from `np.unique(..., return_counts=True)`
  (branch `transform.py:1220`), already aligned to voxel index 0..num_voxels-1.
  **Int inputs promote to `float32`** (document: a `mean` of int columns is no
  longer integer-typed). Float inputs keep their float dtype.

- **`first`** (NET-NEW; deterministic representative, NOT the random survivor):
  ```python
  # idx_sort orders points by hash key; cumsum(insert(count,0,0)[:-1]) is the
  # start offset of each voxel's run in sorted order. The first point of each
  # run is a deterministic, hash-stable representative.
  first_in_voxel = idx_sort[np.cumsum(np.insert(count, 0, 0)[:-1])]
  agg = vals[first_in_voxel]
  ```
  This is **deterministic** (independent of `np.random`), unlike the voxel
  survivor chosen for `coord`/`segment` in train mode (`idx_unique` uses
  `np.random.randint`, branch `transform.py:1222-1226`). Document the divergence:
  a `first`-reduced column is generally NOT the same point as the surviving
  `coord` row in the same voxel. `first` exists for columns where any in-voxel
  representative is acceptable and reproducibility across seeds matters (e.g.
  `plane_id`, a constant-within-voxel id). Preserves input dtype and trailing
  shape.

**Reducer key ≠ `index_valid_keys` interaction.** A key in `self.reducers` is
aggregated and overwritten regardless of whether it is in `index_valid_keys`;
`index_operator` will also try to subset it (to `idx_unique`) but the subsequent
overwrite replaces that with the aggregate. This double-touch is harmless and
matches the branch. If a reduced key is *not* in `index_valid_keys`, the
`index_operator` subset is a no-op for it and only the aggregate write lands —
also fine.

### 3.3 `LogTransform.clip` (D11/D31)

Port the branch form (`transform.py:240,251-252,260-261`):
- ctor adds `clip=False`; store `self.clip = clip`.
- `log_transform(x)`: if `self.clip`, `x = np.clip(x, 0.0, self.max_val)` **before**
  the `log10` mapping (lower bound `0.0`, not `min_val` — the log mapping adds
  `min_val` internally so the valid input domain floor is `0`).
- `linear_transform(x)`: if `self.clip`, `x = np.clip(x, self.min_val, self.max_val)`
  (linear floor is `min_val`).
- Default `clip=False` ⇒ existing PILArNet/energy behavior unchanged (no
  regression).

### 3.4 `MultiViewGenerator.get_view` guard (D11)

Port the branch (`transform.py:1529-1548`). Replace pimm-data's
`get_view` body lines 1414-1418 with:
```python
coord = point["coord"]
max_size = min(self.max_size, coord.shape[0])
if max_size <= 0:
    raise ValueError("Cannot generate a view from an empty point cloud")
size = int(np.random.uniform(*scale) * max_size) if size_override is None \
    else int(size_override)
size = max(1, min(max_size, size))
```
Rest of `get_view` (the `index = argsort(...)[:size]`, view dict build,
`index_valid_keys` inherit) unchanged. **Only `get_view` changes** — the rest of
`MultiViewGenerator` (`__call__`, `get_center`, anchor logic) is already identical
to the branch. Effect: an empty cloud raises a clear error instead of an opaque
downstream failure; a 1-point cloud yields a size-1 view instead of size-0
(`int(uniform * 1)` can floor to 0).

### 3.5 v3 `vertex`/`is_primary` plumbing (D11)

**Module helpers** (port verbatim, branch `transform.py:31-53`):
- `_valid_vertex_mask(data_dict)`: returns `None` if no `vertex` key, if
  `vertex.ndim != 2`, or if `vertex.shape[1] != 3`; else
  `~(vertex == -1).all(axis=1)` (v3 uses `(-1,-1,-1)` as missing-vertex sentinel;
  rows that are all-`-1` are skipped).
- `_apply_to_v3_vertex(data_dict, transform)`: if mask is `None` or `not mask.any()`,
  no-op; else `data_dict["vertex"][mask] = transform(data_dict["vertex"][mask])`.
- `_translate_axis(points, dim, value)`: copy, `points[:, dim] += value`, return.

**`index_operator` default list + revision append** (branch `transform.py:66-103`):
- Add `"vertex"` to the default `index_valid_keys` list (branch has it at
  `transform.py:81`).
- Add the `revision=="v3"` block (branch `transform.py:98-103`): if
  `data_dict.get("revision") == "v3"`, coerce `index_valid_keys` to a list and
  append `"is_primary"` if absent. This runs **before** the prefix-match append
  of §3.2 (order: default list → branch v3 `is_primary` → §3.2 prefix-match +
  explicit ids → §3.2 per-event exclusion). `is_primary` is a per-point bool/int
  column that must subset with `coord` on every N-changing transform; it is added
  only under `revision=="v3"` so non-v3 datasets are unaffected.

**Per-transform vertex insertion** (each guarded by `_apply_to_v3_vertex`, so a
no-op when no valid `vertex` key — dormant until a v3 dataset stamps `vertex`):

| Transform | Insertion (branch ref) | Vertex transform lambda |
|---|---|---|
| `NormalizeCoord` | after coord divide (`transform.py:234`) | `lambda v: (v - centroid) / scale` — requires the single-compute refactor (§2) so `centroid`/`scale` are in scope |
| `PositiveShift` | after `coord -= coord_min` (`transform.py:340`) | `lambda v: v - coord_min` |
| `CenterShift` | per active axis, after the axis shift (`transform.py:360-377`) | `lambda v, shift=shift: _translate_axis(v, dim, -shift)` (dim ∈ {0,1,2}; capture `shift` as default arg) |
| `ConditionalRandomTransform` | inside the `if t_low <= t_high:` translation branch (`transform.py:435-440`) | `lambda v, dim=dim, translation=translation: _translate_axis(v, dim, translation)` |
| `RandomShift` | after `coord += shift` (`transform.py:457`); build `shift=np.array([sx,sy,sz])` | `lambda v: v + shift` |
| `RandomRotate` | after the rotate-about-center (`transform.py:532-535`); `center=np.asarray(center)` | `lambda v: np.dot(v - center, rot_t.T) + center` |
| `RandomRotateTargetAngle` | as `RandomRotate` (`transform.py:576-579`) | same |
| `RandomScale` | after `coord *= scale` (`transform.py:597`) | `lambda v: v * scale` (scale is `(1,)` or `(3,)` per `anisotropic`) |
| `RandomFlip` | per flipped axis (`transform.py:613-638`) | `lambda v: np.column_stack((-v[:,0], v[:,1], v[:,2]))` etc. per axis |

**`PointClip` stays vertex-blind** (D11 explicit; branch `PointClip`
`transform.py:461-473` has no vertex call and is byte-identical to pimm-data
`transform.py:387-399`). Clipping coordinates to a detector box must not drag the
vertex — the vertex is a physics label, not a coordinate to be clamped. Leave it.

**Why guarded.** Every call goes through `_apply_to_v3_vertex`, which returns
early when there is no `vertex` key (the common case today). So porting these has
**zero observable effect** on current PILArNet/LUCiD/JAXTPC runs until a dataset
actually stamps a `(N,3)` `vertex` array — this is the "dormant until a v3 dataset
stamps revision" property from §3.1 #5 of the impl plan.

### 3.6 `MixedScaleGeometryMultiViewGenerator` (NET-NEW)

Port verbatim from the branch (`transform.py:1681-1824`). Subclass of
`MultiViewGenerator`. New ctor args: `fine_local_view_num=3`,
`fine_local_view_scale=(0.01, 0.04)`, `fine_center_mode="geometry"` (or
`"random"`), `fine_center_top_frac=0.05`, `fine_center_k=24`; asserts
`0 <= fine_local_view_num <= self.local_view_num` and
`fine_center_mode in ("geometry","random")`. Static
`_directional_complexity(coord, k)` (kNN-PCA λ2/λ3 ratio via `cKDTree`) and
`_geometry_pool(coord, major_index)` (top-`fine_center_top_frac` complexity
points within the major view). `__call__` replaces some random locals with
fine-scale local crops centered on high-complexity points. It reuses the now-guarded
`get_view` (§3.4). Optional — not on the SSL critical path; no config depends on
it yet. Port for parity with the branch and future use.

### 3.7 `index_operator` prefix-match (D25, §3.2 of impl plan)

This is the keystone of N-changing safety and is **net-new on top of the branch**
(the branch only has a hardcoded list + the `revision` `is_primary` append; it
still omits `time`/`sensor_idx`/`particle_idx`/`plane_id` and any decorated
`segment_*`/`instance_*`/`target_*` axis the config didn't manually `Update`).

**Problem.** `index_operator` (pimm-data `transform.py:38-84`) subsets only keys
in `index_valid_keys`. N-changing transforms — `GridSample` train
(`transform.py:1141`/`1256`), `RandomDropout` (`transform.py:421`), `SphereCrop`
(`transform.py:1259`), `HardExampleCrop`, `ShufflePoint` (`transform.py:1334`),
`CropBoundary` (`transform.py:1344`) — all route through it. Any per-point column
NOT in the list silently desyncs (keeps the old N while `coord` shrinks/reorders),
corrupting labels. Decorated label keys (`segment_pid`, `instance_particle`,
`target_*`, …) and FK columns (`particle_idx`, `sensor_idx`, `plane_id`) must ride
along.

**Fix.** After the default-list init **and** the branch `revision` append, before
the subset loop, append matching dict keys:

```python
def _underscore_prefixed(name, prefixes):
    # match "segment", "segment_pid"; NOT "segmentation". Boundary = exact
    # name or name + "_".
    return any(name == p or name.startswith(p + "_") for p in prefixes)

_LABEL_PREFIXES = ("segment", "instance", "target")
_EXTRA_POINT_KEYS = ("particle_idx", "sensor_idx", "plane_id")

ivk = data_dict["index_valid_keys"]
if not isinstance(ivk, list):
    ivk = list(ivk); data_dict["index_valid_keys"] = ivk
n_points = data_dict["coord"].shape[0] if "coord" in data_dict else None
for name, val in data_dict.items():
    if name in ivk:
        continue
    matches = _underscore_prefixed(name, _LABEL_PREFIXES) or name in _EXTRA_POINT_KEYS
    if not matches:
        continue
    # per-event target_* shape-exclusion: a per-event target (e.g. target_vertex
    # of shape (3,) or (1,D)) must NOT be point-subset. Only carry arrays whose
    # leading dim == n_points.
    if isinstance(val, np.ndarray) and n_points is not None \
            and (val.ndim == 0 or val.shape[0] != n_points):
        continue
    ivk.append(name)
```

Then proceed to the existing subset loop unchanged.

**Underscore-boundary match (reversible default, D34).** Match is `name == prefix`
OR `name.startswith(prefix + "_")`. So `segment`, `segment_pid`,
`instance_particle`, `target_energy` match; `segmentation`, `targeted`,
`instances` do **not**. This is stricter than bare `startswith(prefix)` and is the
chosen default (documented per D34 as reversible — a registry of explicit axis
names is the fallback if a real key ever legitimately collides).

**Per-event `target_*` shape-exclusion (D28/D25).** Per-event targets
(`target_vertex` `(3,)` or `(1,3)`, `target_energy` scalar/`(1,)`,
`target_contained`) are **not** per-point and must survive an N-changing transform
unchanged. The leading-dim check `val.shape[0] != n_points` excludes them. A
scalar (`ndim == 0`) is also excluded. Caveat: a per-event `target_*` whose first
dim *coincidentally* equals `n_points` would be wrongly subset — extremely
unlikely for `(3,)`/scalar targets, and the per-event targets are carried as
`_`-prefixed list-collated metadata downstream anyway (§3.5 of impl plan, not this
part). Document this as a known soft edge.

**Interaction with `ApplyToStream`.** N-changing transforms run **per-stream**
inside `ApplyToStream` (D25). `ApplyToStream` (`detector_transforms.py:53-63`)
calls `self.inner(data_dict[self.stream])` — it passes the *sub-dict* to the inner
`Compose`, so `index_operator` sees the stream's own `coord`/labels and its own
`index_valid_keys`. A config that needs an explicit `Update(index_valid_keys=...)`
must place that `Update` **inside** the same `ApplyToStream` (the list does not
propagate across the stream boundary). With the prefix-match in place, the
per-config `Update(index_valid_keys=...)` is **no longer load-bearing** for the
standard decorated axes — it remains available for exotic custom columns.

**Idempotency.** The append is guarded by `if name in ivk: continue`, so repeated
`index_operator` calls (e.g. `GridSample` then `ShufflePoint`) don't duplicate
entries. Matches the branch's `if key not in ...index_valid_keys` guard style
(`transform.py:102`).

---

## 4. Expected behavior (concrete input→output + invariants)

### 4.1 `RelativeLogNormalize`
Input `time = np.array([-240., 0., 50., 8000.], dtype=np.float32)`, defaults
(`scale=50, max_val=4000, out=[-1,1]`):
- subtract min: `[0, 240, 290, 8240]`
- clip[0,4000]: `[0, 240, 290, 4000]`
- `log1p(x/50)/log1p(4000/50)` = `/log1p(80)` (`denom≈4.394`):
  `log1p(0)=0`→`0`; `log1p(4.8)=1.7579`→`0.4001`; `log1p(5.8)=1.9169`→`0.4363`;
  `log1p(80)=4.3944`→`1.0`.
- affine `[-1,1]`: `[-1.0, -0.1998, -0.1274, 1.0]` (final clip is a no-op here).
**Invariants:** output in `[out_min, out_max]`; finite (no NaN/Inf) for any
finite input including all-negative or all-equal inputs (all-equal ⇒ subtract-min
gives all-zero ⇒ output all `out_min`); dtype `float32`; shape preserved.

### 4.2 `GridSample` reducers
Two points in one voxel with `charge=[10., 30.]` (float32 `(2,1)`) and `plane_id=[2,2]`:
- `reducers={'charge':'sum'}` → voxel `charge = [40.]`.
- `'min'` → `[10.]`; `'max'` → `[30.]`; `'mean'` → `[20.]` (float32).
- `plane_id` with `'first'` → `[2]` (int dtype kept).
Mixed: `reducers={'charge':'mean','energy':'sum'}` aggregates each per its op in
one pass. **Invariants:** every reduced key has leading dim == surviving voxel
count (`num_voxels`); `mean` of int promotes to float32; `first` is deterministic
across `np.random` seeds; `sum`/`min` with no `reducers=` (only `sum_keys`/
`min_keys`) byte-match the branch.

### 4.3 `LogTransform.clip`
`energy = [-5., 0., 1e6]`, `min_val=1e-2, max_val=20, log=True`:
- `clip=False` (default): `log10(x+1e-2)` of `-5` → `log10(-4.99)` = NaN (current
  behavior; out-of-domain).
- `clip=True`: clamp to `[0, 20]` first → `[0, 0, 20]` → finite output in `[-1,1]`.
**Invariant:** `clip=False` output is bit-identical to pre-merge pimm-data
(`transform.py:244-253`) for in-domain input.

### 4.4 `get_view` guard
- empty cloud (`coord.shape[0]==0`) → `raise ValueError`.
- 1-point cloud, `scale=(0.1,0.4)` → `int(uniform*1)` may be 0 → clamped to
  `size=1` → returns a 1-row view (no longer an empty crash).

### 4.5 v3 vertex co-transform
`coord` shape `(N,3)`, `vertex = [[10,20,30],[-1,-1,-1]]` (a valid + a sentinel),
`revision` unset:
- `RandomFlip(axes=('x',))` flips coord x AND `vertex[0]` x → `[-10,20,30]`;
  `vertex[1]` (all-`-1`) untouched.
- `NormalizeCoord` centers/scales coord and `vertex[0]` by the same centroid/scale;
  `vertex[1]` untouched.
- `PointClip` clips coord only; `vertex` unchanged.
**Invariants:** with NO `vertex` key, every geometric transform is bit-identical
to pre-merge (vertex calls are no-ops); sentinel `(-1,-1,-1)` rows are never
transformed; `vertex` subsets with `coord` under N-changing transforms (it's in
`index_valid_keys`).

### 4.6 `index_operator` prefix-match
data_dict with `coord (N,3)`, `segment_pid (N,1)`, `instance_particle (N,1)`,
`particle_idx (N,)`, `target_vertex (3,)`, no manual `Update`:
- `ShufflePoint`/`GridSample(train)` → `coord`, `segment_pid`, `instance_particle`,
  `particle_idx` all subset/reordered to the new N; `target_vertex` (leading dim
  3 ≠ N for N≠3) untouched, still `(3,)`.
- a key named `segmentation_meta` is NOT picked up (boundary match).
**Invariants:** every per-point decorated key co-moves with `coord`; per-event
`target_*` preserved; no duplicate `index_valid_keys` entries across chained
transforms.

---

## 5. Edge cases & error handling

1. **`RelativeLogNormalize` all-equal / single-point.** subtract-min → all zeros
   → output all `out_min`. No NaN. A single-element array works (min == the
   element).
2. **`RelativeLogNormalize` non-idempotent + order-sensitive.** Applying it twice
   re-subtracts the (now 0) min — second pass maps `[out_min..]` through `log1p`
   again, garbage. **Must run once, before any reorder/subset of the same column**
   (it relies on the per-event min of the *raw* `time`). Place it early in the
   pipeline, before `GridSample`/`ShufflePoint`. Document in the class docstring.
3. **`GridSample` empty `data_dict[key]`.** A `reducers` key absent from
   `data_dict` is skipped (`if rk not in data_dict: continue`) — matches branch.
4. **`mean`/`max` fill leakage.** No voxel is empty (every unique key has ≥1
   point), so `mean` never divides by zero and `min`/`max` fills are never
   observed. Documented as a non-issue but the fills are still the correct
   identities so any future empty-voxel path stays sane.
5. **`mean` int→float promotion.** A `mean`-reduced int column changes dtype to
   `float32`. Callers expecting int must not `mean` an int column (use `first` or
   `min`). Document.
6. **`first` ≠ survivor.** `first` picks the hash-sorted first point of the voxel;
   the voxel's surviving `coord` row is a `np.random` pick. They generally differ.
   `first` is for representative-agnostic, reproducibility-sensitive columns only.
7. **`LogTransform.clip=False` regression guard.** Default off; must not change
   any existing config's output.
8. **`get_view` empty cloud** raises `ValueError` (was an opaque downstream
   error). Callers (`MultiViewGenerator.__call__`) already require a non-empty
   `center_mask` region, so the raise is a real misconfig signal.
9. **Vertex shape guards.** `_valid_vertex_mask` returns `None` for wrong ndim/shape
   ⇒ vertex transforms no-op silently. A malformed `vertex` therefore can't crash
   a geometric transform; it just isn't co-transformed (acceptable — the dataset
   stamping `vertex` owns its shape).
10. **Prefix-match per-event coincidence.** A per-event `target_*` whose leading
    dim equals `n_points` would be subset incorrectly. Mitigated by: (a) per-event
    targets are `(3,)`/scalar (dim ≠ N for typical N), (b) downstream they're
    `_`-prefixed list-collated metadata. Known soft edge; flag if a config ever
    stamps an `(N,...)` `target_*`.
11. **`index_valid_keys` as tuple.** Some configs pass a tuple via `Update`. Both
    the branch revision block (`transform.py:99-100`) and §3.2 coerce to list
    before appending. Preserve that coercion.
12. **`ApplyToStream` list non-propagation.** `Update(index_valid_keys=...)` outside
    `ApplyToStream` does not reach the stream sub-dict. With prefix-match this is
    rarely needed, but document: put per-stream `Update` inside the stream's
    `ApplyToStream`.

---

## 6. Tests (numbered; setup → action → EXPECTED; tag; fixture)

Add to `tests/test_transforms.py` unless noted. Reference-parity tests
([parity-vs-branch]) import the branch module and assert equality; the impl plan
gates the *assertions* (not the design) on the branch being importable. If the
branch path isn't importable in CI, the parity tests `pytest.skip` with a clear
reason and the [new-behavior] tests still run. Helper proposed: a session fixture
`branch_transform_module` that tries
`importlib`-loading `…/particle-imaging-models/pimm/datasets/transform.py` and
skips if absent.

**T1 — `RelativeLogNormalize` negatives & no-NaN.** [new-behavior]. Fixture: none.
Setup: `time = np.array([-240, 0, 50, 8000], np.float32)`. Action: apply
`RelativeLogNormalize()`. EXPECTED: output `≈ [-1.0, -0.1998, -0.1274, 1.0]`
(atol 1e-3); all finite; dtype float32; in `[-1,1]`.

**T2 — `RelativeLogNormalize` all-equal / single.** [new-behavior]. Setup:
`time=[5,5,5]` and `time=[7.0]`. Action: apply. EXPECTED: `[5,5,5]→[-1,-1,-1]`;
`[7.0]→[-1.0]`; no NaN.

**T3 — `RelativeLogNormalize` constructor validation.** [new-behavior]. Action:
construct with `scale=0`, `max_val=-1`, `out_max<out_min`. EXPECTED: each raises
`ValueError`.

**T4 — `RelativeLogNormalize` missing key strict.** [new-behavior]. Setup:
`data_dict` without `time`. Action: apply `RelativeLogNormalize(keys=("time",))`.
EXPECTED: `ValueError` "Key time not found".

**T5 — `RelativeLogNormalize` parity.** [parity-vs-branch]. Setup: random
`time` incl. negatives. Action: pimm-data vs branch `RelativeLogNormalize` (same
ctor). EXPECTED: `assert_array_equal`.

**T6 — `GridSample` `min_keys`/`sum_keys` back-compat byte-equal.**
[parity-vs-branch]. Fixture: `jaxtpc_data_root` event or a synthetic
`coord`/`charge`. Setup: seed `random`/`np.random` identically. Action: pimm-data
`GridSample(grid_size=g, sum_keys=['charge'], min_keys=['t0_us'])` vs branch same.
EXPECTED: `assert_array_equal` for `coord`, `charge`, `t0_us`, `grid_coord`.

**T7 — `GridSample` `max` vs hand-rolled groupby.** [new-behavior]. Setup: known
`coord` placing 5 points into 2 voxels, `charge` known. Action:
`GridSample(reducers={'charge':'max'})`. EXPECTED: per-voxel max equals a manual
`np.maximum.reduceat`/groupby on the voxel assignment; dtype preserved.

**T8 — `GridSample` `mean` count-divide + int promotion.** [new-behavior]. Setup:
int `count_col` and float `charge` in known voxels. Action:
`reducers={'count_col':'mean','charge':'mean'}`. EXPECTED: `count_col` output dtype
float32 and equals sum/count; `charge` equals sum/count in its float dtype.

**T9 — `GridSample` `first` determinism across seeds.** [new-behavior]. Setup:
known multi-point voxels with a `plane_id` column. Action: run
`reducers={'plane_id':'first'}` under several distinct `np.random` seeds.
EXPECTED: the `first`-reduced `plane_id` is identical across all seeds (the
deterministic hash-sorted representative), even though the surviving `coord` rows
differ across seeds.

**T10 — `GridSample` `sum`/`min` via `reducers=` == via `sum_keys`/`min_keys`.**
[new-behavior]. Setup: same input, two `GridSample` configs (one using `reducers`,
one using `sum_keys`/`min_keys`), same seed. EXPECTED: identical output (shim
equivalence; explicit `reducers` wins when both given — separate assertion that
`reducers={'k':'max'}` + `min_keys=['k']` yields max).

**T11 — `GridSample` unknown op raises.** [new-behavior]. Action:
`GridSample(reducers={'charge':'median'})`. EXPECTED: `ValueError` listing allowed
ops.

**T12 — `LogTransform.clip` clamps domain.** [new-behavior]. Setup:
`energy=[-5,0,1e6]`. Action: `clip=True` vs `clip=False`. EXPECTED: `clip=True`
all finite, in `[-1,1]`; `clip=False` matches branch (parity) and is non-finite
for the `-5` entry. Add a [parity-vs-branch] sub-assert for `clip=True` vs branch.

**T13 — `LogTransform` default unchanged.** [parity-vs-branch / regression]. Setup:
in-domain `energy`. Action: `LogTransform()` (default `clip=False`). EXPECTED:
byte-identical to the pre-merge pimm-data output (capture a golden array) AND to
the branch.

**T14 — `get_view` empty raises.** [new-behavior]. Setup: a `MultiViewGenerator`
and `point={'coord': np.zeros((0,3))}`. Action: `get_view(point, center, scale)`.
EXPECTED: `ValueError`.

**T15 — `get_view` 1-point size clamp.** [new-behavior]. Setup:
`point={'coord': np.zeros((1,3))}`, `scale=(0.1,0.4)`, `view_keys=('coord',)`.
Action: `get_view`. EXPECTED: returned `view['coord'].shape[0] == 1`.

**T16 — v3 vertex co-transform under flip/rotate/scale/shift.** [parity-vs-branch].
Fixture: synthetic `coord (N,3)` + `vertex (M,3)` with one sentinel row. Setup:
seed identically. Action: run each of `RandomFlip`, `RandomRotate`,
`RandomScale`, `RandomShift`, `CenterShift`, `NormalizeCoord`, `PositiveShift`,
`ConditionalRandomTransform` on pimm-data vs branch. EXPECTED: `assert_array_equal`
on both `coord` and `vertex`; sentinel row unchanged in both.

**T17 — `PointClip` leaves vertex unchanged.** [new-behavior]. Setup: `coord`
exceeding range + `vertex`. Action: `PointClip`. EXPECTED: `coord` clipped,
`vertex` bit-identical to input.

**T18 — Geometric transforms no-op vertex when absent.** [parity-vs-branch /
regression]. Setup: data_dict with `coord` only (no `vertex`). Action: run all the
geometric transforms. EXPECTED: identical to pre-merge pimm-data (golden) — porting
the vertex hooks changed nothing without a `vertex` key.

**T19 — `index_operator` prefix-match subsets per-point keys.** [new-behavior].
Setup: `coord (10,3)`, `segment_pid (10,1)`, `instance_particle (10,1)`,
`particle_idx (10,)`, `sensor_idx (10,)`, no `Update`. Action: `ShufflePoint` then
`GridSample(train)`. EXPECTED: all four label/FK keys have the new leading dim and
are consistently permuted/subset with `coord` (verify via a tagged identity column
carried as `particle_idx`).

**T20 — `index_operator` per-event `target_*` NOT subset.** [new-behavior]. Setup:
`coord (10,3)` + `target_vertex (3,)` + `target_energy` scalar/`(1,)`. Action:
`GridSample(train)` reducing N to <10. EXPECTED: `target_vertex` still shape `(3,)`,
`target_energy` unchanged.

**T21 — `index_operator` underscore-boundary.** [new-behavior]. Setup: keys
`segment_pid (N,1)` (should carry) and `segmentation_meta (N,2)` (should NOT).
Action: `ShufflePoint`. EXPECTED: `segment_pid` permuted with `coord`;
`segmentation_meta` left as-is (not in `index_valid_keys`, unchanged length/order).

**T22 — `index_operator` no duplicate entries on chaining.** [new-behavior].
Setup: data_dict with `segment_pid`. Action: `ShufflePoint` then `RandomDropout`
then `GridSample`. EXPECTED: `index_valid_keys` contains `segment_pid` exactly once.

**T23 — `MixedScaleGeometryMultiViewGenerator` parity.** [parity-vs-branch].
Fixture: synthetic point cloud with `coord`/`energy`. Setup: seed identically.
Action: run pimm-data vs branch with matching ctor (`fine_center_mode='geometry'`
and `'random'`). EXPECTED: equal `global_*`/`local_*` arrays and `global_offset`/
`local_offset`.

**T24 — registered-count bump.** [new-behavior]. Update the existing
`test_transforms_registered_count` (`test_transforms.py:48-56`): assert
`RelativeLogNormalize` and `MixedScaleGeometryMultiViewGenerator` are now in the
registry and bump the `>= 48` floor to `>= 50`.

**T25 — end-to-end JAXTPC seg with reducers + prefix-match.** [new-behavior].
Fixture: `jaxtpc_data_root`. Setup: pipeline
`ApplyToStream(stream='step', [GridSample(reducers={'charge':'sum'}, return_grid_coord=True), ToTensor]) → Collect(stream='step', keys=('coord','grid_coord','segment'), feat_keys=('coord','energy'))`.
Action: `collate_fn([ds[0], ds[1]])`. EXPECTED: batch has `coord`/`feat`/`segment`/
`offset`; `segment` length == `coord` length (prefix-match kept `segment_*` aligned
through `GridSample`); runs without desync error. (Mirrors existing
`test_end_to_end_transform_collate`, `test_transforms.py:68-91`.)

---

## 7. Reversible defaults & risks (D34)

**Reversible defaults chosen (documented in code, not relitigated):**
- **Reducer API shape:** `reducers={key: op}` dict; `sum_keys`/`min_keys` kept as a
  shim; **explicit `reducers` wins** on conflict. Reversible: could instead make
  the shim error on conflict.
- **`first` semantics:** deterministic hash-sorted first point
  (`idx_sort[cumsum(insert(count,0,0)[:-1])]`), NOT the random survivor. Reversible:
  could alias `first` to the `coord` survivor — rejected for reproducibility.
- **`mean` int promotion:** int→`float32`. Reversible: could keep `float64` or
  round back to int.
- **`max` unsigned fill:** `iinfo(uint).min == 0` (correct identity for
  non-negative data). Reversible if a signed-in-unsigned case ever appears (it
  can't).
- **Prefix-match token rule:** underscore-boundary (`name == p or name.startswith(p+"_")`),
  prefixes `("segment","instance","target")` + explicit
  `("particle_idx","sensor_idx","plane_id")`. Reversible: a registry of explicit
  axis names if a collision ever appears.
- **Per-event exclusion rule:** leading-dim `!= n_points` (or `ndim==0`). Reversible:
  an explicit per-event allowlist.

**Risks:**
- **`RelativeLogNormalize` non-idempotent ordering** (§5.2) — must precede any
  same-column reorder/subset. Pipeline-author responsibility; documented in
  docstring.
- **Prefix-match per-event coincidence** (§5.10) — an `(N,...)` `target_*` would be
  wrongly subset. Flag if any config stamps one.
- **`mean` dtype change** (§5.5) silently breaks int-expecting consumers.
- **Parity tests depend on branch importability** — when the branch path is
  unavailable, parity assertions skip; the [new-behavior] tests still gate. Do not
  let a skipped parity test mask a real regression — pair each parity test with a
  golden-array regression where feasible (T13, T18).
- **Do NOT touch `Collect`** — pimm-data's is ahead (`stream=`, autoconvert,
  passthrough); a merge-from-branch would regress it.

---

## 8. Dependencies on other parts

- **`label_config` decoration (Part: datasets/labels, §3.5 of impl plan)** produces
  the `segment_*`/`instance_*`/`target_*`/`particle_idx`/`sensor_idx`/`plane_id`
  keys that §3.7's prefix-match carries. This part assumes those names; the decorator
  must emit exactly the named schema keys (`segment_pid`, `instance_particle`,
  `instance_interaction`, `target_vertex`, `target_energy`, …) for the prefix-match
  to pick them up automatically.
- **Per-event `target_*` carriage (§3.5 of impl plan)** — per-event targets are
  also carried as `_`-prefixed list-collated metadata; §3.7's shape-exclusion is the
  in-transform half of keeping them point-subset-safe.
- **`ApplyToStream` (`detector_transforms.py`)** is the per-stream wrapper (D25):
  N-changing transforms + their `index_valid_keys`/`Update` must live inside it.
  Unchanged by this part; this part's prefix-match makes most explicit `Update`s
  unnecessary.
- **`RelativeLogNormalize` consumer:** the LUCiD-SSL config (`time`/`T` column)
  and the reader surfacing `T`/`T_reco` (§3.4 of impl plan). The `max_val`/`scale`
  window values are a model/config choice (D13); only the negative-time correctness
  (subtract-min/clip) is fixed here (D31).
- **v3 `vertex`/`is_primary` consumer:** dormant until the pilarnet v2→v3 merge (Rb,
  §5 of impl plan) and a dataset that stamps `revision="v3"` + a `(N,3)` `vertex`.
  Until then the hooks are no-ops; this part only ports them so the seam is ready.
- **Test matrix (Step 0, §6 of impl plan):** the parity/golden tests here are part
  of the gate that must be green before any pimm-side flip.
