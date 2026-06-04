# Plan: densification + noise for pimm-data → pimm (v3)

Status: design proposal, converged after 5-agent review **and** clarification
from the data owner. Companion to `gpu_batch_transforms_handoff.md` (neutral
fact base). v1 proposed a `BatchTransform` framework + torch noise port,
framework-first. v2 (post-review) split the work and surfaced two foundational
problems. **v3 incorporates owner clarifications that dissolve both problems and
collapse the design to two small, composable, device-agnostic stages.**

Repos:
- `pimm-data` — data library: `/sdf/group/neutrino/omara/pimm-data`
- `pimm` (particle-imaging-models, Pointcept fork):
  `/sdf/home/o/omara/.claude/jobs/21ffc656/particle-imaging-models`
- `JAXTPC` — simulator (owns noise physics): `/sdf/group/neutrino/omara/JAXTPC`

---

## 0. Owner clarifications that shaped v3

1. **Stored sensor is signal-only (no noise);** noise is added downstream.
2. **`sensor` and `hits` are kept separate by choice — no `hits`→`sensor`
   conversion** (no response convolution in the training path). The v2 "label
   contradiction" and "response-operator crux" are both **void**: nothing is
   converted between streams.
3. **Sparse is the default methodology for everything**, including labels and
   3D `step` (standard Pointcept: per-point `segment`, `GridSample`, sparse
   voxelization). Untouched by this work.
4. **Densification is an optional operation, wanted on BOTH CPU and GPU** (for
   flexibility), **separated per plane**, primarily for `sensor`. `hits` is
   densifiable as an **opt-in**, not the default; labels are *not* produced by
   densifying hits.
5. **Noise is a separate step** from densification, **sensor-only**, fresh per
   epoch, statistically faithful to JAXTPC `tools/noise.py`.
6. **No re-sparsify** — densified output is handed to the (dense) model as-is.
7. 2D/dense consumer is **near-term**. pimm may depend on pimm-data (submodule
   OK).

---

## 1. The design (two decoupled stages)

### Stage A — `Densify` (generic, device-agnostic, per-plane, batched)
Pure sparse→dense scatter. Stream-agnostic (`sensor` default; `hits`/others
opt-in). Output is **per-plane**: `{plane_id: (B, [C,] W_p, T)}`. Device follows
the input tensors → the **same code is the CPU and GPU implementation**.

**Optimal batched scatter** — vectorized over hits *and* batch (no per-sample
Python loop), one `index_add_` per plane (P ≤ 6):

```python
# post-collate sparse sensor batch:
#   wire,time,value,plane_id : (ΣN,)   offset : (B,)
#   config (keyed by plane_id): num_wires[p], T = num_time
batch_idx = offset2batch(offset)              # (ΣN,) sample of each hit
grids = {}
for p in planes:
    m = plane_id == p
    b, w, t, v = batch_idx[m], wire[m], time[m], value[m]
    Wp = num_wires[p]
    flat = (b * Wp + w) * T + t               # int64 flat index into (B,Wp,T)
    buf = v.new_zeros(B * Wp * T)
    buf.index_add_(0, flat, v)                # one scatter, CPU or CUDA unchanged
    grids[p] = buf.view(B, Wp, T)
```

- O(ΣN) traffic; P scatters (tiny); zero per-sample loops; `index_add_`/views
  are device-agnostic. Generalizes to `feat (ΣN,C)` → `(B,C,W_p,T)`.
- Micro-opt (optional): gather per-hit `W_p`/`base[plane_id]` and `index_add_`
  **once** into a ragged concatenated buffer, slice per plane — a single kernel.
  Unnecessary for P ≤ 6.
- Clamp `wire∈[0,W_p)`, `time∈[0,T)`; `flat` int64; `value` float32.

### Stage B — `AddIntrinsicNoise` (sensor-only, separate)
Operates on the densified sensor grids; **shape-preserving** (no N-change — we
don't re-sparsify). Per plane:
`grids[p] += intrinsic_noise((B,W_p,T), wire_lengths[p], spectrum, generator,
device=grids[p].device)`. Torch-FFT port of `tools/noise.py`; fresh
`torch.Generator` seeded from `(base_seed, epoch, iter, rank, p)`; statistically
faithful (see §4). Skipped for non-sensor cases ("noise is a different step").

### Composition
- **SSL/denoising:** `Densify(sensor)` → clean grids (target); clone →
  `AddIntrinsicNoise` → noised grids (input). One densify, stages reused.
- **Supervised:** standard **sparse** path (per-point `segment`, GridSample);
  densify optional for a dense sensor input, labels from the sparse side.
- **Opt-in hits-densify:** `Densify(stream='hits')`, no noise stage.

---

## 2. CPU/GPU flexibility = placement, not a second implementation

Both stages are device-agnostic torch (device = input device), so "CPU vs GPU
densify" is **where the stage sits relative to the `.cuda()` transfer**, one
code path. The driver exposes a **pre-transfer** and a **post-transfer**
post-collate stage list; drop `Densify`/`AddIntrinsicNoise` into whichever.

- **GPU (recommended for training):** sparse batch → `.cuda()` → `Densify` →
  `AddNoise` → model. **Only sparse hits cross PCIe;** dense grids are born on
  the GPU.
- **CPU (tests / no-GPU / debug):** stages run on the collated CPU batch. Note:
  CPU-densify + GPU-model transfers the **dense** grids (~100s MB) — expensive;
  use CPU densify for CPU-only/validation, not the fast path.

---

## 3. Where things live (ownership)

| Concern | Owner |
|---|---|
| `Densify`, `AddIntrinsicNoise` (torch, device-agnostic), parity test | **pimm-data** |
| Readers, datasets, per-sample transforms, `collate_fn`, registry, `Compose` | **pimm-data** |
| Noise physics SoT (JAXTPC `tools/noise.py` + `noise_spectrum.npz`) | **JAXTPC**; pimm-data hosts a parity-tested torch port |
| Engine/trainer/hooks/models, DataLoader build | **pimm** |
| Post-collate driver (pre/post-transfer stage lists) + per-rank/epoch RNG seeds | **pimm** |

pimm-data never imports `torch.cuda`; device follows the batch.

---

## 4. Noise fidelity (statistical, not bitwise)

Port `_noise_core` (`tools/noise.py:89-140`): per-wire rfft spectral shaping +
white term + per-wire series-RMS scaling from `wire_lengths`. Traps: force **DC
real** (Nyquist only if `num_time` even — production `num_time=4321` is odd, so
DC-only); renormalize per-wire *series* RMS but **not** the white term; port
`_get_noise_spectrum_shape` energy-normalization exactly, keyed to `num_time`.
Units: kernels are ADC/electron so the stored signal is already ADC; noise (ADC)
adds directly. Stored sensor has electronics+noise OFF — assert that invariant.
JAX Threefry ≠ torch Philox, so the **parity test is statistical**: per-wire
series RMS ≈ `y+z·L`, white RMS ≈ `x`, total ≈ `sqrt(series²+white²)`, ensemble
PSD ≈ interpolated `spectrum_shape`. Name an owner who reruns on model change.
(Cleaner long-term: a shared "detector-effects" package consumed by production
*and* training, eliminating the port — open option.)

Pixel readout: noise is `_noop` in JAXTPC; `Densify` still works (3D `(Py,Pz,T)`
grid, memory-heavy, opt-in) but `AddIntrinsicNoise` does not apply.

---

## 5. Memory / batching

Per-plane separation bounds memory — process planes sequentially if the model is
per-plane (never all planes resident). Wire grid ≈ `W_p×T` (~2000×4321 ≈ 34 MB
fp32); `(B,W_p,T)` at B=8 ≈ 270 MB/plane. If the full dense batch exceeds
budget, chunk the `B` axis in the scatter and concatenate; the model forward
usually dominates peak anyway.

---

## 6. Track A — De-fork (do now, ungated)

Independent of all densify/noise work; pays off immediately by killing the
stale-schema drift. (Owner confirmed pimm may depend on pimm-data.)

- pimm imports pimm-data's `TRANSFORMS`/`Compose`/`Collect`/collate **directly**
  (registry option (a); federation by dotted name is dead code in both repos)
  and deletes its vendored data layer (`pimm/datasets/transform.py`,
  `detector_transforms.py`, the collate in `utils.py`); repoint
  `dataloader.py`'s collate import. The suite is a superset; `anchors.py` and
  collate are byte-identical → low risk.
- **Real work (not suite reconciliation):** the **flat→nested `JAXTPCDataset`
  output change** breaks every existing JAXTPC config — migrate them
  (`modalities='seg'→'step'`, wrap per-sample transforms in
  `ApplyToStream(stream=...)`, `Collect(stream=...)`, repoint data-root). Same
  for **`LUCiDDataset`** (base-class + reader-set rewrite). Enumerate
  `configs/detector/_base_/jaxtpc_seg.py` + inheritors.
- Packaging: pin `torch` as a floor (or extra) so pimm controls the CUDA build;
  **vendor pimm-data as a git submodule** and snapshot its SHA (`scripts/train.sh`
  only `cp -r`s `scripts tools pimm`; `__version__` is a moving `0.1.0`).

---

## 7. pimm changes for Track B

- **Post-collate driver, config-described,** with pre-/post-transfer stage
  lists. Insert after the `.cuda()` loop (`train.py:225-228`, before `model()`
  at 233) for the GPU list, and before it for the CPU list. Build the `Compose`
  once in `before_train`.
- **Seeding:** independent `torch.Generator` from `(base_seed,epoch,iter,rank,p)`
  — not the global RNG (`set_seed` once → not fresh), not the
  `cfg.seed+rank*num_workers` worker namespace.
- **Eval/test:** noise is **train-only** by default (documented); eval stays
  clean/deterministic. If robustness-eval is later wanted, route the five
  eval/test `.cuda()` sites through a shared helper (or add one
  `after_batch_transfer` hook point) rather than duplicating.
- Note `avg_pts` throughput (`train.py:236-237`) is unaffected (densify/noise
  are shape-preserving on the dense grids; the sparse `coord` count is unchanged
  until the model consumes the dense grids).

---

## 8. Sequencing

1. **Track A** (de-fork) — now, ungated.
2. **Track B** — prototype `Densify` + `AddIntrinsicNoise` end-to-end
   (wire, single geometry), validate the **statistical** parity test, then wire
   the config-described driver + CPU/GPU placement. Near-term consumer exists.

---

## 9. Open / minor

- Sensor densify channels: single value `(B,W_p,T)` vs multi-channel `feat`.
- `B`-axis chunking threshold / VRAM budget.
- Detector-effects shared package vs torch port + statistical parity (§4).
- Geometry source: baked into transform config keyed by `plane_id` (duplicates
  detector geometry into pimm config; no auto consistency check vs data — flag).

## 10. Scope guards (NOT doing)

Not converting `hits`↔`sensor` (no response in training). Not densifying hits by
default. Not re-sparsifying. Not noising anything but `sensor`. Not a
`BatchTransform` class hierarchy (stages are plain device-agnostic callables;
position is config). Not a pimm-data CUDA device-loader. Not touching the
standard sparse path (labels, 3D `step`). Not gating the de-fork on any of this.
