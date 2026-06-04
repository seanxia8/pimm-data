# Handoff: GPU / batched transforms (e.g. train-time noise simulation)

Status: design exploration, no code written. This document states facts and
options for a follow-up agent/person to investigate. It deliberately does
**not** pick a winner.

## 1. The goal

We want to add train-time data transforms that are faster on GPU — the
motivating example is **detector noise simulation** applied as an on-the-fly
augmentation (fresh noise realization per epoch on a stored *noiseless*
signal). Decide the most general / generalizable way to fit such transforms
into the pipeline. An explicit sub-question: **should these run in the
trainer, in the data pipeline, or as a standalone component?** Evaluate; do
not assume the trainer.

## 2. Facts — current pimm-data transform pipeline

(Repo `/sdf/group/neutrino/omara/pimm-data`.)

- **Registry + dict-config + `Compose`.** Transforms are classes decorated
  `@TRANSFORMS.register_module()` and instantiated from dicts
  (`dict(type='GridSample', ...)`). `Compose` folds a list:
  `for t in transforms: data_dict = t(data_dict)`
  (`src/pimm_data/transform.py`). `Compose` accepts any callable, not just
  registered ones.
- **Where transforms run:** per-sample inside `DefaultDataset.__getitem__`
  → `prepare_train_data` (`src/pimm_data/defaults.py`), i.e. in DataLoader
  **worker processes**, on **CPU**, on **numpy** arrays. `Collect`
  (`transform.py`) is the numpy→torch boundary and is documented as the
  **last** transform (enables zero-copy tensor IPC from workers).
- **Streams:** datasets emit a nested dict
  `{'step':{...},'hits':{...},'sensor':{...},'labl':{...},'bridges':{...}}`
  (`src/pimm_data/jaxtpc.py`). `ApplyToStream(stream=..., transforms=[...])`
  (`src/pimm_data/detector_transforms.py`) runs an inner `Compose` on one
  stream's sub-dict. It is therefore inherently **pre-Collect, per-sample,
  per-stream**.
- **Collation:** `collate_fn` (`src/pimm_data/collate.py`) concatenates
  variable-length samples into a single batched cloud — `coord (ΣN, D)`,
  `feat (ΣN, F)`, `segment`, and an `offset` tensor (cumulative per-sample
  point counts) that marks sample boundaries. After `Collect`+collate the
  per-stream structure is gone; you have one flat batched cloud.
- **No device/CUDA anywhere** in transform.py / detector_transforms.py /
  collate.py / defaults.py / jaxtpc.py (grep: zero hits). The library has
  **no model code**; it ends at `collate_fn`.
- Standard data-library characteristics: numpy/scipy internals in several
  transforms (`GridSample` FNV hashing, `cKDTree`, `np.unique`) — these are
  CPU-bound and not trivially device-agnostic.

## 3. Facts — loader performance context

- Stage-by-stage profiling (this session; artifacts under
  `/sdf/home/o/omara/neutrino_data/omara/doraemon/profiling/`) found loading
  is **CPU-decompression-bound**: ~60% of `__getitem__` is HDF5 decode +
  numpy assembly, ~1% is the existing transforms, and the GPU sits idle
  waiting for data. Thread-scaling is flat (h5py global lock); only
  *process* parallelism helps.
- Consequence: **moving the existing cheap transforms to GPU would not speed
  loading** (Amdahl: optimizing ~1%). Noise is **new compute**, not a
  relocation of existing work — a different question.

## 4. Facts — the noise model (lives in JAXTPC, JAX)

(Repo `/sdf/group/neutrino/omara/JAXTPC`.)

- `tools/noise.py`: MicroBooNE intrinsic-noise model
  (`RMS_ADC = sqrt(x² + (y + z·L)²)`, L = wire length) with empirical FFT
  **spectral shaping** from `config/noise_spectrum.npz`. Implemented in
  **JAX**. There is also coherent (per-group correlated) noise and an
  electronics response (`tools/electronics.py`, RC⊗RC via FFT).
- Applied **densely**: inside the JIT sim (`tools/simulation.py`) on the
  response array of shape `(num_wires, num_time)` per plane, **before**
  digitization and thresholding.
- Therefore noise is: **dense per-plane (wire × time), FFT-based,
  batchable, and *generative*** — it adds signal everywhere, so thresholding
  produces new hits and the per-event point count **N changes**.
- Production (`production/run_batch.py`, `production/save.py`) can write with
  noise/electronics ON or OFF. The current doraemon dataset was produced
  with **noise + electronics OFF** (digitization ON) → the stored sensor is
  effectively **noiseless**. So "store clean once, add noise per-epoch on
  GPU" is feasible with existing data.

## 5. Candidate execution placements (neutral tradeoffs)

| Option | Where it runs | Owner | Notes |
|---|---|---|---|
| A. Trainer hook (`on_after_batch_transfer` / `forward`) | training process, on device, post-collate | the consumer's trainer | Simplest to add. Pipeline becomes split across loader-config + trainer-code (dataset no longer self-describing); couples to a specific trainer framework. |
| B. Device-loader wrapper / iterator owned by pimm-data (DALI-style) | a thin iterator around the DataLoader that transfers + applies batch transforms; yields on-device batches | pimm-data | Framework-agnostic driver (works with any loop or trainer); keeps the full pipeline config-described. Means pimm-data optionally touches CUDA. |
| C. `collate_fn` | DataLoader worker / main process | pimm-data | CUDA in DataLoader workers is unsafe with `fork`; collate is CPU. Not viable for GPU. |
| D. Model `forward` (Kornia-style `nn.Module`) | model, on device | model author | Differentiable/on-device, but ties augmentation to a specific model; not reusable across models. |
| E. Offline pre-bake (transcode a noised copy) | batch job | production | Not augmentation (noise fixed, not per-epoch-fresh); loses the diversity/storage benefit. |

Note: options A, B, and D can all drive the *same* transform objects if those
objects are batched + device-agnostic; they differ mainly in **who owns the
driver** and **how portable/reproducible** the result is.

## 6. Cross-cutting design concerns (facts/constraints to resolve)

1. **Generality / reproducibility.** Today a dataset's full transform
   pipeline is describable by a config list. Any placement that lives in
   trainer code (A) or model code (D) makes the augmentation *not* captured
   by the dataset config. A config-described, driver-pluggable component is
   more portable across consumers.
2. **Single source of truth for the physics.** The noise model exists once,
   in JAX (JAXTPC). pimm-data is torch. Options: (a) reuse the JAX impl via
   dlpack/zero-copy torch↔JAX on GPU — one source of truth, but JAX+torch in
   one process has caveats (XLA memory preallocation, init order); (b) port
   noise to torch in pimm-data — clean integration but **drift risk** (cf.
   the `charges_i16` pixel bug found this session, where the synthetic
   fixture/reader diverged from the writer); (c) factor the noise/electronics
   math into a framework-neutral spec both consume. Either way a **parity
   test** (same seed → same noise within tolerance, vs `tools/noise.py`) is
   advisable. Statistical match between *baked* (production) and *augmented*
   (train) noise matters if both are used.
3. **Generative / N-changing.** A noise transform adds hits → changes the
   point count. The batch-transform API must allow returning a *new* batch
   (new `coord`/`feat`/`offset`), not just in-place same-shape edits. Most
   image-augmentation APIs assume shape-preserving ops; this one does not.
4. **Data representation.** Noise needs a **dense** per-plane `wire × time`
   tensor (FFT) + plane **geometry** (`num_wires`, `num_time`), then
   re-threshold → re-sparsify. The loader currently emits a **sparse** cloud;
   geometry lives in each file's `/config`. So the stage must either densify
   on device (carrying geometry through collate) or the loader must expose a
   dense sensor buffer. **Memory:** a full `B × planes × wires × time` grid
   is large (e.g. ~2k wires × ~2.7k ticks × 6 planes × batch); active-region
   + sampled-noise-wire schemes reduce it.
5. **Device-agnostic.** Ideally the transform runs on CPU (tests / no-GPU)
   and GPU (training) from one code path (whatever device the batch is on).
6. **Seeding / reproducibility.** Needs per-batch deterministic RNG that
   works regardless of where the stage runs; note CPU-worker RNG vs CUDA RNG
   differences and per-worker seeding pitfalls.
7. **Overlap with training.** GPU augmentation on the training device
   contends with the model and is **not** hidden behind worker prefetch
   (unlike CPU transforms), unless run on a separate CUDA stream or via a
   dedicated engine (DALI). It still avoids redoing decode, but it is not
   "free."
8. **Repo ownership / scope.** pimm-data is a data library (ends at
   `collate_fn`). JAXTPC is the simulator (owns the noise physics). Decide
   which repo owns: (i) the noise physics, (ii) the batch-transform
   framework/registry, (iii) the execution driver. A "detector-effects"
   layer (noise/electronics/digitize) usable by *both* production and
   training is one way to keep a single source of truth.

## 7. Prior art to check (the next agent should search)

- How LArTPC / LiDAR / sparse-detector ML frameworks apply *stochastic
  detector effects* (noise) as train-time augmentation, if at all — vs
  baking them into the dataset.
- NVIDIA DALI's external-source / GPU-pipeline model as a reference for a
  device-driver that overlaps preprocessing with training.
- torch↔JAX interop in practice (dlpack), and whether anyone runs a JAX
  physics kernel inside a torch training loop on the same GPU.
- Pointcept / MinkowskiEngine / spconv: do any expose a post-collate,
  on-device, batched transform hook, and what is its API shape?
- (Earlier session research, for context: general CPU-vs-GPU transform
  placement, point-cloud framework conventions, and CPU-bound-loader
  remedies — all pointed to CPU-per-sample as the norm and GPU reserved for
  genuinely heavy/batched compute. Noise is the latter.)

## 8. Open questions to answer before writing code

1. **What does the model consume** — the sparse sensor cloud, or a dense
   readout? This drives the representation and whether N changes.
2. **Physics reuse:** JAX-via-dlpack vs torch port vs shared spec? Benchmark
   interop cost and verify parity with `tools/noise.py`.
3. Should there be a **shared detector-effects component** (noise +
   electronics + digitize) callable by both production and training?
4. **Memory budget** for dense per-plane noise at realistic train batch
   sizes; dense-everywhere vs active-region+sampled-noise.
5. **Driver:** trainer hook (A), pimm-data device wrapper (B), or both
   driving the same transform objects? What trainer(s) must be supported?
6. **API for N-changing batch transforms** (return new batch dict; how
   `offset`/segment/instance are rebuilt).

## 9. Key file references

- pimm-data: `src/pimm_data/transform.py` (Compose, Collect, TRANSFORMS
  registry, GridSample), `src/pimm_data/detector_transforms.py`
  (ApplyToStream), `src/pimm_data/collate.py` (offset construction),
  `src/pimm_data/defaults.py` (`__getitem__`/`prepare_train_data`),
  `src/pimm_data/jaxtpc.py` (nested-dict schema), `README.md` (transform
  recipes + perf guidance).
- JAXTPC: `tools/noise.py`, `tools/electronics.py`, `tools/simulation.py`
  (where noise applies, dense, pre-digitize), `config/noise_spectrum.npz`,
  `production/run_batch.py` + `production/save.py` (noise toggles, `--codec`).
- Profiling evidence: `…/doraemon/profiling/` (decode-bound breakdown).
