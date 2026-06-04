# Plan v4 — GPU batch transforms for pimm-data (consolidated, post 8-agent review)

Supersedes the *timing* ("near-term consumer") and *driver-placement* ("pimm owns
the driver") framing of `gpu_batch_transforms_plan.md` v3. v3 remains valid only
for the Track-B internal two-stage shape + noise fidelity (§4) + memory (§5).
Authority for placement/sequencing: `DESIGN.md`/`ROADMAP.md` + the
`engagement_plan_transform_dataset_placement.md` Part VIII decision log.

## 0. Status & governance — PHASE 0

**IMPLEMENTATION STATUS (2026-06): Phases 1–4 DONE** (uncommitted).

*Phase 3 (pimm trainer):* `engines/train.py` — `Trainer.before_train` builds
`self.gpu_transforms` from `cfg.gpu_transforms` + `dataset.plane_geometry()`;
`run_step` diverts to `apply_batch_transforms(... device='cuda', base_seed, epoch,
rank=comm.get_rank())` when stages exist, else the **unchanged** `.cuda()` loop
(sparse configs are byte-identical). Compiles; full pimm run blocked by *pre-existing,
unrelated* env skew (`MultiModalEventDataset` not in this pimm-data; missing
`tensorboardX`) — the dense flow itself is covered by pimm-data's tests. Config schema
(top-level, policy-only, no geometry):

    gpu_transforms = dict(coherent=True, incoherent=True, n_bits=12)   # omit/empty => sparse

*Phase 4 (perf/deployment):* the per-event loop in `add_intrinsic_noise` already
**bounds the 9× irfft transient to one (event, plane)** — no large-B transient blowup;
peak VRAM is the output grids (B·ΣW_p·T) + model. DDP-per-GPU is the multi-GPU mode
(seed folds `rank`; a dedicated transform-GPU re-incurs the dense PCIe cost — avoid).
Offline-bake fallback: omit `AddIntrinsicNoise` from the stage list and pre-noise once
via the numpy `pimm_data.noise` path. `cache=True` + GPU-noise is the synergistic combo.

*10-agent adversarial review (2026-06) — FIXED:* `avg_pts` KeyError on the dense
path (pimm `train.py`, guard on `coord`); `densify` now asserts length-consistency
+ plane-in-registry + in-bounds (catches GridSample-on-sensor corruption, missing
planes, geometry mismatch loudly); removed `wire/time/value/plane_gid` from
`index_valid_keys` (immutable scatter inputs — kept `plane_id` for the sparse path);
`canonical_plane_id` rejects malformed labels; honest docstrings + a loud warning
for incoherent CPU↔CUDA device-specificity and the missing-`name` seed fallback;
sorted cross-plane draw order; salt masked to 63 bits. Added 13 tests
(de-tautologized coherent-vs-JAXTPC at **odd+even** num_time, per-event seed
variation, volume-filter/empty-plane, densify guards, CUDA noise/digitize/e2e).
**206 passed/6 skipped.** *Remaining (documented, not blocking):* eval/test/probe
sites don't densify yet → needed when a dense model exists (Phase 3 follow-on);
registry built from event-0 can miss planes absent there (now a LOUD densify error,
deep fix = build from config); coherent host→device transfer is ~B×P small copies
(Phase-4 perf); no OOM guard (peak ≈ B·ΣW_p·T·4B); `pin_memory=True` required for
async transfer; helix `DetectorConfig.pedestals` defaults are stale (2048/2048/400
vs 1843/1843/410) — harmless (file-populated).

*Phase 1 + Phase 2 (pimm-data):*
`canonical_plane_id` + flat scatter inputs + `plane_geometry()` registry; `Densify`
uniqueness/integer asserts + `Digitize` no-double guard + unique fixture; `dense_ops.py`
(torch `densify`/`add_intrinsic_noise` coherent+incoherent/`digitize`) + `batch_transforms.py`
(`move_to_device`/`apply_batch_transforms`/`build_sensor_gpu_stages`/`BatchTransformMixin`).
193 passed / 6 skipped; densify torch↔numpy bit-exact (CPU **and** CUDA-verified), coherent
bit-exact vs JAXTPC, incoherent statistical, no-`torch.cuda` gate green; GPU end-to-end
born-on-GPU smoke OK (6-plane dict on cuda). **Deferred:** `Collect(into=)` multi-stream
(only needed for supervised sensor+labels; SSL/denoise is sensor-only); **Phase 3** trainer
hookup in `pimm`; **Phase 4** perf (B-chunking for the 9× irfft transient, DDP-per-GPU).

**Owner decision (2026-06):** Track B is **revived but sequenced BEHIND remaining
in-flight data-layer work** (not the immediate priority; queued). When executed, the
**first scope is Phase 1 + Phase 2** (pimm-data-only: CPU correctness/plumbing +
torch `dense_ops` + reference runner + parity), with the trainer hookup (Phase 3)
after. The de-fork is already done, so Track B can land in parallel with no gating
once its turn comes. Remaining Phase-0 items below (oracle disposition,
noise-SoT owner, hybrid-RNG split) stand as **recommended defaults** — confirm at
kickoff, not blocking the queued status.

**Resolved design answers (2026-06):**
- **Fresh-per-epoch noise on the GPU path** — confirmed (the justification for Track B); not offline-baked.
- **Dense output = per-plane dict** `{plane_id: (B, W_p, T)}`; ragged W_p (U/V=1969, Y=1443) handled naturally; plane count set by `volume=` (3 or 6).
- **BOTH coherent + incoherent** noise are in scope — this **reverses §1.4's scope-out**. The dense training data is the **noise-free** doraemon sensor (`include_intrinsic/coherent/electronics=False`), so adding both fresh is correct (no double-count). Requires surfacing per-wire `wire_lengths` (§1.4 revised). Keep a `coherent`/`incoherent` toggle for datasets that already carry baked noise.
- **Runner home = pimm-data.** It owns `apply_batch_transforms`/`build_batch_transforms`/`move_to_device`/seed-derivation/stages (device-agnostic, never imports `torch.cuda`, tested end-to-end on CPU device + CUDA `importorskip`), plus a thin `on_after_batch_transfer`-signature adapter. The current custom Pointcept loop calls `apply_batch_transforms` **inline** in `run_step`. Rejected "pimm owns the driver" (untested orphaned callables).

### (original Phase-0 framing follows — no code until confirmed)

**Two facts the other docs don't state and that reframe everything:**
1. An **uncommitted numpy-host prototype already exists** in the tree:
   `src/pimm_data/noise.py` (numpy `generate_noise`/`incoherent_noise`/`coherent_noise`/`digitize`)
   and per-sample `Densify`/`AddNoise`/`Digitize` in `detector_transforms.py` (run in the
   worker via `ApplyToStream`). This plan **re-architects** it (post-collate, device-agnostic
   torch) while **keeping the numpy version LIVE** as both the CPU per-sample path and the
   bit-exact parity oracle. The numpy code is *not* deleted.
2. **Track B was deferred + re-homed** by `engagement_plan` D1/D33 (wire-TPC-only, NOT LUCiD,
   sequenced after the de-fork). **The de-fork (Track A) is already DONE** — pimm imports
   `pimm_data` directly (`pimm/datasets/__init__.py`).

**Phase 0 owner confirmations (gate the whole plan):**
- Is there a committed **dense-sensor (wire-TPC) consumer** now? Track B starts only if yes.
- Track B **parallel** to / **behind** any remaining de-fork? (Recommended: parallel — it
  touches disjoint files and the de-fork is done.)
- Bless the **runner living in pimm-data** (device-agnostic, never imports `torch.cuda`) —
  this contradicts v3 §3/§7 and is what fixes the orphaned-API/testability critique.
- Keep `noise.py` as the committed **reference oracle** (recommended) vs delete+re-derive.
- **Noise SoT stays JAXTPC**; pimm-data hosts a parity-tested torch port; name the owner who
  re-runs parity on any JAXTPC noise change.
- Bless the **hybrid-RNG split**: coherent = numpy-host **bit-exact**; incoherent =
  torch-device **statistical-only** (and incoherent is **scoped out** for now — see §1.4).

---

## 1. The corrected design

### 1.1 Two phases, split at the collate boundary
- **Phase A — per-sample, SPARSE, in CPU workers (unchanged flow):**
  `get_data` → `ApplyToStream` (sparse, index-preserving ops only on sensor) →
  `Collect` → `collate_fn`. Workers stay numpy/CPU; **no CUDA in workers** (fork-unsafe).
- **Phase B — post-collate, device-agnostic torch, in the main process:** a pimm-data
  **reference runner** `apply_batch_transforms(batch, stages, *, device, base_seed, epoch, rank)`
  = `move_to_device` (idempotent) → `Densify` → `AddIntrinsicNoise` (coherent) → `Digitize`.
  Born-on-GPU: only the SPARSE batch (~17 MB/event) crosses PCIe; the `(B,W_p,T)` dense grids
  (~186 MB/event) are created on-device.

### 1.2 One torch implementation + numpy oracle
The dense stages are **one device-agnostic torch path** (device follows `input.device`; runs on
CPU tensors or CUDA tensors with the same code). The numpy `noise.py`/`detector_transforms.py`
chain stays **live** (CPU per-sample path) and is the **bit-exact parity oracle** the torch port
is tested against. No third implementation is introduced.

### 1.3 Multi-stream: a NAMED SUB-DICT, not a string prefix
The only multi-stream problem is a flat-key collision (one `coord`/`offset`/`feat`). Resolve it by
collecting a secondary stream into a **named sub-dict** (`collate_fn` already recurses into
Mappings and cumsums any `*offset*` inside them — verified `collate.py:34-48`):

```python
transform=[
  dict(type='Collect', stream='edep', keys=('coord','segment'),       # PRIMARY → bare keys (back-compat)
       feat_keys=('coord','energy')),
  dict(type='Collect', stream='sensor', into='sensor',                # SECONDARY → batch['sensor']={...}
       keys=('wire','time','value','plane_id')),
]
# batch = {coord, feat, offset, segment,                  # model + avg_pts read these (bare)
#          sensor: {wire, time, value, plane_id, offset}} # one handle for Densify
```
`into=None` (default) = bare keys, **byte-identical** to today. Primary stream stays bare so the
`Point`/model contract and `avg_pts` (`train.py:236-237`) are untouched. (Flat string-prefix is the
alternative; the sub-dict is preferred — structured, one handle, mirrors the dataset's own nesting.)

**Plumbing the scatter needs:** Densify requires **flat integer** `wire`/`time`/`value` — the merged
`coord` is float32 `(M,2)` and `raw` is a nested `{plane:{...}}` collate can't batch. So the
sensor `Collect` surfaces `wire`/`time`/`value` as flat int/float tensors (concatenated, parallel to
`plane_id`), from the reader's raw integer arrays — never derived from the float `coord`.

### 1.4 Geometry & pedestal: registry, validated against the file (NOT carried through collate)
`(W_p, T)`, `pedestal`, (future) `wire_lengths` are **fixed per plane**. Carrying them per-sample
through collate is collate-hostile (a `shape` dict gets stacked B× or, if `_`-prefixed, dropped —
`collate.py:46`). Instead: the dense stages read geometry from a **canonical plane registry**
(sourced from `DetectorConfig`), **validated at load against the file's `n_wires`/`pedestal`/
`num_time_steps` attrs** (mismatch raises). Data is the source of truth, the registry is the lookup,
drift is caught — without fighting collate.

**Incoherent noise is IN scope** (owner decision — reverses the earlier scope-out). The dense
training data is the **noise-free** doraemon sensor, so both incoherent (per-wire) and coherent
(per-group) are synthesized fresh per epoch. This requires **per-wire `wire_lengths_m`**, which is
part of the geometry registry: per plane either an explicit `(n_wires,)` array or a `(lo, hi)` pair
expanded `linspace(lo, hi, n_wires)` (the convention the doraemon study used), in METERS — sourced
from `DetectorConfig`/registry, validated against the file where available. Keep a
`coherent`/`incoherent` config toggle so a dataset that already carries baked incoherent noise can
run coherent-only (avoids double-count). Note: incoherent is the **FFT-heavy, ~9×-memory-transient,
bandwidth-bound** component — so the memory/B-cap hardening (§2 Phase 4) is load-bearing here, not
optional.

### 1.5 Stable `plane_id` — ADDITIVE (sparse path untouched)
Densify must key per-plane grids by a **stable global** id (positional `plane_id` shifts when a plane
is empty / under `volume=` filtering — silent geometry corruption). Make it **additive**: the
dense-sensor `Collect(into='sensor')` emits a **canonical** `plane_id` (from `canonical_plane_id(label)`)
in `batch['sensor']` *only*; the existing positional `plane_id` semantics on other streams are
unchanged, and `test_jaxtpc_semantics.py:207` (`test_plane_id_is_dense_index`) stays green. No sparse
model Collects sensor `plane_id` today, so this is invisible to the sparse path.

### 1.6 Scatter semantics: deterministic `index_add_`, reference updated
Use **accumulate (`index_add_`)** as the canonical densify semantics on **both** the torch path and
the numpy oracle, and update the reference test to match. Rationale: production data is duplicate-free
(verified), but the CI fixture has duplicate `(wire,time)` cells, so a uniqueness **assert would break
CI**; `index_add_` is deterministic only with atomics care on CUDA — so still pin determinism
(`torch.use_deterministic_algorithms` or sort-then-segment) and document it. (Last-wins is the
alternative if a downstream requires it; then dedup deterministically — not via assert.)

### 1.7 Seeding: content-addressed, owned by pimm-data
Per-event seed = `blake2b(event_name) ⊕ base_seed ⊕ epoch ⊕ rank` — **content-addressed**, so the same
event gets the same noise regardless of batch position/worker/resume/world-size; `epoch` = fresh-per-
epoch, `rank` = per-replica decorrelation. **Coherent draws on a numpy `Generator`** (bit-exact to
JAXTPC; ~6 groups/plane, cheap on host; transfer only the per-group waveforms then broadcast on device).
pimm-data owns the **seed-derivation function** (testable); the trainer supplies `base_seed/epoch/rank`.

### 1.8 Correctness fixes (carried into the port)
`torch.std(unbiased=False)` for the per-wire RMS renorm (numpy uses ddof=0); densify reads **immutable**
raw int coords (+ integer-dtype assert); **no double-`Digitize`** (guard with a marker — non-idempotent
when gain≠1; stored sensor is already digitized); force DC real, Nyquist real only if `num_time` even
(production 4321 is odd → DC-only).

---

## 2. Phase sequencing

| Phase | Deliverable | Gating | Ships alone? |
|---|---|---|---|
| **0. Reconcile** | The §0 owner confirmations; re-version v3→v4; promote Track B in ROADMAP/DESIGN. | — | doc only |
| **1. Correctness + plumbing (CPU, no trainer)** | `index_add_` semantics + unbiased std + immutable-coord + no-double-digitize fixes; canonical `plane_id` (additive); geometry registry (incl. **`wire_lengths`**) + file-validation; sensor `Collect(into=, flat wire/time/value)`; `plane_id`→`index_valid_keys`. | Phase 0 (consumer + oracle disposition) | **yes** — standalone data-layer hardening, CPU-testable |
| **2. Torch stages + runner + parity (pimm-data)** | `dense_ops.py` (device-agnostic `densify` → **per-plane dict** `{pid:(B,W_p,T)}` / `add_intrinsic_noise` **coherent + incoherent** / `digitize`); `batch_transforms.py` (`move_to_device`, `apply_batch_transforms`, `build_batch_transforms`, `on_after_batch_transfer` adapter); parity matrix (§5). | Phase 1 | **yes** — pimm-data only; CUDA `importorskip`, CPU-device always-run |
| **3. Trainer integration (pimm)** | Replace the `train.py:225-228` `.cuda()` loop with `apply_batch_transforms(...)`; build stages in `before_train`; `gpu_transforms` config (policy only); train-only noise; DDP seed inputs. | Phase 2 | no — only phase touching pimm |
| **4. Perf hardening** | B-axis chunking for the **9× irfft memory transient** (load-bearing now that incoherent is in Phase 2); DDP-per-GPU; offline-bake fallback; cache+GPU synergy. | Phase 3 | each knob additive |

---

## 3. Ownership

| Piece | Owner |
|---|---|
| `dense_ops` torch stages; reference runner; numpy oracle; parity test; geometry registry+validation; seed-derivation fn | **pimm-data** (runner tested end-to-end on CPU device; never imports `torch.cuda`) |
| Trainer driver (where stages run, `.cuda()` placement, DDP, hooks); `base_seed/epoch/rank` values; config stage lists | **pimm** |
| Noise physics SoT (`tools/noise.py`, `coherent_noise.py`, `noise_spectrum.npz`) | **JAXTPC** (read-only; pimm-data hosts the parity-tested port) |

---

## 4. Everything affected

**pimm-data** — `noise.py` (commit as oracle; unbiased-std/DC-real fixes), `dense_ops.py` (new),
`batch_transforms.py` (new: runner+adapter), `detector_transforms.py` (correctness fixes; numpy path
kept as CPU/oracle), `jaxtpc.py` (`canonical_plane_id`; surface flat `wire/time/value`; registry hook),
`transform.py` (`Collect(into=)`; `plane_id`→`index_valid_keys`), `collate.py` (named-sub-dict already
works; offset2batch helper), `readers/jaxtpc_sensor.py` (confirm `num_time_steps`/`n_wires`/`pedestal`
surfacing + "noise OFF" assert), `testing.py` (fixture geometry attrs), `tests/test_noise.py` +
`tests/test_batch_transforms.py` (new), `docs/` (this v4 + ROADMAP/DESIGN promotion), `pyproject.toml`
(torch floor).
**pimm** — `engines/train.py` (run_step swap + `before_train` stage build), eval/test `.cuda()` sites
(route through the runner with empty/noise-free stages), config (`gpu_transforms` list).
**JAXTPC** — none (read-only SoT).

---

## 5. Verification

- **Parity matrix:** densify torch↔numpy **bit-exact** (CPU + CUDA `importorskip`, after fixing
  semantics to `index_add_` on both); **coherent** torch-host vs numpy vs JAXTPC **bit-exact** (same
  numpy Generator); **incoherent** (when enabled) **statistical** (per-wire RMS≈√(x²+(y+zL)²), ensemble
  PSD≈spectrum, tolerance tight enough to catch the unbiased-std bug); **digitize** bit-exact + no-double
  guard; DC/Nyquist odd-T trap.
- **Plumbing invariants:** geometry registry == file attrs (raises on mismatch); raw coords immutable
  through densify; canonical `plane_id` stable under empty-plane/`volume=`; seed reproducible across
  workers/restart; **no `torch.cuda` import in pimm-data** (grep gate); `offset2batch` handles empties.
- **Perf gate:** scatter ≈1.9 ms, irfft ≈18 ms/event (band); B-chunking keeps peak under the 9×
  transient; sparse ~17 MB crosses PCIe vs dense ~186 MB (born-on-GPU assert); B-invariance.
- **Regression:** full sparse suite green (Track B opt-in); `import pimm_data` torch-device/JAXTPC-free;
  DDP 2-rank smoke (rank-distinct seeds, no cross-rank correlation); **empty `gpu_transforms` ⇒
  byte-equivalent to the current `.cuda()` loop**.

---

## 6. Risk register

| # | Risk | Mitigation |
|---|---|---|
| R1 | cuFFT ≠ numpy irfft (not bit-exact); Philox≠PCG64 | incoherent parity **statistical**; coherent stays numpy-host bit-exact; documented |
| R2 | GPU-augmentation contention tax (un-overlappable; RAW dep) | DDP-per-GPU (not a dedicated transform GPU — re-incurs dense PCIe); optional CUDA-stream; offline-bake fallback |
| R3 | 9× irfft memory transient → OOM | B-axis chunking + per-plane sequential; measure peak before enabling |
| R4 | `plane_id` semantics | **additive** canonical id in the dense sub-dict only → sparse path + `test_plane_id_is_dense_index` untouched |
| R5 | Doc governance / undocumented prototype | Phase 0 names authority, records prototype→oracle decision in v4 + the decision log |
| R6 | Noise-model drift JAXTPC↔port | named owner re-runs statistical parity on JAXTPC noise changes; long-term shared detector-effects package |
| R7 | Double-digitize / stage order | digitize-once marker; runner enforces Densify→AddNoise→Digitize; immutable-coord assert |

---

## 7. Scope guards (NOT doing)
Not converting `hits`↔`sensor`. Not densifying hits by default. Not re-sparsifying. Not noising anything
but `sensor` (both coherent + incoherent ARE in scope; `wire_lengths` surfaced via the registry). Not a
CUDA device-loader in pimm-data (device follows the batch). Not touching the sparse path (default,
opt-in gated). Not changing positional `plane_id` for non-sensor streams.
