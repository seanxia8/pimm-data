# Real-data validation findings (de-fork Step 1)

The Step 1 work (joint index, transform merge, multimodal base, label-decoration
framework) was built and green against the synthetic fixtures in `testing.py`.
A subsequent audit ran the same code against the **real** detector datasets —
JAXTPC `doraemon` and LUCiD `WAND` (`/sdf/.../neutrino_data/omara/...`) — because
synthetic fixtures encode our *expectations*, not the data's actual quirks. The
audit found real bugs the synthetic suite could not. This doc records them and
their fixes/decisions.

## Method

- **JAXTPC**: `doraemon` (sharded, no `labl`, shard-local `source_event_idx` +
  `global_event_offset`), `doraemon_pixel` (gaps / corrupt shards),
  `sample_production` (has `labl`).
- **LUCiD**: `WAND` SK-like (`config/source_event_idx` vector, `file_index`,
  per-interaction labl tables). Many shards on this filesystem are symlinks to a
  source mount that is intermittently available.
- Each finding was reproduced against real data, fixed, and re-validated against
  real data plus a synthetic regression test shaped like the real failure.

## Critical (fixed, commit e41e9f5)

- **F1 — identity collision.** `MultiModalEventDataset` keyed the holdout on
  `(config_id, source_event_idx)`, reading `source_event_idx` only from a config
  vector. doraemon has no such vector → positional `event_num` fallback →
  shard-local indices → collisions across shards (e.g. 50 collisions / 3 shards).
  Two different physics events hashed to the same split bucket.
  **Fix:** identity is now `(config_id, file_index, source_event_idx)`;
  `source_event_idx` resolves vector → `global_event_offset + event_num` →
  `event_num`. `file_index` is the intrinsic shard id stamped in `config`.
  **Validated:** 600 doraemon events → 600 unique identities (was colliding).

- **F2 — GridSample `sum` overflow.** The sum reducer accumulated in the input
  dtype; real `de` is float16, `delta_times` int8, sensor values uint16 — so a
  busy voxel saturated (374771 → 4096 for float16; int8 wrapped).
  **Fix:** accumulate in int64 / float64, then cast back.
  **Validated:** float16 voxel sum 500000 (not 4096); int8 sum 5000 (not wrapped).

- **F3 — `gather_with_fill` dtype crash.** `np.full(..., fill=-1, dtype=column.dtype)`
  raised `OverflowError` on real uint8 `category` and silently became `True` on
  bool `contained`.
  **Fix:** widen the output dtype to signed/int64 when the column is bool/unsigned.
  **Validated:** real uint8 category gather → `[0, -1]` (no crash); bool → signed −1 sentinel.

## Medium (fixed, this commit)

- **F6 — LUCiD readers not gap-tolerant.** All four LUCiD readers indexed
  `np.arange(n_events)` from the (never-decremented) `config/n_events` attr, so a
  missing `event_NNN` meant (a) opening a non-existent group → crash, and (b)
  off-by-one misalignment of every later event against the other modalities.
  JAXTPC readers already used `present_events`.
  **Fix:** all four LUCiD readers now index
  `read_shard_meta(path)['present_events']` (sorted real event numbers), which
  also wires in the A1 metadata cache. `lucid_edep`'s `min_segments` branch
  iterates `present_events`.
  **Validated on real WAND:** deleting `event_010` from a 744-event shard → reader
  reports 743, skips 10 cleanly, local index 10 remaps to event 11 (monotonic).
  Regression: `test_missing_event_group_gap_tolerant`.

- **F4 — `label_config` contract divergence.** LUCiD routes label decoration
  through the shared `decorate_labels` (full contract: `source='self'`,
  `scope='event'/'event_broadcast'`, `source=(table, col)`, `keyed_by`). JAXTPC
  used a private path (`_track_axes`) that honored only `source=('track', col)`
  point specs and **silently dropped** everything else — the same `label_config`
  produced different streams on the two detectors.
  **Fix:** (a) JAXTPC now honors `source='self'` (emits the bare per-modality
  `instance` under the named key, matching LUCiD where `self == instance`);
  (b) `_validate_label_config()` raises at construction for any spec JAXTPC
  cannot faithfully honor (event scope; non-`track` tables like `particle`/
  `event` that have no JAXTPC analog; `keyed_by` other than `track_ids`). Specs
  are never silently dropped.
  Regressions: `test_jaxtpc_label_config_self_source`,
  `test_jaxtpc_label_config_rejects_unsupported_specs`.

- **F5 — `per_interaction` not surfaced (LUCiD).** The labl reader exposed
  `per_event` / `per_particle` / `per_track` but not the fourth scope,
  `per_interaction` (per-neutrino-vertex). The `per_particle.interaction_idx` FK
  pointed at a table that never reached the output.
  **Fix:** the reader now surfaces `labl_interaction_*` (vertex, neutrino
  kinematics, `source_type`, `contained`, `n_{particles,primaries}`, and the
  ragged CSR primary `pdgs`/`energies`/`track_ids`), cast to the v3 writer dtypes
  (`neutrino_pdg` int16→int32, `source_type` uint8→int32, offsets uint32, bool
  preserved). `lucid.py::_build_labl` rebuilds the nested `interaction` table so a
  `source=('interaction', col)` axis resolves. The synthetic fixture now writes a
  `per_interaction` group.
  **Validated on real WAND:** all 16 interaction keys surface with correct dtypes
  and monotone CSR offsets (single-µ event: primary pdg 13, vertex resolved).
  Regressions: `test_labl_per_interaction_surfaced`,
  `test_label_config_interaction_event_broadcast`.

## Decision (no behavior change)

- **F7 — `RelativeLogNormalize` hard ceiling (D13).** `max_val=4000` clips the
  PMT hit-time long tail; values beyond it saturate to `out_max` and are
  indistinguishable. This is a deliberate model choice (compress the late-time
  reflection/afterpulse tail rather than dilate early-time dynamic range), not a
  bug. **Surfaced** in the class docstring: lossy by design, raise `max_val`
  (and re-tune `scale`) if a task must resolve the tail. Clipping is covered by
  `test_relative_log_normalize_handles_negatives_no_nan`.

## Low (F8–F16)

Fixed (code) — each a clean, low-risk win:

- **F13 — `volume=` filter leaked the other volume's bridges.** With `volume=0`,
  the hits reader still loaded `group_to_track_v1` / `deposit_to_group_v1` /
  `qs_fractions_v1` into `data['bridges']`, though no v1 points are loaded.
  Harmless to correctness but wasted I/O + confusing payload. `_build_bridges`
  now keeps only the selected volume's `*_v{N}` tables.
  Test: `test_volume_filter_prunes_bridges`.

- **F15 — hits `_detect_readout_type` bypassed the A1 cache.** It opened the
  first shard directly to read the `readout_type` attr (the sensor reader uses
  `read_shard_meta`). Now takes the cached `config_attrs` fast path and only
  opens to scan plane datasets when the attr is absent. **Real doraemon:**
  returns `wire` from the cache, no extra open.

- **F16 — `min_deposits` index scan bypassed the A1 cache.** The
  `min_deposits>0` branch opened every shard and looped per-event `n_actual`
  (~75 ms/file, the dominant index-build cost). New cached
  `read_deposit_counts(path)` memoizes the per-event per-volume counts by
  `(path, mtime, size)`, so train/val/test (and tiered) datasets over the same
  shards share one scan. Volume-aware semantics preserved.
  **Real doraemon:** 800 events indexed, cache reused, filtered read works.
  Tests: `test_min_deposits_cached_scan_matches_direct`.

Decision / documentation (no behavior change):

- **F9 — `LogTransform.clip` and real PE.** Real WAND PE reaches ~424; ~0.1 % of
  hits exceed the `max_val=50` default, mapping to ~+1.5 (outside `[-1, 1]`)
  when `clip=False`. Documented in the `LogTransform` docstring: pass
  `clip=True` (saturate at +1) or raise `max_val`; default left off for
  back-compat — choose per task.

- **F11 — schema version mismatch.** Fixtures/docstrings said
  `format_version 3`; real WAND is `format_version 5`. The readers gate on
  *structure*, not the version int (confirmed: F5/F6 validated on real fv5
  files), so the fixtures now stamp 5 and the LUCiD test/docstrings say "v3+
  schema; real WAND is fv5".

- **F8 — v3 vertex / `is_primary` plumbing unvalidated on real data.** No
  PILArNet-v3 / panda data exists in `neutrino_data`, so the
  `_apply_to_v3_vertex` co-transforms + the `index_operator` carry branch are
  green only on synthetic `vertex = coord.copy()`. The panda flip (de-fork
  Step 3) is the first real exercise — call it out there.

- **F10 — named keys are `(N, 1)`, bare keys are `(N,)`.** On real data bare
  `segment`/`instance` are 1-D while `segment_pid`/`instance_*` are 2-D `(N, 1)`;
  values match (`bare segment == segment_pid` confirmed). Left as-is — changing
  a shape is exactly the kind of edit that breaks a downstream consumer
  silently. Documented so consumers `.reshape(-1)` / `[:, None]` deliberately.

- **F12 — run-nested layout + WAND units.** (a) Real doraemon is
  `{modality}/run_NNNNNNNN/sim_*.h5`; pointing `data_root` at the modality dir
  finds nothing. Workaround today: `split='run_NNNNNNNN'` (loads one run). A
  cross-run glob is a deliberate *feature* (merge-all-runs semantics + the F1
  identity interaction — safe on doraemon because `global_event_offset` makes
  `sei` globally unique), not a cleanup; deferred for its own change. (b) WAND
  coords are in **meters** (detector spans ~±10 m), so `GridSample`/voxel sizes
  must be O(0.5) m there, not the LAr-cm scale — a config note for the LUCiD
  pipeline.

## New finding — follow-up (not yet fixed)

- **F17 — eager-open crash on dangling shards.** `_build_index` tolerates an
  unopenable shard (logs a warning, contributes 0 events), but
  `h5py_worker_init` later opens *every* globbed file and raises on the first
  dangling one. On WAND this filesystem, many shards are symlinks to a source
  mount that is intermittently absent, so a single missing shard crashes the
  whole reader at worker init instead of being skipped. Pre-existing (not
  introduced by Step 1); affects the LUCiD readers (and likely JAXTPC). Fix would
  drop unopenable files from `h5_files` in `_build_index` (keeping
  `cumulative_lengths`/`indices` aligned) so worker init only opens survivors.
  Deferred — flagged for the next robustness pass.

## Takeaway

Synthetic fixtures validated our model of the data; real data validated the data.
F1–F3 were silent correctness bugs (wrong splits, saturated sums, wrong gathers)
that passed the synthetic suite because the fixtures shared the code's
assumptions (dense indices, wide dtypes, signed columns). Every fix shipped with
a regression test shaped like the real failure, and the medium findings were each
re-validated against the real datasets, not only the fixtures.
