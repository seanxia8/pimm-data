# Handoff: shard- and event-level dataset filtering (JAXTPC / LUCiD)

Status: design + audit complete, **no code written for this feature**. One
related fix already landed independently (gap-tolerant indexing, see §3).
This document states verified facts, the open bug it exposed, a sequenced
TODO with reasons, and the design decisions still owed by the user.

Verified against the tree at commit `33d9c48` (branch
`jaxtpc-loader-codec-opt`) on 2026-05-28.

## 1. The goal

A user wants to choose which shard files and which events are included when
constructing a dataset, uniformly across any modality combination
(`step` only, `step+sensor`, `hits+labl`, all four, …). Motivating dataset:

```
/sdf/data/neutrino/omara/doraemon/
  {step,sensor,hits}/run_00266285{46,48,50,51}/sim_<mod>_NNNN.h5
  logs/{run_files_0_300.log, run_files_300_600.log, overflow_events_shard000.csv}
```

- Production still running; uneven shard counts per run (100 / 37 / 100 / 27).
- `labl/` modality **not yet produced**.
- 4 separate `run_*` directories — today `split=` selects exactly one.
- One overflow event logged (run 50, shard 54, event 131).

## 2. Facts — current filtering surface (what exists today)

All confirmed by reading source, not inferred.

| Knob | Granularity | Where | Notes |
|---|---|---|---|
| `split` | directory | reader `_find_files` | selects ONE subdir under each modality dir |
| `dataset_name` | filename prefix | reader `_find_files` | e.g. `sim` → `sim_step_*.h5`; one string, all modalities |
| `volume` | per-event channel | dataset + step reader | loads one TPC volume |
| `min_deposits` | per-event | **step reader only** (`jaxtpc_step.py:84-100`) | masks step index; see §4 bug |
| `min_segments` | per-event | **lucid step reader only** (`lucid_step.py:85-94`) | same shape as min_deposits |
| `max_len` | global cap | dataset `get_data_list` (`jaxtpc.py:226-231`) | applied AFTER indexing; correct |

**There is no shard allowlist/denylist, no `file_list`, no `event_filter`,
no `runs=`, no `manifest=`.** Grep for any of those names returns nothing
(`src/`, `tests/`). The four readers each `glob.glob` their own modality dir
and take everything that matches.

## 3. Facts — what the gap-tolerant commit already changed (`0757ee0`)

This landed independently and **partially overlaps** the proposed work, so
it changes the starting point:

- **All four JAXTPC readers** now build their index from the `event_*` groups
  **actually present** in each shard (`sorted(int(k.rsplit('_',1)[1]) …)`),
  not `arange(n_events)`. (`jaxtpc_{step,sensor,hits,labl}.py` `_build_index`.)
  → A production-skipped event (a gap, `n_events` unchanged) no longer
  `KeyError`s at read time. **This resolves the "missing-event crash" item
  from the prior analysis** — but per-reader, not jointly (see §4).
- blosc/zstd/lz4 read support via `import hdf5plugin` in `__init__.py`
  (`hdf5plugin` now a declared dependency). Production default codec is
  blosc-zstd.
- Perf: in-place decode, preallocated `_merge_plane_dotted` (no
  stack+concatenate), single-volume step fast path.
- Pixel decode fix: reads `charges_i16` (/32767) to match the writer.
- Tests (`33d9c48`, `tests/test_jaxtpc_robustness.py`):
  `test_reader_reads_blosc_compressed`, and
  `test_reader_tolerates_missing_event[step,hits]` — the latter deletes
  `event_001` and asserts a **single reader** indexes `[0,2]` and reads both.
  It does **not** test cross-modality alignment when a gap is in some
  modalities but not others.

## 4. Facts — the open bug (cross-modality event desync)

The dataset passes the **same global `idx` to every reader** in `get_data`
(`jaxtpc.py:233-269`) and sets `_n_events = min(len(r) for r in
active_readers)` (`jaxtpc.py:180`). This is only correct if every reader's
index is the **same contiguous list of physics events**. Two ways it breaks:

1. **`min_deposits > 0` desync (live since before the gap commit).**
   The step reader masks its index to a non-contiguous subset
   (`jaxtpc_step.py:84-100`); sensor/hits/labl index all present events.
   `get_data(k)` then reads step event `valid[k]` but sensor/hits/labl event
   `present[k]` — different physics events. `bridges`, `deposit_to_track`,
   `group_to_track` joins become meaningless. **No test covers this** —
   every multi-modality test passes `min_deposits=0`
   (`test_jaxtpc.py:57`, `test_jaxtpc_semantics.py:13`,
   `test_jaxtpc_transforms.py:22`).

2. **Gap-induced desync (newly possible after `0757ee0`).**
   Now that each reader indexes its own present keys, a gap in some-but-not-
   all modalities misaligns them. e.g. step `[0..199]`, hits `[0..130,132..199]`
   → `_n_events=199`; `get_data(131)` → step `event_131`, hits `event_132`.
   Silent. The per-reader gap fix solved the crash but moved the hazard.

3. **`volume=N` + `min_deposits` is volume-blind.**
   `jaxtpc_step.py:91-97` sums `n_actual` across **all** volumes regardless
   of `self.volume`. A user filtering volume 0 may keep an event whose
   deposits all live in volume 1, then `read_event` returns empty for that
   volume.

4. **`min_deposits>0` without `step` in modalities is silently dropped.**
   The step reader is only built when `'step' in modalities`
   (`jaxtpc.py:132`). `modalities=('hits','labl'), min_deposits=N` → filter
   has no effect, no warning. Same for LUCiD `min_segments`.

**doraemon status (verified):** shard 54 of run 50 has `event_131` **present
in all three modalities**, `n_events=200`, **zero gaps anywhere in the
dataset**. So the overflow was logged but the event was still written, and
the gap-induced desync (#2) is currently **latent, not triggered**. The
`min_deposits>0` desync (#1) triggers the moment anyone uses it with >1
modality.

## 5. Done vs. TODO

### Done (no further work)
- [x] Per-reader gap-tolerant indexing — no KeyError on skipped events (`0757ee0`).
- [x] blosc/zstd read support (`0757ee0`).
- [x] Single-reader gap robustness test (`33d9c48`).
- [x] Existing coarse filters: `split`, `dataset_name`, `volume`, `max_len`.

### TODO — Phase A: correctness foundation (do first, standalone PR)
Reason: this is a **bug fix**, independently valuable, and the joint index it
builds is the prerequisite for the Phase B feature. Shipping B on per-reader
indices would bake in §4 #1 and #2.

- [ ] **A1.** Module-level `@lru_cache` `_read_shard_meta(path) ->
  (n_events, n_volumes, present_event_keys, readout_type)`. ~20 LOC, no API
  change. Reason: 4 readers each open every shard at `__init__` (~800 files
  at doraemon scale, 10–30 ms cold each on SDF); memoizing collapses ~3×
  redundant opens. Highest ROI / lowest risk change.
- [ ] **A2.** Joint event index at the **dataset** level: intersect present
  `event_*` keys across loaded modalities, then apply event filters; hand the
  resulting per-shard index arrays to every reader via new reader kwargs
  `(file_list, events_per_file)`. Reason: makes one index the source of truth
  → fixes §4 #1 and #2 by construction. Readers fall back to their own
  `_find_files`/`_build_index` when the kwargs are absent (backward compat).
- [ ] **A3.** Make `min_deposits` volume-aware when `volume=N` set; raise if
  `min_deposits>0` (or `min_segments>0`) is passed without the source
  modality loaded. Reason: §4 #3, #4.
- [ ] **A4.** Length-mismatch handling: warn with concrete per-modality counts
  instead of the silent `min(...)`; add `strict_lengths=True` to hard-error.
  Reason: a 1-shard mismatch silently drops data today.
- [ ] **A5.** Regression test: `modalities=('step','sensor','hits','labl')`
  with `min_deposits>0`, assert the SAME physics event (by an identifying
  attr, e.g. `event_id`) is returned across every modality for every idx.
  Plus a gap-in-one-modality variant for §4 #2. **Both fail on `master`/HEAD
  today.** Reason: lock the fix; this class of bug is currently uncaught.

### TODO — Phase B: the feature (shard/event selection)
Reason: this is the user's actual ask; built on A's joint index.

- [ ] **B1.** `include_shards` / `exclude_shards` / `shard_filter` at the
  dataset level. Unit = shard **tag** (`'0054'`), the only identifier
  invariant across modalities (filename is per-modality, path leaks layout).
  Scope per-run (a tag is a logical shard present in each run). Satisfies the
  doraemon shard-selection request.
- [ ] **B2.** `runs=` first-class (`['run_…', …]` or `'*'`), replacing the
  ConcatDataset workaround for the 4-run layout; extend `get_data_name` to
  carry run identity so sample names survive filter changes. *(Gated on the
  multi-run decision, §6.)*
- [ ] **B3.** `event_filter`: registered-class form (`dict(type='MinDeposits',
  n=10)`) for config round-trip via `build_dataset`, plus a raw-callable
  escape hatch for in-Python use. Signature `(shard_tag, event_idx,
  attrs_dict) -> bool`; called O(N) at `__init__`, never in `__getitem__`.
  Built-ins: `MinDeposits`, `MaxDeposits`, `MinHits`. Reason: callables don't
  survive the dict-config round trip (`_registry.build_from_cfg` does no
  coercion) — the registered form is the shareable one.
- [ ] **B4.** `manifest=` (YAML/JSON, `mode: include|exclude`, list of
  `{run, shard, event_idx}`) + `ds.resolved_manifest()` snapshot. Reason: the
  reproducibility / collaborator-handoff contract; also the right home for
  the overflow CSV (converted once via CLI, not auto-parsed).

### TODO — Phase C: tooling (separate, deferrable)
- [ ] `pimm-data audit <root>` — per-modality shard counts per run, cross-
  modality gaps, overflow summary, draft manifest.
- [ ] `pimm-data manifest from-overflow <csv>` — convert producer CSV → manifest.
- [ ] Shared `pimm_manifest.py` schema vendored in both pimm-data and the producer.
Reason: keep curation/audit out of the runtime import path; own package extra.

## 6. Open decisions owed by the user (asked, dismissed 2026-05-28)

These gate scope; nothing past Phase A should start until answered.

1. **How far now?** Phase A only / A + B1 (recommended first cut, covers
   doraemon) / A + full B / everything incl. CLI.
2. **Multi-run:** one dataset spanning all 4 runs (`runs=`, adds B2) vs.
   ConcatDataset is fine (defer B2) vs. one run at a time.
3. **Backward compat:** is a downstream training repo importing
   `JAXTPCDataset`/`LUCiDDataset`? Determines deprecation-shim vs.
   change-in-place for `min_deposits` semantics.

Independent of the above, recommendation stands: **Phase A is a bug fix and
should land first as its own PR**, regardless of how much of B/C is chosen.

## 7. Design rationale captured (from 4 expert-lens reviews)

- **API:** shard **tag** is the cross-modality-invariant unit; `split=` is the
  wrong abstraction for 4 runs; raw callables break `build_dataset` config
  round-trip and should be the escape hatch, not the primary surface.
- **Correctness:** verdict matrix says step-bearing combos are BROKEN-TODAY
  under `min_deposits>0`; sensor/hits-only combos have no step facts to filter
  on (NEEDS-NEW-API if wanted); `('labl',)` and `('sensor','labl')` stay
  FORBID. Cross-modality event predicates deferred (need richer protocol).
- **Performance:** `_read_shard_meta` memoization is the quick win (~3× init);
  joint index avoids redundant per-reader walks; persist-to-disk index
  deferred until needed; index built in main proc, workers reopen handles
  (keep the `_initted` lazy-open pattern — do not leave open handles on `self`).
- **Operations:** snapshot at `__init__` (not per-epoch); fail loud on missing
  `labl` (don't silently disable); manifest is the producer/consumer contract;
  do NOT auto-parse `logs/overflow_*.csv` — convert via CLI.

## 8. Key source references

- `src/pimm_data/jaxtpc.py:132-180` — reader construction, `_n_events = min(...)`
  (desync hides here); `:226-231` `max_len`; `:233-269` `get_data` (same idx to
  all readers); `:492-499` `get_data_name`; `:220-224` `_modality_root`
  (extension point for `runs=`).
- `src/pimm_data/readers/jaxtpc_step.py:84-100` — only step masks its index
  (§4 #1); `:91-97` volume-blind `n_actual` sum (§4 #3).
- `src/pimm_data/readers/jaxtpc_{sensor,hits,labl}.py` `_build_index` —
  present-key indexing from `0757ee0` (§3, §4 #2).
- `src/pimm_data/lucid.py:121-125` — `min_segments` ignored without step
  (§4 #4); `src/pimm_data/readers/lucid_step.py:85-94` — LUCiD filter.
- `src/pimm_data/_registry.py` `build_from_cfg` — no callable coercion
  (why event_filter needs a registered-class form).
- `tests/test_jaxtpc_robustness.py` — current gap/blosc tests (single-reader).
- `/sdf/data/neutrino/omara/doraemon/logs/overflow_events_shard000.csv` —
  the one overflow row; columns `timestamp,source_path,src_file_idx,
  event_idx,event_id,n_deposits,error_type,error_message`.
