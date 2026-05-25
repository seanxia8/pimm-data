#!/usr/bin/env python3
"""Why doesn't N workers give N× speedup?

If the bottleneck were "per-reader I/O latency, fully parallelizable",
we'd expect linear scaling up to filesystem saturation. We measure
~9× at 20 workers instead of 20×.

This script bypasses PyTorch's DataLoader to isolate the cause. It runs
the same raw h5py work as the pipeline using:

  * forked processes via multiprocessing.Pool  — IPC-free reads
  * native threads via ThreadPoolExecutor       — Python-GIL bound but
                                                 I/O-wait releases the GIL
  * direct os.read of a large flat dataset      — pure FS bandwidth
                                                 (no per-event API overhead)

If raw-process scaling is linear and DataLoader's isn't → DataLoader IPC.
If raw-process scaling plateaus early                  → FS saturation.
If threads scale well but processes don't              → IPC dominates.
"""
import argparse
import multiprocessing as mp
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import _profile_common as pc

# The per-event read worker is shared (and picklable for mp.Pool) in
# _profile_common.per_event_edep_read; tasks carry the dataset name as the
# first tuple element so the worker reads the right schema.
_per_event_h5py_read = pc.per_event_edep_read


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=pc.DATASETS, default='lucid',
                   help='Which dataset edep shards to read (default: lucid)')
    p.add_argument('--data-root', default=None,
                   help='Dataset root (default: per-dataset standard root)')
    p.add_argument('--split', default=None,
                   help='Split subdir (default: "" for lucid, doraemon run '
                        'for jaxtpc)')
    p.add_argument('--n-events-per-worker', type=int, default=200)
    p.add_argument('--worker-counts', nargs='+', type=int,
                   default=[1, 2, 4, 8, 16, 24, 32])
    args = p.parse_args()

    if args.data_root is None:
        args.data_root = pc.default_root(args.dataset)
    if args.split is None:
        args.split = pc.default_split(args.dataset)

    edep_files = pc.modality_files(args.dataset, args.data_root, 'edep',
                                   split=args.split)
    assert len(edep_files) > 0, (
        f'no edep shards under {args.data_root} (split={args.split!r})')
    print(f'dataset            = {args.dataset}')
    print(f'data_root          = {args.data_root}')
    print(f'split              = {args.split!r}')
    print(f'edep shards        = {len(edep_files)}')
    print(f'events per worker  = {args.n_events_per_worker}')
    print(f'worker counts      = {args.worker_counts}')
    print()

    # -- 0) Filesystem cache warmup pass on the first few shards ----------
    print('--- FS warmup (read first 5 shards once, sequential) ---')
    for f in edep_files[:5]:
        _per_event_h5py_read((args.dataset, f, 0, 50))
    print('  done\n')

    # -- 1) Single-process baseline ---------------------------------------
    print('--- baseline: single process, raw h5py ---')
    t0 = time.perf_counter()
    elapsed, n_evt, n_bytes = _per_event_h5py_read(
        (args.dataset, edep_files[0], 0, args.n_events_per_worker))
    base_evt_per_s = n_evt / elapsed
    base_mb_per_s  = (n_bytes / 1e6) / elapsed
    print(f'  {n_evt} events in {elapsed:.2f}s  '
          f'= {base_evt_per_s:6.1f} events/s  '
          f'= {base_mb_per_s:6.1f} MB/s  '
          f'({n_bytes / n_evt / 1024:.1f} KB/event)')
    print()

    # -- 2) Process scaling (multiprocessing.Pool, separate file per worker) ---
    print('--- multiprocessing.Pool: N workers, each on a distinct shard ---')
    print(f'  {"workers":>7}  {"wall s":>7}  {"agg evt/s":>10}  '
          f'{"agg MB/s":>9}  {"speedup":>8}  {"efficiency":>10}')
    proc_results = []
    for w in args.worker_counts:
        if w > len(edep_files):
            print(f'  {w:>7}  (skipped: only {len(edep_files)} shards)')
            continue
        # Each worker gets a different shard so we exercise read-parallelism
        # without forcing same-file contention.
        tasks = [(args.dataset, edep_files[i], 0, args.n_events_per_worker)
                 for i in range(w)]
        t0 = time.perf_counter()
        with mp.Pool(w) as pool:
            results = pool.map(_per_event_h5py_read, tasks)
        wall = time.perf_counter() - t0
        total_evt = sum(r[1] for r in results)
        total_bytes = sum(r[2] for r in results)
        evt_s = total_evt / wall
        mb_s = (total_bytes / 1e6) / wall
        speedup = evt_s / base_evt_per_s
        eff = speedup / w * 100
        proc_results.append((w, evt_s, mb_s, speedup, eff))
        print(f'  {w:>7}  {wall:7.2f}  {evt_s:>10.1f}  {mb_s:>9.1f}  '
              f'{speedup:>7.2f}×  {eff:>9.1f}%')
    print()

    # -- 3) Thread scaling (ThreadPoolExecutor, same shard each thread) ---
    # GIL releases on h5py I/O, so threads can overlap.
    print('--- ThreadPoolExecutor: N threads, each on a distinct shard ---')
    print(f'  {"threads":>7}  {"wall s":>7}  {"agg evt/s":>10}  '
          f'{"speedup":>8}  {"efficiency":>10}')
    for w in args.worker_counts:
        if w > len(edep_files):
            continue
        tasks = [(args.dataset, edep_files[i], 0, args.n_events_per_worker)
                 for i in range(w)]
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=w) as ex:
            results = list(ex.map(_per_event_h5py_read, tasks))
        wall = time.perf_counter() - t0
        total_evt = sum(r[1] for r in results)
        evt_s = total_evt / wall
        speedup = evt_s / base_evt_per_s
        eff = speedup / w * 100
        print(f'  {w:>7}  {wall:7.2f}  {evt_s:>10.1f}  '
              f'{speedup:>7.2f}×  {eff:>9.1f}%')
    print()

    # -- 4) Same-shard contention: N processes all hitting one file ------
    print('--- multiprocessing.Pool: N workers, ALL on the same shard ---')
    print('  (tests whether per-file contention is the cap)')
    print(f'  {"workers":>7}  {"wall s":>7}  {"agg evt/s":>10}  '
          f'{"speedup":>8}')
    for w in args.worker_counts:
        tasks = [(args.dataset, edep_files[0], i * 100, args.n_events_per_worker)
                 for i in range(w)]
        t0 = time.perf_counter()
        with mp.Pool(w) as pool:
            results = pool.map(_per_event_h5py_read, tasks)
        wall = time.perf_counter() - t0
        total_evt = sum(r[1] for r in results)
        evt_s = total_evt / wall
        speedup = evt_s / base_evt_per_s
        print(f'  {w:>7}  {wall:7.2f}  {evt_s:>10.1f}  {speedup:>7.2f}×')
    print()

    # -- 5) Summary -------------------------------------------------------
    if proc_results:
        peak = max(proc_results, key=lambda r: r[1])
        print(f'PEAK process throughput: {peak[1]:.0f} evt/s  '
              f'({peak[2]:.0f} MB/s) at {peak[0]} workers '
              f'= {peak[3]:.1f}× single-proc, {peak[4]:.1f}% efficiency')


if __name__ == '__main__':
    mp.set_start_method('fork', force=True)
    main()
