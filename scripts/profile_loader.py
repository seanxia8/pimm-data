#!/usr/bin/env python3
"""Bottleneck analysis for the LUCiD data-loading pipeline.

Times each layer of get_data() in isolation and compares against a
raw-h5py baseline that does the same I/O without pimm-data overhead.
This lets us decide whether the bottleneck is:

  * the network HDF5 I/O itself           (raw baseline is slow)
  * pimm-data reader-class overhead       (readers slower than raw)
  * dataset assembly + FK joins           (get_data slower than sum of readers)
  * the transform pipeline                (__getitem__ slower than get_data)

Also runs cProfile and shows the top functions by cumulative time.

Usage:
    scripts/profile_loader.py
    scripts/profile_loader.py --data-root /path/to/config_NNNNNN
    scripts/profile_loader.py --n-events 500 --warmup 50
"""
import argparse
import cProfile
import glob
import io
import os
import pstats
import sys
import time

import numpy as np

import _profile_common as pc


def _stat(name, times_ms, n_events, baseline_ms=None):
    """Format one row of the summary table."""
    arr = np.asarray(times_ms)
    p50, p25, p75, p95 = np.percentile(arr, [50, 25, 75, 95])
    mean = arr.mean()
    rate = 1000.0 / mean if mean > 0 else float('nan')
    rel = ''
    if baseline_ms is not None and baseline_ms > 0:
        rel = f'  ×{mean / baseline_ms:.2f}'
    print(f'  {name:38s}  '
          f'mean={mean:7.2f} ms  '
          f'p50={p50:7.2f}  p25–p75={p25:6.2f}–{p75:6.2f}  '
          f'p95={p95:7.2f}  '
          f'({rate:6.1f} events/s){rel}')


def _time_callable(fn, n, warmup):
    """Call fn(i) for i in 0..warmup+n; return times for the last n."""
    for i in range(warmup):
        fn(i)
    times = []
    for i in range(n):
        t0 = time.perf_counter()
        fn(warmup + i)
        times.append((time.perf_counter() - t0) * 1000)
    return times


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__.split('\n\n')[0])
    p.add_argument('--dataset', choices=pc.DATASETS, default='lucid',
                   help='Which dataset/loader to profile (default: lucid)')
    p.add_argument('--data-root', default=None,
                   help='Dataset root (default: per-dataset standard root)')
    p.add_argument('--split', default=None,
                   help='Split subdir under each modality dir '
                        '(default: "" for lucid, the doraemon run for jaxtpc)')
    p.add_argument('--n-events', type=int, default=500,
                   help='number of timed events per stage')
    p.add_argument('--warmup', type=int, default=50,
                   help='events discarded to even out FS cache')
    p.add_argument('--cprofile', action='store_true',
                   help='also dump cProfile top-20 by cumulative time '
                        '(runs over an extra n_events events)')
    args = p.parse_args()

    if args.data_root is None:
        args.data_root = pc.default_root(args.dataset)
    if args.split is None:
        args.split = pc.default_split(args.dataset)

    print(f'dataset   = {args.dataset}')
    print(f'data_root = {args.data_root}')
    print(f'split     = {args.split!r}')
    print(f'n_events  = {args.n_events}   warmup = {args.warmup}')
    print()

    # ---------------------------------------------------------------
    # 1) Build everything once
    # ---------------------------------------------------------------
    print('--- building objects ---')
    readers = []
    for name, reader in pc.build_readers(args.dataset, args.data_root, args.split):
        t0 = time.perf_counter()
        # build_readers already opened handles; time a no-op touch instead.
        _ = len(reader)
        readers.append((name, reader))
        print(f'  {name:6s} reader     opened, {len(reader):,} events  '
              f'({1000*(time.perf_counter()-t0):.1f} ms)')

    cls = 'JAXTPCDataset' if args.dataset == 'jaxtpc' else 'LUCiDDataset'

    t0 = time.perf_counter()
    ds_raw = pc.build_dataset(args.dataset, args.data_root, split=args.split,
                              transform=None)
    print(f'  {cls} (no transform) built in {1000*(time.perf_counter()-t0):.1f} ms')

    t0 = time.perf_counter()
    ds = pc.build_dataset(args.dataset, args.data_root, split=args.split)
    print(f'  {cls} (+ Collect) built in {1000*(time.perf_counter()-t0):.1f} ms')

    t0 = time.perf_counter()
    raw_read, raw_close, n_total = pc.raw_reader(
        args.dataset, args.data_root, split=args.split)
    print(f'  raw h5py reader  built in {1000*(time.perf_counter()-t0):.1f} ms  '
          f'(spans {n_total:,} events across all shards)')

    # bound how many events we can use to n_total - warmup
    max_events = min(args.n_events, n_total - args.warmup)
    if max_events < args.n_events:
        print(f'  trimming n_events to {max_events} (dataset has {n_total:,})')

    print()

    # ---------------------------------------------------------------
    # 2) Per-stage timing
    # ---------------------------------------------------------------
    print('--- per-event timing (lower is better) ---')

    raw_times = _time_callable(raw_read,         max_events, args.warmup)
    raw_mean = float(np.mean(raw_times))

    reader_times = {name: _time_callable(r.read_event, max_events, args.warmup)
                    for name, r in readers}
    sum_readers = np.sum([np.array(t) for t in reader_times.values()], axis=0)

    getdata_times    = _time_callable(ds_raw.get_data, max_events, args.warmup)
    getitem_times    = _time_callable(ds.__getitem__,  max_events, args.warmup)

    print()
    _stat('raw h5py (I/O ceiling)',          raw_times,     max_events)
    print()
    for name, t in reader_times.items():
        _stat(f'{cls[:-7]}{name.capitalize()}Reader.read_event',
              t, max_events, raw_mean)
    _stat(f'  Σ {len(readers)} readers (independent)',
          sum_readers, max_events, raw_mean)
    print()
    _stat(f'{cls}.get_data (assemble)', getdata_times, max_events, raw_mean)
    _stat(f'{cls}.__getitem__ (+ Collect)',
                                             getitem_times, max_events, raw_mean)

    # ---------------------------------------------------------------
    # 3) Where does the overhead come from? Breakdown
    # ---------------------------------------------------------------
    print()
    print('--- breakdown (mean ms per event) ---')
    raw_m   = np.mean(raw_times)
    sumr_m  = float(sum_readers.mean())
    gd_m    = np.mean(getdata_times)
    gi_m    = np.mean(getitem_times)

    print(f'  raw I/O                     {raw_m:7.2f} ms')
    print(f'  + reader wrappers           +{sumr_m - raw_m:6.2f} ms  '
          f'(Σreaders - raw)')
    print(f'  + dataset assembly + FKs    +{gd_m - sumr_m:6.2f} ms  '
          f'(get_data - Σreaders)')
    print(f'  + transforms (Collect)      +{gi_m - gd_m:6.2f} ms  '
          f'(__getitem__ - get_data)')
    print(f'  ──────────────────────────────────')
    print(f'  total (__getitem__)         {gi_m:7.2f} ms')
    if gi_m > 0:
        print()
        print(f'  raw I/O is {100*raw_m/gi_m:.1f}% of __getitem__')
        print(f'  → pimm-data wrappers add {100*(gi_m-raw_m)/gi_m:.1f}% on top')

    # ---------------------------------------------------------------
    # 4) cProfile hotspots
    # ---------------------------------------------------------------
    if args.cprofile:
        print()
        print('--- cProfile: top 20 by cumulative time over '
              f'{max_events} __getitem__ calls ---')
        pr = cProfile.Profile()
        pr.enable()
        for i in range(max_events):
            ds[args.warmup + max_events + i % max_events]
        pr.disable()
        s = io.StringIO()
        pstats.Stats(pr, stream=s).strip_dirs().sort_stats('cumulative').print_stats(20)
        # Only print the data lines for compactness.
        for line in s.getvalue().splitlines():
            if line.strip():
                print('  ' + line)

    raw_close()


if __name__ == '__main__':
    sys.exit(main())
