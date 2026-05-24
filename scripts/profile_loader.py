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
import h5py

from pimm_data import LUCiDDataset
from pimm_data.readers.lucid_edep import LUCiDEdepReader
from pimm_data.readers.lucid_hits import LUCiDHitsReader
from pimm_data.readers.lucid_sensor import LUCiDSensorReader
from pimm_data.readers.lucid_labl import LUCiDLablReader


# Match what benchmark_loader.py uses so timings are comparable.
DEFAULT_TRANSFORM = [
    dict(type='Collect', stream='hits',
         keys=['coord', 'energy', 'time'],
         feat_keys=['energy', 'time']),
]


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


def _raw_h5py_reader(data_root):
    """Closure that reads the same per-event arrays as the four readers
    but without going through pimm-data reader classes. Establishes the
    pure I/O ceiling.
    """
    edep_files   = sorted(glob.glob(f'{data_root}/edep/wc_edep_*.h5'))
    sensor_files = sorted(glob.glob(f'{data_root}/sensor/wc_sensor_*.h5'))
    hits_files   = sorted(glob.glob(f'{data_root}/hits/wc_hits_*.h5'))
    labl_files   = sorted(glob.glob(f'{data_root}/labl/wc_labl_*.h5'))
    # Build the per-file event-count index up front (matches readers).
    counts = []
    for p in edep_files:
        with h5py.File(p, 'r') as f:
            counts.append(int(f['config'].attrs['n_events']))
    cumlens = np.cumsum(counts)
    # Open all files once and keep handles.
    fhs = {
        'edep':   [h5py.File(p, 'r', libver='latest', swmr=True) for p in edep_files],
        'sensor': [h5py.File(p, 'r', libver='latest', swmr=True) for p in sensor_files],
        'hits':   [h5py.File(p, 'r', libver='latest', swmr=True) for p in hits_files],
        'labl':   [h5py.File(p, 'r', libver='latest', swmr=True) for p in labl_files],
    }

    def locate(idx):
        i = int(np.searchsorted(cumlens, idx, side='right'))
        local = idx - (int(cumlens[i - 1]) if i > 0 else 0)
        return i, f'event_{local:03d}'

    def read(idx):
        fi, ek = locate(idx)
        # Mirror the per-modality reads the four readers do.
        e = fhs['edep'][fi][ek]
        _ = e['start_x'][:]; _ = e['start_y'][:]; _ = e['start_z'][:]
        _ = e['end_x'][:];   _ = e['end_y'][:];   _ = e['end_z'][:]
        _ = e['edep'][:];    _ = e['time'][:];    _ = e['track_idx'][:]
        _ = e['dir_x'][:];   _ = e['dir_y'][:];   _ = e['dir_z'][:]
        _ = e['beta_start'][:]; _ = e['n_cherenkov'][:]
        if 'contained' in e:
            _ = e['contained'][:]

        s = fhs['sensor'][fi][ek]
        _ = s['sensor_idx'][:]; _ = s['PE'][:]; _ = s['T'][:]

        h = fhs['hits'][fi][ek]
        _ = h['sensor_idx'][:]; _ = h['particle_idx'][:]
        _ = h['PE'][:]; _ = h['T'][:]

        l = fhs['labl'][fi][ek]
        if 'per_event' in l:
            _ = l['per_event']['t0'][()]; _ = l['per_event']['contained'][()]
        if 'per_particle' in l:
            pp = l['per_particle']
            for k in ('category', 'contained',
                      'genealogy_data', 'genealogy_offsets',
                      'ext_genealogy_data', 'ext_genealogy_offsets'):
                if k in pp:
                    _ = pp[k][:]
        if 'per_track' in l:
            pt = l['per_track']
            for k in ('track_id', 'pdg', 'parent_id', 'particle_idx',
                      'ancestor', 'interaction', 'initial_energy',
                      'n_cherenkov'):
                if k in pt:
                    _ = pt[k][:]

    def close():
        for handles in fhs.values():
            for f in handles:
                f.close()

    return read, close, sum(counts)


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__.split('\n\n')[0])
    p.add_argument('--data-root',
                   default='/sdf/data/neutrino/omara/wand_sk_like/config_000013')
    p.add_argument('--n-events', type=int, default=500,
                   help='number of timed events per stage')
    p.add_argument('--warmup', type=int, default=50,
                   help='events discarded to even out FS cache')
    p.add_argument('--cprofile', action='store_true',
                   help='also dump cProfile top-20 by cumulative time '
                        '(runs over an extra n_events events)')
    args = p.parse_args()

    print(f'data_root = {args.data_root}')
    print(f'n_events  = {args.n_events}   warmup = {args.warmup}')
    print()

    # ---------------------------------------------------------------
    # 1) Build everything once
    # ---------------------------------------------------------------
    print('--- building objects ---')
    t0 = time.perf_counter()
    edep_r = LUCiDEdepReader(data_root=f'{args.data_root}/edep',
                             dataset_name='wc')
    edep_r.h5py_worker_init()
    print(f'  edep reader      built+opened in {1000*(time.perf_counter()-t0):.1f} ms')

    t0 = time.perf_counter()
    sensor_r = LUCiDSensorReader(data_root=f'{args.data_root}/sensor',
                                 dataset_name='wc')
    sensor_r.h5py_worker_init()
    print(f'  sensor reader    built+opened in {1000*(time.perf_counter()-t0):.1f} ms')

    t0 = time.perf_counter()
    hits_r = LUCiDHitsReader(data_root=f'{args.data_root}/hits',
                             dataset_name='wc')
    hits_r.h5py_worker_init()
    print(f'  hits reader      built+opened in {1000*(time.perf_counter()-t0):.1f} ms')

    t0 = time.perf_counter()
    labl_r = LUCiDLablReader(data_root=f'{args.data_root}/labl',
                             dataset_name='wc')
    labl_r.h5py_worker_init()
    print(f'  labl reader      built+opened in {1000*(time.perf_counter()-t0):.1f} ms')

    t0 = time.perf_counter()
    ds_raw = LUCiDDataset(data_root=args.data_root, split='',
                          dataset_name='wc',
                          modalities=('edep', 'sensor', 'hits', 'labl'))
    print(f'  LUCiDDataset (no transform) built in {1000*(time.perf_counter()-t0):.1f} ms')

    t0 = time.perf_counter()
    ds = LUCiDDataset(data_root=args.data_root, split='',
                      dataset_name='wc',
                      modalities=('edep', 'sensor', 'hits', 'labl'),
                      transform=DEFAULT_TRANSFORM)
    print(f'  LUCiDDataset (+ Collect) built in {1000*(time.perf_counter()-t0):.1f} ms')

    t0 = time.perf_counter()
    raw_read, raw_close, n_total = _raw_h5py_reader(args.data_root)
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

    edep_times   = _time_callable(edep_r.read_event,   max_events, args.warmup)
    sensor_times = _time_callable(sensor_r.read_event, max_events, args.warmup)
    hits_times   = _time_callable(hits_r.read_event,   max_events, args.warmup)
    labl_times   = _time_callable(labl_r.read_event,   max_events, args.warmup)
    sum_readers  = (np.array(edep_times) + np.array(sensor_times)
                    + np.array(hits_times) + np.array(labl_times))

    getdata_times    = _time_callable(ds_raw.get_data, max_events, args.warmup)
    getitem_times    = _time_callable(ds.__getitem__,  max_events, args.warmup)

    print()
    _stat('raw h5py (I/O ceiling)',          raw_times,     max_events)
    print()
    _stat('LUCiDEdepReader.read_event',      edep_times,    max_events, raw_mean)
    _stat('LUCiDSensorReader.read_event',    sensor_times,  max_events, raw_mean)
    _stat('LUCiDHitsReader.read_event',      hits_times,    max_events, raw_mean)
    _stat('LUCiDLablReader.read_event',      labl_times,    max_events, raw_mean)
    _stat('  Σ four readers (independent)',  sum_readers,   max_events, raw_mean)
    print()
    _stat('LUCiDDataset.get_data (assemble)', getdata_times, max_events, raw_mean)
    _stat('LUCiDDataset.__getitem__ (+ Collect)',
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
