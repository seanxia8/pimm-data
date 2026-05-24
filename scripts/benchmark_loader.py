#!/usr/bin/env python3
"""Throughput benchmark for the LUCiD DataLoader pipeline.

Measures end-to-end "shard → ready-to-feed sample dict" throughput for
the actual transform stack a model would consume, across (num_workers,
batch_size) combinations. Optionally sweeps prefetch_factor and
persistent_workers at the best (workers, batch) point.

Default behavior:
  - Dataset: WAND config_000013 (GENIE νμ, 101k events, 77 shards)
  - Transform: GridSample on hits → Collect to torch tensors
  - Grid: num_workers ∈ {0,1,2,4,8,16}  ×  batch_size ∈ {1,4,16,32,64}
  - For each cell: 5 warmup batches discarded, 50 batches timed
  - Custom collate_fn = list of dicts (default torch collate cannot
    stack variable-size point clouds; collation cost is a constant
    overhead in the main process — it does not affect worker scaling)

Caveats:
  - Time-to-first-batch (worker spinup) is excluded from the steady-state
    number. Persistent_workers=True keeps workers alive across the warmup
    + timed phases so spinup is paid exactly once per cell.
  - CPU-only timing; data is NOT moved to GPU. Pin_memory + transfer
    costs are GPU-side concerns and would add a constant per-batch
    overhead that doesn't change with worker count.
  - Filesystem cache state affects absolute numbers but not relative
    scaling. The script does a one-time read-through pass before timing
    to even out cold/warm effects across cells.

Usage:
  scripts/benchmark_loader.py
  scripts/benchmark_loader.py --data-root /path/to/config_NNNNNN
  scripts/benchmark_loader.py --workers 0 4 8 --batches 8 32
  scripts/benchmark_loader.py --csv /tmp/bench.csv --plot
  scripts/benchmark_loader.py --prefetch-sweep --persistent-sweep
"""
import argparse
import csv
import os
import sys
import time
from contextlib import contextmanager

import numpy as np
import torch
from torch.utils.data import DataLoader

from pimm_data import LUCiDDataset


# Minimum realistic transform: extract one stream and tensorize.
# This is the bare requirement for tensor-batched training; heavier
# augmentations (GridSample, jitter, etc.) are model-dependent and
# would conflate loader throughput with transform throughput.
DEFAULT_TRANSFORM = [
    dict(type='Collect', stream='hits',
         keys=['coord', 'energy', 'time'],
         feat_keys=['energy', 'time']),
]


def _list_of_dicts_collate(batch):
    """Pass batch through as a list; default collate can't stack ragged."""
    return batch


def _build_dataset(data_root):
    return LUCiDDataset(
        data_root=data_root, split='', dataset_name='wc',
        modalities=('edep', 'sensor', 'hits', 'labl'),
        transform=DEFAULT_TRANSFORM,
    )


def _warmup_filesystem(ds, n_events=200):
    """Touch the first N events once so per-cell timings see warm cache."""
    n = min(n_events, len(ds))
    print(f'  filesystem warmup: reading {n} events ...', flush=True)
    t0 = time.perf_counter()
    for i in range(n):
        _ = ds[i]
    dt = time.perf_counter() - t0
    print(f'  warmup done in {dt:.1f}s ({n/dt:.1f} samples/s, single-proc)',
          flush=True)


@contextmanager
def _loader(ds, *, batch_size, num_workers, prefetch_factor,
            persistent_workers):
    kw = dict(batch_size=batch_size, num_workers=num_workers,
              shuffle=False, drop_last=True,
              collate_fn=_list_of_dicts_collate)
    if num_workers > 0:
        kw['prefetch_factor'] = prefetch_factor
        kw['persistent_workers'] = persistent_workers
    dl = DataLoader(ds, **kw)
    try:
        yield dl
    finally:
        # Tear down worker processes explicitly so persistent_workers
        # doesn't leak between cells.
        del dl


def _time_cell(ds, *, batch_size, num_workers, prefetch_factor,
               persistent_workers, n_warmup, n_timed):
    """Run one (config) cell. Returns dict of timings + throughputs."""
    with _loader(ds, batch_size=batch_size, num_workers=num_workers,
                 prefetch_factor=prefetch_factor,
                 persistent_workers=persistent_workers) as dl:
        it = iter(dl)
        # Warmup: also captures worker-spinup cost. Excluded from timing.
        t_first = time.perf_counter()
        for _ in range(n_warmup):
            next(it)
        t_after_warmup = time.perf_counter()
        first_batch_overhead = t_after_warmup - t_first  # incl. spinup

        # Timed phase
        per_batch = []
        t_phase = time.perf_counter()
        for _ in range(n_timed):
            t0 = time.perf_counter()
            next(it)
            per_batch.append(time.perf_counter() - t0)
        phase_dt = time.perf_counter() - t_phase

    per_batch_ms = 1000 * np.asarray(per_batch)
    return dict(
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        n_timed=n_timed,
        warmup_total_s=first_batch_overhead,
        phase_total_s=phase_dt,
        batches_per_s=n_timed / phase_dt,
        samples_per_s=n_timed * batch_size / phase_dt,
        per_batch_ms_mean=float(per_batch_ms.mean()),
        per_batch_ms_std=float(per_batch_ms.std()),
        per_batch_ms_p25=float(np.percentile(per_batch_ms, 25)),
        per_batch_ms_p50=float(np.percentile(per_batch_ms, 50)),
        per_batch_ms_p75=float(np.percentile(per_batch_ms, 75)),
        per_batch_ms_p95=float(np.percentile(per_batch_ms, 95)),
    )


def _print_table(rows, fixed=()):
    """Pretty table; 'fixed' lists (label, value) pairs to print above."""
    if fixed:
        for k, v in fixed:
            print(f'  {k} = {v}')
        print()
    cols = ['num_workers', 'batch_size', 'samples_per_s',
            'batches_per_s', 'per_batch_ms_mean', 'per_batch_ms_p95',
            'warmup_total_s']
    headers = ['workers', 'batch', 'samples/s',
               'batches/s', 'ms/batch', 'ms/batch p95', 'warmup s']
    widths = [max(len(h), 9) for h in headers]
    fmt_row = '  '.join('{{:>{w}}}'.format(w=w) for w in widths)
    print(fmt_row.format(*headers))
    print(fmt_row.format(*['-' * w for w in widths]))
    for r in rows:
        vals = [r['num_workers'], r['batch_size'],
                f'{r["samples_per_s"]:.1f}',
                f'{r["batches_per_s"]:.2f}',
                f'{r["per_batch_ms_mean"]:.1f}',
                f'{r["per_batch_ms_p95"]:.1f}',
                f'{r["warmup_total_s"]:.1f}']
        print(fmt_row.format(*vals))
    print()


def _save_csv(rows, path):
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'wrote {path}')


def _plot_lines(rows, out_prefix, title_suffix):
    """Two line plots: latency vs batch_size and throughput vs batch_size,
    one line per num_workers, asymmetric error bars from per-batch IQR.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    workers = sorted({r['num_workers'] for r in rows})
    cmap = matplotlib.colormaps['viridis']
    colors = [cmap(i / max(len(workers) - 1, 1)) for i in range(len(workers))]

    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linestyle': '--',
        'grid.color': '#999',
        'axes.axisbelow': True,
        'figure.dpi': 100,
        'savefig.dpi': 160,
        'savefig.bbox': 'tight',
    })

    # --- Plot 1: per-batch latency ---
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for w, color in zip(workers, colors):
        cells = sorted([r for r in rows if r['num_workers'] == w],
                       key=lambda r: r['batch_size'])
        xs = np.array([r['batch_size'] for r in cells])
        med = np.array([r['per_batch_ms_p50'] for r in cells])
        lo = med - np.array([r['per_batch_ms_p25'] for r in cells])
        hi = np.array([r['per_batch_ms_p75'] for r in cells]) - med
        ax.errorbar(xs, med, yerr=np.vstack([lo, hi]),
                    label=f'workers={w}', color=color,
                    marker='o', markersize=5, linewidth=1.6,
                    capsize=3, capthick=1.2, elinewidth=1.0, alpha=0.95)
    ax.set_xscale('log', base=2)
    ax.set_xticks(sorted({r['batch_size'] for r in rows}))
    ax.set_xticklabels([str(b) for b in sorted({r['batch_size'] for r in rows})])
    ax.set_xlabel('batch_size')
    ax.set_ylabel('per-batch latency [ms]  (median, IQR bars)')
    ax.set_title(f'LUCiDDataset per-batch latency {title_suffix}')
    ax.legend(title='num_workers', loc='upper left', frameon=True,
              ncol=2 if len(workers) <= 6 else 3, fontsize=9)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    path = f'{out_prefix}_latency.png'
    fig.savefig(path)
    plt.close(fig)
    print(f'wrote {path}')

    # --- Plot 2: throughput ---
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for w, color in zip(workers, colors):
        cells = sorted([r for r in rows if r['num_workers'] == w],
                       key=lambda r: r['batch_size'])
        xs = np.array([r['batch_size'] for r in cells])
        # Convert per-batch IQR latency to throughput uncertainty.
        # samples/s = batch_size / (batch_time_s). Use IQR endpoints to
        # get matched throughput bounds (slow batch → low samples/s).
        thr_med = xs / (np.array([r['per_batch_ms_p50'] for r in cells]) / 1000)
        thr_lo  = xs / (np.array([r['per_batch_ms_p75'] for r in cells]) / 1000)
        thr_hi  = xs / (np.array([r['per_batch_ms_p25'] for r in cells]) / 1000)
        yerr = np.vstack([thr_med - thr_lo, thr_hi - thr_med])
        ax.errorbar(xs, thr_med, yerr=yerr,
                    label=f'workers={w}', color=color,
                    marker='o', markersize=5, linewidth=1.6,
                    capsize=3, capthick=1.2, elinewidth=1.0, alpha=0.95)
    ax.set_xscale('log', base=2)
    ax.set_xticks(sorted({r['batch_size'] for r in rows}))
    ax.set_xticklabels([str(b) for b in sorted({r['batch_size'] for r in rows})])
    ax.set_xlabel('batch_size')
    ax.set_ylabel('throughput [samples / s]  (median, IQR bars)')
    ax.set_title(f'LUCiDDataset throughput {title_suffix}')
    ax.legend(title='num_workers', loc='upper left', frameon=True,
              ncol=2 if len(workers) <= 6 else 3, fontsize=9)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    path = f'{out_prefix}_throughput.png'
    fig.savefig(path)
    plt.close(fig)
    print(f'wrote {path}')


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__.split('\n\n')[0])
    p.add_argument('--data-root',
                   default='/sdf/data/neutrino/omara/wand_sk_like/config_000013',
                   help='LUCiD v3 dataset root (default: WAND GENIE numu)')
    p.add_argument('--workers', nargs='+', type=int,
                   default=[0, 1, 2, 4, 8, 16],
                   help='num_workers values to sweep')
    p.add_argument('--batches', nargs='+', type=int,
                   default=[1, 4, 16, 32, 64],
                   help='batch_size values to sweep')
    p.add_argument('--prefetch-factor', type=int, default=2)
    p.add_argument('--n-warmup', type=int, default=5,
                   help='batches discarded before timing each cell')
    p.add_argument('--n-timed', type=int, default=50,
                   help='batches timed per cell')
    p.add_argument('--no-fs-warmup', action='store_true',
                   help='skip the one-time filesystem-warmup pass')
    p.add_argument('--prefetch-sweep', action='store_true',
                   help='also sweep prefetch_factor at (workers, batch) = '
                        '(highest workers, batch_size=16)')
    p.add_argument('--prefetch-values', nargs='+', type=int,
                   default=[1, 2, 4, 8],
                   help='prefetch_factor values used by the prefetch sweep')
    p.add_argument('--persistent-sweep', action='store_true',
                   help='also compare persistent_workers True/False at '
                        '(workers, batch) = (4, 16)')
    p.add_argument('--csv', default=None, help='write timings as CSV')
    p.add_argument('--plot', default=None,
                   help='write heatmap PNG of samples/s')
    args = p.parse_args()

    print(f'data_root = {args.data_root}')
    print(f'workers   = {args.workers}')
    print(f'batches   = {args.batches}')
    print(f'warmup={args.n_warmup} timed={args.n_timed}')
    print()

    ds = _build_dataset(args.data_root)
    print(f'len(ds) = {len(ds)}\n')

    if not args.no_fs_warmup:
        _warmup_filesystem(ds, n_events=200)
        print()

    print('=== main grid: num_workers × batch_size ===')
    grid_rows = []
    for w in args.workers:
        for b in args.batches:
            print(f'  cell: workers={w} batch={b} ...', flush=True)
            row = _time_cell(
                ds, batch_size=b, num_workers=w,
                prefetch_factor=args.prefetch_factor,
                persistent_workers=(w > 0),
                n_warmup=args.n_warmup, n_timed=args.n_timed)
            grid_rows.append(row)
    print()
    _print_table(grid_rows,
                 fixed=[('prefetch_factor', args.prefetch_factor),
                        ('persistent_workers', 'True (when workers>0)')])

    extra_rows = []
    if args.prefetch_sweep:
        w_top = max(args.workers)
        if w_top == 0:
            print('skipping prefetch sweep (no worker count > 0)')
        else:
            print(f'=== prefetch sweep: workers={w_top}, batch=16 ===')
            for pf in args.prefetch_values:
                print(f'  cell: prefetch_factor={pf}', flush=True)
                row = _time_cell(
                    ds, batch_size=16, num_workers=w_top, prefetch_factor=pf,
                    persistent_workers=True,
                    n_warmup=args.n_warmup, n_timed=args.n_timed)
                extra_rows.append(row)
            print()
            _print_table(extra_rows,
                         fixed=[('cell', f'workers={w_top}, batch=16')])

    if args.persistent_sweep:
        print('=== persistent_workers sweep: workers=4, batch=16 ===')
        persistent_rows = []
        for pw in (False, True):
            print(f'  cell: persistent_workers={pw}', flush=True)
            row = _time_cell(
                ds, batch_size=16, num_workers=4,
                prefetch_factor=args.prefetch_factor,
                persistent_workers=pw,
                n_warmup=args.n_warmup, n_timed=args.n_timed)
            persistent_rows.append(row)
        _print_table(persistent_rows,
                     fixed=[('cell', 'workers=4, batch=16')])
        extra_rows.extend(persistent_rows)

    all_rows = grid_rows + extra_rows
    if args.csv:
        _save_csv(all_rows, args.csv)
    if args.plot:
        # args.plot is treated as a path prefix; two PNGs are written:
        #   {prefix}_latency.png and {prefix}_throughput.png
        prefix = args.plot
        if prefix.endswith('.png'):
            prefix = prefix[:-4]
        title_suffix = (
            f'·  {os.path.basename(args.data_root)}'
            f'  ·  prefetch_factor={args.prefetch_factor}')
        _plot_lines(grid_rows, prefix, title_suffix)


if __name__ == '__main__':
    sys.exit(main())
