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

# Allow `python scripts/benchmark_loader.py` from any CWD (sibling import).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _profile_common as pc


def _list_of_dicts_collate(batch):
    """Pass batch through as a list; default collate can't stack ragged."""
    return batch


def _build_dataset(dataset, data_root, split, transform_variant='loading_only'):
    return pc.build_dataset(
        dataset, data_root, split=split,
        transform=pc.transform_variant(dataset, transform_variant))


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
               persistent_workers, n_warmup, n_timed=None,
               target_events=None, max_seconds=None, return_times=False):
    """Run one (config) cell. Returns a timings dict (and the raw per-batch
    times in ms if ``return_times``).

    The timed phase stops at whichever comes first: ``n_timed`` /
    ``target_events`` batches collected, ``max_seconds`` of wall time, or the
    end of one epoch. ``target_events`` is converted to a batch count via
    ``ceil(target_events / batch_size)`` so every batch size gets comparable
    statistical power (same number of events sampled).
    """
    target_batches = n_timed
    if target_batches is None and target_events is not None:
        target_batches = max(1, int(np.ceil(target_events / batch_size)))

    with _loader(ds, batch_size=batch_size, num_workers=num_workers,
                 prefetch_factor=prefetch_factor,
                 persistent_workers=persistent_workers) as dl:
        it = iter(dl)
        # Warmup: also captures worker-spinup cost. Excluded from timing.
        t_first = time.perf_counter()
        for _ in range(n_warmup):
            next(it)
        first_batch_overhead = time.perf_counter() - t_first  # incl. spinup

        # Timed phase
        per_batch = []
        t_phase = time.perf_counter()
        while True:
            if target_batches is not None and len(per_batch) >= target_batches:
                break
            if max_seconds is not None and (time.perf_counter() - t_phase) >= max_seconds:
                break
            t0 = time.perf_counter()
            try:
                next(it)
            except StopIteration:
                break  # exhausted the dataset (one epoch)
            per_batch.append(time.perf_counter() - t0)
        phase_dt = time.perf_counter() - t_phase

    per_batch_ms = 1000 * np.asarray(per_batch)
    n = len(per_batch)
    result = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        n_timed=n,
        events_timed=n * batch_size,
        warmup_total_s=first_batch_overhead,
        phase_total_s=phase_dt,
        batches_per_s=n / phase_dt if phase_dt else 0.0,
        samples_per_s=n * batch_size / phase_dt if phase_dt else 0.0,
        per_batch_ms_mean=float(per_batch_ms.mean()) if n else 0.0,
        per_batch_ms_std=float(per_batch_ms.std()) if n else 0.0,
        per_batch_ms_p25=float(np.percentile(per_batch_ms, 25)) if n else 0.0,
        per_batch_ms_p50=float(np.percentile(per_batch_ms, 50)) if n else 0.0,
        per_batch_ms_p75=float(np.percentile(per_batch_ms, 75)) if n else 0.0,
        per_batch_ms_p95=float(np.percentile(per_batch_ms, 95)) if n else 0.0,
    )
    if return_times:
        return result, per_batch_ms
    return result


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


def _load_csv(path):
    """Load a timings CSV back into row dicts, casting numeric columns."""
    int_cols = {'batch_size', 'num_workers', 'prefetch_factor', 'n_timed'}
    rows = []
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            out = {}
            for k, v in r.items():
                if k in int_cols:
                    out[k] = int(float(v))
                elif k == 'persistent_workers':
                    out[k] = v == 'True'
                else:
                    try:
                        out[k] = float(v)
                    except (ValueError, TypeError):
                        out[k] = v
            rows.append(out)
    return rows


def _title_suffix(args):
    """'· dataset: root[/split]  ·  prefetch_factor=N' for plot titles."""
    loc = os.path.basename(args.data_root)
    if args.split:
        loc += f'/{args.split}'
    return f'·  {args.dataset}: {loc}  ·  prefetch_factor={args.prefetch_factor}'


def _plot_lines(rows, out_prefix, title_suffix, latency_ymax=None):
    """Two figures, one line per num_workers with a shaded IQR band:

      * ``{prefix}_latency.png``    — per-sample latency [ms]
        (per-batch latency / batch_size, so cells with different batch
        sizes are directly comparable)
      * ``{prefix}_throughput.png`` — throughput [samples / s]

    The band spans the p25–p75 inter-quartile range of per-batch timings
    (propagated to per-sample / throughput); the line is the median.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    workers = sorted({r['num_workers'] for r in rows})
    batch_sizes = sorted({r['batch_size'] for r in rows})
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
        'grid.alpha': 0.3,
        'grid.linewidth': 0.6,
        'axes.axisbelow': True,
        'legend.frameon': False,
        'figure.dpi': 120,
        'savefig.dpi': 200,
        'savefig.bbox': 'tight',
    })

    def _panel(y_of, ylabel, title, path, ymax=None, clamp_median=False):
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        max_med = 0.0
        for w, color in zip(workers, colors):
            cells = sorted([r for r in rows if r['num_workers'] == w],
                           key=lambda r: r['batch_size'])
            xs = np.array([r['batch_size'] for r in cells], dtype=float)
            med, lo, hi = y_of(cells, xs)
            max_med = max(max_med, float(np.max(med)))
            ax.fill_between(xs, lo, hi, color=color, alpha=0.15, linewidth=0)
            ax.plot(xs, med, color=color, marker='o', markersize=5,
                    linewidth=1.9, alpha=0.95, label=f'{w}')
        ax.set_xscale('log', base=2)
        ax.set_xticks(batch_sizes)
        ax.set_xticklabels([str(b) for b in batch_sizes])
        ax.set_xlabel('batch size')
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc='left', fontweight='bold')
        # Frame to the median lines (+15% headroom) so a few high-variance
        # configs' IQR bands clip at the top instead of stretching the axis.
        if ymax is not None:
            ax.set_ylim(0, ymax)
        elif clamp_median and max_med > 0:
            ax.set_ylim(0, max_med * 1.15)
        else:
            ax.set_ylim(bottom=0)
        ax.margins(x=0.03)
        leg = ax.legend(title='num_workers', loc='best', fontsize=9,
                        ncol=2 if len(workers) > 4 else 1,
                        labelspacing=0.3, handlelength=1.6)
        leg.get_title().set_fontsize(9)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        print(f'wrote {path}')

    # per-sample latency: per-batch percentiles divided by batch size
    def _latency(cells, xs):
        p50 = np.array([r['per_batch_ms_p50'] for r in cells])
        p25 = np.array([r['per_batch_ms_p25'] for r in cells])
        p75 = np.array([r['per_batch_ms_p75'] for r in cells])
        return p50 / xs, p25 / xs, p75 / xs

    # throughput band from matched IQR latency endpoints
    # (slow p75 batch → low samples/s, fast p25 batch → high samples/s)
    def _throughput(cells, xs):
        med = xs / (np.array([r['per_batch_ms_p50'] for r in cells]) / 1000)
        lo  = xs / (np.array([r['per_batch_ms_p75'] for r in cells]) / 1000)
        hi  = xs / (np.array([r['per_batch_ms_p25'] for r in cells]) / 1000)
        return med, lo, hi

    _panel(_latency, 'per-sample latency [ms]  (median, IQR band)',
           f'Per-sample latency  {title_suffix}', f'{out_prefix}_latency.png',
           ymax=latency_ymax, clamp_median=True)
    _panel(_throughput, 'throughput [samples / s]  (median, IQR band)',
           f'Throughput  {title_suffix}', f'{out_prefix}_throughput.png')


def _plot_hist(times_ms, num_workers, batch_size, title_suffix, out_path,
               bins=40, logx=False):
    """Histogram of per-batch latency [ms] for one (workers, batch) cell.

    Shows the full timing distribution — the shape of the right tail that
    drives the IQR band — with median / mean / p95 marked.

    ``logx`` uses a log x-axis with log-spaced bins, which is the right
    view for a heavy-tailed latency distribution: the dense bulk near the
    median and the long slow-event tail are both legible on one plot.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    times_ms = np.asarray(times_ms, dtype=float)
    n = len(times_ms)
    med = float(np.median(times_ms))
    mean = float(times_ms.mean())
    p95 = float(np.percentile(times_ms, 95))
    cv = float(times_ms.std() / mean) if mean else float('nan')

    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 11,
        'axes.titlesize': 12.5, 'axes.labelsize': 12,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.alpha': 0.3, 'grid.linewidth': 0.6,
        'axes.axisbelow': True, 'legend.frameon': False,
        'figure.dpi': 120, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
    })
    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    if logx:
        lo = max(1.0, float(times_ms.min()) * 0.9)
        hi = float(times_ms.max()) * 1.1
        bin_edges = np.logspace(np.log10(lo), np.log10(hi), bins + 1)
        ax.set_xscale('log')
        ax.set_xlim(lo, hi)
    else:
        bin_edges = bins
        ax.set_xlim(left=0)
    ax.hist(times_ms, bins=bin_edges, color='#3b6fb0', alpha=0.85,
            edgecolor='white', linewidth=0.4)
    for val, c, lab in [(med, '#1a1a1a', f'median  {med:.0f} ms'),
                        (mean, '#d1495b', f'mean  {mean:.0f} ms'),
                        (p95, '#e8a33d', f'p95  {p95:.0f} ms')]:
        ax.axvline(val, color=c, linestyle='--', linewidth=1.7, label=lab)
    ax.set_xlabel('per-batch latency [ms]' + ('  (log scale)' if logx else ''))
    ax.set_ylabel('count  (batches)')
    ax.set_title(f'Per-batch latency distribution  ·  workers={num_workers}, '
                 f'batch={batch_size}\n{title_suffix}',
                 loc='left', fontweight='bold')
    leg = ax.legend(loc='upper right', fontsize=9,
                    title=f'n = {n} batches  ·  CV = {cv:.2f}\n'
                          f'median/sample = {med / batch_size:.1f} ms')
    leg.get_title().set_fontsize(9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f'wrote {out_path}  (n={n}, median={med:.0f} ms, CV={cv:.2f})')


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__.split('\n\n')[0])
    p.add_argument('--dataset', choices=pc.DATASETS, default='lucid',
                   help='Which dataset/loader to benchmark (default: lucid)')
    p.add_argument('--data-root', default=None,
                   help='Dataset root. Default: WAND GENIE numu (lucid) or '
                        'doraemon (jaxtpc).')
    p.add_argument('--split', default=None,
                   help='Split subdir under each modality dir. Default: "" '
                        '(lucid) or the doraemon run (jaxtpc).')
    p.add_argument('--transform-variant', default='loading_only',
                   choices=['loading_only', 'collect_first', 'totensor_first'],
                   help='Collect/ToTensor ordering (see _profile_common.'
                        'transform_variant). loading_only=[Collect] (prior '
                        'baseline), collect_first=[Collect,ToTensor] (the '
                        'fix), totensor_first=[ToTensor,Collect] (the '
                        'training-config order that tensorizes discarded '
                        'streams). Default: loading_only.')
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
                   help='batches timed per cell (ignored if --target-events set)')
    p.add_argument('--target-events', type=int, default=None,
                   help='sample this many EVENTS per cell instead of a fixed '
                        'batch count (n_timed = ceil(target / batch_size)), '
                        'so every batch size gets equal statistical power')
    p.add_argument('--max-seconds-per-cell', type=float, default=None,
                   help='also stop a cell after this many wall seconds '
                        '(bounds total runtime; slow cells get fewer events)')
    # Histogram mode: per-batch timing distribution for one worker count.
    p.add_argument('--hist-workers', type=int, default=None,
                   help='if set, after the grid, emit a per-batch latency '
                        'histogram (one figure per --hist-batches) at this '
                        'worker count')
    p.add_argument('--hist-only', action='store_true',
                   help='skip the grid (and its CSV / line plots) and only '
                        'produce the histograms; requires --hist-workers')
    p.add_argument('--hist-logx', action='store_true',
                   help='log x-axis + log-spaced bins for the histograms '
                        '(best view for the heavy-tailed latency)')
    p.add_argument('--hist-replot', action='store_true',
                   help='re-draw histograms from previously saved '
                        '{prefix}_hist_w{W}_b{B}.csv files (no data '
                        'collection); needs --hist-workers, --hist-batches, '
                        '--plot')
    p.add_argument('--hist-batches', nargs='+', type=int, default=[32, 64],
                   help='batch sizes to histogram (default: 32 64)')
    p.add_argument('--hist-max-batches', type=int, default=600,
                   help='max batches to collect per histogram (capped at one '
                        'epoch; default 600)')
    p.add_argument('--hist-bins', type=int, default=40,
                   help='histogram bin count (default 40)')
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
                   help='path prefix for the latency / throughput PNGs')
    p.add_argument('--replot', default=None,
                   help='regenerate plots from an existing CSV and exit '
                        '(no dataset build, no timing); needs --plot')
    p.add_argument('--only-workers', nargs='+', type=int, default=None,
                   help='replot: draw only these num_workers lines '
                        '(default: all present in the CSV)')
    p.add_argument('--latency-ymax', type=float, default=None,
                   help='cap the per-sample latency y-axis [ms] (default: '
                        'auto = 1.15x the max median line, so wide IQR bands '
                        'clip instead of stretching the axis)')
    args = p.parse_args()

    if args.data_root is None:
        args.data_root = pc.default_root(args.dataset)
    if args.split is None:
        args.split = pc.default_split(args.dataset)
    if args.hist_only and args.hist_workers is None:
        p.error('--hist-only requires --hist-workers')

    # Histogram replot fast path: redraw from saved raw-times CSVs, no timing.
    if args.hist_replot:
        if args.hist_workers is None or not args.plot:
            p.error('--hist-replot requires --hist-workers and --plot')
        prefix = args.plot[:-4] if args.plot.endswith('.png') else args.plot
        for b in args.hist_batches:
            src = f'{prefix}_hist_w{args.hist_workers}_b{b}.csv'
            times = np.loadtxt(src, delimiter=',', skiprows=1)
            _plot_hist(times, args.hist_workers, b, _title_suffix(args),
                       f'{prefix}_hist_w{args.hist_workers}_b{b}.png',
                       bins=args.hist_bins, logx=args.hist_logx)
        return

    # Replot-only fast path: load a prior CSV and redraw, skip all timing.
    if args.replot:
        if not args.plot:
            p.error('--replot requires --plot (output prefix)')
        rows = _load_csv(args.replot)
        # The CSV is grid rows first, then prefetch/persistent sweep extras.
        # The sweeps always run at the max worker count (and a batch size
        # that may not be in the grid), so the clean main grid = batch sizes
        # seen at the minimum worker count. Filter to those, then keep the
        # first row per (workers, batch).
        min_w = min(r['num_workers'] for r in rows)
        grid_batches = {r['batch_size'] for r in rows if r['num_workers'] == min_w}
        seen, grid_rows = set(), []
        for r in rows:
            key = (r['num_workers'], r['batch_size'])
            if r['batch_size'] in grid_batches and key not in seen:
                seen.add(key)
                grid_rows.append(r)
        if args.only_workers is not None:
            keep = set(args.only_workers)
            grid_rows = [r for r in grid_rows if r['num_workers'] in keep]
        prefix = args.plot[:-4] if args.plot.endswith('.png') else args.plot
        _plot_lines(grid_rows, prefix, _title_suffix(args),
                    latency_ymax=args.latency_ymax)
        return

    # Per-cell timing budget shared by the grid and the sweeps. target_events
    # (if set) overrides the fixed n_timed; max_seconds bounds slow cells.
    budget = dict(
        n_timed=None if args.target_events else args.n_timed,
        target_events=args.target_events,
        max_seconds=args.max_seconds_per_cell,
    )

    print(f'dataset   = {args.dataset}')
    print(f'data_root = {args.data_root}')
    print(f'split     = {args.split!r}')
    print(f'transform = {args.transform_variant}')
    print(f'workers   = {args.workers}')
    print(f'batches   = {args.batches}')
    if args.target_events:
        print(f'target_events={args.target_events}  '
              f'max_s/cell={args.max_seconds_per_cell}  warmup={args.n_warmup}')
    else:
        print(f'warmup={args.n_warmup} timed={args.n_timed}')
    print()

    ds = _build_dataset(args.dataset, args.data_root, args.split,
                        transform_variant=args.transform_variant)
    print(f'len(ds) = {len(ds)}\n')

    if not args.no_fs_warmup:
        _warmup_filesystem(ds, n_events=200)
        print()

    grid_rows = []
    if not args.hist_only:
        print('=== main grid: num_workers × batch_size ===')
    for w in ([] if args.hist_only else args.workers):
        for b in args.batches:
            print(f'  cell: workers={w} batch={b} ...', flush=True)
            row = _time_cell(
                ds, batch_size=b, num_workers=w,
                prefetch_factor=args.prefetch_factor,
                persistent_workers=(w > 0),
                n_warmup=args.n_warmup, **budget)
            print(f'    -> {row["n_timed"]} batches, {row["events_timed"]} ev, '
                  f'{row["samples_per_s"]:.1f} samples/s', flush=True)
            grid_rows.append(row)
    if grid_rows:
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
                    n_warmup=args.n_warmup, **budget)
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
                n_warmup=args.n_warmup, **budget)
            persistent_rows.append(row)
        _print_table(persistent_rows,
                     fixed=[('cell', 'workers=4, batch=16')])
        extra_rows.extend(persistent_rows)

    all_rows = grid_rows + extra_rows
    if args.csv and not args.hist_only:
        _save_csv(all_rows, args.csv)
    if args.plot and not args.hist_only:
        # args.plot is treated as a path prefix; two PNGs are written:
        #   {prefix}_latency.png and {prefix}_throughput.png
        prefix = args.plot
        if prefix.endswith('.png'):
            prefix = prefix[:-4]
        _plot_lines(grid_rows, prefix, _title_suffix(args),
                    latency_ymax=args.latency_ymax)

    # --- Histogram mode: per-batch timing distribution at one worker count ---
    if args.hist_workers is not None:
        prefix = (args.plot[:-4] if args.plot and args.plot.endswith('.png')
                  else (args.plot or f'hist_{args.dataset}'))
        print(f'\n=== histograms: workers={args.hist_workers}, '
              f'batches={args.hist_batches} (<= {args.hist_max_batches} '
              f'batches/epoch each) ===')
        for b in args.hist_batches:
            print(f'  collecting: workers={args.hist_workers} batch={b} ...',
                  flush=True)
            # No max_seconds here — let it run to hist_max_batches (or epoch)
            # so the histogram has plenty of samples.
            _, times = _time_cell(
                ds, batch_size=b, num_workers=args.hist_workers,
                prefetch_factor=args.prefetch_factor,
                persistent_workers=(args.hist_workers > 0),
                n_warmup=args.n_warmup, n_timed=args.hist_max_batches,
                return_times=True)
            out_png = f'{prefix}_hist_w{args.hist_workers}_b{b}.png'
            out_csv = f'{prefix}_hist_w{args.hist_workers}_b{b}.csv'
            np.savetxt(out_csv, times, delimiter=',',
                       header='per_batch_ms', comments='')
            _plot_hist(times, args.hist_workers, b, _title_suffix(args),
                       out_png, bins=args.hist_bins, logx=args.hist_logx)


if __name__ == '__main__':
    sys.exit(main())
