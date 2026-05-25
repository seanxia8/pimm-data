#!/usr/bin/env python3
"""Re-encode JAXTPC production HDF5 files to a different compression codec.

Reads each dataset, rewrites it under a new codec, preserving the group
tree and all attributes. Used to measure (and apply) the gzip -> blosc/zstd
speedup from the loader profiling — gzip is the slowest codec on both read
and write; blosc:zstd+shuffle is smaller AND ~2.3x faster to read,
blosc:lz4+shuffle is ~4x faster to read for ~19% more disk.

Usage:
    # one file
    scripts/transcode_codec.py --src a.h5 --dst b.h5 --codec blosc-lz4
    # a run dir (mirrors {edep,sensor,hits}/<run>/ structure), first N shards
    scripts/transcode_codec.py --src <root> --dst <root2> --run run_00266285XX \
        --modalities hits edep sensor --shards 1 --codec blosc-lz4

Downstream readers of blosc/zstd/lz4 files must `import hdf5plugin`.
"""
import argparse
import glob
import os
import time

import h5py
import numpy as np
import hdf5plugin


def codec_kwargs(name):
    """Map a codec name to create_dataset kwargs."""
    if name == 'gzip':       return dict(compression='gzip', compression_opts=4)
    if name == 'gzip-1':     return dict(compression='gzip', compression_opts=1)
    if name == 'lzf':        return dict(compression='lzf')
    if name == 'lz4':        return dict(**hdf5plugin.LZ4())
    if name == 'zstd':       return dict(**hdf5plugin.Zstd(clevel=3))
    if name == 'blosc-lz4':
        return dict(**hdf5plugin.Blosc(cname='lz4', clevel=5,
                                       shuffle=hdf5plugin.Blosc.SHUFFLE))
    if name == 'blosc-zstd':
        return dict(**hdf5plugin.Blosc(cname='zstd', clevel=3,
                                       shuffle=hdf5plugin.Blosc.SHUFFLE))
    if name == 'none':       return dict()
    raise ValueError(f'unknown codec {name}')


# Compression needs a chunked layout and ≥1 element; tiny/scalar datasets
# are copied verbatim (compressing them adds overhead for no gain).
_MIN_COMPRESS_BYTES = 2048


def _transcode_group(src, dst, comp_kw):
    for ak, av in src.attrs.items():
        dst.attrs[ak] = av
    for key in src:
        item = src[key]
        if isinstance(item, h5py.Group):
            _transcode_group(item, dst.create_group(key), comp_kw)
        else:
            data = item[()]
            kw = comp_kw if (item.ndim >= 1 and item.size > 0
                             and item.nbytes >= _MIN_COMPRESS_BYTES) else {}
            d = dst.create_dataset(key, data=data, **kw)
            for ak, av in item.attrs.items():
                d.attrs[ak] = av


def transcode_file(src_path, dst_path, comp_kw):
    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
    with h5py.File(src_path, 'r') as fs, h5py.File(dst_path, 'w') as fd:
        _transcode_group(fs, fd, comp_kw)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True, help='source file or dataset root')
    p.add_argument('--dst', required=True, help='dest file or dataset root')
    p.add_argument('--codec', default='blosc-lz4',
                   help='gzip|gzip-1|lzf|lz4|zstd|blosc-lz4|blosc-zstd|none')
    p.add_argument('--run', default=None, help='run subdir (dir mode)')
    p.add_argument('--modalities', nargs='+', default=['hits', 'edep', 'sensor'])
    p.add_argument('--dataset-name', default='sim')
    p.add_argument('--shards', type=int, default=None,
                   help='limit to first N shards per modality (dir mode)')
    args = p.parse_args()

    comp_kw = codec_kwargs(args.codec)

    if os.path.isfile(args.src):
        t0 = time.perf_counter()
        transcode_file(args.src, args.dst, comp_kw)
        print(f'{args.src} -> {args.dst}  ({time.perf_counter()-t0:.1f}s, '
              f'{os.path.getsize(args.dst)/2**20:.0f} MB)')
        return

    # directory mode: mirror {modality}/{run}/{name}_{modality}_*.h5
    for m in args.modalities:
        base = os.path.join(args.src, m, args.run or '')
        files = sorted(glob.glob(os.path.join(base, f'{args.dataset_name}_{m}_*.h5')))
        if args.shards:
            files = files[:args.shards]
        for sp in files:
            rel = os.path.relpath(sp, args.src)
            dp = os.path.join(args.dst, rel)
            t0 = time.perf_counter()
            transcode_file(sp, dp, comp_kw)
            print(f'  {rel}: {os.path.getsize(sp)/2**20:.0f} -> '
                  f'{os.path.getsize(dp)/2**20:.0f} MB  '
                  f'({time.perf_counter()-t0:.1f}s)', flush=True)
    print('done')


if __name__ == '__main__':
    main()
