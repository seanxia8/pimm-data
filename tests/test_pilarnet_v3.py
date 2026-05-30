"""PILArNet Rb merge: v3 is_primary plumbing, width-6 guard, v1/v2 unchanged.

v3 adds a per-cluster is_primary flag (6-wide cluster_extra, column 5) emitted
as a per-point is_primary key. The v3 branch is additive — v1/v2 paths and the
rotations=None default are byte-unchanged.
"""
import os

import numpy as np
import h5py
import pytest

from pimm_data.pilarnet import PILArNetH5Dataset

_VLEN = h5py.special_dtype(vlen=np.dtype("float32"))
_VLEN_INT = h5py.special_dtype(vlen=np.dtype("int32"))


def _make_pilarnet_h5(root, revision, extra_width=6, npoints=10,
                      cluster_sizes=(6, 4)):
    """One-event PILArNet shard at {root}/foo_train/data.h5 (+ _points.npy).

    point: (npoints*8,) flat — cols [0,1,2,3] = x,y,z,e.
    cluster: (n_clusters*6,) — reshape(-1,6)[:,[0,2,3,4,5]] = size,group,inter,sem,pid.
    cluster_extra: (n_clusters*extra_width,) — [:, [1,2,3,4(,5)]] = mom,vtx_x/y/z(,is_primary).
    """
    d = os.path.join(root, "foo_train")
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(0)
    n_cl = len(cluster_sizes)

    point = rng.uniform(0, 700, size=(npoints, 8)).astype(np.float32)
    point[:, 3] = 1.0                                  # energy above threshold
    cluster = np.zeros((n_cl, 6), dtype=np.int32)
    cluster[:, 0] = cluster_sizes                      # cluster_size
    cluster[:, 2] = np.arange(n_cl)                    # group_id
    cluster[:, 3] = np.arange(n_cl)                    # interaction_id
    cluster[:, 4] = [1, 0]                             # semantic_id (track, shower)
    cluster[:, 5] = [2, 0]                             # pid (muon, photon)
    extra = np.zeros((n_cl, extra_width), dtype=np.float32)
    extra[:, 1] = [0.5, 0.3]                           # mom
    extra[:, 2:5] = rng.uniform(0, 700, size=(n_cl, 3))  # vertex
    if extra_width >= 6:
        extra[:, 5] = [1, 0]                           # is_primary

    path = os.path.join(d, "data.h5")
    with h5py.File(path, "w") as f:
        for name, arr, dt in (("point", point, _VLEN),
                              ("cluster", cluster, _VLEN_INT),
                              ("cluster_extra", extra, _VLEN)):
            ds = f.create_dataset(name, shape=(1,), dtype=dt)
            ds[0] = arr.reshape(-1)
    np.save(os.path.join(d, "data_points.npy"), np.array([npoints]))
    return path


def test_v3_emits_is_primary_repeated_by_cluster_size(tmp_path):
    _make_pilarnet_h5(str(tmp_path), "v3")
    ds = PILArNetH5Dataset(data_root=str(tmp_path), split="train",
                           revision="v3", min_points=1, transform=[])
    d = ds.get_data(0)
    assert "is_primary" in d
    # per-point, shape (N,1), repeated from per-cluster [1,0] by sizes (6,4)
    assert d["is_primary"].shape == (10, 1)
    assert d["is_primary"][:, 0].tolist() == [1] * 6 + [0] * 4


def test_v3_width5_cluster_extra_raises(tmp_path):
    _make_pilarnet_h5(str(tmp_path), "v3", extra_width=5)
    ds = PILArNetH5Dataset(data_root=str(tmp_path), split="train",
                           revision="v3", min_points=1, transform=[])
    with pytest.raises(ValueError, match="width 6"):
        ds.get_data(0)


def test_v2_has_no_is_primary(tmp_path):
    _make_pilarnet_h5(str(tmp_path), "v2", extra_width=5)
    ds = PILArNetH5Dataset(data_root=str(tmp_path), split="train",
                           revision="v2", min_points=1, transform=[])
    d = ds.get_data(0)
    assert "is_primary" not in d            # v2 path untouched by the v3 merge
    assert "vertex" in d and "segment_pid" in d


def test_unknown_revision_raises(tmp_path):
    _make_pilarnet_h5(str(tmp_path), "v2", extra_width=5)
    ds = PILArNetH5Dataset(data_root=str(tmp_path), split="train",
                           revision="v2", min_points=1, transform=[])
    ds.revision = "v9"                      # force the else branch
    with pytest.raises(ValueError, match="Unsupported PILArNet revision"):
        ds.get_data(0)
