"""Generic label decoration (Part 04): label_config → named schema keys.

Validates decorate_labels and its opt-in wiring in LUCiDDataset against
hand-computed FK gathers on the synthetic fixtures (whose FK invariants are
guaranteed by testing.py).
"""
import os

import numpy as np
import h5py
import pytest

from pimm_data.testing import make_lucid_sample
from pimm_data.lucid import LUCiDDataset
from pimm_data._label_decorate import gather_with_fill, decorate_labels


_LUCID_SEG_CONFIG = [
    dict(out="segment_pid", scope="point", fk="particle_idx",
         source=("particle", "category")),
    dict(out="instance_particle", scope="point", fk="particle_idx",
         source="self"),
]


# ---------------------------------------------------------------------------
# gather_with_fill primitive
# ---------------------------------------------------------------------------

def test_gather_positional():
    col = np.array([10, 20, 30], dtype=np.int32)
    fk = np.array([0, 2, -1, 5])           # -1 and 5 out of range → fill
    out = gather_with_fill(fk, col, fill=-1)
    assert out.tolist() == [10, 30, -1, -1]


def test_gather_value_keyed():
    keys = np.array([5, 9, 13], dtype=np.int32)     # track_ids
    vals = np.array([100, 200, 300], dtype=np.int32)
    fk = np.array([9, 13, 7])               # 7 absent → fill
    out = gather_with_fill(fk, vals, keyed_by=keys, fill=-1)
    assert out.tolist() == [200, 300, -1]


def test_decorate_event_broadcast():
    sub = {"coord": np.zeros((4, 3), dtype=np.float32)}
    labl = {"event": {"contained": np.array(True)}}
    cfg = [dict(out="target_contained", scope="event_broadcast",
                source=("event", "contained"))]
    decorate_labels(sub, labl, lambda n: None, cfg)
    assert sub["target_contained"].shape == (4, 1)
    assert sub["target_contained"].all()


# ---------------------------------------------------------------------------
# LUCiDDataset opt-in wiring
# ---------------------------------------------------------------------------

def test_lucid_label_config_emits_named_keys(tmp_path):
    root = make_lucid_sample(str(tmp_path), n_events=2)
    ds = LUCiDDataset(data_root=root, split="", modalities=("hits", "labl"),
                      label_config=_LUCID_SEG_CONFIG)
    sample = ds.get_data(0)
    hits = sample["hits"]
    assert "segment_pid" in hits and "instance_particle" in hits

    # Hand-compute the expected gather from the raw fixture.
    with h5py.File(os.path.join(root, "labl", "wc_labl_0000.h5"), "r") as f:
        category = f["event_000/per_particle/category"][()]
    with h5py.File(os.path.join(root, "hits", "wc_hits_0000.h5"), "r") as f:
        particle_idx = f["event_000/particle_idx"][()]
    expected_seg = gather_with_fill(particle_idx, category)
    assert hits["segment_pid"][:, 0].tolist() == expected_seg.tolist()
    assert hits["instance_particle"][:, 0].tolist() == particle_idx.tolist()


def test_lucid_label_config_default_off_unchanged(tmp_path):
    """Without label_config the bare segment/instance behavior is preserved."""
    root = make_lucid_sample(str(tmp_path), n_events=2)
    ds = LUCiDDataset(data_root=root, split="", modalities=("hits", "labl"))
    hits = ds.get_data(0)["hits"]
    assert "segment" in hits and "instance" in hits
    assert "segment_pid" not in hits        # named keys only when opted in


def test_lucid_instance_interaction_one_hop(tmp_path):
    """instance_interaction = per_particle.interaction_idx[particle_idx],
    a one-hop point gather via the generic decorator (no new code)."""
    root = make_lucid_sample(str(tmp_path), n_events=2)
    cfg = [dict(out="instance_interaction", scope="point", fk="particle_idx",
                source=("particle", "interaction_idx"))]
    ds = LUCiDDataset(data_root=root, split="", modalities=("hits", "labl"),
                      label_config=cfg)
    hits = ds.get_data(0)["hits"]
    assert "instance_interaction" in hits

    with h5py.File(os.path.join(root, "labl", "wc_labl_0000.h5"), "r") as f:
        inter = f["event_000/per_particle/interaction_idx"][()]
    with h5py.File(os.path.join(root, "hits", "wc_hits_0000.h5"), "r") as f:
        particle_idx = f["event_000/particle_idx"][()]
    expected = gather_with_fill(particle_idx, inter)
    assert hits["instance_interaction"][:, 0].tolist() == expected.tolist()


def test_lucid_label_config_edep(tmp_path):
    root = make_lucid_sample(str(tmp_path), n_events=2)
    ds = LUCiDDataset(data_root=root, split="", modalities=("edep", "labl"),
                      label_config=_LUCID_SEG_CONFIG)
    edep = ds.get_data(0)["edep"]
    assert "segment_pid" in edep and "instance_particle" in edep
    # edep particle_idx is resolved track_idx → per_track.particle_idx; the
    # named instance_particle must equal that resolved index.
    assert edep["instance_particle"][:, 0].tolist() == \
        edep["particle_idx"].tolist()
