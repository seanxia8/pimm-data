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


# ---------------------------------------------------------------------------
# JAXTPCDataset opt-in label_config (value-keyed, per-volume)
# ---------------------------------------------------------------------------

def test_jaxtpc_label_config_named_keys(tmp_path):
    from pimm_data.jaxtpc import JAXTPCDataset
    from pimm_data.testing import make_jaxtpc_sample
    root = make_jaxtpc_sample(str(tmp_path), n_events=2, n_volumes=2)
    cfg = [
        dict(out="segment_pid", scope="point", source=("track", "pdg")),
        dict(out="instance_interaction", scope="point",
             source=("track", "interaction")),
    ]
    ds = JAXTPCDataset(data_root=root, split="",
                       modalities=("edep", "hits", "labl"),
                       dataset_name="sim", label_config=cfg)
    sample = ds.get_data(0)
    for stream in ("edep", "hits"):
        s = sample[stream]
        assert "segment_pid" in s and "instance_interaction" in s
        # named keys are (N,1); bare segment still present (back-compat)
        assert s["segment_pid"].shape == (s["coord"].shape[0], 1)
        assert "segment" in s
    # segment_pid (track_pdg axis) should equal the bare segment when
    # label_key='pdg' (the default) — same gather, different key spelling.
    edep = sample["edep"]
    assert edep["segment_pid"][:, 0].tolist() == edep["segment"].tolist()


def test_jaxtpc_label_config_self_source(tmp_path):
    """F4: source='self' is honored on JAXTPC too (parity with LUCiD) — it
    emits the per-point resolved track id, == the bare 'instance' axis."""
    from pimm_data.jaxtpc import JAXTPCDataset
    from pimm_data.testing import make_jaxtpc_sample
    root = make_jaxtpc_sample(str(tmp_path), n_events=2, n_volumes=2)
    cfg = [dict(out="instance_particle", scope="point", source="self")]
    ds = JAXTPCDataset(data_root=root, split="",
                       modalities=("edep", "hits", "labl"),
                       dataset_name="sim", label_config=cfg)
    sample = ds.get_data(0)
    for stream in ("edep", "hits"):
        s = sample[stream]
        assert "instance_particle" in s                 # not silently dropped
        assert s["instance_particle"].shape == (s["coord"].shape[0], 1)
        # 'self' is the resolved track id == the bare instance axis
        assert s["instance_particle"][:, 0].tolist() == \
            np.asarray(s["instance"]).ravel().tolist()


def test_jaxtpc_label_config_rejects_unsupported_specs(tmp_path):
    """F4: specs with no JAXTPC analog raise at construction instead of being
    silently dropped (the LUCiD/JAXTPC contract-divergence bug)."""
    from pimm_data.jaxtpc import JAXTPCDataset
    from pimm_data.testing import make_jaxtpc_sample
    root = make_jaxtpc_sample(str(tmp_path), n_events=2)
    kw = dict(data_root=root, split="", modalities=("edep", "labl"),
              dataset_name="sim")
    # event scope — no event-level labl table on JAXTPC
    with pytest.raises(ValueError, match="scope"):
        JAXTPCDataset(label_config=[dict(out="t", scope="event_broadcast",
                                         source=("event", "x"))], **kw)
    # ('particle', col) — LUCiD-only table
    with pytest.raises(ValueError, match="source"):
        JAXTPCDataset(label_config=[dict(out="t", scope="point",
                                         source=("particle", "category"))], **kw)
    # keyed_by other than track_ids
    with pytest.raises(ValueError, match="keyed_by"):
        JAXTPCDataset(label_config=[dict(out="t", scope="point",
                                         source=("track", "pdg"),
                                         keyed_by="group_id")], **kw)


def test_jaxtpc_label_config_default_off(tmp_path):
    from pimm_data.jaxtpc import JAXTPCDataset
    from pimm_data.testing import make_jaxtpc_sample
    root = make_jaxtpc_sample(str(tmp_path), n_events=2)
    ds = JAXTPCDataset(data_root=root, split="",
                       modalities=("edep", "labl"), dataset_name="sim")
    edep = ds.get_data(0)["edep"]
    assert "segment" in edep and "segment_pid" not in edep


def test_gather_empty_table_returns_fill_no_crash():
    """Consolidation: the shared gather guards an empty value table — the JAXTPC
    inline searchsorted it replaced crashed (clip to len-1 == -1 → IndexError)."""
    out = gather_with_fill(np.array([3, 7]), np.array([], dtype=np.int32),
                           keyed_by=np.array([], dtype=np.int32))
    assert out.tolist() == [-1, -1]
    # positional empty table too
    out2 = gather_with_fill(np.array([0, 1]), np.array([], dtype=np.int32))
    assert out2.tolist() == [-1, -1]


def test_gather_bool_and_unsigned_columns():
    """F3: bool/unsigned columns must not crash (uint OverflowError) or
    silently fill True (bool). Real category is uint8, contained is bool."""
    # unsigned uint8 (real category) with an out-of-range FK → fill -1, no crash
    cat = np.array([0, 2, 3], dtype=np.uint8)
    out = gather_with_fill(np.array([0, 5, 2]), cat, fill=-1)
    assert out.tolist() == [0, -1, 3]
    # bool (real contained): unresolved must be the -1 sentinel, not True
    contained = np.array([True, False, True])
    out2 = gather_with_fill(np.array([0, 9, 1]), contained, fill=-1)
    assert out2.tolist() == [1, -1, 0]           # not [True, True, False]
