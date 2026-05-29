"""De-fork Step 1 (Part 01): merged transforms ported/generalized from the
research branch — RelativeLogNormalize, LogTransform.clip, generalized
GridSample reducers, the MultiViewGenerator.get_view guard, and the
index_operator prefix-match (D25).

Golden / invariant tests (stronger than cross-import parity for correctness).
"""
import numpy as np
import pytest

from pimm_data.transform import (
    TRANSFORMS, LogTransform, RelativeLogNormalize, GridSample,
    index_operator,
)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_relative_log_normalize_registered():
    assert TRANSFORMS.get("RelativeLogNormalize") is RelativeLogNormalize


# ---------------------------------------------------------------------------
# RelativeLogNormalize (D11/D31)
# ---------------------------------------------------------------------------

def test_relative_log_normalize_handles_negatives_no_nan():
    t = RelativeLogNormalize(keys=("time",), scale=50.0, max_val=4000.0)
    x = np.array([-240.0, 0.0, 100.0, 88000.0], dtype=np.float32)[:, None]
    out = t({"time": x})["time"]
    assert np.isfinite(out).all()
    assert out.min() >= -1.0 - 1e-6 and out.max() <= 1.0 + 1e-6
    # min subtracted → smallest input maps to out_min.
    assert out[0, 0] == pytest.approx(-1.0, abs=1e-6)
    # value beyond max_val saturates at out_max.
    assert out[3, 0] == pytest.approx(1.0, abs=1e-6)


def test_relative_log_normalize_golden():
    t = RelativeLogNormalize(keys=("time",), scale=50.0, max_val=4000.0,
                             out_min=-1.0, out_max=1.0)
    x = np.array([0.0, 50.0], dtype=np.float32)[:, None]
    out = t({"time": x})["time"]
    # x-min=0 → [0,50]; log1p(50/50)/log1p(4000/50) = ln(2)/ln(81).
    denom = np.log1p(4000.0 / 50.0)
    expected1 = -1.0 + 2.0 * (np.log1p(50.0 / 50.0) / denom)
    assert out[0, 0] == pytest.approx(-1.0, abs=1e-6)
    assert out[1, 0] == pytest.approx(expected1, abs=1e-5)


def test_relative_log_normalize_validates():
    with pytest.raises(ValueError):
        RelativeLogNormalize(scale=0)
    with pytest.raises(ValueError):
        RelativeLogNormalize(out_min=1.0, out_max=0.0)


# ---------------------------------------------------------------------------
# LogTransform.clip (D11/D31)
# ---------------------------------------------------------------------------

def test_log_transform_clip_clamps_domain():
    base = dict(min_val=0.01, max_val=20.0, keys=("energy",))
    e = np.array([1000.0], dtype=np.float32)[:, None]
    no_clip = LogTransform(**base, clip=False)({"energy": e.copy()})["energy"]
    clip = LogTransform(**base, clip=True)({"energy": e.copy()})["energy"]
    # clip → value clamped to max_val → maps to +1; unclipped overshoots +1.
    assert clip[0, 0] == pytest.approx(1.0, abs=1e-5)
    assert no_clip[0, 0] > 1.0


def test_log_transform_clip_default_off_unchanged():
    e = np.array([0.5], dtype=np.float32)[:, None]
    a = LogTransform(keys=("energy",))({"energy": e.copy()})["energy"]
    b = LogTransform(keys=("energy",), clip=False)({"energy": e.copy()})["energy"]
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# GridSample reducers (D29)
# ---------------------------------------------------------------------------

def _voxel_fixture():
    # pts 0,1 → voxel (0,0,0); pt 2 → voxel (5,0,0)
    coord = np.array([[0.1, 0, 0], [0.2, 0, 0], [5.0, 0, 0]], dtype=np.float32)
    energy = np.array([[2.0], [3.0], [7.0]], dtype=np.float32)
    time = np.array([[10.0], [4.0], [9.0]], dtype=np.float32)
    return coord, energy, time


def _by_voxel(out):
    """Map output grid_coord tuple → row index."""
    return {tuple(int(c) for c in gc): i
            for i, gc in enumerate(out["grid_coord"])}


@pytest.mark.parametrize("op,key,expect", [
    ("sum", "energy", {(0, 0, 0): 5.0, (5, 0, 0): 7.0}),
    ("min", "time", {(0, 0, 0): 4.0, (5, 0, 0): 9.0}),
    ("max", "energy", {(0, 0, 0): 3.0, (5, 0, 0): 7.0}),
    ("mean", "energy", {(0, 0, 0): 2.5, (5, 0, 0): 7.0}),
])
def test_gridsample_reducers(op, key, expect):
    coord, energy, time = _voxel_fixture()
    np.random.seed(0)
    gs = GridSample(grid_size=1.0, mode="train", return_grid_coord=True,
                    reducers={key: op})
    out = gs({"coord": coord.copy(), "energy": energy.copy(),
              "time": time.copy()})
    rows = _by_voxel(out)
    for gc, val in expect.items():
        assert out[key][rows[gc], 0] == pytest.approx(val)


def test_gridsample_first_is_deterministic_across_seeds():
    coord, energy, time = _voxel_fixture()
    res = []
    for seed in (0, 1, 7):
        np.random.seed(seed)
        gs = GridSample(grid_size=1.0, mode="train", return_grid_coord=True,
                        reducers={"energy": "first"})
        out = gs({"coord": coord.copy(), "energy": energy.copy()})
        rows = _by_voxel(out)
        res.append(out["energy"][rows[(0, 0, 0)], 0])
    # 'first' independent of the random survivor — same value every seed.
    assert len(set(res)) == 1


def test_gridsample_sum_keys_backcompat_matches_reducers():
    coord, energy, _ = _voxel_fixture()
    np.random.seed(0)
    a = GridSample(grid_size=1.0, mode="train", return_grid_coord=True,
                   sum_keys=("energy",))({"coord": coord.copy(),
                                          "energy": energy.copy()})
    np.random.seed(0)
    b = GridSample(grid_size=1.0, mode="train", return_grid_coord=True,
                   reducers={"energy": "sum"})({"coord": coord.copy(),
                                                "energy": energy.copy()})
    ra, rb = _by_voxel(a), _by_voxel(b)
    for gc in ra:
        assert a["energy"][ra[gc], 0] == pytest.approx(b["energy"][rb[gc], 0])


def test_gridsample_unknown_reducer_raises():
    with pytest.raises(ValueError):
        GridSample(grid_size=1.0, reducers={"energy": "median"})


# ---------------------------------------------------------------------------
# index_operator prefix-match (D25)
# ---------------------------------------------------------------------------

def test_index_operator_carries_schema_and_fk_keys():
    n = 5
    data = {
        "coord": np.arange(n * 3).reshape(n, 3).astype(np.float32),
        "segment_pid": np.arange(n, dtype=np.int32)[:, None],
        "instance_particle": np.arange(n, dtype=np.int32)[:, None],
        "particle_idx": np.arange(n, dtype=np.int32),
        "sensor_idx": np.arange(n, dtype=np.int32),
        # per-event target (length != n_points) must NOT be point-subset.
        "target_vertex": np.array([1.0, 2.0, 3.0], dtype=np.float32),
    }
    idx = np.array([0, 2, 4])
    out = index_operator(data, idx)
    assert out["coord"].shape[0] == 3
    assert out["segment_pid"].tolist() == [[0], [2], [4]]
    assert out["instance_particle"].tolist() == [[0], [2], [4]]
    assert out["particle_idx"].tolist() == [0, 2, 4]
    assert out["sensor_idx"].tolist() == [0, 2, 4]
    # per-event target untouched (still length 3, original values).
    assert out["target_vertex"].tolist() == [1.0, 2.0, 3.0]


def test_gridsample_preserves_schema_keys_through_subsample():
    coord, energy, _ = _voxel_fixture()
    seg = np.array([[1], [1], [2]], dtype=np.int32)   # segment_pid per point
    np.random.seed(0)
    gs = GridSample(grid_size=1.0, mode="train", return_grid_coord=True)
    out = gs({"coord": coord.copy(), "energy": energy.copy(),
              "segment_pid": seg.copy()})
    # 2 voxels survive; segment_pid subsetted to match (not stale length 3).
    assert out["segment_pid"].shape[0] == out["coord"].shape[0] == 2
