"""v3 vertex/is_primary plumbing (Part 01): a per-point ``vertex`` label must
co-transform with ``coord`` through every geometric augmentation, and the
(-1,-1,-1) missing-vertex sentinel must stay untouched.

Invariant used: seed ``vertex = coord.copy()`` (each vertex row equals its
point). Any affine/linear geometric op applies the SAME map to coord and
vertex, so post-transform ``vertex[i] == coord[i]`` for non-sentinel rows.
"""
import random

import numpy as np
import pytest

from pimm_data.transform import (
    NormalizeCoord, PositiveShift, CenterShift, RandomShift, RandomRotate,
    RandomRotateTargetAngle, RandomScale, RandomFlip, PointClip,
    ConditionalRandomTransform, MixedScaleGeometryMultiViewGenerator,
    index_operator,
)

_SENTINEL = np.array([-1.0, -1.0, -1.0], dtype=np.float32)


def _seed(s=0):
    random.seed(s)
    np.random.seed(s)


def _coord_vertex(n=20):
    rng = np.random.default_rng(0)
    coord = rng.uniform(-0.8, 0.8, size=(n, 3)).astype(np.float32)
    vertex = coord.copy()
    vertex[0] = _SENTINEL          # one missing-vertex sentinel
    return coord, vertex


# Transforms that always apply (or forced via p=1) and are affine in coord.
_AFFINE = [
    lambda: NormalizeCoord(),
    lambda: NormalizeCoord(center=[0.1, 0.2, 0.3], scale=2.0),
    lambda: PositiveShift(),
    lambda: CenterShift(),
    lambda: RandomShift(shift=((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2))),
    lambda: RandomRotate(axis="z", p=1.0),
    lambda: RandomRotate(axis="x", p=1.0),
    lambda: RandomRotateTargetAngle(axis="z", p=1.0),
    lambda: RandomScale(),
    lambda: RandomScale(anisotropic=True),
    lambda: RandomFlip(p=1.0, axes=("x", "y", "z")),
]


@pytest.mark.parametrize("make", _AFFINE)
def test_vertex_co_transforms_with_coord(make):
    _seed(1)
    coord, vertex = _coord_vertex()
    out = make()({"coord": coord.copy(), "vertex": vertex.copy()})
    # non-sentinel vertices track their coord; sentinel untouched.
    np.testing.assert_allclose(out["vertex"][1:], out["coord"][1:],
                               rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(out["vertex"][0], _SENTINEL)


def test_conditional_random_transform_vertex():
    _seed(0)
    coord = np.full((10, 3), 0.0, dtype=np.float32)
    coord[:, 0] = np.linspace(-0.99, -0.9, 10)   # near the lower wall → triggers
    vertex = coord.copy()
    vertex[0] = _SENTINEL
    t = ConditionalRandomTransform(p=1.0, axes=("x",),
                                   bounds=((-1, 1), (-1, 1), (-1, 1)))
    out = t({"coord": coord.copy(), "vertex": vertex.copy()})
    np.testing.assert_allclose(out["vertex"][1:], out["coord"][1:],
                               rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(out["vertex"][0], _SENTINEL)


def test_point_clip_leaves_vertex_untouched():
    coord, vertex = _coord_vertex()
    v0 = vertex.copy()
    out = PointClip(point_cloud_range=(-0.5, -0.5, -0.5, 0.5, 0.5, 0.5))(
        {"coord": coord.copy(), "vertex": vertex.copy()})
    # PointClip is intentionally NOT vertex-aware (vertices may lie outside).
    np.testing.assert_array_equal(out["vertex"], v0)


def test_no_vertex_is_noop():
    """Transforms without a 'vertex' key behave exactly as before."""
    _seed(2)
    coord = np.random.default_rng(0).uniform(-1, 1, size=(8, 3)).astype(np.float32)
    out = RandomRotate(axis="z", p=1.0)({"coord": coord.copy()})
    assert "vertex" not in out
    assert out["coord"].shape == (8, 3)


def test_index_operator_carries_vertex_and_is_primary():
    n = 6
    data = {
        "coord": np.zeros((n, 3), dtype=np.float32),
        "vertex": np.arange(n * 3).reshape(n, 3).astype(np.float32),
        "is_primary": np.arange(n, dtype=np.int32)[:, None],
        "revision": "v3",
    }
    out = index_operator(data, np.array([0, 2, 4]))
    assert out["vertex"].shape == (3, 3)
    assert out["vertex"][:, 0].tolist() == [0.0, 6.0, 12.0]
    assert out["is_primary"][:, 0].tolist() == [0, 2, 4]


def test_mixed_scale_multiview_smoke():
    _seed(3)
    rng = np.random.default_rng(0)
    n = 400
    coord = rng.uniform(-1, 1, size=(n, 3)).astype(np.float32)
    energy = rng.uniform(0, 1, size=(n, 1)).astype(np.float32)
    gen = MixedScaleGeometryMultiViewGenerator(
        fine_local_view_num=2,
        fine_local_view_scale=(0.05, 0.1),
        fine_center_mode="geometry",
        global_view_num=2,
        global_view_scale=(0.5, 1.0),
        local_view_num=4,
        local_view_scale=(0.2, 0.4),
        view_keys=("coord", "energy"),
        max_size=n,
    )
    out = gen({"coord": coord.copy(), "energy": energy.copy()})
    assert "global_coord" in out and "local_coord" in out
    assert "global_offset" in out and "local_offset" in out
    assert out["global_offset"][-1] == out["global_coord"].shape[0]
    assert out["local_offset"][-1] == out["local_coord"].shape[0]
