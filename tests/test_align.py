"""Align — reorder parts row-for-row to a reference (multi-task voxel alignment)."""
import numpy as np
import pytest

from pimm_data.transform import TRANSFORMS


def test_align_reorders_to_reference():
    coords = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], np.float32)
    perm = [2, 0, 1]                                   # cluster rows in a different order
    d = {'image':   {'coord': coords.copy(), 'value': np.array([10., 20., 30.])},
         'cluster': {'coord': coords[perm].copy(),
                     'cluster_id': np.array([2, 0, 1])}}   # aligned to its own (shuffled) coord
    out = TRANSFORMS.build(dict(type='Align', to='image', parts=('cluster',)))(d)
    # after Align, cluster.coord == image.coord row-for-row
    assert np.array_equal(out['cluster']['coord'], coords)
    # and cluster_id rode the same permutation -> row i lines up with image row i
    assert out['cluster']['cluster_id'].tolist() == [0, 1, 2]


def test_align_rejects_mismatched_coords():
    d = {'image':   {'coord': np.array([[0, 0, 0]], np.float32)},
         'cluster': {'coord': np.array([[9, 9, 9]], np.float32)}}
    with pytest.raises(ValueError, match="coord sets differ|not found"):
        TRANSFORMS.build(dict(type='Align', to='image', parts=('cluster',)))(d)
