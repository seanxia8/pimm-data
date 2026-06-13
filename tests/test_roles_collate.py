"""Phase 1 — role-driven collate (flat-prefixed parts + _roles).

Locks each role's batching: point/raw concat, offset cumsum (B,), edge shift
(self + cross-store), label compact+distinct-renumber, event stack, and that a
batch with no _roles is untouched (legacy path).
"""
import torch

from pimm_data.collate import collate_fn, collate_with_roles
from pimm_data import _roles


def test_point_and_offset_no_leading_zero():
    s0 = {'step_coord': torch.zeros(3, 3), 'step_offset': torch.tensor([3]), '_roles': {}}
    s1 = {'step_coord': torch.zeros(2, 3), 'step_offset': torch.tensor([2]), '_roles': {}}
    b = collate_fn([s0, s1])
    assert b['step_coord'].shape[0] == 5
    assert b['step_offset'].tolist() == [3, 5]          # (B,), NO leading 0


def test_edge_self_shift():
    s0 = {'step_coord': torch.zeros(3, 3), 'step_offset': torch.tensor([3]),
          'step_edge_index': torch.tensor([[0, 1], [1, 2]]),
          '_roles': {'step_edge_index': ('edge', 'self')}}
    s1 = {'step_coord': torch.zeros(2, 3), 'step_offset': torch.tensor([2]),
          'step_edge_index': torch.tensor([[0], [1]]),
          '_roles': {'step_edge_index': ('edge', 'self')}}
    b = collate_fn([s0, s1])
    # event1's edge shifted by 3 (its node base)
    assert b['step_edge_index'].tolist() == [[0, 1, 3], [1, 2, 4]]
    assert b['step_offset'].tolist() == [3, 5]


def test_edge_cross_store_shift():
    def ev(nh, ns, edges):
        return {'hit_pos': torch.zeros(nh, 3), 'hit_offset': torch.tensor([nh]),
                'sp_pos': torch.zeros(ns, 3),  'sp_offset': torch.tensor([ns]),
                'nexus_edge_index': torch.tensor(edges),
                '_roles': {'nexus_edge_index': ('edge', ('hit', 'sp'))}}
    b = collate_fn([ev(3, 2, [[0, 2], [1, 0]]), ev(2, 1, [[1], [0]])])
    # row0 (hit): ev1 '1' -> 1+3=4 ; row1 (sp): ev1 '0' -> 0+2=2
    assert b['nexus_edge_index'].tolist() == [[0, 2, 4], [1, 0, 2]]
    assert b['hit_offset'].tolist() == [3, 5] and b['sp_offset'].tolist() == [2, 3]


def test_label_compact_distinct_renumber_preserves_hierarchy():
    s0 = {'cl_coord': torch.zeros(3, 3), 'cl_offset': torch.tensor([3]),
          'cl_cluster_id': torch.tensor([0, 0, 1]), 'cl_group_id': torch.tensor([0, 0, 0]),
          '_roles': {'cl_cluster_id': ('label', 'g'), 'cl_group_id': ('label', 'g')}}
    s1 = {'cl_coord': torch.zeros(2, 3), 'cl_offset': torch.tensor([2]),
          'cl_cluster_id': torch.tensor([5, 7]), 'cl_group_id': torch.tensor([5, 9]),  # raw, non-dense
          '_roles': {'cl_cluster_id': ('label', 'g'), 'cl_group_id': ('label', 'g')}}
    b = collate_fn([s0, s1])
    cid, gid = b['cl_cluster_id'].tolist(), b['cl_group_id'].tolist()
    assert cid == [0, 0, 1, 2, 3]            # compacted + running distinct-count
    assert gid == [0, 0, 0, 1, 2]
    # hierarchy: each cluster maps to exactly one group
    mapping = {}
    for c, g in zip(cid, gid):
        assert mapping.setdefault(c, g) == g


def test_event_stack_and_unprefixed():
    s0 = {'step_coord': torch.zeros(3, 3), 'step_offset': torch.tensor([3]),
          'step_count': torch.tensor([3]), 'target': torch.tensor([1., 2., 3.]),
          'name': 'e0', '_roles': {'step_count': 'event'}}
    s1 = {'step_coord': torch.zeros(2, 3), 'step_offset': torch.tensor([2]),
          'step_count': torch.tensor([2]), 'target': torch.tensor([4., 5., 6.]),
          'name': 'e1', '_roles': {'step_count': 'event'}}
    b = collate_fn([s0, s1])
    assert b['step_count'].tolist() == [[3], [2]]        # prefixed event -> stack (B,1)
    assert b['target'].shape == (2, 3)                   # unprefixed -> stack (B,3)
    assert b['name'] == ['e0', 'e1']                     # unprefixed non-tensor -> list


def test_roles_carried_through():
    s = {'step_coord': torch.zeros(2, 3), 'step_offset': torch.tensor([2]),
         'step_edge_index': torch.tensor([[0], [1]]),
         '_roles': {'step_edge_index': ('edge', 'self')}}
    b = collate_fn([s, s])
    assert '_roles' in b and b['_roles']['step_edge_index'] == ('edge', 'self')


def test_no_roles_uses_legacy_path():
    # a Mapping without '_roles' must hit the legacy collate unchanged
    s0 = {'coord': torch.zeros(3, 3), 'offset': torch.tensor([3])}
    s1 = {'coord': torch.zeros(2, 3), 'offset': torch.tensor([2])}
    b = collate_fn([s0, s1])
    assert '_roles' not in b
    assert b['offset'].tolist() == [3, 5]


def test_offset_to_batch_helper():
    assert _roles.offset_to_batch(torch.tensor([3, 5])).tolist() == [0, 0, 0, 1, 1]
    assert _roles.node_bases(torch.tensor([3, 5])).tolist() == [0, 3]
