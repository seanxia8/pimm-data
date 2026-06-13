"""Phase 5a — boundary helpers: to_batched_coords + split_event/batch[i]."""
import torch

from pimm_data.collate import collate_fn
from pimm_data.batch_ops import to_batched_coords, split_event


def _mk(n, edges, name):
    return {'step_coord': torch.arange(n * 3, dtype=torch.float32).reshape(n, 3),
            'step_offset': torch.tensor([n]),
            'step_edge_index': torch.tensor(edges),
            'target': torch.tensor([float(n)]),
            'name': name,
            '_roles': {'step_edge_index': ('edge', 'self')}}


def test_to_batched_coords_prepends_batch_id():
    b = collate_fn([_mk(3, [[0, 1], [1, 2]], 'a'), _mk(2, [[0], [1]], 'b')])
    xb = to_batched_coords(b, 'step')
    assert xb.shape == (5, 4)                       # [batch_id, x, y, z]
    assert xb[:, 0].tolist() == [0, 0, 0, 1, 1]     # batch ids from offset (no-leading-0)


def test_split_event_rebases_edges_and_slices():
    b = collate_fn([_mk(3, [[0, 1], [1, 2]], 'a'), _mk(2, [[0], [1]], 'b')])
    # collated edges: event1 shifted by 3 -> [[0,1,3],[1,2,4]]
    assert b['step_edge_index'].tolist() == [[0, 1, 3], [1, 2, 4]]

    ev1 = split_event(b, 1)
    assert ev1['step_coord'].shape[0] == 2                  # only event1's 2 points
    assert ev1['step_offset'].tolist() == [2]               # single-event offset
    assert ev1['step_edge_index'].tolist() == [[0], [1]]    # rebased back to 0
    assert ev1['target'].tolist() == [2.0]                  # event1's target
    assert ev1['name'] == 'b'                               # event1's name
    assert ev1['_roles']['step_edge_index'] == ('edge', 'self')


def test_split_event_zeroth():
    b = collate_fn([_mk(3, [[0, 1], [1, 2]], 'a'), _mk(2, [[0], [1]], 'b')])
    ev0 = split_event(b, 0)
    assert ev0['step_coord'].shape[0] == 3
    assert ev0['step_edge_index'].tolist() == [[0, 1], [1, 2]]   # unchanged (base 0)
    assert ev0['name'] == 'a'


def test_split_event_cross_store():
    def ev(nh, ns, edges):
        return {'hit_pos': torch.zeros(nh, 3), 'hit_offset': torch.tensor([nh]),
                'sp_pos': torch.zeros(ns, 3), 'sp_offset': torch.tensor([ns]),
                'nexus_edge_index': torch.tensor(edges),
                '_roles': {'nexus_edge_index': ('edge', ('hit', 'sp'))}}
    b = collate_fn([ev(3, 2, [[0, 2], [1, 0]]), ev(2, 1, [[1], [0]])])
    ev1 = split_event(b, 1)
    assert ev1['nexus_edge_index'].tolist() == [[1], [0]]   # rebased: hit-3, sp-2
    assert ev1['hit_pos'].shape[0] == 2 and ev1['sp_pos'].shape[0] == 1


# --- instance role: a part's SECOND row-space (REDESIGN §3) ---------------
def _mk_inst(n_pts, inst_idx, bbox, name):
    """A 'step' part with two row-spaces: per-point instance index (point role,
    sliced by step_offset) + a per-instance bbox (instance role, sliced by the
    distinct step_inst_offset)."""
    return {'step_coord': torch.arange(n_pts * 3, dtype=torch.float32).reshape(n_pts, 3),
            'step_offset': torch.tensor([n_pts]),
            'step_instance': torch.tensor(inst_idx),
            'step_bbox': bbox,
            'step_inst_offset': torch.tensor([bbox.shape[0]]),
            'name': name,
            '_roles': {'step_bbox': ('instance', 'step_inst_offset')}}


def test_instance_role_collate_and_split_uses_inst_offset():
    a = _mk_inst(3, [0, 0, 1], torch.tensor([[10., 11.], [12., 13.]]), 'a')  # 3 pts, 2 inst
    b = _mk_inst(2, [0, 0], torch.tensor([[20., 21.]]), 'b')                  # 2 pts, 1 inst
    batch = collate_fn([a, b])

    # two independent row-spaces survive collate, each its OWN cumulative offset
    assert batch['step_offset'].tolist() == [3, 5]              # points
    assert batch['step_inst_offset'].tolist() == [2, 3]         # instances
    assert batch['step_bbox'].shape == (3, 2)                   # 2+1 instance rows concat
    assert batch['step_instance'].tolist() == [0, 0, 1, 0, 0]   # per-point, per-event-local

    ev1 = split_event(batch, 1)
    # bbox sliced by the INSTANCE span (2,3) -> b's single row, NOT the point span (3,5)
    assert ev1['step_bbox'].tolist() == [[20., 21.]]
    assert ev1['step_inst_offset'].tolist() == [1]
    assert ev1['step_coord'].shape[0] == 2
    assert ev1['step_instance'].tolist() == [0, 0]              # point span, per-event-local
    assert ev1['step_offset'].tolist() == [2]

    ev0 = split_event(batch, 0)
    assert ev0['step_bbox'].tolist() == [[10., 11.], [12., 13.]]
    assert ev0['step_inst_offset'].tolist() == [2]
    assert ev0['step_instance'].tolist() == [0, 0, 1]
