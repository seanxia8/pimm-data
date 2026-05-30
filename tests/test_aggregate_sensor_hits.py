"""AggregateSensorHits: per-PMT aggregation of a LUCiD sensor event.

Replaces the inline LUCiDEventSSLDataset aggregation (D32 dissolve). Groups the
sensor stream by sensor_idx, sums PE, aggregates time per strategy, and lifts
the result to the top-level flat keys the event-SSL pipeline consumes.
"""
import numpy as np
import pytest

from pimm_data.detector_transforms import AggregateSensorHits


def _event():
    # PMT 5 hit twice, PMT 2 hit once (stable order: the two 5s keep input order)
    return {
        'sensor': {
            'coord': np.array([[1, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32),
            'energy': np.array([[2.0], [3.0], [7.0]], dtype=np.float32),
            'time': np.array([[10.0], [4.0], [9.0]], dtype=np.float32),
            'sensor_idx': np.array([5, 5, 2], dtype=np.int64),
        },
        'event_label': np.array([1], dtype=np.int64),
    }


def test_aggregate_lifts_to_top_level_and_sums_pe():
    out = AggregateSensorHits(time_aggregation='earliest')(_event())
    assert 'sensor' not in out                      # consumed sub-dict dropped
    # one point per PMT, sorted by sensor_idx (2, then 5)
    assert out['sensor_idx'].tolist() == [2, 5]
    assert out['coord'].tolist() == [[2, 0, 0], [1, 0, 0]]
    assert out['energy'][:, 0].tolist() == [7.0, 5.0]   # PMT5: 2+3 summed
    assert out['event_label'].tolist() == [1]           # top-level label kept


@pytest.mark.parametrize("strategy,pmt5_time", [
    ("earliest", 4.0),                      # min(10, 4)
    ("mean", 7.0),                          # (10 + 4) / 2
    ("pe_weighted", 6.4),                   # (10*2 + 4*3) / (2 + 3)
    ("first", 10.0),                        # time of the first hit (stable)
])
def test_time_aggregation_strategies(strategy, pmt5_time):
    out = AggregateSensorHits(time_aggregation=strategy)(_event())
    # row 1 is PMT 5 (sorted order); PMT 2 unchanged at 9.0
    assert out['time'][:, 0].tolist() == pytest.approx([9.0, pmt5_time])


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="time_aggregation"):
        AggregateSensorHits(time_aggregation="median")


def test_empty_event_is_noop_shape():
    data = {'sensor': {
        'coord': np.zeros((0, 3), np.float32), 'energy': np.zeros((0, 1), np.float32),
        'time': np.zeros((0, 1), np.float32), 'sensor_idx': np.zeros((0,), np.int64)}}
    out = AggregateSensorHits()(data)
    assert out['coord'].shape == (0, 3) and out['sensor_idx'].shape == (0,)


def test_operates_in_place_when_already_flat():
    flat = {k: v for k, v in _event()['sensor'].items()}
    out = AggregateSensorHits(stream='sensor')(flat)   # no 'sensor' key → flat
    assert out['sensor_idx'].tolist() == [2, 5]
