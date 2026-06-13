"""Collect build-time validation (REDESIGN §9) — fail at construction, not epoch 1."""
import pytest

from pimm_data.transform import Collect


def test_namespaced_spec_requires_keys():
    with pytest.raises(ValueError, match="needs 'keys='"):
        Collect(modalities={'step': dict(feat_keys=('coord',))})   # no keys=


def test_roles_for_non_collected_key_rejected():
    with pytest.raises(ValueError, match="non-collected keys"):
        Collect(modalities={'step': dict(keys=('coord',),
                                          roles={'edge_index': ('edge', 'self')})})  # not in keys


def test_valid_namespaced_spec_ok():
    Collect(modalities={'step': dict(keys=('coord', 'edge_index'),
                                     roles={'edge_index': ('edge', 'self')})})


def test_rejects_both_forms():
    with pytest.raises(AssertionError):
        Collect(keys=['coord'], modalities={'step': dict(keys=['coord'])})
