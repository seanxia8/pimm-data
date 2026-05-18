"""Tests for LUCiDDataset on format_version=3 data (edep/sensor/hits/labl)."""

import numpy as np
import pytest

from pimm_data import LUCiDDataset


def make_ds(lucid_data_root, **kwargs):
    defaults = dict(data_root=lucid_data_root, split='', dataset_name='wc')
    defaults.update(kwargs)
    return LUCiDDataset(**defaults)


# ---------------------------------------------------------------------------
# Single-modality smoke tests
# ---------------------------------------------------------------------------

def test_sensor_only(lucid_data_root):
    """Sensor alone: sparse PMT point cloud, no labels."""
    ds = make_ds(lucid_data_root, modalities=('sensor',))
    d = ds.get_data(0)
    s = d['sensor']
    assert s['coord'].shape[1] == 3
    assert s['coord'].shape[0] == s['energy'].shape[0] == s['time'].shape[0]
    assert s['sensor_idx'].shape[0] == s['coord'].shape[0]
    assert 'segment' not in s and 'instance' not in s
    # No instance-bearing modality → no labl decoration possible
    assert 'hits' not in d and 'edep' not in d and 'labl' not in d


def test_edep_only(lucid_data_root):
    """Edep alone: raw geometry + physics, no decoration from labl."""
    ds = make_ds(lucid_data_root, modalities=('edep',))
    d = ds.get_data(0)
    seg = d['edep']
    assert seg['coord'].shape[1] == 3
    assert seg['energy'].shape[1] == 1
    assert 'track_idx' in seg
    assert seg['direction'].shape[1] == 3
    assert seg['beta_start'].shape[1] == 1
    assert seg['n_cherenkov'].shape[1] == 1
    # pdg moved to labl in v3 — regression guard
    assert 'pdg' not in seg
    assert 'segment' not in seg and 'instance' not in seg


def test_hits_only(lucid_data_root):
    """Hits alone: per-particle hit decomposition, particle_idx as instance.

    ``segment`` requires labl; without it we still expose particle_idx
    / instance."""
    ds = make_ds(lucid_data_root, modalities=('hits',))
    d = ds.get_data(0)
    inst = d['hits']
    assert inst['coord'].shape[1] == 3
    assert 'particle_idx' in inst
    assert 'instance' in inst
    assert np.array_equal(inst['instance'], inst['particle_idx'])
    # labl absent → segment cannot be computed
    assert 'segment' not in inst


# ---------------------------------------------------------------------------
# Invalid modality combinations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('mods', [(), ('labl',), ('sensor', 'labl'),
                                   ('nope',)])
def test_invalid_modalities(lucid_data_root, mods):
    with pytest.raises(ValueError):
        make_ds(lucid_data_root, modalities=mods)


# ---------------------------------------------------------------------------
# Labl decoration
# ---------------------------------------------------------------------------

def test_hits_plus_labl_labels(lucid_data_root):
    """Hits + labl: segment and instance populated at particle level."""
    ds = make_ds(lucid_data_root, modalities=('hits', 'labl'))
    d = ds.get_data(0)
    inst = d['hits']
    assert 'segment' in inst and 'instance' in inst
    assert inst['segment'].shape == inst['instance'].shape
    # inst has particle-level duplicates: >1 unique instance expected
    assert np.unique(inst['instance']).size > 1
    # segment values come from per_particle.category
    cats = d['labl']['particle']['category']
    expected = cats[inst['particle_idx']]
    assert np.array_equal(inst['segment'], expected)


def test_edep_plus_labl_labels(lucid_data_root):
    """Edep + labl: track_idx joins through per_track to particle category."""
    ds = make_ds(lucid_data_root, modalities=('edep', 'labl'))
    d = ds.get_data(0)
    seg = d['edep']
    assert 'particle_idx' in seg and 'instance' in seg and 'segment' in seg
    # instance alias of particle_idx
    assert np.array_equal(seg['instance'], seg['particle_idx'])
    # Cross-check the join by recomputing
    tpidx = d['labl']['track']['particle_idx']
    cats = d['labl']['particle']['category']
    expected_pidx = tpidx[seg['track_idx']]
    expected_seg = cats[expected_pidx]
    assert np.array_equal(seg['particle_idx'], expected_pidx)
    assert np.array_equal(seg['segment'], expected_seg)


def test_all_four_modalities(lucid_data_root):
    """Full multimodal load: all four sub-dicts present and consistent."""
    ds = make_ds(lucid_data_root,
                 modalities=('edep', 'sensor', 'hits', 'labl'))
    d = ds.get_data(0)
    assert set(d.keys()) >= {'edep', 'sensor', 'hits', 'labl',
                             'name', 'split'}
    # All modalities agree on the same particle_idx index space
    inst_pids = set(np.unique(d['hits']['particle_idx']).tolist())
    seg_pids = set(np.unique(d['edep']['particle_idx']).tolist())
    n_particles = d['labl']['particle']['category'].shape[0]
    assert inst_pids <= set(range(n_particles))
    assert seg_pids <= set(range(n_particles))


# ---------------------------------------------------------------------------
# Labl derived columns (ancestor reduction)
# ---------------------------------------------------------------------------

def test_labl_ancestor_columns_present(lucid_data_root):
    """Derived ancestor_particle_idx arrays ship with labl."""
    ds = make_ds(lucid_data_root, modalities=('hits', 'labl'))
    d = ds.get_data(0)
    labl = d['labl']
    pap = labl['particle']['ancestor_particle_idx']
    tap = labl['track']['ancestor_particle_idx']
    n_particles = labl['particle']['category'].shape[0]
    n_tracks = labl['track']['track_id'].shape[0]
    assert pap.shape == (n_particles,)
    assert tap.shape == (n_tracks,)
    # All ancestor particle_idx values must be valid particle indices
    assert pap.max() < n_particles
    assert tap.max() < n_particles
    # Primary particles are their own ancestor (self-ancestors must exist)
    primaries_p = np.where(pap == np.arange(n_particles))[0]
    assert primaries_p.size >= 1


def test_ancestor_remap_one_liner(lucid_data_root):
    """The documented ancestor remap should be a single lookup."""
    ds = make_ds(lucid_data_root,
                 modalities=('hits', 'edep', 'labl'))
    d = ds.get_data(0)

    # hits
    pap = d['labl']['particle']['ancestor_particle_idx']
    hits_anc = pap[d['hits']['particle_idx']]
    assert hits_anc.shape == d['hits']['instance'].shape
    # ancestor grouping must not be finer than particle grouping
    assert np.unique(hits_anc).size <= np.unique(d['hits']['instance']).size

    # edep
    tap = d['labl']['track']['ancestor_particle_idx']
    edep_anc = tap[d['edep']['track_idx']]
    assert edep_anc.shape == d['edep']['instance'].shape
    assert np.unique(edep_anc).size <= np.unique(d['edep']['instance']).size

    # hits and edep share the ancestor index space
    assert set(np.unique(hits_anc).tolist()) == set(np.unique(edep_anc).tolist())


# ---------------------------------------------------------------------------
# Dataset plumbing
# ---------------------------------------------------------------------------

def test_len_and_getitem(lucid_data_root):
    ds = make_ds(lucid_data_root, modalities=('sensor',))
    assert len(ds) > 0
    sample = ds[0]
    assert isinstance(sample, dict)
    assert 'sensor' in sample
    assert isinstance(sample['sensor']['coord'], np.ndarray)


def test_dataloader_workers(lucid_data_root):
    """Fork-safe via lazy h5py_worker_init()."""
    import torch
    ds = make_ds(lucid_data_root, modalities=('sensor',))
    if len(ds) < 2:
        pytest.skip("Need at least 2 events")
    loader = torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=2,
        collate_fn=lambda batch: batch)
    seen = 0
    for batch in loader:
        assert isinstance(batch[0], dict)
        assert 'sensor' in batch[0]
        seen += 1
        if seen >= 2:
            break
    assert seen >= 1


# ---------------------------------------------------------------------------
# Modality combination matrix — every valid subset decorates correctly
# ---------------------------------------------------------------------------

_VALID_COMBOS = [
    # singles (labl alone is invalid; covered separately)
    ('sensor',), ('edep',), ('hits',),
    # pairs without labl
    ('sensor', 'edep'), ('sensor', 'hits'), ('edep', 'hits'),
    # pairs with labl (sensor+labl is invalid; skip)
    ('edep', 'labl'), ('hits', 'labl'),
    # triples without labl
    ('sensor', 'edep', 'hits'),
    # triples with labl
    ('sensor', 'edep', 'labl'),
    ('sensor', 'hits', 'labl'),
    ('edep', 'hits', 'labl'),
    # all four
    ('sensor', 'edep', 'hits', 'labl'),
]


@pytest.mark.parametrize('mods', _VALID_COMBOS)
def test_modality_combo_loads(lucid_data_root, mods):
    """Every valid subset of modalities loads and produces the right sub-dicts."""
    ds = make_ds(lucid_data_root, modalities=mods)
    d = ds.get_data(0)
    for m in mods:
        assert m in d, f"missing modality {m} for combo {mods}"
    # Modalities not requested must not appear
    for m in ('sensor', 'edep', 'hits', 'labl'):
        if m not in mods:
            assert m not in d, f"unexpected modality {m} for combo {mods}"
    # If labl present with an instance-bearer, decoration must happen
    if 'labl' in mods and 'hits' in mods:
        assert 'segment' in d['hits'] and 'instance' in d['hits']
    if 'labl' in mods and 'edep' in mods:
        assert 'segment' in d['edep'] and 'instance' in d['edep']
    # If labl absent, no decoration anywhere
    if 'labl' not in mods:
        if 'hits' in d:
            assert 'segment' not in d['hits']
        if 'edep' in d:
            assert 'segment' not in d['edep']


# ---------------------------------------------------------------------------
# Full-event iteration — every event in the shard loads without error
# ---------------------------------------------------------------------------

def test_iterate_all_events_all_four(lucid_data_root):
    """Walk every event in the shard with all four modalities active."""
    ds = make_ds(lucid_data_root,
                 modalities=('edep', 'sensor', 'hits', 'labl'))
    assert len(ds) > 0
    n_particles_per_evt = []
    for i in range(len(ds)):
        d = ds.get_data(i)
        for m in ('sensor', 'edep', 'hits', 'labl'):
            assert m in d
        # instance IDs must stay within the event's particle table
        P = d['labl']['particle']['category'].shape[0]
        assert d['hits']['instance'].max() < P
        assert d['edep']['instance'].max() < P
        # ancestor reduction must not introduce out-of-range IDs
        pap = d['labl']['particle']['ancestor_particle_idx']
        assert pap.max() < P
        n_particles_per_evt.append(P)
    # Shard should have non-trivial variety in particle counts
    assert max(n_particles_per_evt) >= 2


# ---------------------------------------------------------------------------
# Reader kwargs surface
# ---------------------------------------------------------------------------

def test_edep_include_physics_false(lucid_data_root):
    """include_physics=False suppresses direction / beta_start / n_cherenkov."""
    ds = make_ds(lucid_data_root, modalities=('edep',),
                 include_physics=False)
    d = ds.get_data(0)
    seg = d['edep']
    for k in ('direction', 'beta_start', 'n_cherenkov'):
        assert k not in seg, f"{k} should be absent with include_physics=False"
    # Core fields still present
    for k in ('coord', 'energy', 'time', 'track_idx'):
        assert k in seg


def test_hits_pe_threshold(lucid_data_root):
    """pe_threshold drops low-PE entries consistently across keys."""
    d0 = make_ds(lucid_data_root, modalities=('hits',)).get_data(0)
    d1 = make_ds(lucid_data_root, modalities=('hits',),
                 pe_threshold=1.5).get_data(0)
    assert d1['hits']['coord'].shape[0] < d0['hits']['coord'].shape[0]
    # All retained PEs must exceed the threshold
    assert float(d1['hits']['energy'].min()) > 1.5
    # Length consistency across all per-row arrays
    n = d1['hits']['coord'].shape[0]
    for k in ('energy', 'time', 'sensor_idx', 'particle_idx', 'instance'):
        assert d1['hits'][k].shape[0] == n


def test_edep_min_segments_filter(lucid_data_root):
    """min_segments drops small events; remaining count is non-increasing."""
    full = make_ds(lucid_data_root, modalities=('edep',), min_segments=0)
    filtered = make_ds(lucid_data_root, modalities=('edep',),
                       min_segments=2000)
    assert len(filtered) <= len(full)


def test_pmt_coord_alignment_hits_vs_sensor(lucid_data_root):
    """hits and sensor must decode coord via the same PMT table."""
    ds = make_ds(lucid_data_root,
                 modalities=('sensor', 'hits'))
    d = ds.get_data(0)
    # Where sensor_idx matches, coords must match (hits is a decomposition
    # of sensor, not an independent geometry).
    sensor_map = {int(s): d['sensor']['coord'][i]
                  for i, s in enumerate(d['sensor']['sensor_idx'])}
    # Spot-check 10 hits rows
    for i in range(0, min(10, d['hits']['coord'].shape[0])):
        s = int(d['hits']['sensor_idx'][i])
        if s in sensor_map:
            np.testing.assert_array_equal(d['hits']['coord'][i],
                                          sensor_map[s])


def test_hits_alone_loads_geometry_from_own_config(lucid_data_root):
    """Hits without sensor must still decode coord via its own config."""
    ds = make_ds(lucid_data_root, modalities=('hits',))
    d = ds.get_data(0)
    assert d['hits']['coord'].shape[1] == 3
    # Non-trivial geometry (not all zeros / fallback)
    assert float(np.abs(d['hits']['coord']).max()) > 0.1
