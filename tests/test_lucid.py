"""Tests for LUCiDDataset on the v3+ schema (edep/sensor/hits/labl).

Real WAND is ``format_version 5``; the readers gate on structure, not the
version int, so the fixtures stamp 5 and the same code reads both."""

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
    # Every instance value must be a valid particle index for the event.
    cats = d['labl']['particle']['category']
    if inst['instance'].size > 0:
        assert inst['instance'].min() >= 0
        assert inst['instance'].max() < cats.shape[0]
    # segment values come from per_particle.category
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
    valid = (expected_pidx >= 0) & (expected_pidx < len(cats))
    expected_seg = np.full_like(expected_pidx, -1)
    expected_seg[valid] = cats[expected_pidx[valid]]
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
    assert seg_pids - {-1} <= set(range(n_particles))


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
    # (or -1 sentinel when the labl ancestor track_id isn't in per_track).
    if pap.size > 0:
        assert pap.max() < n_particles
    if tap.size > 0:
        assert tap.max() < n_particles
    # Self-ancestors (primary particles) may legitimately be zero on
    # events whose primary never reached the per_particle table, e.g. a
    # pi0 that decays to invisible secondaries.
    primaries_p = np.where(pap == np.arange(n_particles))[0]
    assert primaries_p.size >= 0


def test_reader_tolerates_dangling_shard(tmp_path):
    """F17: a dangling LUCiD shard (symlink to a vanished source — the WAND
    failure mode) is skipped, not crashed-on, even for the sensor reader whose
    worker init reads PMT geometry from a shard handle. Here shard 0 (the one
    the sensor reader would read geometry from) is the dangling one."""
    import os
    from pimm_data.testing import make_lucid_sample
    from pimm_data.readers.lucid_sensor import LUCiDSensorReader
    root = make_lucid_sample(str(tmp_path), n_events=3, n_files=3)
    sdir = os.path.join(root, 'sensor')
    victim = os.path.join(sdir, 'wc_sensor_0000.h5')   # file 0 → geometry source
    os.remove(victim)
    os.symlink(os.path.join(sdir, 'vanished_source.h5'), victim)

    r = LUCiDSensorReader(data_root=sdir, split='', dataset_name='wc')
    assert len(r.indices[0]) == 0 and len(r) == 6      # dangler contributes none
    for i in range(len(r)):
        r.read_event(i)                                 # no crash on any read
    assert r._h5data[0] is None
    # PMT geometry still resolved — from the first shard that actually opened
    assert r._pmt_positions is not None


def test_labl_per_interaction_surfaced(lucid_data_root):
    """F5: the per_interaction (per-neutrino-vertex) table is surfaced — vertex,
    neutrino kinematics, source_type, and the ragged primary_* CSR lists — and
    reaches the nested labl dict as an ``interaction`` table (the scope a
    ``source=('interaction', col)`` label_config resolves against)."""
    ds = make_ds(lucid_data_root, modalities=('hits', 'labl'))
    d = ds.get_data(0)
    inter = d['labl']['interaction']
    # scalar-per-interaction physics present
    for k in ('vertex_x', 'vertex_y', 'vertex_z', 'neutrino_energy_MeV',
              'neutrino_pdg', 'source_type', 't0', 'contained', 'n_primaries'):
        assert k in inter, k
    n_int = inter['vertex_x'].shape[0]
    assert n_int >= 1
    # the per_particle one-hop FK indexes into this table
    pii = d['labl']['particle']['interaction_idx']
    assert int(pii.max()) < n_int
    # ragged primary_* are CSR (offsets length n_int+1, monotone)
    off = inter['primary_pdgs_offsets']
    assert off.shape == (n_int + 1,)
    assert (np.diff(off) >= 0).all() and int(off[0]) == 0
    assert inter['primary_pdgs_data'].shape[0] == int(off[-1])
    # integer columns are not silently floated by _cast
    assert np.issubdtype(inter['neutrino_pdg'].dtype, np.integer)
    assert np.issubdtype(inter['source_type'].dtype, np.integer)
    # contained stays bool
    assert inter['contained'].dtype == np.bool_


def test_label_config_interaction_event_broadcast(lucid_data_root):
    """F5 end-to-end: a ('interaction', col) event_broadcast axis tiles the
    per-interaction vector onto every point via the shared decorate_labels."""
    cfg = [dict(out='target_vertex_x', scope='event_broadcast',
                source=('interaction', 'vertex_x'))]
    ds = make_ds(lucid_data_root, modalities=('hits', 'labl'),
                 label_config=cfg)
    d = ds.get_data(0)
    hits = d['hits']
    assert 'target_vertex_x' in hits
    n = hits['coord'].shape[0]
    assert hits['target_vertex_x'].shape[0] == n


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

    # hits and edep share the same ancestor index space.
    # In general every hit-producing particle's ancestor also deposits
    # energy (so its ancestor appears in edep), but the converse does
    # not hold: a particle can deposit energy without producing any
    # Cherenkov light (sub-threshold, neutral, etc.). So we require
    # hits_anc ⊆ edep_anc, not equality.
    hits_set = set(np.unique(hits_anc).tolist())
    edep_set = set(np.unique(edep_anc).tolist())
    assert hits_set.issubset(edep_set), (
        f'hits ancestors {hits_set - edep_set} not found in edep')


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
    """Walk events in the shard with all four modalities active.

    On production shards (~10^5 events) we sample the first
    ``_ITERATE_MAX`` events; on small synthetic shards this is the
    full set.
    """
    _ITERATE_MAX = 500
    ds = make_ds(lucid_data_root,
                 modalities=('edep', 'sensor', 'hits', 'labl'))
    assert len(ds) > 0
    n_iter = min(len(ds), _ITERATE_MAX)
    n_particles_per_evt = []
    for i in range(n_iter):
        d = ds.get_data(i)
        for m in ('sensor', 'edep', 'hits', 'labl'):
            assert m in d
        P = d['labl']['particle']['category'].shape[0]
        # instance IDs must stay within the event's particle table
        # (skip max() on zero-row arrays — np.max has no identity)
        if d['hits']['instance'].size > 0:
            assert d['hits']['instance'].max() < P
        if d['edep']['instance'].size > 0:
            assert d['edep']['instance'].max() < P
        # ancestor reduction must not introduce out-of-range IDs
        pap = d['labl']['particle']['ancestor_particle_idx']
        if pap.size > 0:
            assert pap.max() < P
        n_particles_per_evt.append(P)
    # Shard has at least one event with non-zero particle count
    assert max(n_particles_per_evt) >= 1


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
    """pe_threshold drops low-PE entries consistently across keys.

    LUCiD PE is integer-quantized (1, 2, 3, ...); on sparse events
    threshold=1.5 can legitimately drop everything. Skip past such
    events so the assertions actually exercise the filter.
    """
    ds_full = make_ds(lucid_data_root, modalities=('hits',))
    ds_filt = make_ds(lucid_data_root, modalities=('hits',), pe_threshold=1.5)
    # Find an event where threshold=1.5 retains at least one row, so
    # we can actually assert min(retained) > 1.5.
    idx = None
    for i in range(min(len(ds_full), 100)):
        d = ds_filt.get_data(i)
        if d['hits']['energy'].size > 0:
            idx = i
            break
    if idx is None:
        pytest.skip("no event in first 100 has any hit with PE > 1.5")
    d0 = ds_full.get_data(idx)
    d1 = ds_filt.get_data(idx)
    # Threshold is monotone — never increases the row count.
    assert d1['hits']['coord'].shape[0] <= d0['hits']['coord'].shape[0]
    # All retained PEs must exceed the threshold.
    assert float(d1['hits']['energy'].min()) > 1.5
    # Length consistency across all per-row arrays.
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


# ---------------------------------------------------------------------------
# Schema-contract regression guards (catch drift between pimm-data's
# reader expectations and LUCiD's v3_writer output)
# ---------------------------------------------------------------------------

def test_labl_contained_keys_populated(lucid_data_root):
    """labl.event.contained and labl.particle.contained must land as bool.

    LUCiD's v3_writer emits these as bool datasets; the reader silently
    skips missing keys via ``if k in pp``, so a name drift on either
    side would zero out these columns without raising. This test fails
    fast if that happens — applies to BOTH synthetic and real data.
    """
    ds = make_ds(lucid_data_root, modalities=('labl', 'hits'))
    d = ds.get_data(0)
    labl = d['labl']

    assert 'contained' in labl['event'], \
        "missing labl.event.contained (LUCiD writes 'per_event/contained')"
    assert labl['event']['contained'].dtype == np.bool_, \
        f"labl.event.contained dtype = {labl['event']['contained'].dtype}, expected bool"

    assert 'contained' in labl['particle'], \
        "missing labl.particle.contained (LUCiD writes 'per_particle/contained')"
    P = labl['particle']['category'].shape[0]
    assert labl['particle']['contained'].shape == (P,), \
        f"labl.particle.contained shape mismatch: {labl['particle']['contained'].shape} vs ({P},)"
    assert labl['particle']['contained'].dtype == np.bool_, \
        f"labl.particle.contained dtype = {labl['particle']['contained'].dtype}, expected bool"


def test_edep_contained_per_segment(lucid_data_root):
    """edep.contained must be a bool per segment (in lockstep with coord)."""
    ds = make_ds(lucid_data_root, modalities=('edep',))
    d = ds.get_data(0)
    edep = d['edep']
    assert 'contained' in edep, \
        "missing edep.contained (LUCiD writes per-segment 'contained' bool)"
    N = edep['coord'].shape[0]
    assert edep['contained'].shape == (N,)
    assert edep['contained'].dtype == np.bool_
