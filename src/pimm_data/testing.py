"""
Synthetic v3 fixture generators for JAXTPC and LUCiD.

The readers in :mod:`pimm_data.readers` expect a specific on-disk HDF5
layout (four modalities per dataset: ``edep/``, ``sensor/``, ``hits/``,
``labl/``). Real fixtures are produced by the JAXTPC and LUCiD
simulation pipelines, but those take GPU-bound physics runs and
gigabytes of edepsim input — overkill for exercising the reader /
dataset plumbing.

This module builds minimal schema-conformant fixtures from pure
numpy + h5py, with no dependency on JAX, edepsim, Geant4, or any
production code. The generated files are tiny (a few KB each) and
satisfy every cross-modality invariant the readers and dataset layer
rely on:

* **JAXTPC.** ``hits.deposit_to_group`` indexes into
  ``hits.group_to_track``; ``labl.deposit_to_track[i] ==
  hits.group_to_track[hits.deposit_to_group[i]]``; every per-deposit
  track id appears in ``labl.track_ids``; per-plane CSR entries
  decode to the declared ``n_pixels``.
* **LUCiD.** Every ``edep.track_idx`` appears in
  ``labl.per_track.track_id``; every ``hits.particle_idx`` and
  ``labl.per_track.particle_idx`` is a valid index into the per-
  particle tables; every ``labl.per_track.ancestor`` is itself a
  ``track_id`` in ``per_track``.

Usage::

    from pimm_data.testing import make_jaxtpc_sample, make_lucid_sample
    make_jaxtpc_sample('/tmp/jaxtpc_synth', dataset_name='sim', n_events=2)
    make_lucid_sample('/tmp/lucid_synth',   dataset_name='wc',  n_events=2)

Both functions are idempotent given the same seed.
"""

from __future__ import annotations

import os

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# JAXTPC
# ---------------------------------------------------------------------------

_JAXTPC_PLANES = ('U', 'V', 'Y')
_JAXTPC_PDG_POOL = np.array([13, 11, 211, 22, 2212], dtype=np.int32)
_JAXTPC_POS_STEP_MM = 0.05
_JAXTPC_POS_ORIGIN = (-2160.0, -2160.0, -2160.0)


def make_jaxtpc_sample(outdir, dataset_name='sim', n_events=2, n_files=1,
                       n_volumes=2, n_deposits=60, n_groups=6, n_tracks=6,
                       n_pixels_per_plane=40, readout_type='wire', seed=0):
    """Write a minimal schema-conformant JAXTPC v3 dataset.

    Creates ``{outdir}/{edep,sensor,hits,labl}/{dataset_name}_{mod}_NNNN.h5``.

    ``readout_type`` is ``'wire'`` (three U/V/Y planes per volume) or
    ``'pixel'`` (single ``Pixel`` plane per volume; coord adds a pz axis).
    """
    assert readout_type in ('wire', 'pixel'), readout_type
    os.makedirs(outdir, exist_ok=True)
    for mod in ('edep', 'sensor', 'hits', 'labl'):
        os.makedirs(os.path.join(outdir, mod), exist_ok=True)

    rng = np.random.default_rng(seed)

    for file_idx in range(n_files):
        events = [
            _build_jaxtpc_event(rng, n_volumes, n_deposits, n_groups,
                                n_tracks, n_pixels_per_plane, readout_type)
            for _ in range(n_events)
        ]
        tag = f'{dataset_name}_{{mod}}_{file_idx:04d}.h5'
        _write_jaxtpc_edep(os.path.join(outdir, 'edep',
                                        tag.format(mod='edep')), events)
        _write_jaxtpc_sensor(os.path.join(outdir, 'sensor',
                                          tag.format(mod='sensor')), events,
                             readout_type)
        _write_jaxtpc_hits(os.path.join(outdir, 'hits',
                                        tag.format(mod='hits')), events,
                           readout_type)
        _write_jaxtpc_labl(os.path.join(outdir, 'labl',
                                        tag.format(mod='labl')), events)

    return outdir


def _build_jaxtpc_event(rng, n_volumes, n_deposits, n_groups, n_tracks,
                        n_pixels_per_plane, readout_type='wire'):
    """Draw a coherent set of volumes with consistent FKs."""
    plane_names = ('Pixel',) if readout_type == 'pixel' else _JAXTPC_PLANES
    volumes = []
    for v in range(n_volumes):
        track_ids = np.sort(rng.choice(
            np.arange(1, 10 * n_tracks + 1), size=n_tracks,
            replace=False)).astype(np.int32)
        # Deterministic cycle guarantees every PDG in the pool appears when
        # n_tracks >= len(pool); downstream tests rely on PDG variety and
        # on at least one value > 20 to distinguish raw from remapped.
        track_pdg = _JAXTPC_PDG_POOL[
            np.arange(n_tracks) % len(_JAXTPC_PDG_POOL)].astype(np.int32)
        # Each group belongs to exactly one track
        group_to_track = track_ids[rng.integers(0, n_tracks, size=n_groups)
                                   ].astype(np.int32)

        # Each deposit belongs to exactly one group
        deposit_to_group = rng.integers(0, n_groups, size=n_deposits).astype(
            np.int32)
        deposit_to_track = group_to_track[deposit_to_group].astype(np.int32)

        # Deposit geometry — uint16 positions scaled by pos_step_mm
        # Keep positions well inside uint16 range and the detector fiducial.
        positions = rng.integers(1000, 60000, size=(n_deposits, 3),
                                 dtype=np.uint16)
        de = rng.uniform(0.05, 2.0, size=n_deposits).astype(np.float16)
        dx = rng.uniform(0.1, 1.0, size=n_deposits).astype(np.float16)
        theta = rng.uniform(-np.pi, np.pi, size=n_deposits).astype(np.float16)
        phi = rng.uniform(-np.pi, np.pi, size=n_deposits).astype(np.float16)
        t0_us = rng.uniform(0.0, 10.0, size=n_deposits).astype(np.float16)
        charge = rng.uniform(1e3, 5e4, size=n_deposits).astype(np.float16)
        photons = rng.uniform(1e3, 5e4, size=n_deposits).astype(np.float16)
        qs_fractions = rng.uniform(0.01, 1.0, size=n_deposits).astype(
            np.float32)

        planes = {}
        for plane in plane_names:
            planes[plane] = _build_jaxtpc_plane(
                rng, n_groups, n_pixels_per_plane, readout_type)

        volumes.append(dict(
            vol_idx=v,
            positions=positions, de=de, dx=dx, theta=theta, phi=phi,
            t0_us=t0_us, charge=charge, photons=photons,
            qs_fractions=qs_fractions,
            deposit_to_group=deposit_to_group,
            deposit_to_track=deposit_to_track,
            group_to_track=group_to_track,
            track_ids=track_ids,
            track_pdg=track_pdg,
            planes=planes,
        ))
    return volumes


def _build_jaxtpc_plane(rng, n_groups, n_pixels_per_plane, readout_type='wire'):
    """One plane: CSR-packed per-group pixel entries + delta-encoded sparse.

    For pixel readout, coord adds a second spatial axis (py/pz) in both
    the CSR centers/deltas and the sensor sparse stream.
    """
    # CSR: split n_pixels across n_groups with random but ≥1 group_size,
    # capped at uint8 max.
    base = np.ones(n_groups, dtype=np.int32)
    remainder = n_pixels_per_plane - n_groups
    if remainder < 0:
        # Fewer pixels than groups — trim groups logically by zero-sizing
        # the tail (keep n_groups rows so lengths match group_to_track).
        group_sizes = base.copy()
        group_sizes[n_pixels_per_plane:] = 0
    else:
        extra = rng.multinomial(remainder, np.ones(n_groups) / n_groups)
        group_sizes = (base + extra).astype(np.uint8)
    total = int(group_sizes.sum())

    group_ids = np.arange(n_groups, dtype=np.int32)
    center_times = rng.integers(50, 2000, size=n_groups).astype(np.int16)
    peak_charges = rng.uniform(1e3, 5e4, size=n_groups).astype(np.float32)
    delta_times = rng.integers(-5, 6, size=total).astype(np.int8)
    charges_u16 = rng.integers(1, 65535, size=total, dtype=np.uint16)

    # Delta-encoded sparse for the sensor file. Strictly ordered so
    # cumsum reconstructs a monotone stream. Made longer than the hits
    # CSR total so sensor != hits, mirroring electronics shaping which
    # spreads each hits pixel across many sensor ticks.
    n_sparse = max(total * 3 + 1, 2)
    delta_time = np.concatenate([
        np.array([0], dtype=np.int16),
        rng.integers(0, 6, size=n_sparse - 1, dtype=np.int16),
    ])
    values = rng.integers(100, 4000, size=n_sparse, dtype=np.uint16)

    out = dict(
        group_ids=group_ids, group_sizes=group_sizes,
        center_times=center_times, peak_charges=peak_charges,
        delta_times=delta_times, charges_u16=charges_u16,
        delta_time=delta_time, values=values,
        time_start=int(rng.integers(0, 100)),
        pedestal=0,
    )

    if readout_type == 'pixel':
        out['center_py'] = rng.integers(10, 200, size=n_groups).astype(np.int16)
        out['center_pz'] = rng.integers(10, 200, size=n_groups).astype(np.int16)
        out['delta_py'] = rng.integers(-2, 3, size=total).astype(np.int8)
        out['delta_pz'] = rng.integers(-2, 3, size=total).astype(np.int8)
        out['delta_py_sparse'] = np.concatenate([
            np.array([0], dtype=np.int16),
            rng.integers(0, 3, size=n_sparse - 1, dtype=np.int16),
        ])
        out['delta_pz_sparse'] = np.concatenate([
            np.array([0], dtype=np.int16),
            rng.integers(0, 3, size=n_sparse - 1, dtype=np.int16),
        ])
        out['py_start'] = int(rng.integers(0, 100))
        out['pz_start'] = int(rng.integers(0, 100))
    else:
        out['center_wires'] = rng.integers(10, 500, size=n_groups).astype(np.int16)
        out['delta_wires'] = rng.integers(-3, 4, size=total).astype(np.int8)
        out['delta_wire'] = np.concatenate([
            np.array([0], dtype=np.int16),
            rng.integers(0, 4, size=n_sparse - 1, dtype=np.int16),
        ])
        out['wire_start'] = int(rng.integers(0, 100))

    return out


def _write_jaxtpc_edep(path, events):
    with h5py.File(path, 'w') as f:
        cfg = f.create_group('config')
        cfg.attrs['n_events'] = len(events)
        cfg.attrs['n_volumes'] = len(events[0])
        cfg.attrs['pos_step_mm'] = _JAXTPC_POS_STEP_MM
        for i, volumes in enumerate(events):
            evt = f.create_group(f'event_{i:03d}')
            for v in volumes:
                vg = evt.create_group(f'volume_{v["vol_idx"]}')
                vg.attrs['n_actual'] = int(v['positions'].shape[0])
                vg.attrs['pos_step_mm'] = _JAXTPC_POS_STEP_MM
                vg.attrs['pos_origin_x'] = _JAXTPC_POS_ORIGIN[0]
                vg.attrs['pos_origin_y'] = _JAXTPC_POS_ORIGIN[1]
                vg.attrs['pos_origin_z'] = _JAXTPC_POS_ORIGIN[2]
                vg.create_dataset('positions', data=v['positions'])
                for k in ('de', 'dx', 'theta', 'phi', 't0_us',
                          'charge', 'photons'):
                    vg.create_dataset(k, data=v[k])


def _write_jaxtpc_sensor(path, events, readout_type='wire'):
    with h5py.File(path, 'w') as f:
        cfg = f.create_group('config')
        cfg.attrs['n_events'] = len(events)
        cfg.attrs['readout_type'] = readout_type
        for i, volumes in enumerate(events):
            evt = f.create_group(f'event_{i:03d}')
            for v in volumes:
                vg = evt.create_group(f'volume_{v["vol_idx"]}')
                for plane_name, plane in v['planes'].items():
                    pg = vg.create_group(plane_name)
                    pg.attrs['time_start'] = plane['time_start']
                    pg.attrs['pedestal'] = plane['pedestal']
                    pg.create_dataset('delta_time', data=plane['delta_time'])
                    pg.create_dataset('values', data=plane['values'])
                    if readout_type == 'pixel':
                        pg.attrs['py_start'] = plane['py_start']
                        pg.attrs['pz_start'] = plane['pz_start']
                        pg.create_dataset('delta_py',
                                          data=plane['delta_py_sparse'])
                        pg.create_dataset('delta_pz',
                                          data=plane['delta_pz_sparse'])
                    else:
                        pg.attrs['wire_start'] = plane['wire_start']
                        pg.create_dataset('delta_wire',
                                          data=plane['delta_wire'])


def _write_jaxtpc_hits(path, events, readout_type='wire'):
    shared_keys = ('group_ids', 'group_sizes', 'center_times',
                   'peak_charges', 'delta_times', 'charges_u16')
    readout_keys = (('center_py', 'center_pz', 'delta_py', 'delta_pz')
                    if readout_type == 'pixel'
                    else ('center_wires', 'delta_wires'))
    with h5py.File(path, 'w') as f:
        cfg = f.create_group('config')
        cfg.attrs['n_events'] = len(events)
        cfg.attrs['readout_type'] = readout_type
        for i, volumes in enumerate(events):
            evt = f.create_group(f'event_{i:03d}')
            for v in volumes:
                vg = evt.create_group(f'volume_{v["vol_idx"]}')
                vg.create_dataset('group_to_track', data=v['group_to_track'])
                vg.create_dataset('deposit_to_group',
                                  data=v['deposit_to_group'])
                vg.create_dataset('qs_fractions', data=v['qs_fractions'])
                for plane_name, plane in v['planes'].items():
                    pg = vg.create_group(plane_name)
                    for key in shared_keys + readout_keys:
                        pg.create_dataset(key, data=plane[key])


def _write_jaxtpc_labl(path, events):
    with h5py.File(path, 'w') as f:
        cfg = f.create_group('config')
        cfg.attrs['n_events'] = len(events)
        cfg.attrs['n_volumes'] = len(events[0])
        for i, volumes in enumerate(events):
            evt = f.create_group(f'event_{i:03d}')
            for v in volumes:
                vg = evt.create_group(f'volume_{v["vol_idx"]}')
                vg.create_dataset('track_ids', data=v['track_ids'])
                vg.create_dataset('track_pdg', data=v['track_pdg'])
                # cluster and interaction need >1 unique value for
                # test_different_label_keys; derive from track position.
                cluster = np.arange(1, 1 + len(v['track_ids']), dtype=np.int32)
                interaction = (np.arange(len(v['track_ids']),
                                         dtype=np.int32) % 3) + 1
                vg.create_dataset('track_cluster', data=cluster)
                vg.create_dataset('track_interaction', data=interaction)
                vg.create_dataset('track_ancestor', data=v['track_ids'])
                vg.create_dataset('deposit_to_track',
                                  data=v['deposit_to_track'])


# ---------------------------------------------------------------------------
# LUCiD
# ---------------------------------------------------------------------------

_LUCID_PDG_POOL = np.array([11, 13, 22, 211, -11], dtype=np.int32)


def make_lucid_sample(outdir, dataset_name='wc', n_events=2, n_files=1,
                      n_segments=80, n_hits=120, n_hits_entries=200,
                      n_sensors=64, n_tracks=8, n_particles=3, seed=0):
    """Write a minimal schema-conformant LUCiD v3 dataset.

    Creates ``{outdir}/{edep,sensor,hits,labl}/{dataset_name}_{mod}_NNNN.h5``.
    """
    os.makedirs(outdir, exist_ok=True)
    for mod in ('edep', 'sensor', 'hits', 'labl'):
        os.makedirs(os.path.join(outdir, mod), exist_ok=True)

    rng = np.random.default_rng(seed)
    # Single PMT geometry reused across all events and both sensor/hits files.
    pmt_positions = rng.uniform(-500.0, 500.0,
                                size=(n_sensors, 3)).astype(np.float32)

    for file_idx in range(n_files):
        events = [
            _build_lucid_event(rng, n_segments, n_hits, n_hits_entries,
                               n_sensors, n_tracks, n_particles)
            for _ in range(n_events)
        ]
        tag = f'{dataset_name}_{{mod}}_{file_idx:04d}.h5'
        _write_lucid_edep(os.path.join(outdir, 'edep',
                                       tag.format(mod='edep')), events)
        _write_lucid_sensor(os.path.join(outdir, 'sensor',
                                         tag.format(mod='sensor')),
                            events, pmt_positions)
        _write_lucid_hits(os.path.join(outdir, 'hits',
                                       tag.format(mod='hits')),
                          events, pmt_positions)
        _write_lucid_labl(os.path.join(outdir, 'labl',
                                       tag.format(mod='labl')), events)

    return outdir


def _build_lucid_event(rng, n_segments, n_hits, n_hits_entries,
                       n_sensors, n_tracks, n_particles):
    # Allocate track ids — ancestor must itself be a valid track id, so we
    # pick one track per particle to be that particle's ancestor and
    # everyone else's ancestor falls back to it.
    track_ids = np.sort(rng.choice(
        np.arange(1, 10 * n_tracks + 1), size=n_tracks,
        replace=False)).astype(np.int32)
    track_particle_idx = rng.integers(0, n_particles, size=n_tracks).astype(
        np.int32)
    # One root per particle — the first track mapped to that particle.
    roots_by_particle = {}
    for tid, pidx in zip(track_ids, track_particle_idx):
        roots_by_particle.setdefault(int(pidx), int(tid))
    ancestor = np.array(
        [roots_by_particle[int(pidx)] for pidx in track_particle_idx],
        dtype=np.int32)

    # Per-particle
    category = rng.integers(0, 5, size=n_particles).astype(np.int32)
    containment = rng.uniform(0.0, 1.0, size=n_particles).astype(np.float32)
    # Trivial genealogy: each particle's entry is just its own index.
    genealogy_offsets = np.arange(n_particles + 1, dtype=np.int32)
    genealogy_data = np.arange(n_particles, dtype=np.int32)
    ext_genealogy_offsets = genealogy_offsets.copy()
    ext_genealogy_data = genealogy_data.copy()

    # Per-track metadata
    pdg = rng.choice(_LUCID_PDG_POOL, size=n_tracks).astype(np.int32)
    parent_id = np.where(
        rng.random(n_tracks) < 0.5, 0, track_ids).astype(np.int32)
    interaction = rng.integers(0, 3, size=n_tracks).astype(np.int32)
    initial_energy = rng.uniform(0.1, 10.0, size=n_tracks).astype(np.float32)
    n_cherenkov_track = rng.integers(0, 100, size=n_tracks).astype(np.int32)

    # Edep — track_idx is a POSITIONAL index into the per_track table
    # (row index, not the Geant4 track_id value). See
    # lucid.py::_lookup_per_track which gathers with track_idx directly.
    seg_track_idx = rng.integers(0, n_tracks, size=n_segments).astype(np.int32)
    start = rng.uniform(-500.0, 500.0, size=(n_segments, 3)).astype(np.float32)
    direction = rng.normal(0.0, 1.0, size=(n_segments, 3)).astype(np.float32)
    norms = np.linalg.norm(direction, axis=1, keepdims=True)
    direction = np.where(norms > 0, direction / norms, direction)
    end = start + direction * rng.uniform(
        0.1, 10.0, size=(n_segments, 1)).astype(np.float32)
    edep = rng.uniform(0.01, 5.0, size=n_segments).astype(np.float32)
    seg_time = rng.uniform(0.0, 100.0, size=n_segments).astype(np.float32)
    beta_start = rng.uniform(0.0, 1.0, size=n_segments).astype(np.float32)
    n_cherenkov_seg = rng.integers(0, 50, size=n_segments).astype(np.int32)

    # Sensor
    sensor_sensor_idx = rng.integers(0, n_sensors, size=n_hits).astype(
        np.int32)
    sensor_pe = rng.uniform(0.1, 50.0, size=n_hits).astype(np.float32)
    sensor_t = rng.uniform(0.0, 100.0, size=n_hits).astype(np.float32)

    # Hits — particle_idx must be valid index into per_particle (< n_particles)
    hits_sensor_idx = rng.integers(0, n_sensors, size=n_hits_entries).astype(
        np.int32)
    hits_particle_idx = rng.integers(0, n_particles,
                                     size=n_hits_entries).astype(np.int32)
    hits_pe = rng.uniform(0.05, 20.0, size=n_hits_entries).astype(np.float32)
    hits_t = rng.uniform(0.0, 100.0, size=n_hits_entries).astype(np.float32)

    return dict(
        edep=dict(
            start=start, end=end, direction=direction, edep=edep,
            time=seg_time, track_idx=seg_track_idx, beta_start=beta_start,
            n_cherenkov=n_cherenkov_seg,
        ),
        sensor=dict(
            sensor_idx=sensor_sensor_idx, PE=sensor_pe, T=sensor_t,
        ),
        hits=dict(
            sensor_idx=hits_sensor_idx, particle_idx=hits_particle_idx,
            PE=hits_pe, T=hits_t,
        ),
        labl=dict(
            t0=np.float32(rng.uniform(-1.0, 1.0)),
            overall_containment=np.float32(rng.uniform(0.5, 1.0)),
            category=category, containment=containment,
            genealogy_data=genealogy_data,
            genealogy_offsets=genealogy_offsets,
            ext_genealogy_data=ext_genealogy_data,
            ext_genealogy_offsets=ext_genealogy_offsets,
            track_id=track_ids, pdg=pdg, parent_id=parent_id,
            particle_idx=track_particle_idx, ancestor=ancestor,
            interaction=interaction, initial_energy=initial_energy,
            n_cherenkov=n_cherenkov_track,
            n_particles=n_particles,
        ),
    )


def _write_lucid_edep(path, events):
    with h5py.File(path, 'w') as f:
        cfg = f.create_group('config')
        cfg.attrs['n_events'] = len(events)
        cfg.attrs['format_version'] = 3
        for i, evt in enumerate(events):
            seg = evt['edep']
            g = f.create_group(f'event_{i:03d}')
            g.attrs['n_segments'] = int(seg['start'].shape[0])
            g.create_dataset('start_x', data=seg['start'][:, 0])
            g.create_dataset('start_y', data=seg['start'][:, 1])
            g.create_dataset('start_z', data=seg['start'][:, 2])
            g.create_dataset('end_x', data=seg['end'][:, 0])
            g.create_dataset('end_y', data=seg['end'][:, 1])
            g.create_dataset('end_z', data=seg['end'][:, 2])
            g.create_dataset('edep', data=seg['edep'])
            g.create_dataset('time', data=seg['time'])
            g.create_dataset('track_idx', data=seg['track_idx'])
            g.create_dataset('dir_x', data=seg['direction'][:, 0])
            g.create_dataset('dir_y', data=seg['direction'][:, 1])
            g.create_dataset('dir_z', data=seg['direction'][:, 2])
            g.create_dataset('beta_start', data=seg['beta_start'])
            g.create_dataset('n_cherenkov', data=seg['n_cherenkov'])


def _write_lucid_sensor(path, events, pmt_positions):
    with h5py.File(path, 'w') as f:
        cfg = f.create_group('config')
        cfg.attrs['n_events'] = len(events)
        cfg.attrs['n_sensors'] = int(pmt_positions.shape[0])
        cfg.attrs['format_version'] = 3
        cfg.create_dataset('sensor_positions', data=pmt_positions)
        for i, evt in enumerate(events):
            s = evt['sensor']
            g = f.create_group(f'event_{i:03d}')
            g.create_dataset('sensor_idx', data=s['sensor_idx'])
            g.create_dataset('PE', data=s['PE'])
            g.create_dataset('T', data=s['T'])


def _write_lucid_hits(path, events, pmt_positions):
    with h5py.File(path, 'w') as f:
        cfg = f.create_group('config')
        cfg.attrs['n_events'] = len(events)
        cfg.attrs['n_sensors'] = int(pmt_positions.shape[0])
        cfg.attrs['format_version'] = 3
        cfg.create_dataset('sensor_positions', data=pmt_positions)
        for i, evt in enumerate(events):
            s = evt['hits']
            g = f.create_group(f'event_{i:03d}')
            g.create_dataset('sensor_idx', data=s['sensor_idx'])
            g.create_dataset('particle_idx', data=s['particle_idx'])
            g.create_dataset('PE', data=s['PE'])
            g.create_dataset('T', data=s['T'])


def _write_lucid_labl(path, events):
    with h5py.File(path, 'w') as f:
        cfg = f.create_group('config')
        cfg.attrs['n_events'] = len(events)
        cfg.attrs['format_version'] = 3
        for i, evt in enumerate(events):
            l = evt['labl']
            g = f.create_group(f'event_{i:03d}')
            g.attrs['n_particles'] = int(l['n_particles'])

            pe = g.create_group('per_event')
            pe.create_dataset('t0', data=np.float32(l['t0']))
            pe.create_dataset('overall_containment',
                              data=np.float32(l['overall_containment']))

            pp = g.create_group('per_particle')
            pp.create_dataset('category', data=l['category'])
            pp.create_dataset('containment', data=l['containment'])
            pp.create_dataset('genealogy_data', data=l['genealogy_data'])
            pp.create_dataset('genealogy_offsets', data=l['genealogy_offsets'])
            pp.create_dataset('ext_genealogy_data', data=l['ext_genealogy_data'])
            pp.create_dataset('ext_genealogy_offsets',
                              data=l['ext_genealogy_offsets'])

            pt = g.create_group('per_track')
            pt.create_dataset('track_id', data=l['track_id'])
            pt.create_dataset('pdg', data=l['pdg'])
            pt.create_dataset('parent_id', data=l['parent_id'])
            pt.create_dataset('particle_idx', data=l['particle_idx'])
            pt.create_dataset('ancestor', data=l['ancestor'])
            pt.create_dataset('interaction', data=l['interaction'])
            pt.create_dataset('initial_energy', data=l['initial_energy'])
            pt.create_dataset('n_cherenkov', data=l['n_cherenkov'])


__all__ = ['make_jaxtpc_sample', 'make_lucid_sample']
