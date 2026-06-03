"""
Detector-specific transforms.

``PDGToSemantic``: derives semantic labels from per-deposit PDG codes
in step data when no labl file is available (fallback path).

``RemapSegment``: maps the values in ``data_dict['segment']`` through a
user-provided ``{old_value: new_value}`` dict, with a default for
unmapped values. Typical use: after loading with ``label_key='pdg'``
(so ``segment`` holds raw PDG codes), remap to a task-specific set of
class indices before loss. Works for any int-valued segment column
(pdg, interaction, cluster) — not PDG-specific.
"""

import numpy as np

from .transform import TRANSFORMS, Compose
from .utils.pdg import pdg_to_semantic, MOTIF_MAP, PID_MAP

_NAMED_SCHEMES = {
    'motif_5cls': (MOTIF_MAP, 4),
    'pid_6cls': (PID_MAP, 5),
}


@TRANSFORMS.register_module()
class ApplyToStream:
    """Dispatch a sub-pipeline to a nested sub-dict keyed by ``stream``.

    :class:`JAXTPCDataset` emits nested dicts of the form
    ``{'step': {'coord': ..., 'segment': ...}, 'hits': {...}, ...}``.
    Wrap transforms that hardcode ``'coord'`` / ``'segment'`` in
    ``ApplyToStream(stream='step', transforms=[...])`` so they operate
    on the sub-dict directly::

        dict(type='ApplyToStream', stream='step', transforms=[
            dict(type='GridSample', grid_size=0.5),
            dict(type='RandomRotate'),
        ])

    If ``data_dict`` has no ``stream`` key, the transform is a no-op.
    This lets a single config run through optional streams without
    branching on modality presence.
    """

    def __init__(self, stream, transforms=None, required=False):
        if transforms is None:
            transforms = []
        self.stream = stream
        self.required = bool(required)
        self.inner = Compose(transforms) if transforms else None

    def __call__(self, data_dict):
        if self.stream not in data_dict:
            if self.required:
                raise KeyError(
                    f"ApplyToStream(stream={self.stream!r}) applied but "
                    f"data_dict has no such stream; "
                    f"available: {sorted(k for k in data_dict if isinstance(data_dict.get(k), dict))}")
            return data_dict
        if self.inner is not None:
            data_dict[self.stream] = self.inner(data_dict[self.stream])
        return data_dict


@TRANSFORMS.register_module()
class PDGToSemantic:
    """Fallback: derive approximate semantic labels from PDG codes.

    Schemes
    -------
    motif_5cls : shower(0), track(1), michel(2), delta(3), led(4)
    pid_6cls : photon(0), electron(1), muon(2), pion(3), proton(4), other(5)
    custom : user-provided {pdg_code: class_index} dict
    """

    def __init__(self, scheme='motif_5cls', custom_map=None):
        if scheme not in ('motif_5cls', 'pid_6cls', 'custom', 'none'):
            raise ValueError(f"Unknown label scheme: {scheme}")
        if scheme == 'custom':
            assert custom_map is not None
        self.scheme = scheme
        self.custom_map = custom_map

    def __call__(self, data_dict):
        if self.scheme == 'none' or 'pdg' not in data_dict:
            return data_dict

        if 'segment' in data_dict or 'segment_motif' in data_dict:
            return data_dict

        pdg = data_dict['pdg']
        labels = pdg_to_semantic(pdg, scheme=self.scheme,
                                 custom_map=self.custom_map)
        data_dict['segment_motif'] = labels[:, None]

        if self.scheme == 'motif_5cls':
            pid = pdg_to_semantic(pdg, scheme='pid_6cls')
            data_dict['segment_pid'] = pid[:, None]
        elif self.scheme == 'pid_6cls':
            data_dict['segment_pid'] = labels[:, None]

        n = len(labels)

        if 'instance_particle' not in data_dict and 'track_ids' in data_dict:
            track_ids = data_dict['track_ids']
            mask = track_ids >= 0
            if mask.any():
                _, inverse = np.unique(track_ids[mask], return_inverse=True)
                out = np.full(n, -1, dtype=np.int32)
                out[mask] = inverse
                data_dict['instance_particle'] = out[:, None]
            else:
                data_dict['instance_particle'] = np.full((n, 1), -1, dtype=np.int32)

        if 'instance_interaction' not in data_dict and 'interaction_ids' in data_dict:
            iids = data_dict['interaction_ids']
            mask = iids >= 0
            if mask.any():
                _, inverse = np.unique(iids[mask], return_inverse=True)
                out = np.full(n, -1, dtype=np.int32)
                out[mask] = inverse
                data_dict['instance_interaction'] = out[:, None]
            else:
                data_dict['instance_interaction'] = np.full((n, 1), -1, dtype=np.int32)

            data_dict['segment_interaction'] = (iids[:, None] != -1).astype(np.int32)

        return data_dict


@TRANSFORMS.register_module()
class RemapSegment:
    """Remap integer values in a segment-like field through a lookup dict.

    Motivating case: production labl files store raw PDG codes in the
    ``pdg`` column. A downstream task wants 5-class indices. Configure
    ``JAXTPCDataset`` with ``label_key='pdg'`` (so ``segment`` contains
    raw PDG codes per deposit), then add::

        dict(type="RemapSegment", scheme="motif_5cls")

    or with an explicit map::

        dict(type="RemapSegment",
             mapping={22: 0, 11: 0, 13: 1, 211: 1, 2212: 1},
             default=2)

    Parameters
    ----------
    mapping : dict[int, int], optional
        Explicit ``{source_value: target_class}`` map. Mutually
        exclusive with ``scheme``.
    default : int, optional
        Class index for values not in ``mapping``. If ``mapping`` is
        given, default is ``max(mapping.values()) + 1``; if ``scheme``
        is given, scheme-specific default (4 for motif_5cls, 5 for
        pid_6cls). Can be overridden explicitly.
    scheme : str, optional
        One of ``'motif_5cls'``, ``'pid_6cls'``. Loads the built-in
        PDG→class map. Overridden by ``mapping`` if both are given.
    key : str, optional
        Which data-dict field to remap. Default ``'segment'``.
    ignore_value : int, optional
        Source values equal to this are written as-is (bypass default).
        Default ``-1`` so ignore-index sentinels survive remapping.
    """

    def __init__(self, mapping=None, default=None, scheme=None,
                 key='segment', ignore_value=-1):
        if mapping is None and scheme is None:
            raise ValueError("RemapSegment needs either `mapping` or `scheme`")
        if mapping is not None:
            self._map = {int(k): int(v) for k, v in mapping.items()}
            self._default = int(default) if default is not None \
                else max(self._map.values()) + 1
        else:
            if scheme not in _NAMED_SCHEMES:
                raise ValueError(f"Unknown scheme {scheme!r}; "
                                 f"pick from {list(_NAMED_SCHEMES)} or pass `mapping`")
            base_map, scheme_default = _NAMED_SCHEMES[scheme]
            self._map = dict(base_map)
            self._default = int(default) if default is not None else scheme_default
        self._key = key
        self._ignore_value = int(ignore_value)

    def __call__(self, data_dict):
        seg = data_dict.get(self._key)
        if seg is None:
            return data_dict

        seg = np.asarray(seg)
        orig_shape = seg.shape
        flat = seg.ravel()

        out = np.full(flat.shape, self._default, dtype=np.int32)
        # Preserve ignore sentinels
        ignore_mask = flat == self._ignore_value
        out[ignore_mask] = self._ignore_value
        # Apply the map
        for src, dst in self._map.items():
            out[(flat == src) & ~ignore_mask] = dst

        data_dict[self._key] = out.reshape(orig_shape)
        return data_dict


@TRANSFORMS.register_module()
class AggregateSensorHits:
    """Aggregate a LUCiD sensor event's multiple hits per PMT into one point.

    A raw LUCiD ``sensor`` event has several hits per PMT; this groups by
    ``sensor_idx`` and emits one point per PMT — ``coord`` (the shared PMT
    position), ``energy`` (summed PE), ``time`` (aggregated by
    ``time_aggregation``), ``sensor_idx`` (unique). It reads the ``stream``
    sub-dict (``MultiModalEventDataset`` emits nested dicts) and writes the
    aggregated arrays to the **top level** — the flat ``coord``/``energy``/
    ``time`` keys the event-SSL pipeline consumes — then drops the consumed
    sub-dict. This replaces the inline aggregation the dissolved
    ``LUCiDEventSSLDataset`` did (D32 / the colleague's registered-transform
    suggestion).

    ``time_aggregation`` ∈ ``{'earliest', 'mean', 'pe_weighted', 'first'}``
    (the LUCiDEventSSLDataset set): minimum / count-mean / PE-weighted-mean /
    first-hit time per PMT. When ``stream`` is absent (already flat) it
    operates on the top-level keys in place.
    """

    _STRATEGIES = ('earliest', 'mean', 'pe_weighted', 'first')

    def __init__(self, stream='sensor', time_aggregation='earliest',
                 coord_key='coord', energy_key='energy', time_key='time',
                 sensor_key='sensor_idx'):
        if time_aggregation not in self._STRATEGIES:
            raise ValueError(
                f"time_aggregation must be one of {self._STRATEGIES}, "
                f"got {time_aggregation!r}")
        self.stream = stream
        self.time_aggregation = time_aggregation
        self.coord_key = coord_key
        self.energy_key = energy_key
        self.time_key = time_key
        self.sensor_key = sensor_key

    def __call__(self, data_dict):
        sub = data_dict.get(self.stream)
        src = sub if isinstance(sub, dict) else data_dict
        sensor_idx = src.get(self.sensor_key)
        if sensor_idx is None:
            return data_dict
        agg = self._aggregate(
            np.asarray(sensor_idx).reshape(-1),
            np.asarray(src[self.coord_key]),
            np.asarray(src[self.energy_key]),
            np.asarray(src[self.time_key]))
        for k, v in agg.items():
            data_dict[k] = v
        if isinstance(sub, dict):                 # lift out of the stream
            data_dict.pop(self.stream, None)
        return data_dict

    def _aggregate(self, sensor_idx, coord, energy, time):
        out = {self.coord_key: coord.astype(np.float32, copy=False),
               self.energy_key: energy.astype(np.float32, copy=False),
               self.time_key: time.astype(np.float32, copy=False),
               self.sensor_key: sensor_idx.astype(np.int64, copy=False)}
        if sensor_idx.size == 0:
            return out
        order = np.argsort(sensor_idx, kind='stable')
        s_sid = sensor_idx[order]
        uniq, starts = np.unique(s_sid, return_index=True)
        e_sorted = energy[order]
        t_sorted = time[order]
        energy_a = np.add.reduceat(e_sorted, starts, axis=0)
        if self.time_aggregation == 'earliest':
            time_a = np.minimum.reduceat(t_sorted, starts, axis=0)
        elif self.time_aggregation == 'mean':
            counts = np.diff(np.r_[starts, s_sid.shape[0]])
            counts = counts.reshape((-1,) + (1,) * (t_sorted.ndim - 1))
            time_a = np.add.reduceat(t_sorted, starts, axis=0) / counts
        elif self.time_aggregation == 'pe_weighted':
            weighted = np.add.reduceat(t_sorted * e_sorted, starts, axis=0)
            time_a = weighted / np.maximum(energy_a, 1.0e-6)
        else:  # 'first'
            time_a = t_sorted[starts]
        out[self.coord_key] = coord[order][starts].astype(np.float32, copy=False)
        out[self.energy_key] = energy_a.astype(np.float32, copy=False)
        out[self.time_key] = time_a.astype(np.float32, copy=False)
        out[self.sensor_key] = uniq.astype(np.int64, copy=False)
        return out
