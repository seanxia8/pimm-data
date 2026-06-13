"""
Post-collate, on-device dense transforms (REDESIGN).

There is **no batch-transform concept and no runner**: the dense ops are ordinary
``scope='sample'`` transforms placed *after* a ``ToDevice`` step and run by the plain
:class:`pimm_data.transform.Compose` (or any loop). The dense grids are born
on-device; pass ``ToDevice('cpu')`` to run the same transforms on CPU.

Registered transforms (all ``scope='sample'`` — per-event independent):

* ``ToDevice``               — move the batch to a device (the device step)
* ``BatchDensify``           — sparse ``<m>_wire/time/value/plane_gid/offset`` -> ``<m>_dense={gid:(B,W,T)}``
* ``BatchAddIntrinsicNoise`` — fresh coherent (+optional incoherent) noise; **self-seeds**
  from ``batch['name']`` (+ optional ``batch['_epoch']``/``_rank``) — no runner injects seeds
* ``BatchDigitize``          — quantize to ADC codes

Each is ``fn(batch) -> batch`` (flat-prefixed keys; ``modality=`` sets the prefix,
``None`` = bare). :func:`sensor_dense_cfg` returns the standard chain as config;
:func:`build_sensor_gpu_stages` wraps it in a ``Compose`` you call on the batch.

Seeds are content-addressed per event (``blake2b(name|base_seed|epoch|rank)``), so
the same event gets the same noise across batch position / worker / resume. The
**incoherent** component uses a torch ``Generator`` whose CPU/CUDA streams differ, so
incoherent realizations are device-specific (coherent uses numpy, device-independent).
Geometry is a registry baked into the transform, not carried per-sample.
"""

from __future__ import annotations

import hashlib
import warnings

import torch

from . import dense_ops
from .transform import TRANSFORMS
from .noise import (DEFAULT_ENC, DEFAULT_SAMPLING_RATE_HZ, DEFAULT_GROUP_SIZE,
                    DEFAULT_COH_RMS_ADC, DEFAULT_COH_CORNER_FREQ_HZ,
                    DEFAULT_COH_SLOPE, DEFAULT_COH_BETA)


# ── device move (idempotent) ──────────────────────────────────────────────
def move_to_device(batch, device, non_blocking=True):
    """Recursively move tensors to ``device``; non-tensors untouched.

    Idempotent — a tensor already on ``device`` is returned by ``.to`` without a
    copy. Use ``DataLoader(pin_memory=True)`` for ``non_blocking`` to be truly async.
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=non_blocking)
    if isinstance(batch, dict):
        return {k: move_to_device(v, device, non_blocking) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_to_device(v, device, non_blocking) for v in batch)
    return batch


# ── content-addressed seeding ─────────────────────────────────────────────
def content_seed(event_name, base_seed=0, epoch=0, rank=0):
    """Stable per-event seed: same event -> same noise across batch/worker/resume."""
    payload = f"{event_name}|{int(base_seed)}|{int(epoch)}|{int(rank)}".encode()
    h = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(h, 'little') & ((1 << 63) - 1)


def _seeds_for(batch, base_seed):
    """Per-event content-addressed seeds from ``batch['name']`` (self-contained — no
    runner injects them). Epoch/rank are read from optional ``batch['_epoch']`` /
    ``batch['_rank']`` the trainer may stamp (default 0 = stable within a run)."""
    names = batch.get('name')
    if isinstance(names, str):
        names = [names]
    epoch = int(batch.get('_epoch', 0)) if not torch.is_tensor(batch.get('_epoch')) else int(batch['_epoch'])
    rank = int(batch.get('_rank', 0)) if not torch.is_tensor(batch.get('_rank')) else int(batch['_rank'])
    if not names:
        warnings.warn(
            "AddNoise: batch has no 'name' -> seeding falls back to batch position, "
            "NOT reproducible across shuffles/epochs/runs. Ensure Collect passes "
            "'name' through.", RuntimeWarning, stacklevel=2)
        # length B from the dense grid is unknown here; caller passes count via name
        return None
    return [content_seed(nm, base_seed, epoch, rank) for nm in names]


# ── batch transforms (post-collate; registered, scope='batch') ───────────────
@TRANSFORMS.register_module()
class ToDevice:
    """Move the whole batch to a device — the device move expressed as a transform.

    So device selection composes uniformly with the other batch transforms instead
    of being a special ``apply_batch_transforms(device=)`` argument: put
    ``dict(type='ToDevice', device='cuda')`` first in the list and everything after
    runs on-device. Idempotent (a tensor already on ``device`` is returned without a
    copy). ``scope='batch'`` so the pre-collate ``Compose`` fence keeps it post-collate.
    """

    scope = 'sample'

    def __init__(self, device, *, non_blocking=True):
        self.device = device
        self.non_blocking = non_blocking

    def __call__(self, batch):
        return move_to_device(batch, self.device, non_blocking=self.non_blocking)


@TRANSFORMS.register_module()
class BatchDensify:
    """Sparse hits -> per-plane dense grids ``batch[dense_key] = {gid: (B,W,T)}``."""

    scope = 'sample'  # per-event independent; placed post-collate by cost

    def __init__(self, geom, *, modality=None, wire_key='wire', time_key='time',
                 value_key='value', plane_key='plane_gid', offset_key='offset',
                 dense_key='dense'):
        self.geom = geom
        self.modality = modality
        self.wire_key, self.time_key, self.value_key = wire_key, time_key, value_key
        self.plane_key, self.offset_key, self.dense_key = plane_key, offset_key, dense_key

    def __call__(self, batch):
        # REDESIGN: flat-prefixed keys. modality=None → bare keys (`wire`, `dense`);
        # modality='sensor' → flat `sensor_wire` … `sensor_dense`.
        pfx = f'{self.modality}_' if self.modality is not None else ''
        m = self.modality if self.modality is not None else '<bare batch>'
        wire = batch[pfx + self.wire_key]
        off = batch[pfx + self.offset_key]
        n = wire.reshape(-1).shape[0]
        # Clear signal for the #1 dense coupling: a coord-mutating transform on this
        # part pre-collate desyncs the raw COO from offset.
        if off.numel() and int(off[-1]) != n:
            raise ValueError(
                f"BatchDensify({m!r}): offset total {int(off[-1])} != {n} raw-COO "
                f"rows ({pfx + self.wire_key!r}). A coord-mutating transform "
                "(GridSample/SphereCrop/RandomDropout) ran on this part pre-collate "
                "— densify needs the raw COO aligned with offset. Voxelize a SEPARATE "
                "sparse view, not the one you densify.")
        batch[pfx + self.dense_key] = dense_ops.densify(
            wire, batch[pfx + self.time_key], batch[pfx + self.value_key],
            batch[pfx + self.plane_key], off, self.geom)
        return batch


@TRANSFORMS.register_module()
class BatchAddIntrinsicNoise:
    """Add fresh per-event coherent (+optional incoherent) noise to the grids."""

    scope = 'sample'  # per-event independent (per-event seed/elementwise)

    def __init__(self, geom, *, modality=None, base_seed=0, coherent=True,
                 incoherent=False, dense_key='dense', enc=DEFAULT_ENC,
                 sampling_rate_hz=DEFAULT_SAMPLING_RATE_HZ,
                 group_size=DEFAULT_GROUP_SIZE, coh_rms=DEFAULT_COH_RMS_ADC,
                 coh_corner_freq_hz=DEFAULT_COH_CORNER_FREQ_HZ,
                 coh_spectral_slope=DEFAULT_COH_SLOPE, beta=DEFAULT_COH_BETA,
                 series_spectrum=None):
        self.geom = geom
        self.modality = modality
        self.base_seed = int(base_seed)
        self.coherent, self.incoherent = bool(coherent), bool(incoherent)
        self.dense_key = dense_key
        self.enc = tuple(enc)
        self.kw = dict(sampling_rate_hz=sampling_rate_hz, group_size=group_size,
                       coh_rms=coh_rms, coh_corner_freq_hz=coh_corner_freq_hz,
                       coh_spectral_slope=coh_spectral_slope, beta=beta,
                       series_spectrum=series_spectrum)

    def __call__(self, batch):
        # SELF-CONTAINED seeding: derive per-event seeds from batch['name'] here, so
        # no runner needs to inject them — this transform is a plain fn(batch)->batch.
        pfx = f'{self.modality}_' if self.modality is not None else ''
        grids = batch[pfx + self.dense_key]
        seeds = _seeds_for(batch, self.base_seed)
        if seeds is None:                                  # no 'name' -> position fallback
            seeds = [content_seed(f"_idx{i}", self.base_seed)
                     for i in range(next(iter(grids.values())).shape[0])]
        dense_ops.add_intrinsic_noise(
            grids, self.geom, seeds=seeds, enc=self.enc,
            coherent=self.coherent, incoherent=self.incoherent, **self.kw)
        return batch


@TRANSFORMS.register_module()
class BatchDigitize:
    """Quantize per-plane grids to ADC codes (pedestal from registry or override)."""

    scope = 'sample'  # per-event independent (per-event seed/elementwise)

    def __init__(self, geom=None, *, modality=None, pedestal=None, n_bits=12,
                 adc_max=None, gain=1.0, dense_key='dense'):
        self.geom = geom or {}
        self.modality = modality
        self.pedestal = pedestal
        self.n_bits, self.adc_max, self.gain = n_bits, adc_max, gain
        self.dense_key = dense_key

    def __call__(self, batch):
        pfx = f'{self.modality}_' if self.modality is not None else ''
        ped = self.pedestal
        if ped is None:
            ped = {gid: e.get('pedestal', 0) for gid, e in self.geom.items()}
        batch[pfx + self.dense_key] = dense_ops.digitize(
            batch[pfx + self.dense_key], ped, n_bits=self.n_bits,
            adc_max=self.adc_max, gain=self.gain)
        return batch


def sensor_dense_cfg(geom, *, modality='sensor', device=None, base_seed=0,
                     coherent=True, incoherent=False, digitize=True, n_bits=12,
                     dense_key=None, **noise_kw):
    """The standard dense sensor chain as plain ``dict(type=…)`` configs:
    ``[ToDevice?, BatchDensify, BatchAddIntrinsicNoise, BatchDigitize?]``.

    Run it with the ordinary ``Compose`` (these are ``scope='sample'`` transforms —
    there is **no batch-transform runner**): ``Compose([*sensor_dense_cfg(geom,
    device='cuda'), my_user_gpu_fn])(batch)``. Noise self-seeds from ``batch['name']``
    (``base_seed`` here; epoch/rank from optional ``batch['_epoch']``/``_rank``).

    Densify is opt-in and additive (the sparse COO is never replaced). ``modality=
    'sensor'`` writes flat ``sensor_dense``; ``modality=None`` writes bare ``dense``.
    """
    if dense_key is None:
        dense_key = 'dense'
    cfg = []
    if device is not None:
        cfg.append(dict(type='ToDevice', device=device))
    cfg.append(dict(type='BatchDensify', geom=geom, modality=modality,
                    dense_key=dense_key))
    cfg.append(dict(type='BatchAddIntrinsicNoise', geom=geom, modality=modality,
                    base_seed=base_seed, coherent=coherent, incoherent=incoherent,
                    dense_key=dense_key, **noise_kw))
    if digitize:
        cfg.append(dict(type='BatchDigitize', geom=geom, modality=modality,
                        n_bits=n_bits, dense_key=dense_key))
    return cfg


def build_sensor_gpu_stages(geom, *, modality=None, **kw):
    """Convenience: the dense sensor chain as a runnable ``Compose``.

    ``stages = build_sensor_gpu_stages(geom, device='cuda'); batch = stages(batch)``.
    There is no separate batch-transform runner — the dense ops are ordinary
    ``scope='sample'`` transforms run by ``Compose`` (``ToDevice`` is the device step;
    noise self-seeds). Default ``modality=None`` (bare-batch); pass ``modality=
    'sensor'`` for flat ``sensor_*``.
    """
    from .transform import Compose
    return Compose(sensor_dense_cfg(geom, modality=modality, **kw))


class BatchTransformMixin:
    """Mixin for a LightningModule: run the post-collate transforms in
    ``on_after_batch_transfer``.

    Set ``self.batch_stages`` to a ``Compose`` (e.g. from
    :func:`build_sensor_gpu_stages`). Lightning calls this with the batch on-device;
    we stamp ``_epoch``/``_rank`` so the self-seeding noise is per-epoch/rank, then
    run the transforms. (For a custom loop, just call your ``Compose`` on the batch.)
    """

    batch_stages = None

    def on_after_batch_transfer(self, batch, dataloader_idx=0):
        if not self.batch_stages:
            return batch
        trainer = getattr(self, 'trainer', None)
        batch['_epoch'] = getattr(trainer, 'current_epoch', 0) or 0
        batch['_rank'] = getattr(trainer, 'global_rank', 0) or 0
        return self.batch_stages(batch)
