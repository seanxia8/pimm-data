"""Smoke-run every campaign data-loader recipe through a TOY model.

Loads each recipe's batch (against synthetic fixtures) and feeds it to a minimal
torch model matched to that challenge's data shape — just to see the loader plugs
into a model and what output shape we expect. The models are deliberately trivial
(an MLP); the point is the *data path*, not the model.

Run:  python examples/toy_run.py
"""
import copy
import os
import runpy
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F

from pimm_data import build_dataset, collate_fn, _roles
from pimm_data.testing import (make_jaxtpc_sample, make_lucid_sample,
                               make_optical_sample)

_CFG = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs')
torch.manual_seed(0)


def load_batch(rel, fixture, n=2, **override):
    spec = dict(copy.deepcopy(runpy.run_path(os.path.join(_CFG, rel))['data']['train']))
    spec.update(dict(data_root=fixture, split=''))
    spec.update(override)
    ds = build_dataset(spec)
    return collate_fn([ds[i] for i in range(min(n, len(ds)))])


# ── toy models (one per data shape) ──────────────────────────────────────────
class ToyPointSeg(nn.Module):
    """Per-point classifier: feat (N,Cin) -> logits (N,ncls)."""
    def __init__(self, cin, ncls):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cin, 32), nn.ReLU(), nn.Linear(32, ncls))

    def forward(self, feat):
        return self.net(feat)


class ToyCropEncoder(nn.Module):
    """Mean-pool each packed crop (by offset) -> per-crop embedding (n_crops,emb)."""
    def __init__(self, cin, emb=16):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(cin, 32), nn.ReLU(), nn.Linear(32, emb))

    def forward(self, feat, offset):
        h = self.f(feat)
        bid = _roles.offset_to_batch(offset)
        n_crops = offset.numel()
        pooled = torch.zeros(n_crops, h.shape[1])
        count = torch.zeros(n_crops, 1)
        pooled.index_add_(0, bid, h)
        count.index_add_(0, bid, torch.ones(h.shape[0], 1))
        return pooled / count.clamp(min=1.0)


class ToyChunkClassifier(nn.Module):
    """Per-chunk classifier: [waveform stats | scalars] -> logits (K,ncls)."""
    def __init__(self, n_scalar, ncls):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3 + n_scalar, 32), nn.ReLU(),
                                 nn.Linear(32, ncls))

    def forward(self, adc, length, scalars):
        # pool each chunk's variable-length waveform -> [mean, std, max]
        stats = []
        for w in torch.split(adc, length.tolist()):
            stats.append(torch.stack([w.mean(), w.std(unbiased=False), w.max()]))
        stats = torch.stack(stats)                       # (K, 3)
        return self.net(torch.cat([stats, scalars], dim=1))


def _ncls(seg):
    valid = seg[seg >= 0]
    return max(2, int(valid.max()) + 1) if valid.numel() else 2


def hr(t):
    print(f"\n{'='*70}\n{t}\n{'='*70}")


# ── 1) point-cloud segmentation ──────────────────────────────────────────────
def run_seg(tag, rel, fixture, part, **ov):
    b = load_batch(rel, fixture, **ov)
    feat, seg, off = b[f'{part}_feat'].float(), b[f'{part}_segment'].long(), b[f'{part}_offset']
    ncls = _ncls(seg)
    model = ToyPointSeg(feat.shape[1], ncls)
    logits = model(feat)
    loss = F.cross_entropy(logits, seg.clamp(min=0), ignore_index=-1)
    print(f"[{tag}]  events={off.numel()}  points={feat.shape[0]}  in_ch={feat.shape[1]}"
          f"  -> logits {tuple(logits.shape)} (ncls={ncls})  CE={loss.item():.3f}")


# ── 2) multi-crop SSL ─────────────────────────────────────────────────────────
def run_ssl(tag, rel, fixture, **ov):
    b = load_batch(rel, fixture, **ov)
    enc = ToyCropEncoder(b['global_feat'].shape[1])
    g = enc(b['global_feat'].float(), b['global_offset'])
    l = enc(b['local_feat'].float(), b['local_offset'])
    cos = F.cosine_similarity(g[0:1], g[1:2]).item() if g.shape[0] >= 2 else float('nan')
    print(f"[{tag}]  global_crops={g.shape[0]}  local_crops={l.shape[0]}"
          f"  in_ch={b['global_feat'].shape[1]}  -> global_emb {tuple(g.shape)},"
          f" local_emb {tuple(l.shape)}  cos(g0,g1)={cos:.3f}")


# ── 3) optical per-chunk ──────────────────────────────────────────────────────
def run_optical(tag, rel, fixture, **ov):
    b = load_batch(rel, fixture, **ov)
    adc, length = b['sensor_adc'].float(), b['sensor_length']
    scalars = torch.cat([b['sensor_pe'].float(),
                         b['sensor_t0_ns'][:, None].float()], dim=1)   # (K,2)
    inst = b['sensor_instance'].long()
    uniq, target = torch.unique(inst, return_inverse=True)             # contiguous classes
    model = ToyChunkClassifier(scalars.shape[1], int(uniq.numel()))
    logits = model(adc, length, scalars)
    loss = F.cross_entropy(logits, target)
    assert int(length.sum()) == adc.shape[0]                          # packed waveform intact
    print(f"[{tag}]  chunks(K)={logits.shape[0]}  packed_samples={adc.shape[0]}"
          f"  classes={uniq.numel()}  -> logits {tuple(logits.shape)}  CE={loss.item():.3f}"
          f"  (chunks/evt offset={b['sensor_offset'].tolist()},"
          f" samples/evt={b['sensor_wave_offset'].tolist()})")


def main():
    tmp = tempfile.mkdtemp()
    jr = make_jaxtpc_sample(os.path.join(tmp, 'jaxtpc'), dataset_name='sim', n_events=3)
    lr = make_lucid_sample(os.path.join(tmp, 'lucid'), dataset_name='wc',
                           n_events=3, n_sensors=64, n_hits=200)
    optr = make_optical_sample(os.path.join(tmp, 'opt'), dataset_name='optical',
                               n_events=3, n_files=2)
    optew = make_optical_sample(os.path.join(tmp, 'optew'), dataset_name='light',
                                n_events=3, n_files=1, n_channels=8, schema='east_west')

    hr("Segmentation (per-point classifier)")
    run_seg("JAXTPC semseg", 'jaxtpc/semseg_5cls.py', jr, 'step', min_deposits=0)
    run_seg("LUCiD seg_step", 'lucid/seg_step.py', lr, 'step')
    run_seg("LUCiD perpmt_seg_hits", 'lucid/perpmt_seg_hits.py', lr, 'hits')

    hr("Self-supervised (multi-crop encoder)")
    run_ssl("JAXTPC ssl_step", 'jaxtpc/ssl_step.py', jr, min_deposits=0)
    run_ssl("JAXTPC ssl_sensor", 'jaxtpc/ssl_sensor.py', jr)
    run_ssl("LUCiD ssl_sensor", 'lucid/ssl_sensor.py', lr)
    run_ssl("LUCiD ssl_hits", 'lucid/ssl_hits.py', lr)
    run_ssl("LUCiD ssl_step", 'lucid/ssl_step.py', lr)

    hr("Cross-modality recon (load both, encode sensor)")
    b = load_batch('lucid/recon_sensor_to_step.py', lr)
    enc = ToyPointSeg(b['sensor_feat'].shape[1], 8)
    print(f"[LUCiD recon]  sensor_pts={b['sensor_feat'].shape[0]} (in {b['sensor_feat'].shape[1]})"
          f"  step_pts={b['step_feat'].shape[0]}  -> sensor embed {tuple(enc(b['sensor_feat'].float()).shape)}")

    hr("Optical (per-chunk classifier over packed waveforms)")
    run_optical("Optical interaction (label)", 'optical/interaction_discrimination.py',
                optr, dataset_name='optical')
    run_optical("Optical east/west", 'optical/eastwest_readout.py', optew,
                dataset_name='light')


if __name__ == '__main__':
    main()
