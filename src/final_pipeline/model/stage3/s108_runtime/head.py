"""S108 accel head: per-cell MLP, spatial attention, symmetric temporal conv."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

D_IN = 768
D_HID = 256
N_CLASSES = 4
N_CELLS = 16


class S108Head(nn.Module):
    """(T,16,768) -> (T,F,4) logits; noncausal symmetric depthwise Conv1d-3.

    F = frames_per_row (2 at 10 Hz rows of 0.2 s, 4 at 5 Hz rows of 0.4 s).
    The Conv1d always spans one neighboring row each side; row duration
    sets the seconds (±0.2 s vs ±0.4 s). Default F=2 preserves v1 behavior.
    """

    def __init__(self, d_in=D_IN, d_hid=D_HID, n_classes=N_CLASSES,
                 frames_per_row=2):
        super().__init__()
        if int(frames_per_row) < 1:
            raise ValueError("frames_per_row must be >= 1")
        self.frames_per_row = int(frames_per_row)
        self.norm = nn.LayerNorm(d_in)
        self.cell = nn.Linear(d_in, d_hid)
        self.pos = nn.Parameter(torch.zeros(N_CELLS, d_hid))
        self.attn = nn.Linear(d_hid, 1)
        self.temp = nn.Conv1d(d_hid, d_hid, kernel_size=3, padding=0,
                              groups=d_hid)
        self.out = nn.Linear(d_hid, self.frames_per_row * n_classes)
        self.n_classes = int(n_classes)

    def forward(self, feat):
        if feat.dim() != 3 or feat.size(1) != N_CELLS:
            raise ValueError("feat must be (T,16,D), got %r" % (tuple(feat.shape),))
        if feat.size(2) != self.norm.normalized_shape[0]:
            raise ValueError("feat dim %d != %d" % (feat.size(2), self.norm.normalized_shape[0]))
        if feat.size(0) < 1:
            raise ValueError("empty time sequence")
        h = self.cell(self.norm(feat))
        h = F.gelu(h)
        h = h + self.pos.unsqueeze(0)
        w = self.attn(h).squeeze(-1)
        w = F.softmax(w, dim=1)
        seq = (h * w.unsqueeze(-1)).sum(dim=1)
        t = seq.size(0)
        stream = seq.transpose(0, 1).unsqueeze(0)
        if t == 1:
            padded = stream.repeat(1, 1, 3)
        else:
            padded = F.pad(stream, (1, 1), mode="replicate")
        conv = self.temp(padded)
        seq2 = conv.squeeze(0).transpose(0, 1)
        logits = self.out(seq2).reshape(t, self.frames_per_row,
                                        self.n_classes)
        return logits


def param_count(head):
    return int(sum(p.numel() for p in head.parameters()))
