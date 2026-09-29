"""Fixed S093 Small / S090 Base probability ensemble over shared decoded RGB.

The builder supplies the scored S079 Small core and the complete S090 package.
Each backbone evaluates native and canonical geometry (72 views per backbone).
"""
import hashlib
import math
import time
from pathlib import Path

import torch

from . import single_runtime as small_core
from .base_runtime import runtime as base_core

BACKEND_POLICY = small_core.BACKEND_POLICY
f = small_core.f

# Sealed S093 and S090 export_plan.json identities, never caller-selected weights.
EXPECTED_IDENTITIES = {
    "small": {
        "head_sha256": "1636c66203a222663b66581fa77d81b75a1e8e5b443e514ed24088fb6b210eb3",
        "backbone_sha256": "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1",
        "backbone_config_sha256": "1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d",
        "head_type": "mlp",
    },
    "base": {
        "head_sha256": "01787ab548b69cb7f7f92291610dff6534a8818cdfd4f72bd9717f43dc555566",
        "backbone_sha256": "d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841",
        "backbone_config_sha256": "f7ff4cfa73d2f70647dbf6950541ad25d73082d54c2e7e9bded160c7656b2a70",
        "head_type": "mlp",
    },
}


def _check_identity(family, identity):
    for key, expected in EXPECTED_IDENTITIES[family].items():
        if identity.get(key) != expected:
            raise ValueError(f"S094 {family} {key} identity mismatch")


def _check_probability(probability):
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Nonfinite/out-of-range ensemble probability; no default produced")


class _SmallPredictor(small_core.Predictor):
    # Exact S079 runtime._geometry_probability; only the owning class differs.
    def _geometry_probability(self, frames, debug):
        branches, geometry = f.extract_frozen_features(
            frames, self.backbone, max_batch=8, output_device=self.device)
        logits = self.head({name: value.unsqueeze(0) for name, value in branches.items()})
        if logits.shape != (1,) or not torch.isfinite(logits).all():
            raise ValueError("Nonfinite or invalid batch1 MLP logit; no default produced")
        probability = float(logits.sigmoid().item())
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Nonfinite/out-of-range geometry probability")
        hashes = ({name: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                   for name, value in branches.items()} if debug else None)
        return probability, geometry, hashes


class Predictor:
    def __init__(self, backbone_path, head_path, config_path, device="cuda:0"):
        base_root = Path(backbone_path).resolve().parent.parent / "stage1_dino_base"
        base_backbone = base_root / "backbone"
        base_head = base_root / "head.bin"
        base_config = base_root / "runtime_config.json"

        # Both configurations must name the exact constituents before loading.
        # Core constructors authenticate all actual backbone/config/head bytes.
        _check_identity("small", small_core._config(config_path))
        _check_identity("base", base_core._config(base_config))
        self.small = _SmallPredictor(backbone_path, head_path, config_path, device=device)
        self.base = base_core.Predictor(base_backbone, base_head, base_config, device=device)
        _check_identity("small", self.small.identities)
        _check_identity("base", self.base.identities)
        self.device = self.small.device
        if self.base.device != self.device:
            raise ValueError("S094 constituent devices differ")
        self.identities = {"small": self.small.identities, "base": self.base.identities}

    def predict_video(self, path, debug=False):
        if not isinstance(debug, bool):
            raise ValueError("debug must be bool")  # noqa: TRY004 - preserve scored caller contract
        started = time.perf_counter()
        frames, decode = small_core.decode_video(path, debug=debug)
        decoded = time.perf_counter()
        components, families, models = {}, {}, {}
        with small_core._fp32_context(self.device), torch.no_grad():
            canonical_frames = small_core.canonical.canonicalize_frames(frames)
            for family, predictor in (("small", self.small), ("base", self.base)):
                native_p, native_geometry, native_hashes = predictor._geometry_probability(frames, debug)
                _check_probability(native_p)
                canonical_p, canonical_geometry, canonical_hashes = predictor._geometry_probability(canonical_frames, debug)
                _check_probability(canonical_p)
                components[family] = {"native": native_p, "canonical": canonical_p}
                families[family] = (native_p + canonical_p) / 2
                _check_probability(families[family])
                if debug:
                    models[family] = {
                        "geometry_descriptor_sha256": {"native": native_hashes, "canonical": canonical_hashes},
                        "descriptor_sha256": canonical_hashes,
                        "geometry": {"native": native_geometry, "canonical": canonical_geometry},
                    }
            probability = (families["base"] + families["small"]) / 2
            _check_probability(probability)
            if debug:
                diagnostics = {
                    "selected_rgb_sha256": [sample["rgb_sha256"] for sample in decode["samples"]],
                    "canonical_rgb_sha256": [hashlib.sha256(frame.tobytes()).hexdigest() for frame in canonical_frames],
                    "models": models,
                }
        finished = time.perf_counter()
        result = {
            "probability": probability, "prediction": int(probability >= .5), "threshold": .5,
            "family_probabilities": families, "component_probabilities": components,
            "decode": dict(decode, views=144, geometry_views={"native": 72, "canonical": 72},
                        backbone_views={"small": 72, "base": 72}),
            "timing": {"decode_seconds": decoded-started, "inference_seconds": finished-decoded,
                        "total_seconds": finished-started},
        }
        if debug:
            result["debug"] = diagnostics
        return result
