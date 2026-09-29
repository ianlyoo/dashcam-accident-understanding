"""One decoded RGB sequence, two geometries, one verified frozen MLP/backbone.

Uses the Base-bound core with S079's unchanged repaired decoder and geometry.
Final probability is exactly (float(native_p) + float(canonical_p)) / 2.
"""
import hashlib
import math
import time

import torch

from .single_runtime import (
    Predictor as _SinglePredictor, decode_video, _fp32_context, _config,
    f, canonical, BACKEND_POLICY,
)


class Predictor(_SinglePredictor):
    def __init__(self, backbone_path, head_path, config_path, device="cuda:0"):
        if _config(config_path)["head_type"] != "mlp":
            raise ValueError("Dual geometry requires the shared MLP head")
        super().__init__(backbone_path, head_path, config_path, device=device)
        if self.head_type != "mlp":
            raise ValueError("MLP config changed during initialization")

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

    def predict_video(self, path, debug=False):
        if not isinstance(debug, bool):
            raise ValueError("debug must be bool")
        started = time.perf_counter()
        frames, decode = decode_video(path, debug=debug)
        decoded = time.perf_counter()
        with _fp32_context(self.device), torch.no_grad():
            native_p, native_geometry, native_hashes = self._geometry_probability(frames, debug)
            canonical_frames = canonical.canonicalize_frames(frames)
            canonical_p, canonical_geometry, canonical_hashes = self._geometry_probability(canonical_frames, debug)
            probability = (native_p + canonical_p) / 2
            diagnostics = None
            if debug:
                diagnostics = dict(
                    selected_rgb_sha256=[sample["rgb_sha256"] for sample in decode["samples"]],
                    canonical_rgb_sha256=[hashlib.sha256(frame.tobytes()).hexdigest() for frame in canonical_frames],
                    descriptor_sha256=canonical_hashes,
                    geometry_descriptor_sha256=dict(native=native_hashes, canonical=canonical_hashes),
                    geometry=dict(native=native_geometry, canonical=canonical_geometry),
                )
        finished = time.perf_counter()
        result = dict(probability=probability, prediction=int(probability >= .5), threshold=.5,
                      component_probabilities=dict(native=native_p, canonical=canonical_p),
                      decode=dict(decode, views=72, geometry_views=dict(native=36, canonical=36)),
                      timing=dict(decode_seconds=decoded-started, inference_seconds=finished-decoded,
                                  total_seconds=finished-started))
        if debug:
            result["debug"] = diagnostics
        return result
