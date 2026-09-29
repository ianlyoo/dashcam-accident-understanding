"""S065 video heads, metadata-only weighting/pairs, and strict Q/V LoRA state."""
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .features import BRANCH_NAMES, BRANCH_CROPS, FEATURE_DIM


def _initialize(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _validate_branch(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != 4:
        raise ValueError("head inputs must have shape (B,F,C,6144)")
    if min(value.shape[:2]) < 1 or value.shape[2:] != (BRANCH_CROPS[name], FEATURE_DIM):
        raise ValueError(f"invalid {name} input shape")
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError("head inputs must be finite floating tensors")


class VideoHead(nn.Module):
    def __init__(self, global_drop_probability: float = .5):
        super().__init__()
        if not 0 <= global_drop_probability <= 1:
            raise ValueError("global drop probability must be in [0,1]")
        self.global_drop_probability = global_drop_probability
        self.shared = nn.Sequential(nn.Linear(FEATURE_DIM, 128), nn.GELU(),
                                    nn.Linear(128, 128), nn.GELU())
        self.classifier = nn.Sequential(nn.Linear(384, 64), nn.GELU(), nn.Linear(64, 1))
        self.apply(_initialize)

    def forward(self, branches: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if set(branches) != set(BRANCH_NAMES):
            raise ValueError("expected exactly native224, native448, global branches")
        reference = branches["native224"]
        for name in BRANCH_NAMES:
            value = branches[name]
            _validate_branch(value, name)
            if value.shape[:2] != reference.shape[:2] or value.device != reference.device or value.dtype != reference.dtype:
                raise ValueError("branches must share B,F,device,dtype")
        pooled = [self.shared(branches[name]).mean(dim=(1, 2)) for name in BRANCH_NAMES]
        if self.training and self.global_drop_probability:
            # Whole branch per video; deliberately no inverted-dropout scaling.
            keep = torch.rand((reference.shape[0], 1), device=reference.device) >= self.global_drop_probability
            pooled[2] = pooled[2]*keep
        return self.classifier(torch.cat(pooled, dim=-1)).squeeze(-1)


class GlobalOnlyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(FEATURE_DIM, 128), nn.GELU(), nn.Linear(128, 1))
        self.apply(_initialize)

    def forward(self, branches: Mapping[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        value = branches["global"] if isinstance(branches, Mapping) else branches
        _validate_branch(value, "global")
        return self.classifier(value.mean(dim=(1, 2))).squeeze(-1)


def _metadata(rows: Sequence[Mapping]) -> list[tuple[str, str, int, str | None]]:
    if not len(rows):
        raise ValueError("metadata must not be empty")
    result = []
    for row in rows:
        corpus, group, label = row.get("corpus"), row.get("source_group"), row.get("label")
        if not isinstance(corpus, str) or not corpus.strip() or not isinstance(group, str) or not group.strip():
            raise ValueError("each row needs nonempty corpus and source_group strings")
        if label not in (0, 1) or isinstance(label, str):
            raise ValueError("labels must be binary 0/1")
        os_hint = row.get("os_hint")
        if os_hint is not None and not isinstance(os_hint, str):
            raise ValueError("os_hint must be string or None")
        os_hint = os_hint.strip().lower() if os_hint is not None else None
        if os_hint in ("", "unknown", "unk", "n/a", "na", "none"):
            os_hint = None
        result.append((corpus, group, int(label), os_hint))
    return result


def balanced_video_weights(rows: Sequence[Mapping]) -> torch.Tensor:
    """Mean-one weights n/(corpora * 2 * groups_in_corpus_class * files_in_group_class)."""
    metadata = _metadata(rows)
    counts, groups = defaultdict(int), defaultdict(set)
    corpora = {r[0] for r in metadata}
    for corpus, group, label, _ in metadata:
        counts[corpus, label, group] += 1
        groups[corpus, label].add(group)
    if any(not groups[corpus, label] for corpus in corpora for label in (0, 1)):
        raise ValueError("every corpus must contain both classes")
    return torch.tensor([len(rows)/(len(corpora)*2*len(groups[c, y])*counts[c, y, g])
                         for c, g, y, _ in metadata], dtype=torch.float32)


@dataclass(frozen=True)
class RankingPair:
    negative: int
    positive: int
    corpus: str
    source_group: str


def build_ranking_pairs(rows: Sequence[Mapping]) -> list[RankingPair]:
    """Same-source pairs only, without claiming pixel alignment.

    Prefer known equal OS pairs within each source group; if none exist, use
    pairs having at least one unknown OS. Known conflicting OS never pair.
    """
    metadata = _metadata(rows)
    groups = defaultdict(lambda: {0: [], 1: []})
    for index, (corpus, group, label, os_hint) in enumerate(metadata):
        groups[corpus, group][label].append((index, os_hint))
    result = []
    for (corpus, group), classes in sorted(groups.items()):
        matched, unknown = [], []
        for negative, negative_os in classes[0]:
            for positive, positive_os in classes[1]:
                pair = RankingPair(negative, positive, corpus, group)
                if negative_os is not None and positive_os is not None:
                    if negative_os == positive_os:
                        matched.append(pair)
                else:
                    unknown.append(pair)
        result.extend(matched or unknown)
    return result


def pair_ranking_loss(logits: torch.Tensor, pairs: Sequence[RankingPair]) -> torch.Tensor:
    if logits.ndim != 1 or not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("logits must be a finite floating vector")
    grouped = defaultdict(list)
    for pair in pairs:
        if not (0 <= pair.negative < len(logits) and 0 <= pair.positive < len(logits)) or pair.negative == pair.positive:
            raise ValueError("pair indices invalid for logits")
        grouped[pair.corpus, pair.source_group].append(F.softplus(logits[pair.negative]-logits[pair.positive]))
    if not grouped:
        return logits.sum()*0
    corpora = defaultdict(list)
    for (corpus, _), values in grouped.items():
        corpora[corpus].append(torch.stack(values).mean())
    return torch.stack([torch.stack(values).mean() for values in corpora.values()]).mean()


def video_loss(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor,
               pairs: Sequence[RankingPair] = ()) -> torch.Tensor:
    if logits.ndim != 1 or not len(logits) or labels.shape != logits.shape or weights.shape != logits.shape:
        raise ValueError("logits, labels and weights must be matching nonempty vectors")
    if not torch.isfinite(logits).all() or not torch.isfinite(weights).all() or not (weights > 0).all():
        raise ValueError("logits and positive weights must be finite")
    if not ((labels == 0) | (labels == 1)).all():
        raise ValueError("labels must be binary")
    bce = F.binary_cross_entropy_with_logits(logits, labels.to(logits), reduction="none")
    return (bce*weights.to(logits)).mean() + .25*pair_ranking_loss(logits, pairs)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.rank, self.alpha = 8, 8
        self.A = nn.Parameter(base.weight.new_empty((self.rank, base.in_features)))
        self.B = nn.Parameter(base.weight.new_zeros((base.out_features, self.rank)))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.base(value) + F.linear(F.linear(value, self.A), self.B)*(self.alpha/self.rank)


def inject_lora(backbone: nn.Module) -> dict[str, nn.Parameter]:
    """Inspect actual HF modules, then wrap Q/V in blocks 8..11 atomically.

    All other parameters freeze. Calling twice fails before making changes.
    """
    if any(isinstance(module, LoRALinear) for module in backbone.modules()):
        raise ValueError("adapters already injected")
    selected = {}
    for name, module in backbone.named_modules():
        parts = name.split(".")
        if len(parts) == 6 and parts[:2] == ["encoder", "layer"] and parts[3:5] == ["attention", "attention"] and parts[-1] in ("query", "value"):
            if parts[2].isdigit() and int(parts[2]) in range(8, 12):
                if not isinstance(module, nn.Linear):
                    raise ValueError(f"expected Linear: {name}")
                selected[name] = module
    expected = {f"encoder.layer.{i}.attention.attention.{kind}" for i in range(8, 12) for kind in ("query", "value")}
    if set(selected) != expected:
        raise ValueError("unsupported backbone: expected HF DINOv2 Q/V modules in blocks 8..11")
    replacements = {name: LoRALinear(module) for name, module in selected.items()}
    backbone.requires_grad_(False)
    for name, replacement in replacements.items():
        parent_path, attribute = name.rsplit(".", 1)
        setattr(backbone.get_submodule(parent_path), attribute, replacement)
    return adapter_parameters(backbone)


def adapter_parameters(backbone: nn.Module) -> dict[str, nn.Parameter]:
    return {f"{name}.{key}": parameter for name, module in backbone.named_modules()
            if isinstance(module, LoRALinear) for key, parameter in (("A", module.A), ("B", module.B))}


def adapter_state_dict(backbone: nn.Module) -> dict[str, torch.Tensor]:
    parameters = adapter_parameters(backbone)
    if not parameters:
        raise ValueError("no adapters installed")
    return {name: parameter.detach().cpu().clone() for name, parameter in parameters.items()}


def load_adapter_state_dict(backbone: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    parameters = adapter_parameters(backbone)
    if not parameters or set(state) != set(parameters):
        raise ValueError("adapter keys must match exactly")
    prepared = {}
    for name, parameter in parameters.items():
        value = state[name]
        if not isinstance(value, torch.Tensor) or value.shape != parameter.shape or value.dtype != parameter.dtype or not torch.isfinite(value).all():
            raise ValueError(f"invalid adapter tensor: {name}")
        prepared[name] = value.detach().to(parameter.device).clone()
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(prepared[name])


def save_adapter_state(backbone: nn.Module, path: str | Path) -> None:
    from safetensors.torch import save_file
    save_file(adapter_state_dict(backbone), str(path))


def load_adapter_state(backbone: nn.Module, path: str | Path) -> None:
    from safetensors.torch import load_file
    load_adapter_state_dict(backbone, load_file(str(path), device="cpu"))
