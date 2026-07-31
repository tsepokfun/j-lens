"""Activation state — pause, snapshot, and restore layer activations."""

import torch
from pathlib import Path
from dataclasses import dataclass, field
from config import DEVICE


@dataclass
class LayerSnapshot:
    """A frozen activation snapshot for one layer at one position."""
    layer: int
    token_pos: int
    activation: torch.Tensor  # [d_model]
    metadata: dict = field(default_factory=dict)


# Module-level registry
_paused_layers: dict[int, LayerSnapshot] = {}
_frozen: bool = False


def pause_forward_pass(layer: int, token_pos: int, activation: torch.Tensor):
    """Store a snapshot of the current layer activation."""
    global _paused_layers
    _paused_layers[layer] = LayerSnapshot(
        layer=layer,
        token_pos=token_pos,
        activation=activation.detach().clone().cpu(),
        metadata={
            "shape": list(activation.shape),
            "mean": float(activation.mean().item()),
            "std": float(activation.std().item()),
        },
    )


def is_paused(layer: int) -> bool:
    return layer in _paused_layers


def get_paused_activation(layer: int) -> torch.Tensor | None:
    snap = _paused_layers.get(layer)
    return snap.activation.to(DEVICE) if snap else None


def save_snapshot(path: str | Path):
    """Persist all paused snapshots to disk."""
    torch.save(
        {layer: snap for layer, snap in _paused_layers.items()},
        Path(path),
    )


def load_snapshot(path: str | Path):
    """Restore snapshots from disk."""
    global _paused_layers
    data = torch.load(Path(path), map_location="cpu", weights_only=False)
    _paused_layers = {int(k): v for k, v in data.items()}


def clear_snapshots():
    global _paused_layers
    _paused_layers = {}


def set_frozen(frozen: bool):
    global _frozen
    _frozen = frozen


def is_frozen() -> bool:
    return _frozen


def list_snapshots() -> list[int]:
    return sorted(_paused_layers.keys())
