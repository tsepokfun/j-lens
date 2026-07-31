"""Model loader — wraps TransformerLens for hook-safe Windows operation."""

import torch
from pathlib import Path
from transformer_lens import HookedTransformer
from config import MODEL_NAME, DEVICE, DTYPE, VRAM_LIMIT_GB


_model: HookedTransformer | None = None


def load_model() -> HookedTransformer:
    """Load (or return cached) HookedTransformer. Thread-safe for Gradio."""
    global _model
    if _model is not None:
        return _model

    print(f"[model_loader] Loading {MODEL_NAME} on {DEVICE} ({DTYPE}) ...")
    _model = HookedTransformer.from_pretrained_no_processing(
        MODEL_NAME,
        device=DEVICE,
        dtype=DTYPE,
        default_prepend_bos=True,
    )
    _model.eval()
    print(f"[model_loader] Model ready — {len(_model.blocks)} layers, d_model={_model.cfg.d_model}")
    return _model


def get_model() -> HookedTransformer:
    if _model is None:
        return load_model()
    return _model


def run_with_cache_hook(model, prompt: str, hook_name: str, hook_fn):
    """Run a single forward pass with a temporary hook; return (logits, cache).
    The hook is automatically removed after the pass to prevent handle leaks."""
    model.reset_hooks()
    model.add_hook(hook_name, hook_fn)
    try:
        logits, cache = model.run_with_cache(prompt)
    finally:
        model.reset_hooks()
    return logits, cache


def free_model():
    """Release GPU memory held by the model."""
    global _model
    if _model is not None:
        del _model
        _model = None
        torch.cuda.empty_cache()
