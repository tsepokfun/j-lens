"""Offline Jacobian precomputation via randomised Monte-Carlo VJP.

Correct formula: J_l = E[ g ⊗ g ] where g = VJP of random-direction logits.

For each position: draw u ~ N(0, I) [vocab]; compute loss = u·logit;
then g = ∂loss/∂h_l = (∂logit/∂h_l)^T @ u  [d_model].
Accumulate outer(g, g) over all samples; final J_l = mean.
"""

import json
import random
import sys
from pathlib import Path
import torch
from safetensors.torch import save_file
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DEVICE, DTYPE, D_MODEL, NUM_LAYERS, JACOBIAN_DIR, CALIBRATION_FILE,
    CALIBRATION_SAMPLES, MAX_SEQ_LEN, JACOBIAN_BATCH_SIZE, VRAM_LIMIT_GB,
)
from core.model_loader import load_model, free_model


def load_calibration_prompts(n: int = CALIBRATION_SAMPLES) -> list[str]:
    if CALIBRATION_FILE.exists():
        with open(CALIBRATION_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            prompts = data
        elif isinstance(data, dict) and "prompts" in data:
            prompts = data["prompts"]
        else:
            prompts = []
        print(f"[jacobian] Loaded {len(prompts)} prompts from {CALIBRATION_FILE}")
        return prompts[:n]

    base_prompts = [
        "The capital of France is", "The chemical symbol for water is",
        "The largest planet in the solar system is", "The color of the sky is",
        "The speed of light is", "The currency of Japan is the",
        "Mount Everest is located in", "The square root of 64 is",
        "Shakespeare wrote Romeo and",
    ]
    prompts = [random.choice(base_prompts) for _ in range(n)]
    CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIBRATION_FILE, "w", encoding="utf-8") as fh:
        json.dump(prompts, fh, indent=2)
    return prompts


def compute_jacobians():
    model = load_model()
    tokenizer = model.tokenizer
    num_layers = len(model.blocks)
    d_model = model.cfg.d_model

    prompts = load_calibration_prompts()
    print(f"[jacobian] Using {len(prompts)} calibration prompts")
    print(f"[jacobian] d_model={d_model}, layers={num_layers}")

    running_sum: list[torch.Tensor | None] = [None] * num_layers
    sample_count = 0
    batch_size = JACOBIAN_BATCH_SIZE

    for batch_start in tqdm(range(0, len(prompts), batch_size), desc="Jacobian"):
        batch_prompts = prompts[batch_start : batch_start + batch_size]
        if not batch_prompts:
            continue

        tokens = tokenizer(batch_prompts, return_tensors="pt", padding=True,
                          truncation=True, max_length=MAX_SEQ_LEN)
        input_ids = tokens["input_ids"].to(DEVICE)
        attention_mask = tokens["attention_mask"].to(DEVICE)
        bsz, seq_len = input_ids.shape

        layer_acts: list[torch.Tensor] = []

        def make_hook():
            def hook_fn(act, hook):
                layer_acts.append(act)
                return act
            return hook_fn

        model.reset_hooks()
        for layer_idx in range(num_layers):
            hook_name = f"blocks.{layer_idx}.hook_resid_post"
            model.add_hook(hook_name, make_hook())

        try:
            output = model(input_ids, attention_mask=attention_mask)

            for seq_idx in range(bsz):
                valid_len = int(attention_mask[seq_idx].sum().item())
                n_pos = min(valid_len, MAX_SEQ_LEN)
                for pos in range(n_pos):
                    # Random direction in vocab space
                    u = torch.randn(model.cfg.d_vocab, device=DEVICE, dtype=output.dtype)
                    loss = (u * output[seq_idx, pos]).sum()

                    # VJP: g_l = d(loss)/dh_l = (dlogit/dh_l)^T @ u
                    grads = torch.autograd.grad(
                        loss, layer_acts,
                        retain_graph=True, allow_unused=True,
                    )

                    for layer_idx, grad_full in enumerate(grads):
                        if grad_full is None:
                            continue
                        g = grad_full[seq_idx, pos].detach()  # [d_model]
                        outer = torch.outer(g, g)               # [d_model, d_model]

                        if running_sum[layer_idx] is None:
                            running_sum[layer_idx] = outer.to(torch.float32).cpu()
                        else:
                            running_sum[layer_idx] += outer.to(torch.float32).cpu()

                    sample_count += 1

            model.reset_hooks()
            layer_acts.clear()

            if sample_count % 100 == 0:
                torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            print(f"\n[jacobian] OOM at batch_size={batch_size}, halving ...")
            torch.cuda.empty_cache()
            model.reset_hooks()
            layer_acts.clear()
            batch_size = max(1, batch_size // 2)
            continue
        finally:
            model.reset_hooks()
            layer_acts.clear()

    print(f"\n[jacobian] Finalising over {sample_count} samples ...")
    JACOBIAN_DIR.mkdir(parents=True, exist_ok=True)

    jacobian_dict: dict[str, torch.Tensor] = {}
    any_jac = False
    for layer_idx in range(num_layers):
        if running_sum[layer_idx] is not None:
            J_l = running_sum[layer_idx] / sample_count
            jacobian_dict[f"J_l{layer_idx:02d}"] = J_l.to(torch.float32)
            print(f"  Layer {layer_idx:02d}: J shape={J_l.shape}, norm={J_l.norm().item():.4f}")
            any_jac = True
        else:
            print(f"  Layer {layer_idx:02d}: NO data accumulated")

    if not any_jac:
        print("[jacobian] ERROR: No Jacobians computed!")
        free_model()
        return None

    out_path = JACOBIAN_DIR / "J_l.safetensors"
    save_file(jacobian_dict, out_path)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[jacobian] Saved {out_path}  ({size_mb:.1f} MB)")

    expected_bytes = d_model * d_model * 4 * num_layers
    expected_mb = expected_bytes / (1024 * 1024)
    print(f"[jacobian] Expected size ~{expected_mb:.1f} MB ({num_layers} x {d_model}x{d_model} float32)")

    free_model()
    return out_path


if __name__ == "__main__":
    compute_jacobians()
