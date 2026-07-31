"""Gradio UI components — generate, click-token analysis, layer table, interventions."""

import gradio as gr
import torch
import torch.nn.functional as F
from typing import Any

from config import DEVICE, DTYPE, SPARSE_K, NUM_LAYERS
from core.model_loader import get_model
from core.j_lens_engine import lens_projection, get_top_tokens, compute_fve
from core.sparse_solver import sparse_decompose
from core.activation_state import (
    pause_forward_pass, is_paused, get_paused_activation,
    clear_snapshots, save_snapshot, load_snapshot, list_snapshots,
    set_frozen, is_frozen,
)
from core.interveners import steer, swap_coordinates, ablate_top_concepts


# -- Global state for generated sequence --
_generated_text: str = ""
_generated_tokens: list[str] = []
_generated_token_ids: list[int] = []
_generated_cache: Any = None
_generated_prompt_len: int = 0


# -- Jacobian loader (cached) --
_jacobian_dict: dict | None = None


def _load_jacobians() -> dict:
    global _jacobian_dict
    if _jacobian_dict is not None:
        return _jacobian_dict
    from config import JACOBIAN_DIR
    from safetensors.torch import load_file
    j_path = JACOBIAN_DIR / "J_l.safetensors"
    if not j_path.exists():
        raise FileNotFoundError(f"Jacobian not found: {j_path}")
    _jacobian_dict = load_file(str(j_path))
    return _jacobian_dict


# -- Generation --
def generate_and_cache(
    prompt: str,
    max_new_tokens: int = 30,
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> tuple[str, str, str, Any]:
    global _generated_text, _generated_tokens, _generated_token_ids, _generated_cache, _generated_prompt_len

    model = get_model()
    tokenizer = model.tokenizer

    # Generate text (returns string)
    full_text = model.generate(
        prompt,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        do_sample=True,
        prepend_bos=True,
        verbose=False,
    )

    # Re-tokenize to get token IDs
    token_ids_tensor = model.to_tokens(full_text)
    total_len = token_ids_tensor.shape[1]

    # Run forward with cache to capture all activations
    _, cache = model.run_with_cache(
        token_ids_tensor,
        names_filter=lambda name: name.endswith("hook_resid_post"),
    )

    # Also tokenize just the prompt to know where it ends
    prompt_tokens = model.to_tokens(prompt)
    _generated_prompt_len = prompt_tokens.shape[1]

    # Extract token strings for each position
    token_strs: list[str] = []
    token_ids_list: list[int] = []
    for pos in range(total_len):
        tid = int(token_ids_tensor[0, pos].item())
        token_ids_list.append(tid)
        tok_str = tokenizer.decode([tid])
        tok_str = tok_str.replace("\n", "\n").replace("|", "\|")
        token_strs.append(tok_str)

    # Build HTML token display
    html_parts = ['<div style="line-height:2.2;padding:10px;border:1px solid #ddd;border-radius:8px;background:#fafafa;font-family:monospace;font-size:13px">']
    for pos in range(total_len):
        if pos < _generated_prompt_len:
            bg = "#eee"
            color = "#666"
            label = "P"
        else:
            bg = "#d4f0ff"
            color = "#000"
            label = "G"
        html_parts.append(
            f'<span style="background:{bg};color:{color};padding:2px 5px;margin:1px;border-radius:3px;display:inline-block" '
            f'title="Position {pos} ({label}): {token_strs[pos]}">'
            f'<sup style="font-size:9px;color:#888">[{pos}]</sup> {token_strs[pos]}'
            f'</span>'
        )
    html_parts.append('</div>')
    html_parts.append(f'<p style="font-size:12px;color:#888">P=prompt ({_generated_prompt_len} tokens), G=generated ({total_len - _generated_prompt_len} tokens)</p>')
    token_html = "\n".join(html_parts)

    # Store globally
    _generated_text = full_text
    _generated_tokens = token_strs
    _generated_token_ids = token_ids_list
    _generated_cache = cache

    # Build dropdown choices
    choices = [f"Position {pos}: \"{token_strs[pos]}\"" for pos in range(total_len)]

    status = (
        f"Generated {total_len} tokens ({_generated_prompt_len} prompt + {total_len - _generated_prompt_len} new). "
        f"Select a position below to see J-Lens per-layer analysis."
    )

    return full_text, token_html, status, total_len  # total token count for position max


# -- Per-token analysis --
def analyze_token_position(
    position: int,
    k_sparse: int,
    alpha: float,
    beta: float,
    lambda_val: float,
    source_token: str,
    target_token: str,
    intervention_mode: str,
    pause_layer: int,
    merge_layer_a: int,
    merge_layer_b: int,
) -> tuple[str, str, Any]:
    global _generated_cache, _generated_tokens

    if _generated_cache is None or not _generated_tokens:
        return "*No generation yet. Click Generate first.*", "No data", None

    model = get_model()
    tokenizer = model.tokenizer
    W_U = model.unembed.W_U
    j_dict = _load_jacobians()
    num_layers = min(len(model.blocks), len(j_dict))
    cache = _generated_cache

    # Parse position
    pos = int(position)
    pos = max(0, min(pos, len(_generated_tokens) - 1))
    token_at_pos = _generated_tokens[pos]

    table_rows = [
        f"### J-Lens per-layer analysis for token **[{pos}] \"{token_at_pos}\"**",
        "",
        "| Layer | FVE% | Top Tokens |",
        "|-------|------|------------|",
    ]

    all_fves = []
    fve_data = {}

    for layer_idx in range(num_layers):
        hook_name = f"blocks.{layer_idx}.hook_resid_post"
        h_l = cache[hook_name][0, pos].clone()

        J_l_key = f"J_l{layer_idx:02d}"
        if J_l_key not in j_dict:
            table_rows.append(f"| {layer_idx:02d} | -- | -- |")
            continue

        J_l = j_dict[J_l_key]

        # FVE via J-Lens top-K probability mass
        logits_proj = lens_projection(h_l, J_l, W_U)
        probs = F.softmax(logits_proj, dim=-1)
        topk_probs, _ = torch.topk(probs, k=int(k_sparse))
        fve_pct = topk_probs.sum().item() * 100
        all_fves.append(fve_pct)
        fve_data[layer_idx] = fve_pct

        # Top tokens
        top_tokens = get_top_tokens(h_l, J_l, W_U, tokenizer, k=5)
        token_str = ", ".join([f'"{t}"' for t, _ in top_tokens[:5]])

        table_rows.append(f"| {layer_idx:02d} | {fve_pct:.1f}% | {token_str} |")

    table_md = "\n".join(table_rows)

    status_msg = (
        f"Token [{pos}] \"{token_at_pos}\" | {num_layers} layers | "
        f"Avg FVE: {sum(all_fves)/max(1,len(all_fves)):.1f}%"
    )

    # Plot
    import plotly.graph_objects as go
    fig = go.Figure()
    layers_list = list(fve_data.keys())
    fves_list = [fve_data[l] for l in layers_list]
    fig.add_trace(go.Bar(x=layers_list, y=fves_list, marker_color="steelblue", name="FVE%"))
    fig.update_layout(
        title=f"Per-Layer FVE for token [{pos}] \"{token_at_pos}\"",
        xaxis_title="Layer", yaxis_title="FVE %", height=350, template="plotly_white",
    )

    return table_md, status_msg, fig


# -- Intervention handlers --
def handle_pause(prompt: str, layer: int):
    model = get_model()
    tokens = model.to_tokens(prompt)
    _, cache = model.run_with_cache(tokens, names_filter=lambda n: n.endswith("hook_resid_post"))
    hook_name = f"blocks.{layer}.hook_resid_post"
    h_l = cache[hook_name][0, -1]
    pause_forward_pass(layer, tokens.shape[1] - 1, h_l)
    return f"Paused layer {layer}. Snapshots: {list_snapshots()}"

def handle_save_snapshot():
    from config import DATA_DIR
    path = DATA_DIR / "snapshot.pt"
    save_snapshot(path)
    return f"Saved {len(list_snapshots())} snapshots to {path}"

def handle_clear():
    clear_snapshots()
    set_frozen(False)
    return "Cleared all snapshots and unfroze."

def handle_freeze():
    set_frozen(True)
    return "Frozen - subsequent layers will use paused activations."
