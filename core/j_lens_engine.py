"""J-Lens engine — projection, top-token lookup, and shrinkage."""

import torch
import torch.nn.functional as F
from config import DEVICE, DTYPE, D_MODEL, SHRINKAGE_LAMBDA


def lens_projection(
    h_l: torch.Tensor,
    J_l: torch.Tensor,
    W_U: torch.Tensor,
    shrinkage_lambda: float = SHRINKAGE_LAMBDA,
) -> torch.Tensor:
    """Apply J-Lens projection:  logit_l ≈ W_U · J_l' · h_l

    Args:
        h_l:     residual stream activation at layer l   [d_model] or [batch, d_model]
        J_l:     precomputed mean Jacobian                [d_out, d_model]
        W_U:     unembedding matrix                       [vocab, d_model]
        shrinkage_lambda:  regularisation for J' = J + λI

    Returns:
        logits:  [vocab] or [batch, vocab]
    """
    # Ensure 2-D
    squeeze = h_l.dim() == 1
    if squeeze:
        h_l = h_l.unsqueeze(0)

    h_l = h_l.to(device=DEVICE, dtype=DTYPE)
    J_l = J_l.to(device=DEVICE, dtype=DTYPE)
    W_U = W_U.to(device=DEVICE, dtype=DTYPE)

    # J' = J_l + λI  (shrinkage for fidelity)
    eye = torch.eye(J_l.shape[0], device=DEVICE, dtype=DTYPE)
    J_prime = J_l + shrinkage_lambda * eye

    # logit_l = W_U.T @ J_prime @ h_l  (W_U is [d_model, vocab] in TLens)
    jh = J_prime @ h_l.T                   # [d_model, batch]
    logits = (W_U.T @ jh).T                # [batch, vocab]

    if squeeze:
        logits = logits.squeeze(0)
    return logits


def get_top_tokens(
    h_l: torch.Tensor,
    J_l: torch.Tensor,
    W_U: torch.Tensor,
    tokenizer,
    k: int = 10,
) -> list[tuple[str, float]]:
    """Return top-k tokens and their log-probabilities from the J-Lens projection.

    Args:
        h_l:          activation [d_model]
        J_l:          Jacobian  [d_model, d_model]
        W_U:          unembedding [vocab, d_model]
        tokenizer:    HF / TransformerLens tokenizer
        k:            number of top tokens

    Returns:
        List of (token_str, log_prob) sorted descending.
    """
    with torch.no_grad():
        logits = lens_projection(h_l, J_l, W_U)
        log_probs = F.log_softmax(logits, dim=-1)
        topk = torch.topk(log_probs, k=k, dim=-1)
        tokens = [
            (tokenizer.decode([idx.item()]).strip(), val.item())
            for idx, val in zip(topk.indices, topk.values)
        ]
    return tokens


def compute_fve(h_l: torch.Tensor, residual: torch.Tensor) -> float:
    """Fraction of variance explained: 1 - ||res||² / ||h_l||²"""
    h_norm2 = (h_l ** 2).sum().item()
    res_norm2 = (residual ** 2).sum().item()
    if h_norm2 == 0:
        return 0.0
    return max(0.0, 1.0 - res_norm2 / h_norm2)


def compute_jlens_fve(
    h_l: torch.Tensor,
    J_l: torch.Tensor,
    W_U: torch.Tensor,
    k: int = 30,
) -> tuple[float, list[int], torch.Tensor]:
    """Compute FVE as top-K concentration of J-Lens projection energy.

    Instead of sparse reconstruction of h_l (which fails when K << d_model),
    measures how concentrated the J-Lens logit projection is in top-K tokens.

    FVE = ||logits_topk||² / ||logits_full||²

    A high FVE means the activation strongly activates a few specific token
    directions; a low FVE means the projection is diffuse across many tokens.

    Returns:
        fve:          fraction of projection energy in top-K (0-1)
        active_ids:   token ids of top-K
        coefficients:  not used (kept for API compatibility)
    """
    h_l = h_l.to(device=DEVICE, dtype=torch.float32)
    J_l = J_l.to(device=DEVICE, dtype=torch.float32)
    W_U = W_U.to(device=DEVICE, dtype=torch.float32)

    # J-Lens projection
    D_full = W_U.T @ J_l                                 # [vocab, d_model]
    logits = D_full @ h_l                                # [vocab]

    # Top-K concentration
    topk_vals, top_k = torch.topk(logits, k=k)           # [K]

    total_energy = (logits ** 2).sum()
    topk_energy = (topk_vals ** 2).sum()
    fve = (topk_energy / total_energy).item() if total_energy > 0 else 0.0
    fve = max(0.0, min(1.0, fve))

    active_ids = top_k.tolist()

    return fve, active_ids, topk_vals
