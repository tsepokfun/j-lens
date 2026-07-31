"""Interveners — steer, coordinate-swap, and ablate activation directions."""

import torch
from config import DEVICE, DTYPE, D_MODEL


def steer(
    h_l: torch.Tensor,
    J_l: torch.Tensor,
    W_U: torch.Tensor,
    token_id: int,
    alpha: float = 0.1,
) -> torch.Tensor:
    """Steer activation toward a target token's J-Lens direction.

    h_l' = h_l + α · (J_l^T · W_U[token_id])
    """
    h_l = h_l.to(device=DEVICE, dtype=DTYPE)
    J_l = J_l.to(device=DEVICE, dtype=DTYPE)
    W_U = W_U.to(device=DEVICE, dtype=DTYPE)

    # W_U is [d_model, vocab] in TransformerLens
    direction = J_l.T @ W_U[:, token_id]           # [d_model]
    direction = direction / (direction.norm() + 1e-8)  # unit-norm
    return h_l + alpha * direction


def swap_coordinates(
    h_l: torch.Tensor,
    source_token_id: int,
    target_token_id: int,
    J_l: torch.Tensor,
    W_U: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Swap the J-Space coordinate of source_token with target_token.

    Projects h_l onto the J-Lens basis, replaces the source coordinate
    with the target coordinate value, then reconstructs.

    Args:
        h_l:              activation [d_model]
        source_token_id:  token whose coordinate is replaced
        target_token_id:  token whose coordinate value is used as replacement
        J_l:              Jacobian  [d_model, d_model]
        W_U:              unembedding [vocab, d_model]
        alpha:            blend strength (1.0 = full swap)

    Returns:
        Modified activation [d_model]
    """
    h_l = h_l.to(device=DEVICE, dtype=DTYPE)
    J_l = J_l.to(device=DEVICE, dtype=DTYPE)
    W_U = W_U.to(device=DEVICE, dtype=DTYPE)

    # J-Lens basis vectors (rows of W_U.T @ J_l)
    basis = W_U.T @ J_l  # [vocab, d_model]

    # Compute coordinates: c_i = dot(basis[i], h_l) / ||basis[i]||²
    basis_norm2 = (basis ** 2).sum(dim=-1) + 1e-8  # [vocab]
    coordinates = (basis @ h_l) / basis_norm2      # [vocab]

    source_coord = coordinates[source_token_id]
    target_coord = coordinates[target_token_id]

    # Remove source direction, add target direction
    basis_source = basis[source_token_id]
    basis_target = basis[target_token_id]

    h_modified = h_l - source_coord * basis_source + alpha * target_coord * basis_target

    return h_modified


def ablate_top_concepts(
    h_l: torch.Tensor,
    J_l: torch.Tensor,
    W_U: torch.Tensor,
    k: int = 5,
) -> tuple[torch.Tensor, list[int]]:
    """Project out the top-k J-Lens directions from h_l.

    Identifies the k directions with largest absolute coordinate,
    then subtracts their contributions. Focus on mid-layer (35%–85%)
    where semantic concepts typically reside.

    Returns:
        h_ablated:   activation with top-k directions removed
        ablated_ids: token ids that were ablated
    """
    h_l = h_l.to(device=DEVICE, dtype=DTYPE)
    J_l = J_l.to(device=DEVICE, dtype=DTYPE)
    W_U = W_U.to(device=DEVICE, dtype=DTYPE)

    basis = W_U.T @ J_l  # [vocab, d_model]
    basis_norm2 = (basis ** 2).sum(dim=-1) + 1e-8
    coordinates = (basis @ h_l) / basis_norm2  # [vocab]
    abs_coords = coordinates.abs()

    _, top_indices = torch.topk(abs_coords, k=k)

    h_ablated = h_l.clone()
    for idx in top_indices:
        coord = coordinates[idx]
        h_ablated = h_ablated - coord * basis[idx]

    return h_ablated, top_indices.tolist()
