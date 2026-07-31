"""Sparse solver — non-negative OMP decomposition of h_l over the J-Lens dictionary."""

import torch
from config import DEVICE, DTYPE, SPARSE_K


def _omp_solve_python(
    h_l: torch.Tensor,
    D: torch.Tensor,
    k: int = SPARSE_K,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Pure-Python GPU OMP (Orthogonal Matching Pursuit).

    Solves   min ||h - D @ x||²   s.t.  x >= 0,  ||x||₀ <= k

    Args:
        h_l:  activation vector    [d_model]
        D:    dictionary           [d_model, k_dim] — columns are dictionary atoms
        k:    sparsity target

    Returns:
        x:        sparse coefficient vector  [k_dim]
        residual: h - D @ x                  [d_model]
        fve:      fraction of variance explained
    """
    # Use float32 internally — lstsq doesn't support bfloat16
    compute_dtype = torch.float32
    h_l = h_l.to(device=DEVICE, dtype=compute_dtype)
    D = D.to(device=DEVICE, dtype=compute_dtype)

    d_model = h_l.shape[0]
    k_dim = D.shape[1]

    residual = h_l.clone()
    support: list[int] = []
    x = torch.zeros(k_dim, device=DEVICE, dtype=compute_dtype)

    for _ in range(k):
        # Correlation with residual
        corr = (D.T @ residual).abs()  # [k_dim]
        # Exclude already-selected atoms
        for s in support:
            corr[s] = -float("inf")
        best = int(corr.argmax().item())
        support.append(best)

        # Solve non-negative least squares on support: min ||h - D_s @ x_s||², x_s >= 0
        D_s = D[:, support]  # [d_model, |S|]
        # lstsq requires float32 or float64
        x_s = torch.linalg.lstsq(D_s, h_l.unsqueeze(1)).solution.squeeze(1)

        # Clamp negative to 0 (simple NNLS — for production use `batched-omp`)
        x_s = torch.clamp(x_s, min=0.0)
        for i, s_idx in enumerate(support):
            x[s_idx] = x_s[i]

        residual = h_l - D @ x

    # Final FVE
    h_norm2 = (h_l ** 2).sum().item()
    res_norm2 = (residual ** 2).sum().item()
    fve = max(0.0, 1.0 - res_norm2 / h_norm2) if h_norm2 > 0 else 0.0

    return x, residual, fve


def sparse_decompose(
    h_l: torch.Tensor,
    J_l: torch.Tensor,
    W_U: torch.Tensor,
    k: int = SPARSE_K,
) -> tuple[torch.Tensor, torch.Tensor, float, list[int]]:
    """Decompose h_l over the J-Lens dictionary (W_U @ J_l).

    Args:
        h_l:  activation       [d_model]
        J_l:  Jacobian         [d_model, d_model]
        W_U:  unembedding      [vocab, d_model]
        k:    sparsity target

    Returns:
        x:          sparse coefficients   [vocab]
        residual:   reconstruction error  [d_model]
        fve:        fraction of variance explained
        active_ids: indices of non-zero coefficients
    """
    # W_U is [d_model, vocab]; D = (W_U.T @ J_l).T gives [d_model, vocab] columns as atoms
    h_l = h_l.to(device=DEVICE, dtype=DTYPE)
    J_l = J_l.to(device=DEVICE, dtype=DTYPE)
    W_U = W_U.to(device=DEVICE, dtype=DTYPE)
    D = (W_U.T @ J_l).T
    x, residual, fve = _omp_solve_python(h_l, D, k)
    active_ids = torch.nonzero(x > 1e-6).squeeze(-1).tolist()
    if isinstance(active_ids, int):
        active_ids = [active_ids]
    return x, residual, fve, active_ids
