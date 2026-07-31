#!/usr/bin/env python
"""J-Lens end-to-end verification and launcher.

Usage:
    python main.py jacobian    ->  compute Jacobians (offline)
    python main.py ui          ->  launch Gradio dashboard
    python main.py test        ->  run smoke tests
    python main.py all         ->  jacobian + ui
"""

import sys
import io
import torch
from pathlib import Path

# Fix Windows CP950 encoding for unicode output
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


BASE = Path(__file__).parent


def cmd_jacobian():
    """Run offline Jacobian precomputation."""
    print("=" * 60)
    print("  Phase 3: Computing J-Lens Jacobians")
    print("=" * 60)

    from compute_jacobians import main as compute_main
    compute_main()

    j_path = BASE / "data" / "precomputed_jacobians" / "J_l.safetensors"
    if j_path.exists():
        size_mb = j_path.stat().st_size / (1024 * 1024)
        print(f"\n[OK] J_l.safetensors created - {size_mb:.1f} MB")
    else:
        print("[FAIL] J_l.safetensors NOT found! Jacobian computation failed.")
        return 1
    return 0


def cmd_ui():
    """Launch the Gradio dashboard."""
    print("=" * 60)
    print("  Phase 5: Launching J-Lens Dashboard")
    print("=" * 60)

    from ui.app import main as ui_main
    ui_main()
    return 0


def cmd_test():
    """Run smoke tests to verify core modules."""
    print("=" * 60)
    print("  Phase 6: J-Lens Smoke Tests")
    print("=" * 60)

    errors = 0

    # 1. Config
    print("\n[1/6] Testing config …")
    try:
        from config import DEVICE, D_MODEL, NUM_LAYERS, JACOBIAN_DIR
        print(f"      DEVICE={DEVICE}, D_MODEL={D_MODEL}, LAYERS={NUM_LAYERS}")
    except Exception as e:
        print(f"      [FAIL] {e}")
        errors += 1

    # 2. Model loader
    print("\n[2/6] Testing model_loader ...")
    try:
        from core.model_loader import load_model
        model = load_model()
        print(f"      Model loaded: {len(model.blocks)} layers, d_model={model.cfg.d_model}")
    except Exception as e:
        print(f"      [WARN] {type(e).__name__} (may be OK if model not yet downloaded)")
        # Not counted as error — model download may require network

    # 3. J-Lens engine
    print("\n[3/6] Testing j_lens_engine ...")
    try:
        from core.j_lens_engine import lens_projection, compute_fve
        d = D_MODEL
        h_test = torch.randn(d)
        J_test = torch.randn(d, d) * 0.1
        W_test = torch.randn(1000, d)
        logits = lens_projection(h_test, J_test, W_test)
        assert logits.shape[-1] == 1000, f"Expected 1000, got {logits.shape}"
        print(f"      lens_projection output shape: {logits.shape} [OK]")
    except Exception as e:
        print(f"      [FAIL] {e}")
        errors += 1

    # 4. Sparse solver
    print("\n[4/6] Testing sparse_solver ...")
    try:
        from core.sparse_solver import sparse_decompose
        h_test = torch.randn(D_MODEL)
        J_test = torch.randn(D_MODEL, D_MODEL) * 0.1
        W_test = torch.randn(1000, D_MODEL)
        x, residual, fve, active = sparse_decompose(h_test, J_test, W_test, k=5)
        assert 0.0 <= fve <= 1.0, f"FVE out of range: {fve}"
        print(f"      FVE={fve:.4f}, active={len(active)}/{5} [OK]")
    except Exception as e:
        print(f"      [FAIL] {e}")
        errors += 1

    # 5. Interveners
    print("\n[5/6] Testing interveners ...")
    try:
        from core.interveners import steer, swap_coordinates, ablate_top_concepts
        h_test = torch.randn(D_MODEL)
        J_test = torch.randn(D_MODEL, D_MODEL) * 0.1
        W_test = torch.randn(1000, D_MODEL)

        h_steered = steer(h_test, J_test, W_test, token_id=42, alpha=0.1)
        assert h_steered.shape == h_test.shape

        h_swapped = swap_coordinates(h_test, 42, 99, J_test, W_test)
        assert h_swapped.shape == h_test.shape

        h_ablated, ids = ablate_top_concepts(h_test, J_test, W_test, k=3)
        assert h_ablated.shape == h_test.shape
        assert len(ids) == 3
        print(f"      steer, swap, ablate all pass [OK]")
    except Exception as e:
        print(f"      [FAIL] {e}")
        errors += 1

    # 6. Activation state
    print("\n[6/6] Testing activation_state ...")
    try:
        from core.activation_state import (
            pause_forward_pass, is_paused, get_paused_activation,
            save_snapshot, load_snapshot, clear_snapshots,
        )
        import tempfile
        h_test = torch.randn(D_MODEL)
        pause_forward_pass(18, 5, h_test)
        assert is_paused(18)
        act = get_paused_activation(18)
        assert act is not None and act.shape[-1] == D_MODEL

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp = f.name
        save_snapshot(tmp)
        clear_snapshots()
        assert not is_paused(18)
        load_snapshot(tmp)
        assert is_paused(18)
        Path(tmp).unlink()
        clear_snapshots()
        print(f"      pause/save/load/clear all pass [OK]")
    except Exception as e:
        print(f"      [FAIL] {e}")
        errors += 1

    # -- Summary --
    print("\n" + "=" * 60)
    if errors == 0:
        print("  [OK] ALL SMOKE TESTS PASSED")
    else:
        print(f"  [FAIL] {errors} test(s) FAILED")
    print("=" * 60)
    return errors


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    cmd = sys.argv[1].lower()
    if cmd == "jacobian":
        return cmd_jacobian()
    elif cmd == "ui":
        return cmd_ui()
    elif cmd == "test":
        return cmd_test()
    elif cmd == "all":
        ret = cmd_jacobian()
        if ret != 0:
            return ret
        return cmd_ui()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        return 1


if __name__ == "__main__":
    sys.exit(main())
