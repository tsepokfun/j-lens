# J-Lens 🔬 — Sparse Transformer Interpretability Dashboard

**J-Lens** is an interactive tool for exploring how transformer language models build meaning layer by layer. It computes **J-Space Jacobians** offline, then lets you **generate text** and **click any token** to see its semantic representation across all layers in real time.

## How It Works

```
1. Offline: Precompute Jacobians J_l via Monte-Carlo VJP
   ├─ Forward pass on 200-500 calibration prompts
   ├─ Random-direction VJP at each layer
   └─ Save: J_l.safetensors (~225-800 MB)

2. Online (Gradio UI):
   ├─ Generate text with the model
   ├─ Cache all layer activations in one forward pass
   ├─ Click any token → J-Space projection per layer
   └─ See: "Which concepts does each layer associate with this token?"
```

## Architecture

```
jspace_workspace/
├── config.py                  # Model, hardware, HF cache path
├── compute_jacobians.py       # Offline Jacobian precomputation
├── main.py                    # CLI launcher (test/jacobian/ui)
├── core/
│   ├── model_loader.py        # HookedTransformer wrapper
│   ├── j_lens_engine.py       # W_U^T · J_l · h_l projection
│   ├── sparse_solver.py       # OMP decomposition over J-Space
│   ├── activation_state.py    # Pause/save/load layer snapshots
│   └── interveners.py         # Steer, swap coordinates, ablate
├── ui/
│   ├── app.py                 # Gradio dashboard
│   └── components.py          # Generate + per-token analysis
└── data/
    ├── calibration_prompts.json
    └── precomputed_jacobians/
        └── J_l.safetensors    # Final Jacobian matrix
```

## Quick Start

### Prerequisites
- Windows 10/11 or Linux with CUDA GPU (≥12 GB VRAM recommended)
- Python 3.10+ (3.12 tested)
- ~10 GB free disk space for the model (`gpt2-xl`) plus the Jacobian — see
  [Model Cache](#model-cache) for where those bytes actually land

### Install
```bash
cd jspace_workspace

# Create venv
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install transformer_lens transformers accelerate safetensors gradio plotly scikit-learn tqdm
```

### Compute Jacobians (offline, ~5 min)
```bash
python compute_jacobians.py
# → data/precomputed_jacobians/J_l.safetensors
```

### Launch Dashboard
```bash
python ui/app.py
# → http://127.0.0.1:7860
```

### Usage
1. Enter a prompt → click **Generate & Cache Activations**
2. The model generates text; each token is displayed with its position index
3. Enter a token position number → per-layer J-Space analysis appears
4. See which concepts each layer associates with that token

## Model Cache

All models live in **one** HuggingFace cache, shared by Windows and WSL:

| OS | `HF_HOME` |
|----|-----------|
| Windows | `%USERPROFILE%\.cache\huggingface` |
| WSL | `/mnt/c/Users/<windows-user>/.cache/huggingface` |

`config.py` sets this only as a **fallback** — an `HF_HOME` already present in
the environment always wins, so you can relocate the cache without touching code:

```bash
setx HF_HOME "C:\path\to\cache"     # Windows — persistent, new shells only
export HF_HOME=/path/to/cache       # WSL / bash — current session
```

Inspect it with:

```bash
python -c "from huggingface_hub import scan_cache_dir; print(round(scan_cache_dir().size_on_disk/2**30, 2), 'GiB')"
```

> **Why a single cache matters.** `huggingface_hub` stores each file's bytes
> **once** in `blobs/<sha256>` and makes `snapshots/<commit>/<file>` a symlink to
> it — but only *within one cache*. Two caches means two complete copies of the
> same weights. This repo previously pinned `HF_HOME` to `D:/huggingface_cache`,
> which drifted into a second copy alongside the default user-profile cache; the
> paths above consolidate them.
>
> Because the snapshots are symlinks, **copying a cache with a tool that follows
> symlinks (plain `robocopy`, `Copy-Item`, most file managers) will write every
> weight file a second time and roughly double the size.** Use `robocopy /SL`,
> `rsync -a`, or just re-download.

## Example Output

**Prompt:** *"The colour of the planet fourth from the Sun is"*
**Model:** gpt2-xl (1.5B, d=1600, L=48)

| Layer | Top J-Space Tokens | Interpretation |
|-------|-------------------|----------------|
| 0-10 | more, rich, very, beautiful | Generic descriptors |
| 18 | very, measured, bright, not, positive | Early semantic formation |
| 22 | very, bright, rich, **sun**, **orange** | 🔥 Mars-color concepts peak |
| 30 | **red**, dark, blue, bright, purple | Color words dominate |
| 35-47 | the, a, in, one | Grammatical convergence |

## Features

| Feature | Description |
|---------|-------------|
| 🔍 **J-Space Projection** | `logits ≈ W_U^T · J_l · h_l` — see what each layer "thinks" |
| 🎯 **Per-Token Analysis** | Click any generated token to see its layer-wise representation |
| ⏸ **Activation Pause** | Freeze any layer's activation mid-computation |
| 🔀 **Coordinate Swap** | Replace token A's J-Space coordinate with token B's |
| ✂️ **Concept Ablation** | Remove top-K J-Space directions from a layer |
| 📊 **FVE Visualization** | Per-layer variance-explained bar chart |
| 💾 **Snapshot Save/Load** | Persist layer states to disk |

## Model Compatibility

| Model | Status | Notes |
|-------|--------|-------|
| gpt2-large (774M) | ✅ Working | d=1280, L=36 |
| gpt2-xl (1.5B) | ✅ Working | d=1600, L=48 — recommended |
| pythia-2.8b | ⚠️ WSL only | Needs `from_pretrained_no_processing` |
| pythia-6.9b | ⚠️ Needs 20GB+ VRAM | WSL kernel unstable |
| Qwen-2.5 series | ❌ | Segfault on Windows/WSL |

For larger models (6B+), use native Linux with CUDA 12.x.

## WSL Setup (Experimental)

```ini
# .wslconfig (Windows side)
[wsl2]
# Leave real headroom for Windows itself.
# Asking WSL for most of the machine's RAM starves the host: it cannot keep
# its working set resident, so it pages continuously and pagefile.sys grows
# without bound until the system drive fills up. On a 32 GB machine, 16GB is
# plenty — 24GB is not, and has been observed to balloon pagefile.sys past
# 45 GB and take C: down to under 2 GB free.
memory=16GB
processors=8
```

```bash
# WSL terminal
cd /mnt/d/J-sp/jspace_workspace
python3 -m venv ~/jspace_venv
source ~/jspace_venv/bin/activate
pip install torch transformer_lens transformers gradio plotly
python compute_jacobians.py
python ui/app.py
```

WSL reads the same `HF_HOME` as Windows (the `/mnt/c/...` path in
[Model Cache](#model-cache)), so a model downloaded on one side is not
re-downloaded on the other.

## API Endpoints

| Endpoint | Function |
|----------|----------|
| `/generate_and_cache` | Generate text + cache activations |
| `/analyze_token_position` | Per-token J-Space analysis |
| `/handle_pause` | Freeze a layer |
| `/handle_freeze` | Freeze all subsequent layers |
| `/handle_save_snapshot` | Persist layer states |
| `/handle_clear` | Reset all state |

## License

MIT — see [LICENSE](LICENSE)
