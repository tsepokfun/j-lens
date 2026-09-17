"""J-Lens Configuration — Windows 10/11 + CUDA 12.x hardware profile."""

import os
from pathlib import Path
import torch

# ── HuggingFace cache — ONE canonical location, shared Windows/WSL ───
# Keeping a single cache means no model weights are ever stored twice.
# An HF_HOME already present in the environment always wins, so this is
# only a fallback; override it freely without editing this file.
#   Windows : %USERPROFILE%\.cache\huggingface
#   WSL     : /mnt/c/Users/<windows-user>/.cache\huggingface
if os.name == 'nt':
    os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
else:
    # WSL: reuse the Windows-side cache so the two never diverge into
    # two separate copies of the same weights.
    _win_caches = sorted(Path("/mnt/c/Users").glob("*/.cache/huggingface"))
    os.environ.setdefault(
        "HF_HOME",
        str(_win_caches[0]) if _win_caches else str(Path.home() / ".cache" / "huggingface"),
    )

# ── Paths (Windows-compatible via pathlib) ───────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
JACOBIAN_DIR = DATA_DIR / "precomputed_jacobians"
CALIBRATION_FILE = DATA_DIR / "calibration_prompts.json"

# ── Hardware ─────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VRAM_LIMIT_GB = 14.5  # leave 1.5 GB headroom for system
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

# ── Model ───────────────────────────────────────────────────────────
# Stable: "gpt2-xl" (1.5B, d=1600, L=48) — works on Windows + WSL
# Larger models need stable Linux/CUDA (WSL2 + CUDA 13.1 crashes)
MODEL_NAME = "gpt2-xl"

D_MODEL = 1600
NUM_LAYERS = 48

# ── J-Space sparse decomposition ─────────────────────────────────────
SPARSE_K = 30           # top-K occupancy dimensions
FVE_THRESHOLD = 0.85    # fraction-of-variance-explained target

# ── Intervention defaults ────────────────────────────────────────────
ALPHA_STEERING = 0.1
BETA_MERGE = 0.5
SHRINKAGE_LAMBDA = 0.05  # improves J-Lens fidelity (J' = J + λI)

# ── Jacobian computation ─────────────────────────────────────────────
CALIBRATION_SAMPLES = 200   # 200–500 short prompts
MAX_SEQ_LEN = 96
JACOBIAN_BATCH_SIZE = 1     # batch_size=1 for gpt2-xl
