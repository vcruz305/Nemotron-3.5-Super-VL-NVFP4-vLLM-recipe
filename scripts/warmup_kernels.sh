#!/usr/bin/env bash
set -euo pipefail
# Pre-compile, validate, and manage FlashInfer and PyTorch Inductor kernels
# for GB10 Blackwell SM120.
#
# Usage:
#   ./scripts/warmup_kernels.sh          # Verify cache, compile missing/updated ops
#   ./scripts/warmup_kernels.sh --force  # Clean existing cache and force full rebuild
#   ./scripts/warmup_kernels.sh --check  # Quick validation test only

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Auto-source .env if present
if [[ -f "$ROOT/.env" ]]; then
  _backup_vars=()
  for _var in VLLM_BIN MODEL_DIR MAX_JOBS; do
    if [[ -n "${!_var+x}" ]]; then _backup_vars+=("$_var=${!_var}"); fi
  done
  set -a
  # shellcheck source=/dev/null
  source "$ROOT/.env"
  set +a
  for _entry in "${_backup_vars[@]}"; do export "$_entry"; done
  unset _backup_vars _entry _var
fi

MODE="warmup"
for arg in "$@"; do
  case "$arg" in
    --force|--clean) MODE="force" ;;
    --check) MODE="check" ;;
    -h|--help)
      echo "Usage: $0 [--force|--check]"
      echo "  --force : Remove existing FlashInfer cache and rebuild all kernels cleanly"
      echo "  --check : Validate existing compiled kernels without compiling"
      exit 0
      ;;
  esac
done

VLLM_BIN="${VLLM_BIN:-$HOME/vllm31-venv/bin/vllm}"
MAX_JOBS="${MAX_JOBS:-4}"
export MAX_JOBS

if [[ -x "$VLLM_BIN" ]]; then
  PYTHON_BIN="$(dirname "$VLLM_BIN")/python3"
else
  PYTHON_BIN="python3"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Error: Python executable not found at $PYTHON_BIN" >&2
  exit 1
fi

export PATH="$(dirname "$PYTHON_BIN"):/usr/local/cuda/bin:$PATH"
CACHE_BASE="$HOME/.cache/flashinfer"

if [[ "$MODE" == "force" ]]; then
  echo "=== Force Mode: Cleaning FlashInfer JIT cache ==="
  if [[ -d "$CACHE_BASE" ]]; then
    rm -rf "$CACHE_BASE"
    echo "  Cleaned $CACHE_BASE"
  fi
fi

echo "=== Step 1/3: Checking compilation prerequisites ==="

# Check for Python.h (required by PyTorch Inductor for multimodal compilation)
"$PYTHON_BIN" - <<'PY'
import os, sys, sysconfig
include_dir = sysconfig.get_path("include")
header = os.path.join(include_dir, "Python.h")
if not os.path.isfile(header):
    print(f"\n[ERROR] Missing {header}", file=sys.stderr)
    print("PyTorch Inductor requires python C headers for multimodal extensions.", file=sys.stderr)
    print("Please install python3-dev via: sudo apt-get install -y python3-dev\n", file=sys.stderr)
    sys.exit(1)
print(f"  [OK] Found Python.h at {header}")
PY

# Check for nvcc and ninja
if command -v nvcc &>/dev/null; then
  echo "  [OK] Found nvcc at $(command -v nvcc) ($(nvcc --version | grep -o 'release [0-9.]*'))"
else
  echo "[ERROR] nvcc not found on PATH. Ensure /usr/local/cuda/bin is on PATH." >&2
  exit 1
fi

if command -v ninja &>/dev/null; then
  echo "  [OK] Found ninja at $(command -v ninja)"
else
  echo "[ERROR] ninja not found. Ensure ninja is installed in the python environment." >&2
  exit 1
fi

echo ""
echo "=== Step 2/3: Checking & Updating FlashInfer SM120 Modules (MAX_JOBS=$MAX_JOBS) ==="

"$PYTHON_BIN" - "$MODE" <<'PY'
import ctypes
import os
import sys
import time
import flashinfer
import flashinfer.jit.gemm as g
import flashinfer.jit.fused_moe as m

mode = sys.argv[1] if len(sys.argv) > 1 else "warmup"

specs = [
    ("gemm", g.gen_gemm_module),
    ("gemm_sm120", g.gen_gemm_sm120_module),
    ("fp4_gemm_cutlass_sm120", g.gen_gemm_sm120_module_cutlass_fp4),
    ("fused_moe_120", m.gen_cutlass_fused_moe_sm120_module),
]

all_valid = True

for name, gen_fn in specs:
    spec = gen_fn()
    lib_path = spec.jit_library_path
    
    # 1. Check if binary exists and has valid ELF load
    is_valid = False
    if lib_path.exists():
        try:
            ctypes.CDLL(str(lib_path))
            is_valid = True
        except Exception as e:
            print(f"  [INVALID] {name} binary exists but failed to load: {e}", file=sys.stderr)
            is_valid = False
            
    if mode == "check":
        status = "[VALID]" if is_valid else "[MISSING/INVALID]"
        size_mb = lib_path.stat().st_size / (1024 * 1024) if lib_path.exists() else 0
        print(f"  {status} {name}: {size_mb:.2f} MB ({lib_path})")
        if not is_valid:
            all_valid = False
        continue

    # Warmup / Update mode:
    # Ninja checks file timestamps, .ninja_deps, and build.ninja to determine
    # if any source files or compilation flags changed.
    t0 = time.monotonic()
    if is_valid:
        # Run build() which invokes ninja. Ninja will report 'no work to do'
        # if sources/headers are untouched, or incrementally recompile if modified.
        spec.build(verbose=False)
        size_mb = lib_path.stat().st_size / (1024 * 1024)
        print(f"  [UP-TO-DATE] {name}: {size_mb:.2f} MB ({lib_path})")
    else:
        print(f"  [BUILD] Compiling {name} (bounded to MAX_JOBS workers)...", flush=True)
        spec.build(verbose=False)
        dt = time.monotonic() - t0
        size_mb = lib_path.stat().st_size / (1024 * 1024) if lib_path.exists() else 0
        print(f"  [DONE]  {name} compiled in {dt:.1f}s ({size_mb:.2f} MB) -> {lib_path}", flush=True)

if mode == "check" and not all_valid:
    sys.exit(1)
PY

echo ""
echo "=== Step 3/3: Verifying PyTorch Inductor CUDA Environment ==="

"$PYTHON_BIN" - <<'PY'
import torch
print(f"  [OK] PyTorch {torch.__version__} with CUDA {torch.version.cuda}")
assert torch.cuda.is_available(), "CUDA device not available"
print(f"  [OK] Target Device: {torch.cuda.get_device_name(0)}")
PY

echo ""
echo "=== Cache Validation Complete ==="
echo "All FlashInfer Blackwell SM120 modules are valid and up to date."
echo "Launch the server with:"
echo "  ./serve.sh mtp"
