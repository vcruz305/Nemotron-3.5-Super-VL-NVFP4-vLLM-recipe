#!/usr/bin/env bash
set -euo pipefail
# Run on a GB10 Spark. This script never stops other models or rexl3-manager.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Auto-source .env if present, allowing caller environment overrides
if [[ -f "$ROOT/.env" ]]; then
  _backup_vars=()
  for _var in VLLM_BIN MODEL_DIR DRAFT_DIR PORT HOST; do
    if [[ -n "${!_var+x}" ]]; then _backup_vars+=("$_var=${!_var}"); fi
  done
  set -a
  # shellcheck source=/dev/null
  source "$ROOT/.env"
  set +a
  for _entry in "${_backup_vars[@]}"; do export "$_entry"; done
  unset _backup_vars _entry _var
fi

UNIT=vllm-super-vl-nvfp4-recipe.service
GUARD=vllm-super-vl-nvfp4-recipe-guard.service
MODEL_DIR="${MODEL_DIR:-$HOME/models/Nemotron-3.5-Super-VL-NVFP4-ga-20261003/nemotron-3.5-super-vl-120b-a12b-ea20261008_vnvfp4-ga-20261003}"
DRAFT_DIR="${DRAFT_DIR:-$HOME/models/Nemotron-3.5-Super-VL-LiLiCorr-ga-20261001/nemotron-3.5-super-vl-120b-a12b-ea20261008_vlilicorr-ga-20261001}"
VLLM_BIN="${VLLM_BIN:-$HOME/vllm31-venv/bin/vllm}"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
profile="${1:-mtp}"

active() { systemctl --user is-active --quiet "$1"; }
unit_state() { systemctl --user is-active "$1" 2>/dev/null || :; }

case "$profile" in
  down)
    systemctl --user stop --no-block "$GUARD" "$UNIT" 2>/dev/null || :
    for _ in {1..20}; do
      case "$(unit_state "$UNIT")" in active|activating|deactivating) sleep 0.5 ;; *) break ;; esac
    done
    case "$(unit_state "$UNIT")" in
      active|activating|deactivating) systemctl --user kill --signal=SIGKILL "$UNIT" ;;
    esac
    printf 'model=%s guard=%s\n' "$(unit_state "$UNIT")" "$(unit_state "$GUARD")"
    exit 0 ;;
  status)
    printf 'model=%s guard=%s\n' "$(unit_state "$UNIT")" "$(unit_state "$GUARD")"
    exit 0 ;;
  mtp|dflash) ;;
  *) printf 'usage: %s [mtp|dflash|status|down]\n' "$0" >&2; exit 2 ;;
esac

gmu=0.71
if [[ "$profile" == mtp ]]; then
  spec='{"method":"mtp","num_speculative_tokens":5}'
else
  gmu=0.74  # 0.71 yields 5.54 GiB KV, below 7.52 GiB needed for 1M.
  spec="$(python3 -c 'import json,sys; print(json.dumps({"method":"dflash","model":sys.argv[1],"num_speculative_tokens":7,"rejection_sample_method":"standard","attention_backend":"FLASHINFER"},separators=(",",":")))' "$DRAFT_DIR")"
fi
args=("$VLLM_BIN" serve "$MODEL_DIR" --served-model-name nemotron-super-vl
  --host "$HOST" --port "$PORT" --trust-remote-code
  --gpu-memory-utilization "$gmu" --mamba-backend flashinfer
  --mamba-ssm-cache-dtype float16 --enable-mamba-cache-stochastic-rounding
  --mamba-cache-philox-rounds 5 --mamba-cache-mode align
  --prefix-match-unit 16 --enable-mamba-shared-prefix-checkpoint
  --limit-mm-per-prompt '{"image": 1, "video": 0}'
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
  --reasoning-parser nemotron_v3 --speculative-config "$spec")
run=(systemd-run --user --unit="${UNIT%.service}"
  --property=MemoryMax=102005473280 --property=MemorySwapMax=12884901888
  --property=OOMPolicy=stop --property=OOMScoreAdjust=500
  --setenv="PATH=$(dirname "$VLLM_BIN"):/usr/local/cuda/bin:$PATH"
  --setenv=VLLM_USE_V2_MODEL_RUNNER=1 --setenv=MAX_JOBS="${MAX_JOBS:-2}" "${args[@]}")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf 'profile=%s model=nemotron-super-vl context=1048576\n' "$profile"
  printf 'guard: systemd-run --user --unit=%s /usr/bin/python3 %q %q\n' "${GUARD%.service}" "$ROOT/scripts/guard.py" "$UNIT"
  printf 'serve: '; printf '%q ' "${run[@]}"; printf '\n'
  exit 0
fi
[[ -x "$VLLM_BIN" ]] || { echo "missing vLLM: $VLLM_BIN" >&2; exit 1; }
[[ -f "$MODEL_DIR/config.json" ]] || { echo "incomplete Super VL NVFP4 target directory" >&2; exit 1; }
if [[ "$profile" == dflash ]]; then
  [[ -f "$DRAFT_DIR/model.safetensors" && -f "$DRAFT_DIR/config.json" ]] || { echo "missing exact LiLiCorr draft" >&2; exit 1; }
fi

# Pre-flight: verify Python.h is available for PyTorch Inductor multimodal compilation
python3 - <<'PY'
import os, sys, sysconfig
h = os.path.join(sysconfig.get_path("include"), "Python.h")
if not os.path.isfile(h):
    print(f"Error: Missing {h}\nPyTorch Inductor requires python C headers.\nPlease install via: sudo apt-get install -y python3-dev", file=sys.stderr)
    sys.exit(1)
PY

if active "$UNIT" || active "$GUARD"; then echo 'recipe already active; use status/down first' >&2; exit 1; fi
python3 - <<'PY'
from pathlib import Path
m={}
for line in Path('/proc/meminfo').read_text().splitlines():
    if line.startswith(('MemAvailable:', 'SwapFree:')):
        k,v=line.split(':',1);m[k]=int(v.split()[0])
assert m['MemAvailable'] >= 8*1024*1024 and m['SwapFree'] >= 2*1024*1024, m
PY
nvidia-smi --query-compute-apps=pid --format=csv,noheader > /dev/null
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo 'GPU already has a compute process; refusing to overlap another model' >&2; exit 1
fi
python3 - "$PORT" "$HOST" <<'PY'
import socket,sys
s=socket.socket();s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
try:s.bind((sys.argv[2],int(sys.argv[1])))
except OSError:raise SystemExit('port already in use')
finally:s.close()
PY

# Optional direct foreground execution (bypasses systemd)
if [[ "${FOREGROUND:-0}" == 1 ]]; then
  echo "Starting vLLM in foreground mode (bypassing systemd)..."
  export PATH="$(dirname "$VLLM_BIN"):/usr/local/cuda/bin:$PATH"
  export VLLM_USE_V2_MODEL_RUNNER=1
  export MAX_JOBS="${MAX_JOBS:-2}"
  exec "${args[@]}"
fi

# Pre-flight: verify systemd user bus is active
if ! systemctl --user status &>/dev/null; then
  echo "Error: systemd user session is inactive or inaccessible (Connection refused)." >&2
  echo "If user@$UID was terminated (e.g. by OOM), restart it using:" >&2
  echo "  sudo systemctl restart user@$UID" >&2
  echo "Alternatively, you can run directly without systemd using:" >&2
  echo "  FOREGROUND=1 ./serve.sh $profile" >&2
  exit 1
fi
systemctl --user reset-failed "$UNIT" "$GUARD" 2>/dev/null || :
systemd-run --user --unit="${GUARD%.service}" --property=MemoryMax=134217728 \
  --property=MemorySwapMax=0 --property=OOMPolicy=stop \
  /usr/bin/python3 "$ROOT/scripts/guard.py" "$UNIT"
"${run[@]}"
printf 'Started %s; status: %s status\n' "$UNIT" "$0"
