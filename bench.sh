#!/usr/bin/env bash
set -euo pipefail
# Unscored Sixcat speed; not the 120-item quality evaluation.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
SIXCAT_PYTHON="${SIXCAT_PYTHON:-/var/lib/sixcat-eval/venv/bin/python}"
RUN_ID="super-vl-nvfp4-$(date +%Y%m%dT%H%M%S%z)"
OUT="${OUT:-$ROOT/bench/results/$RUN_ID.json}"
cmd=("$SIXCAT_PYTHON" -m sixcat speed --base-url "$BASE_URL"
  --model nemotron-super-vl --policy strict --thinking on
  --profile all --samples 32 --curve-seconds 300 --max-seconds 1200
  --out "$OUT")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${cmd[@]}"; printf '\n'
  exit 0
fi
[[ -x "$SIXCAT_PYTHON" ]] || { echo "missing Sixcat Python: $SIXCAT_PYTHON" >&2; exit 1; }
smoke_args=(--base-url "$BASE_URL" --model nemotron-super-vl --context 1048576)
if [[ "${EXPECT_DRAFT:-1}" == 1 ]]; then smoke_args+=(--expect-spec); fi
python3 "$ROOT/scripts/smoke.py" "${smoke_args[@]}"
mkdir -p "$(dirname "$OUT")"
"${cmd[@]}"
python3 - "$OUT" <<'PY'
import json,sys
p=sys.argv[1];d=json.load(open(p))
expected={'decode','balanced','prefill'}
assert set(d['profiles'])==expected and d['model']=='nemotron-super-vl'
for name,v in d['profiles'].items():
    c=v['confirmation']
    assert c['succeeded']==c['requested']==32,(name,c['succeeded'],c['requested'])
print('validated receipt:',p)
PY
