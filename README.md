# Nemotron 3.5 Super VL NVFP4 — vLLM recipe

Unofficial text-serving recipe for NVIDIA Nemotron 3.5 Super VL on one DGX Spark (GB10, 128 GB unified memory). The **measured** target and LiLiCorr draft were obtained under private NGC access. **This repository contains no weights, private NGC org/team identifier, credentials, or calibration data.** The official Hugging Face releases are linked below, but their weights have **not** been verified byte-identical to the measured NGC versions; do not transfer the speed/quality figures to a different checkpoint without retesting. Nothing here starts, stops, or configures `rexl3-manager`.

## Hugging Face models

- **Target:** [NVIDIA Nemotron 3.5 Super VL 120B-A12B NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4) — public release; not the pinned private NGC target used for the measurements below.
- **Optional draft:** [NVIDIA Nemotron 3.5 Super VL NVFP4 LiLiCorr](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr) — public release; equivalence to the measured NGC draft has not been checked.

## Measured configurations

- Target: NGC `nemotron-3.5-super-vl-120b-a12b-ea20261008`, version `nvfp4-ga-20261003` (107 SHA-256-verified files, **75,061,211,348 bytes**). The model config advertises **1,048,576** tokens; the direct vLLM route also advertised 1,048,576. An actual 1M-token request was **not** tested.
- Optional separate draft: the same NGC model's `lilicorr-ga-20261001` version (four SHA-256-verified files, **1,255,056,137 bytes**). `LiLiCorrDraftModel`, NVFP4, trained block size 8, served with vLLM DFlash K=7. This is not another target checkpoint.
- Engine: isolated `vllm==0.31.0`, `flashinfer-python==0.7.0.post1`, PyTorch 2.13.0, Transformers 5.17.0. One GB10, TP=1, FlashInfer attention and Mamba backend. Target used NVFP4, logged `fp8_e4m3` KV. No `--max-model-len` override; GPU memory utilization **0.71 with MTP5** (the default) and **0.74 with DFlash7**. The latter is necessary for the 1M KV allocation: 0.71 supplied 5.54 GiB while 7.52 GiB was required. The exact flag sets are in [`serve.sh`](serve.sh). The server ran on the Spark; the recorded Sixcat client used `127.0.0.1:18000/v1` on a separate Linux host over an SSH forward to the Spark's `127.0.0.1:8000/v1`. Client TTFT and effective prefill include that tunnel.
- Tested speed prompts are **text-only**. During DFlash startup, vLLM warned that the draft does not accept external multimodal embeddings. Do **not** infer vision-draft correctness or multimodal speed from these receipts.

## Results

Sixcat **speed** used synthetic prompts, strict/temperature-0 policy with thinking on, 32 successful confirmation requests per profile. It is an unscored speed suite, **not** a model-quality score. Decode and prefill p50 are the C=1 curve (four requests per arm); balanced throughput is C=8 confirmation (32 requests). Historical on-host flag sets are reproduced by these scripts, but the new scripts have not yet been replayed end-to-end on a GPU.

- **Decode, C=1 p50:** built-in MTP5 **37.31 tok/s** → LiLiCorr/DFlash7 **37.09 tok/s** (−0.6%). No meaningful median improvement.
- **Balanced, C=8 aggregate output:** **90.87** → **93.55 tok/s** (+3.0%; small, not established beyond run-to-run noise). Includes prefill/queue wall time.
- **Long-prompt prefill, C=1 client-effective p50:** **1,449.44** → **1,342.15 tok/s** (−7.4%). Includes network and first-token handoff, not a server-native prefill metric.
- **Balanced TTFT, C=8 p50:** **1,686** → **1,761 ms** (+4.4%).
- vLLM `/metrics` after DFlash speed: 31,087 accepted of 85,393 draft tokens (**36.4%**); counters include smoke/probes. All three confirmation profiles were 32/32 in both runs.

[Raw speed receipts](bench/receipts/) replace only the operator's absolute `server_props` root path with `${MODEL_DIR}`; numerical measurements are unchanged. The [quality summary](bench/quality-summary.json) reports **84.17/100** (120/120, six categories including executed Code; strict/thinking-on). **It belongs to the MTP5 server, not DFlash; DFlash quality was not rerun.** Truncation/loop flags remain disclosed.

## Quick start on a DGX Spark

This recipe expects an already installed compatible vLLM 0.31.0/FlashInfer runtime and two exact NGC version directories. Fresh installation/download is **not** validated by this repo. A private NGC key stays in your normal credential manager; never paste it into a script or commit it. Under your own authorized NGC org/team scope, inspect manifests first, check there is room for both version directories plus a safety reserve, then download with the NGC CLI (which creates one nested version directory under each `--dest`):

```bash
: "${NGC_CLI_ORG:?configure private NGC org locally}"
: "${NGC_CLI_TEAM:?configure private NGC team locally}"
ngc registry model info "$NGC_CLI_ORG/$NGC_CLI_TEAM/nemotron-3.5-super-vl-120b-a12b-ea20261008:nvfp4-ga-20261003" \
  --files --format_type json --org "$NGC_CLI_ORG" --team "$NGC_CLI_TEAM"
ngc registry model info "$NGC_CLI_ORG/$NGC_CLI_TEAM/nemotron-3.5-super-vl-120b-a12b-ea20261008:lilicorr-ga-20261001" \
  --files --format_type json --org "$NGC_CLI_ORG" --team "$NGC_CLI_TEAM"
ngc registry model download-version "$NGC_CLI_ORG/$NGC_CLI_TEAM/nemotron-3.5-super-vl-120b-a12b-ea20261008:nvfp4-ga-20261003" \
  --dest "$HOME/models/Nemotron-3.5-Super-VL-NVFP4-ga-20261003" \
  --org "$NGC_CLI_ORG" --team "$NGC_CLI_TEAM"
ngc registry model download-version "$NGC_CLI_ORG/$NGC_CLI_TEAM/nemotron-3.5-super-vl-120b-a12b-ea20261008:lilicorr-ga-20261001" \
  --dest "$HOME/models/Nemotron-3.5-Super-VL-LiLiCorr-ga-20261001" \
  --org "$NGC_CLI_ORG" --team "$NGC_CLI_TEAM"
```

The CLI download/verification path above has **not** been replayed from this repository. Verify the nested trees against each private NGC manifest (size and per-file SHA-256) before serving. On the measured box, the directories were named as below; override them if yours differ:

```bash
export VLLM_BIN="$HOME/vllm31-venv/bin/vllm"
export MODEL_DIR="$HOME/models/Nemotron-3.5-Super-VL-NVFP4-ga-20261003/nemotron-3.5-super-vl-120b-a12b-ea20261008_vnvfp4-ga-20261003"
export DRAFT_DIR="$HOME/models/Nemotron-3.5-Super-VL-LiLiCorr-ga-20261001/nemotron-3.5-super-vl-120b-a12b-ea20261008_vlilicorr-ga-20261001"
DRY_RUN=1 ./serve.sh mtp  # inspect exact command; no GPU access
./serve.sh mtp            # measured default; user systemd unit and guard
./serve.sh status
python3 scripts/smoke.py --model nemotron-super-vl --context 1048576 --expect-spec
# Requires an installed Sixcat checkout and its Python environment:
SIXCAT_PYTHON=/var/lib/sixcat-eval/venv/bin/python ./bench.sh
./serve.sh down
```

To reproduce the optional DFlash arm instead, run `./serve.sh dflash`, smoke, and `./bench.sh` with the same benchmarking policy. `bench.sh` defaults to the Spark-local `:8000/v1`. To match the **historical client transport**, run the same checked-out `bench.sh` from a separate Linux client with Sixcat installed, create an SSH local forward (`ssh -N -L 18000:127.0.0.1:8000 <your-spark-host>` in another terminal), then use `BASE_URL=http://127.0.0.1:18000/v1 ./bench.sh`. The tunnel alters client-observed latency; do not compare it interchangeably with a direct-on-Spark run. Do not serve both arms or another model simultaneously; `serve.sh` refuses a GPU compute process or occupied port and never stops someone else's service. It starts a persistent guard that stops **only this recipe's unit** below 4 GiB available RAM, below 2 GiB free swap, or if a manager that was active at launch exits. A first-use FlashInfer top-k JIT can add several minutes to startup. Unit logs: `journalctl --user -u vllm-super-vl-nvfp4-recipe -f`; guard log: `~/.local/state/model-serving-recipes/vllm-super-vl-nvfp4-recipe.service.jsonl`.

## Limits

This is not an NVIDIA publication or a public-weight mirror. Do not deploy the draft for image requests on the strength of the text-only speed test. The MTP and DFlash arms differ not just in draft method/depth but also in GPU-memory reservation (0.71 versus 0.74) to preserve the same configured 1M context. The Super VL DFlash run left about 20 GiB host `MemAvailable` with swapping allowed; it does not establish that Bolt fits alongside it. The scripts do not modify model files.
