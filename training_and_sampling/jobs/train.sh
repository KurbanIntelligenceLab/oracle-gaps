#!/usr/bin/env bash
# Train one policy with the light recipe. ITEM is the seed. Run from the repository root.
# DATA_DIR holds the base model, the datasets and the outputs.
set -euo pipefail
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${DATA_DIR}/CSM/hf_cache" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1

export CSM_SEED="${ITEM:-1}"
export CSM_DATASET="${CSM_DATASET:-geometry3k}"
export CSM_N_TRAIN="${CSM_N_TRAIN:-400}"
export CSM_NUM_GENERATIONS="${CSM_NUM_GENERATIONS:-8}"   # GRPO group size
export CSM_MICRO_BATCH="${CSM_MICRO_BATCH:-1}"           # prompt groups per device step
export CSM_GRAD_ACCUM="${CSM_GRAD_ACCUM:-4}"
export CSM_BETA="${CSM_BETA:-0.0}"                        # no KL penalty
export CSM_LR="${CSM_LR:-1e-5}"
export CSM_MAX_STEPS="${CSM_MAX_STEPS:-500}"
export CSM_RUN_TAG="${CSM_RUN_TAG:-_lr1e5}"
export CSM_MAX_COMPLETION="${CSM_MAX_COMPLETION:-512}"
export CSM_SAVE_STEPS="${CSM_SAVE_STEPS:-25}"

python training_and_sampling/rlvr_train.py
