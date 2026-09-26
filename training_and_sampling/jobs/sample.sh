#!/usr/bin/env bash
# Sample and score one prompt slice for one policy of a seed pool.
# ITEM is model|dataset|split|offset|count, where model is "base" or a seed name (seed1, ...).
set -euo pipefail
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${DATA_DIR}/CSM/hf_cache" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1

IFS='|' read -r MODEL DS SPLIT OFFSET COUNT <<< "$ITEM"
IDS="$(mktemp)"; trap 'rm -f "$IDS"' EXIT
awk -v s="$((OFFSET + 1))" -v c="$COUNT" 'NR >= s && NR < s + c' \
    "analysis/splits/${DS}_${SPLIT}_probe.txt" > "$IDS"
n_ids=$(wc -l < "$IDS")
[ "$n_ids" -eq "$COUNT" ] || { echo "slice has ${n_ids} prompts, expected ${COUNT}" >&2; exit 3; }

if [ "$MODEL" != "base" ]; then
  export CSM_ADAPTER_PATH="${DATA_DIR}/CSM/runs/${DS}__${MODEL}${CSM_RUN_TAG:-_lr1e5}/final"
  [ -d "$CSM_ADAPTER_PATH" ] || { echo "adapter missing: $CSM_ADAPTER_PATH" >&2; exit 4; }
fi

export CSM_MODEL_NAME="$MODEL"
export CSM_DATASET="$DS" CSM_SPLIT="$SPLIT" CSM_PROMPT_IDS="$IDS" CSM_N_ITEMS=0
export CSM_K="${CSM_K:-16}" CSM_TEMP=1.0 CSM_MAX_NEW_TOKENS=1024 CSM_SEED="${CSM_SEED:-0}"
python training_and_sampling/rollout_gen.py
