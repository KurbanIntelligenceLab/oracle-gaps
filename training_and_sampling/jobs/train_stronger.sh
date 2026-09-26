#!/usr/bin/env bash
# Stronger recipe: 1,500 steps, LoRA rank 64 (alpha 128), the full training list
# (1,200 Geometry3K or all 366 MathVista prompts). ITEM is dataset|seed.
set -euo pipefail
IFS='|' read -r DS SEED <<< "$ITEM"
export CSM_DATASET="$DS" CSM_N_TRAIN=1200 CSM_MAX_STEPS=1500 CSM_LORA_R=64 CSM_LORA_ALPHA=128
export CSM_RUN_TAG=_long1500_r64 ARROW_IO_THREADS=2
ITEM="$SEED" exec bash training_and_sampling/jobs/train.sh
