#!/usr/bin/env bash
# Score the stronger-recipe policies on both datasets' prompts. Same ITEM format as sample_cross_dataset.sh.
set -euo pipefail
export CSM_RUN_TAG=_long1500_r64 CSM_NAME_SUFFIX=L ARROW_IO_THREADS=2
exec bash training_and_sampling/jobs/sample_cross_dataset.sh
