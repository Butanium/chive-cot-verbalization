#!/bin/bash
# Production entrypoint for the resampling job (catalogue image job-core; SGLang baked in).
# Usage (inside the job, cwd = delivered worktree root):
#   bash "$SILICO_EXPERIMENT_RELATIVE_DIR/src/resample_job.sh" [extra resample.py args]
set -euo pipefail
EXP="${SILICO_EXPERIMENT_RELATIVE_DIR:?}"
OUT="${SILICO_EXPERIMENT_ARTIFACTS_DIR:?}/resample"
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv
exec uv run python -u "$EXP/src/resample.py" --prompts "$EXP/results/prompts.jsonl" --out-dir "$OUT" "$@"
