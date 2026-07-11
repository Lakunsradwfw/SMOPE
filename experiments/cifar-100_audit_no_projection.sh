#!/usr/bin/env bash
# Causal audit control: preserve v4 anchors but disable Split-Lite projection.
# Example: GPUID=0 bash experiments/cifar-100_audit_no_projection.sh

DATASET=${DATASET:-cifar-100}
OUTDIR=${OUTDIR:-outputs/${DATASET}/10-task/audit-no-projection}
GPUID=${GPUID:-0}
REPEAT=${REPEAT:-3}
OVERWRITE=${OVERWRITE:-1}
MAX_TASK=${MAX_TASK:-10}
CRCT_EPOCHS=${CRCT_EPOCHS:-50}
ACTIVE_TOPK=${ACTIVE_TOPK:-16}

mkdir -p "${OUTDIR}"
python -u run.py --config configs/cifar-100_prompt_smope.yaml --gpuid "${GPUID}" --repeat "${REPEAT}" --overwrite "${OVERWRITE}" \
  --learner_type prompt --learner_name OnePrompt --prompt_param 50 5 1e-5 1e-5 0.4 --seeds 0 1 2 \
  --max_task "${MAX_TASK}" --crct_epochs "${CRCT_EPOCHS}" --ca_batch_size_ratio 1 \
  --split_lite_alpha 0.2 --split_lite_rank 4 --split_lite_min_task 1 --split_lite_active_topk "${ACTIVE_TOPK}" \
  --split_lite_basis_source functional_tangent --split_lite_projection_scope protected_only \
  --functional_tangent_max_memories 0 --functional_tangent_seed 1729 --expert_usage_mode old_union \
  --disable_split_lite --enable_causal_audit \
  --experiment_version audit_no_projection \
  --log_dir "${OUTDIR}"
