#!/usr/bin/env bash
# v6 Exp1: legacy gradient-basis control with the shared old_union top-16 scope.
# Example: GPUID=0 MAX_TASK=5 REPEAT=3 bash experiments/cifar-100_v6_exp1_legacy_topk16.sh

DATASET=${DATASET:-cifar-100}
OUTDIR=${OUTDIR:-outputs/${DATASET}/10-task}
GPUID=${GPUID:-0}
REPEAT=${REPEAT:-3}
OVERWRITE=${OVERWRITE:-1}
MAX_TASK=${MAX_TASK:-5}
CRCT_EPOCHS=${CRCT_EPOCHS:-20}
ACTIVE_TOPK=${ACTIVE_TOPK:-16}

mkdir -p "${OUTDIR}"
python -u run.py --config configs/cifar-100_prompt_smope.yaml --gpuid "${GPUID}" --repeat "${REPEAT}" --overwrite "${OVERWRITE}" \
  --learner_type prompt --learner_name OnePrompt --prompt_param 50 5 1e-5 1e-5 0.4 --seeds 0 1 2 3 4 \
  --max_task "${MAX_TASK}" --crct_epochs "${CRCT_EPOCHS}" --ca_batch_size_ratio 1 \
  --split_lite_alpha 0.2 --split_lite_rank 4 --split_lite_min_task 1 --split_lite_active_topk "${ACTIVE_TOPK}" \
  --split_lite_basis_source gradient --split_lite_projection_scope protected_only \
  --expert_usage_mode old_union \
  --experiment_version v6_exp1_legacy_topk16 \
  --log_dir "${OUTDIR}/v6-exp1-legacy-topk${ACTIVE_TOPK}"
