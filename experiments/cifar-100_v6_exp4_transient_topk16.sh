#!/usr/bin/env bash
# v6 Exp4: Exp3 plus risk-reward transient router bias; transient risk is not
# fed into alpha yet, so this isolates the router contribution.
# Example: GPUID=3 MAX_TASK=5 REPEAT=3 bash experiments/cifar-100_v6_exp4_transient_topk16.sh

DATASET=${DATASET:-cifar-100}
OUTDIR=${OUTDIR:-outputs/${DATASET}/10-task}
GPUID=${GPUID:-3}
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
  --split_lite_basis_source functional_tangent --split_lite_projection_scope protected_only \
  --split_lite_adaptive_conflict --split_lite_adaptive_alpha_max 0.5 --split_lite_conflict_weight 0.6 \
  --functional_tangent_max_memories 0 --functional_tangent_seed 1729 --expert_usage_mode old_union \
  --use_transient_prompt --transient_mode risk_reward --transient_warmup_batches 20 --transient_eval_batches 2 \
  --transient_lr 1e-3 --transient_cp_bias_weight 0.1 --transient_protect_scale 0.0 \
  --experiment_version v6_exp4_transient_topk16 \
  --log_dir "${OUTDIR}/v6-exp4-transient-topk${ACTIVE_TOPK}"
