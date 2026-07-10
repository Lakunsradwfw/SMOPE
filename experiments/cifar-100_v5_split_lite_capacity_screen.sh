# Small split-lite capacity screen for CIFAR-100 v5.
#
# This is the second-stage check after sensitivity overlap:
#   alpha in {0, 0.2, 0.5} x rank in {4, 8}
#
# Defaults are intentionally MAX_TASK=5 for screening. Use REPEAT=3 or
# MAX_TASK=10 only after one or two settings look promising.

DATASET=${DATASET:-cifar-100}
OUTDIR=${OUTDIR:-outputs/${DATASET}/10-task}

GPUID=${GPUID:-0}
REPEAT=${REPEAT:-1}
OVERWRITE=${OVERWRITE:-1}
MAX_TASK=${MAX_TASK:-5}
CRCT_EPOCHS=${CRCT_EPOCHS:-50}
ACTIVE_TOPK=${ACTIVE_TOPK:-16}
SPLIT_MIN_TASK=${SPLIT_MIN_TASK:-1}
USAGE_MODE=${USAGE_MODE:-old_union}
STRICT_CURRENT=${STRICT_CURRENT:-0}
ALPHAS=${ALPHAS:-"0 0.2 0.5"}
RANKS=${RANKS:-"4 8"}
ENABLE_SENSITIVITY=${ENABLE_SENSITIVITY:-0}
SENSITIVITY_MAX_MEMORIES=${SENSITIVITY_MAX_MEMORIES:-4}

STRICT_LABEL=nonstrict
STRICT_ARG=
if [ "${STRICT_CURRENT}" = "1" ]; then
  STRICT_LABEL=strictcurrent
  STRICT_ARG=--split_lite_strict_current_topk
fi

SENSITIVITY_ARG=
if [ "${ENABLE_SENSITIVITY}" = "1" ]; then
  SENSITIVITY_ARG=--enable_sensitivity_diagnostics
fi

mkdir -p "${OUTDIR}"

for SPLIT_RANK in ${RANKS}; do
  for SPLIT_ALPHA in ${ALPHAS}; do
    VERSION="v5_capacity_${USAGE_MODE}_rank${SPLIT_RANK}_alpha${SPLIT_ALPHA}_${STRICT_LABEL}"
    LOG_DIR="${OUTDIR}/v5-capacity-${USAGE_MODE}-${STRICT_LABEL}-topk${ACTIVE_TOPK}-rank${SPLIT_RANK}-alpha${SPLIT_ALPHA}"

    python -u run.py --config configs/cifar-100_prompt_smope.yaml --gpuid "${GPUID}" \
      --repeat "${REPEAT}" --overwrite "${OVERWRITE}" \
      --learner_type prompt --learner_name OnePrompt \
      --prompt_param 50 5 1e-5 1e-5 0.4 --seeds 0 1 2 3 4 \
      --max_task "${MAX_TASK}" --crct_epochs "${CRCT_EPOCHS}" --ca_batch_size_ratio 1 \
      --split_lite_alpha "${SPLIT_ALPHA}" \
      --split_lite_rank "${SPLIT_RANK}" \
      --split_lite_min_task "${SPLIT_MIN_TASK}" \
      --split_lite_active_topk "${ACTIVE_TOPK}" \
      ${STRICT_ARG} \
      --expert_usage_mode "${USAGE_MODE}" \
      ${SENSITIVITY_ARG} \
      --sensitivity_rank "${SPLIT_RANK}" \
      --sensitivity_max_memories "${SENSITIVITY_MAX_MEMORIES}" \
      --experiment_version "${VERSION}" \
      --log_dir "${LOG_DIR}"
  done
done
