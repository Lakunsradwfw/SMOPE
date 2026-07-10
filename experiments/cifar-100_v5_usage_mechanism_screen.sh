# Screen whether task-local expert usage fixes the weak v5/v4 signal.
#
# Runs four small controls:
#   cumulative vs task-local expert frequencies
#   strict-current projection gate off vs on
#
# Override from the shell when needed, for example:
#   MAX_TASK=5 REPEAT=5 ACTIVE_TOPK=16 bash experiments/cifar-100_v5_usage_mechanism_screen.sh
#   USAGE_MODES="task old_union" bash experiments/cifar-100_v5_usage_mechanism_screen.sh
#   STRICT_MODES="0" bash experiments/cifar-100_v5_usage_mechanism_screen.sh

DATASET=${DATASET:-cifar-100}
OUTDIR=${OUTDIR:-outputs/${DATASET}/10-task}

GPUID=${GPUID:-0}
REPEAT=${REPEAT:-3}
OVERWRITE=${OVERWRITE:-1}
MAX_TASK=${MAX_TASK:-5}
CRCT_EPOCHS=${CRCT_EPOCHS:-50}
ACTIVE_TOPK=${ACTIVE_TOPK:-16}
SPLIT_ALPHA=${SPLIT_ALPHA:-0.2}
SPLIT_MIN_TASK=${SPLIT_MIN_TASK:-1}
USAGE_MODES=${USAGE_MODES:-"cumulative task old_union"}
STRICT_MODES=${STRICT_MODES:-"0 1"}

mkdir -p "${OUTDIR}"

for USAGE_MODE in ${USAGE_MODES}; do
  for STRICT_CURRENT in ${STRICT_MODES}; do
    STRICT_LABEL=nonstrict
    STRICT_ARG=
    if [ "${STRICT_CURRENT}" = "1" ]; then
      STRICT_LABEL=strictcurrent
      STRICT_ARG=--split_lite_strict_current_topk
    fi

    VERSION="v5_usage_${USAGE_MODE}_${STRICT_LABEL}"
    LOG_DIR="${OUTDIR}/v5-usage-${USAGE_MODE}-${STRICT_LABEL}-max${MAX_TASK}-topk${ACTIVE_TOPK}"

    python -u run.py --config configs/cifar-100_prompt_smope.yaml --gpuid "${GPUID}" \
      --repeat "${REPEAT}" --overwrite "${OVERWRITE}" \
      --learner_type prompt --learner_name OnePrompt \
      --prompt_param 50 5 1e-5 1e-5 0.4 --seeds 0 1 2 3 4 \
      --max_task "${MAX_TASK}" --crct_epochs "${CRCT_EPOCHS}" --ca_batch_size_ratio 1 \
      --split_lite_alpha "${SPLIT_ALPHA}" \
      --split_lite_min_task "${SPLIT_MIN_TASK}" \
      --split_lite_active_topk "${ACTIVE_TOPK}" \
      ${STRICT_ARG} \
      --expert_usage_mode "${USAGE_MODE}" \
      --experiment_version "${VERSION}" \
      --log_dir "${LOG_DIR}"
  done
done
