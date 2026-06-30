# exp1: strict-current-topk v4/v5 check on CIFAR-100 10-task.
# Usage:
#   bash experiments/cifar-100_exp1_baseline.sh
# Optional overrides:
#   GPUID=3 REPEAT=3 ACTIVE_TOPK=12 bash experiments/cifar-100_exp1_baseline.sh

DATASET=cifar-100
OUTDIR=outputs/${DATASET}/10-task

GPUID=${GPUID:-0}
REPEAT=${REPEAT:-1}
OVERWRITE=${OVERWRITE:-1}
MAX_TASK=${MAX_TASK:-10}
CRCT_EPOCHS=${CRCT_EPOCHS:-50}
ACTIVE_TOPK=${ACTIVE_TOPK:-16}
SPLIT_LITE_ALPHA=${SPLIT_LITE_ALPHA:-0.2}
SPLIT_LITE_MIN_TASK=${SPLIT_LITE_MIN_TASK:-1}
EXPERIMENT_VERSION=${EXPERIMENT_VERSION:-v5_exp1_strict_current_topk}
LOG_DIR=${LOG_DIR:-${OUTDIR}/exp1-max${MAX_TASK}-alpha02-mintask${SPLIT_LITE_MIN_TASK}-topk${ACTIVE_TOPK}-strictcurrent}

mkdir -p "${OUTDIR}"

python -u run.py --config configs/cifar-100_prompt_smope.yaml --gpuid "${GPUID}" --repeat "${REPEAT}" --overwrite "${OVERWRITE}" \
    --learner_type prompt --learner_name OnePrompt \
    --prompt_param 50 5 1e-5 1e-5 0.4 --seeds 0 1 2 3 4 \
    --max_task "${MAX_TASK}" --crct_epochs "${CRCT_EPOCHS}" --ca_batch_size_ratio 1 \
    --split_lite_alpha "${SPLIT_LITE_ALPHA}" \
    --split_lite_min_task "${SPLIT_LITE_MIN_TASK}" \
    --split_lite_active_topk "${ACTIVE_TOPK}" \
    --split_lite_strict_current_topk \
    --experiment_version "${EXPERIMENT_VERSION}" \
    --log_dir "${LOG_DIR}"
