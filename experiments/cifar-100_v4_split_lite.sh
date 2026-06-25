# bash experiments/cifar-100_v4_split_lite.sh
# v4-split-lite run.
# Main stdout: outputs/cifar-100/10-task/one-prompt/v4_split_lite_output.log
# Effective metrics: outputs/cifar-100/10-task/one-prompt/v4_split_lite_effective.log
# Projection diagnostics: outputs/cifar-100/10-task/one-prompt/v4_split_lite_projection.log

DATASET=cifar-100
OUTDIR=outputs/${DATASET}/10-task

GPUID='0'
REPEAT=5
OVERWRITE=1
MAX_TASK=-1
CRCT_EPOCHS=50

mkdir -p $OUTDIR

python -u run.py --config configs/cifar-100_prompt_smope.yaml --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name OnePrompt \
    --prompt_param 50 5 1e-5 1e-5 0.4 --seeds 0 1 2 3 4 \
    --max_task $MAX_TASK --crct_epochs $CRCT_EPOCHS --ca_batch_size_ratio 1 \
    --log_dir ${OUTDIR}/one-prompt
