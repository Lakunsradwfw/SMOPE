# exp1: v4 baseline — MAX_TASK=5, alpha=0.2, min_task=1
# bash experiments/cifar-100_exp1_baseline.sh

DATASET=cifar-100
OUTDIR=outputs/${DATASET}/10-task

GPUID='0'
REPEAT=5
OVERWRITE=1
MAX_TASK=5
CRCT_EPOCHS=50
ACTIVE_TOPK=12

mkdir -p $OUTDIR

python -u run.py --config configs/cifar-100_prompt_smope.yaml --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name OnePrompt \
    --prompt_param 50 5 1e-5 1e-5 0.4 --seeds 0 1 2 3 4 \
    --max_task $MAX_TASK --crct_epochs $CRCT_EPOCHS --ca_batch_size_ratio 1 \
    --split_lite_alpha 0.2 --split_lite_min_task 1 --split_lite_active_topk $ACTIVE_TOPK \
    --log_dir ${OUTDIR}/exp1-max5-alpha02-mintask1-topk${ACTIVE_TOPK}
