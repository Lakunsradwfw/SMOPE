# exp2: reduced projection strength — alpha=0.1
# bash experiments/cifar-100_exp2_lowalpha.sh

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
    --split_lite_alpha 0.1 --split_lite_min_task 1 --split_lite_active_topk $ACTIVE_TOPK \
    --log_dir ${OUTDIR}/exp2-max5-alpha01-mintask1-topk${ACTIVE_TOPK}
