# bash experiments/cifar-100_v5_exp1_v4_delay_topk.sh
# v5 control: v4-split-lite with delayed projection and active top-k.

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
    --split_lite_alpha 0.2 --split_lite_min_task 3 --split_lite_active_topk $ACTIVE_TOPK \
    --experiment_version v5_exp1_v4_delay_topk \
    --log_dir ${OUTDIR}/v5-exp1-v4-delay-topk${ACTIVE_TOPK}
