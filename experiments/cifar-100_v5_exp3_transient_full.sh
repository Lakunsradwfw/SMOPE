# bash experiments/cifar-100_v5_exp3_transient_full.sh
# v5 full: transient prompt guides router bias and e_pv protection strength.

DATASET=cifar-100
OUTDIR=outputs/${DATASET}/10-task

GPUID='0'
REPEAT=5
OVERWRITE=1
MAX_TASK=5
CRCT_EPOCHS=50
ACTIVE_TOPK=12
WARMUP_BATCHES=20

mkdir -p $OUTDIR

python -u run.py --config configs/cifar-100_prompt_smope.yaml --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name OnePrompt \
    --prompt_param 50 5 1e-5 1e-5 0.4 --seeds 0 1 2 3 4 \
    --max_task $MAX_TASK --crct_epochs $CRCT_EPOCHS --ca_batch_size_ratio 1 \
    --split_lite_alpha 0.2 --split_lite_min_task 3 --split_lite_active_topk $ACTIVE_TOPK \
    --use_transient_prompt --transient_min_task 1 --transient_warmup_batches $WARMUP_BATCHES \
    --transient_lr 1e-3 --transient_cp_bias_weight 0.1 --transient_protect_scale 1.0 \
    --experiment_version v5_exp3_transient_full \
    --log_dir ${OUTDIR}/v5-exp3-transient-full-topk${ACTIVE_TOPK}
