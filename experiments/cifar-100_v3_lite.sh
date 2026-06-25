# bash experiments/cifar-100_v3_lite.sh
# v3-lite run. Main stdout is written by run.py to v3_lite_output.log.
# Compact optimization signals are written to v3_lite_effective.log.

DATASET=cifar-100
OUTDIR=outputs/${DATASET}/10-task

GPUID='0'
REPEAT=5
OVERWRITE=1

mkdir -p $OUTDIR

python -u run.py --config configs/cifar-100_prompt_smope.yaml --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name OnePrompt \
    --prompt_param 50 5 1e-5 1e-5 0.4 --seeds 0 1 2 3 4 \
    --crct_epochs 50 --ca_batch_size_ratio 1 \
    --log_dir ${OUTDIR}/one-prompt
