#!/bin/bash

set -e
set -x

source ~/anaconda3/etc/profile.d/conda.sh
conda activate lerobot

MEMORY=${MEMORY:-false}
EPISODIC=${EPISODIC:-false}
BATCH_SIZE=${BATCH_SIZE:-400}
SEED=${SEED:-42}
MIKASA_COLOR=${MIKASA_COLOR:-false}
BASE_MODEL=${BASE_MODEL:-"lerobot/smolvla_base"}
NOTES=${NOTES:-""}

DATA_SET="mikasa_all"
if [ "$EPISODIC" = "true" ] && [ "$MIKASA_COLOR" = "true" ]; then
    DATA_SET="mikasa_color"
fi

if [ "$MIKASA_COLOR" = "true" ] && [ "$EPISODIC" = "false" ]; then
    echo "Warning: MIKASA_COLOR is true but EPISODIC is false. Exiting."
    exit 1
fi

BASE_MODEL_NAME=$BASE_MODEL
if [ -d "$BASE_MODEL" ]; then
    IFS='/' read -ra PATH_CHUNKS <<< "$BASE_MODEL"
    STEP_NUMBER="${PATH_CHUNKS[${#PATH_CHUNKS[@]}-2]}"
    CHECKPOINT_TIMESTAMP="${PATH_CHUNKS[${#PATH_CHUNKS[@]}-4]}"
    BASE_MODEL_NAME="${CHECKPOINT_TIMESTAMP}-${STEP_NUMBER}"
    BASE_MODEL_NAME="${BASE_MODEL_NAME//_/-}"
fi

BASE_MODEL_CONFIG_FIELD_NAME="path"
if [ "$BASE_MODEL" = "smolvla" ]; then
    BASE_MODEL_CONFIG_FIELD_NAME="type"
fi

TS=$(date +"%y%m%d_%H%M%S")
RUN_ID="${DATA_SET}_${BASE_MODEL_NAME}_B${BATCH_SIZE}_E${EPISODIC}_M${MEMORY}_S${SEED}_${TS}"

SLURM_JOBID=${SLURM_JOBID:-"1234567"}
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

python src/lerobot/scripts/train.py \
    --policy.$BASE_MODEL_CONFIG_FIELD_NAME=$BASE_MODEL \
    --policy.chunk_size=50 \
    --policy.n_action_steps=1 \
    --policy.num_steps=10 \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --policy.use_amp=false \
    --policy.memory=$MEMORY \
    --policy.scheduler_type="cosine_annealing_with_warm_restarts" \
    --dataset.repo_id=mikasa_robo_tfds_all_1.0.0_lerobot \
    --dataset.root=/home/lz307/rds/hpc-work/lerobot/data/mikasa_robo_tfds_all_1.0.0_lerobot \
    --wandb.run_id=$RUN_ID \
    --wandb.enable="true" \
    --wandb.project "smolvla" \
    --wandb.entity "leothemagnificent-university-of-cambridge" \
    --wandb.notes="$NOTES" \
    --output_dir=lerobot_storage/$TS \
    --episodic=$EPISODIC \
    --batch_size=$BATCH_SIZE \
    --log_freq=1 \
    --save_freq=10000 \
    --eval_freq=1000 \
    --num_workers=30 \
    --steps=100000 \
    --job_name="$RUN_ID" \
    --seed=$SEED \
    --num_envs=100 \
    --num_trials_per_task=200 \
    --remember_color_only=$MIKASA_COLOR \
