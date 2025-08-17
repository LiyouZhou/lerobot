#!/bin/bash

set -e
set -x

source ~/anaconda3/etc/profile.d/conda.sh
conda activate lerobot

python src/lerobot/scripts/train.py  \
    --policy.type=smolvla  \
    --dataset.repo_id=mikasa_robo_tfds_all_1.0.0_lerobot  \
    --dataset.root=/home/liyouzhou/study/any4lerobot/openx2lerobot/data/mikasa_robo_tfds_all_1.0.0_lerobot  \
    --batch_size=128  \
    --steps=200000  \
    --policy.push_to_hub=false \
    --wandb.run_id="mikasa_all_smolvla_train_$(date +"%Y-%m-%d_%H-%M-%S")" \
    --wandb.enable="true" \
    --wandb.project "smolvla" \
    --wandb.entity "leothemagnificent-university-of-cambridge" \
    --output_dir=logs/$(date +"%Y-%m-%d_%H-%M-%S")/ \
    --policy.use_amp=false \
    --log_freq=10 \
    --policy.device=cuda \
    --num_workers=8 \
    --job_name="mikasa_all_smolvla_train" \
