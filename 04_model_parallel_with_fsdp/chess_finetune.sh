#!/bin/bash -l
#SBATCH -J chess_finetune
#SBATCH -N 1
#SBATCH --output=%x-%j_%n.out
#SBATCH --gres=gpu:a100:4
#SBATCH --constraint gpu80
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 0-1

total_batch_size=${TOTAL_BATCH_SIZE:-64} # total batch size per optimization
batch_size_per_device=${BATCH_SIZE_PER_DEVICE:-1} # batch size per device

num_gpus=$(jq -n "[$CUDA_VISIBLE_DEVICES] | length")
gradient_accumulation_steps=$(($total_batch_size / $batch_size_per_device / $num_gpus))


free_port=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
torchrun \
    --nnodes=1 \
    --nproc-per-node=$num_gpus \
    chess_finetune.py \
        --batch_size_per_device $batch_size_per_device \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        $@
