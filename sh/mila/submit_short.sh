#!/bin/bash
#SBATCH -J dLLM-Pretraining
#SBATCH -o logs/%x_%A_%a_%j.out          # output file (job name, jobarrayID, array index, jobid)
#SBATCH -e logs/%x_%A_%a_%j.err          # error file
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --array=1-99%1                   # array task IDs used (e.g. as different seeds). Run one array task at a time.
#SBATCH --mem-per-gpu=32G               # 32 GB per GPU
#SBATCH -t 3:00:00                       # walltime (HH:MM:SS)
#SBATCH -c 12                            # CPU cores per job
#SBATCH --partition=short-unkillable
#SBATCH --open-mode=append

export TOKENIZERS_PARALLELISM=false
export HF_ALLOW_CODE_EVAL=1

module load cudatoolkit

# Determine MASTER_ADDR (first hostname in the allocation) so torch.distributed can connect reliably
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
# Choose a stable port (modify if your cluster restricts ports). Use jobid to reduce collisions.
MASTER_PORT=$((12000 + (SLURM_JOB_ID % 20000)))

# number of GPUs on this node
NPROC_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)


NNODES=${SLURM_NNODES:-1}
NODE_RANK=${SLURM_NODEID:-0}

exp_name=1 # Qwen2.5-Coder-0.5B_mdm-repr-align-10
OUTPUT_DIR=/network/scratch/a/alexander.tong/dllm/${exp_name}
mkdir -p "$OUTPUT_DIR"


torchrun \
  --nnodes=$NNODES \
  --nproc_per_node=$NPROC_PER_NODE \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  tasks/train_torch.py \
  configs/pretrain/qwen2_5_coder_500M.yaml \
  --data.train_path=/network/scratch/a/alexander.tong/sft_data/Nemotron-SFT-Code \
  --train.ckpt_manager=dcp \
  --train.micro_batch_size=12 \
  --train.global_batch_size=48 \
  --train.output_dir="$OUTPUT_DIR" \
  --train.repr_align_wt=10.0 \
  --train.save_steps=100000 \
  --train.eval_every=10000 \
  --train.wandb_name=${exp_name} \
  --data.datasets_type=iterable