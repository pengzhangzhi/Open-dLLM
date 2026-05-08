#!/bin/bash
#SBATCH -J dLLM-Pretraining
#SBATCH -o nlogs/%x_%j.out
#SBATCH -e nlogs/%x_%j.err
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1            # one launcher per node (important)
#SBATCH --gres=gpu:l40s:4
#SBATCH --partition=long
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH -c 32                          # CPUs per task (per node)
#SBATCH --wait-all-nodes=1
#SBATCH --open-mode=append
#SBATCH --array=1-99%1                   # array task IDs used (e.g. as different seeds). Run one array task at a time.

export TOKENIZERS_PARALLELISM=false
export HF_ALLOW_CODE_EVAL=1

RDV_ADDR=$(hostname)
NPROC_PER_NODE=$(nvidia-smi -L 2>/dev/null | wc -l)

exp_name=Qwen3-0.6B-repr-align-10
OUTPUT_DIR=/network/scratch/a/alexander.tong/dllm/logs/${exp_name}
mkdir -p "${OUTPUT_DIR}"

micro_batch_size=3
global_batch_size=$((micro_batch_size * NPROC_PER_NODE * SLURM_JOB_NUM_NODES))
srun -l torchrun \
   --nproc_per_node=$NPROC_PER_NODE \
   --nnodes=$SLURM_JOB_NUM_NODES \
   --rdzv_id=$SLURM_JOB_ID \
   --rdzv_backend=c10d \
   --rdzv_endpoint=$RDV_ADDR \
    tasks/train_torch.py \
    configs/pretrain/qwen2_5_coder_500M.yaml \
    --data.train_path=/network/scratch/a/alexander.tong/sft_data/Nemotron-SFT-Code \
    --train.ckpt_manager=dcp \
    --train.micro_batch_size=${micro_batch_size} \
    --train.global_batch_size=${global_batch_size} \
    --train.output_dir="${OUTPUT_DIR}" \
    --train.repr_align_wt=10.0 \
    --train.save_steps=100000 \
    --train.eval_every=10000 \
    --train.wandb_name=${exp_name} \
    --data.datasets_type=iterable \
    --model.model_path=Qwen/Qwen3-0.6B \
    --train.freeze_layers="lm_head,embed_tokens"
