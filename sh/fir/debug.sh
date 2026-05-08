#!/bin/bash
#SBATCH --job-name=dLLM-Pretraining
#SBATCH --account=rrg-bengioy-ad
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1               # one launcher per node (important)
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --wait-all-nodes=1
#SBATCH --open-mode=append
#SBATCH --output=nlogs/%x_%j.out
#SBATCH --error=nlogs/%x_%j.err
# Optional: run different seeds serially on the same allocation
#SBATCH --array=1-99%1


# --- Environment on Fir ---
module load rust cuda gcc arrow python/3.11
source "$SCRATCH/venvs/dllm/bin/activate"

# --- Sanity / performance env ---
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_ALLOW_CODE_EVAL=1

# Recommended NCCL settings on CC multi-node (adjust iface if needed)
# If your nodes use InfiniBand and the mlx interface is visible as ibX:
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
# Set your main ethernet/infiniband device; try one of: ib0, eno1, ens3, enp134s0f0, etc.
# If unsure, leave commented or set via sbatch:  --export=ALL,NCCL_SOCKET_IFNAME=ib0
# export NCCL_SOCKET_IFNAME=ib0

# Rendezvous: use the first host in the allocation
RDV_ADDR="$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)"
NPROC_PER_NODE="$(nvidia-smi -L 2>/dev/null | wc -l)"

# --- Experiment naming / paths ---
exp_name=Qwen3-0.6B-repr-align-10-freeze-embed-mlp
OUTPUT_DIR="$SCRATCH/dllm/logs/${exp_name}"
mkdir -p "$OUTPUT_DIR" nlogs

# Data: point this to your Fir dataset location
DATA_DIR="${SCRATCH}/sft_data/Nemotron-SFT-Code"

# Batching: mirror Mila logic
micro_batch_size=3
accumulation_steps=2
global_batch_size=$(( micro_batch_size * accumulation_steps * NPROC_PER_NODE * SLURM_JOB_NUM_NODES ))

echo "[Info] Nodes: ${SLURM_JOB_NUM_NODES}  GPUs/node: ${NPROC_PER_NODE}  GBS: ${global_batch_size}"
echo "[Info] RDZV endpoint: ${RDV_ADDR}"

# If you want per-array-task seeding inside your script, expose SLURM_ARRAY_TASK_ID
export SEED="${SLURM_ARRAY_TASK_ID:-1}"

# --- Launch ---
srun -l torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --nnodes="${SLURM_JOB_NUM_NODES}" \
  --rdzv_id="${SLURM_JOB_ID}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${RDV_ADDR}:29400" \
  tasks/train_torch.py \
  configs/pretrain/qwen2_5_coder_500M.yaml \
  --train.wandb_project=dLLM-Pretraining \
  --data.train_path="${DATA_DIR}" \
  --train.ckpt_manager=dcp \
  --train.micro_batch_size="${micro_batch_size}" \
  --train.global_batch_size="${global_batch_size}" \
  --train.output_dir="${OUTPUT_DIR}" \
  --train.repr_align_wt=10.0 \
  --train.save_steps=100000 \
  --train.eval_every=10000 \
  --train.wandb_name="${exp_name}" \
  --data.datasets_type=iterable \
  --model.model_path=Qwen/Qwen3-0.6B \
  --train.freeze_layers="lm_head,embed_tokens,mlp"

