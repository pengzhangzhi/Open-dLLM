#!/bin/bash
#SBATCH -J dLLM-Pretraining
#SBATCH -o logs/%x_%A_%a_%j.out          # output file (job name, jobarrayID, array index, jobid)
#SBATCH -e logs/%x_%A_%a_%j.err          # error file
#SBATCH --ntasks-per-node=1            # one launcher per node (important)
#SBATCH --account=h200ea
#SBATCH --partition=h200ea
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:4
#SBATCH --array=1-99%1                   # array task IDs used (e.g. as different seeds). Run one array task at a time.
#SBATCH -t 3-00:00:00                       # walltime (HH:MM:SS)
#SBATCH --open-mode=append
#SBATCH --wait-all-nodes=1

export TOKENIZERS_PARALLELISM=false
export HF_ALLOW_CODE_EVAL=1

RDV_ADDR=$(hostname)
NPROC_PER_NODE=$(nvidia-smi -L 2>/dev/null | wc -l)


# init micromamba environment
source /hpc/home/zp70/.bashrc
micromamba activate dllm
exp_name=Qwen3-0.6B-repr-align-10-freeze-embed-mlp-h200
OUTPUT_DIR=/hpc/group/dallagolab/fred/dllm/logs/${exp_name}
mkdir -p "${OUTPUT_DIR}"

micro_batch_size=24
accumulation_steps=1
global_batch_size=$((micro_batch_size * accumulation_steps * NPROC_PER_NODE * SLURM_JOB_NUM_NODES))


srun -l torchrun \
   --nproc_per_node=$NPROC_PER_NODE \
   --nnodes=$SLURM_JOB_NUM_NODES \
   --rdzv_id=$SLURM_JOB_ID \
   --rdzv_backend=c10d \
   --rdzv_endpoint=$RDV_ADDR \
    tasks/train_torch.py \
    configs/pretrain/qwen2_5_coder_500M.yaml \
    --train.wandb_project='dLLM-Pretraining' \
    --data.train_path=/hpc/group/dallagolab/fred/data/sft_data/Nemotron-SFT-Code/ \
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
    --train.freeze_layers="lm_head,embed_tokens,mlp"
    # --train.wandb_entity='openproblems-comp' \
