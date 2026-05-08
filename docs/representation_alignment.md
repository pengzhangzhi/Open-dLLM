# Representation Alignment Tutorial

Open-dLLM supports **representation alignment** for adapting autoregressive language models into diffusion language models. This feature is based on our recent paper, **Don’t Retrain—Align: Adapting Autoregressive LMs to Diffusion LMs via Representation Alignment**.

## Data

For code pretraining and instruction-style code data, we recommend the NVIDIA Nemotron pretraining datasets:

- `nvidia/Nemotron-Pretraining-Code-v1`
- `nvidia/Nemotron-Pretraining-SFT-v1`

These datasets may require permission from NVIDIA before access is granted. Make sure your Hugging Face account has been approved for access before running the download commands.

You should also authenticate with Hugging Face first:

```bash
huggingface-cli login
````

Then download the datasets:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='nvidia/Nemotron-Pretraining-Code-v1', repo_type='dataset', local_dir='$SCRATCH/code_data', resume_download=True, max_workers=4)" \
&& \
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='nvidia/Nemotron-Pretraining-SFT-v1', repo_type='dataset', local_dir='$SCRATCH/sft_data', resume_download=True, max_workers=4)"
```

The example below uses the SFT code subset located at:

```bash
$SCRATCH/sft_data/Nemotron-SFT-Code
```

Adjust the path according to your local data layout.

## Training Configuration

The complete training configuration is provided at:

```bash
configs/pretrain/qwen2_5_coder_500M.yaml
```

This config supports both Qwen2.5 and Qwen3-style models. In the example below, we use Qwen3:

```bash
--model.model_path=Qwen/Qwen3-0.6B
```

Representation alignment is controlled by:

```bash
--train.repr_align_wt=10.0
```

When `train.repr_align_wt > 0`, representation alignment is automatically enabled. Setting it to `0.0` disables the representation-alignment loss.

## Run Training

The following command launches distributed training with `torchrun`. It automatically sets the number of processes per node to the number of visible GPUs.

```bash
export TOKENIZERS_PARALLELISM=false
export HF_ALLOW_CODE_EVAL=1

NNODES=1
NPROC_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
NODE_RANK=${NODE_RANK:=0}
MASTER_ADDR=${MASTER_ADDR:=0.0.0.0}
MASTER_PORT=${MASTER_PORT:=12345}

torchrun \
  --nnodes=$NNODES \
  --nproc-per-node=$NPROC_PER_NODE \
  --node-rank=$NODE_RANK \
  --master-addr=$MASTER_ADDR \
  --master-port=$MASTER_PORT \
  tasks/train_torch.py \
  configs/pretrain/qwen2_5_coder_500M.yaml \
  --model.model_path=Qwen/Qwen3-0.6B \
  --data.train_path=$SCRATCH/sft_data/Nemotron-SFT-Code \
  --train.ckpt_manager=dcp \
  --train.micro_batch_size=2 \
  --train.global_batch_size=2 \
  --train.output_dir=logs/mdm \
  --train.repr_align_wt=10.0 \
  --train.save_steps=100000 \
  --train.eval_every=20000 \
  --data.datasets_type=iterable
```

## Notes

* `train.repr_align_wt` controls the strength of the representation-alignment loss.
* The demo uses `Qwen/Qwen3-0.6B`, but the same config can be adapted to Qwen2.5-style checkpoints.
* `data.train_path` should point to the local dataset directory.


## Slurm submission
Below is an example script to reproduce the paper. We use 16 H100s, total batch size 96, Qwen3-0.6B, repr-align-weight 10, and freeze `lm_head,embed_tokens`.
```bash
#!/bin/bash
#SBATCH -J dLLM-Pretraining
#SBATCH -o nlogs/%x_%j.out
#SBATCH -e nlogs/%x_%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1            # one launcher per node (important)
#SBATCH --gres=gpu:h100:4

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

```