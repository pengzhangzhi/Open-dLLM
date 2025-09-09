# Code Infilling Evaluation

This directory contains tools for evaluating code infilling models on HumanEval and SantaCoder datasets.

## Installation

### 1. Environment Setup
```bash
# Create conda environment with Python 3.11.13
conda create -n dllm python=3.11.13
conda activate dllm
```

### 2. Install Evaluation Tools
```bash
# Install HumanEval infilling evaluation tool
cd human-eval-infilling
pip install -e .

# Verify installation
which evaluate_infilling_functional_correctness
```

### 3. Required Dependencies
The evaluation system uses:
- `evaluate_infilling_functional_correctness` for HumanEval infilling
- `compute_em_santa.py` for SantaCoder FIM (included)
- WandB for experiment tracking (optional)

## Quick Start

### Using the Shell Script
```bash
cd eval/eval_infill
bash run_eval.sh
```

## Key Parameters

- `--model_path`: HuggingFace model path
- `--task`: Choose between `humaneval_infill` or `santacoder-fim`
- `--temperature`: Sampling temperature
- `--steps`: Diffusion steps
- `--alg`: Sampling algorithm: `p2`, `origin`, `maskgit_plus`, `topk_margin`, `entropy`

## Distributed Evaluation
```bash
# Multi-GPU evaluation
torchrun --nproc_per_node=4 eval_infill.py --use_ddp --task humaneval_infill
```

## Evaluation & Grading

### Automatic Evaluation
The system automatically evaluates generated code using task-specific metrics:

#### HumanEval Infilling (`humaneval_infill`)
- **Method**: Functional correctness testing
- **Tool**: `evaluate_infilling_functional_correctness`
- **Metrics**: `pass@k`

#### SantaCoder FIM (`santacoder-fim`)
- **Method**: Exact string matching
- **Tool**: `compute_em_santa.py`
- **Metrics**: `exact_match` rate


### Evaluation Configuration
```bash
# Enable/disable automatic evaluation
python eval_infill.py --task humaneval_infill  # auto_eval=True (default)
python eval_infill.py --task humaneval_infill --no_auto_eval  # disable

# Disable WandB logging
python eval_infill.py --task humaneval_infill --no_wandb
```

### Output Files
Results are saved to `infill_results/{task}/{model_name}/{temperature}/`:
- `{task}_results_{timestamp}.jsonl` - Generated predictions
- `{task}_results_{timestamp}_eval_results.json` - Evaluation metrics
- Automatic WandB logging (if enabled)
