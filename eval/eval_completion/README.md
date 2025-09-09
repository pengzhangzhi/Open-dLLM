# LM-Eval-Harness Evaluation

This directory contains tools for evaluating code generation models using the LM-Eval-Harness framework on coding benchmarks like HumanEval and MBPP.

## Installation

### 1. Environment Setup
```bash
# Create conda environment with Python 3.11.13
conda create -n dllm python=3.11.13
conda activate dllm
```

### 2. Install Dependencies
```bash
# Install LM-Evaluation-Harness from source
cd lm-evaluation-harness
pip install -e .

# Install additional required packages
pip install accelerate torch
pip install wandb  # optional, for experiment tracking
```

## Quick Start

### Using the Shell Script
```bash
cd eval/lm_eval_harness
bash run_eval.sh
```

## Key Parameters

- `--model_args`: Model configuration string with comma-separated parameters:
  - `pretrained`: HuggingFace model path
  - `max_new_tokens`: Maximum tokens to generate
  - `steps`: Diffusion steps
  - `temperature`: Sampling temperature
  - `alg`: Sampling algorithm: `p2`, `origin`, `maskgit_plus`, `topk_margin`, `entropy`
- `--tasks`: Available tasks: `humaneval`, `humaneval_plus`, `mbpp`, `mbpp_plus`
- `--num_fewshot`: Number of few-shot examples (typically 0 for code generation)
- `--batch_size`: Batch size for evaluation

## Evaluation & Grading

### Automatic Evaluation
The system uses LM-Eval-Harness built-in evaluation for code generation tasks:

#### HumanEval (`humaneval`, `humaneval_plus`)
- **Method**: Functional correctness testing
- **Metrics**: `pass@k` scores
- **Process**: Executes generated code against test cases

#### MBPP (`mbpp`, `mbpp_plus`)
- **Method**: Functional correctness testing
- **Metrics**: `pass@k` scores
- **Process**: Executes generated code against test cases

### Output Files
Results are saved to `evals_results/{task}-ns{shots}/`:
- Model predictions and evaluation metrics
- Detailed logs when using `--log_samples`
- JSON format results compatible with LM-Eval-Harness

### WandB Integration
```bash
# Enable WandB logging (optional)
python eval.py [other_args] --wandb_project_name "your-project-name"
```
