#!/usr/bin/env python3
"""
Grid search evaluation script for dLLM training
Based on eval.sh, this script runs parameter combinations sequentially
"""

import subprocess
import os
import itertools
from datetime import datetime

os.environ['NCCL_TIMEOUT'] = '1000'


def run_evaluation(task, nshots, length, steps, temperature, model,
                   num_processes, alg, wandb_project_name=None,
                   enable_wandb=False):
    """Run a single evaluation with given parameters"""

    # Set environment variables
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
    env['HF_ALLOW_CODE_EVAL'] = '1'

    # Create output path
    output_path = f"evals_results/{task}-ns{nshots}"

    print(f"\n{'='*80}")
    print("Running evaluation:")
    print(f"  Task: {task}")
    print(f"  Shots: {nshots}")
    print(f"  Length: {length}")
    print(f"  Steps: {steps}")
    print(f"  Temperature: {temperature}")
    print(f"  Output: {output_path}")
    print(f"  GPUs: {num_processes}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")

    # Build the command
    cmd = [
        'accelerate', 'launch',
        '--num_processes', str(num_processes),
        'eval.py',
        '--model', 'custom_coder',
        '--model_args', (
            f'pretrained={model},max_new_tokens={length},steps={steps},'
            f'add_bos_token=true,temperature={temperature},'
            f'top_p=0.95,alg={alg}'
        ),
        '--tasks', task,
        '--num_fewshot', str(nshots),
        '--batch_size', '20',
        '--output_path', output_path,
        '--log_samples',
        '--confirm_run_unsafe_code'
    ]

    if enable_wandb and wandb_project_name:
        cmd.extend(['--wandb_project_name', wandb_project_name])

    # Run the command - any error will cause the program to stop
    subprocess.run(cmd, env=env, check=True)
    print("✅ Evaluation completed successfully")


def main():
    """Main function to run grid search"""

    # Fixed model and configuration
    model = "fredzzp/open-dcoder-0.5B"
    num_processes = 4
    enable_wandb = True
    wandb_project_name = "dllm-eval-step128-latest"

    # Option 1: Use predefined parameter combinations (hardcoded)
    USE_CUSTOM_COMBINATIONS = False
    
    if USE_CUSTOM_COMBINATIONS:
        # Custom parameter combinations
        custom_combinations = [
            # {"task": "humaneval", "temperature": 0.3, "length": 128, 
            #  "steps": 128, "nshots": 0, "alg": "entropy"},
            # {"task": "humaneval", "temperature": 1.2, "length": 128, 
            #  "steps": 128, "nshots": 0, "alg": "entropy"},
            # {"task": "humaneval_plus", "temperature": 0.6, "length": 128, 
            #  "steps": 128, "nshots": 0, "alg": "entropy"},
            # {"task": "humaneval_plus", "temperature": 1.1, "length": 128, 
            #  "steps": 128, "nshots": 0, "alg": "entropy"},
            # {"task": "mbpp", "temperature": 0.7, "length": 128, 
            #  "steps": 128, "nshots": 0, "alg": "entropy"},
            # {"task": "mbpp", "temperature": 1.0, "length": 128, 
            #  "steps": 128, "nshots": 0, "alg": "entropy"},
            # {"task": "mbpp_plus", "temperature": 0.6, "length": 128, 
            #  "steps": 128, "nshots": 0, "alg": "entropy"},
            # {"task": "mbpp_plus", "temperature": 1.2, "length": 128,
            #  "steps": 128, "nshots": 0, "alg": "entropy"},

            # {"task": "humaneval", "temperature": 0.3, "length": 128,
            #  "steps": 128, "nshots": 0, "alg": "topk_margin"},
            # {"task": "humaneval", "temperature": 1.2, "length": 128, 
            #  "steps": 128, "nshots": 0, "alg": "topk_margin"},
            # {"task": "humaneval_plus", "temperature": 0.6, "length": 128, 
            #  "steps": 128, "nshots": 0, "alg": "topk_margin"},
            # {"task": "humaneval_plus", "temperature": 1.1, "length": 128, 
            #  "steps": 128, "nshots": 0, "alg": "topk_margin"},
            # {"task": "mbpp", "temperature": 0.7, "length": 128, 
            #  "steps": 128, "nshots": 0, "alg": "topk_margin"},
            # {"task": "mbpp", "temperature": 1.0, "length": 128, 
            #  "steps": 128, "nshots": 0, "alg": "topk_margin"},
            # {"task": "mbpp_plus", "temperature": 0.6, "length": 128, 
            #  "steps": 128, "nshots": 0, "alg": "topk_margin"},
            # {"task": "mbpp_plus", "temperature": 1.2, "length": 128, 
            #  "steps": 128, "nshots": 0, "alg": "topk_margin"},
        ]
        
        print(f"Starting custom combinations run with "
              f"{len(custom_combinations)} combinations:")
        for i, combo in enumerate(custom_combinations, 1):
            print(f"  {i}: {combo}")
        
        # Run each combination sequentially
        for i, combo in enumerate(custom_combinations, 1):
            print(f"\n🔄 Running combination {i}/{len(custom_combinations)}")
            
            run_evaluation(
                task=combo["task"],
                nshots=combo["nshots"],
                length=combo["length"],
                steps=combo["steps"],
                temperature=combo["temperature"],
                model=model,
                num_processes=num_processes,
                wandb_project_name=wandb_project_name,
                enable_wandb=enable_wandb,
                alg=combo["alg"]
            )
        
        total_combinations = len(custom_combinations)
    
    else:
        # Option 2: Original grid search approach
        tasks = [
            # "humaneval", 
            "humaneval_plus",
            "mbpp", 
            "mbpp_plus"
            ]
        nshots = [0]
        lengths = [128]
        steps = [128]
        temperatures = [
            1.2, 
            1.1, 
            1.0, 0.9, 
            0.8,
             0.7, 
            0.6, 
            0.5, 
            0.4, 
            0.3,
             0.2, 
            0.1
        ]
        algs = [
        "p2", 
        "origin",
        # "maskgit_plus", 
        # "topk_margin", "entropy"
        ]
        

        # Generate all combinations
        combinations = list(itertools.product(
            tasks, nshots, lengths, steps, temperatures, algs
        ))

        print(f"Starting grid search with {len(combinations)} combinations:")
        print(f"  Tasks: {tasks}")
        print(f"  Shots: {nshots}")
        print(f"  Lengths: {lengths}")
        print(f"  Steps: {steps}")
        print(f"  Temperatures: {temperatures}")
        print(f"  Algorithms: {algs}")

        # Run each combination sequentially
        for i, (task, nshot, length, step, temp, alg) in enumerate(
                combinations, 1):
            print(f"\n🔄 Running combination {i}/{len(combinations)}")

            run_evaluation(
                task=task,
                nshots=nshot,
                length=length,
                steps=step,
                temperature=temp,
                model=model,
                num_processes=num_processes,
                wandb_project_name=wandb_project_name,
                enable_wandb=enable_wandb,
                alg=alg
            )
        
        total_combinations = len(combinations)

    # Final summary
    print(f"\n{'='*80}")
    print("GRID SEARCH COMPLETED")
    print(f"{'='*80}")
    print(f"Total combinations: {total_combinations}")
    print("All runs completed successfully!")
    print(f"Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
