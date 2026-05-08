#!/usr/bin/env python3
"""
Grid search script for evaluating infilling models with different hyperparameters.
"""

import subprocess
import argparse
import json
from datetime import datetime
from itertools import product
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

def run_eval_infill(model_path, task, temperature, steps, alg):
    """Run eval_infill.py with specified parameters"""
    
    cmd = ["torchrun", "--nproc_per_node", "4", "eval_infill.py"]
    cmd.extend([
        "--use_ddp",
        "--model_path", model_path,
        "--task", task, 
        "--temperature", str(temperature),
        "--steps", str(steps),
        "--alg", alg
    ])
    
    print(f"Running: {model_path.split('/')[-1]} {task} temp={temperature} steps={steps}")
    
    result = subprocess.run(cmd, check=True)
    return {"model": model_path, "task": task, "temperature": temperature, "steps": steps}


def main():
    parser = argparse.ArgumentParser(description="Grid search for eval_infill.py")
    args = parser.parse_args()
    
    # Fixed model
    model = "fredzzp/open-dcoder-0.5B"
    
    # Option 1: Use predefined parameter combinations (hardcoded)
    USE_CUSTOM_COMBINATIONS = False
    
    if USE_CUSTOM_COMBINATIONS:
        # Custom parameter combinations
        custom_combinations = [
            # {"task": "humaneval_infill", "temperature": 0.6, "steps": 64, "alg": "maskgit_plus"},
            # {"task": "santacoder-fim", "temperature": 0.1, "steps": 64, "alg": "p2"},
            # {"task": "santacoder-fim", "temperature": 0, "steps": 64, "alg": "maskgit_plus"},
            # {"task": "humaneval_infill", "temperature": 0.6, "steps": 64, "alg": "entropy"},
            # {"task": "santacoder-fim", "temperature": 0.7, "steps": 64, "alg": "entropy"},
            # {"task": "humaneval_infill", "temperature": 0.6, "steps": 64, "alg": "topk_margin"},
            # {"task": "santacoder-fim", "temperature": 0.7, "steps": 64, "alg": "topk_margin"},
        ]
        
        print(f"Starting custom combinations run with "
              f"{len(custom_combinations)} combinations:")
        for i, combo in enumerate(custom_combinations, 1):
            print(f"  {i}: {combo}")
        
        start_time = datetime.now()
        results = []
        
        # Run each combination sequentially
        for i, combo in enumerate(custom_combinations, 1):
            print(f"\n🔄 Running combination {i}/{len(custom_combinations)}")
            result = run_eval_infill(
                model_path=model,
                task=combo["task"],
                temperature=combo["temperature"],
                steps=combo["steps"],
                alg=combo["alg"]
            )
            results.append(result)
        
        total_combinations = len(custom_combinations)
    
    else:
        tasks = ["santacoder-fim", "humaneval_infill"]
        temperatures = [
            # 0.1, 
            0.2, 
            # 0.3, 0.4, 0.5, 
            0.6, 
            # 0.7, 0.8, 0.9, 1.0, 1.1, 1.2
        ]
        steps = [64]
        algs = [
            # "p2", 
        "origin", "maskgit_plus", "entropy", "topk_margin"]
        
        param_combinations = list(product([model], tasks, temperatures,
                                          steps, algs))
        
        print(f"Grid search: {len(param_combinations)} combinations")
        print(f"Model: {model}")
        print(f"Tasks: {tasks}")
        print(f"Temperatures: {temperatures}")
        print(f"Steps: {steps}")
        print(f"Algorithms: {algs}")
        
        start_time = datetime.now()
        results = []
        
        for i, (model_path, task, temp, steps_val, alg) in enumerate(
                param_combinations, 1):
            print(f"{i}/{len(param_combinations)}: ", end="")
            result = run_eval_infill(model_path, task, temp, steps_val,
                                     alg)
            results.append(result)
        
        total_combinations = len(param_combinations)
    
    # Save results
    results_file = (f"grid_search_results_"
                    f"{start_time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    duration = (datetime.now() - start_time).total_seconds() / 60
    print(f"\n{'='*80}")
    print("GRID SEARCH COMPLETED")
    print(f"{'='*80}")
    print(f"Total combinations: {total_combinations}")
    print(f"Completed {len(results)} runs in {duration:.1f} minutes")
    print(f"Results saved to: {results_file}")
    print(f"Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()