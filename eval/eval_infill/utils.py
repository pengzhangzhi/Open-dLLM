import ast
import datetime
import os
import re
import subprocess
from typing import Any, Dict

import wandb


def upload_to_wandb(
    task: str,
    model_path: str,
    prediction_path: str,
    eval_results: Dict[str, Any] = None,
    config_params: Dict[str, Any] = None,
    rank: int = 0,
):
    """Upload results to wandb artifacts"""
    if rank != 0:  # Only upload from main process
        return

    # Extract model name from path
    model_name = model_path.split("/")[-1] if "/" in model_path else model_path

    # Create run name with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{task}_{model_name}_{timestamp}"

    try:
        # Initialize wandb
        wandb.init(project="eval-infill-dllm-step64-latest", name=run_name, config=config_params or {})

        # Log evaluation metrics if available
        if eval_results:
            wandb.log(eval_results)
            print(f"Logged metrics to wandb: {eval_results}")

        # Create artifact for prediction results
        artifact = wandb.Artifact(
            name=f"{task}_{model_name}_results",
            type="predictions",
            description=f"Prediction results for {task} using {model_name}",
        )

        # Add prediction file
        if os.path.exists(prediction_path):
            artifact.add_file(prediction_path)
            print(f"Added prediction file to artifact: {prediction_path}")

        # Add evaluation results file if exists
        eval_file = prediction_path.replace(".jsonl", "_eval_results.json")
        if os.path.exists(eval_file):
            artifact.add_file(eval_file)
            print(f"Added evaluation file to artifact: {eval_file}")

        # Log artifact
        wandb.log_artifact(artifact)
        print(f"Uploaded artifact: {artifact.name}")

        # Finish wandb run
        wandb.finish()

    except Exception as e:
        print(f"Failed to upload to wandb: {e}")
        if "wandb" in globals() and wandb.run is not None:
            wandb.finish()


def run_evaluation_tool(task: str, prediction_path: str, rank: int = 0):
    """Run the appropriate evaluation tool based on task and return results"""
    if rank != 0:  # Only run evaluation on main process
        return None

    print(f"\nRunning evaluation for {task}...")
    eval_results = {}

    try:
        if task == "humaneval_infill":
            cmd = ["evaluate_infilling_functional_correctness", prediction_path, "--benchmark_name=single-line"]
            print(f"Running: {' '.join(cmd)}")
            # Run with stdin=subprocess.DEVNULL to prevent keyboard input interference
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, stdin=subprocess.DEVNULL)
            print("Evaluation output:")
            print(result.stdout)

            # Parse HumanEval results (looking for pass@k format)
            for line in result.stdout.split("\n"):
                if "pass@" in line and "{" in line:
                    try:
                        # Look for dict-like pattern: {'pass@1': np.float64(0.7744433688286544)}
                        match = re.search(r"\{[^}]+\}", line)
                        if match:
                            dict_str = match.group()
                            # Handle numpy types by replacing them
                            dict_str = re.sub(r"np\.float64\(([^)]+)\)", r"\1", dict_str)
                            # Use ast.literal_eval for safer evaluation
                            eval_dict = ast.literal_eval(dict_str)
                            # Convert values to float if they aren't already
                            for k, v in eval_dict.items():
                                if "pass@" in k:
                                    eval_results[k] = float(v)
                    except Exception as e:
                        print(f"Failed to parse HumanEval result line: {line}")
                        print(f"Error: {e}")
                        # Fallback: try to parse "pass@1: 0.423" format
                        match = re.search(r"pass@(\d+):\s*([0-9.]+)", line)
                        if match:
                            k, score = match.groups()
                            eval_results[f"pass@{k}"] = float(score)

            if result.stderr:
                print("Evaluation stderr:")
                print(result.stderr)

        elif task == "santacoder-fim":
            cmd = ["python", "compute_em_santa.py", "--result_path", prediction_path]
            print(f"Running: {' '.join(cmd)}")
            # Run with stdin=subprocess.DEVNULL to prevent keyboard input interference
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, stdin=subprocess.DEVNULL)
            print("Evaluation output:")
            print(result.stdout)

            # Parse SantaCoder EM results
            for line in result.stdout.split("\n"):
                if "{" in line and ("count" in line or "exact_match" in line):
                    try:
                        # Look for dict-like pattern: {'count': 1043, 'exact_match_rate': 56.08820709491851}
                        match = re.search(r"\{[^}]+\}", line)
                        if match:
                            dict_str = match.group()
                            # Use ast.literal_eval for safer evaluation
                            eval_dict = ast.literal_eval(dict_str)
                            # Extract relevant metrics
                            if "exact_match_rate" in eval_dict:
                                eval_results["exact_match"] = (
                                    float(eval_dict["exact_match_rate"]) / 100.0
                                )  # Convert percentage to decimal
                            if "count" in eval_dict:
                                eval_results["count"] = int(eval_dict["count"])
                    except Exception as e:
                        print(f"Failed to parse SantaCoder result line: {line}")
                        print(f"Error: {e}")
                        # Fallback: try legacy patterns
                        em_match = re.search(r"EM:\s*([0-9.]+)", line)
                        if em_match:
                            eval_results["exact_match"] = float(em_match.group(1))

                        pct_match = re.search(r"Exact Match:\s*([0-9.]+)%", line)
                        if pct_match:
                            eval_results["exact_match"] = float(pct_match.group(1)) / 100.0

            if result.stderr:
                print("Evaluation stderr:")
                print(result.stderr)
        else:
            print(f"No automatic evaluation available for task: {task}")
            return None

        # Print parsed results
        if eval_results:
            print("\nParsed evaluation results:")
            for metric, score in eval_results.items():
                print(f"  {metric}: {score:.4f}")
        else:
            print("Could not parse evaluation results from output")

        return eval_results

    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed with exit code {e.returncode}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return None
    except FileNotFoundError as e:
        print(f"Evaluation tool not found: {e}")
        print("Please make sure the evaluation tools are installed and in PATH")
        return None
