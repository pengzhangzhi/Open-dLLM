import datetime
import json
import os
from typing import List

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import AutoTokenizer
from utils import run_evaluation_tool, upload_to_wandb

from veomni.models.transformers.qwen2.generation_utils import MDMGenerationConfig
from veomni.models.transformers.qwen2.modeling_qwen2 import Qwen2ForCausalLM


def infilling(
    model,
    tokenizer,
    prompts: List[str],
    middle_lens: List[int],
    suffixs: List[str],
    max_length: int = 2048,
    batch_size: int = 32,
    steps: int = 100,
    temperature: float = 0.5,
    top_p: float = 0.95,
    alg: str = "p2",
    alg_temp: float = 0.5,
    device: str = "cuda",
    rank: int = 0,
) -> List[str]:
    """Code infilling function similar to generator_temp.py"""
    # Tokenize prompts
    tokenized_prompts = [tokenizer.encode(p, add_special_tokens=True) for p in prompts]
    prefix_lens = [len(p) for p in tokenized_prompts]

    # Construct sequences: prefix + masks + suffix
    sequences = [
        p + [tokenizer.mask_token_id] * m + tokenizer.encode(s, add_special_tokens=False)
        for p, m, s in zip(tokenized_prompts, middle_lens, suffixs)
    ]
    #     import pdb; pdb.set_trace()

    # Truncate if needed
    sequences = [seq[-max_length:] for seq in sequences]

    generations = []
    num_batches = (len(sequences) + batch_size - 1) // batch_size

    for i in tqdm(
        range(0, len(sequences), batch_size), desc="Generating batches", total=num_batches, disable=(rank != 0)
    ):
        batch_seqs = sequences[i : i + batch_size]
        batch_prefix_lens = prefix_lens[i : i + batch_size]
        batch_middle_lens = middle_lens[i : i + batch_size]

        # Pad sequences to same length
        max_len = max(len(seq) for seq in batch_seqs)
        padded_seqs = torch.LongTensor(
            [seq + [tokenizer.pad_token_id] * (max_len - len(seq)) for seq in batch_seqs]
        ).to(device)

        # Generate using diffusion
        generation_config = MDMGenerationConfig(
            mask_token_id=tokenizer.mask_token_id,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=10000000,  # No new tokens, just to avoid transformer error
            steps=steps,
            temperature=temperature,
            top_p=top_p,
            alg=alg,
            alg_temp=alg_temp,
            return_dict_in_generate=True,
        )

        with torch.no_grad():
            # Handle DDP wrapped model
            actual_model = model.module if hasattr(model, "module") else model
            outputs = actual_model._mdm_sample(x=padded_seqs, attention_mask=None, generation_config=generation_config)

        # Extract middle parts and decode
        batch_results = outputs.sequences
        for j, (result, pl, ml) in enumerate(zip(batch_results, batch_prefix_lens, batch_middle_lens)):
            middle_part = result[pl : pl + ml]
            decoded = tokenizer.decode(middle_part.tolist(), skip_special_tokens=True)
            generations.append(decoded)

    return generations


def setup_ddp():
    """Initialize DDP using environment variables set by torchrun"""
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=3600))
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup():
    dist.destroy_process_group()


def eval_infill(
    model_path: str = "fredzzp/open-dcoder-0.5B",
    task: str = "humaneval_infill",
    prediction_path: str = "infill_results.jsonl",
    fix_middle_length: int = None,
    steps: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.95,
    alg: str = "p2",
    alg_temp: float = 0.5,
    batch_size: int = 32,
    use_ddp: bool = False,
    auto_eval: bool = True,
    use_wandb: bool = True,
):
    """Evaluate infilling performance on humaneval_infill dataset"""

    # Extract model name from path
    model_name = model_path.split("/")[-1] if "/" in model_path else model_path

    # Create results directory structure: infill_results/task/model_name/temperature
    results_dir = os.path.join("infill_results", task, model_name, str(temperature))
    os.makedirs(results_dir, exist_ok=True)

    prediction_path = os.path.join(
        results_dir, f"{task}_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    )

    # Collect configuration parameters for wandb
    config_params = {
        "model_path": model_path,
        "model_name": model_name,
        "task": task,
        "fix_middle_length": fix_middle_length,
        "steps": steps,
        "temperature": temperature,
        "top_p": top_p,
        "alg": alg,
        "alg_temp": alg_temp,
        "batch_size": batch_size,
        "use_ddp": use_ddp,
        "auto_eval": auto_eval,
        "results_dir": results_dir,
    }

    # Initialize DDP if requested
    if use_ddp:
        rank, local_rank, world_size = setup_ddp()
        device = f"cuda:{local_rank}"

        # Only print on rank 0 to avoid clutter
        if rank == 0:
            print(f"Initialized DDP: rank {rank}, local_rank {local_rank}, world_size {world_size}")
    else:
        rank, local_rank, world_size = 0, 0, 1
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and tokenizer
    if rank == 0:
        print(f"Loading model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Initialize parallel state for custom model if using DDP
    if use_ddp:
        from veomni.distributed.parallel_state import init_parallel_state

        init_parallel_state(
            dp_size=world_size,
            tp_size=1,
            ep_size=1,
            pp_size=1,
            cp_size=1,
            ulysses_size=1,
            dp_mode="ddp",
            device_type="cuda",
            include_sp_in_fsdp=True,
        )

    model = Qwen2ForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)

    if use_ddp:
        model = model.to(local_rank)
        model = DDP(model, device_ids=[local_rank])
    else:
        model = model.to(device)

    model.eval()

    # Set mask token if needed
    if tokenizer.mask_token is None:
        raise ValueError("Mask token not found in tokenizer")

    # Load dataset
    if rank == 0:
        print(f"Loading {task} dataset...")

    if task == "humaneval_infill":
        from human_eval_infilling.data import read_problems

        problems = read_problems(benchmark_name="single-line")
        #         from itertools import islice
        #         problems = dict(islice(problems.items(), 3))
        prefixs = [problems[task_id]["prompt"] for task_id in problems]
        suffixs = [problems[task_id]["suffix"] for task_id in problems]
        ground_truth_middles = [problems[task_id]["canonical_solution"] for task_id in problems]
        task_ids = list(problems.keys())
    elif task == "santacoder-fim":
        # Load SantaCoder FIM dataset
        fim_data = load_dataset("bigcode/santacoder-fim-task", split="train")
        fim_data = [d for d in fim_data if d["language"] == "py"]
        prefixs = [d["prompt"] + "\n" for d in fim_data]
        suffixs = [d["suffix"] for d in fim_data]
        ground_truth_middles = [d["canonical_solution"] for d in fim_data]
        task_ids = [i for i in range(len(fim_data))]

        if rank == 0:
            print(f"Loaded {len(fim_data)} examples from santacoder-fim")
            print("Example prefix:", prefixs[0][:100] + "...")
            print("Example suffix:", suffixs[0][:100] + "...")
    else:
        raise ValueError(f"Unsupported task: {task}")

    # Shard dataset by rank for DDP
    if use_ddp:
        total = len(prefixs)
        per_rank = total // world_size
        start = rank * per_rank
        end = start + per_rank if rank < world_size - 1 else total

        prefixs_shard = prefixs[start:end]
        suffixs_shard = suffixs[start:end]
        ground_truth_middles_shard = ground_truth_middles[start:end]
        task_ids_shard = task_ids[start:end]
    else:
        prefixs_shard = prefixs
        suffixs_shard = suffixs
        ground_truth_middles_shard = ground_truth_middles
        task_ids_shard = task_ids

    # Calculate middle lengths
    if fix_middle_length:
        middle_lens = [fix_middle_length] * len(task_ids_shard)
    else:
        # Oracle setting: use ground truth lengths
        middle_lens = [len(tokenizer.encode(gt, add_special_tokens=False)) for gt in ground_truth_middles_shard]

    # Print maximum middle_lens
    if rank == 0:
        max_middle_len = max(middle_lens) if middle_lens else 0
        print(f"Maximum middle_lens: {max_middle_len}")
        # import pdb; pdb.set_trace()

    # Generate completions
    if rank == 0:
        print(f"Generating completions for {len(prefixs_shard)} examples...")

    generations = infilling(
        model=model,
        tokenizer=tokenizer,
        prompts=prefixs_shard,
        middle_lens=middle_lens,
        suffixs=suffixs_shard,
        batch_size=batch_size,
        steps=steps,
        temperature=temperature,
        top_p=top_p,
        alg=alg,
        alg_temp=alg_temp,
        device=device,
        rank=rank,
    )

    # Prepare samples
    samples = [
        {
            "task_id": task_id,
            "completion": pred,
            "ground_truth_middle": ground_truth_middle,
            "prefix": prefix,
            "suffix": suffix,
        }
        for task_id, pred, ground_truth_middle, prefix, suffix in zip(
            task_ids_shard, generations, ground_truth_middles_shard, prefixs_shard, suffixs_shard
        )
    ]

    if use_ddp:
        # Gather results from all ranks
        gathered_samples = [None] * world_size
        dist.all_gather_object(gathered_samples, samples)

        if local_rank == 0:
            merged_samples = []
            for s in gathered_samples:
                merged_samples.extend(s)

            print(f"Saving results to {prediction_path}")
            with open(prediction_path, "w") as f:
                for sample in merged_samples:
                    f.write(json.dumps(sample) + "\n")

            print(f"Evaluation complete! Results saved to {prediction_path}")

            # Run automatic evaluation if enabled
            eval_results = None
            if auto_eval:
                eval_results = run_evaluation_tool(task, prediction_path, local_rank)

            # Save evaluation results if available
            if eval_results:
                eval_file = prediction_path.replace(".jsonl", "_eval_results.json")
                eval_data = {"config": config_params, "results": eval_results}
                with open(eval_file, "w") as f:
                    json.dump(eval_data, f, indent=2)
                print(f"Evaluation results saved to {eval_file}")

            # Upload to wandb if enabled
            if use_wandb:
                upload_to_wandb(
                    task=task,
                    model_path=model_path,
                    prediction_path=prediction_path,
                    eval_results=eval_results,
                    config_params=config_params,
                    rank=local_rank,
                )

        # Cleanup DDP
        cleanup()

        # Return both samples and evaluation results
        final_results = {
            "samples": gathered_samples if local_rank == 0 else samples,
            "eval_results": eval_results if local_rank == 0 else None,
        }
        return final_results
    else:
        print(f"Saving results to {prediction_path}")
        with open(prediction_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        print(f"Evaluation complete! Results saved to {prediction_path}")

        # Run automatic evaluation if enabled
        eval_results = None
        if auto_eval:
            eval_results = run_evaluation_tool(task, prediction_path, rank)

        # Save evaluation results if available
        if eval_results:
            eval_file = prediction_path.replace(".jsonl", "_eval_results.json")
            eval_data = {"config": config_params, "results": eval_results}
            with open(eval_file, "w") as f:
                json.dump(eval_data, f, indent=2)
            print(f"Evaluation results saved to {eval_file}")

        # Upload to wandb if enabled
        if use_wandb:
            upload_to_wandb(
                task=task,
                model_path=model_path,
                prediction_path=prediction_path,
                eval_results=eval_results,
                config_params=config_params,
                rank=rank,
            )

        # Return both samples and evaluation results
        return {"samples": samples, "eval_results": eval_results}


if __name__ == "__main__":
    # Simple CLI interface
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate code infilling on humaneval_infill or santacoder-fim")
    parser.add_argument("--model_path", default="fredzzp/open-dcoder-0.5B", help="Model path or HF model name")
    parser.add_argument(
        "--task", default="humaneval_infill", choices=["humaneval_infill", "santacoder-fim"], help="Evaluation task"
    )
    parser.add_argument("--prediction_path", default="infill_results.jsonl", help="Output file for results")
    parser.add_argument(
        "--fix_middle_length", type=int, default=None, help="Fixed middle length (default: use oracle)"
    )
    parser.add_argument("--steps", type=int, default=64, help="Number of diffusion steps")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument(
        "--alg",
        default="p2",
        choices=["p2", "origin", "maskgit_plus", "topk_margin", "entropy"],
        help="Sampling algorithm",
    )
    parser.add_argument("--alg_temp", type=float, default=0.5, help="Algorithm temperature for Gumbel noise")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--use_ddp", action="store_true", help="Enable distributed data parallel")
    parser.add_argument("--no_auto_eval", action="store_true", help="Disable automatic evaluation")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")

    args = parser.parse_args()

    # Convert no_auto_eval to auto_eval and no_wandb to use_wandb
    args.auto_eval = not args.no_auto_eval
    args.use_wandb = not args.no_wandb
    delattr(args, "no_auto_eval")
    delattr(args, "no_wandb")

    # Path generation is now handled in eval_infill function

    eval_infill(**vars(args))
