import logging
from typing import List, Optional, Union

import torch
import transformers
from accelerate import Accelerator
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import get_dtype
from tqdm import tqdm

from veomni.models.transformers.qwen2.generation_utils import MDMGenerationConfig

# Import your custom model and generation config
from veomni.models.transformers.qwen2.modeling_qwen2 import Qwen2ForCausalLM


eval_logger = logging.getLogger("eval_logger")


@register_model("custom_coder")
class CustomCoder(LM):
    def __init__(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        batch_size: Optional[Union[int, str]] = 1,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        # Generation parameters for diffusion_generate
        max_new_tokens: Optional[int] = 128,
        steps: Optional[int] = 100,
        temperature: Optional[float] = 0.5,
        top_k: Optional[int] = 200,
        alg: Optional[str] = "p2",
        alg_temp: Optional[float] = 0.5,
        trust_remote_code: Optional[bool] = True,
        # Other lm-harness params
        max_length: Optional[int] = 2048,
        **kwargs,
    ) -> None:
        super().__init__()

        # Initialize accelerator for multi-GPU support
        self.accelerator = Accelerator()
        self._device = self.accelerator.device
        self.batch_size_per_gpu = int(batch_size)
        self.max_length = max_length
        self.trust_remote_code = trust_remote_code

        # Initialize parallel state for custom model
        self._init_parallel_state()

        # Store generation parameters
        self.max_new_tokens = max_new_tokens
        self.steps = steps
        self.temperature = temperature
        self.top_k = top_k
        self.alg = alg
        self.alg_temp = alg_temp

        # Load the custom model and tokenizer
        self._create_model_and_tokenizer(pretrained, dtype)

    def _init_parallel_state(self):
        """Initialize parallel state for the custom model."""
        import torch.distributed as dist

        from veomni.distributed.parallel_state import init_parallel_state

        # If only 1 process, skip distributed init
        if self.accelerator.num_processes == 1:
            eval_logger.info("Single GPU detected. Skipping distributed init.")
            return

        # Multi-GPU: ensure process group is initialized
        if not dist.is_initialized():
            # import pdb; pdb.set_trace()  # Commented out debug statement
            dist.init_process_group(
                backend="nccl",  # use "gloo" if CPU-only
                init_method="env://",
            )

        world_size = self.accelerator.num_processes
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

    def _create_model_and_tokenizer(self, pretrained, dtype):
        """Loads the Qwen2ForCausalLM model and its tokenizer."""
        eval_logger.info(f"Loading tokenizer from: {pretrained}")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained,
            trust_remote_code=self.trust_remote_code,
        )

        eval_logger.info(f"Loading model from: {pretrained}")
        self.model = Qwen2ForCausalLM.from_pretrained(
            pretrained,
            torch_dtype=get_dtype(dtype),
            trust_remote_code=self.trust_remote_code,
        )

        # Set the mask token if not already set. This is crucial for
        # generation.
        if self.tokenizer.mask_token is None:
            self.tokenizer.add_special_tokens({"mask_token": "[MASK]"})
            self.model.resize_token_embeddings(len(self.tokenizer))
            eval_logger.info("Added new [MASK] token.")

        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            eval_logger.info("Set pad_token to eos_token")

        self.tokenizer.padding_side = "left"

        # Prepare model for distributed training/inference
        self.model = self.accelerator.prepare(self.model)
        self.model.eval()

    def _generate_batch(self, prompts: List[str], gen_kwargs: dict = None) -> List[str]:
        """Generates text for a batch of prompts using diffusion_generate."""

        # Tokenize the batch of prompts
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length - self.max_new_tokens,
        ).to(self.device)

        prompt_lengths = inputs.input_ids.shape[1]

        # Extract specific parameters from yaml if provided
        if gen_kwargs is None:
            gen_kwargs = {}

        # Use specific yaml parameters: num_return_sequences
        # Other parameters still come from eval.sh (model_args)
        # Note: do_sample is automatically set to True by MDMGenerationConfig
        num_return_sequences = gen_kwargs.get("num_return_sequences", 1)

        # Create a generation configuration object
        generation_config = MDMGenerationConfig(
            mask_token_id=self.tokenizer.mask_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            # Parameters from eval.sh (model_args) - unchanged
            max_new_tokens=self.max_new_tokens,
            steps=self.steps,
            temperature=self.temperature,
            top_k=self.top_k,
            alg=self.alg,
            alg_temp=self.alg_temp,
            # Parameters from yaml - override model_args
            num_return_sequences=num_return_sequences,
            return_dict_in_generate=True,
            output_history=False,
        )

        with torch.no_grad():
            # Access the underlying model when wrapped in DDP
            model = self.model.module if hasattr(self.model, "module") else self.model
            outputs = model.diffusion_generate(
                inputs=inputs.input_ids,
                generation_config=generation_config,
            )

        # Decode the generated sequences, skipping the prompt
        generated_sequences = outputs.sequences

        # Reshape to group by original prompts
        # Shape: [batch_size * num_return_sequences, seq_len]
        batch_size = len(prompts)
        num_seqs = num_return_sequences

        # Decode all sequences
        all_generated_texts = self.tokenizer.batch_decode(
            generated_sequences[:, prompt_lengths:], skip_special_tokens=True
        )

        # For now, just return the first sequence for each prompt
        # Since we changed to repeats=10, each call should generate 1 sequence
        generated_texts = []
        for i in range(batch_size):
            start_idx = i * num_seqs
            generated_texts.append(all_generated_texts[start_idx])

        return generated_texts

    @torch.no_grad()
    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False) -> List[str]:
        """
        The main generation method called by lm-harness.
        It processes requests in batches and uses our _generate_batch method.
        """
        res = []

        # Only show progress bar on main process
        disable_tqdm = disable_tqdm or not self.accelerator.is_main_process
        pbar = tqdm(
            total=len(requests),
            disable=disable_tqdm,
            desc="Running generate_until requests",
        )

        for i in range(0, len(requests), self.batch_size):
            batch_requests = requests[i : i + self.batch_size]
            contexts = [req.args[0] for req in batch_requests]

            # Extract yaml generation kwargs from first request
            first_req_args = batch_requests[0].args
            gen_kwargs = first_req_args[1] if len(first_req_args) > 1 else {}

            # Generate responses for the batch
            batch_responses = self._generate_batch(contexts, gen_kwargs)

            # Process 'until' stopping criteria
            for resp, req in zip(batch_responses, batch_requests):
                stop_sequences = req.args[1].get("until", [])
                if stop_sequences:
                    for stop_seq in stop_sequences:
                        if stop_seq in resp:
                            resp = resp.split(stop_seq)[0]
                res.append(resp)

            pbar.update(len(batch_requests))

        pbar.close()
        return res

    # The loglikelihood methods are not required for generation-based tasks
    # like HumanEval. We can leave them as not implemented to simplify the
    # initial effort.
    def loglikelihood(self, requests):
        raise NotImplementedError("Loglikelihood not implemented for this model.")

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        raise NotImplementedError("Loglikelihood rolling not implemented for this model.")

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self.accelerator.process_index

    @property
    def world_size(self):
        return self.accelerator.num_processes


if __name__ == "__main__":
    # This allows us to run the evaluation directly from the command line
    # using lm-harness's built-in argument parser.
    import sys

    from eval_utils import upload_results_after_eval

    # Remove wandb_project_name from sys.argv before calling cli_evaluate
    # since lm-harness doesn't recognize this argument
    wandb_project_name = None
    if "--wandb_project_name" in sys.argv:
        idx = sys.argv.index("--wandb_project_name")
        if idx + 1 < len(sys.argv):
            wandb_project_name = sys.argv[idx + 1]
            # Remove both the flag and its value
            sys.argv = sys.argv[:idx] + sys.argv[idx + 2 :]

    cli_evaluate()

    # Only upload to wandb if wandb_project_name was provided
    if wandb_project_name:
        upload_results_after_eval(wandb_project_name)
    else:
        print("Wandb project name not provided - skipping wandb logging")
