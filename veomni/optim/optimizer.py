# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.optim.optimizer import Optimizer

from ..utils.import_utils import is_torch_npu_available

import math

class SCALE(torch.optim.Optimizer):
    """
     SCALE: Stochastic Column-normalized Last-layer Momentum Optimizer
    """

    def __init__(
        self,
        lr=1e-4,
        wd=0.0,
        main_params=None,
        secondary_params=None,
        oned_params=None,
        id_to_name=None,
        debug=False,
        momentum=0.9,
        adam_lr=None,
        adamw_betas=(0.9, 0.999),
        adamw_eps=1e-8,
        
    ):

        if adam_lr is None:
            adam_lr = lr 

        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            adam_lr=adam_lr,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        params = list(main_params)
        
        secondary_params = list(secondary_params) if secondary_params is not None else []
        params.extend(secondary_params)
 
        oned_params = list(oned_params) if oned_params is not None else []
        params.extend(oned_params)

        self.id_to_name = id_to_name
        self.debug = debug
        self.max_lr = lr
        
        super().__init__(params, defaults)
        
        for p in main_params:
            self.state[p]["param_type"] = "main_param"
        for p in secondary_params:
            self.state[p]["param_type"] = "secondary_param"
        for p in oned_params:
            self.state[p]["param_type"] = "oned_param"

    def step(self, closure=None):
        """Perform a single optimization step.
        
        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
                
        for group in self.param_groups:

            #############################
            #   Main+Secondary Params   #
            #############################

            params = [p for p in group["params"] if (self.state[p]["param_type"] == "main_param") or (self.state[p]["param_type"] == "secondary_param") ]
            lr = group["lr"]
            wd = group["wd"]
            beta1 = group["momentum"]

            for p in params:    
                # sanity check
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g is not None
 
                if self.debug:
                    print("Main: ", self.id_to_name[id(p)])
                
                if self.id_to_name is not None and "embed_tokens" in self.id_to_name[id(p)]:  
                    col_dim = 0
                else:
                    col_dim = 1
                
                # calc update
                state = self.state[p]

                if self.state[p]["param_type"] == "secondary_param":
                    # add momentum 
                    if "moment1" not in state:
                        state["moment1"] = torch.zeros_like(g)
                    buf1 = state["moment1"]
                    buf1.lerp_(g, 1 - beta1)
                    g = buf1

                var = torch.mean(torch.square(g), dim=col_dim, keepdim=True)
                s = torch.sqrt(var).clamp_min_(1e-8)
                u = g / s
                
                # apply weight decay
                p.data.mul_(1 - lr * wd)
                
                # apply update
                p.data.add_(u, alpha=-lr)
                
                if self.debug:
                    print("p.data.dtype: ", p.data.dtype, "u.dtype: ",  u.dtype)


            ############################
            #       Oned Params        #
            ############################


            params = [p for p in group["params"] if (self.state[p]["param_type"] == "oned_param") ]

            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]
            
            lr =  group['lr']
            adam_lr = group['adam_lr'] 
            max_lr = self.max_lr
            
            lr = (adam_lr / max_lr) * lr 
            
            for p in params:          
                g = p.grad
                if g is None:
                    continue
                
                if self.debug:
                    print("1D (AdamW): ", self.id_to_name[id(p)])
                    print("Adam lr = ", lr) 
                
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())
                
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)                

        return loss
    
# https://github.com/meta-llama/llama-recipes/blob/v0.0.4/src/llama_recipes/policies/anyprecision_optimizer.py
class AnyPrecisionAdamW(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
        use_kahan_summation=True,
        momentum_dtype=torch.bfloat16,
        variance_dtype=torch.bfloat16,
        compensation_buffer_dtype=torch.bfloat16,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "use_kahan_summation": use_kahan_summation,
            "momentum_dtype": momentum_dtype,
            "variance_dtype": variance_dtype,
            "compensation_buffer_dtype": compensation_buffer_dtype,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """

        if closure is not None:
            with torch.enable_grad():
                closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            use_kahan_summation = group["use_kahan_summation"]

            momentum_dtype = group["momentum_dtype"]
            variance_dtype = group["variance_dtype"]
            compensation_buffer_dtype = group["compensation_buffer_dtype"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.grad.is_sparse:
                    raise RuntimeError("AnyPrecisionAdamW does not support sparse gradients.")

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0)

                    # momentum - EMA of gradient values
                    state["exp_avg"] = torch.zeros_like(p, dtype=momentum_dtype)

                    # variance uncentered - EMA of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=variance_dtype)

                    # optional Kahan summation - accumulated error tracker
                    if use_kahan_summation:
                        state["compensation"] = torch.zeros_like(p, dtype=compensation_buffer_dtype)

                # Main processing
                # update the steps for each param group update
                state["step"] += 1
                step = state["step"]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                grad = p.grad

                if weight_decay:  # weight decay, AdamW style
                    p.data.mul_(1 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)  # update momentum
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)  # update uncentered variance

                bias_correction1 = 1 - beta1**step  # adjust using bias1
                step_size = lr / bias_correction1

                denom_correction = (1 - beta2**step) ** 0.5  # adjust using bias2 and avoids math import
                centered_variance = (exp_avg_sq.sqrt() / denom_correction).add_(eps, alpha=1)

                if use_kahan_summation:  # lr update to compensation
                    compensation = state["compensation"]
                    compensation.addcdiv_(exp_avg, centered_variance, value=-step_size)

                    # update weights with compensation (Kahan summation)
                    # save error back to compensation for next iteration
                    temp_buffer = p.detach().clone()
                    p.data.add_(compensation)
                    compensation.add_(temp_buffer.sub_(p.data))
                else:  # usual AdamW updates
                    p.data.addcdiv_(exp_avg, centered_variance, value=-step_size)


def build_scale_param_groups(
    model: nn.Module,
    *,
    target_modules: Sequence[str] = ("attn", "mlp", "attention", "embed_tokens"),
    debug: bool = False,
) -> Tuple[List[nn.Parameter], List[nn.Parameter], List[nn.Parameter], Dict[int, str]]:
    """
    Build SCALE param groups:
      - Main params: weights from Linear/Embedding modules matching target_modules
      - Oned params: 1D parameters (biases, layer norms, etc.) not in main params
      - Secondary params: other 2D+ parameters not in main params

    Returns:
      (main_params, secondary_params, oned_params, id_to_name)
    """
    main_params: List[nn.Parameter] = []
    oned_params: List[nn.Parameter] = []
    secondary_params: List[nn.Parameter] = []
    
    id_to_name_main_params = {}
    id_to_name_secondary_params = {}
    id_to_name_oned_params = {}

    # Collect main params from Linear and Embedding modules
    for module_name, module in model.named_modules():
        if not (isinstance(module, nn.Linear) or isinstance(module, nn.Embedding)):
            continue
        if not any(target_key in module_name for target_key in target_modules):
            continue
        
        main_params.append(module.weight)
        id_to_name_main_params[id(module.weight)] = module_name

    # Collect secondary and oned params
    for param_name, p in model.named_parameters():
        if id(p) in id_to_name_main_params:
            continue
        
        if p.ndim == 1:
            oned_params.append(p)
            id_to_name_oned_params[id(p)] = param_name
        else:
            secondary_params.append(p)
            id_to_name_secondary_params[id(p)] = param_name

    # Debug printing
    if debug:
        print(f"MAIN MODULES = {target_modules} !!!")
        for module_name, module in model.named_modules():
            if hasattr(module, 'weight'):
                p = module.weight
                if id(p) in id_to_name_main_params:
                    print("Main module: ", module_name)
                if id(p) in id_to_name_oned_params:
                    print("1D module: ", module_name)
                if id(p) in id_to_name_secondary_params:
                    print("Secondary module: ", module_name)

    id_to_name = {**id_to_name_main_params, **id_to_name_secondary_params, **id_to_name_oned_params}
    
    # Print parameter counts
    print(f"Number of main parameters: {sum(p.numel() for p in main_params if p.requires_grad)}")
    print(f"Number of secondary parameters: {sum(p.numel() for p in secondary_params if p.requires_grad)}")
    print(f"Number of 1D parameters: {sum(p.numel() for p in oned_params if p.requires_grad)}")
    
    return main_params, secondary_params, oned_params, id_to_name


def build_galore_param_groups(
    model: nn.Module,
    *,
    # selection
    target_modules: Sequence[str] = ("attn", "mlp"),
    # GaLore hyperparameters (defaults match your parse_args)
    rank: int = 128,                 # --rank
    update_proj_gap: int = 50,       # --update_proj_gap
    galore_scale: float = 1.0,       # --galore_scale
    proj_type: str = "std",          # --proj_type
    # misc
    include_bias: bool = False,      # keep False to mirror your script (weights only)
) -> Tuple[List[dict], List[nn.Parameter], List[nn.Parameter]]:
    """
    Build GaLore param groups:
      - GaLore group: weights (and optionally biases) of nn.Linear modules whose name
        contains any token in `target_modules` (e.g., "attn", "mlp").
      - Regular group: all remaining parameters.

    Returns:
      (param_groups, galore_params, regular_params)
    """
    galore_params: List[nn.Parameter] = []

    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not any(tok in module_name for tok in target_modules):
            continue

        # Mirror your training script: only the weight is adapted by GaLore
        galore_params.append(module.weight)
        if include_bias and getattr(module, "bias", None) is not None:
            galore_params.append(module.bias)

    id_galore = {id(p) for p in galore_params}
    regular_params: List[nn.Parameter] = [p for p in model.parameters() if id(p) not in id_galore]

    param_groups: List[dict] = [
        {"params": regular_params},
        {
            "params": galore_params,
            "rank": rank,
            "update_proj_gap": update_proj_gap,
            "scale": galore_scale,
            "proj_type": proj_type,
        },
    ]
    # print the number of parameters in the galore group and regular group
    print(f"Number of parameters in the galore group: {sum(p.numel() for p in galore_params if p.requires_grad)}")
    print(f"Number of parameters in the regular group: {sum(p.numel() for p in regular_params if p.requires_grad)}")
    return param_groups, galore_params, regular_params


def build_apollo_param_groups(
    model: nn.Module,
    *,
    # selection
    target_modules: Sequence[str] = ("attn", "mlp"),
    # APOLLO hyperparameters (explicit, no args object)
    rank: int = 256,                 # --rank
    update_proj_gap: int = 200,      # --update_proj_gap
    apollo_scale: float = 1.0,       # --apollo_scale
    proj_type: str = "std",          # --proj_type
    proj: str = "random",            # --proj
    scale_type: str = "channel",     # --scale_type
) -> Tuple[List[dict], List[nn.Parameter], List[nn.Parameter]]:
    """
    Build APOLLO param groups:
      - Low-rank group: weights of nn.Linear whose module name contains any token in `target_modules`
      - Regular group: all remaining parameters

    Returns:
      (param_groups, lowrank_params, regular_params)
    """
    lowrank_params: List[nn.Parameter] = []

    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not any(tk in module_name for tk in target_modules):
            continue
        # only the weight goes to low-rank group (mirrors your training script)
        lowrank_params.append(module.weight)

    id_low = {id(p) for p in lowrank_params}
    regular_params: List[nn.Parameter] = [p for p in model.parameters() if id(p) not in id_low]

    param_groups = [
        {"params": regular_params},
        {
            "params": lowrank_params,
            "rank": rank,
            "update_proj_gap": update_proj_gap,
            "scale": apollo_scale,
            "proj_type": proj_type,
            "proj": proj,
            "scale_type": scale_type,
        },
    ]
    # print the number of parameters in the low-rank group and regular group
    print(f"Number of parameters in the low-rank group: {sum(p.numel() for p in lowrank_params if p.requires_grad)}")
    print(f"Number of parameters in the regular group: {sum(p.numel() for p in regular_params if p.requires_grad)}")
    return param_groups, lowrank_params, regular_params


def build_optimizer(
    model: "nn.Module",
    lr: float = 1e-3,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    weight_decay: float = 1e-2,
    fused: bool = False,
    optimizer_type: str = "adamw",
    param_groups: Optional[Sequence[Dict[str, Any]]] = None,
) -> "torch.optim.Optimizer":
    if param_groups is None:
        param_groups = filter(lambda p: p.requires_grad, model.parameters())

    if optimizer_type == "adamw":
        foreach = False if is_torch_npu_available() else (not fused)
        fused = False if is_torch_npu_available() else fused
        optim = AdamW(param_groups, lr, betas, eps, weight_decay, fused=fused, foreach=foreach)
    elif optimizer_type == "anyprecision_adamw":
        optim = AnyPrecisionAdamW(param_groups, lr, betas, eps, weight_decay)
    elif optimizer_type == "apollo":
        from apollo_torch import APOLLOAdamW
        print("Building APOLLO param groups...")
        param_groups, _, _ = build_apollo_param_groups(
            model,
            rank=1, #256,
            update_proj_gap=200,
            apollo_scale=128, # 1.0,
            proj_type="std",
            proj="random",
            scale_type="tensor", #"channel",
        )
        # Create APOLLO optimizer using training args for lr, betas, weight_decay
        optim = APOLLOAdamW(param_groups, lr=lr)
    elif optimizer_type == "galore":
        from galore_torch import GaLoreAdamW, GaLoreAdamW8bit, GaLoreAdafactor
        print("Building GaLore param groups...")
        param_groups, _, _ = build_galore_param_groups(
            model,
            rank=128,
            update_proj_gap=50,
            galore_scale=1.0,
            proj_type="std",
            target_modules=("attn", "mlp"),
        )
        optim = GaLoreAdamW(param_groups, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    elif optimizer_type == "scale":
        print("Building SCALE param groups...")
        main_params, secondary_params, oned_params, id_to_name = build_scale_param_groups(
            model,
            target_modules=("attn", "mlp", "attention", "embed_tokens"),
            debug=False,
        )
        # Create SCALE optimizer with training args
        optim = SCALE(
            lr=lr,
            wd=weight_decay,
            main_params=main_params,
            secondary_params=secondary_params,
            oned_params=oned_params,
            id_to_name=id_to_name,
            debug=False,
            momentum=0.9,
            adam_lr=lr,  # Use same lr as main by default
            adamw_betas=betas,
            adamw_eps=eps,
        )
    else:
        raise ValueError("Only adamw, anyprecision_adamw, apollo, galore, and scale are supported as optimizers.")

    return optim
