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

"""
ColaReprAlignWrapper — drop-in `nn.Module` that bundles a Repr-Align LM
with a Cola DLM auxiliary head (`ColaDLMHead`).

Design:
- Pass-through for *every* base-LM behavior. `tasks/train_torch.py` calls
  this wrapper exactly the way it calls the raw LM; existing Repr-Align
  losses (MDM + repr_align + aux) are unchanged.
- When `self.training and cola_wt > 0`, the wrapper forces
  `output_hidden_states=True`, picks `cola_source_layer` from the
  returned tuple, and feeds it (detached by default) through the head.
  The Cola loss is scaled by `cola_wt` and added to `outputs.loss`.
- All cola_* scalars + small tensors are surfaced for wandb (see
  `tasks/train_torch.py` for the histogram-logging hook).

Why detach by default?
The student's hidden states are already shaped by MDM + Repr-Align.
Letting an under-trained Cola head also pull on them risks destabilizing
convergence. Set `cola_detach_student=False` after the head has warmed
up to enable end-to-end gradient flow.

This wrapper exposes the small pass-through methods needed by
`build_parallelize_model` (FSDP1/2), `build_optimizer`, and the
checkpoint manager.
"""

from typing import Any

import torch
import torch.nn as nn


class ColaReprAlignWrapper(nn.Module):
    def __init__(
        self,
        lm: nn.Module,
        cola_head: nn.Module,
        cola_wt: float = 0.5,
        cola_source_layer: int = -3,
        cola_detach_student: bool = True,
    ):
        super().__init__()
        self.lm = lm
        self.cola_head = cola_head
        self.cola_wt = float(cola_wt)
        self.cola_source_layer = int(cola_source_layer)
        self.cola_detach_student = bool(cola_detach_student)
        # Cheap CPU counter for gating periodic diagnostics.
        self._cola_fwd_count = 0

    # ---------------------------------------------------------------------
    # Pass-through attributes that downstream infra (FSDP, optimizer,
    # checkpointer, EnvironMeter) expects on a "model" object.
    # ---------------------------------------------------------------------

    @property
    def config(self):
        return self.lm.config

    @property
    def _no_split_modules(self):
        return list(getattr(self.lm, "_no_split_modules", []) or [])

    @property
    def teacher_model(self):
        # Repr-Align teacher is attached on the base LM (see
        # build_foundation_model with make_teacher=True). Surface it
        # through the wrapper so introspecting code keeps working.
        return getattr(self.lm, "teacher_model", None)

    @teacher_model.setter
    def teacher_model(self, value):
        self.lm.teacher_model = value

    def get_input_embeddings(self):
        return self.lm.get_input_embeddings() if hasattr(self.lm, "get_input_embeddings") else None

    def get_output_embeddings(self):
        return self.lm.get_output_embeddings() if hasattr(self.lm, "get_output_embeddings") else None

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.lm, "gradient_checkpointing_enable"):
            self.lm.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.lm, "gradient_checkpointing_disable"):
            self.lm.gradient_checkpointing_disable()

    def get_optimizer_pre_hook(self, *args, **kwargs):
        if hasattr(self.lm, "get_optimizer_pre_hook"):
            return self.lm.get_optimizer_pre_hook(*args, **kwargs)
        return None

    def get_ignore_modules_in_mixed_precision(self):
        if hasattr(self.lm, "get_ignore_modules_in_mixed_precision"):
            return self.lm.get_ignore_modules_in_mixed_precision()
        return ()

    def get_parallel_plan(self):
        if hasattr(self.lm, "get_parallel_plan"):
            return self.lm.get_parallel_plan()
        return None

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------

    def forward(self, **kwargs) -> Any:
        active = self.training and self.cola_wt > 0
        if active:
            kwargs["output_hidden_states"] = True

        outputs = self.lm(**kwargs)

        if not active:
            return outputs

        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None or len(hidden_states) == 0:
            return outputs  # nothing to do — shouldn't happen given the flag

        h = hidden_states[self.cola_source_layer]
        if self.cola_detach_student:
            h = h.detach()

        cola_out = self.cola_head(h)
        aux_loss = self.cola_wt * cola_out["loss"]

        if outputs.loss is not None:
            outputs.loss = outputs.loss + aux_loss
        else:
            outputs.loss = aux_loss

        # ------------------------------------------------------------------
        # Scalar diagnostics → loss_components (the existing training loop
        # reduces this dict across DP ranks and logs each entry as
        # `losses/<name>` to wandb).
        # ------------------------------------------------------------------
        loss_components = getattr(outputs, "loss_components", None)
        if loss_components is None or not isinstance(loss_components, dict):
            loss_components = {}

        with torch.no_grad():
            z = cola_out["z"]
            z_global = cola_out["z_global"]
            z_local = cola_out["z_local"]
            z_pred = cola_out["z_pred"]
            target = cola_out["target"]

            loss_components["cola_diff"] = float(cola_out["loss"].detach())
            loss_components["cola_t_mean"] = float(cola_out["t_mean"].detach())

            # Combined latent geometry
            loss_components["cola_z_norm"] = float(z.norm(dim=-1).mean().detach())
            loss_components["cola_z_std"] = float(z.std().detach())
            loss_components["cola_z_mean"] = float(z.mean().detach())
            loss_components["cola_z_max"] = float(z.abs().max().detach())

            # Per-scale latent stats (catch one stream collapsing)
            loss_components["cola_z_global_norm"] = float(z_global.norm(dim=-1).mean().detach())
            loss_components["cola_z_local_norm"] = float(z_local.norm(dim=-1).mean().detach())
            loss_components["cola_z_global_std"] = float(z_global.std().detach())
            loss_components["cola_z_local_std"] = float(z_local.std().detach())

            # Diffusion fit: cosine(prediction, target) and signal/error-std SNR
            #   target = (noise - z)  under FM (velocity)
            #   target = z            under x0-prediction
            tgt_flat = target.float().flatten()
            zp_flat = z_pred.float().flatten()
            loss_components["cola_pred_cosine"] = float(
                torch.nn.functional.cosine_similarity(tgt_flat, zp_flat, dim=0).detach()
            )
            err = (target - z_pred).float()
            loss_components["cola_pred_snr"] = float(
                (target.float().std() / (err.std() + 1e-8)).detach()
            )

        outputs.loss_components = loss_components

        # ------------------------------------------------------------------
        # Side channel for expensive metrics that the train loop logs
        # only periodically (histograms). Detached CPU tensors so they
        # survive the optimizer step. Small by design — z_global is
        # ~G*dim, z_local ~L*dim.
        # ------------------------------------------------------------------
        self._cola_fwd_count += 1
        outputs.cola_extras = {
            "z_global": z_global.detach().float().cpu(),
            "z_local": z_local.detach().float().cpu(),
            "t_mean": float(cola_out["t_mean"].detach()),
            "fwd_count": self._cola_fwd_count,
        }

        return outputs
