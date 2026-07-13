"""MultiNILM multitask loss (power regression + ON/OFF classification).

Paper-style objective (equation 16):

    L = sum_{i=1}^{A} ( L_power^i + lambda_state * L_state^i )

where A is the number of appliances.

---------------------------------------------------------------------------
TENSOR SHAPES (one training / validation batch)
---------------------------------------------------------------------------

All four inputs come from adapters/multinilm.py -> step() after the model
forward pass and the dataloader batch (x, y, z).

    power_pred    : (B, T, A)  model gated power output (normalized watts)
    state_logits  : (B, T, A)  raw ON/OFF logits (sigmoid NOT applied yet)
    power_true    : (B, T, A)  z-score normalized appliance power targets
    state_true    : (B, T, A)  binary ON/OFF targets in {0.0, 1.0}

    B = batch size          (e.g. 32 from multinilm.yaml)
    T = output timesteps    (e.g. 256 center window for UK-DALE)
    A = num appliances      (e.g. 5 UK-DALE, 4 REDD)

power_pred is already state-gated inside MultiNILM.ApplianceHead:

    soft: power_pred = power_raw * sigmoid(state_logits)
    hard: power_pred = power_raw * 1{sigmoid(state_logits) >= gate_threshold}
          (straight-through estimator during training)

Loss still supervises state_logits directly with BCEWithLogits so the
classification head receives its own gradient path.

---------------------------------------------------------------------------
HOW ONE BATCH IS SCORED (step by step)
---------------------------------------------------------------------------

1) Per-appliance POWER loss (MSE over batch and time)

   For each appliance i in {0, ..., A-1}:

       L_power^i = mean over (b, t) of ( power_pred[b,t,i] - power_true[b,t,i] )^2

   Implementation: square error tensor (B, T, A), then mean(dim=(0, 1))
   -> vector of length A.

   Example with B=2, T=3, A=2:

       err[b,t,i] = (pred[b,t,i] - true[b,t,i])^2   # shape (2, 3, 2)
       L_power^i  = mean of err[:,:,i] over all 6 values (2*3)

2) Per-appliance STATE loss (BCEWithLogits over batch and time)

   For each appliance i:

       L_state^i = mean over (b, t) of BCEWithLogits(
                       state_logits[b,t,i],
                       state_true[b,t,i],
                       pos_weight = pos_weight[i]   # optional
                   )

   BCEWithLogits internally applies sigmoid to logits, then binary cross
   entropy. We pass logits (not probabilities) for numerical stability.

   pos_weight[i] up-weights the ON class (target=1) when appliance i is
   rarely ON in the training set (see adapters/multinilm.py pos_weight:auto).

3) Sum across appliances (NOT a single global mean over all B*T*A terms)

       loss_power = L_power^0 + L_power^1 + ... + L_power^{A-1}
       loss_state = L_state^0 + L_state^1 + ... + L_state^{A-1}

   Each appliance contributes one scalar regression term and one scalar
   classification term before the final sum. This matches the paper layout
   where every appliance has its own multitask pair.

4) Total loss used for backpropagation

       loss = loss_power + lambda_state * loss_state

   lambda_state comes from config/models/multinilm.yaml (default 1.0).
   runner.py calls loss.backward() on this scalar.

5) MAE (logging only — NOT part of the training loss)

   MAE converts normalized errors back toward watts using per-appliance
   std from experiment normalization, then averages for console / CSV logs:

       mae_per_app[i] = mean_{b,t} |power_pred - power_true| * scale[i]
       mae            = mean_i mae_per_app[i]

   power_scale is target_std per appliance from the dataloader.

---------------------------------------------------------------------------
WHY SUM APPLIANCES BUT MEAN BATCH×TIME?
---------------------------------------------------------------------------

    mean_{b,t}  -> every window contributes equally regardless of batch size
    sum_i       -> each appliance adds its own task; more appliances increase
                   loss magnitude (same as summing terms in paper eq. 16)

This differs from a single MSE over all (B, T, A) elements at once, which
would implicitly average over appliances and down-weight each device when A
is large.

---------------------------------------------------------------------------
CALL SITE
---------------------------------------------------------------------------

    adapter.step() in adapters/multinilm.py
        -> loss_fn(power_pred, state_logits, y, z)
        -> StepOutput.loss  -> runner backward / checkpointing
        -> StepOutput.logs  -> loss_power, loss_state, mae, per-app breakdown
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MultiNILMLossOutput:
    """All tensors returned by MultiNILMLoss.forward for training and logging.

    Fields
    ------
    loss : scalar tensor with grad_fn — the value passed to .backward()
    loss_power : scalar — sum of per-appliance MSE terms (detached from per-app vector sum)
    loss_state : scalar — sum of per-appliance BCE terms
    mae : scalar — denormalized mean absolute power error (logging only)
    loss_power_per_appliance : shape (A,) — detached per-appliance MSE
    loss_state_per_appliance : shape (A,) — detached per-appliance BCE
    """

    loss: torch.Tensor
    loss_power: torch.Tensor
    loss_state: torch.Tensor
    mae: torch.Tensor
    loss_power_per_appliance: torch.Tensor
    loss_state_per_appliance: torch.Tensor


class MultiNILMLoss(nn.Module):
    """Per-appliance MSE + per-appliance BCEWithLogits, summed for backprop.

    Hyperparameters (from config/models/multinilm.yaml -> loss:)
    -----------------------------------------------------------
    lambda_state : float
        Multiplier on the total state loss before adding to power loss.
    pos_weight : Tensor (A,) or None
        Per-appliance positive-class weight for BCEWithLogits. When set,
        ON timesteps (target=1) are weighted more heavily than OFF.
    power_scale : float or Tensor (A,)
        Per-appliance std used only to report MAE in watts during training.
        Does not change the MSE loss (computed in normalized space).
    """

    def __init__(
        self,
        lambda_state: float = 0.1,
        pos_weight: torch.Tensor | list[float] | None = None,
        power_scale: float | list[float] | torch.Tensor = 1.0,
    ) -> None:
        super().__init__()
        self.lambda_state = float(lambda_state)

        # target_std per appliance — used for MAE logging, not for MSE.
        self.register_buffer("power_scale", torch.as_tensor(power_scale, dtype=torch.float32))

        if pos_weight is not None:
            pos_weight = torch.as_tensor(pos_weight, dtype=torch.float32)
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def _per_appliance_power_loss(
        self,
        power_pred: torch.Tensor,
        power_true: torch.Tensor,
    ) -> torch.Tensor:
        """Mean squared error per appliance, averaged over batch and time.

        Parameters
        ----------
        power_pred, power_true : (B, T, A) float tensors in normalized watts

        Returns
        -------
        Tensor of shape (A,) where entry i is:

            L_power^i = (1 / (B*T)) * sum_{b=1..B} sum_{t=1..T}
                        ( power_pred[b,t,i] - power_true[b,t,i] )^2

        Reduction is over dim 0 (batch) and dim 1 (time). Dim 2 (appliances)
        is preserved so each device gets its own scalar loss.
        """
        squared_error = (power_pred - power_true) ** 2  # (B, T, A)
        return torch.mean(squared_error, dim=(0, 1))  # (A,)

    def _per_appliance_state_loss(
        self,
        state_logits: torch.Tensor,
        state_true: torch.Tensor,
    ) -> torch.Tensor:
        """Binary cross-entropy with logits per appliance, mean over batch×time.

        Parameters
        ----------
        state_logits : (B, T, A) raw logits from MultiNILM state head
        state_true   : (B, T, A) float targets in {0.0, 1.0}

        Returns
        -------
        Tensor of shape (A,) where entry i is:

            L_state^i = mean_{b,t} BCEWithLogits( logit[b,t,i], target[b,t,i] )

        We loop over appliances because PyTorch pos_weight is per-call scalar
        or per-element; each appliance can have a different pos_weight[i].

        BCEWithLogits for one element:
            p = sigmoid(logit)
            loss = -[ y*log(p) + (1-y)*log(1-p) ]
        With pos_weight w on the positive class:
            loss = -[ w*y*log(p) + (1-y)*log(1-p) ]

        The function returns the mean over all B*T elements for appliance i.
        """
        losses: list[torch.Tensor] = []
        n_apps = state_logits.shape[-1]

        for app_i in range(n_apps):
            # Slice one appliance channel: both become (B, T).
            logits_i = state_logits[..., app_i]
            target_i = state_true[..., app_i]

            weight_i = None
            if self.pos_weight is not None:
                weight_i = self.pos_weight[app_i] if self.pos_weight.ndim > 0 else self.pos_weight

            losses.append(
                F.binary_cross_entropy_with_logits(
                    logits_i,
                    target_i,
                    pos_weight=weight_i,
                )
            )

        return torch.stack(losses)  # (A,)

    def forward(
        self,
        power_pred: torch.Tensor,
        state_logits: torch.Tensor,
        power_true: torch.Tensor,
        state_true: torch.Tensor,
    ) -> MultiNILMLossOutput:
        """Compute total multitask loss and logging metrics for one batch.

        Called once per batch from MultiNILMAdapter.step() during train/val.

        Parameters
        ----------
        power_pred   : (B, T, A) gated normalized power from model
        state_logits : (B, T, A) raw state logits from model
        power_true   : (B, T, A) normalized power from dataloader
        state_true   : (B, T, A) ON/OFF labels (threshold or CSV per config)

        Returns
        -------
        MultiNILMLossOutput — use .loss for backward(), other fields for logs
        """
        # Force FP32 for loss math (stable BCE / MSE even if model uses AMP).
        power_pred = power_pred.float()
        state_logits = state_logits.float()
        power_true = power_true.float()
        state_true = state_true.float()

        # Step 1: per-appliance vectors of shape (A,).
        loss_power_per_app = self._per_appliance_power_loss(power_pred, power_true)
        loss_state_per_app = self._per_appliance_state_loss(state_logits, state_true)

        # Step 2: paper eq. (16) — sum appliance losses into two scalars.
        loss_power = loss_power_per_app.sum()  # scalar
        loss_state = loss_state_per_app.sum()  # scalar

        # Step 3: scalar used by optimizer; only this tensor needs gradients.
        loss = loss_power + self.lambda_state * loss_state

        # Step 4: MAE in approximate watts for human-readable monitoring.
        # Not differentiated; does not affect training.
        scale = self.power_scale.to(device=power_pred.device, dtype=power_pred.dtype)
        if scale.ndim > 0:
            # Per-appliance std: (A,) broadcast over (B, T, A).
            mae_per_app = torch.mean(torch.abs(power_pred - power_true), dim=(0, 1)) * scale
            mae = mae_per_app.mean()
        else:
            mae = torch.mean(torch.abs((power_pred - power_true) * scale))

        return MultiNILMLossOutput(
            loss=loss,
            loss_power=loss_power,
            loss_state=loss_state,
            mae=mae,
            # Detach per-appliance breakdowns for logging (no grad needed).
            loss_power_per_appliance=loss_power_per_app.detach(),
            loss_state_per_appliance=loss_state_per_app.detach(),
        )
