"""
AdaBN: Adaptive BatchNorm for unsupervised domain adaptation.

Reference: Li, Wang, Shi, Liu, Hou, Tian, "Revisiting Batch Normalization
for Practical Domain Adaptation", 2016 (arXiv:1603.04779).

Idea
----
Every ``BatchNorm1d`` in the trained model stores ``running_mean`` /
``running_var`` estimated purely from the *source* domain (H1/H5). At
inference time on a different house (H2), those statistics are wrong for
H2's actual activation distribution, even though every learned weight is
otherwise fine.

Fix: re-estimate just those running stats by forward-passing **unlabeled**
target-domain inputs through the network with BatchNorm layers in "train"
mode (so they use/update batch statistics) while every other layer (conv
weights, dropout, heads, ...) stays exactly as trained — no gradients, no
optimizer step, no labels. This is the cheapest, lowest-risk domain
adaptation trick available: it cannot change *what* the model has learned,
only *which distribution it assumes activations come from*.

Usage (see ``runner.evaluate_model``)::

    n_batches = adapt_batchnorm_running_stats(model, target_loader, device)
    model.eval()
    # ... run normal evaluation ...
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


@torch.no_grad()
def adapt_batchnorm_running_stats(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    max_batches: int | None = None,
) -> int:
    """Re-estimate BatchNorm running mean/var on unlabeled target-domain data.

    Only ``BatchNorm*d`` submodules are switched to train mode (so they use
    the current batch's statistics and update their running buffers via the
    usual momentum rule); everything else keeps whatever mode the caller set
    beforehand (e.g. ``model.eval()`` for dropout/heads). No gradients are
    computed and no optimizer step happens, so learned weights never change.

    Each BatchNorm's running stats are reset first and its momentum is set
    to ``None`` (PyTorch's "cumulative moving average" mode), so the result
    is a plain, order-independent average over every batch seen here — not
    an exponential moving average biased toward the last few batches.

    Args:
        model: trained model (call this *after* loading checkpoint weights).
        loader: DataLoader over the *target* split (aggregate input only is
            used; labels in the batch, if any, are ignored).
        device: device to run the forward passes on.
        max_batches: optional cap (None = use the whole loader once).

    Returns:
        Number of batches actually used for recalibration.
    """
    bn_modules = [m for m in model.modules() if isinstance(m, _BN_TYPES)]
    if not bn_modules:
        return 0

    prev_momentum = [bn.momentum for bn in bn_modules]
    for bn in bn_modules:
        bn.reset_running_stats()
        bn.momentum = None  # cumulative average over all calibration batches
        bn.train()

    n_batches = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
        model(x)
        n_batches += 1

    for bn, momentum in zip(bn_modules, prev_momentum):
        bn.eval()
        bn.momentum = momentum

    return n_batches
