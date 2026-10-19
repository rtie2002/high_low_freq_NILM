"""Shape and objective checks for the paper-faithful TransferNILM baseline."""

from __future__ import annotations

import torch

from model.TransferNILM import TransferMultiApplianceModel, TransferNILMConfig
from model.TransferNILM_loss import TransferNILMLoss


def test_transfer_nilm_returns_one_power_and_state_sequence_per_appliance() -> None:
    cfg = TransferNILMConfig(
        window_size=16,
        hidden=16,
        attn_heads=2,
        encoder_layers=1,
        head_transformer_layers=1,
        conv_kernel_size=5,
        dropout=0.0,
    )
    model = TransferMultiApplianceModel(cfg, num_appliances=5)

    power, state = model(torch.randn(2, 16, 1))

    assert power.shape == (2, 16, 5)
    assert state.shape == (2, 16, 5)
    assert torch.all((state >= 0.0) & (state <= 1.0))


def test_transfer_nilm_loss_matches_official_mse_plus_bce() -> None:
    power_pred = torch.tensor([[[0.5, -0.5]]])
    power_true = torch.tensor([[[1.0, 0.0]]])
    state_prob = torch.tensor([[[0.8, 0.2]]])
    state_true = torch.tensor([[[1.0, 0.0]]])
    loss_fn = TransferNILMLoss(lambda_state=1.0, power_scale=[2.0, 4.0])

    output = loss_fn(power_pred, state_prob, power_true, state_true)

    expected_power = torch.nn.functional.mse_loss(power_pred, power_true)
    expected_state = torch.nn.functional.binary_cross_entropy(state_prob, state_true)
    assert torch.allclose(output.loss_power, expected_power)
    assert torch.allclose(output.loss_state, expected_state)
    assert torch.allclose(output.loss, expected_power + expected_state)
    assert torch.allclose(output.mae, torch.tensor(1.5))
