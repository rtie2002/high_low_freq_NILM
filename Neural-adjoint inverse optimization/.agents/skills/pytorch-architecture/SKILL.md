---
name: pytorch-architecture
description: "Design, edit, or explain PyTorch nn.Module neural-network architectures, forward passes, tensor-shape transformations, feature fusion, temporal encoders, attention, gradient flow and model output heads. Use when creating or changing model.py/layers, especially for NILM or inverse-model experiments. Do not trigger for routine Python formatting."
---

# PyTorch Architecture

## Goal
An architecture where a reader can follow the actual tensor flow from model input through each meaningful block to the prediction.

## Workflow
1. Inspect expected input/output shapes, data layout (batch, channel, time), precision/device, and task objective.
2. Identify the conceptual blocks as they truly exist. Keep names aligned with their mathematical function, not marketing labels.
3. Define `__init__` in architecture order; write `forward()` in matching top-down execution order.
4. Use clear intermediate names (`temporal_features`, `context_features`, `prediction`) and inline shape comments at transformations where needed.
5. Explain non-obvious residuals, skip connections, attention scaling, normalization, multi-branch concatenation and pooling (axis included).
6. Do not introduce generic layer factories, registries, dynamic dispatch or deeply nested wrappers for a single research model.
7. Test with a representative synthetic tensor: output shape, dtype/device, finite values, and backward pass when appropriate. Match old predictions for semantics-preserving refactors when possible.

## Example
```python
# power_window: [batch, channels=1, time]
local_features = self.temporal_encoder(power_window)  # [B, hidden, T']
context_features = self.context_block(local_features)  # [B, hidden, T']
predicted_power = self.output_head(context_features)   # shape depends on task
```
These lines are illustrative only; derive *actual* shapes from the architecture. Do not imply a model has a particular module if it doesn't.

## Architectural clarity
- Keep mathematical parameters discoverable in one place when that improves research iteration.
- Use `torch.nn.functional` explicitly if it improves readability; do not hide operations behind multi-hop helpers.
- Preserve tensor gradients; avoid unnecessary `.detach()`, `.item()`, `.numpy()` or in-place operations in differentiable paths.

## Deliverable
A readable architecture narrative, true tensor shapes, and concrete smoke-test evidence or limitations.
