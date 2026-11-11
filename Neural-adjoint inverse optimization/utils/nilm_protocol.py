"""Backward-compatible name for :mod:`utils.protocol`.

New code should import ``utils.protocol``.  This file preserves any notebook
or early experiment that already imports ``utils.nilm_protocol``.
"""

from .protocol import NILMProtocol, load_protocol, save_inverse_evaluation

__all__ = ["NILMProtocol", "load_protocol", "save_inverse_evaluation"]
