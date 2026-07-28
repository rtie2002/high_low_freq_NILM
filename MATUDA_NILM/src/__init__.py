"""MATUDA source package: independent multi-appliance UDA (FC-layer domain loss)."""

from .matuda_model import MATUDANet, count_parameters
from .matuda_loss import MATUDACriterion, multilayer_domain_loss

__all__ = [
    "MATUDANet",
    "count_parameters",
    "MATUDACriterion",
    "multilayer_domain_loss",
]
