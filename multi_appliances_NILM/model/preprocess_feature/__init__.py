"""Hand-designed features for MultiNILM.

Only the Grünwald–Letnikov front-end is used by the current model:

    FractionalFrontEnd : (B, 1, T) → (B, C, T)
"""

from .fractional import (
    FractionalFrontEnd,
    FractionalSettings,
    default_schirmer_alphas,
    fractional_derivative,
    fractional_derivative_batch,
    fractional_stack,
    fractional_stack_batch,
    gl_binomial_weights,
    parse_fractional_architecture,
)

__all__ = [
    "gl_binomial_weights",
    "default_schirmer_alphas",
    "fractional_derivative",
    "fractional_derivative_batch",
    "fractional_stack",
    "fractional_stack_batch",
    "FractionalFrontEnd",
    "FractionalSettings",
    "parse_fractional_architecture",
]
