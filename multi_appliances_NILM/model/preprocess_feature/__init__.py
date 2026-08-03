"""
Hand-designed invariant features for transferable multi-appliance NILM.

Package layout
--------------
``fractional`` — Grünwald–Letnikov (time-shift)
  1. Core:     ``gl_binomial_weights``, ``default_schirmer_alphas``
  2. NumPy:    ``fractional_derivative``, ``fractional_stack`` (+ batch)
  3. Torch:    ``FractionalFrontEnd``, ``parse_fractional_architecture``

``kle`` — Karhunen–Loève (scale / brand)
  - ACM / eig: ``autocorrelation``, ``kle_eigensystem``, ``kle_coefficients``
  - Spectrum:  ``kle_magnitude_phase``, ``kle_spectrogram_column``, ``normalize_spectrum``
  - MultiNILM channels: ``kle_subspace_channels`` (+ batch) → e.g. C=9

``schirmer_frontend`` — combine fractional (+ optional KLE maps)
  - ``fractional_channels_for_tcn`` (+ batch)
  - ``schirmer_kle_maps`` (+ batch)
"""

from .fractional import (
    FractionalFrontEnd,
    default_schirmer_alphas,
    fractional_derivative,
    fractional_derivative_batch,
    fractional_stack,
    fractional_stack_batch,
    gl_binomial_weights,
    parse_fractional_architecture,
)
from .kle import (
    autocorrelation,
    autocorrelation_matrix,
    kle_coefficients,
    kle_eigensystem,
    kle_magnitude_phase,
    kle_spectrogram_column,
    kle_subspace_channels,
    kle_subspace_channels_batch,
    normalize_spectrum,
)
from .schirmer_frontend import (
    fractional_channels_for_tcn,
    fractional_channels_for_tcn_batch,
    schirmer_kle_maps,
    schirmer_kle_maps_batch,
)

__all__ = [
    # fractional — core
    "gl_binomial_weights",
    "default_schirmer_alphas",
    # fractional — numpy
    "fractional_derivative",
    "fractional_derivative_batch",
    "fractional_stack",
    "fractional_stack_batch",
    # fractional — torch
    "FractionalFrontEnd",
    "parse_fractional_architecture",
    # kle
    "autocorrelation",
    "autocorrelation_matrix",
    "kle_coefficients",
    "kle_eigensystem",
    "kle_magnitude_phase",
    "kle_spectrogram_column",
    "kle_subspace_channels",
    "kle_subspace_channels_batch",
    "normalize_spectrum",
    # schirmer combine
    "fractional_channels_for_tcn",
    "fractional_channels_for_tcn_batch",
    "schirmer_kle_maps",
    "schirmer_kle_maps_batch",
]
