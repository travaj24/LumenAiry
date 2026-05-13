"""
lumenairy.ao -- backward-compatibility shim.

This module moved to :mod:`lumenairy.analysis.ao` in 4.3.0 because
adaptive optics is conceptually an analysis / control layer, not a
top-level element family.  Imports from ``lumenairy.ao`` continue
to work unchanged.

Prefer :mod:`lumenairy.analysis.ao` (or the top-level
``import lumenairy as la; la.DeformableMirror``) in new code.
"""

from .analysis.ao import (
    DeformableMirror,
    apply_dm,
    zernike_modal_basis,
    slope_to_modal,
    LeakyIntegrator,
)

__all__ = [
    'DeformableMirror',
    'apply_dm',
    'zernike_modal_basis',
    'slope_to_modal',
    'LeakyIntegrator',
]
