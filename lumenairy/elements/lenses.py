"""
Lens and focusing element phase screens.

This module provides phase-screen models for refractive focusing elements:
thin lenses (paraxial through aplanatic), thick singlets (spherical and
aspheric), multi-surface real lenses with split-step propagation through
glass, cylindrical lenses, axicons, and GRIN rod lenses.

All functions follow the exp(-i*omega*t) time convention and use SI meters
for spatial quantities.

Backends
--------
Most functions use NumPy by default.  ``apply_thin_lens``,
``apply_spherical_lens``, ``apply_aspheric_lens``, and
``apply_real_lens`` accept a *use_gpu* flag (or auto-dispatch when
the input is a CuPy array) and run entirely on the GPU when CuPy
is installed.  Per-surface FFT propagation, phase-screen multiply,
and aperture clipping all use the array-API ``xp.*`` namespace, so
NumPy and CuPy arrays follow the same code path.  At N=32768 with
a 16-surface prescription, the GPU path is typically 5-10x faster
than the threaded pyFFTW CPU path (NVIDIA RTX 4090 vs Ryzen 7950X).

``apply_real_lens_traced`` accepts ``use_gpu=True`` for its
Newton-inversion polynomial-fit stage; the inner ray trace and
amplitude leg can also be GPU-accelerated independently via
``amp_use_gpu=True``.

Author: Andrew Traverso
"""

from __future__ import annotations

# ``_sys`` and ``_types`` are for the two-way live forward at the bottom of
# this import block; nothing else in the module uses them.
import sys as _sys
import types as _types
from typing import TYPE_CHECKING, Tuple

import numpy as np

# The lens-family LEAF: the grid-versus-aperture bookkeeping, the optional
# CuPy / numba / numexpr plumbing, the two surface-sag builders and the two
# fitting helpers.  All of it USED to live in this file, and ``_lens_real``
# (the sag builders) and ``lenses_maslov`` (the numexpr gate and the two
# fitting helpers) reached back here for it at module scope -- which is exactly
# what made those two edges import 2-cycles with this facade (WP-B11c).
# Re-exported so every existing
# ``from lumenairy.elements.lenses import surface_sag_general`` keeps
# resolving, to the SAME object.
from ._lens_kernels import (  # noqa: F401 -- re-export, see the block below
    CUPY_AVAILABLE as CUPY_AVAILABLE,
    _collect_semi_diameters as _collect_semi_diameters,
    _ensure_cupy_loaded as _ensure_cupy_loaded,
    _ensure_numexpr_loaded as _ensure_numexpr_loaded,
    _fit_normaliser as _fit_normaliser,
    _get_aspheric_sag_accum_numba as _get_aspheric_sag_accum_numba,
    _is_cupy_array as _is_cupy_array,
    _load_numba as _load_numba,
    _multi_indices_total_degree as _multi_indices_total_degree,
    _surface_sag_general as _surface_sag_general,
    _warn_if_aperture_exceeds_grid as _warn_if_aperture_exceeds_grid,
    check_grid_vs_apertures as check_grid_vs_apertures,
    recommend_grid_for_prescription as recommend_grid_for_prescription,
    surface_sag_biconic as surface_sag_biconic,
    surface_sag_general as surface_sag_general,
)

# THE LIVE HALF, which a plain re-export CANNOT carry.
#
# Some names in the leaf are not definitions but STATE.  ``cp``, ``_ne``,
# ``_numba``, ``_njit`` and ``_prange`` are ``None`` until their first use and
# are then REBOUND; ``_NUMBA_AVAILABLE`` and ``NUMEXPR_AVAILABLE`` are gates the
# test suite sets to ``False`` to take the pure-NumPy arm on a box that HAS the
# accelerator, and the kernels read them at call time.  A
# ``from ._lens_kernels import cp`` here would bind the import-time ``None``
# for ever, and a ``lenses._NUMBA_AVAILABLE = False`` would set an attribute
# that nothing reads -- turning every such test into a silent no-op.
#
# So this module forwards those names to the leaf in BOTH directions.  A module
# ``__getattr__`` (PEP 562) covers the read; the WRITE needs the module
# object's TYPE to define ``__setattr__``, which PEP 562 does not provide --
# hence the ``__class__`` assignment below, the documented way to customise a
# module's attribute protocol.  Reads and writes both land on the leaf, this
# module keeps no copy of its own, and
# ``monkeypatch.setattr(lenses, '_NUMBA_AVAILABLE', False)`` -- save, set and
# undo alike -- reaches the code that reads it.
if TYPE_CHECKING:  # pragma: no cover -- a static binding for mypy only;
    # at run time the name is a LIVE forward to the leaf (see below), so
    # this import must never execute, or the facade would keep a copy.
    from ._lens_kernels import NUMEXPR_AVAILABLE as NUMEXPR_AVAILABLE

_LIVE_FORWARD_NAMES = frozenset({
    'cp', '_ne', 'NUMEXPR_AVAILABLE',
    '_NUMBA_AVAILABLE', '_numba', '_njit', '_prange', '_NUMBA_KERNELS',
})

# THE OTHER HALF, and the one the first half's design note did not finish
# (VERIFY-WP-B11c defect D3).
#
# These eight are not state.  They are DEFINITIONS -- a constant and seven
# functions -- and the re-export above binds each of them to the SAME object
# the leaf holds, which is right for reading and is what keeps
# ``from .lenses import _warn_if_aperture_exceeds_grid`` resolving.
#
# What the re-export cannot carry is a WRITE.  Every one of them is read at
# CALL TIME out of ``_lens_kernels``'s globals by the code that moved with
# them, so ``monkeypatch.setattr(lenses, '_is_cupy_array', fake)`` binds a
# shadow in THIS module's dict that the kernel never looks at.  At the commit
# before the move that same line DID change ``surface_sag_general``'s
# behaviour, because the kernel lived here -- so the move turned a working
# substitution into a silent no-op, which is precisely the failure mode
# ``rcwa/_blas.py``'s half of WP-B11c was arranged to make LOUD (there the
# name is absent from ``_core``, so ``monkeypatch.setattr`` raises).
#
# This module cannot be loud the same way: the name has to stay readable.  So
# it is loud on the WRITE instead.  ``lenses.<one of these> = x`` and ``del
# lenses.<one of these>`` raise ``AttributeError`` naming ``_lens_kernels`` as
# the address to patch.  Reading is untouched -- same object, same dict entry,
# same ``import *`` surface, same ``dir()``.
#
# WHY NOT FORWARD THE WRITE, as the live half does?  Because forwarding a write
# while the read still comes from this module's dict ALIASES: ``lenses.X``
# would keep returning the original while ``_lens_kernels.X`` held the fake.
# Removing the re-export to fix that (making all eight live forwards) was
# measured and rejected: ``CUPY_AVAILABLE`` is the one public name among the
# eight, and taking it out of ``lenses.__dict__`` removes it from ``from
# lumenairy.elements.lenses import *`` -- a public-surface change in a
# durability fix.  Refusing the write costs nothing observable and says the
# same thing.
#
# The set and its message live INSIDE ``_LensesFacade`` (below), not at module
# scope: ``dir(lenses)`` is a bit-identity key of WP-B11c's own verification,
# and a module-level name -- private or not -- would grow it.  A class
# attribute of the module's TYPE does not appear in a module's ``__dir__``.

_KERNELS = _sys.modules[__package__ + '._lens_kernels']


class _LensesFacade(_types.ModuleType):
    """``lumenairy.elements.lenses``'s own module type: everything a plain
    module does, plus a two-way forward of ``_LIVE_FORWARD_NAMES`` to
    ``_lens_kernels``, and a refusal of any WRITE to the eight names the leaf
    owns (see the block above ``_KERNELS``)."""

    #: The eight leaf-owned re-exports.  A class attribute, not a module-level
    #: constant, so ``dir(lenses)`` does not move (see the comment above).
    _LEAF_OWNED_NAMES = frozenset({
        'CUPY_AVAILABLE', '_is_cupy_array', '_ensure_cupy_loaded',
        '_load_numba', '_get_aspheric_sag_accum_numba',
        '_ensure_numexpr_loaded', '_collect_semi_diameters',
        '_warn_if_aperture_exceeds_grid',
    })

    @staticmethod
    def _leaf_owned_refusal(name, verb):
        """The message both refusing paths raise.  One text, one address."""
        return (
            f"lumenairy.elements.lenses.{name} is a re-export of "
            f"{_KERNELS.__name__}.{name}, and the code that reads it reads it "
            f"out of {_KERNELS.__name__}'s globals at call time -- so {verb} "
            f"it HERE would bind a shadow that nothing looks at.  This is "
            f"refused rather than accepted silently (VERIFY-WP-B11c D3): the "
            f"same statement against {_KERNELS.__name__} does what you mean.  "
            f"Reading lumenairy.elements.lenses.{name} is unaffected by this "
            f"refusal; note that it returns the object bound at IMPORT time, "
            f"so read it from {_KERNELS.__name__} while a substitute is in "
            f"place.")

    def __getattr__(self, name):
        if name in _LIVE_FORWARD_NAMES:
            return getattr(_KERNELS, name)
        raise AttributeError(
            f'module {self.__name__!r} has no attribute {name!r}')

    def __setattr__(self, name, value):
        if name in _LIVE_FORWARD_NAMES:
            setattr(_KERNELS, name, value)
            return
        if name in type(self)._LEAF_OWNED_NAMES:
            raise AttributeError(
                type(self)._leaf_owned_refusal(name, 'rebinding'))
        super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in _LIVE_FORWARD_NAMES:
            delattr(_KERNELS, name)
            return
        if name in type(self)._LEAF_OWNED_NAMES:
            raise AttributeError(
                type(self)._leaf_owned_refusal(name, 'deleting'))
        super().__delattr__(name)

    def __dir__(self):
        # A module's default ``__dir__`` lists ``__dict__`` only, so without
        # this the forwarded names would VANISH from ``dir(lenses)`` -- an
        # observable surface change in a refactor whose contract is that
        # nothing observable moves.  Introspection therefore sees exactly what
        # it saw while the names were defined here.  ``_LEAF_OWNED_NAMES`` is
        # NOT unioned in: those eight are real entries in this module's dict
        # already, so adding them would be a no-op that reads as if it were
        # not.
        return sorted(set(super().__dir__()) | _LIVE_FORWARD_NAMES)


_sys.modules[__name__].__class__ = _LensesFacade

# ---------------------------------------------------------------------------
# Grid-vs-aperture safety check
#
# The census, the public check, the recommendation and the warning live in
# ``_lens_kernels.py`` -- a LEAF that imports nothing from ``lumenairy``, so a
# family module can reach it without closing a module-level import cycle with
# this facade.  They are re-exported here (the import at the top of this file),
# so every ``lenses.check_grid_vs_apertures`` and
# ``from .lenses import _warn_if_aperture_exceeds_grid`` still resolves.
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Shared Chebyshev helpers used by the Maslov and asymptotic propagators.
#
# The three Chebyshev Vandermonde helpers live in
# ``lumenairy/_math/chebyshev.py`` (see
# docs/history/lumenairy.elements.lenses.md).  The underscore-prefixed
# aliases below preserve every internal call site in this module --
# and every external import of the form
# ``from lumenairy.elements.lenses import _chebyshev_vandermonde`` --
# without forcing those callers to update their import paths.
# ---------------------------------------------------------------------------

from .._math.chebyshev import (  # noqa: F401 -- back-compat alias re-export (v5.2)
    chebyshev_derivative_vandermonde as _chebyshev_derivative_vandermonde,  # noqa: F401
    chebyshev_second_derivative_vandermonde as _chebyshev_second_derivative_vandermonde,  # noqa: F401
    chebyshev_vandermonde as _chebyshev_vandermonde,  # noqa: F401
)


def _evaluate_polynomial_4d(coeffs: np.ndarray,
                              multi_indices,
                              u1: np.ndarray, u2: np.ndarray,
                              u3: np.ndarray, u4: np.ndarray,
                              max_order: int) -> np.ndarray:
    """
    Evaluate a 4-variable Chebyshev tensor-product polynomial in
    total-degree subspace at arbitrary (u1, u2, u3, u4) samples.

    Parameters
    ----------
    coeffs : ndarray, shape (M,)
        Polynomial coefficients in the same order as ``multi_indices``.
    multi_indices : list of 4-tuples
        Multi-indices (k1, k2, k3, k4) enumerating the basis.
    u1, u2, u3, u4 : ndarrays with identical shape
        Evaluation points in [-1, 1]^4.
    max_order : int
        Maximum individual index (== total-degree cap in this call).

    Returns
    -------
    value : ndarray with the broadcast shape of (u1, ..., u4)
    """
    # 3.5.6: vectorised over basis terms (no Python loop).  T_i has
    # shape (max_k+1, *u_shape); T_i[K_i] has shape (M, *u_shape);
    # product is (M, *u_shape); contracting over basis axis with
    # coeffs gives the result.  ~3x faster than the previous loop on
    # M=70 basis terms x 1024-pt u arrays.
    T1 = _chebyshev_vandermonde(u1, max_order)
    T2 = _chebyshev_vandermonde(u2, max_order)
    T3 = _chebyshev_vandermonde(u3, max_order)
    T4 = _chebyshev_vandermonde(u4, max_order)
    K = np.asarray(multi_indices, dtype=np.int64)
    K1, K2, K3, K4 = K[:, 0], K[:, 1], K[:, 2], K[:, 3]
    basis = T1[K1] * T2[K2] * T3[K3] * T4[K4]
    return np.tensordot(np.asarray(coeffs, dtype=np.float64),
                         basis, axes=([0], [0]))


def _evaluate_polynomial_4d_and_grad34(coeffs: np.ndarray,
                                         multi_indices,
                                         u1, u2, u3, u4,
                                         max_order: int
                                         ) -> Tuple[np.ndarray,
                                                    np.ndarray,
                                                    np.ndarray]:
    """
    Evaluate the 4-variable polynomial at (u1, u2, u3, u4) and also its
    partial derivatives d/du3 and d/du4 (used for the Jacobian w.r.t.
    the v2 coordinates).

    Returns
    -------
    f, df_du3, df_du4
    """
    # 4.14.0 (Tier-2 perf, audit group): vectorised over basis terms,
    # mirroring the sibling ``_evaluate_polynomial_4d``.  The previous
    # version walked the M-term basis in a Python loop computing a
    # fused multiply-add over the evaluation grid; that loop overhead
    # is the dominant cost when this helper is hot inside the
    # asymptotic propagator's Newton iterations.
    #
    # The (M, *u_shape) basis tensors trade memory for three
    # tensordots; measured speedup vs the scalar loop at M=70 is
    # ~4.5x at an 8x8 grid, ~3.5x at 16x16, ~1.7x at 32x32 (and
    # marginally slower above ~48x48 where the (M, *u_shape)
    # intermediates spill L2 cache).  Typical
    # ``apply_real_lens_maslov`` callers run with n_v2=32, so the
    # 32x32 regime is the operational point; the cache-bound large-
    # grid regime is not exercised in production.
    T1 = _chebyshev_vandermonde(u1, max_order)
    T2 = _chebyshev_vandermonde(u2, max_order)
    T3 = _chebyshev_vandermonde(u3, max_order)
    T4 = _chebyshev_vandermonde(u4, max_order)
    dT3 = _chebyshev_derivative_vandermonde(u3, max_order)
    dT4 = _chebyshev_derivative_vandermonde(u4, max_order)
    K = np.asarray(multi_indices, dtype=np.int64)
    K1, K2, K3, K4 = K[:, 0], K[:, 1], K[:, 2], K[:, 3]
    # T12 has shape (M, *u_shape); reused for all three outputs.
    T12 = T1[K1] * T2[K2]
    c_arr = np.asarray(coeffs, dtype=np.float64)
    basis_f = T12 * T3[K3] * T4[K4]
    basis_d3 = T12 * dT3[K3] * T4[K4]
    basis_d4 = T12 * T3[K3] * dT4[K4]
    f = np.tensordot(c_arr, basis_f, axes=([0], [0]))
    df3 = np.tensordot(c_arr, basis_d3, axes=([0], [0]))
    df4 = np.tensordot(c_arr, basis_d4, axes=([0], [0]))
    return f, df3, df4


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Maslov propagator -- moved to lumenairy.elements.lenses_maslov in v3.5.5
# to reduce lenses.py bloat.  Re-exported here for backwards-compatible
# `from lumenairy.elements.lenses import apply_real_lens_maslov` imports.
# ---------------------------------------------------------------------------

# Every name below is spelled ``X as X``.  That is not redundancy: it is the
# PEP 484 / mypy marker for a DELIBERATE re-export.  Without it
# ``mypy --strict`` (which sets ``no_implicit_reexport``) reports
# "Module ... does not explicitly export attribute X" for each of the 31
# names the package root re-imports from here, and this module -- a
# compatibility shell whose whole job is to keep the legacy import paths
# working -- would be the reason ``lumenairy/__init__.py`` cannot join the
# strict whitelist.  The alternative, an ``__all__`` on this 1 100-line
# module, would also change ``import *`` behaviour; this does not.
#
# ---------------------------------------------------------------------------
# JAX-traceable real-lens propagators moved to
# lumenairy.elements._lens_jax in v3.5.5.  Re-exported here so existing
# `from lumenairy.elements.lenses import apply_real_lens_traced_jax` /
# `apply_real_lens_maslov_jax` imports continue to work.
# ---------------------------------------------------------------------------
from ._lens_jax import (  # noqa: E402
    apply_real_lens_maslov_jax as apply_real_lens_maslov_jax,
    apply_real_lens_traced_jax as apply_real_lens_traced_jax,
)

# ---------------------------------------------------------------------------
# Analytic split-step real-lens propagator moved to
# lumenairy.elements._lens_real in v3.5.5.  Re-exported here so
# existing `from lumenairy.elements.lenses import apply_real_lens`
# imports continue to work.
# ---------------------------------------------------------------------------
from ._lens_real import (  # noqa: E402
    PreparedAnalyticLens as PreparedAnalyticLens,
    apply_real_lens as apply_real_lens,
    clear_pointwise_cos_grid_cache as clear_pointwise_cos_grid_cache,
    get_lens_sag_dtype as get_lens_sag_dtype,
    get_pointwise_cos_grid_cache_budget as get_pointwise_cos_grid_cache_budget,
    lens_sag_float32_opd_error as lens_sag_float32_opd_error,
    prepare_real_lens as prepare_real_lens,
    set_lens_sag_dtype as set_lens_sag_dtype,
    set_pointwise_cos_grid_cache_budget as set_pointwise_cos_grid_cache_budget,
)

# ---------------------------------------------------------------------------
# Thin-lens / single-element phase screens were moved to
# lumenairy.elements._lens_thin in v3.5.5.  Re-exported here so existing
# `from lumenairy.elements.lenses import apply_thin_lens` etc. continue
# to work.
# ---------------------------------------------------------------------------
from ._lens_thin import (  # noqa: E402
    apply_aspheric_lens as apply_aspheric_lens,
    apply_axicon as apply_axicon,
    apply_cylindrical_lens as apply_cylindrical_lens,
    apply_grin_lens as apply_grin_lens,
    apply_spherical_lens as apply_spherical_lens,
    apply_thin_lens as apply_thin_lens,
)

# ---------------------------------------------------------------------------
# Per-pixel ray-traced apply_real_lens variant moved to
# lumenairy.elements._lens_traced in v3.5.5.  Re-exported here so
# existing
#   from lumenairy.elements.lenses import apply_real_lens_traced
#   from lumenairy.elements.lenses import close_worker_pool
# imports continue to work.
# ---------------------------------------------------------------------------
from ._lens_traced import (  # noqa: E402
    PreparedTracedLens as PreparedTracedLens,
    TiltedCarrier as TiltedCarrier,
    apply_real_lens_traced as apply_real_lens_traced,
    apply_real_lens_traced_multi as apply_real_lens_traced_multi,
    apply_real_lens_traced_segmented as apply_real_lens_traced_segmented,
    close_worker_pool as close_worker_pool,
    get_lens_parallel_amp as get_lens_parallel_amp,
    prepare_real_lens_traced as prepare_real_lens_traced,
    set_lens_parallel_amp as set_lens_parallel_amp,
)
from ._lens_traced_multibranch import (  # noqa: E402
    apply_real_lens_traced_multibranch as apply_real_lens_traced_multibranch,
)
from ._lens_traced_uniform import (  # noqa: E402
    apply_real_lens_traced_uniform as apply_real_lens_traced_uniform,
)
from .lenses_gbd import apply_real_lens_gbd as apply_real_lens_gbd  # noqa: E402
from .lenses_maslov import (  # noqa: E402
    apply_real_lens_maslov as apply_real_lens_maslov,
)
