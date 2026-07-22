"""
Core Optical Propagation Module (v5.1.0 thin re-export shell)
==============================================================

Provides exact and approximate free-space propagation of coherent optical
fields on discrete 2-D grids.

Methods implemented:
    - Angular Spectrum Method (ASM) -- exact, band-limited
    - Tilted / off-axis ASM        -- ASM with carrier frequency removal
    - Single-FFT Fresnel           -- paraxial, changes output grid spacing
    - Fraunhofer (far-field)       -- single FFT, large-z limit
    - Rayleigh-Sommerfeld          -- convolution with Green's function
    - Scalable Angular Spectrum    -- variable output pitch (Heintzmann 2023)
    - MFT propagators              -- arbitrary output grid via Bluestein

Convention
----------
Time dependence:  exp(-i*omega*t)  throughout.
Units:            SI meters for all spatial quantities.

Return-type contract
--------------------
Propagators that **preserve the grid spacing** (ASM, tilted ASM,
Rayleigh-Sommerfeld) return the field as a bare ``ndarray``::

    E_out = angular_spectrum_propagate(E, z, lam, dx)
    E_out = rayleigh_sommerfeld_propagate(E, z, lam, dx)

Propagators that **change the grid spacing** (Fresnel, Fraunhofer)
return a 3-tuple ``(E_out, dx_out, dy_out)`` so callers can resample
or update their pixel pitch::

    E_out, dx_out, dy_out = fresnel_propagate(E, z, lam, dx)
    E_out, dx_out, dy_out = fraunhofer_propagate(E, z, lam, dx)

This split is intentional and stable -- code that treats the bare
return as iterable will fail loudly rather than silently miscompute.

Backends
--------
    NumPy   -- CPU, always available (default)
    CuPy    -- GPU, auto-detected at import time
    pyFFTW  -- multi-threaded CPU FFT, opt-in via USE_PYFFTW flag

v5.1.0 split (Agent C)
----------------------
The formerly-monolithic 4103-line propagation.py was reorganised into
six submodules sharing one FFT/cache/config infrastructure layer:

* :mod:`lumenairy.propagators.fft_infra` -- backend dispatch + plan
  cache + ASM transfer-function caches + DEFAULT_* knobs + setters /
  getters + ``_validate_propagator_inputs``.
* :mod:`lumenairy.propagators.asm`       -- ASM, tilted ASM, batched
  ASM, ``apply_fresnel_curvature``, ``_build_asm_H_square``.
* :mod:`lumenairy.propagators.fresnel`   -- single-FFT Fresnel and
  Fraunhofer (natural FFT output grid).
* :mod:`lumenairy.propagators.rs`        -- Rayleigh-Sommerfeld.
* :mod:`lumenairy.propagators.sas`       -- Scalable Angular Spectrum.
* :mod:`lumenairy.propagators.mft`       -- MFT / Bluestein variants of
  Fresnel / Fraunhofer / ASM, plus :func:`resample_field`.

This module is now a thin re-export shell.  Every name that pre-v5.1.0
was importable from ``lumenairy.propagators.propagation`` continues to
resolve from here bit-for-bit; the split is purely a file-level
reorganisation with no behaviour change.

The 4 ``set_default_*`` setters and 4 ``get_default_*`` accessors keep
their canonical module path here.  The mutable ``DEFAULT_*`` globals
live in :mod:`fft_infra` and are reachable via attribute access on
this module (``propagation.DEFAULT_COMPLEX_DTYPE``) thanks to the
module-level ``__getattr__`` below, which forwards unknown names to
``fft_infra`` so that setter-driven updates are observed live.

Author:  Andrew Traverso
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Module-level __getattr__ for live attribute-access forwarding.
# ---------------------------------------------------------------------------
#
# Setters in ``fft_infra`` mutate the canonical globals in that module.
# Callers who do ``import lumenairy.propagators.propagation as p;
# p.DEFAULT_COMPLEX_DTYPE`` expect to see the latest value.  PEP 562's
# module-level ``__getattr__`` is only invoked when the named attribute
# is NOT found in the module's normal namespace; the re-exports above
# bound the names to their import-time values, so attribute lookup
# normally hits those stale bindings.
#
# To make attribute-access return the LIVE value we forward to
# ``fft_infra`` via a small whitelist for the names whose value can
# change at runtime.  The ``from X import Y`` bindings above still
# capture the import-time snapshot (matching pre-v5.1.0 behaviour for
# the ``from lumenairy.propagators.propagation import Y`` pattern),
# but ``module.X`` lookup gets the current value -- which is what
# every in-library consumer does.
#
# We delete the stale module-level bindings for the live-forwarded
# names so attribute lookup falls through to ``__getattr__`` and
# returns the current value rather than the import-time snapshot.
from . import fft_infra as _fft_infra
from .asm import (
    _build_asm_H_square,
    angular_spectrum_propagate,
    angular_spectrum_propagate_batch,
    angular_spectrum_propagate_tilted,
    apply_fresnel_curvature,
)
from .carrier import (
    CarrierReferencedField,
    TracedCarrierChainResult,
    carrier_referenced_aperture,
    carrier_referenced_envelope,
    carrier_referenced_exact_focus_readout,
    carrier_referenced_fit_radius,
    carrier_referenced_focus_readout,
    carrier_referenced_reconstruct,
    propagate_carrier_referenced,
    propagate_traced_carrier_chain,
)

# ---------------------------------------------------------------------------
# Re-exports from the v5.1.0 infrastructure / kernel submodules.
# ---------------------------------------------------------------------------
#
# The public surface of pre-v5.1.0 ``propagation.py`` is exactly the
# union of the explicit names below.  Every external import path
# ``from lumenairy.propagators.propagation import <NAME>`` resolves
# through this re-export.
#
# Note on ``DEFAULT_*`` globals: ``from .fft_infra import
# DEFAULT_COMPLEX_DTYPE`` binds the name HERE to whatever ``fft_infra``
# has at this import moment.  Subsequent ``set_default_*`` calls mutate
# ``fft_infra.DEFAULT_*`` but NOT this module's local binding -- which
# matches the pre-v5.1.0 semantic (callers who did ``from
# lumenairy.propagators.propagation import DEFAULT_COMPLEX_DTYPE`` got
# a stale snapshot too).  For LIVE forwarding via attribute access
# (``propagation.DEFAULT_COMPLEX_DTYPE``) the module-level
# ``__getattr__`` at the bottom handles the lookup.
from .fft_infra import (
    _ASM_CACHE_LOCK,
    _BANDLIMIT_CACHE,
    _DEFAULT_DY_NO_CONSUMER_WARNED,
    # Back-compat latches
    _DEFAULT_WAVE_PROPAGATOR_NO_CONSUMER_WARNED,
    # ASM caches + helpers
    _FREQ_GRID_CACHE,
    _H_CACHE,
    _PYFFTW_BAD_SHAPES,
    # pyFFTW plan-cache internals
    _PYFFTW_PLAN_CACHE,
    _PYFFTW_PLAN_LOCK,
    # Backend flags + loaders
    CUPY_AVAILABLE,
    # Default-config knobs
    DEFAULT_COMPLEX_DTYPE,
    DEFAULT_DY,
    DEFAULT_REAL_DTYPE,
    DEFAULT_WAVE_PROPAGATOR,
    FFTW_MIN_SIZE,
    # FFT backend config (globals + setters)
    FFTW_THREADS,
    PYFFTW_AVAILABLE,
    PYFFTW_FALLBACK_ON_ERROR,
    SCIPY_FFT_AVAILABLE,
    SCIPY_FFT_WORKERS,
    USE_PYFFTW,
    USE_SCIPY_FFT,
    _build_plan_entry,
    _clear_local_asm_caches,
    _ensure_cupy_loaded,
    _ensure_pyfftw_loaded,
    _entry_bytes,
    # FFT dispatchers
    _fft2,
    _fft2_nd,
    _get_or_make_bandlimit,
    _get_or_make_freq_grids,
    _get_or_make_plan,
    _h_cache_lookup,
    _h_cache_store,
    _handle_pyfftw_failure,
    _ifft2,
    _ifft2_nd,
    _is_cupy_array,
    _promote_entry_to_measure,
    # JAX dtype resolvers
    _resolve_jax_complex_dtype,
    _resolve_jax_real_dtype,
    _scipy_or_numpy_fft2,
    _scipy_or_numpy_ifft2,
    # Validation
    _validate_propagator_inputs,
    clear_asm_caches,
    cp,
    get_asm_cache_size,
    get_default_complex_dtype,
    get_default_dy,
    get_default_real_dtype,
    get_default_wave_propagator,
    get_fft_auto_promote,
    get_fft_double_buffer,
    get_fft_plan_cache_size,
    get_fft_threads,
    get_pyfftw_planner,
    pyfftw,
    reset_fft_backend,
    restore_fft_state,
    set_asm_cache_size,
    set_default_complex_dtype,
    set_default_dy,
    set_default_real_dtype,
    set_default_wave_propagator,
    set_fft_auto_promote,
    set_fft_double_buffer,
    set_fft_fallback,
    set_fft_plan_cache_size,
    set_fft_threads,
    set_pyfftw_planner,
    snapshot_fft_state,
    warmup_fft_plans,
)
from .fresnel import (
    fraunhofer_propagate,
    fresnel_propagate,
    fresnel_tf_propagate,
)
from .mft import (
    angular_spectrum_propagate_mft,
    fraunhofer_propagate_mft,
    fresnel_propagate_mft,
    resample_field,
)
from .rs import (
    rayleigh_sommerfeld_propagate,
)
from .sas import (
    scalable_angular_spectrum_propagate,
)

# Names whose value can change at runtime via a setter.  Listed here
# explicitly so we don't accidentally shadow unrelated module attrs.
#
# v5.1.1 (audit P3-NEW-F1-1): ``_PYFFTW_BAD_SHAPES`` belongs here.
# ``reset_fft_backend()`` rebinds it via ``_PYFFTW_BAD_SHAPES = set()``
# (line 647 of fft_infra.py) -- a new set object, not just an in-place
# ``.clear()`` on the existing one.  Consumers reading
# ``propagation._PYFFTW_BAD_SHAPES`` after a reset would see the
# pre-reset snapshot (still holding the old "skip this shape" memos)
# instead of the fresh empty set.  Live-forwarding routes the lookup
# through ``__getattr__`` so the current value is always returned.
_LIVE_FORWARD_NAMES = frozenset({
    'DEFAULT_COMPLEX_DTYPE',
    'DEFAULT_REAL_DTYPE',
    'DEFAULT_WAVE_PROPAGATOR',
    'DEFAULT_DY',
    'FFTW_THREADS',
    'USE_PYFFTW',
    'USE_SCIPY_FFT',
    'SCIPY_FFT_WORKERS',
    'FFTW_MIN_SIZE',
    'PYFFTW_FALLBACK_ON_ERROR',
    '_PYFFTW_PLAN_FLAGS',
    '_PYFFTW_AUTO_PROMOTE',
    '_PYFFTW_AUTO_PROMOTE_THRESHOLD',
    '_PYFFTW_AUTO_PROMOTE_LOGGED',
    '_PYFFTW_PLAN_CACHE_SIZE',
    # v5.17.1: single/double aligned-buffer mode, mutated by
    # set_fft_double_buffer() -- live-forwarded so consumer reads of
    # ``propagation._PYFFTW_DOUBLE_BUFFER`` track the current value
    # (V14 walker requirement; caught by the release verify gate).
    '_PYFFTW_DOUBLE_BUFFER',
    '_PYFFTW_BAD_SHAPES',
    '_H_CACHE_SIZE',
    '_FREQ_GRID_CACHE_SIZE',
    '_BANDLIMIT_CACHE_SIZE',
    '_H_CACHE_MAX_BYTES_PER_ENTRY',
    '_H_CACHE_MAX_TOTAL_BYTES',
    'cp',
    'pyfftw',
})

# Drop the stale import-time-snapshot bindings so attribute lookup
# falls through to ``__getattr__``.  Without this step, the
# ``from .fft_infra import DEFAULT_COMPLEX_DTYPE`` above pins
# ``propagation.DEFAULT_COMPLEX_DTYPE`` to its import-time value and
# subsequent setter calls aren't observed via attribute access.
for _name in _LIVE_FORWARD_NAMES:
    if _name in globals():
        del globals()[_name]
del _name


def __getattr__(name):
    if name in _LIVE_FORWARD_NAMES:
        return getattr(_fft_infra, name)
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# __all__: every legacy public + private name that this module exposes.
# ---------------------------------------------------------------------------

__all__ = [
    # ASM family
    'angular_spectrum_propagate',
    'angular_spectrum_propagate_tilted',
    # v5.1.0 (V9 walker symmetry): ``angular_spectrum_propagate_batch``
    # + internal FFT config flags accessible as module attributes but
    # not in ``__all__`` (they're not in lumenairy.__all__ either).
    'angular_spectrum_propagate_mft',
    'apply_fresnel_curvature',
    # Fresnel / Fraunhofer
    'fresnel_propagate',
    'fresnel_tf_propagate',
    'fresnel_propagate_mft',
    'fraunhofer_propagate',
    'fraunhofer_propagate_mft',
    # Carrier-referenced (Sziklas-Siegman) free-space step
    'CarrierReferencedField',
    'TracedCarrierChainResult',
    'propagate_carrier_referenced',
    'carrier_referenced_reconstruct',
    'carrier_referenced_envelope',
    'carrier_referenced_aperture',
    'carrier_referenced_fit_radius',
    'carrier_referenced_focus_readout',
    'carrier_referenced_exact_focus_readout',
    'propagate_traced_carrier_chain',
    # Rayleigh-Sommerfeld
    'rayleigh_sommerfeld_propagate',
    # SAS
    'scalable_angular_spectrum_propagate',
    # Grid bridge
    'resample_field',
    # Default-config knobs
    'DEFAULT_COMPLEX_DTYPE',
    'DEFAULT_REAL_DTYPE',
    'DEFAULT_WAVE_PROPAGATOR',
    'DEFAULT_DY',
    'set_default_complex_dtype',
    'get_default_complex_dtype',
    'set_default_real_dtype',
    'get_default_real_dtype',
    'set_default_wave_propagator',
    'get_default_wave_propagator',
    'set_default_dy',
    'get_default_dy',
    # FFT backend
    'CUPY_AVAILABLE',
    'PYFFTW_AVAILABLE',
    'set_fft_threads',
    'get_fft_threads',
    'set_fft_fallback',
    'set_pyfftw_planner',
    'get_pyfftw_planner',
    'set_fft_plan_cache_size',
    'get_fft_plan_cache_size',
    'set_fft_double_buffer',
    'get_fft_double_buffer',
    'snapshot_fft_state',
    'restore_fft_state',
    'warmup_fft_plans',
    'set_fft_auto_promote',
    'get_fft_auto_promote',
    'reset_fft_backend',
    # ASM cache control
    'set_asm_cache_size',
    'get_asm_cache_size',
    'clear_asm_caches',
]
