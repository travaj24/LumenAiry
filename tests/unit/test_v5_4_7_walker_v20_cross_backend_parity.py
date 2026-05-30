"""V20 -- cross-backend (NumPy <-> JAX) parity walker.

v5.4.7 (audit AUDIT_V5_4_6 #3 / Part 9).  Closes the dominant k=6
meta-pattern the deep self-audits named: "canonical-path fix,
peripheral-path drift" -- a physics fix lands on the NumPy/canonical path
but leaks past its JAX twin (the v5.4.1 intersect root-pick that never
reached ``_intersect_jax``; the JonesField ``dy`` drop; the JAX RNG dtype;
the ``_lens_jax`` grid transpose; ...).

This walker installs the structural defense the audit said was missing:
a REGISTRY of NumPy<->JAX physics-path twin pairs.  It enforces three
contracts on every commit:

1. **Completeness** -- every public ``*jax*`` physics function discovered
   in the JAX-backed modules must be REGISTERED (with its NumPy sibling)
   or explicitly exempted.  A new JAX twin that lands without registering
   its NumPy sibling fails here, forcing the author to declare the parity
   contract.
2. **Both-backends-present** -- for every registered pair, BOTH the JAX
   twin and its NumPy sibling must resolve.  Removing or renaming one
   twin without the other fails here.
3. **Parity-fix pins** -- the specific cross-backend fixes the audits
   landed (JAX intersect direction-aware root pick; x64-aware JAX RNG
   dtype) must stay in place.

The walker needs no JAX install: the twin modules import fine without
``jax`` (it is lazy-imported at call time), and the checks are symbol
existence / source pins, not execution.
"""
from __future__ import annotations

import importlib
import inspect

# JAX-backed modules that host physics twins.
_JAX_TWIN_MODULES = (
    'lumenairy.raytrace.jax_trace',
    'lumenairy.elements._lens_jax',
    'lumenairy.propagators.asymptotic_jax_twin',
    'lumenairy.analysis.through_focus',
    'lumenairy.propagators.system',
)

# Non-physics JAX helpers (ray-state builders, cache clearers) -- excluded
# from the parity contract by prefix/suffix.
_UTILITY_PREFIXES = ('make_', 'clear_', 'raybundle_')
_UTILITY_SUFFIXES = ('_cache', '_state')

# Registry: public JAX twin -> its NumPy sibling.  Both are
# ``module:attr`` strings.
_PARITY_REGISTRY = {
    'lumenairy.raytrace.jax_trace:trace_jax':
        'lumenairy.raytrace.trace:trace',
    'lumenairy.raytrace.jax_trace:trace_jax_with_params':
        'lumenairy.raytrace.trace:trace_prescription',
    'lumenairy.elements._lens_jax:apply_real_lens_traced_jax':
        'lumenairy.elements._lens_traced:apply_real_lens_traced',
    'lumenairy.elements._lens_jax:apply_real_lens_maslov_jax':
        'lumenairy.elements.lenses_maslov:apply_real_lens_maslov',
    'lumenairy.analysis.through_focus:through_focus_scan_jax':
        'lumenairy.analysis.through_focus:through_focus_scan',
    'lumenairy.analysis.through_focus:monte_carlo_tolerancing_jax':
        'lumenairy.analysis.through_focus:monte_carlo_tolerancing',
    'lumenairy.propagators.system:propagate_through_system_jax':
        'lumenairy.propagators.system:propagate_through_system',
    'lumenairy.propagators.asymptotic_jax_twin:aberration_tensor_lg00_jax':
        'lumenairy.propagators.asymptotic_aberration_tensor:aberration_tensor',
    'lumenairy.propagators.asymptotic_jax_twin:propagate_modal_asymptotic_lg00_jax':
        'lumenairy.propagators.asymptotic:propagate_modal_asymptotic',
    'lumenairy.propagators.asymptotic_jax_twin:solve_envelope_stationary_jax_ift':
        'lumenairy.propagators.asymptotic_canonical_fit:solve_envelope_stationary',
    'lumenairy.propagators.asymptotic_jax_twin:fit_canonical_polynomials_jax':
        'lumenairy.propagators.asymptotic_canonical_fit:fit_canonical_polynomials',
}

# Discovered ``*jax*`` functions intentionally WITHOUT a physics NumPy
# sibling.  Additions must carry a comment explaining why no sibling exists.
_V20_PARITY_EXEMPTIONS = frozenset({
    # State converter (JaxRayState -> RayBundle), not a physics path.  Its
    # inverse ``raybundle_to_jax_state`` is excluded by the make_/raybundle_
    # utility-prefix filter; this direction reads ``jax_state_...`` so it is
    # listed explicitly.
    'lumenairy.raytrace.jax_trace:jax_state_to_raybundle',
})


def _resolve(ref):
    """Resolve a ``module:attr`` ref to its object, or None."""
    modname, attr = ref.split(':')
    try:
        return getattr(importlib.import_module(modname), attr, None)
    except ImportError:
        return None


def _discover_jax_twins():
    """All public ``*jax*`` physics functions defined in the twin modules
    (excluding ray-state / cache utility helpers)."""
    found = set()
    for modname in _JAX_TWIN_MODULES:
        m = importlib.import_module(modname)
        for name, obj in inspect.getmembers(m, inspect.isfunction):
            if getattr(obj, '__module__', None) != modname:
                continue  # imported symbol, not defined here
            if 'jax' not in name.lower() or name.startswith('_'):
                continue
            if name.startswith(_UTILITY_PREFIXES) or name.endswith(_UTILITY_SUFFIXES):
                continue
            found.add(f'{modname}:{name}')
    return found


# ----------------------------------------------------------------------
# Contract 1 -- registry completeness
# ----------------------------------------------------------------------

def test_every_jax_twin_is_registered():
    discovered = _discover_jax_twins()
    known = set(_PARITY_REGISTRY) | _V20_PARITY_EXEMPTIONS
    missing = discovered - known
    assert not missing, (
        f"V20: JAX physics twin(s) not registered: {sorted(missing)}.  "
        f"Add each to ``_PARITY_REGISTRY`` mapping it to its NumPy sibling "
        f"(so cross-backend fixes can't silently land on one path only), or "
        f"to ``_V20_PARITY_EXEMPTIONS`` with a rationale if it has no NumPy "
        f"sibling.")


def test_registry_has_no_stale_entries():
    """Every registered JAX twin must still be discoverable (catch a twin
    removed/renamed without updating the registry)."""
    discovered = _discover_jax_twins()
    stale = set(_PARITY_REGISTRY) - discovered
    assert not stale, (
        f"V20: registry entries no longer found among the discovered JAX "
        f"twins: {sorted(stale)}.  Update ``_PARITY_REGISTRY`` to match the "
        f"current twin names.")


# ----------------------------------------------------------------------
# Contract 2 -- both backends present for every pair
# ----------------------------------------------------------------------

def test_registered_pairs_have_both_backends():
    for jax_ref, np_ref in _PARITY_REGISTRY.items():
        assert _resolve(jax_ref) is not None, (
            f"V20: JAX twin {jax_ref} is missing")
        assert _resolve(np_ref) is not None, (
            f"V20: NumPy sibling {np_ref} for {jax_ref} is missing -- a fix "
            f"on the JAX path now has no canonical counterpart to mirror")


# ----------------------------------------------------------------------
# Contract 3 -- specific cross-backend parity-fix pins
# ----------------------------------------------------------------------

def test_jax_intersect_direction_aware_root_pick_present():
    """The v5.4.1/v5.4.6 direction-aware near-root pick must remain in
    BOTH JAX intersect kernels (the P1-1 parity fix)."""
    import lumenairy.raytrace.jax_trace as jt
    src = inspect.getsource(jt)
    n = src.count('jnp.where(jnp.abs(t1) <= jnp.abs(t2), t1, t2)')
    assert n >= 2, (
        f"V20: expected the direction-aware root pick in both JAX intersect "
        f"kernels (_intersect_jax + _intersect_jax_param); found {n}.  A "
        f"direction-blind ``t1 if R>0 else t2`` regressed the JAX twin.")


def test_jax_rng_default_dtype_is_x64_aware():
    """The JAX RNG default dtype must follow ``result_type`` (x64-aware),
    matching NumPy/CuPy -- not a hard-coded 32-bit dtype (F-30/F-31)."""
    from lumenairy.backend.random import RandomState
    src = inspect.getsource(RandomState)
    assert 'result_type(float)' in src and 'result_type(int)' in src, (
        "V20: the JAX RandomState default dtype must use "
        "``jax.numpy.result_type(float/int)`` for NumPy/CuPy parity")
    assert 'jax.numpy.float32' not in src and 'jax.numpy.int32' not in src, (
        "V20: hard-coded 32-bit JAX RNG default dtype regressed the "
        "cross-backend parity fix")
