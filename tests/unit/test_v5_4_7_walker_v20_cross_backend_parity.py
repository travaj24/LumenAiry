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

import ast
import importlib
import inspect
import re
import textwrap

# JAX-backed modules that host physics twins.
_JAX_TWIN_MODULES = (
    'lumenairy.raytrace.jax_trace',
    'lumenairy.elements._lens_jax',
    'lumenairy.propagators.asymptotic_jax_twin',
    'lumenairy.analysis.through_focus',
    'lumenairy.propagators.system',
    'lumenairy.elements.rcwa.oned',     # v5.5.3: RCWA jax twin (rcwa/ split 1D/2D)
    'lumenairy.elements.coatings',      # v5.5.3: cover the differentiable TMM twin
    'lumenairy.elements.pmm.oned',      # 2026-06-07 audit P2-C: PMM jax twin (pmm/ split)
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
    # audit-2609 WP-A1: the shared exit-vertex transfer (signed t = -z/N to the
    # vertex plane) has one NumPy body and one JAX body; both must move together.
    'lumenairy.raytrace.jax_trace:exit_vertex_transfer_jax':
        'lumenairy.raytrace.exit_vertex:exit_vertex_transfer',
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
    # v5.5.3: the RCWA jax twin is now a deprecated thin wrapper over the
    # unified, backend-dispatched rcwa_efficiency_1d (its NumPy sibling), so a
    # cross-backend fix can't land on one path only.
    'lumenairy.elements.rcwa.oned:rcwa_efficiency_1d_jax':
        'lumenairy.elements.rcwa.oned:rcwa_efficiency_1d',
    # v5.5.3: the differentiable thin-film TMM companion mirrors the NumPy
    # real-Snell coating_reflectance to machine precision, so a physics fix
    # (Snell chain, Abeles factors, TIR cap) can't drift between the two.
    'lumenairy.elements.coatings:coating_reflectance_jax':
        'lumenairy.elements.coatings:coating_reflectance',
    # 2026-06-07 audit P2-C: the PMM jax twin is a thin wrapper over the unified,
    # backend-dispatched pmm_efficiency_1d (its NumPy sibling), so the differentiable
    # PMM path is no longer structurally invisible to the parity walker.
    'lumenairy.elements.pmm.oned:pmm_efficiency_1d_jax':
        'lumenairy.elements.pmm.oned:pmm_efficiency_1d',
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

# ----------------------------------------------------------------------
# The root-pick contract, expressed STRUCTURALLY (AST), not as a grep.
#
# The property under test is about the EXPRESSION that picks the surface
# intersection root, so the check is written against the expression.  A text
# search cannot tell the difference between code and the comment above it: the
# previous form of this pin searched ``inspect.getsource`` for the literal
# ``'t1 if R'`` and for ``'R_finite > 0'``, which a comment quoting the
# forbidden selector trips (it needed a hand-written ``str.replace`` to hide
# one such comment from itself), while a REAL regression spelled
# ``t1 if radius > 0`` or ``jnp.where(R_safe > 0, t1, t2)`` slips through
# untouched.  Both failure directions go away once the matcher walks the AST,
# where comments do not exist and the shape is what is compared.
#
# REQUIRED: the near root is assigned from a direction-aware expression --
# either the explicit ``jnp.where(|t1| <= |t2|, t1, t2)`` min-modulus pick, or
# the Spencer-Murty stable-quadratic quotient ``e / guard(q)``, which IS the
# near root by construction (|e/q| <= |q/a|).
#
# FORBIDDEN: a root chosen from the SIGN OF THE RADIUS -- ``t1 if R > 0 else
# t2`` and its array twin ``jnp.where(R > 0, t1, t2)``.  That selector is
# direction-blind: it ignores where the ray actually starts, so a ray
# travelling the other way takes the far root.
# ----------------------------------------------------------------------

#: The operand a direction-BLIND selector compares against zero: a radius or a
#: curvature, under any of the spellings these kernels use.
_RADIUS_NAME = re.compile(r'^(r|radius|curv|curvature|cc|c)(_|$)', re.I)


def _attr_chain(node):
    """``jnp.where`` -> ``'jnp.where'``; a bare Name -> its id; else ``''``."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return '.'.join(reversed(parts))
    return ''


def _is_where_call(node):
    return (isinstance(node, ast.Call)
            and _attr_chain(node.func).split('.')[-1] == 'where'
            and len(node.args) == 3)


def _is_abs_call(node):
    return (isinstance(node, ast.Call)
            and _attr_chain(node.func).split('.')[-1] in ('abs', 'absolute')
            and len(node.args) == 1)


def _is_min_modulus_pick(node):
    """``where(abs(a) <= abs(b), a, b)`` -- the explicit min-|t| root pick."""
    if not _is_where_call(node):
        return False
    test, lo, hi = node.args
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1
            and isinstance(test.ops[0], (ast.LtE, ast.Lt))):
        return False
    if not (_is_abs_call(test.left) and _is_abs_call(test.comparators[0])):
        return False
    a = _attr_chain(test.left.args[0])
    b = _attr_chain(test.comparators[0].args[0])
    return bool(a) and bool(b) and (_attr_chain(lo), _attr_chain(hi)) == (a, b)


def _is_stable_quadratic_quotient(node):
    """``e / where(<guard>, q, <const>)`` -- the Spencer-Murty near root.

    The guarded divisor is the whole point: it is what makes the quotient the
    NEAR root for a ray going either way, and the ``where`` is the
    divide-by-zero guard a traced kernel needs.
    """
    if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)):
        return False
    if not _is_where_call(node.right):
        return False
    _guard, divisor, fallback = node.right.args
    return (bool(_attr_chain(node.left)) and bool(_attr_chain(divisor))
            and isinstance(fallback, ast.Constant))


def _is_direction_blind_pick(node):
    """``t1 if R > 0 else t2`` / ``where(R > 0, t1, t2)`` -- the regression.

    Both branches must be plain names (the two roots) and the test must
    compare a radius/curvature against zero; that combination is the defect
    and nothing else in these kernels has that shape.
    """
    if isinstance(node, ast.IfExp):
        test, lo, hi = node.test, node.body, node.orelse
    elif _is_where_call(node):
        test, lo, hi = node.args
    else:
        return False
    if not (isinstance(lo, ast.Name) and isinstance(hi, ast.Name)):
        return False
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1
            and isinstance(test.ops[0], (ast.Gt, ast.GtE, ast.Lt, ast.LtE))):
        return False
    rhs = test.comparators[0]
    if not (isinstance(rhs, ast.Constant)
            and isinstance(rhs.value, (int, float)) and rhs.value == 0):
        return False
    return bool(_RADIUS_NAME.match(_attr_chain(test.left).split('.')[-1]))


def _root_pick_verdict(tree):
    """``(has_direction_aware_pick, [blind sites])`` for one function's AST."""
    aware = False
    blind = []
    for node in ast.walk(tree):
        if _is_min_modulus_pick(node) or _is_stable_quadratic_quotient(node):
            aware = True
        if _is_direction_blind_pick(node):
            blind.append(ast.unparse(node))
    return aware, blind


def _kernel_tree(fn):
    return ast.parse(textwrap.dedent(inspect.getsource(fn)))


class _BlindRewriter(ast.NodeTransformer):
    """Replace every direction-AWARE root pick with the forbidden selector.

    Keyed on the EXPRESSION, not on the name it is assigned to, because the
    two kernels spell that name differently (``t_near`` / ``t_sphere``) and a
    falsifiability fixture that depended on the spelling would quietly stop
    exercising a kernel the day someone renamed a local.
    """

    def __init__(self):
        self.n = 0

    def generic_visit(self, node):
        node = super().generic_visit(node)
        if _is_min_modulus_pick(node) or _is_stable_quadratic_quotient(node):
            self.n += 1
            return ast.parse('t1 if R_finite > 0 else t2').body[0].value
        return node


def _make_direction_blind(tree):
    """The falsifiability fixture for the matchers above: it produces exactly
    the regression the pin exists to catch, from the REAL kernel, so the
    assertions are shown to discriminate rather than merely to pass."""
    rewriter = _BlindRewriter()
    return ast.fix_missing_locations(rewriter.visit(tree)), rewriter.n


def test_jax_intersect_direction_aware_root_pick_present():
    """The v5.4.1/v5.4.6 direction-aware near-root pick must remain in
    BOTH JAX intersect kernels (the P1-1 parity fix).

    audit-2609 WP-A1 (R4): the kernels take the near root from the
    Spencer-Murty stable quadratic ``t = e/q`` with
    ``q = -(b + sign(b) sqrt(disc))/2`` -- the near root by construction
    (|e/q| <= |q/a|), direction-aware without an explicit ``min |t|`` -- and
    ``_intersect_jax`` keeps the explicit ``min |t|`` pick on its
    pure-spherical branch.  Either spelling is the direction-aware pick; a
    direction-blind ``t1 if R>0 else t2`` is neither.

    The check is STRUCTURAL (see the matchers above): it walks each kernel's
    AST for the root-pick EXPRESSION, so a comment quoting the forbidden
    selector cannot fail it and a regression spelled with different variable
    names cannot pass it.
    """
    import lumenairy.raytrace.jax_trace as jt
    for kernel in (jt._intersect_jax, jt._intersect_jax_param):
        aware, blind = _root_pick_verdict(_kernel_tree(kernel))
        assert aware, (
            f"V20: {kernel.__name__} carries neither the explicit min-|t| root "
            f"pick nor the Spencer-Murty near-root quotient; a direction-blind "
            f"``t1 if R>0 else t2`` regressed the JAX twin.")
        assert not blind, (
            f"V20: {kernel.__name__} reintroduced the direction-blind "
            f"selector: {blind}")


def test_the_root_pick_matcher_rejects_a_direction_blind_kernel():
    """Falsifiability for the pin above (TESTING_STANDARDS V1).

    A structural matcher that accepted everything would pass the pin forever
    while the kernel was rewritten underneath it.  Each real kernel is
    mutated IN MEMORY into the exact regression the audit found -- every
    direction-aware root-pick expression replaced by
    ``t1 if R_finite > 0 else t2`` -- and the verdict must flip on BOTH
    counts: the direction-aware expression gone, the blind selector reported.

    Nothing is written; the mutation is on a parsed copy of the source.
    """
    import lumenairy.raytrace.jax_trace as jt
    for kernel in (jt._intersect_jax, jt._intersect_jax_param):
        mutated, n = _make_direction_blind(_kernel_tree(kernel))
        assert n >= 1, (
            f"{kernel.__name__}: no direction-aware root-pick expression to "
            f"mutate, so this falsifiability check would prove nothing about "
            f"it")
        aware, blind = _root_pick_verdict(mutated)
        assert not aware, (
            f"{kernel.__name__}: the direction-aware matcher still fires on a "
            f"kernel whose root pick is ``t1 if R_finite > 0 else t2`` -- it "
            f"is matching something other than the root-pick expression")
        assert len(blind) == n, (
            f"{kernel.__name__}: the direction-blind matcher found "
            f"{len(blind)} of {n} injected selectors")


def test_the_root_pick_matcher_is_not_a_text_search():
    """The regression this test file's own previous form could not see.

    A comment quoting the forbidden selector must NOT fail the pin, and a
    regression spelled with different variable names must NOT pass it.  Both
    are asserted on synthetic sources, so neither claim depends on how the
    shipped kernels happen to be written today.
    """
    quoting_comment = ast.parse(textwrap.dedent("""
        def k(t1, t2, R_finite):
            # The direction-blind ``t1 if R_finite > 0 else t2`` selector is
            # exactly what this must not do.
            t_near = jnp.where(jnp.abs(t1) <= jnp.abs(t2), t1, t2)
            return t_near
    """))
    aware, blind = _root_pick_verdict(quoting_comment)
    assert aware and not blind, (
        'a comment quoting the forbidden selector must not fail the pin')

    renamed_regression = ast.parse(textwrap.dedent("""
        def k(t_a, t_b, curvature):
            t_near = jnp.where(curvature > 0, t_a, t_b)
            return t_near
    """))
    aware, blind = _root_pick_verdict(renamed_regression)
    assert not aware and len(blind) == 1, (
        'a direction-blind pick spelled with different names must fail the '
        f'pin (aware={aware}, blind={blind})')


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
