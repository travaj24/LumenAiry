"""The modal branch cut, round 2: ONE ``_sqrt_decay``, and what the five PMM
copies of it cost.

WHAT ROUND 1 DID.  ``rcwa/_core._sqrt_decay`` maps a layer eigenvalue ``lam^2``
to its modal decay constant ``lam = sqrt(lam^2)``.  A PROPAGATING mode of a
LOSSLESS layer has ``lam^2`` exactly real NEGATIVE -- on the principal square
root's branch cut, where the OUTGOING root ``+i|kz|`` (the branch the region
modes are built on) and the INCOMING root ``-i|kz|`` differ only by the sign of
``Im(lam^2)``, which for an ``eig`` output is the eigensolver's backward error
and not physics.  Round 1 replaced an exact ``Re(r) == 0`` pin -- which an
``eig`` output never satisfies -- with a band relative to the mode spectrum and
a ``conj`` flip that keeps ``Re(lam) >= 0``.  See
``docs/audits/FIX_RCWA_EVEN_SECTOR_WSL_2026_09_11.md``.

WHAT ROUND 2 FOUND.  Its independent verification (defect D1,
``docs/audits/VERIFY_RCWA_EVEN_SECTOR_2026_09_11.md``) established that FIVE
modules under ``lumenairy/elements/pmm/`` carried their own ``_sqrt_decay``
with the exact-zero pin still in it, and that 5-9 propagating modes per
``pmm_efficiency_2d_cell`` solve came back INCOMING as a result.  It could not
establish whether that could make an answer WRONG rather than merely
build-dependent.  It can, and the partner is not a region:

  a three-layer ``PMM2DStackHybrid`` -- uniform ``eps = 2.25`` spacer, a
  weakly modulated ``eps = 2.25`` cell, uniform spacer, ``n_substrate = 1.63``
  so NOTHING coincides with a half-space -- returned per-order efficiencies
  **2.0035e-03** away from an independent RCWA solve of the same geometry, at
  a lossless closure defect of **-1.665e-04**, where the same stack without the
  spacers reads **1.0e-14**.  A uniform LAYER's modes are built by the analytic
  Rayleigh helper in EXACT arithmetic exactly as a half-space REGION's are, so
  a mis-rooted mode of the neighbouring STRUCTURED layer is that uniform
  layer's own BACKWARD mode.

WHAT ROUND 2 SHIPS.  ONE definition.  ``rcwa/_core._sqrt_decay`` gained an
``xp`` and a ``band`` parameter (defaults reproducing the round-1 body bit for
bit) and the five private copies were deleted: the NumPy modules import it, the
three JAX twins call it with ``jax.numpy`` so the traced body is the same
object the eager path runs.  ``pmm/twod_staggered.py``'s copy was DEAD and is
simply gone.

MEASURED POPULATIONS.  Every bar below was derived on BOTH builds -- Windows
py3.14 / numpy 2.4.4 / scipy-openblas Haswell and WSL py3.12 / numpy 2.4.6 /
scipy-openblas SkylakeX -- over ``OPENBLAS_NUM_THREADS`` in {1, 2, 4, 8} and
unpinned, because the thread count is exactly what moved the pre-fix readings.
Probes and JSON: ``validation/probe_fix_branch_cut_round2/``; report:
``docs/audits/FIX_BRANCH_CUT_ROUND2_2026_09_11.md``.
"""
from __future__ import annotations

import pathlib
import re
import warnings

import numpy as np
import pytest

import lumenairy.elements.pmm.twod as _ptw
from lumenairy.elements.pmm import PMM2DStackHybrid
from lumenairy.elements.rcwa import RCWAStack
from lumenairy.elements.rcwa import _core as _rc

_PKG = pathlib.Path(_rc.__file__).resolve().parents[3] / "lumenairy"

# --------------------------------------------------------------- the fixture
# The uniform-spacer stack.  ``HOST`` is the patterned layer's background AND
# the spacers' permittivity; the pillar is a relative 1e-6 from it, so the
# hybrid PMM's own Fourier truncation error is at the arithmetic floor and the
# closure defect is a clean instrument rather than a reading dominated by
# truncation.  ``n_substrate = 1.63`` coincides with nothing.
_WL, _P, _D = 0.6e-6, 0.5e-6, 0.2e-6
_HOST, _N_SUB = 2.25, 1.63
_WEAK = _HOST * (1.0 + 1e-6)
_S = 6


def _cell(pillar=_WEAK):
    c = np.full((_S, _S), _HOST + 0j)
    c[2:4, 2:4] = pillar
    return c


def _pmm_stack(n_orders=4, pillar=_WEAK, spacer=_HOST, n_sub=_N_SUB):
    st = PMM2DStackHybrid(_P, _P, n_substrate=n_sub, n_superstrate=1.0,
                          degree=7, n_orders=n_orders, symmetry=False)
    if spacer is not None:
        st.add_layer(0.1e-6, eps=spacer)
    st.add_layer(_D, eps_cell=_cell(pillar))
    if spacer is not None:
        st.add_layer(0.1e-6, eps=spacer)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return st.set_source(_WL, theta=0.0).solve()


def _closure(res):
    """``sum R + sum T - 2``.  A provably lossless cell conserves energy
    EXACTLY at any truncation, so this is an independent oracle whose error
    floor is the arithmetic -- it needs no reference solve and no prior
    reading.  Two incident polarizations, hence 2 rather than 1."""
    return float(np.sum(np.asarray(res[1])) + np.sum(np.asarray(res[2])) - 2.0)


def _pre_round1_sqrt_decay(x, xp=None, band=1e-8):
    """The pre-round-1 body, re-typed: the EXACT ``Re(r) == 0`` pin and the
    ``-r`` flip.  Installed by :func:`_pre_arm` so every fail-before in this
    file is ENGINEERED rather than hoped for from a build -- the shape
    ``docs/TESTING_STANDARDS.md`` asks for, and the only one available now that
    the defect no longer occurs anywhere in the shipped tree."""
    from lumenairy.backend.array import array_namespace
    if xp is None:
        xp = array_namespace(x)
    x = xp.asarray(x).astype(complex)
    r = xp.sqrt(x)
    on_cut = r.real == 0
    return xp.where(on_cut & (r.imag < 0), -r, r)


_BOUND = ("lumenairy.elements.rcwa._core", "lumenairy.elements.rcwa.oned",
          "lumenairy.elements.rcwa.stack", "lumenairy.elements.pmm.twod",
          "lumenairy.elements.berreman")


class _pre_arm:
    """Reinstate the pre-round-1 branch body at every module that binds the
    shared function, for the duration of a ``with`` block."""

    def __init__(self):
        self._saved = []

    def __enter__(self):
        import importlib
        for name in _BOUND:
            mod = importlib.import_module(name)
            if hasattr(mod, "_sqrt_decay"):
                self._saved.append((mod, mod._sqrt_decay))
                mod._sqrt_decay = _pre_round1_sqrt_decay
        return self

    def __exit__(self, *a):
        for mod, fn in self._saved:
            mod._sqrt_decay = fn
        return False


# ==================================================================== GATE 1
# The consolidation itself, asserted the way the library's other multi-copy
# regressions are pinned: on the SOURCE, so a copy cannot come back.

def test_exactly_one_definition_of_sqrt_decay_exists_in_the_library():
    """SIX bodies of one function is the shape that bred the factor-i defect
    (audit S1-8, six generator copies) and the branch-cut defect (this one,
    six copies again -- five of which round 1 did not reach).  The decision is
    that there is exactly ONE.

    Grep-based on purpose, in the shape of the census / dispatcher pins: an
    identity check on imported names cannot see a copy that is never imported,
    and ``pmm/twod_staggered.py``'s copy -- DEAD code that still carried the
    old pin -- was exactly that.
    """
    defs = []
    for path in sorted(_PKG.rglob("*.py")):
        for i, line in enumerate(
                path.read_text(encoding="utf-8", errors="replace")
                .splitlines(), 1):
            if re.match(r"\s*def\s+_sqrt_decay\b", line):
                defs.append("%s:%d" % (path.relative_to(_PKG).as_posix(), i))
    assert defs == ["elements/rcwa/_core.py:%d"
                    % _rc._sqrt_decay.__code__.co_firstlineno], (
        "expected exactly one definition of _sqrt_decay, in "
        "elements/rcwa/_core.py; found: " + ", ".join(defs))


def test_no_exact_zero_branch_pin_survives_anywhere_in_the_library():
    """The defect in one line: ``on_cut = r.real == 0`` on a value that came
    out of ``eig``, whose real part is the backward error and not zero.  No
    EXECUTABLE comparison may test a float's real or imaginary part against an
    exact zero.

    Parsed with ``ast`` rather than grepped, so that the comments and
    docstrings which quote the removed pin -- and which are how the fix records
    what it removed -- are invisible to it while a reintroduced line of code is
    not.

    SCOPE: equality only.  ``eps.imag != 0`` on a CALLER-SUPPLIED permittivity
    asks "did the user hand me a lossy material?", which is an exact question
    about an exact input and is correct where it appears
    (``elements/eme/_jax_modes.py``'s lossy-discard warning).  The defect is an
    exact ``==`` used to SELECT A BRANCH of a value that came out of a
    floating-point eigensolve.
    """
    import ast

    def is_zero(node):
        if isinstance(node, ast.Constant) and isinstance(
                node.value, (int, float)) and node.value == 0:
            return True
        return (isinstance(node, ast.UnaryOp)
                and isinstance(node.op, ast.USub) and is_zero(node.operand))

    def is_re_im(node):
        return (isinstance(node, ast.Attribute)
                and node.attr in ("real", "imag"))

    bad = []
    for path in sorted(_PKG.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8",
                                            errors="replace"))
        except SyntaxError:                              # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            operands = [node.left] + list(node.comparators)
            for op, left, right in zip(node.ops, operands, operands[1:]):
                if not isinstance(op, ast.Eq):
                    continue
                if ((is_re_im(left) and is_zero(right))
                        or (is_re_im(right) and is_zero(left))):
                    bad.append("%s:%d" % (path.relative_to(_PKG).as_posix(),
                                          node.lineno))
    assert not bad, ("exact-zero branch test(s) reintroduced at: "
                     + ", ".join(sorted(set(bad))))


def test_every_former_copy_site_now_resolves_to_the_shared_function():
    """The NumPy PMM module must bind the SAME object, and the three JAX twins
    must import it rather than define one.  This is the identity half of the
    claim whose source half is gate 1."""
    assert _ptw._sqrt_decay is _rc._sqrt_decay
    import inspect
    for name in ("_jax_twod", "_jax_stack2d", "_jax_twod_jones"):
        mod = __import__("lumenairy.elements.pmm." + name, fromlist=[name])
        src = inspect.getsource(mod)
        code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
        assert "def _sqrt_decay" not in code, \
            "%s defines its own _sqrt_decay again" % name
        assert "_sqrt_decay(" in code, \
            "%s no longer calls the shared _sqrt_decay at all" % name
        assert "_sqrt_decay(-jnp.concatenate" in code or \
               "_sqrt_decay(lam2, jnp)" in code, \
            "%s calls _sqrt_decay without handing it jnp" % name


# ==================================================================== GATE 2
# The refactor half: the shared body must be BIT-IDENTICAL to round 1's.
#
# ``xp=None`` and ``band=_CUT_BAND_REL`` are defaults chosen so that every
# round-1 call site is unchanged.  That is a claim about bits, so it is
# asserted on bits, over an engineered set that includes every corner the
# round-1 verification exercised.

def _round1_body(x):
    from lumenairy.backend.array import array_namespace
    xp = array_namespace(x)
    x = xp.asarray(x).astype(complex)
    r = xp.sqrt(x)
    scale = xp.maximum(xp.max(xp.abs(r)), 1.0) if r.size else 1.0
    on_cut = xp.abs(r.real) <= _rc._CUT_BAND_REL * scale
    return xp.where(on_cut & (r.imag < 0), xp.conj(r), r)


def test_the_shared_body_is_bit_identical_to_the_round_one_body():
    """Measured over 4,010 values spanning fifteen decades plus every corner
    (signed zeros, a denormal, ``lam^2`` exactly on the cut, ``nan``): zero
    differing bits, and the same at five array SIZES because the band is
    relative to the array."""
    rng = np.random.default_rng(20260911)
    mags = 10.0 ** rng.uniform(-15.0, 6.0, 4000)
    ang = rng.uniform(-np.pi, np.pi, 4000)
    vals = list(mags * np.exp(1j * ang))
    vals += [0 + 0j, 0 - 0j, 5e-324 + 0j, -1e-300 + 0j, -2.25 + 0j,
             -2.25 - 0j, -2.25 - 2.911e-15j, 2.25 - 2.911e-15j,
             complex("nan"), 1e300 + 1e300j]
    z = np.array(vals, dtype=complex)
    for n in (1, 2, 7, 64, 243, z.size):
        a = np.nan_to_num(np.asarray(_rc._sqrt_decay(z[:n])), nan=-7.0)
        b = np.nan_to_num(np.asarray(_round1_body(z[:n])), nan=-7.0)
        assert np.array_equal(a, b), (
            "the shared body differs from the round-1 body at n = %d "
            "(max |d| = %.3e)" % (n, float(np.max(np.abs(a - b)))))


def test_the_band_parameter_is_live_and_defaults_to_the_module_constant():
    """``band`` is a parameter so a caller whose modal population differs can
    carry its own derived value.  A parameter that nothing reads is a comment,
    so this asserts it is read: a band of 0 must let an on-cut incoming root
    through, and the default must equal ``_CUT_BAND_REL``."""
    z = np.array([-2.25 - 2.911e-15j], dtype=complex)
    assert _rc._sqrt_decay(z)[0].imag > 0            # default band: pinned
    assert _rc._sqrt_decay(z, band=0.0)[0].imag < 0  # band 0: not pinned
    assert np.array_equal(np.asarray(_rc._sqrt_decay(z)),
                          np.asarray(_rc._sqrt_decay(
                              z, band=_rc._CUT_BAND_REL)))


# ==================================================================== GATE 3
# THE HEADLINE.  Bar 1e-9 on the lossless closure of the uniform-spacer stack.
#
#   PRE arm (the exact-zero pin reinstated), |sum R + T - 2|, over 7
#   (build, thread) samples at n_orders = 4:
#     WIN  1.665e-04 (1 thr)  5.599e-06 (2)  6.367e-05 (4)  3.597e-05 (8)
#          4.582e-04 (unpinned)
#     WSL  2.161e-05 (1 thr, n_orders 3)  ... and 1.8e-15 at n_orders 4,
#          which is the point: WHICH truncation breaks is a per-build fact
#   POST arm, same samples: <= 6.7e-15, every one.
#
# The bar sits 5.2 decades above the post envelope and 3.7 decades below the
# smallest PRE reading that fires.  It is a conservation law with an arithmetic
# error floor, not a tolerance between two code paths.
_CLOSURE_BAR = 1e-9


@pytest.mark.parametrize("n_orders", [3, 4, 5])
def test_a_uniform_spacer_of_the_layer_background_closes_energy(n_orders):
    """A uniform layer whose permittivity equals the neighbouring structured
    layer's BACKGROUND is a coincidence partner exactly as a half-space region
    is -- both are built by the analytic Rayleigh helper in exact arithmetic.
    The substrate index here coincides with nothing, so this is the LAYER-LAYER
    coincidence alone."""
    d = _closure(_pmm_stack(n_orders))
    assert abs(d) < _CLOSURE_BAR, (
        "lossless closure defect %+.3e on the uniform-spacer stack at "
        "n_orders = %d: a propagating layer mode is carrying the incoming "
        "root again" % (d, n_orders))


def test_the_spacer_coincidence_is_what_breaks_the_pre_round_one_branch():
    """The fail-before, ENGINEERED and two-sided.

    With the pre-round-1 body installed the stack misses closure by orders;
    walking ONLY the spacer off the layer background -- leaving the substrate,
    the cell and the truncation exactly where they were -- makes the
    pre-round-1 arm clean.  So the partner is the LAYER, which is the scope
    extension this round records.

    WHICH TRUNCATION MANIFESTS IS A PER-BUILD FACT, exactly as it is on the
    THIN ladder of ``test_m1_conditioning_guard.py``: whether a given
    interface's smallest singular value falls above or below the arithmetic
    floor is decided by the BLAS reduction order.  Measured on the pre-round-1
    arm at one thread, ``|sum R + T - 2|``:

        n_orders      3           4           5
        WIN       4.893e-06   1.665e-04   4.419e-05
        WSL       2.161e-05   2.220e-15   7.690e-06

    ``n_orders = 4`` is broken on Windows and clean on WSL.  So this test scans
    the three truncations and asserts on the WORST -- the claim that survives
    every build is that the coincidence breaks the pre-round-1 branch SOMEWHERE
    on the ladder while the detuned control is clean EVERYWHERE on it.

    The DETUNE LADDER is in the report: a relative 1e-6 -- the detune the
    library's own remedy text used to recommend -- does NOT cure it here
    (8.1e-05 on Windows, 2.3e-04 on WSL); 1e-3 does (8.7e-14 / 1.3e-12).
    """
    ladder = (3, 4, 5)
    with _pre_arm():
        bad = {M: abs(_closure(_pmm_stack(M))) for M in ladder}
        good = {M: abs(_closure(_pmm_stack(M, spacer=_HOST * 1.01)))
                for M in ladder}
    worst = max(bad.values())
    control = max(good.values())
    assert worst > 1e3 * max(control, 1e-15), (
        "the pre-round-1 arm does not reproduce the defect at any truncation "
        "on this build: coincident spacer %s against detuned %s"
        % ({M: "%.3e" % v for M, v in bad.items()},
           {M: "%.3e" % v for M, v in good.items()}))
    assert control < _CLOSURE_BAR, (
        "the DETUNED control is not clean on the pre-round-1 arm (%.3e), so "
        "it cannot show that the coincidence is what breaks it" % control)


def _pixel_stack(n_orders=3, spacer=_HOST, S=32):
    """The same device as ``_pmm_stack`` sampled on a 32-pixel grid instead of
    a 6-pixel one -- three strips per axis rather than three coarse cells.  The
    device is identical; only the sampling differs, and the pre-round-2 error
    on it is three decades larger.  That is worth a gate of its own: the SIZE
    of the error is a property of the mount, not of the defect."""
    e = np.full((S, S), _HOST + 0j)
    x = (np.arange(S) + 0.5) / S - 0.5
    m = (np.abs(x[:, None]) < 0.25) & (np.abs(x[None, :]) < 0.25)
    e[m] = _WEAK
    st = PMM2DStackHybrid(_P, _P, n_substrate=_N_SUB, n_superstrate=1.0,
                          degree=7, n_orders=n_orders, symmetry=False)
    if spacer is not None:
        st.add_layer(0.1e-6, eps=spacer)
    st.add_layer(_D, eps_cell=e)
    if spacer is not None:
        st.add_layer(0.1e-6, eps=spacer)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T, J = st.set_source(_WL, theta=0.0).solve()
    return float(np.sum(np.asarray(R)) + np.sum(np.asarray(T)))


#: Passivity bar for the pixel-sampled spacer stack, derived from a ladder run
#: at `n_orders` 2..6 with and without the spacers, on both builds at 1 / 4 / 8
#: threads (`b10_manufactured_energy.py`).  The POST envelope over ALL of them
#: is 8.8539e-10 -- IDENTICALLY, on every sample -- because it is the
#: `n_orders = 4` no-spacer mount's own Fourier truncation error, which is
#: deterministic and arm-independent (the PRE no-spacer control reads the same
#: 8.8539e-10 everywhere).  The smallest PRE SPACER reading over the same
#: samples is 6.6385e-04 (WSL, 1 thread, `n_orders = 5`), so the bar has
#: 1.05 decades above the floor it must clear and 4.8 decades below the
#: smallest defect it must catch.
_PASSIVITY_BAR = 1e-8
_SPACER_LADDER = (2, 3, 4, 5, 6)


def test_the_coincident_spacer_stack_does_not_manufacture_energy():
    """The loudest reading of the defect, and the loudest fail-before.

    A PASSIVE lossless stack cannot return more power than comes in.  On the
    pre-round-2 branch this one does, by a lot:

        build / threads   worst PRE `sum R + T`     worst per-order motion
        WIN 1             5.812454299  (2.9x)       2.788e+00
        WIN 4             1.098858e+02 (55x)        5.755e+01
        WIN 8             3.567077e+01 (18x)        4.674e+00
        WSL 1             1.999336151                2.076e-02

    -- five decades of spread with the BLAS thread count, which is the
    branch-cut signature, and a per-build partition of WHICH truncation breaks
    (Windows breaks `n_orders` 3 and 4; WSL breaks 5 and 6).  So this test
    asserts on the LADDER, not on one mount: what survives every build is that
    the SPACER mounts break somewhere and the NO-SPACER controls never do.

    The no-spacer control reads 8.8539e-10 on every arm, build and thread count
    measured, which is the mount's own truncation error and is what
    ``_PASSIVITY_BAR`` is derived against.
    """
    post = {M: (_pixel_stack(M), _pixel_stack(M, spacer=None))
            for M in _SPACER_LADDER}
    worst = max(max(abs(a - 2.0), abs(b - 2.0)) for a, b in post.values())
    assert worst < _PASSIVITY_BAR, (
        "a mount of the coincident-spacer ladder misses passivity by %.4e: "
        "%s" % (worst, {M: ("%.9f" % a, "%.9f" % b)
                        for M, (a, b) in post.items()}))
    # ---- fail-before, ENGINEERED: the spacer mounts break, the controls do not.
    with _pre_arm():
        pre = {M: (_pixel_stack(M), _pixel_stack(M, spacer=None))
               for M in _SPACER_LADDER}
    bad = max(abs(a - 2.0) for a, _b in pre.values())
    ctrl = max(abs(b - 2.0) for _a, b in pre.values())
    assert bad > 1e3 * max(ctrl, 1e-15), (
        "the pre-round-1 branch body breaks no mount of this ladder on this "
        "build (worst spacer %.4e against worst control %.4e); the "
        "fail-before demonstrates nothing" % (bad, ctrl))
    assert ctrl < _PASSIVITY_BAR, (
        "the pre-round-1 arm's NO-SPACER controls are not clean (%.4e), so "
        "the spacer is not isolated as the cause" % ctrl)


# ==================================================================== GATE 4
# Conservation says the PRE answer was WRONG; it does not say the POST answer
# is RIGHT.  A DIFFERENT METHOD does: the Fourier RCWA stack on the same
# geometry, whose modal branch round 1 pinned, at a truncation where its own
# answer has stopped moving.
#
#   RCWA self-convergence, n_orders 6 vs 8: 4.4409e-16
#   PMM POST vs the RCWA reference: 5.1e-15 / 6.0e-15 / 1.1e-15 at
#     n_orders 3 / 4 / 5
#   PMM PRE  vs the same reference: 2.9e-05 / 2.0e-03 / 2.5e-04
#
# The bar is 1e-9: 5.2 decades above the POST readings and 4.5 decades below
# the smallest PRE one.
_REFERENCE_BAR = 1e-9


def _order_map(res):
    o = np.asarray(res[0])
    R = np.asarray(res[1], dtype=float)
    T = np.asarray(res[2], dtype=float)
    if R.ndim == 2:
        R, T = R.sum(axis=0), T.sum(axis=0)
    return {(int(a), int(b)): (float(R[i]), float(T[i]))
            for i, (a, b) in enumerate(o)}


def _rcwa_reference(n_orders=6, rep=12):
    """The same device through the Fourier RCWA stack.  The 6 x 6 cell is
    block-replicated ``rep`` x so the Fourier path has enough samples; the cell
    is piecewise constant on those walls, so the replication is EXACT and
    changes the sampling, not the device."""
    st = RCWAStack(_P, period_y=_P, n_substrate=_N_SUB, n_superstrate=1.0,
                   n_orders=n_orders, n_orders_y=n_orders)
    st.add_layer(0.1e-6, eps=_HOST)
    st.add_layer(_D, eps_cell=np.kron(_cell(),
                                      np.ones((rep, rep), dtype=complex)))
    st.add_layer(0.1e-6, eps=_HOST)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return st.set_source(_WL, theta=0.0).solve().efficiencies()


def test_the_repaired_pmm_answer_matches_an_independent_method():
    """Per-order agreement with a solve that shares no eigenproblem with the
    PMM -- and a self-convergence check on the reference first, so the
    comparison cannot pass because the reference is loose."""
    ref = _order_map(_rcwa_reference(6))
    ref8 = _order_map(_rcwa_reference(8))
    keys = sorted(set(ref) & set(ref8))
    conv = max(max(abs(ref[k][j] - ref8[k][j]) for j in (0, 1))
               for k in keys)
    assert conv < 1e-12, (
        "the RCWA reference has not converged (6 vs 8 orders differ by "
        "%.3e); it cannot calibrate the PMM answer" % conv)
    for n_orders in (3, 4, 5):
        got = _order_map(_pmm_stack(n_orders))
        shared = sorted(set(got) & set(ref8))
        assert shared, "the two methods share no diffraction order"
        d = max(max(abs(got[k][j] - ref8[k][j]) for j in (0, 1))
                for k in shared)
        assert d < _REFERENCE_BAR, (
            "the hybrid PMM disagrees with an independent RCWA solve of the "
            "same geometry by %.3e per order at n_orders = %d" % (d, n_orders))


# ==================================================================== GATE 5
# The invariant, read off the PMM's OWN operator rather than a constructed
# array: no layer mode of a lossless cell may come back on the incoming root.
#
# Pre-round-2 this count read 5..9 of 12..39 on-cut modes per solve on Windows
# and 1..8 on WSL, summed over the solve's eigenvalue arrays.

def test_no_pmm_layer_mode_of_a_lossless_cell_carries_the_incoming_root(
        monkeypatch):
    """Reads the eigenvalue arrays the PMM layer solve actually produces
    (through the module binding it resolves at call time), and asserts the
    invariant where it matters instead of on an engineered input."""
    seen = []
    orig = _rc._sqrt_decay

    def spy(x, xp=None, band=_rc._CUT_BAND_REL):
        seen.append(np.asarray(x, dtype=complex).copy())
        return orig(x, xp, band)

    monkeypatch.setattr(_ptw, "_sqrt_decay", spy)
    _pmm_stack(4)
    _pmm_stack(4, pillar=6.0)
    assert seen, "the PMM layer eigenproblem did not run"
    bad = total = 0
    for lam2 in seen:
        lam = np.asarray(orig(lam2))
        scale = max(float(np.max(np.abs(lam))), 1.0)
        on_cut = np.abs(lam.real) <= _rc._CUT_BAND_REL * scale
        total += int(on_cut.sum())
        bad += int(np.sum(on_cut & (lam.imag < 0)))
    assert total > 0, "no mode of this cell was on the cut -- wrong fixture"
    assert bad == 0, "%d of %d on-cut PMM modes carry the incoming root" % (
        bad, total)


# ==================================================================== GATE 6
# The other side: where the sign of ``Im(r)`` is PHYSICS, nothing may move.
# Measured over the round-2 census, every LOSSY surface on both builds is
# bit-identical between the arms (|post - pre| = 0.00e+00, 12 of 12).

@pytest.mark.parametrize("eps_im", [1e-1, 1e-3])
def test_a_lossy_pmm_stack_is_bit_identical_between_the_two_branch_bodies(
        eps_im):
    """For a lossy layer the acted-on population is EMPTY -- every mode's
    ``Im(lam^2)`` takes the sign the loss dictates -- so the flip has nothing
    to act on and the answer must not move by a bit.  A change that "fixed"
    the sign by flipping everything would be caught here."""
    def run():
        st = PMM2DStackHybrid(_P, _P, n_substrate=_N_SUB, n_superstrate=1.0,
                              degree=7, n_orders=3, symmetry=False)
        st.add_layer(0.1e-6, eps=_HOST + 1j * eps_im)
        st.add_layer(_D, eps_cell=_cell(6.0) + 1j * eps_im)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, R, T, _J = st.set_source(_WL, theta=0.0).solve()
        return np.asarray(R), np.asarray(T)

    R0, T0 = run()
    with _pre_arm():
        R1, T1 = run()
    assert float(np.max(np.abs(R0 - R1))) == 0.0
    assert float(np.max(np.abs(T0 - T1))) == 0.0


def test_the_pure_staggered_engine_never_reaches_this_function():
    """SCOPE, pinned.  The no-floor staggered 2-D PMM -- the library's
    accuracy-leading 2-D engine -- selects its forward branch with
    ``pmm/_core._forward_branch_flip`` (already a relative band) and, on the
    out-of-plane path, with ``rcwa/_core._select_forward_flux`` (a relative
    z-flux bar).  It calls ``_sqrt_decay`` NOWHERE, so neither round of this
    campaign can have moved it.  Measured: 0 calls on every staggered surface
    of the round-2 census, and its closure reads 1.0e-14 on the same
    coincidence that broke the hybrid.

    Pinned because a future refactor that routed the staggered layer solve
    through this function would silently change the engine the user's device
    work depends on.
    """
    from lumenairy.elements.pmm import pmm_efficiency_2d_staggered
    calls = []
    orig = _rc._sqrt_decay

    def spy(x, xp=None, band=_rc._CUT_BAND_REL):
        calls.append(np.asarray(x).size)
        return orig(x, xp, band)

    import lumenairy.elements.pmm.twod_staggered as _ts
    saved = [(m, getattr(m, "_sqrt_decay", None)) for m in (_rc, _ptw, _ts)]
    for m, fn in saved:
        if fn is not None:
            m._sqrt_decay = spy
    try:
        cell = np.full((4, 4), _HOST + 0j)
        cell[1:3, 1:3] = _WEAK
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _o, R, T = pmm_efficiency_2d_staggered(
                _P, _P, cell, 1.5, 1.0, _D, _WL, degree=6, n_orders=4)
    finally:
        for m, fn in saved:
            if fn is not None:
                m._sqrt_decay = fn
    assert calls == [], (
        "the staggered engine now routes %d array(s) through _sqrt_decay; if "
        "that is intended, re-derive the band on the staggered pencil's own "
        "gamma^2 population first" % len(calls))
    d = float(np.sum(R) + np.sum(T) - 1.0)
    assert abs(d) < 1e-9, "staggered closure %+.3e on the coincidence" % d


# ==================================================================== GATE 7
# The JAX twins trace the SAME body.  Round 2's whole reason for passing ``xp``
# explicitly is that the traced path and the eager path must not be able to
# drift apart again.
#
#   PRE : the traced stack disagreed with its NumPy sibling by 2.489e-05 on the
#         coincidence (and by <= 1.6e-15 off it) -- the two arms had DIFFERENT
#         wrong answers, because their eigensolvers round differently.
#   POST: parity <= 3.331e-16 on the same surface, <= 3.3e-15 over all seven.

def test_the_jax_twin_agrees_with_its_numpy_sibling_on_the_coincidence():
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    layout = np.zeros((_S, _S), dtype=np.int64)
    layout[2:4, 2:4] = 1

    def solve(traced):
        st = PMM2DStackHybrid(_P, _P, n_substrate=_N_SUB, n_superstrate=1.0,
                              degree=7, n_orders=3, symmetry=False)
        st.add_layer(0.1e-6, eps=_HOST)
        if traced:
            c = jnp.asarray(np.zeros((_S, _S), complex))
            c = c.at[layout == 0].set(jnp.asarray(_HOST + 0j))
            c = c.at[layout == 1].set(jnp.asarray(_WEAK + 0j))
            st.add_layer(_D, eps_cell=c, region_layout=layout)
        else:
            st.add_layer(_D, eps_cell=_cell())
        st.add_layer(0.1e-6, eps=_HOST)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, R, T, J = st.set_source(_WL, theta=0.0).solve()
        return np.asarray(R, dtype=float), np.asarray(T, dtype=float)

    Rn, Tn = solve(False)
    Rj, Tj = solve(True)
    d = max(float(np.max(np.abs(Rn - Rj))), float(np.max(np.abs(Tn - Tj))))
    # 4.5 decades above the measured envelope (3.3e-15 over seven surfaces,
    # both builds) and 4.9 decades below the pre-round-2 reading (2.489e-05).
    assert d < 1e-10, "JAX / NumPy parity %.3e on the coincidence" % d


def test_the_jax_gradient_is_unchanged_off_the_coincidence():
    """The consolidation must not move a derivative where nothing was wrong.

    ``d sum(T) / d eps_pillar`` on an OFF-coincidence lossless cell reads
    ``-1.851320700230e-02`` on the pre-round-2 tree and
    ``-1.851320700230e-02`` after -- identical to all 13 recorded digits,
    hence the 1e-12 relative bar below.  Its agreement with a central finite
    difference improves from 1.9e-08 to 8.2e-10, which is the same statement
    from the accuracy side.
    """
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from lumenairy.elements.pmm import pmm_efficiency_2d_cell
    layout = np.zeros((_S, _S), dtype=np.int64)
    layout[2:4, 2:4] = 1

    def sumT(e1):
        c = jnp.asarray(np.zeros((_S, _S), complex))
        c = c.at[layout == 0].set(jnp.asarray(_HOST + 0j))
        c = c.at[layout == 1].set(e1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _o, _R, T = pmm_efficiency_2d_cell(
                _P, _P, c, _N_SUB, 1.0, _D, _WL, degree=7, n_orders=4,
                region_layout=layout)
        return jnp.sum(T)

    g = complex(jax.grad(sumT)(jnp.asarray(6.0 + 0j))).real
    assert abs(g / -1.851320700230e-02 - 1.0) < 1e-12, (
        "d sum(T) / d eps moved to %.12e from the recorded -1.851320700230e-02"
        % g)
    h = 1e-5
    fd = (float(sumT(jnp.asarray(6.0 + h + 0j)))
          - float(sumT(jnp.asarray(6.0 - h + 0j)))) / (2.0 * h)
    assert abs(g - fd) < 1e-6 * max(abs(fd), 1e-12), (
        "AD %.12e against central FD %.12e" % (g, fd))
