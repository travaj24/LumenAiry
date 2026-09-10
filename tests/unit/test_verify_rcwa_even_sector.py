"""Independent verification gates for the RCWA modal branch-cut fix.

These are the claims that the fix's own file
(``tests/unit/test_fix_rcwa_even_sector_wsl.py``) leaves ungated, found by
re-measuring rather than by reading -- see
``docs/audits/VERIFY_RCWA_EVEN_SECTOR_2026_09_11.md`` for the full evidence,
the probes in ``validation/probe_verify_rcwa_even_sector/`` for the numbers.

TERMS.  ``_core._sqrt_decay`` maps a layer's modal eigenvalue ``lam^2`` to the
decay constant ``lam`` that drives the layer propagator ``X = exp(-lam k0 L)``.
It returns the PRINCIPAL square root (``Re(lam) >= 0``, so ``|X| <= 1``) and,
for a mode numerically ON the branch cut, pins the OUTGOING root by replacing
``r`` with ``conj(r)`` when ``Im(r) < 0`` and
``|Re(r)| <= _CUT_BAND_REL * max(max|r|, 1)``.  "On the cut" means ``lam^2`` is
real negative -- a PROPAGATING mode of a LOSSLESS layer -- where the two roots
``+i|kz|`` (outgoing) and ``-i|kz|`` (incoming) differ only by the sign of a
rounding-level imaginary part.

WHAT IS NEW HERE, and why each one is a decision rather than a reading:

1.  The ``|X| <= 1`` CONTRACTION over the whole complex plane.  ``conj(r)``
    rather than ``-r`` was chosen precisely to keep ``Re(lam) >= 0``; the fix's
    own file checks that on 96 engineered points.  Gate 1 checks the SIGN --
    a discrete property, not a digit -- over 4,096 pseudo-random eigenvalues
    plus every degenerate corner (exact zero, signed-zero imaginary part,
    denormal, deep evanescent), and bounds the root residual by a DERIVED
    floating-point floor rather than by any build's reading.

2.  ``lam^2 -> 0`` (a layer mode at CUTOFF).  Untested upstream; it is the one
    place where the band's discriminating ratio grows without limit, because
    ``Re(r) = Im(lam^2) / (2 sqrt(|Re lam^2|))`` diverges as the mode reaches
    cutoff.  Measured on a mount driven onto a cutoff by trisection, the ratio
    climbs to 6.7e-09 against the 1e-08 bar -- 1.5x, not the fourteen decades
    the populations show away from cutoff.

3.  A UNIFORM SPACER LAYER is a coincidence partner, exactly as a REGION is.
    The library's warning text and the fix's audit both describe the failure as
    "a LAYER permittivity EXACTLY EQUAL to a REGION's".  But a uniform layer of
    an ``RCWAStack`` is built by ``_homogeneous_eigenmodes`` in exact
    arithmetic, just as a half-space is, so a uniform ``eps = 2.25`` spacer
    beside a structured layer whose BACKGROUND is 2.25 is the same degeneracy
    with NO region involved -- and it reproduces at ``n_substrate = 1.63``,
    where nothing coincides with either half-space.  Pre-fix, this file's
    three-layer fixture is REFUSED by the library's own energy tripwire at
    ``sum R + T = 2.600e+01`` on BOTH builds at 1, 4 and 8 threads, for all
    three ``symmetry`` settings; post-fix it reads 2 to <= 1.2e-15 at every
    setting measured.  (The same geometry at ``n_orders = 4`` is refused at
    some thread counts and RETURNED at 1.866 / 2.0028 at others -- the
    thread-order dependence of the pre-fix arm, which is the whole point.)

4.  The band must not touch a mode whose negative imaginary part is PHYSICS.
    The fix's file asserts this on a CONSTRUCTED root; this gate asserts it on
    the modes a real lossy solve produces, over a loss ladder six decades
    deeper than the one the fix measured (down to ``Im(eps) = 1e-12``): the
    acted-on population -- modes with ``Im(r) < 0`` AND inside the band -- must
    be EMPTY, so the flip cannot re-sign a physical decay rate.  The first
    ladder rung that puts a mode inside the band is ``Im(eps) = 1e-15``, below
    the eigensolver's own backward error on this operator, where the sign is
    not physics on any build.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from lumenairy.elements.rcwa import _core as _rc

_EPS = float(np.finfo(np.float64).eps)


# ==================================================================== GATE 1
# The contraction guarantee, over the plane.
#
# DERIVED BOUND, not a reading.  ``lam = sqrt(z)`` carries at most ~1 ulp, and
# forming ``lam**2`` is a complex multiply (two products and one sum per
# component), at most ~4 ulp; propagating the sqrt's error through the square
# doubles it.  So ``|lam**2 - z| <= C eps |z|`` with C of order ten.  The bar
# below takes C = 128 -- more than a decade of head-room over the largest
# residual measured (2.59e-16 relative, IDENTICAL on Windows py3.14 /
# OpenBLAS-Haswell and WSL py3.12 / OpenBLAS-SkylakeX, 2026-09-11), and it is a
# property of IEEE double arithmetic rather than of any LAPACK build.
_ROOT_RESIDUAL_C = 128.0


def _spectrum():
    """4,096 pseudo-random eigenvalues spanning fifteen decades, plus every
    degenerate corner.  Seeded, so the set is the same on every build."""
    rng = np.random.default_rng(20260911)
    mag = 10.0 ** rng.uniform(-8.0, 7.0, size=4096)
    ang = rng.uniform(-np.pi, np.pi, size=4096)
    z = mag * np.exp(1j * ang)
    corners = np.array([
        0.0 + 0.0j,                 # exact cutoff
        complex(0.0, -0.0),         # cutoff, negative signed zero
        complex(-2.25, 0.0),        # on the cut, outgoing
        complex(-2.25, -0.0),       # on the cut, incoming signed zero
        complex(-2.25, -2.911e-15),  # on the cut through eigensolver noise
        complex(-2.25, +2.911e-15),
        complex(2.25, -2.911e-15),  # evanescent through the same noise
        complex(-1e-300, 0.0), complex(1e-300, 0.0),
        complex(5e-324, 0.0),       # denormal
        complex(-4900.0, 0.0),      # deep evanescent / deep propagating
        complex(4900.0, 0.0),
        complex(-2.25, -0.5),       # genuinely lossy
        complex(-2.25, +0.5),
    ], dtype=complex)
    return np.concatenate([z, corners])


def test_sqrt_decay_never_returns_a_growing_root():
    """``|X| = |exp(-lam k0 L)| <= 1`` for every mode, i.e. ``Re(lam) >= 0``.

    This is the guarantee the principal branch exists to provide and the reason
    the on-cut flip takes ``conj(r)`` and not ``-r`` (``-r`` hands back
    ``Re(lam) = -1e-16`` and gives it up).  A DECISION on a sign, so no build
    can move it; the count of violations must be exactly zero.
    """
    z = _spectrum()
    lam = np.asarray(_rc._sqrt_decay(z))
    assert np.all(np.isfinite(lam)), "non-finite root returned"
    bad = int(np.sum(lam.real < 0.0))
    assert bad == 0, (
        "%d of %d roots have Re(lam) < 0, i.e. a GROWING propagator "
        "exp(+|Re lam| k0 L); worst %.3e" % (bad, lam.size,
                                             float(np.min(lam.real))))
    # ... and it is still a square root.  Where the on-cut flip fires the
    # returned root squares to conj(lam^2) rather than to lam^2 -- that IS the
    # flip, and on a mode genuinely on the cut the two differ only by the
    # rounding-level imaginary part the flip exists to discard.  So the
    # residual is taken against the nearer of the two.
    resid = np.minimum(np.abs(lam ** 2 - z), np.abs(lam ** 2 - np.conj(z)))
    allow = _ROOT_RESIDUAL_C * _EPS * np.abs(z) + 8.0 * 5e-324
    worst = int(np.argmax(resid - allow))
    assert np.all(resid <= allow), (
        "root residual %.3e exceeds the derived floor %.3e at lam^2 = %r"
        % (resid[worst], allow[worst], z[worst]))


def test_the_flip_only_ever_moves_a_root_along_the_imaginary_axis():
    """Whatever the band does, it may not change ``|lam|`` or ``Re(lam)``.

    ``conj`` is an isometry that fixes the real part, so this holds by
    construction -- which is exactly why it is worth pinning: a future
    "simplification" back to ``-r`` breaks both halves at once, and the
    ``Re(lam) >= 0`` gate above would catch only one of them.
    """
    z = _spectrum()
    lam = np.asarray(_rc._sqrt_decay(z))
    principal = np.sqrt(z)
    assert np.array_equal(np.abs(lam), np.abs(principal)), (
        "the selector changed |lam|; conj() is an isometry and must not")
    assert np.array_equal(lam.real, principal.real), (
        "the selector changed Re(lam); conj() fixes the real part and must")
    assert not np.array_equal(lam, principal), (
        "the selector never fired on this spectrum -- wrong fixture")


@pytest.mark.parametrize("z", [0.0 + 0.0j, complex(0.0, -0.0),
                               complex(5e-324, 0.0), complex(-5e-324, 0.0),
                               complex(1e-300, 0.0), complex(-1e-300, 0.0)])
def test_a_mode_at_cutoff_is_finite_and_non_growing(z):
    """``lam^2 -> 0`` is a layer mode AT CUTOFF.  ``_inv_lam`` floors ``|lam|``
    downstream, so the only thing required here is that the root itself is
    finite, non-growing, and does not acquire a spurious magnitude."""
    lam = np.asarray(_rc._sqrt_decay(np.array([z], dtype=complex)))[0]
    assert np.isfinite(lam.real) and np.isfinite(lam.imag)
    assert lam.real >= 0.0
    assert abs(lam) <= np.sqrt(abs(z)) * (1.0 + 8.0 * _EPS) + 5e-324


# ==================================================================== GATE 2
# The coincidence partner the audit does not name: a UNIFORM SPACER LAYER.
#
# Bar: a provably lossless stack conserves energy EXACTLY under the Laurent
# rule at any truncation, so ``|sum R + T - 2|`` is an independent oracle whose
# error floor is the arithmetic -- it needs no reference solve and no prior
# reading.  Measured post-fix on THIS fixture: |defect| <= 1.2e-15 over
# (Windows py3.14 / OpenBLAS-Haswell, WSL py3.12 / OpenBLAS-SkylakeX) x
# (1, 4, 8, unpinned) threads, 2026-09-11.  Pre-fix the same fixture is REFUSED
# at ``sum R + T = 2.600e+01`` -- a 24-fold violation -- on both builds at 1, 4
# and 8 threads.  So the bar carries ~6 decades below it and ~9 above it.
_STACK_CLOSURE_BAR = 1e-9

_SPACER_EPS = 2.25          # equals the structured layer's BACKGROUND
_OFF_SPACER_EPS = 2.56      # the control: no layer-layer coincidence


def _tensor_cell(n=24, twist=0.7, no=1.5, ne=1.7, bg=2.25):
    tc = np.zeros((n, n, 3, 3), complex)
    for i in range(3):
        tc[:, :, i, i] = bg
    no2, ne2 = no ** 2, ne ** 2
    c0, s0 = np.cos(twist), np.sin(twist)
    x = (np.arange(n) + 0.5) / n - 0.5
    m = (np.abs(x[:, None]) < 0.25) & (np.abs(x[None, :]) < 0.25)
    tc[m, 0, 0] = ne2 * c0 * c0 + no2 * s0 * s0
    tc[m, 1, 1] = ne2 * s0 * s0 + no2 * c0 * c0
    tc[m, 0, 1] = tc[m, 1, 0] = (ne2 - no2) * c0 * s0
    tc[m, 2, 2] = no2
    return tc


def _spacer_stack(spacer_eps, symmetry, n_substrate=1.63, n_orders=3):
    from lumenairy.elements.rcwa import RCWAStack
    st = RCWAStack(period=0.5e-6, period_y=0.5e-6, n_superstrate=1.0,
                   n_substrate=n_substrate, n_orders=n_orders,
                   n_orders_y=n_orders)
    st.add_layer(0.05e-6, eps=spacer_eps)
    st.add_layer(0.12e-6, eps_tensor_cell=_tensor_cell())
    st.add_layer(0.06e-6, eps=spacer_eps)
    return st.set_source(0.6e-6).solve(symmetry=symmetry)


@pytest.mark.parametrize("symmetry", ["auto", True, False])
def test_a_uniform_spacer_layer_is_a_coincidence_partner_too(symmetry):
    """A uniform ``eps = 2.25`` spacer beside a structured layer whose
    BACKGROUND is 2.25 must not destroy the solve -- and the substrate index is
    free, so this is NOT the layer-vs-region case the warning text describes.
    """
    res = _spacer_stack(_SPACER_EPS, symmetry)
    e = res.efficiencies()
    tot = float(np.sum(np.asarray(e[1])) + np.sum(np.asarray(e[2])))
    assert abs(tot - 2.0) < _STACK_CLOSURE_BAR, (
        "lossless closure defect %+.3e on the uniform-spacer stack "
        "(symmetry=%s)" % (tot - 2.0, symmetry))


def test_the_spacer_control_agrees_with_the_coincident_stack_on_the_physics():
    """The other side of the claim: moving the SPACER off the structured
    layer's background must not change what the solve is worth.  Both stacks
    conserve energy, and they differ -- so the coincident one is not passing by
    accidentally being the same problem."""
    a = _spacer_stack(_SPACER_EPS, "auto")
    b = _spacer_stack(_OFF_SPACER_EPS, "auto")
    ea, eb = a.efficiencies(), b.efficiencies()
    for e in (ea, eb):
        tot = float(np.sum(np.asarray(e[1])) + np.sum(np.asarray(e[2])))
        assert abs(tot - 2.0) < _STACK_CLOSURE_BAR
    assert not np.allclose(np.asarray(ea[1]), np.asarray(eb[1])), (
        "the control stack returned the same answer as the coincident one -- "
        "the two spacers are not distinguishable, so this fixture proves "
        "nothing")


# ==================================================================== GATE 3
# The band must not reach a mode whose negative imaginary part is PHYSICS.
#
# Asserted through a real solve rather than on a constructed root, over a loss
# ladder six decades deeper than the fix's own (which stopped at 1e-8).
# Measured: the acted-on population is EMPTY at every rung from 1e-2 to 1e-12
# on both builds; the first rung that puts a mode inside the band is 1e-15,
# below the eigensolver's backward error on this operator (max |Im(lam^2)| of
# an on-cut mode ~3e-15 against ||M|| ~ 72), where no bar could separate the
# sign from the noise.
_LOSS_LADDER = (1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12)


def _lossy_eig_spectrum(eps_imag):
    """Every layer eigenvalue the lossy solve produces, via the eig factory the
    layer solve resolves at call time."""
    from lumenairy.elements.rcwa import rcwa_jones_2d
    seen = []
    orig_core = _rc._eig_for

    def factory(xp):
        base = orig_core(xp)

        def wrapped(A):
            w, v = base(A)
            seen.append(np.asarray(w).astype(complex).copy())
            return w, v
        return wrapped

    cell = _tensor_cell(n=24)
    for i in range(3):
        cell[:, :, i, i] = cell[:, :, i, i] + 1j * eps_imag
    _rc._eig_for = factory
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rcwa_jones_2d(0.5e-6, 0.5e-6, cell, 1.5, 1.0, 0.2e-6, 0.6e-6,
                          n_orders_x=3, n_orders_y=3)
    finally:
        _rc._eig_for = orig_core
    assert seen, "the layer eigenproblem did not run"
    return np.concatenate(seen)


@pytest.mark.parametrize("eps_imag", _LOSS_LADDER)
def test_the_band_never_acts_on_a_lossy_layers_modes(eps_imag):
    """No mode of an ABSORBING layer may fall inside the on-cut band while
    carrying ``Im(r) < 0`` -- if one did, the flip would conjugate a physical
    root.  A DECISION on emptiness of the acted-on set, not on any digit."""
    band = getattr(_rc, "_CUT_BAND_REL", 1e-8)
    lam2 = _lossy_eig_spectrum(eps_imag)
    r = np.sqrt(lam2)
    scale = max(float(np.max(np.abs(r))), 1.0)
    acted = (np.abs(r.real) <= band * scale) & (r.imag < 0)
    n = int(acted.sum())
    assert n == 0, (
        "%d modes of an Im(eps) = %.0e layer sit inside the band with "
        "Im(r) < 0, so the flip re-signs them; smallest ratio %.3e"
        % (n, eps_imag,
           float(np.min(np.abs(r.real)[acted]) / scale)))


def test_the_lossy_solve_is_untouched_by_the_selector():
    """The same statement from the other side, end to end: with a layer that
    absorbs, the shipped answer must be BIT-IDENTICAL to the one a selector
    that never fires would give."""
    from lumenairy.elements.rcwa import rcwa_jones_2d
    cell = _tensor_cell(n=24)
    for i in range(3):
        cell[:, :, i, i] = cell[:, :, i, i] + 1e-4j

    def solve():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return rcwa_jones_2d(0.5e-6, 0.5e-6, cell, 1.5, 1.0, 0.2e-6,
                                 0.6e-6, n_orders_x=3, n_orders_y=3)
    shipped = solve()
    orig = _rc._sqrt_decay
    # The control is the PRE-FIX selector (the exact ``Re(r) == 0`` test), not
    # a bare ``sqrt``: the homogeneous REGION modes are built in exact
    # arithmetic and DO rely on a pin firing there, so a bare sqrt makes the
    # region mode-match singular and proves nothing about the layer.
    _rc._sqrt_decay = _pre_fix_sqrt_decay
    try:
        pre = solve()
    finally:
        _rc._sqrt_decay = orig
    for a, b in zip(shipped, pre):
        assert np.array_equal(np.asarray(a), np.asarray(b)), (
            "the selector moved a LOSSY solve; it must only ever act where the "
            "sign of Im(lam^2) is rounding")


def _pre_fix_sqrt_decay(x):
    """The ``48c8747`` body: the exact-zero pin the fix replaced."""
    x = np.asarray(x, dtype=complex)
    r = np.sqrt(x)
    on_cut = r.real == 0
    return np.where(on_cut & (r.imag < 0), -r, r)


# ==================================================================== GATE 4
# LOUDNESS at a layer cutoff.
#
# Away from cutoff the band separates its two populations by many decades.  AT
# a layer cutoff (``lam^2 -> 0``) it does not: the discriminating ratio for the
# mode at cutoff climbs to 6.7e-09 against the 1e-08 bar, and the lossless
# closure of such a mount degrades in BOTH arms (measured post-fix +4.25e-02 at
# ``|lam^2| = 4.5e-15``, pre-fix REFUSED at the same mount).  That is a
# pre-existing conditioning limit of a measure-zero mount, not this change's --
# but what MUST hold either way is that the library is never SILENTLY wrong
# there.  This gate drives a mount onto a cutoff by trisection and asserts the
# implication, over every mount the search visits, on whatever build runs it.
_LOUDNESS_CLOSURE_BAR = 1e-6     # measured: silent rows <= 1.3e-07, loud rows
#                                  >= 2.5e-05, on both builds (2026-09-11)


def _te_mount(period):
    from lumenairy.elements.rcwa import rcwa_efficiency_1d
    return rcwa_efficiency_1d(period, 2.1, 1.5, 1.5, 1.0, 0.2e-6, 0.5, 0.6e-6,
                              polarization="te", n_orders=11)


def _min_abs_lam2_and_verdict(period):
    """``(min |lam^2|, verdict, closure)`` for one mount, in one solve."""
    from lumenairy.elements.rcwa import oned as _oned
    seen = []
    orig = _rc._eig_for

    def factory(xp):
        base = orig(xp)

        def wrapped(A):
            w, v = base(A)
            seen.append(np.asarray(w).astype(complex).copy())
            return w, v
        return wrapped
    # ``_eig_for`` is imported BY NAME into ``rcwa.oned``, so patching only
    # ``_core`` leaves the whole 1-D fast path unseen (measured: 0 arrays).
    _rc._eig_for = factory
    _oned._eig_for = factory
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                res = _te_mount(period)
                clo = float(np.sum(np.asarray(res[1]))
                            + np.sum(np.asarray(res[2])) - 1.0)
                verdict = "WARNED" if caught else "SILENT"
            except Exception:
                clo, verdict = None, "RAISED"
    finally:
        _rc._eig_for = orig
        _oned._eig_for = orig
    m = min((float(np.min(np.abs(w))) for w in seen), default=float("inf"))
    return m, verdict, clo


def test_a_mount_driven_onto_a_layer_cutoff_is_never_silently_wrong():
    """Engineer the state rather than hope a fixture produces it: trisect the
    period toward a LAYER mode at cutoff, and assert at every mount visited
    that a lossless-closure violation above the bar came with a warning or a
    refusal.  Build-free: the assertion is an IMPLICATION over whatever mounts
    this build's arithmetic leads the search to."""
    grid = np.linspace(0.20e-6, 1.60e-6, 121)
    scan = [_min_abs_lam2_and_verdict(float(p)) for p in grid]
    k = int(np.argmin([r[0] for r in scan]))
    a0 = float(grid[max(k - 1, 0)])
    b0 = float(grid[min(k + 1, len(grid) - 1)])
    reached = min(r[0] for r in scan)
    silent_bad = [(float(grid[i]), r[0], r[2]) for i, r in enumerate(scan)
                  if r[1] == "SILENT" and r[2] is not None
                  and abs(r[2]) > _LOUDNESS_CLOSURE_BAR]
    for _ in range(120):
        m1 = a0 + (b0 - a0) / 3.0
        m2 = b0 - (b0 - a0) / 3.0
        r1 = _min_abs_lam2_and_verdict(m1)
        r2 = _min_abs_lam2_and_verdict(m2)
        for p, r in ((m1, r1), (m2, r2)):
            reached = min(reached, r[0])
            if r[1] == "SILENT" and r[2] is not None and                     abs(r[2]) > _LOUDNESS_CLOSURE_BAR:
                silent_bad.append((p, r[0], r[2]))
        if r1[0] < r2[0]:
            b0 = m2
        else:
            a0 = m1
        if b0 - a0 < 1e-22:
            break
    assert reached < 1e-8, (
        "the search did not reach a layer cutoff (min |lam^2| = %.3e); the "
        "gate would be vacuous" % reached)
    assert not silent_bad, (
        "%d mount(s) violated lossless closure above %.0e with NO warning; "
        "worst |sum R+T - 1| = %.3e at period %.6e (min |lam^2| = %.3e)"
        % (len(silent_bad), _LOUDNESS_CLOSURE_BAR,
           max(abs(c) for _, _, c in silent_bad),
           max(silent_bad, key=lambda t: abs(t[2]))[0],
           max(silent_bad, key=lambda t: abs(t[2]))[1]))
