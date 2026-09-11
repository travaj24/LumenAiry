"""Gaps found by the INDEPENDENT VERIFICATION of the 5.45.1 BOR multilayer
guards (``docs/audits/VERIFY_BOR_MULTILAYER_GUARDS_2026_09_12.md``).

WHAT THIS FILE IS AND IS NOT.  It is not a second copy of
``test_fix_bor_multilayer_guards.py``: every claim that file makes was
re-measured during the verification and is not re-asserted here.  What is here
is the three places where the re-measurement found the shipped guards deciding
something the physics does not support, plus one bound on the SEM warn edge
that the shipped census does not carry.

WHY THREE OF THEM ARE ``xfail(strict=True)``.  They assert the behaviour the
physics requires, which the tree does NOT yet have.  Strict xfail is the right
marker for that: the suite stays green while the gap is open, and the moment
the gap is closed the marker itself fails, so nobody has to remember to come
back.  Each one is PREMISE-GATED: if the arm running it does not exhibit the
underlying numerical pathology at all (the 5.45.0 CI lesson -- the release
runner's LAPACK returns correct answers from solves that are ill-conditioned
on every local mount), the test SKIPS rather than reporting a defect that is
not there.

TERMS USED BELOW.

*Passive medium* -- one with ``Im eps >= 0`` in this library's
``exp(-i omega t)`` convention, i.e. absorbing or lossless but not amplifying.
For a stack of passive media energy conservation reads ``R + T + A = 1`` with
the absorbed fraction ``A >= 0``, so ``R + T <= 1`` is a THEOREM there --
whether or not the stack is lossless.  On a LOSSLESS passive stack closed by a
PEC wall there is no other exit and no absorption, so ``R + T = 1`` is an
EQUALITY.

*Channel* -- a propagating diffraction channel of the cascade; one entry of the
``R`` / ``T`` arrays.  For a UNIFORM PEC-walled half-space the number of them
is a closed form (below), which is what makes it usable as an oracle.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import warnings

import numpy as np
import pytest
from scipy.special import jn_zeros, jnp_zeros

import lumenairy.elements.bor.bor_solve as _bs
from lumenairy.elements.bor.bor_solve import BORNodalPassivityError, build_layer, solve
from lumenairy.elements.bor.bor_stack import BORStack

pytestmark = pytest.mark.slow          # eig-heavy BOR solves


# --------------------------------------------------------------------------- #
#  fixtures
# --------------------------------------------------------------------------- #
def _uni(val):
    return lambda r: np.full_like(r, val, dtype=complex)


def _ring(period, e_lo, e_hi, duty=0.5):
    def f(r):
        e = np.full_like(r, e_lo, dtype=complex)
        e[(r % period) < duty * period] = e_hi
        return e
    return f


def _nodal_stack(e_hi, Rbig=4.0, N=200, k0=2.0, m=1, e_out=2.0 + 0j):
    """The stack the shipped gate ``test_structured_stack_energy_floor_nodal``
    built, with the ring's HIGH permittivity left free so an infinitesimal
    imaginary part can be put on it."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return [build_layer(m, Rbig, N, _uni(e_out), k0, basis="nodal"),
                build_layer(m, Rbig, N, _ring(0.8, e_out, e_hi), k0,
                            thickness=0.5, basis="nodal"),
                build_layer(m, Rbig, N, _uni(e_out), k0, basis="nodal")]


def _solve_disarmed(layers, k0):
    prev = _bs.BOR_NODAL_PASSIVITY_GUARD
    _bs.BOR_NODAL_PASSIVITY_GUARD = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return solve(layers, k0)
    finally:
        _bs.BOR_NODAL_PASSIVITY_GUARD = prev


# --------------------------------------------------------------------------- #
#  GAP 1 -- a negligible loss disarms the passivity screen entirely
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(strict=True, reason=(
    "VERIFY 2026-09-12 GAP 1 (open).  bor_solve._stack_is_provably_passive "
    "disarms the whole screen unless EVERY layer's max|Im eps| / max|Re eps| "
    "is <= 1e-12, on the stated reasoning that 'on a lossy stack there is no "
    "theorem to violate'.  That reasoning covers the BELOW-unity direction "
    "only: R + T <= 1 holds for every PASSIVE medium, absorbing or not.  "
    "Measured on this fixture: the screen REFUSES at rel Im = 1e-12 and "
    "RETURNS max(R+T) = 1.02882 with no warning at 3e-12, where the staggered "
    "twin reads 0.999999999998 -- i.e. the physical absorption is 2e-12 and "
    "the returned violation is the same 2.9 % either side of the threshold."))
def test_a_negligible_loss_must_not_defeat_the_nodal_passivity_screen():
    k0 = 2.0
    # PREMISE: the lossless arm must actually exhibit the pathology, i.e. the
    # guard must refuse it here.  On an arm where the nodal cascade does not
    # blow up there is nothing to be defeated and nothing to assert.
    try:
        solve(_nodal_stack(6.0 + 0j), k0)
    except BORNodalPassivityError:
        pass
    else:
        pytest.skip("premise absent on this arm: the LOSSLESS nodal stack is "
                    "not refused, so there is no screen to defeat")
    # A loss three times the 1e-12 disarm threshold: physically nothing (the
    # staggered twin of this stack absorbs 2e-12 of the incident power), but
    # enough to take the whole stack out of `provably passive`.
    rel = 3e-12
    layers = _nodal_stack(complex(6.0, 6.0 * rel))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = solve(layers, k0)                        # must not reach here
    worst = float(np.max(np.asarray(res["R"]) + np.asarray(res["T"]))) - 1.0
    assert worst <= 1e-3 or w, (
        "a %g relative loss disarmed the passivity screen: max(R+T) - 1 = %.6g "
        "was returned with %d warnings" % (rel, worst, len(w)))


# --------------------------------------------------------------------------- #
#  GAP 2 -- the screen is one-sided on a stack it has already proven lossless
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(strict=True, reason=(
    "VERIFY 2026-09-12 GAP 2 (open).  _check_nodal_passivity tests only "
    "max(R+T) - 1 > bar.  On a stack it has ALREADY established is lossless "
    "and passive, and which is closed by a PEC wall (so the propagating "
    "channels of the two half-spaces are the only exits), R + T = 1 is an "
    "EQUALITY, not an inequality -- there is no absorption for a deficit to "
    "hide in.  Measured: this fixture returns R + T = 0.537349 (a 46 % energy "
    "DEFICIT) with no error and no warning, while its staggered twin returns "
    "1.000000000000."))
def test_the_nodal_passivity_screen_must_be_two_sided_on_a_lossless_stack():
    k0, m = 2.0, 2
    Rbig = 0.5 * 2.0 * np.pi / k0                 # half a vacuum wavelength
    layers = _nodal_stack(6.0 + 0j, Rbig=Rbig, N=200, k0=k0, m=m)
    # PREMISE: the deficit has to be there on this arm.
    ref = _solve_disarmed(_nodal_stack(6.0 + 0j, Rbig=Rbig, N=200, k0=k0, m=m),
                          k0)
    E = np.asarray(ref["R"]) + np.asarray(ref["T"])
    if E.size == 0 or float(np.min(E)) > 1.0 - 1e-3:
        pytest.skip("premise absent on this arm: the nodal cascade does not "
                    "under-close on this fixture (min R+T = %.12g)"
                    % (float(np.min(E)) if E.size else float("nan")))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = solve(layers, k0)
    Eg = np.asarray(res["R"]) + np.asarray(res["T"])
    assert w or float(np.min(Eg)) > 1.0 - 1e-3, (
        "a LOSSLESS nodal stack returned min(R+T) = %.6g with %d warnings"
        % (float(np.min(Eg)), len(w)))


# --------------------------------------------------------------------------- #
#  GAP 3 -- the channel COUNT, against a closed form
# --------------------------------------------------------------------------- #
def _exact_channel_count(m, n_index, k0, Rbig, nz=200):
    """The number of propagating channels of a uniform PEC-walled cylinder.

    TE modes have ``J_m'(gamma Rbig) = 0``, TM modes ``J_m(gamma Rbig) = 0``,
    and a mode propagates when ``gamma Rbig < n k0 Rbig``.  Returns the count
    and the BASIN: the distance from the cut to the nearest zero on either
    side, in units of the cut.  Matching by basin rather than by a tolerance is
    what makes this build-free -- the count can only move if a zero crosses,
    and the nearest one is decades away in relative terms.
    """
    x = float(np.real(n_index)) * float(np.real(k0)) * float(Rbig)
    z = np.sort(np.concatenate([jnp_zeros(m, nz), jn_zeros(m, nz)]))
    n = int(np.sum(z < x))
    below = x - z[n - 1] if n > 0 else x
    above = z[n] - x if n < z.size else float("inf")
    return n, float(min(below, above) / x)


def _counts(basis, m, Rbig, N, k0, e_out=2.0, e_mid=4.0):
    prev = _bs.BOR_NODAL_PASSIVITY_GUARD
    _bs.BOR_NODAL_PASSIVITY_GUARD = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            layers = [build_layer(m, Rbig, N, _uni(e_out), k0, basis=basis),
                      build_layer(m, Rbig, N, _uni(e_mid), k0, thickness=0.4,
                                  basis=basis),
                      build_layer(m, Rbig, N, _uni(e_out), k0, basis=basis)]
            r = solve(layers, k0)
        R, T = np.asarray(r["R"]), np.asarray(r["T"])
        return int(R.size), float(np.max(R + T)) - 1.0 if R.size else None
    finally:
        _bs.BOR_NODAL_PASSIVITY_GUARD = prev


@pytest.mark.parametrize("m", [1, 2])
def test_the_staggered_channel_count_is_the_closed_form_count(m):
    """THE REFERENCE THE PASSIVITY BAR DOES NOT HAVE.  Before asking whether
    ``R + T`` is physical it is worth asking whether ``R`` and ``T`` are the
    right observables at all, and for a uniform half-space that has a closed
    form.  The div-conforming staggered basis returns exactly it.

    Build-free by construction: the assertion is an INTEGER count matched
    against Bessel zeros, and the basin (the relative distance from the
    propagation cut to the nearest zero) is asserted to be wide enough that no
    arithmetic could move the count.
    """
    k0, N = 2.0, 200
    Rbig = 2.0 * 2.0 * np.pi / k0                  # two vacuum wavelengths
    exact, basin = _exact_channel_count(m, np.sqrt(2.0), k0, Rbig)
    assert basin > 1e-3, (
        "this fixture sits too near a mode cutoff to pin a count (basin %.3g)"
        % basin)
    n_stag, _ = _counts("staggered", m, Rbig, N, k0)
    assert n_stag == exact, (
        "m=%d, Rbig=%.6g, k0=%.6g: the staggered basis returned %d channels "
        "against the closed-form %d (basin %.3g)"
        % (m, Rbig, k0, n_stag, exact, basin))


@pytest.mark.parametrize("m", [1, 2])
def test_the_nodal_channel_count_is_wrong_where_the_passivity_screen_says_ok(m):
    """THE GAP, stated as a decision rather than as a magnitude.

    The 5.45.1 passivity bar refuses a nodal solve when ``max(R+T) - 1`` passes
    1e-3 and warns above 1e-6.  On this fixture it reads ~1e-9, i.e. the guard
    says OK -- and the returned channel SET is nevertheless wrong against the
    closed form.  The energy excess is therefore not a detector of the nodal
    basis's defect; it only sees the part of it that happens to carry power.

    This is not marked xfail: the assertion is that the gap EXISTS and is
    measured, so the number below is a finding, not a target.  It is
    premise-gated -- an arm on which the nodal basis happens to return the
    right count has nothing to report.
    """
    k0, N = 2.0, 200
    Rbig = 2.0 * 2.0 * np.pi / k0
    exact, basin = _exact_channel_count(m, np.sqrt(2.0), k0, Rbig)
    assert basin > 1e-3
    n_nodal, excess = _counts("nodal", m, Rbig, N, k0)
    if n_nodal == exact:
        pytest.skip("premise absent on this arm: the nodal basis returned the "
                    "closed-form count (%d)" % exact)
    assert excess is not None and excess <= 1e-6, (
        "the premise of this finding is that the passivity screen is SILENT "
        "here; it reads max(R+T) - 1 = %.6g, which is above the 1e-6 warning "
        "edge, so the screen is not silent and the finding has to be "
        "re-stated" % excess)
    assert n_nodal > exact, (
        "m=%d: nodal returned %d channels against the closed-form %d while "
        "the passivity screen read %.4g and said OK"
        % (m, n_nodal, exact, excess))


# --------------------------------------------------------------------------- #
#  GAP 4 -- the SEM warn edge is entered by an ordinary shallow taper
# --------------------------------------------------------------------------- #
def _taper(n_slices, dr, Rbig=24.0, k0=2.0, degree=8, r_top=8.0, height=0.8):
    s = BORStack(Rbig=Rbig, m=1, N=120, n_superstrate=1.0, n_substrate=1.0,
                 basis="sem", degree=degree)
    for j in range(n_slices):
        r = r_top - dr * (j + 0.5) / n_slices
        s.add_layer(height / n_slices, segments=[(r, 4.0), (Rbig, 1.0)])
    s.set_source(k0=k0)
    return s


def test_the_sem_warn_edge_is_reached_by_a_shallow_taper_at_32_slices():
    """WHAT THE SHIPPED CENSUS'S MARGIN IS A PROPERTY OF.

    ``_BOR_SLIVER_BAND_FRAC = 1e-4`` was fixed from a census whose binding
    ordinary geometry was a taper staircase, and the build's note says the edge
    "can never REFUSE one, because ... a taper reaches [1e-6 of Rbig] only at
    ~6 million slices".  Both statements are about the census's own cone, which
    changes radius by 6 over its height: the manufactured cell of a taper is
    ``dr / n_slices``, so it is set by the cone's STEEPNESS as much as by the
    slice count, and a SHALLOW cone reaches the same cell at an ordinary slice
    count.

    This gate pins that as a decision: a 32-slice taper whose ring radius
    changes by 0.05 in total (0.6 % of the ring radius, 0.2 % of ``Rbig``) --
    ordinary conical geometry, the kind a taper-angle sweep walks through --
    lands inside the warn band and warns.  The band membership is DERIVED here
    from the library's own two constants, so the gate tracks them if they
    move.
    """
    from lumenairy.elements.bor import _sem_contract as C
    Rbig, n_slices = 24.0, 32
    dr = 0.05                   # a 0.6 % change of the ring radius, total
    cell = dr / n_slices / Rbig                 # the manufactured cell / Rbig
    assert C._BOR_MIN_ELEM_FRAC <= cell < C._BOR_SLIVER_BAND_FRAC, (
        "this fixture is meant to sit INSIDE the degradation band [%.0e, "
        "%.0e); it reads %.3e -- re-derive it from the current constants"
        % (C._BOR_MIN_ELEM_FRAC, C._BOR_SLIVER_BAND_FRAC, cell))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _taper(n_slices, dr, Rbig=Rbig).solve()
    msgs = [str(x.message) for x in w
            if "MANUFACTURED element" in str(x.message)]
    assert msgs, (
        "a %d-slice taper of total radius change %.4g on Rbig = %.4g "
        "manufactures a cell of %.3e of Rbig, inside the contract's own "
        "degradation band, and emitted no warning" % (n_slices, dr, Rbig, cell))
    # and the warning is emitted once per AFFECTED LAYER, not once per cause:
    # one geometric coincidence, n_slices messages.
    assert len(msgs) >= n_slices - 1, (
        "expected one message per affected layer (%d), got %d"
        % (n_slices, len(msgs)))


# --------------------------------------------------------------------------- #
#  GAP 5 -- warn_own is attributed to a layer that asked for nothing
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(strict=True, reason=(
    "VERIFY 2026-09-12 GAP 5 (open).  The 'warn_own' branch of "
    "_sem_contract.verdict fires on w_min_frac, the narrowest cell of the "
    "POST-WINDOW mesh, and its message tells the caller that \"the LAYER'S OWN "
    "segment list asked for\" that cell.  A neighbour of the liner layer has "
    "the cell only because the +-1 enrichment window put it there -- its own "
    "segment list is a single full-radius entry -- yet it receives the same "
    "message.  The justification for never refusing a warn_own cell ('you "
    "prescribed the geometry') does not hold for that layer."))
def test_warn_own_is_only_reported_for_the_layer_that_prescribed_the_cell():
    Rbig, w = 24.0, 1e-6 * 24.0
    s = BORStack(Rbig=Rbig, m=1, N=120, n_superstrate=1.0, n_substrate=1.0,
                 basis="sem", degree=8)
    s.add_layer(0.5, segments=[(9.0, 1.0), (9.0 + w, 4.0), (Rbig, 1.0)])
    s.add_layer(0.5, eps=1.21)              # asks for NO wall at all
    s.set_source(k0=2.0)
    with warnings.catch_warnings(record=True) as w_rec:
        warnings.simplefilter("always")
        s.solve()
    own = [str(x.message) for x in w_rec
           if "LAYER'S OWN segment list" in str(x.message)]
    if not own:
        pytest.skip("premise absent on this arm: no warn_own message was "
                    "emitted for this liner")
    blamed = sorted({int(msg.split("layer ")[1].split("'s")[0])
                     for msg in own})
    assert blamed == [0], (
        "warn_own blamed layers %s; only layer 0 has a segment list that "
        "mentions the %.3e-wide cell" % (blamed, w / Rbig)
    )


# --------------------------------------------------------------------------- #
#  GAP 6 -- the EME band's literal 1.0 floor is unit-dependent
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(strict=True, reason=(
    "VERIFY 2026-09-12 GAP 6 (open, NEW in 5.45.1).  elements/eme/_branch."
    "cut_band floors its band at a literal 1.0, justified by 'ky here is "
    "DIMENSIONLESS'.  It is not: strip_x_modes assembles d2/dx2 + eps k0^2 on "
    "a spacing Lx/Nx, so lam carries 1/length^2 and ky carries 1/length, and "
    "k0 is a free argument carrying units rather than a normalisation.  The "
    "floor therefore engages whenever max|ky| < 1 in the caller's units -- the "
    "ordinary case for a sub-micron cell written in nanometres -- and the "
    "branch decision moves with the unit system.  The BOR peer, "
    "elements/bor/_orient.orient_band_scale, floors at k0 for exactly this "
    "reason (audit P2-06).  Measured: the same 1 um cell at lambda = 1550 nm "
    "with eps_hi = 12 - 1e-6j returns 3 of 96 modes on a DIFFERENT ROOT when "
    "written in nm rather than um (worst |d ky| = 26.4382 /um); the pre-5.45.1 "
    "exact-zero pin, having no scale, returned 0 of 96."))
def test_the_eme_branch_band_is_unit_system_invariant():
    from lumenairy.elements.eme import eme_2d as _e2

    def roots(scale):
        """The same physics in two unit systems: lengths x scale, wavenumbers
        / scale.  ``ky`` then scales as 1/scale, so the comparison is made on
        ``ky * scale``, which is the unit-free observable."""
        Lx, Nx = 1.0 * scale, 96
        k0 = 2.0 * np.pi / (1.55 * scale)
        eps = np.full(Nx, 1.0 + 0j)
        eps[Nx // 3:2 * Nx // 3] = 12.0 - 1e-6j          # weak GAIN
        lam = np.asarray(_e2.strip_x_modes(eps, Lx, Nx, k0)[0], complex)
        return np.asarray(_e2._ky_forward(lam, 0.0)) * scale

    a, b = roots(1.0), roots(1e-3)          # micrometres against nanometres
    # PREMISE: the two unit systems must describe the same spectrum to begin
    # with.  If the eigensolve itself has moved, there is nothing to attribute.
    if a.size != b.size:
        pytest.skip("premise absent on this arm: the two unit systems returned "
                    "different spectrum sizes (%d, %d)" % (a.size, b.size))
    same = np.isclose(a, b, rtol=1e-6, atol=1e-6 * max(np.max(np.abs(a)), 1.0))
    n_diff = int((~same).sum())
    assert n_diff == 0, (
        "%d of %d forward roots differ between a micrometre and a nanometre "
        "statement of the SAME cell; worst |d ky * scale| = %.6g"
        % (n_diff, a.size, float(np.max(np.abs(a - b)))))
