"""Audit S5-3 [P2][seam]: the patterned-medium argument silently switches
between refractive INDEX and PERMITTIVITY across the engine entry points.

The scalar ``*_efficiency_1d`` entries take the refractive INDEX ``n`` for the
patterned regions, whereas the ``*_jones_1d`` family and EVERY 2-D entry take
the PERMITTIVITY ``eps = n**2`` -- while ``n_substrate`` / ``n_superstrate``
are INDEX everywhere, so a single 2-D / Jones call mixes both conventions.  A
wrong-convention value (``n=2.1`` handed to a Jones/2-D slot as ``eps=2.1``) is
silently accepted and yields a plausible wrong answer (``n_eff ~ 1.45``).
``rcwa_efficiency_1d`` / ``rcwa_jones_1d`` already carried a CONVENTION WARNING;
the fix mirrors it onto ``pmm_efficiency_1d`` / ``pmm_jones_1d`` and documents
the mixed signature on ``rcwa_efficiency_2d`` / ``pmm_efficiency_2d``.

The tests below verify (1) the CONVENTION WARNING is present on every seam entry
point (guards the doc change from silent removal), (2) an INDEPENDENT physics
oracle that the documented mapping is the physically correct one -- the scalar
``rcwa_efficiency_1d(n)`` reproduces ``rcwa_jones_1d(eps=n**2)`` per order, and
the two scalar engines agree under the shared INDEX convention, and (3) the
footgun the warning describes: feeding an INDEX where PERMITTIVITY is expected
is silently accepted and returns a materially different (wrong) answer.
"""
import numpy as np

from lumenairy.elements.pmm import (
    pmm_efficiency_1d,
    pmm_efficiency_2d,
    pmm_jones_1d,
)
from lumenairy.elements.rcwa import (
    rcwa_efficiency_1d,
    rcwa_efficiency_2d,
    rcwa_jones_1d,
)


# --------------------------------------------------------------------------- #
# (1) the CONVENTION WARNING reaches every seam entry point
# --------------------------------------------------------------------------- #
def test_convention_warning_present_on_all_seam_entrypoints():
    # Pre-existing anchors (must not regress).
    for fn in (rcwa_efficiency_1d, rcwa_jones_1d):
        assert "CONVENTION WARNING" in fn.__doc__

    # Newly documented by the S5-3 fix: the scalar PMM entry (INDEX) and the
    # Jones / 2-D entries (PERMITTIVITY, mixed with INDEX half-spaces).
    for fn in (pmm_efficiency_1d, pmm_jones_1d,
               rcwa_efficiency_2d, pmm_efficiency_2d):
        doc = fn.__doc__
        assert "CONVENTION WARNING" in doc, f"{fn.__name__} lost its seam note"
        # both halves of the seam must be named so the reader can act
        assert "INDEX" in doc and "PERMITTIVITY" in doc, fn.__name__
        assert "n**2" in doc, fn.__name__


# --------------------------------------------------------------------------- #
# (2) independent oracle: the DOCUMENTED convention is the physically correct
#     one -- n for the scalar efficiency == n**2 for the Jones tensor.
# --------------------------------------------------------------------------- #
def test_index_efficiency_matches_permittivity_jones():
    # A propagating multi-order dielectric grating (a few orders open in the
    # n=1.5 substrate) at normal incidence.
    period, wl = 0.8e-6, 0.6e-6
    n_r, n_g, n_sub, n_sup = 2.1, 1.0, 1.5, 1.0
    depth, dc, M = 0.30e-6, 0.5, 11

    orders_e, R_e, T_e = rcwa_efficiency_1d(
        period, n_r, n_g, n_sub, n_sup, depth, dc, wl,
        angle=0.0, polarization="te", n_orders=M)

    # Jones takes PERMITTIVITY -> square the index; isotropic == n**2 * I3.
    eps_r = (n_r ** 2) * np.eye(3)
    eps_g = (n_g ** 2) * np.eye(3)
    orders_j, R_j, T_j, _J = rcwa_jones_1d(
        period, eps_r, eps_g, n_sub, n_sup, depth, dc, wl,
        angle=0.0, n_orders=M)

    assert np.array_equal(orders_e, orders_j)
    # TE = E along the grooves (y); that is the incident-E_y row (index 1) of
    # the Jones efficiency block.  If the index-vs-eps mapping were wrong the
    # spectra would not coincide.
    assert np.allclose(R_e, R_j[1], atol=1e-9), (R_e, R_j[1])
    assert np.allclose(T_e, T_j[1], atol=1e-9), (T_e, T_j[1])
    # sanity: a non-trivial, energy-passive response (not a degenerate all-zero
    # match that would pass vacuously).
    tot = float(R_e.sum() + T_e.sum())
    assert 0.99 < tot < 1.01
    assert R_e.sum() > 1e-3


def test_scalar_engines_share_the_index_convention():
    # The finding: n_ridge/n_groove are INDEX and "probed identical results
    # across engines".  PMM (spectral element) vs RCWA (Fourier) under the
    # SHARED index convention must agree to the PMM convergence tolerance.
    period, wl = 0.8e-6, 0.6e-6
    n_r, n_g, n_sub, n_sup = 2.1, 1.0, 1.5, 1.0
    depth, dc = 0.30e-6, 0.5

    o_r, R_r, T_r = rcwa_efficiency_1d(
        period, n_r, n_g, n_sub, n_sup, depth, dc, wl,
        angle=0.0, polarization="te", n_orders=15)
    o_p, R_p, T_p = pmm_efficiency_1d(
        period, n_r, n_g, n_sub, n_sup, depth, dc, wl,
        angle=0.0, polarization="te", degree=16, far_field_orders=15)

    # total reflectance / transmittance agree across engines (per-order sums are
    # the convention-independent physical observables).
    assert abs(float(R_r.sum()) - float(R_p.sum())) < 2e-3
    assert abs(float(T_r.sum()) - float(T_p.sum())) < 2e-3


# --------------------------------------------------------------------------- #
# (3) the footgun the warning names: an INDEX fed into a PERMITTIVITY slot is
#     silently accepted and returns a plausible-but-WRONG answer.
# --------------------------------------------------------------------------- #
def test_wrong_convention_is_silently_accepted_and_wrong():
    period, wl = 0.8e-6, 0.6e-6
    n_r, n_g, n_sub, n_sup = 2.1, 1.0, 1.5, 1.0
    depth, dc, M = 0.30e-6, 0.5, 11

    # correct: eps = n**2
    _o, R_ok, _T, _J = rcwa_jones_1d(
        period, (n_r ** 2) * np.eye(3), (n_g ** 2) * np.eye(3),
        n_sub, n_sup, depth, dc, wl, angle=0.0, n_orders=M)
    # wrong (the footgun): the INDEX handed straight into the eps slot ->
    # the ridge is modelled as n_eff = sqrt(2.1) ~ 1.449 instead of 2.1.
    _o, R_bad, _T, _J = rcwa_jones_1d(
        period, n_r * np.eye(3), n_g * np.eye(3),
        n_sub, n_sup, depth, dc, wl, angle=0.0, n_orders=M)

    # no exception was raised (silently accepted) and the zeroth-order TE
    # reflectance is materially different -> a wrong but plausible answer.
    p0 = M  # zeroth order lives at the array centre
    assert abs(float(R_ok[1, p0]) - float(R_bad[1, p0])) > 1e-2
