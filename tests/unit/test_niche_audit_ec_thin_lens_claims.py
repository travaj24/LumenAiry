"""AUDIT_ADVERSARIAL_CODEBASE_2026_07_25 (E-C1 / E-C2): behavioural pins for
the documented physics of ``apply_spherical_lens`` / ``apply_aspheric_lens``.

Both findings were DOCUMENTATION-physics bugs -- the code was (and remains)
the orientation-blind single-plane thin-element screen
``phi = -k (n-1) (sag1 - sag2)``; only the docstrings claimed more than that.

* E-C1 -- ``apply_aspheric_lens`` prescribed the ``-n**2`` conic on the
  CURVED FIRST surface of a plano-convex singlet as a third-order-SA null
  for collimated input.  Measured with an exact meridional Snell + eikonal
  trace (R = 50 mm N-BK7, f/3.9, sphere-referenced): that advice gives
  10.35 waves PV against 3.93 for a plain sphere in the same orientation
  (2.6x WORSE), while the ``-n**2`` HYPERBOLOID on the EXIT surface of a
  FLAT-FIRST lens is exactly stigmatic (1.7e-7 waves).  The conic that
  minimises the SCREEN's own OPD on a curved-first singlet is a third
  value, ``-1 - (n-1)**2``.  (Numbers quoted here are this file's 200-ray
  oracle; the audit's 400-ray run reads 10.38 / 3.94 -- the ray count
  shifts the sphere-referenced PV by ~0.3%, well inside every margin
  asserted below.)
* E-C2 -- both functions claimed to compute "the exact OPD ... naturally
  including spherical aberration and all higher-order monochromatic
  aberrations".  The screen never reads the centre thickness ``d`` and
  cannot distinguish the two orientations of a plano-convex singlet (1.7%
  apart in the screen, 4.0x apart in truth).

The tests below pin (1) the "``d`` is unused" fact the docstrings now state,
(2) the corrected conic guidance against an INDEPENDENT exact-trace oracle
written in this file, (3) the screen model's own SA-null location, and
(4) that the two retracted claims stay out of the docstrings.

Self-contained: no external design assets, ~2 s.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import least_squares

from lumenairy.elements._lens_thin import (
    apply_aspheric_lens,
    apply_spherical_lens,
)

# N-BK7 at 632.8 nm, hardcoded so the oracle does not depend on the glass
# catalogue (``get_glass_index('N-BK7', 632.8e-9)`` = 1.5150890... ).
N_GLASS = 1.515089
WL_REF = 632.8e-9

_R = 50e-3          # front/back radius magnitude [m]
_D = 5e-3           # centre thickness [m]
_HA = 12.5e-3       # semi-aperture [m]  -> f/3.9


# ---------------------------------------------------------------------------
# Independent oracle: exact meridional Snell refraction + eikonal OPL
# ---------------------------------------------------------------------------

def _sphere(h, R):
    """Library's signed exact spherical wavefront S(R) (_lens_thin.py:255)."""
    return np.sign(R) * (np.sqrt(h ** 2 + R ** 2) - abs(R))


def _sphere_referenced_pv(h, W):
    """PV of *W* after removing piston + its best-fit EXACT sphere.

    Referencing to a sphere (not a parabola) matters: at NA 0.13 a parabola
    is itself ~6 waves from a sphere, and a parabola-referenced metric would
    mislabel that as aberration.
    """
    p = np.polyfit(h ** 2, W, 1)
    R0 = 1.0 / (2.0 * p[0]) if p[0] != 0 else 1.0
    sol = least_squares(lambda q: W - (q[0] + _sphere(h, q[1])), [0.0, R0],
                        xtol=1e-15, ftol=1e-15)
    return float(np.ptp(sol.fun))


def _trace_singlet(R1, k1, R2, k2, d, n_g, h_max, npts=200):
    """Exact 2-D meridional trace of a collimated on-axis pencil.

    Returns ``(h_exit, W)`` at the exit-vertex plane ``z = d``, with the
    axial glass piston ``n_g * d`` removed.  Conic surfaces are intersected
    by Newton iteration on ``z(t) - z_vertex - sag(x(t)) = 0``; refraction is
    the vector Snell law; OPL accumulates ``n * path``.
    """
    surfs = [(0.0, R1, 1.0, n_g, k1), (d, R2, n_g, 1.0, k2)]
    h0 = np.linspace(1e-9, h_max, npts)
    h_exit = np.full(npts, np.nan)
    opl_a = np.full(npts, np.nan)
    for j, x0 in enumerate(h0):
        P = np.array([x0, 0.0])          # (x, z)
        u = np.array([0.0, 1.0])         # collimated, +z
        opl = 0.0
        ok = True
        for (z_v, R_s, n1, n2, kc) in surfs:
            if np.isfinite(R_s):
                def sag(h, R_s=R_s, kc=kc):
                    q = 1.0 - (1.0 + kc) * h * h / R_s ** 2
                    if q < 0:
                        return np.nan
                    return h * h / (R_s * (1.0 + np.sqrt(q)))

                def dsag(h, R_s=R_s, kc=kc):
                    q = 1.0 - (1.0 + kc) * h * h / R_s ** 2
                    if q <= 0:
                        return np.nan
                    return h / (R_s * np.sqrt(q))
                t = (z_v - P[1]) / u[1]
                for _ in range(80):
                    Q = P + t * u
                    f = Q[1] - z_v - sag(Q[0])
                    if not np.isfinite(f):
                        ok = False
                        break
                    df = u[1] - dsag(Q[0]) * u[0]
                    if abs(df) < 1e-18:
                        ok = False
                        break
                    dt = -f / df
                    t += dt
                    if abs(dt) < 1e-16:
                        break
                if not ok:
                    break
                P2 = P + t * u
                m = dsag(P2[0])
                nrm = np.array([-m, 1.0]) / np.hypot(m, 1.0)
            else:
                t = (z_v - P[1]) / u[1]
                P2 = P + t * u
                nrm = np.array([0.0, 1.0])
            opl += n1 * t
            ci = float(np.dot(u, nrm))
            eta = n1 / n2
            disc = 1.0 - eta * eta * (1.0 - ci * ci)
            if disc < 0:                 # TIR
                ok = False
                break
            u = eta * u + (np.sqrt(disc) - eta * ci) * nrm
            u /= np.linalg.norm(u)
            P = P2
        if not ok:
            continue
        t = (d - P[1]) / u[1]            # coast to the exit-vertex plane
        opl += 1.0 * t
        h_exit[j] = (P + t * u)[0]
        opl_a[j] = opl
    good = np.isfinite(h_exit)
    return h_exit[good], opl_a[good] - n_g * d


def _exact_sa_waves(R1, k1, R2, k2):
    h, W = _trace_singlet(R1, k1, R2, k2, _D, N_GLASS, _HA)
    assert h.size > 150, f"oracle lost rays: only {h.size} survived"
    return _sphere_referenced_pv(h, W) / WL_REF


# ---------------------------------------------------------------------------
# Screen-model probe: read the OPD the library actually imprints
# ---------------------------------------------------------------------------

# The screen OPD ``(n-1)*(sag1-sag2)`` is WAVELENGTH-INDEPENDENT (the
# wavelength only enters as the ``-k`` prefactor), so we probe with a long
# wavelength that keeps ``|phi| < pi``.  ``np.angle`` then recovers the OPD
# with no unwrapping ambiguity at all (max |phi| ~ 0.5 rad here).
_WL_PROBE = 1e-2
_N_PROBE = 1250
_DX_PROBE = 2e-5     # covers h = 0 .. 12.48 mm on the y = 0 row


def _screen_opd_cut(**kwargs):
    """(h, OPD[m]) on the y = 0 row of ``apply_aspheric_lens``'s screen."""
    E = np.ones((4, _N_PROBE), dtype=np.complex128)
    out = apply_aspheric_lens(E, wavelength=_WL_PROBE, dx=_DX_PROBE, **kwargs)
    row = out[2]                                       # y = 0 (yc = 0)
    phi = np.angle(row)
    assert np.abs(phi).max() < np.pi * 0.9, (
        "probe wavelength too short -- phase wrapped, np.angle is ambiguous")
    opd = -phi / (2 * np.pi / _WL_PROBE)
    h = np.abs((np.arange(_N_PROBE) - _N_PROBE / 2) * _DX_PROBE)
    sl = slice(_N_PROBE // 2, _N_PROBE)                # h >= 0 half
    return h[sl], opd[sl]


# ---------------------------------------------------------------------------
# 1. ``d`` is accepted but not used  (E-C2, the documented fact)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('func', [apply_spherical_lens, apply_aspheric_lens])
def test_ec2_thickness_d_is_not_read_by_the_screen(func):
    """d=1 nm and d=1 m must give BIT-IDENTICAL fields.

    This pins the ``d`` parameter documentation ("ACCEPTED FOR SIGNATURE
    COMPATIBILITY BUT NOT USED BY THIS MODEL") as a behavioural fact rather
    than a comment.

    IF THIS TEST EVER FAILS BECAUSE ``d`` WAS DELIBERATELY WIRED INTO THE
    MODEL (finding E-L6, deferred), DO NOT JUST RELAX THE ASSERTION: the
    ``d`` parameter docstrings of BOTH functions, and the "Validity
    boundary" section of :func:`apply_spherical_lens`, state that ``d`` has
    no effect and must be revised in the same commit.
    """
    rng = np.random.default_rng(725)
    N = 48
    E = (rng.standard_normal((N, N))
         + 1j * rng.standard_normal((N, N))).astype(np.complex128)
    kw = dict(R1=_R, R2=-_R, n_lens=N_GLASS, wavelength=WL_REF, dx=100e-6)
    a = func(E, d=1e-9, **kw)
    b = func(E, d=1.0, **kw)
    assert np.array_equal(a, b), (
        f"{func.__name__}: d is now read by the model "
        f"(max|dE| = {np.abs(a - b).max():.3e}) -- update the `d` parameter "
        f"docstrings, which promise it is unused")


# ---------------------------------------------------------------------------
# 2. E-C1: the -n**2 conic belongs on the EXIT surface of a FLAT-FIRST lens
# ---------------------------------------------------------------------------

def test_ec1_minus_n_squared_conic_is_a_flat_first_exit_surface_null():
    """Independent exact-trace oracle for the corrected docstring guidance.

    * flat-first, hyperbolic EXIT surface ``k2 = -n**2`` -> stigmatic.
    * curved-first, ``k1 = -n**2`` (the RETRACTED advice) -> far worse than
      a plain sphere in the same orientation.
    """
    k_hyper = -N_GLASS ** 2

    pv_flat_first_hyperbola = _exact_sa_waves(np.inf, 0.0, -_R, k_hyper)
    pv_curved_first_hyperbola = _exact_sa_waves(_R, k_hyper, np.inf, 0.0)
    pv_curved_first_sphere = _exact_sa_waves(_R, 0.0, np.inf, 0.0)

    # Exactly stigmatic (measured 1.7e-7 waves; the residual is Newton /
    # least-squares roundoff, not physics).
    assert pv_flat_first_hyperbola < 0.01, (
        f"flat-first k2=-n^2 should be stigmatic, got "
        f"{pv_flat_first_hyperbola:.6f} waves PV")

    # The retracted advice: measured 10.35 waves.
    assert pv_curved_first_hyperbola > 5.0, (
        f"curved-first k1=-n^2 should be badly aberrated, got "
        f"{pv_curved_first_hyperbola:.6f} waves PV")

    # ... and specifically WORSE than doing nothing (measured 3.93 waves).
    assert pv_curved_first_hyperbola > 2.0 * pv_curved_first_sphere, (
        f"curved-first k1=-n^2 ({pv_curved_first_hyperbola:.4f}) should be "
        f">2x the plain sphere ({pv_curved_first_sphere:.4f})")


# ---------------------------------------------------------------------------
# 3. E-C1: the SCREEN model's own SA null is at k = -1 - (n-1)**2
# ---------------------------------------------------------------------------

def test_ec1_screen_model_sa_null_is_at_minus_one_minus_n_minus_one_squared():
    """Scan k on the OPD ``apply_aspheric_lens`` actually imprints.

    The screen's stationary point is displaced from the real lens's because
    the screen is the normal (z) sag PROJECTION; the docstring says so and
    quotes this value.
    """
    ks = np.linspace(-2.0, -0.5, 151)
    pv = np.empty(ks.size)
    for i, kk in enumerate(ks):
        h, opd = _screen_opd_cut(R1=_R, R2=np.inf, d=_D, n_lens=N_GLASS,
                                 k1=float(kk))
        pv[i] = _sphere_referenced_pv(h, opd) / WL_REF
    k_min = float(ks[int(np.argmin(pv))])
    k_pred = -1.0 - (N_GLASS - 1.0) ** 2          # = -1.26532
    assert abs(k_min - k_pred) < 0.05, (
        f"screen SA null moved: argmin k = {k_min:+.5f}, predicted "
        f"{k_pred:+.5f} (PV {pv.min():.5f} waves)")

    # The null is a real, deep minimum, not a flat floor: k = 0 (sphere) is
    # measured at 8.55 waves PV on the same grid.
    h0, opd0 = _screen_opd_cut(R1=_R, R2=np.inf, d=_D, n_lens=N_GLASS, k1=0.0)
    pv_sphere = _sphere_referenced_pv(h0, opd0) / WL_REF
    assert pv_sphere > 100.0 * pv.min(), (
        f"screen null not a deep minimum: sphere {pv_sphere:.4f} vs null "
        f"{pv.min():.5f} waves")

    # And it is NOT the real lens's null: the same conic in an exact trace
    # still leaves several waves (measured 4.34).
    assert _exact_sa_waves(_R, k_pred, np.inf, 0.0) > 1.0


# ---------------------------------------------------------------------------
# 4. The retracted claims must stay out of the docstrings
# ---------------------------------------------------------------------------

def _squash(text):
    """Collapse all whitespace so substring checks survive re-wrapping."""
    return ' '.join(text.split())


@pytest.mark.parametrize('func', [apply_spherical_lens, apply_aspheric_lens])
def test_ec1_ec2_retracted_claims_absent_from_docstrings(func):
    assert func.__doc__, f"{func.__name__} lost its docstring"
    doc = _squash(func.__doc__)

    # E-C2: the false "exact OPD" framing.
    assert 'exact OPD' not in doc, (
        f"{func.__name__}: the retracted 'exact OPD' claim is back -- this "
        f"function computes the thin-element sag-projection screen, not an "
        f"exact through-element OPD (audit 2026-07-25 E-C2)")
    assert 'all higher-order monochromatic aberrations' not in doc

    # E-C1: the wrong-surface conic prescription.
    assert 'k1 = -n_lens**2' not in doc, (
        f"{func.__name__}: the retracted curved-first -n^2 SA-null advice is "
        f"back (audit 2026-07-25 E-C1)")

    # ... and the honest wording must be present.
    assert 'THIN-ELEMENT' in doc
    assert 'orientation' in doc.lower(), (
        f"{func.__name__}: the orientation-blindness caveat is missing")
    assert 'apply_real_lens_traced' in doc, (
        f"{func.__name__}: must point users needing real-lens accuracy at "
        f"apply_real_lens_traced")
    assert 'NOT USED BY THIS MODEL' in doc, (
        f"{func.__name__}: the `d`-is-unused note is missing")


def test_ec1_corrected_conic_guidance_present_in_aspheric_doc():
    doc = _squash(apply_aspheric_lens.__doc__)
    low = doc.lower()
    # the corrected placement (exit surface of a flat-first lens)
    assert 'exit surface' in low and 'flat-first' in low, (
        "apply_aspheric_lens must state that the -n^2 hyperboloid belongs on "
        "the EXIT surface of a FLAT-FIRST plano-convex lens")
    assert 'stigmatic' in low
    # the screen model's own (different) null, quoted symbolically
    assert '-1 - (n_lens - 1)**2' in doc, (
        "apply_aspheric_lens must quote the screen model's SA-minimising "
        "conic -1 - (n_lens - 1)**2")


def test_ec2_validity_boundary_section_present_in_spherical_doc():
    doc = _squash(apply_spherical_lens.__doc__)
    assert 'Validity boundary' in doc
    # mirrors apply_real_lens's "Oblique validity boundary" wording
    assert 'normal-projected thin phase screen' in doc
    assert 'sag * theta**2' in doc
    assert 'design-dependent' in doc
