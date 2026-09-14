"""VERIFY-A6 -- independent re-verification of WP-A6 (audit findings C1-C5 on
``lumenairy/propagators/carrier.py`` and ``carrier_field.py``).

These tests were written by the VERIFIER, not by the engineer whose fixes they
check, and every oracle here is built in this file from first principles --
``np.fft.fftfreq``, a hand-written Gaussian ABCD ``q``-parameter propagation, a
dense numerical scan of the containment inequality, a hand-written copy of the
PRE-FIX moment estimator -- so that none of them can inherit an error from the
code under test.  Fixtures deliberately differ from the WP's own
(``lambda = 0.85 um`` and ``0.633 um`` rather than 1.31 um; NA 0.08; odd and
anamorphic grids) so that the claims are re-measured, not re-read.

Two defects in WP-A6's own files are pinned here with their fail-before arms:

* the C1 containment solver returned ``0.0`` ("margin unreachable at ANY leg
  length") whenever the quadratic opened downward, although the containment set
  is then the interval BETWEEN the roots and a qualifying leg exists whenever
  the input plane itself is contained -- see
  ``TestVerifyC1Quadratic::test_the_downward_quadratic_still_resolves_a_leg``;
* the C4 separable-screen agreement was documented, and barred, as a FIXED
  absolute number, while the measured difference scales with the screen's own
  argument -- see ``TestVerifyC4SeparableBound``.

Per ``docs/TESTING_STANDARDS.md``: no wall-clock assertion, no ``pytest.skip``
on a resource precondition, every bar derived from its oracle's own floor with
the measured value and the decades of clearance in the comment.
"""

import dataclasses
import warnings

import numpy as np
import pytest

from lumenairy.propagators import carrier as C
from lumenairy.propagators import carrier_field as CF
from lumenairy.propagators.carrier_field import (
    CarrierField,
    CarrierSpec,
    FieldGrid,
)

EPS = float(np.finfo(np.float64).eps)


# ===========================================================================
# oracles (written here; nothing below calls the library to build a truth)
# ===========================================================================
def _grid(n, dx):
    return (np.arange(n, dtype=np.float64) - n / 2) * dx


def _gauss(n, dx, w, xc=0.0, yc=0.0):
    g = _grid(n, dx)
    return np.exp(-(((g[None, :] - xc) ** 2 + (g[:, None] - yc) ** 2) / w ** 2))


def _abcd_field(xo, w_in, r_beam, z, lam):
    """Analytic Gaussian at distance ``z``, in the WHOLE-FUNCTION form

        ``E(r, z) = exp(i k z) / (1 + z/q) * exp(i k r^2 / (2 q(z)))``

    with ``1/q = 1/R + i lam / (pi w^2)`` and ``q(z) = q + z``.

    THE SIGN OF THE IMAGINARY PART IS THE WHOLE POINT.  This library's
    ``exp(-i omega t)`` convention (``CONVENTIONS.md`` Section 7) pairs a
    forward-propagating wave with ``exp(+i k z)``, and the complex curvature
    that goes with it is ``1/q = 1/R + i lam/(pi w^2)``.  Siegman's
    ``exp(+i omega t)`` pairing takes the conjugate, ``1/q = 1/R - i
    lam/(pi w^2)``, and carrying the Gouy term separately as
    ``angle(q/q2)`` on top of THAT gives the Gouy phase the WRONG SIGN in
    this convention -- a 2 x arctan(z/zR) error, reaching exactly pi across a
    focus.  Every assertion in this file is piston-free (each compares a
    normalised field or a width), so nothing here failed; the oracle was
    simply not the convention it was oracle FOR.  The whole-function form
    above cannot carry the two halves in different conventions, because there
    is only one ``q``: the amplitude ``|q/q2| = w_in/w(z)``, the wavefront
    curvature and the Gouy phase are all the argument of ONE complex number.

    ``wz`` is read back from the same ``q``: ``w(z) = sqrt(lam / (pi *
    Im(1/q(z))))``, positive by construction here where the old spelling
    needed a minus sign to undo its own conjugation.

    Returns ``(E, w(z))``.  ``E`` is a complete field -- amplitude, curvature
    and Gouy phase -- not an intensity profile, and it carries the absolute
    piston ``exp(i k z)``, which
    :func:`test_the_gaussian_oracle_carries_this_librarys_phase_convention`
    pins.
    """
    kk = 2.0 * np.pi / lam
    q = 1.0 / (1.0 / r_beam + 1j * lam / (np.pi * w_in ** 2))
    q2 = q + z
    wz = float(np.sqrt(lam / (np.pi * np.imag(1.0 / q2))))
    xx, yy = np.meshgrid(xo, xo, indexing='xy')
    r2 = xx ** 2 + yy ** 2
    return (np.exp(1j * kk * z) / (1.0 + z / q)
            * np.exp(1j * kk * r2 / (2.0 * q2))), wz


def _brute_containment_edge(half, zeta_cf, w_env, c, zr, margin, n=400001):
    """The edge of the containment region that CONTAINS THE INPUT PLANE, found
    by a dense scan plus 200 bisections of the inequality
    ``half |zeta - zeta_cf| / zeta_cf >= margin * w_beam(zeta)``.

    Written from the geometry (co-moving contraction x Gaussian ABCD width), so
    it shares no expression with the closed form under test.  Returns ``None``
    when the input plane itself is not contained."""
    def ok(zz):
        hs = half * abs(zz - zeta_cf) / zeta_cf
        wb = w_env * np.sqrt((1.0 + zz * c) ** 2 + (zz / zr) ** 2)
        return hs >= margin * wb

    if not ok(0.0):
        return None
    z = np.linspace(0.0, zeta_cf * (1.0 - 1e-12), n)
    hs = half * np.abs(z - zeta_cf) / zeta_cf
    wb = w_env * np.sqrt((1.0 + z * c) ** 2 + (z / zr) ** 2)
    bad = np.flatnonzero(hs < margin * wb)
    if bad.size == 0:
        return float(z[-1])
    lo, hi = float(z[bad[0] - 1]), float(z[bad[0]])
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _prefix_fit_inv(E, wavelength, dx, dy, axis=None, estimator='gradient'):
    """The PRE-C2 estimator, transcribed from the audit's quotation of the
    shipped source (grid origin, ``meshgrid``, no tilt projection).  Used to
    prove that ``centre='origin'`` is the historical arithmetic BIT for bit --
    a claim WP-A6 makes against its own new code path."""
    E = np.asarray(E)
    ny, nx = E.shape[-2], E.shape[-1]
    k = 2.0 * np.pi / wavelength
    x = _grid(nx, dx)
    y = _grid(ny, dy)
    Y, X = np.meshgrid(y, x, indexing='ij')
    if estimator == 'increment':
        num = den = 0.0
        if axis is None or axis == 1:
            dphi = np.angle(E[:, 1:] * np.conj(E[:, :-1]))
            wgt = np.abs(E[:, 1:]) * np.abs(E[:, :-1])
            xm = 0.5 * (X[:, 1:] + X[:, :-1])
            num += float(np.sum(wgt * xm * (dphi / dx)))
            den += k * float(np.sum(wgt * xm * xm))
        if axis is None or axis == 0:
            dphi = np.angle(E[1:, :] * np.conj(E[:-1, :]))
            wgt = np.abs(E[1:, :]) * np.abs(E[:-1, :])
            ym = 0.5 * (Y[1:, :] + Y[:-1, :])
            num += float(np.sum(wgt * ym * (dphi / dy)))
            den += k * float(np.sum(wgt * ym * ym))
    else:
        inten = np.abs(E) ** 2
        dE = np.gradient(E, dx, axis=1)
        num = np.imag(np.sum(np.conj(E) * X * dE))
        dE = np.gradient(E, dy, axis=0)
        num += np.imag(np.sum(np.conj(E) * Y * dE))
        den = k * float(np.sum(inten * (X * X + Y * Y)))
    return 0.0 if den == 0.0 else float(num) / float(den)


# ===========================================================================
# C1 -- the closed-form containment quadratic
# ===========================================================================
class TestVerifyC1Quadratic:
    """``_beam_containment_standoff`` solves ``half|zeta-zeta_cf|/zeta_cf >=
    M w_beam(zeta)`` in closed form.  The verification is against a dense
    numerical scan of that same inequality, built from the geometry here."""

    LAM = 0.85e-6

    @pytest.mark.parametrize('r_mm', [-7.5, -20.0, -3.0])
    @pytest.mark.parametrize('inv_env', [7.0, -7.0, 120.0, 1e-9])
    def test_the_closed_form_is_the_numerical_boundary(self, r_mm, inv_env):
        """ORACLE: ``_brute_containment_edge`` -- 4e5 samples plus 200
        bisections, i.e. an edge located to ~1e-60 of the interval, so the
        oracle's own floor is float64 round-off on ``zeta`` (~1e-16 relative).

        BAR 1e-12 relative: measured worst 1.25e-14 over the 28 comparable
        cells of the verifier's full matrix (3 radii x 2 widths x 3 extents x
        6 residual curvatures), i.e. ~80x under the bar, while a sign error or
        a wrong root is O(1) -- 12 decades up."""
        lam = self.LAM
        r = r_mm * 1e-3
        w_env = 200e-6
        half = 6.0 * w_env                      # ext 6: gamma > 0 by design
        zr = np.pi * w_env * w_env / lam
        zeta_cf = -r
        c = 1.0 / r + inv_env
        z_far = zeta_cf * 0.999
        edge = _brute_containment_edge(half, zeta_cf, w_env, c, zr,
                                       C._FOCUS_STANDOFF_MARGIN)
        s = C._beam_containment_standoff(
            np.ones((8, 8), complex), r, z_far, lam, 1e-6, w_env,
            (0.0, 0.0), half, inv_env=inv_env)
        assert edge is not None, 'fixture must contain the input plane'
        got = z_far - s
        assert got == pytest.approx(edge, rel=1e-12), (r_mm, inv_env, got, edge)

    def test_the_carrier_focus_is_always_between_the_roots(self):
        """The claim the closed form's root SELECTION rests on.  Checked as an
        identity rather than a sample: ``q(zeta_cf)`` must equal
        ``-Q[(1 + c zeta_cf)^2 + (zeta_cf/zR)^2]``, which is negative unless
        both terms vanish (impossible for a finite ``zR``).

        The identity's own accuracy is CANCELLATION-limited (the ``h2`` terms
        sum to zero exactly), so it is bounded against the dominant term rather
        than relatively -- see the bar at the assertion.  20000 log-uniform
        cells spanning |R| 1 mm..100 mm, w 10 um..1 mm, ext 1..16 and
        |1/R_env| up to 1e3."""
        rng = np.random.default_rng(20260912)
        lam = self.LAM
        m = C._FOCUS_STANDOFF_MARGIN
        worst_rel = 0.0
        worst_sign = -np.inf
        for _ in range(20000):
            r = -10.0 ** rng.uniform(-3, -1)
            w_env = 10.0 ** rng.uniform(-5, -3)
            half = w_env * 10.0 ** rng.uniform(0.0, 1.2)
            inv_env = rng.uniform(-1, 1) * 10.0 ** rng.uniform(0, 3)
            zr = np.pi * w_env * w_env / lam
            c = 1.0 / r + inv_env
            zcf = -r
            q = (m * w_env) ** 2
            h2 = half * half
            a = h2 / zcf ** 2 - q * (c * c + 1.0 / zr ** 2)
            b = -2.0 * h2 / zcf - 2.0 * q * c
            g = h2 - q
            val = a * zcf ** 2 + b * zcf + g
            pred = -q * ((1.0 + c * zcf) ** 2 + (zcf / zr) ** 2)
            # the h2 terms cancel EXACTLY in exact arithmetic
            # (h2 - 2 h2 + h2), so the achievable accuracy of the identity is
            # set by the largest term in the sum, not by the residual: bound
            # the comparison by 64 eps times that scale rather than by a
            # relative tolerance the cancellation cannot deliver.
            scale = max(h2, q * ((1.0 + abs(c) * zcf) ** 2 + (zcf / zr) ** 2))
            worst_rel = max(worst_rel,
                            abs(val - pred) / (64.0 * EPS * scale))
            worst_sign = max(worst_sign, val / abs(pred))
        # BAR 1.0 on that derived ratio: measured worst 0.051 over the 20000
        # cells (i.e. the identity holds to 20 eps of the dominant term); a
        # wrong coefficient would be O(1) relative, ~1e14 on this ratio.
        assert worst_rel < 1.0, worst_rel
        assert worst_sign < 0.0, worst_sign

    def test_the_downward_quadratic_still_resolves_a_leg(self):
        """DEFECT FOUND AND FIXED BY THE VERIFIER (WP-A6's own C1 code).

        The shipped solver returned ``0.0`` for every ``alpha <= 0``, on the
        stated grounds that "the grid is too narrow for this beam at ANY leg
        length".  That is only true when the INPUT plane is also uncontained.
        With ``alpha < 0`` the parabola opens DOWNWARD, so the containment set
        is the closed interval between the roots; when ``gamma = q(0) >= 0``
        the input plane is inside it and the larger root is a perfectly good
        stop plane.  ``alpha`` flips sign with the carrier mismatch (it is
        ``half^2/zeta_cf^2 - Q(c^2 + 1/zR^2)``), so this was reachable from the
        same fixture family the finding is about.

        FAIL-BEFORE, in process: restore the ``alpha > 0`` gate and confirm the
        leg collapses to the shipped one and the guard refuses.

        MEASURED on this fixture (R = -20 mm, w_env = 200 um, ext = 6,
        residual 1/R_env = -60 /m, lambda = 0.85 um):
            pre  : leg 1705.9 um, containment 0.866 measured / 0.500 modelled,
                   guard REFUSES, peak 1.408x low
            post : leg 5929.9 um (the brute-force boundary to 8 significant
                   figures), containment 3.1996 / 3.2000, guard silent
        """
        lam, k = self.LAM, 2.0 * np.pi / self.LAM
        n, w2, ext, r = 512, 200e-6, 6.0, -20e-3
        inv_res = -60.0
        dx = 2.0 * ext * w2 / n
        g = _grid(n, dx)
        r2 = g[None, :] ** 2 + g[:, None] ** 2
        env = np.exp(-r2 / w2 ** 2) * np.exp(1j * k * r2 * 0.5 * inv_res)
        z = -r
        cen = C._envelope_amp_centroid(env, dx, dx)
        w_env = C._envelope_amp_radius(env, dx, dx, centre=cen)
        half = 0.5 * n * dx - max(abs(cen[0]), abs(cen[1]))
        inv_fit = C._fit_carrier_inv(env, lam, dx, dx, axis=None,
                                     estimator='increment', centre=cen)
        c = 1.0 / r + inv_fit
        zr = np.pi * w_env * w_env / lam
        q = (C._FOCUS_STANDOFF_MARGIN * w_env) ** 2
        alpha = half ** 2 / z ** 2 - q * (c * c + 1.0 / zr ** 2)
        gamma = half ** 2 - q
        # the fixture must actually be in the branch under test
        assert alpha < 0.0 and gamma > 0.0, (alpha, gamma)

        edge = _brute_containment_edge(half, z, w_env, c, zr,
                                       C._FOCUS_STANDOFF_MARGIN)
        assert edge is not None
        s_new = C._beam_containment_standoff(env, r, z, lam, dx, w_env, cen,
                                             half, inv_env=inv_fit)
        # BAR 2e-6 relative: the brute edge is bisected to float64 round-off
        # but its SCAN start is on a 4e5-point lattice, so the comparison is
        # limited by the bisection bracket, not by the solver; measured 1.2e-11.
        assert (z - s_new) == pytest.approx(edge, rel=2e-6), (s_new, edge)

        pd = {}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            C.carrier_referenced_focus_readout(
                env, r, z, lam, dx, dx_out=1e-6, N_out=48,
                on_replica='ignore', on_focus_containment='warn',
                _period_out=pd)
        assert pd['containment'] > C._FOCUS_READOUT_CONTAINMENT_MIN, pd
        assert pd['containment'] == pytest.approx(C._FOCUS_STANDOFF_MARGIN,
                                                  rel=2e-3), pd
        assert not [x for x in w if 'co-moving' in str(x.message)]

        # FAIL-BEFORE: the pre-fix gate, restored in process
        real = C._beam_containment_standoff

        def _gated(*a, **kw):
            r_ = a[1]
            zeta_cf = -float(r_)
            w_, half_ = a[5], a[7]
            iv = kw.get('inv_env')
            cc = 1.0 / float(r_) + float(iv)
            zr_ = np.pi * w_ * w_ / a[3]
            al = (half_ ** 2 / zeta_cf ** 2
                  - (C._FOCUS_STANDOFF_MARGIN * w_) ** 2
                  * (cc * cc + 1.0 / zr_ ** 2))
            return real(*a, **kw) if al > 0.0 else 0.0

        try:
            C._beam_containment_standoff = _gated
            pd2 = {}
            with pytest.raises(RuntimeError, match='does not fit the co-moving'):
                C.carrier_referenced_focus_readout(
                    env, r, z, lam, dx, dx_out=1e-6, N_out=48,
                    on_replica='ignore', _period_out=pd2)
            assert pd2['containment'] < 1.0, pd2
        finally:
            C._beam_containment_standoff = real

    def test_the_resolver_never_shortens_and_never_moves_a_real_envelope(self):
        """The invariant the C1 change rests on -- ``max(s_shipped, s_beam)``
        with ``s_beam == 0.0`` exactly whenever the envelope is flat.  Probed
        OUTSIDE the resolver's own 6x10 calibration matrix (NA down to 0.01 and
        up to 0.45, extents 0.8..16) and on envelopes the WP did not use: a
        top-hat, a two-lobe pair, a decentred Gaussian, a 10 %-noise Gaussian
        and residual curvature of BOTH signs.

        No tolerance: the flat/real arms must be EQUAL to the pre-fix leg (the
        beam term short-circuits on ``inv_env == 0.0``), and no arm may be
        shorter.  Measured over the verifier's full 910-cell sweep: 0 shorter,
        0 moved on the five real-amplitude kinds, 81 lengthened on the two
        curved ones."""
        lam, k = 1.31e-6, 2.0 * np.pi / 1.31e-6
        rmag, n = 20e-3, 256

        def pre_fix(env, dx):
            real = C._beam_containment_standoff
            try:
                C._beam_containment_standoff = lambda *a, **kw: 0.0
                return C._default_focus_standoff(env, -rmag, rmag, lam, dx)
            finally:
                C._beam_containment_standoff = real

        shorter, moved_real, lengthened = [], [], 0
        for na in (0.01, 0.05, 0.20, 0.45):
            for ext in (0.8, 1.2, 2.0, 4.0, 10.0, 16.0):
                w = na * rmag
                dx = 2.0 * ext * w / n
                g = _grid(n, dx)
                r2 = g[None, :] ** 2 + g[:, None] ** 2
                rng = np.random.default_rng(5)
                kinds = {
                    'flat': _gauss(n, dx, w).astype(complex),
                    'noisy': (_gauss(n, dx, w)
                              * (1.0 + 0.1 * rng.standard_normal((n, n)))
                              ).astype(complex),
                    'tophat': (r2 <= w * w).astype(complex),
                    'twolobe': (_gauss(n, dx, 0.3 * w, xc=0.9 * w)
                                + _gauss(n, dx, 0.3 * w, xc=-0.9 * w)
                                ).astype(complex),
                    'decentred': _gauss(n, dx, w, xc=0.8 * w).astype(complex),
                    'resid+': _gauss(n, dx, w) * np.exp(1j * k * r2 / 0.8),
                    'resid-': _gauss(n, dx, w) * np.exp(-1j * k * r2 / 0.8),
                }
                for kind, env in kinds.items():
                    new = C._default_focus_standoff(env, -rmag, rmag, lam, dx)
                    old = pre_fix(env, dx)
                    if new < old:
                        shorter.append((na, ext, kind, old, new))
                    if kind in ('flat', 'noisy', 'tophat', 'twolobe',
                                'decentred') and new != old:
                        moved_real.append((na, ext, kind, old, new))
                    if new > old:
                        lengthened += 1
        assert shorter == [], shorter
        assert moved_real == [], moved_real
        assert lengthened > 0, 'the beam term must be live on curved envelopes'


# ===========================================================================
# C1 -- against an analytic Gaussian-ABCD oracle on a NEW fixture
# ===========================================================================
class TestVerifyC1AgainstTheAnalyticFocus:

    LAM = 0.85e-6                       # WP and audit both used 1.31 um

    def _fixture(self, n=1024, w_in=0.6e-3, na=0.08, ext=4.0):
        k = 2.0 * np.pi / self.LAM
        r0 = -w_in / na
        dx = 2.0 * ext * w_in / n
        g = _grid(n, dx)
        r2 = g[None, :] ** 2 + g[:, None] ** 2
        e = np.exp(-r2 / w_in ** 2) * np.exp(1j * k * r2 / (2.0 * r0))
        return e, r0, dx, w_in

    @pytest.mark.parametrize('frac,peak_bar,rel_bar,pre_peak', [
        # (carrier/truth, post-fix peak bar, post-fix relL2 bar,
        #  MEASURED pre-fix peak on this fixture)
        (1.00, 0.99, 0.010, 0.999799),
        (0.97, 0.90, 0.080, 0.216767),
        (0.93, 0.85, 0.150, 0.024932),
    ])
    def test_the_readout_matches_the_analytic_focus_at_a_new_na(
            self, frac, peak_bar, rel_bar, pre_peak):
        """ORACLE: ``_abcd_field`` -- the analytic Gaussian ABCD focal field,
        amplitude, curvature and Gouy phase, written in this file.  Fixture:
        lambda = 0.85 um, NA = 0.08, w = 0.6 mm, N = 1024, ext = 4 -- none of
        which the WP or the audit used (they used 1.31 um at NA 0.05).

        MEASURED here, post-fix -> pre-fix (the pre-fix arm is the resolver
        with its beam term reverted in process, its only change):

            R/R0   peak vs the oracle      piston-free relL2
            1.00   0.999799 -> 0.999799    2.41e-03 -> 2.41e-03
            0.99   0.995468 -> 0.891233    1.57e-02 -> 6.08e-02
            0.97   0.973251 -> 0.216767    4.27e-02 -> 5.54e-01
            0.93   0.919521 -> 0.024932    8.54e-02 -> 8.37e-01
            0.90   0.885027 -> 0.002060    1.10e-01 -> 1.00e+00

        BARS are per row because the post-fix residual GROWS with the mismatch
        (it is the hand-off model error of the longer leg the fix resolves, not
        clipping): each peak bar sits 1.01x-1.08x under its measured post-fix
        value and 4.3x-37x over the measured pre-fix one, and each relL2 bar
        1.4x-1.9x over the post-fix value and 3.9x-13x under the pre-fix one.
        The higher NA is why the post-fix numbers here are worse than the WP's
        0.9995..0.9852 at NA 0.05 -- the leg error scales as NA^3."""
        lam = self.LAM
        e_phys, r0, dx, w_in = self._fixture()
        z = -r0
        _, w0 = _abcd_field(np.zeros(1), w_in, r0, z, lam)
        n_out, dx_out = 96, w0 / 8.0
        xo = _grid(n_out, dx_out)
        truth, _ = _abcd_field(xo, w_in, r0, z, lam)

        r = frac * r0
        env = C.carrier_referenced_envelope(e_phys, r, lam, dx)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            f = C.carrier_referenced_focus_readout(
                env, r, z, lam, dx, dx_out=dx_out, N_out=n_out,
                on_replica='ignore')
        pist = np.angle(np.vdot(truth, f))
        rel = float(np.linalg.norm(f * np.exp(-1j * pist) - truth)
                    / np.linalg.norm(truth))
        peak = float((np.abs(f) ** 2).max() / (np.abs(truth) ** 2).max())
        assert peak > peak_bar, (frac, peak)
        assert peak < 1.02, (frac, peak)        # and no manufactured energy
        assert rel < rel_bar, (frac, rel)
        assert not [x for x in w if 'co-moving' in str(x.message)]
        # FAIL-BEFORE on the same fixture: the resolver's beam term reverted
        real = C._beam_containment_standoff
        try:
            C._beam_containment_standoff = lambda *a, **kw: 0.0
            s_pre = C._default_focus_standoff(env, r, z, lam, dx)
        finally:
            C._beam_containment_standoff = real
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            f_pre = C.carrier_referenced_focus_readout(
                env, r, z, lam, dx, dx_out=dx_out, N_out=n_out,
                standoff=s_pre, on_replica='ignore',
                on_focus_containment='ignore')
        got_pre = float((np.abs(f_pre) ** 2).max()
                        / (np.abs(truth) ** 2).max())
        assert got_pre == pytest.approx(pre_peak, rel=5e-2), (frac, got_pre)

    @pytest.mark.parametrize('frac,peak_bar,pre_peak', [
        # (carrier/truth, post-fix peak bar, MEASURED peak on the pre-fix leg)
        (0.99, 0.97, 0.912560),
        (0.97, 0.92, 0.300483),
        (0.93, 0.85, 0.025460),
    ])
    def test_a_narrow_grid_resolves_and_the_guard_sees_the_pre_fix_leg(
            self, frac, peak_bar, pre_peak):
        """VERIFY-A6 OI-1, implemented 2026-09-12.

        Before: the beam term could only ask for ``_FOCUS_STANDOFF_MARGIN``
        = 3.2 beam radii, so on a grid holding fewer than that at the INPUT
        plane (``gamma = half^2 - (M w)^2 < 0``, i.e. ext < 3.2 -- the module's
        own small-extent branch V1/V2) it had no margin to resolve against and
        returned 0.0; and the guard's floor was a bare 1.0 radius, which a
        landing at 1.627 clears.  Net: at ext = 3.0 a 1 % carrier mismatch cost
        8.7 % of the focal peak with ZERO warnings.

        After: both halves target ``_achievable_focus_margin`` -- the same
        ``m_req = min(M, sat*ext)`` the carrier-referenced law already resolves
        against -- so the resolver lengthens, and the guard's floor is the
        worse of 1.0 and ``_FOCUS_READOUT_CONTAINMENT_FRAC`` of that target.

        MEASURED here (lambda 0.85 um, NA 0.08, ext 3.0, target margin 2.598):

            R/R0   leg (um)  post -> pre   containment   peak vs the oracle
            1.00     94.76 ->   94.76       2.745         0.999324 (control)
            0.99    534.48 ->  167.37    2.598 -> 1.627   0.990892 -> 0.912560
            0.97   1408.84 ->  312.71    2.598 -> 0.984   0.954704 -> 0.300483
            0.93   2698.45 ->  603.87    2.598 -> 0.865   0.883644 -> 0.025460

        BARS: each peak bar sits 1.02x under its measured post-fix value and
        1.06x / 3.1x / 35x over the measured pre-fix one; the containment claim
        is equality with the target to 1e-3 (the resolver solves for it); and
        the pre-fix leg must be REFUSED by the default disposition, which is
        the two-sided half of the bar."""
        lam = self.LAM
        e_phys, r0, dx, w_in = self._fixture(ext=3.0)
        z = -r0
        _, w0 = _abcd_field(np.zeros(1), w_in, r0, z, lam)
        n_out, dx_out = 96, w0 / 8.0
        truth, _ = _abcd_field(_grid(n_out, dx_out), w_in, r0, z, lam)
        r = frac * r0
        env = C.carrier_referenced_envelope(e_phys, r, lam, dx)

        pd = {}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            f = C.carrier_referenced_focus_readout(
                env, r, z, lam, dx, dx_out=dx_out, N_out=n_out,
                on_replica='ignore', _period_out=pd)
        peak = float((np.abs(f) ** 2).max() / (np.abs(truth) ** 2).max())
        assert peak > peak_bar, (frac, peak)
        assert not [x for x in w if 'co-moving' in str(x.message)]
        # the grid cannot give 3.2, and the resolver knows it
        assert pd['containment_target'] < C._FOCUS_STANDOFF_MARGIN
        assert pd['containment'] == pytest.approx(pd['containment_target'],
                                                  rel=1e-3), pd

        # FAIL-BEFORE: the pre-fix leg, and the guard must now refuse it
        real = C._beam_containment_standoff
        try:
            C._beam_containment_standoff = lambda *a, **kw: 0.0
            s_pre = C._default_focus_standoff(env, r, z, lam, dx)
        finally:
            C._beam_containment_standoff = real
        assert s_pre < pd['standoff'], (s_pre, pd['standoff'])
        with pytest.raises(RuntimeError, match='does not fit the co-moving'):
            C.carrier_referenced_focus_readout(
                env, r, z, lam, dx, dx_out=dx_out, N_out=n_out,
                standoff=s_pre, on_replica='ignore')
        pd2 = {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            f2 = C.carrier_referenced_focus_readout(
                env, r, z, lam, dx, dx_out=dx_out, N_out=n_out,
                standoff=s_pre, on_replica='ignore',
                on_focus_containment='ignore', _period_out=pd2)
        got_pre = float((np.abs(f2) ** 2).max() / (np.abs(truth) ** 2).max())
        assert got_pre == pytest.approx(pre_peak, rel=5e-2), (frac, got_pre)

    def test_the_contained_control_on_the_same_narrow_grid_is_untouched(self):
        """The other side of the bar: on the SAME ext = 3.0 grid the MATCHED
        carrier is contained (2.745 radii against a 2.598 target), so the leg
        must not move and the guard must stay silent.  A guard that fired here
        would be refusing the module's own documented narrow-grid branch."""
        lam = self.LAM
        e_phys, r0, dx, w_in = self._fixture(ext=3.0)
        z = -r0
        env = C.carrier_referenced_envelope(e_phys, r0, lam, dx)
        real = C._beam_containment_standoff
        try:
            C._beam_containment_standoff = lambda *a, **kw: 0.0
            pre = C._default_focus_standoff(env, r0, z, lam, dx)
        finally:
            C._beam_containment_standoff = real
        assert C._default_focus_standoff(env, r0, z, lam, dx) == pre
        pd = {}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            C.carrier_referenced_focus_readout(
                env, r0, z, lam, dx, dx_out=1e-6, N_out=48,
                on_replica='ignore', _period_out=pd)
        assert pd['containment'] > pd['containment_target'], pd
        assert not [x for x in w if 'co-moving' in str(x.message)]

    def test_the_achievable_margin_leaves_every_wide_grid_alone(self):
        """``_achievable_focus_margin`` must be the IDENTITY above
        ``M/sat`` = 3.695 beam radii, or OI-1 would have moved the validated
        wide-grid surface.  Asserted as an exact equality on a ladder that
        straddles the knee, plus the saturation law below it."""
        m = C._FOCUS_STANDOFF_MARGIN
        f_cap = np.sqrt(max(C._FOCUS_STANDOFF_WAIST_GROWTH ** 2 - 1.0, 0.0))
        sat = f_cap / np.sqrt(1.0 + f_cap * f_cap)
        w = 1e-3
        for ext in (3.70, 4.0, 6.0, 10.0, 100.0):
            assert C._achievable_focus_margin(ext * w, w) == m, ext
        for ext in (0.5, 1.2, 2.0, 3.0, 3.69):
            got = C._achievable_focus_margin(ext * w, w)
            assert got == pytest.approx(sat * ext, rel=1e-15), ext
            assert got < m, ext
        # degenerate inputs fall back to the constant rather than dividing
        assert C._achievable_focus_margin(0.0, w) == m
        assert C._achievable_focus_margin(w, 0.0) == m

    def test_the_relative_floor_is_scoped_to_grids_below_the_knee(self):
        """The false positive this scoping exists to prevent (found by WP-A3
        on ``test_niche_d1_tilted_carrier.py``, 2026-09-12).

        A fraction-of-target floor applied on a WIDE grid is not a clipping
        test.  On a real relay the envelope carries a broad low-level skirt, so
        the measured ``sqrt(2<r^2>)`` reads the WINDOW rather than the core:
        the niche-D1 tilted relay's worst readout measures 1.033 radii against
        a MODELLED 3.506 on an input grid of 4.64 beam radii, and that landing
        satisfies the fixture's own ray-trace, diffraction-limit and
        energy-conservation oracles (33/33 with the historical floor).  An
        unscoped ``0.9 * 3.2`` = 2.88 floor refused 7 of those tests.

        So the relative arm governs ONLY ``m_target < _FOCUS_STANDOFF_MARGIN``
        -- the grids where the resolver aims lower by design and the absolute
        floor is therefore blind.  Asserted on both sides of the knee
        (``M/sat`` = 3.695 beam radii) as an exact statement about the floor,
        not about any one fixture."""
        m = C._FOCUS_STANDOFF_MARGIN
        frac = C._FOCUS_READOUT_CONTAINMENT_FRAC
        mn = C._FOCUS_READOUT_CONTAINMENT_MIN
        w = 1e-3

        def floor_at(ext):
            t = C._achievable_focus_margin(ext * w, w)
            return (mn if t >= m else max(mn, frac * t)), t

        for ext in (3.70, 4.0, 4.64, 6.0, 100.0):     # wide: historical floor
            f, t = floor_at(ext)
            assert t == m and f == mn, (ext, f, t)
        for ext in (1.2, 2.0, 3.0, 3.69):             # narrow: relative arm
            f, t = floor_at(ext)
            assert t < m, (ext, t)
            assert f == max(mn, frac * t), (ext, f, t)
        # ... and it BITES (rises above the absolute floor) over the band where
        # the resolver's target is worth a fraction of: frac*t > 1 needs
        # ext > 1/(frac*sat) = 1.283 beam radii.
        for ext in (1.5, 2.0, 3.0, 3.69):
            f, t = floor_at(ext)
            assert f == pytest.approx(frac * t, rel=1e-15), (ext, f, t)
            assert f > mn, (ext, f, t)
        assert floor_at(1.2)[0] == mn                  # below that, 1.0 rules
        # ... and the d1 reading itself is accepted on its own grid extent
        f, _ = floor_at(4.64)
        assert 1.033 > f, f

    def test_the_sixty_cell_matrix_is_untouched_and_silent(self):
        """The acceptance condition for OI-1: the shipped defaults must not
        move on a single flat-envelope calibration cell, and the stricter
        floor must not fire on one either.

        MEASURED over the resolver's own 6 NA x 10 extent matrix: 0 cells whose
        resolved leg differs from the pre-fix one (the beam term still
        short-circuits on ``inv_env == 0.0``, which a real envelope gives
        exactly), 0 guard firings, worst achieved/target containment ratio
        **0.9969** against the 0.9 bar -- 1.108x of clearance -- and worst
        absolute containment 1.2566 against the 1.0 floor."""
        lam, rmag, n = 1.31e-6, 20e-3, 512
        worst_ratio, worst_abs, moved = np.inf, np.inf, []
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            for na in (0.03, 0.05, 0.10, 0.15, 0.278, 0.35):
                for ext in (1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 6.0, 10.0):
                    wb = na * rmag
                    dx = 2.0 * ext * wb / n
                    env = _gauss(n, dx, wb).astype(complex)
                    real = C._beam_containment_standoff
                    try:
                        C._beam_containment_standoff = lambda *a, **kw: 0.0
                        pre = C._default_focus_standoff(env, -rmag, rmag,
                                                        lam, dx)
                    finally:
                        C._beam_containment_standoff = real
                    if C._default_focus_standoff(env, -rmag, rmag,
                                                 lam, dx) != pre:
                        moved.append((na, ext))
                    pd = {}
                    C.carrier_referenced_focus_readout(
                        env, -rmag, rmag, lam, dx,
                        dx_out=(lam * rmag / (np.pi * wb)) / 8.0, N_out=32,
                        on_replica='ignore', _period_out=pd)
                    worst_ratio = min(worst_ratio, pd['containment']
                                      / pd['containment_target'])
                    worst_abs = min(worst_abs, pd['containment'])
        assert moved == [], moved
        assert not [x for x in w if 'co-moving' in str(x.message)]
        assert worst_ratio > C._FOCUS_READOUT_CONTAINMENT_FRAC, worst_ratio
        assert worst_abs > C._FOCUS_READOUT_CONTAINMENT_MIN, worst_abs


# ===========================================================================
# C2 -- the fit centre and the tilt projection
# ===========================================================================
class TestVerifyC2:

    LAM = 1.31e-6
    N, DX, W = 512, 2e-6, 100e-6

    def _field(self, r=np.inf, tilt=0.0, x0=0.0):
        k = 2.0 * np.pi / self.LAM
        g = _grid(self.N, self.DX)
        xx, yy = g[None, :] - x0, g[:, None]
        r2 = xx * xx + yy * yy
        ph = (0.0 if not np.isfinite(r) else k * r2 / (2.0 * r)) + k * tilt * xx
        return np.exp(-r2 / self.W ** 2) * np.exp(1j * ph)

    @pytest.mark.parametrize('est', ['gradient', 'increment'])
    @pytest.mark.parametrize('x0f', [0.0, 0.5, 2.0])
    def test_origin_is_the_hand_written_pre_fix_estimator_bit_for_bit(
            self, est, x0f):
        """WP-A6 asserts byte-identity of the default branch against its OWN
        ``centre='origin'`` path.  That is circular; here the comparison is
        against ``_prefix_fit_inv``, a transcription of the pre-C2 source, so a
        regrouping inside the new branch would show.  No tolerance: equality."""
        for r in (50e-3, -20e-3):
            e = self._field(r=r, x0=x0f * self.W)
            assert (C._fit_carrier_inv(e, self.LAM, self.DX, self.DX,
                                       estimator=est)
                    == _prefix_fit_inv(e, self.LAM, self.DX, self.DX,
                                       estimator=est)), (est, x0f, r)

    @pytest.mark.parametrize('x0f', [0.5, 1.0, 2.0])
    @pytest.mark.parametrize('tilt', [0.0, 0.002, 0.02])
    def test_tilt_and_curvature_together_on_a_decentred_beam(self, x0f, tilt):
        """A fixture the WP did not build: the SAME wavefront carries a true
        parabola about its own centre AND a uniform tilt.  Truth R = 50 mm by
        construction, independent of both the decentre and the tilt.

        BAR 1e-6 relative: 'increment' is EXACT on a parabola (the midpoint
        increment is ``k x_mid/R`` term for term), so the only error is
        float64 round-off in the moment -- measured 2.2e-16 at every one of the
        nine cells.  The bar is 10 decades over that.

        The 'origin' arm is scored against the ANALYTIC pre-fix reading
        ``1/R_fit = [(1/R)(w^2/2) + L x0] / (x0^2 + w^2/2)`` (the moment with
        the ``x0 <x>`` cross term missing and the tilt unprojected), not merely
        asserted to be wrong -- at ``x0 = w, L = 0.002`` the two biases happen
        to CANCEL and 'origin' reads the right answer, which a "must differ"
        assertion would call a failure.  BAR 2e-3 relative: the fixture's own
        truncation at 2.56 waists leaves the second moment 1e-4 short of the
        analytic ``w^2/2``; measured worst 3.4e-4."""
        x0 = x0f * self.W
        e = self._field(r=50e-3, tilt=tilt, x0=x0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            auto = C.carrier_referenced_fit_radius(
                e, self.LAM, self.DX, estimator='increment',
                on_aliased='silent')
            orig = C.carrier_referenced_fit_radius(
                e, self.LAM, self.DX, estimator='increment',
                on_aliased='silent', centre='origin')
            expl = C.carrier_referenced_fit_radius(
                e, self.LAM, self.DX, estimator='increment',
                on_aliased='silent', centre=(x0, 0.0))
        assert auto / 50e-3 == pytest.approx(1.0, rel=1e-6), (x0f, tilt, auto)
        assert expl / 50e-3 == pytest.approx(1.0, rel=1e-6), (x0f, tilt, expl)
        s2 = 0.5 * self.W ** 2                      # 2 sigma^2 for exp(-r^2/w^2)
        inv_pre = ((1.0 / 50e-3) * s2 + tilt * x0) / (x0 * x0 + s2)
        assert 1.0 / orig == pytest.approx(inv_pre, rel=2e-3), (x0f, tilt, orig)

    @pytest.mark.parametrize('kind', ['tophat', 'clipped', 'twolobe',
                                      'triangular'])
    def test_non_gaussian_envelopes_read_their_own_radius(self, kind):
        """``centre='auto'`` is an INTENSITY CENTROID, so it has to work on
        amplitudes that are not Gaussian.  Truth R = 50 mm by construction.

        BAR 1e-3 relative: the increment estimator is exact on a parabola for
        ANY real amplitude weight (the weight cancels between numerator and
        denominator only in the limit, but the midpoint increment is exact
        pointwise), measured 2e-16..3e-4 across these four shapes; 'origin'
        reads 2.25x..3.05x on the same fields."""
        k = 2.0 * np.pi / self.LAM
        g = _grid(self.N, self.DX)
        x0 = 1.0 * self.W
        xx, yy = g[None, :] - x0, g[:, None]
        r2 = xx * xx + yy * yy
        amp = {
            'tophat': (r2 <= self.W ** 2).astype(float),
            'clipped': np.exp(-r2 / self.W ** 2) * (g[None, :] > -0.2 * self.W),
            'twolobe': (np.exp(-((xx - 0.8 * self.W) ** 2 + yy ** 2)
                               / (0.3 * self.W) ** 2)
                        + np.exp(-((xx + 0.8 * self.W) ** 2 + yy ** 2)
                                 / (0.3 * self.W) ** 2)),
            'triangular': np.clip(1.0 - np.sqrt(r2) / (2 * self.W), 0.0, None),
        }[kind]
        e = amp * np.exp(1j * k * r2 / (2.0 * 50e-3))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            auto = C.carrier_referenced_fit_radius(
                e, self.LAM, self.DX, estimator='increment',
                on_aliased='silent')
            orig = C.carrier_referenced_fit_radius(
                e, self.LAM, self.DX, estimator='increment',
                on_aliased='silent', centre='origin')
        assert auto / 50e-3 == pytest.approx(1.0, rel=1e-3), (kind, auto)
        assert abs(orig / 50e-3 - 1.0) > 1.0, (kind, orig)

    @pytest.mark.parametrize('x0_px', [0.0, 0.2, 0.49, 0.51, 1.0, 25.0])
    def test_a_sub_pixel_decentre_no_longer_hides_a_pure_tilt(self, x0_px):
        """VERIFY-A6 OI-2, implemented 2026-09-12.

        The tilt projection used to be gated on ``centre != (0, 0)``, and
        ``_envelope_amp_centroid`` SNAPS any decentre under half a pixel to
        exactly ``(0, 0)``.  A beam decentred by a fifth of a pixel while
        carrying a tilt therefore got neither the centring nor the projection
        and read a finite radius, with a discontinuity at the snap.  Measured
        before, for ``L = 0.02`` on a 100 um waist at dx = 2 um, truth ``inf``:

            x0/dx   0.00        0.20      0.49      0.51        1.00
            'auto'  -4.9e+14 m  0.625 m   0.255 m   2.5e+14 m   8.7e+13 m

        i.e. a 0.26 m radius at the snap boundary.  ``'auto'`` now projects
        whatever the snap did, and ``'origin'`` keeps the unprojected fit as
        the byte-identity escape hatch (and still reads the finite radius --
        asserted below, so the fix cannot be mistaken for the fixture being
        benign).

        BAR |R_fit| > 1e6 m, the same absolute form the WP's own tilt test
        uses: the truth is infinite, so a ratio is not available; 1e6 m is
        5e7 beam radii of radius, i.e. flat to 2e-8 waves of sag across the
        beam.  Measured post-fix worst 8.7e+13 m across these six cells, 8
        decades over the bar; the pre-fix readings are 6 decades UNDER it."""
        x0 = x0_px * self.DX
        e = self._field(r=np.inf, tilt=0.02, x0=x0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            auto = C.carrier_referenced_fit_radius(
                e, self.LAM, self.DX, estimator='increment',
                on_aliased='silent')
            orig = C.carrier_referenced_fit_radius(
                e, self.LAM, self.DX, estimator='increment',
                on_aliased='silent', centre='origin')
        assert abs(auto) > 1e6, (x0_px, auto)
        if x0_px > 0.0:
            # the fixture really does carry the cross term the projection
            # removes -- ``'origin'`` still reads it, analytically
            # ``1/R = L x0/(x0^2 + w^2/2)``
            s2 = 0.5 * self.W ** 2
            assert 1.0 / orig == pytest.approx(0.02 * x0 / (x0 * x0 + s2),
                                               rel=2e-3), (x0_px, orig)

    def test_refit_carrier_on_a_decentred_beam_is_self_consistent(self):
        """WP-A6 records as a residual risk that
        ``carrier_referenced_aperture(refit_carrier=True)`` keeps its internal
        fit about the grid ORIGIN.  Measured here on a decentred aperture, on
        BOTH a flat envelope and one carrying residual curvature about its own
        decentred centre:

        * the PHYSICAL field is preserved to <= 2.3e-16 relL2 at every decentre
          out to 2 waists -- ``_rereference`` applies exactly the screen the
          radius change asks for, so the refit can never produce a wrong field,
          only a different carrier/envelope SPLIT;
        * a flat envelope refits to the input radius to 12 digits and leaves
          exactly zero residual (nothing to mis-attribute);
        * a curved one leaves zero residual ABOUT THE ORIGIN (the fit and the
          screen are the matched pair the docstring describes) and up to
          1.78 /m about the beam's own centre at 2 waists -- i.e. the split is
          self-consistent but not the beam's own.

        So it is a documented limitation, not a latent C2.  BAR 1e-14 on the
        field invariance (measured 2.2e-16, the complex-multiply round-off of a
        unit-modulus screen) and EQUALITY on the origin-referred residual."""
        k = 2.0 * np.pi / self.LAM
        g = _grid(self.N, self.DX)
        for x0f in (0.0, 0.5, 1.0, 2.0):
            for curved in (False, True):
                xx, yy = g[None, :] - x0f * self.W, g[:, None]
                r2 = xx * xx + yy * yy
                env = np.exp(-r2 / self.W ** 2).astype(complex)
                if curved:
                    env = env * np.exp(1j * k * r2 / (2.0 * 0.5))
                mask = (r2 <= (2.2 * self.W) ** 2)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    ref = C.carrier_referenced_reconstruct(
                        env, 50e-3, self.LAM, self.DX) * mask
                    (e2, r2o, dx2) = C.carrier_referenced_aperture(
                        env, 50e-3, self.LAM, self.DX, mask=mask,
                        refit_carrier=True, refit_estimator='increment')
                    got = C.carrier_referenced_reconstruct(
                        e2, r2o, self.LAM, dx2)
                    left = C._fit_carrier_inv(e2, self.LAM, dx2, dx2,
                                              estimator='increment')
                rel = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
                assert rel < 1e-14, (x0f, curved, rel)
                assert abs(left) < 1e-13, (x0f, curved, left)
                if not curved:
                    assert r2o == pytest.approx(50e-3, rel=1e-12), (x0f, r2o)


# ===========================================================================
# C4 -- the separable bound, stated relative to the argument
# ===========================================================================
class TestVerifyC4SeparableBound:
    """DEFECT FOUND BY THE VERIFIER (documentation / bar, not arithmetic).

    WP-A6 barred the separable regrouping at a FIXED ``1e-11`` rad and
    justified it as "the float64 representation noise of the ``k r^2/2R`` ~
    1e5-1e6 rad arguments these screens carry", with the measured difference
    "three orders BELOW" it.  Re-measured, the difference is
    ``1.0-1.6 x eps |arg|`` -- i.e. AT that floor, not three decades under it --
    and at the upper end of the stated operating range (5.2e5 rad) it measures
    1.75e-10, which the 1e-11 bar would FAIL.  The arithmetic is fine; the bar
    and its derivation were not scale-free.  Restated here relative to the
    argument, and the module comment corrected."""

    LAM = 1.31e-6

    @pytest.mark.parametrize('n,dx,R,arg_rad', [
        # (N, dx, R, max|k r^2/2R| measured on the fixture)
        (256, 2e-6, 50e-3, 6.2866e+00),
        (2048, 2e-6, 50e-3, 4.0234e+02),
        (2048, 8e-6, 50e-3, 6.4375e+03),
        (1024, 16e-6, 10e-3, 3.2188e+04),
    ])
    def test_the_separable_screen_sits_on_the_argument_s_own_floor(
            self, n, dx, R, arg_rad):
        """ORACLE: the whole-grid ``x^2[None,:] + y^2[:,None]`` expression with
        a single ``np.exp``, written here.

        BAR ``4 eps |arg|``: the identity is exact in exact arithmetic, so the
        only error is that the separable form rounds the two half-arguments
        apart -- which cannot exceed a small multiple of the ulp of the total.
        Measured ratio ``|diff| / (eps |arg|)`` = 1.02 / 1.27 / 1.27 / 1.53 on
        these four cells (and 1.53 at 5.15e5 rad, the largest argument the
        module's own docstring claims), so the bar has 2.5x of headroom;
        a genuine regrouping blunder (a wrong sign, a dropped cross term) is
        O(|arg|), 15 decades up."""
        k = 2.0 * np.pi / self.LAM
        x = _grid(n, dx)
        r2 = x[None, :] ** 2 + x[:, None] ** 2
        arg = k * r2 / (2.0 * R)
        assert float(np.abs(arg).max()) == pytest.approx(arg_rad, rel=1e-3)
        orc = np.exp(1j * arg)
        got = C._radial_carrier_phase((n, n), dx, dx, self.LAM, R, +1)
        d = float(np.abs(got - orc).max())
        assert d <= 4.0 * EPS * float(np.abs(arg).max()), (n, dx, R, d)

    def test_a_fixed_absolute_bar_would_not_have_held(self):
        """The reason the bar above is relative.  At the upper end of the
        argument range the module's own docstring quotes (``k r^2/2R`` up to
        ~1e6 rad) the regrouping difference measures 1.75e-10 rad, which is
        17x OVER the 1e-11 the WP's test uses.  Asserted as an inequality on
        the running build, not as a remembered number, so it tracks."""
        k = 2.0 * np.pi / self.LAM
        n, dx, R = 2048, 16e-6, 10e-3
        x = _grid(n, dx)
        arg = k * (x[None, :] ** 2 + x[:, None] ** 2) / (2.0 * R)
        am = float(np.abs(arg).max())
        got = C._radial_carrier_phase((n, n), dx, dx, self.LAM, R, +1)
        d = float(np.abs(got - np.exp(1j * arg)).max())
        assert am > 1e5, am                       # inside the quoted range
        assert d > 1e-11, (am, d)                 # the fixed bar is exceeded
        assert d <= 4.0 * EPS * am, (am, d)       # the derived one holds

    def test_the_flag_off_build_is_the_hand_written_whole_grid_one(self):
        """``_SEPARABLE_CARRIER_PHASE = False`` must restore the historical
        build against an oracle written HERE (the WP compares against its own
        fallback).  No tolerance: raw-byte equality, at both parities."""
        k = 2.0 * np.pi / self.LAM
        for n, cen in ((257, (0.0, 0.0)), (256, (7e-6, -3e-6))):
            dx, R = 2e-6, 50e-3
            x = _grid(n, dx) - cen[0]
            y = _grid(n, dx) - cen[1]
            Y, X = np.meshgrid(y, x, indexing='ij')
            hand = np.exp(1j * k * (X * X + Y * Y) / (2.0 * R))
            old = C._SEPARABLE_CARRIER_PHASE
            try:
                C._SEPARABLE_CARRIER_PHASE = False
                got = C._radial_carrier_phase((n, n), dx, dx, self.LAM, R, +1,
                                              centre=cen)
            finally:
                C._SEPARABLE_CARRIER_PHASE = old
            assert np.array_equal(got.view(np.float64),
                                  hand.view(np.float64)), (n, cen)


# ===========================================================================
# C5 -- the P3 cluster, re-checked against independent oracles
# ===========================================================================
class TestVerifyC5:

    LAM = 0.633e-6                      # not the 1.31 um the WP used

    @pytest.mark.parametrize('n', [64, 65, 127, 128, 129])
    @pytest.mark.parametrize('axis', [0, 1])
    def test_asm_axis_against_an_fftfreq_oracle(self, n, axis):
        """WP-A6's own test builds its oracle on ``(arange(n) - n//2)/(n d)``,
        which is the expression under test restated; this one is built on
        ``np.fft.fftfreq`` itself, which is what the module claims equivalence
        to, and it adds N = 128/129 to the WP's 64/65/127.

        BAR 1e-12 relL2: the only legitimate difference is the 2-4 ulp between
        ``fftfreq``'s multiply-by-reciprocal and the module's divide, which
        enters the transfer function as ``z d(kz)`` -- measured 0.0 at
        N = 64/65 and <= 5.9e-14 at 127/128/129, identical with the band limit
        ON and OFF (so it is the frequency axis, not the mask).  A half-bin
        mask misregistration -- the defect -- is O(1) on the two edge bins,
        12 decades up."""
        d, z = 2e-6, 4.0e-4
        k = 2.0 * np.pi / self.LAM
        rng = np.random.default_rng(3)
        e = (rng.standard_normal((n, n))
             + 1j * rng.standard_normal((n, n)))
        f = np.fft.fftfreq(n, d)
        kz_sq = k * k - (2.0 * np.pi * f) ** 2
        kz = np.where(kz_sq > 0, np.sqrt(np.maximum(kz_sq, 0.0)), 0.0)
        for bandlimit in (True, False):
            h = np.where(kz_sq > 0, np.exp(1j * z * kz), 0.0)
            if bandlimit:
                h = np.where(np.abs(f) < (n * d) / (2.0 * self.LAM * abs(z)),
                             h, 0.0)
            sh = [1, 1]
            sh[axis] = n
            orc = np.fft.ifft(np.fft.fft(e, axis=axis) * h.reshape(sh),
                              axis=axis)
            got = C._asm_axis(e, z, self.LAM, d, axis, bandlimit=bandlimit)
            rel = float(np.linalg.norm(got - orc) / np.linalg.norm(orc))
            assert rel < 1e-12, (n, axis, bandlimit, rel)

    @pytest.mark.parametrize('n', [7, 64, 65, 127, 129, 1025])
    def test_the_frequency_builders_are_fftfreq_to_a_few_ulp(self, n):
        """The surviving builders' docstring claimed EXACT equality with
        ``fftfreq`` at both parities; measured it is exact only where
        ``1/(N d)`` is representable and 2-4 ulp otherwise (N = 7/65/127/129/
        1025 here).  The docstring is corrected; the PROPERTY that matters --
        the ``N // 2`` offset -- is asserted directly below.

        BAR 4 ulp: measured worst 3 ulp on the square builder and 2 on the
        linear one.  A ``- N/2`` offset (the deleted twin's defect) is half a
        BIN, i.e. ~1e12 in these units: 28 decades up."""
        d = 2e-6
        a = np.fft.ifftshift(C._freq_sq_1d_bld(n, d, np))
        b = (2.0 * np.pi * np.fft.fftfreq(n, d)) ** 2
        np.testing.assert_allclose(a, b, rtol=4 * EPS, atol=0.0)
        # ... and the OFFSET itself, stated structurally so it cannot be
        # satisfied by restating the implementation: the DC bin of a centred
        # axis sits at index ``n // 2`` and is exactly zero, and the axis is
        # the integer ladder about it.  A ``- n/2`` offset (the deleted twin's
        # defect) puts the zero half a bin away at odd n, so no element is 0.
        lin = C._freq_1d_bld(n, d, np)
        assert lin[n // 2] == 0.0, (n, lin[n // 2])
        assert int(np.count_nonzero(lin == 0.0)) == 1
        step = 2.0 * np.pi / (n * d)
        # the spacing is a DIFFERENCE of numbers up to ``pi/d`` = 1.57e6, so
        # its own floor is ``eps * max|lin|`` = 3.5e-10, not ``eps * step``;
        # bar at 4x that (measured worst 2.5e-10 at n = 1025).
        np.testing.assert_allclose(
            np.diff(lin), np.full(n - 1, step), rtol=0.0,
            atol=4.0 * EPS * float(np.abs(lin).max()))
        assert not hasattr(C, '_freq_sq_1d')

    def test_the_readout_really_forwards_the_kernel_and_the_tilt(self):
        """WP-A6 pins this comparatively (the argument must change the answer).
        Here the inner call is SPIED, so the claim is that the values arrive at
        ``propagate_carrier_referenced`` -- which is what C5 is about -- and not
        merely that something downstream changed."""
        lam = self.LAM
        n, dx, w, rmag = 256, 8e-6, 200e-6, 20e-3
        env = _gauss(n, dx, w).astype(complex)
        seen = []
        real = C.propagate_carrier_referenced

        def spy(*a, **kw):
            seen.append((kw.get('gap_kernel'), kw.get('tilt')))
            return real(*a, **kw)

        try:
            C.propagate_carrier_referenced = spy
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                C.carrier_referenced_focus_readout(
                    env, -rmag, rmag, lam, dx,
                    dx_out=(lam * rmag / (np.pi * w)) / 8.0, N_out=32,
                    on_replica='ignore', gap_kernel='fresnel',
                    tilt=(0.03, -0.02))
        finally:
            C.propagate_carrier_referenced = real
        assert seen == [('fresnel', (0.03, -0.02))], seen

    def test_the_sphere_parabola_conversion_dy_is_wired_to_every_caller(self):
        """VERIFY-A6 OI-3, implemented 2026-09-12.  ``dy=`` was added by WP-A6
        but no call site forwarded a pitch, so a ``dy != dx`` chain would still
        have converted its y axis on ``dx``.  All seven sites now pass one --
        ``dx_fine`` on the two retrace sites, and the chain's ``cur_dy``, which
        is tracked beside ``cur_dx`` and picks up the y component of an
        astigmatic leg's ``(dx_x, dx_y)`` instead of discarding it.

        Asserted three ways: every call site passes a ``dy``; the helper is
        BIT-identical for ``dy == dx`` (so no square chain moves); and the
        ``dy != dx`` answer matches a numerically stable hand-built eikonal.

        BAR 1e-12 on the phase: the oracle uses ``S = sign(R) r^2/(|R| +
        sqrt(R^2+r^2))``, which avoids the catastrophic cancellation of the
        textbook ``sqrt(R^2+r^2) - |R|`` (that form's own error is
        ``eps |R| k`` = 4e-11 rad here), so the comparison is limited by the
        library, not by the oracle; measured 0.0."""
        import inspect
        import re
        src = inspect.getsource(C)
        # every call site of the helper must now hand it a y pitch
        calls = re.findall(
            r'_sphere_parab_conversion\((?:[^()]|\([^()]*\))*\)',
            src.split('def _sphere_parab_conversion')[0]
            + src.split('def _sphere_parab_conversion', 1)[1].split('\ndef ', 1)[1])
        assert len(calls) == 7, [c[:60] for c in calls]
        missing = [c[:70] for c in calls if 'dy=' not in c]
        assert missing == [], missing
        lam, k = self.LAM, 2.0 * np.pi / self.LAM
        sh, dx, r = (96, 128), 2e-6, -20e-3
        dy = 3.0 * dx
        a = C._sphere_parab_conversion(sh, dx, lam, r, +1)
        assert np.array_equal(
            a, C._sphere_parab_conversion(sh, dx, lam, r, +1, dy=dx))
        got = C._sphere_parab_conversion(sh, dx, lam, r, +1, dy=dy)
        x = _grid(sh[1], dx)
        y = _grid(sh[0], dy)
        r2 = x[None, :] ** 2 + y[:, None] ** 2
        s = np.sign(r) * (r2 / (np.sqrt(r2 + r * r) + abs(r)))
        orc = np.exp(1j * k * (s - r2 / (2.0 * r)))
        assert float(np.abs(got - orc).max()) < 1e-12
        assert float(np.abs(got - a).max()) > 1e-6      # dy really is used

    def test_an_astigmatic_leg_really_produces_the_two_pitches(self):
        """Where a ``dy != dx`` chain comes FROM, and what the conversion then
        sees.  ``propagate_carrier_referenced`` on an astigmatic carrier
        returns ``(dx_x, dx_y)`` -- the chain used to reduce that to its x
        component and throw the y pitch away, which is what made the
        conversion's ``dy`` unreachable.  Measured on this fixture:
        ``(4.160, 4.100) um``, 1.5 % apart.

        Pinned as the pair: the leg's two pitches are distinct, and the screen
        built from them matches a hand-built eikonal on the SAME two pitches
        while differing from the dx-only one by 100x the bar."""
        lam, k = self.LAM, 2.0 * np.pi / self.LAM
        n, dx = 128, 4e-6
        env = _gauss(n, dx, 80e-6).astype(complex)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cr = C.propagate_carrier_referenced(env, (50e-3, 80e-3), 2e-3,
                                                lam, dx, gap_kernel='fresnel')
        assert isinstance(cr.dx, tuple) and cr.dx[0] != cr.dx[1], cr.dx
        dxo, dyo = float(cr.dx[0]), float(cr.dx[1])
        assert abs(dyo / dxo - 1.0) > 1e-3, (dxo, dyo)
        r = -20e-3
        got = C._sphere_parab_conversion((n, n), dxo, lam, r, +1, dy=dyo)
        blind = C._sphere_parab_conversion((n, n), dxo, lam, r, +1)
        x = _grid(n, dxo)
        y = _grid(n, dyo)
        r2 = x[None, :] ** 2 + y[:, None] ** 2
        s = np.sign(r) * (r2 / (np.sqrt(r2 + r * r) + abs(r)))
        orc = np.exp(1j * k * (s - r2 / (2.0 * r)))
        assert float(np.abs(got - orc).max()) < 1e-12
        assert float(np.abs(got - blind).max()) > 1e-10

    def test_the_c64_sphere_builder_survives_its_own_del(self):
        """DEFECT FOUND BY THE VERIFIER (reported independently by WP-A5 as a
        ruff F821).  ``CarrierSpec.phasor_on``'s complex64 branch passed
        ``_phasor_rows`` a lambda that CLOSED OVER ``S`` and then executed
        ``del S`` two lines later.  It happened to work because the helper is
        eager, but the body referred to a name the function goes on to unbind:
        ruff reports ``F821 undefined name 'S'`` and the first lazy or deferred
        builder turns it into a ``NameError`` on the memory-campaign path.

        FAIL-BEFORE, executable: capture the builder ``phasor_on`` hands over
        and invoke it AFTER ``phasor_on`` has returned (and therefore after its
        ``del``).  With the closure it raises ``NameError``; with the default
        argument it returns the same rows.  The value is checked too, so the
        rebind cannot have changed the arithmetic."""
        grid = FieldGrid((64, 64), 2e-6)
        spec = CarrierSpec(R=-20e-3, tilt=(0.01, -0.005))
        captured = []
        real = CF._phasor_rows

        def capture(arg_rows, shape, dtype):
            captured.append(arg_rows)
            return real(arg_rows, shape, dtype)

        try:
            CF._phasor_rows = capture
            ph = spec.phasor_on(grid, self.LAM, dtype=np.complex64)
        finally:
            CF._phasor_rows = real
        assert ph.dtype == np.complex64
        assert len(captured) == 1, captured
        rows = captured[0](0, 4)                    # after phasor_on returned
        assert np.shape(rows) == (4, 64)
        # it is the SPHERE argument: purely imaginary, so its exponential is a
        # unit phasor (bar 1e-15 -- these are exact by construction, the only
        # error being the complex multiply's own round-off)
        assert float(np.abs(np.real(rows)).max()) == 0.0
        assert float(np.abs(np.abs(np.exp(rows)) - 1.0).max()) < 1e-15

    def test_the_carrier_field_deprecation_cycle_is_complete(self):
        """The announcement half of the C5 freeze, checked on every route a
        caller actually uses: construction, one warning PER assignment,
        ``dataclasses.replace``, ``with_provenance``, ``deepcopy`` and
        ``pickle`` (the multi orchestrator ships fields to workers), and the
        accumulation idiom the pipeline driver was changed to."""
        import copy
        import pickle
        lam, n, dx = self.LAM, 32, 2e-6
        env = _gauss(n, dx, 8e-6).astype(complex)
        grid = FieldGrid((n, n), dx)
        spec = CarrierSpec(R=-1e-2)
        with warnings.catch_warnings():
            warnings.simplefilter('error', DeprecationWarning)
            f = CarrierField(env, grid, spec, lam)
            copy.deepcopy(f)
            pickle.loads(pickle.dumps(f))
            dataclasses.replace(f, wavelength=1.55e-6)
            f.with_provenance(note='verify')
            np.add(f.envelope, env, out=f.envelope)     # the driver's idiom
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            f.wavelength = 1.55e-6
            f.wavelength = 1.60e-6
        assert len(w) == 2 and all(x.category is DeprecationWarning for x in w)
        assert f.wavelength == 1.60e-6              # still takes effect
        # the horizon must be ahead of the running library, via the shared
        # resolver rather than a literal
        from lumenairy._deprecation import resolve_removal_version
        assert (resolve_removal_version(CF._CARRIER_FIELD_FROZEN_IN)
                == CF._CARRIER_FIELD_FROZEN_IN)
        # '_built' is a gate, not a field: it must not reach the dataclass API
        assert '_built' not in [fl.name for fl in dataclasses.fields(f)]
        assert '_built' not in repr(f)

    def test_the_in_place_accumulation_is_bit_identical_to_the_rebinding_one(
            self):
        """``validation/pipeline/driver.py`` was changed (at WP-A6's request)
        from ``acc.envelope += ...`` to ``np.add(..., out=...)``.  The claim is
        bit-identity AND warning-freedom; both are asserted here so the request
        is verified in this repository rather than taken on trust."""
        rng = np.random.default_rng(1)
        a = rng.standard_normal((64, 64)) + 1j * rng.standard_normal((64, 64))
        b = rng.standard_normal((64, 64)) + 1j * rng.standard_normal((64, 64))
        c1 = a.copy()
        c1 += b
        c2 = a.copy()
        np.add(c2, b, out=c2)
        assert np.array_equal(c1.view(np.float64), c2.view(np.float64))


# ===========================================================================
# C3 -- dtype, on the entry points the WP's own tests do not reach
# ===========================================================================
class TestVerifyC3:

    LAM = 1.31e-6

    def test_every_readout_keeps_complex64(self):
        n, dx, w, rmag = 256, 8e-6, 200e-6, 20e-3
        env = _gauss(n, dx, w).astype(np.complex64)
        kw = dict(dx_out=(self.LAM * rmag / (np.pi * w)) / 8.0, N_out=32)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            a = C.carrier_referenced_focus_readout(
                env, -rmag, rmag, self.LAM, dx, on_replica='ignore', **kw)
            b = C.carrier_referenced_focus_readout(
                env, -rmag, rmag, self.LAM, dx, on_replica='ignore',
                tilt=(0.02, -0.01), **kw)
            c = C.carrier_referenced_exact_focus_readout(
                env, -rmag, rmag, self.LAM, dx, **kw)
        for name, arr in (('paraxial', a), ('paraxial+tilt', b), ('exact', c)):
            assert np.asarray(arr).dtype == np.complex64, name

    def test_the_carrier_field_verbs_keep_complex64(self):
        n, dx, w, rmag = 128, 8e-6, 200e-6, 20e-3
        grid = FieldGrid((n, n), dx)
        s1 = CarrierSpec(R=-rmag, tilt=(0.01, -0.005), piston=3e-3)
        s2 = CarrierSpec(R=-1.2 * rmag)
        f = CarrierField(_gauss(n, dx, w).astype(np.complex64), grid, s1,
                         self.LAM)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rr = CF.re_reference(f, s2, grid)
            ag = CF.aggregate([f, f], s2, grid)
        assert f.full_field().dtype == np.complex64
        assert np.asarray(rr.envelope).dtype == np.complex64
        assert ag.field.envelope.dtype == np.complex64
        assert s1.phasor_on(grid, self.LAM).dtype == np.complex128   # default

    def test_the_tilted_landing_ramp_is_built_in_the_field_s_dtype(self):
        """DEFECT FOUND BY THE VERIFIER (reported independently by WP-A5).

        The chain's TILTED paraxial landing multiplied the readout field by a
        chief-ray ramp ``np.exp(i k0 (L u + M v))`` built whole-grid at
        complex128, which promotes a complex64 readout right at the end of the
        chain.  WP-A6's report attributed that promotion to
        ``angular_spectrum_propagate_mft``; WP-A5 measured that function to
        preserve complex64, and the readout itself returns complex64 (pinned in
        ``test_every_readout_keeps_complex64``), so the ramp was the only site
        left.  Unlike the two obliquity pistons it is an ARRAY, so a
        ``complex(...)`` cast cannot weaken it -- it has to be BUILT narrow.

        The mechanism, stated executably (the fail-before), then the property:
        the banded build must equal the narrowed whole-grid one BIT for bit, at
        both parities of the output grid, so the complex64 path costs exactly
        one float32 rounding and the complex128 path is the historical
        expression untouched."""
        k0 = 2.0 * np.pi / self.LAM
        a64 = np.ones((4, 4), dtype=np.complex64)
        # fail-before mechanism: the whole-grid complex128 ramp promotes
        assert (a64 * np.exp(1j * np.ones((4, 4)))).dtype == np.complex128
        for nn, dxo, L, M, cx, cy in ((32, 2e-6, 0.02, -0.01, 1e-4, -2e-4),
                                      (33, 3e-6, -0.05, 0.02, 5e-5, 5e-5)):
            u = _grid(nn, dxo) + cx
            v = _grid(nn, dxo) + cy
            r128 = np.exp(1j * k0 * (L * u[None, :] + M * v[:, None]))
            r64 = C._phasor_rows(
                lambda r0, r1, _u=u, _v=v: 1j * k0 * (L * _u[None, :]
                                                      + M * _v[r0:r1, None]),
                (nn, nn), np.complex64)
            assert np.array_equal(r64, r128.astype(np.complex64)), nn
            assert (a64[:1, :1] * r64[:1, :1]).dtype == np.complex64

    @pytest.mark.parametrize('dt', [np.complex64, np.complex128])
    def test_the_chain_tilted_paraxial_landing_keeps_its_dtype(self, dt):
        """End to end, through the branch the unit test above isolates: a
        TILTED congruence landing on ``final_leg='paraxial'`` with a
        ``focus_readout`` window -- the one configuration that reaches the
        chief-ray ramp.  (The audit's own ``p8_c64chain.py`` supplies no
        ``focus_readout``, so it lands on the bare final leg and never
        exercised this site.)

        Measured stage tilt on this fixture: L = 0.018554, i.e. the tilted
        branch is genuinely taken.  Pre-fix the complex64 arm returned
        complex128; complex128 in must still give complex128 out."""
        from lumenairy.elements._lens_traced import TiltedCarrier
        from lumenairy.glass import GLASS_REGISTRY
        GLASS_REGISTRY.setdefault('_VA6GLASS', (lambda wl: 1.5168))
        presc = {
            'wavelength': self.LAM, 'aperture_diameter': 8e-3,
            'surfaces': [
                {'radius': 51.68e-3, 'thickness': 5e-3, 'glass_before': 'air',
                 'glass_after': '_VA6GLASS', 'semi_diameter': 4e-3},
                {'radius': -51.68e-3, 'thickness': 0.0,
                 'glass_before': '_VA6GLASS', 'glass_after': 'air',
                 'semi_diameter': 4e-3}],
            'thicknesses': [5e-3], 'stop_index': 0}
        n, dx = 256, 8e-6
        env = _gauss(n, dx, 300e-6).astype(dt)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = C.propagate_traced_carrier_chain(
                env, [{'prescription': presc, 'gap_before': 2e-3}], self.LAM,
                dx, r_in=TiltedCarrier(np.inf, 0.02, 0.0, 0.0, 0.0),
                ray_subsample=4, final_distance=40e-3,
                traced_kwargs=dict(amplitude_model='ray_density',
                                   preserve_input_phase='remap',
                                   remap_sampling='full',
                                   fit_radius_beam_factor=2.0),
                final_leg='paraxial',
                focus_readout=dict(dx_out=2e-6, N_out=32,
                                   on_replica='ignore',
                                   on_focus_containment='ignore'),
                on_decentred_fit='ignore', on_gap_paraxial='ignore',
                on_gap_frame='ignore', on_multi_congruence='ignore')
        assert res.stages and res.stages[-1].get('L'), res.stages[-1]
        assert np.asarray(res.field).dtype == np.dtype(dt)

    def test_the_complex64_accumulator_costs_one_float32_rounding(self):
        """A DEFAULT CHANGE worth stating: ``aggregate`` used to sum in
        complex128 whatever the members were stored as, and now follows them.
        The price is float32 accumulation.

        BAR: the difference against a complex128 sum of the SAME fields must be
        at most ``sqrt(K) * eps32`` = 4 x 1.2e-7 = 4.8e-07 for K = 16 -- the
        random-walk bound for K roundings -- and must be non-zero (or the
        accumulator did not narrow).  Measured 7.6e-08, i.e. 6x inside the
        bound."""
        n, dx = 64, 8e-6
        grid = FieldGrid((n, n), dx)
        spec = CarrierSpec(R=np.inf)
        rng = np.random.default_rng(4)
        raw = [(rng.standard_normal((n, n))
                + 1j * rng.standard_normal((n, n))).astype(np.complex64)
               for _ in range(16)]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            s64 = CF.aggregate([CarrierField(e, grid, spec, self.LAM)
                                for e in raw], spec, grid).field.envelope
            s128 = CF.aggregate([CarrierField(e.astype(np.complex128), grid,
                                              spec, self.LAM) for e in raw],
                                spec, grid).field.envelope
        assert s64.dtype == np.complex64 and s128.dtype == np.complex128
        rel = float(np.linalg.norm(s64.astype(np.complex128) - s128)
                    / np.linalg.norm(s128))
        assert 0.0 < rel < 4.0 * float(np.finfo(np.float32).eps), rel


# ===========================================================================
# The oracle's own convention (WP-B11a item 17)
# ===========================================================================

def test_the_gaussian_oracle_carries_this_librarys_phase_convention():
    """ONE ABSOLUTE, PISTON-INCLUDED assertion on :func:`_abcd_field`.

    Every other use of the oracle in this file compares a NORMALISED field or
    a width, so the Gouy phase's SIGN was invisible: the pre-B11 spelling built
    ``1/q = 1/R - i lam/(pi w^2)`` -- Siegman's ``exp(+i omega t)`` pairing --
    and then added ``angle(q/q2)`` on top of it, which in this library's
    ``exp(-i omega t)`` convention (``CONVENTIONS.md`` Section 7) is the Gouy
    phase with the wrong sign.  The error is ``2 arctan(z/zR)``, i.e. EXACTLY
    pi across a focus, and no assertion in the file could see it.

    This is the assertion that can.  For a beam at its own waist the whole
    field on axis is ``exp(i k z) * exp(-i arctan(z / zR))`` -- the plane-wave
    piston times a Gouy RETARDATION -- and the transverse profile at radius r
    carries ``k r^2 / (2 R(z))`` of curvature on top of it.

    MEASURED 2026-09-13 (lam = 1 um, w0 = 200 um, zR = 125.664 mm): the
    corrected oracle reads an on-axis Gouy of 0.000000 / -0.463648 /
    -0.785398 / -1.249046 rad at z/zR = 0 / 0.5 / 1 / 3, against
    ``arctan(z/zR)`` = 0.000000 / +0.463648 / +0.785398 / +1.249046 -- i.e.
    exactly ``-arctan(z/zR)``, the sign this convention requires and the
    OPPOSITE of what the old spelling produced.  The widths and the
    piston-free profiles are unchanged (w(z) identical to all printed digits,
    normalised-profile difference <= 1.8e-10, which is the round-off between
    the two algebraic spellings of the same curvature).
    """
    lam, w_in = 1.0e-6, 200.0e-6
    z_r = np.pi * w_in ** 2 / lam
    k = 2.0 * np.pi / lam
    on_axis = np.zeros(1)
    for z_over_zr in (0.0, 0.5, 1.0, 3.0):
        z = z_over_zr * z_r
        E, wz = _abcd_field(on_axis, w_in, np.inf, z, lam)
        # (a) the width is the textbook one
        assert abs(wz - w_in * np.sqrt(1.0 + z_over_zr ** 2)) <= 1e-12 * w_in
        # (b) the ABSOLUTE on-axis phase is the piston MINUS the Gouy shift
        got = complex(E[0, 0])
        want = np.exp(1j * (k * z - np.arctan(z_over_zr)))
        # DERIVED bar.  The piston ``k z`` reaches 3.95e+05 rad at z = 3 zR
        # here, so ``exp(1j k z)`` carries ``eps * k z`` ~ 8.8e-11 rad of
        # representation error before any physics -- a tighter bar would be
        # measuring float64, not the convention.  1e-9 is ~11x that and nine
        # decades below the 2 arctan(z/zR) (0.93 rad at z = 0.5 zR) the wrong
        # convention produces.
        assert abs(got / abs(got) - want) <= 1e-9, (
            f'z = {z_over_zr} zR: on-axis phase {np.angle(got):+.9f} rad '
            f'against the convention\'s {np.angle(want):+.9f}; the difference '
            f'{np.angle(got / want):+.9f} rad is the Gouy sign.')
        # (c) and the sign is not accidentally symmetric: the WRONG convention
        # differs by exactly 2 arctan(z/zR), which is the falsification arm
        wrong = np.exp(1j * (k * z + np.arctan(z_over_zr)))
        if z_over_zr > 0.0:
            assert abs(got / abs(got) - wrong) > 1e-3, (
                'the oracle agrees with BOTH sign conventions, so this test '
                'cannot tell them apart')

    # (d) across a focus the total swing is exactly pi -- the statement the
    # brief makes, and the reason a piston-free file never noticed
    far = 1e9 * z_r
    E_minus, _ = _abcd_field(on_axis, w_in, np.inf, -far, lam)
    E_plus, _ = _abcd_field(on_axis, w_in, np.inf, +far, lam)
    gouy = (np.angle(complex(E_plus[0, 0]) * np.exp(-1j * k * far))
            - np.angle(complex(E_minus[0, 0]) * np.exp(1j * k * far)))
    assert abs(abs(gouy) - np.pi) < 1e-6, (
        f'the Gouy swing across a focus is {gouy:+.9f} rad, not +/-pi')
