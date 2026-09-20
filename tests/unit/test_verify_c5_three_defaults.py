"""VERIFY-WP-C5 -- the decision tests this verification added, one per gap it
closed in the three default flips of ``feat/c5-three-defaults``.

Nothing here restates what ``tests/unit/test_c5_three_defaults.py`` already
asserts.  Each id below is a property that branch's own tests do not reach,
and each was found by re-measuring rather than by reading:

ITEM 1 (``carrier._GAP_KERNEL_ACCURACY_TAU``)
  * the rule is a band in ``k |z_eff| theta_env^4``, NOT a distance to a
    focus, and the suite already contains a leg that proves it: on one
    fixture family the leg FURTHEST from its own ``A = 0`` plane falls back
    while the leg CLOSEST to it does not;
  * on that leg the fallback moves the answer TOWARD the exact scalar
    (non-paraxial) field rather than away from it -- measured against a
    closed-form angular-spectrum oracle, which is what the WP-C5 report lists
    under "could not be measured" -- and the leg's own paraxial modelling
    error is two decades LARGER than the kernel decision, so the decision is
    not a physics choice there;
  * the closed-form band holds on a third fixture (a different lambda, w0
    and f), bisected on the running build.

ITEM 2 (``gbd.DENSE_MEM_BUDGET_ACCOUNTING``)
  * the published floor bounds the process RESIDENT SET, not only
    ``tracemalloc`` (the report states the floor for ``tracemalloc`` only);
  * "warn, not refuse" is checked where it actually binds: the floor crosses
    the SHIPPED default budget AT ``N = 1706``, and a DEFAULT-path call past
    that completes, warns exactly once, and names both remedies;
  * the mutation "the floor helper forgets its fixed term" is caught by the
    at-the-floor rung of the sweep.

ITEM 3 (``replica_fill``)
  * the zeroed set is the complement of one period about the FIELD's origin
    over six OFF-AXIS and ANAMORPHIC windows, with the period measured
    EMPIRICALLY off the returned array rather than taken from the same
    expression the library's mask uses;
  * the mutation "the fill keyed on the WINDOW's centre" is caught;
  * the mutation "the exact readout's faithful band is blanked" is caught;
  * an off-axis window is REFUSED at a window/period ratio that is SERVED on
    axis, identically under both fills.

Every bar is derived at runtime from what the running build measures.
"""
import gc
import inspect
import threading
import tracemalloc
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.propagators import carrier as CA
from lumenairy.propagators import gbd as G


# ===========================================================================
# shared helpers
# ===========================================================================
def _ax(n, d):
    return (np.arange(int(n), dtype=np.float64) - int(n) / 2) * float(d)


def _gauss(n, dx, w):
    g = _ax(n, dx)
    return np.exp(-(g[None, :] ** 2 + g[:, None] ** 2)
                  / float(w) ** 2).astype(np.complex128)


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _rel_pf(a, b):
    """Piston-free relative L2 -- an absolute phase is not a difference."""
    a, b = np.asarray(a), np.asarray(b)
    ov = np.vdot(b, a)
    return _rel(a / (ov / abs(ov)) if abs(ov) > 0 else a, b)


# ===========================================================================
# Item 1 -- the band is in k |z_eff| theta^4, and what it costs there
# ===========================================================================
#: A converging Gaussian re-enveloped against a MISMATCHED carrier.  The
#: envelope then carries the residual lens ``1/R0 - 1/(fr R0)``, so
#: ``theta_env`` grows with the mismatch while the leg stays the same length
#: -- which is how one fixture family can separate "near a focus" from "inside
#: the band".  Numbers are this verification's own (WP-A6's fixture is 1.31 um
#: at NA 0.05; this is 1.55 um at NA 0.04).
_M = dict(lam=1.55e-6, n=1024, w=0.80e-3, na=0.04, ext=4.0)
_M['R0'] = -_M['w'] / _M['na']
_M['dx'] = 2.0 * _M['ext'] * _M['w'] / _M['n']
_M['w0'] = _M['lam'] * abs(_M['R0']) / (np.pi * _M['w'])
_M['dxo'] = _M['w0'] / 8.0
_M['nout'] = 64


def _mismatch_env(fr):
    k = 2.0 * np.pi / _M['lam']
    g = _ax(_M['n'], _M['dx'])
    r2 = g[None, :] ** 2 + g[:, None] ** 2
    return (np.exp(-r2 / _M['w'] ** 2)
            * np.exp(1j * k * r2 / (2.0 * _M['R0']))
            * np.exp(-1j * k * r2 / (2.0 * fr * _M['R0']))).astype(
                np.complex128)


def _mismatch_leg(fr, gap_kernel, stats=None):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.asarray(CA._collins_transport(
            _mismatch_env(fr), fr * _M['R0'], -_M['R0'], _M['lam'],
            _M['dx'], _M['dx'], dx_out=_M['dxo'], dy_out=_M['dxo'],
            N_out_x=_M['nout'], N_out_y=_M['nout'], R_ref=float('inf'),
            gap_kernel=gap_kernel, on_collins_sampling='ignore',
            stats_out=stats))


def _distance_to_A0(fr):
    """How far this leg lands from its OWN ``A = 0`` plane -- the plane the
    ledger's "near a focus" means.  ``A = 1 + z/R``, so ``A = 0`` at
    ``z = -R = -fr R0``."""
    return abs(-_M['R0'] - (-fr * _M['R0']))


@pytest.fixture(scope='module')
def _exact_scalar_oracle():
    """The EXACT scalar (Helmholtz) field of the fixture's physical input, on
    the readout's own lattice.

    ``exp(-r^2/w^2) exp(i k r^2 / 2 R0)`` is ``exp(-a r^2)`` with complex
    ``a``, whose 2-D Fourier transform is ``(pi/a) exp(-pi^2 f^2 / a)``
    ANALYTICALLY -- so nothing has to be sampled on the input lattice, which
    the carrier aliases anyway.  That spectrum is propagated with the exact
    transfer function ``exp(i k z sqrt(1 - lambda^2 f^2))`` and inverted by a
    1-D Hankel quadrature on the grid's unique radii.  Shares no machinery
    with the library.

    The quadrature's own error is measured here by halving the sample count;
    the ids below refuse to conclude anything if it is not far below the
    quantity they compare.
    """
    from scipy.special import j0

    lam, R0, w = _M['lam'], _M['R0'], _M['w']
    z = -R0
    k = 2.0 * np.pi / lam
    a = 1.0 / w ** 2 - 1j * k / (2.0 * R0)
    inv_a = 1.0 / a
    F = min(float(np.sqrt(46.0 / (np.pi ** 2 * inv_a.real))), 0.90 / lam)
    xo = _ax(_M['nout'], _M['dxo'])
    rr = np.round(np.sqrt(xo[None, :] ** 2 + xo[:, None] ** 2), 15)
    rho = np.unique(rr)

    def _quad(n_f):
        f = np.linspace(0.0, F, int(n_f))
        H = np.exp(1j * k * z
                   * np.sqrt(np.maximum(1.0 - (lam * f) ** 2, 0.0)))
        wf = (np.pi * inv_a) * np.exp(-(np.pi ** 2) * inv_a * f * f) * H * f
        vals = np.empty(rho.size, dtype=np.complex128)
        step = max(1, int(4.0e7 // max(1, f.size)))
        for i0 in range(0, rho.size, step):
            blk = rho[i0:i0 + step]
            B = j0(2.0 * np.pi * np.outer(blk, f)) * wf[None, :]
            vals[i0:i0 + step] = 2.0 * np.pi * np.trapezoid(B, f, axis=-1)
        look = {v: c for v, c in zip(rho.tolist(), vals.tolist())}
        return np.array([look[v] for v in rr.ravel().tolist()],
                        dtype=np.complex128).reshape(rr.shape)

    full = _quad(200_000)
    half = _quad(100_001)
    return full, float(np.linalg.norm(full - half) / np.linalg.norm(full))


def test_the_rule_is_a_band_in_k_z_theta4_and_not_a_distance_to_a_focus():
    """ITEM 1, SCOPE.  The ledger describes the armed rule as firing "only
    for a leg that lands within about 100 micrometres of a geometric focus".
    The law is ``sqrt(3/2) k |z_eff| theta_env^4 / 8``, which is a band in
    ``k |z_eff| theta^4``, so a WIDE envelope far from a focus trips it too.

    Demonstrated on one fixture family, so nothing but the envelope's angle
    differs in spirit: three carrier mismatches of the same physical beam.
    The leg that falls back is the one FURTHEST from its own ``A = 0`` plane
    and the legs that do not are CLOSER to it -- the opposite ordering to a
    distance rule.  Measured 2026-09-20 on both builds: 3.00 mm from ``A = 0``
    at ``theta_env`` 7.09e-03 rad falls back (departure 1.77e-04), 2.00 mm at
    4.49e-03 does not (4.53e-05), 1.00 mm at 2.19e-03 does not (5.46e-06) --
    against VERIFY-B4 F3, whose ladder falls back only inside 23.5 um of its
    own ``A = 0`` plane.  The three distances and the three departures are
    both re-measured here; nothing is pasted.
    """
    tau = CA._GAP_KERNEL_ACCURACY_TAU
    assert tau is not None, (
        "PREMISE: the accuracy rule is disarmed on this tree, so there is no "
        "band to place")
    rows = []
    for fr in (0.85, 0.90, 0.95):
        st = {}
        _mismatch_leg(fr, 'auto', st)
        assert 'kernel_departure' in st, (
            f"the k4 gate did not resolve fr={fr} to 'exact', so the "
            f"accuracy rule never saw it and this fixture cannot place the "
            f"band (k4={st.get('k4')})")
        rows.append(dict(fr=fr, d0=_distance_to_A0(fr),
                         dep=float(st['kernel_departure']),
                         kernel=st['kernel'], k4=float(st['k4'])))

    far, mid, near = rows          # 0.85 is furthest from A = 0
    assert far['d0'] > mid['d0'] > near['d0'] > 0.0, rows
    assert far['kernel'] == 'fresnel', (
        f"the leg {far['d0'] * 1e3:.2f} mm from its A=0 plane did NOT fall "
        f"back (departure {far['dep']:.4e} against tau {tau:.1e})")
    assert mid['kernel'] == 'exact' and near['kernel'] == 'exact', (
        f"a leg CLOSER to its A=0 plane fell back too, so this fixture no "
        f"longer separates the two rules: {rows}")
    # the separation is in the ANGLE, and it is quartic: the ratio of the two
    # departures must be the ratio of the fourth powers times the ratio of
    # the reduced distances, which is what makes this a band in k|z|theta^4.
    assert far['dep'] > tau > mid['dep'], rows
    assert far['dep'] / mid['dep'] > 2.0, (
        f"PREMISE: the two departures differ by only "
        f"{far['dep'] / mid['dep']:.2f}x, so this fixture is sitting on the "
        f"bar rather than either side of it")
    # and the k4 REPRESENTABILITY gate sees none of this: every leg is
    # decades under its bar of 1, which is the gap the accuracy rule closes.
    assert max(r['k4'] for r in rows) < 1e-2, rows
    # the rule fires two decades further from A = 0 than the near-focus
    # picture allows.  100 um is the ledger's own figure; the bar here is a
    # tenth of it, re-derived as "the leg that fires is not near-focus by any
    # reading of the word".
    assert far['d0'] > 1.0e-4, (
        f"the falling-back leg sits {far['d0'] * 1e6:.1f} um from its A=0 "
        f"plane, which IS near-focus, so this id is not demonstrating the "
        f"scope it claims")


def test_the_rule_fires_on_a_leg_that_is_near_no_focus_at_all():
    """ITEM 1, SCOPE, without the escape hatch.

    The id above reads out AT the beam's own focus, so "it is not near a
    focus" rests on the CARRIER's ``A = 0`` plane alone.  This one removes
    that: the readout plane sits a QUARTER of the way to the focus, where the
    beam is still 0.600 mm wide -- 48.7 focal waists, i.e. 48.7 Rayleigh
    ranges short of its focus -- and 9.00 mm from the carrier's own ``A = 0``
    plane.  There is no focus within two decades of this plane by either
    reading, and the rule fires anyway, because ``theta_env`` is 1.1e-02 rad.

    Measured 2026-09-20 on both builds, readout at ``z = 0.25 |R0|``:
    ``fr = 0.70`` reads ``|z_eff|`` 7.778e-03 m, ``k4`` 8.95e-05 and a
    departure 4.179e-04 -> ``'fresnel'``; ``fr = 0.80``, which is FURTHER
    from the ``A = 0`` plane (11.0 mm against 9.00 mm), reads 4.548e-05 and
    keeps ``'exact'``.  Both bars are re-derived here.
    """
    tau = CA._GAP_KERNEL_ACCURACY_TAU
    assert tau is not None, "PREMISE: the rule is disarmed on this tree"
    z = 0.25 * abs(_M['R0'])
    w_here = _M['w'] * (1.0 - z / abs(_M['R0']))        # geometric radius
    z_R = np.pi * _M['w0'] ** 2 / _M['lam']
    to_focus = abs(_M['R0']) - z
    assert to_focus / z_R > 20.0, (
        f"PREMISE: the readout plane is only {to_focus / z_R:.1f} Rayleigh "
        f"ranges from the beam's focus, so it IS near one")
    assert w_here / _M['w0'] > 20.0, (
        f"PREMISE: the beam is only {w_here / _M['w0']:.1f} focal waists "
        f"wide here")

    rows = {}
    for fr in (0.70, 0.80):
        st = {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            CA._collins_transport(
                _mismatch_env(fr), fr * _M['R0'], z, _M['lam'], _M['dx'],
                _M['dx'], dx_out=_M['dxo'], dy_out=_M['dxo'],
                N_out_x=_M['nout'], N_out_y=_M['nout'], R_ref=float('inf'),
                gap_kernel='auto', on_collins_sampling='ignore',
                stats_out=st)
        assert 'kernel_departure' in st, (
            f"the k4 gate did not resolve fr={fr} to 'exact' (k4="
            f"{st.get('k4')}), so the accuracy rule never saw this leg")
        rows[fr] = dict(dep=float(st['kernel_departure']),
                        kernel=st['kernel'], k4=float(st['k4']),
                        d0=abs(z - (-fr * _M['R0'])))

    assert rows[0.70]['d0'] > 1.0e-3, (
        f"the falling-back leg is {rows[0.70]['d0'] * 1e3:.2f} mm from its "
        f"A=0 plane, which is not the two decades this id claims")
    assert rows[0.70]['kernel'] == 'fresnel', rows
    assert rows[0.80]['kernel'] == 'exact', rows
    assert rows[0.80]['d0'] > rows[0.70]['d0'], (
        f"the leg that KEPT the exact kernel is not further from the A=0 "
        f"plane than the one that fell back, so this id does not separate "
        f"the band from a distance rule: {rows}")
    assert rows[0.70]['dep'] > tau > rows[0.80]['dep'], rows
    assert max(r['k4'] for r in rows.values()) < 1e-2, rows


def test_an_explicit_exact_survives_the_band_on_the_wide_angle_leg():
    """The opt-out, on the leg the rule actually moves in the suite.  An
    explicit ``gap_kernel='exact'`` must be honoured there, or the caller's
    request is being silently replaced -- the shape the vocabulary gates
    exist to remove.  Two-sided: ``'auto'`` on the same leg does fall back,
    so the id is not vacuous."""
    st_auto, st_exact = {}, {}
    E_auto = _mismatch_leg(0.85, 'auto', st_auto)
    E_exact = _mismatch_leg(0.85, 'exact', st_exact)
    E_fres = _mismatch_leg(0.85, 'fresnel')
    assert st_auto['kernel'] == 'fresnel', st_auto
    assert st_exact['kernel'] == 'exact', st_exact
    assert np.array_equal(E_auto, E_fres), (
        "'auto' fell back but did not return the 'fresnel' answer")
    assert not np.array_equal(E_exact, E_fres), (
        "PREMISE: the refinement changed nothing on this leg, so honouring "
        "it is not observable here")


def test_on_the_wide_angle_leg_the_fallback_is_not_a_step_away_from_physics(
        _exact_scalar_oracle):
    """ITEM 1, the question the WP-C5 report files under "could not be
    measured": on a leg the rule moves, which kernel is closer to the truth?

    The report's oracle is PARAXIAL, so it can only say how far the exact
    kernel departs from the paraxial answer.  This id uses a NON-PARAXIAL
    oracle instead -- the closed-form angular spectrum of the fixture's own
    input, propagated with ``exp(i k z sqrt(1 - lambda^2 f^2))`` -- and reads
    two things off it.

    (1) The fallback does not make the leg worse.  Measured 2026-09-20 on
    both builds: the plain kernel reads 2.89933e-02 against the exact scalar
    field and the refined one 2.91439e-02, so the refinement is 1.5e-04
    FURTHER away and the rule removes it.

    (2) And the choice is not a physics choice at this fixture's NA: BOTH
    kernels sit 2.9e-02 from the exact field, because the refinement lives in
    the REDUCED frame on the ENVELOPE's angle while the leg's own
    non-paraxiality is set by the beam's NA.  The decision the rule takes is
    192x smaller than the error neither kernel addresses (1.506e-04 against
    2.899e-02).

    Bars: (1) is an inequality between two measurements taken here, with a
    premise that the oracle's own quadrature error is two decades under the
    gap being read; (2) is a ratio with a bar of 10 against a measured 192.
    """
    oracle, quad_err = _exact_scalar_oracle
    st = {}
    E_auto = _mismatch_leg(0.85, 'auto', st)
    E_fres = _mismatch_leg(0.85, 'fresnel')
    E_exact = _mismatch_leg(0.85, 'exact')
    assert st['kernel'] == 'fresnel', (
        "PREMISE: the rule did not fire on this leg, so there is no decision "
        "to score")

    d_auto = _rel_pf(E_auto, oracle)
    d_fres = _rel_pf(E_fres, oracle)
    d_exact = _rel_pf(E_exact, oracle)
    gap = abs(d_exact - d_fres)
    assert quad_err < 0.01 * gap, (
        f"PREMISE: the oracle's own quadrature error {quad_err:.3e} is not "
        f"far enough below the {gap:.3e} gap it is being asked to resolve")
    assert d_auto <= d_exact, (
        f"the accuracy rule moved this leg AWAY from the exact scalar field: "
        f"'auto' reads {d_auto:.6e}, the refinement it replaced reads "
        f"{d_exact:.6e}.  The rule's scope, not its constant, would be the "
        f"defect.")
    assert d_auto == pytest.approx(d_fres, rel=1e-12)
    # (2) the scale of what neither kernel models
    assert d_fres > 10.0 * gap, (
        f"PREMISE: the leg's own paraxial modelling error {d_fres:.3e} is "
        f"not large against the {gap:.3e} the kernel decision moves, so the "
        f"'not a physics choice' reading does not hold on this fixture")
    # and the oracle is a real one: a paraxial oracle would read ~0 here
    assert d_fres > 1e-3, (
        f"PREMISE: the exact-scalar oracle agrees with the paraxial answer "
        f"to {d_fres:.3e}, i.e. this fixture is paraxial and cannot "
        f"discriminate")


def test_the_closed_form_band_holds_on_a_third_fixture():
    """ITEM 1, the band on a fixture neither WP-C5 nor VERIFY-B4 used: HeNe
    at 0.633 um, w 0.20 mm, R -30 mm.

    The rule fires where ``|z_eff| > 8 tau / (sqrt(3/2) k theta_env^4)``.
    The switch-over distance is found by BISECTION on the running build (the
    leg is re-run at each rung and its own ``A`` and ``B`` are read), and the
    ``|z_eff|`` at that distance is compared with the closed form.  Measured
    2026-09-20, identical on both builds: the bisected edge is 14.082 um from
    the geometric focus and the closed-form threshold is 63.881 m in
    ``z_eff``; they agree to 1e-4 relative.  Bar 1 %, four decades of room.
    """
    tau = CA._GAP_KERNEL_ACCURACY_TAU
    assert tau is not None, "PREMISE: the rule is disarmed on this tree"
    lam, n, dx, w, R = 0.633e-6, 1024, 3e-6, 0.20e-3, -30e-3
    dxo, nout = 3.1e-6, 128
    env = _gauss(n, dx, w)
    theta = max(CA._collins_envelope_half_angle(
        np.fft.fft2(np.ascontiguousarray(env, dtype=np.complex128)),
        dx, dx, lam))

    def leg(dz, gk='auto'):
        st = {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            CA._collins_transport(
                env, R, -R - dz, lam, dx, dx, dx_out=dxo, dy_out=dxo,
                N_out_x=nout, N_out_y=nout, R_ref=float('inf'),
                gap_kernel=gk, on_collins_sampling='ignore', stats_out=st)
        return st

    lo, hi = 1e-9, 1e-3
    assert leg(lo)['kernel'] == 'fresnel' and leg(hi)['kernel'] == 'exact', (
        "PREMISE: the ladder does not bracket the switch-over on this build")
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        if leg(mid)['kernel'] == 'fresnel':
            lo = mid
        else:
            hi = mid
    edge = 0.5 * (lo + hi)
    st = leg(edge)
    z_eff = abs(float(st['abcd'][1]) / float(st['abcd'][0]))
    k = 2.0 * np.pi / lam
    predicted = 8.0 * tau / (CA._QUARTIC_RMS_MOMENT * k * theta ** 4)
    assert z_eff == pytest.approx(predicted, rel=0.01), (
        f"the observed switch-over sits at |z_eff| = {z_eff:.6g} m, against "
        f"a closed form {predicted:.6g} m (theta_env {theta:.4e} rad, "
        f"tau {tau:.1e}) -- the rule is not the law it documents")
    # two-sided on the DEPARTURE as well, and the bar is tau itself
    assert CA._collins_exact_kernel_departure(
        z_eff, theta, lam) == pytest.approx(tau, rel=0.01)
    assert 1e-9 < edge < 1e-3, edge


# ===========================================================================
# Item 2 -- the floor, RSS, and where "warn, not refuse" binds
# ===========================================================================
_PROC = None


def _proc():
    global _PROC
    if _PROC is None:
        import psutil
        _PROC = psutil.Process()
    return _PROC


class _RssPeak:
    """A SAMPLED high-water mark on the process's resident set.

    A before/after RSS difference is not a peak: the transient the chunk
    arithmetic is about is freed before the call returns and the allocator
    keeps the arena, so the difference reads near zero however large the
    transient was.  Sampling in a thread makes "peak" mean the same thing for
    RSS as it does for ``tracemalloc``.  It is a PROCESS-wide reading, so the
    ids below take the MINIMUM over repeats -- another thread's allocation
    can only push it up.
    """

    def __init__(self, interval=0.002):
        self.interval, self.peak, self.base = float(interval), 0, 0
        self._stop, self._th = threading.Event(), None

    def __enter__(self):
        gc.collect()
        self.base = self.peak = _proc().memory_info().rss
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        return self

    def _loop(self):
        while not self._stop.is_set():
            try:
                r = _proc().memory_info().rss
            except Exception:                            # pragma: no cover
                return
            self.peak = max(self.peak, r)
            self._stop.wait(self.interval)

    def __exit__(self, *a):
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=2.0)
        return False

    @property
    def delta(self):
        return int(self.peak - self.base)


@pytest.fixture
def _restore_accounting():
    old = G.DENSE_MEM_BUDGET_ACCOUNTING
    yield
    G.DENSE_MEM_BUDGET_ACCOUNTING = old


def _bundle(n=384, seed=17):
    rng = np.random.default_rng(seed)
    return G.BeamletBundle(
        positions=rng.normal(0.0, 1.5e-4, size=(n, 3)),
        directions=np.zeros((n, 3)),
        Q=np.full(n, 1.0 / (1.2e-3 - 0.03j), dtype=np.complex128),
        amplitude=(rng.normal(size=n)
                   + 1j * rng.normal(size=n)).astype(np.complex128),
        waist0=np.full(n, 1.2e-3))


def _dense(b, mode, budget_mb, N, chunk=4096, dx=2.5e-6, trace=True,
           catch=False):
    """``(field, tracemalloc_peak, rss_peak_delta, [messages])``."""
    G.DENSE_MEM_BUDGET_ACCOUNTING = mode
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        peak = base = 0
        with _RssPeak() as rw:
            if trace:
                tracemalloc.start()
            try:
                if trace:
                    tracemalloc.reset_peak()
                    base = tracemalloc.get_traced_memory()[0]
                out = G.reconstruct_field_from_beamlets(
                    b, Ny=N, Nx=N, dx=dx, wavelength=1.064e-6,
                    chunk_beamlets=chunk, mem_budget_mb=budget_mb)
                if trace:
                    peak = tracemalloc.get_traced_memory()[1]
            finally:
                if trace:
                    tracemalloc.stop()
    msgs = [str(w.message) for w in rec]
    return (np.asarray(out), int(peak - base), rw.delta,
            msgs if catch else [])


def test_the_flip_bounds_the_resident_set_and_not_only_tracemalloc(
        _restore_accounting):
    """ITEM 2, the instrument the WP-C5 report lists under "could not be
    measured".  Every reading there is ``tracemalloc``, which counts what the
    CPython allocator handed out; a caller who sets ``mem_budget_mb`` to fit a
    machine cares about the RESIDENT SET.

    Two things make this a measurement rather than a hope.  The transient is
    kept LARGE on both arms (a 300 MB budget on a 512-square grid: 214 MB
    under ``'measured'``, 1.80 GB under ``'legacy'``), because a small
    transient that follows a large one is served out of pages the allocator
    already holds and an in-process RSS delta then reads ZERO -- measured on
    WSL glibc, a 24.6 MB transient after a 192 MB one read 4 KB.  And the
    honest arm is the ``'measured'`` one FIRST, so its pages are the fresh
    ones.

    MEASURED 2026-09-20 at this id's own settings.  Windows py3.14:
    ``'measured'`` 0.7135x the budget on ``tracemalloc`` and 0.7146x on RSS;
    ``'legacy'`` 5.9983x and 5.9924x.  WSL py3.12: 0.6573x / 0.7789x and
    5.5017x / 5.4679x.  Bars, all derived here: the two modes must separate by
    3x in RESIDENT SET (measured 8.39x and 7.02x), ``'measured'`` must come in
    under the budget on the RSS reading (0.715 / 0.779), and the two
    instruments must agree within 2x on each arm (worst 1.185) -- otherwise
    one of them is not measuring this loop and nothing here can be
    concluded.
    """
    b = _bundle(n=256)
    N, budget_mb = 512, 300.0
    _dense(b, 'legacy', 1.0e7, N, chunk=1)              # warm-up, discarded
    out = {}
    for mode in ('measured', 'legacy'):                 # honest arm FIRST
        _f, pk, rs, _m = _dense(b, mode, budget_mb, N)
        out[mode] = (pk / (budget_mb * 1e6), rs / (budget_mb * 1e6), pk, rs)
        gc.collect()
    (t_leg, r_leg, pk_leg, rs_leg) = out['legacy']
    (t_mea, r_mea, pk_mea, rs_mea) = out['measured']

    for tag, t, r, pk, rs in (('legacy', t_leg, r_leg, pk_leg, rs_leg),
                              ('measured', t_mea, r_mea, pk_mea, rs_mea)):
        assert 0.5 < r / t < 2.0, (
            f"PREMISE ({tag}): tracemalloc reads {t:.3f}x the budget "
            f"({pk / 1e6:.1f} MB) and the sampled RSS peak {r:.3f}x "
            f"({rs / 1e6:.1f} MB) -- the two instruments disagree by more "
            f"than 2x, so RSS is not measuring this loop on this build and "
            f"nothing about the resident set can be concluded here")
    assert r_leg > 2.0, (
        f"PREMISE: 'legacy' only reached {r_leg:.2f}x the {budget_mb} MB "
        f"budget in RESIDENT SET, so the under-count this item removes is "
        f"not visible on this cell")
    assert r_mea < 1.0, (
        f"'measured' peaked at {r_mea:.3f}x the {budget_mb} MB budget in "
        f"RESIDENT SET ({rs_mea / 1e6:.1f} MB), so the honest accounting "
        f"does not bound what the machine feels")
    assert r_leg / r_mea > 3.0, (
        f"the two accountings separate by only {r_leg / r_mea:.2f}x in "
        f"resident set ({r_leg:.2f}x against {r_mea:.2f}x)")


def test_the_default_budget_binds_at_N_1706_and_the_call_still_completes(
        _restore_accounting):
    """ITEM 2, the premise under "warn, not refuse", checked where it binds.

    The branch's own ids exercise the notice at an explicitly small budget.
    The decision that a refusal would break callers rests on a different
    statement: at the SHIPPED default ``mem_budget_mb`` the floor crosses the
    budget on a grid a caller can plausibly ask for, and the call has to
    complete anyway.  Both halves are measured here.

    Measured 2026-09-20: with ``mem_budget_mb`` defaulting to 512.0 and the
    floor at 176 B/cell, ``N* = 1705.6``, so the floor is 511.64 MB at
    N = 1705 and 512.24 MB at N = 1706 -- it binds AT 1706, not past it.  A
    default-path call at N = 1707 completes, returns a finite field and emits
    exactly one floor notice naming 738/512-style numbers, ``window=5.0``,
    ``raise mem_budget_mb`` and the helper that publishes the floor.

    Two-sided: under ``'legacy'`` the same grid and budget emit NO floor
    notice, which is the mode gate.
    """
    assert G.DENSE_MEM_BUDGET_ACCOUNTING == 'measured', (
        "PREMISE: the shipped accounting is no longer 'measured', so the "
        "floor notice this id exercises is not on the default path and "
        "'warn, not refuse' is a decision about something else")
    default_budget = inspect.signature(
        G.reconstruct_field_from_beamlets).parameters['mem_budget_mb'].default
    per_cell = (G._DENSE_FIXED_CELL_BYTES + G._DENSE_CELL_BYTES_MEASURED)
    n_star = float(np.sqrt(default_budget * 1e6 / per_cell))
    below, at = int(np.floor(n_star)), int(np.floor(n_star)) + 1
    assert G._dense_budget_floor_bytes(below, below) < default_budget * 1e6
    assert G._dense_budget_floor_bytes(at, at) > default_budget * 1e6
    assert at == 1706, (
        f"the shipped default budget {default_budget} MB and the "
        f"{per_cell:.0f} B/cell floor now bind at N = {at}, not 1706; the "
        f"prose that quotes 1706 needs re-deriving")

    N = at + 1
    b = _bundle(n=24, seed=5)
    # dx chosen so the bundle's own profile still decays across the grid --
    # at a coarser pitch the Gaussian argument overflows at the edge and the
    # arm would be measuring the fixture, not the budget.
    G.DENSE_MEM_BUDGET_ACCOUNTING = 'measured'
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        E = np.asarray(G.reconstruct_field_from_beamlets(
            b, Ny=N, Nx=N, dx=5.0e-7, wavelength=1.064e-6))
    msgs = [str(w.message) for w in rec]
    notes = [m for m in msgs if 'CANNOT meet' in m and 'mem_budget_mb' in m]
    assert np.isfinite(E).all() and E.shape == (N, N), (
        "the default-path call at the binding grid did not return a finite "
        "field, so 'it completes' is not the reason to warn rather than "
        "refuse")
    assert len(notes) == 1, (
        f"expected exactly one floor notice from a DEFAULT call at N = {N}; "
        f"got {len(notes)}: {msgs}")
    floor_mb = G._dense_budget_floor_bytes(N, N) / 1e6
    assert ('%.6g' % floor_mb) in notes[0], notes[0]
    assert 'window=5.0' in notes[0] and 'raise mem_budget_mb' in notes[0], (
        f"the notice does not name both remedies, so a caller cannot act on "
        f"it: {notes[0]}")
    assert '_dense_budget_floor_bytes' in notes[0]
    del E
    gc.collect()

    # the mode gate: 'legacy' is silent on the same grid and budget.  The
    # chunk is capped at one here so the counter-arm does not allocate the
    # multi-gigabyte transient that mode would otherwise take -- the notice
    # does not depend on the chunk, only on the mode and the budget.
    G.DENSE_MEM_BUDGET_ACCOUNTING = 'legacy'
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        E2 = np.asarray(G.reconstruct_field_from_beamlets(
            b, Ny=N, Nx=N, dx=5.0e-7, wavelength=1.064e-6,
            chunk_beamlets=1, mem_budget_mb=default_budget))
    assert not [m for m in (str(w.message) for w in rec)
                if 'CANNOT meet' in m], "'legacy' emitted the floor notice"
    assert np.isfinite(E2).all()
    del E2
    gc.collect()


def test_a_floor_that_forgot_its_fixed_term_would_be_caught(
        _restore_accounting):
    """ITEM 2, MUTATION.  ``_dense_budget_floor_bytes`` is
    ``Ny*Nx*(48 + 128)``.  The plausible way to get it wrong is to publish
    only the chunk term, ``Ny*Nx*128`` -- the number the budget arithmetic
    itself uses -- which reads 0.727x the true floor.

    The arm that catches it is the AT-THE-FLOOR rung of the two-sided sweep:
    at a budget equal to the TRUE floor the loop stays under it, and at a
    budget equal to the MUTANT floor it does not.  Measured 2026-09-20 at
    N = 320: 0.826x at the true floor against 1.136x at the mutant's.

    The mutation is applied to the module so the failing arm runs against a
    real answer, and it is restored whatever happens.
    """
    b = _bundle()
    N = 320
    G.DENSE_MEM_BUDGET_ACCOUNTING = 'measured'
    _dense(b, 'measured', 1.0e7, N, chunk=1)            # warm-up, discarded
    true_floor = G._dense_budget_floor_bytes(N, N)
    _f, pk_true, _r, _m = _dense(b, 'measured', true_floor / 1e6, N)
    assert pk_true < true_floor, (
        f"at the published floor the loop peaked at {pk_true / 1e6:.3f} MB "
        f"against {true_floor / 1e6:.3f} MB")

    real = G._dense_budget_floor_bytes
    try:
        G._dense_budget_floor_bytes = (
            lambda Ny, Nx: float(Ny) * float(Nx) * G._DENSE_CELL_BYTES_MEASURED)
        mutant_floor = G._dense_budget_floor_bytes(N, N)
        assert mutant_floor < true_floor
        _f, pk_mut, _r, _m = _dense(b, 'measured', mutant_floor / 1e6, N)
    finally:
        G._dense_budget_floor_bytes = real
    assert pk_mut > mutant_floor, (
        f"the mutant floor {mutant_floor / 1e6:.3f} MB still bounded the "
        f"loop ({pk_mut / 1e6:.3f} MB), so dropping the fixed term is not "
        f"observable and the sweep's at-the-floor rung is not the arm that "
        f"catches it")
    assert G._dense_budget_floor_bytes(N, N) == true_floor


# ===========================================================================
# Item 3 -- the zeroing is confined to the complement of ONE period about the
#           FIELD's origin, off axis and anamorphic
# ===========================================================================
_WL3, _R3, _W3 = 1.55e-6, -25.0e-3, 0.60e-3
_N3, _DX3 = 512, 5.0e-6
#: ``lambda |z| / dx`` -- the Collins readout's period, from the documented
#: closed form rather than from the dict the library fills in.
_P3 = _WL3 * abs(_R3) / _DX3


def _pupil3(dy=None):
    x, y = _ax(_N3, _DX3), _ax(_N3, dy if dy else _DX3)
    return np.exp(-(x[None, :] ** 2 + y[:, None] ** 2)
                  / _W3 ** 2).astype(np.complex128)


def _read3(dx_out, n_out, centre=(0.0, 0.0), fill=None, dy=None,
           on_replica='ignore'):
    pd = {}
    kw = {} if fill is None else {'replica_fill': fill}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        F = CA._collins_focus_readout(
            _pupil3(dy), _R3, -_R3, _WL3, _DX3, (dy if dy else _DX3),
            dx_out=dx_out, N_out=n_out, centre_out=centre,
            on_replica=on_replica, on_collins_sampling='ignore',
            _period_out=pd, **kw)
    return np.asarray(F), pd


def _empirical_period(F, dx_out, axis, tol=1e-10, max_frac=0.75):
    """The period MEASURED off the ``'repeat'`` array: the SMALLEST integer
    sample shift that reproduces its MODULUS to ``tol``.

    Independent of the expression the library's mask uses, which is the whole
    point of doing it this way.  Two guards make it a measurement rather than
    a search for the smallest number: the FIRST shift meeting ``tol`` is
    returned rather than the best one, and shifts past ``max_frac`` of the
    axis are not considered at all -- a shift that leaves a sliver overlapping
    always "wins" on a windowed array, which is how a period measured by
    "smallest residual" reads 254 samples on a 256-sample window.

    WHY THE MODULUS.  ``_fill_readout_replicas`` states the periodicity as
    ``E(u + period) == E(u)``, and on the two readouts that finish on
    ``angular_spectrum_propagate_mft`` it holds on the complex field to
    1e-13.  On the COLLINS readout it holds on the modulus to 2e-14 and on
    the complex field only where the field is bright (measured 2026-09-20:
    3.0e-11 absolute on the core and its replica, against a wing pair at
    0.4 % of the peak whose phases differ by 1.7 rad).  A replica being a
    FULL-AMPLITUDE image of the core is what the fill is about and what every
    reduction it protects reads, so the modulus is both the safe reading and
    the load-bearing one.
    """
    A = np.abs(np.asarray(F))
    n, scale = A.shape[axis], float(A.max())
    assert scale > 0.0
    best = (None, np.inf)
    for m in range(2, int(max_frac * n) + 1):
        if axis == 1:
            a, c = A[:, m:], A[:, :n - m]
        else:
            a, c = A[m:], A[:n - m]
        if a.size == 0 or max(float(a.max()), float(c.max())) <= 0.0:
            break
        r = float(np.max(np.abs(a - c))) / scale
        if r < best[1]:
            best = (m, r)
        if r < tol:
            return m, r
    return best


def _bands(n, dx_out, period, centre):
    """The two candidate bands: about the FIELD's origin (the contract) and
    about the WINDOW's centre (the misapplication)."""
    u = _ax(n, dx_out)
    tol = 1.0 + 1e-9
    fx = np.abs(u + float(centre[0])) <= 0.5 * float(period[0]) * tol
    fy = np.abs(u + float(centre[1])) <= 0.5 * float(period[1]) * tol
    wx = np.abs(u) <= 0.5 * float(period[0]) * tol
    wy = np.abs(u) <= 0.5 * float(period[1]) * tol
    return (np.logical_and(fy[:, None], fx[None, :]),
            np.logical_and(wy[:, None], wx[None, :]))


#: Six windows past one period whose FIELD-centred and WINDOW-centred bands
#: differ: four off-axis, two anamorphic (``dy != dx``, so the two axes have
#: different periods and the mask has to be per-axis).
#: ``wp`` is chosen so that ``N_out/wp`` is a whole number of samples on both
#: axes -- then ONE period is an integer shift and the empirical measurement
#: above can read it to round-off instead of to the grid's own coarseness.
_OFFAXIS = [
    ('x030', 1.60, (0.30, 0.00), None),
    ('y030', 1.60, (0.00, 0.30), None),
    ('xy', 1.60, (0.30, -0.45), None),
    ('edge', 1.60, (0.62, 0.62), None),
    ('wide', 2.00, (0.55, 0.15), None),
    ('ana2', 1.60, (0.25, 0.40), 2.0),
    ('ana4', 1.60, (-0.33, 0.18), 4.0),
]


@pytest.mark.parametrize('tag,wp,cfrac,dyf', _OFFAXIS,
                         ids=[r[0] for r in _OFFAXIS])
def test_the_zeroed_set_is_one_period_about_the_field_origin(tag, wp, cfrac,
                                                             dyf):
    """ITEM 3, the misapplication the maintainer named, on SIX windows.

    ``E(u + period) == E(u)`` holds in ABSOLUTE output coordinates, so a
    window pushed off axis SPENDS its period rather than carrying it.  A fill
    keyed on the window's own centre passes every on-axis reading and is
    wrong here.

    What makes this an independent check rather than a restatement: the
    period is measured EMPIRICALLY off the returned ``'repeat'`` array (the
    integer sample shift that reproduces it) and cross-checked against the
    transport's documented closed form ``lambda |z| / dx``; the library's own
    ``_period_out`` is only compared, never used to build the band.

    Measured 2026-09-20 on both builds: on all six the zeroed set is exactly
    the complement of the FIELD-centred band and never the complement of the
    window-centred one, the two fills are byte-identical inside it, the part
    outside is exactly 0.0 under ``'zero'``, and under ``'repeat'`` it reaches
    29.04 against an in-band peak of 29.04 -- a full-amplitude copy of the
    core, which is what a wing-weighted reduction would have scored.
    """
    dy = dyf * _DX3 if dyf else None
    px = _P3
    py = _WL3 * abs(_R3) / (dy if dy else _DX3)
    n = 256
    dxo = wp * px / n
    centre = (cfrac[0] * px, cfrac[1] * py)

    F_rep, pd_r = _read3(dxo, n, centre=centre, fill='repeat', dy=dy)
    F_zero, pd_z = _read3(dxo, n, centre=centre, fill='zero', dy=dy)
    F_def, _pd_d = _read3(dxo, n, centre=centre, fill=None, dy=dy)

    # the period, three ways, before anything is asserted about the mask
    reported = (float(pd_r['period'][0]), float(pd_r['period'][1]))
    assert reported[0] == pytest.approx(px, rel=1e-12)
    assert reported[1] == pytest.approx(py, rel=1e-12)
    for axis, p in ((1, px), (0, py)):
        m, resid = _empirical_period(F_rep, dxo, axis)
        assert m is not None and resid < 1e-10, (
            f"axis {axis}: no shift reproduces the 'repeat' array to "
            f"round-off (best residual {resid} at shift {m}), so the array "
            f"is not periodic and the contract cannot be checked on it")
        assert m * dxo == pytest.approx(p, rel=1e-9), (
            f"axis {axis}: the array repeats every {m * dxo:.6e} m, not "
            f"every {p:.6e} m")

    field_band, window_band = _bands(n, dxo, reported, centre)
    assert not np.array_equal(field_band, window_band), (
        "PREMISE: this window is not off axis enough for the two candidate "
        "bands to differ, so it cannot discriminate between them")
    assert field_band.any() and (~field_band).any(), (
        "PREMISE: the window is entirely inside or outside one period")

    zeroed = (F_zero == 0) & (F_rep != 0)
    assert np.array_equal(zeroed, ~field_band), (
        "the zeroed set is not the complement of one period about the "
        f"FIELD's origin ({int(zeroed.sum())} zeroed against "
        f"{int((~field_band).sum())} expected)")
    assert not np.array_equal(zeroed, ~window_band), (
        "the zeroed set is the complement of the WINDOW-centred band -- the "
        "fill is keyed on centre_out rather than on the absolute coordinate")
    assert np.array_equal(F_rep[field_band], F_zero[field_band]), (
        "the two fills differ INSIDE one period")
    assert float(np.max(np.abs(F_zero[~field_band]))) == 0.0
    assert float(np.max(np.abs(F_rep[~field_band]))) > 0.0, (
        "PREMISE: 'repeat' is already zero outside the period here")
    assert np.array_equal(F_def, F_zero), "the default is not 'zero'"
    assert pd_z['replica_fill'] == 'zero'
    assert pd_r['replica_fill'] == 'repeat'
    assert tuple(pd_r['faithful_samples']) == (int(field_band.any(axis=0)
                                                   .sum()),
                                               int(field_band.any(axis=1)
                                                   .sum()))


def test_keying_the_fill_on_the_window_centre_would_be_caught():
    """ITEM 3, MUTATION (this verification's own).  The branch's matrix
    mutates the PERIOD; the maintainer's stated concern is misapplication,
    whose sharpest form is a fill keyed on the window rather than on the
    field.  Built through the library's own ``_fill_readout_replicas`` by
    dropping ``centre_out``, so the mutant is the array that mistake would
    really have produced.
    """
    n, wp = 256, 1.60
    dxo = wp * _P3 / n
    centre = (0.30 * _P3, 0.0)
    F_rep, pd = _read3(dxo, n, centre=centre, fill='repeat')
    F_zero, _ = _read3(dxo, n, centre=centre, fill='zero')
    period = tuple(float(v) for v in pd['period'])
    field_band, window_band = _bands(n, dxo, period, centre)
    assert not np.array_equal(field_band, window_band)

    mutant = np.asarray(CA._fill_readout_replicas(
        np.array(F_rep), period, dxo, n, (0.0, 0.0), 'zero'))
    assert not np.array_equal(mutant, F_zero), (
        "PREMISE: dropping centre_out changed nothing, so this window does "
        "not discriminate")
    mut_zeroed = (mutant == 0) & (F_rep != 0)
    assert np.array_equal(mut_zeroed, ~window_band), (
        "the mutant did not do what the mistake would do")
    assert not np.array_equal(mut_zeroed, ~field_band), (
        "the mutant is indistinguishable from the correct answer on this "
        "window")
    # the RIGHT call passes the same reading, so this id is about the key
    good = np.asarray(CA._fill_readout_replicas(
        np.array(F_rep), period, dxo, n, centre, 'zero'))
    assert np.array_equal((good == 0) & (F_rep != 0), ~field_band)
    # and the mutant keeps measurement it should have blanked AND blanks
    # measurement it should have kept -- both halves, so neither direction
    # can pass unnoticed
    assert float(np.max(np.abs(mutant[~field_band]))) > 0.0
    kept_wrongly = field_band & ~window_band
    assert kept_wrongly.any()
    assert float(np.max(np.abs(F_rep[kept_wrongly]))) > 0.0
    assert float(np.max(np.abs(mutant[kept_wrongly]))) == 0.0


def test_blanking_the_exact_readouts_faithful_band_would_be_caught():
    """ITEM 3, MUTATION (this verification's own), on the OTHER public
    readout.  The exact high-NA readout has its own period (the fine crop
    window) and its own fill call; a confinement check that only ever ran on
    the paraxial readout would not see a mistake made there.

    The mutation blanks one column INSIDE the exact readout's faithful band
    -- the shape an off-by-one in the comparison produces -- and the check has
    to reject it, with a premise that the column carried signal.
    """
    n, dx, w, R = 512, 0.5e-6, 30e-6, -0.2e-3
    x = _ax(n, dx)
    r2 = x[:, None] ** 2 + x[None, :] ** 2
    S = np.sign(R) * (np.sqrt(r2 + R * R) - abs(R))
    E = (np.exp(-r2 / w ** 2)
         * np.exp(1j * 2.0 * np.pi / _WL3 * S)).astype(np.complex128)
    kw = dict(dx_out=0.05e-6, window_factor=4.0, on_replica='ignore',
              N_out=3072)
    out = {}
    pd = {}
    for fill in ('repeat', 'zero'):
        d = {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out[fill] = np.asarray(la.carrier_referenced_exact_focus_readout(
                E, R, -R, _WL3, dx, _period_out=d, replica_fill=fill, **kw))
        pd = d
    period = tuple(float(v) for v in pd['period'])
    n_out, dxo = 3072, 0.05e-6
    band, _w = _bands(n_out, dxo, period, (0.0, 0.0))
    assert band.any() and (~band).any(), (
        "PREMISE: this window is entirely inside or entirely outside the "
        "exact readout's own period, so it cannot show confinement there")
    assert tuple(pd['faithful_samples']) == (int(band.any(axis=0).sum()),
                                             int(band.any(axis=1).sum()))
    assert pd['replica_fill'] == 'zero'
    # the unmutated state, first: equal inside, blanked outside
    assert np.array_equal(out['repeat'][band], out['zero'][band])
    assert float(np.max(np.abs(out['zero'][~band]))) == 0.0
    assert float(np.max(np.abs(out['repeat'][~band]))) > 0.0, (
        "PREMISE: 'repeat' is already zero outside this readout's period")

    inx = band.any(axis=0)
    edge = int(np.flatnonzero(inx)[-1])          # last faithful column
    assert float(np.max(np.abs(out['repeat'][:, edge]))) > 0.0, (
        "PREMISE: the column being blanked is already zero, so blanking it "
        "is not a mutation")
    mutant = np.array(out['zero'])
    mutant[:, edge] = 0.0
    assert not np.array_equal(out['repeat'][band], mutant[band]), (
        "blanking a column inside the EXACT readout's faithful band was not "
        "observable, so a confinement check on this readout could not catch "
        "an off-by-one")
    # and the mutation is invisible to the other two readings, which is why
    # the inside-the-band comparison is the one that has to be made
    assert float(np.max(np.abs(mutant[~band]))) == 0.0
    assert float(np.sum(np.abs(mutant) ** 2)) < float(
        np.sum(np.abs(out['zero']) ** 2))


def test_an_off_axis_window_is_refused_where_the_same_ratio_is_served(
):
    """ITEM 3, the REFUSAL, on the axis the branch's census does not walk.

    The guard's condition is ``2|centre_out| + N_out dx_out <= period``, so
    the ratio at which a window is refused DEPENDS ON THE OFFSET: a window
    that is served on axis is refused once it is pushed off it.  If the fill
    had leaked into the guard this is where it would show, and the census has
    to be identical under both fills here too -- not only on axis.

    Measured 2026-09-20: at 0.90 periods the window is served on axis and
    refused at 0.20 periods off it (``2*0.20 + 0.90 = 1.30`` periods of
    reach), while a 0.50-period window at the same offset is still served
    (``1.30 -> 0.90``) -- identically under ``'repeat'`` and ``'zero'``, on
    both builds.  The bar is the guard's own condition, re-evaluated here.
    """
    n = 128
    census = {}
    for wp in (0.50, 0.90, 1.00, 1.02):
        for coff in (0.0, 0.20):
            for fill in ('repeat', 'zero'):
                dxo = wp * _P3 / n
                try:
                    _read3(dxo, n, centre=(coff * _P3, 0.0), fill=fill,
                           on_replica='error')
                    verdict = 'served'
                except (ValueError, RuntimeError) as exc:
                    verdict = ('refused'
                               if 'ALIAS' in str(exc).upper() else 'other')
                census['%.2f/%.2f/%s' % (wp, coff, fill)] = verdict
    for key, verdict in census.items():
        wp, coff, _fill = key.split('/')
        twin = '%s/%s/%s' % (wp, coff,
                             'zero' if _fill == 'repeat' else 'repeat')
        assert census[twin] == verdict, (
            f"the refusal depends on the FILL at {key}: "
            f"{verdict} against {census[twin]}")
    assert census['0.90/0.00/zero'] == 'served', census
    assert census['0.90/0.20/zero'] == 'refused', (
        f"a window 0.20 periods off axis at 0.90 of the period was SERVED, "
        f"so the guard is not reading centre_out: {census}")
    assert census['0.50/0.20/zero'] == 'served', census
    assert census['1.00/0.00/zero'] == 'served', census
    assert census['1.02/0.00/zero'] == 'refused', census
    # the guard's own condition, re-evaluated: 2|c| + N dx <= period
    for key, verdict in census.items():
        wp, coff, _fill = (float(key.split('/')[0]),
                           float(key.split('/')[1]), key.split('/')[2])
        reach = 2.0 * coff + wp
        assert verdict == ('served' if reach <= 1.0 + 1e-9 else 'refused'), (
            f"{key}: reach {reach:.4f} periods but the guard said "
            f"{verdict!r}")
