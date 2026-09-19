"""Wave-5 hygiene-2, H2-3 (audit item 20) -- the near-focus exact-kernel table.

WP-B11 section 2.20 built the fixture (a converging Gaussian carrier,
``f = 20 mm``, ``w0 = 15.915 um``, ``theta = 20.0 mrad``, evaluated 1 um .. 5 mm
short of the geometric focus) and computed the dropped quartic, but could not
publish the table: ``propagate_carrier_referenced`` takes and returns an
ENVELOPE referenced to a carrier, and two spellings of the reference
bookkeeping gave O(1) residuals and then a ``ValueError``.

So the bookkeeping is validated FIRST here, on cases whose answer is known
independently, and only the claims that survive that are made about the table.

THE FIXTURE IS DERIVED, NOT QUOTED.  ``theta = lambda/(pi w0)`` fixes
``lambda = pi w0 theta = 1.0000e-06 m``; ``zR = pi w0^2/lambda = 795.77 um``;
placing the input plane one focal length before the waist fixes
``q_in = -f - i zR`` and hence ``R_in``, ``w_in`` and -- this is the whole
bookkeeping -- the fact that the input ENVELOPE is exactly the real Gaussian
``exp(-r^2/w_in^2)``, because ``1/q = 1/R + i lambda/(pi w^2)`` splits into a
carrier and an amplitude with no cross term.

THE ORACLE is the whole-function ``q`` form
``E(r,z) = exp(i k z)/(1 + z/q) exp(i k r^2/(2(q+z)))`` in the convention
WP-B11 section 2.17 established for this library (``exp(-i omega t)``, Gouy a
RETARDATION).  It carries the absolute piston, so every comparison below is
PISTON-INCLUDED and a convention error cannot cancel.

WHAT THE ORACLE CAN AND CANNOT REFEREE.  It is a PARAXIAL solution.  So it can
say whether the propagator reproduces the paraxial truth when asked for the
paraxial kernel (``gap_kernel='fresnel'``), and it can measure how far the
exact kernel departs from that truth -- but it cannot say which of the two
kernels is more physical.  Every claim here is phrased accordingly.

The full table, both builds, is
``validation/probe_wave5_hyg2/near_focus_{win,wsl}.json`` and the report's
section H2-3.
"""
from __future__ import annotations

import numpy as np
import pytest

from lumenairy.propagators.carrier import (
    _collins_transport, carrier_referenced_reconstruct,
    propagate_carrier_referenced)

# --- the fixture, derived ---------------------------------------------------
W0 = 15.915e-6
THETA = 20.0e-3
LAM = float(np.pi * W0 * THETA)
ZR = float(np.pi * W0 ** 2 / LAM)
F = 20.0e-3
K = 2.0 * np.pi / LAM
N_IN = 512
DX_IN = 8e-6
EPS = float(np.finfo(np.float64).eps)

#: six output widths of window, so the truncated tail is exp(-9) = 1.2e-04 of
#: peak and the window stays inside the chirp-Z's own spatial period.
WINDOW_WIDTHS = 6.0


def _q_in():
    return complex(-F, -ZR)


def _w_of_q(q):
    return float(np.sqrt(LAM / (np.pi * np.imag(1.0 / q))))


def _R_of_q(q):
    re = float(np.real(1.0 / q))
    return float('inf') if re == 0.0 else 1.0 / re


def _axis(n, d):
    return (np.arange(int(n), dtype=np.float64) - int(n) / 2) * float(d)


def _q_field(xo, yo, q_in, z):
    q2 = q_in + z
    xx, yy = np.meshgrid(xo, yo, indexing='xy')
    r2 = xx ** 2 + yy ** 2
    return (np.exp(1j * K * z) / (1.0 + z / q_in)
            * np.exp(1j * K * r2 / (2.0 * q2)))


def _pitch_for(w, R, n=N_IN):
    """Window, amplitude and curvature, whichever binds -- see the probe."""
    p_win = WINDOW_WIDTHS * w / float(n)
    p_amp = w / 8.0
    p_curv = (LAM * abs(R) / (4.0 * w)) if np.isfinite(R) else p_amp
    return float(min(p_win, p_amp, p_curv))


def _floor_bar(z, n_out, dx_out, w_out, decades=10.0):
    """The oracle's OWN error floor, derived, times a stated factor.

    ``eps * k * |z|`` is the representation error of the absolute piston
    ``exp(i k z)`` before any physics (2.8e-11 at z = 20 mm, lambda = 1 um);
    ``exp(-(N dx/2)^2 / w^2)`` is the Gaussian tail the grid truncates, which
    the propagator transports and the oracle does not.  Their sum is a floor,
    not an estimate of the residual, so the bar is ten times it -- the FFT
    chain in between adds its own ``eps log2 N`` on a quantity of order one.
    The gap that matters is to the failure mode, and a wrong carrier spelling
    gives an O(1) residual twelve decades above.
    """
    piston = EPS * K * abs(float(z))
    edge = float(n_out) * float(dx_out) / 2.0
    trunc = float(np.exp(-(edge / w_out) ** 2)) if w_out > 0 else 1.0
    return decades * (piston + trunc)


def _rel(got, ref):
    return float(np.linalg.norm(np.asarray(got) - np.asarray(ref))
                 / np.linalg.norm(np.asarray(ref)))


@pytest.fixture(scope="module")
def fixture_env():
    q = _q_in()
    w_in = _w_of_q(q)
    x = _axis(N_IN, DX_IN)
    env = np.exp(-(x[None, :] ** 2 + x[:, None] ** 2)
                 / w_in ** 2).astype(np.complex128)
    return q, _R_of_q(q), w_in, env


# ===========================================================================
# 1.  The bookkeeping, on cases with a known answer -- DECISIONS
# ===========================================================================

def test_the_fixture_is_the_one_the_three_numbers_determine():
    """``w0``, ``theta`` and ``f`` fix everything else; nothing below is a
    quoted constant.  Pinned so a future edit that changes one of the three
    cannot leave the derived quantities behind."""
    assert LAM == pytest.approx(np.pi * W0 * THETA, rel=0, abs=0)
    assert ZR == pytest.approx(np.pi * W0 ** 2 / LAM, rel=1e-15)
    assert LAM == pytest.approx(1.0e-6, rel=1e-4)
    assert ZR == pytest.approx(795.77e-6, rel=1e-4)
    q = _q_in()
    assert _R_of_q(q) == pytest.approx(-(F ** 2 + ZR ** 2) / F, rel=1e-12)
    assert _w_of_q(q) == pytest.approx(
        np.sqrt(LAM * (F ** 2 + ZR ** 2) / (np.pi * ZR)), rel=1e-12)


def test_a_collimated_gaussian_lands_at_the_oracle_floor_on_the_paraxial_arm():
    """CASE A, the case with no carrier at all: the envelope IS the field, so
    the only way to get an O(1) residual is to apply a carrier that should not
    be there.

    ``gap_kernel='fresnel'`` is the arm compared with the floor, because the
    oracle is a paraxial solution and 'fresnel' is the paraxial kernel.
    """
    q_a = complex(0.0, -ZR)
    w_a = _w_of_q(q_a)
    z = 1.5 * ZR
    w_out = _w_of_q(q_a + z)
    dx = 10.0 * w_out / N_IN
    x = _axis(N_IN, dx)
    env = np.exp(-(x[None, :] ** 2 + x[:, None] ** 2)
                 / w_a ** 2).astype(np.complex128)
    got = propagate_carrier_referenced(env, float('inf'), z, LAM, dx,
                                       gap_kernel='fresnel')
    fld = carrier_referenced_reconstruct(got.env, got.R, LAM, got.dx)
    xo = _axis(N_IN, got.dx)
    rel = _rel(fld, _q_field(xo, xo, q_a, z))
    bar = _floor_bar(z, N_IN, got.dx, w_out)
    assert rel < bar, (f"collimated paraxial arm {rel:.3e} past the derived "
                       f"floor bar {bar:.3e}")


def test_the_exact_kernels_departure_on_a_collimated_leg_is_the_quartic():
    """The same case with ``gap_kernel='auto'`` (which resolves to the EXACT
    kernel) does NOT land at the oracle's floor, and the amount by which it
    misses is the beam's own dropped quartic ``k z theta^4 / 8``.

    That is the independent confirmation that the bookkeeping is right AND
    that the exact kernel does what theory says: the residual is not a bug, it
    is the paraxial oracle's own error, of the predicted size.  Asserted as a
    RATIO within a factor of two, which is the honest claim for an L2 average
    against a peak-angle prediction.
    """
    q_a = complex(0.0, -ZR)
    w_a = _w_of_q(q_a)
    z = 1.5 * ZR
    w_out = _w_of_q(q_a + z)
    dx = 10.0 * w_out / N_IN
    x = _axis(N_IN, dx)
    env = np.exp(-(x[None, :] ** 2 + x[:, None] ** 2)
                 / w_a ** 2).astype(np.complex128)
    theta = LAM / (np.pi * w_a)
    predicted = K * abs(z) * theta ** 4 / 8.0
    got = propagate_carrier_referenced(env, float('inf'), z, LAM, dx,
                                       gap_kernel='auto')
    fld = carrier_referenced_reconstruct(got.env, got.R, LAM, got.dx)
    xo = _axis(N_IN, got.dx)
    rel = _rel(fld, _q_field(xo, xo, q_a, z))
    assert 0.5 < rel / predicted < 2.0, (
        f"the exact kernel's departure from the paraxial oracle is "
        f"{rel:.4e}, not the predicted quartic {predicted:.4e} "
        f"(ratio {rel / predicted:.4f})")


def test_both_spellings_of_the_output_reference_give_the_same_field(
        fixture_env):
    """CASE B, far from focus: the two ways to read the transport's output
    must agree, and both must land at the oracle floor.

    ``carrier_out=inf`` asks the Collins transport to return the reconstructed
    FIELD on the chosen lattice; the default geometric carrier returns an
    ENVELOPE that :func:`carrier_referenced_reconstruct` rebuilds.  These are
    the two spellings WP-B11 could not get right.
    """
    q, R_in, _w_in, env = fixture_env
    d = 5e-3
    z = F - d
    q_out = q + z
    w_out = _w_of_q(q_out)
    dx_out = _pitch_for(w_out, _R_of_q(q_out))
    common = dict(wavelength=LAM, dx=DX_IN, transport='collins',
                  gap_kernel='fresnel', dx_out=dx_out,
                  on_collins_sampling='warn')

    a = propagate_carrier_referenced(env, R_in, z,
                                     **dict(common, carrier_out=float('inf')))
    assert np.isinf(float(a.R)), (
        "carrier_out=inf must return a FLAT reference; it came back "
        f"{a.R!r}, so the returned array is not the field")
    b = propagate_carrier_referenced(env, R_in, z, **common)
    b_fld = carrier_referenced_reconstruct(b.env, b.R, LAM, b.dx)

    xo = _axis(N_IN, float(a.dx))
    ref = _q_field(xo, xo, q, z)
    bar = _floor_bar(z, N_IN, float(a.dx), w_out)
    rel_a, rel_b = _rel(a.env, ref), _rel(b_fld, ref)
    assert rel_a < bar, f"carrier_out=inf spelling {rel_a:.3e} past {bar:.3e}"
    assert rel_b < bar, f"reconstruct spelling {rel_b:.3e} past {bar:.3e}"
    assert abs(rel_a - rel_b) <= 1e-3 * max(rel_a, rel_b), (
        "the two spellings do not agree with each other; one of them is "
        "carrying the reference differently")


def test_the_sziklas_transport_reproduces_the_oracle_far_from_focus(
        fixture_env):
    """The OTHER transport, whose output pitch is forced to ``m*dx``, on the
    same case.  Its bookkeeping is the reconstruct spelling and nothing else,
    so this is the control for the test above."""
    q, R_in, _w_in, env = fixture_env
    d = 5e-3
    z = F - d
    w_out = _w_of_q(q + z)
    got = propagate_carrier_referenced(env, R_in, z, LAM, DX_IN,
                                       gap_kernel='fresnel')
    fld = carrier_referenced_reconstruct(got.env, got.R, LAM, got.dx)
    xo = _axis(N_IN, float(got.dx))
    rel = _rel(fld, _q_field(xo, xo, q, z))
    bar = _floor_bar(z, N_IN, float(got.dx), w_out)
    assert rel < bar, f"sziklas far-field {rel:.3e} past {bar:.3e}"


def test_a_double_applied_carrier_is_the_failure_mode_the_bar_excludes(
        fixture_env):
    """The falsification arm.  Reconstructing a FIELD (``carrier_out=inf``) as
    if it were an envelope applies the carrier twice -- one of the spellings
    that produced WP-B11's O(1) residuals.

    Asserted to be at least eight decades ABOVE the bars used in this file, so
    those bars are demonstrably discriminating and not merely small.
    """
    q, R_in, _w_in, env = fixture_env
    d = 5e-3
    z = F - d
    w_out = _w_of_q(q + z)
    dx_out = _pitch_for(w_out, _R_of_q(q + z))
    a = propagate_carrier_referenced(
        env, R_in, z, wavelength=LAM, dx=DX_IN, transport='collins',
        gap_kernel='fresnel', dx_out=dx_out, carrier_out=float('inf'),
        on_collins_sampling='warn')
    xo = _axis(N_IN, float(a.dx))
    ref = _q_field(xo, xo, q, z)
    right = _rel(a.env, ref)
    wrong = _rel(carrier_referenced_reconstruct(
        a.env, R_in + z, LAM, float(a.dx)), ref)
    # Stated against the CORRECT spelling rather than against the bar: the bar
    # at this rung is dominated by the window's own Gaussian truncation
    # (exp(-9) = 1.2e-04 at six widths), so "decades above the bar" would
    # understate the separation.  The two SPELLINGS are what the reader has to
    # tell apart, and they are twelve decades apart.
    assert wrong > 0.1, (
        f"the double-carrier spelling reads {wrong:.3e}; it is supposed to be "
        f"an O(1) failure, so this fixture no longer demonstrates one")
    assert wrong / right > 1e10, (
        f"the right spelling reads {right:.3e} and the wrong one "
        f"{wrong:.3e} -- only {wrong / right:.1e}x apart, so nothing here "
        f"discriminates the two")


# ===========================================================================
# 2.  The table -- premise-gated claims
# ===========================================================================

D_LADDER = (1e-6, 1e-5, 1e-4, 1e-3, 5e-3)


def _collins_row(env, R_in, q, d, gap_kernel):
    z = F - d
    q_out = q + z
    w_out = _w_of_q(q_out)
    dx_out = _pitch_for(w_out, _R_of_q(q_out))
    got = propagate_carrier_referenced(
        env, R_in, z, wavelength=LAM, dx=DX_IN, transport='collins',
        gap_kernel=gap_kernel, dx_out=dx_out, carrier_out=float('inf'),
        on_collins_sampling='ignore')
    xo = _axis(N_IN, float(got.dx))
    return _rel(got.env, _q_field(xo, xo, q, z)), \
        _floor_bar(z, N_IN, float(got.dx), w_out)


def test_the_paraxial_arm_holds_the_oracle_floor_all_the_way_to_the_focus(
        fixture_env):
    """The first publishable row of the table, and the one that makes the rest
    readable: on ``transport='collins'`` with ``gap_kernel='fresnel'``, the
    residual against the analytic Gaussian stays at the oracle floor from 5 mm
    short of the focus down to 1 um short of it.

    That is the near-focus claim WP-B11 could not make.  It is also what
    licenses reading the 'auto' / 'exact' rows as the exact kernel's own
    departure rather than as a transport failure.
    """
    q, R_in, _w, env = fixture_env
    for d in D_LADDER:
        rel, bar = _collins_row(env, R_in, q, d, 'fresnel')
        assert rel < bar, (
            f"collins/fresnel at d={d:.1e} reads {rel:.3e}, past the derived "
            f"floor bar {bar:.3e}")


def test_auto_and_exact_are_the_same_row_everywhere_on_the_ladder(
        fixture_env):
    """``gap_kernel='auto'`` resolving to anything other than 'exact' would be
    the fallback VERIFY-B4 F3 asks about.  Measured: it does not, anywhere on
    this ladder -- so the maintainer decision is a decision to CHANGE
    behaviour, not to ratify it."""
    q, R_in, _w, env = fixture_env
    for d in D_LADDER:
        a, _ = _collins_row(env, R_in, q, d, 'auto')
        e, _ = _collins_row(env, R_in, q, d, 'exact')
        assert a == e, (
            f"at d={d:.1e} 'auto' ({a:.6e}) and 'exact' ({e:.6e}) differ, so "
            f"'auto' took a fallback -- the F3 decision may already be made")


def test_the_kernel_gate_never_fires_on_this_fixture(fixture_env):
    """WHY 'auto' never falls back: the gate is ``k4 <= 1``, and ``k4`` is
    measured from the ENVELOPE's angular half-width, which is two and a half
    decades below the BEAM's 20 mrad because the carrier has been referenced
    out.  ``k4`` reads 2.2e-05 at 1 um from focus -- four decades below the
    gate.

    This is the correction the table makes to WP-B11 section 2.20's framing:
    the quartic that matters to a carrier-referenced leg is
    ``k |z_eff| theta_envelope^4 / 8``, not ``k |z_eff| theta_beam^4 / 8``, and
    the two differ by ``(theta_beam/theta_env)^4`` -- five orders of magnitude
    on this fixture.
    """
    q, R_in, _w, env = fixture_env
    theta_beam = THETA
    for d in D_LADDER:
        z = F - d
        w_out = _w_of_q(q + z)
        st = {}
        _collins_transport(
            env, R_in, z, LAM, DX_IN, DX_IN,
            dx_out=_pitch_for(w_out, _R_of_q(q + z)),
            dy_out=_pitch_for(w_out, _R_of_q(q + z)),
            N_out_x=N_IN, N_out_y=N_IN, R_ref=float('inf'),
            gap_kernel='auto', on_collins_sampling='ignore', stats_out=st)
        assert st['kernel'] == 'exact'
        assert st['k4'] < 1.0
        assert st['k4'] < 1e-3, (
            f"k4 = {st['k4']:.3e} at d={d:.1e} is within three decades of the "
            f"gate; the claim 'the gate never comes close' needs re-deriving")
        theta_env = max(st['theta_x'], st['theta_y'])
        assert theta_env < theta_beam / 5.0, (
            f"the envelope's measured angle {theta_env:.4e} is not well below "
            f"the beam's {theta_beam:.4e}; the premise of this test is gone")


def test_the_exact_kernels_departure_falls_monotonically_away_from_the_focus(
        fixture_env):
    """PREMISE-GATED.  The claim is that the exact kernel's departure from the
    paraxial answer decreases monotonically as the leg stops further from the
    focus -- which is what makes "fall back near a focus" a coherent rule at
    all.

    The premise is that there IS a departure to order: the span across the
    ladder must cover at least two decades.  If it does not (a build whose
    exact kernel degenerated, or a fixture that stopped reaching the focus),
    the test fails at the premise and says so, rather than passing on a flat
    line.
    """
    q, R_in, _w, env = fixture_env
    dep = [_collins_row(env, R_in, q, d, 'exact')[0] for d in D_LADDER]
    assert dep[0] / dep[-1] > 100.0, (
        f"PREMISE: the departure spans only {dep[0] / dep[-1]:.1f}x across "
        f"the ladder ({dep[0]:.3e} .. {dep[-1]:.3e}); there is no ordering to "
        f"assert")
    for i in range(len(dep) - 1):
        assert dep[i] > dep[i + 1], (
            f"departure is not monotonic: d={D_LADDER[i]:.1e} -> "
            f"{dep[i]:.3e}, d={D_LADDER[i + 1]:.1e} -> {dep[i + 1]:.3e}")


def test_the_departure_is_linear_in_the_reduced_distance(fixture_env):
    """THE DERIVED LAW, and the reason a threshold can be stated at all.

    The exact-kernel refinement is a diagonal phase over the REDUCED frame
    ``z_eff = B/A``, whose leading departure from the paraxial kernel is the
    dropped quartic ``k |z_eff| theta^4 / 8`` -- linear in ``z_eff``.  The
    ladder moves ``z_eff`` over two decades at fixed envelope, so the exponent
    is FITTED here and must come out 1.

    Bar: the fit's own worst relative deviation from the power law must be
    below 1e-3.  MEASURED 2026-09-15 on Windows py3.14 and WSL py3.12, 9 rungs
    spanning ``z_eff`` 0.0597 .. 12.27 m: slope 0.9999985, worst deviation
    1.3e-06 -- three decades of room.
    """
    q, R_in, _w, env = fixture_env
    zs, deps = [], []
    for d in (1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 5e-3):
        z = F - d
        w_out = _w_of_q(q + z)
        dx_out = _pitch_for(w_out, _R_of_q(q + z))
        st = {}
        common = dict(wavelength=LAM, dx=DX_IN, transport='collins',
                      dx_out=dx_out, carrier_out=float('inf'),
                      on_collins_sampling='ignore')
        a = propagate_carrier_referenced(env, R_in, z,
                                         **dict(common, gap_kernel='exact'))
        b = propagate_carrier_referenced(env, R_in, z,
                                         **dict(common, gap_kernel='fresnel'))
        _collins_transport(
            env, R_in, z, LAM, DX_IN, DX_IN, dx_out=dx_out, dy_out=dx_out,
            N_out_x=N_IN, N_out_y=N_IN, R_ref=float('inf'),
            gap_kernel='auto', on_collins_sampling='ignore', stats_out=st)
        A, B = float(st['abcd'][0]), float(st['abcd'][1])
        zs.append(abs(B / A))
        deps.append(_rel(a.env, b.env))
    assert max(zs) / min(zs) > 100.0, (
        f"PREMISE: z_eff spans only {max(zs) / min(zs):.1f}x; the exponent is "
        f"not determined")
    p = np.polyfit(np.log(zs), np.log(deps), 1)
    resid = np.log(deps) - np.polyval(p, np.log(zs))
    worst = float(np.max(np.abs(np.exp(resid) - 1.0)))
    assert worst < 1e-3, (
        f"the departure is not a power law in z_eff (worst deviation "
        f"{worst:.3e}); the derived threshold below rests on it")
    assert abs(p[0] - 1.0) < 1e-3, (
        f"the departure scales as z_eff^{p[0]:.6f}, not z_eff^1")


def test_the_departure_is_quartic_in_the_envelopes_own_angle(fixture_env):
    """The other half of the law, and the half that says WHOSE angle.

    At ONE distance to focus, the input envelope's width is scaled: a narrower
    envelope on the same carrier has a wider angular spectrum while the
    geometry -- and therefore ``z_eff`` -- is held.  The claim is an exponent
    of 4 against the envelope's ANALYTIC ``1/e^2`` half-angle
    ``lambda/(pi w_env)``.

    The output window is sized from EACH rung's own output width: sizing it
    from the fixture's clipped the narrow-envelope rungs and fitted 3.07 with
    55 per cent scatter, which is what a clipped fixture looks like.

    MEASURED 2026-09-15, both builds, 6 rungs over a 5.3x span of angle: slope
    3.99970, worst deviation 1.9e-04.
    """
    q, R_in, w_in, _env = fixture_env
    d = 1e-4
    z = F - d
    x = _axis(N_IN, DX_IN)
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    ths, deps = [], []
    for scale in (0.25, 0.35, 0.5, 0.7, 1.0, 1.4):
        w_env = scale * w_in
        q_e = 1.0 / (1.0 / R_in + 1j * LAM / (np.pi * w_env ** 2))
        w_out_e = _w_of_q(q_e + z)
        dx_out = _pitch_for(w_out_e, _R_of_q(q_e + z))
        env = np.exp(-r2 / w_env ** 2).astype(np.complex128)
        common = dict(wavelength=LAM, dx=DX_IN, transport='collins',
                      dx_out=dx_out, carrier_out=float('inf'),
                      on_collins_sampling='ignore')
        a = propagate_carrier_referenced(env, R_in, z,
                                         **dict(common, gap_kernel='exact'))
        b = propagate_carrier_referenced(env, R_in, z,
                                         **dict(common, gap_kernel='fresnel'))
        ths.append(float(LAM / (np.pi * w_env)))
        deps.append(_rel(a.env, b.env))
    assert max(ths) / min(ths) > 4.0
    p = np.polyfit(np.log(ths), np.log(deps), 1)
    resid = np.log(deps) - np.polyval(p, np.log(ths))
    worst = float(np.max(np.abs(np.exp(resid) - 1.0)))
    assert worst < 1e-2, (
        f"the departure is not a clean power law in the envelope angle "
        f"(worst deviation {worst:.3e})")
    assert abs(p[0] - 4.0) < 0.05, (
        f"the departure scales as theta^{p[0]:.5f}, not theta^4 -- the "
        f"quartic is the term the exact kernel adds, so this is the law the "
        f"threshold is derived from")


def test_the_measured_departure_is_the_quartic_times_one_constant(
        fixture_env):
    """The two halves together, as ONE number a threshold can be written with.

    ``departure_relL2 = C * k |z_eff| theta_env^4 / 8``.  ``C`` is fitted on
    the converging fixture and then checked, unchanged, against the
    COLLIMATED case of section 1 -- a different transport
    (``sziklas``), a different carrier (none) and a different geometry.  One
    constant serving both is what makes the law a law.

    MEASURED 2026-09-15: C = 1.228 on the converging ladder and 1.225 on the
    collimated leg, both builds.
    """
    q, R_in, w_in, env = fixture_env
    d = 1e-4
    z = F - d
    w_out = _w_of_q(q + z)
    dx_out = _pitch_for(w_out, _R_of_q(q + z))
    st = {}
    common = dict(wavelength=LAM, dx=DX_IN, transport='collins',
                  dx_out=dx_out, carrier_out=float('inf'),
                  on_collins_sampling='ignore')
    a = propagate_carrier_referenced(env, R_in, z,
                                     **dict(common, gap_kernel='exact'))
    b = propagate_carrier_referenced(env, R_in, z,
                                     **dict(common, gap_kernel='fresnel'))
    _collins_transport(env, R_in, z, LAM, DX_IN, DX_IN, dx_out=dx_out,
                       dy_out=dx_out, N_out_x=N_IN, N_out_y=N_IN,
                       R_ref=float('inf'), gap_kernel='auto',
                       on_collins_sampling='ignore', stats_out=st)
    z_eff = abs(float(st['abcd'][1]) / float(st['abcd'][0]))
    theta = float(LAM / (np.pi * w_in))
    quartic = K * z_eff * theta ** 4 / 8.0
    C_conv = _rel(a.env, b.env) / quartic

    q_a = complex(0.0, -ZR)
    w_a = _w_of_q(q_a)
    z2 = 1.5 * ZR
    dx2 = 10.0 * _w_of_q(q_a + z2) / N_IN
    x2 = _axis(N_IN, dx2)
    env2 = np.exp(-(x2[None, :] ** 2 + x2[:, None] ** 2)
                  / w_a ** 2).astype(np.complex128)
    e2 = propagate_carrier_referenced(env2, float('inf'), z2, LAM, dx2,
                                      gap_kernel='exact')
    f2 = propagate_carrier_referenced(env2, float('inf'), z2, LAM, dx2,
                                      gap_kernel='fresnel')
    C_coll = (_rel(e2.env, f2.env)
              / (K * abs(z2) * (LAM / (np.pi * w_a)) ** 4 / 8.0))

    assert 0.5 < C_conv < 3.0, f"C on the converging ladder is {C_conv:.4f}"
    assert abs(C_conv - C_coll) / C_conv < 0.05, (
        f"the constant is not shared: converging {C_conv:.4f} vs collimated "
        f"{C_coll:.4f}; the law does not carry across geometries")


def test_the_sziklas_transport_loses_the_focus_and_the_collins_one_does_not(
        fixture_env):
    """The near-focus comparison the table exists to make, at FIXED
    ``gap_kernel`` so the two transports are the only thing moving.

    The Sziklas step's output pitch is forced to ``m*dx`` with ``m -> 0`` at
    the geometric focus, so its window collapses faster than the beam does;
    the Collins step chooses its own pitch.  Measured at 1 um from focus:
    Sziklas 7.3e-04 against Collins 9.8e-13, nine decades.

    Premise-gated on the pitch actually collapsing, so the claim cannot pass
    for some other reason.
    """
    q, R_in, _w, env = fixture_env

    def _sziklas(d):
        z = F - d
        sz = propagate_carrier_referenced(env, R_in, z, LAM, DX_IN,
                                          gap_kernel='fresnel')
        fld = carrier_referenced_reconstruct(sz.env, sz.R, LAM, sz.dx)
        xo = _axis(N_IN, float(sz.dx))
        return sz, _rel(fld, _q_field(xo, xo, q, z)),             _floor_bar(z, N_IN, float(sz.dx), _w_of_q(q + z))

    # PREMISE 1 -- the near rung really does reach the auto-split.  At 1 um
    # from focus the co-moving magnification is m = 1.63e-03, so an unsplit
    # leg would return a pitch of m*dx = 1.30e-08 m; the leg returns
    # 1.9098e-06 m instead (MEASURED 2026-09-15, both builds), i.e. the
    # carrier -> through-waist ASM bridge -> carrier split ran.  That bridge,
    # not a collapsed grid, is what this row is about.
    d_near = 1e-6
    sz_near, rel_sz_near, _ = _sziklas(d_near)
    m = abs((R_in + (F - d_near)) / R_in)
    assert float(sz_near.dx) > 10.0 * m * DX_IN, (
        f"PREMISE: the returned pitch {float(sz_near.dx):.4e} is the plain "
        f"co-moving m*dx = {m * DX_IN:.4e}; the auto-split did not engage, so "
        f"this row is measuring something else")

    # PREMISE 2 -- the SAME transport is at its own floor far from focus, so
    # what the near rung shows is a near-focus effect and not a broken leg.
    _sz_far, rel_sz_far, bar_far = _sziklas(5e-3)
    assert rel_sz_far < bar_far, (
        f"PREMISE: the Sziklas transport reads {rel_sz_far:.3e} against a "
        f"floor bar of {bar_far:.3e} even 5 mm from focus; the degradation "
        f"below is not attributable to the focus")

    # THE CLAIM.
    rel_co, _bar = _collins_row(env, R_in, q, d_near, 'fresnel')
    assert rel_co < rel_sz_near / 1e6, (
        f"at 1 um from focus the Collins transport reads {rel_co:.3e} and the "
        f"Sziklas one {rel_sz_near:.3e}; the six-decade separation this row "
        f"publishes is gone")
    assert rel_sz_near / rel_sz_far > 1e6, (
        f"the Sziklas transport reads {rel_sz_near:.3e} at 1 um and "
        f"{rel_sz_far:.3e} at 5 mm; the near-focus degradation this row is "
        f"about is gone")
