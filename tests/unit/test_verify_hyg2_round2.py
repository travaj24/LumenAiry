"""VERIFY-WAVE5-HYGIENE2 ROUND 2 -- the gates this verification found missing.

Seven ids, one per gap measured in
``docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/
VERIFY_WAVE5_HYGIENE2_ROUND2.md``.  Every bar here is derived at runtime from a
quantity the running build measures, two-sided, and premise-gated; the probe
that produced the numbers quoted in each docstring is
``validation/probe_verify_hyg2_round2/``.

WHY EACH ONE EXISTS, in one line:

* D-1  a phase-budget reference that forms its phase the way the DENSE route
       does cannot see the dense route's phase error, and the shipped
       ``method='direct'`` advice rests on exactly that reading;
* D-2  the ``(s.q)/N`` chief-ray term inside the ONE consolidated kernel is
       reachable by every tilted leg and gated by nothing -- deleting it moves
       136 of this verification's 342 byte-identity keys and leaves all 216
       ids of the five touched test files green;
* D-3  a SITE-LOCAL conjugation of the kernel is still writable (the
       consolidation removes the second COPY, not the unary minus), so the
       absence of one is asserted structurally as well as numerically;
* D-4  the evanescent clamp is only reached at a sub-wavelength pitch, which
       no shipped fixture uses, and the evanescent REFUSAL beside it is
       gated only by a token census;
* D-5  the cross-backend bar reduces to its ``32 * eps`` floor on both
       builds, so it is not today a function of the two backends it names;
* D-6  ``_GAP_KERNEL_ACCURACY_TAU = None`` is asserted by reading the constant
       and by the absence of a stats key, never by the rule not RUNNING.
"""
from __future__ import annotations

import ast
import cmath
import math
import pathlib
import warnings
from fractions import Fraction

import numpy as np
import pytest

import lumenairy.propagators._bluestein as BL
import lumenairy.propagators.carrier as CA

EPS = float(np.finfo(np.float64).eps)
TAU = 2.0 * math.pi
WL = 1.55e-6
K = 2.0 * np.pi / WL


# ===========================================================================
# D-2 / D-3.  The ONE kernel: its chief-ray term, and its sign at each site
# ===========================================================================

def test_the_chief_ray_term_inside_the_one_kernel_is_gated_numerically():
    r"""``_exact_dispersion_phase`` must have NO term linear in ``q``.

    THE PHYSICS.  The kernel returns the ``q``-dependent remainder of the
    non-paraxial dispersion expanded about the carrier wavevector ``k s``::

        sqrt(k^2 - |k s + q|^2) - k N + (s.q)/N
            = -|q|^2/(2 k N) - (s.q)^2/(2 k N^3) + O(q^3/k^2)

    The ``(s.q)/N`` term exists precisely to cancel the radical's own LINEAR
    term, because the caller already applies the chief-ray advance
    ``x_c += L z / N`` in real space.  So the kernel's odd part in ``q`` must
    vanish to third order, and that is a property of the physics rather than
    of the spelling: a token census cannot see it.

    THE STATISTIC.  Along ``x`` at a small ``q0``, take the antisymmetric and
    symmetric parts of the kernel at ``+-q0``.  The symmetric part IS the
    quadratic term; the first surviving odd term is ``O(q0^3)``, i.e. of order
    ``|sym| * (q0/k)``; and the arithmetic floor is ``eps * k`` (the kernel
    subtracts two numbers of size ``k N``).  The ratio

        |anti| / max(eps k, |sym| q0/k)

    is therefore of order one when the term is present.

    MEASURED 2026-09-20 on both builds, over four tilts and three ``q0``:
    **0.000 .. 0.345**.  BAR 10 -- 1.5 decades above the worst reading.  THE
    OTHER SIDE: with the term deleted the odd part becomes ``-L q0 / N``
    exactly, and the same ratio reads **1.0e+05 .. 2.3e+08** on the same
    rungs, four decades above the bar at its weakest.  Deleting it moves 136
    of 342 byte-identity keys, so the mutation is live and not cosmetic.
    """
    worst = 0.0
    rows = []
    for tilt in ((0.0, 0.0), (0.05, 0.0), (0.11, 0.07), (0.3, -0.2)):
        L, M = tilt
        Nz = float(np.sqrt(1.0 - L * L - M * M))
        for frac in (1e-3, 1e-4, 1e-6):
            q0 = K * frac
            qx = np.array([-q0, 0.0, q0])
            qy = np.array([0.0])
            ph = np.asarray(CA._exact_dispersion_phase(
                qx, qy, K, tilt, np, 'test_verify_hyg2_round2'))[0]
            anti = float(ph[2] - ph[0]) / 2.0
            sym = float(ph[2] + ph[0]) / 2.0
            floor = EPS * K
            denom = max(floor, abs(sym) * frac)
            ratio = abs(anti) / denom
            rows.append((tilt, frac, anti, sym, ratio))
            worst = max(worst, ratio)
            # PREMISE: the quadratic term is real and far above the floor, or
            # the ratio below is a statement about round-off.
            assert abs(sym) > 100.0 * floor, (
                f"PREMISE: at tilt={tilt} q0/k={frac:.0e} the kernel's "
                f"quadratic part {sym:.4e} is within 100x the arithmetic "
                f"floor {floor:.4e}; this rung cannot discriminate")

    assert worst < 10.0, (
        f"the exact-dispersion kernel carries a term LINEAR in q: the "
        f"antisymmetric part reaches {worst:.4g}x the third-order scale it "
        f"should sit at (measured 0.000 .. 0.345 on 2026-09-20).  The "
        f"(s.q)/N chief-ray subtraction is what cancels that term, and the "
        f"caller applies the chief-ray advance x_c += L z/N in real space, "
        f"so a linear term here is applied TWICE.  Rows: "
        + "; ".join(f"tilt={t} q0/k={f:.0e} anti={a:.3e} ratio={r:.3g}"
                    for t, f, a, _s, r in rows))

    # THE OTHER SIDE, engineered rather than hoped for: the ratio a dropped
    # term would produce, computed from the same readings.  It must be
    # decades above the bar, or the bar is not discriminating.
    gaps = []
    for tilt, frac, _anti, sym, _r in rows:
        L, M = tilt
        if L == 0.0 and M == 0.0:
            continue
        Nz = float(np.sqrt(1.0 - L * L - M * M))
        dropped = abs(L * K * frac / Nz)
        gaps.append(dropped / max(EPS * K, abs(sym) * frac))
    assert min(gaps) > 1e3, (
        f"the failure this bar must catch -- the (s.q)/N term deleted -- "
        f"would read only {min(gaps):.3g}x on its weakest rung, against a "
        f"bar of 10; the two readings are not separable")


def test_no_call_site_negates_the_one_kernel():
    """A SITE-LOCAL conjugation is still writable; assert there is not one.

    V-D22 removed the second and third COPIES of the dispersion.  It did not
    and could not remove the unary minus: ``phase = -_exact_dispersion_phase(
    ...)`` at one site is four characters and reintroduces exactly the
    cross-site disagreement the consolidation was supposed to end.  MEASURED
    2026-09-20 on the merged tip: written at
    ``_collins_exact_kernel_correction`` it moves 63 of 342 byte-identity keys
    and 13 ids across the five touched files DO catch it -- so this id is not
    the only guard, it is the one that says WHICH four characters.

    The check is on the AST with docstrings stripped, so a cross-reference in
    prose is not a call and a re-spelling (``- _exact_dispersion_phase``,
    ``(-1) * _exact_dispersion_phase``) is still one.
    """
    src = pathlib.Path(CA.__file__).read_text(encoding='cp1252')
    tree = ast.parse(src)
    sites = {'_exact_tf_2d_xp', '_exact_envelope_tf_step',
             '_collins_exact_kernel_correction'}
    seen = {}
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        body = ast.Module(body=fn.body, type_ignores=[])
        # strip the docstring so a :func: reference is not parsed as code
        if (body.body and isinstance(body.body[0], ast.Expr)
                and isinstance(body.body[0].value, ast.Constant)):
            body.body = body.body[1:]
        calls = [n for n in ast.walk(body)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == '_exact_dispersion_phase']
        if not calls:
            continue
        negated = [ast.unparse(n) for n in ast.walk(body)
                   if isinstance(n, ast.UnaryOp)
                   and isinstance(n.op, ast.USub)
                   and '_exact_dispersion_phase' in ast.unparse(n.operand)]
        scaled = [ast.unparse(n) for n in ast.walk(body)
                  if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult)
                  and '_exact_dispersion_phase' in ast.unparse(n)
                  and isinstance(n.left, ast.UnaryOp)]
        seen[fn.name] = {'n_calls': len(calls),
                         'negated': negated + scaled}

    # PREMISE: the census is reading the module, and all three sites call it.
    assert set(seen) >= sites, (
        f"only {sorted(seen)} call _exact_dispersion_phase; the three sites "
        f"{sorted(sites)} must all reach the one kernel")
    for name, info in seen.items():
        assert not info['negated'], (
            f"{name} applies a unary minus to _exact_dispersion_phase "
            f"({info['negated']}).  The consolidation removed the second "
            f"COPY of the dispersion, not the sign: one negated call site "
            f"reintroduces the cross-site disagreement V-D22 ended, and the "
            f"sign pins in test_wave5_h2_near_focus_table.py and "
            f"test_verify_wave5_hyg2.py only read the COLLINS leg.")


# ===========================================================================
# D-4.  The evanescent clamp, at a pitch that reaches the band edge
# ===========================================================================

def test_the_evanescent_clamp_is_reached_and_holds_at_a_sub_wavelength_pitch():
    """``maximum(rad, 0)`` bites only when ``pi/dx > k``, i.e. ``dx < lam/2``.

    Every carrier fixture in this campaign has ``dx`` of micrometres against a
    wavelength of 1.55 um, so ``q_max/k`` is 0.25 and the clamp is never
    reached -- deleting it moves NONE of this verification's 342 byte-identity
    keys and leaves the five touched test files green.  MEASURED 2026-09-20,
    both builds: with the clamp deleted and ``dx = lam/4``, the kernel and the
    step both return NaN; with it, the step's norm is finite and equal to the
    well-sampled case's.

    The PREMISE is asserted first and separately: the grid really does resolve
    frequencies past ``k``, and a non-empty set of them is clamped.
    """
    n = 64
    for dx_over_lam, expect_clamped in ((0.25, True), (0.4, True),
                                        (2.0, False)):
        dx = dx_over_lam * WL
        qx = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
        qmax = float(np.max(np.abs(qx)))
        for tilt in ((0.0, 0.0), (0.3, -0.2)):
            L, M = tilt
            root0 = float(np.sqrt(max(K * K * (1.0 - L * L - M * M), 0.0)))
            ph = np.asarray(CA._exact_dispersion_phase(
                qx, qx, K, tilt, np, 'test_verify_hyg2_round2'))
            n_clamped = int(np.sum(np.isclose(ph, -root0, rtol=0.0,
                                              atol=1e-9 * root0)))
            # PREMISE: this rung is (or is not) in the evanescent regime for
            # the stated reason, measured rather than assumed.
            assert (qmax > K) == expect_clamped, (
                f"PREMISE: dx = {dx_over_lam} lam gives q_max/k = "
                f"{qmax / K:.4f}; this rung is not the regime it is here for")
            if expect_clamped:
                assert n_clamped > 0, (
                    f"PREMISE: dx = {dx_over_lam} lam, tilt {tilt}: no "
                    f"frequency was clamped, so this rung does not exercise "
                    f"the evanescent band at all")
            assert np.all(np.isfinite(ph)), (
                f"the kernel returned a non-finite phase at dx = "
                f"{dx_over_lam} lam, tilt {tilt}: k^2 - |k s + q|^2 went "
                f"negative and was not clamped to zero, so the evanescent "
                f"band produces NaN instead of a band limit")
            x = (np.arange(n) - n / 2.0) * dx
            env = np.exp(-(x[None, :] ** 2 + x[:, None] ** 2)
                         / (8 * dx) ** 2).astype(np.complex128)
            out = CA._exact_envelope_tf_step(env, 1e-5, WL, dx, dx, tilt=tilt)
            assert np.all(np.isfinite(out)), (
                f"the envelope step returned NaN at dx = {dx_over_lam} lam, "
                f"tilt {tilt}")



def test_the_evanescent_carrier_refusal_is_gated_by_behaviour():
    """``|s|^2 < 1`` must be a BEHAVIOUR, not a token the census greps for.

    MEASURED 2026-09-20: weakening the guard to ``if not (s2 <= 1.0)`` -- so
    the grazing ``|s| = 1`` direction is ACCEPTED instead of refused -- is
    caught by exactly ONE id on each build, and on both it is
    ``test_wave5_h2_collins_jax.py::test_the_exact_dispersion_is_written_once``
    -- the single-definition census, which notices only because the literal
    string ``s2 < 1.0`` is one of the three tokens it greps for.  No numerical
    id notices that the refusal stopped refusing, and my 342-key byte-identity
    probe sees it as exactly one kind flip.

    A census is the wrong instrument for a behaviour.  Re-spell the guard as
    ``if s2 >= 1.0: raise`` and the census fails while the behaviour is
    correct; keep the token and change the comparison somewhere else and the
    census passes while the behaviour is wrong.  This id asserts the two
    decisions instead, and it asserts them at BOTH sides of the boundary.

    ``N = sqrt(1 - |s|^2)`` divides the chief-ray term, so ``|s| = 1`` is a
    division by zero and not merely an unphysical input; the refusal is what
    keeps an infinity out of the kernel.
    """
    qx = 2.0 * np.pi * np.fft.fftfreq(16, d=2e-6)
    qy = qx

    # --- REFUSED: |s|^2 >= 1 -------------------------------------------
    for tilt in ((1.0, 0.0), (0.0, -1.0), (0.6, 0.8), (0.8, 0.8), (2.0, 0.0)):
        s2 = tilt[0] ** 2 + tilt[1] ** 2
        assert s2 >= 1.0, f"PREMISE: tilt {tilt} has |s|^2 = {s2} < 1"
        with pytest.raises(ValueError) as exc:
            CA._exact_dispersion_phase(qx, qy, K, tilt, np, 'PROBE_NAME')
        msg = str(exc.value)
        assert 'PROBE_NAME' in msg, (
            f"the refusal for tilt {tilt} does not name its caller: {msg!r}")

    # --- ACCEPTED: just inside the boundary ----------------------------
    # Engineered, not hoped for: take |s| one ULP below 1 along x.
    near = float(np.nextafter(1.0, 0.0))
    for tilt in ((near, 0.0), (0.0, -near), (0.9999999, 0.0)):
        s2 = tilt[0] ** 2 + tilt[1] ** 2
        assert s2 < 1.0, f"PREMISE: tilt {tilt} has |s|^2 = {s2} >= 1"
        ph = np.asarray(CA._exact_dispersion_phase(
            qx, qy, K, tilt, np, 'test_verify_hyg2_round2'))
        assert np.all(np.isfinite(ph)), (
            f"tilt {tilt} is INSIDE the propagating cone (|s|^2 = {s2!r} < 1) "
            f"and the kernel returned a non-finite phase; the guard has "
            f"started refusing -- or admitting -- the wrong side")

    # --- and the two shipped phrasings are both reachable ---------------
    msgs = {}
    for fn in ('_exact_tf_2d_xp', '_exact_envelope_tf_step'):
        with pytest.raises(ValueError) as exc:
            CA._exact_dispersion_phase(qx, qy, K, (1.0, 0.0), np, fn)
        msgs[fn] = str(exc.value)
    assert msgs['_exact_tf_2d_xp'] != msgs['_exact_envelope_tf_step'], (
        "the two shipped refusal phrasings have collapsed into one; both are "
        "reproduced verbatim on purpose, because an archive-to-archive "
        "byte-identity probe folds an exception's MESSAGE into its digest")

# ===========================================================================
# D-6.  "OFF" asserted by the rule not RUNNING
# ===========================================================================

def test_the_accuracy_rule_is_never_executed_while_tau_is_none(monkeypatch):
    """``_GAP_KERNEL_ACCURACY_TAU is None`` means the rule does not RUN.

    The shipped id reads the constant and asserts that ``stats_out`` has not
    grown a ``kernel_departure`` key.  Both are consequences; neither says the
    rule was not evaluated, and a future edit that measured the half-angle and
    then discarded it would satisfy both while paying for a full-grid FFT
    moment on every ``'exact'`` leg.

    Here the two functions the rule calls are replaced by raising sentinels,
    so "not executed" is the assertion itself.  MEASURED 2026-09-20 on both
    builds: zero calls on ``'auto'``, ``'exact'`` and ``'fresnel'``, and the
    rule's four source lines are absent from a ``sys.settrace`` line trace of
    the same legs while the guard line that short-circuits them IS present --
    which is what makes the trace live rather than vacuous.
    """
    assert CA._GAP_KERNEL_ACCURACY_TAU is None, (
        "the accuracy-keyed fallback is ARMED in the shipped module")

    calls = []

    def boom_half(*a, **k):
        calls.append('_collins_envelope_half_angle')
        raise AssertionError('the accuracy rule measured the envelope '
                             'half-angle while tau is None')

    def boom_dep(*a, **k):
        calls.append('_collins_exact_kernel_departure')
        raise AssertionError('the accuracy rule evaluated the departure '
                             'while tau is None')

    monkeypatch.setattr(CA, '_collins_envelope_half_angle', boom_half)
    monkeypatch.setattr(CA, '_collins_exact_kernel_departure', boom_dep)

    n, dx = 128, 4e-6
    x = (np.arange(n) - n / 2.0) * dx
    env = np.exp(-(x[None, :] ** 2 + x[:, None] ** 2)
                 / (40e-6) ** 2).astype(np.complex128)
    kernels_seen = []
    for gk in ('auto', 'exact', 'fresnel'):
        st = {}
        CA._collins_transport(env, 0.05, 5e-3, WL, dx, dx, dx_out=dx,
                              dy_out=dx, N_out_x=n, N_out_y=n,
                              R_ref=float('inf'), gap_kernel=gk,
                              on_collins_sampling='ignore', stats_out=st)
        kernels_seen.append(st['kernel'])
        assert 'kernel_departure' not in st, (
            f"gap_kernel={gk!r} grew a kernel_departure key while the rule "
            f"is off; the stats dict is a byte-identity key of this campaign")
    assert not calls, (
        f"the accuracy rule called {calls} while "
        f"_GAP_KERNEL_ACCURACY_TAU is None")
    # PREMISE: at least one of those legs actually RESOLVED to 'exact', or
    # the rule had nothing to guard and this id proves nothing.
    assert 'exact' in kernels_seen, (
        f"PREMISE: none of the three legs resolved to the exact kernel "
        f"({kernels_seen}); the accuracy rule only guards 'exact', so this "
        f"fixture cannot say whether it would have run")


# ===========================================================================
# D-5.  A cross-backend bar built on a quantity that reads exactly zero
# ===========================================================================

def test_the_cross_backend_bar_is_the_legs_own_last_bit_sensitivity():
    """A bar derived from a quantity that is not the `32 * eps` FLOOR.

    ``test_wave5_h2_collins_jax.py::_fft_spread_bar`` takes one forward
    transform through each backend, multiplies by a chain depth of 6, and
    floors the result at ``32 * eps``.  MEASURED 2026-09-20, the shipped
    fixture, both builds:

        library ``_fft2`` vs ``jnp.fft.fft2``   2.685e-16 (WIN) 2.509e-16 (WSL)
        ``np.fft.fft2``   vs ``jnp.fft.fft2``   EXACTLY 0.0 on both
        ``_fft_spread_bar(env)``                7.105427357601002e-15, i.e.
                                                ``32 * eps`` EXACTLY, on both

    So the floor is what is asserted against -- 4.41x above ``6 x`` the
    measured spread, which is the ratio the author's V-D16 entry already
    records -- and the whole of the nonzero reading comes from pyFFTW sitting
    on the NumPy side: the two libraries' own transforms agree BIT FOR BIT.
    Nothing is wrong with that bar; it is simply not, today, a function of the
    backends it names.

    This id asserts the same comparison against a quantity that is a property
    of the LEG rather than of which FFT wrapper NumPy happens to use: the
    leg's own response to a one-ULP change of its input, which is exactly what
    a cross-backend difference IS.  MEASURED: 3.454e-16 (WIN) / 3.469e-16
    (WSL), against cross-backend differences of 5.486e-16 / 4.029e-16 through
    the public Collins leg.  Two-sided, as the shipped id is: the bar must
    also sit decades below the smallest real signal on the fixture, the
    exact-vs-paraxial kernel departure at 3.078e-05.

    The PREMISE is the shipped id's: the two arms must be two backends.  It
    is what M10 (the V-D3 regression) trips.
    """
    jnp = pytest.importorskip("jax.numpy")
    jax = pytest.importorskip("jax")
    jax.config.update('jax_enable_x64', True)

    n, dx, R_in, z = 128, 3e-6, 0.04, 5e-3
    x = (np.arange(n) - n / 2.0) * dx
    env = np.exp(-(x[None, :] ** 2 + x[:, None] ** 2)
                 / (48e-6) ** 2).astype(np.complex128)
    kw = dict(dx_out=dx, on_collins_sampling='ignore', transport='collins')

    def rel(a, b):
        a, b = np.asarray(a), np.asarray(b)
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))

    a = CA.propagate_carrier_referenced(env, R_in, z, WL, dx, **kw)
    b = CA.propagate_carrier_referenced(jnp.asarray(env), R_in, z, WL, dx,
                                        **kw)
    # PREMISE, as the shipped id has it: two backends, not one twice.
    assert not isinstance(b.env, np.ndarray), (
        f"the JAX arm came back as {type(b.env).__name__}: the public leg "
        f"demoted the caller's array and the comparison is vacuous")

    # THE BAR, from a quantity that is not zero: perturb every entry of the
    # envelope by one ULP and measure how far the answer moves.
    env_ulp = np.nextafter(env.real, np.inf) + 1j * env.imag
    ulp = rel(CA.propagate_carrier_referenced(env_ulp, R_in, z, WL, dx,
                                              **kw).env, a.env)
    assert ulp > 0.0, (
        f"PREMISE: the leg's response to a 1-ULP input change reads "
        f"{ulp:.3e}; a bar derived from it would be zero too")
    bar = 10.0 * ulp
    got = rel(b.env, a.env)
    assert got <= bar, (
        f"JAX vs NumPy through the public Collins leg is {got:.4e}, above "
        f"10x the leg's own measured 1-ULP sensitivity {ulp:.4e}")

    # Two-sided: the bar must be decades below the smallest real signal.
    sig = rel(CA.propagate_carrier_referenced(
        env, R_in, z, WL, dx, **dict(kw, gap_kernel='exact')).env,
        CA.propagate_carrier_referenced(
            env, R_in, z, WL, dx, **dict(kw, gap_kernel='fresnel')).env)
    assert bar < sig / 1e3, (
        f"the bar {bar:.3e} is not three decades below the smallest real "
        f"signal on this fixture ({sig:.3e}, the exact-vs-paraxial kernel "
        f"departure); the comparison is reading noise")


# ===========================================================================
# D-1.  A phase-budget reference that cannot see the dense route's error
# ===========================================================================

def _exact_phase_kernel(alpha, n_in, n_out, sign=-1):
    """``W[k, n] = exp(sign 2 pi i frac(alpha n k))`` with an EXACT reduction.

    ``alpha`` is a float64 and therefore an exact rational; ``n`` and ``k``
    are integers.  The product and its fractional part are computed in
    ``Fraction``, so the single float64 rounding happens on a number already
    inside ``[-1/2, 1/2)`` and the phase keeps all 53 bits whatever the
    budget.  This is the ONE thing the shipped reference does not do.
    """
    fa = Fraction(alpha)
    W = np.empty((n_out, n_in), dtype=np.complex128)
    for k in range(n_out):
        for n in range(n_in):
            t = fa * (n * k)
            t -= math.floor(t)
            if t >= Fraction(1, 2):
                t -= 1
            W[k, n] = cmath.exp(sign * TAU * 1j * float(t))
    return W


def _fsum_sandwich(E, Wy, Wx):
    """``Wy . E . Wx^T`` with every output entry accumulated by ``fsum``."""
    My, Ny = Wy.shape
    Mx, Nx = Wx.shape
    F = np.empty((My, Mx), dtype=np.complex128)
    for ky in range(My):
        wy = Wy[ky]
        for kx in range(Mx):
            wx = Wx[kx]
            re, im = [], []
            for ny in range(Ny):
                a = wy[ny]
                row = E[ny]
                for nx in range(Nx):
                    v = row[nx] * a * wx[nx]
                    re.append(v.real)
                    im.append(v.imag)
            F[ky, kx] = complex(math.fsum(re), math.fsum(im))
    return F


def test_a_phase_budget_reference_must_not_reduce_its_phase_like_the_route():
    r"""The dense route is NOT immune to the chirp phase budget.

    ``_direct_matrix_2d`` forms ``t = alpha * n * k`` in float64 and then
    reduces it by ``t - rint(t)``.  The SECOND step is exact (Sterbenz); the
    FIRST has already discarded the low bits of a product that needs ~63 of
    them, and no later reduction can recover a bit that is gone.  So the dense
    route's phase error is ``~eps * alpha * n * k <= eps * budget`` -- the
    same law the chirp-Z routes obey, with a smaller constant.

    ``test_wave5_h2_mft_direct.py::_fsum_reference`` forms ``t`` the SAME way
    (``ty = alpha * ky * n_y``; ``ty - np.rint(ty)``).  ``math.fsum`` then
    makes the SUMMATION correctly rounded, which is what that reference is
    for -- but it leaves the reference's phase exactly as wrong as the dense
    route's, so the two agree by construction and
    ``test_the_chirp_phase_error_is_linear_in_the_budget_and_dense_is_immune``
    asserts ``rd < 1e-14`` against a reading of the instrument.

    MEASURED 2026-09-20, shipped geometry N=24 M=12, against a reference whose
    phase is reduced EXACTLY (``fractions.Fraction``), both builds:

        budget     1e5      1e9      1e12     1e15
        chirp-Z    1.8e-11  1.9e-07  1.5e-04  1.8e-01
        DENSE      2.3e-12  2.8e-08  3.4e-05  2.0e-02
        eps*budget 2.2e-11  2.2e-07  2.2e-04  2.2e-01

    fitted slope 1.0038 for the dense route and 1.0045 for the chirp-Z one.
    Against the naive-phase reference the dense route instead reads
    3.0e-16 .. 3.6e-16 at every one of those budgets.

    THIS ID GATES THE INSTRUMENT, not the defect: it asserts that the two
    references DISAGREE by decades at a large budget, so that any future claim
    about a route's phase accuracy has to say which reference it used.  If the
    dense route is ever made exact in the product as well, this id's premise
    (the two references disagreeing) still holds -- it is a fact about the
    references -- and the second assertion, that the dense route tracks the
    exact reference to within the same law, is the one that would then be
    re-derived with its new measurement.
    """
    n_in, n_out = 24, 12
    rng = np.random.default_rng(20260920)
    E = (rng.standard_normal((n_in, n_in))
         + 1j * rng.standard_normal((n_in, n_in))).astype(np.complex128)
    rows = []
    for budget in (1e9, 1e12):
        alpha = budget / float(n_in) ** 2
        W_exact = _exact_phase_kernel(alpha, n_in, n_out)
        t = alpha * np.arange(n_out)[:, None] * np.arange(n_in)[None, :]
        W_naive = np.exp(-1j * TAU * (t - np.rint(t)))
        ref_exact = _fsum_sandwich(E, W_exact, W_exact)
        ref_naive = _fsum_sandwich(E, W_naive, W_naive)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            dense = BL._bluestein_2d(
                E, alpha, alpha, n_out, n_out, sign=-1, xp=np,
                fft2=np.fft.fft2, ifft2=np.fft.ifft2, method='direct')

        def rel(a, b):
            return float(np.linalg.norm(np.asarray(a) - np.asarray(b))
                         / np.linalg.norm(np.asarray(b)))

        rows.append({
            'budget': budget,
            'dense_vs_exact': rel(dense, ref_exact),
            'dense_vs_naive': rel(dense, ref_naive),
            'refs_disagree': rel(ref_naive, ref_exact),
            'eps_budget': EPS * budget,
        })

    for r in rows:
        # PREMISE: the naive reference really is the thing the shipped
        # reference builds, i.e. it agrees with the dense route to round-off.
        assert r['dense_vs_naive'] < 1e-14, (
            f"PREMISE: at budget {r['budget']:.0e} the dense route and a "
            f"naive-phase reference differ by {r['dense_vs_naive']:.3e}; "
            f"this id is about a reference that agrees with the route by "
            f"CONSTRUCTION and this one does not")
        # THE CLAIM, one: the two references disagree by decades, so which
        # one a phase-budget measurement used is not a detail.
        assert r['refs_disagree'] > 1e3 * r['dense_vs_naive'], (
            f"at budget {r['budget']:.0e} the exact-phase and naive-phase "
            f"references agree to {r['refs_disagree']:.3e}, within 1e3 of "
            f"the round-off floor {r['dense_vs_naive']:.3e}; the instrument "
            f"distinction this id exists for has gone away and the shipped "
            f"'dense is immune' reading would be sound")
        # THE CLAIM, two: measured against the EXACT reference the dense
        # route obeys the same eps*budget law, within a factor of 20.
        assert 0.005 * r['eps_budget'] < r['dense_vs_exact'] \
            < 20.0 * r['eps_budget'], (
            f"at budget {r['budget']:.0e} the dense route reads "
            f"{r['dense_vs_exact']:.3e} against an exact-phase reference, "
            f"which is not the measured law eps*budget = "
            f"{r['eps_budget']:.3e} (factor "
            f"{r['dense_vs_exact'] / r['eps_budget']:.3f}; measured 0.10 and "
            f"0.15 at these two budgets on 2026-09-20).  Either the route "
            f"changed or the reference did.")

    # ... and the law itself, over the two rungs: the dense error grows with
    # the budget rather than staying at round-off.
    growth = rows[1]['dense_vs_exact'] / rows[0]['dense_vs_exact']
    span = rows[1]['budget'] / rows[0]['budget']
    assert growth > 0.1 * span, (
        f"the dense route's error against an exact-phase reference grew only "
        f"{growth:.3g}x over {span:.0e} decades of budget; that would make it "
        f"budget-INDEPENDENT, which is the claim this id refutes")
