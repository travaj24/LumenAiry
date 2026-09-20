"""VERIFY Wave-5 hygiene 2 -- the decisions the shipped package's own tests
leave open.

Adversarial verification of ``refactor/wave5-hygiene-2`` (H2-1 the
direct-matrix MFT route, H2-2 ``_collins_transport`` on the field's backend,
H2-3 the near-focus table, H2-4 the loud stale-patch refusal).  Report:
``docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/
VERIFY_WAVE5_HYGIENE2.md``; probes and JSON in
``validation/probe_verify_wave5_hyg2/``.

Every id here closes a gap found by RE-MEASURING, and every bar is derived at
runtime from the quantity it bounds.  Four gaps, and what each one asserts:

1. **H2-2 reaches the private transport and not the public leg.**  MEASURED on
   both trees: at ``f4f18851`` an eager JAX array through
   ``_collins_transport`` came back as ``numpy.ndarray``; on the branch it
   comes back as a JAX array -- the port is real.  But
   ``propagate_carrier_referenced(transport='collins')`` still converts to the
   host in ``_collins_carrier_leg`` (``env_a = np.asarray(env)``), so at the
   PUBLIC surface a JAX array is still silently demoted and a CuPy array still
   raises ``TypeError: Implicit conversion to a NumPy array is not allowed``
   with no mention of the transport.  The ids below pin the part that is real
   and BOUND the part that is not: the demotion must stay a backend downgrade
   and never a change of answer, and a traced call must never return a
   silently wrong field.

2. **The gradient-shape bar.**  ``test_wave5_h2_collins_jax.py`` asserts
   ``corr > 1.0 - 1e-6`` on a constant with no stated origin.  The merit is
   ``P(a) = a^T (L^H L) a``, so ``grad P = 2 (L^H L) a`` is proportional to
   ``a`` exactly when ``a`` is an eigenvector of ``L^H L`` -- i.e. when the leg
   is a scaled isometry on the fixture.  The shortfall is that failure, and it
   is QUADRATIC in the residual, which the running build can measure.
   MEASURED 2026-09-19, both builds: ``1 - corr`` = 3.168867e-07 against the
   pinned 1e-6, i.e. **0.50 decades**; a 17 % change of the fixture's envelope
   width (60 -> 70 um) takes it to 6.93e-06, SEVEN TIMES over that bar.  The
   derived form below has decades.

3. **The central difference's floor model.**  The same test derives its bar
   from ``eps^(2/3)``, the minimum of a U-curve balancing truncation
   ``P''' h^2/6`` against cancellation ``eps|P|/h``.  For this merit ``P'''``
   is identically zero -- the transport is LINEAR in the envelope, so the
   merit is an exact quadratic form and a central difference of it has no
   truncation error.  MEASURED: the disagreement falls monotonically over the
   whole ladder (7.23e-10 at h = 1e-5 to 7.55e-15 at h = 1e-1 on WIN), so the
   ladder never brackets a minimum and the bar is ~4.7 decades looser than the
   measurement.  The id below asserts the PREMISE that makes the cancellation
   branch the right floor.

4. **Where H2-1's derived summation bar holds.**  The bar
   ``(g_a + g_b) * eps * sum|E|`` is claimed at 1.47 .. 2.59 decades of room.
   RE-MEASURED over a wider ladder the two halves part company: the DENSE
   route's margin WIDENS with ``n`` (1.84 -> 3.60 decades from 16x16->8x8 to
   256x256->128x128) while the chirp-Z routes' margin NARROWS monotonically
   (1.93 -> 0.73 decades over the same ladder).  The report's explanation --
   "the actual errors grow more slowly" -- has the direction backwards: the
   chirp-Z error grows FASTER than ``g_chirp * eps * sum|E|``, which is why
   the margin closes.  The ids below assert both trends and that the shapes
   the shipped file actually uses are still decades inside, so the shipped
   assertions are sound where they are made.
"""
from __future__ import annotations

import numpy as np
import pytest

import lumenairy.propagators.carrier as CA

WL = 633e-9
N = 64
DX = 8e-6
R_IN = -0.05
Z = 5e-3
R_REF = -0.045
W_ENV = 60e-6


def _gauss(n=N, dx=DX, w=W_ENV):
    ax = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(ax, ax)
    return np.exp(-(X ** 2 + Y ** 2) / (w * w)).astype(np.complex128)


def _rel(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _priv(env, **kw):
    return CA._collins_transport(
        env, R_IN, Z, WL, DX, DX, dx_out=DX, dy_out=DX, N_out_x=N, N_out_y=N,
        R_ref=R_REF, gap_kernel='fresnel', on_collins_sampling='ignore', **kw)


def _fft_bar(env, jnp, chain_depth=6.0):
    """Everything the two backends are entitled to differ by, per transform,
    MEASURED on this build -- floored by the transform's own round-off.

    One forward transform of the fixture through the library's dispatcher and
    through ``jax.numpy.fft`` gives the irreducible per-transform spread.  On
    an easy fixture the two agree BIT FOR BIT (measured here: exactly 0.0 on
    WIN-py3.14 and WSL-py3.12), which would make a bar built on the spread
    alone identically zero and the comparison a coin toss.  The floor is the
    FFT's own relative round-off, ``eps * log2(Ny*Nx)`` for a radix-2
    decomposition -- derived from the transform's depth rather than picked, so
    it tracks the grid instead of pinning a constant.  ``chain_depth`` is an
    upper bound on the transforms the transport applies (five on the 2-D arm,
    six on the separable one); round-off through unitary transforms adds at
    worst linearly, so the product is a bound.

    (``test_wave5_h2_collins_jax.py::_fft_spread_bar`` uses ``32 * eps``
    instead.  On this fixture that floor DOMINATES its measured term by 4.4x,
    so the bar the shipped file actually asserts against is 7.105e-15 and not
    the 1.611e-15 its report tabulates.)
    """
    from lumenairy.propagators.fft_infra import _fft2
    a = np.asarray(_fft2(np.ascontiguousarray(env, dtype=np.complex128)))
    b = np.asarray(jnp.fft.fft2(jnp.asarray(env, dtype=jnp.complex128)))
    spread = float(np.linalg.norm(a - b) / np.linalg.norm(a))
    eps = float(np.finfo(np.float64).eps)
    floor = eps * float(np.log2(env.shape[-1] * env.shape[-2]))
    return chain_depth * max(spread, floor)


# ===========================================================================
# 1.  H2-2's reach: the private transport moved, the public leg did not
# ===========================================================================

def test_the_private_collins_transport_stays_on_an_eager_jax_backend():
    """H2-2's actual win, gated.

    This FAILS at ``f4f18851`` -- measured there, the same call returns
    ``numpy.ndarray`` -- so it is a real before/after gate and not a
    restatement of today's behaviour.
    """
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    jax.config.update("jax_enable_x64", True)
    env = _gauss()
    out = _priv(jnp.asarray(env))
    assert not isinstance(out, np.ndarray), (
        "an eager JAX envelope came back as a host numpy array; the transport "
        "round-tripped through the host, which is what H2-2 removed")
    assert type(out).__module__.split('.')[0] in ('jax', 'jaxlib'), (
        f"expected a JAX array, got "
        f"{type(out).__module__}.{type(out).__name__}")
    # ... and it is the SAME field, against a bar measured from the two
    # backends' own single-transform spread rather than a remembered residual.
    ref = _priv(env)
    bar = _fft_bar(env, jnp)
    got = _rel(np.asarray(out), ref)
    # The smallest REAL signal on this fixture: the exact kernel against the
    # paraxial one, which is a physical difference the leg is meant to show.
    exact = CA._collins_transport(
        env, R_IN, Z, WL, DX, DX, dx_out=DX, dy_out=DX, N_out_x=N, N_out_y=N,
        R_ref=R_REF, gap_kernel='exact', on_collins_sampling='ignore')
    signal = _rel(exact, ref)
    assert got < bar, (
        f"JAX and NumPy disagree by {got:.4e}, past the {bar:.4e} the two "
        f"backends' own FFT spread allows")
    assert bar < 1e-3 * signal, (
        f"the bar {bar:.3e} is not decades below the smallest real signal on "
        f"this fixture ({signal:.3e}); the comparison has become noise")


def test_the_public_collins_leg_demotes_the_backend_but_never_the_answer():
    """The boundary of H2-2's reach, BOUNDED rather than merely noted.

    WHEN THIS WAS WRITTEN (2026-09-19):
    ``propagate_carrier_referenced(transport='collins')`` reached
    ``_collins_carrier_leg``, which opened with ``env_a = np.asarray(env)``.
    H2-2 had not touched it, so at the public surface an eager JAX array was
    demoted to the host.  A demotion is a placement defect; a WRONG ANSWER
    would be a correctness defect.  This id said which one it was, so the
    defect could not quietly become the other.

    SINCE ROUND 2 the leg runs in the field's own namespace and the demotion
    is gone -- the same call now returns a ``jaxlib`` array, and the
    comparison below has become an ordinary cross-backend parity claim
    (measured 4.1309e-16 WIN / 4.1251e-16 WSL against the same bar, where it
    read exactly 0.0 while both arms were NumPy).  The ASSERTIONS are
    unchanged and still hold, which is the property a defect-bounding id
    should have: it was written so that fixing the defect does not falsify
    it.
    """
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    jax.config.update("jax_enable_x64", True)
    env = _gauss()
    kw = dict(transport='collins', gap_kernel='fresnel',
              on_collins_sampling='ignore')
    host = CA.propagate_carrier_referenced(env, R_IN, Z, WL, DX, **kw).env
    dev = CA.propagate_carrier_referenced(jnp.asarray(env), R_IN, Z, WL, DX,
                                          **kw).env
    bar = _fft_bar(env, jnp)
    got = _rel(np.asarray(dev), np.asarray(host))
    assert got <= bar, (
        f"the public Collins leg returns a different field for a JAX input "
        f"({got:.4e} > {bar:.4e}); the host round-trip has stopped being a "
        f"pure demotion")
    assert bar < 1e-6, (
        f"the bar {bar:.3e} is too loose to tell a demotion from a change of "
        f"answer; re-derive it")


def test_a_traced_collins_call_never_returns_a_silently_wrong_field():
    """Two-sided, and the two sides are different functions.

    The PRIVATE transport takes the designed refusal and must name both ways
    out.  When this was written the PUBLIC leg had no such refusal -- it met
    JAX's own ``TracerArrayConversionError`` at ``np.asarray`` first -- so all
    that could be asserted there was that it RAISES rather than materialising
    the tracer and computing off the copy, with the note that "if the leg is
    ever threaded, the first arm still passes and the second becomes 'it
    returned a traced array', which this id accepts".

    SINCE ROUND 2 the leg IS threaded and the public refusal is designed: a
    ``ValueError`` naming ``_collins_transport``, ``dx_out``, ``gap_kernel``
    and ``on_collins_sampling``.  This id still takes the raising disposition,
    unchanged; the id that pins WHICH exception and WHAT it names is
    ``tests/unit/test_wave5_h2_collins_jax.py::
    test_the_public_collins_leg_refuses_a_trace_by_name``.
    """
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    jax.config.update("jax_enable_x64", True)
    env = jnp.asarray(_gauss())

    with pytest.raises(ValueError) as exc:
        jax.jit(lambda e: CA._collins_transport(
            e, R_IN, Z, WL, DX, DX, dx_out=DX, dy_out=DX, N_out_x=N,
            N_out_y=N, R_ref=R_REF))(env)
    msg = str(exc.value)
    for needle in ("gap_kernel", "'fresnel'", "on_collins_sampling",
                   "'ignore'"):
        assert needle in msg, (
            f"the traced refusal does not name {needle!r}; a caller cannot "
            f"act on it.  Message: {msg[:300]}")

    def pub(e):
        return CA.propagate_carrier_referenced(
            e, R_IN, Z, WL, DX, transport='collins', gap_kernel='fresnel',
            on_collins_sampling='ignore').env

    try:
        out = jax.jit(pub)(env)
    except Exception:            # noqa: BLE001 -- see the docstring: raising
        # is the accepted disposition today (a raw TracerArrayConversionError
        # out of np.asarray inside _collins_carrier_leg).
        return
    assert not isinstance(out, np.ndarray), (
        "the public Collins leg returned a concrete host array from inside a "
        "jax.jit trace -- it materialised the tracer and computed off the "
        "copy, which is the one disposition that is not allowed")


# ===========================================================================
# 2.  The gradient bars, derived instead of pinned
# ===========================================================================

def test_the_gradient_shape_bar_is_the_legs_departure_from_a_scaled_isometry():
    """``grad P`` is proportional to ``a`` exactly when ``a`` is an
    eigenvector of ``L^H L``; the correlation's shortfall IS that failure, and
    it is quadratic in the residual -- so the bar is the residual, measured,
    and not a constant.

    PREMISE-GATED on both sides: the residual must be non-zero (or the claim
    is vacuous and a pinned bar would pass on a degenerate fixture) and small
    (or the fixture is not the near-isometric leg the claim is about).
    """
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    jax.config.update("jax_enable_x64", True)

    def merit(amp):
        return jnp.sum(jnp.abs(_priv(amp.astype(jnp.complex128))) ** 2)

    a = np.real(_gauss())
    g = np.asarray(jax.grad(merit)(jnp.asarray(a)))
    assert np.all(np.isfinite(g))
    m = a > 0.05 * a.max()
    gm, am = g[m], a[m]
    c = float(np.dot(gm, am) / np.dot(am, am))
    r = float(np.linalg.norm(gm - c * am) / np.linalg.norm(c * am))
    corr = float(np.corrcoef(gm, am)[0, 1])

    assert 0.0 < r < 1e-1, (
        f"PREMISE: the leg's departure from a scaled isometry on this fixture "
        f"is {r:.3e}; at 0 the shape claim is vacuous and above 1e-1 the "
        f"fixture is not the near-isometric leg the claim is about")
    # Pearson's shortfall is second order in the residual.  The factor of ten
    # covers the mean subtraction Pearson does and the norm above does not.
    # MEASURED 2026-09-19 on BOTH builds: 1 - corr = 3.1689e-07, r^2/2 =
    # 1.2490e-07, ratio 2.54.
    assert (1.0 - corr) < 10.0 * r * r, (
        f"the gradient's departure from proportionality (1 - corr = "
        f"{1.0 - corr:.4e}) is larger than the leg's own non-isometry allows "
        f"({10.0 * r * r:.4e}); the gradient is not the one it should be")
    assert (1.0 - corr) > 0.01 * r * r, (
        f"1 - corr = {1.0 - corr:.4e} is far BELOW the leg's measured "
        f"non-isometry {r:.4e}; the correlation has stopped reading the "
        f"quantity this bar is derived from, so the bar gates nothing")


def test_the_central_difference_through_this_merit_has_no_truncation_branch():
    """Why ``eps^(2/3)`` is the wrong floor here, asserted as its premise.

    ``_collins_transport`` is LINEAR in the envelope, so ``P(a) = sum|L a|^2``
    is an exact quadratic form and ``P''' == 0``.  A central difference of a
    quadratic is EXACT, so the ladder has no truncation branch and no
    minimum: the achievable floor is the cancellation branch alone, which at
    the ladder's top is decades below ``eps^(2/3)``.

    Asserted on the arithmetic rather than on a residual -- the third central
    difference must vanish to round-off relative to the second -- and then on
    its observable consequence, that the ladder improves monotonically with
    ``h``.
    """
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    jax.config.update("jax_enable_x64", True)

    def merit(amp):
        return float(jnp.sum(jnp.abs(_priv(amp.astype(jnp.complex128))) ** 2))

    a = np.real(_gauss())
    ij = np.unravel_index(int(np.argmax(a)), a.shape)
    h = 1e-2

    def at(step):
        ap = a.copy()
        ap[ij] += step * h
        return merit(jnp.asarray(ap))

    f = {s: at(s) for s in (-2, -1, 0, 1, 2)}
    d2 = f[1] - 2.0 * f[0] + f[-1]
    d3 = f[2] - 2.0 * f[1] + 2.0 * f[-1] - f[-2]
    assert abs(d2) > 0.0, "PREMISE: the merit is flat here; pick another entry"
    # eps*|P|/|d2| is the largest relative round-off the third difference can
    # carry; anything at that level IS round-off and not curvature.
    floor = float(np.finfo(np.float64).eps) * abs(f[0]) / abs(d2)
    assert abs(d3) / abs(d2) < 1e3 * floor, (
        f"the third difference is {abs(d3) / abs(d2):.3e} of the second, past "
        f"{1e3 * floor:.3e} = 1000 x round-off; the merit has a real cubic "
        f"term after all and eps^(2/3) IS the right floor -- re-derive the "
        f"bar in test_wave5_h2_collins_jax.py")

    g = np.asarray(jax.grad(lambda x: jnp.sum(jnp.abs(
        _priv(x.astype(jnp.complex128))) ** 2))(jnp.asarray(a)))
    rel = []
    for hh in (1e-3, 1e-2, 1e-1):
        ap, am = a.copy(), a.copy()
        ap[ij] += hh
        am[ij] -= hh
        fd = (merit(jnp.asarray(ap)) - merit(jnp.asarray(am))) / (2.0 * hh)
        rel.append(abs(fd - g[ij]) / abs(g[ij]))
    assert rel[0] > rel[1] > rel[2], (
        f"the central difference does not improve monotonically with h "
        f"({rel}); the ladder brackets a minimum after all and the U-curve "
        f"model applies")


# ===========================================================================
# 3.  Where H2-1's derived summation bar holds
# ===========================================================================

def _pairwise_reference(E, alpha, My, Mx, sign=-1):
    """The SAME sum, one output point at a time, summed by ``np.sum`` --
    pairwise, blocked at 128, growth factor ``log2(n/128) + 8``."""
    ny, nx = E.shape
    n_x = np.arange(nx, dtype=np.float64)
    n_y = np.arange(ny, dtype=np.float64)
    out = np.empty((My, Mx), dtype=np.complex128)
    for ky in range(My):
        ty = alpha * float(ky) * n_y
        wy = np.exp(1j * sign * 2.0 * np.pi * (ty - np.rint(ty)))
        Ewy = E * wy[:, None]
        for kx in range(Mx):
            tx = alpha * float(kx) * n_x
            wx = np.exp(1j * sign * 2.0 * np.pi * (tx - np.rint(tx)))
            out[ky, kx] = np.sum(Ewy * wx[None, :])
    return out


def _margins(n_in, m_out):
    """``(chirp decades, dense decades, bar/signal)`` at one shape, every
    quantity derived here from the summation and none remembered."""
    from scipy.fft import next_fast_len

    from lumenairy.propagators._bluestein import (
        _bluestein_2d, _clear_h_fft_cache)
    from lumenairy.propagators.fft_infra import _fft2, _ifft2
    eps = float(np.finfo(np.float64).eps)
    rng = np.random.default_rng(20260919)
    E = (rng.standard_normal((n_in, n_in))
         + 1j * rng.standard_normal((n_in, n_in))).astype(np.complex128)
    alpha = 1.0 / 64.0
    ref = _pairwise_reference(E, alpha, m_out, m_out)
    n = n_in * n_in
    L = float(next_fast_len(int(n_in + m_out - 1)))
    g_pair = float(np.log2(max(n / 128.0, 2.0)) + 8.0)
    s = float(np.sum(np.abs(E)))
    bars = {'chirp': (3.0 * np.log2(L * L) + g_pair) * eps * s,
            'dense': (float(np.sqrt(n)) + g_pair) * eps * s}
    out = {}
    for name, kw in (('chirp', dict(separable=False, method='auto')),
                     ('dense', dict(method='direct'))):
        _clear_h_fft_cache()
        F = _bluestein_2d(E, alpha, alpha, m_out, m_out, sign=-1, xp=np,
                          fft2=_fft2, ifft2=_ifft2, **kw)
        err = float(np.max(np.abs(F - ref)))
        assert err > 0.0, (
            f"PREMISE: the {name} route reproduced the reference EXACTLY at "
            f"N={n_in}; there is no summation difference to bound")
        out[name] = float(np.log10(bars[name] / err))
    out['bar_over_signal'] = bars['chirp'] / float(np.max(np.abs(ref)))
    return out


def test_the_summation_bar_is_two_sided_where_the_shipped_file_asserts_it():
    """The bar must sit decades BELOW the smallest real signal as well as
    above the residual, or it is not gating anything.

    ``test_wave5_h2_mft_direct.py`` asserts ``err < bar`` and argues the other
    side in prose.  The prose is right -- and now it fails if it stops being.
    """
    for (n_in, m_out) in ((16, 8), (32, 16)):
        m = _margins(n_in, m_out)
        assert m['bar_over_signal'] < 1e-8, (
            f"at N={n_in} M={m_out} the derived bar is "
            f"{m['bar_over_signal']:.2e} of the signal; an O(1) wrong answer "
            f"would be inside it")
        assert m['chirp'] > 1.0 and m['dense'] > 1.0, (
            f"at N={n_in} M={m_out} a route is within one decade of its own "
            f"bar (chirp {m['chirp']:.2f}, dense {m['dense']:.2f} decades)")


def test_the_two_reductions_margins_move_in_opposite_directions_with_n():
    """The range-of-validity claim H2-1's report states backwards.

    A bar that grows faster than its quantity gives a WIDENING margin.  The
    chirp-Z margin narrows, so its error grows FASTER than
    ``g_chirp * eps * sum|E|``; the dense route's grows more slowly and its
    margin widens.  Both asserted, premise-gated on the ladder spanning enough
    ``n`` to determine a trend at all.
    """
    shapes = ((16, 8), (48, 24), (128, 64))
    ms = [_margins(a, b) for a, b in shapes]
    span = (shapes[-1][0] ** 2) / (shapes[0][0] ** 2)
    assert span >= 16.0, (
        f"PREMISE: n spans only {span:.1f}x across this ladder; no trend is "
        f"determined")
    chirp = [m['chirp'] for m in ms]
    dense = [m['dense'] for m in ms]
    assert chirp[0] > chirp[1] > chirp[2], (
        f"the chirp-Z margin against its derived bar does not narrow with n "
        f"({chirp}); the bar's range of validity has moved and the report's "
        f"1.47-2.59 decades must be re-derived")
    assert dense[0] < dense[1] < dense[2], (
        f"the dense margin against its derived bar does not widen with n "
        f"({dense}); the two routes' error growth is no longer what the bar "
        f"is built on")
    assert chirp[2] > 0.3, (
        f"the chirp-Z route is within {chirp[2]:.2f} decades of its own "
        f"derived bar at N=128; the bar has stopped being a bound and the "
        f"shipped tolerance claim needs its range of validity stated")


# ===========================================================================
# 4.  The sign of the exact-kernel correction -- invisible to every magnitude
# ===========================================================================

def test_the_exact_kernel_retards_the_envelope_and_the_sign_is_gated():
    """The gap ``test_wave5_h2_near_focus_table.py`` leaves open.

    ``|exp(i phi) - 1|`` is EVEN in ``phi``, so every magnitude that file
    reads -- the relative L2 against the oracle, the exact-minus-fresnel
    departure, the monotonicity, both power-law slopes and the constant ``C``
    -- is invariant under ``phi -> -phi``.  MEASURED 2026-09-19 on a copy of
    the tree: flipping the sign of ``z_eff`` inside
    ``_collins_exact_kernel_correction`` leaves all 14 of that file's ids
    GREEN on both builds while the intensity-weighted mean of
    ``arg(exact/fresnel)`` flips sign.  A conjugated non-paraxial kernel would
    ship.

    The sign is not a convention.  ``sqrt(k^2 - q^2) < k - q^2/(2k)`` for every
    real ``q``, so the exact kernel RETARDS relative to the paraxial one and
    on a leg with ``z_eff > 0`` the mean correction phase is NEGATIVE.  Its
    size is the same law the report derives, with the RMS factor
    ``sqrt(<u^8>) = sqrt(3/2)`` replaced by the MEAN factor ``<u^4> = 1/2`` --
    both exact for a 2-D circular Gaussian envelope under the spectral weight
    ``exp(-2 u^2)``:

        ``<phi> = -k |z_eff| theta_env^4 / 16`` .

    MEASURED here: -4.38331e-07 against a predicted -4.38323e-07 (0.002 %),
    and the magnitude law ``sqrt(3/2) k z_eff theta^4 / 8`` = 1.07367e-06
    against a measured departure of 1.07407e-06 (0.04 %).  Both builds.
    """
    env = _gauss()
    st = {}
    exact = CA._collins_transport(
        env, R_IN, Z, WL, DX, DX, dx_out=DX, dy_out=DX, N_out_x=N, N_out_y=N,
        R_ref=R_REF, gap_kernel='exact', on_collins_sampling='ignore',
        stats_out=st)
    fres = _priv(env)
    assert st.get('kernel') == 'exact', (
        f"PREMISE: the leg resolved to {st.get('kernel')!r}, so there is no "
        f"exact-kernel correction on it to read a sign from")
    A, B = float(st['abcd'][0]), float(st['abcd'][1])
    z_eff = B / A
    assert z_eff > 0.0, (
        f"PREMISE: z_eff = {z_eff:.4e} is not positive; the sign of the "
        f"retardation is stated for a forward reduced frame")

    k = 2.0 * np.pi / WL
    theta = WL / (np.pi * W_ENV)        # the ENVELOPE's analytic 1/e^2 angle
    wgt = np.abs(fres) ** 2
    mean_phi = float((wgt * np.angle(exact / fres)).sum() / wgt.sum())
    predicted = -k * z_eff * theta ** 4 / 16.0

    assert mean_phi < 0.0, (
        f"the exact-kernel correction ADVANCES the envelope (mean phase "
        f"{mean_phi:+.4e}); sqrt(k^2-q^2) - k + q^2/(2k) is negative for every "
        f"real q, so the refinement's z_eff carries the wrong sign")
    assert abs(mean_phi - predicted) < 0.01 * abs(predicted), (
        f"the mean correction phase is {mean_phi:.6e}, not the derived "
        f"-k z_eff theta_env^4/16 = {predicted:.6e} (the <u^4> = 1/2 moment "
        f"of a 2-D circular Gaussian envelope)")
    # ... and the MAGNITUDE law, with the RMS moment, on the same reading, so
    # the two moments of the same quartic are gated together and a fixture
    # that satisfied one by accident could not satisfy both.
    dep = _rel(exact, fres)
    rms_pred = float(np.sqrt(1.5)) * k * z_eff * theta ** 4 / 8.0
    assert abs(dep - rms_pred) < 0.02 * rms_pred, (
        f"the exact-vs-paraxial departure is {dep:.6e}, not the derived "
        f"sqrt(3/2) k z_eff theta_env^4/8 = {rms_pred:.6e}")
    assert dep > 1e3 * float(np.finfo(np.float64).eps), (
        f"PREMISE: the departure {dep:.3e} sits at round-off; the sign of a "
        f"quantity that small is not a claim about the kernel")


# ===========================================================================
# 5.  The two H2-2 threadings no value test can see
# ===========================================================================

def test_the_axis_chirp_builds_on_the_bld_it_is_handed():
    """``_backend_of`` returns ``bld = np`` for a NumPy field AND for a JAX
    one, so hardcoding ``bld = np`` inside ``_collins_axis_chirp``'s body is
    invisible on every arm this box can run -- MEASURED: that mutation leaves
    all 21 ids of ``test_wave5_h2_collins_jax.py`` green and all 93 measured
    bit-identity keys unchanged.  Only CuPy would differ, and CuPy's FFT
    cannot run here.

    A recording namespace gates the threading itself, needs no backend and
    costs nothing.  The second half is the byte-identity that makes ``bld=np``
    the historical build verbatim.
    """
    class _Recording:
        def __init__(self):
            self.seen = []

        def __getattr__(self, name):
            self.seen.append(name)
            return getattr(np, name)

    rec = _Recording()
    got = CA._collins_axis_chirp(16, DX, WL, -0.02, bld=rec)
    assert 'arange' in rec.seen and 'exp' in rec.seen, (
        f"_collins_axis_chirp did not build on the namespace it was handed; "
        f"it reached for {rec.seen!r}")
    ref = CA._collins_axis_chirp(16, DX, WL, -0.02)
    assert np.array_equal(np.ascontiguousarray(got).view(np.float64),
                          np.ascontiguousarray(ref).view(np.float64)), (
        "bld=np is not the historical NumPy build byte for byte")


def test_as_c_order_makes_the_numpy_path_contiguous():
    """``np.asarray`` and ``np.ascontiguousarray`` agree on VALUES for every
    input, so replacing one with the other is value-inert -- MEASURED: that
    mutation leaves all 21 ids and all 93 bit-identity keys unchanged.  The
    contract is a LAYOUT contract and only a layout assertion gates it.
    """
    a = np.asfortranarray(np.arange(12.0).reshape(3, 4) + 1j)
    for src in (a, a.T, a[:, ::2]):
        got = CA._as_c_order(src, np.complex128, np)
        assert got.flags.c_contiguous, (
            f"_as_c_order left a non-contiguous array for an input with "
            f"strides {src.strides}")
        assert got.dtype == np.complex128
        assert np.array_equal(
            got, np.ascontiguousarray(src, dtype=np.complex128))
    # The JAX branch: jax.numpy has no ascontiguousarray on either version
    # here (0.11.0 and 0.10.2, both measured), so the helper must fall through
    # to asarray rather than raise -- and if jax ever grows one, the
    # docstring's stated reason stops holding and this says so.
    jnp = pytest.importorskip("jax.numpy")
    assert not hasattr(jnp, 'ascontiguousarray'), (
        "jax.numpy grew ascontiguousarray; _as_c_order's fallback branch is "
        "now dead code and its docstring's reason no longer holds")
    out = CA._as_c_order(jnp.asarray(a), np.complex128, jnp)
    assert type(out).__module__.split('.')[0] in ('jax', 'jaxlib')


# ===========================================================================
# 6.  The chirp phase budget is an error LAW, not a cliff
# ===========================================================================

def test_the_chirp_phase_error_is_linear_and_a_shared_phase_reference_is_blind(
):
    """What ``_bluestein_2d``'s phase-budget guard is guarding, and what a
    reference that shares a route's phase can and cannot see.

    (The threshold read ``1e15`` when this was written; Round 2 derived it
    from the law below and it is now ``1e-6/eps = 4.5036e9``.  This id is
    unaffected, which is what "deliberately not a pin on the threshold" was
    for.)

    The chirp signal's phase is rounded to float64 BEFORE ``exp``, so the
    chirp-Z routes' relative error tracks the budget LINEARLY --
    ``rel ~ eps * alpha * N_max^2`` -- and there is no cliff to sit just
    below.

    CORRECTED 2026-09-20 (VERIFY-WAVE5-HYGIENE2 round 2, D-1).  This id was
    called ``..._and_dense_is_immune`` and its last two assertions read the
    dense route against :func:`_pairwise_reference`, which forms its phase as
    ``alpha*k*n`` in float64 and then reduces it -- the same two roundings
    :func:`_direct_matrix_2d` commits.  The dense route therefore agreed with
    it to the summation floor by CONSTRUCTION, at every budget, and calling
    that immunity was reading the instrument.  Measured against an
    exactly-reduced reference the dense route obeys the SAME ``eps*budget``
    law with a constant 1.5x to 11.8x smaller; that measurement and its bars
    live in ``tests/unit/test_wave5_h2_mft_direct.py::
    test_both_routes_follow_the_budget_law_and_dense_wins_by_a_bounded_factor``
    and are not duplicated here.

    What the dense arm asserts NOW is the blindness itself, as a decision
    about the instrument: against a reference sharing its phase the dense
    route sits at the summation floor at EVERY budget, and the chirp route
    does not.  That is a real, falsifiable statement -- it fails the moment
    the dense route stops reducing its phase modulo one turn -- and it is the
    premise the corrected reading rests on.

    Deliberately NOT a pin on the warning threshold: moving that threshold is
    a behaviour change owed to the maintainer, and this id stays true whatever
    it becomes.
    """
    import warnings

    from lumenairy.propagators._bluestein import (
        _bluestein_2d, _clear_h_fft_cache)
    from lumenairy.propagators.fft_infra import _fft2, _ifft2
    eps = float(np.finfo(np.float64).eps)
    ny = nx = 24
    my = mx = 12
    rng = np.random.default_rng(20260919)
    E = (rng.standard_normal((ny, nx))
         + 1j * rng.standard_normal((ny, nx))).astype(np.complex128)
    budgets = (1e9, 1e11, 1e13, 1e15)
    chirp_rel, dense_rel = [], []
    for budget in budgets:
        alpha = budget / float(max(ny, nx, my, mx)) ** 2
        ref = _pairwise_reference(E, alpha, my, mx)
        nrm = float(np.linalg.norm(ref))
        _clear_h_fft_cache()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c = _bluestein_2d(E, alpha, alpha, my, mx, sign=-1, xp=np,
                              fft2=_fft2, ifft2=_ifft2)
            d = _bluestein_2d(E, alpha, alpha, my, mx, sign=-1, xp=np,
                              fft2=_fft2, ifft2=_ifft2, method='direct')
        chirp_rel.append(float(np.linalg.norm(c - ref)) / nrm)
        dense_rel.append(float(np.linalg.norm(d - ref)) / nrm)

    assert min(chirp_rel) > 0.0, (
        "PREMISE: the chirp route reproduced the reference exactly at some "
        "budget; there is no error law to fit")
    slope = float(np.polyfit(np.log10(budgets), np.log10(chirp_rel), 1)[0])
    assert abs(slope - 1.0) < 0.05, (
        f"the chirp-Z relative error scales as budget^{slope:.4f}, not "
        f"budget^1; the guard's premise -- and any threshold derived from it "
        f"-- rests on that linear law")
    for b, r in zip(budgets, chirp_rel):
        assert 0.1 * eps * b < r < 10.0 * eps * b, (
            f"at budget {b:.0e} the chirp route reads {r:.3e}, outside the "
            f"decade around eps*budget = {eps * b:.3e}")
    # THE INSTRUMENT, not the route: a reference whose phase is formed the
    # way the dense route forms it agrees with the dense route to the
    # summation floor at every budget, and with the chirp route not at all.
    assert max(dense_rel) < 100.0 * eps, (
        f"the dense route's worst residual against a reference that shares "
        f"its phase is {max(dense_rel):.3e}, past 100*eps = "
        f"{100.0 * eps:.3e}; either its modulo-one-turn reduction has "
        f"stopped being exact or the reference has stopped sharing it")
    assert max(chirp_rel) / max(dense_rel) > 1e6, (
        f"PREMISE: against this reference the two routes are only "
        f"{max(chirp_rel) / max(dense_rel):.1e}x apart at the top of the "
        f"ladder; the blindness this id records is not being exercised")


def test_the_public_mft_method_vocabulary_is_closed_and_documented():
    """A value the docstring names must work, and a value nothing names must
    raise -- on the PUBLIC entry points, which validate ``method`` only by
    pass-through to the primitives.

    The shipped docstrings name ``{'auto', 'direct'}`` while the primitives
    accept four.  The SUBSET direction is what this asserts, so it stays true
    whether the docstrings widen or the public surface narrows.
    """
    import re

    from lumenairy.propagators._bluestein import _SUM_METHODS
    from lumenairy.propagators.mft import (
        angular_spectrum_propagate_mft, fraunhofer_propagate_mft,
        fresnel_propagate_mft)
    accepted = set(_SUM_METHODS) | {'auto'}
    E = _gauss(64, 8e-6, 60e-6)
    for fn in (fresnel_propagate_mft, fraunhofer_propagate_mft,
               angular_spectrum_propagate_mft):
        doc = fn.__doc__ or ''
        m = re.search(r"method : \{([^}]*)\}", doc)
        assert m, f"{fn.__name__} documents no method vocabulary"
        documented = {v.strip().strip("'\"") for v in m.group(1).split(',')}
        assert documented <= accepted, (
            f"{fn.__name__}'s docstring names {sorted(documented - accepted)}, "
            f"which the primitives do not accept")
        for value in sorted(documented):
            out = fn(E, 0.05, WL, 8e-6, 4e-6, 16, method=value)
            assert out.shape == (16, 16), (
                f"{fn.__name__}(method={value!r}) -- a value its own docstring "
                f"names -- did not produce an output")
        with pytest.raises(ValueError, match="method must be one of"):
            fn(E, 0.05, WL, 8e-6, 4e-6, 16, method='no-such-route')
