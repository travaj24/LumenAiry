"""WP-B1 regression pins: the Maslov asymptotic saddle follows the INPUT
field's local wavevector.

AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11 finding **S6**, the half WP-A4
deferred (its report section 6 item 1 carries the design).  WP-A4 shipped a
``RuntimeWarning``; the saddle itself was still that of the OPD alone.

The defect.  The v2 integrand is
``E_in(s1(s2, v2)) |det ds1/dv2|^(1/2) exp(2 pi i OPD_waves)``, so the phase
that is stationary in ``v2`` is ``arg E_in(s1(v2)) + 2 pi OPD_waves``.  Both
asymptotic evaluators solved ``grad_v2 OPD = 0``, and the symplectic identity
``dOPD/dv2 = -n1 (v1 . ds1/dv2)`` makes that the ``v1 = 0`` on-axis collimated
launch ray at EVERY pixel and for EVERY input -- while the driver sizes the
pupil chart specifically to cover a diverging / converging / tilted one.

Oracles, none of them produced by the code under test:

* a **lumenairy-free** oracle, inline below: an exact sequential conic
  raytrace of the INPUT's own rays (launch direction = its local wavevector
  ``(1/k0) grad arg E_in``) through the singlet, then a direct
  Rayleigh--Sommerfeld surface sum from the exit surface to the readout
  points.  Method (not code) copied from the inline oracle of
  ``tests/unit/test_niche_d6_exact_tilted_leg.py``, generalised to a launch
  direction that varies across the pupil.  ``_test_the_oracle_is_grid_...``
  measures its own convergence floor, which is what every bar below is
  allowed to sit above.
* the **symplectic identity itself**, which turns the converged saddle back
  into the launch direction of the ray it selected:
  ``v1 = -(lambda / n1) (ds1/du_v2)^-T grad_u_v2 OPD_waves``.  That is the
  audit's own ``|v1|`` census and it needs no field metric at all.
* a **central finite difference** of the fitted input phase along the chart
  coordinates, for the gradient / Hessian terms in isolation.

Every bar is measured on BOTH sides of the fix on this fixture, with the
seam ``lenses_maslov._S6_INPUT_WAVEVECTOR_SADDLE`` holding the two saddles
side by side on one build (``False`` = the 5.46.0 saddle), so nothing here
pins a prior release's numbers.
"""
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements import lenses_maslov as LM
from lumenairy.glass import get_glass_index

# ===========================================================================
# Fixture -- the WP-A4 f = 6 mm N-BK7 singlet on its NA 0.05 chart
# ===========================================================================
_LAM = 1.0e-6
_K0 = 2.0 * np.pi / _LAM
_GLASS = 'N-BK7'
_R1, _R2, _THICK = 6.0e-3, -6.0e-3, 0.7e-3
_APER = 0.60e-3                      # clear aperture -> chart NA 0.05
_N, _DX = 256, 3.2e-6                # 0.819 mm span, 1.37 x the aperture
_W = 0.15e-3                         # input 1/e amplitude radius
_D_READ = 6.40e-3                    # 0.6 mm PAST best focus (5.80 mm):
                                     # a ~19 um geometric disc, so the saddle
                                     # is non-degenerate (at the focus of this
                                     # diffraction-limited singlet it is not,
                                     # and no leading-order expansion is)
_ROI_HW = 32.0e-6
_NOUT = int(round(2 * _ROI_HW / _DX))


def _n_glass():
    return float(get_glass_index(_GLASS, _LAM))


def _prescription():
    return la.make_singlet(_R1, _R2, _THICK, _GLASS, aperture=_APER)


def _grid():
    ax = (np.arange(_N) - _N / 2) * _DX
    return np.meshgrid(ax, ax, indexing='xy')


def _field(tilt_x=0.0, curv_f=None, speckle=0.0, hard_aperture=None, seed=7):
    """Gaussian, optionally tilted / curved / speckled / hard-apertured.

    ``curv_f > 0`` CONVERGES.  A real non-negative field (all four knobs off)
    has ``arg E_in == 0``, so its local wavevector is EXACTLY zero -- which is
    the premise of the byte-identity pins below.
    """
    X, Y = _grid()
    r2 = X * X + Y * Y
    E = np.exp(-r2 / (_W * _W)).astype(np.complex128)
    if hard_aperture is not None:
        E = E * (r2 <= (hard_aperture * 0.5 * _APER) ** 2)
    ph = np.zeros_like(X)
    if tilt_x:
        ph = ph + _K0 * tilt_x * X
    if curv_f:
        ph = ph - _K0 * r2 / (2.0 * float(curv_f))
    if speckle:
        ph = ph + speckle * np.random.default_rng(seed).standard_normal(
            (_N, _N))
    return E * np.exp(1j * ph) if np.any(ph) else E


def _na_input():
    """The driver's own ``input_na`` for the collimated envelope: the 3-sigma
    second moment of ``|FFT(E_in)|^2``."""
    P = np.abs(np.fft.fft2(_field())) ** 2
    f = np.fft.fftfreq(_N, d=_DX)
    FX, FY = np.meshgrid(f, f, indexing='xy')
    return 3.0 * float(np.sqrt(
        ((_LAM ** 2 * (FX ** 2 + FY ** 2)) * P).sum() / P.sum()))


# ===========================================================================
# The lumenairy-free oracle: conic raytrace of the input's own rays + a
# direct Rayleigh--Sommerfeld sum from the exit surface.
# ===========================================================================
def _sag(r2, R, K):
    if not np.isfinite(R):
        return np.zeros_like(r2)
    return r2 / (R * (1.0 + np.sqrt(np.maximum(
        1.0 - (1.0 + K) * r2 / (R * R), 0.0))))


def _dsag_over_r(r2, R, K):
    if not np.isfinite(R):
        return np.zeros_like(r2)
    return 1.0 / (R * np.sqrt(np.maximum(1.0 - (1.0 + K) * r2 / (R * R), 0.0)))


def _cosines(E):
    """``(1/k0) grad arg E`` by the conjugate-product forward difference --
    written here, independent of the library's own estimator."""
    ux = np.zeros(E.shape)
    uy = np.zeros(E.shape)
    ux[:, :-1] = np.angle(E[:, 1:] * np.conj(E[:, :-1])) / (_K0 * _DX)
    ux[:, -1] = ux[:, -2]
    uy[:-1, :] = np.angle(E[1:, :] * np.conj(E[:-1, :])) / (_K0 * _DX)
    uy[-1, :] = uy[-2, :]
    dead = np.abs(E) <= 0.0
    ux[dead] = 0.0
    uy[dead] = 0.0
    return ux, uy


def _bilinear(G, XQ, YQ):
    o = -(_N / 2) * _DX
    fx, fy = (XQ - o) / _DX, (YQ - o) / _DX
    ix = np.clip(np.floor(fx).astype(int), 0, _N - 2)
    iy = np.clip(np.floor(fy).astype(int), 0, _N - 2)
    wx, wy = fx - ix, fy - iy
    return ((1 - wx) * (1 - wy) * G[iy, ix] + wx * (1 - wy) * G[iy, ix + 1]
            + (1 - wx) * wy * G[iy + 1, ix] + wx * wy * G[iy + 1, ix + 1])


def _trace_input_rays(E, n_pupil=141):
    """The INPUT's own congruence, traced to the exit surface."""
    n = _n_glass()
    surfaces = [(_R1, 0.0), (_R2, 0.0)]
    idx = [1.0, n, 1.0]
    ux, uy = _cosines(E)
    h = np.linspace(-0.5 * _APER, 0.5 * _APER, int(n_pupil))
    dh = float(h[1] - h[0])
    HX, HY = np.meshgrid(h, h, indexing='ij')
    amp = _bilinear(np.abs(E), HX, HY)
    cpx = _bilinear(E, HX, HY)
    phase = np.where(np.abs(cpx) > 0,
                     cpx / np.maximum(np.abs(cpx), 1e-300), 1.0 + 0j)
    L, M = _bilinear(ux, HX, HY), _bilinear(uy, HX, HY)
    D = np.stack([L / idx[0], M / idx[0],
                  np.sqrt(np.maximum(1.0 - (L / idx[0]) ** 2
                                     - (M / idx[0]) ** 2, 0.0))])
    P = np.stack([HX.copy(), HY.copy(), np.zeros_like(HX)])
    opl = np.zeros_like(HX)
    alive = np.ones(HX.shape, dtype=bool)
    z_vertex = 0.0
    for si, (R, K) in enumerate(surfaces):
        if si:
            z_vertex += _THICK
        t = (z_vertex - P[2]) / np.where(np.abs(D[2]) > 1e-12, D[2], 1.0)
        for _ in range(60):
            XX, YY, ZZ = P[0] + t * D[0], P[1] + t * D[1], P[2] + t * D[2]
            rr2 = XX * XX + YY * YY
            g = _dsag_over_r(rr2, R, K)
            df = D[2] - g * (XX * D[0] + YY * D[1])
            t = t - (ZZ - z_vertex - _sag(rr2, R, K)) / np.where(
                np.abs(df) > 1e-14, df, 1.0)
        XX, YY, ZZ = P[0] + t * D[0], P[1] + t * D[1], P[2] + t * D[2]
        rr2 = XX * XX + YY * YY
        opl = opl + idx[si] * t
        P = np.stack([XX, YY, ZZ])
        g = _dsag_over_r(rr2, R, K)
        gx, gy = g * XX, g * YY
        nrm = np.sqrt(1.0 + gx * gx + gy * gy)
        Nv = np.stack([-gx / nrm, -gy / nrm, np.ones_like(nrm) / nrm])
        if si == len(surfaces) - 1:
            break
        ci = -(D * Nv).sum(axis=0)
        Nv2 = np.where((ci < 0)[None, :, :], -Nv, Nv)
        ci = np.abs(ci)
        eta = idx[si] / idx[si + 1]
        kk = 1.0 - eta * eta * (1.0 - ci * ci)
        alive &= kk > 0.0
        D = eta * D + (eta * ci - np.sqrt(np.maximum(kk, 0.0))) * Nv2
        D = D / np.sqrt((D * D).sum(axis=0))
    return {'HX': HX, 'HY': HY, 'dh': dh, 'amp': amp, 'phase': phase,
            'P': P, 'N': Nv, 'nrm': nrm, 'opl': opl, 'D': D, 'alive': alive,
            'n_last': idx[-2]}


def _oracle_landing(E, n_pupil=141):
    """Intensity-weighted mean landing of the INPUT's own rays at the readout
    plane -- an independent CHIEF-RAY oracle for the centroid."""
    tr = _trace_input_rays(E, n_pupil)
    D, Nv = tr['D'], tr['N']
    ci = -(D * Nv).sum(axis=0)
    Nv = np.where((ci < 0)[None, :, :], -Nv, Nv)
    ci = np.abs(ci)
    eta = tr['n_last']
    kk = 1.0 - eta * eta * (1.0 - ci * ci)
    Dout = eta * D + (eta * ci - np.sqrt(np.maximum(kk, 0.0))) * Nv
    Dout = Dout / np.sqrt((Dout * Dout).sum(axis=0))
    P = tr['P']
    t = ((_THICK + _D_READ) - P[2]) / np.where(np.abs(Dout[2]) > 1e-12,
                                               Dout[2], 1.0)
    xl, yl = P[0] + t * Dout[0], P[1] + t * Dout[1]
    inside = (tr['HX'] ** 2 + tr['HY'] ** 2) <= (0.5 * _APER) ** 2
    w = (tr['amp'] ** 2) * inside * (kk > 0.0) * tr['alive']
    s = float(w.sum())
    cx, cy = float((w * xl).sum() / s), float((w * yl).sum() / s)
    rms = float(np.sqrt((w * ((xl - cx) ** 2 + (yl - cy) ** 2)).sum() / s))
    return cx, cy, rms


def _oracle_field(E, centre, n_pupil=141):
    """Rayleigh--Sommerfeld sum over the exit surface onto the SAME window
    ``apply_real_lens_maslov(roi=...)`` returns."""
    tr = _trace_input_rays(E, n_pupil)
    P, Nv, dh = tr['P'], tr['N'], tr['dh']
    X, Y = P[0], P[1]
    jac = np.ones_like(X)
    jac[1:-1, 1:-1] = np.abs(
        (X[2:, 1:-1] - X[:-2, 1:-1]) * (Y[1:-1, 2:] - Y[1:-1, :-2])
        - (X[1:-1, 2:] - X[1:-1, :-2]) * (Y[2:, 1:-1] - Y[:-2, 1:-1])
    ) / (2.0 * dh) ** 2
    jac[0, :], jac[-1, :] = jac[1, :], jac[-2, :]
    jac[:, 0], jac[:, -1] = jac[:, 1], jac[:, -2]
    inside = (tr['HX'] ** 2 + tr['HY'] ** 2) <= (0.5 * _APER) ** 2
    w = (tr['amp'] / np.sqrt(np.maximum(jac, 1e-30))) * inside * tr['alive'] \
        * tr['nrm'] * jac * dh * dh
    keep = w > 1e-9 * w.max()
    Xk, Yk, Zk, wk = X[keep], Y[keep], P[2][keep], w[keep]
    phk = tr['phase'][keep] * np.exp(1j * _K0 * tr['opl'][keep])
    nx, ny, nz = Nv[0][keep], Nv[1][keep], Nv[2][keep]
    u = (np.arange(_NOUT) - _NOUT / 2) * _DX
    XS, YS = np.meshgrid(u + centre[0], u + centre[1], indexing='xy')
    xs, ys = XS.ravel(), YS.ravel()
    out = np.empty(xs.size, dtype=np.complex128)
    chunk = max(1, int(3e7 // max(Xk.size, 1)))
    for i0 in range(0, xs.size, chunk):
        sl = slice(i0, min(i0 + chunk, xs.size))
        dxp = xs[sl][:, None] - Xk[None, :]
        dyp = ys[sl][:, None] - Yk[None, :]
        dzp = ((_THICK + _D_READ) - Zk)[None, :]
        rr = np.sqrt(dxp * dxp + dyp * dyp + dzp * dzp)
        cos = (dxp * nx[None, :] + dyp * ny[None, :]
               + dzp * nz[None, :]) / rr
        out[sl] = ((wk[None, :] * cos / rr) * phk[None, :]
                   * np.exp(1j * _K0 * rr)).sum(axis=1)
    return out.reshape(_NOUT, _NOUT)


# ===========================================================================
# Small shared helpers
# ===========================================================================
def _maslov(E, method, centre=(0.0, 0.0), seam='auto', **kw):
    """One ``apply_real_lens_maslov`` call with the S6 seam held at ``seam``
    (``'auto'`` leaves the shipped default).  Returns ``(field, messages)``."""
    args = dict(prescription=_prescription(), wavelength=_LAM, dx=_DX,
                output_plane_distance=_D_READ, integration_method=method,
                roi=(centre[0], centre[1], _ROI_HW), normalize_output='none',
                poly_order=4, ray_field_samples=16, ray_pupil_samples=16)
    args.update(kw)
    old = LM._S6_INPUT_WAVEVECTOR_SADDLE
    if seam != 'auto':
        LM._S6_INPUT_WAVEVECTOR_SADDLE = seam
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            out = la.apply_real_lens_maslov(E.copy(), **args)
    finally:
        LM._S6_INPUT_WAVEVECTOR_SADDLE = old
    return out, [str(c.message) for c in caught]


def _s6_warnings(msgs):
    return [m for m in msgs if 'saddle of the OPD alone' in m]


def _fidelity(A, B):
    a, b = np.asarray(A).ravel(), np.asarray(B).ravel()
    na = np.sqrt(float(np.vdot(a, a).real))
    nb = np.sqrt(float(np.vdot(b, b).real))
    return 0.0 if na <= 0.0 or nb <= 0.0 else float(
        abs(np.vdot(a, b)) / (na * nb))


def _centroid(E, centre):
    I = np.abs(np.asarray(E)) ** 2
    ax = (np.arange(_NOUT) - _NOUT / 2) * _DX
    XS, YS = np.meshgrid(ax + centre[0], ax + centre[1], indexing='xy')
    tot = float(I.sum())
    if not np.isfinite(tot) or tot <= 0.0:
        return np.nan, np.nan
    return float((I * XS).sum() / tot), float((I * YS).sum() / tot)


def _ee(E, centre, about, radius):
    I = np.abs(np.asarray(E)) ** 2
    ax = (np.arange(_NOUT) - _NOUT / 2) * _DX
    XS, YS = np.meshgrid(ax + centre[0], ax + centre[1], indexing='xy')
    rr = np.sqrt((XS - about[0]) ** 2 + (YS - about[1]) ** 2)
    tot = float(I.sum())
    if not np.isfinite(tot) or tot <= 0.0:
        return np.nan
    return float(I[rr <= radius].sum() / tot)


def _window_on(E):
    """Readout window centred on the input's own geometric landing, snapped to
    the output pixel grid (so every arm reads the SAME pixels)."""
    gx, gy, _ = _oracle_landing(E)
    return round(gx / _DX) * _DX, round(gy / _DX) * _DX


# ===========================================================================
# 0.  The oracle's own floor, and the premise the byte-identity pins rest on
# ===========================================================================
def test_b1_the_inline_oracle_is_grid_converged():
    """The bars below may only sit above the ORACLE's own error, so measure
    it: refine the entrance sampling and watch ``1 - fidelity`` between
    consecutive grids.

    MEASURED 2026-09-13 (collimated input, this fixture): 81 -> 111 -> 141 ->
    181 -> 221 entrance samples per axis give ``1 - fid`` = 7.80e-08 /
    1.29e-08 / 9.90e-09 / 3.77e-09.  So the oracle is converged to ~1e-08 --
    five decades below the 1e-03 envelope the fidelity pins use and four
    below the 6.3e-05 at which it agrees with the library's own converged
    uniform quadrature.
    """
    E = _field()
    prev = _oracle_field(E, (0.0, 0.0), n_pupil=111)
    deltas = []
    for npu in (141, 181, 221):
        cur = _oracle_field(E, (0.0, 0.0), n_pupil=npu)
        deltas.append(1.0 - _fidelity(cur, prev))
        prev = cur
    assert deltas[-1] < 1e-6, (
        f'the oracle is not grid-converged: consecutive-grid 1-fid = '
        f'{deltas} (want the last < 1e-6; measured 3.77e-09)')
    assert deltas[-1] < deltas[0], (
        f'refining the oracle must shrink its own spread; got {deltas}')


def test_b1_the_oracle_and_the_librarys_exact_integrator_agree():
    """Cross-validation of two independent computations of the same field:
    the lumenairy-free Rayleigh--Sommerfeld oracle above, and the library's
    uniform ``'quadrature'`` (which has no saddle and is exact once resolved).

    MEASURED 2026-09-13: ``1 - fid`` = 6.2862e-05 at ``n_v2 = 128`` and
    6.2876e-05 at 192 -- converged in ``n_v2`` to 1.4e-08, so the residual
    6.3e-05 is the two methods' genuine disagreement (the oracle carries the
    field to the exit SURFACE geometrically before diffracting from it), not
    a sampling artefact.  It is the floor under every fidelity bar here.
    """
    E = _field()
    Eo = _oracle_field(E, (0.0, 0.0))
    q128, _ = _maslov(E, 'quadrature', n_v2=128)
    q192, _ = _maslov(E, 'quadrature', n_v2=192)
    d128 = 1.0 - _fidelity(q128, Eo)
    d192 = 1.0 - _fidelity(q192, Eo)
    assert abs(d192 - d128) < 1e-5, (
        f'the library quadrature is not converged in n_v2: 1-fid '
        f'{d128:.4e} -> {d192:.4e}')
    assert d192 < 1e-3, (
        f'the lumenairy-free oracle and the library exact integrator disagree '
        f'by 1-fid = {d192:.4e}; measured 6.29e-05')


def test_b1_a_real_input_has_exactly_zero_local_wavevector():
    """The premise under every byte-identity claim below: for a real,
    non-negative ``E_in`` the local-wavevector estimator returns EXACTLY
    zero, so the S6 term is not approximately absent, it is absent."""
    E = _field()
    assert np.all(E.imag == 0.0) and np.all(E.real >= 0.0)
    ux, uy = LM._local_direction_cosines(E, _DX, _DX, _LAM)
    assert np.array_equal(ux, np.zeros_like(ux))
    assert np.array_equal(uy, np.zeros_like(uy))
    assert LM._wavefront_na(E, _DX, _DX, _LAM) == 0.0


# ===========================================================================
# 1.  The defect itself: WHICH RAY does the saddle select?
# ===========================================================================
def _saddle_launch_census(E, seam):
    """Mean ``|v1|`` of the ray the converged saddle selects, and mean
    ``|v1 - v1_in|`` against the input's own local wavevector there.

    ``v1`` comes from the symplectic identity, not from the code under test:
    ``grad_u_v2 OPD_waves = -(n1/lambda) (ds1/du_v2)^T v1``.
    """
    cap = {}
    orig_sp = LM._integrate_stationary_phase
    orig_nt = LM._maslov_newton_saddle_cpu

    def sp(*a, **kw):
        cap['coef_s1x'], cap['coef_s1y'] = a[1], a[2]
        return orig_sp(*a, **kw)

    def nt(opd_eval, coef_opd, u1, u2, inbox, ni, tol, l3, l4,
           progress=None, k1_fit=None):
        out = orig_nt(opd_eval, coef_opd, u1, u2, inbox, ni, tol, l3, l4,
                      progress=progress, k1_fit=k1_fit)
        cap.update(ev=opd_eval, cop=coef_opd, u1=u1, u2=u2, lin=(l3, l4),
                   mask=inbox & out[2], u3=out[0], u4=out[1],
                   engaged=k1_fit is not None)
        return out

    LM._integrate_stationary_phase = sp
    LM._maslov_newton_saddle_cpu = nt
    try:
        _maslov(E, 'stationary_phase', centre=_window_on(E), seam=seam)
    finally:
        LM._integrate_stationary_phase = orig_sp
        LM._maslov_newton_saddle_cpu = orig_nt
    m = cap['mask']
    u1, u2 = cap['u1'][m], cap['u2'][m]
    u3, u4 = cap['u3'][m], cap['u4'][m]
    _, g3, g4, _, _, _ = cap['ev'](cap['cop'], u1, u2, u3, u4)
    g = np.stack([g3 + cap['lin'][0], g4 + cap['lin'][1]], axis=1)
    sx = cap['ev'](cap['coef_s1x'], u1, u2, u3, u4)
    sy = cap['ev'](cap['coef_s1y'], u1, u2, u3, u4)
    J = np.empty((u1.size, 2, 2))
    J[:, 0, 0], J[:, 0, 1] = sx[1], sx[2]
    J[:, 1, 0], J[:, 1, 1] = sy[1], sy[2]
    v1 = -_LAM * np.linalg.solve(np.swapaxes(J, 1, 2), g[..., None])[..., 0]
    kx, ky = LM._local_direction_cosines(E, _DX, _DX, _LAM)
    k1x = LM._sample_real_bilinear(kx, _N, _DX, _DX, sx[0], sy[0])
    k1y = LM._sample_real_bilinear(ky, _N, _DX, _DX, sx[0], sy[0])
    return {'n': int(u1.size), 'engaged': cap['engaged'],
            'v1': float(np.mean(np.hypot(v1[:, 0], v1[:, 1]))),
            'k1': float(np.mean(np.hypot(k1x, k1y))),
            'err': float(np.mean(np.hypot(v1[:, 0] - k1x, v1[:, 1] - k1y)))}


@pytest.mark.parametrize('label,kw', [
    ('tilted', dict(tilt_x=2.0 * 4.501579e-03)),
    ('converging', dict(curv_f=40.0e-3)),
])
def test_b1_the_saddle_selects_the_inputs_own_launch_ray(label, kw):
    """The audit's ``|v1|`` census, read off the converged saddle.

    The stationary point of the TOTAL integrand phase is where
    ``(v1_in - v1) . ds1/dv2 = 0``, so turning the converged saddle back into
    a launch direction through the symplectic identity has to return the
    input's own local wavevector.

    MEASURED 2026-09-13 (this fixture, ~400 in-box converged pixels):

    ===================  ==========  ==========  ==============
    input                 mean |v1|   mean |k1|  mean |v1 - k1|
    ===================  ==========  ==========  ==============
    tilted, 5.46 saddle    7.0e-19     9.00e-03     9.00e-03
    tilted, S6 saddle      9.00e-03    9.00e-03     1.1e-13
    converging, 5.46       7.0e-19     8.48e-03     8.48e-03
    converging, S6         3.43e-03    3.43e-03     6.0e-15
    ===================  ==========  ==========  ==============

    The pre-fix column is the ``v1 = 0`` collimated ray at machine zero, at
    EVERY pixel, which IS the defect; the post-fix error is ten decades below
    the input's own wavevector.  (The two ``mean |k1|`` rows differ for the
    converging input because the two saddles land on DIFFERENT entrance
    points, which sample different parts of a curved wavefront; for a uniform
    tilt they cannot.)  Both bars below sit two decades inside the measured
    values.
    """
    E = _field(**kw)
    before = _saddle_launch_census(E, seam=False)
    after = _saddle_launch_census(E, seam='auto')
    assert not before['engaged'] and after['engaged'], (
        f'premise: seam False must not engage and the default must; got '
        f'{before["engaged"]} / {after["engaged"]}')
    assert before['n'] > 100 and after['n'] > 100, 'premise: pixels converged'
    assert before['v1'] < 1e-6 * before['k1'], (
        f'premise of S6: the OPD-only saddle sits at v1 = 0 whatever the '
        f'input; measured mean |v1| = {before["v1"]:.3e} against the input\'s '
        f'own {before["k1"]:.3e}')
    assert after['err'] < 1e-2 * after['k1'], (
        f'the S6 saddle must select the input\'s own launch ray: mean '
        f'|v1 - v1_in| = {after["err"]:.3e} against |v1_in| = '
        f'{after["k1"]:.3e} (measured 1.1e-13 / 6.0e-15 against ~9e-03)')
    assert after['err'] < 1e-4 * before['err'], (
        f'the launch-direction error must collapse: {before["err"]:.3e} -> '
        f'{after["err"]:.3e}')


# ===========================================================================
# 2.  The field: fidelity, centroid and encircled energy against the oracle
# ===========================================================================
_NA_IN = None


def _na_in_cached():
    global _NA_IN
    if _NA_IN is None:
        _NA_IN = _na_input()
    return _NA_IN


@pytest.mark.parametrize('method,floor_tol,before_max', [
    ('stationary_phase', 1.5e-3, 0.75),
    ('local_quadrature', 1.5e-3, 0.95),
])
def test_b1_a_tilted_input_returns_to_the_collimated_asymptotic_floor(
        method, floor_tol, before_max):
    """The DERIVED envelope: a leading-order expansion about the RIGHT ray is
    exactly as accurate on a tilted input as on a collimated one, because a
    uniform tilt changes nothing about the chart it expands on.  So the bar is
    the method's OWN collimated fidelity against the oracle -- measured in the
    same run, never assumed -- not a constant.

    MEASURED 2026-09-13 (fidelity against the lumenairy-free oracle; the
    oracle's own floor is 1-fid ~ 1e-08 and it agrees with the library's
    converged quadrature to 6.3e-05):

    ==================  ===========  =============  =============
    method              collimated   tilt 0.5 x NA  tilt 1.0 x NA
    ==================  ===========  =============  =============
    stationary_phase       0.910229
      5.46 saddle                       0.618907       0.230122
      S6 saddle                         0.910131       0.910469
    local_quadrature       0.982759
      5.46 saddle                       0.923426       0.729791
      S6 saddle                         0.982702       0.982865
    ==================  ===========  =============  =============

    So ``|fid(tilted) - fid(collimated)|`` is 9.8e-05 / 2.4e-04 for
    stationary_phase and 5.7e-05 / 1.1e-04 for local_quadrature -- the bar is
    set at 1.5e-03, a factor 6 above the largest of them, and three decades
    below the 0.29 / 0.69 the 5.46 saddle loses at 1 x NA.  ``before_max``
    is the other side: the 5.46 saddle must be visibly WORSE than the floor,
    with a gap of 0.16 (local_quadrature, the milder of the two) to 0.52.
    """
    na = _na_in_cached()
    E0 = _field()
    floor = _fidelity(_maslov(E0, method)[0], _oracle_field(E0, (0.0, 0.0)))
    assert floor > 0.85, (
        f'premise: {method} must be usable at all on this chart for a '
        f'collimated input; measured fidelity {floor:.6f}')
    for mult in (0.5, 1.0):
        E = _field(tilt_x=mult * na)
        ctr = _window_on(E)
        Eo = _oracle_field(E, ctr)
        before = _fidelity(_maslov(E, method, ctr, seam=False)[0], Eo)
        after = _fidelity(_maslov(E, method, ctr)[0], Eo)
        assert abs(after - floor) < floor_tol, (
            f'{method} at tilt {mult} x NA: the corrected saddle must be as '
            f'accurate as on a collimated input -- fidelity {after:.6f} '
            f'against the collimated floor {floor:.6f} (measured spread '
            f'<= 2.4e-04)')
        assert before < before_max, (
            f'fail-before missing: the 5.46 saddle scores {before:.6f} at '
            f'tilt {mult} x NA, which is not distinguishable from the '
            f'{floor:.6f} floor -- the fixture no longer exhibits S6')
        assert after > before, (
            f'{method}: the corrected saddle must beat the OPD-only one '
            f'({after:.6f} vs {before:.6f})')


@pytest.mark.parametrize('method', ['stationary_phase', 'local_quadrature'])
def test_b1_the_focal_centroid_follows_an_independent_ray_trace(method):
    """Fail-before, stated as the thing a user sees: WHERE the spot lands.

    The oracle is the intensity-weighted landing point of the INPUT's own
    rays, from the inline conic raytrace -- no library propagator involved.
    A uniform input tilt ``theta`` moves it by ~``f theta`` and leaves
    ``|E_in|`` untouched, so the OPD-only saddle (whose entrance point does
    not depend on the input's phase) returns a spot that does not move at all.

    MEASURED 2026-09-13, centroid error against the ray trace, tilt
    0.5 x NA / 1.0 x NA (landing 14.889 um / 29.779 um; the traced bundle's
    own rms radius is 10.78 um):

    ==================  ===============  ==============
    method              5.46 saddle      S6 saddle
    ==================  ===============  ==============
    stationary_phase    14.657 / 26.329  0.014 / 0.066
    local_quadrature     6.228 / 12.154  0.007 / 0.029
    ==================  ===============  ==============

    Both bars are stated in the fixture's OWN measured scales, taken in the
    same run.  After: one tenth of a readout pixel, 0.32 um -- measured 0.007
    to 0.066 um, so 4.8x inside at worst.  Before: a quarter of the traced
    bundle's rms radius (10.78 um here, measured by the same ray trace) --
    measured 0.58 / 1.13 / 1.36 / 2.44 of that radius, so 2.3x outside at
    worst.  The two bars are 8.4x apart and the measured values 400-1000x.
    ``local_quadrature`` is the milder arm: its window reaches part of the
    true stationary region even when centred on the wrong ray, so it moves
    ~42 % of the way (6.23 / 12.15 um) instead of not at all.
    """
    na = _na_in_cached()
    for mult in (0.5, 1.0):
        E = _field(tilt_x=mult * na)
        gx, gy, grms = _oracle_landing(E)
        ctr = _window_on(E)
        cb = _centroid(_maslov(E, method, ctr, seam=False)[0], ctr)
        ca = _centroid(_maslov(E, method, ctr)[0], ctr)
        assert abs(ca[0] - gx) < 0.1 * _DX, (
            f'{method} at tilt {mult} x NA: corrected centroid '
            f'{ca[0] * 1e6:.4f} um vs the independent ray trace '
            f'{gx * 1e6:.4f} um (error {abs(ca[0] - gx) * 1e6:.4f} um, '
            f'bar {0.1 * _DX * 1e6:.3f} um)')
        assert abs(cb[0] - gx) > 0.25 * grms, (
            f'fail-before missing: the 5.46 saddle put the spot at '
            f'{cb[0] * 1e6:.4f} um against the {gx * 1e6:.4f} um the ray '
            f'trace demands -- an error of {abs(cb[0] - gx) / grms:.3f} of '
            f'the traced bundle rms radius {grms * 1e6:.2f} um, which is '
            f'inside the 0.25 bar, so this fixture no longer exhibits S6')
        assert abs(cb[0] - gx) > 20.0 * abs(ca[0] - gx), (
            f'{method}: the centroid error must collapse, '
            f'{abs(cb[0] - gx) * 1e6:.4f} -> {abs(ca[0] - gx) * 1e6:.4f} um')
        assert grms > 5e-6, 'premise: the readout plane carries a real disc'


@pytest.mark.parametrize('method,ee_tol', [('stationary_phase', 0.02),
                                           ('local_quadrature', 0.02)])
def test_b1_a_converging_input_gets_the_right_encircled_energy(method,
                                                               ee_tol):
    """A CONVERGING input moves the focus axially, so it changes the SIZE of
    the readout disc, not its position -- the half of S6 a centroid cannot
    see.  ``f_in = +40 mm`` on this ``f = 6 mm`` singlet puts best focus at
    5.2 mm, so at the 6.4 mm readout plane the bundle is 2.6 x wider than the
    collimated one (traced rms 28.31 um vs 10.78 um).

    MEASURED 2026-09-13, encircled energy about the oracle's own centroid
    (oracle EE(2 um) = 0.00479, EE(10 um) = 0.13663):

    ==================  ==================  ==================
    method              5.46 saddle         S6 saddle
    ==================  ==================  ==================
    stationary_phase    0.02861 / 0.56542   0.00517 / 0.14583
    local_quadrature    0.01480 / 0.24726   0.00460 / 0.13098
    ==================  ==================  ==================

    The 5.46 rows are the COLLIMATED disc (EE(10 um) 0.580 / 0.381): the
    OPD-only saddle returns an intensity pattern that does not know the input
    is converging at all.  The bar is 0.02 absolute on EE(10 um), a factor 4
    inside the smallest post-fix error (0.0092) and a factor 5 outside the
    largest pre-fix one (0.429).
    """
    E = _field(curv_f=40.0e-3)
    ctr = _window_on(E)
    Eo = _oracle_field(E, ctr)
    ref = _centroid(Eo, ctr)
    ee_o = _ee(Eo, ctr, ref, 10e-6)
    ee_b = _ee(_maslov(E, method, ctr, seam=False)[0], ctr, ref, 10e-6)
    ee_a = _ee(_maslov(E, method, ctr)[0], ctr, ref, 10e-6)
    assert abs(ee_a - ee_o) < ee_tol, (
        f'{method}: EE(10 um) {ee_a:.5f} against the oracle {ee_o:.5f}')
    assert abs(ee_b - ee_o) > 2.0 * ee_tol, (
        f'fail-before missing: the 5.46 saddle already gets EE(10 um) '
        f'{ee_b:.5f} against the oracle {ee_o:.5f}')
    assert abs(ee_a - ee_o) < abs(ee_b - ee_o)


# ===========================================================================
# 3.  Nothing that did not engage moved a bit
# ===========================================================================
@pytest.mark.parametrize('method', ['stationary_phase', 'local_quadrature',
                                    'quadrature'])
def test_b1_a_collimated_input_is_byte_identical_to_the_opd_only_saddle(
        method):
    """A flat input has ``k1 = 0`` EXACTLY, so the S6 term is not small, it is
    absent -- and the driver never builds the fit.  Pin that as byte-identity
    between the shipped default and the seam forced to the 5.46 saddle, on
    the same build, plus the absence of the S6 warning.

    (The same comparison against the 5.46.0 module itself, loaded out of
    ``git show HEAD:``, is in the WP-B1 report: 24/24 rows ``array_equal``,
    covering both A4 S6 fixture arms on all four integrators.)
    """
    E = _field()
    kw = {'n_v2': 48} if method == 'quadrature' else {}
    a, ma = _maslov(E, method, seam=False, **kw)
    b, mb = _maslov(E, method, seam='auto', **kw)
    assert np.array_equal(a, b), (
        f'{method}: a collimated input must be byte-identical with and '
        f'without the S6 machinery; max |diff| = '
        f'{np.max(np.abs(a - b)):.3e}')
    assert not _s6_warnings(ma) and not _s6_warnings(mb), (
        f'a collimated input must not trip the S6 warning; got {ma + mb}')


@pytest.mark.parametrize('method', ['quadrature', 'levin'])
def test_b1_the_integrators_without_a_saddle_are_untouched(method):
    """``'quadrature'`` and ``'levin'`` integrate the true integrand
    pointwise and have no saddle, so S6 must not reach them EVEN FOR A
    NON-COLLIMATED INPUT.

    This is why the ``(k1x, k1y)`` columns are a SEPARATE least-squares solve
    rather than two more columns on the stacked OPD / s1x / s1y right-hand
    side: widening a GEMM's right-hand side is entitled to move the existing
    columns in the last bits, which would have moved every one of these
    answers too.
    """
    E = _field(tilt_x=_na_in_cached())
    kw = {'n_v2': 48} if method == 'quadrature' else {'levin_tol': 3e-2}
    a, _ = _maslov(E, method, seam=False, **kw)
    b, _ = _maslov(E, method, seam='auto', **kw)
    assert np.array_equal(a, b), (
        f'{method} moved under S6 on a tilted input; max |diff| = '
        f'{np.max(np.abs(a - b)):.3e}')


@pytest.mark.parametrize('method', ['stationary_phase', 'local_quadrature'])
def test_b1_a_declared_collimated_input_keeps_the_opd_only_saddle(method):
    """``collimated_input=True`` is the caller declaring the input flat; it
    already re-sizes the pupil chart, and it must keep the 5.46 saddle too --
    the S6 fit is never built."""
    E = _field(tilt_x=_na_in_cached())
    a, ma = _maslov(E, method, seam=False, collimated_input=True)
    b, mb = _maslov(E, method, seam='auto', collimated_input=True)
    assert np.array_equal(a, b)
    assert not _s6_warnings(ma) and not _s6_warnings(mb)


def test_b1_a_sub_threshold_wavefront_keeps_the_opd_only_saddle():
    """Below ``_SADDLE_FLAT_INPUT_NA`` the driver treats the input as flat and
    ships the 5.46 arithmetic unchanged -- a FLATNESS declaration, not a claim
    that the residual error is zero (the constant's comment carries the
    measured size of what is being left on the table).  Pinned two-sided at
    the bar: 1.00e-03 is unchanged, 1.01e-03 is not."""
    na = LM._SADDLE_FLAT_INPUT_NA
    just_under = _field(tilt_x=na / 3.0)            # NA_wf = 1.000e-03
    just_over = _field(tilt_x=1.02 * na / 3.0)      # NA_wf = 1.020e-03
    assert LM._wavefront_na(just_under, _DX, _DX, _LAM) <= na
    assert LM._wavefront_na(just_over, _DX, _DX, _LAM) > na
    a, ma = _maslov(just_under, 'stationary_phase', seam=False)
    b, mb = _maslov(just_under, 'stationary_phase', seam='auto')
    assert np.array_equal(a, b), (
        'an input at the flatness bar must ship the 5.46 saddle unchanged')
    assert not _s6_warnings(ma) and not _s6_warnings(mb)
    c, _ = _maslov(just_over, 'stationary_phase', seam=False)
    d, _ = _maslov(just_over, 'stationary_phase', seam='auto')
    assert not np.array_equal(c, d), (
        'the gate is vacuous: an input just OVER the flatness bar did not '
        'engage the S6 saddle')


# ===========================================================================
# 4.  The two saddle solvers (CPU active-subset, xp freeze-all) stay twins
# ===========================================================================
def _synthetic_saddle_chart(order=3, seed=11):
    """A small random chart (OPD, s1x, s1y, k1x, k1y over one shared
    tensor-Chebyshev basis) plus its evaluators -- enough to drive the two
    saddle solvers and a finite difference without tracing anything."""
    from lumenairy.elements.lenses import _multi_indices_total_degree
    rng = np.random.default_rng(seed)
    mi = _multi_indices_total_degree(4, order)
    K = [np.array([k[i] for k in mi], np.int64) for i in range(4)]
    idx = {k: j for j, k in enumerate(mi)}
    M = len(mi)

    def smooth(scale):
        c = rng.standard_normal(M) * scale
        # decay with total degree so the chart is smooth, not noise
        deg = np.array([sum(k) for k in mi], float)
        return c * 0.25 ** deg

    coef_opd = smooth(4.0)
    coef_opd[idx[(0, 0, 2, 0)]] += 6.0        # a real curvature in v2
    coef_opd[idx[(0, 0, 0, 2)]] += 4.5
    coef_s1x = smooth(2.0e-5)
    coef_s1x[idx[(0, 0, 1, 0)]] += 2.0e-4
    coef_s1y = smooth(2.0e-5)
    coef_s1y[idx[(0, 0, 0, 1)]] += 2.0e-4
    coef_k1x = smooth(4.0e-3)
    coef_k1y = smooth(4.0e-3)
    return mi, K, order, (coef_opd, coef_s1x, coef_s1y, coef_k1x, coef_k1y)


def test_b1_the_cpu_and_xp_saddle_solvers_agree_with_the_s6_term():
    """``_maslov_newton_saddle_cpu`` shrinks an active subset; its GPU twin
    ``_maslov_newton_saddle_xp`` evaluates every pixel and freezes the
    converged ones.  The S6 term had to go into BOTH, identically, or the
    CuPy / NumPy backends would answer differently for a non-collimated
    input.  Driven here with ``xp = np`` and ONE shared evaluator, so the only
    difference under test is the loop structure."""
    mi, K, order, coefs = _synthetic_saddle_chart()
    coef_opd, csx, csy, ckx, cky = coefs
    rng = np.random.default_rng(3)
    n = 512
    u_s2x = rng.uniform(-0.9, 0.9, n)
    u_s2y = rng.uniform(-0.9, 0.9, n)
    inbox = np.ones(n, dtype=bool)
    inbox[::37] = False

    def ev(coef, u1, u2, u3, u4):
        return LM._opd6_xp(np, coef, K[0], K[1], K[2], K[3],
                           u1, u2, u3, u4, order)

    k1 = (csx, csy, ckx, cky, 1.0 / _LAM)
    for fit in (None, k1):
        a = LM._maslov_newton_saddle_cpu(ev, coef_opd, u_s2x, u_s2y, inbox,
                                         12, 1e-10, 0.3, -0.2, k1_fit=fit)
        b = LM._maslov_newton_saddle_xp(np, ev, coef_opd, u_s2x, u_s2y,
                                        inbox, 12, 1e-10, 0.3, -0.2,
                                        k1_fit=fit)
        tag = 'without' if fit is None else 'with'
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]), (
            f'the CPU and xp saddle solvers diverged {tag} the S6 term: '
            f'max |du_v2| = {max(np.max(np.abs(a[0] - b[0])), np.max(np.abs(a[1] - b[1]))):.3e}')
        assert np.array_equal(a[2], b[2]), (
            f'the convergence masks differ {tag} the S6 term')
    # ... and the term is not vacuous on this chart
    z = LM._maslov_newton_saddle_cpu(ev, coef_opd, u_s2x, u_s2y, inbox,
                                     12, 1e-10, 0.3, -0.2, k1_fit=None)
    w = LM._maslov_newton_saddle_cpu(ev, coef_opd, u_s2x, u_s2y, inbox,
                                     12, 1e-10, 0.3, -0.2, k1_fit=k1)
    assert np.max(np.abs(z[0] - w[0])) > 1e-6, (
        'premise: the S6 term must actually move the saddle on this chart')


def test_b1_the_input_phase_hessian_is_the_derivative_of_its_gradient():
    """``_input_phase_terms`` returns ``(g3, g4, a33, a34, a44)`` where the
    three ``a`` are the v2-derivatives of the two ``g``.  Check them against a
    central finite difference of the gradient the same function returns --
    which is what catches a dropped product-rule term (``k1`` varies along the
    chart, so ``d(k1 . ds1/du)/du`` is NOT ``k1 . d2s1/du2``).

    The mixed term is SYMMETRISED by construction, so it is compared with the
    mean of the two finite-differenced cross derivatives, and the asymmetry
    the fit leaves behind is reported.
    """
    mi, K, order, coefs = _synthetic_saddle_chart(seed=5)
    _, csx, csy, ckx, cky = coefs
    inv = 1.0 / _LAM

    def ev(coef, u1, u2, u3, u4):
        return LM._opd6_xp(np, coef, K[0], K[1], K[2], K[3],
                           u1, u2, u3, u4, order)

    def terms(u1, u2, u3, u4):
        return LM._input_phase_terms(ev(csx, u1, u2, u3, u4),
                                     ev(csy, u1, u2, u3, u4),
                                     ev(ckx, u1, u2, u3, u4),
                                     ev(cky, u1, u2, u3, u4), inv)

    rng = np.random.default_rng(17)
    n = 256
    u1 = rng.uniform(-0.8, 0.8, n)
    u2 = rng.uniform(-0.8, 0.8, n)
    u3 = rng.uniform(-0.8, 0.8, n)
    u4 = rng.uniform(-0.8, 0.8, n)
    h = 1e-5
    g3, g4, a33, a34, a44 = terms(u1, u2, u3, u4)
    g3p, g4p = terms(u1, u2, u3 + h, u4)[:2]
    g3m, g4m = terms(u1, u2, u3 - h, u4)[:2]
    g3q, g4q = terms(u1, u2, u3, u4 + h)[:2]
    g3r, g4r = terms(u1, u2, u3, u4 - h)[:2]
    fd33 = (g3p - g3m) / (2 * h)
    fd44 = (g4q - g4r) / (2 * h)
    fd34 = 0.5 * ((g3q - g3r) / (2 * h) + (g4p - g4m) / (2 * h))
    scale = max(float(np.max(np.abs(a33))), float(np.max(np.abs(a44))),
                float(np.max(np.abs(a34))), 1e-300)
    g_scale = max(float(np.max(np.abs(g3))), float(np.max(np.abs(g4))))
    # The central difference's own error floor is eps |g| / h.  MEASURED on
    # this chart: |g| = 1.870 waves per unit u, so the floor is 4.15e-11
    # absolute = 1.0e-10 of the 0.4154 Hessian scale, and it moves with h as
    # a floor should (1.1e-09 / 1.9e-10 / 2.4e-09 at h = 1e-4 / 1e-5 / 1e-6).
    # The bar is 1e-08 relative, two decades above that floor and seven below
    # the term itself.
    fd_floor = np.finfo(float).eps * g_scale / h
    assert scale > 1e6 * fd_floor, (
        f'premise: the Hessian terms ({scale:.4f}) must stand well clear of '
        f'the finite-difference floor ({fd_floor:.3e})')
    for name, an, fd in (('a33', a33, fd33), ('a44', a44, fd44),
                         ('a34', a34, fd34)):
        err = float(np.max(np.abs(an - fd))) / scale
        assert err < 1e-8, (
            f'{name} is not the derivative of the gradient it is paired '
            f'with: max relative FD mismatch {err:.3e} (measured 1.9e-10 / '
            f'5.5e-11 / 1.1e-10 for a33 / a44 / a34)')
    # The symmetrisation is load-bearing here, not decorative: this chart's
    # k1 is NOT curl-free (the two coefficient sets are independent), so the
    # two one-sided cross derivatives differ by 1.45 x the Hessian scale and
    # only their MEAN is what a34 returns.
    asym = float(np.max(np.abs((g3q - g3r) - (g4p - g4m)))) / (2 * h)
    assert asym > 0.1 * scale, (
        f'premise: the mixed derivative must be genuinely asymmetric on this '
        f'chart, or the symmetrisation is untested; got {asym / scale:.3f} '
        f'of the Hessian scale (measured 1.446)')


# ===========================================================================
# 5.  The fallback: when the chart cannot carry the input's wavevector
# ===========================================================================
def _k1_fit_report(E, **kw):
    """The driver's own ``(ray NA, fit residual, engaged)`` for this input."""
    rec = []

    def prog(phase, frac, note=''):
        if 'S6 input-wavevector saddle' in str(note):
            rec.append(str(note))
    out, msgs = _maslov(E, 'stationary_phase', progress=prog, **kw)
    if not rec:
        return None, None, False, msgs
    s = rec[0]
    return (float(s.split('ray NA')[1].split(',')[0]),
            float(s.split('residual')[1].split('(')[0]),
            'engaged' in s, msgs)


def test_b1_the_fallback_criterion_separates_the_measured_families():
    """The criterion, with both families measured on this fixture.

    The fit residual is the intensity-weighted RMS of ``A @ coef - k1`` as a
    fraction of the intensity-weighted RMS of ``k1`` itself, so
    ``_K1_FIT_RESIDUAL_MAX = 0.5`` reads "the fit explains at least 75 % of
    the local wavevector's power".

    MEASURED 2026-09-13 (this fixture, order 4):

    ==========================  ==========  ========
    input                       residual    decision
    ==========================  ==========  ========
    tilt 1 x NA                 1.44e-11    engage
    converging f = +40 mm       1.07e-05    engage
    diverging f = -25 mm        1.27e-05    engage
    hard aperture 0.6 + tilt    2.66e-01    engage
    tilt + 0.02 rad rms speckle 2.77e-01    engage
    tilt + 0.3 rad rms speckle  9.20e-01    refuse
    white-noise phase           9.64e-01    refuse
    ==========================  ==========  ========

    The gap the bar sits in is 2.77e-01 to 9.20e-01, a factor 3.3; it is
    placed 1.8x above the highest engaged case and 1.8x below the lowest
    refused one.  The larger fixture's speckle ladder (WP-B1 report) locates
    the point where the fitted saddle stops helping between residual 0.91
    and 0.96, which is why the bar is conservative rather than centred.
    """
    na = _na_in_cached()
    smooth = [('tilt', _field(tilt_x=na)),
              ('converging', _field(curv_f=40.0e-3)),
              ('diverging', _field(curv_f=-25.0e-3))]
    rough = [('0.3 rad speckle', _field(tilt_x=na, speckle=0.3)),
             ('white-noise phase', _field(tilt_x=na, speckle=6.0, seed=3))]
    smooth_res, rough_res = [], []
    for tag, E in smooth:
        na_r, res, eng, msgs = _k1_fit_report(E)
        smooth_res.append(res)
        assert eng, f'{tag}: a chart-representable wavefront must engage'
        assert not _s6_warnings(msgs), f'{tag} must not warn; got {msgs}'
    for tag, E in rough:
        na_r, res, eng, msgs = _k1_fit_report(E)
        rough_res.append(res)
        assert not eng, (
            f'{tag}: residual {res:.3e} was accepted; the fit does not '
            f'describe this input')
        assert _s6_warnings(msgs), (
            f'{tag}: the fallback must announce itself -- the caller is '
            f'getting the 5.46 saddle')
    assert max(smooth_res) < LM._K1_FIT_RESIDUAL_MAX < min(rough_res), (
        f'the bar {LM._K1_FIT_RESIDUAL_MAX} no longer sits in the measured '
        f'gap: smooth family up to {max(smooth_res):.3e}, refused family '
        f'from {min(rough_res):.3e}')
    assert min(rough_res) / max(smooth_res) > 2.0, (
        f'the two families overlap: {max(smooth_res):.3e} vs '
        f'{min(rough_res):.3e}')


@pytest.mark.parametrize('method', ['stationary_phase', 'local_quadrature'])
def test_b1_the_fallback_really_is_the_opd_only_saddle(method):
    """When the criterion refuses, the answer must be EXACTLY what 5.46
    returned -- not a third thing.  Byte-identity against the seam."""
    E = _field(tilt_x=_na_in_cached(), speckle=0.3)
    a, ma = _maslov(E, method, seam=False)
    b, mb = _maslov(E, method, seam='auto')
    assert np.array_equal(a, b), (
        f'{method}: the refused path is not the OPD-only saddle; max |diff| '
        f'= {np.max(np.abs(a - b)):.3e}')
    assert _s6_warnings(mb), 'the refused path must warn'
    # and forcing it through does change the answer, so the gate is not
    # silently inert
    c, _ = _maslov(E, method, seam=True)
    assert not np.array_equal(a, c), (
        'the fallback gate is vacuous: forcing the fit changed nothing')


def test_b1_the_s6_warning_is_gone_exactly_where_the_fix_applies():
    """The "retire or restate" half.  5.46 warned whenever the input's
    wavefront NA exceeded ``_SADDLE_FLAT_INPUT_NA``; 5.47 computes that case
    correctly and must be SILENT there, and must still warn on the fallback
    regime -- otherwise the warning has simply been deleted."""
    na = _na_in_cached()
    quiet = _field(tilt_x=na)
    loud = _field(tilt_x=na, speckle=0.3)
    for E, want, tag in ((quiet, False, 'a tilted input now computed right'),
                         (loud, True, 'a speckled input, fit refused')):
        assert LM._wavefront_na(E, _DX, _DX, _LAM) > LM._SADDLE_FLAT_INPUT_NA
        for method in ('stationary_phase', 'local_quadrature'):
            _, msgs = _maslov(E, method)
            got = bool(_s6_warnings(msgs))
            assert got is want, (
                f'{tag} ({method}): S6 warning fired={got}, want {want}')
            # 5.46 warned on BOTH, which is what makes this a restatement
            _, old = _maslov(E, method, seam=False)
            assert _s6_warnings(old), (
                f'premise: 5.46 warned on {tag} ({method})')


# ===========================================================================
# 6.  The per-call keyword and the process seam say the same thing
# ===========================================================================
@pytest.mark.parametrize('method', ['stationary_phase', 'local_quadrature'])
def test_b1_the_keyword_overrides_the_seam_in_both_directions(method):
    """``input_wavevector_saddle=`` is the per-call spelling of the process
    seam ``_S6_INPUT_WAVEVECTOR_SADDLE``, and the keyword WINS.

    Both directions are pinned, and each against the seam-driven result
    byte-for-byte -- the point of the keyword is to select one of exactly two
    computations, not to open a third:

    * seam ``True`` + ``input_wavevector_saddle=False``  ==  seam ``False``
    * seam ``False`` + ``input_wavevector_saddle=True``  ==  seam ``True``

    plus the two no-op arms (keyword agreeing with the seam, and the keyword
    on its own against the shipped default), so a reading of the precedence
    that simply ignored one of the two inputs would fail here.  The fixture is
    a tilted input, which engages: on a flat one both settings collapse to the
    same arithmetic and the test would be vacuous, which the last assertion
    guards against.
    """
    E = _field(tilt_x=_na_in_cached())
    opd_only, _ = _maslov(E, method, seam=False)
    fitted, _ = _maslov(E, method, seam=True)
    assert not np.array_equal(opd_only, fitted), (
        'premise: the two saddles must differ on this fixture, or the '
        'override has nothing to select between')

    # the keyword overrides a seam that says the opposite ...
    kw_off, msgs_off = _maslov(E, method, seam=True,
                               input_wavevector_saddle=False)
    assert np.array_equal(kw_off, opd_only), (
        f'{method}: input_wavevector_saddle=False did not override the '
        f'process seam; max |diff| = {np.max(np.abs(kw_off - opd_only)):.3e}')
    assert _s6_warnings(msgs_off), (
        'asking for the OPD-only saddle on a non-flat input must still warn')
    kw_on, msgs_on = _maslov(E, method, seam=False,
                             input_wavevector_saddle=True)
    assert np.array_equal(kw_on, fitted), (
        f'{method}: input_wavevector_saddle=True did not override the '
        f'process seam; max |diff| = {np.max(np.abs(kw_on - fitted)):.3e}')
    assert not _s6_warnings(msgs_on), (
        'the fitted saddle on a chart-representable wavefront must be silent')

    # ... and agrees with itself when the two do not conflict
    assert np.array_equal(
        _maslov(E, method, seam=False, input_wavevector_saddle=False)[0],
        opd_only)
    assert np.array_equal(
        _maslov(E, method, seam='auto', input_wavevector_saddle=None)[0],
        _maslov(E, method, seam='auto')[0])


def test_b1_the_keyword_is_classified_by_lens_config():
    """``lens_config`` must carry a written decision for every keyword-only
    parameter of the seven lens entry points (WP-A16's "silently ignored
    kwarg" detector).  This one is deliberately NOT a config field: which
    stationary point to expand about is a property of the INPUT FIELD, so it
    cannot travel in a ``LensNumerics`` that is reused across fields.

    Pinned here as well as in
    ``test_audit2609_a16_lens_config_round_trip.py`` so the two halves of the
    change -- the keyword and its classification -- cannot land apart.
    """
    from lumenairy.elements import lens_config as lc
    entry = lc.KWARG_ONLY.get('apply_real_lens_maslov', {})
    assert 'input_wavevector_saddle' in entry, (
        'lens_config.KWARG_ONLY does not classify input_wavevector_saddle; '
        'the A16 census will fail on apply_real_lens_maslov')
    reason = entry['input_wavevector_saddle']
    assert isinstance(reason, str) and len(reason) >= 30, (
        f'the exclusion reason must be a real sentence; got {reason!r}')
    import inspect
    params = inspect.signature(la.apply_real_lens_maslov).parameters
    assert params['input_wavevector_saddle'].kind is inspect.Parameter.KEYWORD_ONLY
    assert params['input_wavevector_saddle'].default is None, (
        'the documented default is None (auto); a default that disagreed with '
        'the signature is exactly what the A15a default-identity detector and '
        'the A16 table comparison exist to catch')


# ===========================================================================
# 7.  VERIFY-B1 -- the adversarial re-verification's own pins
# ===========================================================================
def _s6_report(E, method='stationary_phase', **kw):
    """``(ray NA, s1 chart fit, k1 fit residual, engaged, messages)`` as the
    driver itself reports them.  Independent of :func:`_k1_fit_report` so the
    two parsers cannot fail together."""
    rec = []

    def prog(phase, frac, note=''):
        if 'S6 input-wavevector saddle' in str(note):
            rec.append(str(note))
    out, msgs = _maslov(E, method, progress=prog, **kw)
    if not rec:
        return out, None, None, None, False, msgs
    s = rec[0]

    def field(name, stop):
        # ``None`` for a field this build does not report, so the pins that
        # only need the k1 half stay readable against a build without the
        # chart half.
        return (float(s.split(name)[1].split(stop)[0]) if name in s else None)
    return (out, field('ray NA', ','), field('s1 chart fit', ','),
            field('k1 fit residual', '('), 'engaged' in s, msgs)


def _phase_rms_waves(A, B):
    """Intensity-weighted RMS of ``arg(A conj(B))`` after removing the best
    global phase, in WAVES, over the pixels above 1 % of ``B``'s peak."""
    a, b = np.asarray(A), np.asarray(B)
    w = np.abs(b) ** 2
    if not np.isfinite(w).all() or w.max() <= 0.0:
        return np.inf
    m = w >= 0.01 * w.max()
    g = np.vdot(b, a)
    if abs(g) == 0.0 or not m.any():
        return np.inf
    z = (a * np.conj(b) * np.conj(g / abs(g)))[m]
    return float(np.sqrt((w[m] * np.angle(z) ** 2).sum() / w[m].sum())
                 / (2.0 * np.pi))


_NA_LENS = 0.5 * _APER / 6.0e-3          # this singlet's own NA, 0.05


@pytest.mark.parametrize('method', ['stationary_phase', 'local_quadrature'])
def test_verify_b1_the_gate_refuses_a_chart_that_cannot_carry_ds1_dv2(method):
    """The S6 term is ``k1 . ds1/dv2``: BOTH factors have to be
    chart-representable, and a uniform tilt passes the ``k1`` half perfectly
    (``k1`` is a constant) however badly the chart itself is fitted.

    The failure this pins, MEASURED 2026-09-13 against an exact conic-raytrace
    + Kirchhoff oracle on this fixture and on an independent f = 13.3 mm
    N-SF11 / 1.55 um one, with the tilt driving ``na_proxy`` (and so the pupil
    box the order-4 chart must span) upwards:

    ==================  ==========  ==========  ==================
    tilt / lens NA      s1 fit      k1 fit      fidelity, sp / lq
    ==================  ==========  ==========  ==================
    1.0                 3.3e-04     7.1e-10     0.937 / 0.985
    1.5                 1.3e-03     1.3e-14     0.932 / 0.984
    2.0                 4.1e-03     2.6e-14     0.000 / 0.006
    4.0                 6.7e-03     1.5e-14     0.000 / 0.000
    ==================  ==========  ==========  ==================

    -- against a collimated floor of 0.931 / 0.984, while the exact pointwise
    ``'quadrature'`` on the SAME chart is still 0.97 at tilt 2x.  So what
    fails is the saddle riding on the fit, and the ``k1`` residual cannot see
    it.  Two-sided: the engaged row must stay engaged and silent, the refused
    row must warn and name the chart, and ``_S1_FIT_RESIDUAL_MAX`` must sit
    strictly between the two measured residuals.
    """
    good = _field(tilt_x=1.0 * _NA_LENS)
    bad = _field(tilt_x=2.0 * _NA_LENS)
    _, _, s1_good, k1_good, eng_good, msg_good = _s6_report(
        good, method, centre=_window_on(good))
    _, _, s1_bad, k1_bad, eng_bad, msg_bad = _s6_report(
        bad, method, centre=_window_on(bad))
    assert eng_good and not _s6_warnings(msg_good), (
        f'{method}: tilt 1x the lens NA (s1 fit {s1_good:.2e}) is well inside '
        f'every bar and must engage silently; warnings {msg_good}')
    assert not eng_bad, (
        f'{method}: tilt 2x the lens NA was accepted at s1 fit residual '
        f'{s1_bad:.2e}; the saddle would be placed on a ray the chart has '
        f'itself misplaced')
    assert _s6_warnings(msg_bad), 'the refusal must announce itself'
    why = _s6_warnings(msg_bad)[0]
    assert 'poly_order' in why and 'input_na' in why, (
        f'the warning must name the two remedies that repair the chart; '
        f'got {why!r}')
    assert max(k1_good, k1_bad) < LM._K1_FIT_RESIDUAL_MAX, (
        'premise: the k1 half of the gate is blind to this failure -- both '
        'rows fit k1 perfectly, which is why the s1 half exists')
    assert s1_good < LM._S1_FIT_RESIDUAL_MAX < s1_bad, (
        f'the bar {LM._S1_FIT_RESIDUAL_MAX:g} no longer sits in the measured '
        f'gap {s1_good:.2e} .. {s1_bad:.2e}')
    # and the caller can still force it, which is what makes this a policy and
    # not a wall.  (The two ANSWERS need not differ here: at this tilt the
    # OPD-only saddle puts the spot outside the window on the input's own
    # landing and the forced saddle does not converge, so both can be the
    # all-zero patch.  What must differ is the DECISION.)
    _, _, _, _, eng_forced, m_forced = _s6_report(
        bad, method, centre=_window_on(bad), input_wavevector_saddle=True)
    assert eng_forced and not _s6_warnings(m_forced), (
        'input_wavevector_saddle=True must override the chart gate')


@pytest.mark.parametrize('method', ['stationary_phase', 'local_quadrature'])
def test_verify_b1_the_input_phase_is_not_double_counted(method):
    """The deviation from the WP-A4 design: the input phase reaches the answer
    ONLY through the complex ``E_in`` the integrand samples, never through
    ``opd_star`` / ``opd_v``.

    Scored as PHASE, not intensity -- a double count is a pure phase error and
    every centroid / EE pin in this file is blind to it.  The reference is the
    exact pointwise ``'quadrature'`` on the same chart, which has no saddle;
    the envelope is the method's OWN collimated phase error, because a
    leading-order expansion about the right ray is exactly as accurate on a
    tilted input as on a flat one.

    MEASURED 2026-09-13, intensity-weighted RMS phase difference from
    ``'quadrature'`` in waves:

    ===================  ==================  ==================
    input                stationary_phase    local_quadrature
    ===================  ==================  ==================
    collimated (floor)   0.0664              0.0271
    tilt 0.5x lens NA    0.0667              0.0271
    tilt 1.0x lens NA    0.0641              0.0251
    WP-A4 design's
      extra term         0.2801 / 0.2898     (same site, lq)
    ===================  ==================  ==================

    So the tilted rows sit at 0.97-1.00 of the floor and the defect scale is
    4.2x it.  The bar is 1.5x the floor -- a factor 2.8 below the defect and
    1.5 above the measurement.
    """
    flat = _field()
    floor_ref, _ = _maslov(flat, 'quadrature', n_v2=96)
    floor_got, _ = _maslov(flat, method)
    floor = _phase_rms_waves(floor_got, floor_ref)
    assert 0.0 < floor < 0.15, (
        f'premise: the collimated floor must be a real number, got {floor}')
    for mult in (0.5, 1.0):
        E = _field(tilt_x=mult * _NA_LENS)
        c = _window_on(E)
        ref, _ = _maslov(E, 'quadrature', centre=c, n_v2=96)
        got, msgs = _maslov(E, method, centre=c)
        assert not _s6_warnings(msgs), 'premise: this tilt must engage'
        ph = _phase_rms_waves(got, ref)
        assert ph <= 1.5 * floor, (
            f'{method}: tilt {mult}x the lens NA carries {ph:.4f} waves RMS '
            f'of phase error against the exact quadrature, above 1.5x the '
            f'collimated floor {floor:.4f} -- the input phase is being '
            f'counted twice (fitted into the exponent AND sampled through '
            f'E_in), or the saddle is on the wrong ray')


def test_verify_b1_the_k1_fit_residual_is_the_statistic_it_claims_to_be():
    """``_K1_FIT_RESIDUAL_MAX`` is compared against the intensity-weighted RMS
    of ``A @ coef - k1`` over the RMS of ``k1`` itself, so for white phase
    noise of ``sigma`` radians on a uniform tilt ``theta`` the residual is
    predictable with no free parameter:

        residual ~ sigma * sqrt(2) / (k0 dx theta)

    (the two axes' forward differences are independent, and a degree-4 chart
    can fit none of the noise).  Pinned to a factor of 2 either side, which is
    what separates a calibration drift from the estimator changing meaning.

    MEASURED 2026-09-13 on this fixture at tilt 0.5x the lens NA:
    sigma = 0.002 / 0.01 / 0.05 rad rms -> 4.7e-03 / 2.3e-02 / 1.1e-01,
    against 5.6e-03 / 2.8e-02 / 1.4e-01 predicted (ratio 0.83, the bilinear
    sampling of the noisy grid).  All three sit one to two DECADES below the
    0.5 bar while the field fidelity against the exact ``'quadrature'`` has
    already fallen from the collimated floor to 0.66 (sigma = 0.01) and 4e-04
    (sigma = 0.05): the bar is a statement about the fit's VALUE, and the
    saddle also consumes its two DERIVATIVES.  See VERIFY_WP-B1.md.
    """
    theta = 0.5 * _NA_LENS
    for sigma in (0.002, 0.01, 0.05):
        E = _field(tilt_x=theta, speckle=sigma)
        _, _, _, res, eng, msgs = _s6_report(E, centre=_window_on(E))
        pred = sigma * np.sqrt(2.0) / (_K0 * _DX * theta)
        assert 0.5 * pred <= res <= 2.0 * pred, (
            f'speckle {sigma} rad rms: the driver reports residual '
            f'{res:.3e}, against {pred:.3e} predicted -- the k1 fit residual '
            f'no longer means what _K1_FIT_RESIDUAL_MAX is compared against')
        assert eng and not _s6_warnings(msgs), (
            f'speckle {sigma} rad rms measures {res:.3e}, far below the '
            f'{LM._K1_FIT_RESIDUAL_MAX} bar, so it must still engage -- this '
            f'records how loose that bar is, not that the answer is good')
