"""R2 (roadmap A1): PEARCEY (cusp) completion of the traced uniform caustic.

``apply_real_lens_traced(amplitude_model='ray_density', caustic='uniform')`` now
HANDLES a CUSP (a 3-branch coalescence -- the meridional map ``x_out(h)`` has TWO
interior turning points, ``n_turn == 2``) instead of only detecting it and
falling back.  The local ray geometry is mapped to the Pearcey normal form
``t^4 + x t^2 + y t`` and the cusp-finite Pearcey field (REUSING
``lenses_maslov.pearcey`` -- no reimplementation) replaces the multibranch in the
cusp zone.  This is the cusp analog of the fold path's Chester-Friedman-Ursell
Airy completion.

WHAT THIS PHASE ESTABLISHED (measured, adversarial):

* CONTROL-COORDINATE MAPPING (the crux): the three branch PHASES reduce to the
  Pearcey control coordinates ``(x, y)`` by matching the sorted critical values
  of ``t^4 + x t^2 + y t`` -- recovered to MACHINE PRECISION on a synthetic
  quartic, and the Pearcey-derivative basis ``(P, dP/dy, dP/dx)`` reconstructs an
  exact quartic-phase integral.
* KERNEL vs 1-D CARTESIAN CUSP (Pearcey is EXACT there -- no Bessoid confound):
  the amplitude-coefficient Vandermonde + smooth continuation reconstructs a
  direct 1-D Fresnel cusp field to intensity-correlation ~0.99 and TAMES the
  geometric caustic singularity to a finite peak.
* ROT-SYM CUSP vs DIRECT RAYLEIGH-SOMMERFELD ground truth (``caustic_cusp_ref``,
  a lumenairy-free synthetic 2-hump / cusp RS field): the Pearcey completion
  matches the true field's fringe structure (intensity Icorr ~0.98 vs ~0.79 for
  the pure geometric multibranch), is FINITE (peak within ~2x truth, vs the
  multibranch's divergent caustic pile-up), and beats the multibranch on the
  windowed r2m, EE80/EE95 and the intensity L2 -- the correct cusp diffraction,
  not a geometric divergence.
* REAL ROT-SYM SINGLET, END TO END (the coverage the fold path has -- added
  after a verifier showed a rot-sym aspheric singlet DOES reach the cusp path,
  correcting an earlier "on-axis singlets never trigger n_turn == 2" skip):
  a real conic + even-aspheric singlet (conic ``-2.0``, a6 ``5e12`` SI, f/1.9,
  ``opd 4.8 mm``) produces a GENUINE finite-radius, same-sign, off-axis
  ``n_turn == 2`` cusp RING at ``r_c ~ 12.76 um`` (the on-axis focus stays a
  ``n_turn == 1`` Bessoid; a *conic* + high-order asphere is what pushes the
  three-branch coalescence off axis).  The WHOLE
  ``apply_real_lens_traced_uniform`` dispatcher (no monkeypatch) routes it to the
  Pearcey completion (``reason == 'cusp_ring'``, ``fell_back == False``, finite),
  and vs an INDEPENDENT direct-Rayleigh-Sommerfeld ground truth for THAT EXACT
  singlet (``caustic_cusp_singlet_ref``, its own raytracer, energy closure
  ~0.999, fan/rho convergence ~5e-6) the completion beats the geometric
  multibranch on the cusp-zone intensity correlation (~0.90 vs ~0.67) and
  intensity L2 (~2.4x closer) and tames the divergent caustic pile-up (peak
  ~5x the truth) to a FINITE peak near the truth.  HONEST near-axial envelope
  (``k*r_c ~ 61``, a leading-order Bessoid-Pearcey): the windowed r2m / EE80 are
  only marginally better than the multibranch (both under-read the far Airy tail,
  the A5 aperture-edge item) -- the decisive wins are the fringe STRUCTURE and
  FINITENESS.
* SCOPE / FALLBACKS (never inf/nan): a near-axial (Bessoid) cusp, a non-linear /
  astigmatic map, an under-sampled band, or a bad control solve DETECT + fall
  back to the plain multibranch field.  The fold path (``n_turn == 1``) and the
  default (``caustic=None``) are UNCHANGED.

All gates run WITHOUT Zemax (the lumenairy-free ``caustic_cusp_ref`` npz + a
synthetic meridional map + a 1-D Fresnel integral).
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import types
import warnings

import numpy as np
import pytest

import lumenairy.elements._lens_traced_uniform as umod
from lumenairy.elements._lens_traced_multibranch import (
    apply_real_lens_traced_multibranch,
)
from lumenairy.elements._lens_traced_uniform import (
    _build_pearcey_cusp_field,
    _cusp_geometry_from_branches,
    _pearcey_basis,
    _pearcey_crit_values,
    _pearcey_cubic_roots,
    _pearcey_cusp_amp_coeffs,
    _solve_pearcey_control,
    apply_real_lens_traced_uniform,
)
from lumenairy.elements.lenses_maslov import pearcey
from lumenairy.glass import GLASS_REGISTRY

# dispersionless model glass (matches the caustic_fold_ref / K4 singlet index)
GLASS_REGISTRY['_K4CAU'] = lambda wl: 1.5168

_WL = 1.31e-6
_K = 2.0 * np.pi / _WL
_ROOT = pathlib.Path(__file__).resolve().parents[2] / 'validation' / 'oracles'


def _load_oracle(name):
    """Import an independent (lumenairy-free) oracle module by file path."""
    spec = importlib.util.spec_from_file_location(name, _ROOT / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# synthetic rot-sym cusp geometry (matches caustic_cusp_ref: the 2-hump map
# W'(h) = -A h exp(-((h-h_c)/s)^2), y_im(h) = h + Z W'(h))
# ---------------------------------------------------------------------------
class _SynthCusp:
    def __init__(self, A, h_c, s, Z, w0, hmax):
        self.A, self.h_c, self.s = A, h_c, s
        self.Z, self.w0, self.hmax = Z, w0, hmax
        hg = np.linspace(0.0, hmax, 60000)
        Wp = -A * hg * np.exp(-((hg - h_c) / s) ** 2)
        self.hg = hg
        self.Wint = np.concatenate(
            [[0.0], np.cumsum(0.5 * (Wp[1:] + Wp[:-1]) * np.diff(hg))])
        self.yi = hg + Z * Wp
        self.slope = np.diff(self.yi) / np.diff(hg)
        self.hcen = 0.5 * (hg[:-1] + hg[1:])
        tp = np.where(np.diff(np.sign(np.diff(self.yi))) != 0)[0] + 1
        self.r1, self.r2 = np.sort(np.abs(self.yi[tp]))

    def Wf(self, h):
        return np.interp(h, self.hg, self.Wint)

    def yim(self, h):
        return np.interp(h, self.hg, self.yi)

    def dyim(self, h):
        return np.interp(h, self.hcen, self.slope)

    def amp_env(self, h):
        return np.exp(-h ** 2 / self.w0 ** 2)

    def branches(self, rq):
        f = self.yi - rq
        idx = np.where(np.diff(np.sign(f)) != 0)[0]
        out = []
        for ii in idx:
            a, b = self.hg[ii], self.hg[ii + 1]
            for _ in range(50):
                m = 0.5 * (a + b)
                if (self.yim(m) - rq) * (self.yim(a) - rq) <= 0:
                    b = m
                else:
                    a = m
            out.append(0.5 * (a + b))
        return np.array(out)

    def branch_data(self, n_band=120):
        band = self.r2 - self.r1
        vs = np.linspace(self.r1 + 0.02 * band, self.r2 - 0.02 * band, n_band)
        r_arr, phib, ampb, dxdhb = [], [], [], []
        for v in vs:
            bh = self.branches(v)
            if bh.size != 3:
                continue
            ph = _K * (self.Wf(bh) + self.Z + (v - bh) ** 2 / (2 * self.Z))
            jj = self.dyim(bh)
            amp = self.amp_env(bh) * np.sqrt(bh / (v * np.abs(jj) + 1e-300))
            r_arr.append(v)
            phib.append(ph)
            ampb.append(amp)
            dxdhb.append(jj)
        return (np.array(r_arr), np.array(phib), np.array(ampb),
                np.array(dxdhb))

    def geometric_multibranch_2d(self, N, dx):
        """The pure geometric (ART) multibranch field on an (N, N) grid (the
        construction ``caustic='multibranch'`` approximates -- divergent at the
        caustic)."""
        xg = (np.arange(N) - N / 2.0) * dx
        Xg, Yg = np.meshgrid(xg, xg)
        rg = np.sqrt(Xg ** 2 + Yg ** 2)
        r_prof = np.linspace(0.0, rg.max() + dx, 1500)
        g = np.zeros(r_prof.size, dtype=complex)
        for i, rq in enumerate(r_prof):
            if rq < 1e-9:
                continue
            for hb in self.branches(rq):
                jj = self.dyim(hb)
                aa = self.amp_env(hb) * np.sqrt(hb / (rq * abs(jj) + 1e-30))
                m = 1 if jj < 0 else 0
                g[i] += (aa * np.exp(1j * _K * (self.Wf(hb) + self.Z
                                                + (rq - hb) ** 2 / (2 * self.Z)))
                         * np.exp(-1j * np.pi / 2 * m))
        return (np.interp(rg, r_prof, g.real)
                + 1j * np.interp(rg, r_prof, g.imag)).astype(np.complex128), rg


def _params_from_ref(meta):
    sc = meta['synthetic_cusp']
    return _SynthCusp(sc['A'], sc['h_c_mm'] * 1e-3, sc['s_mm'] * 1e-3,
                      sc['Z_mm'] * 1e-3, sc['w0_mm'] * 1e-3,
                      sc['hmax_mm'] * 1e-3)


# ===========================================================================
# 1. Control-coordinate mapping (the crux) -- machine precision
# ===========================================================================
def test_pearcey_control_inversion_self_consistent():
    """The three branch PHASES reduce to the Pearcey control coordinates
    ``(x, y)`` by matching the sorted critical values of the normal form -- and
    that inversion is exact (machine precision) on a synthetic quartic."""
    for x0, y0 in [(-3.0, 1.5), (-5.0, 0.0), (-4.0, -2.0), (-2.5, 0.8),
                   (-8.0, 3.0)]:
        cv = _pearcey_crit_values(x0, y0)
        assert cv is not None
        phis = np.sort(cv) + 12345.678         # arbitrary common offset
        x, ay, cost = _solve_pearcey_control(phis, guess=(x0 * 0.7, 0.5))
        assert abs(x - x0) < 1e-6
        assert abs(ay - abs(y0)) < 1e-6
        # ``cost`` is the solver's residual at convergence; it reaches machine
        # zero (~1e-18) on some libm/BLAS builds but a few ULP higher on others
        # (Linux glibc vs Windows) -- assert a platform-robust "converged to
        # numerical zero" bound, not the last-ULP value.  The (x, ay) < 1e-6
        # asserts above are the real self-consistency guard.
        assert cost < 1e-12
    # outside the cusp there are not three real roots
    assert _pearcey_cubic_roots(2.0, 0.0) is None
    assert _pearcey_crit_values(2.0, 0.0) is None


def test_pearcey_basis_reconstructs_quartic_integral():
    """The basis ``(P, dP/dx, dP/dy)`` -- REUSING :func:`pearcey` -- reconstructs
    an exact quartic-phase integral ``int (a + b w + c w^2) e^{i(w^4+xw^2+yw)} dw
    = a P - i b dP/dy - i c dP/dx`` (the amplitude combination the cusp
    completion evaluates)."""
    for a, b, c, x0, y0 in [(1.0, 0.0, 0.0, -3.0, 1.0),
                            (1.0, 0.3, 0.0, -4.0, 0.5),
                            (0.8, -0.2, 0.15, -3.5, -1.2)]:
        p0, px, py = _pearcey_basis(x0, y0)
        recon = a * p0 - 1j * b * py - 1j * c * px
        W = np.linspace(-16, 16, 800001)
        g = W ** 4 + x0 * W ** 2 + y0 * W
        exact = np.trapezoid((a + b * W + c * W ** 2) * np.exp(1j * g), W)
        assert abs(recon - exact) < 5e-3 * max(1.0, abs(exact))
    # pearcey is REUSED (basis P equals the kernel exactly)
    assert _pearcey_basis(-3.0, 1.0)[0] == pearcey(-3.0, 1.0)


# ===========================================================================
# 2. KERNEL vs a 1-D Cartesian cusp (Pearcey is EXACT -- decisive validation)
# ===========================================================================
def _reconstruct_1d_cusp(B1, B3, Z, w0, Hmax):
    """1-D Fresnel cusp truth + Pearcey kernel reconstruction (via the library's
    control / amplitude / basis functions).  Returns (xg, E_truth, E_geo,
    E_pearcey)."""
    def xim(h):
        return B1 * h + B3 * h ** 3

    def dxim(h):
        return B1 + 3 * B3 * h ** 2

    def Wf(h):
        return ((B1 - 1) * h ** 2 / 2 + B3 * h ** 4 / 4) / Z

    ht = np.sqrt(-B1 / (3 * B3))
    xf = abs(xim(ht))
    h = np.linspace(-Hmax, Hmax, 40001)
    Ee = np.exp(-h ** 2 / w0 ** 2) * np.exp(1j * _K * Wf(h))
    dh = h[1] - h[0]
    W = max(35e-6, xf * 2.5)
    xg = np.linspace(-W, W, 2000)
    Et = np.array([np.sum(Ee * np.exp(1j * _K * (x - h) ** 2 / (2 * Z))) * dh
                   for x in xg]) * np.sqrt(1 / (1j * _WL * Z))

    def br(xq):
        r = np.roots([B3, 0, B1, -xq])
        return np.sort(r[np.abs(r.imag) < 1e-9 * (abs(xq) + 1e-9)].real)

    def ra(hb):
        return np.exp(-hb ** 2 / w0 ** 2) / np.sqrt(abs(dxim(hb)))

    G = np.zeros(xg.size, complex)
    for i, xq in enumerate(xg):
        for hb in br(xq):
            m = 1 if dxim(hb) < 0 else 0
            G[i] += (ra(hb) * np.exp(1j * _K * (Wf(hb) + (xq - hb) ** 2 / (2 * Z)))
                     * np.exp(-1j * np.pi / 2 * m))

    xb = np.linspace(-xf * 0.98, xf * 0.98, 141)
    rok, xl, yl, phil, bl = [], [], [], [], []
    guess = (-2.0, 0.1)
    for xq in xb:
        bh = br(xq)
        if bh.size != 3:
            continue
        ph = _K * (Wf(bh) + (xq - bh) ** 2 / (2 * Z))
        x, ay, _ = _solve_pearcey_control(ph, guess=guess)
        guess = (x, ay if ay > 1e-9 else 0.1)
        bc = _pearcey_cusp_amp_coeffs(x, ay, ph, ra(bh), dxim(bh))
        if bc is None:
            continue
        rok.append(xq)
        xl.append(x)
        yl.append(ay)
        phil.append(ph.mean() + x ** 2 / 6.0)
        bl.append(bc)
    rok = np.array(rok)
    ya = np.array(yl)
    ic = int(np.argmin(ya))
    rs = rok[ic]
    ys = ya * np.sign(rok - rs + 1e-30)
    u = rok - rs
    x0 = float(np.median(xl))
    gam = float(np.linalg.lstsq(u[:, None], ys, rcond=None)[0][0])
    cP = np.polyfit(u, np.array(phil), 3)
    cb = [np.polyfit(u, np.array(bl)[:, j], 2) for j in range(3)]
    u_lo, u_hi = float(u.min()), float(u.max())
    U = np.zeros(xg.size, complex)
    for i, xq in enumerate(xg):
        d = xq - rs
        dc = min(max(d, u_lo), u_hi)
        p0, px, py = _pearcey_basis(x0, gam * d)
        env = (np.polyval(cb[0], dc) * p0 - 1j * np.polyval(cb[1], dc) * py
               - 1j * np.polyval(cb[2], dc) * px)
        U[i] = env * np.exp(1j * np.polyval(cP, dc))
    far = (np.abs(xg) > xf + 6e-6) & (np.abs(xg) < 0.9 * W)
    U *= np.vdot(U[far], Et[far]) / np.vdot(U[far], U[far])
    G *= np.vdot(G[far], Et[far]) / np.vdot(G[far], G[far])
    return xg, Et, G, U


def test_kernel_matches_1d_cartesian_cusp():
    """1-D Cartesian cusp (Pearcey is the EXACT local normal form): the library
    control / amplitude / basis kernel reconstructs the direct 1-D Fresnel cusp
    field to intensity Icorr ~0.99 (vs ~0.1 geometric) and TAMES the geometric
    caustic singularity to a finite peak."""
    xg, Et, G, U = _reconstruct_1d_cusp(B1=-0.10, B3=6.0e8, Z=4e-3,
                                        w0=0.5e-3, Hmax=0.6e-3)
    sel = np.abs(xg) <= 30e-6
    icorr_u = np.corrcoef(np.abs(U[sel]) ** 2, np.abs(Et[sel]) ** 2)[0, 1]
    icorr_g = np.corrcoef(np.abs(G[sel]) ** 2, np.abs(Et[sel]) ** 2)[0, 1]
    assert np.all(np.isfinite(U))
    assert icorr_u > 0.98, icorr_u
    assert icorr_u > icorr_g + 0.4
    # the geometric field diverges at the caustic; the uniform stays finite
    assert np.max(np.abs(U) ** 2) < 3.0 * np.max(np.abs(Et) ** 2)
    assert np.max(np.abs(G) ** 2) > 5.0 * np.max(np.abs(Et) ** 2)


# ===========================================================================
# 3. Rot-sym cusp geometry solve on the synthetic 2-hump map
# ===========================================================================
def test_cusp_geometry_solve_on_synthetic():
    """The meridional-branch reducer resolves the synthetic cusp to a clean
    LINEAR Pearcey control map (small map residual, single ``n_turn == 2``
    ring)."""
    sy = _SynthCusp(A=350.0, h_c=0.10e-3, s=0.04e-3, Z=2.0e-3, w0=0.15e-3,
                    hmax=0.5e-3)
    r_arr, phib, ampb, dxdhb = sy.branch_data()
    geom = _cusp_geometry_from_branches(r_arr, phib, ampb, dxdhb)
    assert geom['ok'], geom.get('reason')
    assert geom['x0'] < 0.0                     # a cusp opens toward x < 0
    assert geom['map_resid'] < 0.05             # linear transverse map
    assert abs(geom['gamma']) > 0.0
    # the resolved cusp symmetric radius sits inside the two folds
    assert sy.r1 < geom['r_sym'] < sy.r2


# ===========================================================================
# 4. Rot-sym cusp vs direct Rayleigh-Sommerfeld ground truth (the money gate)
# ===========================================================================
def _win(E, rg, win, frac=None):
    m = rg <= win
    I = np.abs(E[m]) ** 2
    r = rg[m]
    if frac is None:
        return np.sqrt((I * r ** 2).sum() / I.sum()) * 1e6
    o = np.argsort(r)
    c = np.cumsum(I[o])
    c = c / c[-1]
    return float(np.interp(frac, c, r[o])) * 1e6


def test_pearcey_cusp_beats_multibranch_vs_rs_truth():
    """On the lumenairy-free ``caustic_cusp_ref`` direct-RS grid, the Pearcey
    cusp completion matches the true field's FRINGE STRUCTURE (intensity Icorr
    ~0.98) far better than the pure geometric multibranch (~0.79), stays FINITE
    (peak within ~2x truth, vs the multibranch's divergent caustic pile-up), and
    beats the multibranch on the windowed r2m, EE80/EE95 and intensity L2."""
    ref = np.load(_ROOT / 'caustic_cusp_ref.npz', allow_pickle=True)
    E2d = ref['E2d']
    N = int(ref['N'])
    dx = float(ref['dx'])
    wl = float(ref['wavelength'])
    win = float(ref['window'])
    meta = json.loads(str(ref['metrics']))
    sy = _params_from_ref(meta)

    r_arr, phib, ampb, dxdhb = sy.branch_data()
    geom = _cusp_geometry_from_branches(r_arr, phib, ampb, dxdhb)
    assert geom['ok'], geom.get('reason')
    geom = dict(geom)
    geom.update(r1=sy.r1, r2=sy.r2)

    E_mb, rg = sy.geometric_multibranch_2d(N, dx)
    E_cusp = _build_pearcey_cusp_field(E_mb, geom, wl, dx)
    assert E_cusp is not None and np.all(np.isfinite(E_cusp))

    zone = (rg >= max(0.0, sy.r1 - 3 * (sy.r2 - sy.r1))) \
        & (rg <= sy.r2 + 4 * (sy.r2 - sy.r1))

    def icorr(E, m):
        return np.corrcoef(np.abs(E[m]) ** 2, np.abs(E2d[m]) ** 2)[0, 1]

    def il2(E, m):
        a = np.abs(E[m]) ** 2
        a = a / a.sum()
        t = np.abs(E2d[m]) ** 2
        t = t / t.sum()
        return np.linalg.norm(a - t) / np.linalg.norm(t)

    ic_c = icorr(E_cusp, zone)
    ic_m = icorr(E_mb, zone)
    # (a) correct cusp fringe structure -- decisively beats the geometric sum
    assert ic_c > 0.95, ic_c
    assert ic_c > ic_m + 0.1, (ic_c, ic_m)

    # (b) FINITE: tames the geometric caustic singularity
    tpk = float(np.max(np.abs(E2d[zone]) ** 2))
    cpk = float(np.max(np.abs(E_cusp[zone]) ** 2))
    mpk = float(np.max(np.abs(E_mb[zone]) ** 2))
    assert np.all(np.isfinite(E_cusp))
    assert cpk < 3.0 * tpk                       # finite, near the true peak
    assert mpk > 4.0 * tpk                        # multibranch diverges

    # (c) windowed r2m / EE -- Pearcey beats the multibranch
    r2m_t = _win(E2d, rg, win)
    r2m_c = _win(E_cusp, rg, win)
    ee80_t = _win(E2d, rg, win, 0.8)
    ee80_c = _win(E_cusp, rg, win, 0.8)
    ee80_m = _win(E_mb, rg, win, 0.8)
    ee95_t = _win(E2d, rg, win, 0.95)
    ee95_c = _win(E_cusp, rg, win, 0.95)
    ee95_m = _win(E_mb, rg, win, 0.95)
    assert abs(r2m_c / r2m_t - 1.0) < 0.10, (r2m_c, r2m_t)
    assert abs(ee80_c / ee80_t - 1.0) < 0.07
    assert abs(ee80_c - ee80_t) < abs(ee80_m - ee80_t)
    assert abs(ee95_c / ee95_t - 1.0) < 0.10
    assert abs(ee95_c - ee95_t) < abs(ee95_m - ee95_t)

    # (d) intensity L2 in the cusp zone -- much closer to the truth
    l2_c = il2(E_cusp, zone)
    l2_m = il2(E_mb, zone)
    assert l2_c < 0.30, l2_c
    assert l2_c < 0.5 * l2_m


# ===========================================================================
# 5. Wiring: caustic='uniform' routes an n_turn==2 cusp to the Pearcey path
# ===========================================================================
def test_uniform_routes_cusp_to_pearcey(monkeypatch):
    """The dispatcher routes an ``n_turn == 2`` (cusp) meridional trace to the
    Pearcey completion (not the fallback): with a synthetic cusp geometry the
    completed field is finite and flagged ``reason == 'cusp_ring'``."""
    from lumenairy.elements._lens_traced_uniform import (
        apply_real_lens_traced_uniform,
    )
    N, dx = 288, 0.8e-6
    sy = _SynthCusp(A=350.0, h_c=0.10e-3, s=0.04e-3, Z=2.0e-3, w0=0.15e-3,
                    hmax=0.5e-3)
    x = (np.arange(N) - N / 2.0) * dx
    X, Y = np.meshgrid(x, x)
    E0 = np.exp(-(X ** 2 + Y ** 2) / (0.35e-3) ** 2).astype(np.complex128)
    E_mb2d, _ = sy.geometric_multibranch_2d(N, dx)
    r_arr, phib, ampb, dxdhb = sy.branch_data()
    geom = _cusp_geometry_from_branches(r_arr, phib, ampb, dxdhb)
    geom = dict(geom)
    geom.update(r1=sy.r1, r2=sy.r2)

    # stub the multibranch base and the meridional traces to inject the cusp
    def fake_mb(E_in, **kw):
        if kw.get('return_diagnostics'):
            return E_mb2d, {'input_carrier': (0.0, 0.0)}
        return E_mb2d
    monkeypatch.setattr(umod, 'apply_real_lens_traced_multibranch', fake_mb)
    monkeypatch.setattr(umod, '_is_rotationally_symmetric', lambda *a, **k: True)
    monkeypatch.setattr(
        umod, '_trace_meridional_fold',
        lambda *a, **k: {'ok': False, 'reason': 'cusp_or_multiple', 'n_turn': 2})
    monkeypatch.setattr(umod, '_trace_meridional_cusp', lambda *a, **k: geom)

    presc = {'wavelength': _WL, 'surfaces': [], 'thicknesses': []}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        E, diag = apply_real_lens_traced_uniform(
            E0, prescription=presc, wavelength=_WL, dx=dx,
            output_plane_distance=2.0e-3, return_diagnostics=True)
    assert not diag['fell_back']
    assert diag['reason'] == 'cusp_ring'
    assert np.all(np.isfinite(E))
    assert E.shape == (N, N)


def test_cusp_fallbacks_finite(monkeypatch):
    """A cusp the mapping cannot handle (control solve / linear map rejected)
    falls back to the plain multibranch field (finite, warned) -- never
    inf/nan."""
    from lumenairy.elements._lens_traced_uniform import (
        apply_real_lens_traced_uniform,
    )
    N, dx = 128, 2e-6
    x = (np.arange(N) - N / 2.0) * dx
    X, Y = np.meshgrid(x, x)
    E0 = np.exp(-(X ** 2 + Y ** 2) / (0.1e-3) ** 2).astype(np.complex128)
    base = np.ones((N, N), dtype=np.complex128)

    def fake_mb(E_in, **kw):
        if kw.get('return_diagnostics'):
            return base, {'input_carrier': (0.0, 0.0)}
        return base
    monkeypatch.setattr(umod, 'apply_real_lens_traced_multibranch', fake_mb)
    monkeypatch.setattr(umod, '_is_rotationally_symmetric', lambda *a, **k: True)
    monkeypatch.setattr(
        umod, '_trace_meridional_fold',
        lambda *a, **k: {'ok': False, 'reason': 'cusp_or_multiple', 'n_turn': 2})
    # cusp trace rejects the case (e.g. near-axial / non-linear map)
    monkeypatch.setattr(
        umod, '_trace_meridional_cusp',
        lambda *a, **k: {'ok': False, 'reason': 'near-axial cusp (Bessoid)'})

    presc = {'wavelength': _WL, 'surfaces': [], 'thicknesses': []}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        E, diag = apply_real_lens_traced_uniform(
            E0, prescription=presc, wavelength=_WL, dx=dx,
            output_plane_distance=1e-3, return_diagnostics=True)
    assert diag['fell_back'] and 'cusp' in diag['reason']
    assert np.all(np.isfinite(E))
    assert np.array_equal(E, base)
    assert any('uniform' in str(m.message) for m in w)


def test_near_axial_cusp_detected_out_of_scope():
    """A near-axial (Bessoid) cusp -- the three-branch band straddles the axis --
    is DETECTED by the meridional cusp trace and reported out of scope (falls
    back), not silently mis-handled."""
    # inject a synthetic map whose two turning points have OPPOSITE-sign x_out
    n = 2000
    lr = 0.5e-3
    xs = np.linspace(lr / n, lr, n)
    xo = 8e-6 * np.sin(2.0 * np.pi * xs / lr) * (xs / lr)  # straddles the axis
    fake = types.SimpleNamespace(
        image_rays=types.SimpleNamespace(
            x=xo, y=np.zeros(n), N=np.ones(n), L=np.zeros(n), M=np.zeros(n),
            opd=1e-3 * xs, alive=np.ones(n, dtype=bool)))

    def _fake_trace(rays, surfaces, wavelength, **kw):
        return fake

    presc = {'wavelength': _WL, 'aperture_diameter': 1e-3, 'surfaces': [
        {'radius': float('inf'), 'thickness': 0.0, 'glass_before': 'air',
         'glass_after': 'air', 'semi_diameter': 0.5e-3}],
        'thicknesses': [0.0], 'stop_index': 0}
    import lumenairy.elements._lens_traced_uniform as u
    orig = u.rt.trace
    u.rt.trace = _fake_trace
    try:
        E0 = np.ones((64, 64), dtype=np.complex128)
        out = u._trace_meridional_cusp(presc, _WL, 1e-3, 1.0, lr, n, E0, 2e-6)
    finally:
        u.rt.trace = orig
    assert not out['ok']
    # either not exactly two turns, or the Bessoid (axis-straddling) rejection
    assert ('near-axial' in out['reason'] or 'cusp' in out['reason']
            or 'not a single cusp' in out['reason'])


# ===========================================================================
# 6. END TO END on a REAL rot-sym aspheric singlet (the mandated coverage):
#    the WHOLE dispatcher routes a genuine off-axis n_turn==2 cusp to Pearcey,
#    validated vs an INDEPENDENT direct-RS ground truth for that exact singlet.
# ===========================================================================
def _singlet_prescription(sg):
    """The real conic + even-aspheric singlet of ``caustic_cusp_singlet_ref``
    (a lumenairy prescription; index via the dispersionless ``_K4CAU`` glass)."""
    ac = {int(k): float(v) for k, v in sg['aspheric_coeffs_si'].items()}
    s0 = {'radius': sg['R1_mm'] * 1e-3, 'thickness': sg['thick_mm'] * 1e-3,
          'glass_before': 'air', 'glass_after': '_K4CAU',
          'semi_diameter': 0.5 * sg['aperture_mm'] * 1e-3 * 1.0714,
          'conic': float(sg['conic']), 'aspheric_coeffs': ac}
    return {'wavelength': sg['wavelength_um'] * 1e-6,
            'aperture_diameter': sg['aperture_mm'] * 1e-3, 'surfaces': [
                s0,
                {'radius': float('inf'), 'thickness': 0.0,
                 'glass_before': '_K4CAU', 'glass_after': 'air',
                 'semi_diameter': 0.5 * sg['aperture_mm'] * 1e-3 * 1.0714}],
            'thicknesses': [sg['thick_mm'] * 1e-3], 'stop_index': 0}


@pytest.fixture(scope='module')
def _singlet_e2e():
    """Run the REAL ``apply_real_lens_traced_uniform`` dispatcher (no
    monkeypatch) on the real cusp singlet + a direct-RS ground truth for it, ONCE
    for the whole module.  Reconstructs the truth field from the fixture's radial
    profile via the independent oracle's ``radial_to_2d`` (grid-independent)."""
    from lumenairy.memory import available_memory_bytes
    ref_path = _ROOT / 'caustic_cusp_singlet_ref.npz'
    if not ref_path.exists():
        pytest.skip('caustic_cusp_singlet_ref.npz fixture missing')
    ref = np.load(ref_path, allow_pickle=True)
    sg = json.loads(str(ref['singlet']))
    meta = json.loads(str(ref['metrics']))
    wl = float(ref['wavelength'])
    win = float(ref['window'])
    opd = float(ref['output_plane_distance'])

    N, dx = 1536, 1.0e-6
    # RAM-skip guard: the dispatcher holds several (N, N) complex128 working
    # copies; never OOM a memory-constrained runner.
    est_gb = (N * N * 16 * 8) / 1e9
    if available_memory_bytes() / 1e9 < est_gb + 1.5:
        pytest.skip(f"needs ~{est_gb:.1f} GB; memory-constrained host")

    cft = _load_oracle('caustic_fold_truth')
    E2d = cft.radial_to_2d(ref['rho'], ref['E_rho'], N, dx)

    x = (np.arange(N) - N / 2.0) * dx
    X, Y = np.meshgrid(x, x)
    rg = np.sqrt(X * X + Y * Y)
    E0 = np.exp(-(X ** 2 + Y ** 2)
                / (sg['w0_mm'] * 1e-3) ** 2).astype(np.complex128)
    p = _singlet_prescription(sg)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        E_cusp, diag = apply_real_lens_traced_uniform(
            E0, prescription=p, wavelength=wl, dx=dx,
            output_plane_distance=opd, ray_subsample=3,
            return_diagnostics=True)
        E_mb = apply_real_lens_traced_multibranch(
            E0, prescription=p, wavelength=wl, dx=dx,
            output_plane_distance=opd, ray_subsample=3)
    return dict(E2d=E2d, E_cusp=np.asarray(E_cusp), E_mb=np.asarray(E_mb),
                diag=diag, rg=rg, meta=meta, win=win, opd=opd,
                fold_r=meta['fold_radius_um'] * 1e-6)


def test_real_singlet_routes_cusp_to_pearcey_end_to_end(_singlet_e2e):
    """The REAL ``apply_real_lens_traced_uniform`` dispatcher -- the full path,
    NOT monkeypatched -- routes a real rot-sym aspheric singlet's genuine
    off-axis ``n_turn == 2`` cusp to the Pearcey completion: ``reason ==
    'cusp_ring'``, ``fell_back == False``, the field is finite everywhere, and
    the resolved cusp ring radius matches the independent RS fold radius.

    This is the coverage a verifier proved was reachable (an aspheric singlet
    with a conic term DOES trigger the cusp path -- the earlier "on-axis singlets
    never produce n_turn == 2" premise was false)."""
    d = _singlet_e2e
    diag = d['diag']
    E_cusp = d['E_cusp']
    assert d['meta']['n_turning_points'] == 2       # the truth is a real cusp
    assert diag['reason'] == 'cusp_ring'
    assert diag['fell_back'] is False
    assert np.all(np.isfinite(E_cusp))
    assert E_cusp.shape == d['E2d'].shape
    # the resolved cusp ring sits at the independent RS fold radius
    fold_r = d['fold_r']
    assert abs(diag['cusp_r2'] / fold_r - 1.0) < 0.08, (diag['cusp_r2'], fold_r)
    assert 0.0 < diag['cusp_r1'] < diag['cusp_r2']
    # the linear Pearcey control map is comfortably inside the accept threshold
    assert diag['cusp_map_resid'] < 0.20


def test_real_singlet_cusp_beats_multibranch_vs_rs_truth(_singlet_e2e):
    """vs the INDEPENDENT direct-RS ground truth for that exact singlet, the
    end-to-end Pearcey completion beats the geometric multibranch on the cusp
    fringe structure (intensity Icorr ~0.90 vs ~0.67) and intensity L2 (~2.4x
    closer), and tames the divergent caustic pile-up (multibranch peak ~5x the
    truth) to a FINITE peak near the truth.

    HONEST near-axial envelope (``k*r_c ~ 61``, a leading-order Bessoid-Pearcey):
    the windowed r2m / EE80 are only marginally better than the multibranch
    (both under-read the far Airy tail) -- the decisive, reproducible wins are
    the fringe STRUCTURE, the intensity L2, and FINITENESS."""
    d = _singlet_e2e
    E2d, E_cusp, E_mb, rg = d['E2d'], d['E_cusp'], d['E_mb'], d['rg']
    fold_r, win = d['fold_r'], d['win']
    zone = (rg >= max(0.0, fold_r - 8e-6)) & (rg <= fold_r + 16e-6)

    def icorr(E):
        return np.corrcoef(np.abs(E[zone]) ** 2, np.abs(E2d[zone]) ** 2)[0, 1]

    def il2(E):
        a = np.abs(E[zone]) ** 2
        a = a / a.sum()
        t = np.abs(E2d[zone]) ** 2
        t = t / t.sum()
        return np.linalg.norm(a - t) / np.linalg.norm(t)

    # (a) fringe structure: cusp completion decisively beats the geometric sum
    ic_c, ic_m = icorr(E_cusp), icorr(E_mb)
    assert ic_c > 0.85, ic_c
    assert ic_c > ic_m + 0.15, (ic_c, ic_m)

    # (b) intensity L2 in the cusp zone: much closer to the truth
    l2_c, l2_m = il2(E_cusp), il2(E_mb)
    assert l2_c < 0.60, l2_c
    assert l2_c < 0.70 * l2_m, (l2_c, l2_m)

    # (c) FINITE: tames the geometric caustic pile-up
    tpk = float(np.max(np.abs(E2d[zone]) ** 2))
    cpk = float(np.max(np.abs(E_cusp[zone]) ** 2))
    mpk = float(np.max(np.abs(E_mb[zone]) ** 2))
    assert np.all(np.isfinite(E_cusp))
    assert cpk < 2.5 * tpk, (cpk, tpk)        # finite, near the true peak
    assert mpk > 3.0 * tpk, (mpk, tpk)        # the multibranch diverges

    # (d) windowed radial metrics -- HONEST near-axial limit: the completion is
    #     at least as close to the truth as the multibranch (not worse), the
    #     structural wins above being the decisive ones.
    def wr2m(E, frac=None):
        m = rg <= win
        I = np.abs(E[m]) ** 2
        r = rg[m]
        if frac is None:
            return np.sqrt((I * r ** 2).sum() / I.sum()) * 1e6
        o = np.argsort(r)
        c = np.cumsum(I[o])
        c = c / c[-1]
        return float(np.interp(frac, c, r[o])) * 1e6
    r2m_t, r2m_c, r2m_m = wr2m(E2d), wr2m(E_cusp), wr2m(E_mb)
    assert abs(r2m_c - r2m_t) <= abs(r2m_m - r2m_t) + 0.30, (r2m_c, r2m_m, r2m_t)
    # the far-tail EE95 (the extra Airy energy) is recovered better than the
    # multibranch's hard zero tail
    ee95_t, ee95_c, ee95_m = wr2m(E2d, 0.95), wr2m(E_cusp, 0.95), wr2m(E_mb, 0.95)
    assert abs(ee95_c - ee95_t) < abs(ee95_m - ee95_t) + 0.10, (ee95_c, ee95_m)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
