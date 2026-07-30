# Does ``preserve_input_phase='remap'`` transport the input residual
# correctly, and is the error decentre-driven?
# (DIAG_LAST_GROUP_DECENTRE_2026_07_30, the decomposition that names the
# broken construction.)
#
# wfe_probe_decentre.py shows that with an input whose carrier-de-chirped
# RESIDUAL is identically 1, design 121's last-group element reproduces the
# EXACT ray trace to 0.0002-0.0007 waves rms at every decentre out to 1.5 w.
# So the ray map, the Chebyshev fits, the Newton inversion and the cubic OPL
# upsample are all sound.  The only remaining construction is the residual leg.
#
# THREE PHASES OF THE SAME INPUT, COMPARED ON THE SAME EXIT NODES
# ---------------------------------------------------------------
#   ELEMENT   arg E_out                                          (what ships)
#   MODEL-A   k0*[W(xe*) + OPL(xe*)] + arg resid(xe*)            (what 'remap'
#             INTENDS: geometric transport of the residual along the CARRIER
#             rays, with (xe*, ye*) the EXACT inverse of the carrier ray map
#             and ``resid`` evaluated by band-limited interpolation)
#   ORACLE-B  k0*[W(xe') + a(xe') + OPL(xe')]                    (the exact-ray
#             oracle the whole study uses: rays launched along the TOTAL input
#             eikonal grad(W + a), a = arg(resid)/k0, and inverted exactly)
#
# ELEMENT - MODEL-A  isolates the IMPLEMENTATION (pullback position, bilinear
#                    phasor sampling, 'full' vs 'lattice' resampling).
# MODEL-A - ORACLE-B isolates the MODEL (transporting the residual along the
#                    carrier congruence instead of its own).
#
# usage:
#   MODE=real DEC=0,0.5,1.079,1.5 python wfe_probe_remap.py
#   MODE=synth ALPHA=0,0.5,1,2 DEC=0,1.079 python wfe_probe_remap.py
import os
import sys
import time
import warnings

import numpy as np
from scipy.ndimage import map_coordinates

import wfe_probe_common as P
import _d121_common as C
from lumenairy.elements import apply_real_lens_traced
from lumenairy.elements._lens_traced import TiltedCarrier
from lumenairy.raytrace import trace
from lumenairy.raytrace.trace import _make_bundle
from lumenairy.glass import get_glass_index

LAM = P.LAM
K0 = P.K0
R_IN = -0.021139185452405257
L_IN, M_IN = 0.04907347265758019, 0.024536736328790096
X0_D, Y0_D = -0.003016240777531001, -0.0015081203887655004
DX = 33.2112e-6
N = 1024
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     '_wfe_probe_g5_input.npz')


# ---------------------------------------------------------------------------
# the REAL residual the chain hands design 121's last group
# ---------------------------------------------------------------------------
def real_input(m=-4, n=-2):
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        return d['E_in'], d['E_out'], tuple(d['car'])
    calls, _res, _msg = P.run_chain_capture(m=m, n=n)
    rec = calls[-1]
    car = rec['kw']['carrier']
    np.savez_compressed(CACHE, E_in=rec['E_in'], E_out=rec['E_out'],
                        car=np.array([car.R, car.L, car.M, car.x0, car.y0]))
    return rec['E_in'], rec['E_out'], (car.R, car.L, car.M, car.x0, car.y0)


class ResidualField(object):
    """A band-limited, smoothly-unwrapped model of the carrier-de-chirped
    input residual, evaluable at arbitrary entrance points.

    Built by Fourier-upsampling a crop of the residual by ``up`` (exact for a
    band-limited envelope -- the chain's envelopes are exactly that, proven by
    the scope doc's UP 1/2/4 step-halving test), then interpolating the PHASOR
    itself with an order-3 spline on that fine lattice, where the per-pixel
    phase step is ``up`` times smaller than the source's.

    NO GLOBAL UNWRAP ANYWHERE.  A first revision of this class unwrapped the
    fine phase with the usual ``unwrap(unwrap(.,axis=1),axis=0)`` idiom and
    FAILED its own self-test by 1.96 rad: the beam skirt, where the residual
    amplitude is ~0 and its phase is numerical noise, makes the row-wise
    unwrap arbitrary and the cubic spline's GLOBAL prefilter then carries that
    into the core.  Only wrapped quantities are interpolated here, and only
    ``exp(i k0 a)`` is ever consumed downstream, so a 2*pi ambiguity in ``a``
    is invisible by construction.
    """

    def __init__(self, resid, dx, x0, y0, half_m, up=8):
        n = resid.shape[0]
        c = n // 2
        h = int(np.ceil(half_m / dx))
        ic = int(round(x0 / dx)) + c
        jc = int(round(y0 / dx)) + c
        self.i0 = max(ic - h, 0)
        self.j0 = max(jc - h, 0)
        i1 = min(self.i0 + 2 * h, n)
        j1 = min(self.j0 + 2 * h, n)
        sub = np.asarray(resid)[self.j0:j1, self.i0:i1]
        nc = sub.shape[0]
        m = nc * up
        # Zero-pad in the UNSHIFTED spectrum by quadrant copy.  The
        # fftshift-then-centre-paste idiom (used by hybrid_localize_121's
        # _fourier_up) mis-places the DC bin by one fine sample whenever the
        # crop size is ODD, which shifts the whole interpolant by dx/up -- it
        # cost this probe a 2.6 rad self-test failure before it was found.
        # This form is exact at the source nodes for any nc.
        F = np.fft.fft2(sub)
        G = np.zeros((m, m), dtype=complex)
        k = (nc + 1) // 2
        t_ = nc - k
        G[:k, :k] = F[:k, :k]
        G[:k, m - t_:] = F[:k, k:]
        G[m - t_:, :k] = F[k:, :k]
        G[m - t_:, m - t_:] = F[k:, k:]
        fine = np.fft.ifft2(G) * (up * up)
        self.up = up
        self.dxf = dx / up
        # fine-grid origin in metres (fine sample k sits at coarse index
        # i0 + k/up, i.e. metre position ((i0 + k/up) - n/2) * dx)
        self.x_org = (self.i0 - n / 2) * dx
        self.y_org = (self.j0 - n / 2) * dx
        amp = np.abs(fine)
        self.amp = amp
        # unit phasor (real / imag), interpolated directly -- no unwrap
        with np.errstate(invalid='ignore', divide='ignore'):
            uz = np.where(amp > 0, fine / np.maximum(amp, 1e-300), 1.0 + 0.0j)
        self.zr = np.real(uz).astype(np.float64)
        self.zi = np.imag(uz).astype(np.float64)
        ang = np.angle(uz)
        # WRAPPED central differences -> the local phase gradient, aliasing
        # free (the fine-grid step is up times the coarse one) and needing no
        # unwrap.  Smooth real fields, so an order-3 spline is safe on them.
        dxr = np.zeros_like(ang)
        dyr = np.zeros_like(ang)
        dxr[:, 1:-1] = np.angle(np.exp(1j * (ang[:, 2:] - ang[:, :-2]))) \
            / (2 * self.dxf)
        dyr[1:-1, :] = np.angle(np.exp(1j * (ang[2:, :] - ang[:-2, :]))) \
            / (2 * self.dxf)
        dxr[:, 0], dxr[:, -1] = dxr[:, 1], dxr[:, -2]
        dyr[0, :], dyr[-1, :] = dyr[1, :], dyr[-2, :]
        self.gx, self.gy = dxr, dyr
        self.shape = fine.shape
        # PRE-FILTER once.  map_coordinates(order=3) re-runs the global spline
        # prefilter on every call; this array is up^2 times the crop (26 M
        # samples at up=8), and the Newton inversion calls it ~150 times.
        from scipy.ndimage import spline_filter as _sf
        self._fzr = _sf(self.zr, order=3, mode='nearest')
        self._fzi = _sf(self.zi, order=3, mode='nearest')
        self._fgx = _sf(self.gx, order=3, mode='nearest')
        self._fgy = _sf(self.gy, order=3, mode='nearest')
        self._famp = _sf(self.amp, order=3, mode='nearest')

    def _coords(self, x, y):
        col = (np.asarray(x) - self.x_org) / self.dxf
        row = (np.asarray(y) - self.y_org) / self.dxf
        return np.vstack([np.asarray(row).ravel(), np.asarray(col).ravel()])

    def ev(self, x, y):
        """(a, La, Ma) -- residual eikonal in METRES (modulo lambda, which is
        all ``exp(i k0 a)`` can see) and its transverse direction cosines."""
        c = self._coords(x, y)
        sh = np.asarray(x).shape
        a = self._angle(c).reshape(sh)
        gx = map_coordinates(self._fgx, c, order=3, mode='nearest',
                             prefilter=False).reshape(sh)
        gy = map_coordinates(self._fgy, c, order=3, mode='nearest',
                             prefilter=False).reshape(sh)
        return a / K0, gx / K0, gy / K0

    def amp_at(self, x, y):
        sh = np.asarray(x).shape
        return map_coordinates(self._famp, self._coords(x, y), order=3,
                               mode='nearest', prefilter=False).reshape(sh)

    def _angle(self, c):
        zr = map_coordinates(self._fzr, c, order=3, mode='nearest',
                             prefilter=False)
        zi = map_coordinates(self._fzi, c, order=3, mode='nearest',
                             prefilter=False)
        return np.angle(zr + 1j * zi)

    def phase(self, x, y):
        sh = np.asarray(x).shape
        return self._angle(self._coords(x, y)).reshape(sh)


def trace_total(xe, ye, carrier, surfs, rf=None):
    """Exact trace launched along grad(W + a).  Returns
    (x_out, y_out, psi_total, L, M, alive) with psi = OPL + W + a."""
    xe = np.asarray(xe, dtype=np.float64).ravel()
    ye = np.asarray(ye, dtype=np.float64).ravel()
    W, L, M = P.carrier_parts(carrier, xe, ye)
    extra = 0.0
    if rf is not None:
        a, La, Ma = rf.ev(xe, ye)
        L = L + La
        M = M + Ma
        extra = a
    rays = _make_bundle(x=xe.copy(), y=ye.copy(),
                        L=np.asarray(L, dtype=np.float64).ravel().copy(),
                        M=np.asarray(M, dtype=np.float64).ravel().copy(),
                        wavelength=LAM)
    fin = trace(rays, surfs, LAM, output_filter='last').image_rays
    n_exit = get_glass_index(surfs[-1].glass_after, LAM)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(fin.alive & (np.abs(fin.N) > 1e-30), -fin.z / fin.N, 0.0)
    opd = np.asarray(fin.opd) + n_exit * t + np.asarray(W) + extra
    xo = np.asarray(fin.x) + np.asarray(fin.L) * t
    yo = np.asarray(fin.y) + np.asarray(fin.M) * t
    return (xo, yo, opd, np.asarray(fin.L), np.asarray(fin.M),
            np.asarray(fin.alive, bool))


def invert_total(Xt, Yt, carrier, surfs, rf, guess, n_iter=16, tol=1e-11,
                 h=5e-7):
    """Newton inversion of the TOTAL-eikonal map (exact residual, exact
    Jacobian by finite differences of the same trace)."""
    sh = Xt.shape
    xt, yt = Xt.ravel(), Yt.ravel()
    xe = np.asarray(guess[0]).ravel().copy()
    ye = np.asarray(guess[1]).ravel().copy()
    for _it in range(n_iter):
        xo, yo, psi, Lo, Mo, alive = trace_total(xe, ye, carrier, surfs, rf)
        rx, ry = xo - xt, yo - yt
        if np.nanmax(np.hypot(rx, ry)) < tol:
            break
        x1, y1, _p, _l, _m, _a = trace_total(xe + h, ye, carrier, surfs, rf)
        x2, y2, _p, _l, _m, _a = trace_total(xe, ye + h, carrier, surfs, rf)
        jxx, jyx = (x1 - xo) / h, (y1 - yo) / h
        jxy, jyy = (x2 - xo) / h, (y2 - yo) / h
        det = jxx * jyy - jxy * jyx
        g = np.isfinite(det) & (np.abs(det) > 1e-14)
        dxe = np.where(g, (jyy * rx - jxy * ry) / np.where(g, det, 1.0), 0.0)
        dye = np.where(g, (-jyx * rx + jxx * ry) / np.where(g, det, 1.0), 0.0)
        s = np.hypot(dxe, dye)
        sc = np.where(s > 2e-3, 2e-3 / np.maximum(s, 1e-30), 1.0)
        xe, ye = xe - dxe * sc, ye - dye * sc
    xo, yo, psi, Lo, Mo, alive = trace_total(xe, ye, carrier, surfs, rf)
    rx, ry = xo - xt, yo - yt
    psi_node = psi - (Lo * rx + Mo * ry)
    return {'xe': xe.reshape(sh), 'ye': ye.reshape(sh),
            'phi': (K0 * psi_node).reshape(sh),
            'resid': np.hypot(rx, ry).reshape(sh),
            'ok': (alive & np.isfinite(psi_node)).reshape(sh)}


def fshift(a, sx, sy, dx):
    if sx == 0.0 and sy == 0.0:
        return np.array(a, copy=True)
    n = a.shape[0]
    f = np.fft.fftfreq(n, d=dx)
    r = np.exp(-2j * np.pi * (f[None, :] * sx + f[:, None] * sy))
    return np.fft.ifft2(np.fft.fft2(a) * r)


def main():
    mode = os.environ.get('MODE', 'real')
    decs = [float(v) for v in os.environ.get('DEC',
                                             '0,0.5,1.079,1.5').split(',')]
    alphas = [float(v) for v in os.environ.get('ALPHA', '1').split(',')]
    half = int(os.environ.get('HALF', '72'))
    up = int(os.environ.get('UP', '8'))
    w = float(os.environ.get('W', '3.1255')) * 1e-3
    kwx = os.environ.get('KW', '')
    kw = {}
    for it in kwx.split(';'):
        if it.strip():
            k, v = it.split('=', 1)
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    v = {'True': True, 'False': False}.get(v, v)
            kw[k.strip()] = v
    _pre, post, _g, _p = C.geometry()
    presc = post[-1]['prescription']
    surfs = P.element_surfaces(presc)
    az = np.arctan2(Y0_D, X0_D)

    E_in_real, E_out_real, car_real = real_input()
    car0 = TiltedCarrier(*car_real)
    ax = (np.arange(N) - N / 2) * DX
    Xg, Yg = np.meshgrid(ax, ax)
    W0, _l, _m = P.carrier_parts(car0, Xg, Yg)
    resid0 = np.asarray(E_in_real) * np.exp(-1j * K0 * W0)
    amp0 = np.abs(resid0)
    print(f"REAL residual at the last group's entrance: peak |resid| "
          f"{amp0.max():.4e}")
    # its own per-pixel phase step over the bright core (aliasing check)
    br = amp0 > 0.05 * amp0.max()
    d1 = np.angle(np.exp(1j * (np.angle(resid0[:, 1:])
                              - np.angle(resid0[:, :-1]))))
    kb = br[:, 1:] & br[:, :-1]
    print(f"  max |d phase| per pixel over |resid|>0.05pk: "
          f"{np.abs(d1[kb]).max():.4f} rad   (pi = 3.1416); "
          f"amplitude-weighted rms "
          f"{np.sqrt((d1[kb]**2 * amp0[:, 1:][kb]**2).sum() / (amp0[:, 1:][kb]**2).sum()):.4f}")
    print(f"  MODE={mode}  UP={up}  element kwargs override: {kw}")
    print()

    hdr = (f"{'alpha':>6} {'dec/w':>7} {'ELEM-MODELA':>12} {'MODELA-ORCL':>12} "
           f"{'ELEM-ORACLE':>12} {'Strehl(E-O)':>12} {'nnA':>7} {'nnB':>7} "
           f"{'npix':>7}")
    print(hdr)
    print('-' * len(hdr))
    for alpha in alphas:
        for d in decs:
            t0 = time.time()
            x0 = d * w * np.cos(az)
            y0 = d * w * np.sin(az)
            car = TiltedCarrier(R_IN, L_IN, M_IN, x0, y0)
            Wn, _l2, _m2 = P.carrier_parts(car, Xg, Yg)
            if mode == 'real':
                rs = fshift(resid0, x0 - X0_D, y0 - Y0_D, DX)
                if alpha != 1.0:
                    rs = np.abs(rs) * np.exp(1j * alpha * np.angle(rs))
            else:
                u, v = Xg - x0, Yg - y0
                r2 = (u * u + v * v) / (w * w)
                rs = np.exp(-r2) * np.exp(1j * alpha * (r2 ** 2))
            E_in = (rs * np.exp(1j * K0 * Wn)).astype(np.complex128)
            opts = dict(amplitude_model='ray_density',
                        preserve_input_phase='remap',
                        fit_radius_beam_factor=2.0, remap_sampling='full',
                        ray_subsample=4, n_workers=8)
            opts.update(kw)
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always')
                E_out = np.asarray(apply_real_lens_traced(
                    E_in, prescription=presc, wavelength=LAM, dx=DX,
                    carrier=car, **opts))
            # patch around the exit chief ray
            xo, yo, _p3, _l3, _m3, _a3 = P.trace_forward([x0], [y0], car,
                                                         surfs)
            ic = int(round(float(xo[0]) / DX + N / 2))
            jc = int(round(float(yo[0]) / DX + N / 2))
            sx = slice(max(ic - half, 0), min(ic + half + 1, N))
            sy = slice(max(jc - half, 0), min(jc + half + 1, N))
            Xp, Yp, Ep = Xg[sy, sx], Yg[sy, sx], E_out[sy, sx]
            rf = ResidualField(rs, DX, x0, y0, 3.4 * w, up=up)
            # seed
            t = np.linspace(-2.6 * w, 2.6 * w, 21)
            U, V = np.meshgrid(t, t)
            gxo, gyo, _q, _q2, _q3, alv = P.trace_forward(
                (U + x0).ravel(), (V + y0).ravel(), car, surfs)
            gg = alv & np.isfinite(gxo)
            A3 = np.stack([np.ones(int(gg.sum())), gxo[gg], gyo[gg]], axis=1)
            cxx = np.linalg.lstsq(A3, (U + x0).ravel()[gg], rcond=None)[0]
            cyy = np.linalg.lstsq(A3, (V + y0).ravel()[gg], rcond=None)[0]
            B3 = np.stack([np.ones(Xp.size), Xp.ravel(), Yp.ravel()], axis=1)
            guess = ((B3 @ cxx).reshape(Xp.shape), (B3 @ cyy).reshape(Xp.shape))
            exA = P.exact_phase_on_nodes(Xp, Yp, car, surfs, guess=guess,
                                         n_iter=16)
            phiA = exA['phi'] + rf.phase(exA['xe'], exA['ye'])
            exB = invert_total(Xp, Yp, car, surfs, rf, guess)
            phiB = exB['phi']
            amp = np.abs(Ep)
            pk = float(amp.max())
            keepA = exA['ok'] & (amp > 0.02 * pk) & (exA['resid'] < 1e-9)
            keepB = keepA & exB['ok'] & (exB['resid'] < 1e-9)
            rA = P.local_wfe(Ep, phiA, keepB, Xp, Yp)
            rB = P.local_wfe(Ep, phiB, keepB, Xp, Yp)
            # MODEL-A vs ORACLE-B, weighted by the element's amplitude
            zAB = np.exp(1j * (phiA - phiB))
            rAB = P.local_wfe(np.abs(Ep) * zAB, np.zeros_like(phiA), keepB,
                              Xp, Yp)
            print(f"{alpha:>6.2f} {d:>7.3f} {rA['rms_waves']:>12.5f} "
                  f"{rAB['rms_waves']:>12.5f} {rB['rms_waves']:>12.5f} "
                  f"{rB['strehl_marechal']:>12.5f} {rA['nn_step_rad']:>7.3f} "
                  f"{rB['nn_step_rad']:>7.3f} {rB['n_pix']:>7d}  "
                  f"[{time.time()-t0:.0f}s]")


if __name__ == '__main__':
    sys.exit(main())
