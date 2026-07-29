# Repro of the DECENTRED-CARRIER exit-slope measurement (roadmap
# ROADMAP_DESIGN121_FULL_CONFIGURATION_2026_07_27, D6 retraction + D7
# correction, 2026-07-29).
#
# LOCAL-ONLY: needs the 121 .zmx and the design-study runner (for the Schott
# Sellmeier coefficients), like its sibling runners in this directory.
#
# ---------------------------------------------------------------------------
# READ THIS FIRST -- what this script showed, and what it actually measured.
# ---------------------------------------------------------------------------
# The D6 revision of this script reported that ``apply_real_lens_traced`` under
# a DECENTRED carrier invents 3.7 -> 408 urad of exit slope from 0.00 to 0.97
# beam radii of decentre (0.029 -> 3.147 um of blur against a 3.5 um
# diffraction FWHM), and the roadmap, the ``on_decentred_fit`` guard and the
# fan runner's banner all quoted that curve.
#
# THE CURVE WAS THIS SCRIPT'S OWN MEASUREMENT ARTEFACT.  It extracted the
# field's local exit direction cosine with a GLOBAL FFT derivative of the
# de-chirped field.  The exit wavefront converges at NA 0.405 on a grid whose
# Nyquist direction cosine is 0.0197 -- 20x beyond -- so the de-chirp has to
# carry the whole burden, and it only succeeds where the residual stays inside
# Nyquist.  It does over the beam CORE, but the FFT derivative is a NON-LOCAL
# operator: the beam skirt, where the residual is not inside Nyquist, aliases
# and leaks into the core.  ``FFT`` below is that estimator, kept verbatim.
#
# The kill: run the SAME estimator on a SYNTHETIC field whose exit-slope error
# is 0.36 urad BY CONSTRUCTION -- amplitude taken from the library's own
# output, phase set to ``k0 * OPL`` from an order-12 fit + tight Newton, whose
# error against the exact ray trace was measured independently.  It reports
# ~400 urad.  Both columns are printed side by side below; when they agree,
# the number is real, and at 0.97 w they do not.
#
# ``LOCAL`` is the aliasing-free replacement: the field's phase is referenced
# against that same order-12 OPL map and differenced between NEIGHBOURING
# PIXELS with a 2*pi wrap.  Per-pixel steps are ~0.06 rad, four orders inside
# pi, so nothing can fold, and no global transform touches the data.  What it
# measures, on design 121's own last group:
#
#   decentre     pre-D7          D7 (order-10 off-centre fit)
#   0.00 w       1.28 urad       1.28 urad   (byte-identical -- on-axis path)
#   0.32 w       5.32            0.78
#   0.64 w       5.88            0.91
#   0.80 w       6.56            0.91
#   0.97 w       7.16            0.90
#
# i.e. the element was never carrying 408 urad, and after D7 the decentred
# figure sits BELOW the on-axis one.  Design 121's per-order chain metric is
# still a LOWER BOUND on the design -- the residual is elsewhere (see the D7
# correction in the roadmap for the four candidates it is measurably NOT).
#
# METHOD (unchanged).  Feed design 121's own LAST GROUP an EXACT spherical wave
# converging to a transversely displaced point (a real congruence; with zero
# residual the ray launch is right by construction whatever the carrier model)
# and compare the exit wavefront the returned FIELD carries against the traced
# map.  ``PRE_D7=1`` restores the pre-D7 off-centre fit order (6) so the
# fail-before / pass-after is one environment variable apart.
#
# The self-contained CI twins are
# tests/unit/test_niche_d7_decentred_fit.py (the element, synthetic stand-in)
# and tests/unit/test_niche_d6_exact_tilted_leg.py::
# test_decentred_carrier_decentre_penalty_envelope (end to end).
#
# usage:  PRE_D7=0|1 python decentred_fit_defect.py [n_ray]
import ast, dataclasses, os, re, sys, warnings

import numpy as np

warnings.filterwarnings('ignore', message='.*prescription aperture.*')
warnings.filterwarnings('ignore', message='.*residual transverse.*')
warnings.filterwarnings('ignore', message='.*under-sampled.*')
warnings.filterwarnings('ignore', message='.*decentre_fit_frac.*')
warnings.filterwarnings('ignore', message='.*DGRATING.*')
sys.path.insert(0, r"D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Lumenairy")
sys.path.insert(0, r"D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Reverse_Symmetric_ASM")

import lumenairy as la
from lumenairy import GLASS_REGISTRY, GLASS_VALIDITY, SELLMEIER_COEFFICIENTS
from lumenairy.elements import apply_real_lens_traced
from lumenairy.elements import _lens_traced as _LT
from lumenairy.elements._lens_traced import TiltedCarrier
from lumenairy.propagators.carrier import _group_abcd, _paraxial_group_r_out
from lumenairy.raytrace import Surface, trace
from lumenairy.raytrace.trace import _make_bundle, surfaces_from_prescription

if os.environ.get('PRE_D7') == '1':
    _LT._DECENTRED_FIT_POLY_ORDER = 6

RUNNER = (r"D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics"
          r"\Reverse_Symmetric_ASM\run_poc_119_120_v518.py")
ZMX = (r"D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics"
       r"\Reverse_Symmetric_ASM\tx4designstudy121\20260707 dll Tx02-MSOP16.zmx")

# --- Schott glasses the 121 design uses, parsed out of the runner ----------
src = open(RUNNER, 'r', encoding='utf-8', errors='ignore').read()
m = re.search(r'_NEW_GLASSES\s*=\s*\{', src)
i0 = m.end() - 1
d = 0
for i in range(i0, len(src)):
    if src[i] == '{':
        d += 1
    elif src[i] == '}':
        d -= 1
        if d == 0:
            break
for g, (c, r, nd) in ast.literal_eval(src[i0:i + 1]).items():
    if g not in SELLMEIER_COEFFICIENTS:
        SELLMEIER_COEFFICIENTS[g] = c
        GLASS_REGISTRY[g] = '__sellmeier__'
        GLASS_VALIDITY[g] = r

import tx_design_study_sim as sim

# The LAST post-DOE group, and the state the chain hands it for the extreme
# order (-4,-2) at N=1024 -- measured, see the roadmap's D6 note.
LAM = 1.31e-6
K0 = 2.0 * np.pi / LAM
R_IN = -21.1392e-3          # entrance carrier radius (m)
W_BEAM = 3.1255e-3          # entrance beam amplitude radius (m)
DX = 33.211e-6              # co-moving pitch (m)
N = 1024
PRESC = [e['prescription'] for e in
         sim.group_elements_into_lenses(la.load_zemax_zmx(ZMX))
         if e['type'] == 'lens'][-1]
NRAY = int(sys.argv[1]) if len(sys.argv) > 1 else 161

# the three fits the element built on the most recent call, in build order
# (Sx, Sy, So) -- captured so the reference OPL map can be rebuilt at a
# different polynomial order without duplicating the launch geometry.
_FITS = []
_orig_cheb_init = _LT._Cheb2DEvaluator.__init__


def _capture(self, xs_in, ys_in, values, order=6, xp=None, weights=None):
    _orig_cheb_init(self, xs_in, ys_in, values, order=order, xp=xp,
                    weights=weights)
    _FITS.append(self)


_LT._Cheb2DEvaluator.__init__ = _capture


def exit_surfaces():
    sf = surfaces_from_prescription(PRESC)
    sf[-1] = dataclasses.replace(sf[-1], thickness=0.0)
    return list(sf) + [Surface(radius=np.inf, conic=0.0, semi_diameter=np.inf,
                               glass_before='air', glass_after='air',
                               is_mirror=False, thickness=0.0, label='exit')]


def _element(x0, tilt_L, **kw):
    """The returned field, and the grid it lives on."""
    _FITS.clear()
    sgn = -1.0
    ax = (np.arange(N) - N / 2) * DX
    X, Y = np.meshgrid(ax, ax, indexing='xy')
    u, v = X - x0, Y
    rho = np.sqrt(u * u + v * v + R_IN * R_IN)
    W = sgn * (rho - abs(R_IN)) + tilt_L * u
    E_in = (np.exp(-(u * u + v * v) / (W_BEAM * W_BEAM))
            * np.exp(1j * K0 * W)).astype(np.complex128)
    opts = dict(ray_subsample=4, n_workers=8, amplitude_model='ray_density',
                preserve_input_phase='remap', remap_sampling='full',
                fit_radius_beam_factor=2.0, on_undersample='silent')
    opts.update(kw)
    E_out = np.asarray(apply_real_lens_traced(
        E_in, prescription=PRESC, wavelength=LAM, dx=DX,
        carrier=TiltedCarrier(R_IN, tilt_L, 0.0, x0, 0.0), **opts))
    return E_out, X, Y, list(_FITS)


def _reference_opl(fits, X, Y, x0):
    """OPL on the wave grid from an order-12 fit + tight Newton.

    Its own error against the EXACT ray trace was measured at 0.36 urad of
    exit slope (against 6.91 urad for the shipped order-6 fit), so it is a
    valid reference for BOTH the pre-D7 and the D7 field.
    """
    Sx, Sy, So = fits[0], fits[1], fits[2]
    xe = np.full(X.size, x0, dtype=np.float64)
    ye = np.zeros(X.size, dtype=np.float64)
    xw, yw = X.ravel(), Y.ravel()
    for _ in range(40):
        fx, jxx, jxy = Sx.ev_value_and_grad(xe, ye)
        fy, jyx, jyy = Sy.ev_value_and_grad(xe, ye)
        rx, ry = fx - xw, fy - yw
        det = jxx * jyy - jxy * jyx
        xe = xe - (jyy * rx - jxy * ry) / det
        ye = ye - (-jyx * rx + jxx * ry) / det
    return So.ev(xe, ye).reshape(X.shape), xe.reshape(X.shape), \
        ye.reshape(X.shape)


def fft_estimator(E_out, x0, tilt_L):
    """The D6 estimator, verbatim: de-chirp against the paraxial exit sphere,
    take a GLOBAL FFT derivative, compare with the traced exit cosines.

    Kept so the artefact stays reproducible -- ``measure`` runs it on a
    synthetic field whose answer is known to 0.36 urad."""
    A, B, C, D = _group_abcd(PRESC, LAM)
    R_out = _paraxial_group_r_out(PRESC, R_IN, LAM)
    sgn = -1.0
    ax = (np.arange(N) - N / 2) * DX
    X, Y = np.meshgrid(ax, ax, indexing='xy')
    hh = 2.0 * W_BEAM
    uu = np.linspace(-hh, hh, NRAY)
    U, V = np.meshgrid(uu, uu, indexing='xy')
    rr = np.sqrt(U * U + V * V + R_IN * R_IN)
    im = trace(_make_bundle((U + x0).ravel(), V.ravel(),
                            (sgn * U / rr + tilt_L).ravel(),
                            (sgn * V / rr).ravel(), LAM),
               exit_surfaces(), LAM, output_filter='last').image_rays
    sh = (NRAY, NRAY)
    xo, yo = np.asarray(im.x).reshape(sh), np.asarray(im.y).reshape(sh)
    Lo = np.asarray(im.L).reshape(sh)
    ali = np.asarray(im.alive).reshape(sh)
    A_in = np.exp(-(U * U + V * V) / (W_BEAM * W_BEAM))
    xco, Lco = A * x0 + B * tilt_L, C * x0 + D * tilt_L
    uo, vo = X - xco, Y
    so = 1.0 if R_out > 0 else -1.0
    rho2 = np.sqrt(uo * uo + vo * vo + R_out * R_out)
    env = E_out * np.exp(-1j * K0 * (so * (rho2 - abs(R_out)) + Lco * uo))
    f = np.fft.fftfreq(N, d=DX)
    dEx = np.fft.ifft(1j * 2 * np.pi * f[None, :] * np.fft.fft(env, axis=1),
                      axis=1)
    Ez = np.where(np.abs(env) < 1e-12 * np.abs(env).max(), 1.0, env)
    Lg = so * uo / rho2 + Lco + np.imag(dEx / Ez) / K0
    gi, gj = xo / DX + N / 2, yo / DX + N / 2
    ok = ali & (gi > 1) & (gj > 1) & (gi < N - 2) & (gj < N - 2)
    i0_ = np.clip(np.floor(gi).astype(int), 0, N - 2)
    j0_ = np.clip(np.floor(gj).astype(int), 0, N - 2)
    fx, fy = gi - i0_, gj - j0_

    def bil(a):
        return ((1 - fx) * (1 - fy) * a[j0_, i0_]
                + fx * (1 - fy) * a[j0_, i0_ + 1]
                + (1 - fx) * fy * a[j0_ + 1, i0_]
                + fx * fy * a[j0_ + 1, i0_ + 1])

    dL = bil(Lg) - Lo
    aout = bil(np.abs(E_out))
    msk = ok & (A_in > 0.05) & (aout > 0.05 * np.abs(E_out).max())
    wt = A_in[msk] ** 2
    Bm = np.stack([np.ones(int(msk.sum())), U[msk], V[msk]], axis=1)
    Bw = Bm * wt[:, None]
    res = dL[msk] - Bm @ np.linalg.solve(Bw.T @ Bm, Bw.T @ dL[msk])
    core = np.hypot(U[msk], V[msk]) <= W_BEAM
    return float(np.sqrt((res[core] ** 2).mean())), abs(R_out)


def local_estimator(E_out, X, Y, opl_ref, xe, ye, x0, half=96):
    """Aliasing-free: wrapped NEAREST-NEIGHBOUR phase differences of
    ``arg(E_out) - k0 * opl_ref``.  Purely local, so beyond-Nyquist skirt
    content cannot reach the core."""
    A, _B, _C, _D = _group_abcd(PRESC, LAM)
    jc = int(round(A * x0 / DX + N / 2))
    ic = N // 2
    sy, sx = slice(ic - half, ic + half), slice(jc - half, jc + half)
    Ew, Xw, Yw = E_out[sy, sx], X[sy, sx], Y[sy, sx]
    ow, xw, yw = opl_ref[sy, sx], xe[sy, sx], ye[sy, sx]
    psi = np.angle(Ew * np.exp(-1j * K0 * ow))
    amp = np.abs(Ew)
    dpx = np.angle(np.exp(1j * (psi[:, 1:] - psi[:, :-1])))
    dLx = dpx / (K0 * DX)
    thr = 0.02 * amp.max()
    keep = (amp[:, 1:] > thr) & (amp[:, :-1] > thr) & np.isfinite(ow[:, 1:])
    wt = (0.5 * (amp[:, 1:] + amp[:, :-1])) ** 2
    re = np.hypot(0.5 * (xw[:, 1:] + xw[:, :-1]) - x0,
                  0.5 * (yw[:, 1:] + yw[:, :-1]))
    Xm = 0.5 * (Xw[:, 1:] + Xw[:, :-1]) - A * x0
    Ym = 0.5 * (Yw[:, 1:] + Yw[:, :-1])
    Bm = np.stack([np.ones(int(keep.sum())), Xm[keep], Ym[keep]], axis=1)
    q, ww = dLx[keep], wt[keep]
    Bw = Bm * ww[:, None]
    res = q - Bm @ np.linalg.solve(Bw.T @ Bm, Bw.T @ q)
    core = re[keep] <= W_BEAM
    return float(np.sqrt((res[core] ** 2).mean()))


def measure(x0, tilt_L=0.0):
    """(local rms, FFT rms on the library field, FFT rms on a field that is
    right to 0.36 urad by construction, |R_out|), all in rad."""
    E_out, X, Y, _fits = _element(x0, tilt_L)
    _E12, _X, _Y, fits12 = _element(x0, tilt_L, newton_poly_order=12,
                                    ray_subsample=1)
    opl_ref, xe, ye = _reference_opl(fits12, X, Y, x0)
    E_syn = np.where(np.abs(E_out) > 0,
                     np.abs(E_out) * np.exp(1j * K0 * opl_ref), 0.0)
    fft_lib, r_out = fft_estimator(E_out, x0, tilt_L)
    fft_syn, _ = fft_estimator(E_syn, x0, tilt_L)
    return (local_estimator(E_out, X, Y, opl_ref, xe, ye, x0),
            fft_lib, fft_syn, r_out)


if __name__ == '__main__':
    print(f"{PRESC.get('name', 'last group')}: R_in {R_IN * 1e3:.4f} mm, beam "
          f"radius {W_BEAM * 1e3:.4f} mm, N={N} dx={DX * 1e6:.3f} um, "
          f"{NRAY}^2 oracle rays")
    print(f"off-centre ray-fit order = {_LT._DECENTRED_FIT_POLY_ORDER} "
          f"({'PRE-D7' if os.environ.get('PRE_D7') == '1' else 'D7 default'})")
    print(f"grid Nyquist direction cosine {LAM / (2 * DX):.5f} against an exit "
          f"NA of ~0.405 -- 20x beyond, which is what breaks the FFT column\n")
    print(f"{'chief |c| (mm)':>15} {'|c|/w':>7} {'LOCAL (urad)':>14} "
          f"{'blur (um)':>10} {'FFT lib':>9} {'FFT synth':>10}  verdict")
    for x0 in (0.0, 1.0e-3, 1.5e-3, 2.0e-3, 2.5e-3, 3.017e-3):
        loc, fl, fs, r_out = measure(-x0)
        bad = abs(fs) > 3.0 * max(abs(loc), 1e-12)
        print(f"{x0 * 1e3:15.3f} {x0 / W_BEAM:7.3f} {loc * 1e6:14.2f} "
              f"{loc * r_out * 1e6:10.3f} {fl * 1e6:9.2f} {fs * 1e6:10.2f}"
              f"  {'FFT ALIASED' if bad else 'both agree'}", flush=True)
    print("\n'FFT synth' is the SAME estimator on a field whose exit-slope")
    print("error is 0.36 urad BY CONSTRUCTION.  Where it reads hundreds of")
    print("urad the estimator is measuring itself, not the element.")
    print("\nCONTROL -- tilt vs decentre, on the LOCAL estimator:")
    loc, _fl, _fs, _r = measure(0.0, tilt_L=0.04872)
    print(f"   48.7 mrad tilt, beam ON AXIS   {loc * 1e6:8.2f} urad")
    loc, _fl, _fs, _r = measure(-3.017e-3, tilt_L=0.0)
    print(f"   3.017 mm decentre, NO tilt     {loc * 1e6:8.2f} urad")
