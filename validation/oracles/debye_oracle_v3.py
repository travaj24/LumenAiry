"""Congruence-fixed Debye/Huygens real-lens oracle (P0 / niche N0.1).

Independent of lumenairy, Zemax, and any FFT grid.  Exact meridional raytrace
through a general rotationally-symmetric conic + even-aspheric surface list,
launched along a signed input congruence (R_in), plus BOTH:

  (a) a geometric transverse-aberration SPOT DIAGRAM at the image plane
      (Gaussian-apodized ray-density r2m / EE), and
  (b) a congruence-fixed DIFFRACTION spot at the image plane.  For a REAL
      (downstream) image this is the grid-free ring-Huygens (+R0) integral.
      For a VIRTUAL (upstream, ``zrel < 0``) image -- e.g. a negative lens
      over-diverging a converging input -- the forward (+R0) kernel cannot
      reach the plane, so the exact ray-traced exit field is built on an FFT
      grid and ANGULAR-SPECTRUM BACK-propagated by the signed (negative)
      distance instead (``huy_method`` in the output records which path ran).

WHY v3 (the fix over debye_oracle2.py).  v2's ring-Huygens integral is BROKEN
for a NON-collimated input congruence in two ways, both fixed here:

  1. MISSING ENTRANCE EIKONAL.  v2 launched each ray with the congruence slope
     but measured OPL from the flat entrance plane, omitting the input
     wavefront's own phase ``W_in(h) = h^2 / (2 R_in)``.  That collapsed the
     exit phase reference (the hammer-H6 class error) and blew the spot up by
     an order of magnitude (~700 um where the truth is ~70 um).
  2. WRONG RING MEASURE.  v2 weighted each exit ring by the ENTRANCE measure
     ``h dh``; the physical exit field on the last surface carries the
     ENERGY-conserving exit measure.  For a launch uniform in ``h`` the correct
     per-ray amplitude weight is ``A_env(h) * sqrt(h * y_exit * dy_exit/dh)``
     (equivalently ``E_exit * y_exit dy_exit`` with the energy-law amplitude
     ``E_exit = A_env sqrt(h/(y_exit J))``).  This reduces EXACTLY to v2 in the
     collimated limit (``y_exit ~ h``), which is why v2 looked correct there.

VALIDATION (self-consistency, all independent of the lens model under test):
  * collimated f/5 singlet EE80 reproduces the debye_oracle.py / dual-oracle
    value to <1%;
  * small-beam (diffraction-limited) EE80 reproduces the ABCD Gaussian image
    size to ~1-4%;
  * heavily-aberrated finite-conjugate cases agree with the geometric spot
    where geometry dominates AND with ZOS POP (adequately sampled) to <10%;
  * grid-converged in the fan / rho sampling.

Usage (CLI, mm / um units to match the LDE conventions)::

    python debye_oracle_v3.py <job.json> [z_img_offset_mm]

Job schema::

    {
      "wavelength_um": 1.31,
      "surfaces": [
        {"radius_mm": 51.68, "thickness_mm": 5.0, "index": 1.5168,
         "conic": 0.0, "aspheric_coeffs_si": {"4": 2000.0}},
        {"radius_mm": -51.68, "thickness_mm": 116.5, "index": "air"}
      ],
      "pop": {"w0_mm": 4.0},
      "R_in_mm": -35.0,          # signed input conjugate (>0 diverging); null=collim
      "aperture_mm": 20.0        # optional clear diameter (ray clip)
    }
"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy.special import j0


def _asm(E, z, wl, dx):
    """From-scratch angular-spectrum free-space propagation (evanescent zeroed).

    Exact for any SIGNED ``z``.  Unlike the forward-only ring-Huygens (+R0)
    kernel, a NEGATIVE ``z`` correctly BACK-propagates to an upstream (virtual)
    plane -- which is what a negative-lens virtual image requires.
    """
    N = E.shape[0]
    k = 2.0 * np.pi / wl
    fx = np.fft.fftfreq(N, d=dx)
    FX, FY = np.meshgrid(fx, fx)
    kz2 = k * k - (2 * np.pi * FX) ** 2 - (2 * np.pi * FY) ** 2
    kz = np.sqrt(np.maximum(kz2, 0.0))
    H = np.where(kz2 > 0, np.exp(1j * kz * z), 0.0)
    return np.fft.ifft2(np.fft.fft2(E) * H)


def _win_metrics_2d(I, dx, win):
    """Peak-centred windowed r2m and EE50/80/95 of a 2-D intensity, matching the
    in-library ``_wave_metrics`` convention the P2 tests use (radius from the
    peak pixel, integrated inside ``win``)."""
    N = I.shape[0]
    x = (np.arange(N) - N / 2) * dx
    jp, ip = np.unravel_index(np.argmax(I), I.shape)
    X, Y = np.meshgrid(x - x[ip], x - x[jp])
    r = np.sqrt(X ** 2 + Y ** 2)
    m = r <= win
    Iw, rw, Pw = I[m], r[m], float(I[m].sum())
    r2m = float(np.sqrt(np.sum(Iw * rw ** 2) / Pw))
    rb = np.linspace(0.0, win, 601)
    cum = np.array([Iw[rw <= rr].sum() for rr in rb]) / Pw
    ee = lambda f: float(np.interp(f, cum, rb))  # noqa: E731
    return r2m, ee(0.5), ee(0.8), ee(0.95), Pw


def _diffraction_backprop(hs, y_ex, opl_ex, A_env, ok, zrel, wl, N, dx, win):
    """Diffraction spot at an UPSTREAM (virtual) image plane (``zrel < 0``).

    The forward ring-Huygens (+R0) kernel radiates only diverging waves and so
    cannot reach an upstream plane; a virtual image requires BACK-propagation.
    This builds the EXACT ray-traced exit field -- energy-conserving amplitude
    ``A_env * sqrt(h / (y_exit * J))`` and exact eikonal phase ``k * opl`` on an
    FFT grid -- and angular-spectrum propagates it by the signed ``zrel`` (< 0)
    to the virtual plane.  It is a DIFFERENT diffraction method than the
    ring-Huygens j0 sum and reuses only the lumenairy-free geometric raytrace,
    so it remains an independent oracle.
    """
    k = 2.0 * np.pi / wl
    m = ok & np.isfinite(y_ex) & np.isfinite(opl_ex)
    h, ys, opl, A = hs[m], y_ex[m], opl_ex[m], A_env[m]
    J = np.abs(np.gradient(ys, h))
    amp = A * np.sqrt(np.clip(h / np.clip(ys * J, 1e-30, None), 0.0, None))
    order = np.argsort(ys)
    yss = ys[order]
    keep = np.concatenate(([True], np.diff(yss) > 1e-12))
    yss, amps, opls = yss[keep], amp[order][keep], opl[order][keep]
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    r = np.sqrt(X ** 2 + Y ** 2)
    amp2d = np.interp(r, yss, amps, left=amps[0], right=0.0)
    amp2d = np.where(r > yss[-1], 0.0, amp2d)
    opl2d = np.interp(r, yss, opls, left=opls[0], right=opls[-1])
    E_exit = amp2d * np.exp(1j * k * opl2d)
    I = np.abs(_asm(E_exit, zrel, wl, dx)) ** 2
    return _win_metrics_2d(I, dx, win)


def _sag(r, R, kc, asph):
    r = np.asarray(r, dtype=float)
    z = np.zeros_like(r)
    if np.isfinite(R) and R != 0.0:
        c = 1.0 / R
        u = (1.0 + kc) * c * c * r * r
        w = np.sqrt(np.maximum(1.0 - u, 0.0))
        z = c * r * r / (1.0 + w)
    for p, a in asph.items():
        z = z + float(a) * r ** int(p)
    return z


def _dsag(r, R, kc, asph):
    r = np.asarray(r, dtype=float)
    dz = np.zeros_like(r)
    if np.isfinite(R) and R != 0.0:
        c = 1.0 / R
        w = np.sqrt(np.maximum(1.0 - (1.0 + kc) * c * c * r * r, 1e-300))
        dwdr = -(1.0 + kc) * c * c * r / w
        dz = (2.0 * c * r * (1.0 + w) - c * r * r * dwdr) / (1.0 + w) ** 2
    for p, a in asph.items():
        dz = dz + float(a) * int(p) * r ** (int(p) - 1)
    return dz


def _prep_surfaces(surfs):
    out = []
    for s in surfs:
        idx = s.get("index", "air")
        out.append(dict(
            n=1.0 if idx == "air" else float(idx),
            R=(s["radius_mm"] * 1e-3 if s["radius_mm"] != 0 else np.inf),
            k=float(s.get("conic", 0.0) or 0.0),
            asph={int(p): float(a)
                  for p, a in (s.get("aspheric_coeffs_si") or {}).items()},
            t=s["thickness_mm"] * 1e-3))
    return out


def trace_ray(h, surfaces, R_in, aper):
    """Exact meridional trace of a ray entering at height ``h`` [m], launched
    along the signed input congruence (slope ``h / R_in``).  Returns
    ``(exit_point, exit_dir, opl)`` with ``opl`` INCLUDING the entrance eikonal
    ``W_in = h^2 / (2 R_in)``, or ``None`` (vignetted / TIR)."""
    if aper is not None and abs(h) > aper:
        return None
    p = np.array([0.0, h])
    if R_in is not None:
        g = h / R_in
        d = np.array([1.0, g]) / np.hypot(1.0, g)
        opl = h * h / (2.0 * R_in)                 # entrance eikonal (FIX 1)
    else:
        d = np.array([1.0, 0.0])
        opl = 0.0
    n_in = 1.0
    z_v = 0.0
    for s in surfaces:
        R, kc, asph, n_out = s["R"], s["k"], s["asph"], s["n"]
        flat = (not np.isfinite(R)) and (not asph)
        if flat:
            t = (z_v - p[0]) / d[0]
            p = p + t * d
            opl += n_in * t
            nrm = np.array([1.0, 0.0])
        else:
            t = (z_v - p[0]) / d[0]
            for _ in range(30):
                y = p[1] + t * d[1]
                r = abs(y)
                gg = p[0] + t * d[0] - z_v - float(_sag(r, R, kc, asph))
                dsag = float(_dsag(r, R, kc, asph)) * np.sign(y if y != 0 else 1.0)
                dgdt = d[0] - dsag * d[1]
                if abs(dgdt) < 1e-30:
                    dgdt = 1e-30
                t = t - gg / dgdt
            p = p + t * d
            opl += n_in * t
            y = p[1]
            r = abs(y)
            dsag = float(_dsag(r, R, kc, asph)) * np.sign(y if y != 0 else 1.0)
            nz, ny = 1.0, -dsag
            nn = np.hypot(nz, ny)
            nrm = np.array([nz / nn, ny / nn])
        if nrm[0] < 0:
            nrm = -nrm
        cos_i = np.dot(d, nrm)
        eta = n_in / n_out
        sin2_t = eta * eta * (1.0 - cos_i * cos_i)
        if sin2_t > 1.0:
            return None
        cos_t = np.sqrt(1.0 - sin2_t)
        d = eta * d + (cos_t - eta * cos_i) * nrm
        d = d / np.linalg.norm(d)
        n_in = n_out
        z_v += s["t"]
    return p, d, opl


def evaluate(job, dz_off_mm=0.0):
    """Run both oracles and return a metrics dict (SI internally, um out)."""
    wl = job["wavelength_um"] * 1e-6
    k = 2 * np.pi / wl
    surfaces = _prep_surfaces(job["surfaces"])
    w0 = job["pop"]["w0_mm"] * 1e-3
    R_in = job.get("R_in_mm")
    R_in = None if R_in in (None, 0) else float(R_in) * 1e-3
    aper = job.get("aperture_mm")
    aper = None if not aper else float(aper) * 1e-3 * 0.5
    win = float(job.get("window_um", 400.0)) * 1e-6
    # Job convention: the LAST surface's thickness is the (last-vertex -> image)
    # distance.  The exit vertex plane is thus the sum of the INTERIOR gaps;
    # the image plane is the full sum (+ optional through-focus offset).
    z_exit = sum(s["t"] for s in surfaces[:-1])
    z_img = sum(s["t"] for s in surfaces) + dz_off_mm * 1e-3

    r_edge = aper if aper is not None else 1.9 * w0
    n_fan = int(job.get("n_fan", 6000))
    hs = np.linspace(r_edge / n_fan, r_edge, n_fan)

    Y_img = np.full(hs.size, np.nan)          # geometric image height
    y_ex = np.full(hs.size, np.nan)           # exit-plane ray height
    opl_ex = np.full(hs.size, np.nan)         # OPL to a flat exit plane
    ok = np.zeros(hs.size, dtype=bool)
    for i, h in enumerate(hs):
        res = trace_ray(h, surfaces, R_in, aper)
        if res is None:
            continue
        p, d, opl = res
        # geometric landing at the image plane
        ti = (z_img - p[0]) / d[0]
        Y_img[i] = p[1] + ti * d[1]
        # reference every ray to the common flat exit vertex plane z_exit
        te = (z_exit - p[0]) / d[0]
        y_ex[i] = p[1] + te * d[1]
        opl_ex[i] = opl + 1.0 * te
        ok[i] = True

    A_env = np.exp(-hs ** 2 / w0 ** 2)

    # ---- (a) geometric transverse-aberration spot ------------------------
    m = ok & np.isfinite(Y_img)
    yi = np.abs(Y_img[m])
    wi = (A_env[m] ** 2) * hs[m]              # intensity x entrance ring measure
    r2m_geo = float(np.sqrt(np.sum(wi * yi ** 2) / np.sum(wi)))
    order = np.argsort(yi)
    cum_g = np.cumsum(wi[order]) / np.sum(wi)
    geo_ee = lambda f: float(np.interp(f, cum_g, yi[order]))  # noqa: E731

    # ---- (b) congruence-fixed diffraction spot ---------------------------
    # ``zrel`` = image plane relative to the flat exit vertex plane.  For a REAL
    # (downstream) image ``zrel > 0`` the grid-free ring-Huygens (+R0) forward
    # kernel applies.  For a VIRTUAL (upstream) image ``zrel < 0`` that forward
    # kernel CANNOT reach the plane (it radiates only diverging waves outward,
    # blowing the spot up to hundreds of um with most energy outside the
    # window); back-propagate the EXACT exit field with the angular spectrum.
    zrel = z_img - z_exit
    mm = ok & np.isfinite(y_ex) & np.isfinite(opl_ex)
    n_rays = int(mm.sum())
    if zrel < 0.0:
        asm_N = int(job.get("asm_N", 4096))
        asm_dx = float(job.get("asm_dx_um", 5.0)) * 1e-6
        r2m_huy, ee50_huy, ee80_huy, ee95_huy, _Pw = _diffraction_backprop(
            hs, y_ex, opl_ex, A_env, ok, zrel, wl, asm_N, asm_dx, win)
        frac_in = float("nan")
        huy_method = "asm_backprop"
    else:
        h = hs[mm]
        ys = y_ex[mm]
        opl = opl_ex[mm]
        A = A_env[mm]
        J = np.abs(np.gradient(ys, h))       # dy_exit/dh (exit ring Jacobian)
        w_amp = A * np.sqrt(np.clip(h * ys * J, 0.0, None))     # FIX 2
        rho_max = float(job.get("rho_max_um", 800.0)) * 1e-6
        n_rho = int(job.get("n_rho", 2400))
        rho = np.linspace(0.0, rho_max, n_rho)
        E = np.zeros(rho.size, dtype=complex)
        for i in range(h.size):
            R0 = np.sqrt(zrel ** 2 + ys[i] ** 2 + rho ** 2)
            E += w_amp[i] * j0(k * ys[i] * rho / R0) \
                * np.exp(1j * k * (opl[i] + R0)) / R0
        Irad = np.abs(E) ** 2
        sel = rho <= win
        P_full = float(np.trapezoid(Irad * rho, rho))
        Pw = float(np.trapezoid((Irad * rho)[sel], rho[sel]))
        r2m_huy = float(np.sqrt(
            np.trapezoid((Irad * rho ** 3)[sel], rho[sel]) / Pw))
        cum_h = np.array([np.trapezoid((Irad * rho)[:j + 1], rho[:j + 1])
                          for j in range(rho.size)])
        cum_h /= cum_h[-1]
        _huy_ee = lambda f: float(np.interp(f, cum_h, rho))  # noqa: E731
        ee50_huy, ee80_huy, ee95_huy = _huy_ee(0.5), _huy_ee(0.8), _huy_ee(0.95)
        frac_in = Pw / P_full if P_full > 0 else float("nan")
        huy_method = "ring_huygens"

    return {
        "z_img_mm": z_img * 1e3, "dz_off_mm": dz_off_mm,
        "R_in_mm": None if R_in is None else R_in * 1e3,
        "w0_mm": w0 * 1e3, "n_rays": n_rays,
        "geo_r2m_um": r2m_geo * 1e6,
        "geo_EE50_um": geo_ee(0.5) * 1e6, "geo_EE80_um": geo_ee(0.8) * 1e6,
        "geo_EE95_um": geo_ee(0.95) * 1e6,
        "huy_method": huy_method,
        "huy_r2m_um": r2m_huy * 1e6,
        "huy_EE50_um": ee50_huy * 1e6, "huy_EE80_um": ee80_huy * 1e6,
        "huy_EE95_um": ee95_huy * 1e6,
        "huy_frac_in_window": frac_in,
    }


def main(argv):
    job = json.load(open(argv[1]))
    dz = float(argv[2]) if len(argv) > 2 else 0.0
    out = evaluate(job, dz)
    print(json.dumps(out, indent=1))
    print("DEBYE_ORACLE_V3_DONE")


if __name__ == "__main__":
    main(sys.argv)
