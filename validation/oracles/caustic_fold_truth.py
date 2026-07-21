"""Multi-valued FOLD-CAUSTIC ground-truth oracle (P5 / niche N0.3).

Independent of lumenairy, Zemax, and any propagator under test.  Builds the
reference wave field at a THROUGH-FOCUS plane of a strongly-aberrated singlet
where the geometric ray map FOLDS (multi-valued -- two ray branches reach the
same transverse radius) by a DENSE, DIRECT Rayleigh-Sommerfeld / Huygens
integral of the exact exit field.  NO stationary phase, NO ray-branch
assumptions enter the PROPAGATION: the integral sums every exit-plane ring's
spherical wave to every observation point, so the fold-caustic interference
(the bright caustic ring + Airy-type fringes on the illuminated side, the
exponential tail on the dark side) emerges automatically.  This is the ground
truth for the FGA vs GBD/traced caustic comparison (niche N6a).

METHOD.

  1. Exact meridional raytrace of a dense COLLIMATED fan through a
     rotationally-symmetric conic + even-aspheric surface list -> the exit-plane
     ray height ``y_exit(h)``, exit optical path length ``opl(h)``, and the
     exit-ring Jacobian ``J = dy_exit/dh`` (all single-valued and smooth AT the
     exit plane -- the fold is DOWNSTREAM).

  2. The exact exit field on the flat exit vertex plane, with the
     energy-conserving ray-tube amplitude
         E_exit(y_exit) = A_env(h) * sqrt(h / (y_exit * J)) * exp(i k opl(h)),
     so that  2 pi INT |E_exit|^2 y_exit dy_exit  =  2 pi INT A_env^2 h dh
     = the launched power (a lossless element).

  3. The DIRECT Rayleigh-Sommerfeld ring integral to the observation plane
     ``z`` (a rotationally-symmetric field, so the azimuthal integral is the
     Debye ``J0`` form, exact to O((y rho / R0^2)^2 k) which is < 1e-2 rad here):
         E(rho, z) = (1/(i lambda)) INT E_exit(y) J0(k y rho / R0)
                     (dz'/R0) exp(i k R0) / R0 * 2 pi y dy,
     R0 = sqrt(dz'^2 + y^2 + rho^2), dz' = z - z_exit.  The (1/(i lambda)),
     obliquity ``dz'/R0`` and ring-area ``2 pi y dy`` factors make the integral
     ENERGY-CONSERVING (the metric-only ``debye_oracle_v3`` ring sum drops them;
     the |E|^2 SHAPE, hence r2m / EE, is identical, cross-checked below).

SELF-VERIFICATION (all independent of any lens model):
  * grid convergence -- doubling the fan / rho sampling changes the windowed
    r2m by < 0.5% (``convergence`` in the output);
  * energy closure -- the full-plane propagated power equals the launched power
    to < ~2% (``energy_closure``);
  * multi-valuedness -- the geometric ray map genuinely folds at this plane
    (``n_branches_at_caustic`` >= 2, ``fold_radius_um`` the caustic ring).

Usage (CLI)::

    python caustic_fold_truth.py <job.json> [--save <prefix>]

writes ``<prefix>.npz`` (reference radial + 2-D field + grid) when ``--save``.

Job schema (mm / um to match the LDE conventions)::

    {
      "wavelength_um": 1.31,
      "surfaces": [
        {"radius_mm": 2.7, "thickness_mm": 1.0, "index": 1.5168,
         "conic": 0.0, "aspheric_coeffs_si": {"4": 0.0}},
        {"radius_mm": 0.0, "thickness_mm": 4.3704, "index": "air"}
      ],
      "input": {"w0_mm": 0.55},        # collimated Gaussian waist
      "aperture_mm": 1.4,              # clear diameter (ray clip)
      "grid": {"N": 1280, "dx_um": 2.0},   # 2-D reference grid for propagators
      "window_um": 60.0,               # metric window radius
      "n_fan": 8000, "n_rho": 3000, "rho_max_um": 120.0
    }

The last surface's ``thickness_mm`` is the (last-vertex -> observation-plane)
distance, i.e. the through-focus plane where the map folds.
"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy.special import j0


# --------------------------------------------------------------------------
# geometry: conic + even-aspheric sag / slope, exact meridional raytrace
# --------------------------------------------------------------------------
def _sag(r, R, kc, asph):
    r = np.asarray(r, dtype=float)
    z = np.zeros_like(r)
    if np.isfinite(R) and R != 0.0:
        c = 1.0 / R
        w = np.sqrt(np.maximum(1.0 - (1.0 + kc) * c * c * r * r, 0.0))
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


def trace_ray(h, surfaces, aper):
    """Exact meridional trace of a COLLIMATED ray entering at height ``h`` [m].
    Returns ``(exit_point, exit_dir, opl)`` or ``None`` (vignetted / TIR)."""
    if aper is not None and abs(h) > aper:
        return None
    p = np.array([0.0, h])
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
            for _ in range(40):
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


# --------------------------------------------------------------------------
# exit field + direct Rayleigh-Sommerfeld ring integral
# --------------------------------------------------------------------------
def build_exit_field(job, n_fan=None):
    """Trace the fan; return ``(h, y_exit, opl_exit, amp_exit, z_exit, extras)``
    with the energy-conserving exit amplitude and the geometric landing height at
    the observation plane (for the fold diagnostic)."""
    surfaces = _prep_surfaces(job["surfaces"])
    w0 = job["input"]["w0_mm"] * 1e-3
    aper = job.get("aperture_mm")
    aper = None if not aper else float(aper) * 1e-3 * 0.5
    n_fan = int(n_fan or job.get("n_fan", 8000))
    z_exit = sum(s["t"] for s in surfaces[:-1])
    z_img = sum(s["t"] for s in surfaces)

    r_edge = aper if aper is not None else 1.9 * w0
    hs = np.linspace(r_edge / n_fan, r_edge, n_fan)
    y_ex = np.full(hs.size, np.nan)
    opl_ex = np.full(hs.size, np.nan)
    y_im = np.full(hs.size, np.nan)      # geometric landing at the obs plane
    for i, h in enumerate(hs):
        res = trace_ray(h, surfaces, aper)
        if res is None:
            continue
        p, d, opl = res
        te = (z_exit - p[0]) / d[0]
        y_ex[i] = p[1] + te * d[1]
        opl_ex[i] = opl + 1.0 * te
        ti = (z_img - p[0]) / d[0]
        y_im[i] = p[1] + ti * d[1]
    ok = np.isfinite(y_ex) & np.isfinite(opl_ex)
    h, ys, opl, yim = hs[ok], y_ex[ok], opl_ex[ok], y_im[ok]
    A_env = np.exp(-h ** 2 / w0 ** 2)
    J = np.abs(np.gradient(ys, h))
    amp = A_env * np.sqrt(np.clip(h / np.clip(ys * J, 1e-30, None), 0.0, None))
    P_in = 2.0 * np.pi * float(np.trapezoid(A_env ** 2 * h, h))
    return h, ys, opl, amp, z_exit, dict(
        z_img=z_img, y_im=yim, A_env=A_env, J=J, P_in=P_in, w0=w0, aper=aper)


def build_exit_field_synthetic(job, n_fan=None):
    """Exit field for a SYNTHETIC rotationally-symmetric CUSP (niche R2 / A1).

    Rather than raytracing a surface list, the exit-plane wavefront is prescribed
    directly as ``W'(h) = -A h exp(-((h - h_c)/s)^2)`` (an even, ``W'(0) = 0``
    wavefront: unit-slope background with a localised converging bump), so the
    downstream geometric landing ``y_im(h) = h + Z W'(h)`` folds TWICE at finite
    radius -- a single CUSP ring (three ray branches between the two folds).  The
    ``2``-hump meridional map is the roadmap's prescribed synthetic cusp oracle;
    the direct Rayleigh-Sommerfeld propagation below (shared with the fold truth)
    is unchanged and remains independent of any lens / propagator model.

    Returns the same ``(h, y_exit, opl_exit, amp, z_exit, extras)`` tuple as
    :func:`build_exit_field`."""
    sc = job["synthetic_cusp"]
    A = float(sc["A"])
    h_c = float(sc["h_c_mm"]) * 1e-3
    s = float(sc["s_mm"]) * 1e-3
    Z = float(sc["Z_mm"]) * 1e-3
    w0 = float(sc["w0_mm"]) * 1e-3
    hmax = float(sc["hmax_mm"]) * 1e-3
    n_fan = int(n_fan or job.get("n_fan", 14000))
    h = np.linspace(hmax / n_fan, hmax, n_fan)
    Wp = -A * h * np.exp(-((h - h_c) / s) ** 2)
    # W(h) = int_0^h W'(s) ds  (additive constant is an irrelevant global phase)
    Wint = np.concatenate([[0.0], np.cumsum(0.5 * (Wp[1:] + Wp[:-1]) * np.diff(h))])
    amp = np.exp(-h ** 2 / w0 ** 2)
    y_ex = h                      # flat exit vertex plane: exit height = launch
    y_im = h + Z * Wp             # geometric landing at the observation plane
    J = np.abs(np.gradient(y_ex, h))
    P_in = 2.0 * np.pi * float(np.trapezoid(amp ** 2 * h, h))
    return h, y_ex, Wint, amp, 0.0, dict(
        z_img=Z, y_im=y_im, A_env=amp, J=J, P_in=P_in, w0=w0, aper=hmax)


def _dispatch_exit_field(job, n_fan=None):
    """Route to the synthetic-cusp or raytraced exit-field builder."""
    if "synthetic_cusp" in job:
        return build_exit_field_synthetic(job, n_fan=n_fan)
    return build_exit_field(job, n_fan=n_fan)


def _rs_integral(h, ys, opl, amp, zrel, wl, rho):
    """Energy-correct DIRECT Rayleigh-Sommerfeld ring integral (vectorized in
    rho, trapezoidal in the exit-ring coordinate ``ys``)."""
    k = 2.0 * np.pi / wl
    order = np.argsort(ys)
    ys_s, amp_s, opl_s = ys[order], amp[order], opl[order]
    Es = amp_s * np.exp(1j * k * opl_s)              # complex exit field E_exit(ys)
    dys = np.gradient(ys_s)                           # exit-ring measure
    E = np.zeros(rho.size, dtype=complex)
    pref = 1.0 / (1j * wl)
    for i in range(ys_s.size):
        y = ys_s[i]
        R0 = np.sqrt(zrel ** 2 + y ** 2 + rho ** 2)
        E += (pref * Es[i] * j0(k * y * rho / R0) * (zrel / R0)
              * np.exp(1j * k * R0) / R0 * (2.0 * np.pi * y * dys[i]))
    return E


def radial_to_2d(rho, E_rho, N, dx):
    """Rotate the radial complex profile onto an (N, N) grid centred on the
    axis (pixel-centre convention matching the propagators' ``x0=-(N/2)dx``)."""
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    r = np.sqrt(X ** 2 + Y ** 2)
    re = np.interp(r, rho, E_rho.real, left=E_rho.real[0], right=0.0)
    im = np.interp(r, rho, E_rho.imag, left=E_rho.imag[0], right=0.0)
    return (re + 1j * im).astype(np.complex128)


def win_metrics_2d(I, dx, win, center="axis"):
    """Windowed r2m + EE50/80/95 of a 2-D intensity.

    ``center='axis'`` (default) measures the radius from the GRID CENTRE (the
    optical axis) -- the correct, robust convention for a rotationally-symmetric
    caustic, where it reproduces the grid-free radial metric to <1%.
    ``center='peak'`` measures from the argmax pixel (the in-library
    ``_wave_metrics`` convention); for a RING-dominated caustic the peak sits on
    the bright fringe OFF-AXIS, so peak-centring inflates r2m ~12% -- an artefact
    of the convention, not the field, that cancels only if BOTH sides use it.
    STEP B therefore compares axis-centred metrics."""
    N = I.shape[0]
    x = (np.arange(N) - N / 2) * dx
    if center == "peak":
        jp, ip = np.unravel_index(np.argmax(I), I.shape)
        X, Y = np.meshgrid(x - x[ip], x - x[jp])
    else:
        X, Y = np.meshgrid(x, x)
    r = np.sqrt(X ** 2 + Y ** 2)
    m = r <= win
    Iw, rw, Pw = I[m], r[m], float(I[m].sum())
    r2m = float(np.sqrt(np.sum(Iw * rw ** 2) / Pw))
    rb = np.linspace(0.0, win, 601)
    cum = np.array([Iw[rw <= rr].sum() for rr in rb]) / Pw
    ee = lambda f: float(np.interp(f, cum, rb))  # noqa: E731
    return r2m, ee(0.5), ee(0.8), ee(0.95), Pw


def _radial_metrics(rho, E, win):
    I = np.abs(E) ** 2
    sel = rho <= win
    Pw = float(np.trapezoid((I * rho)[sel], rho[sel]))
    r2m = float(np.sqrt(np.trapezoid((I * rho ** 3)[sel], rho[sel]) / Pw))
    cum = np.array([np.trapezoid((I * rho)[:j + 1], rho[:j + 1])
                    for j in range(rho.size)])
    cum /= cum[-1]
    ee = lambda f: float(np.interp(f, cum, rho))  # noqa: E731
    return r2m, ee(0.5), ee(0.8), ee(0.95), Pw


def _fold_diagnostic(extras, win):
    """Genuine multi-valuedness: count geometric ray branches reaching radii
    inside the caustic; report the fold (caustic ring) radius."""
    yim = np.abs(extras["y_im"])
    # turning points of |y_im(h)| = fold(s); interior branches = turns + 1
    d = np.diff(yim)
    turns = int(np.sum(np.diff(np.sign(d)) != 0))
    # caustic ring radius = the smallest turning-point radius (envelope)
    tp = np.where(np.diff(np.sign(d)) != 0)[0]
    fold_r = float(yim[tp[0]]) if tp.size else 0.0
    # count branches reaching a test radius just inside the fold ring
    n_br = 0
    if fold_r > 0:
        rtest = 0.5 * fold_r
        # number of h intervals crossing rtest
        s = np.sign(yim - rtest)
        n_br = int(np.sum(np.diff(s) != 0))
    return dict(n_branches_at_caustic=max(n_br, turns + 1),
                fold_radius_um=fold_r * 1e6, n_turning_points=turns)


def evaluate(job, save_prefix=None):
    """Build the reference field, self-verify, return a metrics dict (+ save)."""
    wl = job["wavelength_um"] * 1e-6
    win = float(job.get("window_um", 60.0)) * 1e-6
    rho_max = float(job.get("rho_max_um", 120.0)) * 1e-6
    n_rho = int(job.get("n_rho", 3000))
    grid = job.get("grid", {})
    N = int(grid.get("N", 1024))
    dx = float(grid.get("dx_um", 2.0)) * 1e-6

    h, ys, opl, amp, z_exit, extras = _dispatch_exit_field(job)
    zrel = extras["z_img"] - z_exit
    rho = np.linspace(0.0, rho_max, n_rho)
    E_rho = _rs_integral(h, ys, opl, amp, zrel, wl, rho)

    r2m_r, e50_r, e80_r, e95_r, Pw_r = _radial_metrics(rho, E_rho, win)
    # energy closure: full-plane propagated power vs launched power
    P_out = 2.0 * np.pi * float(np.trapezoid(np.abs(E_rho) ** 2 * rho, rho))
    energy_closure = P_out / extras["P_in"]

    # 2-D reference on the propagator grid
    E2d = radial_to_2d(rho, E_rho, N, dx)
    I2d = np.abs(E2d) ** 2
    r2m_2d, e50_2d, e80_2d, e95_2d, _ = win_metrics_2d(I2d, dx, win)

    # ---- self-verification: fan/rho grid convergence (halve dx = double N) ----
    h2, ys2, opl2, amp2, ze2, _ = _dispatch_exit_field(job, n_fan=2 * int(
        job.get("n_fan", 8000)))
    rho2 = np.linspace(0.0, rho_max, 2 * n_rho)
    E_rho2 = _rs_integral(h2, ys2, opl2, amp2, zrel, wl, rho2)
    r2m_r2, _, e80_r2, _, _ = _radial_metrics(rho2, E_rho2, win)
    conv_r2m = abs(r2m_r2 - r2m_r) / r2m_r
    conv_ee80 = abs(e80_r2 - e80_r) / e80_r

    fold = _fold_diagnostic(extras, win)

    out = {
        "z_img_mm": extras["z_img"] * 1e3,
        "z_exit_mm": z_exit * 1e3,
        "output_plane_distance_mm": zrel * 1e3,
        "wavelength_um": job["wavelength_um"],
        "window_um": win * 1e6, "N": N, "dx_um": dx * 1e6,
        "n_rays": int(h.size),
        # radial (grid-free) reference metrics
        "huy_r2m_um": r2m_r * 1e6, "huy_EE50_um": e50_r * 1e6,
        "huy_EE80_um": e80_r * 1e6, "huy_EE95_um": e95_r * 1e6,
        # 2-D reference on the propagator grid (what STEP B compares against)
        "grid_r2m_um": r2m_2d * 1e6, "grid_EE50_um": e50_2d * 1e6,
        "grid_EE80_um": e80_2d * 1e6, "grid_EE95_um": e95_2d * 1e6,
        # self-verification
        "energy_closure": energy_closure,
        "convergence_r2m_frac": conv_r2m,
        "convergence_ee80_frac": conv_ee80,
        "grid_vs_radial_r2m_frac": abs(r2m_2d - r2m_r) / r2m_r,
        **fold,
    }
    if "synthetic_cusp" in job:
        out["synthetic_cusp"] = job["synthetic_cusp"]
    if save_prefix:
        np.savez_compressed(
            save_prefix + ".npz",
            rho=rho, E_rho=E_rho, E2d=E2d, N=N, dx=dx,
            wavelength=wl, window=win,
            output_plane_distance=zrel, z_img=extras["z_img"], z_exit=z_exit,
            metrics=json.dumps(out))
    return out


def main(argv):
    job = json.load(open(argv[1]))
    save = None
    if "--save" in argv:
        save = argv[argv.index("--save") + 1]
    out = evaluate(job, save_prefix=save)
    print(json.dumps(out, indent=1))
    print("CAUSTIC_FOLD_TRUTH_DONE")


if __name__ == "__main__":
    main(sys.argv)
