# Lumenairy-free 3-D geometric spot-diagram oracle for DECENTERED / TILTED
# conic + even-aspheric singlets and compound elements (niche N2 / P3).
#
# Traces a 2-D Gaussian-apodized ray bundle through a surface list in which
# each surface may carry a lateral DECENTER (dx, dy) [mm] and a small-angle
# TILT (tx, ty) [mrad], to the image plane, and returns the geometric spot
# CENTROID (xc, yc), the RMS spot radius about the centroid, and the
# encircled-energy (EE) curve about the centroid.  Pure geometric optics,
# independent of lumenairy / Zemax / any FFT grid.
#
# This is the 3-D generalisation of debye_oracle3.py (which is meridional and
# rotationally symmetric): it is the robust independent oracle for the
# decenter/tilt-induced CENTROID SHIFT and COMATIC spot that a rotationally-
# symmetric oracle structurally cannot represent.  Near a strong reconvergence
# caustic the geometric ray-density spot over-estimates the true wave spot
# (documented in the P2 note), so for absolute wave-spot fidelity pair it with
# a diffraction oracle / ZOS; but the CENTROID and the coma FLARE DIRECTION /
# relative EE it reports are geometric-exact and model-independent.
#
# Convention: the small-angle surface tilt adds a linear sag ramp
# ``tx*(x-dx) + ty*(y-dy)`` (radians) plus the correspondingly tilted normal --
# the SAME field-frame small-angle convention the lumenairy displaced-pointwise
# model uses, so the two are comparing the same physical element.  Decenter
# enters as ``sag(x-dx, y-dy)``.
#
# Usage (CLI, mm / mrad / um units):
#   python geom_spot_decenter_oracle.py <job.json> [z_img_offset_mm]
# or import geom_spot(job) directly (returns a dict).
from __future__ import annotations

import json
import sys

import numpy as np


def _sag(r2, R, kc, asph):
    r2 = np.asarray(r2, dtype=float)
    z = np.zeros_like(r2)
    if np.isfinite(R) and R != 0.0:
        c = 1.0 / R
        u = (1.0 + kc) * c * c * r2
        w = np.sqrt(np.maximum(1.0 - u, 0.0))
        z = c * r2 / (1.0 + w)
    for p, a in (asph or {}).items():
        z = z + float(a) * r2 ** (int(p) // 2)
    return z


def _dsag_dr(r, R, kc, asph):
    """d(sag)/dr at radius r (>=0)."""
    r = np.asarray(r, dtype=float)
    dz = np.zeros_like(r)
    if np.isfinite(R) and R != 0.0:
        c = 1.0 / R
        w = np.sqrt(np.maximum(1.0 - (1.0 + kc) * c * c * r * r, 1e-300))
        dwdr = -(1.0 + kc) * c * c * r / w
        dz = (2.0 * c * r * (1.0 + w) - c * r * r * dwdr) / (1.0 + w) ** 2
    for p, a in (asph or {}).items():
        pw = int(p)
        dz = dz + float(a) * pw * r ** (pw - 1)
    return dz


def _surface_z_grad(s, x, y):
    """z-departure f(x,y) [m] from the vertex plane and its (df/dx, df/dy),
    honouring decenter + small-angle tilt (linear ramp + tilted normal)."""
    dcx, dcy = s["_dec"]
    tx, ty = s["_tilt"]
    xs = np.asarray(x, dtype=float) - dcx
    ys = np.asarray(y, dtype=float) - dcy
    R, kc, asph = s["_R"], s["_k"], s["_asph"]
    r2 = xs * xs + ys * ys
    r = np.sqrt(r2)
    f = _sag(r2, R, kc, asph)
    sp = _dsag_dr(r, R, kc, asph)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_r = np.where(r > 0.0, 1.0 / r, 0.0)
    dfdx = sp * xs * inv_r
    dfdy = sp * ys * inv_r
    if tx != 0.0 or ty != 0.0:
        f = f + tx * xs + ty * ys
        dfdx = dfdx + tx
        dfdy = dfdy + ty
    return f, dfdx, dfdy


def _prep(job):
    surfs = job["surfaces"]
    for s in surfs:
        idx = s.get("index", "air")
        s["_n"] = 1.0 if idx == "air" else float(idx)
        s["_R"] = s["radius_mm"] * 1e-3 if s["radius_mm"] != 0 else np.inf
        s["_k"] = float(s.get("conic", 0.0) or 0.0)
        s["_asph"] = {int(p): float(a)
                      for p, a in (s.get("aspheric_coeffs_si") or {}).items()}
        dec = s.get("decenter_mm") or (0.0, 0.0)
        s["_dec"] = (float(dec[0]) * 1e-3, float(dec[1]) * 1e-3)
        tl = s.get("tilt_mrad") or (0.0, 0.0)
        s["_tilt"] = (float(tl[0]) * 1e-3, float(tl[1]) * 1e-3)
    return surfs


def _trace_bundle(x0, y0, surfs, R_in):
    """Vectorised 3-D trace of a ray bundle launched at (x0, y0, 0) along the
    input congruence (R_in) through ``surfs``.  Returns (px, py, dx, dy, dz,
    alive) at the exit vertex plane."""
    n = x0.size
    if R_in is None:
        gx = np.zeros(n)
        gy = np.zeros(n)
    else:
        gx = x0 / R_in
        gy = y0 / R_in
    nrm = np.sqrt(1.0 + gx * gx + gy * gy)
    dxr, dyr, dzr = gx / nrm, gy / nrm, 1.0 / nrm
    px, py, pz = x0.copy(), y0.copy(), np.zeros(n)
    alive = np.ones(n, dtype=bool)
    n_in = 1.0
    z_v = 0.0
    for s in surfs:
        n_out = s["_n"]
        flat = ((not np.isfinite(s["_R"])) and (not s["_asph"])
                and s["_tilt"] == (0.0, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (z_v - pz) / dzr
        if flat:
            px, py, pz = px + t * dxr, py + t * dyr, pz + t * dzr
            nxc, nyc, nzc = np.zeros(n), np.zeros(n), np.ones(n)
        else:
            for _ in range(40):
                xq, yq = px + t * dxr, py + t * dyr
                f, dfdx, dfdy = _surface_z_grad(s, xq, yq)
                f = np.where(np.isnan(f), 0.0, f)
                dfdx = np.where(np.isnan(dfdx), 0.0, dfdx)
                dfdy = np.where(np.isnan(dfdy), 0.0, dfdy)
                g = pz + t * dzr - z_v - f
                dgdt = dzr - (dfdx * dxr + dfdy * dyr)
                dgdt = np.where(np.abs(dgdt) < 1e-30, 1e-30, dgdt)
                t = t - g / dgdt
            px, py, pz = px + t * dxr, py + t * dyr, pz + t * dzr
            f, dfdx, dfdy = _surface_z_grad(s, px, py)
            nxc, nyc, nzc = -dfdx, -dfdy, np.ones(n)
            nn = np.sqrt(nxc * nxc + nyc * nyc + nzc * nzc)
            nxc, nyc, nzc = nxc / nn, nyc / nn, nzc / nn
            alive = alive & np.isfinite(f)
        cos_i = dxr * nxc + dyr * nyc + dzr * nzc
        eta = n_in / n_out
        sin2 = eta * eta * (1.0 - cos_i * cos_i)
        alive = alive & np.isfinite(px) & (sin2 <= 1.0)
        cos_t = np.sqrt(np.maximum(1.0 - sin2, 0.0))
        ndx = eta * dxr + (cos_t - eta * cos_i) * nxc
        ndy = eta * dyr + (cos_t - eta * cos_i) * nyc
        ndz = eta * dzr + (cos_t - eta * cos_i) * nzc
        nn2 = np.sqrt(ndx * ndx + ndy * ndy + ndz * ndz)
        nn2 = np.where(nn2 == 0.0, 1.0, nn2)
        dxr, dyr, dzr = ndx / nn2, ndy / nn2, ndz / nn2
        n_in = n_out
        z_v += s["thickness_mm"] * 1e-3
    return px, py, dxr, dyr, dzr, alive


def geom_spot(job, dz_off_mm=0.0, n_side=221, ee_fracs=(0.5, 0.8, 0.95)):
    """Geometric spot metrics at the image plane.  Returns a dict with the
    centroid (um), RMS radius about the centroid (um), and EE radii (um)."""
    surfs = _prep(job)
    W0 = job["pop"]["w0_mm"] * 1e-3
    R_in = job.get("R_in_mm")
    R_in = None if R_in in (None, 0) else float(R_in) * 1e-3
    aper = job.get("aperture_mm")
    aper = None if not aper else float(aper) * 1e-3 * 0.5
    z_img = sum(s["thickness_mm"] for s in surfs) * 1e-3 + dz_off_mm * 1e-3

    hw = 1.6 * W0 if aper is None else min(1.6 * W0, aper)
    ax = np.linspace(-hw, hw, int(n_side))
    LX, LY = np.meshgrid(ax, ax)
    x0 = LX.ravel()
    y0 = LY.ravel()
    h2 = x0 * x0 + y0 * y0
    keep = np.ones(x0.size, dtype=bool)
    if aper is not None:
        keep &= h2 <= aper * aper
    x0, y0, h2 = x0[keep], y0[keep], h2[keep]
    wgt = np.exp(-2.0 * h2 / W0 ** 2)                 # Gaussian intensity weight

    px, py, dxr, dyr, dzr, alive = _trace_bundle(x0, y0, surfs, R_in)
    # The trace stops at the exit vertex plane z = sum(thicknesses[:-1]); the
    # last surface's thickness is the image distance.  Propagate the remaining
    # gap from the exit vertex to the image plane.
    z_exit = sum(s["thickness_mm"] for s in surfs[:-1]) * 1e-3
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (z_img - z_exit) / dzr
    Xi = px + t * dxr
    Yi = py + t * dyr
    m = alive & np.isfinite(Xi) & np.isfinite(Yi)
    Xi, Yi, w = Xi[m], Yi[m], wgt[m]
    W = w.sum()
    xc = float((w * Xi).sum() / W)
    yc = float((w * Yi).sum() / W)
    rr = np.sqrt((Xi - xc) ** 2 + (Yi - yc) ** 2)
    rms = float(np.sqrt((w * rr ** 2).sum() / W))
    order = np.argsort(rr)
    cum = np.cumsum(w[order])
    cum /= cum[-1]
    rs = rr[order]
    ee = {f"EE{int(f * 100)}_um": float(np.interp(f, cum, rs) * 1e6)
          for f in ee_fracs}
    out = {"z_img_mm": z_img * 1e3, "dz_off_mm": dz_off_mm,
           "centroid_x_um": xc * 1e6, "centroid_y_um": yc * 1e6,
           "rms_um": rms * 1e6, "n_rays": int(m.sum())}
    out.update(ee)
    return out


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rigid_tilt_centroid(job, surf_index=0, n_side=161):
    """INDEPENDENT rigid-body (full-rotation) tilt reference: trace a Gaussian
    bundle where surface ``surf_index`` is RIGIDLY ROTATED by its ``tilt_mrad``
    about its vertex (R = Rx(tx) @ Ry(ty)) -- the Zemax / optomechanical
    convention -- rather than the small-angle linear-ramp field-frame
    convention ``geom_spot`` (and the lumenairy displaced model) use.  Returns
    the geometric spot centroid (um).  Its MAGNITUDE cross-checks the linear-
    ramp centroid to leading order (they agree to <1% at moderate tilt, opposite
    sign by the differing 'positive tilt' definition); a large divergence would
    indicate the linear ramp is an inadequate tilt model.  Pure geometric,
    lumenairy-free."""
    surfs = _prep(job)
    W0 = job["pop"]["w0_mm"] * 1e-3
    R_in = job.get("R_in_mm")
    R_in = None if R_in in (None, 0) else float(R_in) * 1e-3
    aper = job.get("aperture_mm")
    aper = None if not aper else float(aper) * 1e-3 * 0.5
    z_img = sum(s["thickness_mm"] for s in surfs) * 1e-3
    hw = 1.6 * W0 if aper is None else min(1.6 * W0, aper)
    ax = np.linspace(-hw, hw, int(n_side))
    LX, LY = np.meshgrid(ax, ax)
    x0, y0 = LX.ravel(), LY.ravel()
    if aper is not None:
        keep = (x0 * x0 + y0 * y0) <= aper * aper
        x0, y0 = x0[keep], y0[keep]
    w = np.exp(-2.0 * (x0 * x0 + y0 * y0) / W0 ** 2)

    def _sphere_grad(x, y, R, kc, asph):
        f, dfdx, dfdy = _surface_z_grad(
            {"_dec": (0.0, 0.0), "_tilt": (0.0, 0.0),
             "_R": R, "_k": kc, "_asph": asph}, x, y)
        return f, dfdx, dfdy

    def _refract(d, nrm, eta):
        ci = float(d @ nrm)
        s2 = eta * eta * (1.0 - ci * ci)
        if s2 > 1.0:
            return None
        ct = np.sqrt(1.0 - s2)
        nd = eta * d + (ct - eta * ci) * nrm
        return nd / np.linalg.norm(nd)

    Xi = np.full(x0.size, np.nan)
    for j in range(x0.size):
        p = np.array([x0[j], y0[j], 0.0])
        d = np.array([0.0, 0.0, 1.0]) if R_in is None else (
            np.array([x0[j] / R_in, y0[j] / R_in, 1.0]))
        d = d / np.linalg.norm(d)
        n_in = 1.0
        z_v = 0.0
        ok = True
        for si, s in enumerate(surfs):
            n_out = s["_n"]
            R, kc, asph = s["_R"], s["_k"], s["_asph"]
            if si == surf_index and s["_tilt"] != (0.0, 0.0):
                Rm = _rot_x(s["_tilt"][1]) @ _rot_y(s["_tilt"][0])
                pv = np.array([0.0, 0.0, z_v])
                ps = Rm.T @ (p - pv)
                ds = Rm.T @ d
                t = (0.0 - ps[2]) / ds[2]
                for _ in range(60):
                    q = ps + t * ds
                    f, gx, gy = _sphere_grad(q[0], q[1], R, kc, asph)
                    g = q[2] - f
                    dg = ds[2] - (gx * ds[0] + gy * ds[1])
                    t = t - g / (dg if abs(dg) > 1e-30 else 1e-30)
                ps = ps + t * ds
                f, gx, gy = _sphere_grad(ps[0], ps[1], R, kc, asph)
                nrm = np.array([-gx, -gy, 1.0])
                nrm = nrm / np.linalg.norm(nrm)
                if nrm[2] < 0:
                    nrm = -nrm
                ds = _refract(ds, nrm, n_in / n_out)
                if ds is None:
                    ok = False
                    break
                p = Rm @ ps + pv
                d = Rm @ ds
            else:
                t = (z_v - p[2]) / d[2]
                for _ in range(60):
                    q = p + t * d
                    f, gx, gy = _sphere_grad(q[0] - s["_dec"][0],
                                             q[1] - s["_dec"][1], R, kc, asph)
                    g = q[2] - z_v - f
                    dg = d[2] - (gx * d[0] + gy * d[1])
                    t = t - g / (dg if abs(dg) > 1e-30 else 1e-30)
                p = p + t * d
                f, gx, gy = _sphere_grad(p[0] - s["_dec"][0],
                                         p[1] - s["_dec"][1], R, kc, asph)
                nrm = np.array([-gx, -gy, 1.0])
                nrm = nrm / np.linalg.norm(nrm)
                if nrm[2] < 0:
                    nrm = -nrm
                d = _refract(d, nrm, n_in / n_out)
                if d is None:
                    ok = False
                    break
            n_in = n_out
            z_v += s["thickness_mm"] * 1e-3
        if not ok:
            continue
        t = (z_img - p[2]) / d[2]
        Xi[j] = p[0] + t * d[0]
    m = np.isfinite(Xi)
    xc = float((w[m] * Xi[m]).sum() / w[m].sum())
    return {"centroid_x_um": xc * 1e6, "n_rays": int(m.sum())}


if __name__ == "__main__":
    job = json.load(open(sys.argv[1]))
    dz = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    print(json.dumps(geom_spot(job, dz_off_mm=dz), indent=1))
    print("GEOM_SPOT_ORACLE_DONE")
