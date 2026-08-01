# POP CROSSCHECK, step 3: score a saved POP irradiance grid with the SAME
# metric code the campaign's own scorers use.
#
# THE EE CONVENTION (checked, not assumed -- the brief guessed "diameter").
# Both in-house scorers take r in MICRONS as a RADIUS:
#
#   focus_scan_121.metrics():
#       ee = {r: I[rr <= r*1e-6].sum() * dxo*dxo / P_in  for r in (3, 6, 12)}
#       rr measured from the PEAK PIXEL, normalised by the CHAIN INPUT POWER.
#
#   hybrid_localize_121.rs_spot()  <- the source of the per-order numbers
#       for rad in (3e-6, 6e-6, 12e-6):
#           out[...] = I[r <= rad].sum() / tot
#       r measured from the intensity CENTROID, normalised by ``tot`` = the sum
#       over the WHOLE OUTPUT WINDOW, which at the campaign's settings
#       (NOUT=61, DXO=0.4 um) is a 24.0 x 24.0 um SQUARE.
#
# So "EE3" = energy inside a 3 um RADIUS (6 um diameter), and the per-order
# reference numbers (0,0) 89.21 / (-4,-2) 88.49 etc. are normalised to a 24 um
# square window, NOT to the launched power.  Comparing a POP EE3 normalised to
# the full POP array against those numbers would be an apples-to-oranges error
# of order the halo fraction.  This script therefore reports BOTH:
#
#   ee_win  -- centroid-centred, divided by the 24 um square window sum
#              (directly comparable with hybrid_localize / the oracle)
#   ee_arr  -- centroid-centred, divided by the whole POP array sum.  That sum
#              is a REAL transmission (see pop_grid_diag.py mode=clip: a forced
#              aperture drops it to 0.327), and it measures 1.00000000 here for
#              every order, so ee_arr is directly comparable with focus_scan's
#              P_in normalisation.
#
# usage:  python pop_metrics_121.py <file.npz> [...]
import glob
import json
import sys

import numpy as np

WIN_HALF = 12.0e-6      # hybrid_localize window half-width, m (NOUT=61,DXO=0.4)
DXO_REF = 0.4e-6        # hybrid_localize output pixel


def _fwhm_interp(I, X, Y, cx, cy, dx):
    """Ring-averaged FWHM with linear interpolation (hybrid_localize form)."""
    r = np.hypot(X - cx, Y - cy)
    nb = int(min(I.shape[0] // 2, 12e-6 / dx))
    if nb < 4:
        return np.nan
    ring = np.clip((r / dx).astype(int), 0, nb)
    s = np.bincount(ring.ravel(), weights=I.ravel(), minlength=nb + 1)
    cn = np.bincount(ring.ravel(), minlength=nb + 1)
    prof = s[:nb] / np.maximum(cn[:nb], 1)
    prof = prof / prof[0]
    idx = np.where(prof < 0.5)[0]
    if not len(idx) or idx[0] == 0:
        return np.nan
    i1 = idx[0]
    f = (prof[i1 - 1] - 0.5) / (prof[i1 - 1] - prof[i1])
    return 2 * ((i1 - 1 + 0.5) * dx + f * dx)


def score(path, verbose=True, rebin=1, profile=False):
    d = np.load(path, allow_pickle=True)
    I = d['I']
    x = d['x'] * 1e-3          # mm -> m
    y = d['y'] * 1e-3
    meta = json.loads(str(d['meta']))
    if rebin > 1:
        # SEPARATE "the field changed" from "the estimator moved".  A finer POP
        # run rebinned onto the coarse run's pixel must reproduce the coarse
        # run's numbers if and only if the underlying field is the same.
        n = (I.shape[0] // rebin) * rebin
        I = I[:n, :n].reshape(n // rebin, rebin, n // rebin,
                              rebin).mean(axis=(1, 3))
        x = x[:n].reshape(-1, rebin).mean(axis=1)
        y = y[:n].reshape(-1, rebin).mean(axis=1)
    dx = float(np.diff(x).mean())
    dy = float(np.diff(y).mean())
    X, Y = np.meshgrid(x, y)

    # when the saved grid is a crop, the FULL-array power is carried in meta so
    # the array-normalised EE stays exact.  meta['P_full'] is in Zemax lens
    # units (W with x,y in mm); this grid is in metres, hence the 1e-6.
    Ptot = (float(meta['P_full']) * 1e-6 if meta.get('P_full')
            else float(I.sum()) * dx * dy)
    # POP centres its array on the chief ray, so (0,0) of this grid is the
    # chief-ray intercept -- the same point hybrid_localize centres on.
    m_win = (np.abs(X) <= WIN_HALF + 0.5 * dx) & (np.abs(Y) <= WIN_HALF
                                                  + 0.5 * dy)
    Iw = np.where(m_win, I, 0.0)
    tot_win = float(Iw.sum())
    cx = float((Iw * X).sum() / tot_win)
    cy = float((Iw * Y).sum() / tot_win)
    r = np.hypot(X - cx, Y - cy)

    ee_win = {k: float(Iw[r <= k * 1e-6].sum()) / tot_win for k in (3, 6, 12)}
    ee_arr = {k: float(I[r <= k * 1e-6].sum()) * dx * dy / Ptot
              for k in (3, 6, 12)}

    iy, ix = np.unravel_index(np.argmax(I), I.shape)
    px, py = float(x[ix]), float(y[iy])
    fw_c = _fwhm_interp(I, X, Y, cx, cy, dx)
    fw_p = _fwhm_interp(I, X, Y, px, py, dx)

    # power outside the 24 um window but inside the array: the halo POP sees
    halo = 1.0 - tot_win * dx * dy / Ptot

    if verbose:
        print(f"--- {path.split(chr(92))[-1]}")
        print(f"    cfg={meta['config']} N={meta['nx']} W={meta['width']} mm "
              f"dz={meta.get('defocus_um', 0):+g} um  swap={meta.get('swap')}")
        print(f"    grid {I.shape[0]}x{I.shape[1]} dx={dx * 1e6:.5f} um  "
              f"array {np.ptp(x) * 1e6:.1f} um wide")
        print(f"    centroid ({cx * 1e6:+.4f},{cy * 1e6:+.4f}) um   peak px "
              f"({px * 1e6:+.4f},{py * 1e6:+.4f}) um")
        print(f"    FWHM(centroid) {fw_c * 1e6:.4f} um   "
              f"FWHM(peak) {fw_p * 1e6:.4f} um")
        print(f"    ee_win  EE3 {ee_win[3] * 100:6.2f}  EE6 "
              f"{ee_win[6] * 100:6.2f}  EE12 {ee_win[12] * 100:6.2f}   "
              f"(/24um-square window)")
        print(f"    ee_arr  EE3 {ee_arr[3] * 100:6.2f}  EE6 "
              f"{ee_arr[6] * 100:6.2f}  EE12 {ee_arr[12] * 100:6.2f}   "
              f"(/whole array; halo outside the window "
              f"{halo * 100:.3f} %)")
        for line in meta.get('header', []):
            if 'Beam Width' in line or 'Point spacing' in line:
                print("      [zemax]", line)
    if profile:
        nb = int(min(I.shape[0] // 2, 20e-6 / dx))
        ring = np.clip((r / dx).astype(int), 0, nb)
        s = np.bincount(ring.ravel(), weights=I.ravel(), minlength=nb + 1)
        cn = np.bincount(ring.ravel(), minlength=nb + 1)
        prof = s[:nb] / np.maximum(cn[:nb], 1)
        prof = prof / prof[0]
        cum = np.cumsum(s[:nb]) / tot_win
        print("      r[um]   I/I0        EE(r)/win")
        for rq in (0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12):
            j = int(rq * 1e-6 / dx)
            if j < nb:
                print(f"      {rq:5.1f}   {prof[j]:.6f}    {cum[j]:.5f}")
    return dict(path=path, meta=meta, dx=dx, width_um=float(np.ptp(x) * 1e6),
                fwhm_c=fw_c, fwhm_p=fw_p, ee_win=ee_win, ee_arr=ee_arr,
                halo=halo, cx=cx, cy=cy, Ptot=Ptot)


def main(argv):
    rebin = 1
    profile = False
    if '--profile' in argv:
        argv = [a for a in argv if a != '--profile']
        profile = True
    if '--rebin' in argv:
        i = argv.index('--rebin')
        rebin = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    files = []
    for a in argv:
        files.extend(sorted(glob.glob(a)))
    rows = [score(f, rebin=rebin, profile=profile) for f in files]
    if len(rows) > 1:
        print("\n=== CONVERGENCE TABLE ===")
        print("  cfg    N   W[mm]  dz[um]  dx[um]  arr[um]  FWHM   EE3w   "
              "EE6w  EE12w   EE3a   EE6a  EE12a   halo%")
        for r in rows:
            m = r['meta']
            print(f"  {m['config']:3d} {m['nx']:5d} {m['width']:6.3f} "
                  f"{m.get('defocus_um', 0):+7.1f} {r['dx'] * 1e6:7.4f} "
                  f"{r['width_um']:8.1f} {r['fwhm_c'] * 1e6:6.3f} "
                  f"{r['ee_win'][3] * 100:6.2f} {r['ee_win'][6] * 100:6.2f} "
                  f"{r['ee_win'][12] * 100:6.2f} {r['ee_arr'][3] * 100:6.2f} "
                  f"{r['ee_arr'][6] * 100:6.2f} {r['ee_arr'][12] * 100:6.2f} "
                  f"{r['halo'] * 100:7.3f}")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
