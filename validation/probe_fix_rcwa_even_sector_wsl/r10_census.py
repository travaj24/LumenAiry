"""R10 -- the 24-fixture census, both arms, one interpreter.

PRE arm = the exact ``_sqrt_decay`` body as it stood at 48c8747 (copied here,
so the fail-before arm needs no second checkout).  POST arm = whatever the
imported library ships.  Both arms run on the SAME interpreter and the SAME
BLAS, so every difference between them is the change and nothing else.

Per fixture it records

* the branch-cut populations -- ``|Re(r)| / max(max|r|, 1)`` split into the
  on-cut (``lam^2`` numerically real NEGATIVE) and off-cut sets -- which is
  where ``_CUT_BAND_REL``'s two-sided separation is READ OFF rather than
  assumed;
* the lossless-closure defect of each path (an independent oracle: a provably
  lossless cell must give ``sum R + T == 2`` exactly under the Laurent rule at
  ANY truncation);
* the full-vs-even disagreement the failing test asserts on;
* ``cond(a + b)`` at each interface;
* whether POST and PRE are bit-identical.
"""
from __future__ import annotations

import itertools
import json
import warnings

import _lib as L
import numpy as np

import lumenairy.elements.rcwa._core as C
import lumenairy.elements.rcwa.twod as TW

_CX = C._C


def sqrt_decay_pre(x):
    """``_sqrt_decay`` verbatim as of 48c8747 (the PRE arm)."""
    xp = C.array_namespace(x)
    x = xp.asarray(x).astype(_CX)
    r = xp.sqrt(x)
    on_cut = r.real == 0
    return xp.where(on_cut & (r.imag < 0), -r, r)


_POST = C._sqrt_decay
IFACE = []
BAND = []


def install():
    orig_if = C._interface_smatrix
    orig_eig_for = C._eig_for

    def iface(Wa, Va, Wb, Vb):
        a = np.linalg.solve(np.asarray(Wb), np.asarray(Wa))
        b = np.linalg.solve(np.asarray(Vb), np.asarray(Va))
        s = np.linalg.svd(np.asarray(a + b), compute_uv=False)
        IFACE.append(float(s[0] / s[-1]) if s[-1] > 0 else float("inf"))
        return orig_if(Wa, Va, Wb, Vb)

    def eig_for(xp):
        base = orig_eig_for(xp)

        def wrapped(A):
            w, v = base(A)
            wn = np.asarray(w).astype(complex)
            r = np.sqrt(wn)
            scale = max(float(np.max(np.abs(r))), 1.0)
            rel = np.abs(r.real) / scale
            oncut = wn.real < 0
            BAND.append((rel[oncut].tolist(), rel[~oncut].tolist()))
            return w, v
        return wrapped

    C._interface_smatrix = iface
    TW._interface_smatrix = iface
    C._eig_for = eig_for


def cell(S=48, twist=0.7, no=1.5, ne=1.7, hw=0.25, eps_bg=2.25, loss=0.0):
    tc = L.even_sector_cell(S=S, twist=twist, no=no, ne=ne, halfwidth=hw,
                            eps_bg=eps_bg)
    if loss:
        for i in range(3):
            tc[:, :, i, i] = tc[:, :, i, i] + 1j * loss
    return tc


def fixtures():
    out = []
    for twist, nord, n_sub in itertools.product((0.0, 0.7, 1.2), (3, 4, 5),
                                                (1.5, 1.0, 1.8)):
        out.append(dict(name="t%.1f_n%d_s%.1f" % (twist, nord, n_sub),
                        tc=dict(twist=twist), nord=nord, n_sub=n_sub,
                        theta=0.0, phi=0.0))
    out.append(dict(name="bg1.44_coinc", tc=dict(eps_bg=1.44, no=1.2, ne=1.4),
                    nord=5, n_sub=1.2, theta=0.0, phi=0.0))
    out.append(dict(name="lossy_coinc", tc=dict(loss=0.05), nord=4,
                    n_sub=1.5, theta=0.0, phi=0.0))
    out.append(dict(name="lossy_strong", tc=dict(loss=0.5), nord=4,
                    n_sub=1.5, theta=0.0, phi=0.0))
    out.append(dict(name="hw0.4_coinc", tc=dict(hw=0.4), nord=5, n_sub=1.5,
                    theta=0.0, phi=0.0))
    out.append(dict(name="oblique_th0.3", tc=dict(), nord=4, n_sub=1.5,
                    theta=0.3, phi=0.0))
    out.append(dict(name="oblique_th0.2_ph0.7", tc=dict(), nord=4, n_sub=1.5,
                    theta=0.2, phi=0.7))
    return out


def solve(f, sym):
    IFACE.clear()
    BAND.clear()
    tc = cell(**f["tc"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = TW.rcwa_jones_2d(L.P_DEFAULT, L.P_DEFAULT, tc, f["n_sub"], 1.0,
                               0.2e-6, L.WL_DEFAULT, n_orders_x=f["nord"],
                               n_orders_y=f["nord"], theta=f["theta"],
                               phi=f["phi"], symmetry=sym)
    return res, list(IFACE), list(BAND)


def main():
    a = L.arm()
    install()
    rows = []
    on_all, off_all = [], []
    print("### build=%s tree=%s" % (a["build"], a["tree"]), flush=True)
    for f in fixtures():
        rec = dict(name=f["name"], nord=f["nord"], n_sub=f["n_sub"],
                   theta=f["theta"], phi=f["phi"], tc=f["tc"])
        keep = {}
        for arm_name, fn in (("pre", sqrt_decay_pre), ("post", _POST)):
            C._sqrt_decay = fn
            full, if_f, band = solve(f, False)
            even, if_e, _ = solve(f, True)
            if arm_name == "post":
                for on, off in band:
                    on_all += on
                    off_all += off
            R, T = np.asarray(full[1]), np.asarray(full[2])
            Rev, Tev = np.asarray(even[1]), np.asarray(even[2])
            keep[arm_name] = (R.copy(), Rev.copy())
            rec[arm_name] = dict(
                dR=float(np.max(np.abs(R - Rev))),
                dT=float(np.max(np.abs(T - Tev))),
                dJ=float(np.max(np.abs(np.asarray(full[3])
                                       - np.asarray(even[3])))),
                defect_full=float(np.sum(R) + np.sum(T) - 2.0),
                defect_even=float(np.sum(Rev) + np.sum(Tev) - 2.0),
                cond_full=if_f, cond_even=if_e)
        C._sqrt_decay = _POST
        pre, post = rec["pre"], rec["post"]
        rec["bit_identical_full"] = bool(np.array_equal(keep["pre"][0],
                                                        keep["post"][0]))
        rec["bit_identical_even"] = bool(np.array_equal(keep["pre"][1],
                                                        keep["post"][1]))
        rec["max_abs_change_full"] = float(np.max(np.abs(keep["pre"][0]
                                                         - keep["post"][0])))
        rec["max_abs_change_even"] = float(np.max(np.abs(keep["pre"][1]
                                                         - keep["post"][1])))
        rows.append(rec)
        print("%-22s pre  dR %.2e dT %.2e def_f %+.2e def_e %+.2e cond %.2e"
              % (f["name"], pre["dR"], pre["dT"], pre["defect_full"],
                 pre["defect_even"], max(pre["cond_full"] or [0])), flush=True)
        print("%-22s post dR %.2e dT %.2e def_f %+.2e def_e %+.2e cond %.2e"
              "  bitid %s/%s  |post-pre| %.2e"
              % ("", post["dR"], post["dT"], post["defect_full"],
                 post["defect_even"], max(post["cond_full"] or [0]),
                 rec["bit_identical_full"], rec["bit_identical_even"],
                 max(rec["max_abs_change_full"], rec["max_abs_change_even"])),
              flush=True)
    on_all = np.asarray(on_all)
    off_all = np.asarray(off_all)
    band = dict(n_oncut=int(on_all.size), n_offcut=int(off_all.size),
                oncut_max=float(on_all.max()) if on_all.size else None,
                offcut_min=float(off_all.min()) if off_all.size else None,
                bar=C._CUT_BAND_REL)
    print("\nBAND CENSUS  on-cut n=%d max=%.3e | off-cut n=%d min=%.3e | "
          "bar=%.1e" % (band["n_oncut"], band["oncut_max"] or -1,
                        band["n_offcut"], band["offcut_min"] or -1,
                        band["bar"]))
    print("R10JSON " + json.dumps(dict(band=band, n=len(rows))))
    L.dump("r10_census", dict(band=band, rows=rows))


if __name__ == "__main__":
    main()
