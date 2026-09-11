"""TASK E / claim (c) -- the LAYER MODE COUNT under an infinitesimal Im(eps).

``layer_modes`` scans ``sigma_min(M(qz2))`` on a real-axis window and refines
each dip.  Every strip S-matrix in that cascade is built from ``_ky_forward``'s
forward set, so a mode that came back on its BACKWARD partner changes the
cascade and therefore the dip structure -- which is how a branch decision turns
into a MODE COUNT.

Adding ``i 1e-30`` to one region is a physical no-op (thirteen decades below
any material) but routes the strip eigensolve from ``eigh`` to ``scipy eig``.
The count must not move.  Measured on the build's own window, on a NARROWED
window and on MY OWN three-strip layer.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _vh  # noqa: E402
import ve_fix  # noqa: E402

PI = np.pi

# (name, strips-builder, Lx, Nx, Ly, k0, window, kx0, ky0, n_scan)
WINDOWS = {
    # the build's OWN scoping fixture and window
    "theirs_full": ("two", 96, 1.0, 1.0, 20 * PI, (26055.8, 35530.6),
                    0.0, 0.37, 300),
    # the build's NARROWED gate window
    "theirs_gate": ("two", 96, 1.0, 1.0, 20 * PI, (26055.8, 28500.0),
                    0.0, 0.37, 60),
    # MY OWN windows on the same layer, different band and density
    "mine_lowband": ("two", 96, 1.0, 1.0, 20 * PI, (12000.0, 16000.0),
                     0.0, 0.37, 80),
    "mine_kx": ("two", 96, 1.0, 1.0, 20 * PI, (26055.8, 28500.0),
                0.21, 0.11, 60),
    # MY OWN three-strip layer (different contrast, a region of eps = 1)
    "mine_three": ("three", 96, 1.0, 1.0, 20 * PI, (14000.0, 20000.0),
                   0.0, 0.37, 80),
    # a COARSER grid, below the resolution the build says the onset needs
    "mine_nx48": ("two", 48, 1.0, 1.0, 20 * PI, (26055.8, 28500.0),
                  0.0, 0.37, 60),
}


def strips(kind, Nx, Ly, imag):
    if kind == "two":
        return ve_fix.strips_two(Nx, Ly, imag)
    if kind == "three":
        return ve_fix.strips_three(Nx, Ly, imag)
    raise SystemExit(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True, choices=["pre", "post"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--windows", default="theirs_gate")
    ap.add_argument("--imags", default="0,1e-30")
    a = ap.parse_args()

    import lumenairy
    print("lumenairy.__file__ =", lumenairy.__file__)
    _vh.require_tree(a.build)
    from lumenairy.elements.eme import eme_2d

    out = dict(task="E", claim="3c", build=a.build, tag=a.tag, arm=_vh.arm(),
               lumenairy_file=lumenairy.__file__)
    rows = []
    for wname in a.windows.split(","):
        kind, Nx, Lx, Ly, k0, win, kx0, ky0, nscan = WINDOWS[wname]
        for s in a.imags.split(","):
            imag = float(s)
            t0 = time.time()
            q = np.asarray(eme_2d.layer_modes(
                strips(kind, Nx, Ly, imag), Lx, Nx, Ly, k0, win,
                kx0=kx0, ky0=ky0, n_scan=nscan))
            rows.append(dict(name="%s_im%g" % (wname, imag), window=wname,
                             imag=imag, n=int(q.size),
                             secs=round(time.time() - t0, 1),
                             hash_q=_vh.hash_arrays(q),
                             q_first=float(q[0]) if q.size else None,
                             q_last=float(q[-1]) if q.size else None,
                             q=[float(v) for v in q]))
            print("  %-24s im=%-8g n=%3d  %5.1fs" % (wname, imag, q.size,
                                                     rows[-1]["secs"]))
    out["layer_modes"] = rows
    _vh.dump(HERE / ("ve_modes_%s.json" % a.tag), out)


if __name__ == "__main__":
    main()
