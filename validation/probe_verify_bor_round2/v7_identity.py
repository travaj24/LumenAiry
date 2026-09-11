"""V7 -- BIT-IDENTITY.  Round 2 must move NOTHING except the rows its new
screen refuses.

An INDEPENDENT battery: 36 BOR fixtures and 12 EME fixtures, hashed to the
SHA-256 of the exact IEEE-754 bytes of the answer.  A fixture that RAISES is
recorded as its exception class plus the first 80 characters of the message,
so a newly refused row shows up as a MOVE with a named cause rather than as a
crash.

The BOR battery deliberately includes the LEGACY NODAL path, which is the only
path the passivity screen can touch -- the round-2 report's own battery is
``BORStack``-only (fd / sem), where the screen is structurally unreachable, so
its "nothing moved" is a weaker statement than it looks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _vfix as F  # noqa: E402

import lumenairy  # noqa: E402
from lumenairy.elements.bor import bor_solve as BS  # noqa: E402
from lumenairy.elements.bor.bor_stack import BORStack  # noqa: E402
from lumenairy.elements.eme import eme_2d as E2  # noqa: E402
from lumenairy.elements.eme import eme_2d_vector as EV  # noqa: E402
from lumenairy.elements.eme.eme_diffraction import mode_match, plane_wave_orders  # noqa: E402


def _prov(name):
    import scipy
    import threadpoolctl
    return dict(tag=name, lumenairy_file=lumenairy.__file__,
                python=sys.version.split()[0], numpy=np.__version__,
                scipy=scipy.__version__,
                blas_arch=sorted({str(d.get("architecture"))
                                  for d in threadpoolctl.threadpool_info()}),
                blas_threads=sorted({int(d.get("num_threads", -1))
                                     for d in threadpoolctl.threadpool_info()}),
                env={k: os.environ.get(k) for k in
                     ("OPENBLAS_CORETYPE", "OMP_NUM_THREADS",
                      "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})


def _ky_fwd(lam, qz2, k0):
    """``_ky_forward`` on either tree -- the PRE tree takes no ``k0``."""
    try:
        return E2._ky_forward(lam, qz2, k0)
    except TypeError:
        return E2._ky_forward(lam, qz2)


def _cellS(sm, qz2, k0):
    try:
        return E2.cell_smatrix(sm, qz2, k0=k0)
    except TypeError:
        return E2.cell_smatrix(sm, qz2)


def _disp(sm, qz2, ky0, Ly, k0):
    try:
        return E2.dispersion(sm, qz2, ky0, Ly, k0=k0)
    except TypeError:
        return E2.dispersion(sm, qz2, ky0, Ly)


def _hash(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a))
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------- #
#  BOR
# --------------------------------------------------------------------- #
def _bor_stack_fixture(basis, m, k0, kind, scale=1.0, N=140, degree=8):
    Rbig = 4.0 * scale
    st = BORStack(Rbig, m, n_substrate=1.5, n_superstrate=1.0, N=N,
                  basis=basis, degree=degree)
    if kind == "uniform":
        st.add_layer(0.5 * scale, eps=4.0)
    elif kind == "rings":
        st.add_layer(0.5 * scale, rings=(1.2 * scale, 0.5, 2.2, 1.45))
    elif kind == "lossy":
        st.add_layer(0.5 * scale, eps=complex(4.0, 1e-3))
    elif kind == "three":
        st.add_layer(0.3 * scale, eps=4.0)
        st.add_layer(0.4 * scale, rings=(1.0 * scale, 0.4, 2.4, 1.3))
        st.add_layer(0.3 * scale, eps=2.25)
    elif kind == "taper":
        for i in range(8):
            st.add_layer(0.06 * scale,
                         segments=[((1.0 + 0.2 * i) * scale, 4.0),
                                   (Rbig, 1.0)])
    elif kind == "aniso":
        st.add_layer(0.5 * scale, eps_tensor=(4.0, 3.6, 4.4))
    st.set_source(k0=k0 / scale)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = st.solve()
    return _hash(np.asarray(r["R"], float), np.asarray(r["T"], float),
                 np.asarray(r["q"]), np.asarray(r["energy"], float))


def _bor_nodal_fixture(kind, m, N, rbl, k0, im_rel=0.0):
    Rbig = F.rbig_of(rbl, k0)
    if kind == "uniform":
        mid = F.uniform(6.0, im_rel)
    elif kind == "ring":
        mid = F.ring(2.0, 6.0, Rbig, n_rings=4, im_rel=im_rel)
    else:
        mid = F.segments([2.0, 6.0, 3.0, 9.0], Rbig, im_rel=im_rel, im_on=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        layers = [BS.build_layer(m, Rbig, N, F.uniform(2.0), k0,
                                 basis="nodal"),
                  BS.build_layer(m, Rbig, N, mid, k0, basis="nodal",
                                 thickness=0.35),
                  BS.build_layer(m, Rbig, N, F.uniform(2.0), k0,
                                 basis="nodal")]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = BS.solve(layers, k0)
    return _hash(np.asarray(r["R"], float), np.asarray(r["T"], float),
                 np.asarray(r["energy"], float))


def bor_fixtures():
    out = {}
    for basis in ("fd", "sem"):
        for m in (0, 1, 2, 5):
            out["stack_%s_m%d_rings" % (basis, m)] = (
                _bor_stack_fixture, (basis, m, 2.0, "rings"))
        for kind in ("uniform", "lossy", "three", "taper"):
            out["stack_%s_%s" % (basis, kind)] = (
                _bor_stack_fixture, (basis, 1, 2.0, kind))
        for k0 in (0.8, 3.5):
            out["stack_%s_k0%g" % (basis, k0)] = (
                _bor_stack_fixture, (basis, 1, k0, "rings"))
        out["stack_%s_nmunits" % (basis,)] = (
            _bor_stack_fixture, (basis, 1, 2.0, "rings", 1000.0))
    out["stack_fd_aniso"] = (_bor_stack_fixture, ("fd", 1, 2.0, "aniso"))
    out["stack_sem_aniso"] = (_bor_stack_fixture, ("sem", 1, 2.0, "aniso"))
    # the LEGACY NODAL path -- the only one the screen can reach
    for kind in ("uniform", "ring", "segment"):
        for m in (0, 1, 2):
            for rbl in (0.5, 2.0):
                out["nodal_%s_m%d_rbl%g" % (kind, m, rbl)] = (
                    _bor_nodal_fixture, (kind, m, 200, rbl, 2.0))
    for im in (1e-13, 3e-12, 1e-6, 1e-2):
        out["nodal_ring_lossy_%g" % (im,)] = (
            _bor_nodal_fixture, ("ring", 1, 200, 2.0, 2.0, im))
    return out


# --------------------------------------------------------------------- #
#  EME
# --------------------------------------------------------------------- #
def _eme_strip(Nx, k0, im_eps, scale=1.0):
    Lx = 1.0 * scale
    eps = np.full(Nx, 1.0 + 0j)
    eps[Nx // 3:2 * Nx // 3] = 12.0 + im_eps
    lam, Phi = E2.strip_x_modes(eps, Lx, Nx, k0 / scale)
    ky = _ky_fwd(lam, 0.0, k0 / scale)
    return _hash(np.asarray(lam), np.asarray(Phi), np.asarray(ky))


def _eme_cell(Nx, k0, im_eps):
    eps = np.full(Nx, 1.0 + 0j)
    eps[Nx // 3:2 * Nx // 3] = 12.0 + im_eps
    sm = [(E2.strip_x_modes(eps, 1.0, Nx, k0), 0.3),
          (E2.strip_x_modes(np.full(Nx, 2.25 + 0j), 1.0, Nx, k0), 0.2)]
    S = _cellS(sm, 0.3 * k0 ** 2, k0)
    d = _disp(sm, 0.3 * k0 ** 2, 0.0, 1.0, k0)
    return _hash(*[np.asarray(b) for b in S], np.asarray([d]))


def _eme_modematch(Nx, k0, depth):
    orders = plane_wave_orders(3, 0)
    K = len(orders)
    ang = np.arange(Nx)
    Psi = np.exp(2j * np.pi * np.outer(ang, np.arange(K)) / Nx)
    qz2 = (k0 ** 2) * np.linspace(0.2, 1.4, K)
    r = mode_match(qz2, Psi, orders, kx0=0.0, ky0=0.0, k0=k0, eps_sup=1.0,
                   eps_sub=2.25, depth=depth, Lx=1.0, Ly=1.0, Nx=Nx, Ny=1)
    return _hash(np.asarray(r["R"], float), np.asarray(r["T"], float))


def _eme_vector(Nx, k0, im_eps):
    eps = np.full(Nx, 1.0 + 0j)
    eps[Nx // 3:2 * Nx // 3] = 12.0 + im_eps
    v = EV.strip_vector_modes(eps, 1.0, Nx, k0, 0.0, 0.2 * k0 ** 2)
    arrs = [np.asarray(x) for x in v] if isinstance(v, tuple) else \
        [np.asarray(v[k]) for k in sorted(v)]
    return _hash(*arrs)


def eme_fixtures():
    out = {}
    for Nx in (48, 96):
        for k0 in (2.0 * np.pi / 1.55, 2.0 * np.pi / 0.633):
            for im in (0.0, -1e-6j, 1e-3j):
                out["strip_%d_k%.3f_im%s" % (Nx, k0, im)] = (
                    _eme_strip, (Nx, k0, im))
    out["strip_nmunits"] = (_eme_strip, (64, 2.0 * np.pi / 1.55, -1e-6j,
                                         1000.0))
    out["cell_64"] = (_eme_cell, (64, 2.0 * np.pi / 1.55, -1e-6j))
    out["cell_48_real"] = (_eme_cell, (48, 2.0 * np.pi / 1.55, 0.0))
    out["modematch_a"] = (_eme_modematch, (64, 2.0 * np.pi / 1.55, 0.4))
    out["modematch_b"] = (_eme_modematch, (48, 2.0 * np.pi / 0.633, 1.2))
    out["vector_48"] = (_eme_vector, (48, 2.0 * np.pi / 1.55, -1e-6j))
    out["vector_64_real"] = (_eme_vector, (64, 2.0 * np.pi / 1.55, 0.0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="local")
    a = ap.parse_args()
    res = {}
    for group, fx in (("BOR", bor_fixtures()), ("EME", eme_fixtures())):
        for name, (fn, args) in fx.items():
            key = "%s:%s" % (group, name)
            try:
                res[key] = fn(*args)
            except Exception as exc:                     # noqa: BLE001
                res[key] = "RAISED:%s:%s" % (type(exc).__name__,
                                             str(exc)[:80])
            print("%-46s %s" % (key, res[key][:40]), flush=True)
    out = dict(provenance=_prov(a.tag), n_bor=len(bor_fixtures()),
               n_eme=len(eme_fixtures()), hashes=res)
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "v7_identity_%s.json" % (a.tag,))
    with open(p, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", p, "n =", len(res))


if __name__ == "__main__":
    main()
