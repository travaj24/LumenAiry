"""V1 -- WITHOUT-ARM BIT IDENTITY.

Runs a fixed fixture battery that uses ONLY the pre-2026-09-11 public API, so
the SAME file runs against the feature tree (C:/tmp/lum_vmortar) and the
read-only main clone at a68a0da (which has none of the new keywords).  Every
result is reduced to a sha256 of raw IEEE bytes; the two JSON files are then
compared byte-for-byte by ``v1_compare.py``.

Two arms:

* ``stack``  -- 18 stack-level fixtures on the SHARED path (scalar / in-plane
  tensor / out-of-plane / magnetic / slanted / gyrotropic; single layer and
  multi-layer; ``retain_internal`` + ``layer_absorption``; both 2-D pure and
  the 1-D ``PMMStack``, which also changed on this branch for the sliver fix).
* ``pencil`` -- Basis1D matrix families and the assembled 2-D pencils
  (``Rmat``/``Lmat``/``Stt``/``Schur``/``Agen``/``Bgen``) at INTEGER ``N``.

Usage:
    V1_TAG=with  python validation/probe_verify_mortar/v1_bit_identity.py
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import hashlib
import json
import sys
import time

import numpy as np

import lumenairy
from lumenairy.elements.pmm.stack import PMMStack
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure
from lumenairy.elements.pmm.twod_staggered import (
    Basis1D,
    Granet2DTransverseE,
    _global_pair_segmat,
    _stag_fourier_projection,
)

HERE = os.path.dirname(os.path.abspath(__file__))
TAG = os.environ.get("V1_TAG", "x")

# EVERY ARM ASSERTS WHICH TREE IT IMPORTED.  ``V1_EXPECT_ROOT`` is the
# with/without discriminator: the WITHOUT arm runs this same file with
# PYTHONPATH pointing at the read-only main clone (a68a0da).
_ROOT = os.path.abspath(os.environ.get(
    "V1_EXPECT_ROOT", os.path.join(HERE, "..", "..")))
_HAVE = os.path.abspath(lumenairy.__file__)
assert _HAVE.lower().startswith(_ROOT.lower()), (
    f"lumenairy.__file__ = {_HAVE} is not under {_ROOT}")
print(f"[arm] lumenairy = {_HAVE}", flush=True)

WL = 1.0e-6
PX = 0.9e-6
PY = 0.75e-6


def h(a):
    a = np.ascontiguousarray(a)
    m = hashlib.sha256()
    m.update(str(a.dtype).encode())
    m.update(str(a.shape).encode())
    m.update(a.tobytes())
    return m.hexdigest()


def hmulti(*arrs):
    m = hashlib.sha256()
    for a in arrs:
        m.update(h(np.asarray(a)).encode())
    return m.hexdigest()


# --------------------------------------------------------------------- cells
def cell_pillar(nx, ny, e_p=6.0, e_h=2.25):
    c = np.full((nx, ny), e_h, dtype=complex)
    c[nx // 2, ny // 2] = e_p
    return c


def cell_stripe(nx, ny, e_p=6.0, e_h=2.25):
    c = np.full((nx, ny), e_h, dtype=complex)
    c[0, :] = e_p
    return c


def tile_inplane(nx, ny):
    """(nx,ny,3,3) IN-PLANE tensor (no e_xz/e_yz) -- an LC director in x-y."""
    t = np.zeros((nx, ny, 3, 3), dtype=complex)
    host = np.diag([2.25, 2.25, 2.25]).astype(complex)
    no, ne = 1.5, 1.7
    psi = 0.4
    c, s = np.cos(psi), np.sin(psi)
    d = np.array([c, s, 0.0])
    eps = (no ** 2) * np.eye(3) + (ne ** 2 - no ** 2) * np.outer(d, d)
    for i in range(nx):
        for j in range(ny):
            t[i, j] = host
    t[nx // 2, ny // 2] = eps
    return t


def tile_oop(nx, ny, tilt=0.61, azi=0.44):
    """(nx,ny,3,3) OUT-OF-PLANE tensor (tilted director -> e_xz != 0)."""
    t = np.zeros((nx, ny, 3, 3), dtype=complex)
    host = np.diag([2.25, 2.25, 2.25]).astype(complex)
    no, ne = 1.5, 1.7
    d = np.array([np.sin(tilt) * np.cos(azi), np.sin(tilt) * np.sin(azi),
                  np.cos(tilt)])
    eps = (no ** 2) * np.eye(3) + (ne ** 2 - no ** 2) * np.outer(d, d)
    for i in range(nx):
        for j in range(ny):
            t[i, j] = host
    t[nx // 2, ny // 2] = eps
    return t


def tile_gyro(nx, ny):
    """Hermitian gyrotropic (lossless, absorbs nothing)."""
    t = np.zeros((nx, ny, 3, 3), dtype=complex)
    base = np.array([[4.0, 0.9j, 0.0], [-0.9j, 4.0, 0.0], [0.0, 0.0, 4.0]],
                    dtype=complex)
    host = np.diag([2.25, 2.25, 2.25]).astype(complex)
    for i in range(nx):
        for j in range(ny):
            t[i, j] = host
    t[nx // 2, ny // 2] = base
    return t


def mu_cell_pattern(nx, ny):
    c = np.full((nx, ny), 1.0, dtype=complex)
    c[nx // 2, ny // 2] = 1.45
    return c


# ---------------------------------------------------------------- fixtures
def _solve(st, retain=False):
    o, R, T, J = st.solve(retain_internal=retain)
    return o, R, T, J


def fx_scalar_pillar_normal():
    st = PMM2DStackPure(PX, PY, n_superstrate=1.0, n_substrate=1.45,
                        n_modes=5, n_orders=3)
    st.add_layer(0.30e-6, eps_cell=cell_pillar(2, 2))
    st.set_source(WL, theta=0.0, phi=0.0)
    return hmulti(*_solve(st))


def fx_scalar_pillar_conical():
    st = PMM2DStackPure(PX, PY, n_superstrate=1.0, n_substrate=1.45,
                        n_modes=5, n_orders=3)
    st.add_layer(0.30e-6, eps_cell=cell_pillar(2, 2))
    st.set_source(WL, theta=0.25, phi=0.7)
    return hmulti(*_solve(st))


def fx_scalar_pillar_n3_m6():
    st = PMM2DStackPure(PX, PY, n_modes=6, n_orders=3)
    st.add_layer(0.24e-6, eps_cell=cell_pillar(3, 3))
    st.set_source(WL, theta=0.18, phi=0.35)
    return hmulti(*_solve(st))


def fx_stripe_ab_stack():
    st = PMM2DStackPure(PX, PY, n_superstrate=1.0, n_substrate=1.45,
                        n_modes=5, n_orders=3)
    st.add_layer(0.20e-6, eps_cell=cell_stripe(2, 2, 6.0))
    st.add_layer(0.15e-6, eps_cell=cell_stripe(2, 2, 4.0, 1.0))
    st.set_source(WL, theta=0.20, phi=0.0)
    return hmulti(*_solve(st))


def fx_uniform_sandwich():
    st = PMM2DStackPure(PX, PY, n_modes=5, n_orders=3)
    st.add_layer(0.10e-6, eps=2.10)
    st.add_layer(0.22e-6, eps_cell=cell_pillar(2, 2))
    st.add_layer(0.08e-6, eps=1.90)
    st.set_source(WL, theta=0.12, phi=0.9)
    return hmulti(*_solve(st))


def fx_inplane_tensor():
    st = PMM2DStackPure(PX, PY, n_modes=5, n_orders=3)
    st.add_layer(0.26e-6, eps_cell=tile_inplane(2, 2))
    st.set_source(WL, theta=0.18, phi=0.35)
    return hmulti(*_solve(st))


def fx_gyrotropic():
    st = PMM2DStackPure(PX, PY, n_modes=5, n_orders=3)
    st.add_layer(0.26e-6, eps_cell=tile_gyro(2, 2))
    st.set_source(WL, theta=0.18, phi=0.35)
    return hmulti(*_solve(st))


def fx_oop_tensor():
    st = PMM2DStackPure(PX, PY, n_modes=5, n_orders=3)
    st.add_layer(0.26e-6, eps_cell=tile_oop(2, 2))
    st.set_source(WL, theta=0.22, phi=0.44)
    return hmulti(*_solve(st))


def fx_oop_tensor_symmetry_off():
    st = PMM2DStackPure(PX, PY, n_modes=5, n_orders=3, symmetry=False)
    st.add_layer(0.26e-6, eps_cell=tile_oop(2, 2))
    st.set_source(WL, theta=0.0, phi=0.0)
    return hmulti(*_solve(st))


def fx_oop_tensor_symmetry_auto_normal():
    st = PMM2DStackPure(PX, PY, n_modes=5, n_orders=3, symmetry="auto")
    st.add_layer(0.26e-6, eps_cell=tile_oop(2, 2))
    st.set_source(WL, theta=0.0, phi=0.0)
    return hmulti(*_solve(st))


def fx_uniform_tensor_oop():
    d = np.array([np.sin(0.6) * np.cos(0.3), np.sin(0.6) * np.sin(0.3),
                  np.cos(0.6)])
    eps = (1.5 ** 2) * np.eye(3) + (1.7 ** 2 - 1.5 ** 2) * np.outer(d, d)
    st = PMM2DStackPure(PX, PY, n_modes=5, n_orders=3)
    st.add_layer(0.30e-6, eps=eps.astype(complex))
    st.set_source(WL, theta=0.20, phi=0.5)
    return hmulti(*_solve(st))


def fx_magnetic_patterned_mu():
    st = PMM2DStackPure(PX, PY, n_modes=5, n_orders=3)
    st.add_layer(0.24e-6, eps_cell=cell_pillar(2, 2),
                 mu_cell=mu_cell_pattern(2, 2))
    st.set_source(WL, theta=0.15, phi=0.25)
    return hmulti(*_solve(st))


def fx_magnetic_uniform_mu():
    st = PMM2DStackPure(PX, PY, n_modes=5, n_orders=3)
    st.add_layer(0.24e-6, eps_cell=cell_pillar(2, 2), mu=1.3)
    st.set_source(WL, theta=0.15, phi=0.25)
    return hmulti(*_solve(st))


def fx_slanted_patterned():
    st = PMM2DStackPure(PX, PY, n_modes=5, n_orders=3)
    st.add_layer(0.28e-6, eps_cell=cell_pillar(2, 2), slant=(0.18, 0.0))
    st.set_source(WL, theta=0.10, phi=0.0)
    return hmulti(*_solve(st))


def fx_slanted_conical_xy():
    st = PMM2DStackPure(PX, PY, n_modes=5, n_orders=3)
    st.add_layer(0.28e-6, eps_cell=cell_pillar(2, 2), slant=(0.12, 0.09))
    st.set_source(WL, theta=0.16, phi=0.4)
    return hmulti(*_solve(st))


def fx_slanted_stack():
    st = PMM2DStackPure(PX, PY, n_modes=5, n_orders=3)
    st.add_layer(0.18e-6, eps_cell=cell_pillar(2, 2), slant=(0.14, 0.0))
    st.add_layer(0.12e-6, eps=2.4, slant=(0.14, 0.0))
    st.set_source(WL, theta=0.16, phi=0.0)
    return hmulti(*_solve(st))


def fx_retain_internal_lossy():
    st = PMM2DStackPure(PX, PY, n_modes=5, n_orders=3)
    c = np.full((2, 2), 2.25 + 0.0j)
    c[1, 1] = 6.0 + 0.35j
    st.add_layer(0.24e-6, eps_cell=c)
    st.set_source(WL, theta=0.18, phi=0.35)
    o, R, T, J = st.solve(retain_internal=True)
    A = st.layer_absorption()
    return hmulti(o, R, T, J, np.asarray(A))


def fx_retain_internal_3layer():
    st = PMM2DStackPure(PX, PY, n_modes=5, n_orders=3)
    c1 = np.full((2, 2), 2.25 + 0.0j)
    c1[1, 1] = 6.0 + 0.35j
    c2 = np.full((2, 2), 2.0 + 0.0j)
    c2[0, 0] = 5.0 + 0.20j
    st.add_layer(0.16e-6, eps_cell=c1)
    st.add_layer(0.10e-6, eps=2.1 + 0.05j)
    st.add_layer(0.14e-6, eps_cell=c2)
    st.set_source(WL, theta=0.18, phi=0.35)
    o, R, T, J = st.solve(retain_internal=True)
    A = st.layer_absorption()
    return hmulti(o, R, T, J, np.asarray(A))


def fx_per_order_amplitudes():
    st = PMM2DStackPure(PX, PY, n_modes=5, n_orders=3)
    st.add_layer(0.24e-6, eps_cell=cell_pillar(2, 2))
    st.set_source(WL, theta=0.18, phi=0.35)
    st.solve()
    out = []
    for port in ("reflection", "transmission"):
        a = st.per_order_amplitudes(port)
        for k in sorted(k for k in a if isinstance(a[k], np.ndarray)):
            out.append(a[k])
    return hmulti(*out)


# ------------------------------------------------------ 1-D PMMStack arms
def fx_pmm1d_te():
    st = PMMStack(PX, n_superstrate=1.0, n_substrate=1.45, degree=8,
                  n_orders=5)
    st.add_layer(0.30e-6, segments=[(0.5, 6.0), (0.5, 2.25)])
    st.add_layer(0.12e-6, eps=2.1)
    st.set_source(WL, theta=0.21)
    o, R, T, J = st.solve()
    return hmulti(o, R, T, J)


def fx_pmm1d_tm_perlayer():
    st = PMMStack(PX, n_superstrate=1.0, n_substrate=1.45, degree=8,
                  n_orders=5, layer_grids="per-layer")
    st.add_layer(0.30e-6, segments=[(0.5, 6.0), (0.5, 2.25)])
    st.add_layer(0.20e-6, segments=[(1 / 3, 4.0), (2 / 3, 2.0)])
    st.set_source(WL, theta=0.21)
    o, R, T, J = st.solve()
    return hmulti(o, R, T, J)


STACK_FIXTURES = [
    ("scalar_pillar_normal", fx_scalar_pillar_normal),
    ("scalar_pillar_conical", fx_scalar_pillar_conical),
    ("scalar_pillar_n3_m6", fx_scalar_pillar_n3_m6),
    ("stripe_ab_stack", fx_stripe_ab_stack),
    ("uniform_sandwich", fx_uniform_sandwich),
    ("inplane_tensor", fx_inplane_tensor),
    ("gyrotropic_hermitian", fx_gyrotropic),
    ("oop_tensor", fx_oop_tensor),
    ("oop_tensor_symmetry_off", fx_oop_tensor_symmetry_off),
    ("oop_tensor_symmetry_auto_normal", fx_oop_tensor_symmetry_auto_normal),
    ("uniform_tensor_oop", fx_uniform_tensor_oop),
    ("magnetic_patterned_mu", fx_magnetic_patterned_mu),
    ("magnetic_uniform_mu", fx_magnetic_uniform_mu),
    ("slanted_patterned", fx_slanted_patterned),
    ("slanted_conical_xy", fx_slanted_conical_xy),
    ("slanted_stack", fx_slanted_stack),
    ("retain_internal_lossy", fx_retain_internal_lossy),
    ("retain_internal_3layer", fx_retain_internal_3layer),
    ("per_order_amplitudes", fx_per_order_amplitudes),
    ("pmm1d_te_shared", fx_pmm1d_te),
    ("pmm1d_tm_perlayer", fx_pmm1d_tm_perlayer),
]


# ----------------------------------------------------------------- pencils
def basis_hashes():
    out = {}
    for (d, N, M, tau) in [(1.0, 2, 5, 1.0 + 0j), (0.9, 3, 5, 1.0 + 0j),
                           (1.2, 4, 6, np.exp(-1j * 0.37)),
                           (0.75, 2, 7, np.exp(-1j * 1.13)),
                           (1.31, 5, 4, np.exp(1j * 0.5)),
                           (0.6, 1, 8, 1.0 + 0j)]:
        b = Basis1D(d, N, M, tau)
        pre = f"b_d{d}_N{N}_M{M}_t{np.angle(tau):.3f}"
        eps_seg = 1.0 + 0.3 * np.arange(N) + 0.11j * np.arange(N)
        out[pre + "_mass_tt"] = h(b.mass(b.Btilde, b.Btilde))
        out[pre + "_mass_bb"] = h(b.mass(b.B, b.B))
        out[pre + "_mass_tb"] = h(b.mass(b.Btilde, b.B))
        out[pre + "_mass_eps"] = h(b.mass(b.Btilde, b.Btilde, eps_seg))
        out[pre + "_stiff_tt"] = h(b.stiff(b.Btilde, b.Btilde))
        out[pre + "_stiff_eps"] = h(b.stiff(b.B, b.B, eps_seg))
        out[pre + "_mixed_tb"] = h(b.mixed(b.Btilde, b.B))
        out[pre + "_mixed_bt"] = h(b.mixed(b.B, b.Btilde))
        out[pre + "_mixed_tt"] = h(b.mixed(b.Btilde, b.Btilde))
        out[pre + "_segmat_m_tt"] = h(
            _global_pair_segmat(b, b.m_ref, b.Btilde, b.Btilde))
        out[pre + "_segmat_m_bb"] = h(
            _global_pair_segmat(b, b.m_ref, b.B, b.B))
        out[pre + "_segmat_s_tt"] = h(
            _global_pair_segmat(b, b.s_ref, b.Btilde, b.Btilde))
        out[pre + "_segmat_c_tb"] = h(
            _global_pair_segmat(b, b.c_ref, b.Btilde, b.B))
        orders = np.arange(-4, 5)
        for a0 in (0.0, 0.31, -1.7):
            asm = _stag_fourier_projection(b, orders, a0)
            out[pre + f"_four_a{a0}"] = hmulti(asm(b.B), asm(b.Btilde))
        out[pre + "_geom"] = hmulti(b.xb, np.asarray([b.d, b.N, b.M]),
                                    np.asarray([b.tau]))
    return out


def pencil_hashes():
    out = {}
    grids = [(2, 2, 5), (3, 3, 5), (2, 2, 6), (4, 4, 4)]
    kinds = {
        "scalar": lambda nx, ny: (cell_pillar(nx, ny), None, None),
        "inplane": lambda nx, ny: (tile_inplane(nx, ny), None, None),
        "oop": lambda nx, ny: (tile_oop(nx, ny), None, None),
        "gyro": lambda nx, ny: (tile_gyro(nx, ny), None, None),
        "magnetic": lambda nx, ny: (cell_pillar(nx, ny),
                                    mu_cell_pattern(nx, ny), None),
        "slanted": lambda nx, ny: (cell_pillar(nx, ny), None, (0.17, 0.05)),
    }
    for (nx, ny, M) in grids:
        for kname, mk in kinds.items():
            cell, mcell, slant = mk(nx, ny)
            sol = Granet2DTransverseE(
                PX / WL, PY / WL, nx, ny, M, cell,
                alpha0x=0.31, alpha0y=-0.17, k0=2.0 * np.pi,
                mu_cell=mcell, slant=slant)
            pre = f"p_{nx}x{ny}_M{M}_{kname}"
            for attr in ("Rmat", "Lmat", "Stt", "Schur", "Agen", "Bgen"):
                v = getattr(sol, attr, None)
                out[pre + "_" + attr] = "None" if v is None else h(v)
            out[pre + "_offplane"] = str(bool(sol.offplane))
            gb = getattr(sol, "Ggram_blocks", None)
            out[pre + "_Ggram"] = ("None" if gb is None
                                   else hmulti(gb[0], gb[1]))
    return out


def main():
    which = sys.argv[1:] or ["stack", "pencil"]
    res = {"tag": TAG, "lumenairy": os.path.abspath(lumenairy.__file__),
           "version": lumenairy.__version__,
           "python": sys.version.split()[0], "numpy": np.__version__}
    if "pencil" in which:
        t0 = time.time()
        res["basis"] = basis_hashes()
        print(f"[basis] {len(res['basis'])} hashes  {time.time()-t0:.1f}s",
              flush=True)
        t0 = time.time()
        res["pencil"] = pencil_hashes()
        print(f"[pencil] {len(res['pencil'])} hashes  {time.time()-t0:.1f}s",
              flush=True)
    if "stack" in which:
        st = {}
        for name, fn in STACK_FIXTURES:
            t0 = time.time()
            try:
                st[name] = fn()
            except Exception as exc:               # noqa: BLE001
                st[name] = f"ERROR: {type(exc).__name__}: {exc}"
            print(f"[stack] {name:38s} {st[name][:16]}  "
                  f"{time.time()-t0:.1f}s", flush=True)
        res["stack"] = st
    path = os.path.join(HERE, f"v1_bit_identity_{TAG}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    print("wrote", path)


if __name__ == "__main__":
    main()
