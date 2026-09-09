"""V3b -- |jones|^2 <-> order-0 efficiency consistency, done properly.

V3's part D compared ``R00`` against the reflected |E|^2 rebuilt from the
Jones and got an exact match at NORMAL incidence but a 6.8e-3 mismatch at
oblique.  The missing piece is the INCIDENT normalization: the Jones columns
are the response to a unit TRANSVERSE incident field, and an obliquely
incident plane wave with transverse part ``e`` carries a longitudinal
component too (``k . E = 0``), so ``|E_inc|^2 = 1 + |k_t . e / kz|^2``.

This probe closes that and uses the per-column normalization as an extra
COLUMN-CONVENTION discriminator: the normalization differs between column 0
(``kx``) and column 1 (``ky``), so feeding the TRANSPOSED Jones into the same
identity misses -- even for a cell whose Jones is symmetric, where a plain
``J`` vs ``J.T`` comparison is blind.
"""
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import lumenairy  # noqa: E402

_ROOT = os.path.abspath("C:/tmp/lum_aniso")
assert os.path.abspath(lumenairy.__file__).startswith(_ROOT), lumenairy.__file__

from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    pmm_jones_2d_staggered,
)
from lumenairy.elements.rcwa._core import uniaxial_tensor  # noqa: E402

LC = uniaxial_tensor(1.45, 1.75, np.pi / 2, phi=0.90)
GY = np.array([[2.60, 0.35j, 0.0], [-0.35j, 2.60, 0.0],
               [0.0, 0.0, 2.40]], dtype=complex)
ISO = 5.0 * np.eye(3, dtype=complex)
P2, WL2, DEP2, NS2 = 0.62e-6, 0.50e-6, 0.31e-6, 1.45


def cell(host, pillar, n=2):
    c = np.empty((n, n, 3, 3), dtype=complex)
    c[:] = host
    c[0, 0] = pillar
    return c


def power_from_jones(J, kx, ky, kz):
    """order-0 reflected power per incident column, flux-normalized."""
    out = []
    for col in (0, 1):
        e = np.array([J[0, col], J[1, col]])
        ez = (kx * e[0] + ky * e[1]) / kz          # k_r = (kx, ky, -kz)
        inc = np.zeros(2, complex)
        inc[col] = 1.0
        ez_i = -(kx * inc[0] + ky * inc[1]) / kz   # k_i = (kx, ky, +kz)
        out.append(float((abs(e[0]) ** 2 + abs(e[1]) ** 2 + abs(ez) ** 2)
                         / (1.0 + abs(ez_i) ** 2)))
    return np.array(out)


OUT = {}
for host, pill, tag in ((LC, ISO, "lc_host"), (GY, ISO, "gyro_host")):
    for th, ph in ((0.0, 0.0), (0.30, 0.70), (0.55, 1.30)):
        o, R, T, J = pmm_jones_2d_staggered(P2, P2, cell(host, pill), NS2,
                                            1.0, DEP2, WL2, degree=7,
                                            n_orders=5, theta=th, phi=ph)
        i0 = int(np.where((np.asarray(o)[:, 0] == 0)
                          & (np.asarray(o)[:, 1] == 0))[0][0])
        kx = np.sin(th) * np.cos(ph)
        ky = np.sin(th) * np.sin(ph)
        kz = np.sqrt(1.0 - kx ** 2 - ky ** 2)
        eng = np.array([float(R[0][i0]), float(R[1][i0])])
        pred = power_from_jones(J, kx, ky, kz)
        predT = power_from_jones(J.T, kx, ky, kz)
        OUT[f"{tag}_th{th:.2f}_ph{ph:.2f}"] = dict(
            R00_engine=eng.tolist(), R00_from_jones=pred.tolist(),
            residual=float(np.max(np.abs(pred - eng))),
            FAILBEFORE_transposed_columns=float(np.max(np.abs(predT - eng))),
            jones_is_symmetric=float(abs(J[0, 1] - J[1, 0])))

fn = ("C:/tmp/lum_aniso/validation/probe_verify_staggered_aniso/"
      "out_v3b_jones_power.json")
json.dump(OUT, open(fn, "w"), indent=1)
for k, v in OUT.items():
    print(f"{k:26s} res={v['residual']:.3e}  "
          f"failbefore(J.T)={v['FAILBEFORE_transposed_columns']:.3e}  "
          f"|J01-J10|={v['jones_is_symmetric']:.3e}")
print("written", fn)
