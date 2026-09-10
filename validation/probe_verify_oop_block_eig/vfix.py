"""Shared fixtures for the INDEPENDENT verification of the normal-incidence
PARITY block reduction (`_stag_block_eig`) of the pure staggered OUT-OF-PLANE
2-D PMM.

These fixtures are written from the PHYSICS, not copied from the build's own
probe or test file: every cell here is constructed by an explicit rule
(``parity_pair`` builds the mirror pair, ``mirror_check`` asserts the cell
really is its own parity image) so that a fixture that silently stopped being
centro-symmetric would be caught rather than assumed.

Import-safe from either arm (the tip worktree or the fb3fd93 main clone): it
imports only numpy and ``uniaxial_tensor``, both of which predate the build.
"""
import hashlib
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402


def assert_arm(expect_prefix):
    """Hard-assert which library copy this process imported."""
    import lumenairy
    got = os.path.abspath(lumenairy.__file__).replace("\\", "/").lower()
    want = expect_prefix.replace("\\", "/").lower()
    if not got.startswith(want):
        raise SystemExit(f"WRONG ARM: lumenairy at {got}, expected {want}")
    return got, lumenairy.__version__


def uni(no, ne, tilt_deg, azim_deg):
    from lumenairy.elements.rcwa._core import uniaxial_tensor
    return uniaxial_tensor(no, ne, np.deg2rad(tilt_deg),
                           phi=np.deg2rad(azim_deg))


AIR = np.eye(3, dtype=complex)

# ---- the tensor zoo.  Every one of these has NON-ZERO e13/e23/e31/e32, which
# is what routes the layer into the out-of-plane branch.
T_LC = None          # filled lazily (uniaxial_tensor needs the library)


def tensors():
    """The tensor zoo, built on the importing arm."""
    t = {}
    t["lc"] = uni(1.5, 1.75, 40.0, 20.0)                 # tilted uniaxial
    t["lc2"] = uni(1.62, 1.44, 55.0, -35.0)              # negative uniaxial
    t["lossy"] = uni(1.5 + 0.08j, 1.8 + 0.02j, 30.0, 65.0)
    t["metal"] = uni(0.6 + 4.2j, 0.9 + 3.1j, 50.0, 10.0)  # strongly lossy
    # NON-RECIPROCAL (anti-symmetric imaginary off-diagonal; the tensor is
    # Hermitian so the layer stays lossless, but eps != eps^T)
    nr = t["lc"].copy()
    nr[0, 2] += 0.35j
    nr[2, 0] -= 0.35j
    nr[1, 2] += 0.18j
    nr[2, 1] -= 0.18j
    t["nonrec"] = nr
    # a GENERIC dense tensor with every out-of-plane entry populated and no
    # symmetry at all (neither Hermitian nor complex-symmetric)
    g = np.array([[2.4 + 0.05j, 0.31 - 0.02j, 0.44 + 0.11j],
                  [0.17 + 0.09j, 2.8 - 0.03j, -0.26 + 0.07j],
                  [-0.38 + 0.02j, 0.52 + 0.13j, 3.1 + 0.06j]], dtype=complex)
    t["generic"] = g
    return t


def inplane_tensor():
    """Genuinely IN-PLANE: e13 = e23 = e31 = e32 = 0 exactly."""
    a = uni(1.5, 1.9, 90.0, 33.0)          # director in the x-y plane
    a = a.copy()
    a[0, 2] = a[1, 2] = a[2, 0] = a[2, 1] = 0.0
    return a


def tile(t, n):
    c = np.zeros((n, n, 3, 3), dtype=complex)
    c[:, :] = t
    return c


def is_parity_image(cell, tol=0.0):
    """Is ``cell`` its own image under ``(i, j) -> (n-1-i, m-1-j)``?"""
    c = np.asarray(cell)
    return float(np.max(np.abs(c - c[::-1, ::-1]))) <= tol


# --------------------------------------------------------------------------- #
# CARRYING cells -- built so they are their own parity image BY CONSTRUCTION
# --------------------------------------------------------------------------- #
def centro_pair(t_pair, n, bg=None, centre=None):
    """Fill the mirror PAIR (0,0)/(n-1,n-1) with ``t_pair`` and (optionally)
    the parity-fixed centre pixel with ``centre`` (odd ``n`` only)."""
    c = tile(AIR if bg is None else bg, n)
    c[0, 0] = t_pair
    c[n - 1, n - 1] = t_pair
    if centre is not None:
        if n % 2 == 0:
            raise ValueError("no parity-fixed centre pixel on an even grid")
        c[n // 2, n // 2] = centre
    assert is_parity_image(c), "fixture is not its own parity image"
    return c


def centro_cross(t_a, t_b, n):
    """A 4-pixel orbit: (0,1)&(n-1,n-2) get ``t_a``, (1,0)&(n-2,n-1) get
    ``t_b``.  Distinct tensors on the two orbits, so a shape-only test would
    not see the difference."""
    c = tile(AIR, n)
    c[0, 1] = t_a
    c[n - 1, n - 2] = t_a
    c[1, 0] = t_b
    c[n - 2, n - 1] = t_b
    assert is_parity_image(c)
    return c


def centro_interior(t_ring, t_centre, n=3):
    """(3,3) with an INTERIOR feature: the centre pixel differs from the ring,
    and the ring itself is a parity orbit."""
    c = tile(AIR, n)
    c[0, 0] = c[2, 2] = t_ring
    c[0, 2] = c[2, 0] = t_ring
    c[1, 1] = t_centre
    assert is_parity_image(c)
    return c


# --------------------------------------------------------------------------- #
# VIOLATING cells
# --------------------------------------------------------------------------- #
def offcentre(t, n):
    c = tile(AIR, n)
    c[0, 0] = t
    assert not is_parity_image(c)
    return c


def parity_breaking_tensor(t_a, t_b, n):
    """Parity-symmetric PATTERN, parity-BREAKING tensors: the two mirror
    pixels are both filled, with DIFFERENT tensors."""
    c = tile(AIR, n)
    c[0, 0] = t_a
    c[n - 1, n - 1] = t_b
    assert not is_parity_image(c), "the two tensors must actually differ"
    # the FILLED-PIXEL PATTERN, however, IS its own parity image
    filled = np.max(np.abs(c - AIR), axis=(2, 3)) > 0
    assert np.array_equal(filled, filled[::-1, ::-1])
    return c


# --------------------------------------------------------------------------- #
# hashing / metrics
# --------------------------------------------------------------------------- #
def sha(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        x = np.ascontiguousarray(np.asarray(a))
        h.update(str(x.dtype).encode())
        h.update(str(x.shape).encode())
        h.update(x.tobytes())
    return h.hexdigest()


def dmax(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def hausdorff(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    d = np.abs(a[:, None] - b[None, :])
    return max(float(np.max(np.min(d, axis=1))),
               float(np.max(np.min(d, axis=0))))


# ---- the mount used everywhere below (deliberately NOT the build's numbers)
PX = 1.05e-6
PY = 0.95e-6
WL = 0.633e-6
DEP = 0.29e-6
NSUP = 1.0
NSUB = 1.46
