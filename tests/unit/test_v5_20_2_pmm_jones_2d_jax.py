"""Differentiable (JAX) 2-D anisotropic PMM Jones solver -- pmm_jones_2d.

``pmm_jones_2d`` gains a JAX twin (``pmm/_jax_twod_jones.py``): gradients flow
through the per-region (3, 3) permittivity tensor VALUES, the half-space
indices, depth, wavelength and the incidence angles.  As with the scalar cell
twin, a traced ``eps_tensor_cell`` cannot define the spectral-element walls, so
a CONCRETE ``region_layout`` (int grid) is passed alongside.

The twin ALWAYS drives the full-3x3 generator path -- exact for an in-plane
tensor (off-plane blocks vanish), correct for an out-of-plane one -- so the
forward and the gradient share ONE branch (the routing-bug lesson from the
RCWA-2D twin: a traced tensor's out-of-plane structure is invisible, so keying
the branch on it would silently drop z-coupling under ``jax.grad``).  A
PATTERNED (>= 2 region) cell lifts the modal degeneracy -> ``jnp.linalg.eig`` is
well-conditioned and the twin matches NumPy to machine precision; a UNIFORM
single-region cell is degenerate and RAISES (a planar problem -> Berreman).

These tests pin: forward parity vs NumPy (elementwise R/T + full complex Jones +
basis-invariant Jones singular values) for BOTH an in-plane and an out-of-plane
patterned cell, the gradient vs central finite difference (in-plane, OOP, depth),
a finite gradient under ``jit`` (no host argsort severs the graph), the absorbed
fraction vs an independent NumPy oracle (the lossless-trap killer), and the
JAX-path API guards (incl. the uniform-cell rejection).
"""
from __future__ import annotations

import numpy as np
import pytest

from lumenairy.elements.pmm import pmm_jones_2d

# A sharp-cornered subwavelength cell is mildly truncation-limited (~1% energy);
# the twin matches the NumPy solve to machine precision regardless, so the
# lossless-closure warning is expected and irrelevant to twin<->numpy parity.
pytestmark = pytest.mark.filterwarnings("ignore:.*energy closure.*")

_P = 0.5e-6
_WL = 0.55e-6
_DEP = 0.40e-6
# v5.30: theta/phi moved 20/25 -> 30/40 deg.  The old geometry put order
# (0,-1) at s = 0.99810 in the substrate -- 0.19% from cutoff -- so its
# grazing kz made the power split build-sensitive: the NumPy-vs-JAX TOTAL
# parity swung green -> 1.4e-3 -> 1.1e-2 across CI runs with no code
# change in this path.  A (theta, phi) scan puts the nearest order at
# |s-1| = 0.111 here (the best margin on this order lattice), so the
# parity bars below measure algorithm agreement, not cutoff roulette.
_TH = np.deg2rad(30.0)
_PH = np.deg2rad(40.0)
_DEG = 9
_NO = 3
_KW = dict(theta=_TH, phi=_PH, degree=_DEG, n_orders=_NO)


def _jax():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    return jax


def _blas_pin(n=1):
    """Pin the process BLAS pool to ``n`` threads, or ``None`` if it cannot be.

    See the forward-parity test for why this is needed.  ``threadpoolctl`` is
    an OPTIONAL dependency of this library (it is what
    ``lumenairy.elements.rcwa.set_blas_threads`` needs too), so the caller must
    handle ``None`` -- on a build without it, the pinned leg is skipped and only
    the build-invariant channels are asserted.
    """
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return None
    return threadpool_limits(limits=n, user_api="blas")


def _sv(J):
    return np.sort(np.linalg.svd(np.asarray(J), compute_uv=False))


# ---- tensor builders (xp = np for the oracle, jnp inside a traced loss) -----
def _iso(xp, n):
    return xp.asarray(np.eye(3) * (n ** 2), dtype=xp.complex128)


def _inpl(xp, ne, loss_im=0.0):
    """In-plane anisotropic tensor (eps_xz = eps_yz = 0)."""
    eo = 1.5 ** 2 + 1j * loss_im
    ee = ne ** 2 + 1j * loss_im
    D = xp.diag(xp.asarray([ee, eo, eo], dtype=xp.complex128))
    ca, sa = xp.cos(xp.deg2rad(25.0)), xp.sin(xp.deg2rad(25.0))
    Rz = xp.asarray([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]],
                    dtype=xp.complex128)
    return Rz @ D @ Rz.T


def _uni_oop(xp, ne):
    """Out-of-plane tilted uniaxial (director in the x-z plane)."""
    eo, ee = 1.5 ** 2, ne ** 2
    D = xp.diag(xp.asarray([ee, eo, eo], dtype=xp.complex128))
    ct, st = xp.cos(xp.deg2rad(35.0)), xp.sin(xp.deg2rad(35.0))
    ca, sa = xp.cos(xp.deg2rad(20.0)), xp.sin(xp.deg2rad(20.0))
    Ry = xp.asarray([[ct, 0.0, st], [0.0, 1.0, 0.0], [-st, 0.0, ct]],
                    dtype=xp.complex128)
    Rz = xp.asarray([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]],
                    dtype=xp.complex128)
    R = Rz @ Ry
    return R @ D @ R.T


def _pillar_layout(s=4):
    """A corner block (region 1) in a host (region 0) -- patterned on BOTH axes
    so the modal degeneracy is lifted and jnp.linalg.eig is well-conditioned."""
    lay = np.zeros((s, s), dtype=np.int64)
    lay[: s // 2, : s // 2] = 1
    return lay


def _cell_np(layout, tensors):
    cell = np.zeros(layout.shape + (3, 3), dtype=np.complex128)
    for r, t in enumerate(tensors):
        cell[layout == r] = np.asarray(t)
    return cell


def _cell_jax(jnp, layout, tensors):
    """Broadcast per-region (3, 3) tensors onto the (Sx, Sy, 3, 3) cell using
    the CONCRETE layout mask (keeps traced region values in the graph)."""
    cell = jnp.zeros(layout.shape + (3, 3), dtype=jnp.complex128)
    for r, t in enumerate(tensors):
        mask = jnp.asarray((layout == r).astype(np.float64))[..., None, None]
        cell = cell + mask * jnp.asarray(t)[None, None]
    return cell


# ============================ forward parity ================================
def _forward_pair(kind, degree=11):
    """Run both engines on the same cell and return the parity metrics."""
    import jax.numpy as jnp
    lay = _pillar_layout()
    build = _inpl if kind == "inplane" else _uni_oop
    cell_np = _cell_np(lay, [_iso(np, 1.5), build(np, 1.9)])
    cell_jx = _cell_jax(jnp, lay, [_iso(jnp, 1.5), build(jnp, 1.9)])
    kw = dict(_KW, degree=degree)
    o_n, R_n, T_n, J_n = pmm_jones_2d(_P, _P, cell_np, 1.5, 1.0, _DEP, _WL,
                                      **kw)
    o_j, R_j, T_j, J_j = pmm_jones_2d(_P, _P, cell_jx, 1.5, 1.0, _DEP, _WL,
                                      region_layout=lay, **kw)
    R_j, T_j, J_j = np.asarray(R_j), np.asarray(T_j), np.asarray(J_j)
    return dict(
        orders_equal=bool(np.array_equal(np.asarray(o_j), o_n)),
        order_R=float(np.max(np.abs(R_j - R_n))),
        order_T=float(np.max(np.abs(T_j - T_n))),
        total_R=abs(float(R_j.sum()) - float(R_n.sum())),
        total_T=abs(float(T_j.sum()) - float(T_n.sum())),
        jones=float(np.max(np.abs(J_j - J_n))),
        sv=float(np.max(np.abs(_sv(J_j) - _sv(J_n)))),
        closure_np=float(R_n.sum() + T_n.sum()),
        closure_jx=float(R_j.sum() + T_j.sum()),
    )


@pytest.mark.parametrize("kind", ["inplane", "oop"])
def test_pmm_jones_2d_jax_forward_matches_numpy(kind):
    """Forward parity of the JAX twin against NumPy, on TWO legs.

    THE BLAS-THREAD-COUNT FINDING (M4, 2026-08-04) -- why this test is split.
    The in-plane arm's total-transmission bar had been re-tuned twice (v5.25.0,
    v5.30) and kept rotting.  Holding the code, the build and the geometry
    FIXED and varying ONLY the BLAS pool (`OPENBLAS_NUM_THREADS`) on
    Windows / scipy-openblas 0.3.31 / py3.14 / numpy 2.4.4, degree 11:

        BLAS threads |  1        |  2        |  24
        -------------+-----------+-----------+-----------
        order R      | 4.80e-11  | 7.25e-08  | 2.52e-07
        order T      | 7.11e-06  | 8.04e-03  | 5.41e-03
        total R      | 5.95e-11  | 9.53e-08  | 2.94e-07
        total T      | 1.01e-05  | 3.19e-03  | 1.83e-02   <- old 5e-3 bar
        jones        | 2.47e-10  | 2.05e-07  | 9.87e-07
        sing. values | 9.04e-11  | 1.55e-07  | 3.20e-07

    So the old ``_PAR_TOTAL`` was an ABSOLUTE bar on a magnitude that the BLAS
    thread count sets: it passes at 2 threads (a 2-core CI runner) and fails at
    24, with identical code.  That is the whole "passes CI, fails locally"
    story, and re-tuning the constant could never fix it.

    WHICH ENGINE MOVES.  The JAX side does not: its lossless energy closure is
    2.0124975650960613 (1 thread) vs 2.012497565096153 (24) -- 9e-14 apart, and
    smooth in degree (2.01237 / 2.01239 / 2.01250 / 2.01256 at deg 7/9/11/13).
    The NumPy side does: 2.0125077 -> 2.0307687 at deg 11, i.e. it manufactures
    ~1.8% extra energy on a LOSSLESS cell when the pool is 24.  The NumPy arm
    takes the symmetric ``eig(P Q)`` route for an in-plane cell while the twin
    keeps the generator route (tracing-consistency), and that route's
    near-degenerate mode pair is where the thread-dependence enters.  Degree
    does not rescue it -- deg 9 is thread-stable on THIS build (closure moves
    1.6e-8) while v5.30 measured deg 9 as the bad one and deg 11 as clean on
    CI's build.  Chasing a "good" degree is the roulette, not a fix.

    LEG 1 (always) -- the BUILD-INVARIANT channels, at the historical bars.
    Every channel except total-T is invariant across 1/2/24 threads with >= 3
    orders of headroom (worst: order-T 8.0e-3 against 2e-2).
    LEG 2 (when the BLAS pool can be pinned) -- ALGORITHM agreement, which is
    what this test's name promises.  With one deterministic thread count the
    two engines agree to ~1e-5 and the bars are TIGHTENED by 2-3 orders, so
    this leg has strictly more power to catch a real forward-parity defect than
    the single loose leg it replaces.

    The out-of-plane arm is unaffected (both engines take the generator route):
    every channel is <= 3.6e-14 at both 1 and 24 threads.
    """
    _jax()
    # --- leg 1: native BLAS pool -------------------------------------------
    m = _forward_pair(kind)
    assert m["orders_equal"]
    # v5.24.4 (audit S5-12 / S4-4) -> v5.25.0 (PR #18 CI): per-order bar 2e-2
    # (an order-1 algorithm bug gives O(0.1-1) per-order errors), Jones by the
    # full complex matrix AND its basis-invariant singular values.  These are
    # the channels the 1/2/24-thread table above shows to be invariant, so they
    # keep their historical values and are asserted unconditionally.
    _PAR_ORDER = 2e-2
    assert m["order_R"] < _PAR_ORDER
    assert m["order_T"] < _PAR_ORDER
    assert m["jones"] < 2.0 * np.sqrt(_PAR_ORDER)
    assert m["sv"] < 5e-3
    # ERA-PIN (M4 2026-08-04).  The v5.30 assertion was, verbatim:
    #     _PAR_TOTAL = 5e-3
    #     assert abs(float(np.asarray(R_j).sum()) - float(R_n.sum())) < _PAR_TOTAL
    #     assert abs(float(np.asarray(T_j).sum()) - float(T_n.sum())) < _PAR_TOTAL
    # The R half is invariant (2.9e-7 at 24 threads) and is RETAINED at its
    # original value below.  The T half is the BLAS-thread lottery documented
    # above and moves to leg 2, where it is deterministic and pinned 5x tighter
    # than it ever was here.
    _PAR_TOTAL = 5e-3
    assert m["total_R"] < _PAR_TOTAL
    # --- leg 2: pinned BLAS pool -> deterministic algorithm agreement -------
    pin = _blas_pin(1)
    if pin is None:                 # threadpoolctl absent: leg 1 only
        return
    with pin:
        p = _forward_pair(kind)
    assert p["orders_equal"]
    # Measured at 1 thread (both builds), degree 11:
    #   inplane: order_R 4.8e-11, order_T 7.1e-06, total_R 5.9e-11,
    #            total_T 1.0e-05, jones 2.5e-10, sv 9.0e-11
    #   oop    : every channel <= 3.6e-14
    # Bars sit ~100x above the in-plane numbers -- tight enough that a
    # per-order or total defect two orders below the leg-1 bars still fails.
    assert p["order_R"] < 1e-3
    assert p["order_T"] < 1e-3
    assert p["total_R"] < 1e-3
    assert p["total_T"] < 1e-3
    assert p["jones"] < 1e-3
    assert p["sv"] < 1e-3
    # The twin's own lossless energy closure is the independent instrument that
    # identified NumPy as the moving side: it must not depend on the thread
    # count.  (Lossless cell, two incident polarizations -> 2.0 up to the
    # n_orders=3 truncation defect, ~1.3e-2 here.)
    assert abs(p["closure_jx"] - m["closure_jx"]) < 1e-9


# ============================ gradient vs FD ================================
@pytest.mark.parametrize("kind", ["inplane", "oop"])
def test_pmm_jones_2d_jax_gradient_matches_fd(kind):
    jax = _jax()
    import jax.numpy as jnp
    lay = _pillar_layout()
    build = _inpl if kind == "inplane" else _uni_oop
    host = _iso(jnp, 1.5)

    def loss(ne):
        cell = _cell_jax(jnp, lay, [host, build(jnp, ne)])
        out = pmm_jones_2d(_P, _P, cell, 1.5, 1.0, _DEP, _WL,
                           region_layout=lay, **_KW)
        # gauge-invariant physical scalars: |Jones|^2 + total reflected power
        return (jnp.sum(jnp.abs(jnp.asarray(out[3])) ** 2)
                + jnp.sum(jnp.asarray(out[1])))

    ne0, h = 1.9, 1e-6
    g = float(jax.grad(loss)(jnp.asarray(ne0)))
    fd = (float(loss(jnp.asarray(ne0 + h)))
          - float(loss(jnp.asarray(ne0 - h)))) / (2.0 * h)
    assert abs(g - fd) <= 1e-5 * max(abs(fd), 1.0) + 1e-8


def test_pmm_jones_2d_jax_grad_finite_under_jit():
    jax = _jax()
    import jax.numpy as jnp
    lay = _pillar_layout()
    host = _iso(jnp, 1.5)

    def loss(ne):
        cell = _cell_jax(jnp, lay, [host, _uni_oop(jnp, ne)])
        out = pmm_jones_2d(_P, _P, cell, 1.5, 1.0, _DEP, _WL,
                           region_layout=lay, **_KW)
        return jnp.sum(jnp.abs(jnp.asarray(out[3])) ** 2)

    # a host argsort in the forward/backward split would sever the graph
    g = float(jax.jit(jax.grad(loss))(jnp.asarray(1.9)))
    assert np.isfinite(g)


def test_pmm_jones_2d_jax_grad_depth_matches_fd():
    jax = _jax()
    import jax.numpy as jnp
    lay = _pillar_layout()
    cell = _cell_jax(jnp, lay, [_iso(jnp, 1.5), _inpl(jnp, 1.9)])

    def loss_depth(d):
        out = pmm_jones_2d(_P, _P, cell, 1.5, 1.0, d, _WL,
                           region_layout=lay, **_KW)
        return jnp.sum(jnp.asarray(out[1]))

    d0, h = _DEP, _DEP * 1e-4
    g = float(jax.grad(loss_depth)(jnp.asarray(d0)))
    fd = (float(loss_depth(jnp.asarray(d0 + h)))
          - float(loss_depth(jnp.asarray(d0 - h)))) / (2.0 * h)
    assert abs(g - fd) <= 1e-5 * max(abs(fd), 1.0) + 1e-3 / _DEP


# ==================== lossy absorptance oracle =============================
def test_pmm_jones_2d_jax_lossy_absorbed_fraction():
    _jax()
    import jax.numpy as jnp
    lay = _pillar_layout()
    cell_np = _cell_np(lay, [_iso(np, 1.5), _inpl(np, 1.9, loss_im=0.4)])
    cell_jx = _cell_jax(jnp, lay, [_iso(jnp, 1.5), _inpl(jnp, 1.9, loss_im=0.4)])
    _o, R_n, T_n, _J = pmm_jones_2d(_P, _P, cell_np, 1.5, 1.0, _DEP, _WL, **_KW)
    _o2, R_j, T_j, _J2 = pmm_jones_2d(_P, _P, cell_jx, 1.5, 1.0, _DEP, _WL,
                                      region_layout=lay, **_KW)
    # per incident polarization: absorbed = 1 - (sum R + sum T)
    a_np = 1.0 - (np.asarray(R_n).sum(1) + np.asarray(T_n).sum(1))
    a_j = 1.0 - (np.asarray(R_j).sum(1) + np.asarray(T_j).sum(1))
    assert np.max(np.abs(a_j - a_np)) < 1e-8
    assert np.all(a_np > 1e-2)             # genuinely lossy (not the trap)


# ============================ API guards ===================================
def test_pmm_jones_2d_jax_missing_region_layout_raises():
    _jax()
    import jax.numpy as jnp
    lay = _pillar_layout()
    cell = _cell_jax(jnp, lay, [_iso(jnp, 1.5), _inpl(jnp, 1.9)])
    with pytest.raises(ValueError, match="region_layout"):
        pmm_jones_2d(_P, _P, cell, 1.5, 1.0, _DEP, _WL, **_KW)


def test_pmm_jones_2d_jax_stabilize_rejected():
    _jax()
    import jax.numpy as jnp
    lay = _pillar_layout()
    cell = _cell_jax(jnp, lay, [_iso(jnp, 1.5), _inpl(jnp, 1.9)])
    with pytest.raises(ValueError, match="stabilize"):
        pmm_jones_2d(_P, _P, cell, 1.5, 1.0, _DEP, _WL, region_layout=lay,
                     stabilize=True, **_KW)


def test_pmm_jones_2d_jax_bad_layout_shape_raises():
    _jax()
    import jax.numpy as jnp
    lay = _pillar_layout()
    cell = _cell_jax(jnp, lay, [_iso(jnp, 1.5), _inpl(jnp, 1.9)])
    with pytest.raises(ValueError, match="region_layout must be"):
        pmm_jones_2d(_P, _P, cell, 1.5, 1.0, _DEP, _WL,
                     region_layout=np.zeros((3, 3), dtype=np.int64), **_KW)


@pytest.mark.parametrize("lay", [
    np.zeros((6, 6), dtype=np.int64),                              # uniform
    np.repeat(np.array([[0, 0, 1, 1]]), 4, axis=0),               # stripe: x only
    np.repeat(np.array([[0], [0], [1], [1]]), 4, axis=1),         # stripe: y only
])
def test_pmm_jones_2d_jax_axis_uniform_cell_rejected(lay):
    """A cell uniform along ANY axis (fully uniform, or a 1-D-grating stripe) is
    degenerate under jnp.linalg.eig -> reject, pointing at the 1-D / planar
    differentiable solvers.  Both-axes coupling is required."""
    _jax()
    import jax.numpy as jnp
    lay = np.ascontiguousarray(lay.astype(np.int64))
    tensors = [_inpl(jnp, 1.9)] if lay.max() == 0 else [_inpl(jnp, 1.9),
                                                        _inpl(jnp, 1.6)]
    cell = _cell_jax(jnp, lay, tensors)
    with pytest.raises(NotImplementedError, match="BOTH axes"):
        pmm_jones_2d(_P, _P, cell, 1.5, 1.0, _DEP, _WL, region_layout=lay,
                     **_KW)


def test_pmm_jones_2d_jax_li_formulation_and_three_regions():
    """formulation='li' and a 3-region cell both reproduce NumPy to machine
    precision (the adversarial-probe coverage)."""
    _jax()
    import jax.numpy as jnp
    # li formulation on the pillar
    lay = _pillar_layout()
    regs_np = [_iso(np, 1.5), _inpl(np, 1.9)]
    regs_jx = [_iso(jnp, 1.5), _inpl(jnp, 1.9)]
    _o, R_n, _T, J_n = pmm_jones_2d(_P, _P, _cell_np(lay, regs_np), 1.5, 1.0,
                                    _DEP, _WL, formulation="li", **_KW)
    _o2, R_j, _T2, J_j = pmm_jones_2d(_P, _P, _cell_jax(jnp, lay, regs_jx), 1.5,
                                      1.0, _DEP, _WL, region_layout=lay,
                                      formulation="li", **_KW)
    # v5.24.4 (audit S5-12 / S4-4): cross-BLAS parity bar -- see the
    # in-plane test above.  The li-formulation in-plane block hits the same
    # near-degenerate eig split (observed ~3.8e-5 on CI OpenBLAS vs ~1e-9
    # on the author's BLAS); tolerate it while still catching order-1 bugs.
    _PAR = 2e-3
    assert np.max(np.abs(np.asarray(R_j) - R_n)) < _PAR
    assert np.max(np.abs(np.asarray(J_j) - J_n)) < _PAR

    # a 3-region cell (host + in-plane block + out-of-plane block)
    l3 = np.zeros((6, 6), dtype=np.int64)
    l3[:2, :2] = 1
    l3[4:, 4:] = 2
    t_np = [_iso(np, 1.5), _inpl(np, 1.9), _uni_oop(np, 1.7)]
    t_jx = [_iso(jnp, 1.5), _inpl(jnp, 1.9), _uni_oop(jnp, 1.7)]
    _o, R_n, _T, J_n = pmm_jones_2d(_P, _P, _cell_np(l3, t_np), 1.5, 1.0, _DEP,
                                    _WL, **_KW)
    _o2, R_j, _T2, J_j = pmm_jones_2d(_P, _P, _cell_jax(jnp, l3, t_jx), 1.5, 1.0,
                                      _DEP, _WL, region_layout=l3, **_KW)
    assert np.max(np.abs(np.asarray(R_j) - R_n)) < 1e-9
    assert np.max(np.abs(np.asarray(J_j) - J_n)) < 1e-9
