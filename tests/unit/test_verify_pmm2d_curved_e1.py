"""INDEPENDENT VERIFIER of curved-cell Phase E1 (out-of-plane tensors and
slanted walls inside curved cells of the pure staggered 2-D PMM) --
decision tests for what the builder's suite
(``tests/unit/test_pmm2d_staggered_curved_e1.py``) does not decide
(``docs/audits/VERIFY_PMM2D_CURVED_E1_2026_10_03.md``; probes and JSON in
``validation/probe_pmm2d_curved/verify_e1/``).

Words.  Under a coordinate map with Jacobian ``J`` (``sg = det J``) every
region is magnetic, so the first-order out-of-plane generator (state
``[E1; E2; G1; G2]``) carries permeability blocks; a slanted layer under a
map is the composite frame ``x = Phi(u, v) + t w`` whose shear reaches the
generator as ``tau = J^-1 t``.  ``_OOP_ROT_SIGN`` / ``_OOP_H_GAUGE`` are
the two measured gauge constants of the out-of-plane path.  A "C2-broken"
cell has no 180-degree rotation symmetry about z.  "The same device two
ways": a pattern whose material walls are all grid lines, solved unmapped
and under a transfinite map whose moved vertices all sit inside ONE
material -- the physical structure is identical, only the discretisation
differs.

Every bar was measured by the verifier on 2026-10-03 (Windows 11, CPython
3.14.6, numpy 2.4.4, scipy 1.17.1, OMP / OPENBLAS / MKL = 1; the readings
named next to each assertion) and is stated with its gap on both sides.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import warnings  # noqa: E402
from contextlib import contextmanager  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lumenairy.elements.pmm import Circle, PMM2DStackPure  # noqa: E402
from lumenairy.elements.pmm import _curvemap as CM  # noqa: E402
from lumenairy.elements.pmm import stack2d_pure as SP  # noqa: E402
from lumenairy.elements.pmm import twod_staggered as TS  # noqa: E402
from lumenairy.elements.rcwa._core import uniaxial_tensor  # noqa: E402

_EYE = np.eye(3, dtype=complex)
#: a director in a GENERAL direction (polar 52 deg, azimuth 37 deg)
_DIR = uniaxial_tensor(1.52, 1.78, np.deg2rad(52.0), phi=np.deg2rad(37.0))
_MU_GYRO = np.array([[1.35, -0.25j, 0], [0.25j, 1.5, 0], [0, 0, 1.1]],
                    complex)
_MU_GYRO_LOSSY = np.array([[1.45 + 0.04j, 0.28j + 0.02, 0],
                           [-0.28j + 0.02, 1.3 + 0.03j, 0],
                           [0, 0, 1.15 + 0.02j]], complex)
_MU_FULL_OOP = np.array([[1.4, 0.1, 0.15], [0.1, 1.3, 0.12],
                         [0.15, 0.12, 1.2]], complex)


@contextmanager
def _patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


# =========================================================================== #
# 1. the derivation: a z-independent map cannot create mu'^{3t}; the
#    composite frame's identities; an out-of-plane mu is refused everywhere
# =========================================================================== #
def test_ve1_map_keeps_mu_block_form_and_the_composite_frame_is_tau():
    """For ANY J (det > 0) and any block-form mu, the map's congruence
    ``sg Lm^-1 mu Lm^-T`` (``Lm = blockdiag(J, 1)``) keeps the t3 / 3t blocks
    EXACTLY zero (measured 0.0 over 2000 random draws, ``v2a_mu3t.json``),
    so the refused out-of-plane permeability can only enter from the
    material.  The composite frame ``Lambda = [[J, t], [0, 1]]`` creates one
    and the generator's two weights are exactly it: ``-mu'^{3t} / mu'^{33}
    = tau^T`` with ``tau = J^-1 t`` (2.7e-15), ``chi'_t3 = chi'_tt tau``,
    ``chi'_tt = J^T mu_t^-1 J / sg`` unchanged by the shear and the Schur
    complement ``1 / (sg mu33)`` (5.8e-13 relative).  Bars 1e-12 / 1e-10."""
    rng = np.random.default_rng(5)
    for _ in range(200):
        J = rng.normal(size=(2, 2)) + 2 * np.eye(2)
        sg = np.linalg.det(J)
        if sg <= 0.1:
            continue
        mu = np.zeros((3, 3), complex)
        mu[:2, :2] = (rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
                      + 3 * np.eye(2))
        mu[2, 2] = 1.5 + 0.2j
        L = np.eye(3)
        L[:2, :2] = J
        Mm = sg * np.linalg.inv(L) @ mu @ np.linalg.inv(L).T
        assert float(np.abs(Mm[:2, 2]).max() + np.abs(Mm[2, :2]).max()) \
            <= 1e-12 * float(np.abs(Mm).max())
        t = rng.normal(size=2) * 0.4
        Lc = L.copy()
        Lc[:2, 2] = t
        Mc = sg * np.linalg.inv(Lc) @ mu @ np.linalg.inv(Lc).T
        chi = np.linalg.inv(Mc)
        tau = np.linalg.solve(J, t)
        assert float(np.abs(-Mc[2, :2] / Mc[2, 2] - tau).max()) <= 1e-12
        s = float(np.abs(chi).max())
        assert float(np.abs(chi[:2, 2] - chi[:2, :2] @ tau).max()) <= 1e-10 * s
        assert float(np.abs(chi[:2, :2] - J.T @ np.linalg.inv(mu[:2, :2]) @ J
                            / sg).max()) <= 1e-10 * s


def _oop_mu_calls():
    P = 1.0
    eps = np.broadcast_to(_EYE, (3, 3, 3, 3)).copy()
    eps[1, 1] = _DIR
    mc = np.broadcast_to(_EYE, (3, 3, 3, 3)).copy()
    mc[1, 1] = _MU_FULL_OOP
    cm = CM._circle_map_3x3(P, 0.3)[0]

    def shape():
        st = PMM2DStackPure(P, P, n_modes=4, n_orders=2)
        st.add_layer(0.3, shapes=[Circle(0.5, 0.5, 0.3, 2.0,
                                         mu=_MU_FULL_OOP)],
                     background_eps=1.0)
        st.set_source(1.0)
        st.solve()
    return {
        "solver": lambda: TS.Granet2DTransverseE(P, P, 3, 3, 4, eps,
                                                 mu_cell=mc),
        "solver_map": lambda: TS.Granet2DTransverseE(
            P, P, cm.u_walls, cm.v_walls, 4, eps, mu_cell=mc, cmap=cm),
        "solver_slant": lambda: TS.Granet2DTransverseE(
            P, P, 3, 3, 4, eps, mu_cell=mc, slant=(0.1, 0.0)),
        "jones_map": lambda: TS.pmm_jones_2d_staggered(
            P, P, eps, 1.5, 1.0, 0.3, 1.0, degree=4, n_orders=2,
            mu_cell=mc, cmap=cm),
        "stack_uniform": lambda: PMM2DStackPure(P, P, n_modes=4).add_layer(
            0.3, eps=_DIR, mu=_MU_FULL_OOP),
        "stack_map": lambda: PMM2DStackPure(P, P, n_modes=4,
                                            cmap=cm).add_layer(
            0.3, eps_cell=eps, mu_cell=mc),
        "stack_slant": lambda: PMM2DStackPure(P, P, n_modes=4).add_layer(
            0.3, eps=2.0, mu=_MU_FULL_OOP, slant=(0.1, 0.0)),
        "shape": shape,
    }


def test_ve1_an_out_of_plane_mu_is_refused_loudly_at_every_entry():
    """Eight entry points (solver with / without a map / with a slant, the
    Jones entry under a map, uniform / mapped / slanted stack layers, a
    shape's ``mu=``): each raises ``NotImplementedError`` naming
    "OUT-OF-PLANE" and the permeability (``v2a_mu3t.json``)."""
    for name, fn in _oop_mu_calls().items():
        with pytest.raises(NotImplementedError) as ei:
            fn()
        msg = str(ei.value)
        assert "OUT-OF-PLANE" in msg, (name, msg)
        assert "permeability" in msg or "mu" in msg, (name, msg)


def test_ve1_the_out_of_plane_mu_refusal_states_the_current_reason():
    for name, fn in _oop_mu_calls().items():
        with pytest.raises(NotImplementedError) as ei:
            fn()
        assert "has no mu blocks" not in str(ei.value), name


# =========================================================================== #
# 2. B stays Hermitian positive definite under the circle map
# =========================================================================== #
def test_ve1_the_permeability_side_of_B_stays_hpd_under_the_circle_map():
    """``-R = C[chi_t]C`` with the Duffy-weighted vacuum ``chi_t = g / sg``
    (pointwise eigenvalues down to ~2e-3 near the singular vertices) and a
    lossless gyrotropic mu in the disk: Hermitian to round-off (5e-17), its
    smallest generalised eigenvalue against the PLAIN Gram 0.30 / 0.26 /
    0.23 at M = 6 / 7 / 8 (``v2b_hpd_6_7_8.json``; r 0.45 p: 0.22 .. 0.17)
    -- bounded away from zero, the Cholesky never fails.  At the unit size
    (M = 5) bar >= 0.1 (a decade below the trend; 0 would mean the
    whitening is unsafe); a LOSSY mu flags ``_bgen_hermitian = False``."""
    import scipy.linalg as sla
    P = 1.0
    cm = CM._circle_map_3x3(P, 0.3)[0]
    eps = np.broadcast_to(_EYE, (3, 3, 3, 3)).copy()
    eps[1, 1] = _DIR
    mc = np.broadcast_to(_EYE, (3, 3, 3, 3)).copy()
    mc[1, 1] = _MU_GYRO
    s = TS.Granet2DTransverseE(P, P, cm.u_walls, cm.v_walls, 5, eps,
                               mu_cell=mc, cmap=cm)
    qq = s.q * s.q
    mR = s.Bgen[:2 * qq, :2 * qq]
    assert float(np.abs(mR - mR.conj().T).max() / np.abs(mR).max()) <= 1e-14
    assert s._bgen_hermitian
    Gp = np.zeros_like(mR)
    Gp[:qq, :qq] = s.Bgen[3 * qq:, 3 * qq:]
    Gp[qq:, qq:] = s.Bgen[2 * qq:3 * qq, 2 * qq:3 * qq]
    gev = sla.eigh(0.5 * (mR + mR.conj().T), Gp, eigvals_only=True)
    assert float(gev[0]) >= 0.1, gev[0]
    np.linalg.cholesky(s.Bgen)
    mc[1, 1] = _MU_GYRO_LOSSY
    s2 = TS.Granet2DTransverseE(P, P, cm.u_walls, cm.v_walls, 4, eps,
                                mu_cell=mc, cmap=cm)
    assert not s2._bgen_hermitian


# =========================================================================== #
# 3. the gauge constants are map-INDEPENDENT -- decided on a C2-broken
#    cell at NORMAL incidence, the same device two ways
# =========================================================================== #
_P3 = 1.0
_TRI = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (0, 2)]


def _tri_solve(cm, M=4):
    e = np.broadcast_to(_EYE, (4, 4, 3, 3)).copy()
    for c in _TRI:
        e[c] = _DIR
    st = PMM2DStackPure(_P3, _P3, n_superstrate=1.0, n_substrate=1.5,
                        n_modes=M, n_orders=2, cmap=cm)
    st.add_layer(0.4, eps_cell=e)
    st.set_source(1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _o, R, T, J = st.solve(jones=True)
    return np.concatenate([np.asarray(R).ravel(), np.asarray(T).ravel(),
                           np.abs(np.asarray(J)).ravel()])


def _tri_map():
    w = np.linspace(0.0, _P3, 5)
    V = np.stack(np.meshgrid(w, w, indexing="ij"), axis=-1)
    # every moved vertex sits inside ONE material: (1,1) in the tensor
    # triangle, (3,3) and (2,3) in the background -- the device is unchanged
    for (i, j), (dx, dy) in {(1, 1): (0.06, -0.04), (3, 3): (-0.05, 0.03),
                             (2, 3): (0.04, 0.05)}.items():
        V[i, j] += (dx * _P3, dy * _P3)
    return CM.TransfiniteMap(w, w, V, None)


def test_ve1_gauge_constants_are_map_independent_on_a_c2_broken_cell():
    """A staircase triangle of a general-director tensor (no 180-degree
    symmetry: the rotation sign IS visible at normal incidence here, unlike
    on a film) solved unmapped and under a non-symmetric 4 x 4 transfinite
    map that leaves every material wall in place.  Measured
    (``v3_same_M{4,5,6,7}.json``, R / T / |Jr| vector): shipped constants
    mapped vs unmapped 8.5e-4 (M = 4) -> 7.3e-5 (M = 5), converging; the
    rotation sign flipped GLOBALLY moves both solves by 2.9e-2 and they
    still agree (9.7e-4: the flip is the device, not the map); the
    rotation sign withheld from the MAPPED weights only (a map-dependent
    gauge) puts the map 2.9e-2 off the unmapped solve, flat in M.  Bars
    (M = 4): shipped <= 3e-3; global flip visible >= 1e-2 and consistent
    <= 3e-3; map-only flip >= 1e-2."""
    cm = _tri_map()
    ok_n, ok_m = _tri_solve(None), _tri_solve(cm)
    assert float(np.abs(ok_m - ok_n).max()) <= 3e-3
    with _patched(TS, "_OOP_ROT_SIGN", 1.0):
        fl_n, fl_m = _tri_solve(None), _tri_solve(cm)
    assert float(np.abs(fl_n - ok_n).max()) >= 1e-2
    assert float(np.abs(fl_m - fl_n).max()) <= 3e-3
    with _patched(TS, "_stag_scale_weight", lambda W, s: W):
        mo_m = _tri_solve(cm)
    assert float(np.abs(mo_m - ok_n).max()) >= 1e-2


# =========================================================================== #
# 4. the QZ fallback is load-bearing for a WEAKLY lossy permeability
# =========================================================================== #
_SLAB = dict(P=0.8, WL=1.0, DEP=0.42, NSUB=1.52, NSUP=1.0)


def _delta(eps, mu, Kx, Ky):
    """Berreman matrix, psi = (Ex, Ey, Hx, Hy), d psi / d(k0 z) = i Delta
    psi (exp(-i w t)); Ez, Hz from the two longitudinal rows."""
    e, m = np.asarray(eps, complex), np.asarray(mu, complex)
    a = np.array([-e[2, 0], -e[2, 1], Ky, -Kx]) / e[2, 2]
    b = np.array([-Ky, Kx, -m[2, 0], -m[2, 1]]) / m[2, 2]
    SE = np.vstack([[1, 0, 0, 0], [0, 1, 0, 0], a])
    SH = np.vstack([[0, 0, 1, 0], [0, 0, 0, 1], b])
    eE, mH = e @ SE, m @ SH
    return np.array([Kx * a + mH[1], Ky * a - mH[0], Kx * b - eE[1],
                     Ky * b + eE[0]])


def _flux(v):
    return float(np.real(v[0] * np.conj(v[3]) - v[1] * np.conj(v[2])))


def _iso(n, Kx, Ky):
    K = np.hypot(Kx, Ky)
    kz = np.sqrt(n * n - K * K + 0j)
    cs, sn = (1.0, 0.0) if K < 1e-14 else (Kx / K, Ky / K)
    out = []
    for sgn in (1, -1):
        k = np.array([Kx, Ky, sgn * kz])
        es = np.array([-sn, cs, 0.0])
        cols = []
        for E in (es, np.cross(k, es) / n):
            H = np.cross(k, E)
            cols.append([E[0], E[1], H[0], H[1]])
        out.append(np.array(cols).T)
    return out


def _eps_mu_slab(eps, mu, theta, phi):
    """(R, T, Jr, Jt) of one (eps, mu) slab of _SLAB: the slab's eigenmodes
    with backward modes referenced at its bottom face and analytic s / p
    half-space modes (the verifier's oracle; = berreman_jones_1d at mu = I
    to 4.6e-15, = the Airy (eps, mu) slab to 4.4e-16, = the builder's expm
    oracle to 7.6e-15 -- ``v0_oracle.json``)."""
    f = _SLAB
    k0d = 2 * np.pi / f["WL"] * f["DEP"]
    Kx = f["NSUP"] * np.sin(theta) * np.cos(phi)
    Ky = f["NSUP"] * np.sin(theta) * np.sin(phi)
    q, W = np.linalg.eig(_delta(eps, mu, Kx, Ky))
    fw = np.array([(q[i].imag > 0) if abs(q[i].imag) > 1e-10
                   else (_flux(W[:, i]) > 0) for i in range(4)])
    Wf, Wb = W[:, fw], W[:, ~fw]
    U1f, U1b = _iso(f["NSUP"], Kx, Ky)
    U3f, _ = _iso(f["NSUB"], Kx, Ky)
    A = np.zeros((8, 8), complex)
    A[:4, 0:2], A[:4, 2:4] = U1b, -Wf
    A[:4, 4:6] = -Wb @ np.diag(np.exp(-1j * q[~fw] * k0d))
    A[4:, 2:4] = Wf @ np.diag(np.exp(1j * q[fw] * k0d))
    A[4:, 4:6], A[4:, 6:8] = Wb, -U3f
    R, T = np.zeros(2), np.zeros(2)
    Jr, Jt = np.zeros((2, 2), complex), np.zeros((2, 2), complex)
    for c in range(2):
        a = np.linalg.solve(U1f[:2], np.eye(2)[c])
        rhs = np.zeros(8, complex)
        rhs[:4] = -U1f @ a
        x = np.linalg.solve(A, rhs)
        vi, vr, vt = U1f @ a, U1b @ x[0:2], U3f @ x[6:8]
        Jr[:, c], Jt[:, c] = vr[:2], vt[:2]
        R[c], T[c] = -_flux(vr) / _flux(vi), _flux(vt) / _flux(vi)
    return R, T, Jr, Jt


def _slab_err(t33, mu, M, theta, phi, cmap=None, slant=None):
    f = _SLAB
    st = PMM2DStackPure(f["P"], f["P"], n_superstrate=f["NSUP"],
                        n_substrate=f["NSUB"], n_modes=M, n_orders=2,
                        cmap=cmap)
    st.add_layer(f["DEP"], eps=t33, mu=mu, slant=slant)
    st.set_source(f["WL"], theta=theta, phi=phi)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _o, R, T, J = st.solve(jones=True)
    md = st._modal
    p0 = md["p0"]
    jt = np.array([[md["tx"][0][p0], md["tx"][1][p0]],
                   [md["ty"][0][p0], md["ty"][1][p0]]])
    Rb, Tb, jr, jtb = _eps_mu_slab(t33, EYE3 if mu is None else mu, theta,
                                   phi)
    return max(float(np.abs(np.asarray(R).sum(1) - Rb).max()),
               float(np.abs(np.asarray(T).sum(1) - Tb).max()),
               float(np.abs(np.asarray(J) - jr).max()),
               float(np.abs(jt - jtb).max()))


EYE3 = _EYE


def test_ve1_the_qz_fallback_is_load_bearing_for_a_weakly_lossy_mu():
    """A symmetric permeability with loss ``Im mu ~ 1e-3``: the shipped
    structural flag (``-R`` non-Hermitian above 1e-12 relative) sends the
    region eig to QZ and the slab meets the oracle (``v7b_qz_weak.json``,
    M = 5 conical: 9.4e-10 unmapped).  With the flag forced True the
    Cholesky -- which reads only the lower triangle -- does NOT fail for a
    weak loss: it returns a SILENTLY wrong answer, 3.3e-3 (and 3.3 x the
    loss all the way down: 3.3e-5 at 1e-5, 3.4e-8 at 1e-8; a strong loss,
    >= 0.03, fails loudly in the Cholesky instead).  Bars at M = 4 conical
    (unmapped): correct <= 1e-6 (measured 3.4e-7), forced-off >= 1e-3
    (measured 3.3e-3), no exception."""
    s = 1e-3
    mu = np.array([[1.4, 0.15, 0], [0.15, 1.25, 0], [0, 0, 1.1]], complex)
    mu = mu + 1j * s * np.diag([1.0, 0.7, 0.9])
    mu[0, 1] = mu[1, 0] = 0.15 + 0.2j * s
    th, ph = np.deg2rad(22.0), np.deg2rad(63.0)
    assert _slab_err(_DIR, mu, 4, th, ph) <= 1e-6
    orig = TS.Granet2DTransverseE._assemble_oop_general

    def forced(self, *a, **k):
        r = orig(self, *a, **k)
        self._bgen_hermitian = True
        return r
    with _patched(TS.Granet2DTransverseE, "_assemble_oop_general", forced):
        assert _slab_err(_DIR, mu, 4, th, ph) >= 1e-3


# =========================================================================== #
# 5. the parity accelerator under a C2-symmetric map: refused (scope), and
#    measured CORRECT when lifted -- the integration's evidence
# =========================================================================== #
def test_ve1_parity_reduction_under_a_centred_circle_map_is_scope_not_correctness():
    """The shipped ``_stag_parity_gauge`` refuses a mapped solver.  With the
    refusal lifted (the gauge computed as for the unmapped cell) on the
    CENTRED 3 x 3 circle map -- a C2-symmetric map -- the structural test
    passes (R A R + A 3.5e-14, R B R - B 2.6e-16), the block eig is accepted
    and reproduces the dense eigenvalues to 1.8e-14 and the full stack to
    1.3e-13 at 2.3x the speed (``v6_parity_M5.json``).  An OFF-CENTRE map
    is refused by the gauge's own wall-parity test.  Bars (M = 4):
    eigenvalues <= 1e-11."""
    P = 1.0
    eps = np.broadcast_to(_EYE, (3, 3, 3, 3)).copy()
    eps[1, 1] = _DIR
    cm = CM._circle_map_3x3(P, 0.3)[0]
    s = TS.Granet2DTransverseE(P, P, cm.u_walls, cm.v_walls, 4, eps, cmap=cm)
    assert TS._stag_parity_gauge(s) is None
    s.cmap = None
    g = TS._stag_parity_gauge(s)
    s.cmap = cm
    assert g is not None
    qq = s.q * s.q
    fac = TS._stag_block_eig(s.Agen, s.Bgen, qq, g)
    assert fac is not None
    dense = TS._region_modes_oop(s, symmetry=False)
    lam = np.concatenate([dense[2], dense[5]])
    q_red = fac[0]
    lam_red = -1j * q_red
    d = np.abs(lam_red[:, None] - lam[None, :]).min(axis=1).max()
    assert float(d / np.abs(lam).max()) <= 1e-11
    cmo = CM._circle_map_3x3(P, 0.27, center=(0.41, 0.56))[0]
    so = TS.Granet2DTransverseE(P, P, cmo.u_walls, cmo.v_walls, 4, eps,
                                cmap=cmo)
    so.cmap = None
    assert TS._stag_parity_gauge(so) is None
