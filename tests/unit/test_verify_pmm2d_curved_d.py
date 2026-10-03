"""INDEPENDENT VERIFIER of curved-cell Phase D (anisotropic and magnetic
materials inside curved cells of the pure staggered 2-D PMM) -- decision
tests that close the gaps the verifier's mutation matrix found in
``tests/unit/test_pmm2d_staggered_curved_d.py``
(``docs/audits/VERIFY_PMM2D_CURVED_D_2026_10_03.md``, probes and JSON in
``validation/probe_pmm2d_curved/verify_d/``).

Words.  Under a coordinate map with Jacobian ``J`` (``sg = det J``) a
block-form permeability enters the solver as ``chi_t = J^T mu_t^-1 J / sg``
and ``chi_33 = 1 / (sg mu_33)`` (``_stag_map_eff_tensor``).  Every Phase D
unit gate uses a DIAGONAL ``mu``, for which ``mu^T = mu``: a transposed
``chi`` (the index order of the inverse permeability) cannot be seen, and
every magnetic gate runs at NORMAL incidence, where ``chi_33`` (the
``H_z`` weight) never enters.  A GYROTROPIC ``mu`` under a SHEARED map
(non-diagonal ``J``) at normal and conical incidence, against an
independent eps+mu Berreman 4x4 written here, decides both.

Every bar was measured by the verifier on 2026-10-03 (Windows 11, CPython
3.14, OMP/OPENBLAS/MKL = 1) and is stated next to its assertion with the
gap on both sides.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import re  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy.linalg import expm  # noqa: E402

from lumenairy.elements.pmm import PMM2DStackPure  # noqa: E402
from lumenairy.elements.pmm import _curvemap as CM  # noqa: E402
from lumenairy.elements.pmm import twod_staggered as TS  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_G3 = dict(P=0.40e-6, WL=1.0e-6, DEP=0.55e-6, NSUB=1.5, NSUP=1.0)
#: gyrotropic (Hermitian) permeability: mu != mu^T
_MU_GYRO = np.array([[1.5, 0.35j, 0], [-0.35j, 1.4, 0], [0, 0, 1.25]],
                    dtype=complex)


# --------------------------------------------------------------------------- #
# an independent eps + mu Berreman 4x4 (equal to the shipped
# berreman_jones_1d to 6.1e-15 on eps-only stacks incl. the complex
# reflection Jones, normal / oblique / conical, eight tensor classes --
# validation/probe_pmm2d_curved/verify_d/v0_oracle.json)
# --------------------------------------------------------------------------- #
def _t3(a):
    a = np.asarray(a, complex)
    return a * np.eye(3) if a.ndim == 0 else a


def _delta(eps, mu, Kx, Ky):
    eps, mu = _t3(eps), _t3(mu)
    D = np.zeros((4, 4), complex)
    for c in range(4):
        Ex, Ey, hx, hy = np.eye(4)[c]
        hz = (Kx * Ey - Ky * Ex - mu[2, 0] * hx - mu[2, 1] * hy) / mu[2, 2]
        Ez = (-(Kx * hy - Ky * hx) - eps[2, 0] * Ex
              - eps[2, 1] * Ey) / eps[2, 2]
        mh = mu @ np.array([hx, hy, hz])
        eE = eps @ np.array([Ex, Ey, Ez])
        D[:, c] = [Kx * Ez + mh[1], Ky * Ez - mh[0],
                   Kx * hz - eE[1], Ky * hz + eE[0]]
    return D


def _halfspace(n, Kx, Ky):
    w, v = np.linalg.eig(_delta(n * n, 1.0, Kx, Ky))
    fwd = (w.real > 1e-12) | ((np.abs(w.real) <= 1e-12) & (w.imag > 0))
    F, B = v[:, fwd], v[:, ~fwd]
    return F @ np.linalg.inv(F[:2]), B @ np.linalg.inv(B[:2])


def _sz(p):
    return float(np.real(p[0] * np.conj(p[3]) - p[1] * np.conj(p[2])))


def _berreman(eps, mu, d, n_sub, n_sup, wl, theta, phi):
    """(R, T, r): per lab input (E_x, E_y), and the reflection Jones."""
    k0 = 2 * np.pi / wl
    Kx = n_sup * np.sin(theta) * np.cos(phi)
    Ky = n_sup * np.sin(theta) * np.sin(phi)
    Fs, Bs = _halfspace(n_sup, Kx, Ky)
    Ft, _ = _halfspace(n_sub, Kx, Ky)
    Mt = expm(1j * k0 * _delta(eps, mu, Kx, Ky) * d)
    X = np.linalg.solve(np.hstack([Ft, -Mt @ Bs]), Mt @ Fs)
    t, r = X[:2], X[2:]
    R = np.array([-_sz(Bs @ r[:, j]) / _sz(Fs[:, j]) for j in (0, 1)])
    T = np.array([_sz(Ft @ t[:, j]) / _sz(Fs[:, j]) for j in (0, 1)])
    return R, T, r


def _shear4(Pp):
    """4 x 4 TransfiniteMap, straight edges, nine interior vertices moved by
    up to 7 % of the period (bilinear cells, NON-diagonal J)."""
    w = np.linspace(0.0, Pp, 5)
    V = np.stack(np.meshgrid(w, w, indexing="ij"), axis=-1)
    rng = np.random.default_rng(20261003)
    for i in range(1, 4):
        for j in range(1, 4):
            V[i, j] += rng.uniform(-0.07, 0.07, 2) * Pp
    return CM.TransfiniteMap(w, w, V, None)


def _mag_film(M, theta, phi):
    """eps 2 + gyrotropic mu film under the 4 x 4 shear: (dRT, dJ)."""
    f = _G3
    st = PMM2DStackPure(f["P"], f["P"], n_superstrate=f["NSUP"],
                        n_substrate=f["NSUB"], n_modes=M, n_orders=2,
                        cmap=_shear4(f["P"]))
    st.add_layer(f["DEP"], eps=2.0, mu=_MU_GYRO)
    st.set_source(f["WL"], theta=theta, phi=phi)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _o, R, T, J = st.solve(jones=True)
    Rb, Tb, rb = _berreman(2.0, _MU_GYRO, f["DEP"], f["NSUB"], f["NSUP"],
                           f["WL"], theta, phi)
    dRT = max(float(np.abs(np.asarray(R).sum(1) - Rb).max()),
              float(np.abs(np.asarray(T).sum(1) - Tb).max()))
    return dRT, float(np.abs(np.asarray(J) - rb).max())


def _patch_kernel(fn):
    orig = TS._stag_map_eff_tensor

    class _Ctx:
        def __enter__(self):
            TS._stag_map_eff_tensor = lambda *a: fn(orig, *a)

        def __exit__(self, *exc):
            TS._stag_map_eff_tensor = orig
    return _Ctx()


def _chi_transposed(orig, eps, mu, xu, xv, yu, yv):
    return orig(eps, None if mu is None else np.swapaxes(mu, -1, -2),
                xu, xv, yu, yv)


def _chi33_no_sg(orig, eps, mu, xu, xv, yu, yv):
    out = orig(eps, mu, xu, xv, yu, yv)
    if mu is not None:
        out["c33"] = out["c33"] * (xu * yv - xv * yu)
    return out


def _chi_sides_swapped(orig, eps, mu, xu, xv, yu, yv):
    out = orig(eps, mu, xu, xv, yu, yv)
    if mu is not None:
        bad = orig(eps, mu, xu, yu, xv, yv)          # J mu^-1 J^T / sg
        out.update({k: bad[k] for k in ("c11", "c12", "c21", "c22")})
    return out


# =========================================================================== #
# V-D1: the index order of chi_t, and chi_33, for a GYROTROPIC mu
# =========================================================================== #
def test_verify_d_gyrotropic_mu_under_a_shear_matches_an_independent_berreman():
    """A uniform eps-2 film with a GYROTROPIC permeability under a 4 x 4
    sheared map (bilinear cells: the covariant plane-wave field is exact) vs
    the eps+mu Berreman above.  Measured 2026-10-03 (``v2_summary.json``,
    ``v_unit_readings.json``): NORMAL incidence M = 4: R / T 2.1e-15 and the
    complex Jones 7.8e-15 (bar 1e-10: round-off, 4 decades of margin);
    CONICAL (20 deg, 40 deg) M = 4: 2.7e-10 / 3.2e-10 (bar 1e-8, 1.5
    decades; the bilinear map is exact at normal incidence and spectral
    off it -- 2.9e-14 / 8.5e-14 at M = 5).

    Fail-before arms, each a one-token bug in the magnetic branch of
    ``_stag_map_eff_tensor`` that EVERY Phase D unit gate passes (they use a
    diagonal mu, and normal incidence for every magnetic gate):
    * mu TRANSPOSED in chi (``J^T mu^-T J``): Jones 0.141 at normal
      incidence while R / T stays 5.4e-15 (blind), bar >= 1e-2;
    * chi_33 without sqrt(g) (``1 / mu_33``): blind at normal incidence
      (7.9e-15, asserted <= 1e-10), conical M = 4 Jones 2.0e-4 / R,T 1.3e-5
      (bar >= 1e-6, 2.3 decades under the Jones reading);
    * the two sides of chi swapped (``J mu^-1 J^T``): R / T 6.2e-4, Jones
      8.3e-3 at normal incidence (bar >= 1e-4)."""
    nRT, nJ = _mag_film(4, 0.0, 0.0)
    assert nRT <= 1e-10 and nJ <= 1e-10, (nRT, nJ)
    th, ph = np.deg2rad(20.0), np.deg2rad(40.0)
    cRT, cJ = _mag_film(4, th, ph)
    assert cRT <= 1e-8 and cJ <= 1e-8, (cRT, cJ)
    with _patch_kernel(_chi_transposed):
        tRT, tJ = _mag_film(4, 0.0, 0.0)
    assert tJ >= 1e-2, tJ
    with _patch_kernel(_chi33_no_sg):
        zn = _mag_film(4, 0.0, 0.0)
        zc = _mag_film(4, th, ph)
    assert max(zn) <= 1e-10, zn                 # blind at normal incidence
    assert max(zc) >= 1e-6, zc
    with _patch_kernel(_chi_sides_swapped):
        sRT, sJ = _mag_film(4, 0.0, 0.0)
    assert max(sRT, sJ) >= 1e-4, (sRT, sJ)


# =========================================================================== #
# V-D2: the Phase B verifier's D-1 check keeps its 1e-6 bar
# =========================================================================== #
class _SlightlyBadSinusoid(CM.Sinusoid):
    def __call__(self, s):
        v, d = super().__call__(s)
        d = d.copy()
        d[:, 0] *= 1.0 + 3e-5                 # a 3e-5 derivative bug
        return v, d


def _sine_edges(cls):
    P = 1.2
    vw = np.array([0.0, 0.36, 0.9, P])
    uw = np.array([0.0, 0.3, 0.9, P])
    V = np.stack(np.meshgrid(uw, vw, indexing="ij"), axis=-1).astype(float)
    k = 2 * np.pi / P
    for i in (1, 2):
        V[i, :, 0] += 0.1 * np.sin(k * vw)
    ed = {("v", i, j): cls(uw[i], 0.1, P, vw[j], vw[j + 1])
          for i in (1, 2) for j in range(3)}
    return uw, vw, V, ed


def test_verify_d_curve_derivative_check_refuses_a_small_derivative_bug():
    """``TransfiniteMap`` refuses an edge curve whose analytic derivative is
    not d/ds of its value at ``1e-6 x max(period, |d|)``.  The only existing
    gate (``test_verify_b_inconsistent_curve_derivative_is_refused``) uses a
    10 % bug, so the bar could be loosened 10^4-fold unseen (the verifier's
    mutation 'd1_tol100' -- the bar x 100 -- passed every existing test).
    Here a 3e-5 derivative bug must be refused (the check's central
    difference is good to ~1e-10 relative at h = 1e-6, so 3e-5 sits 1.5
    decades above the bar and 0.5 decades below the loosened one), and the
    correct curve must be accepted."""
    CM.TransfiniteMap(*_sine_edges(CM.Sinusoid))
    with pytest.raises(ValueError, match="derivative"):
        CM.TransfiniteMap(*_sine_edges(_SlightlyBadSinusoid))


# =========================================================================== #
# V-D3: the walker exemption of material_key keeps its stated reason
# =========================================================================== #
def test_verify_d_material_key_walker_exemption_states_its_reason():
    """The ``__all__`` walker's exemption registry is a frozenset of
    ``(module, name)`` tuples: the REASON for an exemption lives only in the
    comment above it, so deleting it is invisible to every test (the
    verifier's mutation 'walker_reason').  This pins that the Phase D entry
    for ``stack2d_pure.material_key`` keeps a comment block that names the
    helper and why it is submodule-level (a viewer palette helper)."""
    with open(os.path.join(_HERE, "test_v4_16_0_walker_all_symmetry.py"),
              encoding="utf-8") as fh:
        src = fh.read()
    m = re.search(r"((?:[ \t]*#[^\n]*\n)+)[ \t]*\('lumenairy\.elements\.pmm\."
                  r"stack2d_pure', 'material_key'\)", src)
    assert m is not None, "material_key exemption lost its comment block"
    block = m.group(1)
    assert "material_key" in block and "viewer" in block, block
    assert len(block.split()) >= 30, block


# =========================================================================== #
# V-D4: F-D3 -- geom[4] of the mapped geometric cache is the INVERSE PLAIN
# block Gram, the one object both of its consumers need
# =========================================================================== #
def test_verify_d_geom_slot_4_is_the_inverse_plain_gram_for_both_consumers():
    """``_homog_geom_cache(solver)[4]`` feeds TWO consumers: the half-spaces'
    Eq.-25 H partner (``_homog_region_modes``) and the mapped incident
    decomposition ``C = W0^-1 G^-1 b`` (``_stag_incident_coeffs_mapped``).
    Both need the SAME operator -- the inverse of the PLAIN (u, v) L2 block
    Gram -- and under a map it is NOT ``inv(-R)`` (every mapped region is
    magnetic: ``-R`` carries chi_t).  This pins the identity of the slot
    (so an edit that swaps it, or reorders the tuple, fails HERE, naming
    both consumers), on the 3 x 3 circle map at M = 4: ``geom[4] @
    blockdiag(G1, G2) = I`` to <= 1e-10 and ``geom[4]``
    differs from ``inv(-R)`` by >= 1e-2 relative (measured 0.40;
    ``v_unit_readings.json``: inverse residual 4.7e-14); with no map the two
    coincide (<= 1e-10)."""
    P = 1.2
    cm = CM._circle_map_3x3(P, 0.36)[0]
    s = TS.Granet2DTransverseE(P, P, cm.u_walls, cm.v_walls, 4,
                               np.ones((3, 3), complex), cmap=cm)
    geom = TS._homog_geom_cache(s)
    assert len(geom) == 6 and geom[5] == s.q * s.q
    G1, G2 = s.Ggram_blocks
    qq = s.q * s.q
    Gp = np.zeros((2 * qq, 2 * qq), complex)
    Gp[:qq, :qq], Gp[qq:, qq:] = G1, G2
    assert np.abs(geom[4] @ Gp - np.eye(2 * qq)).max() <= 1e-10
    iR = np.linalg.inv(-s.Rmat)
    rel = np.abs(geom[4] - iR).max() / np.abs(iR).max()
    assert rel >= 1e-2, rel
    s0 = TS.Granet2DTransverseE(P, P, 3, 3, 4, np.ones((3, 3), complex))
    g0 = TS._homog_geom_cache(s0)
    assert np.abs(g0[4] - np.linalg.inv(-s0.Rmat)).max() <= 1e-10
