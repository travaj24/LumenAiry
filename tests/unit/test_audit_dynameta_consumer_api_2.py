"""AUDIT_DYNAMETA_CONSUMER_API_GAPS_2026_07_13 -- remainder campaign gates
(B-PMM2D, C1, C2, C3, D1; the A1/A2/B-PMMStack gates live in
``test_audit_dynameta_consumer_api.py``).

B (PMM2D leg) -- ``per_order_amplitudes`` / ``jones_transmission`` on
``PMM2DStackHybrid`` + ``PMM2DStackPure`` via the shared
``PerOrderAmplitudesMixin``: per-order COMPLEX-amplitude parity against RCWA
on an identical crossed-pillar cell, exact flux-recipe closure, public
decaying-branch ``kz``.

C1 -- ``BORStack.per_mode_amplitudes`` (deterministic pinned eigenvector
gauge; the diagonal is gauge-invariant) + ``BORStack.layer_absorption``
(z-flux difference on the staggered two-grid quadrature).  The phase gates
pin the fundamental mode's COMPLEX reflection against the analytic Fresnel /
Fabry-Perot values at the mode's own local angle -- machine-precision
matches, not fits.

C2 -- Berreman OOP-tensor-at-oblique ``retain_internal``: the generalized
(Li 2003) cascade now retains the same internals shape as the native core
(asymmetric modes sliced from the M blocks, generalized partial cascades,
public-gauge conj + modal-H sign map), so ``internal_field`` /
``layer_absorption`` serve the flagship tilted-director regime.  Gates:
closed absorption budget at oblique AND conical (machine precision),
lossless zero, theta -> 0 continuity against the NATIVE path for both the
absorption and all six field components.

C3 -- ``PMM2DStackPure.solve(retain_internal=True)`` + ``layer_absorption``
via the staggered block-field-Gram flux (probe-pinned ``Re`` pairing).

D1 -- RCWAStack JAX path: traced uniform ``eps=`` and traced
wavelength/theta/phi (set_source) now flow gradients; forward parity with
the concrete solve and AD-vs-FD on all three.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pytest

from lumenairy.elements.berreman import BerremanStack
from lumenairy.elements.pmm.stack2d import PMM2DStackHybrid
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure
from lumenairy.elements.rcwa import RCWAStack

_C = complex
_WL = 1.55e-6


# ======================================================================== #
# B (PMM2D leg): per-order amplitudes for Hybrid + Pure                    #
# ======================================================================== #

_P2 = 1.0e-6
_DEPTH2 = 0.3e-6
_NSUB2 = 1.45
_EPS_PILL, _EPS_BG = 6.0 + 0j, 2.0 + 0j


def _cell4(ep=_EPS_PILL):
    c = np.full((4, 4), _EPS_BG, dtype=_C)
    c[1:3, 1:3] = ep
    return c


def _rcwa2d(theta, phi, S=512):
    s = RCWAStack(_P2, period_y=_P2, n_substrate=_NSUB2, n_orders=11)
    x = (np.arange(S) + 0.5) / S
    X, Y = np.meshgrid(x, x, indexing="ij")
    inside = (np.abs(X - 0.5) < 0.25) & (np.abs(Y - 0.5) < 0.25)
    s.add_layer(_DEPTH2, eps_cell=np.where(inside, _EPS_PILL, _EPS_BG))
    s.set_source(_WL, theta=theta, phi=phi)
    return s.solve()


def _common2d(po_p, po_r):
    op, orc = np.asarray(po_p["orders"]), np.asarray(po_r["orders"])
    kp = {tuple(mn): i for i, mn in enumerate(op.tolist())}
    kr = {tuple(mn): i for i, mn in enumerate(orc.tolist())}
    common = sorted(set(kp) & set(kr))
    return (np.array([kp[c] for c in common]),
            np.array([kr[c] for c in common]))


def _assert_amp_parity_2d(sp, res_r, tol):
    for port in ("reflection", "transmission"):
        pp = sp.per_order_amplitudes(port)
        pr = res_r.per_order_amplitudes(port)
        ip, ir = _common2d(pp, pr)
        assert np.max(np.abs(pp["kx"][ip] - pr["kx"][ir])) < 1e-12
        assert np.max(np.abs(pp["kz"][ip] - pr["kz"][ir])) < 1e-9
        prop = np.real(pp["kz"][ip]) > 1e-9
        assert np.any(prop)
        for key in ("Ex", "Ey"):
            d = np.abs(pp[key][:, ip] - pr[key][:, ir])[:, prop]
            assert np.max(d) < tol, f"{port}/{key}: {np.max(d):.3e}"


def _assert_flux_closure_2d(sp, R_eff, T_eff, tol=1e-12):
    pr = sp.per_order_amplitudes("reflection")
    p0 = int(np.where((pr["orders"][:, 0] == 0)
                      & (pr["orders"][:, 1] == 0))[0][0])
    kz_inc = float(np.real(pr["kz"][p0]))
    kx0 = float(np.real(pr["kx"][p0]))
    ky0 = float(np.real(pr["ky"][p0]))
    for port, eff in (("reflection", R_eff), ("transmission", T_eff)):
        po = sp.per_order_amplitudes(port)
        kz = po["kz"]
        safe = np.where(np.abs(kz) < 1e-12, 1.0, kz)
        out = np.zeros_like(np.asarray(eff))
        for col, (ex0, ey0) in enumerate(((1, 0), (0, 1))):
            einc = 1.0 + ((kx0 * ex0 + ky0 * ey0) / kz_inc) ** 2
            Ex, Ey = po["Ex"][col], po["Ey"][col]
            Ez = -(po["kx"] * Ex + po["ky"] * Ey) / safe
            e = np.real(kz / kz_inc) * (np.abs(Ex) ** 2 + np.abs(Ey) ** 2
                                        + np.abs(Ez) ** 2) / einc
            out[col] = np.where(np.real(kz) > 0, np.real(e), 0.0)
        assert np.max(np.abs(out - np.asarray(eff))) < tol


@pytest.mark.slow      # 2-D eig-heavy (CI fast-gate budget)
def test_b2_hybrid_amplitudes_vs_rcwa_conical():
    th, ph = np.deg2rad(20.0), np.deg2rad(30.0)
    sh = PMM2DStackHybrid(_P2, n_substrate=_NSUB2, degree=9, n_orders=7)
    sh.add_layer(_DEPTH2, eps_cell=_cell4())
    sh.set_source(_WL, theta=th, phi=ph)
    _o, R, T, Jr = sh.solve()
    rr = _rcwa2d(th, ph)
    # cross-engine convergence gap ~3e-3 at these settings (probe); 2e-2 bar
    _assert_amp_parity_2d(sh, rr, tol=2e-2)
    _assert_flux_closure_2d(sh, R, T)
    assert np.max(np.abs(sh.jones_transmission()
                         - rr.jones_transmission())) < 2e-2


@pytest.mark.slow      # 2-D eig-heavy (CI fast-gate budget)
def test_b2_pure_amplitudes_vs_rcwa_conical():
    th, ph = np.deg2rad(20.0), np.deg2rad(30.0)
    sp = PMM2DStackPure(_P2, n_substrate=_NSUB2, n_modes=9, n_orders=5)
    sp.add_layer(_DEPTH2, eps_cell=_cell4())
    sp.set_source(_WL, theta=th, phi=ph)
    _o, R, T, _j = sp.solve()
    rr = _rcwa2d(th, ph)
    # 5e-2 bar: cross-engine CONVERGENCE gap in the weak higher orders,
    # dominated by the RCWA REFERENCE (probe: the gap is flat in Pure's M
    # from 8 -> 13 at 1.91e-2..1.92e-2 but shrinks with the reference's
    # n_orders, 11 -> 15 gives 1.5e-2 -- RCWA converging toward the
    # no-floor answer, the established pattern on patterned cells); a
    # gauge/sign/contract error is O(0.1-1) and still fails loudly.  The
    # order-0 Jones (below, 3e-4 measured) and the EXACT flux closure pin
    # the rest.
    _assert_amp_parity_2d(sp, rr, tol=5e-2)
    _assert_flux_closure_2d(sp, R, T)
    assert np.max(np.abs(sp.jones_transmission()
                         - rr.jones_transmission())) < 1e-2


@pytest.mark.slow      # 2-D eig-heavy (CI fast-gate budget)
def test_b2_evanescent_kz_public_branch_and_contract():
    sp = PMM2DStackPure(_P2, n_substrate=_NSUB2, n_modes=8, n_orders=5)
    sp.add_layer(_DEPTH2, eps_cell=_cell4())
    with pytest.raises(ValueError, match="no per-order amplitudes"):
        sp.per_order_amplitudes()
    sp.set_source(_WL, theta=0.2, phi=0.4)
    sp.solve()
    for port in ("reflection", "transmission"):
        kz = sp.per_order_amplitudes(port)["kz"]
        ev = np.abs(np.real(kz)) < 1e-9
        assert np.any(ev)
        assert np.min(np.imag(kz[ev])) > 0.0
    sp.set_source(_WL, theta=0.1)          # re-source invalidates
    with pytest.raises(ValueError, match="no per-order amplitudes"):
        sp.jones_transmission()


# ======================================================================== #
# C1: BORStack per_mode_amplitudes + layer_absorption                      #
# ======================================================================== #

_BOR = pytest.importorskip("lumenairy.elements.bor")
BORStack = _BOR.BORStack
_K0 = 2.0 * np.pi          # lam = 1
_RB, _NB = 8.0, 160        # Rbig = 8 lam


@pytest.mark.slow
def test_c1_bor_absorption_budget_and_split():
    """Lossy ring layer: R + T + sum A = 1 (machine precision); identically
    ~0 per layer for the lossless twin; splitting the layer in two conserves
    R and the total A."""
    def _stack(splits, ring_n):
        st = BORStack(Rbig=_RB, m=1, N=_NB, n_superstrate=1.41,
                      n_substrate=1.41)
        for d in splits:
            st.add_layer(d, rings=(1.5, 0.5, ring_n, 1.41))
        st.set_source(k0=_K0)
        res = st.solve(retain_internal=True)
        return res, st.layer_absorption()

    res0, A0 = _stack([0.5], 2.45 + 0j)
    assert np.abs(A0).max() < 1e-10
    assert np.abs(res0["R"] + res0["T"] + A0.sum(0) - 1.0).max() < 1e-9

    res1, A1 = _stack([0.5], 2.45 + 0.15j)
    assert np.abs(res1["R"] + res1["T"] + A1.sum(0) - 1.0).max() < 1e-9
    assert A1.max() > 0.05                      # real absorption happened

    res2, A2 = _stack([0.25, 0.25], 2.45 + 0.15j)
    assert np.abs(res1["R"] - res2["R"]).max() < 1e-12
    assert np.abs(A1.sum(0) - A2.sum(0)).max() < 1e-12


@pytest.mark.slow
def test_c1_bor_pinned_gauge_fresnel_phase():
    """The pinned-gauge fundamental-mode COMPLEX reflection matches the
    analytic transverse-E Fresnel coefficient at the mode's own local angle
    (including the spacer round-trip phase) -- the machine-precision phase
    oracle (probe: 5e-16)."""
    n1, n2, d = 1.0, 2.25, 0.4
    st = BORStack(Rbig=_RB, m=1, N=_NB, n_superstrate=n1, n_substrate=n2)
    st.add_layer(d, eps=n1 ** 2)               # index-matched spacer
    st.set_source(k0=_K0)
    res = st.solve()
    amps = st.per_mode_amplitudes("reflection")
    q = np.real(amps["q_inc"])
    jf = int(np.argmax(q))
    th1 = np.arcsin(np.sqrt(max(n1 ** 2 * _K0 ** 2 - q[jf] ** 2, 0.0))
                    / (n1 * _K0))
    th2 = np.arcsin(n1 * np.sin(th1) / n2)
    rte = ((n1 * np.cos(th1) - n2 * np.cos(th2))
           / (n1 * np.cos(th1) + n2 * np.cos(th2)))
    r_ana = rte * np.exp(2j * q[jf] * d)
    assert abs(amps["amplitude"][jf, jf] - r_ana) < 1e-10
    # row sums reproduce the solve's R / T exactly
    assert np.abs(np.sum(np.abs(amps["amplitude"]) ** 2, axis=0)
                  - res["R"]).max() < 1e-12
    at = st.per_mode_amplitudes("transmission")
    assert np.abs(np.sum(np.abs(at["amplitude"]) ** 2, axis=0)
                  - res["T"]).max() < 1e-12


@pytest.mark.slow
def test_c1_bor_pinned_gauge_fabry_perot_phase():
    """Complex Fabry-Perot reflection of a uniform slab, fundamental mode --
    the multi-interface phase oracle (probe: 1.7e-14)."""
    n_s, d = 2.0, 0.35
    st = BORStack(Rbig=_RB, m=1, N=_NB, n_superstrate=1.0, n_substrate=1.0)
    st.add_layer(d, eps=n_s ** 2)
    st.set_source(k0=_K0)
    st.solve()
    af = st.per_mode_amplitudes("reflection")
    q = np.real(af["q_inc"])
    jf = int(np.argmax(q))
    th = np.arcsin(np.sqrt(max(_K0 ** 2 - q[jf] ** 2, 0.0)) / _K0)
    ths = np.arcsin(np.sin(th) / n_s)
    r12 = ((np.cos(th) - n_s * np.cos(ths))
           / (np.cos(th) + n_s * np.cos(ths)))
    beta = n_s * _K0 * d * np.cos(ths)
    r_fp = (r12 - r12 * np.exp(2j * beta)) / (1 - r12 ** 2 * np.exp(2j * beta))
    assert abs(af["amplitude"][jf, jf] - r_fp) < 1e-10
    # determinism: a fresh identical stack pins the same amplitudes exactly
    st2 = BORStack(Rbig=_RB, m=1, N=_NB, n_superstrate=1.0, n_substrate=1.0)
    st2.add_layer(d, eps=n_s ** 2)
    st2.set_source(k0=_K0)
    st2.solve()
    af2 = st2.per_mode_amplitudes("reflection")
    assert np.array_equal(af["amplitude"], af2["amplitude"])


# ======================================================================== #
# C3: PMM2DStackPure layer_absorption                                      #
# ======================================================================== #

@pytest.mark.slow      # 2-D eig-heavy (CI fast-gate budget)
def test_c3_pure_absorption_budget_and_hybrid_crossgate():
    eps_lossy = 6.0 + 0.8j
    sp = PMM2DStackPure(_P2, n_substrate=_NSUB2, n_modes=8, n_orders=5)
    sp.add_layer(_DEPTH2, eps_cell=_cell4(eps_lossy))
    sp.add_layer(0.2e-6, eps=2.25 + 0.1j)
    sp.set_source(_WL, theta=0.15, phi=0.3)
    _o, R, T, _j = sp.solve(retain_internal=True)
    A = sp.layer_absorption()
    assert np.abs(R.sum(1) + T.sum(1) + A.sum(0) - 1.0).max() < 1e-10
    assert A.min() > 0.0 and A.max() > 0.05
    sh = PMM2DStackHybrid(_P2, n_substrate=_NSUB2, degree=11, n_orders=9)
    sh.add_layer(_DEPTH2, eps_cell=_cell4(eps_lossy))
    sh.add_layer(0.2e-6, eps=2.25 + 0.1j)
    sh.set_source(_WL, theta=0.15, phi=0.3)
    sh.solve(retain_internal=True)
    Ah = sh.layer_absorption()
    assert np.abs(A - Ah).max() < 5e-2          # cross-engine (probe 6.7e-3)


@pytest.mark.slow      # 2-D eig-heavy (CI fast-gate budget)
def test_c3_pure_lossless_zero_and_contract():
    sp = PMM2DStackPure(_P2, n_substrate=_NSUB2, n_modes=8, n_orders=5)
    sp.add_layer(_DEPTH2, eps_cell=_cell4(6.0 + 0j))
    sp.set_source(_WL, theta=0.15, phi=0.3)
    with pytest.raises(ValueError, match="retain_internal"):
        sp.layer_absorption()                  # not retained yet
    sp.solve(retain_internal=True)
    A = sp.layer_absorption()
    assert np.abs(A).max() < 1e-10
    sp.solve()                                 # plain re-solve invalidates
    with pytest.raises(ValueError, match="retain_internal"):
        sp.layer_absorption()


# ======================================================================== #
# C2: Berreman OOP-tensor-at-oblique internals                             #
# ======================================================================== #

def _tilted(tilt_deg, loss=0.0):
    no2, ne2 = (1.5 ** 2 + loss * 1j), (1.7 ** 2 + loss * 1j)
    th = np.deg2rad(tilt_deg)
    c, s = np.cos(th), np.sin(th)
    Rm = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return Rm @ np.diag([ne2, no2, no2]) @ Rm.T


def _director_stack(loss, ang, phi=0.0):
    st = BerremanStack(n_substrate=1.45)
    st.add_layer(400e-9, eps=_tilted(35.0, loss))
    st.add_layer(150e-9, eps=2.25 + (0.5 * loss) * 1j)
    st.add_layer(300e-9, eps=_tilted(-20.0, loss))
    st.set_source(_WL, angle=ang, phi=phi)
    return st


@pytest.mark.parametrize("phi", [0.0, 0.6], ids=["oblique", "conical"])
def test_c2_oop_oblique_absorption_budget(phi):
    """The audit's acceptance gate: the DynaMeta tilted-director reproducer
    flips from asserting the warn to asserting a CLOSED absorption budget --
    sum A_i == 1 - R - T at machine precision (probe: 9e-16)."""
    st = _director_stack(0.08, 0.45, phi)
    R, T, _Jr = st.solve(retain_internal=True)
    A = st.layer_absorption()
    assert np.abs(A.sum(axis=0) + R + T - 1.0).max() < 1e-12
    assert A.min() > 0.0
    # lossless twin: absorption identically zero
    st0 = _director_stack(0.0, 0.45, phi)
    R0, T0, _ = st0.solve(retain_internal=True)
    assert np.abs(st0.layer_absorption()).max() < 1e-12
    assert np.abs(R0 + T0 - 1.0).max() < 1e-12


def test_c2_theta_to_zero_continuity_vs_native():
    """The generalized retention must agree with the NATIVE path in the
    theta -> 0 limit -- absorption AND all six field components (two fully
    independent code paths; probe: dA 1.8e-7, dF 4.9e-5 at theta = 1e-4)."""
    st_n = _director_stack(0.08, 0.0)         # native serves OOP at normal
    st_n.solve(retain_internal=True)
    st_g = _director_stack(0.08, 1e-4)        # generalized path
    st_g.solve(retain_internal=True)
    assert np.abs(st_n.layer_absorption()
                  - st_g.layer_absorption()).max() < 1e-5
    fn = st_n.internal_field(250e-9, incident=(1.0, 0.0))
    fg = st_g.internal_field(250e-9, incident=(1.0, 0.0))
    for c in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        assert abs(fn[c] - fg[c]) < 1e-3, f"{c}: {abs(fn[c] - fg[c]):.3e}"


# ======================================================================== #
# D1: RCWAStack traced uniform eps + traced source                         #
# ======================================================================== #

def _d1_solve(eps_u, wl, theta):
    S = 128
    x = (np.arange(S) + 0.5) / S
    cell = np.where(x < 0.5, 4.0 + 0j, 1.0 + 0j)[:, None]
    s = RCWAStack(1.0e-6, n_substrate=1.45, n_orders=7)
    s.add_layer(0.20e-6, eps=eps_u)
    s.add_layer(0.30e-6, eps_cell=cell)
    s.set_source(wl, theta=theta)
    _o, R, T = s.solve().efficiencies()
    return R, T


def test_d1_traced_uniform_eps_and_source():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    R_c, T_c = _d1_solve(2.25 + 0.01j, _WL, 0.3)
    # forward parity: traced uniform eps
    R_j, T_j = _d1_solve(jnp.asarray(2.25 + 0.01j), _WL, 0.3)
    assert np.abs(np.asarray(R_j) - R_c).max() < 1e-12
    # forward parity: traced wavelength + theta
    R_s, T_s = _d1_solve(2.25 + 0.01j, jnp.asarray(_WL), jnp.asarray(0.3))
    assert np.abs(np.asarray(R_s) - R_c).max() < 1e-12
    assert np.abs(np.asarray(T_s) - T_c).max() < 1e-12

    # AD vs FD: d(sum R)/d(eps_u), d/d(wl), d/d(theta)
    def loss_eps(e):
        return jnp.sum(_d1_solve(e + 0.01j, _WL, 0.3)[0])

    def loss_wl(w):
        return jnp.sum(_d1_solve(2.25 + 0.01j, w, 0.3)[0])

    def loss_th(t):
        return jnp.sum(_d1_solve(2.25 + 0.01j, _WL, t)[0])

    g = float(jax.grad(loss_eps)(jnp.asarray(2.25)))
    h = 1e-6
    fd = float((np.sum(np.asarray(_d1_solve(2.25 + h + 0.01j, _WL, 0.3)[0]))
                - np.sum(np.asarray(_d1_solve(2.25 - h + 0.01j, _WL,
                                              0.3)[0]))) / (2 * h))
    assert abs(g - fd) < 1e-5 * max(abs(fd), 1.0)

    g = float(jax.grad(loss_wl)(jnp.asarray(_WL)))
    h = 1e-13
    fd = float((np.sum(np.asarray(_d1_solve(2.25 + 0.01j, _WL + h, 0.3)[0]))
                - np.sum(np.asarray(_d1_solve(2.25 + 0.01j, _WL - h,
                                              0.3)[0]))) / (2 * h))
    assert abs(g - fd) < 1e-4 * max(abs(fd), 1.0)

    g = float(jax.grad(loss_th)(jnp.asarray(0.3)))
    h = 1e-7
    fd = float((np.sum(np.asarray(_d1_solve(2.25 + 0.01j, _WL, 0.3 + h)[0]))
                - np.sum(np.asarray(_d1_solve(2.25 + 0.01j, _WL,
                                              0.3 - h)[0]))) / (2 * h))
    assert abs(g - fd) < 1e-5 * max(abs(fd), 1.0)
