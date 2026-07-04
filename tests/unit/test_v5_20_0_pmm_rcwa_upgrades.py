"""PMM/RCWA upgrade program (PMM_RCWA_AUDIT_2026_07_02 + conical audit).

Phase-1 defect regressions (B3 angle reuse, B4/B5 covered by existing suites +
the parity harness) and the bit-exact perf ports (F1/P2 covered by the parity
harness).  Later phases add their own gate files.
"""
import numpy as np
import pytest

import lumenairy as la


# --------------------------------------------------------------------------- #
# S5-P1 -- shared eps-free geometric eig for uniform half-spaces               #
# --------------------------------------------------------------------------- #
def test_shared_uniform_geo_eig_reproduces_sem_modes():
    """The shared geometric-eig half-space modes reproduce the per-eps
    _sem_modes q-spectrum (as a set) for both polarizations and at oblique
    incidence -- one eps-free eig, q^2 = eps - mu (audit S5-P1)."""
    from lumenairy.elements.pmm._core import (
        _build_sem,
        _scalar_uniform_geo_eig,
        _sem_modes,
        _sem_modes_uniform_scalar,
    )
    period, degree = 1.0e-6, 15
    k0 = 2.0 * np.pi / 0.8e-6
    eps = 2.3 + 0.0j
    mats = _build_sem(period, 0.5e-6, eps, eps, degree, 1, 1, True)
    for kx0 in (0.0, 0.31 * k0):
        geo = _scalar_uniform_geo_eig(mats, k0, kx0)
        for pol in ("te", "tm"):
            _A, _lam, q_ref, _iv = _sem_modes(mats, k0, pol, kx0)
            _X, _l2, q_new, _iv2 = _sem_modes_uniform_scalar(
                mats, k0, pol, eps, kx0, geo=geo)
            a = np.sort_complex(np.round(q_ref, 10))
            b = np.sort_complex(np.round(q_new, 10))
            assert np.allclose(a, b, atol=1e-9), (
                f"pol={pol} kx0={kx0}: shared-eig q spectrum differs")


# --------------------------------------------------------------------------- #
# F4 -- PMM2DStack repeated-layer dedup is transparent (bit-exact physics)     #
# --------------------------------------------------------------------------- #
def test_stack_repeated_layer_dedup_is_transparent():
    """A stack of N IDENTICAL patterned layers (dedup fires -> one eig reused)
    equals a single layer of the total thickness: the interfaces between
    identical-material layers reflect nothing, so the physics is unchanged and
    the dedup is exact (audit F4)."""
    P, WL = 0.6e-6, 0.55e-6
    S = 6
    cell = np.full((S, S), 1.0 + 0j)
    cell[1:4, 1:4] = 6.0

    def _stack(n, t_each):
        st = la.PMM2DStack(period_x=P, period_y=P, n_substrate=1.5,
                           n_superstrate=1.0, degree=9, n_orders=3)
        for _ in range(n):
            st.add_layer(t_each, eps_cell=cell)
        return st.set_source(WL, theta=0.0, phi=0.0).solve()

    o4, R4, T4, J4 = _stack(4, 0.1e-6)     # 4 identical layers -> 1 dedup'd eig
    o1, R1, T1, J1 = _stack(1, 0.4e-6)     # single 0.4um layer
    assert np.array_equal(o4, o1)
    assert np.max(np.abs(np.asarray(J4) - np.asarray(J1))) < 1e-9, (
        "identical-material internal interfaces must be transparent")
    assert np.max(np.abs(R4 - R1)) < 1e-9 and np.max(np.abs(T4 - T1)) < 1e-9


# --------------------------------------------------------------------------- #
# F8 -- circular truncation (Lalanne 1997) for the 2-D PMM hybrid              #
# --------------------------------------------------------------------------- #
def _pillar(trunc, **kw):
    from lumenairy.elements.pmm.twod import pmm_efficiency_2d
    P, DEP, WL = 0.6e-6, 0.25e-6, 0.55e-6
    return pmm_efficiency_2d(
        P, P, 6.0, 2.0, (0.2 * P, 0.6 * P), (0.2 * P, 0.6 * P),
        1.5, 1.0, DEP, WL, truncation=trunc, **kw)[:3]


def test_circular_truncation_reduces_orders_and_converges():
    """truncation='circular' drops the high-|G| corner orders, conserves
    energy, and its zeroth order agrees with 'rectangular' (audit F8)."""
    kw = dict(degree=11, n_orders=8)
    o_r, R_r, T_r = _pillar("rectangular", **kw)
    o_c, R_c, T_c = _pillar("circular", **kw)
    assert len(o_c) < len(o_r), "circular must keep fewer orders"
    assert abs(float(R_c.sum() + T_c.sum()) - 1.0) < 1e-2   # energy at floor
    ir = (o_r[:, 0] == 0) & (o_r[:, 1] == 0)
    ic = (o_c[:, 0] == 0) & (o_c[:, 1] == 0)
    assert abs(float(R_r[ir][0]) - float(R_c[ic][0])) < 5e-3


def test_rectangular_truncation_is_the_default_and_unchanged():
    """The default is 'rectangular'; passing it explicitly is identical."""
    a = _pillar("rectangular", degree=9, n_orders=4)
    from lumenairy.elements.pmm.twod import pmm_efficiency_2d
    P, DEP, WL = 0.6e-6, 0.25e-6, 0.55e-6
    b = pmm_efficiency_2d(P, P, 6.0, 2.0, (0.2 * P, 0.6 * P),
                          (0.2 * P, 0.6 * P), 1.5, 1.0, DEP, WL,
                          degree=9, n_orders=4)[:3]
    assert len(a[0]) == len(b[0])
    assert np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2])


def test_invalid_truncation_raises():
    with pytest.raises(ValueError, match="truncation"):
        _pillar("bogus", degree=7, n_orders=3)


# --------------------------------------------------------------------------- #
# F5 -- factorized sandwich == dense (kron(Ty,Tx)*v) @ kron(Typ,Txp)            #
# --------------------------------------------------------------------------- #
def test_factorized_sandwich_matches_dense():
    """_sandwich_factorized reproduces the dense Kronecker sandwich to machine
    precision, without materializing kron(Ty, Tx) (audit F5)."""
    from lumenairy.elements.pmm.twod import _sandwich_factorized
    rng = np.random.default_rng(0)
    NyO, NxO, Ny, Nx = 5, 7, 11, 9
    Tx = rng.standard_normal((NxO, Nx)) + 1j * rng.standard_normal((NxO, Nx))
    Txp = rng.standard_normal((Nx, NxO)) + 1j * rng.standard_normal((Nx, NxO))
    Ty = rng.standard_normal((NyO, Ny)) + 1j * rng.standard_normal((NyO, Ny))
    Typ = rng.standard_normal((Ny, NyO)) + 1j * rng.standard_normal((Ny, NyO))
    v = rng.standard_normal(Ny * Nx) + 1j * rng.standard_normal(Ny * Nx)
    dense = (np.kron(Ty, Tx) * v[None, :]) @ np.kron(Typ, Txp)
    fac = _sandwich_factorized(Tx, Txp, Ty, Typ, v, NyO, NxO, Ny, Nx)
    assert fac.shape == (NyO * NxO, NyO * NxO)
    assert np.max(np.abs(dense - fac)) < 1e-11 * np.max(np.abs(dense))


# --------------------------------------------------------------------------- #
# B3 -- solve_vs_wavelength reuses a previously set_source()-configured angle   #
# --------------------------------------------------------------------------- #
def _stack():
    st = la.PMM2DStack(period_x=0.6e-6, period_y=0.6e-6, n_substrate=1.5,
                       n_superstrate=1.0, degree=7, n_orders=3)
    st.add_layer(0.25e-6, eps=2.25)
    return st


def test_solve_vs_wavelength_reuses_set_source_angle():
    """After set_source(theta=T), a sweep with no explicit theta must run at T
    (audit B3), not silently reset to normal incidence."""
    wl = 0.55e-6
    # explicit-angle sweep (the intended geometry)
    a = _stack()
    a.set_source(wl, theta=0.3)
    o_e, R_e, T_e = a.solve_vs_wavelength([wl], theta=0.3)[:3]
    # reuse: set_source(theta=0.3) then sweep with NO theta -> must equal explicit
    b = _stack()
    b.set_source(wl, theta=0.3)
    o_r, R_r, T_r = b.solve_vs_wavelength([wl])[:3]
    assert np.array_equal(o_e, o_r)
    assert np.max(np.abs(R_e - R_r)) < 1e-12
    assert np.max(np.abs(T_e - T_r)) < 1e-12
    # and the reused angle genuinely differs from normal incidence
    c = _stack()
    _o0, R0, _T0 = c.solve_vs_wavelength([wl], theta=0.0)[:3]
    assert np.max(np.abs(R_r - R0)) > 1e-6, (
        "the reused oblique angle must change R vs normal incidence")


def test_solve_vs_wavelength_explicit_theta_still_wins():
    """An explicit theta overrides any set_source angle (B3 must not hijack)."""
    wl = 0.55e-6
    st = _stack()
    st.set_source(wl, theta=0.3)          # configure oblique...
    _o, R_norm, _T = st.solve_vs_wavelength([wl], theta=0.0)[:3]  # ...override to normal
    ref = _stack()
    _o2, R_ref, _T2 = ref.solve_vs_wavelength([wl], theta=0.0)[:3]
    assert np.max(np.abs(R_norm - R_ref)) < 1e-12


def test_solve_vs_wavelength_defaults_normal_without_set_source():
    """No set_source() -> the sweep still defaults to normal incidence (the
    pre-B3 behavior is preserved when nothing was configured)."""
    wl = 0.55e-6
    st = _stack()
    _o, R_a, _T = st.solve_vs_wavelength([wl])[:3]
    ref = _stack()
    _o2, R_b, _T2 = ref.solve_vs_wavelength([wl], theta=0.0)[:3]
    assert np.max(np.abs(R_a - R_b)) < 1e-12


# --------------------------------------------------------------------------- #
# B1 -- pmm_efficiency_2d_cell honours max_nodal_dof on the JAX path           #
# --------------------------------------------------------------------------- #
def test_cell_jax_path_honours_max_nodal_dof():
    """The JAX cell dispatch used to skip _validate_cell_cost and drive a dense
    N x N assembly -> OOM.  It now rejects an oversized cell with the same
    max_nodal_dof error as the NumPy path (audit B1)."""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from lumenairy.elements.pmm.twod import pmm_efficiency_2d_cell
    P, DEP, WL = 0.6e-6, 0.25e-6, 0.55e-6
    lay = np.zeros((8, 8), dtype=int)
    lay[2:6, 2:6] = 1
    eps = np.where(lay == 1, 6.0, 2.0).astype(complex)
    kw = dict(period_x=P, period_y=P, region_layout=lay, n_substrate=1.5,
              n_superstrate=1.0, depth=DEP, wavelength=WL, degree=7, n_orders=3)
    # normal cell within the cap works on the JAX path
    _o, R, T = pmm_efficiency_2d_cell(eps_cell=jnp.asarray(eps), **kw)
    assert np.isfinite(float(np.asarray(R).sum() + np.asarray(T).sum()))
    # oversized cell (tiny cap) raises on BOTH paths, with the same message
    for e in (jnp.asarray(eps), eps):
        with pytest.raises(ValueError, match="max_nodal_dof"):
            pmm_efficiency_2d_cell(eps_cell=e, max_nodal_dof=50, **kw)
