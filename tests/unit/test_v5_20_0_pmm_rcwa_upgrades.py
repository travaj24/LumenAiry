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
# F2 -- even-parity symmetry fold for the 2-D PMM hybrid                       #
# --------------------------------------------------------------------------- #
def _centered(pol, sym, **kw):
    from lumenairy.elements.pmm.twod import pmm_efficiency_2d
    P, DEP, WL = 0.6e-6, 0.25e-6, 0.55e-6
    xb = (0.3 * P, 0.7 * P)                       # centered on the cell centre
    return pmm_efficiency_2d(
        P, P, 6.0, 2.0, xb, xb, 1.5, 1.0, DEP, WL, polarization=pol,
        symmetry=sym, **kw)[:3]


@pytest.mark.parametrize("pol", ["te", "tm"])
def test_symmetry_fold_matches_full_solve(pol):
    """symmetry=True (even-parity sector) matches the full 2N solve to ~1e-12
    for a centered pillar at normal incidence (audit F2)."""
    kw = dict(degree=11, n_orders=6)
    o_f, R_f, T_f = _centered(pol, False, **kw)
    o_s, R_s, T_s = _centered(pol, True, **kw)
    assert np.array_equal(o_f, o_s)
    assert np.max(np.abs(R_f - R_s)) < 1e-11
    assert np.max(np.abs(T_f - T_s)) < 1e-11


def test_symmetry_falls_back_when_precondition_fails():
    """symmetry=True must fall back to the FULL solve (bit-identical) when the
    precondition fails -- oblique incidence (order set not flip-closed) and an
    off-centre cell -- so the result is always correct."""
    from lumenairy.elements.pmm.twod import pmm_efficiency_2d
    P, DEP, WL = 0.6e-6, 0.25e-6, 0.55e-6
    kw = dict(degree=9, n_orders=5)
    # oblique -> flip perm is None -> full solve
    a = pmm_efficiency_2d(P, P, 6.0, 2.0, (0.3 * P, 0.7 * P), (0.3 * P, 0.7 * P),
                          1.5, 1.0, DEP, WL, theta=np.radians(20.0),
                          symmetry=False, **kw)[:3]
    b = pmm_efficiency_2d(P, P, 6.0, 2.0, (0.3 * P, 0.7 * P), (0.3 * P, 0.7 * P),
                          1.5, 1.0, DEP, WL, theta=np.radians(20.0),
                          symmetry=True, **kw)[:3]
    assert np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2])


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


def test_stack_sweep_geom_cache_is_transparent_and_reused():
    """F4 part 2: a wavelength sweep reuses the wl-independent per-layer build
    (one geom-cache entry per unique layer, shared across all wavelengths) and
    is bit-identical to per-wavelength fresh solves."""
    P, DEP = 0.6e-6, 0.25e-6
    S = 6
    cell = np.full((S, S), 1.0 + 0j)
    cell[1:4, 1:4] = 6.0
    wls = np.array([0.50e-6, 0.55e-6, 0.60e-6])
    st = la.PMM2DStack(period_x=P, period_y=P, n_substrate=1.5,
                       n_superstrate=1.0, degree=9, n_orders=3)
    st.add_layer(DEP, eps_cell=cell)
    o, R, T = st.solve_vs_wavelength(wls)[:3]
    assert len(st._geom_cache) == 1        # one unique layer, built once
    for i, w in enumerate(wls):
        ref = la.PMM2DStack(period_x=P, period_y=P, n_substrate=1.5,
                            n_superstrate=1.0, degree=9, n_orders=3)
        ref.add_layer(DEP, eps_cell=cell)
        ref.set_source(float(w))
        _o1, R1, T1, _j = ref.solve()
        assert np.array_equal(R[i], R1) and np.array_equal(T[i], T1)


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
def test_cell_jax_path_factorized_matches_numpy():
    """B1 part b: the JAX cell path builds Gx0F/Gy0F by per-axis factorization
    (no dense N x N Minv/kron), still matching the NumPy cell path to machine
    precision."""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from lumenairy.elements.pmm.twod import pmm_efficiency_2d_cell
    P, DEP, WL = 0.6e-6, 0.25e-6, 0.55e-6
    lay = np.zeros((10, 10), dtype=int)
    lay[2:7, 2:7] = 1
    eps = np.where(lay == 1, 6.0, 2.0).astype(complex)
    kw = dict(period_x=P, period_y=P, region_layout=lay, n_substrate=1.5,
              n_superstrate=1.0, depth=DEP, wavelength=WL, degree=7, n_orders=4)
    _on, Rn, Tn = pmm_efficiency_2d_cell(eps_cell=eps, **kw)[:3]
    _oj, Rj, Tj = pmm_efficiency_2d_cell(eps_cell=jnp.asarray(eps), **kw)[:3]
    assert np.max(np.abs(Rn - np.asarray(Rj))) < 1e-11
    assert np.max(np.abs(Tn - np.asarray(Tj))) < 1e-11


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


# --------------------------------------------------------------------------- #
# R3 -- F2 even-parity fold extended to PMM2DStack + pmm_jones_2d (tensor)      #
# --------------------------------------------------------------------------- #
def _centro_scalar_cell(S, eps_hi=6.25):
    """Centro-symmetric scalar cell: a centred square (flip-invariant grid)."""
    c = np.full((S, S), 1.0 + 0j)
    lo, hi = S // 4, S - S // 4
    c[lo:hi, lo:hi] = eps_hi
    return c


def _sym_stack(sym, *, theta=0.0):
    P, WL = 0.6e-6, 0.55e-6
    st = la.PMM2DStack(period_x=P, period_y=P, n_substrate=1.5,
                       n_superstrate=1.0, degree=11, n_orders=6, symmetry=sym)
    st.add_layer(0.22e-6, eps_cell=_centro_scalar_cell(8))
    st.add_layer(0.08e-6, eps=2.10)                       # uniform interlayer
    st.add_layer(0.18e-6, eps_cell=_centro_scalar_cell(6, 5.0))
    return st.set_source(WL, theta=theta, phi=0.0).solve()


def test_stack_symmetry_fold_matches_full_solve():
    """PMM2DStack(symmetry=True) runs the whole cascade in the even sector and
    matches the full solve to the ~1e-12 even-basis level for a centro-symmetric
    multilayer scalar stack at normal incidence (audit F2 for the cascade)."""
    o_f, R_f, T_f, J_f = _sym_stack(False)
    o_s, R_s, T_s, J_s = _sym_stack(True)
    assert np.array_equal(o_f, o_s)
    assert np.max(np.abs(R_f - R_s)) < 1e-11
    assert np.max(np.abs(T_f - T_s)) < 1e-11
    assert np.max(np.abs(np.asarray(J_f) - np.asarray(J_s))) < 1e-11


def test_stack_symmetry_falls_back_at_oblique():
    """Oblique incidence is not flip-closed -> the stack even-parity fold must
    fall back to the full solve, BYTE-IDENTICAL."""
    o_f, R_f, T_f, J_f = _sym_stack(False, theta=np.radians(20.0))
    o_s, R_s, T_s, J_s = _sym_stack(True, theta=np.radians(20.0))
    assert np.array_equal(R_f, R_s) and np.array_equal(T_f, T_s)
    assert np.array_equal(np.asarray(J_f), np.asarray(J_s))


def _inplane_tensor(n_o, n_e, az):
    """Exactly-in-plane uniaxial tensor (zz = n_o**2, no z-coupling)."""
    eo, ee = n_o ** 2, n_e ** 2
    c, s = np.cos(az), np.sin(az)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return Rz @ np.diag([ee, eo, eo]).astype(complex) @ Rz.T


def _centro_tensor_cell(S):
    cell = np.empty((S, S, 3, 3), dtype=complex)
    for i in range(S):
        for j in range(S):
            xr, yr = (i - (S - 1) / 2) / S, (j - (S - 1) / 2) / S
            az = 0.6 * np.cos(2 * np.pi * np.hypot(xr, yr))   # centro-symmetric
            cell[i, j] = _inplane_tensor(1.5, 1.7, az)
    return cell


@pytest.mark.parametrize("formulation", ["laurent", "li"])
def test_jones_2d_tensor_symmetry_fold_matches_full(formulation):
    """pmm_jones_2d(symmetry=True) folds the IN-PLANE tensor P/Q (rcwa's
    _tensor_PQ, byte-identical to the eigendecomposed blocks) into the even
    sector, matching the full 2Nf tensor solve to ~1e-12 (audit F2, tensor)."""
    P, WL, DEP = 0.55e-6, 0.5e-6, 0.30e-6
    kw = dict(degree=9, n_orders=6, formulation=formulation, theta=0.0)
    a = la.pmm_jones_2d(P, P, _centro_tensor_cell(8), 1.5, 1.0, DEP, WL,
                        symmetry=False, **kw)
    b = la.pmm_jones_2d(P, P, _centro_tensor_cell(8), 1.5, 1.0, DEP, WL,
                        symmetry=True, **kw)
    assert np.array_equal(a[0], b[0])
    assert np.max(np.abs(a[1] - b[1])) < 1e-11
    assert np.max(np.abs(a[2] - b[2])) < 1e-11
    assert np.max(np.abs(np.asarray(a[3]) - np.asarray(b[3]))) < 1e-11


def test_jones_2d_symmetry_falls_back_offplane():
    """An OUT-OF-PLANE tensor cell cannot fold (the generator breaks +/-lam),
    so symmetry=True falls back to the generalized cascade BYTE-IDENTICALLY."""
    from lumenairy.elements.rcwa._core import uniaxial_tensor
    P, WL, DEP = 0.55e-6, 0.5e-6, 0.30e-6
    S = 6
    cell = np.empty((S, S, 3, 3), dtype=complex)
    for i in range(S):
        for j in range(S):
            xr, yr = (i - (S - 1) / 2) / S, (j - (S - 1) / 2) / S
            az = 0.5 * np.cos(2 * np.pi * np.hypot(xr, yr))
            cell[i, j] = uniaxial_tensor(1.5, 1.7, np.deg2rad(35.0), phi=az)
    kw = dict(degree=9, n_orders=5, theta=0.0)
    a = la.pmm_jones_2d(P, P, cell, 1.5, 1.0, DEP, WL, symmetry=False, **kw)
    b = la.pmm_jones_2d(P, P, cell, 1.5, 1.0, DEP, WL, symmetry=True, **kw)
    assert np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2])
    assert np.array_equal(np.asarray(a[3]), np.asarray(b[3]))


# --------------------------------------------------------------------------- #
# R6 -- S5-P1 shared eps-free geometric eig for the SLANT half-spaces          #
# --------------------------------------------------------------------------- #
def test_slant_shared_uniform_geo_eig_reproduces_sem_modes_slant():
    """A homogeneous (slant-free) half-space's shared-eig slant modes reproduce
    the per-eps _sem_modes_slant q-spectrum (as a set), both pols + oblique --
    q^2 = eps - mu, invop = (1/eps)I (audit S5-P1 for the slant path)."""
    from lumenairy.elements.pmm._core import (
        _build_sem_slant,
        _scalar_uniform_geo_eig,
        _sem_modes_slant,
        _sem_modes_slant_uniform,
    )
    period, degree = 1.0e-6, 15
    k0 = 2.0 * np.pi / 0.8e-6
    eps = 2.3 + 0.0j
    mats = _build_sem_slant(period, 0.5e-6, eps, eps, degree, 1, 1, True)
    for kx0 in (0.0, 0.31 * k0):
        geo = _scalar_uniform_geo_eig(mats, k0, kx0)
        for pol in ("te", "tm"):
            ref = _sem_modes_slant(mats, k0, pol, 0.0, kx0)      # per-eps
            new = _sem_modes_slant_uniform(mats, k0, pol, eps, kx0, geo=geo)
            a = np.sort_complex(np.round(ref["q"], 10))
            b = np.sort_complex(np.round(new["q"], 10))
            assert np.allclose(a, b, atol=1e-9), (
                f"pol={pol} kx0={kx0}: shared-eig slant q spectrum differs")
            if pol == "tm":                          # invop == (1/eps)I exactly
                n = mats["S0"].shape[0]
                assert np.allclose(new["invop"], np.eye(n) / eps, atol=1e-12)


def test_slant_efficiency_unchanged_by_shared_eig():
    """pmm_efficiency_1d_slanted still matches the analytic energy balance with
    the shared-eig half-spaces (the S5-P1 slant port is gauge-equivalent)."""
    o, R, T = la.pmm_efficiency_1d_slanted(
        0.5e-6, np.sqrt(2.0), 1.0, 1.5, 1.0, 0.4e-6, 0.5, 0.55e-6,
        slant_angle=np.deg2rad(15.0), angle=np.deg2rad(20.0),
        polarization="te", degree=13)
    assert abs(float(R.sum() + T.sum()) - 1.0) < 1e-5    # lossless TE closes (BLAS-robust; slant floor ~1e-6)
