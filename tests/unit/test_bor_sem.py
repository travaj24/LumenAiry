"""BOR spectral-element basis (sem_radial + BORStack basis='sem') gates.

Every gate is an INDEPENDENT oracle (Bessel ladders, the open-cladding fiber
characteristic equation, the analytic slab Airy response, the uniaxial
dispersion relation) -- never energy conservation alone (the lossless trap).
The spurious-mode gate counts EVERY discrete eigenvalue in a window against
the analytic ladder, so kernel pollution cannot hide between checked modes.
"""
import warnings

import numpy as np
import pytest
from scipy.special import jn_zeros, jnp_zeros

from lumenairy.elements.bor import BORStack
from lumenairy.elements.bor.fiber_oracle import fiber_modes
from lumenairy.elements.bor.sem_radial import (
    SemRadialMesh,
    equalize_meshes,
    sem_interface_smatrix,
    sem_layer_modes,
)
from lumenairy.elements.bor.zcascade import propagation_smatrix, redheffer_star

K0 = 2 * np.pi
RBIG = 8.0
EPS = 2.25


def _ladder(m, n=40):
    if m == 0:
        g = np.concatenate([jn_zeros(1, n), jn_zeros(0, n)]) / RBIG
    else:
        g = np.concatenate([jnp_zeros(m, n), jn_zeros(m, n)]) / RBIG
    return np.sort(K0 * K0 * EPS - g ** 2)[::-1]


@pytest.mark.parametrize("m", [0, 1, 2, 3, 4])
def test_uniform_bessel_ladder_and_no_interlopers(m):
    """Top-25 eigenvalues match the analytic TE+TM ladder IN COUNT AND VALUE
    (an extra discrete mode anywhere in the window = a spurious mode)."""
    mesh = SemRadialMesh(np.linspace(0, RBIG, 5), [(EPS,)] * 4, 12)
    L = sem_layer_modes(mesh, m, K0)
    q2c = L["q"] ** 2
    q2 = np.sort(q2c.real[np.abs(q2c.imag) < 1e-6 * np.abs(q2c).max()])[::-1]
    q2e = _ladder(m)
    win = q2e[24]
    got, want = q2[q2 > win], q2e[q2e > win]
    assert got.size == want.size, (
        f"m={m}: {got.size} modes in window vs {want.size} analytic -- "
        "spurious mode(s)")
    # 5e-8: the p=12 tail-mode quadrature floor at m=3-4 sits at ~6e-9;
    # the count assertion above is the spurious-mode gate, this one is
    # accuracy only
    assert np.max(np.abs(got - want) / np.abs(want)) < 5e-8


def test_p_convergence_is_spectral():
    """Error must fall by orders of magnitude from p=6 to p=12 (spectral),
    not by the ~4x a 2nd-order method would give."""
    errs = {}
    for p in (6, 12):
        mesh = SemRadialMesh(np.linspace(0, RBIG, 5), [(EPS,)] * 4, p)
        L = sem_layer_modes(mesh, 1, K0)
        q2c = L["q"] ** 2
        q2 = np.sort(q2c.real[np.abs(q2c.imag)
                              < 1e-6 * np.abs(q2c).max()])[::-1]
        q2e = _ladder(1)
        errs[p] = np.max(np.abs(q2[:16] - q2e[:16]) / np.abs(q2e[:16]))
    assert errs[6] > 1e-6                       # coarse is visibly inexact
    assert errs[12] < 1e-8                      # fine is near machine
    assert errs[6] / errs[12] > 1e3             # spectral collapse


def test_fiber_oracle_guided_modes_to_1e_9():
    """Open-cladding step-index fiber (m=1): the guided q's must match the
    canonical vector characteristic equation.  The FD basis floors at
    1e-4..1e-2 here; the SEM must reach <= 1e-9 -- the headline gate."""
    a, e1, e2, k0 = 1.0, 6.0, 2.0, 3.0
    q_oracle = fiber_modes(1, a, e1, e2, k0)
    mesh = SemRadialMesh(np.array([0.0, a, 3.0, 6.0, 10.0, 14.0]),
                         [(e1,), (e2,), (e2,), (e2,), (e2,)], 14)
    L = sem_layer_modes(mesh, 1, k0)
    q = L["q"]
    qg = np.sort(q[(np.abs(q.imag) < 1e-8)
                   & (q.real > np.sqrt(e2) * k0 + 1e-9)
                   & (q.real < np.sqrt(e1) * k0)].real)[::-1]
    assert qg.size == q_oracle.size            # exactly the oracle's modes
    for i, qo in enumerate(q_oracle):
        assert abs(qg[i] - qo) / qo < 1e-9


def test_anisotropic_uniaxial_dispersion_and_slot_swap():
    """m=0 TM ladder must follow gamma^2/eps_zz + q^2/eps_rr = k0^2; putting
    eps_e in the WRONG slot must break it (discriminating power)."""
    eo, ee = 2.25, 3.24

    def q_ladder(tri):
        mesh = SemRadialMesh(np.linspace(0, RBIG, 5), [tri] * 4, 12)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # slot-swap trips the axis guard
            L = sem_layer_modes(mesh, 0, K0)
        q = L["q"]
        return np.sort(q[(np.abs(q.imag) < 1e-8)
                         & (q.real > 1e-6)].real)[::-1]

    gtm = jn_zeros(0, 10) / RBIG
    pred = np.sqrt(eo * (K0 * K0 - gtm ** 2 / ee))
    qa = q_ladder((eo, eo, ee))
    err_ok = max(np.min(np.abs(qa - p_)) / p_ for p_ in pred[:8])
    assert err_ok < 1e-9
    qw = q_ladder((ee, eo, eo))                 # eps_e in the WRONG slot
    err_bad = max(np.min(np.abs(qw - p_)) / p_ for p_ in pred[:8])
    # relative discrimination: the wrong slot must miss by >= 5 orders more
    # than the correct one hits (min-distance vs a rich ladder can land a
    # coincidental few-1e-3 neighbour, so an absolute bar is too blunt)
    assert err_bad > 1e5 * err_ok, (
        f"wrong-slot tensor still matched -- oracle blind "
        f"(ok={err_ok:.2e}, bad={err_bad:.2e})")


def test_axis_anisotropy_guard_warns():
    with pytest.warns(UserWarning, match="axis"):
        SemRadialMesh(np.linspace(0, RBIG, 3), [(3.24, 2.25, 2.25)] * 2, 4)


def test_same_medium_interface_is_identity():
    mesh = SemRadialMesh(np.linspace(0, 6.0, 4), [(EPS,)] * 3, 8)
    L = sem_layer_modes(mesh, 1, K0)
    S = sem_interface_smatrix(L, L)
    n = L["q"].size
    assert np.abs(S[0]).max() < 1e-12
    assert np.abs(S[1] - np.eye(n)).max() < 1e-12


def test_slab_airy_and_energy():
    """Uniform dielectric slab: R at every propagating mode's angle must match
    the analytic Airy TE/TM response; energy must conserve to ~1e-10."""
    R = 6.0
    e_slab, thk = 4.0, 0.31
    bnds = np.linspace(0, R, 4)
    La = sem_layer_modes(SemRadialMesh(bnds, [(1.0,)] * 3, 10), 1, K0)
    Ls = sem_layer_modes(SemRadialMesh(bnds, [(e_slab,)] * 3, 10), 1, K0)
    S = redheffer_star(redheffer_star(
        sem_interface_smatrix(La, Ls), propagation_smatrix(Ls["q"], thk)),
        sem_interface_smatrix(Ls, La))
    qa = La["q"]
    prop = np.where((np.abs(qa.imag) < 5e-5 * np.abs(qa.real).max())
                    & (qa.real > 1e-6))[0]
    Rr = np.array([np.sum(np.abs(S[0][prop, j]) ** 2) for j in prop])
    Tt = np.array([np.sum(np.abs(S[2][prop, j]) ** 2) for j in prop])
    assert np.abs(Rr + Tt - 1.0).max() < 1e-10
    th = np.arcsin(np.clip(
        np.sqrt(np.maximum(K0 ** 2 - qa[prop].real ** 2, 0)) / K0, 0, 1))

    def airy(theta, pol):
        n1, n2 = 1.0, np.sqrt(e_slab)
        c1, s1 = np.cos(theta), np.sin(theta)
        c2 = np.sqrt(1 - (s1 * n1 / n2) ** 2 + 0j)
        r12 = ((n1 * c1 - n2 * c2) / (n1 * c1 + n2 * c2) if pol == "te"
               else (n2 * c1 - n1 * c2) / (n2 * c1 + n1 * c2))
        beta = K0 * n2 * c2 * thk
        r = (r12 * (1 - np.exp(2j * beta))) / (1 - r12 ** 2 * np.exp(2j * beta))
        return abs(r) ** 2
    worst = max(min(abs(Rr[i] - airy(th[i], "te")),
                    abs(Rr[i] - airy(th[i], "tm")))
                for i in range(prop.size))
    assert worst < 1e-6


def test_per_layer_meshes_reproduce_shared_mesh_physics():
    """The mortar (cross-tested Galerkin) per-layer interface must reproduce
    the shared-mesh R/T on smooth propagating channels, with conserved
    energy -- the per-layer-grids transfer-function gate."""
    R = 6.0
    e_slab, thk, m = 4.0, 0.31, 1
    bnds = np.linspace(0, R, 4)
    La = sem_layer_modes(SemRadialMesh(bnds, [(1.0,)] * 3, 10), m, K0)
    Ls_shared = sem_layer_modes(
        SemRadialMesh(bnds, [(e_slab,)] * 3, 10), m, K0)
    Ls_own = sem_layer_modes(
        SemRadialMesh(np.array([0.0, 2.6, 4.1, R]), [(e_slab,)] * 3, 10),
        m, K0)

    def rt(Ls):
        S = redheffer_star(redheffer_star(
            sem_interface_smatrix(La, Ls),
            propagation_smatrix(Ls["q"], thk)),
            sem_interface_smatrix(Ls, La))
        qa = La["q"]
        prop = np.where((np.abs(qa.imag) < 5e-5 * np.abs(qa.real).max())
                        & (qa.real > 1e-6))[0]
        Rr = np.array([np.sum(np.abs(S[0][prop, j]) ** 2) for j in prop])
        Tt = np.array([np.sum(np.abs(S[2][prop, j]) ** 2) for j in prop])
        return Rr, Tt
    R1, T1 = rt(Ls_shared)
    R2, T2 = rt(Ls_own)
    assert np.abs(R2 + T2 - 1.0).max() < 1e-5      # mortar keeps energy
    assert np.abs(R1 - R2).max() < 5e-3            # spectral-remainder scale


def test_borstack_sem_ring_grating_energy_and_fd_agreement():
    """End-to-end BORStack(basis='sem'): ring-grating stack conserves energy
    and agrees with the (much finer) FD answer at matched angles to the FD's
    own convergence scale."""
    k0 = 2.0
    wl = 2 * np.pi / k0
    lam, duty, n_r, n_g = 3.0, 0.5, 2.45, 1.41
    rbig = 8 * lam

    def run(basis, **kw):
        s = BORStack(rbig, 1, n_substrate=1.41, n_superstrate=1.41,
                     basis=basis, **kw)
        s.add_layer(0.4, eps=1.41 ** 2)
        s.add_layer(0.5, rings=(lam, duty, n_r, n_g))
        s.add_layer(0.4, eps=1.41 ** 2)
        s.set_source(wavelength=wl)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return s.solve()
    r_sem = run("sem", degree=8)
    en = np.abs(np.atleast_1d(r_sem["energy"]) - 1)
    assert en.max() < 1e-6
    r_fd = run("fd", N=512)
    a_s = np.degrees(np.atleast_1d(r_sem["angles"]))
    a_f = np.degrees(np.atleast_1d(r_fd["angles"]))
    R_s, R_f = np.atleast_1d(r_sem["R"]), np.atleast_1d(r_fd["R"])
    ds = []
    for i in range(a_s.size):
        j = int(np.argmin(np.abs(a_f - a_s[i])))
        if abs(a_f[j] - a_s[i]) < 0.3:
            ds.append(abs(R_s[i] - R_f[j]))
    assert len(ds) >= 10
    assert max(ds) < 0.15                       # FD's own 2nd-order error


def test_borstack_sem_segments_spec_and_anisotropy():
    """segments= layers (the SEM-native spec) with a diagonal-cylindrical
    tensor annulus cascade and conserve energy."""
    rbig, k0 = 6.0, 2.0
    s = BORStack(rbig, 0, n_substrate=1.5, n_superstrate=1.5,
                 basis="sem", degree=8)
    s.add_layer(0.4, segments=[(rbig, 1.5 ** 2)])
    s.add_layer(0.5, segments=[(2.0, 2.25), (3.0, (2.25, 3.24, 2.25)),
                               (rbig, 2.25)])
    s.set_source(wavelength=2 * np.pi / k0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = s.solve()
    en = np.abs(np.atleast_1d(res["energy"]) - 1)
    # the cross-mesh mortar remainder at degree 8 (spectrally decaying;
    # the shared-mesh path in the ring-grating gate sits at ~1e-8)
    assert en.size > 0 and en.max() < 1e-5


def test_borstack_sem_rejects_bare_profile_layers():
    s = BORStack(4.0, 0, basis="sem", degree=4)
    s.add_layer(0.2, eps_profile=lambda r: np.full(np.size(r), 2.0, complex))
    s.set_source(wavelength=1.0)
    with pytest.raises(ValueError, match="segments"):
        s.solve()


def test_equalize_meshes_pads_to_common_count():
    m1 = SemRadialMesh(np.linspace(0, 4.0, 6), [(2.0,)] * 5, 4)
    m2 = SemRadialMesh(np.array([0.0, 4.0]), [(1.0,)], 4)
    e1, e2 = equalize_meshes([m1, m2])
    assert e1.ne == e2.ne == 5
    assert e1.n0 == e2.n0 and e1.n1 == e2.n1


@pytest.mark.parametrize("pol", ["te", "tm"])
def test_gate4_sem_per_order_vs_planar(pol):
    """GATE 4 on the SEM basis: ring-grating per-order efficiencies must
    match planar pmm_efficiency_1d at each cylindrical order's local oblique
    angle -- at a TIGHTER bar than the FD gate's 0.015/0.025 (the spurious
    floor the SEM was built to remove)."""
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_gate4 import _planar_buckets
    k0, n_sup = 2.0, 1.41
    n_ridge, n_groove, depth, duty, m_az = 2.45, 1.41, 0.5, 0.5, 1
    lam, rfac = 3.0, 16
    eps = n_sup ** 2
    wavelength = 2 * np.pi / k0
    s = BORStack(rfac * lam, m_az, n_substrate=n_sup, n_superstrate=n_sup,
                 basis="sem", degree=10)
    s.add_layer(depth, rings=(lam, duty, n_ridge, n_groove))
    s.set_source(k0=k0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = s.solve()
    inc = res["inc"]
    S11, S21 = res["S"][0], res["S"][2]
    gam = res["gamma"]
    sup = s._last["sup"]
    n1, W = sup["n1"], sup["W"]
    pte = np.array([
        np.linalg.norm(W[n1:, j]) ** 2
        / max(np.linalg.norm(W[:n1, j]) ** 2
              + np.linalg.norm(W[n1:, j]) ** 2, 1e-300) for j in inc])
    sel = np.where(pte > 0.6 if pol == "te" else pte < 0.4)[0]
    assert sel.size >= 3
    jsel = sel[np.argmax(gam[sel])]
    jglob = inc[jsel]
    th = float(np.arcsin(np.clip(gam[jsel] / (np.sqrt(eps) * k0), 0, 1)))
    bk, _sumRT = _planar_buckets(lam, th, n_sup, k0, n_ridge, n_groove,
                                 n_sup, depth, duty, wavelength, pol)
    centers = np.array([b["absk"] for b in bk])
    cR = np.zeros(len(bk))
    cT = np.zeros(len(bk))
    for kk in sel:
        bi = int(np.argmin(np.abs(centers - gam[kk])))
        cR[bi] += abs(S11[inc[kk], jglob]) ** 2
        cT[bi] += abs(S21[inc[kk], jglob]) ** 2
    worst = max(max(abs(b["R"] - cR[i]), abs(b["T"] - cT[i]))
                for i, b in enumerate(bk))
    assert worst < 0.010, f"per-bucket eta off by {worst:.4f}"
    assert np.abs(np.atleast_1d(res["energy"]) - 1).max() < 1e-6


def test_wavelength_resolution_cap_on_sparse_wall_layers():
    """Regression: a wall-sparse layer (all rings near the axis) once left a
    ~7-wavelength element whose starved basis broke the cascade -- NEGATIVE
    absorption in a lossless stack.  The wavelength cap must keep energy
    conserved and R hp-stable."""
    k0 = 2.0

    def run(**kw):
        s = BORStack(24.0, 1, n_substrate=1.41, n_superstrate=1.41,
                     basis="sem", **kw)
        s.add_layer(0.4, eps=1.41 ** 2)
        s.add_layer(0.5, segments=[(1.5, 6.0), (3.0, 2.25),
                                   (24.0, 1.41 ** 2)])
        s.set_source(k0=k0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return s.solve()
    r1 = run(degree=8)
    r2 = run(degree=8, elements_per_segment=2)
    assert np.abs(np.atleast_1d(r1["energy"]) - 1).max() < 1e-6
    a1 = np.degrees(np.atleast_1d(r1["angles"]))
    a2 = np.degrees(np.atleast_1d(r2["angles"]))
    R1, R2 = np.atleast_1d(r1["R"]), np.atleast_1d(r2["R"])
    ds = [abs(R1[i] - R2[int(np.argmin(np.abs(a2 - a1[i])))])
          for i in range(a1.size) if np.min(np.abs(a2 - a1[i])) < 0.2]
    assert len(ds) >= 10 and max(ds) < 1e-3


def test_layer_absorption_closure_on_sem():
    """C1b identity on the SEM basis: R + T + sum_i A_i = 1 per incident
    mode; per-layer A ~ 0 for a lossless stack, >= 0 for a passive one."""
    k0 = 2.0

    def run(core):
        s = BORStack(24.0, 1, n_substrate=1.41, n_superstrate=1.41,
                     basis="sem", degree=8)
        s.add_layer(0.4, eps=1.41 ** 2)
        s.add_layer(0.5, segments=[(1.5, core), (3.0, 2.25),
                                   (24.0, 1.41 ** 2)])
        s.add_layer(0.3, eps=1.41 ** 2)
        s.set_source(k0=k0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = s.solve(retain_internal=True)
            return res, s.layer_absorption()
    res, A = run(6.0)                                    # lossless
    R, T = np.atleast_1d(res["R"]), np.atleast_1d(res["T"])
    assert np.abs(R + T + A.sum(axis=0) - 1.0).max() < 1e-6
    assert np.abs(A).max() < 1e-6
    res, A = run(-80.0 + 2.5j)                           # lossy metal
    R, T = np.atleast_1d(res["R"]), np.atleast_1d(res["T"])
    assert np.abs(R + T + A.sum(axis=0) - 1.0).max() < 1e-6
    # passivity: per-mode noise in the ~zero-A cladding layers measured
    # -1.1e-6 at deg8, so the bar sits at -5e-6 (closure is the tight gate)
    assert A.min() > -5e-6


def test_sem_pml_m3_bound_mode_invariance():
    """M3 gate on the SEM PML: a truly bound fiber mode's q must be invariant
    to sigma_max (FD bar was 1e-6; the SEM reaches ~1e-13), and the radiation
    continuum must be absorbed (complex q)."""
    a, e1, e2, k0 = 1.0, 6.0, 2.0, 3.0
    q_oracle = fiber_modes(1, a, e1, e2, k0)[0]
    qs = []
    for sig in (3.0, 20.0):
        mesh = SemRadialMesh(np.array([0.0, a, 4.0, 8.0, 12.0]),
                             [(e1,), (e2,), (e2,), (e2,)], 12,
                             R_pml=8.0, sigma_max=sig)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            L = sem_layer_modes(mesh, 1, k0)
        q = L["q"]
        qs.append(complex(q[int(np.argmin(np.abs(q - q_oracle)))]))
        n_abs = int(np.sum((np.abs(q.imag) > 1e-3) & (q.real > 0)
                           & (q.real < np.sqrt(e2) * k0)))
        assert n_abs > 20                    # continuum absorbed
    assert abs(qs[1] - qs[0]) < 1e-9
    assert abs(qs[0] - q_oracle) < 1e-7


def test_longitudinal_resonance_pencil_fallback():
    """At an EXACT longitudinal resonance the Schur elimination is undefined;
    the unreduced QZ pencil must fire (warning) and reproduce the propagating
    ladder of a detuned elimination solve."""
    from scipy.linalg import eig as _eig

    from lumenairy.elements.bor.sem_radial import _assemble, _keeps
    mesh = SemRadialMesh(np.linspace(0, 8.0, 5), [(2.25,)] * 4, 8)
    m = 1
    ops = _assemble(mesh, m, 1.0)
    _ip, iz = _keeps(mesh, m)
    Sz = ops["S_z"][np.ix_(iz, iz)]
    Me = ops["Mz_eps"][np.ix_(iz, iz)]
    k0_res = float(np.sqrt(np.sort(np.real(_eig(Sz, Me)[0]))[10]))
    with pytest.warns(UserWarning, match="UNREDUCED"):
        L_res = sem_layer_modes(mesh, m, k0_res)
    L_det = sem_layer_modes(mesh, m, k0_res * (1 + 1e-7))

    def ladder(L, k0):
        q = L["q"]
        return np.sort(q[(np.abs(q.imag) < 1e-6 * k0)
                         & (q.real > 1e-6)].real)[::-1]
    qa, qb = ladder(L_res, k0_res), ladder(L_det, k0_res)
    k = min(qa.size, qb.size, 10)
    assert max(abs(qa[i] - qb[i]) / qb[i] for i in range(k)) < 1e-5


def test_sem_farfield_parseval_on_quad_grid():
    """Oversampled quad_eval projection: Parseval identity on a non-uniform
    SEM grid (the far-field deferral gate)."""
    from scipy.special import jv

    from lumenairy.elements.bor.farfield import fourier_bessel
    R, m = 8.0, 1
    mesh = SemRadialMesh(np.array([0.0, 1.3, 2.2, 5.0, R]), [(1.0,)] * 4, 12)
    r0, _w0 = mesh.nodes0()
    f_dof = np.exp(-((r0 - 2.5) / 1.1) ** 2) * jv(1, 2.3 * r0)
    rq, wq, E0, _E1 = mesh.quad_eval()
    fq = E0 @ f_dof
    with warnings.catch_warnings():
        warnings.simplefilter("error")       # any Nyquist warning = failure
        c, _kt, norm = fourier_bessel(fq, rq, 0.0, m, 40, wq=wq, R=R)
    total = np.sum(np.abs(c) ** 2 * norm)
    direct = np.sum(np.abs(fq) ** 2 * wq)
    assert abs(total - direct) / direct < 1e-8
