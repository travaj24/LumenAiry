"""G03-guards-em: robustness remediation for the v5.24.2 exhaustive audit.

Covers findings S1-7, S1-14, S1-15, S1-17, S1-20, S5-11.  Each test FAILS on
the pre-fix code (silent-wrong / crash / silent-non-convergence) and PASSES
after.  Oracles are INDEPENDENT where possible: a hand geometry rule (back-side
alias), the documented ``eps * eye(3)`` workaround, energy conservation, and a
cross-entry reduction (scalar 2-D solver vs the tensor Jones solver).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import warnings

import numpy as np
import pytest


# --------------------------------------------------------------------------- #
#  S1-7: PMMStack source setters reject back-side incidence (was silent)        #
# --------------------------------------------------------------------------- #
class TestS1_7BackSideIncidence:
    def test_set_source_rejects_back_side_angle(self):
        """A back-side angle (|angle| >= pi/2) aliases BYTE-IDENTICALLY to the
        supplementary front-side angle pi - angle (kx0 ~ sin(angle)); the
        checked resolver rejects it instead of silently solving the wrong
        geometry."""
        from lumenairy.elements.pmm.stack import PMMStack
        st = PMMStack(1e-6, n_substrate=1.5, n_superstrate=1.0)
        with pytest.raises(ValueError, match="pi/2"):
            st.set_source(0.5e-6, angle=2.5)          # 2.5 rad > pi/2

    def test_set_source_front_side_and_theta_alias_unchanged(self):
        """The fix must not disturb the front-side path or the theta-wins
        alias contract."""
        from lumenairy.elements.pmm.stack import PMMStack
        st = PMMStack(1e-6, n_substrate=1.5, n_superstrate=1.0)
        st.set_source(0.5e-6, angle=0.3)
        assert st._src["angle"] == pytest.approx(0.3)
        st.set_source(0.5e-6, angle=0.3, theta=0.4)   # theta wins
        assert st._src["angle"] == pytest.approx(0.4)

    def test_solve_vs_wavelength_rejects_back_side(self):
        from lumenairy.elements.pmm.stack import PMMStack
        st = PMMStack(1e-6, n_substrate=1.5, n_superstrate=1.0)
        st.add_layer(0.3e-6, eps=2.25)
        with pytest.raises(ValueError, match="pi/2"):
            st.solve_vs_wavelength([0.5e-6], angle=2.5)

    def test_pmmstack_and_1d_entry_agree_on_back_side(self):
        """Independent parity: the SAME back-side angle is rejected by both the
        PMMStack setter and the 1-D entry point (they now share ONE checked
        resolver)."""
        from lumenairy.elements.pmm.oned import pmm_jones_1d
        from lumenairy.elements.pmm.stack import PMMStack
        st = PMMStack(1e-6, n_substrate=1.5, n_superstrate=1.0)
        raised_stack = raised_1d = False
        try:
            st.set_source(0.5e-6, angle=2.0)
        except ValueError:
            raised_stack = True
        try:
            pmm_jones_1d(1e-6, 2.25, 1.0, 1.5, 1.0, 0.3e-6, 0.5, 0.5e-6,
                         angle=2.0)
        except ValueError:
            raised_1d = True
        assert raised_stack and raised_1d

    def test_checked_resolver_is_single_source(self):
        """The move to ._core must be single-source: oned re-exports the SAME
        object (no drifted second copy -- the multi-copy pattern the audit warns
        about)."""
        from lumenairy.elements.pmm import _core as core
        from lumenairy.elements.pmm import oned
        assert oned._resolve_incidence_checked is core._resolve_incidence_checked


# --------------------------------------------------------------------------- #
#  S1-14: uniform-tensor entry to RCWA (scalar eps / (3,3) eps)                 #
# --------------------------------------------------------------------------- #
class TestS1_14UniformTensorEntry:
    def test_rcwa_jones_1d_scalar_eps_matches_explicit_tensor(self):
        """rcwa_jones_1d used to IndexError on a scalar eps; it now promotes to
        eps * I3.  Oracle: the documented ``scalar * np.eye(3)`` workaround must
        give a BYTE-IDENTICAL result."""
        from lumenairy.elements.rcwa.oned import rcwa_jones_1d
        o_s, R_s, T_s, J_s = rcwa_jones_1d(
            1e-6, 2.25, 1.0, 1.5, 1.0, 0.3e-6, 0.5, 0.5e-6, n_orders=5)
        o_t, R_t, T_t, J_t = rcwa_jones_1d(
            1e-6, 2.25 * np.eye(3), 1.0 * np.eye(3), 1.5, 1.0, 0.3e-6, 0.5,
            0.5e-6, n_orders=5)
        assert np.array_equal(o_s, o_t)
        assert np.array_equal(R_s, R_t)
        assert np.array_equal(T_s, T_t)
        assert np.array_equal(J_s, J_t)
        assert R_s[0].sum() + T_s[0].sum() == pytest.approx(1.0, abs=1e-9)

    def test_rcwastack_uniform_iso_tensor_matches_scalar_slab(self):
        """RCWAStack.add_layer(eps=<3x3>) used to crash at solve; it now builds a
        uniform tensor cell.  Oracle: an ISOTROPIC uniform tensor slab must equal
        the scalar-eps slab to machine precision (a DIFFERENT layer kind /
        solver path -> an independent cross-check)."""
        from lumenairy.elements.rcwa.stack import RCWAStack

        def solve(eps):
            st = RCWAStack(1e-6, n_superstrate=1.0, n_substrate=1.5, n_orders=5)
            st.add_layer(0.3e-6, eps=eps)
            st.set_source(0.5e-6, theta=0.1)
            return st.solve().efficiencies()

        oT, RT, TT = solve(np.diag([2.25, 2.25, 2.25]).astype(complex))
        oS, RS, TS = solve(2.25)
        assert np.array_equal(oT, oS)
        assert np.max(np.abs(RT - RS)) < 1e-12
        assert np.max(np.abs(TT - TS)) < 1e-12

    def test_rcwastack_uniform_birefringent_conserves_energy(self):
        """A genuinely anisotropic (birefringent) uniform tensor slab solves and
        conserves energy for both incident polarizations (lossless oracle)."""
        from lumenairy.elements.rcwa.stack import RCWAStack
        st = RCWAStack(1e-6, n_superstrate=1.0, n_substrate=1.0, n_orders=5)
        st.add_layer(0.3e-6, eps=np.diag([2.5, 2.1, 2.3]).astype(complex))
        st.set_source(0.5e-6, theta=0.2)
        o, R, T = st.solve().efficiencies()
        assert R[0].sum() + T[0].sum() == pytest.approx(1.0, abs=1e-9)
        assert R[1].sum() + T[1].sum() == pytest.approx(1.0, abs=1e-9)

    def test_rcwastack_add_layer_rejects_bad_shape(self):
        from lumenairy.elements.rcwa.stack import RCWAStack
        st = RCWAStack(1e-6, n_superstrate=1.0, n_substrate=1.5, n_orders=5)
        with pytest.raises(ValueError, match=r"\(3, 3\)"):
            st.add_layer(0.3e-6, eps=np.ones((2, 2)))


# --------------------------------------------------------------------------- #
#  S5-11: apply_transmission companion; port default documented                 #
# --------------------------------------------------------------------------- #
class TestS5_11TransmissionPort:
    def test_apply_transmission_uses_transmission_jones(self):
        """apply_transmission must carry the TRANSMISSION Jones (self._Jt), and
        differ from apply_reflection (self._Jr) on a transmissive stack -- the
        observable a transmit metasurface actually wants."""
        from lumenairy.elements.polarization import JonesField
        from lumenairy.elements.rcwa.stack import RCWAStack
        st = RCWAStack(1e-6, n_superstrate=1.0, n_substrate=1.0, n_orders=5)
        st.add_layer(0.3e-6, eps=6.0)
        st.set_source(0.5e-6, theta=0.0)
        res = st.solve()
        ex = np.ones((4, 4), dtype=complex)
        ey = np.zeros((4, 4), dtype=complex)
        fr = res.apply_reflection(JonesField(ex.copy(), ey.copy(), dx=1e-6))
        ft = res.apply_transmission(JonesField(ex.copy(), ey.copy(), dx=1e-6))
        Jr, Jt = res.jones_reflection(), res.jones_transmission()
        assert np.allclose(fr.Ex, Jr[0, 0] * ex + Jr[0, 1] * ey)
        assert np.allclose(ft.Ex, Jt[0, 0] * ex + Jt[0, 1] * ey)
        # a non-trivial transmit stack: the two ports are genuinely different
        assert not np.allclose(ft.Ex, fr.Ex)


# --------------------------------------------------------------------------- #
#  S1-15: BOR nodal basis warns on large cells (was a silent ~1e29 blow-up)     #
# --------------------------------------------------------------------------- #
class TestS1_15BorNodalGuard:
    @staticmethod
    def _uni(v):
        return lambda r: np.full_like(r, v, dtype=complex)

    def test_nodal_warns_on_large_cell(self):
        from lumenairy.elements.bor.bor_solve import build_layer
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            build_layer(1, 20.0, 60, self._uni(2.0), 2.0, basis="nodal")
        assert any("nodal" in str(x.message) and "1e29" in str(x.message)
                   for x in w)

    def test_nodal_quiet_on_small_cell(self):
        """The documented small-cell floor regime (~1-4%) must NOT warn -- the
        legacy gates run there."""
        from lumenairy.elements.bor.bor_solve import build_layer
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            build_layer(1, 4.0, 60, self._uni(2.0), 2.0, basis="nodal")
        assert not any("nodal" in str(x.message) for x in w)

    def test_staggered_never_warns(self):
        from lumenairy.elements.bor.bor_solve import build_layer
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            build_layer(1, 20.0, 60, self._uni(2.0), 2.0)   # staggered default
        assert not any("nodal" in str(x.message) for x in w)


# --------------------------------------------------------------------------- #
#  S1-17: EME diffraction warns on a structured (non-convergent) layer          #
# --------------------------------------------------------------------------- #
class TestS1_17EmeStructuredGuard:
    def test_uniform_layer_does_not_warn_and_is_exact(self):
        from lumenairy.elements.eme import diffraction_fd
        Lx = Ly = 1.0
        Nx = Ny = 16
        eps_xy = np.full((Nx, Ny), 4.0, dtype=complex)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            res = diffraction_fd(eps_xy, Lx, Ly, Nx, Ny, 4.0, 1.0, 2.25, 0.5,
                                 1, 1)
        assert not any("STRUCTURED" in str(x.message) for x in w)
        assert abs(res["energy"] - 1.0) < 1e-9        # uniform stays exact

    def test_structured_layer_warns(self):
        from lumenairy.elements.eme import diffraction_fd, strips_to_eps_xy
        Nx = 16
        xg = (np.arange(Nx) + 0.5) / Nx * 2.0
        block = np.where((xg >= 0.5) & (xg < 1.3), 6.0, 1.0)
        strips = [(np.full(Nx, 1.0), 0.7), (block, 0.6), (np.full(Nx, 1.0), 0.7)]
        eps_xy = strips_to_eps_xy(strips, 2.0, Nx, 2.0, 48)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            diffraction_fd(eps_xy, 2.0, 2.0, Nx, 48, 5.0, 1.0, 2.25, 0.4, 1, 1,
                           kx0=0.0, ky0=2.0)
        assert any("STRUCTURED" in str(x.message) for x in w)

    def test_structure_classifier(self):
        """The guard's decision logic (shared by both drivers): a single uniform
        strip is not structured; distinct-eps strips are."""
        from lumenairy.elements.eme.eme_diffraction import _eme_layer_is_structured
        assert _eme_layer_is_structured(np.full(8, 2.0)) is False
        assert _eme_layer_is_structured(np.full(8, 1.0), np.full(8, 6.0)) is True

    def test_diffraction_eme_warns_on_structured_strips(self):
        """diffraction_eme emits the warning at entry (before its heavy modal
        scan); catch it regardless of any later solve outcome."""
        from lumenairy.elements.eme import diffraction_eme
        Nx = 12
        xg = (np.arange(Nx) + 0.5) / Nx * 2.0
        block = np.where(xg < 1.0, 6.0, 1.0)
        strips = [(block, 2.0)]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                diffraction_eme(strips, 2.0, Nx, 2.0, 5.0, 1.0, 2.25, 0.4, 1, 1,
                                n_scan=40)
            except Exception:
                pass
        assert any("STRUCTURED" in str(x.message) for x in w)


# --------------------------------------------------------------------------- #
#  S1-20: circular truncation exposed on rcwa_jones_2d and RCWAStack            #
# --------------------------------------------------------------------------- #
def _scalar_cross_cell(Sx=41, Sy=41, lo=2.25, hi=6.0):
    cell = np.full((Sx, Sy), lo, dtype=complex)
    cell[Sx // 4:3 * Sx // 4, Sy // 4:3 * Sy // 4] = hi
    return cell


class TestS1_20CircularTruncation:
    def test_rcwa_jones_2d_circular_is_subset_and_conserves_energy(self):
        from lumenairy.elements.rcwa.twod import _harmonic_orders_2d, rcwa_jones_2d
        cell = _scalar_cross_cell()
        tcell = np.zeros((41, 41, 3, 3), dtype=complex)
        for i in range(3):
            tcell[:, :, i, i] = cell
        px = py = 1e-6
        common = dict(theta=0.0, phi=0.0, n_orders_x=4, n_orders_y=4)
        oR = rcwa_jones_2d(px, py, tcell, 1.5, 1.0, 0.3e-6, 0.5e-6,
                           truncation="rectangular", **common)[0]
        oC, RC, TC, _ = rcwa_jones_2d(px, py, tcell, 1.5, 1.0, 0.3e-6, 0.5e-6,
                                      truncation="circular", **common)
        assert len(oC) < len(oR)                       # strictly fewer harmonics
        oref = _harmonic_orders_2d(4, 4, truncation="circular",
                                   period_x=px, period_y=py)[0]
        assert np.array_equal(np.asarray(oC), oref)     # exactly the Lalanne set
        assert RC[0].sum() + TC[0].sum() == pytest.approx(1.0, abs=1e-9)

    def test_rcwa_jones_2d_circular_uniform_truncation_invariant(self):
        """Independent structural oracle: a UNIFORM (isotropic) tensor cell has
        only the DC harmonic, so the specular (0, 0) order is TRUNCATION-INVARIANT
        -- rectangular and circular must give byte-identical (0, 0) efficiencies
        and Jones -- and the specular order carries all the power (lossless)."""
        from lumenairy.elements.rcwa.twod import rcwa_jones_2d
        S = 21
        tcell = np.zeros((S, S, 3, 3), dtype=complex)
        for i in range(3):
            tcell[:, :, i, i] = 4.0
        px = py = 1e-6

        def run(tr):
            o, R, T, J = rcwa_jones_2d(
                px, py, tcell, 1.5, 1.0, 0.3e-6, 0.5e-6, theta=0.0, phi=0.0,
                n_orders_x=3, n_orders_y=3, truncation=tr)
            i00 = int(np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
            return R[:, i00], T[:, i00], J

        Rr, Tr, Jr = run("rectangular")
        Rc, Tc, Jc = run("circular")
        assert np.max(np.abs(Rr - Rc)) < 1e-12
        assert np.max(np.abs(Tr - Tc)) < 1e-12
        assert np.max(np.abs(Jr - Jc)) < 1e-12
        assert Rr[0] + Tr[0] == pytest.approx(1.0, abs=1e-9)

    def test_rcwa_jones_2d_rejects_bad_truncation(self):
        from lumenairy.elements.rcwa.twod import rcwa_jones_2d
        tcell = np.zeros((21, 21, 3, 3), dtype=complex)
        for i in range(3):
            tcell[:, :, i, i] = 2.25
        with pytest.raises(ValueError, match="rectangular.*circular|truncation"):
            rcwa_jones_2d(1e-6, 1e-6, tcell, 1.5, 1.0, 0.3e-6, 0.5e-6,
                          n_orders_x=2, n_orders_y=2, truncation="oval")

    def test_rcwastack_circular_2d_subset_and_energy(self):
        from lumenairy.elements.rcwa.stack import RCWAStack
        from lumenairy.elements.rcwa.twod import _harmonic_orders_2d
        cell = _scalar_cross_cell()

        def solve(trunc):
            st = RCWAStack(1e-6, period_y=1e-6, n_superstrate=1.0,
                           n_substrate=1.5, n_orders=4, n_orders_y=4,
                           truncation=trunc)
            st.add_layer(0.3e-6, eps_cell=cell)
            st.set_source(0.5e-6, theta=0.0, phi=0.0)
            return st.solve().efficiencies()

        oR = solve("rectangular")[0]
        oC, RC, TC = solve("circular")
        assert len(oC) < len(oR)
        oref = _harmonic_orders_2d(4, 4, truncation="circular",
                                   period_x=1e-6, period_y=1e-6)[0]
        assert np.array_equal(np.asarray(oC), oref)
        assert RC[0].sum() + TC[0].sum() == pytest.approx(1.0, abs=1e-9)

    def test_rcwastack_circular_cache_no_collision(self):
        """A rectangular solve, then a circular solve, then a rectangular solve
        must reproduce the first rectangular result -- truncation is in the mode
        cache key, so the differing order sets never alias."""
        from lumenairy.elements.rcwa.stack import RCWAStack
        cell = _scalar_cross_cell()

        def solve(trunc):
            st = RCWAStack(1e-6, period_y=1e-6, n_superstrate=1.0,
                           n_substrate=1.5, n_orders=4, n_orders_y=4,
                           truncation=trunc)
            st.add_layer(0.3e-6, eps_cell=cell)
            st.set_source(0.5e-6, theta=0.0, phi=0.0)
            return st.solve().efficiencies()

        _, R1, T1 = solve("rectangular")
        solve("circular")
        _, R2, T2 = solve("rectangular")
        assert np.array_equal(R1, R2)
        assert np.array_equal(T1, T2)

    def test_rcwastack_circular_rejected_on_1d(self):
        """Circular is a 2-D concept; on a 1-D stack the inscribed-circle radius
        collapses to (0, 0), so it is rejected rather than silently dropping
        every diffracted order."""
        from lumenairy.elements.rcwa.stack import RCWAStack
        with pytest.raises(ValueError, match="2-D"):
            RCWAStack(1e-6, n_superstrate=1.0, n_substrate=1.5, n_orders=5,
                      truncation="circular")
