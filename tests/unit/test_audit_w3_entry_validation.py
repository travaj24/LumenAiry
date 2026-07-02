"""Wave-3 audit fixes: ENTRY-VALIDATION cluster (guard-mirroring drift).

Pins the v5.17.0 deep-audit findings where one sibling validated an input and
another did not:

* P3-29 -- every 1-D PMM entry silently aliased ``|angle| >= pi/2`` to the
  supplementary front-side angle (``sin`` symmetry): now ``ValueError``.
* P3-30 -- ``PMMStack.add_layer`` accepted ``thickness <= 0`` / NaN while
  ``PMM2DStack.add_layer`` validated: now mirrors the 2-D guard.
* P3-31 -- the covariant (uniform-slant) dispatch returned before the
  ``retain_internal`` / ``stabilize`` handling: ``stabilize`` is now validated
  eagerly, ``retain_internal=True`` raises ``NotImplementedError``, and
  ``stabilize='slices'`` is honoured (consensus check runs after the solve).
* P3-36 -- ``pmm_efficiency_2d_staggered`` crashed with a deep cryptic
  ``AssertionError`` for non-square cells: now a clear entry ``ValueError``
  naming the restriction and the ``pmm_efficiency_2d_cell`` alternative.
* P3-10 -- ``BORStack`` had no input validation (Rbig/N/m/thickness/
  wavelength/k0): now mirrors the PMM stacks' builder guards.

JAX-path guards are trace-safe: a CONCRETE jax value is checked, a TRACED
tracer skips the guard (the rcwa ``_reject_jax_offplane`` carve-out).  The
jax-runtime tests importorskip and run only where jax is installed.
"""
import warnings

import numpy as np
import pytest

from lumenairy.elements.bor import BORStack
from lumenairy.elements.pmm import (
    PMMStack,
    pmm_efficiency_1d,
    pmm_efficiency_2d_staggered,
    pmm_jones_1d,
)
from lumenairy.elements.pmm.oned import _resolve_incidence_checked

# --------------------------------------------------------------------------- #
# P3-29: 1-D PMM back-side incidence angle
# --------------------------------------------------------------------------- #
_EFF_KW = dict(period=1.0e-6, n_ridge=2.0, n_groove=1.0, n_substrate=1.5,
               n_superstrate=1.0, depth=0.4e-6, duty_cycle=0.5,
               wavelength=0.8e-6, degree=8, stabilize=False)


class TestP329BacksideAngle:
    @pytest.mark.parametrize("bad", [1.75, -1.75, np.pi / 2, -np.pi / 2, 3.0])
    def test_efficiency_rejects_backside_angle(self, bad):
        with pytest.raises(ValueError, match=r"\|angle\| < pi/2"):
            pmm_efficiency_1d(angle=bad, **_EFF_KW)

    def test_theta_alias_also_checked(self):
        with pytest.raises(ValueError, match=r"pmm_jones_1d: incidence angle"):
            pmm_jones_1d(1.0e-6, 4.0, 1.0, 1.5, 1.0, 0.4e-6, 0.5, 0.8e-6,
                         theta=2.0, degree=8, stabilize=False)

    def test_nan_angle_rejected(self):
        with pytest.raises(ValueError, match=r"\|angle\| < pi/2"):
            pmm_efficiency_1d(angle=float("nan"), **_EFF_KW)

    def test_valid_oblique_angle_still_solves(self):
        orders, R, T = pmm_efficiency_1d(angle=0.3, **_EFF_KW)
        tot = R.sum() + T.sum()
        assert np.isfinite(tot) and abs(tot - 1.0) < 1e-2

    def test_steep_but_frontside_angle_still_solves(self):
        # composes with (does not swallow) the grazing guard: 1.5 rad is
        # steep but propagating, so it must SOLVE, not raise.
        orders, R, T = pmm_efficiency_1d(angle=1.5, **_EFF_KW)
        assert np.isfinite(R.sum() + T.sum())

    def test_helper_error_names_the_entry_point(self):
        with pytest.raises(ValueError, match=r"^some_entry: incidence angle"):
            _resolve_incidence_checked("some_entry", 1.75, None)

    def test_helper_passes_non_numeric_through(self):
        # a non-numeric angle is left for the solver's own coercion to raise on
        assert _resolve_incidence_checked("f", "not-a-number", None) == \
            "not-a-number"


# --------------------------------------------------------------------------- #
# P3-30: PMMStack.add_layer thickness guard (mirror of PMM2DStack)
# --------------------------------------------------------------------------- #
class TestP330StackThickness:
    @pytest.mark.parametrize("bad", [-0.3e-6, 0.0, float("nan"), float("inf")])
    def test_rejects_nonpositive_or_nonfinite(self, bad):
        st = PMMStack(period=1.0e-6)
        with pytest.raises(ValueError,
                           match="PMMStack.add_layer: thickness must be > 0"):
            st.add_layer(bad, eps=4.0)

    def test_valid_thickness_accepted(self):
        st = PMMStack(period=1.0e-6)
        st.add_layer(0.3e-6, eps=4.0)
        assert len(st._layers) == 1

    def test_message_mirrors_2d_sibling(self):
        from lumenairy.elements.pmm import PMM2DStack
        with pytest.raises(ValueError, match="thickness must be > 0") as e1:
            PMMStack(period=1.0e-6).add_layer(-1e-7, eps=4.0)
        with pytest.raises(ValueError, match="thickness must be > 0") as e2:
            PMM2DStack(period_x=1.0e-6, period_y=1.0e-6).add_layer(
                -1e-7, eps=4.0)
        # same message family, differing only in the class prefix
        assert str(e1.value).split(":", 1)[1] == str(e2.value).split(":", 1)[1]


# --------------------------------------------------------------------------- #
# P3-31: covariant dispatch kwarg handling
# --------------------------------------------------------------------------- #
def _covariant_stack():
    st = PMMStack(period=1.0e-6, degree=6)
    st.add_layer(0.3e-6, segments=[(0.5, 4.0), (0.5, 1.0)], slant_angle=0.3)
    st.set_source(wavelength=0.8e-6, angle=0.1)
    return st


class TestP331CovariantKwargs:
    def test_bogus_stabilize_raises_eagerly(self):
        st = _covariant_stack()
        with pytest.raises(ValueError,
                           match="stabilize must be None or 'slices'"):
            st.solve(stabilize="bogus_value")

    def test_retain_internal_raises_not_implemented(self):
        st = _covariant_stack()
        with pytest.raises(NotImplementedError,
                           match=r"retain_internal=True"):
            st.solve(retain_internal=True)

    def test_stabilize_slices_is_honoured(self):
        # no taper recipe on a hand-built stack -> the consensus check runs
        # and WARNS that it was skipped (previously: silently ignored).
        st = _covariant_stack()
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            orders, R, T, jones = st.solve(stabilize="slices")
        assert any("no taper builder" in str(w.message) for w in wlist)
        assert np.isfinite(R.sum() + T.sum())

    def test_plain_covariant_solve_unaffected(self):
        st = _covariant_stack()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            orders, R, T, jones = st.solve()
        assert np.isfinite(R.sum() + T.sum())


# --------------------------------------------------------------------------- #
# P3-36: staggered non-square eps_cell
# --------------------------------------------------------------------------- #
class TestP336StaggeredSquareCell:
    @pytest.mark.parametrize("cell", [np.array([[4.0], [1.0]]),
                                      np.ones((3, 2)) * 2.0])
    def test_non_square_cell_raises_clear_valueerror(self, cell):
        with pytest.raises(ValueError, match=r"eps_cell must be SQUARE"):
            pmm_efficiency_2d_staggered(1.2, 1.2, cell, 1.5, 1.0, 0.3, 1.0,
                                        degree=6)

    def test_message_names_the_alternative(self):
        with pytest.raises(ValueError,
                           match="pmm_efficiency_2d_cell") as exc:
            pmm_efficiency_2d_staggered(1.2, 1.2, np.array([[4.0], [1.0]]),
                                        1.5, 1.0, 0.3, 1.0, degree=6)
        assert "pmm_efficiency_2d_staggered" in str(exc.value)

    def test_square_cell_still_solves(self):
        cell = np.array([[4.0, 1.0], [1.0, 1.0]])
        orders, R, T = pmm_efficiency_2d_staggered(1.2, 1.2, cell, 1.5, 1.0,
                                                   0.3, 1.0, degree=4)
        assert np.isfinite(R.sum() + T.sum())


# --------------------------------------------------------------------------- #
# P3-10: BORStack input validation
# --------------------------------------------------------------------------- #
class TestP310BorValidation:
    @pytest.mark.parametrize("kw, msg", [
        (dict(Rbig=-4.0, m=1, N=80), "Rbig"),
        (dict(Rbig=0.0, m=1, N=80), "Rbig"),
        (dict(Rbig=float("nan"), m=1, N=80), "Rbig"),
        (dict(Rbig=4.0, m=1.5, N=80), "m .azimuthal order. must be an integer"),
        (dict(Rbig=4.0, m=1, N=1), "N .radial grid points."),
        (dict(Rbig=4.0, m=1, N=80.5), "N .radial grid points."),
    ])
    def test_constructor_rejects_bad_inputs(self, kw, msg):
        with pytest.raises(ValueError, match=msg):
            BORStack(**kw)

    @pytest.mark.parametrize("bad", [-0.5, 0.0, float("nan")])
    def test_add_layer_rejects_nonpositive_thickness(self, bad):
        s = BORStack(Rbig=4.0, m=1, N=80)
        with pytest.raises(ValueError,
                           match="BORStack.add_layer: thickness must be > 0"):
            s.add_layer(bad, eps=6.0)

    @pytest.mark.parametrize("kw, msg", [
        (dict(wavelength=-1.0), "wavelength must be > 0"),
        (dict(wavelength=0.0), "wavelength must be > 0"),
        (dict(k0=-2.0), "k0 must be > 0"),
        (dict(k0=0.0), "k0 must be > 0"),
        (dict(), "give wavelength or k0"),
    ])
    def test_set_source_rejects_bad_source(self, kw, msg):
        s = BORStack(Rbig=4.0, m=1, N=80)
        with pytest.raises(ValueError, match=msg):
            s.set_source(**kw)

    def test_valid_stack_still_builds_and_solves(self):
        s = BORStack(Rbig=4.0, m=1, N=60, n_superstrate=1.4142,
                     n_substrate=1.4142)
        s.add_layer(0.5, rings=(0.8, 0.5, 2.449, 1.414))
        s.set_source(wavelength=2 * np.pi / 2.0)
        res = s.solve()
        assert len(res["R"]) > 0 and np.all(np.isfinite(res["energy"]))
        assert np.max(res["energy"]) <= 1.0 + 1e-6

    def test_m_zero_and_integer_float_accepted(self):
        # m may legitimately be 0 or a whole-number float; N likewise.
        s = BORStack(Rbig=4.0, m=0, N=60)
        assert s.m == 0
        s2 = BORStack(Rbig=4.0, m=2.0, N=60.0)
        assert s2.m == 2 and s2.N == 60


# --------------------------------------------------------------------------- #
# JAX-runtime tests (trace-safety of the concrete-only guards); skip w/o jax
# --------------------------------------------------------------------------- #
class TestJaxTraceSafety:
    def test_concrete_jax_angle_is_checked(self):
        jnp = pytest.importorskip("jax.numpy")
        with pytest.raises(ValueError, match=r"\|angle\| < pi/2"):
            _resolve_incidence_checked("f", jnp.asarray(1.75), None)

    def test_traced_angle_skips_the_guard(self):
        jax = pytest.importorskip("jax")
        import jax.numpy as jnp

        def f(a):
            # under trace the guard must SKIP (no concrete value): the
            # returned angle feeds sin() untouched.
            return jnp.sin(_resolve_incidence_checked("f", a, None))

        out = jax.jit(f)(jnp.asarray(1.75))   # traced: no raise
        assert np.isfinite(float(out))

    def test_concrete_jax_thickness_is_checked(self):
        jnp = pytest.importorskip("jax.numpy")
        st = PMMStack(period=1.0e-6)
        with pytest.raises(ValueError,
                           match="PMMStack.add_layer: thickness must be > 0"):
            st.add_layer(jnp.asarray(-0.3e-6), eps=4.0)

    def test_traced_thickness_skips_the_guard(self):
        jax = pytest.importorskip("jax")
        import jax.numpy as jnp

        def build(t):
            st = PMMStack(period=1.0e-6)
            st.add_layer(t, eps=4.0)          # tracer: guard must skip
            return t

        jax.jit(build)(jnp.asarray(0.3e-6))   # must not raise
