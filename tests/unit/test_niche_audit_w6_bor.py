"""Wave-6 adversarial audit of ``lumenairy/elements/bor/**`` -- the
body-of-revolution (axisymmetric) PMM cluster, territory never numerically
validated by the 2026-07-25 adversarial audit.

Two halves.

**Fix pins** (each verified FAILING on a pre-fix ``3a1da2b`` worktree):

* ``W6-B1`` -- LATTICE ANCHOR, now the DEFAULT (``STAGGERED_WALL_ANCHOR``).
  ``_fd_grid_staggered`` used ``h = Rbig/N`` with a "ghost node = 0" outer
  stencil, so the PEC wall of the *production* ``BORStack`` basis sat at
  ``Rbig + h/2``, not ``Rbig``: the transverse wavenumbers converged to
  ``j_{m,n}/(Rbig + h/2)`` and the scheme was FIRST-order (measured p = 0.99,
  -8.3e-3 relative at N = 60) despite the module's documented 2nd order.  The
  fix ``h = Rbig/(N + 0.5)`` puts the ghost node exactly on ``Rbig`` -- second
  order (p = 1.99), 6.5e-7 at N = 60 -- while keeping the exact discrete
  ``curl.grad == 0`` and the machine-precision cascade energy that the
  antisymmetric-ghost alternative destroys (it breaks de Rham to 1.1e-3
  relative and the energy to 2.1e-4).  Shipped opt-in first, then flipped to
  the default by owner decision (2026-07-26) under the better-physics-default
  policy; ``'ghost'`` remains as a documented legacy escape hatch.  The pins
  below lock the defect, the fix, the flip, and the hatch's bit-identity with
  the pre-flip cavity.  The downstream deliberate update lives in
  ``tests/unit/test_audit_bor_grazing_cutoff.py``.
* ``W6-B2`` -- ``guided_modes`` used ABSOLUTE margins (1e-2, 1e-3) on ``q``,
  which has units 1/length: the same fiber written in nanometres has its whole
  guided window below 1e-2, so the function silently returned ``[]``.
* ``W6-B3`` -- an unrecognized ``wall`` fell through to the leaky
  ``'natural'`` wall silently (``wall='PEC'`` bought open-boundary physics).
* ``W6-B4`` -- the staggered path silently IGNORED ``R_pml`` /
  ``inverse_rule=False`` / ``wall='natural'`` (measured bit-identical output):
  a caller asking for an open radial boundary got the closed Dirichlet wall.
* ``W6-B5`` -- ``fourier_bessel`` aliased silently past the grid Nyquist; the
  Parseval power sum over-counted 3.0x at nmax = 250 on an N = 100 grid and
  ``order_power_fractions`` renormalized ``frac`` by that inflated total.
* ``W6-B6`` -- ``BORStack`` validated Rbig/m/N/wavelength/k0/thickness but not
  the half-space indices: ``n_superstrate=0`` returned EMPTY R/T, ``-1.5``
  silently meant ``+1.5``, NaN/inf surfaced as raw LAPACK errors.
* ``W6-B7`` -- ``add_layer`` / ``set_source`` documented "exactly one of" but
  silently applied a precedence and discarded the rest.
* ``W6-B8`` -- the modal LRU's documented "one eig per DISTINCT profile"
  collapsed to a ZERO hit rate once the distinct-profile count exceeded
  ``_MODAL_CACHE_SIZE`` (cyclic access is the LRU worst case): 41 eigs instead
  of 21 for 20 profiles x 2 repetitions, on EVERY solve.
* ``W6-B9`` -- ``_normal_eps`` wrote the inverse-rule pair in place, so a ring
  exactly ONE node wide had the inner interface's harmonic mean clobbered by
  the outer one and the operator depended on the indexing direction.
* ``W6-B10`` -- ``far_field_angles`` with a complex ``eps`` masked by numpy's
  LEXICOGRAPHIC complex comparison and filled ``theta`` from a complex
  ``arcsin`` whose imaginary part was dropped with only a ``ComplexWarning``.
* ``W6-B11`` -- ``bor_solve.build_layer``/``solve`` had none of the thickness
  guards its ``BORStack.add_layer`` sibling has had since P3-10.
* ``W6-B12`` -- ``radial_spectrum`` accepted ``R <= 0`` (returning the ``|R|``
  spectrum) and died on ``n_el = 0`` with a bare ``IndexError``.

**Oracle pins** (new coverage; these PASS pre-fix -- they are the numerical
validation this territory never had, and they gate the fixes):
analytic Bessel/Hankel pairs, ``r dr`` orthonormality under an INDEPENDENT
quadrature, a 2-D transform oracle, the ASM angle map, radial energy /
two-grid flux, layer-absorption + per-mode-amplitude closure, the
``BORStack`` <-> ``bor_solve`` twin, and the PML stretch identity.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import warnings

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.interpolate import BarycentricInterpolator
from scipy.special import jn_zeros, jnp_zeros, jv

import lumenairy.elements.bor.bor_stack as bor_stack_mod
import lumenairy.elements.bor.coupled_radial_eigensolver as cre_mod
from lumenairy.elements.bor.bor_solve import build_layer
from lumenairy.elements.bor.bor_solve import solve as bor_solve_solve
from lumenairy.elements.bor.bor_stack import BORStack
from lumenairy.elements.bor.coupled_radial_eigensolver import (
    _fd_grid_staggered,
    _normal_eps,
    _pml_stretch,
    guided_modes,
    radial_coupled_modes,
)
from lumenairy.elements.bor.farfield import (
    far_field_angles,
    fourier_bessel,
    order_power_fractions,
)
from lumenairy.elements.bor.fiber_oracle import fiber_modes
from lumenairy.elements.bor.radial_eigensolver import radial_spectrum
from lumenairy.elements.bor.zcascade import layer_modes


def _uniform(val):
    return lambda r: np.full_like(r, complex(val), dtype=complex)


def _cell_grid(R, N):
    h = R / N
    return (np.arange(N) + 0.5) * h, h


def _grazing_reproducer(rbig_um, N, scale=1e6):
    """The AUDIT_BOR_PROPAGATING_CUTOFF_ENERGY_2026_07_13 ring grating (um
    scale), radius parametrized so both wall anchors can be compared on the
    same physical cavity."""
    k0 = 2.0 * np.pi / (1.0e-6 * scale)
    s = BORStack(Rbig=rbig_um * 1e-6 * scale, m=1, N=N,
                 n_superstrate=1.41 + 0j, n_substrate=1.41 + 0j)
    s.add_layer(0.5e-6 * scale, rings=(3.0e-6 * scale, 0.5, 2.45 + 0j,
                                       1.41 + 0j))
    s.set_source(k0=k0)
    out = s.solve()
    out["_k0"] = k0
    return out


def _box_gammas(m, eps, R, N, k0):
    """Transverse wavenumbers gamma = sqrt(eps k0^2 - q^2) of the staggered
    homogeneous-cylinder box modes (the analytic set is {j_mn, j'_mn}/R)."""
    L = layer_modes(m, R, N, _uniform(eps), k0, staggered=True)
    g2 = eps * k0 ** 2 - L["q"] ** 2
    sel = (np.abs(g2.imag) < 1e-8 * np.abs(g2.real)) & (g2.real > 1e-9)
    return np.sort(np.sqrt(g2[sel].real))


# ===================================================================== #
#  W6-B1 -- the staggered lattice anchor (PEC wall at Rbig, 2nd order)   #
# ===================================================================== #
@pytest.fixture
def anchor(monkeypatch):
    """Set ``STAGGERED_WALL_ANCHOR`` for the duration of a test."""
    def _set(value):
        monkeypatch.setattr(cre_mod, "STAGGERED_WALL_ANCHOR", value)
    return _set


@pytest.mark.parametrize("N", [17, 60, 128, 301])
@pytest.mark.parametrize("Rbig", [1.0, 4.0, 8.0, 137.0])
def test_w6_b1_wall_radius_per_anchor_setting(Rbig, N, anchor):
    """The staggered outer stencil forces the tangential field to zero at the
    GHOST NODE (index N, radius ``(N + 0.5) h``), so THAT radius is the PEC
    wall.  The legacy ``'ghost'`` hatch puts it half a cell OUTSIDE the
    requested domain; the ``'rbig'`` DEFAULT puts it exactly on ``Rbig``."""
    anchor("ghost")                                    # legacy escape hatch
    r_n, r_f, h, Dn2f, Df2n, An2f, Af2n = _fd_grid_staggered(Rbig, N)
    assert h == pytest.approx(Rbig / N, rel=1e-15)
    assert (N + 0.5) * h == pytest.approx(Rbig + h / 2.0, rel=1e-14)
    assert (N + 0.5) * h > Rbig                        # THE DEFECT
    assert r_f[-1] == pytest.approx(Rbig, rel=1e-14)   # last face == Rbig

    anchor("rbig")                                     # corrected DEFAULT
    r_n2, r_f2, h2, Dn2f2, _Df, An2f2, _Af = _fd_grid_staggered(Rbig, N)
    assert h2 == pytest.approx(Rbig / (N + 0.5), rel=1e-15)
    assert (N + 0.5) * h2 == pytest.approx(Rbig, rel=1e-14, abs=0.0)
    assert r_f2[-1] == pytest.approx(Rbig - h2 / 2.0, rel=1e-13)

    # BOTH settings keep the ghost-value-ZERO stencil -- the convention the
    # exact discrete curl.grad == 0 depends on (see the de Rham pin below).
    for D, A, hh in ((Dn2f, An2f, h), (Dn2f2, An2f2, h2)):
        assert D[N - 1, N - 1] == pytest.approx(-1.0 / hh, rel=1e-14)
        assert A[N - 1, N - 1] == pytest.approx(0.5, rel=1e-14)


@pytest.mark.slow
def test_w6_b1_legacy_anchor_is_first_order_and_default_is_second(anchor):
    """MEASURED, both ways.  Homogeneous PEC cylinder vs the analytic Bessel
    set ``{j_{m,n}, j'_{m,n}}/Rbig``.  The legacy ``'ghost'`` hatch converges
    to ``j/(Rbig + h/2)`` -- FIRST order, -8.3e-3 at N = 60 -- contradicting the
    module docstring's "Convergence is 2nd-order in N (FD)".  The ``'rbig'``
    default is 4 decades better at identical cost."""
    m, eps, R, k0 = 1, 4.0, 8.0, 2.0
    exact = np.sort(np.concatenate([jn_zeros(m, 6) / R, jnp_zeros(m, 6) / R]))

    def sweep():
        out = {}
        for N in (60, 240):
            g = _box_gammas(m, eps, R, N, k0)
            matched = np.array([g[np.argmin(np.abs(g - e))] for e in exact[:4]])
            out[N] = (matched / exact[:4] - 1.0, matched[0])
        return out

    anchor("ghost")
    sh = sweep()
    e60, e240 = sh[60][0], sh[240][0]
    assert np.all(e60 < 0.0) and np.all(e240 < 0.0)      # biased LOW
    assert 5e-3 < abs(e60[0]) < 2e-2, e60                # measured -8.26e-3
    assert 1e-3 < abs(e240[0]) < 5e-3, e240              # measured -2.08e-3
    p_ship = np.log(abs(e60[0]) / abs(e240[0])) / np.log(4.0)
    assert p_ship < 1.2, f"shipped order {p_ship} (measured ~0.99)"
    # the whole error IS the half-cell wall offset: R_eff == Rbig + h/2
    for N in (60, 240):
        R_eff = exact[0] * R / sh[N][1]
        assert abs((R_eff - R) / (0.5 * R / N) - 1.0) < 1e-3, (N, R_eff)

    anchor("rbig")
    rp = sweep()
    f60, f240 = rp[60][0], rp[240][0]
    assert abs(f60[0]) < 1e-5, f60                       # measured 6.52e-7
    assert abs(f240[0]) < 1e-6, f240                     # measured 4.14e-8
    assert np.abs(f60).max() < 2e-3 and np.abs(f240).max() < 2e-4
    p_fix = np.log(abs(f60[0]) / abs(f240[0])) / np.log(4.0)
    assert p_fix > 1.85, f"repaired order {p_fix} (measured 1.99)"
    assert abs(f60[0]) < abs(e60[0]) / 1000.0            # >= 3 decades better


@pytest.mark.slow
def test_w6_b1_both_anchors_keep_de_rham_and_machine_precision_energy(anchor):
    """The two properties that make this basis worth having -- the EXACT
    discrete ``curl.grad == 0`` and machine-precision cascade energy -- hold at
    BOTH anchor settings.  That is the whole reason the repair moves the
    anchor rather than the stencil: the competing antisymmetric-ghost repair
    reaches p = 2.000 too but was measured to break de Rham to 1.05e-3 relative
    and the cascade energy from 1.5e-14 to 2.1e-4."""
    for setting in ("ghost", "rbig"):
        anchor(setting)
        for Rbig, N in ((8.0, 60), (4.0, 150)):
            r_n, r_f, h, Dn2f, Df2n, An2f, Af2n = _fd_grid_staggered(Rbig, N)
            A_n2f = Dn2f + np.diag(1.0 / r_f) @ An2f
            # curl(grad psi) == 0 as a MATRIX identity, boundary row included
            resid = A_n2f @ np.diag(1.0 / r_n) - np.diag(1.0 / r_f) @ Dn2f
            assert np.max(np.abs(resid)) < 1e-12 * np.max(np.abs(A_n2f)), setting
        s = BORStack(Rbig=4.0, m=1, N=60, n_superstrate=1.4142,
                     n_substrate=1.4142)
        s.add_layer(0.5, rings=(0.8, 0.5, 2.449, 1.414))
        s.set_source(k0=2.0)
        res = s.solve()
        assert len(res["R"]) >= 4
        assert np.max(np.abs(res["energy"] - 1.0)) < 1e-11, setting


def test_w6_b1_default_is_the_corrected_anchor():
    """THE FLIP PIN (owner decision, 2026-07-26): the shipped default is the
    CORRECTED anchor, and the default code path is bit-identical to asking for
    ``'rbig'`` explicitly (i.e. the switch has no separate default branch)."""
    assert cre_mod.STAGGERED_WALL_ANCHOR == "rbig"
    for Rbig, N in ((1.0, 17), (8.0, 60), (48.09375, 256)):
        default = _fd_grid_staggered(Rbig, N)
        assert default[2] == Rbig / (N + 0.5)
        assert (N + 0.5) * default[2] == pytest.approx(Rbig, rel=1e-14)


def test_w6_b1_default_path_is_bit_identical_to_explicit_rbig(anchor):
    """Setting the switch to its own default must change nothing, bit for bit
    -- grids, stencils and a full ``BORStack`` solve."""
    ref_grid = _fd_grid_staggered(8.0, 60)
    s = BORStack(Rbig=4.0, m=1, N=60, n_superstrate=1.4142, n_substrate=1.4142)
    s.add_layer(0.5, rings=(0.8, 0.5, 2.449, 1.414))
    s.set_source(k0=2.0)
    ref = s.solve()
    anchor("rbig")                                   # explicit == default
    got_grid = _fd_grid_staggered(8.0, 60)
    for a, b in zip(ref_grid, got_grid):
        assert np.array_equal(a, b) if np.ndim(a) else a == b
    s2 = BORStack(Rbig=4.0, m=1, N=60, n_superstrate=1.4142,
                  n_substrate=1.4142)
    s2.add_layer(0.5, rings=(0.8, 0.5, 2.449, 1.414))
    s2.set_source(k0=2.0)
    got = s2.solve()
    for key in ("R", "T", "q", "energy"):
        assert np.array_equal(np.asarray(ref[key]), np.asarray(got[key])), key


@pytest.mark.slow
def test_w6_b1_legacy_hatch_reproduces_the_pre_flip_cavity_bit_for_bit(anchor):
    """The escape hatch must be exactly that: ``'ghost'`` at radius ``Rbig`` is
    the SAME discretization as the default ``'rbig'`` at ``Rbig + h/2``, since
    that is the cavity the legacy anchor was really simulating.  Verified on
    the AUDIT_BOR_PROPAGATING_CUTOFF_ENERGY geometry (48 um, N = 256, so
    h/2 = 0.09375 um): grids, all four stencils, and all 319 R/T/q/energy
    values are BIT-identical -- which is what lets that file's near-grazing
    gates keep the audit's published numbers (319 orders, min q/k0 = 0.049293,
    fundamental R = 0.146135, shipped-bug 0.145113, 2.28e-2 leak)."""
    Rbig_um, N = 48.0, 256
    anchor("ghost")
    g_legacy = _fd_grid_staggered(Rbig_um, N)
    leg = _grazing_reproducer(Rbig_um, N)
    anchor("rbig")
    g_default = _fd_grid_staggered(Rbig_um + Rbig_um / N / 2.0, N)
    cur = _grazing_reproducer(Rbig_um + Rbig_um / N / 2.0, N)
    for a, b in zip(g_legacy, g_default):
        assert np.array_equal(a, b) if np.ndim(a) else a == b
    for key in ("R", "T", "q", "energy"):
        assert np.array_equal(np.asarray(leg[key]), np.asarray(cur[key])), key
    assert len(cur["R"]) == 319                          # the audit's count


@pytest.mark.slow
def test_w6_b1_flip_cost_on_the_grazing_reproducer_is_pinned(anchor):
    """The measured cost of the flip on the nominal 48 um cavity, i.e. the
    deliberate update carried by ``test_audit_bor_grazing_cutoff.py``:
    319 -> 318 incident orders, fundamental R 0.146135 -> 0.142290, min q/k0
    0.049293 -> 0.051165 -- with energy closure unaffected either way."""
    got = {}
    for setting in ("ghost", "rbig"):
        anchor(setting)
        res = _grazing_reproducer(48.0, 256)
        q = np.asarray(res["q"], float)
        got[setting] = (len(res["R"]),
                        float(np.asarray(res["R"])[int(np.argmax(q))]),
                        float((q / res["_k0"]).min()),
                        float(np.max(np.abs(np.asarray(res["energy"]) - 1.0))))
    assert got["ghost"][0] == 319 and got["rbig"][0] == 318, got
    assert abs(got["ghost"][1] - 0.146135) < 1e-4, got["ghost"]
    assert abs(got["rbig"][1] - 0.142290) < 1e-4, got["rbig"]
    assert abs(got["ghost"][2] - 0.049293) < 1e-5, got["ghost"]
    assert abs(got["rbig"][2] - 0.051165) < 1e-5, got["rbig"]
    for setting, row in got.items():
        assert row[3] < 1e-9, (setting, row)     # measured ~7e-12 both ways


# ===================================================================== #
#  W6-B2 -- guided_modes unit invariance                                #
# ===================================================================== #
@pytest.mark.slow
def test_w6_b2_guided_modes_unit_invariant():
    """The validated V = 4 fiber (m=1, a=1, eps 6/2, k0=2) re-expressed in four
    length units: ``L' = L*scale``, ``k0' = k0/scale``.  Pre-fix the absolute
    ``qlo + 1e-2 < q < qhi - 1e-2`` window was EMPTY for scale >= 1e3 (the whole
    guided window is 2.8e-3..4.9e-3 there), so ``guided_modes`` silently
    returned ``[]`` while the raw spectrum held the right modes; the ANSWER (the
    dimensionless q/oracle ratio) must be the same in every unit system."""
    m, e1, e2, N = 1, 6.0, 2.0, 250
    ratios = {}
    for label, scale in (("micron", 1.0), ("nanometre", 1e3),
                         ("angstrom", 1e4), ("metre", 1e-6)):
        k0, a, Rbig = 2.0 / scale, 1.0 * scale, 8.0 * scale
        q_or = fiber_modes(m, a, e1, e2, k0, n_scan=3000)
        q_or = q_or[q_or > np.sqrt(e2) * k0 * (1 + 1e-9)]
        assert len(q_or) >= 1                  # the oracle is scale-invariant
        gm = guided_modes(m, a, Rbig, N, e1, e2, k0)
        qs = np.array([md["q"].real for md in gm])
        assert len(qs) >= 1, f"guided_modes returned nothing at {label} scale"
        # the nodal-FD + interface-quantization floor on this grid
        assert abs(qs[0] / q_or[0] - 1.0) < 5e-3, (label, qs[0], q_or[0])
        ratios[label] = qs[0] / q_or[0]
    vals = np.array(list(ratios.values()))
    assert np.max(np.abs(vals / vals[0] - 1.0)) < 1e-5, ratios


def test_w6_b2_margin_recast_is_bit_exact_at_the_validated_k0():
    """The recast keeps the historical constants BIT-exact at k0 = 2.0, so the
    micron-scale gate results are unchanged rather than merely close."""
    assert 5e-3 * 2.0 == 1e-2
    assert 5e-4 * 2.0 == 1e-3


# ===================================================================== #
#  W6-B3 / W6-B4 -- wall validation and inert staggered params           #
# ===================================================================== #
@pytest.mark.parametrize("bad", ["PEC", "Natural", "banana", "", 42, 0])
def test_w6_b3_unrecognized_wall_is_rejected(bad):
    """Pre-fix any unrecognized ``wall`` fell through to the leaky
    ``'natural'`` wall, bit-identically -- a typo silently bought
    open-boundary physics instead of the closed box."""
    with pytest.raises(ValueError, match="wall"):
        radial_coupled_modes(1, 8.0, 30, _uniform(2.0), 2.0, wall=bad)
    with pytest.raises(ValueError, match="wall"):
        layer_modes(1, 8.0, 30, _uniform(2.0), 2.0, wall=bad)
    with pytest.raises(ValueError, match="wall"):
        build_layer(1, 8.0, 30, _uniform(2.0), 2.0, wall=bad, basis="nodal")


def test_w6_b3_supported_walls_still_differ():
    """GUARD: validation must not collapse the two walls into one."""
    q = {}
    for w in ("pec", "natural", None):
        q[w] = layer_modes(1, 8.0, 30, _uniform(2.0), 2.0, wall=w)["q"]
    assert np.array_equal(q[None], q["natural"])          # None == nodal default
    assert not np.allclose(np.sort_complex(q["pec"]),
                           np.sort_complex(q["natural"]))


@pytest.mark.parametrize("kwargs", [{"R_pml": 5.0},
                                    {"R_pml": 5.0, "sigma_max": 50.0},
                                    {"inverse_rule": False},
                                    {"wall": "natural"}])
def test_w6_b4_staggered_rejects_nodal_only_params(kwargs):
    """The staggered basis has the closed Dirichlet wall and the FACE inverse
    rule built in.  Pre-fix these kwargs were silently DROPPED (measured:
    bit-identical ``q`` with and without ``R_pml=5.0, sigma_max=50``), so a
    caller asking for an open radial boundary got the closed wall."""
    with pytest.raises(ValueError):
        radial_coupled_modes(1, 8.0, 30, _uniform(2.0), 2.0, staggered=True,
                             **kwargs)


def test_w6_b4_layer_modes_staggered_rejects_pml():
    with pytest.raises(ValueError, match="PML"):
        layer_modes(1, 8.0, 30, _uniform(2.0), 2.0, staggered=True, R_pml=5.0)
    # ... and the plain staggered call is untouched
    L = layer_modes(1, 8.0, 30, _uniform(2.0), 2.0, staggered=True)
    assert L["W"].shape == (60, 60)


# ===================================================================== #
#  W6-B5 -- fourier_bessel aliasing past the grid Nyquist                #
# ===================================================================== #
def test_w6_b5_fourier_bessel_warns_and_the_aliasing_is_real():
    """MEASURED: N = 100 grid, nmax = 250 -> 151 orders above ``pi/h``, the
    Parseval sum reports 3.0x the true field power and the reconstruction is
    100% wrong (L2 residual 2.0).  Pre-fix that was completely silent."""
    m, R, N, w = 1, 4.0, 100, 0.7
    r, h = _cell_grid(R, N)
    f = (r / w) * np.exp(-(r / w) ** 2)
    with pytest.warns(UserWarning, match="Nyquist"):
        c, kt, norm = fourier_bessel(f, r, h, m, 250)
    ratio = np.sum(np.abs(c) ** 2 * norm) / np.sum(np.abs(f) ** 2 * r * h)
    assert ratio > 2.0, ratio                 # the over-count is not subtle
    assert int(np.sum(kt > np.pi / h)) > 100
    # order_power_fractions renormalizes frac, so it still sums to 1 -- which is
    # exactly why the warning (not a self-check) is the fix
    with pytest.warns(UserWarning, match="Nyquist"):
        d = order_power_fractions(f, r, h, m, 2.25, 2.0, 250)
    assert d["frac"].sum() == pytest.approx(1.0, abs=1e-12)


def test_w6_b5_no_warning_when_the_orders_are_resolved():
    m, R, N, w = 1, 4.0, 400, 0.7
    r, h = _cell_grid(R, N)
    f = (r / w) * np.exp(-(r / w) ** 2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fourier_bessel(f, r, h, m, 60)


# ===================================================================== #
#  W6-B6 / W6-B7 -- BORStack builder guards                             #
# ===================================================================== #
@pytest.mark.parametrize("kw", [{"n_superstrate": 0.0},
                                {"n_substrate": 0.0},
                                {"n_superstrate": -1.5},
                                {"n_substrate": -2.0},
                                {"n_superstrate": np.nan},
                                {"n_substrate": np.inf}])
def test_w6_b6_bad_half_space_index_rejected(kw):
    """Pre-fix: ``n = 0`` solved happily and returned EMPTY R/T (0 orders);
    ``n = -1.5`` solved with 5 orders because only ``n**2`` is used, so it
    silently meant ``+1.5``; NaN/inf reached LAPACK as
    ``ValueError: array must not contain infs or NaNs`` / ``OverflowError``
    from deep inside the eigensolve rather than from the builder."""
    with pytest.raises(ValueError, match="n_su"):
        BORStack(Rbig=3.0, m=1, N=30, **kw)


def test_w6_b6_lossy_half_space_still_accepted():
    """GUARD: a complex index with a positive real part is legitimate."""
    s = BORStack(Rbig=3.0, m=1, N=30, n_superstrate=1.5,
                 n_substrate=1.5 + 0.01j)
    assert s.eps_sub.imag != 0.0


def test_w6_b7_add_layer_enforces_exactly_one_profile_spec():
    s = BORStack(Rbig=3.0, m=1, N=30)
    with pytest.raises(ValueError, match="EXACTLY ONE"):
        s.add_layer(0.3, eps=2.0, rings=(0.5, 0.5, 2.0, 1.0))
    with pytest.raises(ValueError, match="EXACTLY ONE"):
        s.add_layer(0.3, eps_profile=_uniform(2.0), eps=9.0)
    with pytest.raises(ValueError, match="EXACTLY ONE"):
        s.add_layer(0.3, eps_profile=_uniform(2.0), rings=(0.5, 0.5, 2.0, 1.0))
    with pytest.raises(ValueError):                     # still need one
        s.add_layer(0.3)
    assert s._layers == []
    s.add_layer(0.3, eps=2.0)                           # and one works
    assert len(s._layers) == 1


def test_w6_b7_set_source_enforces_exactly_one_of_wavelength_k0():
    """Pre-fix ``set_source(wavelength=1.0, k0=7.0)`` silently discarded the
    wavelength and used k0 = 7.0."""
    s = BORStack(Rbig=3.0, m=1, N=30)
    with pytest.raises(ValueError, match="not both"):
        s.set_source(wavelength=1.0, k0=7.0)
    assert s.k0 is None
    s.set_source(wavelength=2 * np.pi)
    assert s.k0 == pytest.approx(1.0)
    s.set_source(k0=3.0)
    assert s.k0 == 3.0


# ===================================================================== #
#  W6-B8 -- the modal LRU interiors, at BOTH cache settings              #
# ===================================================================== #
def _counting_layer_modes(monkeypatch):
    calls = {"n": 0}
    orig = bor_stack_mod.layer_modes

    def counted(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(bor_stack_mod, "layer_modes", counted)
    return calls


def _repeat_stack(n_distinct, repeats, N=30):
    s = BORStack(Rbig=3.0, m=1, N=N, n_superstrate=1.4, n_substrate=1.4)
    for _ in range(repeats):
        for j in range(n_distinct):
            s.add_layer(0.2, eps=2.0 + 0.1 * j)
    s.set_source(k0=2.0)
    return s


@pytest.mark.slow
@pytest.mark.parametrize("cap", [1, 2, 4, 16, 64])
def test_w6_b8_within_solve_dedup_is_cache_size_independent(monkeypatch, cap):
    """The documented promise -- "an ABAB... periodic stack pays one eig per
    DISTINCT profile, not per repetition" -- was FALSE whenever the distinct
    count exceeded ``_MODAL_CACHE_SIZE``: cyclic access is the LRU worst case,
    so the hit rate collapsed to ZERO.  MEASURED pre-fix: 20 distinct profiles
    x 2 repetitions cost 41 eigs (ideal 21) at cap 16, 13 (ideal 5) for 4 x 3 at
    cap 1 -- and the same again on every re-solve."""
    monkeypatch.setattr(bor_stack_mod, "_MODAL_CACHE_SIZE", cap)
    calls = _counting_layer_modes(monkeypatch)
    n_distinct = 20
    s = _repeat_stack(n_distinct, 2)
    calls["n"] = 0
    s.solve()
    assert calls["n"] == n_distinct + 1, (
        f"cap={cap}: {calls['n']} eigs for {n_distinct} distinct profiles "
        f"(+1 shared half-space); ideal {n_distinct + 1}")


@pytest.mark.slow
def test_w6_b8_cache_size_never_changes_the_physics(monkeypatch):
    """GUARD (passes pre-fix): a cache is an optimisation -- R/T must be
    BIT-identical at every ``_MODAL_CACHE_SIZE``, and across re-solves."""
    ref = None
    for cap in (1, 3, 16, 64):
        monkeypatch.setattr(bor_stack_mod, "_MODAL_CACHE_SIZE", cap)
        s = _repeat_stack(4, 3)
        r1 = s.solve()
        r2 = s.solve()
        if ref is None:
            ref = (r1["R"].copy(), r1["T"].copy())
        assert np.array_equal(r1["R"], ref[0])
        assert np.array_equal(r1["T"], ref[1])
        assert np.array_equal(r2["R"], ref[0])


def test_w6_b8_cache_key_covers_every_geometry_and_source_field(monkeypatch):
    """GUARD (passes pre-fix): ``k0``, ``m``, ``Rbig``, ``N`` and the profile
    fingerprint are all in the key, so mutating any of the public attributes
    forces a recompute rather than a stale hit."""
    calls = _counting_layer_modes(monkeypatch)
    s = BORStack(Rbig=3.0, m=1, N=30, n_superstrate=1.4, n_substrate=1.4)
    s.add_layer(0.2, eps=2.0)
    s.add_layer(0.2, eps=3.0)
    s.set_source(k0=2.0)
    calls["n"] = 0
    s.solve()
    assert calls["n"] == 3                       # 2 layers + 1 half-space
    calls["n"] = 0
    s.solve()
    assert calls["n"] == 0                       # all hit
    for mutate in (lambda: s.set_source(k0=2.5), lambda: setattr(s, "m", 2),
                   lambda: setattr(s, "N", 32),
                   lambda: setattr(s, "Rbig", 3.5)):
        mutate()
        calls["n"] = 0
        s.solve()
        assert calls["n"] == 3


# ===================================================================== #
#  W6-B9 -- the nodal inverse rule on back-to-back interfaces            #
# ===================================================================== #
def test_w6_b9_normal_eps_is_mirror_symmetric():
    """A ring exactly ONE node wide has jumps on BOTH sides.  Pre-fix the
    in-place ``en[i] = en[i-1] = ...`` let the outer interface CLOBBER the
    inner one, so ``[2,6,3,3] -> [3,4,4,3]`` while the mirrored profile gave
    ``[3,4,3,3]`` reversed -- the operator depended on the indexing
    direction."""
    for prof in ([2.0, 6.0, 3.0, 3.0], [1.0, 9.0, 4.0, 2.0, 2.0],
                 [2.0, 6.0, 2.0, 6.0, 2.0]):
        e = np.array(prof, dtype=complex)
        fwd = _normal_eps(e)
        mir = _normal_eps(e[::-1])[::-1]
        assert np.array_equal(fwd, mir), (prof, fwd, mir)


def _legacy_normal_eps(eps):
    """The PRE-FIX ``_normal_eps`` verbatim -- the reference the fix must
    reproduce bit-for-bit on every isolated interface."""
    en = (1.0 / eps).copy()
    for i in range(1, len(eps)):
        if eps[i] != eps[i - 1]:
            hm = 2.0 / (1.0 / eps[i] + 1.0 / eps[i - 1])
            en[i] = en[i - 1] = 1.0 / hm
    return 1.0 / en


def test_w6_b9_isolated_interface_is_bit_identical_to_the_legacy_rule():
    """GUARD: every profile the gates exercise (rings two or more nodes wide)
    must be BIT-identical to the historical in-place rule, so the fix changes
    no validated number -- it only removes the clobbering."""
    for a, b in ((2.0, 6.0), (1.0, 12.25), (2.25 + 0.1j, 1.0),
                 (6.0, 1.9993)):
        for e in (np.array([a, a, a, b, b, b], dtype=complex),
                  np.array([a, a, b, b, a, a, a], dtype=complex),
                  np.array([b, b, b, b, a, a], dtype=complex)):
            assert np.array_equal(_normal_eps(e), _legacy_normal_eps(e)), e
    # uniform profile untouched (and identical to the legacy round trip)
    u = np.full(5, 3.0, dtype=complex)
    assert np.array_equal(_normal_eps(u), _legacy_normal_eps(u))
    # ... while the ADJACENT-jump case is exactly where they must differ
    thin = np.array([2.0, 6.0, 3.0, 3.0], dtype=complex)
    assert not np.array_equal(_normal_eps(thin), _legacy_normal_eps(thin))


# ===================================================================== #
#  W6-B10 -- far_field_angles with a complex permittivity                #
# ===================================================================== #
def test_w6_b10_complex_eps_no_complex_warning_and_real_index_convention():
    """Pre-fix a lossy ``eps`` made ``s`` complex: the propagating mask fell
    back to numpy's LEXICOGRAPHIC complex ordering and ``theta`` came from a
    complex ``arcsin`` truncated with only a ``ComplexWarning``."""
    kt = np.array([0.5, 1.0, 2.0, 5.0])
    eps, k0 = 2.25 + 0.1j, 2.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # ComplexWarning would raise
        theta, prop = far_field_angles(kt, eps, k0)
    n_real = np.sqrt(complex(eps)).real
    expect_prop = kt <= n_real * k0
    assert np.array_equal(prop, expect_prop)
    assert theta.dtype == np.float64
    assert np.allclose(theta[prop], np.arcsin(kt[prop] / (n_real * k0)),
                       rtol=0, atol=1e-15)
    assert np.all(np.isnan(theta[~prop]))


def test_w6_b10_real_eps_unchanged():
    """GUARD: bit-identical for a real ``eps`` (``sqrt(x)`` and
    ``sqrt(complex(x)).real`` agree bitwise for positive x)."""
    kt = np.array([0.0, 1.0, 2.0, 2.9699, 4.5])
    a = far_field_angles(kt, 2.25, 2.0)
    b = far_field_angles(kt, complex(2.25), 2.0)
    assert np.array_equal(np.nan_to_num(a[0], nan=-1.0),
                          np.nan_to_num(b[0], nan=-1.0))
    assert np.array_equal(a[1], b[1])


@pytest.mark.parametrize("bad", [0.0, -2.0, np.nan, np.inf, -1.0 + 0.0j])
def test_w6_b10_non_physical_eps_rejected(bad):
    """Pre-fix ``eps = 0`` divided by zero into ``s = inf`` and silently
    reported every order evanescent; ``eps < 0`` gave a purely imaginary index
    and the lexicographic mask."""
    with pytest.raises(ValueError, match="eps"):
        far_field_angles(np.array([1.0]), bad, 2.0)


# ===================================================================== #
#  W6-B11 / W6-B12 -- sibling guard gaps in bor_solve / radial_eigensolver
# ===================================================================== #
@pytest.mark.parametrize("thk", [-0.5, 0.0, np.nan, np.inf])
def test_w6_b11_build_layer_thickness_guard(thk):
    """``BORStack.add_layer`` has guarded this since P3-10 ("a NEGATIVE
    thickness flips exp(iqL) so forward-oriented evanescent modes GROW,
    silently destabilizing the Redheffer cascade"); the ``bor_solve`` twin
    accepted ``thickness=-0.5`` and cascaded it."""
    with pytest.raises(ValueError, match="thickness"):
        build_layer(1, 4.0, 24, _uniform(2.0), 2.0, thickness=thk)


def test_w6_b11_solve_requires_mid_layer_thickness():
    """Pre-fix this died inside ``propagation_smatrix`` as
    ``TypeError: unsupported operand type(s) for *: 'complex' and 'NoneType'``."""
    lay = [build_layer(1, 4.0, 24, _uniform(2.0), 2.0),
           build_layer(1, 4.0, 24, _uniform(6.0), 2.0),      # no thickness
           build_layer(1, 4.0, 24, _uniform(2.0), 2.0)]
    with pytest.raises(ValueError, match="thickness"):
        bor_solve_solve(lay, 2.0)
    with pytest.raises(ValueError, match="half-spaces"):
        bor_solve_solve(lay[:1], 2.0)
    lay[1]["thickness"] = 0.4                                # and it then works
    res = bor_solve_solve(lay, 2.0)
    assert np.max(np.abs(res["energy"] - 1.0)) < 1e-9


@pytest.mark.parametrize("kw", [{"R": -1.0}, {"R": 0.0}, {"R": np.nan},
                                {"n_el": 0}, {"n_el": -2}])
def test_w6_b12_radial_spectrum_domain_guard(kw):
    """Pre-fix ``R = -1`` silently returned the ``|R| = 1`` spectrum (Jacobian,
    ``r`` measure and ``1/r`` stiffness all flip together) and ``n_el = 0`` died
    with a bare ``IndexError`` from the local->global map."""
    args = {"m": 1, "R": 1.0, "degree": 6, "n_el": 4}
    args.update(kw)
    with pytest.raises(ValueError):
        radial_spectrum(args["m"], args["R"], args["degree"], args["n_el"],
                        n_low=2)


# ===================================================================== #
#  ORACLE SUITE -- the numerical validation this territory never had     #
#  (all of these PASS pre-fix; they are new coverage + fix gates)        #
# ===================================================================== #
def _independent_mass_matrix(vecs, r_nodes, degree, n_el, R):
    """``INT_0^R psi_i psi_j r dr`` by barycentric interpolation of the nodal
    values per element + 40-point Gauss-Legendre -- deliberately built WITHOUT
    the module's own Lagrange/quadrature helpers, so it cannot be circular."""
    bnds = np.linspace(0.0, R, n_el + 1)
    p = degree + 1
    xq, wq = leggauss(40)
    nmod = vecs.shape[1]
    M = np.zeros((nmod, nmod))
    for e in range(n_el):
        xl, xr = bnds[e], bnds[e + 1]
        J = 0.5 * (xr - xl)
        rq = 0.5 * (xr + xl) + J * xq
        g0 = e * (p - 1)
        loc = r_nodes[g0:g0 + p]
        vals = np.empty((len(rq), nmod))
        for j in range(nmod):
            vals[:, j] = BarycentricInterpolator(loc, vecs[g0:g0 + p, j])(rq)
        w = wq * J * rq
        M += vals.T @ (w[:, None] * vals)
    return M


@pytest.mark.parametrize("m", [0, 1, 3])
def test_w6_oracle_radial_spectrum_rdr_orthonormal_and_regular(m):
    """ORACLE (1): the classic BOR defect is a missing or doubled ``r`` weight.
    The spectral-element modes are mutually orthogonal under the CORRECT
    ``r dr`` measure (independent quadrature), the eigenvalues are the Bessel
    zeros, and the axis DOF enforces the ``r^|m|`` regularity exactly."""
    R, degree, n_el = 1.0, 8, 6
    ev, vecs, rn = radial_spectrum(m, R, degree, n_el, bc="dirichlet",
                                   n_low=5, return_modes=True)
    M = _independent_mass_matrix(vecs, rn, degree, n_el, R)
    dg = np.sqrt(np.abs(np.diag(M)))
    off = np.abs(M / np.outer(dg, dg) - np.eye(len(dg)))
    assert off.max() < 1e-11, off.max()
    assert np.max(np.abs(ev / (jn_zeros(m, 5) ** 2 / R ** 2) - 1.0)) < 1e-10
    # axis regularity: psi(0) == 0 EXACTLY for m != 0, non-zero for m == 0
    assert rn[0] == 0.0
    if m == 0:
        assert np.all(np.abs(vecs[0, :3]) > 1e-3)
    else:
        assert np.all(vecs[0, :] == 0.0)
    # Neumann (TE) branch hits the derivative zeros
    ev_n = radial_spectrum(m, R, degree, n_el, bc="neumann", n_low=6)
    ref = jnp_zeros(m, 6) ** 2 if m else np.r_[0.0, jnp_zeros(0, 5) ** 2]
    scale = np.maximum(np.abs(ref), 1.0)
    assert np.max(np.abs(ev_n[:len(ref)] - ref) / scale) < 1e-9


def test_w6_oracle_hankel_analytic_gaussian_pair():
    """ORACLE (2): the raw Hankel integral inside ``fourier_bessel`` is
    ``c_n * norm_n``.  For ``f = exp(-r^2/w^2)`` at m = 0 the analytic pair is
    ``(w^2/2) exp(-kt^2 w^2/4)``.  Two ABLATION controls (drop the ``r``, or
    double it) prove the test discriminates the radial weight."""
    m, R, N, w = 0, 6.0, 600, 1.0
    r, h = _cell_grid(R, N)
    f = np.exp(-(r / w) ** 2)
    c, kt, norm = fourier_bessel(f, r, h, m, 30)
    I_num = (c * norm).real
    I_exact = (w ** 2 / 2.0) * np.exp(-kt ** 2 * w ** 2 / 4.0)
    scale = I_exact[0]
    assert np.max(np.abs(I_num - I_exact)) / scale < 1e-4
    for wgt, floor in ((np.ones_like(r), 0.5), (r ** 2, 0.1)):
        I_bad = np.array([np.sum(f * jv(m, kt[n] * r) * wgt * h)
                          for n in range(5)])
        assert np.max(np.abs(I_bad - I_exact[:5])) / scale > floor


@pytest.mark.slow
@pytest.mark.parametrize("m", [0, 1, 2, 3])
def test_w6_oracle_hankel_vs_independent_2d_quadrature(m):
    """ORACLE (2b): the 2-D transform identity
    ``FT2{f(r) e^{i m phi}}(kt) = 2 pi (-i)^m e^{i m phi_k} INT f J_m(kt r) r dr``
    evaluated by direct 2-D Cartesian quadrature at the EXACT ``kt_n`` -- an
    oracle that shares no code with the 1-D radial kernel."""
    Ng, Lg = 420, 20.0
    dx = Lg / Ng
    xs = (np.arange(Ng) - (Ng - 1) / 2.0) * dx
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    rho, phi = np.hypot(X, Y), np.arctan2(Y, X)

    def prof(t):
        return (t ** m) * np.exp(-t ** 2)

    F2 = prof(rho) * np.exp(1j * m * phi)
    Rm, Nm, nord = 10.0, 700, 8
    rm, hm = _cell_grid(Rm, Nm)
    c, ktm, norm = fourier_bessel(prof(rm), rm, hm, m, nord)
    I_mod = (c * norm).real
    I_2d = np.array([(np.sum(F2 * np.exp(-1j * kt * X)) * dx ** 2
                      / (2 * np.pi * (-1j) ** m)).real for kt in ktm])
    assert np.max(np.abs(I_mod - I_2d)) / np.abs(I_mod).max() < 2e-4


def test_w6_oracle_parseval_and_roundtrip_on_a_non_basis_function():
    """ORACLE (3): radial energy.  Parseval for the discrete scheme ACTUALLY
    used (midpoint ``r dr``), on a function that is NOT a basis member, plus
    the transform/inverse round trip to the quadrature floor."""
    m, R, N, w = 1, 4.0, 500, 0.7
    r, h = _cell_grid(R, N)
    f = (r / w) * np.exp(-(r / w) ** 2)
    alpha = jn_zeros(m, 60)
    c, kt, norm = fourier_bessel(f, r, h, m, 60)
    lhs = np.sum(np.abs(f) ** 2 * r * h)
    assert abs(np.sum(np.abs(c) ** 2 * norm) / lhs - 1.0) < 1e-8
    rec = np.sum([c[n].real * jv(m, alpha[n] * r / R) for n in range(60)],
                 axis=0)
    assert np.linalg.norm(rec - f) / np.linalg.norm(f) < 1e-5
    # the discrete power itself matches the analytic integral
    true_pow = quad(lambda x: ((x / w) * np.exp(-(x / w) ** 2)) ** 2 * x,
                    0.0, R, limit=400, epsabs=1e-15)[0]
    assert abs(lhs / true_pow - 1.0) < 1e-6


def test_w6_oracle_far_field_angle_matches_asm_kz():
    """ORACLE (4): the module's ``kt -> theta`` map must be the same angle the
    validated 2-D angular-spectrum propagator's ``kz`` implies:
    ``theta = arcsin(kt/k) == arccos(kz/k)`` with ``kz = sqrt(k^2 - kt^2)``."""
    R, wl, m = 40e-6, 1.0e-6, 1
    k = 2 * np.pi / wl
    for n_ord in (0, 3, 6):
        kt = jn_zeros(m, 8)[n_ord] / R
        theta, prop = far_field_angles(np.array([kt]), 1.0, k)
        kz = np.sqrt(k ** 2 - kt ** 2)
        assert prop[0]
        assert abs(theta[0] - np.arccos(kz / k)) < 1e-12
    # a single FB order decomposes back onto itself
    Nr = 400
    rr, hr = _cell_grid(R, Nr)
    alpha = jn_zeros(m, 8)
    c, ktv, norm = fourier_bessel(jv(m, alpha[3] * rr / R), rr, hr, m, 8)
    frac = np.abs(c) ** 2 * norm / np.sum(np.abs(c) ** 2 * norm)
    assert frac.argmax() == 3
    assert frac[3] > 1 - 1e-12


@pytest.mark.slow
def test_w6_oracle_two_grid_flux_normalisation_is_load_bearing():
    """ORACLE (3b): every kept propagating staggered mode carries UNIT z-flux
    under the two-grid ``r dr`` quadrature -- and the single-grid quadrature is
    measurably wrong (the P3-14 half-cell error), so the pin discriminates."""
    m, Rbig, N, k0, eps = 1, 4.0, 60, 2.0, 2.0
    L = layer_modes(m, Rbig, N, _uniform(eps), k0, staggered=True)
    W, V, q = L["W"], L["V"], L["q"]
    wq_f, wq_n = np.real(L["wq_face"]), np.real(L["wq_node"])
    good = np.real(np.sum(W[:N] * np.conj(V[N:]) * wq_f[:, None], axis=0)
                   - np.sum(W[N:] * np.conj(V[:N]) * wq_n[:, None], axis=0))
    bad = np.real(np.sum(W[:N] * np.conj(V[N:]) * wq_n[:, None], axis=0)
                  - np.sum(W[N:] * np.conj(V[:N]) * wq_n[:, None], axis=0))
    qn = q / k0
    keep = ((np.abs(qn.imag) < 5e-5) & (qn.real > 1e-6)
            & (np.sqrt(eps) - qn.real > -5e-10))
    assert keep.sum() >= 4
    assert np.max(np.abs(good[keep] - 1.0)) < 1e-12
    assert np.max(np.abs(bad[keep] - 1.0)) > 1e-3
    # the exposed grids are the ones the rows actually live on
    assert np.allclose(L["r_face"], L["r"] + (L["r"][1] - L["r"][0]) / 2.0)


@pytest.mark.slow
def test_w6_oracle_layer_absorption_and_amplitude_closure():
    """ORACLE: ``R + T + sum_i A_i == 1`` per incident order (machine
    precision), ``A_i == 0`` for a lossless stack, and
    ``sum |per_mode_amplitudes|^2 == R`` / ``T``."""
    for eps_mid, lossless in (((2.449 + 0j) ** 2, True),
                              ((2.449 + 0.05j) ** 2, False)):
        s = BORStack(Rbig=4.0, m=1, N=60, n_superstrate=1.4142,
                     n_substrate=1.4142)
        s.add_layer(0.5, rings=(0.8, 0.5, np.sqrt(eps_mid), 1.414))
        s.add_layer(0.3, eps=eps_mid)
        s.set_source(k0=2.0)
        res = s.solve(retain_internal=True)
        A = s.layer_absorption()
        assert A.shape == (2, len(res["R"]))
        assert np.max(np.abs(res["R"] + res["T"] + A.sum(axis=0) - 1.0)) < 1e-11
        if lossless:
            assert np.max(np.abs(A)) < 1e-11
        else:
            assert A.sum(axis=0).min() > 1e-3
    s = BORStack(Rbig=4.0, m=1, N=60, n_superstrate=1.4142, n_substrate=1.6)
    s.add_layer(0.5, rings=(0.8, 0.5, 2.449, 1.414))
    s.set_source(k0=2.0)
    res = s.solve()
    for port, ref in (("reflection", res["R"]), ("transmission", res["T"])):
        d = s.per_mode_amplitudes(port)
        assert np.max(np.abs(np.sum(np.abs(d["amplitude"]) ** 2, axis=0)
                             - ref)) < 1e-12
    d = s.per_mode_amplitudes("reflection")
    inc = res["inc"]
    assert np.max(np.abs(np.diag(d["amplitude"])
                         - res["S"][0][inc, inc])) < 1e-14


@pytest.mark.slow
def test_w6_oracle_borstack_bor_solve_staggered_twin_parity():
    """ORACLE: the two staggered cascade drivers must agree exactly (measured
    max|dR| = max|dT| = 0.0) -- they share the modal basis and the S-matrix
    primitives, so any divergence is a wiring bug in one of them."""
    Rbig, m, N, k0, n_h = 4.0, 1, 60, 2.0, 1.4142

    def ring_eps(r):
        return np.where((r % 0.8) < 0.4, complex(2.449) ** 2,
                        complex(1.414) ** 2).astype(complex)

    s = BORStack(Rbig=Rbig, m=m, N=N, n_superstrate=n_h, n_substrate=n_h)
    s.add_layer(0.5, eps_profile=ring_eps)
    s.set_source(k0=k0)
    r1 = s.solve()
    lay = [build_layer(m, Rbig, N, _uniform(complex(n_h) ** 2), k0),
           build_layer(m, Rbig, N, ring_eps, k0, thickness=0.5),
           build_layer(m, Rbig, N, _uniform(complex(n_h) ** 2), k0)]
    r2 = bor_solve_solve(lay, k0)
    assert np.array_equal(r1["inc"], r2["inc"])
    assert np.max(np.abs(r1["R"] - r2["R"])) < 1e-13
    assert np.max(np.abs(r1["T"] - r2["T"])) < 1e-13


def test_w6_oracle_pml_stretch_is_the_identity_in_the_physical_region():
    """ORACLE: ``rt = INT_0^r s dr'`` must equal ``r`` wherever sigma == 0, or a
    bound mode would feel the absorber."""
    N, Rbig, R_pml = 200, 10.0, 6.0
    h = Rbig / N
    r = (np.arange(N) + 0.5) * h
    sinv, rt = _pml_stretch(r, h, R_pml, Rbig, 5.0, 2)
    phys = r <= R_pml
    assert np.max(np.abs(rt[phys] - r[phys])) < 1e-13 * Rbig
    assert np.all(np.abs(sinv[phys] - 1.0) < 1e-15)
    assert rt[-1].imag > 1.0                       # absorption downstream
