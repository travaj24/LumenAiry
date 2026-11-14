"""``RCWAStack.add_layer(formulation='auto')`` -- the default factorization.

``'auto'`` resolves per layer: ``'li'`` for a patterned isotropic layer of a
1-D stack (``eps_cell``; a y-invariant ``shapes`` / ``segments`` list), ``'li'``
on a 2-D stack only for a METALLIC layer that ``'li'`` supports, and
``'laurent'`` for everything else (uniform and tensor layers, y-varying
layers of a 1-D stack, dielectric 2-D layers, disks / ellipses, dispersive
2-D layers, a traced JAX cell, an exact zero permittivity).  The resolved
value is ``layers[i].formulation``; the request is
``layers[i].formulation_requested``.

Fixture A: a resonant metal-strip stack -- period 370 nm, a 175 nm wide,
50 nm thick strip (eps -74+6i) in air over 1 / 7 / 5 / 7 / 1 nm of
eps 3.05 / 3.53 / 2.5+0.5i / 3.53 / 3.05 over a 70 nm mirror (eps -160+30i),
air above, n = 1.447 below, 1300 nm, TM.  Its converged zeroth-order
reflection ``r_ref = 0.0571572 + 0.1312207i`` comes from a geometrically
graded ``PMMStack`` (degree 20, 12 elements) and agrees with an independent
finite-element solve to about 1e-5.  Measured 2026-11-14 (Windows/MKL):
``'li'`` errs 1.06e-2 at N = 100 against 1.67e-1 for ``'laurent'``
(1.0553e-2 on both Windows/MKL and WSL/OpenBLAS).
"""
import warnings

import numpy as np
import pytest

import lumenairy as lum
from lumenairy.elements.rcwa import RCWAStack
from lumenairy.elements.rcwa import stack as _stmod
from lumenairy.elements.rcwa.stack import RCWAYAverageWarning

WL, P, W = 1300e-9, 370e-9, 175e-9
AU, AL, AO, HF, OX, NS = -74 + 6j, -160 + 30j, 3.05, 3.53, 2.5 + 0.5j, 1.447
LAYERS = [(50e-9, AU, 1.0), (1e-9, AO, AO), (7e-9, HF, HF), (5e-9, OX, OX),
          (7e-9, HF, HF), (1e-9, AO, AO), (70e-9, AL, AL)]
R_REF = 0.0571572 + 0.1312207j
STRIP = [dict(shape="rectangle", eps=AU, center=(W / 2, P / 2), size=(W, P))]


def _fix_a(N, formulation=None, theta=0.0):
    st = RCWAStack(P, n_superstrate=1.0, n_substrate=NS, n_orders=N)
    kw = {} if formulation is None else dict(formulation=formulation)
    for t, es, eg in LAYERS:
        if es == eg:
            st.add_layer(t, eps=es)
        else:
            st.add_layer(t, shapes=STRIP, eps_background=eg, **kw)
    return st.set_source(WL, theta=theta)


def _bits(res):
    o, R, T = res.efficiencies()
    return tuple(np.asarray(a).tobytes() for a in
                 (R, T, res.jones_reflection(), res.jones_transmission()))


# ---------------------------------------------------------------------------
# 1-3: what the default computes
# ---------------------------------------------------------------------------
def test_default_is_li_on_the_metal_strip_and_much_closer():
    """The default call IS the explicit 'li' call (same code path, bit for
    bit), and it is 16x closer to the converged value than the direct rule
    at N = 100 (measured 1.06e-2 against 1.67e-1)."""
    st = _fix_a(100)
    assert st.layers[0].formulation == "li"
    assert st.layers[0].formulation_requested == "auto"
    r_def = st.solve()
    r_li = _fix_a(100, "li").solve()
    assert _bits(r_def) == _bits(r_li)
    err = abs(complex(np.asarray(r_def.jones_reflection())[0, 0]) - R_REF)
    assert err < 1.2e-2
    r_la = _fix_a(100, "laurent").solve()
    err_la = abs(complex(np.asarray(r_la.jones_reflection())[0, 0]) - R_REF)
    assert err_la > 0.15
    assert err_la > 10.0 * err


def test_te_is_unchanged_by_the_default():
    """E_y is tangential to every wall of a 1-D grating, so Li's rule is the
    direct rule there: TE moves only at round-off (measured at N = 50:
    2.0e-13 Windows/MKL, 2.9e-13 WSL/OpenBLAS; bar 1e-11)."""
    Jd = np.asarray(_fix_a(50).solve().jones_reflection())
    Jl = np.asarray(_fix_a(50, "laurent").solve().jones_reflection())
    assert abs(Jd[1, 1] - Jl[1, 1]) < 1e-11
    # ... while TM moves by O(0.1): the change is the point of the default
    assert abs(Jd[0, 0] - Jl[0, 0]) > 0.1


def test_uniform_only_stack_is_bit_identical_to_laurent():
    def build(**kw):
        st = RCWAStack(P, n_superstrate=1.0, n_substrate=NS, n_orders=5)
        for t, es, _eg in LAYERS[1:]:
            st.add_layer(t, eps=es, **kw)
        return st.set_source(WL, theta=0.2)
    a, b = build(), build(formulation="laurent")
    assert all(L.formulation == "laurent" for L in a.layers)
    assert all(L.formulation_requested == "auto" for L in a.layers)
    assert _bits(a.solve()) == _bits(b.solve())


# ---------------------------------------------------------------------------
# 4: the resolution table, one row per assertion
# ---------------------------------------------------------------------------
def _one(stack, **kw):
    stack.add_layer(0.1e-6, **kw)
    L = stack.layers[-1]
    return L.formulation, L.formulation_requested


def _s1():
    return RCWAStack(1e-6, n_orders=4)


def _s2():
    return RCWAStack(1e-6, period_y=1e-6, n_orders=3, n_orders_y=3)


_X = (np.arange(24) + 0.5) / 24
_LINE = np.where(_X < 0.4, 4.0, 1.0).astype(complex)
_METAL_LINE = np.where(_X < 0.4, -20 + 2j, 1.0).astype(complex)
_PILLAR = np.where((_X[:, None] < 0.4) & (_X[None, :] < 0.5), 4.0, 1.0
                   ).astype(complex)
_METAL_PILLAR = np.where((_X[:, None] < 0.4) & (_X[None, :] < 0.5),
                         -20 + 2j, 1.0).astype(complex)
_RECT = [dict(shape="rectangle", eps=-20 + 2j, center=(0.3e-6, 0.5e-6),
              size=(0.4e-6, 0.3e-6))]
_DIEL_RECT = [dict(shape="rectangle", eps=4.0, center=(0.3e-6, 0.5e-6),
                   size=(0.4e-6, 0.3e-6))]
_DISK = [dict(shape="disk", eps=-20 + 2j, center=(0.5e-6, 0.5e-6),
              radius=0.3e-6)]
_STRIPE = [dict(shape="rectangle", eps=-20 + 2j, center=(0.3e-6, 0.5e-6),
                size=(0.4e-6, 1e-6))]


def test_resolution_table_1d():
    assert _one(_s1(), eps_cell=_LINE) == ("li", "auto")
    assert _one(_s1(), eps_cell=_METAL_LINE) == ("li", "auto")
    assert _one(_s1(), shapes=_STRIPE, eps_background=1.0) == ("li", "auto")
    assert _one(_s1(), segments=[(0.4, 4.0), (0.6, 1.0)]) == ("li", "auto")
    # y-varying layers of a 1-D stack: the y-average contract keeps 'laurent'
    # (bit-identical to pre-averaging the cell) and still warns
    with pytest.warns(RCWAYAverageWarning):
        assert _one(_s1(), shapes=_DISK, eps_background=1.0) == (
            "laurent", "auto")
    with pytest.warns(RCWAYAverageWarning):
        assert _one(_s1(), eps_cell=_PILLAR) == ("laurent", "auto")
    # an exact zero: the inverse rule would read 1/0
    assert _one(_s1(), shapes=_STRIPE, eps_background=0.0) == (
        "laurent", "auto")
    zero_cell = np.where(_X < 0.4, 4.0, 0.0).astype(complex)
    assert _one(_s1(), eps_cell=zero_cell) == ("laurent", "auto")
    # dispersive 1-D cell: 'li' (inspected per wavelength)
    assert _one(_s1(), eps_cell=lambda wl: _LINE) == ("li", "auto")
    # uniform / tensor ignore the setting
    assert _one(_s1(), eps=2.25) == ("laurent", "auto")
    assert _one(_s1(), eps=2.25 * np.eye(3)) == ("laurent", "auto")
    # the companion pair is read only by the inverse rule
    assert _one(_s1(), eps_cell=_LINE,
                eps_cell_normal=(_LINE, _LINE)) == ("li", "auto")
    # explicit requests pass through ('fff' is 'li')
    assert _one(_s1(), eps_cell=_LINE, formulation="laurent") == (
        "laurent", "laurent")
    assert _one(_s1(), eps_cell=_LINE, formulation="fff") == ("li", "li")


def test_resolution_table_2d():
    assert _one(_s2(), eps_cell=_PILLAR) == ("laurent", "auto")
    assert _one(_s2(), eps_cell=_METAL_PILLAR) == ("li", "auto")
    assert _one(_s2(), shapes=_RECT, eps_background=1.0) == ("li", "auto")
    assert _one(_s2(), shapes=_DIEL_RECT, eps_background=1.0) == (
        "laurent", "auto")
    # a metal disk: 2-D 'li' takes axis-aligned rectangles only
    assert _one(_s2(), shapes=_DISK, eps_background=1.0) == (
        "laurent", "auto")
    # a metallic BACKGROUND counts too (holes in a metal film)
    holes = [dict(shape="rectangle", eps=1.0, center=(0.5e-6, 0.5e-6),
                  size=(0.3e-6, 0.3e-6))]
    assert _one(_s2(), shapes=holes, eps_background=-20 + 2j) == (
        "li", "auto")
    # dispersive 2-D metal: no per-wavelength factorization switch
    disp = [dict(_RECT[0], eps=lambda wl: -20 + 2j)]
    assert _one(_s2(), shapes=disp, eps_background=1.0) == ("laurent", "auto")
    assert _one(_s2(), eps_cell=lambda wl: _METAL_PILLAR) == (
        "laurent", "auto")
    tens = np.broadcast_to(np.diag([2.0, 2.5, 2.0]).astype(complex),
                           (13, 13, 3, 3)).copy()
    tens[:6, :, 0, 0] = 3.0
    assert _one(_s2(), eps_tensor_cell=tens) == ("laurent", "auto")


def test_bogus_formulation_and_li_on_uniform_still_raise():
    with pytest.raises(ValueError, match="formulation must be"):
        _s1().add_layer(0.1e-6, eps_cell=_LINE, formulation="bogus")
    with pytest.raises(ValueError, match="isotropic patterned"):
        _s1().add_layer(0.1e-6, eps=2.25, formulation="li")


def test_dispersive_sweep_keeps_the_request():
    st = _s1().add_layer(0.1e-6, eps_cell=lambda wl: _METAL_LINE)
    st.set_source(1.0e-6, theta=0.1)
    base = list(st._layers)
    mat = st._materialized_layers(1.0e-6, base)
    assert (mat[0].formulation, mat[0].formulation_requested) == ("li", "auto")


# ---------------------------------------------------------------------------
# 5: builders resolve once per call
# ---------------------------------------------------------------------------
def _taper(formulation=None, raster="hard"):
    st = RCWAStack(P, n_superstrate=1.0, n_substrate=NS, n_orders=40)
    kw = {} if formulation is None else dict(formulation=formulation)
    st.add_tapered_grating(50e-9, eps_ridge=AU, eps_groove=1.0,
                           duty_bottom=W / P, n_slices=1, n_x=8 * 40 + 8,
                           raster=raster, **kw)
    for t, es, _eg in LAYERS[1:]:
        st.add_layer(t, eps=es)
    return st.set_source(WL, theta=0.0)


def test_builder_default_equals_explicit_li():
    a, b = _taper(), _taper("li")
    assert a.layers[0].formulation == "li"
    assert a.layers[0].formulation_requested == "auto"
    assert b.layers[0].formulation_requested == "li"
    assert _bits(a.solve()) == _bits(b.solve())
    # raster='harmonic' under the resolved 'li' rides the companion pair
    h = _taper(raster="harmonic")
    assert h.layers[0].normal_cells is not None


def test_2d_builder_dielectric_keeps_laurent_metal_takes_li():
    def pillars(eps, **kw):
        st = RCWAStack(1e-6, period_y=1e-6, n_superstrate=1.0,
                       n_substrate=1.5, n_orders=2, n_orders_y=2)
        st.add_tapered_pillars(0.2e-6, pillars=[((0.5e-6, 0.5e-6),
                                                 (0.3e-6, 0.3e-6),
                                                 (0.4e-6, 0.35e-6), eps)],
                               eps_host=1.0, n_slices=2, **kw)
        return st.set_source(0.7e-6, theta=0.1, phi=0.2)
    d, dl = pillars(4.0), pillars(4.0, formulation="laurent")
    assert {L.formulation for L in d.layers} == {"laurent"}
    assert _bits(d.solve()) == _bits(dl.solve())
    m = pillars(-20 + 2j)
    assert {L.formulation for L in m.layers} == {"li"}
    assert {L.formulation_requested for L in m.layers} == {"auto"}


def test_graded_layer_resolves_per_slice():
    st = _s1()
    st.add_graded_layer(0.2e-6, lambda z: _LINE[:, None] if z < 0.5 else 2.0,
                        n_slices=2)
    assert [L.formulation for L in st.layers] == ["li", "laurent"]
    assert [L.formulation_requested for L in st.layers] == ["auto", "auto"]


# ---------------------------------------------------------------------------
# 6: the even-parity fold engages for the resolved 'li' layers
# ---------------------------------------------------------------------------
def test_even_parity_fold_engages_under_the_default(monkeypatch):
    calls = []
    orig = _stmod._symmetric_cascade_rt

    def spy(*a, **k):
        out = orig(*a, **k)
        calls.append(out is not None)
        return out

    monkeypatch.setattr(_stmod, "_symmetric_cascade_rt", spy)
    st = _fix_a(40)
    folded = st.solve(symmetry="auto")
    assert calls == [True]                   # the fold ran and was used
    full = st.solve(symmetry=False)
    o, R0, T0 = folded.efficiencies()
    o, R1, T1 = full.efficiencies()
    # measured |dJ| 4.0e-13 (Windows/MKL) / 4.6e-13 (WSL/OpenBLAS), |dR|
    # 1.4e-13 / 1.2e-13; the suite's fold bar for R / T is 1e-11
    assert np.allclose(R0, R1, atol=1e-11, rtol=0)
    assert np.allclose(T0, T1, atol=1e-11, rtol=0)
    assert np.max(np.abs(np.asarray(folded.jones_reflection())
                         - np.asarray(full.jones_reflection()))) < 1e-10


# ---------------------------------------------------------------------------
# 7: JAX -- a traced cell resolves to 'laurent' and differentiates; a concrete
# cell resolved to 'li' differentiates and jits
# ---------------------------------------------------------------------------
_JP, _JS = 600e-9, 64
_JX = np.arange(_JS) / _JS


def _jax_stack(cell, *, wl=1310e-9, **kw):
    st = RCWAStack(_JP, n_superstrate=1.0, n_substrate=1.5, n_orders=7)
    st.add_layer(150e-9, eps_cell=cell, **kw)
    return st.set_source(wl, theta=0.0)


def test_traced_cell_resolves_to_laurent_and_its_gradient_matches_fd():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    seen = []

    def total_r(s):
        cell = jnp.where(jnp.asarray(_JX) < 0.4, (-20 + 2j) * s, 1.0 + 0j)
        st = _jax_stack(cell)
        seen.append(st.layers[0].formulation)
        _o, R, _T = st.solve().efficiencies()
        return jnp.sum(R[0])

    g = float(jax.grad(total_r)(1.0))
    assert seen[-1] == "laurent"
    h = 1e-6
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fd = (float(total_r(1.0 + h)) - float(total_r(1.0 - h))) / (2 * h)
    # the EAGER evaluation (a concrete jnp cell) resolves the same way, so the
    # finite difference and the gradient describe one function
    assert seen[-1] == "laurent"
    assert np.isfinite(g)
    assert abs(g - fd) < 1e-5 * max(abs(fd), 1.0)


def test_concrete_li_cell_differentiates_and_jits_through_the_wavelength():
    """A concrete cell resolved to 'li' keeps working when the trace starts
    elsewhere (here the wavelength): its contrast test is taken when the layer
    is added, not read back from a tracer during the solve."""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    cell = np.where(_JX < 0.4, -20 + 2j, 1.0).astype(complex)
    assert _jax_stack(cell).layers[0].formulation == "li"

    def total_r(wl):
        _o, R, _T = _jax_stack(cell, wl=wl).solve().efficiencies()
        return jnp.sum(R[0])

    wl0 = 1310e-9
    r_np = float(np.sum(np.asarray(
        _jax_stack(cell).solve().efficiencies()[1])[0]))
    assert abs(float(jax.jit(total_r)(jnp.asarray(wl0))) - r_np) < 1e-12
    g = float(jax.grad(total_r)(jnp.asarray(wl0)))
    h = 1e-13
    fd = (float(np.sum(np.asarray(_jax_stack(cell, wl=wl0 + h).solve()
                                  .efficiencies()[1])[0]))
          - float(np.sum(np.asarray(_jax_stack(cell, wl=wl0 - h).solve()
                                    .efficiencies()[1])[0]))) / (2 * h)
    assert np.isfinite(g)
    assert abs(g - fd) < 1e-4 * max(abs(fd), 1.0)
