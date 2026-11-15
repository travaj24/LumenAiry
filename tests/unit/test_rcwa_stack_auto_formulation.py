"""``RCWAStack.add_layer(formulation='auto')`` -- the default factorization.

``'auto'`` resolves per layer: ``'li'`` for a patterned isotropic layer of a
1-D stack (``eps_cell``; a y-invariant ``shapes`` / ``segments`` list), ``'li'``
on a 2-D stack for every layer ``'li'`` supports (an ``eps_cell``; a list of
axis-aligned rectangles), whatever its material, and ``'laurent'`` for
everything else (uniform and tensor layers, y-varying layers of a 1-D stack,
2-D disks / ellipses, an exact zero permittivity).  The 2-D rule is per
shape class, not per material: the measurements behind it (a metal-only rule
left high-index dielectric layers on the less accurate direct rule) are in
``docs/audits/AUDIT_RCWA_2D_LI_FACTORIZATION_2026_11_14.md``.  The resolved
value is ``layers[i].formulation``; the request is
``layers[i].formulation_requested``.

The strip / spacer / mirror stack: a resonant metal-strip stack -- period
500 nm, a 180 nm wide, 40 nm thick strip (eps -90+7i) in air over
2 / 6 / 3 / 6 / 2 nm of eps 2.8 / 4.2 / 2.2+0.6i / 4.2 / 2.8 over an 80 nm
mirror (eps -130+20i), air above, n = 1.5 below, 1500 nm, TM.  Its converged
zeroth-order reflection ``r_ref = -0.1012241 - 0.1763084i`` comes from a
geometrically graded ``PMMStack`` (degree 20, 12 elements; degree 24 agrees to
1.3e-7).  (The fixture was replaced by a generic one; every reading below was
re-measured on it, Windows/MKL.)  ``'li'`` errs 1.05e-2 at N = 120 against
2.58e-1 for ``'laurent'``.
"""
import warnings

import numpy as np
import pytest

import lumenairy as lum
from lumenairy.elements.rcwa import RCWAStack
from lumenairy.elements.rcwa import stack as _stmod
from lumenairy.elements.rcwa.stack import RCWAYAverageWarning

WL, P, W = 1500e-9, 500e-9, 180e-9
E_STRIP, E_MIRROR = -90 + 7j, -130 + 20j
E_LOW, E_HIGH, E_LOSSY, N_SUB = 2.8, 4.2, 2.2 + 0.6j, 1.5
LAYERS = [(40e-9, E_STRIP, 1.0), (2e-9, E_LOW, E_LOW),
          (6e-9, E_HIGH, E_HIGH), (3e-9, E_LOSSY, E_LOSSY),
          (6e-9, E_HIGH, E_HIGH), (2e-9, E_LOW, E_LOW),
          (80e-9, E_MIRROR, E_MIRROR)]
# fixture replaced by a generic one; re-measured (PMMStack, degree 20 x 12)
R_REF = -0.1012241 - 0.1763084j
STRIP = [dict(shape="rectangle", eps=E_STRIP, center=(W / 2, P / 2),
              size=(W, P))]


def _strip_stack(N, formulation=None, theta=0.0):
    st = RCWAStack(P, n_superstrate=1.0, n_substrate=N_SUB, n_orders=N)
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
    bit), and it is 25x closer to the converged value than the direct rule
    at N = 120 (re-measured on the generic fixture: 1.05e-2 against 2.58e-1;
    the rung moved from 100, where the direct rule happens to dip to 2.8e-2
    on this stack, with the bars unchanged)."""
    st = _strip_stack(120)
    assert st.layers[0].formulation == "li"
    assert st.layers[0].formulation_requested == "auto"
    r_def = st.solve()
    r_li = _strip_stack(120, "li").solve()
    assert _bits(r_def) == _bits(r_li)
    err = abs(complex(np.asarray(r_def.jones_reflection())[0, 0]) - R_REF)
    assert err < 1.2e-2
    r_la = _strip_stack(120, "laurent").solve()
    err_la = abs(complex(np.asarray(r_la.jones_reflection())[0, 0]) - R_REF)
    assert err_la > 0.15
    assert err_la > 10.0 * err


def test_te_is_unchanged_by_the_default():
    """E_y is tangential to every wall of a 1-D grating, so Li's rule is the
    direct rule there: TE moves only at round-off (re-measured on the generic
    fixture at N = 50: 6.1e-13 Windows/MKL; bar 1e-11), while TM moves by
    0.149."""
    Jd = np.asarray(_strip_stack(50).solve().jones_reflection())
    Jl = np.asarray(_strip_stack(50, "laurent").solve().jones_reflection())
    assert abs(Jd[1, 1] - Jl[1, 1]) < 1e-11
    # ... while TM moves by O(0.1): the change is the point of the default
    assert abs(Jd[0, 0] - Jl[0, 0]) > 0.1


def test_uniform_only_stack_is_bit_identical_to_laurent():
    def build(**kw):
        st = RCWAStack(P, n_superstrate=1.0, n_substrate=N_SUB, n_orders=5)
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
    # 2-D: per SHAPE class, not per material (measured, 2026-11-14:
    # docs/audits/AUDIT_RCWA_2D_LI_FACTORIZATION_2026_11_14.md and
    # tests/unit/test_rcwa_li_2d_crossed.py) -- every layer 'li' supports
    # resolves to 'li', dielectric included
    assert _one(_s2(), eps_cell=_PILLAR) == ("li", "auto")
    assert _one(_s2(), eps_cell=_METAL_PILLAR) == ("li", "auto")
    assert _one(_s2(), shapes=_RECT, eps_background=1.0) == ("li", "auto")
    assert _one(_s2(), shapes=_DIEL_RECT, eps_background=1.0) == (
        "li", "auto")
    # a metal disk: 2-D 'li' takes axis-aligned rectangles only
    assert _one(_s2(), shapes=_DISK, eps_background=1.0) == (
        "laurent", "auto")
    # a metallic BACKGROUND counts too (holes in a metal film)
    holes = [dict(shape="rectangle", eps=1.0, center=(0.5e-6, 0.5e-6),
                  size=(0.3e-6, 0.3e-6))]
    assert _one(_s2(), shapes=holes, eps_background=-20 + 2j) == (
        "li", "auto")
    # dispersive 2-D layers: the rule does not read the material, so it
    # cannot switch inside a spectrum
    disp = [dict(_RECT[0], eps=lambda wl: -20 + 2j)]
    assert _one(_s2(), shapes=disp, eps_background=1.0) == ("li", "auto")
    assert _one(_s2(), eps_cell=lambda wl: _METAL_PILLAR) == ("li", "auto")
    # an exact zero background: the inverse rule would read 1/0
    assert _one(_s2(), shapes=_RECT, eps_background=0.0) == (
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
    st = RCWAStack(P, n_superstrate=1.0, n_substrate=N_SUB, n_orders=40)
    kw = {} if formulation is None else dict(formulation=formulation)
    st.add_tapered_grating(40e-9, eps_ridge=E_STRIP, eps_groove=1.0,
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


def test_2d_builder_takes_li_for_dielectric_and_metal():
    """A 2-D builder resolves like ``add_layer``: per shape class, so a
    rasterized dielectric pillar and a metal one both take 'li', and the
    default call is the explicit 'li' call bit for bit (same route)."""
    def pillars(eps, **kw):
        st = RCWAStack(1e-6, period_y=1e-6, n_superstrate=1.0,
                       n_substrate=1.5, n_orders=2, n_orders_y=2)
        st.add_tapered_pillars(0.2e-6, pillars=[((0.5e-6, 0.5e-6),
                                                 (0.3e-6, 0.3e-6),
                                                 (0.4e-6, 0.35e-6), eps)],
                               eps_host=1.0, n_slices=2, **kw)
        return st.set_source(0.7e-6, theta=0.1, phi=0.2)
    d, dl = pillars(4.0), pillars(4.0, formulation="li")
    assert {L.formulation for L in d.layers} == {"li"}
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
    st = _strip_stack(40)
    folded = st.solve(symmetry="auto")
    assert calls == [True]                   # the fold ran and was used
    full = st.solve(symmetry=False)
    o, R0, T0 = folded.efficiencies()
    o, R1, T1 = full.efficiencies()
    # re-measured on the generic fixture: |dJ| 5.0e-13, |dR| 1.9e-13
    # (Windows/MKL); the suite's fold bar for R / T is 1e-11
    assert np.allclose(R0, R1, atol=1e-11, rtol=0)
    assert np.allclose(T0, T1, atol=1e-11, rtol=0)
    assert np.max(np.abs(np.asarray(folded.jones_reflection())
                         - np.asarray(full.jones_reflection()))) < 1e-10


# ---------------------------------------------------------------------------
# 7: JAX -- a traced cell resolves as its NumPy twin ('li') and differentiates;
# a concrete cell resolved to 'li' differentiates and jits
# ---------------------------------------------------------------------------
_JP, _JS = 600e-9, 64
_JX = np.arange(_JS) / _JS


def _jax_stack(cell, *, wl=1310e-9, **kw):
    st = RCWAStack(_JP, n_superstrate=1.0, n_substrate=1.5, n_orders=7)
    st.add_layer(150e-9, eps_cell=cell, **kw)
    return st.set_source(wl, theta=0.0)


def test_traced_cell_resolves_like_numpy_and_its_gradient_matches_fd():
    """Re-pinned when a JAX cell started to resolve as its NumPy twin: the
    traced and the concrete JAX cell both run 'li' (they ran 'laurent'
    before, while the same values as a NumPy array ran 'li')."""
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
    assert seen[-1] == "li"
    h = 1e-6
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fd = (float(total_r(1.0 + h)) - float(total_r(1.0 - h))) / (2 * h)
    # the EAGER evaluation (a concrete jnp cell) resolves the same way, so the
    # finite difference and the gradient describe one function
    assert seen[-1] == "li"
    cell_np = np.where(_JX < 0.4, -20 + 2j, 1.0 + 0j)
    assert _jax_stack(cell_np).layers[0].formulation == "li"
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


# ---------------------------------------------------------------------------
# 8: layer_absorption stays right on the resolved 'li' layers
# ---------------------------------------------------------------------------
def _absorber(formulation=None, N=8):
    """A lossy bar over a metal line (eps -20+2i), 1.31 um, theta 0.25."""
    p = 0.6e-6
    kw = {} if formulation is None else dict(formulation=formulation)
    st = RCWAStack(p, n_substrate=1.5, n_orders=N)
    st.add_layer(0.06e-6, shapes=[dict(shape="rectangle", size=(0.3 * p, p),
                                       eps=complex(4.0, 0.5),
                                       center=(0.2 * p, p / 2))],
                 eps_background=complex(2.25, 0.02), **kw)
    st.add_layer(0.04e-6, shapes=[dict(shape="rectangle", size=(0.5 * p, p),
                                       eps=complex(-20.0, 2.0),
                                       center=(0.6 * p, p / 2))],
                 eps_background=1.0, **kw)
    st.set_source(1.31e-6, theta=0.25)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return st.solve(retain_internal=True, symmetry=False)


def test_li_layer_absorption_is_the_face_flux_and_closes():
    """An 'li' layer is attributed the Poynting-flux difference between its
    faces.  The Im(eps)|E|^2 volume integral is not consistent with the
    inverse-rule E_x: on this stack in TM it summed to 0.2505 against
    A = 0.2088 (measured on the base release with explicit 'li').  The face
    fluxes close on A to round-off (measured 6e-15)."""
    res = _absorber()
    assert {L.formulation for L in res._require_modal()["internal"]["layers"]
            } == {"li"}
    raw = res.layer_absorption(return_raw=True)
    A = res.absorptance()
    assert np.max(np.abs(raw.sum(axis=1) - A)) < 1e-12
    assert np.all(raw > 0)
    # independent check where the volume integral IS consistent: in TE (E_y
    # tangential to every wall, the two rules give the same solve) the 'li'
    # face flux and the 'laurent' volume integral attribute the same power
    # to each layer -- measured agreement 4e-6 relative at nz_per_layer 128
    vol = _absorber("laurent").layer_absorption(nz_per_layer=128,
                                                return_raw=True)
    assert np.max(np.abs(raw[1] / vol[1] - 1)) < 1e-4


def test_li_layer_absorption_closes_on_a_2d_metal_layer():
    """The face-flux attribution on a 2-D stack ('auto' -> 'li' for the metal
    rectangle; the lossy spacer keeps the volume integral): the 'li' layer's
    share is exact, so the raw split misses the absorptance only by the
    spacer's midpoint depth rule (measured 8.0e-7 / 3.5e-7 relative for the
    two incident polarizations at nz_per_layer 256, Windows/MKL)."""
    p = 0.6e-6
    st = RCWAStack(p, period_y=p, n_substrate=1.5, n_orders=3, n_orders_y=3)
    st.add_layer(0.05e-6, shapes=[dict(shape="rectangle", eps=-10 + 1j,
                                       center=(0.35 * p, 0.6 * p),
                                       size=(0.3 * p, 0.4 * p))],
                 eps_background=1.0)
    st.add_layer(0.05e-6, eps=complex(2.1, 0.1))
    st.set_source(1.31e-6, theta=0.2, phi=0.5)
    assert st.layers[0].formulation == "li"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = st.solve(retain_internal=True, symmetry=False)
    raw = res.layer_absorption(nz_per_layer=256, return_raw=True)
    assert np.all(raw > 0)
    assert np.max(np.abs(raw.sum(axis=1) / res.absorptance() - 1)) < 1e-4
