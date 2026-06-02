"""v5.11.0 RCWA internal-layer E/H field reconstruction (audit GAP1).

``RCWAStack.solve(retain_internal=True)`` + ``RCWAResult.internal_field(z, ...)``
recover the real-space E and H field INSIDE the structure (not just the
far-field superposition).  Derived + validated by a derive/prototype/synthesize
workflow; the winning recovery uses the gap-free S-matrix partials, the
``exp(+lam k0 z)`` backward reference, the ``-i*eta0`` magnetic map, and -- the
decisive correction -- the CURL-H longitudinal ``Ez`` (the div-D form is wrong
for oblique TM).

Validation is library-grounded + self-contained (no external TMM, which is
where the prototype's standalone oracle mislabeled TM E/H): the field must (1)
be continuous (tangential E,H) across interfaces, (2) match the library's own
(grcwa/inkstone/Airy-validated) reflected/transmitted amplitudes at the
boundaries, (3) carry constant Poynting flux in a lossless stack, and (4) have
continuous normal D (=eps*Ez) -- the curl-H Ez check.
"""
from __future__ import annotations

import numpy as np
import pytest

import lumenairy as la

THICK = [0.15e-6, 0.2e-6, 0.1e-6]
EPS = [2.1, 6.0 + 0.3j, 4.0]


def _stack(theta=0.0, grating=False, lossless=False, n_orders=8):
    st = la.RCWAStack(0.5e-6, n_substrate=1.5, n_superstrate=1.0,
                      n_orders=n_orders)
    epss = [2.1, 6.0, 4.0] if lossless else EPS
    st.add_layer(THICK[0], eps=epss[0])
    if grating:
        x = (np.arange(256) + 0.5) / 256
        cell = np.where(np.abs(x - 0.5) < 0.25, epss[1], 1.0).astype(complex)
        st.add_layer(THICK[1], eps_cell=cell[:, None])
    else:
        st.add_layer(THICK[1], eps=epss[1])
    st.add_layer(THICK[2], eps=epss[2])
    return st.set_source(0.633e-6, theta=theta)


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("grating", [False, True])
@pytest.mark.parametrize("theta", [0.0, np.deg2rad(30)])
def test_tangential_field_continuous_across_interfaces(theta, grating):
    res = _stack(theta, grating).solve(retain_internal=True)
    worst = 0.0
    for inc in ((1.0, 0.0), (0.0, 1.0)):
        for k in range(2):                       # interfaces 0|1 and 1|2
            fa = res.internal_field(THICK[k], nx=32, layer=k, incident=inc)
            fb = res.internal_field(0.0, nx=32, layer=k + 1, incident=inc)
            for c in ("Ex", "Ey", "Hx", "Hy"):
                worst = max(worst, float(np.max(np.abs(fa[c] - fb[c]))))
    assert worst < 1e-10


@pytest.mark.parametrize("theta", [0.0, np.deg2rad(30)])
def test_boundary_matches_library_reflection_transmission(theta):
    res = _stack(theta).solve(retain_internal=True)
    o, R, T = res.efficiencies()
    ra = res.per_order_amplitudes("reflection")
    ta = res.per_order_amplitudes("transmission")
    i0 = int(np.where(o == 0)[0][0])
    ic = 4
    worst = 0.0
    for inc, row in (((1.0, 0.0), 0), ((0.0, 1.0), 1)):
        top = res.internal_field(0.0, nx=8, layer=0, incident=inc)
        bot = res.internal_field(THICK[2], nx=8, layer=2, incident=inc)
        for c in ("Ex", "Ey"):
            inc0 = {"Ex": inc[0], "Ey": inc[1]}[c]
            worst = max(worst,
                        abs(top[c][0, ic] - (inc0 + ra[c][row, i0])),
                        abs(bot[c][0, ic] - ta[c][row, i0]))
    assert worst < 1e-10


def test_poynting_flux_constant_in_lossless_stack():
    res = _stack(theta=np.deg2rad(25), lossless=True).solve(retain_internal=True)
    zs = np.linspace(0.005e-6, sum(THICK) - 0.005e-6, 25)
    for inc in ((1.0, 0.0), (0.0, 1.0)):
        f = res.internal_field(zs, nx=8, incident=inc)
        Sz = np.real(f["Ex"] * np.conj(f["Hy"]) - f["Ey"] * np.conj(f["Hx"]))
        Sz0 = Sz.mean(axis=1)
        drift = np.max(np.abs(Sz0 - Sz0.mean())) / (abs(Sz0.mean()) + 1e-30)
        assert drift < 1e-9


def test_curl_h_Ez_via_normal_D_continuity_oblique_tm():
    # the decisive Ez check: Dz = eps*Ez is continuous across interfaces (normal
    # D), where Ez itself jumps.  Oblique TM makes Ez nonzero; the div-D Ez form
    # would FAIL this.
    res = _stack(theta=np.deg2rad(40)).solve(retain_internal=True)
    ez_seen, worst = 0.0, 0.0
    for inc in ((1.0, 0.0), (0.0, 1.0)):
        for k in range(2):
            fa = res.internal_field(THICK[k], nx=16, layer=k, incident=inc)
            fb = res.internal_field(0.0, nx=16, layer=k + 1, incident=inc)
            ez_seen = max(ez_seen, float(np.max(np.abs(fa["Ez"]))))
            dz_jump = EPS[k] * fa["Ez"] - EPS[k + 1] * fb["Ez"]
            worst = max(worst, float(np.max(np.abs(dz_jump))))
    assert ez_seen > 0.1                          # Ez is genuinely exercised
    assert worst < 1e-10


# ---------------------------------------------------------------------------
# API behaviour
# ---------------------------------------------------------------------------

def test_internal_field_requires_retain_flag():
    res = _stack().solve()                        # retain_internal defaults off
    with pytest.raises(ValueError, match="retain_internal"):
        res.internal_field(0.1e-6)


def test_internal_field_component_selection_and_shapes():
    res = _stack().solve(retain_internal=True)
    zs = np.linspace(0, sum(THICK), 7)
    fE = res.internal_field(zs, component="E", nx=16)
    assert set(fE) >= {"Ex", "Ey", "Ez"} and "Hx" not in fE
    assert fE["Ex"].shape == (7, 16)              # 1-D stack -> ny dropped
    fH = res.internal_field(zs, component="H", nx=16)
    assert set(fH) >= {"Hx", "Hy", "Hz"} and "Ex" not in fH


def test_internal_field_te_has_no_Ex():
    # pure TE incidence (E along y) -> Ex == 0 everywhere
    res = _stack().solve(retain_internal=True)
    f = res.internal_field(np.linspace(0, sum(THICK), 5), nx=16,
                           incident=(0.0, 1.0))
    assert np.max(np.abs(f["Ex"])) < 1e-12
    assert np.max(np.abs(f["Ey"])) > 0.1
