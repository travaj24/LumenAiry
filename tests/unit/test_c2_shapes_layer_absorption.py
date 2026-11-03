"""Plan C2 (audit L-3): ``RCWAResult.layer_absorption`` on analytic
``shapes=`` layers.

Pins:

* a stack holding ``shapes`` layers no longer raises, and the default split
  still closes on ``absorptance()``;
* the per-layer split agrees with an INDEPENDENT, frame-free split -- the
  Poynting-flux difference between the two faces of each layer, which IS that
  layer's absorbed power -- on OFF-CENTRE shapes at oblique incidence.  The
  off-centre geometry is what discriminates the frame: evaluating the weight
  at the point mirror ``(-x, -y)`` fails these pins (the solve and
  ``internal_field`` are registered to the DECLARED structure since the
  point-mirror fix, ``test_rcwa_point_mirror_fix.py``);
* a MIXED stack (shapes + an asymmetric eps_cell raster + uniform) splits
  consistently against the same flux oracle, and ``return_raw=True`` closes
  on an asymmetric eps_cell layer too;
* the in-plane rule is EXACT: with a Gauss-Legendre depth rule the
  un-normalised integral closes on ``A`` to ~1e-13 (measured), so the only
  residual of ``return_raw=True`` is the midpoint depth rule;
* an existing-kind stack (uniform + eps_cell) is BYTE-identical to the
  pre-C2 algorithm, re-implemented here verbatim.
"""
import warnings

import numpy as np
import pytest

from lumenairy.elements import rcwa
from lumenairy.elements.rcwa.stack import RCWAResult

P = 0.6e-6
WL = 1.31e-6


def _solve(st):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return st.solve(retain_internal=True, symmetry=False)


def _flux_split(res):
    """Per-layer absorption from the Poynting flux difference (2, n_layers)."""
    m = res._require_modal()
    info = m["internal"]
    nx = 2 * info["nox"] + 2
    ny = 1 if info["noy"] == 0 else 2 * info["noy"] + 2
    out = np.zeros((2, len(info["thick"])))
    for p, inc in enumerate(((1.0, 0.0), (0.0, 1.0))):
        kz = float(m["kz_inc"])
        li = float(m["kx0"]) * inc[0] + float(m["ky0"]) * inc[1]
        s_inc = kz * (1.0 + (li / kz) ** 2)
        for i, t in enumerate(info["thick"]):
            f = res.internal_field([0.0, t], component="all", nx=nx, ny=ny,
                                   layer=i, incident=inc)
            sz = np.real(f["Ex"] * np.conj(f["Hy"]) - f["Ey"] * np.conj(f["Hx"]))
            sz = sz.reshape(2, -1).mean(axis=1)
            out[p, i] = (sz[0] - sz[1]) / s_inc
    return out


def _stack_1d():
    """Off-centre lossy bar on a lossy background over a metal line."""
    st = rcwa.RCWAStack(period=P, n_substrate=1.5, n_orders=8)
    st.add_layer(0.06e-6, shapes=[{"shape": "rectangle", "size": (0.3 * P, P),
                                   "eps": complex(4.0, 0.5),
                                   "center": (0.2 * P, P / 2)}],
                 eps_background=complex(2.25, 0.02))
    st.add_layer(0.04e-6, shapes=[{"shape": "rectangle", "size": (0.5 * P, P),
                                   "eps": complex(-20.0, 2.0),
                                   "center": (0.6 * P, P / 2)}],
                 eps_background=1.0)
    st.set_source(WL, theta=0.25)
    return _solve(st)


def _stack_2d():
    """Off-centre lossy ellipse + metal rectangle, oblique, conical."""
    st = rcwa.RCWAStack(period=P, period_y=P, n_substrate=1.5, n_orders=3,
                        n_orders_y=3)
    st.add_layer(0.08e-6, shapes=[
        {"shape": "ellipse", "semi_axes": (0.15 * P, 0.1 * P),
         "eps": complex(4.0, 0.8), "center": (0.25 * P, 0.35 * P)},
        {"shape": "rectangle", "size": (0.2 * P, 0.3 * P),
         "eps": complex(-10.0, 1.0), "center": (0.7 * P, 0.7 * P)}],
        eps_background=1.0)
    st.add_layer(0.05e-6, eps=complex(2.1, 0.1))
    st.set_source(WL, theta=0.2, phi=0.5)
    return _solve(st)


@pytest.mark.parametrize("make", [_stack_1d, _stack_2d], ids=["1d", "2d"])
def test_shapes_split_matches_poynting_flux_difference(make):
    res = make()
    a = res.layer_absorption(nz_per_layer=64)       # no NotImplementedError
    assert np.all(np.isfinite(a)) and np.all(a > 0)
    assert np.allclose(a.sum(axis=1), res.absorptance(), rtol=0, atol=1e-14)
    fl = _flux_split(res)
    # flux faces telescope onto A (the reference is itself closed)
    assert np.allclose(fl.sum(axis=1), res.absorptance(), rtol=0, atol=1e-12)
    # midpoint depth rule, O(nz^-2): measured 1.2e-4 (1d) / 3.9e-4 (2d);
    # evaluating the weight at the point mirror (-x, -y) fails by > 0.1
    assert np.max(np.abs(a / fl - 1)) < 1e-3


def test_shapes_in_plane_rule_is_exact():
    """GL depth rule + the shipped weight on the shipped grid: closure ~1e-13
    (the point-sampled indicator was measured at 1.5e-3 for a disk even at
    nx = 1024)."""
    res = _stack_2d()
    m = res._require_modal()
    info = m["internal"]
    L, t = info["layers"][0], info["thick"][0]
    M = info["nox"]
    n = 4 * M + 2
    zg, wg = np.polynomial.legendre.leggauss(32)
    fl = _flux_split(res)
    for p, inc in enumerate(((1.0, 0.0), (0.0, 1.0))):
        f = res.internal_field(0.5 * t * (zg + 1), component="E", nx=n, ny=n,
                               layer=0, incident=inc)
        e2 = np.abs(f["Ex"]) ** 2 + np.abs(f["Ey"]) ** 2 + np.abs(f["Ez"]) ** 2
        w = RCWAResult._layer_im_eps(L, f["x"], f["y"], P, P,
                                     n_orders=(M, M))
        kz = float(m["kz_inc"])
        li = float(m["kx0"]) * inc[0] + float(m["ky0"]) * inc[1]
        raw = (info["k0"] / (kz * (1 + (li / kz) ** 2)) * t * 0.5
               * np.sum(wg * np.mean(w[None] * e2, axis=(1, 2))))
        assert abs(raw / fl[p, 0] - 1) < 1e-10


def test_return_raw_closes_on_a_and_converges_with_depth():
    res = _stack_1d()
    A = res.absorptance()
    errs = []
    for nz in (32, 128):
        raw = res.layer_absorption(nz_per_layer=nz, return_raw=True)
        errs.append(np.max(np.abs(raw.sum(axis=1) / A - 1)))
    # measured 9.7e-4 (nz 32) -> 6.1e-5 (nz 128): the 16x of O(nz^-2)
    assert errs[1] < 3e-4
    assert errs[1] < errs[0] / 8


def test_existing_kinds_byte_identical_to_pre_c2_algorithm():
    st = rcwa.RCWAStack(period=P, period_y=0.5e-6, n_substrate=1.5,
                        n_orders=3, n_orders_y=2)
    cell = np.ones((24, 20), dtype=complex) * 2.1
    cell[6:15, 4:12] = complex(-12.0, 1.5)
    st.add_layer(0.06e-6, eps_cell=cell)
    st.add_layer(0.05e-6, eps=complex(2.25, 0.08))
    st.set_source(WL, theta=0.2, phi=0.3)
    res = _solve(st)
    nx, ny, nz = 20, 16, 4
    got = res.layer_absorption(nx=nx, ny=ny, nz_per_layer=nz)
    # --- the pre-C2 body, verbatim (scalar kinds) ---
    info = res._require_modal()["internal"]
    px, py = info["period_x"], info["period_y"]
    dx, dy = px / nx, py / ny
    xg = (np.arange(nx) - nx // 2) * dx
    yg = (np.arange(ny) - ny // 2) * dy
    im_eps = [RCWAResult._layer_im_eps(L, xg, yg, px, py)
              for L in info["layers"]]
    A_tot = res.absorptance()
    ref = np.zeros((2, 2))
    for p, inc in enumerate(((1.0, 0.0), (0.0, 1.0))):
        raw = np.zeros(2)
        for i in range(2):
            zl = (np.arange(nz) + 0.5) / nz * info["thick"][i]
            f = res.internal_field(zl, component="E", nx=nx, ny=ny, dx=dx,
                                   dy=dy, layer=i, incident=inc)
            e2 = (np.abs(f["Ex"]) ** 2 + np.abs(f["Ey"]) ** 2
                  + np.abs(f["Ez"]) ** 2)
            raw[i] = info["thick"][i] * float(np.mean(im_eps[i][None] * e2))
        ref[p] = A_tot[p] * raw / raw.sum()
    assert got.tobytes() == ref.tobytes()


def _mixed_stack(nx_cell=64):
    """shapes (off-centre lossy bar) + ASYMMETRIC eps_cell raster (lossy
    staircase) + uniform lossy spacer, oblique."""
    st = rcwa.RCWAStack(period=P, n_substrate=1.5, n_orders=8)
    st.add_layer(0.06e-6, shapes=[{"shape": "rectangle", "size": (0.3 * P, P),
                                   "eps": complex(4.0, 0.5),
                                   "center": (0.2 * P, P / 2)}],
                 eps_background=complex(2.25, 0.02))
    x = np.arange(nx_cell) / nx_cell
    cell = np.where(x < 0.3, complex(5.0, 0.8),
                    np.where(x < 0.5, complex(2.5, 0.2), 1.0 + 0j))
    st.add_layer(0.05e-6, eps_cell=cell[:, None])
    st.add_layer(0.04e-6, eps=complex(2.1, 0.1))
    st.set_source(WL, theta=0.25)
    return _solve(st)


def test_mixed_shapes_and_cell_stack_split_matches_flux():
    """Before the point-mirror fix the shapes and eps_cell branches sampled
    different frames, so a mixed stack could not be right in both."""
    res = _mixed_stack()
    fl = _flux_split(res)
    # nx = S: the cell's nearest-node map IS the solved series at the nodes
    a = res.layer_absorption(nx=64, nz_per_layer=64)
    # measured 5.8e-6 (max over layers and pols)
    assert np.max(np.abs(a / fl - 1)) < 1e-4


def test_return_raw_closes_on_an_asymmetric_eps_cell_layer():
    """Verifier finding 3: 200-311 % off for asymmetric cells while the
    solve was mirrored; now closes to the depth-rule error."""
    res = _mixed_stack()
    fl = _flux_split(res)
    A = res.absorptance()
    raw = res.layer_absorption(nx=64, nz_per_layer=128, return_raw=True)
    # measured 3.0e-6 (cell layer vs flux), 1.8e-6 (sum vs A)
    assert np.max(np.abs(raw[:, 1] / fl[:, 1] - 1)) < 1e-4
    assert np.max(np.abs(raw.sum(axis=1) / A - 1)) < 1e-4
