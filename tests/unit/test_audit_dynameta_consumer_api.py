"""AUDIT_DYNAMETA_CONSUMER_API_GAPS_2026_07_13 regression gates (A1 / A2 / B).

A1 -- ``BerremanStack.jones_transmission()``: the class solve computes the
transmission Jones on every path and used to discard it, forcing consumers to
run a SECOND functional solve just for ``t``.  Gate: the accessor is
bit-identical to the functional ``berreman_jones_1d`` ``jones_t`` across the
audit's five case classes, and survives a ``retain_internal=True`` solve (the
one-solve consolidation the consumer wants).

A2 -- ``RCWAStack.layers``: the public read-only view of the per-layer records
(reverse translators previously read the private ``_layers`` slot under a
version ceiling).

B -- ``PMMStack.per_order_amplitudes`` / ``PMMStack.jones_transmission``: the
:meth:`RCWAResult.per_order_amplitudes` contract mirrored for the PMM family.
Gates: per-order COMPLEX-amplitude parity against RCWA on identical physics
(classical oblique, conical patterned via the nodal cascade, conical uniform
via the Fourier path); the documented flux recipe rebuilds the returned
efficiencies exactly; the rotated conical s-hat superposition synthesized from
the amplitudes matches the RCWA-amplitude oracle while the naive per-order
POWER sum misses the cross terms (the physics that forced DynaMeta to
hard-raise conical for patterned PMM cells); the exported evanescent ``kz``
carries the PUBLIC decaying branch (``Im >= 0``).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pytest

from lumenairy.elements.berreman import BerremanStack, berreman_jones_1d
from lumenairy.elements.pmm import PMMStack
from lumenairy.elements.rcwa import RCWAStack

_C = complex
_WL = 1.55e-6


# ======================================================================== #
# A1: BerremanStack.jones_transmission                                     #
# ======================================================================== #

def _lc_tensor(rot_oop=False):
    """An anisotropic tensor: in-plane rotated uniaxial, optionally tilted
    OUT of plane (eps_xz != 0)."""
    no2, ne2 = 1.5 ** 2, 1.7 ** 2
    if rot_oop:
        th = np.deg2rad(35.0)
        c, s = np.cos(th), np.sin(th)
        Rm = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    else:
        ph = np.deg2rad(25.0)
        c, s = np.cos(ph), np.sin(ph)
        Rm = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return Rm @ np.diag([ne2, no2, no2]) @ Rm.T


_A1_CASES = [
    ("iso-normal", 2.25, 0.0, 0.0),
    ("iso-oblique", 2.25, 0.5, 0.0),
    ("inplane-tensor", _lc_tensor(), 0.4, 0.0),
    ("conical-rotated", _lc_tensor(), 0.4, 0.6),
    ("oop-oblique", _lc_tensor(rot_oop=True), 0.4, 0.0),
]


@pytest.mark.parametrize("tag,eps,ang,phi", _A1_CASES,
                         ids=[c[0] for c in _A1_CASES])
def test_a1_jones_transmission_matches_functional(tag, eps, ang, phi):
    """The class accessor == the functional ``jones_t``, bit-identical (the
    audit's pre-proven consolidation, pinned as a gate)."""
    st = BerremanStack(n_substrate=1.45)
    st.add_layer(300e-9, eps=eps)
    st.set_source(_WL, angle=ang, phi=phi)
    R, T, Jr = st.solve()
    Jt = st.jones_transmission()
    _R, _T, _Jr, Jt_fn = berreman_jones_1d(
        [(eps, 300e-9)], 1.45, 1.0, _WL, angle=ang, phi=phi)
    assert np.array_equal(np.asarray(Jt), np.asarray(Jt_fn)), \
        f"{tag}: class Jt != functional jones_t"
    assert np.max(np.abs(R - _R)) == 0.0 and np.max(np.abs(T - _T)) == 0.0


def test_a1_jones_transmission_requires_solve():
    st = BerremanStack(n_substrate=1.45)
    st.add_layer(300e-9, eps=2.25)
    with pytest.raises(ValueError, match="call solve"):
        st.jones_transmission()


def test_a1_retained_alongside_internal_observables():
    """The one-solve consolidation: a retain_internal=True solve serves the
    far field (incl. Jt) AND layer_absorption -- no second functional solve."""
    eps_lossy = 2.25 + 0.05j
    st = BerremanStack(n_substrate=1.45)
    st.add_layer(300e-9, eps=eps_lossy)
    st.set_source(_WL, angle=0.4)
    R, T, _Jr = st.solve(retain_internal=True)
    Jt = st.jones_transmission()
    _R, _T, _Jr2, Jt_fn = berreman_jones_1d(
        [(eps_lossy, 300e-9)], 1.45, 1.0, _WL, angle=0.4)
    assert np.array_equal(np.asarray(Jt), np.asarray(Jt_fn))
    A = st.layer_absorption()
    budget = np.abs(np.sum(A, axis=0) + R + T - 1.0)
    assert np.max(budget) < 1e-10      # closed absorption budget, same solve


def test_a1_jax_twin_stores_jt_both_branches():
    """The differentiable twin mirrors the NumPy path: ``_jones_t`` is
    populated on the plain solve AND the retain_internal solve, matching
    NumPy to solver precision."""
    pytest.importorskip("jax")
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    eps = 2.25 + 0.02j
    ref = BerremanStack(n_substrate=1.45)
    ref.add_layer(300e-9, eps=eps)
    ref.set_source(_WL, angle=0.4)
    ref.solve()
    Jt_np = np.asarray(ref.jones_transmission())
    for retain in (False, True):
        st = BerremanStack(n_substrate=1.45)
        st.add_layer(300e-9, eps=jnp.asarray(eps))
        st.set_source(_WL, angle=0.4)
        st.solve(retain_internal=retain)
        Jt_jx = np.asarray(st.jones_transmission())
        assert np.max(np.abs(Jt_jx - Jt_np)) < 1e-12, f"retain={retain}"


# ======================================================================== #
# A2: RCWAStack.layers                                                     #
# ======================================================================== #

def test_a2_public_layers_view():
    s = RCWAStack(1.0e-6, n_substrate=1.45, n_orders=5)
    s.add_layer(0.2e-6, eps=2.25)
    cell = np.full((64, 1), 4.0 + 0j)
    s.add_layer(0.3e-6, eps_cell=cell, formulation="li")
    recs = s.layers
    assert isinstance(recs, tuple) and len(recs) == 2
    assert recs[0].kind == "uniform" and recs[1].kind == "iso"
    assert recs[0].thickness == pytest.approx(0.2e-6)
    assert recs[1].formulation == "li"
    assert recs[0].dispersive is False
    # a snapshot view of the SAME records the solver consumes
    assert all(a is b for a, b in zip(recs, s._layers))


# ======================================================================== #
# B: PMMStack per-order amplitudes + transmission Jones                    #
# ======================================================================== #

_PERIOD = 1.1e-6
_DUTY, _DEPTH = 0.45, 0.35e-6
_EPS_R, _EPS_G = 4.2 + 0j, 1.0 + 0j
_NSUB = 1.45


def _pmm_grating(theta, phi):
    s = PMMStack(_PERIOD, n_substrate=_NSUB, degree=14, far_field_orders=11)
    s.add_layer(_DEPTH, segments=[(_DUTY, _EPS_R), (1 - _DUTY, _EPS_G)])
    s.set_source(_WL, theta=theta, phi=phi)
    return s


def _rcwa_grating(theta, phi, S=2048):
    s = RCWAStack(_PERIOD, n_substrate=_NSUB, n_orders=41)
    x = (np.arange(S) + 0.5) / S
    cell = np.where(x < _DUTY, _EPS_R, _EPS_G)[:, None]
    s.add_layer(_DEPTH, eps_cell=cell, formulation="li")
    s.set_source(_WL, theta=theta, phi=phi)
    return s.solve()


def _common_orders(po_p, po_r):
    op, orc = np.asarray(po_p["orders"]), np.asarray(po_r["orders"])
    common = sorted(set(op.tolist()) & set(orc.tolist()))
    ip = np.array([int(np.where(op == m)[0][0]) for m in common])
    ir = np.array([int(np.where(orc == m)[0][0]) for m in common])
    return ip, ir


def _assert_amp_parity(sp, res_r, tol):
    """Per-order COMPLEX amplitude + k-vector parity on common orders."""
    for port in ("reflection", "transmission"):
        pp = sp.per_order_amplitudes(port)
        pr = res_r.per_order_amplitudes(port)
        ip, ir = _common_orders(pp, pr)
        assert np.max(np.abs(pp["kx"][ip] - pr["kx"][ir])) < 1e-12
        assert np.max(np.abs(pp["kz"][ip] - pr["kz"][ir])) < 1e-9
        prop = np.real(pp["kz"][ip]) > 1e-9
        assert np.any(prop)
        for key in ("Ex", "Ey"):
            d = np.abs(pp[key][:, ip] - pr[key][:, ir])[:, prop]
            assert np.max(d) < tol, f"{port}/{key}: {np.max(d):.3e} >= {tol}"


def _assert_flux_closure(sp, R_eff, T_eff, tol=1e-12):
    """The documented recipe (flux weight + longitudinal Ez + incident |E|^2)
    rebuilds the returned efficiencies from the raw tangential amplitudes."""
    pr = sp.per_order_amplitudes("reflection")
    p0 = int(np.where(pr["orders"] == 0)[0][0])
    kz_inc = float(np.real(pr["kz"][p0]))
    kx0 = float(np.real(pr["kx"][p0]))
    ky0 = float(np.real(pr["ky"][p0]))
    for port, eff in (("reflection", R_eff), ("transmission", T_eff)):
        po = sp.per_order_amplitudes(port)
        kz = po["kz"]
        safe = np.where(np.abs(kz) < 1e-12, 1.0, kz)
        out = np.zeros_like(np.asarray(eff))
        for col, (ex0, ey0) in enumerate(((1, 0), (0, 1))):
            einc = 1.0 + ((kx0 * ex0 + ky0 * ey0) / kz_inc) ** 2
            Ex, Ey = po["Ex"][col], po["Ey"][col]
            Ez = -(po["kx"] * Ex + po["ky"] * Ey) / safe
            e = np.real(kz / kz_inc) * (np.abs(Ex) ** 2 + np.abs(Ey) ** 2
                                        + np.abs(Ez) ** 2) / einc
            out[col] = np.where(np.real(kz) > 0, np.real(e), 0.0)
        assert np.max(np.abs(out - np.asarray(eff))) < tol


def test_b_classical_oblique_amplitudes_vs_rcwa():
    th = np.deg2rad(20.0)
    sp = _pmm_grating(th, 0.0)
    o, R, T, Jr = sp.solve()
    rr = _rcwa_grating(th, 0.0)
    _assert_amp_parity(sp, rr, tol=1e-3)
    _assert_flux_closure(sp, R, T)
    assert np.max(np.abs(sp.jones_transmission()
                         - rr.jones_transmission())) < 1e-3
    assert np.max(np.abs(Jr - rr.jones_reflection())) < 1e-3


def test_b_conical_patterned_amplitudes_vs_rcwa():
    """The nodal-conical cascade (the AUDIT_PMM_CONICAL fix path) exposes
    per-order amplitudes that agree with RCWA in COMPLEX value -- the
    per-order cross-engine referee the order-summed gates could not give."""
    th, ph = np.deg2rad(30.0), np.deg2rad(25.0)
    sp = _pmm_grating(th, ph)
    o, R, T, Jr = sp.solve()
    rr = _rcwa_grating(th, ph)
    _assert_amp_parity(sp, rr, tol=1e-3)
    _assert_flux_closure(sp, R, T)
    assert np.max(np.abs(sp.jones_transmission()
                         - rr.jones_transmission())) < 1e-3


def test_b_conical_shat_superposition_synthesis():
    """The audit's driving physics: the rotated conical eigen-polarization
    ``s-hat = (-sin phi, cos phi)`` total R synthesized from the amplitudes
    matches the RCWA-amplitude oracle, while the naive power sum
    ``ux^2 R_x + uy^2 R_y`` misses the per-order cross terms."""
    th, ph = np.deg2rad(30.0), np.deg2rad(25.0)
    sp = _pmm_grating(th, ph)
    _o, R, _T, _J = sp.solve()
    rr = _rcwa_grating(th, ph)
    ux, uy = -np.sin(ph), np.cos(ph)

    def _r_total(po):
        p0 = int(np.where(po["orders"] == 0)[0][0])
        kz_inc = float(np.real(po["kz"][p0]))
        kx0 = float(np.real(po["kx"][p0]))
        ky0 = float(np.real(po["ky"][p0]))
        Ex = ux * po["Ex"][0] + uy * po["Ex"][1]
        Ey = ux * po["Ey"][0] + uy * po["Ey"][1]
        kz = po["kz"]
        safe = np.where(np.abs(kz) < 1e-12, 1.0, kz)
        Ez = -(po["kx"] * Ex + po["ky"] * Ey) / safe
        einc = (abs(ux) ** 2 + abs(uy) ** 2
                + abs((kx0 * ux + ky0 * uy) / kz_inc) ** 2)
        return float(np.sum(np.where(
            np.real(kz) > 0,
            np.real(kz / kz_inc) * (np.abs(Ex) ** 2 + np.abs(Ey) ** 2
                                    + np.abs(Ez) ** 2) / einc, 0.0)))

    R_u = _r_total(sp.per_order_amplitudes("reflection"))
    R_oracle = _r_total(rr.per_order_amplitudes("reflection"))
    R_naive = ux ** 2 * R[0].sum() + uy ** 2 * R[1].sum()
    assert abs(R_u - R_oracle) < 1e-4
    assert abs(R_u - R_naive) > 5e-3       # the cross terms are REAL physics


def test_b_conical_uniform_fourier_path_exact():
    """A uniform slab at conical incidence (the analytic Fourier path) must
    match RCWA exactly -- both engines are exact for constants."""
    th, ph = np.deg2rad(30.0), np.deg2rad(25.0)
    sp = PMMStack(_PERIOD, n_substrate=_NSUB, degree=10)
    sp.add_layer(_DEPTH, eps=2.25 + 0j)
    sp.set_source(_WL, theta=th, phi=ph)
    _o, R, T, _J = sp.solve()
    rr = RCWAStack(_PERIOD, n_substrate=_NSUB, n_orders=5)
    rr.add_layer(_DEPTH, eps=2.25 + 0j)
    rr.set_source(_WL, theta=th, phi=ph)
    res = rr.solve()
    _assert_amp_parity(sp, res, tol=1e-12)
    _assert_flux_closure(sp, R, T)
    assert np.max(np.abs(sp.jones_transmission()
                         - res.jones_transmission())) < 1e-12


def test_b_evanescent_kz_public_branch():
    """The exported kz must carry the PUBLIC decaying branch (Im >= 0) on
    every evanescent order, both ports, both solve paths (the naive
    conj-gauge-map exported the GROWING branch on the nodal-conical path)."""
    for theta, phi in ((np.deg2rad(20.0), 0.0),
                       (np.deg2rad(30.0), np.deg2rad(25.0))):
        sp = _pmm_grating(theta, phi)
        sp.solve()
        for port in ("reflection", "transmission"):
            kz = sp.per_order_amplitudes(port)["kz"]
            ev = np.abs(np.real(kz)) < 1e-9
            assert np.any(ev)                       # gate exercises the branch
            assert np.min(np.imag(kz[ev])) > 0.0


def test_b_accessor_contract():
    """Raise before solve; port validation; supersede-on-re-source (the
    audit-P1-04 invalidation contract, extended to the modal slot)."""
    sp = _pmm_grating(np.deg2rad(20.0), 0.0)
    with pytest.raises(ValueError, match="no per-order amplitudes"):
        sp.per_order_amplitudes()
    with pytest.raises(ValueError, match="no per-order amplitudes"):
        sp.jones_transmission()
    sp.solve()
    with pytest.raises(ValueError, match="port must be"):
        sp.per_order_amplitudes(port="up")
    sp.per_order_amplitudes()                       # retained after solve
    sp.set_source(_WL, theta=0.1)                   # re-source invalidates
    with pytest.raises(ValueError, match="no per-order amplitudes"):
        sp.per_order_amplitudes()
