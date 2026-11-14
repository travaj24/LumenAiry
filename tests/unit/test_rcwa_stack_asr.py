"""Adaptive spatial resolution (ASR) on the 1-D ``RCWAStack`` (``asr=``).

Formulation: ``lumenairy/elements/rcwa/_asr_stack.py`` (module docstring) and
``docs/rcwa_stack_asr.md``.  One coordinate map for the whole stack (walls of
every patterned layer on coordinate lines, ``'sine4'`` profile), covariant
Li factorization, the half-spaces through an oversampled outgoing admittance,
the patterned layers through a balanced first-order eigensolve.

Fixtures
--------
* A: a 175 nm metal strip (eps -74+6i, 50 nm) over 1/7/5/7/1 nm dielectric and
  lossy-oxide layers over a 70 nm metal mirror (eps -160+30i); period 370 nm,
  air above, n = 1.447 below, 1300 nm.  TM sits on a gap resonance.
* B: fixture A's stack at period 1200 nm with a 500 nm strip (orders +/-1
  propagate in the substrate).
* D: a lossless lamellar grating (eps 4 / air, duty 0.4, 300 nm) on n = 1.5,
  period 1000 nm, 800 nm, theta 0.2.

References (measured 2026-11-14):
* A TM: PMMStack, geometric grading ratio 0.17, degree 24 x 12 elements:
  0.05715711017 + 0.13122118072i (degree 20 sits 5.2e-7 from it and ASR at
  n_orders 150 / 200 / 300 within 9e-8 of it: its error is about 1e-7).
* A TM at theta 0.3 and B TM: the same PMM setting.
* A TE: single-element PMM degree 60 (-0.8996752029 - 0.4078972484i; degree 80
  and Chebyshev 10 x 8 agree to 5e-12).
* D TE: single-element PMM degree 60 (degree 40 agrees to 8.6e-12).

Measured errors |r - r_ref| at n_orders 20 (2026-11-14; Windows/MKL and
WSL/OpenBLAS agree to the printed digits on every reading in this file):
A TM ASR 4.7e-6 / 'li' 4.2e-2; A TM theta 0.3 4.2e-6 / 4.5e-2; A TE 5.0e-11 /
1.1e-5; B TM 5.6e-7 / 4.5e-3; D TE 1.7e-8 / 3.7e-5.  Every physics bar sits
at least 4x above its reading and at least a decade below the plain-basis
value it is paired with.
"""
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

from lumenairy.elements.rcwa import RCWAStack, rcwa_convergence
from lumenairy.elements.rcwa._asr_stack import (
    _AsrMap,
    _normalize_asr,
    _resolved_eta,
)
from lumenairy.elements.rcwa.oned import _asr_metric_profile

WL, P, W = 1300e-9, 370e-9, 175e-9
F = W / P
AU, AL, AO, HF, OX, NS = -74 + 6j, -160 + 30j, 3.05, 3.53, 2.5 + 0.5j, 1.447
LAYERS = [(50e-9, AU, 1.0), (1e-9, AO, AO), (7e-9, HF, HF), (5e-9, OX, OX),
          (7e-9, HF, HF), (1e-9, AO, AO), (70e-9, AL, AL)]
R_A_TM = 0.05715711017388516 + 0.13122118072055405j
R_A_TM_TH03 = 0.058489067279635376 + 0.12269609664861383j
R_A_TE = -0.8996752029 - 0.4078972484j
R_B_TM = -0.6572375177011002 - 0.6842506177778588j
R_D_TE = -0.14291410405334798 + 0.13785652848412056j


def _stack(N, *, period=P, frac=F, asr=True, layers=LAYERS, dispersive=False,
           **kw):
    st = RCWAStack(period, n_superstrate=1.0, n_substrate=NS, n_orders=N,
                   asr=asr, **kw)
    for t, es, eg in layers:
        e = (lambda wl, v=es: v) if dispersive else es
        if es == eg:
            st.add_layer(t, eps=e)
        else:
            st.add_layer(t, segments=[(frac, e), (1.0 - frac, eg)])
    return st


def _r(N, *, pol="tm", theta=0.0, **kw):
    st = _stack(N, **kw)
    st.set_source(WL, theta=theta)
    J = np.asarray(st.solve().jones_reflection())
    return complex(J[0, 0] if pol == "tm" else J[1, 1])


def _diel(N, asr=True, theta=0.2, phi=0.0, three=False, retain=False):
    st = RCWAStack(1000e-9, n_superstrate=1.0, n_substrate=1.5, n_orders=N,
                   asr=asr)
    st.add_layer(300e-9, segments=[(0.4, 4.0), (0.6, 1.0)])
    if three:
        st.add_layer(100e-9, eps=2.0)
        st.add_layer(150e-9, segments=[(0.25, 2.25), (0.5, 1.0), (0.25, 3.0)])
    st.set_source(800e-9, theta=theta, phi=phi)
    return st.solve(retain_internal=retain)


# ---------------------------------------------------------------------------
# accuracy: the decision pairs
# ---------------------------------------------------------------------------

def test_metal_strip_stack_tm_normal():
    # measured 4.7e-6 (ASR) vs 4.2e-2 ('li') at 20; 2.0e-6 at 40
    e20 = abs(_r(20) - R_A_TM)
    e20_li = abs(_r(20, asr=None) - R_A_TM)
    e40 = abs(_r(40) - R_A_TM)
    assert e20 < 2e-5
    assert e40 < 1e-5
    assert e20_li > 1e-2
    assert e20_li / e20 > 500


def test_metal_strip_stack_tm_oblique():
    # classical mount, theta 0.3: measured 4.2e-6 vs 4.5e-2 at 20
    e = abs(_r(20, theta=0.3) - R_A_TM_TH03)
    e_li = abs(_r(20, theta=0.3, asr=None) - R_A_TM_TH03)
    assert e < 2e-5
    assert e_li > 1e-2


def test_metal_strip_stack_te_no_worse_than_plain():
    # measured 5.0e-11 vs 1.1e-5 at 20 (the reference agrees with its own
    # degree-80 rung to 5e-12)
    e = abs(_r(20, pol="te") - R_A_TE)
    e_li = abs(_r(20, pol="te", asr=None) - R_A_TE)
    assert e < 1e-8
    assert e < e_li / 100


def test_second_geometry_wide_period():
    # period 1200 nm, 500 nm strip: measured 5.6e-7 vs 4.5e-3 at 20
    e = abs(_r(20, period=1200e-9, frac=500 / 1200) - R_B_TM)
    e_li = abs(_r(20, period=1200e-9, frac=500 / 1200, asr=None) - R_B_TM)
    assert e < 5e-6
    assert e_li > 1e-3


def test_lossless_lamellar_grating_te_and_closure():
    # D TE: measured 1.7e-8 vs 3.7e-5 at 20; closure <= 1e-8 at 10 and 20
    for N in (10, 20):
        res = _diel(N)
        _o, R, T = res.efficiencies()
        assert np.max(np.abs(1.0 - R.sum(axis=1) - T.sum(axis=1))) < 1e-7
    J = _diel(20).jones_reflection()
    J_li = _diel(20, asr=None).jones_reflection()
    assert abs(J[1, 1] - R_D_TE) < 2e-7
    assert abs(J_li[1, 1] - R_D_TE) > 1e-5


def test_lossless_closure_converges_on_two_patterned_layers():
    # two patterned layers with different walls (union map, 4 sub-intervals):
    # measured |1 - R - T| <= 1.3e-7 at 20, <= 2.5e-10 at 40
    for N, bar in ((20, 1e-6), (40, 1e-8)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _o, R, T = _diel(N, three=True).efficiencies()
        assert np.max(np.abs(1.0 - R.sum(axis=1) - T.sum(axis=1))) < bar


def test_conical_mount():
    # phi -> 0 reproduces the classical mount (measured 2.2e-11); at phi 0.3
    # ASR at 30 sits on the plain solve at 120 (2.5e-6, about that plain
    # solve's own error) and closes energy (6e-12).
    Jp = _diel(30, theta=0.3).jones_reflection()
    Jc = _diel(30, theta=0.3, phi=1e-9).jones_reflection()
    assert np.max(np.abs(Jp - Jc)) < 1e-9
    res = _diel(30, theta=0.3, phi=0.3)
    _o, R, T = res.efficiencies()
    assert np.max(np.abs(1.0 - R.sum(axis=1) - T.sum(axis=1))) < 1e-9
    J_li = _diel(120, asr=None, theta=0.3, phi=0.3).jones_reflection()
    assert np.max(np.abs(res.jones_reflection() - J_li)) < 2e-5
    assert abs(res.jones_reflection()[0, 1]) > 1e-3   # genuinely conical


def test_sine_profile_also_beats_plain():
    e = abs(_r(40, asr=dict(eta=1.0, profile="sine")) - R_A_TM)
    assert e < 1e-3 and e < abs(_r(40, asr=None) - R_A_TM) / 10


# ---------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------

def test_default_off_bit_identical():
    a = _r(20, asr=None)
    b = _r(20, asr=False)
    st = RCWAStack(P, n_superstrate=1.0, n_substrate=NS, n_orders=20)
    for t, es, eg in LAYERS:
        if es == eg:
            st.add_layer(t, eps=es)
        else:
            st.add_layer(t, segments=[(F, es), (1.0 - F, eg)])
    st.set_source(WL)
    c = complex(np.asarray(st.solve().jones_reflection())[0, 0])
    assert a == b == c


def test_nothing_to_adapt_is_the_standard_solve():
    # uniform-only stack, and eta='auto' at n_orders <= 5 (eta 0)
    for kw in (dict(layers=[(80e-9, OX, OX)], theta=0.3), dict(N=5)):
        N = kw.pop("N", 10)
        assert _r(N, asr=True, **kw) == _r(N, asr=None, **kw)


def test_symmetry_flag_and_stabilize():
    st = _stack(20)
    st.set_source(WL)
    a = st.solve(symmetry="auto").jones_reflection()
    b = st.solve(symmetry=False).jones_reflection()
    assert np.array_equal(a, b)       # the even fold never engages under ASR
    res = st.solve(stabilize=True)
    assert abs(res.jones_reflection()[0, 0] - R_A_TM) < 1e-4


def test_dispersive_sweep_matches_concrete():
    st = _stack(20, dispersive=True)
    st.set_source(WL)
    _o, _R, _T, J = st.solve_vs_wavelength([1290e-9, 1300e-9])
    st2 = _stack(20)
    st2.set_source(WL)
    assert np.max(np.abs(J[1] - st2.solve().jones_reflection())) < 1e-9


def test_rcwa_convergence_runs_with_asr():
    st = _stack(20)
    st.set_source(WL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _res, rep = rcwa_convergence(st)
    assert rep["converged"]
    assert st.n_orders_x == 20


def _two_layer_fields(N, asr):
    st = RCWAStack(1000e-9, n_superstrate=1.0, n_substrate=1.5, n_orders=N,
                   asr=asr)
    st.add_layer(300e-9, segments=[(0.4, 4.0), (0.6, 1.0)])
    st.add_layer(100e-9, eps=2.0)
    st.set_source(800e-9, theta=0.2)
    return st.solve(retain_internal=True)


def test_internal_field_self_converged():
    a30, a60 = _two_layer_fields(30, True), _two_layer_fields(60, True)
    for inc in ((1.0, 0.0), (0.0, 1.0)):
        f30 = a30.internal_field([150e-9, 350e-9], nx=64, incident=inc)
        f60 = a60.internal_field([150e-9, 350e-9], nx=64, incident=inc)
        for c in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            sc = np.max(np.abs(f60[c])) + 1e-300
            assert np.all(np.isfinite(f30[c]))
            # measured <= 1.8e-6 relative (Ex included: E_x = D_x / eps)
            assert np.max(np.abs(f30[c] - f60[c])) < 2e-5 * sc


@pytest.mark.slow
def test_internal_field_converged_against_plain():
    # dielectric two-layer stack: ASR at 30 equals the plain solve at 300 to
    # that plain solve's own error (measured 7e-5 / 2e-5 / 5e-5 relative for
    # Ez / Hy / Hz, while plain 300 and plain 600 differ 1.5e-4 / 4.7e-5 /
    # 1.2e-4)
    a30, p300 = _two_layer_fields(30, True), _two_layer_fields(300, None)
    keep = np.ones(64, bool)
    keep[[25, 26, 27, 31, 32, 33]] = False          # nodes next to the walls
    for inc, comps in (((1.0, 0.0), ("Ez", "Hy")), ((0.0, 1.0), ("Hz", "Ey"))):
        f30 = a30.internal_field([150e-9, 350e-9], nx=64, incident=inc)
        fp = p300.internal_field([150e-9, 350e-9], nx=64, incident=inc)
        for c in comps:
            sc = np.max(np.abs(fp[c]))
            assert np.max(np.abs(f30[c][:, keep] - fp[c][:, keep])) < 5e-4 * sc


def test_layer_absorption_face_fluxes():
    st = _stack(30)
    st.set_source(WL)
    res = st.solve(retain_internal=True)
    raw = res.layer_absorption(return_raw=True)
    # face fluxes telescope: the raw split sums to the absorptance (measured
    # 2e-12 / 6e-13 on the two builds); lossless layers read 8e-15
    assert np.max(np.abs(raw.sum(axis=1) - res.absorptance())) < 1e-9
    # lossless layers absorb nothing
    assert np.max(np.abs(raw[:, [1, 2, 4, 5]])) < 1e-9
    assert np.all(raw[:, [0, 3, 6]] > -1e-9)


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------

def test_refusals():
    with pytest.raises(ValueError, match="1-D stacks only"):
        RCWAStack(P, period_y=P, n_orders=3, asr=True)
    with pytest.raises(ValueError, match="NumPy only"):
        RCWAStack(P, n_orders=3, asr=True, use_gpu=True)
    for bad in (1.5, 0.0, -0.1, "x", dict(bad=1), dict(profile="x")):
        with pytest.raises(ValueError):
            RCWAStack(P, n_orders=3, asr=bad)
    cases = {
        "eps_cell raster": dict(eps_cell=np.where(np.arange(64) < 30, AU,
                                                  1.0)),
        "tensor": dict(eps=np.diag([2.0, 3.0, 2.0])),
        "formulation='laurent'": dict(segments=[(F, AU), (1 - F, 1.0)],
                                      formulation="laurent"),
    }
    for what, kw in cases.items():
        st = RCWAStack(P, n_orders=10, asr=True)
        st.add_layer(50e-9, **kw)
        st.set_source(WL)
        with pytest.raises(ValueError, match=re.escape(what)):
            st.solve()


def test_jax_refused():
    jnp = pytest.importorskip("jax.numpy")
    st = RCWAStack(P, n_orders=10, asr=True)
    st.add_layer(50e-9, segments=[(F, AU), (1 - F, 1.0)])
    st.add_layer(20e-9, eps=jnp.asarray(2.0 + 0j))
    st.set_source(WL)
    with pytest.raises(ValueError):
        st.solve()


# ---------------------------------------------------------------------------
# the coordinate map
# ---------------------------------------------------------------------------

def _legacy_metric_profile(duty_cycle, eta, n_samples):
    """Frozen copy of the 5.50 ``oned._asr_metric_profile`` arithmetic."""
    u = (np.arange(n_samples) + 0.5) / n_samples
    u1 = float(duty_cycle)
    f = np.ones(n_samples)
    x = u.copy()
    tp = 2.0 * np.pi
    in_ridge = u < u1
    if u1 > 0.0:
        a = u[in_ridge]
        f[in_ridge] = 1.0 - eta * np.cos(tp * a / u1)
        x[in_ridge] = a - (eta * u1 / tp) * np.sin(tp * a / u1)
    w = 1.0 - u1
    if w > 0.0:
        b = u[~in_ridge] - u1
        f[~in_ridge] = 1.0 - eta * np.cos(tp * b / w)
        x[~in_ridge] = u[~in_ridge] - (eta * w / tp) * np.sin(tp * b / w)
    return u, f, x, in_ridge


def test_single_layer_profile_is_the_shared_map():
    for duty in (0.0, 0.3, 0.5, 1.0):
        for eta in (0.0, 0.5, 0.9):
            new = _asr_metric_profile(duty, eta, 512)
            old = _legacy_metric_profile(duty, eta, 512)
            for a, b in zip(new, old):
                assert np.allclose(a, b, rtol=0.0, atol=1e-14)


def test_one_map_implementation():
    root = Path(__file__).resolve().parents[2] / "lumenairy"
    defs = [p for p in root.rglob("*.py")
            if re.search(r"^class _AsrMap\b", p.read_text(encoding="utf-8"),
                         re.M)]
    assert [p.name for p in defs] == ["_asr_stack.py"]
    oned = (root / "elements" / "rcwa" / "oned.py").read_text(encoding="utf-8")
    body = oned.split("def _asr_metric_profile", 1)[1].split("\ndef ", 1)[0]
    assert "_AsrMap(" in body and "np.cos" not in body


@pytest.mark.parametrize("profile", ["sine", "sine4"])
def test_map_invariants(profile):
    mp = _AsrMap([0.1, 0.35, 0.8], 1.0, profile)
    x, f = mp.samples(mp.b)
    assert np.allclose(x, mp.b, atol=1e-15)          # walls stay put
    assert np.all(f >= -1e-15) and np.max(f[:3]) < 1e-12
    u = np.linspace(-0.3, 1.7, 401)
    xu, fu = mp.samples(u)
    assert np.all(np.diff(xu) >= 0.0)
    ub, fb = mp.inverse(xu)
    # x(u) is flat to s^3 / s^5 at a wall, so u is recoverable only where f
    # is not tiny; x(u(x)) = x everywhere
    assert np.max(np.abs(mp.samples(ub)[0] - xu)) < 1e-14
    ok = fu > 1e-3
    assert np.max(np.abs(ub - u)[ok]) < 1e-9
    assert np.allclose(fb[ok], fu[ok], atol=1e-9)
    # exact coefficients vs a fine quadrature of the same piecewise function
    vals = np.array([2.0 - 1j, 5.0, -3.0 + 0.5j])
    n = 1 << 16
    us = (np.arange(n) + 0.5) / n
    j = np.clip(np.searchsorted(np.append(mp.b, mp.b[0] + 1),
                                mp.b[0] + np.mod(us - mp.b[0], 1.0),
                                side="right") - 1, 0, 2)
    _x, fs = mp.samples(us)
    ks = np.arange(-8, 9)
    c_q = (np.fft.fft(vals[j] * fs) / n)[ks % n] * np.exp(-1j * np.pi * ks / n)
    c = mp.fourier_coeffs(vals, 9)
    assert np.max(np.abs(c - c_q)) < 2e-4               # O(1/n) step error
    assert abs(mp.fourier_coeffs(np.ones(3), 1)[0] - 1.0) < 1e-14  # <f> = 1


def test_normalize_and_auto_rule():
    assert _normalize_asr(None) is None and _normalize_asr(False) is None
    assert _normalize_asr(True) == ("sine4", "auto")
    assert _normalize_asr(0.7) == ("sine4", 0.7)
    assert _normalize_asr(dict(profile="sine")) == ("sine", "auto")
    assert [_resolved_eta("auto", n) for n in (3, 5, 10, 15, 40)] == \
        [0.0, 0.0, 0.5, 1.0, 1.0]
    assert _resolved_eta(0.3, 40) == 0.3
    assert RCWAStack(P, asr=True).asr == dict(profile="sine4", eta="auto")
    assert RCWAStack(P).asr is None


# ---------------------------------------------------------------------------
# heavier ladders
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_floor_and_self_convergence():
    # measured: 100 / 150 differ 4.4e-8 / 5.0e-8 (Windows / WSL); 150 sits
    # 8.9e-8 / 8.5e-8 from the reference, whose own error is about 1e-7
    r100, r150 = _r(100), _r(150)
    assert abs(r100 - r150) < 2e-7
    assert abs(r150 - R_A_TM) < 4e-7


@pytest.mark.slow
def test_neighbouring_n_scatter_removed():
    # 'li' over N = 40..50 spans 1.7e-2 .. 4.5e-2; ASR 8e-7 .. 5e-6
    e_asr = [abs(_r(N) - R_A_TM) for N in range(40, 51)]
    e_li = [abs(_r(N, asr=None) - R_A_TM) for N in range(40, 51)]
    assert max(e_asr) < 1e-5
    assert min(e_li) > 1e-2 and max(e_li) / min(e_li) > 2
