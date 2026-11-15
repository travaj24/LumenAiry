"""PMMStack laterally-uniform-layer fast path (``pmm/_uniform_cascade.py``).

Uniform isotropic vertical layers take their modes from the shared geometric
eig, and every run of consecutive uniform media (half-spaces included) is
cascaded elementwise and starred in as one S-matrix.  The dense per-layer path
is still reachable through the private switch ``stack._UNIFORM_FAST_PATH``,
which is what every cross-path comparison below forces.

The strip / spacer / mirror stack (a generic fixture): period 500 nm, a 180 nm
wide / 40 nm thick metal strip (eps -90+7i) in air over 2 / 6 / 3 / 6 / 2 nm
dielectric and lossy films (eps 2.8 / 4.2 / 2.2+0.6i / 4.2 / 2.8) over an
80 nm metal mirror (eps -130+20i), air above, n = 1.5 below, 1500 nm.  The
"graded" variant replaces the 3 nm lossy film by 52 graded sub-layers.
"""
import time
import warnings

import numpy as np
import pytest

import lumenairy.elements.pmm._core as C
import lumenairy.elements.pmm.stack as S
from lumenairy.elements.pmm import PMMStack
from lumenairy.elements.pmm._uniform_cascade import (
    UNIFORM_LAYER_LABEL,
    _diag_interface,
    _uniform_iso_eps,
    _uniform_medium_diag,
)

WL, P, W = 1500e-9, 500e-9, 180e-9
F = W / P
E_STRIP, E_MIRROR = -90.0 + 7.0j, -130.0 + 20.0j
E_LOW, E_HIGH, E_LOSSY = 2.8 + 0j, 4.2 + 0j, 2.2 + 0.6j
N_SUB = 1.5


def _lossy_sublayers(n=52, t=3e-9):
    i = np.arange(n + 1)
    z = 0.5 * (1.0 - np.cos(np.pi * i / n)) * t
    zc = 0.5 * (z[:-1] + z[1:]) / t
    eps = (2.0 + 0.3j) + np.sin(np.pi * zc) ** 2 * (1.0 + 0.3j)
    return list(zip(np.diff(z), eps))


def _layers(graded=0):
    L = [(40e-9, E_STRIP, 1.0 + 0j), (2e-9, E_LOW, E_LOW),
         (6e-9, E_HIGH, E_HIGH)]
    L += ([(t, e, e) for t, e in _lossy_sublayers(graded)] if graded
          else [(3e-9, E_LOSSY, E_LOSSY)])
    L += [(6e-9, E_HIGH, E_HIGH), (2e-9, E_LOW, E_LOW),
          (80e-9, E_MIRROR, E_MIRROR)]
    return L


def _fixture(degree=20, epr=3, graded=0, theta=0.0, **kw):
    st = PMMStack(P, n_substrate=N_SUB, n_superstrate=1.0, degree=degree,
                  elements_per_region=epr, **kw)
    for t, es, eg in _layers(graded):
        if es == eg:
            st.add_layer(t, eps=es)
        else:
            st.add_layer(t, segments=[(F, es), (1.0 - F, eg)])
    st.set_source(WL, theta=theta)
    return st


@pytest.fixture
def dense_switch(monkeypatch):
    """Run ``fn`` on the fast path and on the dense path."""
    def run(fn):
        monkeypatch.setattr(S, "_UNIFORM_FAST_PATH", True)
        fast = fn()
        monkeypatch.setattr(S, "_UNIFORM_FAST_PATH", False)
        dense = fn()
        monkeypatch.setattr(S, "_UNIFORM_FAST_PATH", True)
        return fast, dense
    return run


def _maxdiff(a, b):
    return max(float(np.max(np.abs(np.asarray(x) - np.asarray(y))))
               for x, y in zip(a, b))


# --------------------------------------------------------------------------- #
# 1. fast path == dense path (strip stack and its graded variant; normal and
#    oblique)
# --------------------------------------------------------------------------- #
#: Cross-path bar on J and on the per-order R / T.  The two paths solve the
#: same modes in different eigenvector gauges (clean ``(w, 0)`` / ``(0, w)``
#: columns against a raw ``2n`` eig's degenerate mixtures), so they agree to
#: round-off, not bit for bit.  Fixture replaced by a generic one;
#: RE-MEASURED 2026-11-15 (Windows/MKL), max over J / R / T: strip stack
#: degree 20 x 3 el 3.7e-12 (theta 0) / 3.2e-12 (theta 0.3), graded variant
#: degree 12 x 3 el 2.2e-13 / 1.9e-12.  (The dense path's degenerate eig is
#: not run-to-run reproducible under threaded MKL, which is why the bar is
#: an envelope and not a reading.)  Bar = 130 x the largest reading; the
#: smallest physical signal it guards is |J| ~ 0.2 (strip stack TM), and a
#: wrong diagonal (e.g. a dropped clamp or a sign) moves J by O(1).
CROSS_PATH_BAR = 5e-10


@pytest.mark.parametrize("theta", [0.0, 0.3])
def test_strip_stack_fast_equals_dense(dense_switch, theta):
    fast, dense = dense_switch(lambda: _fixture(20, 3, theta=theta).solve())
    assert np.array_equal(fast[0], dense[0])
    d = _maxdiff(fast[1:], dense[1:])
    assert d < CROSS_PATH_BAR, d


@pytest.mark.parametrize("theta", [0.0, 0.3])
def test_graded_stack_fast_equals_dense(dense_switch, theta):
    fast, dense = dense_switch(
        lambda: _fixture(12, 3, graded=52, theta=theta).solve())
    d = _maxdiff(fast[1:], dense[1:])
    assert d < CROSS_PATH_BAR, d


def test_stack_without_uniform_layers_is_bit_identical(dense_switch):
    """No uniform layer -> the fast path changes nothing (same arithmetic)."""
    def run():
        st = PMMStack(P, n_substrate=N_SUB, degree=10, elements_per_region=2)
        st.add_layer(40e-9, segments=[(F, E_STRIP), (1 - F, 1.0)])
        st.add_layer(30e-9, segments=[(0.3, 2.25), (0.7, 1.0)])
        st.set_source(WL, theta=0.2)
        return st.solve()
    fast, dense = dense_switch(run)
    for x, y in zip(fast, dense):
        assert np.array_equal(np.asarray(x), np.asarray(y))


# --------------------------------------------------------------------------- #
# 2. all-uniform Bragg stack vs an independent analytic TMM
# --------------------------------------------------------------------------- #
def _tmm_r(layers, n_sup, n_sub, wl, theta, pol):
    """Analytic thin-film reflection, ``exp(-i w t)``: the tangential-E
    reflection ``(Y0 - Yin) / (Y0 + Yin)`` with the admittance ``Y = kz``
    (TE) or ``eps / kz`` (TM) -- the lab-basis Jones entries ``J[1, 1]`` and
    ``J[0, 0]`` (equal at normal incidence, as isotropy requires)."""
    k0 = 2.0 * np.pi / wl
    kx = n_sup * np.sin(theta)

    def kz(eps):
        v = np.sqrt(complex(eps - kx ** 2))
        return -v if v.imag < 0 else v

    def Y(eps):
        return kz(eps) if pol == "te" else eps / kz(eps)

    Yin = Y(n_sub ** 2)
    for t, e in reversed(layers):
        Yl = Y(e)
        tn = np.tan(k0 * kz(e) * t)
        Yin = Yl * (Yin - 1j * Yl * tn) / (Yl - 1j * Yin * tn)
    Y0 = Y(n_sup ** 2)
    return (Y0 - Yin) / (Y0 + Yin)


def _bragg_layers():
    na, nb, wl0 = 1.45, 2.1, 1300e-9
    L = []
    for _ in range(10):
        L += [(wl0 / (4 * na), na ** 2), (wl0 / (4 * nb), nb ** 2)]
    return L + [(37e-9, -20.0 + 1.5j)]


#: Library vs analytic TMM on the all-uniform Bragg stack (10 quarter-wave
#: pairs + a 37 nm metal film, 1250 nm, degree 8).  MEASURED 2026-11-14 (max
#: over TE / TM / cross terms, theta 0 and 0.4): fast path 4.36e-13 /
#: 1.78e-13, dense path 5.48e-14 / 8.06e-14 on Windows/MKL; 4.37e-13 /
#: 1.78e-13 and 5.53e-14 / 7.99e-14 on WSL/OpenBLAS.  Bar = 100 x the
#: larger (fast-path) envelope, held by BOTH paths; |r| ~ 1 here.
TMM_BAR = 5e-11


@pytest.mark.parametrize("theta", [0.0, 0.4])
def test_bragg_stack_matches_analytic_tmm(dense_switch, theta):
    L = _bragg_layers()

    def run():
        st = PMMStack(400e-9, n_substrate=1.5, n_superstrate=1.0, degree=8)
        for t, e in L:
            st.add_layer(t, eps=e)
        st.set_source(1250e-9, theta=theta)
        return np.asarray(st.solve()[3])
    fast, dense = dense_switch(run)
    rte = _tmm_r(L, 1.0, 1.5, 1250e-9, theta, "te")
    rtm = _tmm_r(L, 1.0, 1.5, 1250e-9, theta, "tm")
    for J in (fast, dense):
        assert abs(J[1, 1] - rte) < TMM_BAR, (J[1, 1], rte)
        assert abs(J[0, 0] - rtm) < TMM_BAR, (J[0, 0], rtm)
        assert abs(J[0, 1]) < TMM_BAR and abs(J[1, 0]) < TMM_BAR


# --------------------------------------------------------------------------- #
# 3. the small-eigenvalue clamp (Rayleigh anomaly inside a uniform medium)
# --------------------------------------------------------------------------- #
def test_diagonal_uses_the_dense_clamp_at_an_exact_anomaly():
    """``lam = 0`` exactly (a mode exactly at cut-off in a uniform medium):
    the diagonal must be the one the dense ``V`` is built with -- finite, with
    ``|lam| < 1e-12`` clamped to ``1e-12`` -- not ``eps / 0``."""
    mats = C._build_sem_tensor_segments(
        P, [F, 1 - F], [C._tensor3_dict(np.eye(3))] * 2, 8, 2, True)
    k0 = 2 * np.pi / WL
    geo = C._uniform_geo_eig(mats, k0, 0.3 * k0)
    mu = geo[0]
    k = int(np.argmin(np.abs(mu - 1.0)))
    eps = complex(mu[k])                 # q^2 = eps - mu_k = 0 exactly
    W2, V2, lam, q = C._sem_modes_uniform(mats, k0, 0.3 * k0, eps, geo)
    assert lam[k] == 0 and lam[mats["n_glob"] + k] == 0
    D = _uniform_medium_diag(eps, lam, mu)
    assert np.all(np.isfinite(D))
    n = mats["n_glob"]
    # D is exactly the V / W ratio the dense path built
    w = geo[1]
    Vx = w * D[None, :n]
    Vy = w * D[None, n:]
    assert np.allclose(V2[n:, :n], Vx, rtol=0, atol=1e-9 * np.max(np.abs(Vx)))
    assert np.allclose(V2[:n, n:], Vy, rtol=0, atol=1e-9 * np.max(np.abs(Vx)))
    # the clamp is the COMPLEX 1e-12 + 0j: divide by that, as the dense
    # path does (Python's complex / float division rounds the imaginary part
    # of this eps one ulp differently).  The E_y-type entry is q^2 / lam with
    # q^2 = 0 exactly: q^2 takes the value the clamped lam implies,
    # -_LAM_CLAMP^2 (_core._safe_q2; re-pinned from 0 when the exact-cut-off
    # pole was removed -- a zero there made the mode-match singular)
    assert D[k] == -eps / np.complex128(1e-12)
    assert D[n + k] == C._safe_q2(np.complex128(0.0)) / np.complex128(1e-12)
    assert D[n + k] != 0


def test_interface_refuses_non_finite_and_the_cascade_falls_back():
    """``1 + b = 0`` (or ``b`` infinite) -> no elementwise S-matrix; the
    cascade then takes the dense kernels for that run."""
    Da = np.array([1.0, 2.0, 3.0], dtype=complex)
    assert _diag_interface(Da, -Da) is None              # 1 + b = 0
    assert _diag_interface(Da, np.array([1.0, 0.0, 1.0])) is None  # b = inf
    S_ = _diag_interface(Da, 2 * Da)
    assert S_ is not None and all(np.all(np.isfinite(x)) for x in S_)


def _spacer_stack(degree, theta):
    """A uniform AIR spacer inside the stack with the -1 order at cut-off in
    that spacer: n_sup 1.2, sin(theta) = (wl / P - 1) / 1.2."""
    st = PMMStack(1000e-9, n_substrate=N_SUB, n_superstrate=1.2,
                  degree=degree, elements_per_region=2)
    st.add_layer(40e-9, segments=[(0.4, E_STRIP), (0.6, 1.0)])
    st.add_layer(120e-9, eps=1.0)
    st.add_layer(30e-9, eps=2.25)
    st.set_source(WL, theta=theta)
    return st


_THETA_CUT = float(np.arcsin((WL / 1000e-9 - 1.0) / 1.2))


def test_mode_at_cutoff_in_a_uniform_spacer(dense_switch, monkeypatch):
    """A uniform layer carrying a mode at its own cut-off.  The fast path must
    be finite, passive (the strip absorbs: R + T <= 1) and stationary in
    degree, and the dense elementwise-fallback route must reproduce it.

    Fixture replaced by a generic one; RE-MEASURED 2026-11-15 (Windows/MKL):
    the fast path reads J_yy = -0.406801 - 0.427851i at degree 10 AND 14
    (same 6 digits; max R + T = 0.982532 in both rows), and
    -0.40731 - 0.42774i a milliradian off the anomaly, where both paths agree
    to round-off.  The DENSE per-layer eig is the path that failed here
    before its H_x partner was fixed (``_core._hx_partner_rows``): on the
    code before that fix, TE R + T = 1.365 (degree 10) / 1.332 (degree 14),
    0.10 / 0.14 off the fast path; with the fix the two paths agree to
    2.5e-13.  Which build a dense path fails on is not asserted
    (build-dependent near-degenerate eig); where it IS passive it must agree
    with the fast path."""
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for deg in (10, 14):
            fast, dense = dense_switch(
                lambda: _spacer_stack(deg, _THETA_CUT).solve())
            out[deg] = (fast, dense)
    for deg, (fast, dense) in out.items():
        for x in fast[1:]:
            assert np.all(np.isfinite(np.asarray(x)))
        tot_f = fast[1].sum(axis=1) + fast[2].sum(axis=1)
        assert float(np.max(tot_f)) <= 1.0 + 1e-6, tot_f
        tot_d = dense[1].sum(axis=1) + dense[2].sum(axis=1)
        if float(np.max(tot_d)) <= 1.0 + 1e-3:
            assert _maxdiff(fast[1:], dense[1:]) < CROSS_PATH_BAR
    # stationary in degree (the spacer's modes are exact; only the strip's
    # TM corner converges algebraically, so TE is the clean witness)
    J10, J14 = (np.asarray(out[d][0][3]) for d in (10, 14))
    assert abs(J10[1, 1] - J14[1, 1]) < CUTOFF_DEGREE_BAR
    # the elementwise run refused -> dense kernels on the shared modes: same
    import lumenairy.elements.pmm._uniform_cascade as U
    monkeypatch.setattr(U, "_diag_run", lambda *a, **k: None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Jfb = np.asarray(_spacer_stack(10, _THETA_CUT).solve()[3])
    assert float(np.max(np.abs(Jfb - J10))) < CROSS_PATH_BAR


#: TE of the cut-off-spacer stack, degree 10 vs 14 on the fast path.
#: RE-MEASURED 2026-11-15 on the generic fixture: 6.1e-7 (Windows/MKL).
#: Bar 1e-4: 160x above the reading, 1000x below the pre-fix dense path's
#: error (0.10-0.14).
CUTOFF_DEGREE_BAR = 1e-4


# --------------------------------------------------------------------------- #
# 4. what stays on the dense path
# --------------------------------------------------------------------------- #
def test_uniform_iso_classifier():
    assert _uniform_iso_eps([2.25 * np.eye(3)] * 3) == 2.25
    assert _uniform_iso_eps([2.25, 2.25]) == 2.25
    assert _uniform_iso_eps([2.25 * np.eye(3), 2.0 * np.eye(3)]) is None
    assert _uniform_iso_eps([np.diag([2.25, 2.0, 2.25])]) is None    # uniaxial
    off = 2.25 * np.eye(3, dtype=complex)
    off[0, 1] = off[1, 0] = 0.1
    assert _uniform_iso_eps([off]) is None
    assert _uniform_iso_eps(["LC"]) is None                         # key
    assert _uniform_iso_eps([lambda wl: 2.25]) is None              # callable
    assert _uniform_iso_eps([np.nan * np.eye(3)]) is None
    assert _uniform_iso_eps([]) is None


def test_traced_layer_is_not_classified_uniform():
    jnp = pytest.importorskip("jax.numpy")
    assert _uniform_iso_eps([jnp.asarray(2.25 * np.eye(3))]) is None


def test_anisotropic_uniform_layer_takes_the_dense_eig(monkeypatch):
    calls = []
    orig = S._sem_modes_tensor

    def spy(mats, *a, **k):
        calls.append({complex(b[2]["exx"]) for b in mats["elem_bnds"]}
                     | {complex(b[2]["eyy"]) for b in mats["elem_bnds"]})
        return orig(mats, *a, **k)
    monkeypatch.setattr(S, "_sem_modes_tensor", spy)
    st = PMMStack(P, n_substrate=N_SUB, degree=8, elements_per_region=2)
    st.add_layer(40e-9, segments=[(F, E_STRIP), (1 - F, 1.0)])
    st.add_layer(20e-9, eps=np.diag([2.25, 2.0, 2.25]))        # uniaxial
    st.add_layer(20e-9, eps=3.0)                               # isotropic
    st.set_source(WL, theta=0.2)
    st.solve()
    flat = [v for c in calls for v in c]
    assert 2.0 in flat and 2.25 in flat         # the uniaxial layer: dense
    assert 3.0 not in flat                      # the isotropic one: shared eig
    assert len(calls) == 2                      # strip + uniaxial only


# --------------------------------------------------------------------------- #
# 5. retained internals (per-layer cascade, uniform modes shared)
# --------------------------------------------------------------------------- #
#: Internal field / absorption, fast vs dense, RELATIVE to the field's own
#: peak.  RE-MEASURED 2026-11-15 on the strip stack (generic fixture),
#: degree 12 x 2 el: fields 3.6e-13 (theta 0) / 2.6e-13 (theta 0.3),
#: absorbed fractions 3.8e-14 / 1.5e-13 on Windows/MKL.  Bar kept: 2700 x
#: the largest reading, far below any physical change of the field.
INTERNAL_BAR = 1e-9


@pytest.mark.parametrize("theta", [0.0, 0.3])
def test_retained_internals_fast_equals_dense(dense_switch, theta):
    z = np.linspace(2e-9, 138e-9, 9)

    def run():
        st = _fixture(12, 2, theta=theta)
        out = st.solve(retain_internal=True)
        A = st.layer_absorption()
        f = [st.internal_field(z, pol=p) for p in ("tm", "te")]
        return out, A, f
    (of, Af, ff), (od, Ad, fd) = dense_switch(run)
    assert _maxdiff(of[1:], od[1:]) < CROSS_PATH_BAR
    assert float(np.max(np.abs(Af - Ad))) < INTERNAL_BAR
    for a, b in zip(ff, fd):
        peak = max(float(np.max(np.abs(b[k]))) for k in ("Ex", "Ey", "Ez"))
        hpeak = max(float(np.max(np.abs(b[k]))) for k in ("Hx", "Hy", "Hz"))
        for key in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            ref = peak if key.startswith("E") else hpeak
            rel = float(np.max(np.abs(a[key] - b[key]))) / ref
            assert rel < INTERNAL_BAR, (key, rel)


# --------------------------------------------------------------------------- #
# 6. speed (a decision, not a reading)
# --------------------------------------------------------------------------- #
def _blas2():
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:                                  # pragma: no cover
        import contextlib
        return contextlib.nullcontext()
    return threadpool_limits(2)


def test_graded_stack_fast_path_skips_the_uniform_layer_eigs(monkeypatch):
    """The deterministic half of the speed claim (no wall clock in the fast
    suite): on the graded variant the fast path runs the dense coupled eig
    only for the patterned layer, the dense path for every layer."""
    st = _fixture(8, 3, graded=52)
    calls = {True: 0, False: 0}
    orig = S._sem_modes_tensor
    for fast in (True, False):
        monkeypatch.setattr(S, "_UNIFORM_FAST_PATH", fast)

        def spy(*a, _f=fast, **k):
            calls[_f] += 1
            return orig(*a, **k)
        monkeypatch.setattr(S, "_sem_modes_tensor", spy)
        st.solve()
    monkeypatch.setattr(S, "_sem_modes_tensor", orig)
    assert calls[True] == 1, calls                 # the strip layer only
    # every byte-distinct layer (the dense path memoizes identical ones):
    # measured 46 of the 58 layers (2026-11-15)
    assert calls[False] >= 40, calls


@pytest.mark.slow
def test_graded_stack_fast_path_is_at_least_5x_faster(monkeypatch):
    """Interleaved in one process, median of 3, BLAS pinned to 2 threads.
    RE-MEASURED 2026-11-15 on the generic fixture (degree 12 x 3 el, loaded
    box): dense 1.04 s vs fast 0.077 s, 13.6x (Windows/MKL).  At degree 20
    the gap is wider (3.5 s vs 0.20 s, 17.6x)."""
    st = _fixture(12, 3, graded=52)
    st.solve()                                    # warm the geometric cache
    t = {True: [], False: []}
    with _blas2():
        for _ in range(3):
            for fast in (True, False):
                monkeypatch.setattr(S, "_UNIFORM_FAST_PATH", fast)
                t0 = time.perf_counter()
                st.solve()
                t[fast].append(time.perf_counter() - t0)
    med = {k: sorted(v)[1] for k, v in t.items()}
    assert med[False] > 5.0 * med[True], med


# --------------------------------------------------------------------------- #
# 7. mode-cut census label
# --------------------------------------------------------------------------- #
def test_census_labels_uniform_layers(monkeypatch):
    monkeypatch.setattr(C, "_MODE_CUT_CENSUS", [])
    _fixture(8, 2, theta=0.2).solve()
    sites = [row["site"] for row in C._MODE_CUT_CENSUS]
    # four DISTINCT uniform media among the six uniform layers (identical
    # layers share one modal solve), one patterned layer, two half-spaces
    assert sites.count(UNIFORM_LAYER_LABEL) == 4, sites
    assert sites.count("PMM modal solve (patterned layer)") == 1, sites
    assert sites.count("PMM modal solve (uniform half-space)") == 2, sites
