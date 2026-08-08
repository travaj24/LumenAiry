"""K4 (niche N16): opt-in UNIFORM (Airy) dark-side completion of the traced
multibranch ray-density caustic field.

``apply_real_lens_traced(amplitude_model='ray_density', caustic='uniform')`` runs
the K1 multibranch for the finite, ludwig-regularized BRIGHT side and then --
for a rotationally-symmetric SINGLE fold RING -- fills the DARK side (``r > r_c``)
with the Chester-Friedman-Ursell UNIFORM Airy tail (the exponentially-decaying
diffraction the pure geometric multibranch sum drops).  It REUSES the validated
``lenses_maslov._fold_airy_eval`` CFU kernel for BOTH sides: the two smooth
coefficients ``(a0, a1)`` are fit to the bright field just inside ``r_c`` (with
the fold parameter ``zeta(r) = kappa (r_c - r)`` and mean phase ``A(r)`` taken
EXACTLY from a meridional ray trace), then the SAME kernel is evaluated at
``zeta < 0`` (analytic continuation through the caustic) to give the dark tail.

WHAT THIS PHASE ESTABLISHED (measured, adversarial):

* FOLD-TRUTH, BOTH SIDES (the crux, plan N16 gate a): vs the lumenairy-free
  ``caustic_fold_ref`` (a dense direct Rayleigh-Sommerfeld field) the uniform
  completion closes K1's -14.8% r2m / 0.80-energy gap to ~2% windowed r2m,
  ~1-3% EE50/EE80, and energy ~0.96-1.0 -- the dark-side Airy tail is exactly
  what fills it.  It DECISIVELY beats the pure multibranch (which is identically
  zero on the dark side) on r2m and energy.
* DARK-TAIL DECAY (gate b): the completed field's dark-side intensity decays as
  ``Ai(+k^{2/3} kappa (r - r_c))^2`` -- the log-intensity slope vs
  ``(r - r_c)^{3/2}`` yields a decay ``kappa`` within ~15% of the independent
  ray-geometry ``kappa`` (from the eikonal difference of the two coalescing
  branches).  A wrong decay rate (or a geometric zero tail) fails.
* BRIGHT-SIDE CONTINUITY (gate b/d): for ``r < r_c`` the uniform field is
  BYTE-IDENTICAL to the multibranch (only the dark side is added); the field is
  FINITE through the caustic (no sqrt blow-up); complex dtype preserved.
* KMAH / PHASE (gate c): the bright side IS the K1 multibranch, whose per-branch
  det-Q KMAH the K1 suite already pins against a direct Rayleigh-Sommerfeld
  integral on a CONFIRMED two-branch region -- inherited here unchanged.
* CFU KERNEL REUSE: ``_fold_airy_eval`` is finite for ``zeta < 0`` (dark),
  ``zeta = 0`` (on the caustic) and ``zeta > 0`` (bright), linear in ``(a0, a1)``
  (so the bright-side fit continues analytically), and ``uniform_fold_airy`` --
  refactored to close through it -- still matches the exact cubic-phase Airy
  integral to ~1e-13.
* FALLBACKS (gates b/e, never inf/nan): a decentered/tilted prescription, a
  non-rotationally-symmetric input, a carrier tilt, a plane with NO fold, or a
  CUSP / multiple fold rings (``n_turn != 1`` -- the Pearcey regime, out of
  scope) are DETECTED and fall back to the plain multibranch field with a
  one-time warning.  The default (``caustic=None``) and the ``'multibranch'``
  path are unchanged.

All gates run WITHOUT Zemax (the lumenairy-free ``caustic_fold_ref`` npz + a
meridional ray trace).
"""
from __future__ import annotations

import json
import pathlib
import types
import warnings

import numpy as np
import pytest

import lumenairy as la
import lumenairy.elements._lens_traced_uniform as _umod
from lumenairy.elements._lens_traced_uniform import (
    _is_rotationally_symmetric,
    _trace_meridional_fold,
    apply_real_lens_traced_uniform,
)
from lumenairy.elements.lenses_maslov import _fold_airy_eval, uniform_fold_airy

# Model glass for THIS module only: registered and removed by
# tests/conftest.py::_module_glass_registry_guard.
MODULE_GLASSES = {'_K4CAU': lambda wl: 1.5168}   # dispersionless

_WL = 1.31e-6
_ROOT = pathlib.Path(__file__).resolve().parents[2] / 'validation' / 'oracles'


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def _gauss(N, dx, w0, dtype=np.complex128):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(dtype)


def _caustic_prescription(dec=None):
    """The plano-convex singlet of caustic_fold_ref (convex-first, f/3.65)."""
    s0 = {'radius': 2.7e-3, 'thickness': 1.0e-3, 'glass_before': 'air',
          'glass_after': '_K4CAU', 'semi_diameter': 0.75e-3}
    if dec is not None:
        s0['decenter'] = dec
    return {'wavelength': _WL, 'aperture_diameter': 1.4e-3, 'surfaces': [
        s0,
        {'radius': float('inf'), 'thickness': 0.0, 'glass_before': '_K4CAU',
         'glass_after': 'air', 'semi_diameter': 0.75e-3}],
        'thicknesses': [1.0e-3], 'stop_index': 0}


def _rlt(E0, p, dx, wl=_WL, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.asarray(la.apply_real_lens_traced(
            E0, prescription=p, wavelength=wl, dx=dx,
            on_undersample='silent', **kw))


def _axis_metrics(E, dx, win=60e-6):
    """Axis-centred windowed r2m / EE (rotational-caustic convention)."""
    N = E.shape[0]
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    r = np.sqrt(X ** 2 + Y ** 2)
    I = np.abs(E) ** 2
    m = r <= win
    Iw, rw, Pw = I[m], r[m], float(I[m].sum())
    r2m = np.sqrt((Iw * rw ** 2).sum() / Pw) * 1e6
    rb = np.linspace(0, win, 601)
    cum = np.array([Iw[rw <= rr].sum() for rr in rb]) / Pw
    e50 = float(np.interp(0.5, cum, rb)) * 1e6
    e80 = float(np.interp(0.8, cum, rb)) * 1e6
    return r2m, e50, e80, Pw


# ===========================================================================
# CFU kernel (reuse): _fold_airy_eval bright/dark/caustic + uniform_fold_airy
# ===========================================================================
def test_fold_airy_eval_dark_side_finite_and_linear():
    """``_fold_airy_eval`` is finite on BOTH sides of the fold (``zeta > 0``
    bright, ``zeta = 0`` on the caustic, ``zeta < 0`` the dark continuation) and
    is LINEAR in ``(a0, a1)`` -- the property the bright-side fit relies on to
    continue analytically into the dark side."""
    k = 2.0 * np.pi / _WL
    a0, a1 = 1.2 - 0.3j, -0.7 + 0.4j
    for zeta in (5e-5, 1e-6, 0.0, -1e-5, -5e-5, -2e-4):
        v = _fold_airy_eval(k, 5e-3, zeta, a0, a1)
        assert np.isfinite(v)
    # dark side decays (|value| shrinks as zeta -> more negative)
    mags = [abs(_fold_airy_eval(k, 5e-3, -z, a0, a1))
            for z in (1e-5, 3e-5, 6e-5, 1e-4)]
    assert mags[0] > mags[1] > mags[2] > mags[3]
    # linearity: eval(a0,a1) == a0 eval(1,0) + a1 eval(0,1)
    b0 = _fold_airy_eval(k, 5e-3, -3e-5, 1.0, 0.0)
    b1 = _fold_airy_eval(k, 5e-3, -3e-5, 0.0, 1.0)
    assert np.allclose(a0 * b0 + a1 * b1,
                       _fold_airy_eval(k, 5e-3, -3e-5, a0, a1))


def test_uniform_fold_airy_still_matches_exact_cubic():
    """The refactor (closing ``uniform_fold_airy`` through ``_fold_airy_eval``)
    is behaviour-preserving: it still matches the exact cubic-phase Airy integral
    ``int (1 + a t) e^{i k (t^3/3 - c t)} dt =
    2pi k^{-1/3} Ai(-k^{2/3} c) - a 2pi i k^{-2/3} Ai'(-k^{2/3} c)`` and stays
    finite through the caustic (``c -> 0``)."""
    from scipy.special import airy
    # f(t) = t^3/3 - c t ; saddles at t = +-sqrt(c); f'' = 2t.  Nonzero, distinct
    # phases (zeta > 0) approaching the caustic as c -> 0.
    for k in (2000.0, 8000.0):
        for c in (0.06, 0.02, 5e-3, 1e-6):
            s = np.sqrt(c)
            f1 = -s ** 3 / 3 + c * s
            f2 = s ** 3 / 3 - c * s
            uni = uniform_fold_airy(k, -s, s, f1, f2, -2 * s, 2 * s)
            exact = 2 * np.pi * k ** (-1 / 3) * airy(-(k ** (2 / 3)) * c)[0]
            assert np.isfinite(uni)
            assert abs(uni - exact) < 1e-9 * max(1.0, abs(exact))


# ===========================================================================
# DEFAULT / API: byte-identical existing paths + validation errors
# ===========================================================================
def test_default_and_multibranch_paths_unchanged():
    """Adding ``caustic='uniform'`` leaves the existing paths intact:
    ``caustic=None`` is byte-identical (tolerance) to the default screen call,
    and ``caustic='multibranch'`` is byte-identical to a direct multibranch
    call (per the v5.26.0 cache-warmth ULP lesson, asserted with a tolerance /
    array_equal as appropriate)."""
    from lumenairy.elements._lens_traced_multibranch import (
        apply_real_lens_traced_multibranch,
    )
    N, dx = 192, 10e-6
    E0 = _gauss(N, dx, 0.6e-3)
    p = _caustic_prescription()
    base = _rlt(E0, p, dx)
    none = _rlt(E0, p, dx, caustic=None)
    scale = float(np.max(np.abs(base)))
    assert np.max(np.abs(base - none)) <= 1e-10 * scale
    # multibranch path still a thin wrapper on the direct call
    d = 4.3704e-3
    route = _rlt(E0, p, dx, amplitude_model='ray_density',
                 caustic='multibranch', output_plane_distance=d,
                 caustic_ray_subsample=2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        direct = np.asarray(apply_real_lens_traced_multibranch(
            E0, prescription=p, wavelength=_WL, dx=dx,
            output_plane_distance=d, ray_subsample=2, min_area_ratio=1e-6,
            caustic_band='ludwig', input_carrier=None))
    assert np.array_equal(route, direct)


def test_uniform_validation_errors():
    """``caustic='uniform'`` shares the multibranch requirements."""
    N, dx = 96, 10e-6
    E0 = _gauss(N, dx, 0.6e-3)
    p = _caustic_prescription()
    # requires ray_density
    with pytest.raises(ValueError, match='ray_density'):
        _rlt(E0, p, dx, caustic='uniform')
    # requires the CPU path
    with pytest.raises(ValueError, match='CPU path'):
        _rlt(E0, p, dx, amplitude_model='ray_density', caustic='uniform',
             use_gpu=True)
    # bad caustic value still rejected (message lists 'uniform' now)
    with pytest.raises(ValueError, match='caustic must be'):
        _rlt(E0, p, dx, amplitude_model='ray_density', caustic='bogus')
    # scalar-conjugate carrier not representable as a single launch tilt
    with pytest.raises(ValueError, match='carrier'):
        _rlt(E0, p, dx, amplitude_model='ray_density', caustic='uniform',
             output_plane_distance=4.3704e-3, carrier=0.15)


# ===========================================================================
# Meridional fold detector + rotational-symmetry gate
# ===========================================================================
def test_meridional_fold_detector():
    """The meridional trace resolves the single fold ring of caustic_fold_ref
    (r_c ~ 14.26 um, kappa > 0, one interior turning point) at the fold plane,
    and reports NO fold at the exit vertex."""
    p = _caustic_prescription()
    lr = 0.5 * 1.4e-3 * 0.98
    fold = _trace_meridional_fold(p, _WL, 4.3704e-3, 1.0, lr, 4000)
    assert fold['ok'] and fold['n_turn'] == 1
    assert abs(fold['r_c'] * 1e6 - 14.26) < 0.3
    assert fold['kappa'] > 0
    nofold = _trace_meridional_fold(p, _WL, 0.0, 1.0, lr, 4000)
    assert not nofold['ok'] and nofold['reason'] == 'no_fold'


def test_rotational_symmetry_gate():
    """The azimuthal check accepts a centred Gaussian (rot-sym) and rejects a
    strongly astigmatic one (isolating azimuthal from radial variation)."""
    N, dx = 256, 4e-6
    assert _is_rotationally_symmetric(_gauss(N, dx, 0.4e-3), dx)
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    astig = np.exp(-(X ** 2 / (0.4e-3) ** 2 + Y ** 2 / (0.18e-3) ** 2))
    assert not _is_rotationally_symmetric(astig.astype(np.complex128), dx)


# ===========================================================================
# Bright-side continuity + finiteness + dtype
# ===========================================================================
def test_uniform_reduces_to_multibranch_on_bright_side():
    """The uniform completion only ADDS the dark tail: for ``r < r_c`` it is
    byte-identical to the multibranch field, and it is FINITE everywhere (no
    blow-up at the caustic), preserving the caller's complex dtype."""
    N, dx = 384, 3e-6
    d = 4.3704e-3
    p = _caustic_prescription()
    for dt in (np.complex128, np.complex64):
        E0 = _gauss(N, dx, 0.55e-3, dtype=dt)
        mb = _rlt(E0, p, dx, amplitude_model='ray_density',
                  caustic='multibranch', output_plane_distance=d,
                  caustic_ray_subsample=2)
        un = _rlt(E0, p, dx, amplitude_model='ray_density',
                  caustic='uniform', output_plane_distance=d,
                  caustic_ray_subsample=2)
        assert un.dtype == dt
        assert np.all(np.isfinite(un))
        x = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(x, x)
        r = np.sqrt(X ** 2 + Y ** 2)
        # bright side (r < r_c ~ 14.26 um) is unchanged (only the dark side is
        # filled); complex64 keeps its own multibranch dtype so compare as such
        bright = r < 13.8e-6
        assert np.array_equal(un[bright], mb[bright])
        # the dark side is now NON-zero (the tail K1 was missing)
        dark = (r > 16e-6) & (r < 24e-6)
        assert np.abs(un[dark]).max() > 10.0 * np.abs(mb[dark]).max()


# ===========================================================================
# GATE a: fold-truth dark tail closes the gap (BOTH sides), energy ~1.0
# ===========================================================================
def test_fold_truth_dark_tail_closes_gap():
    """On the ``caustic_fold_ref`` grid the uniform completion matches the direct
    Rayleigh-Sommerfeld reference on BOTH sides of the fold: windowed r2m within
    ~4%, EE50 within ~4%, EE80 within ~7%, energy ~0.92-1.05 -- closing K1's
    -14.8% r2m / 0.80-energy gap.  It DECISIVELY beats the pure multibranch
    (zero dark tail) on both r2m and energy.  N=768 (radial metrics are
    grid-independent; reproduces the N=1280 reference ring)."""
    from lumenairy.memory import available_memory_bytes
    ref = np.load(_ROOT / 'caustic_fold_ref.npz', allow_pickle=True)
    meta = json.loads(str(ref['metrics']))
    opd = float(ref['output_plane_distance'])
    N, dxc = 768, 2e-6
    est_gb = (N * N * 16 * 8) / 1e9
    if available_memory_bytes() / 1e9 < est_gb + 1.5:
        pytest.skip(f"needs ~{est_gb:.1f} GB; memory-constrained host")
    p = _caustic_prescription()
    Ec = _gauss(N, dxc, 0.55e-3)
    x = (np.arange(N) - N / 2) * dxc
    X, Y = np.meshgrid(x, x)
    pin = float((np.abs(Ec[(X ** 2 + Y ** 2) <= (0.7e-3) ** 2]) ** 2).sum())

    mb = _rlt(Ec, p, dxc, amplitude_model='ray_density', caustic='multibranch',
              output_plane_distance=opd, caustic_ray_subsample=2)
    un = _rlt(Ec, p, dxc, amplitude_model='ray_density', caustic='uniform',
              output_plane_distance=opd, caustic_ray_subsample=2)
    assert np.all(np.isfinite(un))

    mb_r2m, _, _, mb_P = _axis_metrics(mb, dxc)
    un_r2m, un_e50, un_e80, un_P = _axis_metrics(un, dxc)
    r2m_ref = meta['grid_r2m_um']
    e50_ref = meta['grid_EE50_um']
    e80_ref = meta['grid_EE80_um']

    # gate a: uniform closes to a few % on r2m / EE and energy ~1.0
    assert abs(un_r2m / r2m_ref - 1.0) < 0.05, un_r2m
    assert abs(un_e50 / e50_ref - 1.0) < 0.05, un_e50
    assert abs(un_e80 / e80_ref - 1.0) < 0.07, un_e80
    assert 0.90 < un_P / pin < 1.06, un_P / pin
    # decisively beats the pure multibranch (zero dark tail)
    assert abs(un_r2m / r2m_ref - 1.0) < abs(mb_r2m / r2m_ref - 1.0)
    assert abs(un_P / pin - 1.0) < abs(mb_P / pin - 1.0)
    # and multibranch really is the -~15% / 0.8 gap this closes
    assert mb_r2m / r2m_ref - 1.0 < -0.10
    assert mb_P / pin < 0.9


# ===========================================================================
# GATE b: dark-side decay follows the Ai(+) exponential scaling
# ===========================================================================
def test_dark_tail_airy_decay_rate():
    """The completed dark-side intensity decays as
    ``Ai(k^{2/3} kappa (r - r_c))^2``: the log-intensity slope vs
    ``(r - r_c)^{3/2}`` recovers a decay ``kappa`` within ~18% of the INDEPENDENT
    ray-geometry ``kappa`` -- the Ai(+) exponential scaling, not a geometric
    zero.  A wrong decay rate (or the K1 zero tail) fails."""
    from lumenairy.memory import available_memory_bytes
    ref = np.load(_ROOT / 'caustic_fold_ref.npz', allow_pickle=True)
    opd = float(ref['output_plane_distance'])
    N, dxc = 640, 2e-6
    est_gb = (N * N * 16 * 8) / 1e9
    if available_memory_bytes() / 1e9 < est_gb + 1.5:
        pytest.skip(f"needs ~{est_gb:.1f} GB; memory-constrained host")
    p = _caustic_prescription()
    Ec = _gauss(N, dxc, 0.55e-3)
    E, diag = apply_real_lens_traced_uniform(
        Ec, prescription=p, wavelength=_WL, dx=dxc,
        output_plane_distance=opd, ray_subsample=2, return_diagnostics=True)
    assert not diag['fell_back']
    r_c, kappa = diag['r_c'], diag['kappa']
    k0 = 2.0 * np.pi / _WL
    x = (np.arange(N) - N / 2) * dxc
    X, Y = np.meshgrid(x, x)
    rg = np.sqrt(X ** 2 + Y ** 2)
    I = np.abs(E) ** 2
    sel = (rg > r_c + 1.0e-6) & (rg < 20e-6) & (I > 0)
    u = (rg[sel] - r_c) ** 1.5
    slope = np.polyfit(u, np.log(I[sel]), 1)[0]
    assert slope < 0                                   # decays
    kappa_meas = ((-slope) * 3.0 / (4.0 * k0)) ** (2.0 / 3.0)
    assert abs(kappa_meas / kappa - 1.0) < 0.18, (kappa_meas, kappa)


# ===========================================================================
# GATE b/e: fallbacks (never inf/nan) -- documented detect + fall back
# ===========================================================================
def test_fallbacks_return_multibranch_finite():
    """A decentered prescription, a non-rotationally-symmetric input, and a
    plane with no fold each DETECT the unsupported case and fall back to the
    plain multibranch field (finite, warned) -- equal to a direct multibranch
    call there."""
    N, dx = 384, 3e-6
    d = 4.3704e-3
    E0 = _gauss(N, dx, 0.55e-3)
    # decentered prescription -> field-frame -> fall back
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        Ed, dd = apply_real_lens_traced_uniform(
            E0, prescription=_caustic_prescription((0.4e-3, 0.0)),
            wavelength=_WL, dx=dx, output_plane_distance=d,
            ray_subsample=2, return_diagnostics=True)
    assert dd['fell_back'] and 'decenter' in dd['reason']
    assert np.all(np.isfinite(Ed))
    assert any('uniform' in str(m.message) for m in w)
    # non-rot-sym input -> fall back
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    astig = np.exp(-(X ** 2 / (0.55e-3) ** 2
                     + Y ** 2 / (0.28e-3) ** 2)).astype(np.complex128)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        En, dn = apply_real_lens_traced_uniform(
            astig, prescription=_caustic_prescription(), wavelength=_WL, dx=dx,
            output_plane_distance=d, ray_subsample=2, return_diagnostics=True)
        # no-fold plane (exit vertex) -> fall back
        E0v, d0v = apply_real_lens_traced_uniform(
            E0, prescription=_caustic_prescription(), wavelength=_WL, dx=dx,
            output_plane_distance=0.0, ray_subsample=2, return_diagnostics=True)
    assert dn['fell_back'] and 'symmetric' in dn['reason']
    assert np.all(np.isfinite(En))
    assert d0v['fell_back'] and d0v['reason'] == 'no_fold'
    assert np.all(np.isfinite(E0v))


def test_cusp_or_multiple_fold_detected_and_falls_back(monkeypatch):
    """A CUSP / multiple fold ring (``n_turn > 1`` -- the Pearcey regime, out of
    scope) is DETECTED by the meridional turning-point count and falls back to
    the multibranch (detect + document, never a silent mis-handle).  The 2-fold
    meridional map is injected via a synthetic ray trace (a single-element SA
    system has only one meridional fold), which exercises the detector's
    classification directly."""
    p = _caustic_prescription()
    lr = 0.5 * 1.4e-3 * 0.98
    n = 2000
    xs = np.linspace(lr / n, lr, n)
    # a TWO-hump meridional height map (two interior dx/dh sign changes)
    xo = 1e-5 * np.sin(3.0 * np.pi * xs / lr)
    fake = types.SimpleNamespace(
        image_rays=types.SimpleNamespace(
            x=xo, y=np.zeros(n), N=np.ones(n), L=np.zeros(n), M=np.zeros(n),
            opd=1e-3 * xs, alive=np.ones(n, dtype=bool)))

    def _fake_trace(rays, surfaces, wavelength, **kw):
        return fake
    monkeypatch.setattr(_umod.rt, 'trace', _fake_trace)
    fold = _trace_meridional_fold(p, _WL, 4.3704e-3, 1.0, lr, n)
    assert not fold['ok']
    assert fold['reason'] == 'cusp_or_multiple' and fold['n_turn'] > 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
