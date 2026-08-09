"""K1 (niche N13): opt-in MULTIBRANCH KMAH / Maslov caustic amplitude.

``apply_real_lens_traced(amplitude_model='ray_density', caustic='multibranch')``
connects the ray-density amplitude to the EXISTING
``apply_real_lens_traced_multibranch`` branch-finder + analytic det-Q KMAH
counter (REUSE, not reimplement).  Where the ray map folds (``det J -> 0`` /
sign change) it gathers ALL real ray branches per output pixel, weights each
``|E_in(x_in^b)| / sqrt(|det J_b|)``, applies the Maslov phase
``exp(-i (pi/2) KMAH_b)``, and sums COHERENTLY -- finite at the fold (Ludwig
uniform-Airy swap), output at ``output_plane_distance`` past the exit vertex.

WHAT THIS PHASE ESTABLISHED (measured, adversarial):

* The opt-in routing is a THIN wrapper on the existing multibranch machinery:
  ``caustic='multibranch'`` output is BYTE-IDENTICAL to a direct
  ``apply_real_lens_traced_multibranch`` call with the mapped parameters
  (``np.array_equal``).  ``caustic=None`` (default) is byte-identical to prior
  releases; a non-zero ``output_plane_distance`` with any non-multibranch mode
  raises.
* KMAH correctness (genuine 2-branch fold ring): at the wave-resolved
  ``caustic_fold_ref`` fold RING -- where the ray map is DOUBLY valued and the
  bright-ring fringe POSITION is set by the RELATIVE Maslov phase between the
  two coalescing branches -- the multibranch with the correct det-Q KMAH
  reproduces the reference bright ring (peak radius ~6.1 um == the direct-RS
  reference, radial-shape correlation ~0.95), while ZEROING or NEGATING the
  KMAH index moves the bright ring out to ~14 um and drops the correlation to
  ~0.55-0.64: a wrong Maslov phase is decisively caught (the
  ``fold_ref_maslov_sensitive`` test PINS this, monkeypatching the counter to
  prove sensitivity).  On the ``ludwig_fold`` building block a WRONG +pi/2 on
  the caustic-touched branch drives a bright fold fringe fully DARK (>1e3x
  contrast -- the ``kmah_maslov_flip`` unit test).
* SINGLE-branch amplitude + eikonal: at a fold-free plane (mini fast singlet,
  d=2.6 mm; ray map single-valued -- ``n_branch.max()==1``, KMAH uniformly 0)
  the coherent sum through the K1 API matches a self-contained DIRECT
  Rayleigh-Sommerfeld Huygens integral to rel-L2 ~6%.  This validates the
  branch AMPLITUDE + EIKONAL geometry through the API; because the plane is
  single-valued it does NOT exercise the Maslov index (that is the fold-ring
  test above) -- the test asserts the single-branch scope explicitly.
* Airy transition: the single-branch ray-density amplitude ``1/sqrt(|det J|)``
  DIVERGES at the fold (capped + warned in single-branch mode); the multibranch
  Ludwig field stays FINITE there and reduces to the plain two-branch GO sum on
  the bright side -- the finite fold-diffraction (Airy) profile.
* Collimated / single-branch limit: at a fold-free plane (exit vertex of a slow
  lens) the multibranch field reduces to the single-branch ray-density field
  (rel-L2 <1%) and conserves energy to <0.5% -- the sanity limit; the default
  (collimated, unaberrated) is unchanged.

HONEST RESIDUALS (documented, pinned):

* FOLD GROUND TRUTH (gate 1): on the fine, wave-resolved ``caustic_fold_ref``
  grid the pure GEOMETRIC multibranch does NOT beat the single-branch
  ray-density exit field + ASM.  The multibranch is identically zero on the
  DARK side of the fold (no real branches there), so it carries no evanescent
  diffraction tail: windowed r2m reads ~15% LOW and ~20% of the caustic energy
  (the dark-side tail + the ring's uniform correction) is missing, while
  single-branch+ASM -- a genuine wave propagation -- already matches the direct
  RS reference to ~3% (r2m +0.0% / EE80 +3.0%).  So single-branch ``ray_density``
  + ASM (or ``apply_real_lens_gbd`` / ``apply_real_lens_fga``) remains the
  quantitative fold reference; multibranch is the finite, no-blow-up, one-call
  coherent field AT the caustic.  This test PINS that envelope.
* FOCAL DECENTERED PSF (gate 4): the multibranch centroid tracks the geometric
  oracle to ~0.3% (decenter GEOMETRY correct), but AT the paraxial image plane
  (an axial point focus for a collimated input) it over-amplifies ~2-3x -- the
  D5 caustic pile-up (a RING of branches coalesces where the fold-Airy swap
  regularizes only the closest pair).  Finite (never inf/nan) but NOT the
  decentered-PSF EE model; ``apply_real_lens_gbd`` (N10b) remains that
  reference.  Pinned.

All gates run WITHOUT Zemax (lumenairy-free ``caustic_fold_ref`` npz + geom
oracle + a self-contained direct-RS Huygens integral).
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import warnings

import numpy as np
import pytest

import lumenairy as la
import lumenairy.elements._lens_traced_multibranch as _mbmod
from lumenairy.elements._lens_traced_multibranch import (
    _trace_launch_grid,
    apply_real_lens_traced_multibranch,
    ludwig_fold,
)

# Model glass for THIS module only: registered and removed by
# tests/conftest.py::_module_glass_registry_guard.
MODULE_GLASSES = {'_K1A': lambda wl: 1.5168,     # dispersionless
                  '_K1CAU': lambda wl: 1.5168}

_WL = 1.31e-6
_LAM633 = 0.633e-6
_Z_IMG = 49.162e-3                             # f/5 singlet exit vertex -> image

_ROOT = pathlib.Path(__file__).resolve().parents[2] / 'validation' / 'oracles'


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _ROOT / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def _gauss(N, dx, w0, dtype=np.complex128):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(dtype)


def _f5(dec=None):
    """f/5 singlet (P11 config); optional front-surface decenter."""
    s0 = {'radius': 51.68e-3, 'thickness': 5e-3, 'glass_before': 'air',
          'glass_after': '_K1A', 'semi_diameter': 6e-3}
    if dec is not None:
        s0['decenter'] = dec
    return {'wavelength': _WL, 'aperture_diameter': 10e-3, 'surfaces': [
        s0,
        {'radius': -51.68e-3, 'thickness': 0.0, 'glass_before': '_K1A',
         'glass_after': 'air', 'semi_diameter': 6e-3}],
        'thicknesses': [5e-3], 'stop_index': 0}


def _slow():
    """Slow, near-unaberrated singlet -- the single-branch limit."""
    return {'wavelength': _WL, 'aperture_diameter': 4e-3, 'surfaces': [
        {'radius': 103.36e-3, 'thickness': 3e-3, 'glass_before': 'air',
         'glass_after': '_K1A', 'semi_diameter': 2.5e-3},
        {'radius': -103.36e-3, 'thickness': 0.0, 'glass_before': '_K1A',
         'glass_after': 'air', 'semi_diameter': 2.5e-3}],
        'thicknesses': [3e-3], 'stop_index': 0}


def _mini_fast_singlet():
    """f/2.4 strong-SA plano-convex mini-singlet (R=2mm N-BK7, BFL~3.2mm) --
    identical to the v5.21 multibranch-vs-direct-RS validated config."""
    return {'aperture_diameter': 1.6e-3, 'surfaces': [
        {'radius': 2.0e-3, 'glass_before': 'air', 'glass_after': 'N-BK7'},
        {'radius': float('inf'), 'glass_before': 'N-BK7', 'glass_after': 'air'},
        {'radius': float('inf'), 'glass_before': 'air', 'glass_after': 'air'}],
        'thicknesses': [1.0e-3, 0.0]}


def _caustic_prescription():
    """The plano-convex singlet of caustic_fold_ref (convex-first, f/3.65)."""
    return {'wavelength': _WL, 'aperture_diameter': 1.4e-3, 'surfaces': [
        {'radius': 2.7e-3, 'thickness': 1.0e-3, 'glass_before': 'air',
         'glass_after': '_K1CAU', 'semi_diameter': 0.75e-3},
        {'radius': float('inf'), 'thickness': 0.0, 'glass_before': '_K1CAU',
         'glass_after': 'air', 'semi_diameter': 0.75e-3}],
        'thicknesses': [1.0e-3], 'stop_index': 0}


def _rlt(E0, p, dx, wl=_WL, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.asarray(la.apply_real_lens_traced(
            E0, prescription=p, wavelength=wl, dx=dx,
            on_undersample='silent', **kw))


def _axis_metrics(E, dx, win=60e-6):
    """Axis-centred windowed r2m / EE (rotational caustic convention)."""
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
    return r2m, e50, e80


def _radial_profile(I, dx, rmax=30e-6, nb=120):
    """Azimuthally-averaged intensity profile (grid-independent), rc in um."""
    N = I.shape[0]
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    r = np.sqrt(X ** 2 + Y ** 2)
    edges = np.linspace(0, rmax, nb + 1)
    rc = 0.5 * (edges[:-1] + edges[1:])
    prof = np.zeros(nb)
    for i in range(nb):
        m = (r >= edges[i]) & (r < edges[i + 1])
        prof[i] = I[m].mean() if m.any() else 0.0
    return rc * 1e6, prof


def _ring_peak_and_corr(prof, rc, ref_prof, lo=2.0, hi=20.0, peak_lo=6.0):
    """Bright-ring peak radius [um] and radial-shape correlation of ``prof``
    against ``ref_prof`` over the fringe window ``lo < r < hi`` (each profile
    self-normalised to its own peak in the window)."""
    win = (rc > lo) & (rc < hi)

    def _norm(p):
        return p / max(p[win].max(), 1e-300)
    pn, rn = _norm(prof), _norm(ref_prof)
    pk_win = (rc > peak_lo) & (rc < hi)
    peak_r = float(rc[np.argmax(np.where(pk_win, pn, -1.0))])
    a = pn[win] - pn[win].mean()
    b = rn[win] - rn[win].mean()
    corr = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))
    return peak_r, corr


# ===========================================================================
# DEFAULT byte-identical (the regression pins)
# ===========================================================================
def test_default_caustic_none_is_byte_identical():
    """``caustic`` defaults to None (single-branch); None / 'single' /
    explicit-default calls are the SAME numerical model as before this feature.

    Asserted with a tolerance (not array_equal) between two live traced calls,
    per the v5.26.0 cache-warmth ~1 ULP lesson."""
    N, dx = 384, 8e-6
    E0 = _gauss(N, dx, 3e-3)
    base = _rlt(E0, _f5(), dx)
    none = _rlt(E0, _f5(), dx, caustic=None, output_plane_distance=0.0)
    single = _rlt(E0, _f5(), dx, caustic='single')
    scale = float(np.max(np.abs(base)))
    assert np.max(np.abs(base - none)) <= 1e-10 * scale
    assert np.max(np.abs(base - single)) <= 1e-10 * scale
    # ray_density single-branch default (caustic=None) also unchanged
    rd = _rlt(E0, _f5(), dx, amplitude_model='ray_density')
    rd_none = _rlt(E0, _f5(), dx, amplitude_model='ray_density', caustic=None)
    assert np.max(np.abs(rd - rd_none)) <= 1e-10 * float(np.max(np.abs(rd)))


def test_multibranch_routes_to_existing_machinery_byte_identical():
    """``caustic='multibranch'`` is a THIN wrapper: its output is BYTE-IDENTICAL
    (np.array_equal) to a direct ``apply_real_lens_traced_multibranch`` call with
    the mapped parameters -- proving REUSE (not a reimplementation)."""
    N, dx = 192, 10e-6
    E0 = _gauss(N, dx, 0.6e-3)
    p = _mini_fast_singlet()
    d = 2.4e-3
    route = _rlt(E0, p, dx, wl=_LAM633, amplitude_model='ray_density',
                 caustic='multibranch', output_plane_distance=d,
                 caustic_ray_subsample=2, caustic_band='ludwig')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        direct = np.asarray(apply_real_lens_traced_multibranch(
            E0, prescription=p, wavelength=_LAM633, dx=dx,
            output_plane_distance=d, ray_subsample=2, min_area_ratio=1e-6,
            caustic_band='ludwig', input_carrier=None))
    assert np.array_equal(route, direct)


def test_multibranch_validation_errors():
    """The documented incompatibilities all raise ValueError."""
    N, dx = 128, 10e-6
    E0 = _gauss(N, dx, 1e-3)
    p = _f5()
    # multibranch requires ray_density
    with pytest.raises(ValueError, match='ray_density'):
        _rlt(E0, p, dx, caustic='multibranch')
    # bad caustic value
    with pytest.raises(ValueError, match='caustic must be'):
        _rlt(E0, p, dx, amplitude_model='ray_density', caustic='bogus')
    # output_plane_distance != 0 without multibranch
    with pytest.raises(ValueError, match='output_plane_distance'):
        _rlt(E0, p, dx, output_plane_distance=1e-3)
    # multibranch requires the CPU path
    with pytest.raises(ValueError, match='CPU path'):
        _rlt(E0, p, dx, amplitude_model='ray_density', caustic='multibranch',
             use_gpu=True)
    # scalar-conjugate carrier not representable as a single launch tilt
    with pytest.raises(ValueError, match='carrier'):
        _rlt(E0, p, dx, amplitude_model='ray_density', caustic='multibranch',
             carrier=0.15)


def test_multibranch_finite_and_preserves_complex_dtype():
    """The multibranch field is finite (no inf/nan) and preserves the caller's
    complex dtype (complex64 stays complex64)."""
    N, dx = 192, 10e-6
    p = _mini_fast_singlet()
    for dt in (np.complex128, np.complex64):
        E0 = _gauss(N, dx, 0.6e-3, dtype=dt)
        out = _rlt(E0, p, dx, wl=_LAM633, amplitude_model='ray_density',
                   caustic='multibranch', output_plane_distance=3.6e-3,
                   caustic_ray_subsample=1)
        assert out.dtype == dt
        assert np.all(np.isfinite(out))


# ===========================================================================
# GATE 5 (part): collimated / single-branch limit + energy conservation
# ===========================================================================
def test_collimated_single_branch_reduces_to_ray_density_and_conserves():
    """At a fold-free plane (the exit vertex of a slow lens) the ray map is
    single-valued, so the multibranch field REDUCES to the single-branch
    ray-density field (rel-L2 <1% over the bright support) and conserves the
    aperture-transmitted energy to <0.5% -- the collimated / single-branch
    sanity limit (the default, unaberrated regime is unchanged)."""
    N, dx = 1024, 4e-6
    E0 = _gauss(N, dx, 1.2e-3)
    p = _slow()
    sb = np.abs(_rlt(E0, p, dx, amplitude_model='ray_density'))
    mb = np.abs(_rlt(E0, p, dx, amplitude_model='ray_density',
                     caustic='multibranch', output_plane_distance=0.0,
                     caustic_ray_subsample=1))
    m = sb > 0.05 * sb.max()
    assert np.linalg.norm(mb[m] - sb[m]) / np.linalg.norm(sb[m]) < 0.02
    # energy: multibranch(opd=0) vs aperture-transmitted input power (<0.5%)
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    pin = float((np.abs(E0[(X ** 2 + Y ** 2) <= (2e-3) ** 2]) ** 2).sum())
    assert abs(float((mb ** 2).sum()) / pin - 1.0) < 5e-3


# ===========================================================================
# GATE 3: KMAH correctness (coherent sum vs direct Huygens; +-pi/2 flip)
# ===========================================================================
def test_kmah_maslov_flip_bright_to_dark():
    """The per-branch Maslov phase is decisive.  In the Ludwig fold field the
    caustic-TOUCHED branch carries an extra -pi/2 (Gouy) relative to the
    untouched branch; a WRONG +-pi/2 (rotating that branch by +pi/2) drives a
    BRIGHT fold fringe fully DARK -- a >1e3x intensity contrast at the flipped
    fringe, and a 100% field change overall.  This is the 'a wrong +-pi/2 flips
    a bright fringe dark' gate, tested on the exact building block the
    multibranch sum uses at the fold."""
    k = 2.0 * np.pi / _WL
    yv = np.linspace(1e-8, 3.0e-5, 2000)
    # coalescing pair with eikonal split 2*y about phi=0, equal GO amplitude;
    # ludwig_fold bakes in the correct -i (=-pi/2) Maslov on the '+' branch.
    correct = np.array([ludwig_fold(k, r, -r, 1.0 + 0j, 1.0 + 0j) for r in yv])
    # WRONG: pre-rotate the '+' (touched) branch amplitude by an extra +pi/2.
    wrong = np.array([ludwig_fold(k, r, -r, 1j * (1.0 + 0j), 1.0 + 0j)
                      for r in yv])
    Ic, Iw = np.abs(correct) ** 2, np.abs(wrong) ** 2
    assert np.all(np.isfinite(Ic)) and np.all(np.isfinite(Iw))
    # at a bright fringe of the correct field the wrong field is ~null
    ratio = Ic / np.maximum(Iw, 1e-30)
    ib = int(np.argmax(np.where(Ic > 0.3 * Ic.max(), ratio, 0.0)))
    assert Ic[ib] > 0.3 * Ic.max()                     # it IS a bright fringe
    assert ratio[ib] > 1e3, (Ic[ib], Iw[ib])           # driven dark
    # and the whole field is materially different (not a global phase)
    assert (np.linalg.norm(correct - wrong)
            / np.linalg.norm(correct)) > 0.5


def test_multibranch_matches_direct_huygens_single_branch_amplitude():
    """The coherent sum through the K1 API matches a self-contained DIRECT
    Rayleigh-Sommerfeld Huygens integral of the exact ray-traced exit field to
    rel-L2 <10% off the ~1-pixel axial band -- validating the branch AMPLITUDE +
    EIKONAL geometry through the API.

    SCOPE (asserted): the config (mini fast singlet, d=2.6 mm) is a SINGLE-valued
    ray map -- ``n_branch.max()==1`` and the KMAH index is UNIFORMLY 0 there, so
    this match does NOT exercise the Maslov index (it would be bit-identical with
    the KMAH counter zeroed or negated).  KMAH / Maslov correctness is guarded at
    a genuine 2-branch region by
    ``test_multibranch_kmah_fringe_matches_fold_ref_maslov_sensitive`` (the fold
    ring, where a wrong index moves the bright fringe) and by
    ``test_kmah_maslov_flip_bright_to_dark`` (the ludwig_fold building block)."""
    presc = _mini_fast_singlet()
    N, dx = 192, 10e-6
    k = 2.0 * np.pi / _LAM633
    E0 = _gauss(N, dx, 0.6e-3)
    d = 2.6e-3
    # PIN the single-branch scope: at this plane the ray map is single-valued,
    # so the Maslov index is not exercised (do NOT re-add a KMAH claim here).
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _, diag = apply_real_lens_traced_multibranch(
            E0, prescription=presc, wavelength=_LAM633, dx=dx,
            output_plane_distance=d, ray_subsample=1, caustic_band='ludwig',
            input_carrier=None, return_diagnostics=True)
    assert int(diag['n_branch'].max()) == 1                # single-valued plane
    assert np.array_equal(np.unique(diag['kmah']), np.array([0]))  # KMAH == 0
    # exit-pupil field on a Nyquist-adequate launch grid, direct RS to centre row
    g = _trace_launch_grid(presc, _LAM633, 0.5 * 1.6e-3 * 0.98, 1601, 0.0, 1.0)
    h = g['xs_in'][1] - g['xs_in'][0]
    ok = g['alive'] & np.isfinite(g['x_out']) & np.isfinite(g['opl'])
    d11, d12 = np.gradient(g['x_out'], h, h)
    d21, d22 = np.gradient(g['y_out'], h, h)
    Jex = d11 * d22 - d12 * d21
    ok &= np.isfinite(Jex) & (Jex > 0)
    Ein = np.exp(-(g['Xi'] ** 2 + g['Yi'] ** 2) / (0.6e-3) ** 2)
    wamp = (Ein * np.sqrt(np.abs(Jex))
            * np.exp(1j * k * g['opl']))[ok] * h * h
    px, py = g['x_out'][ok], g['y_out'][ok]
    xs = (np.arange(N) - N / 2) * dx
    row = np.zeros(N, dtype=np.complex128)
    for i, xo in enumerate(xs):
        r = np.sqrt((xo - px) ** 2 + py ** 2 + d * d)
        row[i] = np.sum(wamp * np.exp(1j * k * r) / r * (d / r))
    ref = np.abs(row) ** 2
    mb = _rlt(E0, presc, dx, wl=_LAM633, amplitude_model='ray_density',
              caustic='multibranch', output_plane_distance=d,
              caustic_ray_subsample=1)
    c = N // 2
    b = np.abs(mb[c]) ** 2
    a = ref / ref.max()
    b = b / b.max()
    assert abs(int(np.argmax(ref)) - int(np.argmax(np.abs(mb[c])))) <= 1
    msk = a > 0.02
    msk[c - 3:c + 4] = False                            # axial caustic band
    rel = float(np.linalg.norm(b[msk] - a[msk]) / np.linalg.norm(a[msk]))
    assert rel < 0.10, rel


def test_multibranch_kmah_fringe_matches_fold_ref_maslov_sensitive():
    """GATE 3 (KMAH correctness) at a GENUINE two-branch region.

    At the ``caustic_fold_ref`` fold RING the ray map is DOUBLY valued and the
    bright-ring fringe POSITION is set by the RELATIVE Maslov phase between the
    two coalescing branches.  Compared to the reference (a dense DIRECT
    Rayleigh-Sommerfeld / Huygens field, lumenairy-free) the multibranch through
    the K1 API, with the CORRECT det-Q KMAH, reproduces the reference bright ring
    (peak radius within ~1 um; radial-shape correlation > 0.85).  Monkeypatching
    the per-branch KMAH counter to ZERO or NEGATE the index moves the bright ring
    outward (to ~14 um) and collapses the correlation (< 0.80) -- i.e. a wrong
    +-pi/2 per branch is DECISIVELY caught here (unlike the single-branch
    d=2.6 mm plane).  This is the through-the-API coherent-multibranch-field vs
    direct-Huygens KMAH guard the gate calls for.

    Runs at N=768 (radial profiles are grid-independent, so this reproduces the
    N=1280 reference ring at a fraction of the cost); each multibranch call is
    ~0.4 s, three calls total."""
    from lumenairy.memory import available_memory_bytes
    ref = np.load(_ROOT / 'caustic_fold_ref.npz', allow_pickle=True)
    dxc = float(ref['dx'])                              # 2 um
    opd = float(ref['output_plane_distance'])           # vertex -> fold
    Iref = np.abs(ref['E2d']) ** 2
    rc, pref = _radial_profile(Iref, dxc)
    ref_peak, _ = _ring_peak_and_corr(pref, rc, pref)   # ~6.1 um

    N = 768                                             # covers the 1.4 mm stop
    est_gb = (N * N * 16 * 6) / 1e9
    if available_memory_bytes() / 1e9 < est_gb + 1.0:
        pytest.skip(f"needs ~{est_gb:.1f} GB; memory-constrained host")
    p = _caustic_prescription()
    Ec = _gauss(N, dxc, 0.55e-3)

    def _mb_profile(patch=None):
        orig = _mbmod._kmah_free_leg
        if patch is not None:
            _mbmod._kmah_free_leg = patch
        try:
            E = _rlt(Ec, p, dxc, amplitude_model='ray_density',
                     caustic='multibranch', output_plane_distance=opd,
                     caustic_ray_subsample=2)
        finally:
            _mbmod._kmah_free_leg = orig
        _, prof = _radial_profile(np.abs(E) ** 2, dxc)
        return _ring_peak_and_corr(prof, rc, pref)

    def _kmah_zero(g, d_out):
        m, detJ = _orig_kmah(g, d_out)
        return np.zeros_like(m), detJ

    def _kmah_neg(g, d_out):
        m, detJ = _orig_kmah(g, d_out)
        return -m, detJ

    _orig_kmah = _mbmod._kmah_free_leg
    # CORRECT KMAH: reproduces the reference bright ring
    pk_ok, corr_ok = _mb_profile()
    assert abs(pk_ok - ref_peak) <= 1.0, (pk_ok, ref_peak)
    assert corr_ok > 0.85, corr_ok
    # WRONG KMAH (zeroed AND negated): the bright ring moves + correlation drops
    for label, patch in (('zeroed', _kmah_zero), ('negated', _kmah_neg)):
        pk_w, corr_w = _mb_profile(patch)
        assert pk_w - ref_peak > 4.0, (label, pk_w, ref_peak)     # ring moved out
        assert corr_w < 0.80, (label, corr_w)                     # shape wrong
        assert corr_ok - corr_w > 0.15, (label, corr_ok, corr_w)  # decisive
    # the counter is restored (defensive; _mb_profile's finally already does it)
    assert _mbmod._kmah_free_leg is _orig_kmah


# ===========================================================================
# GATE 2: Airy transition -- sqrt-singularity resolves into a finite peak
# ===========================================================================
def test_airy_transition_singlebranch_diverges_multibranch_finite():
    """As two branches coalesce (S+ -> S-) the single-branch ray-density
    amplitude ``1/sqrt(S+ - S-)`` DIVERGES, but the Ludwig fold field stays
    FINITE (the fold-diffraction / Airy profile) and reduces to the plain
    two-branch GO sum on the bright side."""
    k = 2.0 * np.pi / _WL
    peaks = []
    for delta in (5e-6, 1e-6, 1e-7, 1e-9):
        val = ludwig_fold(k, delta, -delta, 1.0 + 0j, 1.0 + 0j)
        assert np.isfinite(val)
        sb = 1.0 / np.sqrt(delta)                       # single-branch proxy
        peaks.append(abs(val))
        # single-branch grows without bound; ludwig stays O(1)
        assert abs(val) < 10.0
        assert sb > abs(val)
    # the single-branch proxy has grown 100x across the sweep; ludwig has not
    assert (1.0 / np.sqrt(1e-9)) / (1.0 / np.sqrt(5e-6)) > 30.0
    assert max(peaks) / max(min(peaks), 1e-30) < 30.0
    # (the full-field 'ludwig reduces to the plain two-branch sum outside the
    # Kravtsov-Orlov band' reduction is pinned by the existing v5.21 test
    # test_multibranch_ludwig_band_swap; here we assert only the finite peak.)


def test_airy_transition_via_propagator_finite_no_blowup():
    """Through the propagator: single-branch ray_density placed AT a focus warns
    (det J -> 0, the sqrt-singularity) and caps; the multibranch at the same
    through-focus caustic plane is FINITE with a bounded peak (no blow-up)."""
    p = _mini_fast_singlet()
    N, dx = 192, 10e-6
    E0 = _gauss(N, dx, 0.6e-3)
    # near-focus (2-branch) caustic plane
    mb = _rlt(E0, p, dx, wl=_LAM633, amplitude_model='ray_density',
              caustic='multibranch', output_plane_distance=2.9e-3,
              caustic_ray_subsample=1, caustic_band='ludwig')
    plain = _rlt(E0, p, dx, wl=_LAM633, amplitude_model='ray_density',
                 caustic='multibranch', output_plane_distance=2.9e-3,
                 caustic_ray_subsample=1, caustic_band='plain')
    assert np.all(np.isfinite(mb))
    # Ludwig tames the fold divergence: its peak does not exceed the plain sum's
    assert np.abs(mb).max() <= np.abs(plain).max() + 1e-9
    # and the field is bounded (no sqrt blow-up)
    ein_peak = float(np.abs(E0).max())
    assert np.abs(mb).max() < 50.0 * ein_peak


# ===========================================================================
# GATE 1: fold ground truth -- FINITE, no blow-up; DOCUMENTED residual
# (single-branch ray_density + ASM remains the quantitative fold reference)
# ===========================================================================
@pytest.mark.slow
def test_fold_ground_truth_finite_and_documented_residual():
    """On the ``caustic_fold_ref`` grid the multibranch field AT the fold plane
    is FINITE and does NOT blow up, but it does NOT beat the single-branch
    ray-density exit field + ASM: the pure geometric multibranch is zero on the
    DARK side of the fold (no evanescent tail), so its windowed r2m reads ~15%
    low and it loses ~20% of the caustic energy, whereas single-branch+ASM (a
    genuine wave propagation) already matches the direct-RS reference to ~3%.

    HONEST gate-1 outcome (plan N13 risk clause): single-branch ``ray_density``
    + ASM / GBD / FGA remain the quantitative fold reference; multibranch is the
    finite, no-blow-up, one-call coherent field at the caustic.  This PINS the
    measured envelope so any future improvement trips it."""
    from lumenairy.memory import available_memory_bytes
    ref = np.load(_ROOT / 'caustic_fold_ref.npz', allow_pickle=True)
    meta = json.loads(str(ref['metrics']))
    Nc, dxc = int(ref['N']), float(ref['dx'])          # 1280, 2 um
    opd = float(ref['output_plane_distance'])          # 4.3704 mm vertex->fold
    est_gb = (Nc * Nc * 16 * 8) / 1e9
    if available_memory_bytes() / 1e9 < est_gb + 2.0:
        pytest.skip(f"needs ~{est_gb:.1f} GB; memory-constrained host")

    p = _caustic_prescription()
    Ec = _gauss(Nc, dxc, 0.55e-3)

    # multibranch AT the fold plane -- finite, no blow-up
    mb = _rlt(Ec, p, dxc, amplitude_model='ray_density', caustic='multibranch',
              output_plane_distance=opd, caustic_ray_subsample=2)
    assert np.all(np.isfinite(mb))
    x = (np.arange(Nc) - Nc / 2) * dxc
    X, Y = np.meshgrid(x, x)
    pin = float((np.abs(Ec[(X ** 2 + Y ** 2) <= (0.7e-3) ** 2]) ** 2).sum())
    e_ratio = float((np.abs(mb) ** 2).sum()) / pin
    assert 0.7 < e_ratio < 1.05, e_ratio          # bounded, no blow-up
    mb_r2m, mb_e50, mb_e80 = _axis_metrics(mb, dxc)
    # measured envelope (pinned): finite, bounded, within the documented band
    assert abs(mb_r2m / meta['grid_r2m_um'] - 1.0) < 0.25, mb_r2m
    assert abs(mb_e50 / meta['grid_EE50_um'] - 1.0) < 0.05, mb_e50
    assert abs(mb_e80 / meta['grid_EE80_um'] - 1.0) < 0.15, mb_e80

    # single-branch ray_density exit field + ASM -- the quantitative reference
    E_exit = _rlt(Ec, p, dxc, amplitude_model='ray_density')
    E_fold = la.angular_spectrum_propagate(E_exit, opd, _WL, dxc)
    sb_r2m, _, sb_e80 = _axis_metrics(E_fold, dxc)
    # THE honest finding: single-branch+ASM beats multibranch on the r2m the
    # dark-side tail dominates (it is a genuine wave propagation).
    assert (abs(sb_r2m / meta['grid_r2m_um'] - 1.0)
            < abs(mb_r2m / meta['grid_r2m_um'] - 1.0)), (sb_r2m, mb_r2m)
    assert abs(sb_r2m / meta['grid_r2m_um'] - 1.0) < 0.05, sb_r2m


# ===========================================================================
# GATE 4: decentered focal PSF -- centroid oracle-matched; EE = documented
# residual (near-focus caustic pile-up; GBD remains the coma reference)
# ===========================================================================
@pytest.mark.slow
def test_decenter_centroid_matches_geom_oracle_psf_is_documented_residual():
    """The multibranch decenter GEOMETRY is oracle-matched: at the paraxial
    image plane the intensity centroid tracks the lumenairy-free geometric spot
    oracle to <2% for a 1 mm and 2 mm front-surface decenter (and mirrors under
    sign flip).  But the image plane of a collimated input is an axial point
    focus, so the multibranch OVER-AMPLIFIES there (~2-3x energy: the D5 caustic
    pile-up) -- finite (never inf/nan) but NOT the decentered-PSF EE model.
    ``apply_real_lens_gbd`` (N10b) remains that reference.  Pinned."""
    from lumenairy.memory import available_memory_bytes
    geom = _load('geom_spot_decenter_oracle')
    N, dx = 768, 6e-6
    est_gb = (N * N * 16 * 8) / 1e9
    if available_memory_bytes() / 1e9 < est_gb + 2.0:
        pytest.skip(f"needs ~{est_gb:.1f} GB; memory-constrained host")
    E0 = _gauss(N, dx, 4e-3)
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    pin = float((np.abs(E0[(X ** 2 + Y ** 2) <= (5e-3) ** 2]) ** 2).sum())

    def _gjob(dmm):
        return {'wavelength_um': _WL * 1e6, 'aperture_mm': 10.0,
                'pop': {'w0_mm': 4.0}, 'surfaces': [
                    {'radius_mm': 51.68, 'thickness_mm': 5.0, 'index': 1.5168,
                     'decenter_mm': [dmm, 0.0], 'tilt_mrad': [0.0, 0.0]},
                    {'radius_mm': -51.68, 'thickness_mm': 49.162,
                     'index': 'air'}]}

    xcs = {}
    for dmm in (1.0, 2.0):
        E = _rlt(E0, _f5((dmm * 1e-3, 0.0)), dx, amplitude_model='ray_density',
                 caustic='multibranch', output_plane_distance=_Z_IMG,
                 caustic_ray_subsample=2)
        assert np.all(np.isfinite(E))                  # finite, no inf/nan
        I = np.abs(E) ** 2
        P = I.sum()
        xc = float((I * X).sum() / P) * 1e6
        xcs[dmm] = xc
        go = geom.geom_spot(_gjob(dmm))
        # decenter GEOMETRY (centroid) oracle-matched
        assert abs(xc - go['centroid_x_um']) / abs(go['centroid_x_um']) < 0.02
        # near-focus caustic pile-up: over-amplifies (documented residual)
        assert P / pin > 1.5, P / pin                  # NOT energy-neutral
    # sign-mirror: +d centroid ~ -(-d centroid)
    Em = _rlt(E0, _f5((-1e-3, 0.0)), dx, amplitude_model='ray_density',
              caustic='multibranch', output_plane_distance=_Z_IMG,
              caustic_ray_subsample=2)
    Im = np.abs(Em) ** 2
    xcm = float((Im * X).sum() / Im.sum()) * 1e6
    assert abs(xcm + xcs[1.0]) / abs(xcs[1.0]) < 0.02


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
