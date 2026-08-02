"""A design-121-CLASS CI gate for the traced carrier chain -- niche D5
(roadmap ``ROADMAP_DESIGN121_FULL_CONFIGURATION_2026_07_27`` P4, third
bullet: "No CI gate asserts design-121 dx-flatness (the .zmx can't ship)").

WHAT WAS MISSING.  The only shipped dx gate is ``self_check='dx'``: ONE
synthetic N=512 singlet, ONE sqrt(2) step, 5 % tolerance on window-power /
peak / r50.  It catches a gross regression and nothing subtler, and it does
not touch EE or FWHM -- the metrics the design-121 acceptance is quoted in.
The real 8-group acceptance (EE6 99.6 / EE3 88.8 / FWHM 3.450 um) cannot be
run in CI because it needs the Tx02-MSOP16 ``.zmx``, which is not
distributable.

WHAT THIS FILE IS.  A four-group synthetic stand-in with design 121's
STRUCTURE, built inline from radii / conics / aspheres / glass so nothing
outside the library is needed:

    small-waist (4 um) diverging launch  ->  G1 collimate  ->  G2 focus
    ->  G3 re-collimate  ->  G4 fast focus  ->  exact high-NA final leg

and -- this is the half a first attempt got wrong -- it is a **CORRECTED
RELAY**, not a chain of individually-perfect groups.  G3's flat EXIT carries
an ``r^4`` pre-shaping term and G4's flat ENTRANCE carries the compensating
term 2 mm downstream in the same collimated space, so the chain has to HAND
OFF a large ``r^4`` residual between two groups and G4 has to consume it.
That is exactly design 121's situation (audit
``AUDIT_TRACED_FROZEN_AMPLITUDE_2026_07_24`` S12: "the residual a
carrier-regime chain carries is the design's own correction, which is
r^4-dominant", A = 9.2 rad on the 121 final leg), and it is what gives the
gate teeth on ``preserve_input_phase`` -- see below.

WHY THE STRUCTURE MATTERS.  The 121's per-group exit NA runs
0.039 / ~0 / ~0 / ~0 / 0.023 / 0.080 / 0.148 with a final-leg ``na_exit`` of
0.405; the stand-in runs 0.000 / 0.078 / 0.000 / 0.189 with ``na_exit`` =
0.2021, so it climbs to the same class, crosses ``na_exact_threshold`` =
0.15, and takes the SAME exact final leg.  The small-waist launch is what
puts the carrier beyond the co-moving grid's Nyquist pitch on the coarse
legs -- the regime the v5.29 default flip exists for.

THE ORACLE IS INDEPENDENT AND ABSOLUTE.  The level half of the gate is
anchored to ``validation/oracles/debye_oracle_v3.py`` -- the lumenairy-FREE
oracle already in the tree (exact meridional raytrace through the same conic
+ aspheric surface list, energy-conserving exit ring measure, ring-Huygens
diffraction integral; validated against the ZOS Huygens PSF to EE50 1.9 % /
EE80 0.72 % in ``test_niche_p8_capstone.py``).  It shares NO code with the
propagator under test: not the readout, not ``window_factor``, not the FFT
grid, not even the wave model.  ``huygens_radial_profile`` (added for this
gate) normalises it ABSOLUTELY -- its own total comes back 1.019 of the
launched power, a measurement, not a fit -- so FWHM and encircled energy are
compared as physical numbers rather than as ratios that cancel any error the
two share.  A previous revision anchored to a "perfect sphere pushed through
the SAME ``carrier_referenced_exact_focus_readout``"; that reference moved in
lockstep with the chain (``window_factor`` 4.0 -> 3.0 broadened BOTH by 6 %
and the gate stayed green), which is precisely why it was replaced.

THE GATE, and what each half catches (all numbers MEASURED 2026-07-29):

* **dx-flatness** across N = 512 / 768 / 1024 at a PITCH-PRESERVING
  ``ray_subsample`` = 2 / 3 / 4 (physical ray pitch 4.074 um on all three,
  launch dx 2.037 / 1.358 / 1.019 um -- two dx steps, total factor 2).
* **level, anchored ABSOLUTELY** -- FWHM within 2 % of the independent
  oracle (measured 0.437 %), EE2 >= 0.70x and EE4 >= 0.80x the oracle's
  (measured 0.790 / 0.867), window power >= 99.0 % of the LAUNCHED power
  (measured 99.88).

TWO MEASURED RESULTS THIS FILE PINS BECAUSE THEY COST A REVISION EACH.

(i) **dx-flatness ALONE is not a sufficient gate.**
``carrier_reference='parabola'`` is dx-FLAT to 0.005 % on this stand-in while
sitting 3.67x wide of the oracle.  A flatness-only gate -- which is all
``self_check='dx'`` is -- passes it silently.

(ii) **A stand-in whose groups are each individually perfect has no teeth on
``preserve_input_phase``.**  With every group exactly stigmatic the residual
each group hands the next is ~0, ``'remap'`` coincides with ``False`` by
construction (see the ``preserve_input_phase`` docstring), and BOTH flips
move the answer by 0.03 EE points -- i.e. the second leg of the v5.29
validated triple could be reverted and the gate would pass with an empty
violation list.  On the corrected relay the same flips cost 23 % / 52 % of
the FWHM.  ``test_gate_has_teeth`` covers the whole triple except
``remap_sampling``; the measured reason that one has no teeth here is
recorded on ``test_remap_sampling_has_no_teeth_here``.

A KNOWN, ROOT-CAUSED GAP, pinned by
``test_the_level_gap_is_the_traced_fit_radius_cliff``: under the shipped
defaults the chain reproduces the oracle's FWHM to 0.437 % but its EE2 to
only 0.790x.  That is NOT oracle error -- it is the documented aperture:beam
/ ray-fit cliff (``fit_radius_beam_factor``, audit
``AUDIT_TRACED_PRODUCTION_READINESS_2026_07_24`` S4) biting on a FAST final
group whose input is COLLIMATED, so no carrier engages and the OPL fit has
to represent the whole sphere.  Measured on a single stigmatic conic singlet
at exit NA 0.20: ``apply_real_lens_traced`` leaves 4.428 rad of exit
wavefront error at r=w unrestricted and 1.122 rad at the chain's validated
``fit_radius_beam_factor`` = 2.0 default, against 0.087 rad at 1.5 and
0.031 rad from ``apply_real_lens_gbd`` -- an independent propagator, which
reproduces the exact ray-traced (Fermat spherical) exit wavefront.  The gap is dx-INDEPENDENT (1.240 /
1.238 / 1.233 rad at N = 1024 / 2048 / 4096), which is the sharpest possible
statement of why a flatness-only gate cannot see it.

PLATFORM ENVELOPE (why the tolerances are what they are -- MEASURED, not
estimated).  The whole ladder was re-run on the WSL Ubuntu-24.04 CI proxy
(CPython 3.12.3, numpy 2.4.6, scipy-openblas 0.3.31 SkylakeX DYNAMIC_ARCH)
against Windows (CPython 3.14.6, numpy 2.4.4).  Maximum disagreement over all
three rows x five metrics: FWHM <= 4e-6 um, EE <= 5e-5 points, window
<= 3e-6 points -- ~200x below the dx spread the gate tolerates and ~1e4 x
below the tolerances themselves.  The ORACLE is identical to every printed
digit on both platforms (it is pure numpy + ``scipy.special.j0``), so the
absolute anchor cannot drift between runners.

COST: measured 137 s wall / 3.16 GiB peak RSS on Windows for the whole file
(13 tests).  Every test that runs the chain is RAM-guarded and skips below
4 GiB available, which the 7 GiB ubuntu-latest runner class clears.
"""
from __future__ import annotations

import functools
import importlib.util
import pathlib
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy import get_glass_index

_ORACLE_PATH = (pathlib.Path(__file__).resolve().parents[2]
                / 'validation' / 'oracles' / 'debye_oracle_v3.py')
_spec = importlib.util.spec_from_file_location('debye_oracle_v3', _ORACLE_PATH)
_oracle_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_oracle_mod)

# ---------------------------------------------------------------------------
# The synthetic stand-in.
# ---------------------------------------------------------------------------
_WL = 1.31e-6
_K0 = 2.0 * np.pi / _WL
_GLASS = 'N-BK7'
_W0 = 4e-6                  # launch waist (design 121 uses 4 um)
_Z1 = 2e-3                  # hand-off plane (design 121 uses 2 mm)
_ZC = 47.9e-3               # G1 front focal distance
_F2 = 62e-3                 # G2 focal length  -> intermediate NA
_F3 = 12e-3                 # G3 (re-collimator) focal length
_GAP34 = 2e-3               # G3 -> G4, the leg that carries the residual
_NA4 = 0.20                 # target exit NA of the fast final group
_FW = 5.0                   # launch window = _FW * w(launch)
_DXO = 0.05e-6              # readout pitch
_NOUT = 512                 # readout points  (+/- 12.8 um)
_NFC = 2048                 # final-leg n_fine_cap
_WFAC = 4.0                 # window_factor (no-op regime, >= 4)
_RADII = (1.0, 2.0, 4.0)    # EE radii in um -- the NA-scaled analogue of the
#                             121's EE3/EE6/EE12 at this 1.3x higher NA

# The carried residual: ``_RESID_RAD`` radians of r^4 at r = w on the G3->G4
# leg (design 121 carries 9.2 rad on its final leg).  2.0 rad is the measured
# knee -- large enough that dropping it costs 52 % of the FWHM, small enough
# that the fast group's ray-FIT still tracks the independent oracle's FWHM to
# 0.44 %.  ``_CORR_RATIO`` is the compensating coefficient on G4's entrance,
# solved offline by bisection on the exact marginal ray and RE-PROVED in CI by
# ``test_the_stand_in_is_a_corrected_relay``.
_RESID_RAD = 2.0
_CORR_RATIO = 0.98480560

# Ladder: (N, ray_subsample).  ray_subsample is chosen so the PHYSICAL ray
# pitch (rs * dx) is identical on all three rows -- the F-0 finding says the
# degradation tracks grid pitch, not ray density, so the ladder must vary
# only the grid.
_LADDER = ((512, 2), (768, 3), (1024, 4))

# Tolerances.  Flatness: ~10x the measured spread.  Level: ABSOLUTE, against
# the lumenairy-free oracle (see the module docstring).
_FLAT_FWHM_REL = 5e-3       # measured 1.4e-5
_FLAT_EE_ABS = 0.30         # points; measured <= 0.011
_FLAT_WIN_ABS = 0.20        # points; measured 0.0074
_LEVEL_FWHM_REL = 0.02      # measured 0.00437
_LEVEL_EE2_RATIO = 0.70     # measured 0.790
_LEVEL_EE4_RATIO = 0.80     # measured 0.867
_LEVEL_WINDOW_MIN = 99.0    # per cent of the LAUNCHED power; measured 99.883


def _n_glass() -> float:
    return float(get_glass_index(_GLASS, _WL))


def _surface(radius, glass_before, glass_after, conic, asph=None):
    return {'radius': float(radius), 'glass_before': glass_before,
            'glass_after': glass_after, 'conic': float(conic),
            'radius_y': None, 'conic_y': None,
            'aspheric_coeffs': asph, 'aspheric_coeffs_y': None}


def _geometry():
    """``(w_collimated, w_last, f4, a4, b4)`` -- the derived constants.

    Every group is a ``K = -n^2`` conic singlet at the conjugate pair its
    conic solves EXACTLY (hyperbola, one side collimated -- the Fermat
    solution: a plane wave in glass imaged to a point in air gives
    ``r^2 = 2 f (n-1) s + (n^2-1) s^2`` for the sag ``s``, i.e.
    ``R = -(n-1) f``, ``K = -n^2``), so the relay's ONLY aberration is the
    ``r^4`` pair deliberately installed on the G3->G4 leg.
    """
    zR = np.pi * _W0 * _W0 / _WL
    w_c = _W0 * np.sqrt(1.0 + (_ZC / zR) ** 2)      # collimated 1/e^2 radius
    w_last = w_c * _F3 / _F2                        # radius at G3 / G4
    f4 = w_last / _NA4
    a4 = _RESID_RAD / _K0 / (_n_glass() - 1.0) / w_last ** 4
    return float(w_c), float(w_last), float(f4), float(a4), float(a4 * _CORR_RATIO)


def _stand_in():
    """``(groups, w_collimated, w_last, f4)`` for the 4-group stand-in."""
    n = _n_glass()
    w_c, w_last, f4, a4, b4 = _geometry()
    groups = [
        {'prescription': {
            'name': 'G1 collimate', 'aperture_diameter': 22e-3,
            'thicknesses': [4e-3],
            'surfaces': [_surface((n - 1.0) * _ZC, 'air', _GLASS, -n * n),
                         _surface(np.inf, _GLASS, 'air', 0.0)]},
         'gap_before': _ZC - _Z1},
        {'prescription': {
            'name': 'G2 focus', 'aperture_diameter': 22e-3,
            'thicknesses': [5e-3],
            'surfaces': [_surface(np.inf, 'air', _GLASS, 0.0),
                         _surface(-(n - 1.0) * _F2, _GLASS, 'air', -n * n)]},
         'gap_before': 20e-3},
        {'prescription': {
            'name': 'G3 recollimate', 'aperture_diameter': 6e-3,
            'thicknesses': [3e-3],
            'surfaces': [_surface(-(n - 1.0) * _F3, 'air', _GLASS, -n * n),
                         _surface(np.inf, _GLASS, 'air', 0.0, {4: a4})]},
         'gap_before': _F2 - _F3},
        {'prescription': {
            'name': 'G4 fast focus', 'aperture_diameter': 6e-3,
            'thicknesses': [3e-3],
            'surfaces': [_surface(np.inf, 'air', _GLASS, 0.0, {4: b4}),
                         _surface(-(n - 1.0) * f4, _GLASS, 'air', -n * n)]},
         'gap_before': _GAP34},
    ]
    return groups, w_c, w_last, f4


def _launch(N):
    """``(envelope, dx, r_in, launch_power)`` -- the design-121 launch, a
    Gaussian handed off ``_Z1`` past a ``_W0`` waist.  ``r_in`` is the EXACT
    sphere (``carrier_reference='sphere'``), i.e. a Gaussian-apodised point
    source ``r_in`` behind the launch plane."""
    zR = np.pi * _W0 * _W0 / _WL
    w1 = _W0 * np.sqrt(1.0 + (_Z1 / zR) ** 2)
    R1 = _Z1 * (1.0 + (zR / _Z1) ** 2)
    dx = _FW * w1 / N
    x = (np.arange(N) - N // 2) * dx
    env = np.exp(-(x[None, :] ** 2 + x[:, None] ** 2) / (w1 * w1)
                 ).astype(np.complex128)
    return env, float(dx), float(R1), float(np.sum(np.abs(env) ** 2)) * dx * dx


# ---------------------------------------------------------------------------
# The lumenairy-FREE oracle: the SAME surfaces as a debye_oracle_v3 job.
# ---------------------------------------------------------------------------
def _oracle_surfaces(groups, final_distance, ng=None):
    """``groups`` -> the oracle's mm-unit surface list.  Surface 0 is the flat
    air-to-air LAUNCH plane, so the oracle's entrance plane is exactly the
    plane the chain's ``E_in`` lives on."""
    n = _n_glass()
    gs = groups if ng is None else groups[:ng]
    out = [{'radius_mm': 0.0, 'thickness_mm': float(gs[0]['gap_before']) * 1e3,
            'index': 'air', 'conic': 0.0}]
    for gi, g in enumerate(gs):
        p = g['prescription']
        if gi:
            out[-1]['thickness_mm'] = float(g['gap_before']) * 1e3
        for si, s in enumerate(p['surfaces']):
            asph = s.get('aspheric_coeffs') or None
            out.append({
                'radius_mm': (0.0 if not np.isfinite(s['radius'])
                              else s['radius'] * 1e3),
                'thickness_mm': (float(p['thicknesses'][0]) * 1e3
                                 if si == 0 else 0.0),
                'index': n if si == 0 else 'air',
                'conic': float(s['conic']),
                'aspheric_coeffs_si': ({str(k): float(v)
                                        for k, v in asph.items()}
                                       if asph else None)})
    out[-1]['thickness_mm'] = float(final_distance) * 1e3
    return out


def _oracle_job(ng=None, final_distance=None, n_fan=6000, n_rho=2400):
    groups, _w_c, _w_last, f4 = _stand_in()
    zR = np.pi * _W0 * _W0 / _WL
    w1 = _W0 * np.sqrt(1.0 + (_Z1 / zR) ** 2)
    R1 = _Z1 * (1.0 + (zR / _Z1) ** 2)
    fd = f4 if final_distance is None else final_distance
    return {'wavelength_um': _WL * 1e6,
            'surfaces': _oracle_surfaces(groups, fd, ng),
            'pop': {'w0_mm': w1 * 1e3}, 'R_in_mm': R1 * 1e3,
            # the fan spans exactly the chain's launch window
            'aperture_mm': _FW * w1 * 1e3,
            'entrance_eikonal': 'sphere',
            'n_fan': int(n_fan), 'n_rho': int(n_rho),
            'rho_max_um': 0.5 * _NOUT * _DXO * 1e6}


@functools.lru_cache(maxsize=None)
def _oracle(n_fan=6000, n_rho=2400):
    """Absolute FWHM (um) / EE fractions (%) / total (%) of the TRUE focal
    spot -- no lumenairy anywhere in the call."""
    rho, inten = _oracle_mod.huygens_radial_profile(_oracle_job(n_fan=n_fan,
                                                                n_rho=n_rho))
    cum = np.concatenate(([0.0], np.cumsum(
        0.5 * (inten[1:] * rho[1:] + inten[:-1] * rho[:-1])
        * np.diff(rho)))) * 2.0 * np.pi
    j = int(np.where(inten < 0.5 * inten[0])[0][0])
    fr = (inten[j - 1] - 0.5 * inten[0]) / (inten[j - 1] - inten[j])
    out = {'fwhm': float(2.0 * (rho[j - 1] + fr * (rho[j] - rho[j - 1]))) * 1e6}
    for r in _RADII:
        out[f'ee{r:g}'] = float(np.interp(r * 1e-6, rho, cum)) * 100.0
    out['total'] = float(cum[-1]) * 100.0
    return out


# ---------------------------------------------------------------------------
# Metrics and the chain runner.
# ---------------------------------------------------------------------------
def _metrics(E, P_in):
    """``dict`` of FWHM (um), EE at ``_RADII`` (%), window power (%) and the
    peak offset (um), all on the ``_DXO`` readout grid."""
    E = np.asarray(E)
    n = E.shape[-1]
    inten = np.abs(E) ** 2
    iy, ix = np.unravel_index(int(np.argmax(inten)), inten.shape)
    xx = (np.arange(n) - ix) * _DXO
    yy = (np.arange(n) - iy) * _DXO
    rr = np.sqrt(xx[None, :] ** 2 + yy[:, None] ** 2)
    nb = n // 2
    ring = np.clip((rr / _DXO).astype(np.int64), 0, nb)
    s = np.bincount(ring.ravel(), weights=inten.ravel(), minlength=nb + 1)
    cnt = np.bincount(ring.ravel(), minlength=nb + 1)
    prof = (s[:nb] / np.maximum(cnt[:nb], 1)) / inten[iy, ix]
    rb = (np.arange(nb) + 0.5) * _DXO
    idx = np.where(prof < 0.5)[0]
    if len(idx) and idx[0] > 0:
        j = int(idx[0])
        fr = (prof[j - 1] - 0.5) / (prof[j - 1] - prof[j])
        fwhm = 2.0 * (rb[j - 1] + fr * (rb[j] - rb[j - 1]))
    else:
        fwhm = np.nan
    out = {'fwhm': float(fwhm) * 1e6}
    for r in _RADII:
        out[f'ee{r:g}'] = float(
            inten[rr <= r * 1e-6].sum()) * _DXO * _DXO / P_in * 100.0
    out['window'] = float(inten.sum()) * _DXO * _DXO / P_in * 100.0
    out['offset'] = (float((ix - n // 2) * _DXO) * 1e6,
                     float((iy - n // 2) * _DXO) * 1e6)
    return out


_CACHE: dict = {}


def _run(N, rs, tag='defaults', chain_kwargs=None, traced_kwargs=None):
    """Run the stand-in through ``propagate_traced_carrier_chain`` and return
    ``(metrics, stages)``.  Memoised: the ladder is shared across tests."""
    key = (N, rs, tag)
    if key in _CACHE:
        return _CACHE[key]
    env, dx, R1, P_in = _launch(N)
    groups, _w_c, _w_last, f4 = _stand_in()
    kw = dict(r_in=R1, ray_subsample=rs, n_workers=1, final_distance=f4,
              focus_readout={'dx_out': _DXO, 'N_out': _NOUT,
                             'n_fine_cap': _NFC, 'window_factor': _WFAC},
              final_leg='auto')
    if chain_kwargs:
        kw.update(chain_kwargs)
    if traced_kwargs:
        kw['traced_kwargs'] = traced_kwargs
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = la.propagate_traced_carrier_chain(env, groups, _WL, dx, **kw)
    val = (_metrics(res.field, P_in), list(res.stages))
    _CACHE[key] = val
    return val


def _ladder(tag='defaults', chain_kwargs=None, traced_kwargs=None,
            ladder=_LADDER):
    return [_run(N, rs, tag, chain_kwargs, traced_kwargs)[0]
            for (N, rs) in ladder]


# ---------------------------------------------------------------------------
# THE GATE.  Returns the list of violations; empty list == PASS.  Both the
# pass test and the teeth tests call this SAME function, so "the gate has
# teeth" is a statement about the shipped gate and not about a parallel
# re-implementation of it.
# ---------------------------------------------------------------------------
def dx_flatness_gate(rows, oracle):
    """Violations of the D5 gate for the metric ``rows`` of a dx ladder
    against the ABSOLUTE, lumenairy-free ``oracle``.

    Half 1 is dx-flatness across the ladder.  Half 2 is a LEVEL check in
    physical units: the oracle's EE fractions are renormalised by its own
    measured total (1.019 of the launched power -- the ring-Huygens kernel
    carries no obliquity factor), so the comparison is
    fraction-of-focused-power on both sides, while the chain's window power is
    checked against the LAUNCHED power with no oracle involved at all.
    """
    bad = []
    keys = [f'ee{r:g}' for r in _RADII]
    fw = [r['fwhm'] for r in rows]
    if not all(np.isfinite(fw)):
        return ['FWHM is not finite on every row of the ladder']
    if len(rows) > 1:
        spread = (max(fw) - min(fw)) / max(np.mean(fw), 1e-300)
        if spread > _FLAT_FWHM_REL:
            bad.append(f'FWHM not dx-flat: {spread * 100:.4f} % across the '
                       f'ladder (tolerance {_FLAT_FWHM_REL * 100:.3f} %)')
        for k in keys + ['window']:
            vals = [r[k] for r in rows]
            tol = _FLAT_WIN_ABS if k == 'window' else _FLAT_EE_ABS
            if max(vals) - min(vals) > tol:
                bad.append(f'{k} not dx-flat: {max(vals) - min(vals):.4f} '
                           f'points across the ladder (tolerance {tol:.3f})')
    ref = rows[-1]                                  # the finest grid
    scale = max(oracle['total'], 1e-300) / 100.0    # oracle -> focused-power
    rel = ref['fwhm'] / oracle['fwhm'] - 1.0
    if abs(rel) > _LEVEL_FWHM_REL:
        bad.append(f'FWHM {ref["fwhm"]:.4f} um is {rel * 100:+.2f} % from the '
                   f'independent oracle {oracle["fwhm"]:.4f} um (tolerance '
                   f'+-{_LEVEL_FWHM_REL * 100:.1f} %)')
    for k, lim in (('ee2', _LEVEL_EE2_RATIO), ('ee4', _LEVEL_EE4_RATIO)):
        ideal = oracle[k] / scale
        rat = ref[k] / max(ideal, 1e-300)
        if rat < lim:
            bad.append(f'{k} {ref[k]:.3f} % is only {rat:.3f} of the '
                       f'independent oracle {ideal:.3f} % (tolerance {lim:.2f})')
    if ref['window'] < _LEVEL_WINDOW_MIN:
        bad.append(f'window power {ref["window"]:.3f} % of the launched power '
                   f'is below {_LEVEL_WINDOW_MIN:.1f} %')
    return bad


# ---------------------------------------------------------------------------
# RAM guard.  The exact final leg retraces the last group on a 2048^2 fine
# grid and the traced element holds ~a dozen wave-grid intermediates, so the
# MEASURED whole-file peak RSS is 3.16 GiB on Windows -- not the 64 MiB a
# single-array estimate suggests.  ubuntu-latest CI runners are the 7 GiB
# (2-core) / 16 GiB (4-core) classes, so this is a real skip, not a
# formality: without it the 7 GiB class is one concurrent job away from OOM.
# ---------------------------------------------------------------------------
def _need_ram(gib=4.0):
    psutil = pytest.importorskip('psutil')
    avail = psutil.virtual_memory().available
    if avail < gib * 1024 ** 3:
        pytest.skip(f'needs ~{gib:.1f} GB free for the chain ladder '
                    f'(have {avail / 1024 ** 3:.2f} GB)')


# ===========================================================================
# 1. The stand-in's premises: a CORRECTED relay, and 121-shaped.
# ===========================================================================
def _sag(r, R, K, asph):
    z = 0.0
    if np.isfinite(R):
        c = 1.0 / R
        disc = max(1.0 - (1.0 + K) * c * c * r * r, 0.0)
        z = c * r * r / (1.0 + np.sqrt(disc))
    for p, a in (asph or {}).items():
        z = z + float(a) * r ** int(p)
    return z


def _dsag(r, R, K, asph):
    dz = 0.0
    if np.isfinite(R):
        c = 1.0 / R
        w = np.sqrt(max(1.0 - (1.0 + K) * c * c * r * r, 1e-30))
        dwdr = -(1.0 + K) * c * c * r / w
        dz = (2.0 * c * r * (1.0 + w) - c * r * r * dwdr) / (1.0 + w) ** 2
    for p, a in (asph or {}).items():
        dz = dz + float(a) * int(p) * r ** (int(p) - 1)
    return dz


def _surface_stack(drop_corrector=False):
    """``[(z_vertex, R, K, asph, n_after), ...]`` in world z from the LAUNCH
    plane (z=0), i.e. the plane ``E_in`` lives on."""
    groups, _w_c, _w_last, _f4 = _stand_in()
    n = _n_glass()
    out = []
    z = 0.0
    for gi, g in enumerate(groups):
        p = g['prescription']
        z += float(g['gap_before'])
        for si, s in enumerate(p['surfaces']):
            asph = s.get('aspheric_coeffs')
            if drop_corrector and gi == 3 and si == 0:
                asph = None
            out.append((z, float(s['radius']), float(s['conic']), asph,
                        n if si == 0 else 1.0))
            if si == 0:
                z += float(p['thicknesses'][0])
    return out


def _meridional_error(h0, z_img, surfs=None):
    """Exact meridional trace of the ray leaving the launch PLANE at height
    ``h0`` along the launch congruence (a point source ``R1`` behind, the
    exact ``carrier_reference='sphere'`` launch); returns its transverse miss
    at ``z_img`` (m).  Snell in 2-D on the exact conic + aspheric sag, no
    paraxial step anywhere."""
    surfs = _surface_stack() if surfs is None else surfs
    _zR = np.pi * _W0 * _W0 / _WL
    R1 = _Z1 * (1.0 + (_zR / _Z1) ** 2)
    u = h0 / R1
    y, z, n = h0, 0.0, 1.0
    for (zv, R, K, asph, n2) in surfs:
        y0 = y + u * (zv - z)
        y = y0
        for _ in range(200):                 # y = y0 + u * sag(|y|)
            y_new = y0 + u * _sag(abs(y), R, K, asph)
            if abs(y_new - y) < 1e-16:
                y = y_new
                break
            y = y_new
        s = _sag(abs(y), R, K, asph)
        m = _dsag(abs(y), R, K, asph) * (1.0 if y >= 0.0 else -1.0)
        nx, nz = -m / np.sqrt(1.0 + m * m), 1.0 / np.sqrt(1.0 + m * m)
        d = np.sqrt(1.0 + u * u)
        dy, dz = u / d, 1.0 / d
        ci = dy * nx + dz * nz
        eta = n / n2
        k = 1.0 - eta * eta * (1.0 - ci * ci)
        if k < 0.0:
            return np.nan
        fy = eta * dy - (eta * ci - np.sqrt(k)) * nx
        fz = eta * dz - (eta * ci - np.sqrt(k)) * nz
        u, n, z = fy / fz, n2, zv + s
    return y + u * (z_img - z)


def test_the_stand_in_is_a_corrected_relay():
    """The two structural premises, both proved by the inline exact raytrace.

    (a) The relay is CORRECTED: the transverse ray error at the image plane
    is <= 0.11 um out to 1.25 launch radii and 0.51 um at 1.5 -- an order
    below the 2.74 um oracle FWHM -- so the focus is diffraction-limited and
    the independent oracle is a meaningful level reference.

    (b) The correction is CARRIED BETWEEN GROUPS, which is what gives the
    gate teeth on ``preserve_input_phase``: removing G4's compensating
    ``r^4`` term (and NOTHING else) moves the same rays to +10.15 um at
    1.0w and +43.1 um at 1.5w -- 100x the corrected error and 16x the ideal
    FWHM.  That difference IS the residual the chain has to hand from G3 to
    G4; measured at G4's entrance it is 2.0 rad of r^4 at r=w (design 121
    carries 9.2 rad on its final leg)."""
    _groups, _w_c, _w_last, f4 = _stand_in()
    tab = _surface_stack()
    z_img = tab[-1][0] + f4
    zR = np.pi * _W0 * _W0 / _WL
    w1 = _W0 * np.sqrt(1.0 + (_Z1 / zR) ** 2)
    fracs = (0.05, 0.25, 0.5, 0.75, 1.0, 1.25)
    errs = [abs(_meridional_error(f * w1, z_img, tab)) for f in fracs]
    assert all(np.isfinite(errs)), errs
    assert max(errs) < 0.5e-6, [f'{e * 1e6:.4f} um' for e in errs]
    assert abs(_meridional_error(1.5 * w1, z_img, tab)) < 1.5e-6
    bare = _surface_stack(drop_corrector=True)
    lost = [abs(_meridional_error(f * w1, z_img, bare)) for f in (1.0, 1.5)]
    assert min(lost) > 20 * max(errs), (lost, errs)
    assert lost[0] > 5e-6, lost           # measured 10.15 um


def test_na_progression_climbs_like_design_121():
    """The structural premise: several groups, exit NA climbing to the
    ~0.15-0.2 class, and the EXACT high-NA final leg engaged -- the same
    route design 121 takes.

    Measured 2026-07-29 at N=1024: per-group exit NA (|w| / |R_out|) =
    0.0000 / 0.0783 / 0.0000 / 0.1890, final-leg ``na_exit`` = 0.2021, i.e.
    35 % above the ``na_exact_threshold`` = 0.15 that ``final_leg='auto'``
    branches on.  Design 121 reads 0.039 / ~0 / ~0 / ~0 / 0.023 / 0.080 /
    0.148 with ``na_exit`` = 0.405."""
    _need_ram()
    _m, stages = _run(1024, 4)
    na_out = [abs(st['w'] / st['R_out']) for st in stages]
    assert len(na_out) == 4, na_out
    assert max(na_out) > 0.15, na_out
    assert 0.02 < sorted(na_out)[-2] < 0.15, na_out   # a genuine mid rung
    na_exit = [st.get('na_exit') for st in stages if st.get('na_exit')]
    assert na_exit, 'the exact final leg did not run (na_exit not reported)'
    assert na_exit[-1] > 0.15, na_exit
    assert stages[-1].get('exact_final'), stages[-1]


# ===========================================================================
# 2. The oracle is independent, absolute and converged.
# ===========================================================================
def test_the_oracle_is_absolute_and_converged():
    """The level half is only as good as its anchor, so the anchor states its
    own error bars.

    * ABSOLUTE: ``huygens_radial_profile`` carries the ``2 pi dh / (i lambda)``
      constant, so the profile's own total is a MEASUREMENT of how much of
      the launched power it accounts for.  Measured 101.907 % -- the ~1.9 %
      excess is the ring-Huygens kernel's missing obliquity factor at NA 0.2,
      and it is why the gate renormalises EE by this total instead of
      trusting it to 0.1 %.
    * CONVERGED: doubling BOTH the fan and the rho sampling (3000 x 1200 ->
      6000 x 2400) moves FWHM by 3.6e-6 relative (2.743298 -> 2.743288 um)
      and EE2 by 0.0012 points (82.4453 -> 82.4465 %).
    * INDEPENDENT: it imports nothing from lumenairy -- in particular not
      ``carrier_referenced_exact_focus_readout``, whose ``window_factor`` a
      shared reference would track in lockstep."""
    import sys
    a = _oracle(3000, 1200)
    b = _oracle(6000, 2400)
    assert 95.0 < b['total'] < 106.0, b
    assert abs(b['fwhm'] - a['fwhm']) / b['fwhm'] < 2e-3, (a, b)
    assert abs(b['ee2'] - a['ee2']) < 0.05, (a, b)
    src = _ORACLE_PATH.read_text(encoding='utf-8')
    assert 'import lumenairy' not in src, 'the oracle stopped being independent'
    assert 'from lumenairy' not in src, 'the oracle stopped being independent'
    assert not any(m.startswith('lumenairy')
                   for m in vars(_oracle_mod)
                   if isinstance(getattr(_oracle_mod, m, None), type(sys))), \
        'the oracle imported a lumenairy module'


# ===========================================================================
# 3. The gate passes on the shipped defaults.
# ===========================================================================
def test_dx_flatness_gate_passes_on_the_shipped_defaults():
    """The whole point of the file.  ``carrier_reference='sphere'`` +
    ``amplitude_model='ray_density'`` + ``preserve_input_phase='remap'`` +
    ``remap_sampling='full'`` (all CHAIN DEFAULTS since v5.29) put the
    stand-in on a dx plateau at the independent oracle's FWHM.

    Measured 2026-07-29 at N = 512 / 768 / 1024, rs = 2 / 3 / 4:

        FWHM   2.75530 / 2.75528 / 2.75526 um  (oracle 2.74329)
        EE1    25.6198 / 25.6207 / 25.6247 %   (oracle 32.5603 renormalised)
        EE2    63.8660 / 63.8677 / 63.8767 %   (oracle 80.9036)
        EE4    86.6129 / 86.6148 / 86.6164 %   (oracle 99.9624)
        window 99.8887 / 99.8908 / 99.8834 %   (of the LAUNCHED power)

    i.e. dx spreads of 0.0014 % on FWHM and <= 0.011 points on EE (the gate
    tolerates 0.5 % and 0.30), and a LEVEL of +0.437 % on FWHM, 0.790x on
    EE2 and 0.867x on EE4 against the independent oracle (tolerances 2 %,
    0.70, 0.80).

    The residual EE distance to the oracle is the ray-FIT cliff on the fast
    final group, root-caused and pinned by
    :func:`test_the_level_gap_is_the_traced_fit_radius_cliff`."""
    _need_ram()
    rows = _ladder()
    oracle = _oracle()
    assert dx_flatness_gate(rows, oracle) == [], (
        dx_flatness_gate(rows, oracle), rows, oracle)


def test_the_focus_lands_on_axis_at_every_grid():
    """A dx-dependent transverse WALK is the other way a chain can be
    non-flat, and EE/FWHM about the PEAK cannot see it (they re-centre).
    Measured: exactly (0.00, 0.00) um on all three rows."""
    _need_ram()
    for row in _ladder():
        assert row['offset'] == (0.0, 0.0), row['offset']


# ===========================================================================
# 4. Teeth.  The SAME gate must reject every reverted leg of the v5.29 triple.
# ===========================================================================
@pytest.mark.parametrize('tag,chain_kw,traced_kw,ladder', [
    ('parabola', {'carrier_reference': 'parabola'}, None,
     ((512, 2), (1024, 4))),
    ('pip_true', None, {'preserve_input_phase': True}, ((1024, 4),)),
    ('pip_false', None, {'preserve_input_phase': False}, ((1024, 4),)),
    ('paraxial_leg', {'final_leg': 'paraxial'}, None, ((1024, 4),)),
    ('legacy', {'carrier_reference': 'parabola'},
     {'amplitude_model': 'screen', 'preserve_input_phase': True,
      'remap_sampling': 'lattice'}, ((1024, 4),)),
])
def test_gate_has_teeth(tag, chain_kw, traced_kw, ladder):
    """Prove the gate is not vacuous: revert one leg of the v5.29 validated
    triple (or the whole legacy configuration, or the exact final leg) and
    the gate must report violations.

    Measured 2026-07-29 at N = 1024 (defaults FWHM 2.75526 um, EE2 63.877 %,
    window 99.883 %; oracle FWHM 2.74329 um, EE2 80.904 % renormalised):

        parabola      FWHM 10.0622 um  EE2  6.189 %  window 75.085 %
        pip_true      FWHM  3.3702 um  EE2 53.859 %  window 98.014 %
        pip_false     FWHM  4.1625 um  EE2 39.906 %  window 90.096 %
        paraxial_leg  FWHM  8.5848 um  EE2  8.437 %  window 82.082 %
        legacy        FWHM 10.5320 um  EE2  6.132 %  window 82.780 %

    Every one of them trips all four terms of the level half.  ``pip_true``
    and ``pip_false`` are the ones a stigmatic-groups stand-in cannot see at
    all (0.03 EE points there): here they cost 23 % / 52 % of the FWHM and
    10 / 24 EE2 points, because the corrected relay actually hands a
    residual across the G3->G4 leg for ``preserve_input_phase`` to
    transport."""
    _need_ram()
    rows = _ladder(tag, chain_kw, traced_kw, ladder)
    bad = dx_flatness_gate(rows, _oracle())
    assert bad, rows
    assert any('independent oracle' in b for b in bad), bad


def test_dx_flatness_alone_is_not_sufficient():
    """MEASURED LESSON, pinned so the oracle anchor is never dropped as
    redundant: the broken configuration above is dx-FLAT.

    ``carrier_reference='parabola'`` reads FWHM 10.06172 / 10.06224 um at
    N = 512 / 1024 -- a 0.005 % spread, INSIDE the 0.5 % flatness tolerance
    and inside every EE/window flatness tolerance too (EE2 6.18876 /
    6.18924, window 75.08426 / 75.08488) -- while sitting 3.67x wide of the
    independent oracle.  A flatness-only gate (which is all the
    shipped ``self_check='dx'`` is, on a coarser metric set) passes it
    silently.  The LEVEL half of :func:`dx_flatness_gate` is what has the
    teeth.

    2026-08-01 -- RE-PRICED AGAINST THE SAME ORACLE AFTER NICHE C6.

    PIN WAS ``FWHM / oracle > 3.0`` (measured 3.6695 / 3.6697 at
    N = 512 / 1024).  PIN IS NOW ``> 2.5``, measured 2.9541 / 2.9538, plus
    two new level assertions with far more margin than the FWHM bar ever had.

    WHY IT MOVED: ``REMAP_STATIONARY_PHASE_LAUNCH`` (niche C6) improves even
    this deliberately-broken configuration -- the broken half is the CARRIER
    REFERENCE, not the residual transport, so the residual the relay hands
    between groups is still carried better than it was.  The oracle did NOT
    move and CANNOT: it is ``validation/oracles/debye_oracle_v3.py``, pure
    numpy + ``scipy.special.j0`` with no lumenairy in the call, and it reads
    FWHM 2.743288 um / EE1 33.18 / EE2 82.45 / EE4 101.87 on both sides of
    the change.  Measured 2026-08-01 by
    ``validation/repro_traced_carrier_121/recon_d5_oracle.py``:

        configuration        N     FWHM um   /oracle    EE2    window
        parabola, C6 OFF   1024   10.06708    3.6697   6.189   75.085
        parabola, C6 ON    1024    8.10312    2.9538   8.772   77.247
        defaults,  C6 OFF  1024    2.75529    1.0044  63.877   99.883
        defaults,  C6 ON   1024    2.72274    0.9925  69.539   99.898

    So the broken configuration is LESS wrong (BETTER against the absolute
    oracle) and the shipped defaults are better too -- and the lesson this
    test exists for is untouched: the parabola row is still dx-FLAT to
    1.14e-04 (44x inside the 5e-03 bar) while sitting 2.95x wide of the
    oracle, delivering 10.6 % of its EE2 and losing 23 % of the launched
    power out of the readout window.  The two new assertions pin the level
    failure in the currency that has the teeth rather than leaning on a
    FWHM bar that a further accuracy improvement could walk through."""
    _need_ram()
    rows = _ladder('parabola', {'carrier_reference': 'parabola'}, None,
                   ((512, 2), (1024, 4)))
    fw = [r['fwhm'] for r in rows]
    spread = (max(fw) - min(fw)) / np.mean(fw)
    assert spread < _FLAT_FWHM_REL, spread          # flat...
    for r in _RADII:
        k = f'ee{r:g}'
        vals = [row[k] for row in rows]
        assert max(vals) - min(vals) < _FLAT_EE_ABS, (k, vals)
    oracle = _oracle()
    assert rows[-1]['fwhm'] / oracle['fwhm'] > 2.5, (rows, oracle)  # ...and wrong
    # ... and wrong in the currencies the gate actually scores, with margin:
    # EE2 is 0.106x the oracle's against a 0.70x gate bar, and the window
    # power is 77.2 % of launched against a 99.0 % bar.
    assert rows[-1]['ee2'] < 0.25 * oracle['ee2'], (rows[-1], oracle)
    assert rows[-1]['window'] < _LEVEL_WINDOW_MIN - 10.0, rows[-1]


def test_remap_sampling_has_no_teeth_here():
    """MEASURED NEGATIVE, recorded rather than papered over.  ``remap_sampling``
    is the fourth leg of the v5.29 default set and this stand-in CANNOT test
    it: 'lattice' vs 'full' moves the answer by 0.001 um of FWHM and 0.04 EE2
    points.

    The reason is quantitative, from audit S12: the lattice route aliases the
    transported residual only outside
    ``r_alias = (pi w^4 / (4 A h))^(1/3)`` with ``A`` the r^4 residual in rad
    at ``r = w`` and ``h`` the ray pitch.  Here A = 2.0 rad, w = 0.9665 mm and
    h = 7.6 um give r_alias = 3.6 mm = 3.7w -- far outside the beam.  Reaching
    r_alias = 1.5w at this pitch needs A ~ 30 rad, which on this geometry puts
    the fast group's ray-FIT so far into the aperture:beam cliff that the
    defaults no longer track the oracle at all (measured at A = 8 rad: EE2
    21.4 % against an oracle 74.3 %).  So the choice was: a stand-in that
    tests ``preserve_input_phase`` honestly, or one that tests
    ``remap_sampling`` while being unable to test anything else.  This test
    pins the measurement so the gap is visible rather than assumed covered."""
    _need_ram()
    a = _ladder(ladder=((1024, 4),))[0]
    b = _ladder('lattice', None, {'remap_sampling': 'lattice'},
                ((1024, 4),))[0]
    assert abs(a['fwhm'] - b['fwhm']) < 0.02, (a, b)
    assert abs(a['ee2'] - b['ee2']) < 0.30, (a, b)


# ===========================================================================
# 5. The level gap: root-caused, not tolerated.
# ===========================================================================
def test_the_level_gap_is_the_traced_fit_radius_cliff():
    """WHY the gate's EE tolerances are 0.70 / 0.80 rather than 0.98.

    Minimal repro, element level, no chain: ONE stigmatic conic singlet
    (K = -n^2, plano-convex, f = 4.83 mm) with a COLLIMATED Gaussian input at
    exit NA 0.20.  Because the system is stigmatic the exact exit wavefront
    is a SPHERE centred on the focus (Fermat), so the truth needs no
    diffraction model at all -- the inline raytrace IS the oracle.

    Measured 2026-07-29 (exit wavefront error at r = w, rad; N = 1024 over a
    3.0 mm window, dx = 2.93 um):

        apply_real_lens_traced, no fit restriction   4.4281
        ... fit_radius_beam_factor = 2.0 (the chain's validated default)
                                                     1.122   (N=2048 probe)
        ... fit_radius_beam_factor = 1.5             0.0874
        apply_real_lens_gbd (independent propagator) 0.0309

    So the gap is the documented aperture:beam ray-FIT cliff (audit
    AUDIT_TRACED_PRODUCTION_READINESS_2026_07_24 S4) biting where no carrier
    can engage -- a fast group fed by a COLLIMATED beam, so the OPL fit has
    to represent the whole exit sphere rather than a small residual.  It is
    dx-INDEPENDENT (measured 1.240 / 1.238 / 1.233 rad at N = 1024 / 2048 /
    4096 in the full chain), which is the sharpest statement of why the
    flatness half of any gate cannot see it."""
    _need_ram(2.5)
    n = _n_glass()
    w = 0.9665e-3
    f = w / 0.20
    N, dx = 1024, 3.0e-3 / 1024
    presc = {'name': 'fast', 'aperture_diameter': 6e-3, 'thicknesses': [3e-3],
             'surfaces': [_surface(np.inf, 'air', _GLASS, 0.0),
                          _surface(-(n - 1.0) * f, _GLASS, 'air', -n * n)]}
    x = (np.arange(N) - N // 2) * dx
    E_in = np.exp(-(x[None, :] ** 2 + x[:, None] ** 2) / (w * w)).astype(complex)

    def resid_at_w(E):
        row = np.asarray(E)[N // 2, :]
        S = -(np.hypot(x, f) - f)                    # converging sphere, R=-f
        ph = np.unwrap(np.angle(row * np.exp(-1j * _K0 * S)))[N // 2:]
        ph = ph - ph[0]
        return abs(float(np.interp(w, x[N // 2:], ph)))

    kw = dict(prescription=presc, wavelength=_WL, dx=dx, ray_subsample=4,
              n_workers=1, amplitude_model='ray_density',
              preserve_input_phase='remap', remap_sampling='full')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        e_def = resid_at_w(la.apply_real_lens_traced(E_in, **kw))
        e_fit = resid_at_w(la.apply_real_lens_traced(
            E_in, fit_radius_beam_factor=1.5, **kw))
        e_gbd = resid_at_w(la.apply_real_lens_gbd(
            E_in, prescription=presc, wavelength=_WL, dx=dx,
            beamlets_per_aperture=64))
    # GBD reproduces the exact (Fermat) spherical exit wavefront ...
    assert e_gbd < 0.2, (e_gbd, e_def, e_fit)
    # ... the traced element does not, unless the ray-fit disc is restricted.
    assert e_def > 10 * e_gbd, (e_def, e_gbd)
    assert e_fit < 0.4 * e_def, (e_fit, e_def)
