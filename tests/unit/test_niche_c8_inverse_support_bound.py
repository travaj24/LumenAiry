"""``REMAP_INVERSE_SUPPORT_BOUND`` -- the structural cure for the class of
defect niche C6 exposed: an exit pixel outside the region the traced rays
actually REACHED must get zero amplitude, not an extrapolated inverse-map
value.

WHAT THE DEFECT IS.  ``apply_real_lens_traced`` fits the traced entrance->exit
map and Newton-inverts the FIT, per exit pixel.  Both backends extrapolate
outside their data -- the polynomial globally, the spline past its last knot --
and on a map that has lost its radial symmetry that extrapolated inverse can
send a far exit pixel BACK INSIDE THE BRIGHT BEAM, where
``_ray_density_amp_grid`` samples ``|E_in|`` and hands it real amplitude.  The
light is not misplaced, it is manufactured: no ray of the call goes there.  On
design 121's on-axis last group (not reproducible from unit-test assets -- see
``docs/audits/C8_INVERSE_SUPPORT_BOUND_2026_08_01.md``) that was +0.486 % of
the input power, deposited at 4-8 mm at 83 % of peak.

WHAT IS PINNED HERE, on self-contained synthetic fixtures:

  1. the bound removes a manufactured lobe that the library's own ENERGY
     self-check cannot see, and does so on a fixture where the library
     manufactures the lobe by itself (the ``REMAP_STATIONARY_PHASE_FIT_GUARD``
     regression cell) -- with the fail-before arm asserted, not assumed;
  2. it is a SUBTRACTION and nothing else -- it never raises any pixel's
     amplitude, never touches a pixel with traced data behind it, and is
     byte-inert on the ``'screen'`` amplitude, which is not derived from the
     inverse map at all;
  3. it reaches BOTH ``newton_fit`` backends identically -- the support comes
     from the traced samples BEFORE any fit, so the two backends are bounded by
     the same hull and must not be driven apart (D7's contract);
  4. the support follows the BEAM, not the grid (a decentred beam is the case a
     grid-referenced radius gets wrong);
  5. the feather is monotone in its width and the flag's OFF state is inert.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements import _lens_traced as LT

_WL = 1.31e-6
_K0 = 2.0 * np.pi / _WL


def _flat():
    return {'radius': np.inf, 'glass_before': 'air', 'glass_after': 'air',
            'conic': 0.0, 'radius_y': None, 'conic_y': None,
            'aspheric_coeffs': None, 'aspheric_coeffs_y': None}


def _surf(r, gb, ga):
    d = _flat()
    d['radius'], d['glass_before'], d['glass_after'] = r, gb, ga
    return d


def _singlet(r1, r2, th, z, ap, glass='N-BK7'):
    """Biconvex singlet followed by a free leg ``z`` to the exit plane."""
    return {'name': 'c8_singlet', 'aperture_diameter': ap,
            'surfaces': [_surf(r1, 'air', glass), _surf(r2, glass, 'air'),
                         _flat()],
            'thicknesses': [th, z]}


def _field(n, dx, w, rc, alpha, cx=0.0, cy=0.0):
    """Gaussian on a converging carrier sphere plus an ``alpha (r/w)^4``
    residual -- the same construction the C6 and C7 fixtures use."""
    ax = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(ax, ax)
    sg = 1.0 if rc > 0 else -1.0
    rho = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2 + rc * rc)
    Wc = sg * (rho - abs(rc))
    r2 = (X - cx) ** 2 + (Y - cy) ** 2
    a = (alpha / _K0) * (r2 / (w * w)) ** 2
    return (np.exp(-r2 / (w * w))
            * np.exp(1j * _K0 * (Wc + a))).astype(np.complex128)


# The CLEAN fixture -- small, well conditioned, no manufactured light anywhere.
_CLEAN = dict(n=384, dx=30e-6, w=0.9e-3, rc=-0.20, alpha=3.0,
              r1=200e-3, r2=-200e-3, th=4e-3, z=5e-3, ap=11e-3)

# The DEFECTIVE fixture -- ``validation/repro_traced_carrier_121/
# probe_ghost_synthetic.py``'s 'medium, finer grid' cell, the same one niche C7
# calibrated its halo check against.  With BOTH the C6 launch and the C6 fit
# guard on, the library manufactures a lobe at ~4.6e-02 of peak beyond 3 w
# while its energy self-check stays silent.  That is this file's fail-before.
_GHOST = dict(n=768, dx=25e-6, w=1.5e-3, rc=-0.15, alpha=5.0,
              r1=150e-3, r2=-150e-3, th=4e-3, z=6e-3, ap=18e-3)

_BASE_KW = dict(wavelength=_WL, amplitude_model='ray_density',
                preserve_input_phase='remap', remap_sampling='full',
                parallel_amp=False, on_undersample='silent',
                on_noncollimated='silent', on_aperture_beam='silent',
                ray_subsample=4, fit_radius_beam_factor=2.0)


def _call(spec, bound=True, feather=None, launch=True, guard=False,
          cx=0.0, cy=0.0, **over):
    """One element call with the C8 bound, its feather and both C6 flags
    controlled and restored.  Returns ``(field, [warning messages])``."""
    E = _field(spec['n'], spec['dx'], spec['w'], spec['rc'], spec['alpha'],
               cx=cx, cy=cy)
    presc = _singlet(spec['r1'], spec['r2'], spec['th'], spec['z'],
                     spec['ap'])
    old = (LT.REMAP_STATIONARY_PHASE_LAUNCH,
           LT.REMAP_STATIONARY_PHASE_FIT_GUARD,
           LT.REMAP_INVERSE_SUPPORT_BOUND,
           LT._SUPPORT_BOUND_FEATHER_CELLS)
    LT.REMAP_STATIONARY_PHASE_LAUNCH = bool(launch)
    LT.REMAP_STATIONARY_PHASE_FIT_GUARD = bool(guard)
    LT.REMAP_INVERSE_SUPPORT_BOUND = bool(bound)
    if feather is not None:
        LT._SUPPORT_BOUND_FEATHER_CELLS = float(feather)
    try:
        kw = dict(_BASE_KW)
        kw['dx'] = spec['dx']
        kw.update(over)
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter('always')
            F = np.asarray(la.apply_real_lens_traced(
                E, prescription=presc, carrier=spec['rc'], **kw))
    finally:
        (LT.REMAP_STATIONARY_PHASE_LAUNCH,
         LT.REMAP_STATIONARY_PHASE_FIT_GUARD,
         LT.REMAP_INVERSE_SUPPORT_BOUND,
         LT._SUPPORT_BOUND_FEATHER_CELLS) = old
    return F, [str(w.message) for w in wl]


def _radii(spec, cx=0.0, cy=0.0):
    ax = (np.arange(spec['n']) - spec['n'] // 2) * spec['dx']
    X, Y = np.meshgrid(ax, ax)
    return np.hypot(X - cx, Y - cy)


def _halo(F, R, r):
    """max |E| beyond radius ``r``, over the peak."""
    a = np.abs(F)
    pk = float(a.max())
    m = R > r
    return (float(a[m].max()) / pk) if (m.any() and pk > 0.0) else 0.0


@pytest.fixture(scope='module')
def _warm():
    """Both members of every byte-identity pair must sit on the same side of
    the traced pipeline's first-call ulp boundary (the W9 determinism
    calibration)."""
    for _ in range(2):
        _call(_CLEAN, bound=False, launch=False)
    return True


# ---------------------------------------------------------------------------
# 1.  The defaults, and the fail-before switch.
# ---------------------------------------------------------------------------
def test_the_bound_ships_on_with_a_one_cell_feather():
    """Unlike ``REMAP_STATIONARY_PHASE_FIT_GUARD``, this one is a fix and not a
    lever: it regresses none of the six fixtures that kept the guard opt-in.
    The feather is measured in EXIT-LATTICE cells and must be positive -- a
    hard cut at the sample hull removes real light (measured: 7.1x more, and
    it is the only setting that moves a pixel by more than 1e-3 of peak)."""
    assert LT.REMAP_INVERSE_SUPPORT_BOUND is True
    assert LT._SUPPORT_BOUND_FEATHER_CELLS == 1.0
    assert 0.0 < LT._SUPPORT_BOUND_FEATHER_CELLS <= 2.0


def test_with_the_flag_off_the_feather_constant_is_inert(_warm):
    """The OFF state has to be a real switch and not merely a small feather:
    the whole support computation is skipped, so the constant cannot reach the
    field.  Byte-identical against an absurd feather width."""
    a, _ = _call(_CLEAN, bound=False, feather=1.0)
    b, _ = _call(_CLEAN, bound=False, feather=1e6)
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# 2.  It removes manufactured light -- with the fail-before arm asserted.
# ---------------------------------------------------------------------------
def test_it_removes_a_lobe_the_energy_self_check_cannot_see(_warm):
    """FAIL-BEFORE.  On this fixture the library manufactures the lobe by
    itself (C6 launch + the C6 fit guard), and its own ``P_out/P_ap`` band is
    quiet throughout -- which is the entire reason a support bound is needed
    rather than a tighter energy tolerance."""
    R = _radii(_GHOST)
    r3w = 3.0 * _GHOST['w']
    before, wa = _call(_GHOST, bound=False, guard=True)
    after, wb = _call(_GHOST, bound=True, guard=True)

    h_before = _halo(before, R, r3w)
    h_after = _halo(after, R, r3w)
    assert h_before > 1.0e-02, (
        f"the fail-before arm stopped manufacturing light ({h_before:.3e}); "
        f"this test no longer proves anything")
    # Amplitude beyond 3 w falls 4.593e-02 -> 8.911e-04, i.e. 51.5x.  The bar
    # is 20x rather than 100x for a reason that is in the design and not in the
    # result: the taper's PLATEAU deliberately keeps a band of
    # ``sqrt(2) sub dx`` outside the hull at full weight (it is what makes the
    # bleed into the support exactly zero -- see ``_support_taper``), and 3 w
    # on this fixture lies inside that band.  What is left there is ray-density
    # skirt with traced data behind it, not the lobe.
    assert h_after < h_before / 20.0, (h_before, h_after)

    # The POWER statement is the strong one and it keeps the 100x bar: the lobe
    # itself is gone (measured 2.576e-05 -> 4.945e-08 of the input power).
    p = np.abs(before) ** 2
    q = np.abs(after) ** 2
    m = R > r3w
    assert float(q[m].sum()) < float(p[m].sum()) / 100.0

    # and the energy self-check is silent in BOTH directions, so this is not a
    # restatement of the power guard.
    assert not [t for t in wa + wb if 'energy self-check FAILED' in t]


def test_the_halo_self_check_goes_silent(_warm):
    """The v5.32 halo self-check is the independent instrument for exactly this
    defect.  It must fire on the unbounded field and be silent on the bounded
    one -- and the bound must not make it fire where it did not."""
    _b, wa = _call(_GHOST, bound=False, guard=True)
    _a, wb = _call(_GHOST, bound=True, guard=True)
    assert [t for t in wa if 'HALO self-check FAILED' in t]
    assert not [t for t in wb if 'HALO self-check FAILED' in t]
    # clean fixture: silent before AND after.
    _c, wc = _call(_CLEAN, bound=False)
    _d, wd = _call(_CLEAN, bound=True)
    assert not [t for t in wc if 'HALO self-check FAILED' in t]
    assert not [t for t in wd if 'HALO self-check FAILED' in t]


# ---------------------------------------------------------------------------
# 3.  It is a subtraction, and only outside the traced support.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('spec,guard', [(_CLEAN, False), (_GHOST, True)])
def test_it_never_raises_any_pixel(spec, guard, _warm):
    """A bound that could ADD amplitude anywhere would be a new model, not a
    restriction.  Pointwise, on a clean fixture and a ghosting one."""
    a, _ = _call(spec, bound=False, guard=guard)
    b, _ = _call(spec, bound=True, guard=guard)
    assert np.all(np.abs(b) <= np.abs(a) * (1.0 + 1e-12) + 1e-300)


def test_it_leaves_the_beam_exactly_alone(_warm):
    """Every pixel with traced data behind it keeps its full amplitude: the
    feather band lies entirely OUTSIDE the sample hull.  Pinned BYTE-wise over
    the region that carries the light."""
    a, _ = _call(_CLEAN, bound=False)
    b, _ = _call(_CLEAN, bound=True)
    R = _radii(_CLEAN)
    core = R <= 3.0 * _CLEAN['w']
    assert core.any()
    assert np.array_equal(a[core], b[core])
    # and the total power is unmoved at the level a clean call has to be
    pa = float((np.abs(a) ** 2).sum())
    pb = float((np.abs(b) ** 2).sum())
    assert abs(pb - pa) <= 1e-9 * pa, (pa, pb)


def test_the_screen_amplitude_is_untouched(_warm):
    """SCOPE.  ``amplitude_model='screen'`` does not derive its amplitude from
    the inverse map -- it comes from ``apply_real_lens``'s analytic transport
    -- so there is nothing for an extrapolated inverse to corrupt and the bound
    must not engage at all."""
    kw = dict(amplitude_model='screen', preserve_input_phase=True)
    a, _ = _call(_CLEAN, bound=False, **kw)
    b, _ = _call(_CLEAN, bound=True, **kw)
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# 4.  Both newton_fit backends, bounded identically.
# ---------------------------------------------------------------------------
def test_both_newton_backends_get_the_same_support(_warm):
    """The support is taken from the traced samples BEFORE the fit-domain
    restriction, so it does not depend on which interpolant is fitted to them.
    D7's contract (the polynomial must track the unrestricted spline map to
    < 1e-3 of peak) has to survive the bound, and the bound must not drive the
    two apart."""
    pa, _ = _call(_GHOST, bound=False, newton_fit='polynomial')
    sa, _ = _call(_GHOST, bound=False, newton_fit='spline')
    pb, _ = _call(_GHOST, bound=True, newton_fit='polynomial')
    sb, _ = _call(_GHOST, bound=True, newton_fit='spline')
    scale = float(np.abs(sa).max())
    d_off = float(np.max(np.abs(np.abs(pa) - np.abs(sa)))) / scale
    d_on = float(np.max(np.abs(np.abs(pb) - np.abs(sb)))) / scale
    assert d_on < 1e-3, d_on
    assert d_on <= max(d_off * 1.05, 1e-9), (
        f"the support bound drove the two newton_fit backends apart: "
        f"{d_off:.3e} -> {d_on:.3e}")


# ---------------------------------------------------------------------------
# 5.  The support follows the BEAM, not the grid.
# ---------------------------------------------------------------------------
def test_a_decentred_beam_is_not_cut(_warm):
    """The case a grid-referenced radius gets wrong, and the case the C4
    transpose defect hid in: the hull is built from the traced landing points
    of the aperture-passing rays, so it moves with the beam.  A beam pushed
    1.5 w off centre must keep its power and its peak."""
    cx = 1.5 * _CLEAN['w']
    a, _ = _call(_CLEAN, bound=False, cx=cx)
    b, _ = _call(_CLEAN, bound=True, cx=cx)
    pa = float((np.abs(a) ** 2).sum())
    pb = float((np.abs(b) ** 2).sum())
    assert abs(pb - pa) <= 1e-6 * pa, (pa, pb)
    assert abs(float(np.abs(b).max()) - float(np.abs(a).max())) <= (
        1e-9 * float(np.abs(a).max()))


# ---------------------------------------------------------------------------
# 6.  The feather.
# ---------------------------------------------------------------------------
def test_the_feather_is_monotone_in_its_width(_warm):
    """The taper is a raised cosine over ``[0, f]`` of the signed distance
    OUTSIDE the hull, so widening ``f`` can only admit more of the band -- the
    returned amplitude must be pointwise non-decreasing in the feather, and a
    hard cut must be the most aggressive setting of all."""
    fields = [_call(_CLEAN, bound=True, feather=f)[0]
              for f in (0.0, 0.5, 1.0, 2.0)]
    for lo, hi in zip(fields[:-1], fields[1:]):
        assert np.all(np.abs(hi) >= np.abs(lo) * (1.0 - 1e-12) - 1e-300)
    powers = [float((np.abs(F) ** 2).sum()) for F in fields]
    assert powers == sorted(powers)


def test_a_hard_cut_still_removes_the_lobe(_warm):
    """The feather is insurance on the SAMPLING, not on the defect: at zero
    feather the bound is a hard binary mask and it must still remove the
    manufactured light.  (What the feather buys is not cutting real light at
    the boundary -- pinned by the monotonicity test above.)"""
    R = _radii(_GHOST)
    r3w = 3.0 * _GHOST['w']
    before, _ = _call(_GHOST, bound=False, guard=True)
    hard, _ = _call(_GHOST, bound=True, feather=0.0, guard=True)
    assert _halo(hard, R, r3w) < _halo(before, R, r3w) / 100.0


# ---------------------------------------------------------------------------
# 7.  It declines rather than guessing.
# ---------------------------------------------------------------------------
def test_a_degenerate_support_declines_instead_of_raising(monkeypatch, _warm):
    """A support with no interior (collinear or duplicated landings) has no
    hull, and Qhull raises.  The bound must then decline -- return the
    unbounded field -- rather than propagate an exception out of a physics
    call.  Forced here by making the hull constructor raise."""
    import scipy.spatial as _sp

    class _Boom(Exception):
        pass

    def _raise(*a, **k):
        raise _Boom('forced degenerate support')

    ref, _ = _call(_CLEAN, bound=False)
    monkeypatch.setattr(_sp, 'ConvexHull', _raise)
    got, _ = _call(_CLEAN, bound=True)
    assert np.array_equal(got, ref)
