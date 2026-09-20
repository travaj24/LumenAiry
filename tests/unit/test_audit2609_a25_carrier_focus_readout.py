"""WP-A25 -- the P2 battery's unclipped doublet cell scores a periodic COPY of
its own focus, and ``replica_fill='zero'`` is the readout knob that stops it.

Finding A25 of ``docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/
WP-A25_REPORT.md``.

WHAT IS GOING ON
----------------
``carrier_referenced_focus_readout`` finishes on a Bluestein zoom whose
reconstruction is PERIODIC in the absolute output coordinate, with a period
equal to the co-moving grid's extent at the stop plane -- a few beam radii near
a focus.  Everything a caller requests beyond ``|u| <= period/2`` of the field's
own origin is a repeat of what the transform already evaluated;
``on_replica='error'`` refuses such a window and has since D3 (2026-08-06).

``test_niche_p2_design_battery.py``'s through-focus helper waives that refusal
-- it wants ``n_out`` wide because the same ``n_out`` sizes its own scan
transform -- so on the unclipped cell (a 2 mm Gaussian through a 50 mm achromat
at a 2.5x aperture) it asks for 512 x 0.5 um = 256.000 um against a 124.113 um
period and gets 2.063 periods of window.  A repeat is not a degraded answer: it
is a full-amplitude image of the core laid down where the real field is weak,
so ``np.argmax`` over the window finds it.  Measured on this box 2026-09-13,
the fixture unchanged and only the library varying:

    tree / request                     FWHM   ratio   EE1w    EE2w    dz_best
    a1ff1e6e, 818251fd (pre-C1)       16.50   0.948   0.843   0.953   +0.131 mm
    a18ab074 .. 6c83ac91              20.50   1.177   0.353   0.495   +0.393 mm
    ... with replica_fill='zero'      18.50   1.062   0.8585  0.9970  +0.131 mm
    ... standoff 768 um (period>=win) 18.50   1.062   0.8585  0.9970  +0.131 mm
    ... standoff 1536 um              18.50   1.062   0.8585  0.9970  +0.131 mm

Neither historical reading was the beam: both scored a copy.  Pre-C1 the
brightest sample sat at output pixel (30, 30), 1.0040 periods off axis on both
axes, far enough inside the window that the encircled-energy disc still fitted
and 0.953 looked plausible.  WP-A6's C1 lengthened the beam-referenced leg
337.468 -> 372.144 um (the correct leg: the envelope carries a fitted residual
curvature of +0.01762 /m against a carrier ``1/R`` of -15.1596 /m), the period
followed by the same 10.27 %, and the brightest copy moved onto the window
CORNER at (0, 0) -- where three quarters of that disc falls off the grid.  That
is the whole of the 0.953 -> 0.495 step, and it is a property of the REQUEST,
not of C1.

The bottom three rows are three INDEPENDENT readout geometries with no copies
in them; they agree to the digit, and with this fixture's own 2026-07-25 record
(18.5 um, 1.062x, EE1w 86.0 %, EE2w 99.7 %, EE3w 99.8 %).

THE ORACLE AND ITS FLOOR
------------------------
The oracle is the analytic Gaussian focus of the beam the chain itself
delivers, ``fwhm_th = 1.177 lambda |R| / (pi w_exit)`` -- the battery's own
oracle, and independent of every readout knob under test.  Its floor: the truth
is the oracle plus the design's residual, and the three faithful geometries put
that at ``fwhm/fwhm_th = 1.0624`` with ZERO spread (18.500 um on all three).
The only spread left is the reading's quantisation: the FWHM comes off a radial
profile binned at ``dx_out``, so it moves in steps of ``2 dx_out`` = 1.0 um =
0.0574 in ratio units.  Bars below are stated in that bin.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import warnings

import numpy as np
import pytest

from lumenairy.propagators import carrier as C

# The fixture under pin is the battery's own, imported by path so the two
# cannot drift apart (the pattern test_audit2609_a22 / a23 already use).
_BATTERY = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'test_niche_p2_design_battery.py')


def _battery():
    if 'a25_battery' in sys.modules:
        return sys.modules['a25_battery']
    spec = importlib.util.spec_from_file_location('a25_battery', _BATTERY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['a25_battery'] = mod
    spec.loader.exec_module(mod)
    return mod


_RUNS = {}


def _cell(fill='zero'):
    """``_through_focus(_d_doublet, 2.0e-3, 2.5)`` plus the readout's own
    inputs and diagnostics.

    ``fill='zero'`` applies the one-key change this work package asks of the
    battery helper -- ``focus_readout=dict(..., replica_fill='zero')`` -- by
    injecting it at the readout call, so the fixture itself is bit-identical
    to the shipped one.  ``fill='repeat'`` is the shipped request verbatim.
    """
    if fill in _RUNS:
        return _RUNS[fill]
    B = _battery()
    B._skip_if_low_ram()                 # the fixture's own N=1024 precondition
    B._CACHE.clear()
    cap = {}
    orig = C.carrier_referenced_focus_readout

    def _spy(env, R, z, wavelength, dx, **kw):
        kw['replica_fill'] = fill
        pd = kw.get('_period_out')
        if pd is None:
            pd = kw['_period_out'] = {}
        out = orig(env, R, z, wavelength, dx, **kw)
        cap.update(env=np.asarray(env), R=float(R), z=float(z), dx=float(dx),
                   wavelength=float(wavelength), pd=dict(pd),
                   dx_out=float(kw['dx_out']), N_out=int(kw['N_out']),
                   centre_out=tuple(kw.get('centre_out', (0.0, 0.0))),
                   field=np.asarray(out))
        return out

    try:
        C.carrier_referenced_focus_readout = _spy
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fwhm, fwhm_th, ee, dz = B._through_focus(B._d_doublet, 2.0e-3, 2.5)
    finally:
        C.carrier_referenced_focus_readout = orig
        B._CACHE.clear()
    cap.update(fwhm=float(fwhm), fwhm_th=float(fwhm_th), ee=dict(ee),
               dz=float(dz))
    # The plane the scan scored, and WHERE in the window it found its peak --
    # the reading that separates the spot from an image of it.  Same transform
    # and same grid the battery's own scan uses.
    from lumenairy.propagators.mft import angular_spectrum_propagate_mft
    Ez = (cap['field'] if cap['dz'] == 0.0 else
          angular_spectrum_propagate_mft(cap['field'], cap['dz'],
                                         cap['wavelength'], cap['dx_out'],
                                         cap['dx_out'], cap['N_out']))
    I = np.abs(np.asarray(Ez)) ** 2
    iy, ix = np.unravel_index(int(np.argmax(I)), I.shape)
    cap['peak_ij'] = (int(iy), int(ix))
    cap['peak_uv'] = ((ix - cap['N_out'] / 2.0) * cap['dx_out'],
                      (iy - cap['N_out'] / 2.0) * cap['dx_out'])
    _RUNS[fill] = cap
    return cap


# Measured truth and quantisation, 2026-09-13 (see the module docstring).
_R_TRUE = 1.0624                  # fwhm / fwhm_th on three faithful geometries
_BIN = 2.0 * 0.5e-6               # FWHM step = 2 dx_out of the battery readout


# ===========================================================================
# 1. The physics: read on a window the transform can deliver, the cell IS the
#    analytic Gaussian focus.
# ===========================================================================
def test_the_unclipped_cell_reproduces_the_analytic_gaussian_focus():
    """ORACLE: the analytic Gaussian focus of the chain's own exit beam, which
    no readout knob can move.  FLOOR: the design's residual puts the truth at
    1.0624x it, with zero spread across three independent faithful readout
    geometries (``replica_fill='zero'`` at the resolved leg, and standoffs of
    768 um and 1536 um, all 18.500 um); the reading is quantised at one radial
    bin, 2 dx_out = 1.0 um = 0.0574 in ratio units.

    TWO-SIDED, both sides derived:

    * below -- an UNCLIPPED Gaussian cannot focus tighter than its own Gaussian
      limit, so the physical floor is 1.000 and the bar sits one bin under it
      at 0.94 (the truth is 2.1 bins above it);
    * above -- the battery's shipped 1.10 (the truth is 0.65 bins under it).

    DEFECT SCALE: the copy-scored readings the same fixture returns without the
    knob -- 1.1773 (20.50 um) and, on the pre-C1 trees, 0.9476 (16.50 um) --
    are 1.2 and 0.4 bins outside those bars.  The encircled-energy arm below is
    the sharp one (0.4953 against a 0.95 bar).
    """
    c = _cell()
    r = c['fwhm'] / c['fwhm_th']
    assert 0.94 <= r <= 1.10, (
        f"best-focus FWHM {c['fwhm'] * 1e6:.3f} um against the analytic "
        f"Gaussian {c['fwhm_th'] * 1e6:.3f} um (ratio {r:.4f}); the truth on "
        f"this fixture is {_R_TRUE:.4f} and one reading bin is "
        f"{_BIN / c['fwhm_th']:.4f}")
    assert abs(r - _R_TRUE) <= 2.0 * _BIN / c['fwhm_th'], (
        f"ratio {r:.4f} is more than two reading bins from the measured truth "
        f"{_R_TRUE:.4f} -- re-derive it, dated, if the readout changed on "
        f"purpose")


def test_the_cell_keeps_the_launched_power_inside_the_focal_waists():
    """The same oracle read as ENERGY -- the arm the copies actually broke.

    An unclipped Gaussian puts ``1 - exp(-8)`` = 99.966 % of its power inside
    two waists and ``1 - exp(-2)`` = 86.47 % inside one; measured here 0.9970
    and 0.8585 (the shortfall is the 2.5x aperture, the doublet's residual and
    the readout's own loss).  Two-sided: a readout cannot deliver MORE than was
    launched, so the upper bar is 1 + 2e-3 (the measured round-off of the same
    energy sum on the CARRIER fixtures); the lower is the battery's shipped
    0.95.  Without the knob this arm reads 0.4953 -- 0.45 under the bar,
    against a 0.047 margin over it.
    """
    ee = _cell()['ee']
    assert 0.95 <= ee[2] <= 1.0 + 2e-3, (
        f'EE inside two waists {ee[2]:.4f} (1w {ee[1]:.4f}, 3w {ee[3]:.4f})')
    assert ee[3] >= ee[2] >= ee[1], ee
    assert 0.80 <= ee[1] <= 0.90, (
        f'EE inside one waist {ee[1]:.4f} against an analytic 0.8647')


def test_the_best_focus_plane_is_where_the_beam_puts_it_not_where_a_copy_does():
    """A DECISION with a stated basin, not a reading: the scan's best plane
    must lie inside the depth of focus of the beam the chain delivered.  The
    target plane is the geometric focus and the design's residual defocus moves
    the true best focus to dz = +0.131 mm = +0.25 z_R (identical on all three
    faithful geometries); scoring a copy instead moves it to +0.393 mm =
    +0.75 z_R, two scan steps away and three quarters of the way to the edge of
    the scan.  The bar is half a Rayleigh range."""
    c = _cell()
    w_f = c['fwhm_th'] / np.sqrt(2.0 * np.log(2.0))
    zR = np.pi * w_f * w_f / c['wavelength']
    assert abs(c['dz']) <= 0.5 * zR, (
        f"best focus at dz = {c['dz'] * 1e3:+.4f} mm = {c['dz'] / zR:+.3f} "
        f"z_R (z_R = {zR * 1e3:.4f} mm)")


def test_the_scan_finds_the_spot_on_axis_and_not_an_image_of_it():
    """The reading that separates a focus from an image of it.  The beam is on
    axis, so the peak of the scan's best plane belongs at the window's centre;
    the nearest image of the core is one whole period away -- 124.113 um =
    8.4 focal waists -- so the basin is half of that and the bar is a quarter
    period (4.2 waists).  Measured at the best plane: 0.0 um from the centre
    with ``replica_fill='zero'``, 181.0 um (the window's corner, pixel (0, 0))
    without it."""
    c = _cell()
    off = float(np.hypot(*c['peak_uv']))
    assert off <= 0.25 * min(c['pd']['period']), (
        f"peak {off * 1e6:.3f} um from the window centre at pixel "
        f"{c['peak_ij']}; one period is "
        f"{min(c['pd']['period']) * 1e6:.3f} um")


# ===========================================================================
# 2. The defect, characterised on the shipped request (FAIL-BEFORE).
# ===========================================================================
def test_the_shipped_request_scores_an_image_of_the_core():
    """The fixture as it ships -- ``on_replica='ignore'`` and the default
    ``replica_fill='repeat'`` -- scores a replica, and this arm says so as a
    DECISION rather than by pinning the defect's size.  Two halves, both about
    WHERE the scan's peak is:

    * it is more than a quarter period from the window centre, so it is not the
      spot (the beam is on axis); measured 181.0 um = 1.46 periods, the
      window's corner pixel (0, 0);
    * reduced into the principal period it lands within ONE focal waist of the
      origin on each axis, which is what makes it an image of the core rather
      than some other feature: measured 3.887 um = 0.26 w_f on both axes,
      against a 14.789 um waist.

    The reading that comes with it -- 20.50 um / 1.177x / EE2w 0.4953 -- is
    quoted with loose bars because its job is to prove the pins above are
    load-bearing, not to pin a defect.  Same leg, same containment, same
    period as the passing run: ONLY the content of the window outside one
    period differs.
    """
    bad, good = _cell('repeat'), _cell()
    per = min(bad['pd']['period'])
    w_f = bad['fwhm_th'] / np.sqrt(2.0 * np.log(2.0))
    off = float(np.hypot(*bad['peak_uv']))
    assert off > 0.25 * per, (off, per, bad['peak_ij'])
    for u in bad['peak_uv']:
        assert abs(u) > 0.25 * per, (u, per)
        assert abs(u - round(u / per) * per) <= w_f, (
            f'{u * 1e6:.3f} um does not reduce to within one waist '
            f'({w_f * 1e6:.3f} um) of the origin')
    assert bad['fwhm'] / bad['fwhm_th'] > 1.12, (bad['fwhm'], bad['fwhm_th'])
    assert bad['ee'][2] < 0.60, bad['ee']
    assert float(bad['pd']['window_energy_frac']) > 3.0, bad['pd']
    assert bad['pd']['standoff'] == pytest.approx(good['pd']['standoff'],
                                                  rel=0, abs=0)
    assert bad['pd']['period'] == good['pd']['period']
    assert bad['pd']['containment'] == pytest.approx(good['pd']['containment'],
                                                     rel=0, abs=0)


def test_the_core_s_replica_enters_the_window_at_exactly_two_periods():
    """WHY the fixture's reading collapsed where it did, as a geometric
    criterion rather than a threshold: the nearest replica's CENTRE sits one
    period from the origin and the window's edge at half its span, so a full
    image of the core is inside the window exactly when
    ``N_out*dx_out > 2*period``.

    Two-sided on the SAME leg and the same fill (``'repeat'``), with only the
    window varied, so nothing but the criterion can move the answer.  The two
    arms bracket it by EIGHT output samples on each side -- the narrow one is
    the largest even ``N_out`` that keeps the nearest replica's centre 8
    samples OUTSIDE the window, and the fixture's own 512 puts it 7.8 samples
    inside.  Measured 2026-09-13: at ``N_out = 480`` the window is 240.0 um =
    1.934 periods and the scan reads 18.500 um / EE2w 0.9970 with the peak on
    axis; at 512 it is 256.0 um = 2.063 periods and reads 20.500 um / 0.4953
    with the peak in the corner.  (The same criterion explains the leg history
    independently: at the 0.8 z_R leg the window is 1.829 periods and the cell
    reads 18.500 um / 0.9970 even though 1.23x of the window's power is
    replicas.)

    This is the pin that stops the next resizing of the readout from silently
    reopening the finding: a standoff change that shrinks the period past
    ``N_out*dx_out / 2`` puts the fixture back on the wrong side of it."""
    B = _battery()
    c = _cell('repeat')
    per = min(c['pd']['period'])
    dxo = c['dx_out']
    n_ok = 2 * int(np.floor(per / dxo - 8.0))       # even; replica centre 8
    assert 2.0 * per < c['N_out'] * dxo, (per, c['N_out'])   # samples outside
    assert n_ok * dxo < 2.0 * per and n_ok > 0, (n_ok, per)
    assert n_ok * dxo > per, (n_ok, per)     # ... and still over ONE period
    B._CACHE.clear()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fwhm, fwhm_th, ee, dz = B._through_focus(B._d_doublet, 2.0e-3, 2.5,
                                                 dx_out=dxo, n_out=n_ok)
    B._CACHE.clear()
    # under two periods the replicas are still in the wings only
    assert 0.94 <= fwhm / fwhm_th <= 1.10, (fwhm, fwhm_th, n_ok)
    assert ee[2] > 0.95, (ee, n_ok)
    # ... and over it, on the very same leg, they are not
    assert c['fwhm'] / c['fwhm_th'] > 1.12 and c['ee'][2] < 0.60, (
        c['fwhm'], c['ee'])


def test_the_cell_really_does_ask_for_more_window_than_one_period():
    """The precondition of everything above, forced and asserted separately
    (S4): if the battery's window ever stops exceeding the period, these pins
    stop testing the fill and must be re-derived rather than quietly passed.
    Measured: 512 x 0.5 um = 256.000 um against 124.113 um = 2.063 periods,
    249 of 512 samples per axis faithful -- and 249 is the period the readout
    reports, not a stored number."""
    c = _cell()
    per = min(c['pd']['period'])
    assert c['N_out'] * c['dx_out'] > per
    nx, ny = c['pd']['faithful_samples']
    assert 0 < nx < c['N_out'] and 0 < ny < c['N_out'], (nx, ny, c['N_out'])
    assert nx == ny == 2 * int(np.floor(0.5 * per / c['dx_out'])) + 1, (
        f'{nx} faithful samples per axis for a {per * 1e6:.4f} um period at '
        f'{c["dx_out"] * 1e6:.4f} um')


def test_every_sample_outside_one_period_is_exactly_zero_under_the_knob():
    """EXACT, no tolerance.  Outside ``|u| <= period/2`` of the field's own
    origin the transport has nothing to report, and with ``replica_fill='zero'``
    the readout writes nothing there."""
    c = _cell()
    per, n = c['pd']['period'], c['N_out']
    u = (np.arange(n) - n / 2.0) * c['dx_out']
    out_x = np.abs(u + c['centre_out'][0]) > 0.5 * float(per[0]) * (1 + 1e-9)
    out_y = np.abs(u + c['centre_out'][1]) > 0.5 * float(per[1]) * (1 + 1e-9)
    assert out_x.any() and out_y.any()
    F = c['field']
    assert np.all(F[out_y, :] == 0.0), 'rows outside one period are not zero'
    assert np.all(F[:, out_x] == 0.0), 'columns outside one period are not zero'
    assert np.abs(F[np.ix_(~out_y, ~out_x)]).max() > 0.0
    # ... and the part that survived is the part the default keeps unchanged
    G = _cell('repeat')['field']
    assert np.array_equal(F[np.ix_(~out_y, ~out_x)], G[np.ix_(~out_y, ~out_x)])


def test_the_window_power_bound_holds_under_the_knob_and_is_broken_without_it():
    """A Bluestein window is a sub-window of one period, so its power is
    bounded by the stop plane's -- the bound the module's own
    ``_FOCUS_READOUT_WINDOW_ENERGY_TOL`` tripwire states.  Two-sided on the
    same cell: 0.99873 with ``replica_fill='zero'`` (inside the 1.01 tripwire,
    and above 0.9 so the beam is not being thrown away), 5.7004 without it --
    5.6x over, i.e. the readout hands back 4.7x the launched power as signal.
    """
    good, bad = _cell(), _cell('repeat')
    wef = float(good['pd']['window_energy_frac'])
    assert 0.9 < wef <= 1.0 + C._FOCUS_READOUT_WINDOW_ENERGY_TOL, wef
    assert float(bad['pd']['window_energy_frac']) > 5.0, bad['pd']


def test_c1s_beam_referenced_leg_is_still_the_leg_this_cell_runs_on():
    """This work package must not have been bought by shortening WP-A6's C1
    leg.  On this cell the envelope's fitted residual curvature is +0.01762 /m
    against a carrier ``1/R`` of -15.1596 /m, so the beam-referenced solve asks
    for a LONGER leg than the carrier-referenced law -- 372.144 um against
    337.468 um, a factor 1.1027 -- and the readout must still be using it."""
    c = _cell()
    env, R, z, dx = c['env'], c['R'], c['z'], c['dx']
    cen = C._envelope_amp_centroid(env, dx, dx)
    w_env = C._envelope_amp_radius(env, dx, dx, centre=cen)
    half = (0.5 * min(env.shape[-1], env.shape[-2]) * dx
            - max(abs(cen[0]), abs(cen[1])))
    inv = C._fit_carrier_inv(env, c['wavelength'], dx, dx, axis=None,
                             estimator='increment', centre=cen,
                             stride=C._fit_carrier_diag_stride(env.shape))
    assert inv != 0.0, 'fixture must carry residual envelope curvature'
    s_beam = C._beam_containment_standoff(env, R, z, c['wavelength'], dx,
                                          w_env, cen, half, inv_env=inv)
    s_used = float(c['pd']['standoff'])
    assert s_beam > 0.0 and s_used == pytest.approx(s_beam, rel=1e-12), (
        f'resolved standoff {s_used:.6e} m is not the beam-referenced leg '
        f'{s_beam:.6e} m')
    w0f = c['wavelength'] * abs(R) / (np.pi * w_env)
    zR = np.pi * w0f * w0f / c['wavelength']
    ext = half / w_env
    f_cap = np.sqrt(C._FOCUS_STANDOFF_WAIST_GROWTH ** 2 - 1.0)
    m_req = min(C._FOCUS_STANDOFF_MARGIN,
                f_cap / np.sqrt(1.0 + f_cap * f_cap) * ext)
    s_carrier = m_req / np.sqrt(ext * ext - m_req * m_req) * zR
    assert s_used > s_carrier, (s_used, s_carrier)
    assert c['pd']['containment'] > C._FOCUS_READOUT_CONTAINMENT_MIN


# ===========================================================================
# 3. The knob itself, on a synthetic fixture (no chain, no prescription).
# ===========================================================================
_WL = 1.31e-6
_RMAG = 20.0e-3


def _gauss_pupil(n=512, na=0.05, ext=4.0):
    w = na * _RMAG
    dx = 2.0 * ext * w / n
    x = (np.arange(n) - n / 2) * dx
    return (np.exp(-(x[None, :] ** 2 + x[:, None] ** 2) / w ** 2)
            .astype(np.complex128), dx, w)


def _read(env, dx, dx_out, n_out, **kw):
    pd = {}
    F = C.carrier_referenced_focus_readout(
        env, -_RMAG, _RMAG, _WL, dx, dx_out=dx_out, N_out=n_out,
        _period_out=pd, **kw)
    return np.asarray(F), pd


def _one_period(env, dx, w):
    _, pd = _read(env, dx, (_WL * _RMAG / (np.pi * w)) / 8.0, 16)
    return min(pd['period'])


def test_a_faithful_window_is_untouched_by_either_fill():
    """Nothing the fill does may reach a window the replica guard would pass.
    The short-circuit is exact -- the SAME object, not merely equal values --
    so every shipped configuration whose window fits one period is
    byte-identical on both settings."""
    env, dx, w = _gauss_pupil()
    dxo = (_WL * _RMAG / (np.pi * w)) / 8.0
    F_rep, pd = _read(env, dx, dxo, 32)
    F_zero, pd2 = _read(env, dx, dxo, 32, replica_fill='zero')
    assert pd['faithful_samples'] == pd2['faithful_samples'] == (32, 32)
    assert 32 * dxo <= min(pd['period'])
    assert np.array_equal(F_rep, F_zero)
    assert C._fill_readout_replicas(F_rep, pd['period'], dxo, 32,
                                    (0.0, 0.0), 'zero') is F_rep


def test_the_two_fills_differ_only_outside_one_period():
    """Two-sided on the knob, on the same call: inside the period the two
    settings are bit-identical, outside it 'repeat' is non-zero everywhere the
    field is and 'zero' is exactly zero.  A future resize of the readout that
    separated the fill from the guard's own condition would fail here."""
    env, dx, w = _gauss_pupil()
    per = _one_period(env, dx, w)
    n = 256
    dxo = per * 1.6 / n
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        F_rep, pd_rep = _read(env, dx, dxo, n, on_replica='ignore',
                              replica_fill='repeat')
        F_zero, pd_zero = _read(env, dx, dxo, n, on_replica='ignore',
                                replica_fill='zero')
    assert pd_rep['faithful_samples'] == pd_zero['faithful_samples']
    assert pd_rep['faithful_samples'][0] < n
    u = (np.arange(n) - n / 2.0) * dxo
    inside = np.abs(u) <= 0.5 * min(pd_rep['period']) * (1 + 1e-9)
    assert inside.any() and (~inside).any()
    assert np.array_equal(F_rep[np.ix_(inside, inside)],
                          F_zero[np.ix_(inside, inside)])
    assert np.abs(F_zero[~inside, :]).max() == 0.0
    assert np.abs(F_rep[~inside, :]).max() > 0.0


def test_the_fill_is_validated_and_not_a_silent_fall_through():
    """A typo must not quietly restore the repeats -- the same defect class as
    the ``on_replica`` / ``gap_kernel`` fall-throughs this campaign fixed.

    The shipped default was ``'repeat'`` when this knob landed (WP-A25) and
    became ``'zero'`` in 5.49.0 (WP-C5, ledger item 1.7).  Both public
    readouts must carry the SAME default -- leaving one behind is the
    asymmetry that let the paraxial readout ship without a replica guard in
    the first place -- and ``tests/unit/test_c5_three_defaults.py`` adds the
    private Collins readout the chain reaches on that transport.
    """
    env, dx, w = _gauss_pupil(n=128)
    dxo = (_WL * _RMAG / (np.pi * w)) / 8.0
    for fn in (C.carrier_referenced_focus_readout,
               C.carrier_referenced_exact_focus_readout):
        import inspect
        assert inspect.signature(fn).parameters[
            'replica_fill'].default == 'zero'
    with pytest.raises(ValueError, match='replica_fill'):
        _read(env, dx, dxo, 16, replica_fill='zeros')


def test_the_refusal_names_the_knob_and_no_longer_promises_a_safe_peak():
    """The refusal is what a caller reads before deciding to waive it, so it
    has to be true.  It said the core survives replicas -- "the spot CORE is
    unaffected -- so a width or a peak still looks right" -- which is what the
    battery's waiver cites; on that cell at 2.063 periods the brightest sample
    IS a copy.  The message must now name the fill and must NOT carry the
    superseded claim (fail-before on the text)."""
    env, dx, w = _gauss_pupil()
    per = _one_period(env, dx, w)
    n = 256
    with pytest.raises(RuntimeError) as ei:
        _read(env, dx, per * 1.10 / n, n)
    msg = str(ei.value)
    assert "replica_fill='zero'" in msg
    # 5.49.0: 'zero' is the default, so the message has to name the OTHER one
    # too -- a caller who wants the periodic reconstruction has to be able to
    # find it from the refusal they are reading.
    assert "replica_fill='repeat'" in msg
    assert 'a width or a peak still looks right' not in msg
    # the V6 measured-overshoot wording the fix_v1_v8 pins parse stays
    assert 'm over' in msg and 'sample(s) per edge' in msg and 'ALIASES' in msg


def test_the_knob_keeps_the_readouts_input_dtype():
    """C3: the fill multiplies by a bool array, which is weak against the
    field's complex dtype under NEP 50, so a complex64 readout stays
    complex64 -- the property the v5.44 memory campaign rests on."""
    env, dx, w = _gauss_pupil(n=256)
    per = _one_period(env.astype(np.complex64), dx, w)
    dxo = per * 1.5 / 64
    F, pd = _read(env.astype(np.complex64), dx, dxo, 64, on_replica='ignore',
                  replica_fill='zero')
    assert pd['faithful_samples'][0] < 64
    assert F.dtype == np.complex64, F.dtype


def test_replica_fill_reaches_the_readout_through_the_chain():
    """The knob has to be usable from where the battery -- and every other
    chain caller -- actually sits: inside ``focus_readout``.  That means it is
    on ``_OUTPUT_GRID_PASSTHROUGH`` (so ``_FOCUS_READOUT_KEYS`` accepts it and
    the multi entry point's ``output_grid`` reaches it) and on the paraxial
    ``_par_kw`` whitelist (so it is forwarded rather than dropped).  Asserted
    by EFFECT, not by reading the lists: the same chain call with and without
    the key must differ exactly outside one period and nowhere inside it."""
    assert 'replica_fill' in C._FOCUS_READOUT_KEYS
    B = _battery()
    c = _cell()
    out = {}
    for fill in ('repeat', 'zero'):
        B._CACHE.clear()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res, _, _, _ = B._run_chain(
                B._d_doublet, 2.0e-3, 2.5, 'default', final_distance=c['z'],
                focus_readout=dict(dx_out=5e-7, N_out=512,
                                   on_replica='ignore', replica_fill=fill))
        out[fill] = np.asarray(res.field)
        st = res.stages[-1]
    B._CACHE.clear()
    n, dxo = 512, 5e-7
    per = min(st['readout_period'])
    u = (np.arange(n) - n / 2.0) * dxo
    inside = np.abs(u) <= 0.5 * per * (1 + 1e-9)
    assert inside.any() and (~inside).any()
    assert np.array_equal(out['repeat'][np.ix_(inside, inside)],
                          out['zero'][np.ix_(inside, inside)])
    assert np.abs(out['zero'][~inside, :]).max() == 0.0
    assert np.abs(out['repeat'][~inside, :]).max() > 0.0


def test_the_chain_publishes_the_faithful_window_size_per_stage():
    """``readout_faithful_samples`` rides beside ``readout_period`` /
    ``readout_containment`` / ``readout_window_energy`` on the chain's own
    stage record, so a caller who waived the refusal can see how much of the
    window it actually got without re-deriving the period."""
    c = _cell()
    B = _battery()
    B._CACHE.clear()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res, _, _, _ = B._run_chain(
            B._d_doublet, 2.0e-3, 2.5, 'default', final_distance=c['z'],
            focus_readout=dict(dx_out=5e-7, N_out=512, on_replica='ignore'))
    B._CACHE.clear()
    st = res.stages[-1]
    assert st['readout_faithful_samples'] == c['pd']['faithful_samples']
    assert st['readout_period'] == c['pd']['period']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
