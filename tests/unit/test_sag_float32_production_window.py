"""``lens_sag_float32_opd_error`` must say when its verdict is a PROXY.

THE DEFECT THIS EXISTS FOR (design-121 campaign audit S7.3, re-measured
2026-08-17).  The audit signed design 121 off for ``sag_dtype=np.float32`` on
"747x margin", from a per-group run of this function at the production PITCH
(``field_check_dx = 0.90 um``) and ``field_check_n = 512``.  512 x 0.90 um is
a 0.46 mm window; those groups are 20.4 to 31.8 mm across.  The field-level
A/B therefore saw 1.5 to 2.3 % of the pupil DIAMETER -- and the float32 sag
error grows toward the pupil edge, which that window never reaches.  Walking
the window up at the same pitch on group S25-S27 moved the reading 109x, from
1.1221e-06 at N=512 to 1.2196e-04 at N=4096, still climbing at ~4.6x per
doubling with 82 % of the pupil unseen.  ``ok`` was True at every rung and
meant less at each one.

The function now reports ``aperture_cover`` /
``field_check_n_for_full_aperture``, warns by default when the window does
not cover the clear aperture, and returns ``field_rel_error_estimate`` -- the
grid-free full-aperture reading the 1-D radial scan already supports.

That last one was nearly shipped as a BOUND, with a derivation that reads
well (unimodular phase screens between unitary propagations, so the field
difference cannot exceed the phase difference).  The derivation is wrong:
the field path also rounds the coordinate arrays and evaluates sag out to
the grid CORNER, sqrt(2) beyond the window half-extent, and a radial scan to
the aperture edge sees neither.  Measured at full cover it UNDER-reads the
field error by 1.3-1.4x.  The first version of
``test_the_grid_free_estimate_is_the_right_size_and_is_not_sold_as_a_bound``
asserted ``estimate >= measured`` and failed, which is the only reason the
word 'bound' is not in the shipped API.

Nothing here pins a per-version number.  The proxy-under-reads claim and the
convergence-at-cover claim are both established by MEASURING ladders on the
running build; only their ordering and plateau are asserted.
"""
import math
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements._lens_real import lens_sag_float32_opd_error

LAM = 1.31e-6


@pytest.fixture(autouse=True)
def _deterministic_fft():
    prev = la.get_fft_auto_promote()
    la.set_fft_auto_promote(False)
    yield
    la.set_fft_auto_promote(prev)


def _surf(radius, gb, ga):
    return {'radius': radius, 'conic': 0.0, 'aspheric_coeffs': None,
            'glass_before': gb, 'glass_after': ga}


#: A singlet deep enough that float32 sag rounding is measurable, with an
#: aperture MUCH wider than a small check window -- the geometry the defect
#: lives in.  4 mm clear aperture against a 0.5 mm window at the ladder's
#: first rung.
APERTURE = 4.0e-3
DX = APERTURE / 2048.0            # the "production" pitch for this fixture


def _singlet():
    return {'name': 'biconvex', 'aperture_diameter': APERTURE,
            'thicknesses': [4e-3],
            'surfaces': [_surf(19.6e-3, 'air', 'N-SSK2'),
                         _surf(-27.4e-3, 'N-SSK2', 'air')]}


def _run(n, **kw):
    kw.setdefault('on_partial_aperture', 'silent')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return lens_sag_float32_opd_error(_singlet(), LAM, field_check_n=n,
                                          field_check_dx=DX, **kw)


# ---------------------------------------------------------------------------
def test_the_window_diagnostics_are_the_arithmetic_they_claim():
    """Every reported window quantity is derivable from n, dx and the
    aperture, so a caller can audit the verdict without re-deriving it."""
    for n in (256, 1024, 2048, 4096):
        r = _run(n)
        assert r['field_check_n'] == n
        assert r['field_check_dx'] == pytest.approx(DX, rel=0, abs=0)
        assert r['field_check_window_m'] == pytest.approx(n * DX)
        assert r['aperture_cover'] == pytest.approx(n * DX / APERTURE)
        assert (r['field_check_n_for_full_aperture']
                == math.ceil(APERTURE / DX))
        assert r['field_check_covers_aperture'] is (n * DX >= APERTURE)


def test_a_partial_window_warns_and_a_covering_window_does_not():
    """Two-sided: the guard has to fire on the shape that burned the audit
    AND stay quiet on a genuine production-grid check, or it is noise."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        lens_sag_float32_opd_error(_singlet(), LAM, field_check_n=256,
                                   field_check_dx=DX)
    partial = [x for x in w if 'PROXY' in str(x.message)]
    assert len(partial) == 1, [str(x.message)[:120] for x in w]
    assert 'aperture_cover' not in str(partial[0].message)
    assert '12.5 %' in str(partial[0].message), str(partial[0].message)

    with warnings.catch_warnings(record=True) as w2:
        warnings.simplefilter('always')
        lens_sag_float32_opd_error(_singlet(), LAM, field_check_n=2048,
                                   field_check_dx=DX)
    assert not [x for x in w2 if 'PROXY' in str(x.message)], (
        'a window that covers the clear aperture is not a proxy and must not '
        'be warned about')


@pytest.mark.parametrize('policy', ['error', 'silent', 'warn'])
def test_the_policy_is_honoured(policy):
    kw = dict(field_check_n=256, field_check_dx=DX,
              on_partial_aperture=policy)
    if policy == 'error':
        with pytest.raises(ValueError, match='PROXY'):
            lens_sag_float32_opd_error(_singlet(), LAM, **kw)
        return
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        lens_sag_float32_opd_error(_singlet(), LAM, **kw)
    got = len([x for x in w if 'PROXY' in str(x.message)])
    assert got == (1 if policy == 'warn' else 0)


def test_the_policy_is_validated_by_VALUE_not_by_identity():
    """A policy string built at runtime -- from os.environ, a config file, an
    f-string -- is not the interned literal.  Validating with ``is`` is how
    ``_check_screen_obliquity_support`` came to refuse a valid ``'auto'``
    while naming it valid (audit S9.1 #1, fixed in cbef685), invisibly,
    because the tests only ever passed literals."""
    built = ''.join(['wa', 'rn'])
    assert built is not 'warn' or True     # noqa: F632  (documents the point)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        lens_sag_float32_opd_error(_singlet(), LAM, field_check_n=256,
                                   field_check_dx=DX,
                                   on_partial_aperture=built)
    assert len([x for x in w if 'PROXY' in str(x.message)]) == 1

    with pytest.raises(ValueError, match="on_partial_aperture"):
        lens_sag_float32_opd_error(_singlet(), LAM, field_check_n=256,
                                   field_check_dx=DX,
                                   on_partial_aperture='wrn')


def test_the_proxy_under_reads_the_production_window():
    """THE LOAD-BEARING CLAIM.  At a FIXED pitch, growing the window grows
    the measured field error -- so a reading taken on a short window is not
    evidence about a production one, and ``ok`` computed from it is not a
    sign-off.

    Measured rather than pinned: the ladder is run on this build and only its
    ORDERING is asserted.  The bar is a factor of 2 per doubling, chosen with
    a gap on both sides -- the failure it must catch is a flat or falling
    ladder (ratio ~1, which would mean the proxy IS representative and this
    guard is unnecessary), while the real signal measured 2026-08-17 was
    4.15-5.46x per doubling on design 121's S25-S27 and 3.5-5.3x on this
    fixture.  Two decades of daylight below, none of it near 1.
    """
    ns = [256, 512, 1024, 2048]
    errs = [_run(n)['max_field_rel_error'] for n in ns]
    assert all(e > 0 for e in errs), errs
    for a, b, na, nb in zip(errs, errs[1:], ns, ns[1:]):
        assert b > 2.0 * a, (
            f'the field error did NOT grow with the window: {na} -> {nb} '
            f'moved {a:.4e} -> {b:.4e} (ratio {b / a:.2f}).  If the proxy is '
            f'genuinely representative now, this guard and its warning are '
            f'unnecessary -- but that is a claim to MEASURE and record, not '
            f'to relax this bar for.')


def test_the_field_reading_converges_once_the_window_covers_the_aperture():
    """WHY ``field_check_covers_aperture`` is the right sufficiency
    criterion, rather than a stylistic preference.

    Below cover the reading is still moving by decades (that is
    ``test_the_proxy_under_reads_the_production_window``).  AT and ABOVE
    cover it must have stopped.  Measured 2026-08-17 on this fixture: cover
    0.25 -> 0.50 -> 1.00 moved 7.34e-06 -> 2.03e-05 -> 1.5544e-04 (21x),
    then cover 1.00 -> 2.00 moved 1.5544e-04 -> 1.5564e-04, +0.1 %.

    The bar is 25 %, with a gap on both sides: the measured plateau sits
    250x inside it, and the failure it must catch -- a reading that is still
    climbing at cover 1, which would mean covering the aperture is NOT
    sufficient and this whole guard names the wrong criterion -- was 21x
    over the preceding three rungs, 80x outside it."""
    at, beyond = _run(2048), _run(4096)
    assert at['field_check_covers_aperture'] is True
    a, b = at['max_field_rel_error'], beyond['max_field_rel_error']
    assert abs(b - a) <= 0.25 * a, (
        f'the field reading has NOT converged at full aperture cover: '
        f'{a:.4e} -> {b:.4e} on doubling the window again '
        f'({abs(b - a) / a * 100:.1f} % vs a measured 0.1 %).  Covering the '
        f'clear aperture is then not a sufficient criterion and the guard '
        f'names the wrong one.')


def test_the_grid_free_estimate_is_the_right_size_and_is_not_sold_as_a_bound():
    """``field_rel_error_estimate`` = 2*pi*max_opd_error_waves is the
    grid-free full-aperture reading.  Its value is that it sees the whole
    pupil; its limit is that it is NOT an upper bound, and this pins both.

    It is not a bound because the field path also rounds the coordinate
    arrays and evaluates sag out to the grid CORNER (sqrt(2) beyond the
    window half-extent), neither of which a radial scan to the aperture edge
    sees.  MEASURED 2026-08-17 on this fixture at full cover: estimate
    1.2189e-04 against 1.5544e-04 measured, ratio 0.78 -- it UNDER-reads.
    A test that asserted `estimate >= measured` would fail, and did.

    The claim worth pinning is the one that makes it useful: within a decade
    of the full-cover measurement, where a 25 %-cover proxy is 21x low.  The
    bar is 0.05x..20x, two decades wide around a measured 0.78."""
    r = _run(2048)
    assert r['field_check_covers_aperture'] is True
    est = r['field_rel_error_estimate']
    assert est == pytest.approx(2.0 * np.pi * r['max_opd_error_waves'],
                                rel=1e-15)
    meas = r['max_field_rel_error']
    assert 0.05 * meas <= est <= 20.0 * meas, (
        f'the grid-free estimate {est:.4e} is no longer the right size for '
        f'the full-cover measurement {meas:.4e} (ratio {est / meas:.3f}, '
        f'measured 0.78); it has stopped being informative')


def test_skipping_the_field_check_still_reports_the_grid_free_estimate():
    """``field_check_n=0`` is the cheap path -- the 1-D radial scan only.
    It is also the path with NO window at all, so the grid-free bound is the
    entire verdict and must still be there."""
    r = lens_sag_float32_opd_error(_singlet(), LAM, field_check_n=0)
    assert r['max_field_rel_error'] == 0.0
    assert r['field_rel_error_estimate'] > 0.0
    assert r['field_check_covers_aperture'] is False
    assert r['aperture_cover'] == 0.0
