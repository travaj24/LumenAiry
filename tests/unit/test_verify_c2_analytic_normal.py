"""VERIFY-WP-C2 -- decision tests for the gaps the WP-C2 verification found.

Every test here closes a gap that the shipped WP-C2 suite leaves open, and
every bar is either an identity, a quantity this build measures for itself,
or a census pinned so it can only move deliberately.  Nothing carries a
number from a prior run as a bar; recorded readings appear in comments with
their date and build, as the campaign requires.

Evidence: ``validation/probe_verify_c2/`` (both builds) and
``docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/VERIFY_WP-C2.md``.
"""
from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import math
import pathlib
import re
from decimal import Decimal, getcontext

import numpy as np
import pytest

from lumenairy.raytrace.intersection import _refract
from lumenairy.raytrace.surface import (
    RayBundle, Surface, _is_pure_spherical, _surface_normal)
from lumenairy.raytrace.trace import trace

REPO = pathlib.Path(__file__).resolve().parents[2]
WL = 587.5618e-9
EPS = float(np.finfo(np.float64).eps)
CLAMP = 0.9999


# ---------------------------------------------------------------- helpers

def _s(R, th, gb, ga, sd=np.inf, conic=0.0, mirror=False):
    return Surface(radius=R, conic=conic, thickness=th, glass_before=gb,
                   glass_after=ga, semi_diameter=sd, is_mirror=mirror)


def _doublet():
    return [_s(0.0517, 0.0090, 'air', 'N-BK7', 0.0125),
            _s(-0.0345, 0.0025, 'N-BK7', 'N-SF5', 0.0125),
            _s(-0.1200, 0.0400, 'N-SF5', 'air', 0.0125)]


def _ladder(n_pairs):
    out = []
    for j in range(n_pairs):
        out.append(_s(0.0800 + 0.003 * j, 0.0055, 'air', 'N-BK7', 0.011))
        out.append(_s(-0.0900 - 0.003 * j, 0.0090, 'N-BK7', 'air', 0.011))
    out.append(_s(0.2500, 0.0300, 'air', 'N-BK7', 0.011))
    return out


def _axial_bundle(n, hmax):
    """Rays along ``(0, 0, 1)`` -- the ONLY direction that is exactly unit
    in float64 (a dyadic ``(L, N)`` with ``L**2 + N**2 = 1`` exactly forces
    ``L = 0``), which is what lets an exact-arithmetic oracle be built."""
    h = np.linspace(-hmax, hmax, n)
    return RayBundle(x=h.copy(), y=(0.37 * h).copy(), z=np.zeros(n),
                     L=np.zeros(n), M=np.zeros(n), N=np.ones(n),
                     wavelength=WL, alive=np.ones(n, dtype=bool),
                     opd=np.zeros(n))


def _nz(x, y, R, analytic):
    a = np.array([float(x)])
    b = np.array([float(y)])
    return float(_surface_normal(a, b, _s(R, 0.01, 'air', 'N-BK7'),
                                 analytic_sphere=analytic)[2][0])


def _gate_threshold(R, analytic, steps=80):
    """LOCATE a route's domain gate by bisection on the running build."""
    lo, hi = 0.99, 1.0
    aR = abs(R)
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if math.isfinite(_nz(mid * aR, 0.0, R, analytic)):
            lo = mid
        else:
            hi = mid
    return lo


# ======================================================================
# 1.  The rim band's SHAPE -- it is not an annulus
# ======================================================================

@pytest.mark.parametrize('R', [0.0020, -0.0125, 0.0515, -0.1200, 1.0])
def test_vc2_the_two_domain_gates_coincide_on_the_axis(R):
    """The two routes gate the sphere from different expressions
    (``(x*x + y*y)/(R*R)`` against ``(1 + k) * sqrt(x*x + y*y)**2 / R**2``),
    and the shipped work package describes the result as "a rim band one ULP
    of ``h`` wide at ``0.99995 |R|``".

    It is narrower than that, and this pins the distinction: ALONG THE AXIS
    (``y = 0``, where ``sqrt(x*x)**2`` round-trips exactly) the two gates are
    the SAME float, at every radius.  The band exists only at azimuths where
    the two expressions round apart -- so a bundle confined to a meridional
    fan cannot enter it at all.

    Bar: exact equality of two located thresholds, which is an identity
    claim and not a tolerance.  MEASURED 2026-09-20, Windows py3.14 /
    numpy 2.4.4 and WSL py3.12 / numpy 2.4.6: both routes locate
    0.9999499987499374 at all five radii.
    """
    ta = _gate_threshold(R, True)
    tg = _gate_threshold(R, False)
    assert ta == tg, (
        f'the two domain gates no longer coincide on the axis at R = {R}: '
        f'analytic {ta!r} against generic {tg!r}.  If that is intended the '
        f'rim band has become reachable by a meridional fan, which the '
        f'Migration note does not say.')
    # and the located threshold IS sqrt(0.9999) to its own last bit
    assert abs(ta - math.sqrt(CLAMP)) <= 4.0 * math.ulp(math.sqrt(CLAMP)), (
        f'the located gate {ta!r} is no longer sqrt({CLAMP}) = '
        f'{math.sqrt(CLAMP)!r}')


def test_vc2_the_rim_band_is_reachable_only_off_axis():
    """The other half of the claim above, two-sided: a straddle point DOES
    exist off the axis, and it reaches ``_refract``'s ``alive`` flag.

    Constructed by a directed ``nextafter`` walk rather than sampled, and
    the arm fails if no straddle is found within the walk -- so this is a
    demonstration, not a search that may quietly return nothing.
    """
    found = None
    for R in (-0.0125, -0.1200):
        h0 = math.sqrt(CLAMP) * abs(R)
        for k in range(8):
            th = 2.0 * math.pi * k / 8 + 0.11
            for sgn in (+1.0, -1.0):
                h = h0
                for _ in range(400):
                    x, y = h * math.cos(th), h * math.sin(th)
                    fa = math.isfinite(_nz(x, y, R, True))
                    fg = math.isfinite(_nz(x, y, R, False))
                    if fa != fg:
                        found = (R, x, y, fa, fg)
                        break
                    h = math.nextafter(h, h + sgn)
                if found:
                    break
            if found:
                break
        if found:
            break
    assert found is not None, (
        'no point where the two domain gates disagree was found by a '
        '400-step nextafter walk at either radius.  Either the two gate '
        'expressions have been made identical -- in which case the '
        'Migration note about the rim band is now wrong -- or the walk no '
        'longer reaches it.')
    R, x, y, fa, fg = found
    assert fa and not fg, (
        f'the straddle found at R = {R} has the CLOSED FORM refusing and '
        f'the generic route accepting ({fa}, {fg}), which is the opposite '
        f'of the documented direction.')
    out = {}
    for route, flag in (('analytic', True), ('generic', False)):
        rb = RayBundle(x=np.array([x]), y=np.array([y]), z=np.zeros(1),
                       L=np.zeros(1), M=np.zeros(1), N=np.ones(1),
                       wavelength=WL, alive=np.ones(1, dtype=bool),
                       opd=np.zeros(1))
        _refract(rb, _s(R, 0.01, 'air', 'N-BK7'), 1.0, 1.5168,
                 renormalize=True,
                 sphere_normal=('analytic' if flag else 'generic'))
        out[route] = (bool(rb.alive[0]), int(rb.error_code[0]))
    assert out['generic'][0] is False and out['generic'][1] == 4, out
    assert out['analytic'][0] is True and out['analytic'][1] == 0, out


# ======================================================================
# 2.  The JAX backend has no clamp at all
# ======================================================================

def test_vc2_the_jax_backend_does_not_apply_the_numpy_domain_clamp():
    """The NumPy normal is NaN outside ``h**2/R**2 < 0.9999`` on BOTH
    routes; ``jax_trace._refract_jax`` builds the pure-spherical normal as
    ``(x, y, z - R)/R`` from the intersection's own ``z`` and applies no
    gate at all.  The two backends' vignetting therefore differs by the
    WHOLE outer annulus ``h > 0.99995 |R|``, not by one ULP.

    That is PRE-EXISTING -- it is the same under all four CPU
    ``(renormalize, sphere_normal)`` settings -- but nothing pinned it, and
    ``jax_gets_a_clamp`` (``validation/probe_verify_c2/vc2_mutplugin.py``)
    survives 294 raytrace and parity tests.  This makes the divergence a
    recorded decision: the counts are derived from the ray set, not
    recorded, so the arm moves only if a backend's gate moves.

    MEASURED 2026-09-20 on a ball lens (R = 12.5 mm, clear semi-diameter
    12.5 mm -- a catalogue part whose aperture reaches the rim), 40 000
    rim-packed rays: 1991 past the clamp, CPU keeps 0 of them on both
    routes, JAX keeps all 1991.  Both builds.
    """
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    from lumenairy.raytrace.jax_trace import make_jax_ray_state, trace_jax
    from lumenairy.raytrace import surfaces_from_prescription as sfp

    R = 0.0125
    pres = dict(name='ball', aperture_diameter=2 * R,
                surfaces=[dict(radius=R, conic=0.0, aspheric_coeffs=None,
                               radius_y=None, conic_y=None,
                               aspheric_coeffs_y=None, glass_before='air',
                               glass_after='N-BK7'),
                          dict(radius=-R, conic=0.0, aspheric_coeffs=None,
                               radius_y=None, conic_y=None,
                               aspheric_coeffs_y=None,
                               glass_before='N-BK7', glass_after='air')],
                thicknesses=[2 * R])
    surfs = sfp(pres)
    n = 4000
    r = R * np.linspace(0.9990, 0.999999, n)
    z = np.zeros(n)
    past = (r / R) > math.sqrt(CLAMP)
    n_past = int(past.sum())
    assert n_past > 100, (
        f'premise: the ray set must actually reach past the clamp for this '
        f'comparison to have content; only {n_past} of {n} do.')

    st = make_jax_ray_state(x=r, y=z.copy(), z=z.copy(), L=z.copy(),
                            M=z.copy(), N=np.ones(n))
    ja = np.asarray(trace_jax(st, pres, WL).alive, dtype=bool)

    for rn, sn in (('surface', 'generic'), ('exit', 'analytic')):
        rb = RayBundle(x=r.copy(), y=z.copy(), z=z.copy(), L=z.copy(),
                       M=z.copy(), N=np.ones(n), wavelength=WL,
                       alive=np.ones(n, dtype=bool), opd=np.zeros(n))
        ir = trace(rb, surfs, WL, output_filter='last',
                   renormalize=rn, sphere_normal=sn).image_rays
        a = np.asarray(ir.alive, dtype=bool)
        ec = np.asarray(ir.error_code)
        assert int((a & past).sum()) == 0, (
            f'the NumPy tracer ({rn}/{sn}) now keeps '
            f'{int((a & past).sum())} of the {n_past} rays past its own '
            f'domain clamp; the clamp is supposed to kill every one.')
        assert int(((ec == 4) & past).sum()) == n_past, (
            f'the NumPy tracer ({rn}/{sn}) no longer reports every ray '
            f'past the clamp as RAY_NAN.')
    assert int((ja & past).sum()) == n_past, (
        f'the JAX tracer now kills {n_past - int((ja & past).sum())} of the '
        f'{n_past} rays past the NumPy domain clamp.  If a clamp has been '
        f'added to jax_trace that is a cross-backend BEHAVIOUR CHANGE and '
        f'VERIFY_WP-C2.md defect D6 has been actioned -- update this arm '
        f'and the Migration Guide together.')


# ======================================================================
# 3.  analysis.ghost keeps the generic normal
# ======================================================================

def test_vc2_analysis_ghost_still_asks_for_the_private_default():
    """``analysis.ghost`` calls ``_refract`` / ``_reflect`` directly, and
    those kept ``sphere_normal='generic'``, so a ghost path and the public
    trace refracted off different arithmetic on the same sphere -- up to
    2.13e-14 mm of RMS spot radius apart on three 2-bounce paths of a
    spherical doublet (defect D5).

    ROUND 2 CLOSED IT and this arm is INVERTED: the ghost leg now asks
    ``raytrace.trace._library_trace_default('sphere_normal')`` -- the
    library's CURRENT public route, not a literal -- so it tracks the next
    default flip instead of pinning this one.  ``renormalize`` stays at the
    private ``True``, which D5 requires: the ghost loop has no exit pass to
    hoist a single rescale to.

    Both halves are still asserted here: the PRIVATE defaults have NOT
    moved (that is what keeps the finite-difference differential path
    unchanged by construction), and the ghost leg's two call sites now name
    the keyword.
    """
    for fn in (_refract,):
        p = inspect.signature(fn).parameters
        assert p['renormalize'].default is True, p['renormalize'].default
        assert p['sphere_normal'].default == 'generic', (
            p['sphere_normal'].default)
    src = (REPO / 'lumenairy' / 'analysis' / 'ghost.py').read_text(
        encoding='utf-8')
    for call in ("_reflect(rays, surfs[s_idx], sphere_normal=_ghost_sphere_normal)",
                 "_refract(rays, surfs[s_idx], n1, n2,\n"
                 "                     sphere_normal=_ghost_sphere_normal)"):
        assert call in src, (
            f'{call!r} is not in analysis/ghost.py.  D5 was closed by '
            f'making the ghost leg ask the library default; if the leg has '
            f'been changed again, re-measure and update this arm.')
    assert "_library_trace_default('sphere_normal')" in src, (
        'the ghost leg no longer asks the LIBRARY for its route.  A '
        "literal 'analytic' here would pin this release's default into "
        'the ghost path for every release after it, which is the failure '
        'D4 and D5 were both written to avoid.')
    # two-sided: the route it gets really is the public one
    from lumenairy.raytrace.trace import _library_trace_default
    assert (_library_trace_default('sphere_normal')
            == inspect.signature(trace).parameters['sphere_normal'].default)


# ======================================================================
# 4.  What the renormalise hoist costs, deterministically
# ======================================================================

class _Counted(np.ndarray):
    """Counts every element-wise operation applied to it.

    Both protocols are needed: ``np.where`` dispatches through
    ``__array_function__`` and, without it, returns a BASE ndarray, after
    which everything downstream is invisible.
    """
    tally = [0]

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        _Counted.tally[0] += max(
            [a.size for a in inputs
             if isinstance(a, np.ndarray) and a.ndim] or [0])
        kw = dict(kwargs)
        if 'out' in kw:
            kw['out'] = tuple(np.asarray(o).view(np.ndarray)
                              for o in kw['out'])
        raw = [np.asarray(a).view(np.ndarray)
               if isinstance(a, np.ndarray) else a for a in inputs]
        r = getattr(ufunc, method)(*raw, **kw)
        return r.view(_Counted) if isinstance(r, np.ndarray) else r

    def __array_function__(self, func, types, args, kwargs):
        _Counted.tally[0] += max(
            [a.size for a in args
             if isinstance(a, np.ndarray) and a.ndim] or [0])

        def strip(o):
            if isinstance(o, np.ndarray):
                return o.view(np.ndarray)
            if isinstance(o, (list, tuple)):
                return type(o)(strip(x) for x in o)
            return o

        impl = getattr(func, '_implementation', func)
        r = impl(*strip(args), **{k: strip(v) for k, v in kwargs.items()})
        return r.view(_Counted) if isinstance(r, np.ndarray) else r


def _elements(surfs, **kw):
    n = 512
    h = np.linspace(-0.0070, 0.0070, n)

    def mk():
        return RayBundle(x=h.copy().view(_Counted),
                         y=(0.3 * h).copy().view(_Counted),
                         z=np.zeros(n).view(_Counted),
                         L=np.zeros(n).view(_Counted),
                         M=np.zeros(n).view(_Counted),
                         N=np.ones(n).view(_Counted), wavelength=WL,
                         alive=np.ones(n, dtype=bool).view(_Counted),
                         opd=np.zeros(n).view(_Counted))

    trace(mk(), surfs, WL, output_filter='last', **kw)      # warm
    _Counted.tally[0] = 0
    trace(mk(), surfs, WL, output_filter='last', **kw)
    return _Counted.tally[0]


def test_vc2_the_renormalise_hoist_is_a_loss_below_three_surfaces():
    """What ``renormalize='exit'`` is worth, counted rather than timed.

    Three timing instruments were tried against this switch on the
    verification box and all three moved under load -- the 15.6 ms Windows
    process tick read the block as exactly ZERO, and two wall-clock runs of
    the same two arms disagreed in SIGN.  The element-op count does not
    depend on load or on the build at all.

    The hoist removes, per refracting surface, one ``np.maximum`` and three
    in-place divides (4 N-sized passes) and adds one
    ``_normalize_directions`` (3 squares + 2 adds + 1 sqrt + 1 maximum + 3
    divides = 10).  So it BREAKS EVEN between two and three surfaces, and
    is a net loss on a two-surface system.  That is the decision, and it
    bounds what the switch can be worth -- WP-B9 reported 1.03x-1.10x;
    measured here, the saving never exceeds 1.024x even at thirteen
    surfaces.

    MEASURED 2026-09-20, both builds, identical to the last digit:
    2 surfaces -8192 elements (0.9910x), 3 surfaces +8192 (1.0050x),
    7 +73728 (1.0189x), 13 +172032 (1.0237x).
    """
    two = [_s(-0.3000, -0.1000, 'air', 'air', 0.060, mirror=True),
           _s(-0.0900, 0.2000, 'air', 'air', 0.020, mirror=True)]
    got = {}
    for name, surfs in (('two', two), ('three', _doublet()),
                        ('seven', _ladder(3)), ('thirteen', _ladder(6))):
        a = _elements(surfs, sphere_normal='analytic',
                      renormalize='surface')
        b = _elements(surfs, sphere_normal='analytic', renormalize='exit')
        got[name] = (a, b, a - b)
    assert got['two'][2] < 0, (
        f'the exit hoist no longer COSTS array work on a two-surface '
        f'system: saved {got["two"][2]} elements.  The exit pass '
        f'recomputes the magnitude with its own sqrt and three squares, '
        f'so below three surfaces it must be a loss; if it is not, either '
        f'_normalize_directions or the per-surface block has changed. {got}')
    for name in ('three', 'seven', 'thirteen'):
        assert got[name][2] > 0, (name, got)
    # strictly increasing in the surface count, and bounded
    saved = [got[k][2] for k in ('two', 'three', 'seven', 'thirteen')]
    assert saved == sorted(saved), (saved, got)
    best = got['thirteen'][0] / got['thirteen'][1]
    assert best < 1.10, (
        f'the deterministic element-op bound on the hoist is now {best:.4f}x '
        f'at thirteen surfaces.  WP-B9 reported 1.03x-1.10x end to end; if '
        f'the count can now reach that the arithmetic has changed.')


def test_vc2_the_sphere_normal_switch_removes_a_sixth_of_the_array_work():
    """The same instrument for the other switch, with a two-sided control.

    MEASURED 2026-09-20, both builds identical: 1.1612x to 1.1973x on
    sphere-bearing prescriptions, and EXACTLY 1.0000x (zero elements) on a
    conic prescription the predicate cannot select.
    """
    conic = [_s(0.0517, 0.0090, 'air', 'N-BK7', 0.0125, conic=-0.6),
             _s(-0.0345, 0.0025, 'N-BK7', 'N-SF5', 0.0125, conic=-1.2),
             _s(-0.1200, 0.0400, 'N-SF5', 'air', 0.0125, conic=0.4)]
    for surfs in (_doublet(), _ladder(3)):
        g = _elements(surfs, sphere_normal='generic', renormalize='exit')
        a = _elements(surfs, sphere_normal='analytic', renormalize='exit')
        assert g > a, (g, a)
        assert 1.05 < g / a < 1.40, (
            f'the closed form now removes {g / a:.4f}x of the array work; '
            f'measured 1.1612x-1.1973x on 2026-09-20.')
    g = _elements(conic, sphere_normal='generic', renormalize='exit')
    a = _elements(conic, sphere_normal='analytic', renormalize='exit')
    assert g == a, (
        f'CONTROL: a prescription with no pure sphere must do exactly the '
        f'same array work either way; got {g} against {a}.  The selection '
        f'predicate has widened.')


# ======================================================================
# 5.  End to end against an exact-arithmetic oracle
# ======================================================================

def _oracle_trace(surfs, idx, x0, y0):
    """Exact-arithmetic trace of one AXIAL ray, at 60 digits."""
    getcontext().prec = 60
    D = Decimal
    px, py, pz = D(float(x0)), D(float(y0)), D(0)
    dl, dm, dn = D(0), D(0), D(1)
    opd = D(0)
    for i, s in enumerate(surfs):
        n1, n2 = D(idx[i][0]), D(idx[i][1])
        R = D(s.radius)
        ax, ay, az = px, py, pz - R
        a = dl * dl + dm * dm + dn * dn
        b = 2 * (dl * ax + dm * ay + dn * az)
        c = ax * ax + ay * ay + az * az - R * R
        disc = b * b - 4 * a * c
        if disc < 0:
            return None
        sq = disc.sqrt()
        t1, t2 = (-b - sq) / (2 * a), (-b + sq) / (2 * a)
        t = t1 if abs(t1) <= abs(t2) else t2
        px += dl * t
        py += dm * t
        pz += dn * t
        opd += n1 * t
        nx, ny, nz = -px / R, -py / R, -(pz - R) / R
        if dl * nx + dm * ny + dn * nz > 0:
            nx, ny, nz = -nx, -ny, -nz
        cos_i = -(dl * nx + dm * ny + dn * nz)
        eta = n1 / n2
        disc_r = 1 - eta * eta * (1 - cos_i * cos_i)
        if disc_r < 0:
            return None
        k = eta * cos_i - disc_r.sqrt()
        dl, dm, dn = (eta * dl + k * nx, eta * dm + k * ny,
                      eta * dn + k * nz)
        if i < len(surfs) - 1 and s.thickness != 0:
            tt = (D(s.thickness) - pz) / dn
            px += dl * tt
            py += dm * tt
            opd += n2 * tt
            pz = D(0)
    return px, py, opd


def test_vc2_no_default_combination_is_further_from_the_truth():
    """All four ``(renormalize, sphere_normal)`` combinations against an
    exact-arithmetic oracle of the same geometry.

    The shipped ladder measures the DIFFERENCE between the two renormalise
    modes, which says how far apart they are but not which is closer to the
    truth.  This measures both against a 60-digit ``decimal`` trace.

    Bar: derived from the trace's own arithmetic -- each of the ~4
    roundings per surface costs at most ``eps`` of the ray height, so
    ``8 * n_surfaces * eps * max|h|`` in position.  ``n_surfaces * eps``
    of the OPL scale for the path length.  MEASURED 2026-09-20: every
    combination within 5.2e-18 m and 7.0e-17 m on both builds, and which
    combination is closest flips with the prescription and with the build
    -- which is the claim: the hoist costs no measurable accuracy.
    """
    from lumenairy.glass import get_glass_index
    for surfs in (_doublet(), _ladder(3)):
        idx = [(get_glass_index(s.glass_before, WL),
                get_glass_index(s.glass_after, WL)) for s in surfs]
        rb = _axial_bundle(9, 0.0080)
        truth = [_oracle_trace(surfs, idx, float(rb.x[i]), float(rb.y[i]))
                 for i in range(rb.n_rays)]
        keep = [i for i, t in enumerate(truth) if t is not None]
        assert len(keep) >= 7, len(keep)
        opl_scale = max(abs(float(t[2])) for t in truth if t is not None)
        pos_bar = 8.0 * len(surfs) * EPS * 0.0080
        opl_bar = 8.0 * len(surfs) * EPS * opl_scale
        worst = {}
        for rn in ('surface', 'exit'):
            for sn in ('generic', 'analytic'):
                ir = trace(rb, surfs, WL, output_filter='last',
                           renormalize=rn, sphere_normal=sn).image_rays
                dx = max(abs(float(ir.x[i]) - float(truth[i][0]))
                         for i in keep if bool(ir.alive[i]))
                do = max(abs(float(ir.opd[i]) - float(truth[i][2]))
                         for i in keep if bool(ir.alive[i]))
                worst[f'{rn}/{sn}'] = (dx, do)
                assert dx <= pos_bar, (
                    f'{rn}/{sn} is {dx:.3e} m from the 60-digit truth in '
                    f'position, past the derived {pos_bar:.3e} m envelope '
                    f'({len(surfs)} surfaces).  Measured <= 5.2e-18 m.')
                assert do <= opl_bar, (
                    f'{rn}/{sn} is {do:.3e} m from the truth in OPL, past '
                    f'the derived {opl_bar:.3e} m envelope.')
        # two-sided: the hoist must not be SYSTEMATICALLY worse either
        ex = max(worst['exit/generic'][0], worst['exit/analytic'][0])
        su = max(worst['surface/generic'][0], worst['surface/analytic'][0])
        assert ex <= 4.0 * max(su, EPS * 0.0080), (
            f'renormalize="exit" is now {ex / max(su, 1e-300):.1f}x further '
            f'from the truth than "surface" in position ({worst}).  The '
            f'hoist is supposed to cost no measurable accuracy.')


# ======================================================================
# 6.  The corrected history-drift bound
# ======================================================================

def test_vc2_the_history_drift_exceeds_one_e_minus_15_by_seven_surfaces():
    """``trace``'s docstring promised ``| |d| - 1 | <= 1e-15`` on the
    intermediate history bundles before WP-C2, and now carries an
    ``n_surfaces * eps`` form instead.  Both halves are pinned here: the
    derived envelope HOLDS, and the retired 1e-15 reading is EXCEEDED --
    so a future edit that puts the old number back is caught.

    MEASURED 2026-09-20, both builds identical: 6.7e-16 at 3 surfaces
    rising to 1.67e-15 at 13, i.e. 0.58 to 1.00 of ``n_surfaces * eps``,
    with 1e-15 first exceeded at SEVEN surfaces (1.22e-15).  Under
    ``'surface'`` every history bundle stays at 2.2e-16 on every rung.
    """
    rung = {}
    for n_pairs in (1, 2, 3, 6):
        surfs = _ladder(n_pairs)
        n_s = len(surfs)
        for rn in ('surface', 'exit'):
            res = trace(_axial_bundle(400, 0.0080), surfs, WL,
                        output_filter='all', renormalize=rn,
                        sphere_normal='analytic')
            worst = 0.0
            for b in res.ray_history[:-1]:
                m = np.sqrt(b.L ** 2 + b.M ** 2 + b.N ** 2)[b.alive]
                if m.size:
                    worst = max(worst, float(np.max(np.abs(m - 1.0))))
            fb = res.ray_history[-1]
            mf = np.sqrt(fb.L ** 2 + fb.M ** 2 + fb.N ** 2)[fb.alive]
            rung[(n_s, rn)] = (worst, float(np.max(np.abs(mf - 1.0))))

    for (n_s, rn), (hist, final) in rung.items():
        if rn == 'surface':
            assert hist <= 4.0 * EPS, (
                f"'surface' mode left {hist:.3e} of drift in a history "
                f'bundle at {n_s} surfaces; it rescales at every surface '
                f'so 4 eps is the whole budget.')
        else:
            assert hist <= n_s * EPS, (
                f"the derived history envelope n_surfaces * eps = "
                f'{n_s * EPS:.3e} is exceeded at {n_s} surfaces '
                f'({hist:.3e}).  The docstring states that form.')
        assert final <= 4.0 * EPS, (
            f'the FINAL bundle is not unit at {n_s} surfaces / {rn} '
            f'({final:.3e}); every mode rescales it.')

    long_hist = rung[(13, 'exit')][0]
    assert long_hist > 1e-15, (
        f'the retired "<= 1e-15" history bound is no longer exceeded at 13 '
        f'surfaces ({long_hist:.3e}).  It was a reading from a short '
        f'stack; if the drift has genuinely shrunk, re-derive the '
        f'docstring rather than restoring the old number.')
    assert rung[(7, 'exit')][0] > rung[(3, 'exit')][0], (
        'the exit-mode history drift no longer grows with surface count, '
        'so the n_surfaces * eps form is the wrong shape.')


# ======================================================================
# 7.  The way-back census
# ======================================================================

_TRACERS = {'trace', 'trace_world', 'trace_prescription', 'raytrace_system',
            'trace_jax', 'trace_jax_world'}
_KEYWORDS = ('sphere_normal', 'renormalize')

#: The SIXTEEN CPU-affected exported entry points VERIFY-WP-C2 defect D4
#: found with no way back -- measured 2026-09-20 by an AST walk of the whole
#: package (``validation/probe_verify_c2/vc2_entrypoints.py``), both builds
#: identical, against the six the WP-C2 report named.  Round 2 threaded
#: ``sphere_normal=`` and ``renormalize=`` through every one of them, so
#: this set is now the set that MUST carry both keywords, and the arm below
#: is inverted from the one that pinned the defect.
_SIXTEEN_WITH_A_WAY_BACK = {
    'apply_real_lens_maslov', 'apply_real_lens_traced',
    'caustic_diagnostic', 'eval_image_plane_wfe',
    'fit_canonical_polynomials', 'fit_hf_polynomials',
    'opd_fan_data', 'opd_fan_data_world', 'paraxial_focus_world',
    'plot_lens_layout', 'ray_fan_data', 'ray_fan_data_world',
    'ray_transfer_jacobian', 'raytrace_system', 'through_focus_rms',
    'trace_prescription',
}

#: The four that reach ``trace_jax``, which has NEITHER switch by design
#: (closed-form normal always, no per-surface rescale to hoist), so they
#: are exempt and stay exempt.  (``trace_jax`` itself is a tracer, not a
#: consumer, so it is not in this census.)
_NO_WAY_BACK = {
    'apply_real_lens_maslov_jax', 'apply_real_lens_traced_jax',
    'fit_canonical_polynomials_jax', 'ray_transfer_jacobian_jax',
}


def _census():
    import lumenairy as la
    root = pathlib.Path(la.__file__).parent
    direct = {}
    for f in sorted(root.rglob('*.py')):
        try:
            tree = ast.parse(f.read_text(encoding='utf-8', errors='replace'))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.startswith('_'):
                continue
            # A bare NAME counts as well as a call: ``ray_fan_data``
            # PASSES ``trace`` to a helper rather than calling it, and a
            # census that reads only Call nodes misses exactly that shape.
            # An ATTRIBUTE call counts too (``mod.trace(...)``).
            hit = any(
                (isinstance(sub, ast.Name) and sub.id in _TRACERS)
                or (isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr in _TRACERS)
                for sub in ast.walk(node))
            if hit:
                direct.setdefault(node.name, set()).add(
                    f.relative_to(root).as_posix())
    public = {}
    for mod in ('lumenairy', 'lumenairy.raytrace', 'lumenairy.analysis',
                'lumenairy.io', 'lumenairy.optimize',
                'lumenairy.elements', 'lumenairy.propagators'):
        try:
            m = importlib.import_module(mod)
        except Exception:
            continue
        for n in dir(m):
            if not n.startswith('_'):
                public.setdefault(n, m)
    out = {}
    for name in direct:
        m = public.get(name)
        if m is None:
            continue
        try:
            params = inspect.signature(getattr(m, name)).parameters
        except (TypeError, ValueError):
            continue
        out[name] = [k for k in _KEYWORDS if k in params]
    return out


def test_vc2_the_entry_points_without_a_way_back_are_pinned():
    """The campaign's rule is that every moved public entry point has a
    one-keyword way back.  ``trace`` / ``trace_world`` moved, so every
    exported function that traces internally moved with them -- and can
    only be put back if it forwards the two keywords.

    The WP-C2 report named SIX.  This verification's AST walk found
    twenty exported directly-tracing functions with neither keyword,
    sixteen of them CPU-affected, including the lens propagators
    ``apply_real_lens_traced`` and ``apply_real_lens_maslov``,
    ``ray_transfer_jacobian``, ``eval_image_plane_wfe``,
    ``plot_lens_layout``, ``caustic_diagnostic``, the two ``*_world`` fan
    twins and both polynomial fitters (defect D4).

    ROUND 2 CLOSED IT, and this arm is INVERTED accordingly: all sixteen
    now carry both keywords, and the only entry points that do not are the
    four ``*_jax`` twins, whose tracer has neither switch by design.  Their
    way back was re-measured archive to archive on both builds --
    ``validation/probe_c2_round2/r2_wayback_summary_{win,wsl}.json``, 742
    of 742 arrays byte-identical against a ``git archive 49ddf4bd`` tree in
    its own process, with all sixteen moving at the default.

    The census is pinned both ways: nothing may silently JOIN the exempt
    set (a new entry point that traces without forwarding), and a jax twin
    that GAINS the keywords is a change that should update D4.
    """
    census = _census()
    assert census, 'the AST census found no exported function that traces'
    missing = {n for n, kw in census.items() if len(kw) < 2}
    assert missing == _NO_WAY_BACK, (
        f'the way-back census moved.\n'
        f'  newly WITHOUT a way back: {sorted(missing - _NO_WAY_BACK)}\n'
        f'  now WITH one (fixed):     {sorted(_NO_WAY_BACK - missing)}\n'
        f'Every exported function that traces on the CPU must forward both '
        f'keywords; only the jax twins are exempt.')
    have = {n for n, kw in census.items() if len(kw) == 2}
    assert _SIXTEEN_WITH_A_WAY_BACK <= have, (
        f'these entry points lost their way back again: '
        f'{sorted(_SIXTEEN_WITH_A_WAY_BACK - have)}')
    assert len(_SIXTEEN_WITH_A_WAY_BACK) == 16, (
        'D4 is about sixteen entry points; this set must stay the sixteen '
        'that were measured, or the defect record and the test disagree.')
    # the two tracers themselves are not in the census (their own bodies
    # do not name a tracer), so their way back is asserted directly
    from lumenairy.raytrace.world_trace import trace_world
    for fn in (trace, trace_world):
        p = inspect.signature(fn).parameters
        assert all(k in p for k in _KEYWORDS), (
            f'{fn.__name__} lost one of the two keywords: {sorted(p)}')


# ======================================================================
# 8.  The EDITED_IN_PLACE override is one-sided
# ======================================================================

def _load_reanchor():
    path = REPO / 'scripts' / 'reanchor_citations.py'
    spec = importlib.util.spec_from_file_location('_vc2_ra', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_vc2_the_edited_in_place_override_accepts_a_reverted_default():
    """``scripts/reanchor_citations.py``'s ``EDITED_IN_PLACE`` map answers
    a citation whose CONTENT changed, guarded by "the current line still
    begins with the same leading token".

    That guard is one-sided.  It refuses an unrelated line and an
    out-of-range coordinate -- both asserted here -- but it accepts ANY
    value after the ``=``, including the value the citation says the
    release moved AWAY from.  So a later release that silently reverts the
    default is re-anchored with the 5.49.0 reason text and reported clean.

    Both the BASE and the CURRENT file contents are supplied here, so the
    arm is a pure unit test of the guard and needs neither a git checkout
    nor the base commit to be present.

    This arm PINS the one-sidedness, so it cannot be relied on by
    accident: it fails the day the override starts checking the new
    content, which is the fix VERIFY_WP-C2.md defect D7 asks for.
    """
    ra = _load_reanchor()
    tgt = 'lumenairy/raytrace/trace.py'
    base = 'SYNTHETIC-BASE'
    assert (tgt, 61) in ra.EDITED_IN_PLACE, (
        'the EDITED_IN_PLACE map no longer carries trace.py:61; if the '
        'map has been retired, delete this arm.')
    assert ra.EDITED_IN_PLACE[(tgt, 61)][0] == 61, ra.EDITED_IN_PLACE

    base_lines = (['# pad'] * 60
                  + ["    sphere_normal: str = 'generic',",
                     '    ) -> None:'])
    assert base_lines[60] == "    sphere_normal: str = 'generic',"
    real_lines = ra.lines

    def _try(current):
        def patched(path, rev=None):
            if path != tgt:
                return real_lines(path, rev)
            return base_lines if rev == base else current
        ra.lines = patched
        try:
            return ra._edited_in_place(tgt, 61, base)[0]
        finally:
            ra.lines = real_lines

    shipped = list(base_lines)
    shipped[60] = "    sphere_normal: str = 'analytic',"
    assert _try(shipped) == 61, (
        'the override no longer fires on the change it was written for.')

    # ABUSE: the default reverted in place -- still accepted
    reverted = list(base_lines)
    assert _try(reverted) == 61, (
        'the EDITED_IN_PLACE override now REFUSES a reverted default, '
        'which is the fix VERIFY_WP-C2.md defect D7 asks for.  Delete '
        'this arm and pin the new two-sided behaviour instead.')

    nonsense = list(base_lines)
    nonsense[60] = "    sphere_normal: str = 'not-a-route',"
    assert _try(nonsense) == 61, (
        'the override now checks the value after the "=" -- D7 actioned.')

    # the guard IS two-sided for an unrelated line, and for a short file
    unrelated = list(base_lines)
    unrelated[60] = '    renormalize: str = "exit",'
    assert _try(unrelated) is None, (
        'the leading-token guard stopped refusing an unrelated line; the '
        'override can now re-anchor a citation onto anything.')
    assert _try(base_lines[:30]) is None, (
        'the override stopped refusing an out-of-range new coordinate.')


# ======================================================================
# 9.  The closed form against an EXACT-INPUT oracle
# ======================================================================

def test_vc2_the_closed_form_wins_against_an_exact_input_oracle():
    """The shipped oracle converts its inputs with
    ``Decimal(repr(float(x)))`` -- the shortest ROUND-TRIPPING decimal, not
    the exact value the library was handed.  That injects up to half an ULP
    of INPUT error, which ``sqrt(1 - u)`` amplifies by ``u / (2(1 - u))``.

    This re-measures the comparison with the EXACT conversion
    (``Decimal(float)`` is exact) and pins both halves of the conclusion:
    out to ``h = 0.95 |R|`` the closed form is never worse than the generic
    route by more than one unit, and above it neither dominates.

    MEASURED 2026-09-20 over 2240 points, both builds identical: closed
    form 1.00 against generic 1.75 out to 0.95 |R| (zero points worse by
    more than 1 of 1280), 41.47 against 75.66 over the whole set.  The
    shipped oracle's own input-conversion error, in the same units, is
    1.00 at 0.95 |R| and 41.6 at the clamp.
    """
    getcontext().prec = 60
    unit = 2.0 ** -52
    lo_a = lo_g = hi_a = hi_g = 0.0
    n_worse_low = 0
    n_low = 0
    for mag in (0.0015, 0.0437, 0.1013, 0.7770):
        for sign in (+1.0, -1.0):
            R = sign * mag
            surf = _s(R, 0.01, 'air', 'N-BK7')
            assert _is_pure_spherical(surf)
            for f in (0.0, 0.1, 0.5, 0.9, 0.95, 0.99, 0.999, 0.99994):
                for a in (0.0, 37.0, 163.0):
                    h = f * abs(R)
                    x = np.array([h * math.cos(math.radians(a))])
                    y = np.array([h * math.sin(math.radians(a))])
                    dx, dy, dR = (Decimal(float(x[0])),
                                  Decimal(float(y[0])), Decimal(R))
                    u = (dx * dx + dy * dy) / (dR * dR)
                    ref = (float(-dx / dR), float(-dy / dR),
                           float((Decimal(1) - u).sqrt()))
                    got = {}
                    for key, flag in (('a', True), ('g', False)):
                        c = _surface_normal(x, y, surf, analytic_sphere=flag)
                        got[key] = max(abs(float(c[i][0]) - ref[i])
                                       for i in range(3)) / unit
                    if not (math.isfinite(got['a'])
                            and math.isfinite(got['g'])):
                        continue
                    if f <= 0.95:
                        n_low += 1
                        lo_a = max(lo_a, got['a'])
                        lo_g = max(lo_g, got['g'])
                        if got['a'] > got['g'] + 1.0:
                            n_worse_low += 1
                    else:
                        hi_a = max(hi_a, got['a'])
                        hi_g = max(hi_g, got['g'])
    assert n_low >= 100, n_low
    assert n_worse_low == 0, (
        f'the closed form is worse than the generic route by more than one '
        f'unit at {n_worse_low} of {n_low} points out to h = 0.95 |R|; the '
        f'work package claims zero.')
    assert lo_a <= lo_g, (
        f'the closed form is no longer at least as good as the generic '
        f'route over the working aperture: {lo_a:.3f} against {lo_g:.3f}.')
    assert lo_a <= 4.0, (
        f'the closed form is {lo_a:.3f} units from an EXACT-input 60-digit '
        f'oracle out to 0.95 |R|; measured 1.00 on both builds.')
    assert hi_a > 4.0 and hi_g > 4.0, (
        f'above 0.95 |R| BOTH routes are supposed to leave the 4-unit band '
        f'together (the shared conditioning limit of sqrt(1 - u)); got '
        f'{hi_a:.2f} and {hi_g:.2f}.  If one still holds, the honest half '
        f'of the accuracy claim has changed.')


# ======================================================================
# 10.  The clamp the ledger left alone
# ======================================================================

def test_vc2_the_clamp_kills_through_the_public_trace_on_both_routes():
    """The domain clamp stays, and it kills on BOTH routes -- which is what
    makes the rim band one ULP wide rather than an annulus.

    Located on the running build rather than read off the literal, and
    asserted through the PUBLIC ``trace`` so a change to the expression
    (not only to the constant) is caught.
    """
    R = 0.0125
    surfs = [_s(R, 0.010, 'air', 'N-BK7', np.inf),
             _s(np.inf, 0.020, 'N-BK7', 'air', np.inf)]
    thr = math.sqrt(CLAMP)
    hs = np.array([0.5, 0.9, 0.99, thr * (1 - 1e-9), thr * (1 + 1e-9),
                   0.99999]) * R
    n = len(hs)
    for rn, sn in (('surface', 'generic'), ('exit', 'analytic')):
        rb = RayBundle(x=hs.copy(), y=np.zeros(n), z=np.zeros(n),
                       L=np.zeros(n), M=np.zeros(n), N=np.ones(n),
                       wavelength=WL, alive=np.ones(n, dtype=bool),
                       opd=np.zeros(n))
        ir = trace(rb, surfs, WL, output_filter='last',
                   renormalize=rn, sphere_normal=sn).image_rays
        ec = np.asarray(ir.error_code)
        assert list(ec[-2:]) == [4, 4], (
            f'{rn}/{sn}: a ray past the {CLAMP} domain clamp is no longer '
            f'killed RAY_NAN ({list(ec)}).')
        assert int(ec[3]) != 4, (
            f'{rn}/{sn}: a ray just INSIDE the clamp is now killed '
            f'RAY_NAN, so the clamp has moved inward ({list(ec)}).')


# ======================================================================
# 11.  The private-layer docstrings the flip left behind
# ======================================================================

def test_vc2_the_private_docstrings_still_describe_the_old_defaults():
    """WP-C2 rewrote the two PUBLIC docstrings and left the private ones.

    Four of their sentences became false the moment the default moved, and
    the first is the one a reader reaches for to answer exactly the question
    the Migration note raises -- ``_sphere_normal``'s own account of the rim
    band said "the shipped default is the generic route on both sides",
    which is the opposite of what the release did (defect D11, the only P1).

    ROUND 2 CLOSED IT and this arm is INVERTED: each of the four sentences
    is now asserted ABSENT and its replacement asserted PRESENT, so the arm
    cannot be satisfied by deleting the paragraph either.  The shipped WP-C2
    file carries the same census with the fail-before demonstration against
    a ``git archive eadc67ba`` tree (4 of 4 stale sentences, both builds).
    """
    def _flat(path):
        return re.sub(r'\s+', ' ', (REPO / 'lumenairy' / 'raytrace'
                                    / path).read_text(encoding='cp1252'))

    surface_src = _flat('surface.py')
    isect_src = _flat('intersection.py')
    trace_src = _flat('trace.py')

    # PREMISE: the two PUBLIC docstrings DO say the defaults moved, so a
    # red here cannot mean "the whole work package was reverted".
    assert 'THIS DEFAULT MOVED' in trace_src, (
        'premise: trace.py no longer says the defaults moved, so this arm '
        'has nothing to compare the private docstrings against.')

    stale = {
        'surface.py::_sphere_normal': (
            surface_src,
            'That band is reachable only under ``sphere_normal='
            "'analytic'``; the shipped default is the generic route on "
            'both sides.'),
        'surface.py::_surface_normal': (
            surface_src,
            'The default is the generic sag-derivative route, which is '
            'the arithmetic every caller has always got'),
        'intersection.py::_intersect_surface': (
            isect_src,
            'It is opt-in because it differs in the last bit from the '
            'sag-derivative route every caller has been getting.'),
        'intersection.py::_refract': (
            isect_src,
            "``'generic'`` is the sag-derivative dispatch every caller has "
            'always used'),
    }
    still_there = [k for k, (src_, txt) in stale.items()
                   if re.sub(r'\s+', ' ', txt) in src_]
    assert not still_there, (
        f'{sorted(still_there)} still claim the generic sag-derivative '
        f'route is what ships.  The shipped default is the CLOSED FORM; '
        f'VERIFY_WP-C2.md defect D11 carries the replacement wording.')

    corrected = {
        'surface.py::_sphere_normal': (
            surface_src,
            'which is the SHIPPED DEFAULT of :func:`trace.trace` / '
            ':func:`world_trace.trace_world` since WP-C2 moved it'),
        'surface.py::_surface_normal': (
            surface_src,
            'THIS PRIVATE DEFAULT DID NOT MOVE when WP-C2 moved the two '
            'public ones'),
        'intersection.py::_intersect_surface': (
            isect_src,
            'It is the DEFAULT for :func:`trace.trace` / ``trace_world`` '
            'since WP-C2 moved it'),
        'intersection.py::_refract': (
            isect_src,
            'is the sag-derivative dispatch and the default of THIS '
            'PRIVATE helper'),
    }
    missing = [k for k, (src_, txt) in corrected.items()
               if re.sub(r'\s+', ' ', txt) not in src_]
    assert not missing, (
        f'{sorted(missing)} lost their corrected wording as well as the '
        f'stale wording.  D11 asks for a REWRITE, not a deletion: a reader '
        f'who opens _sphere_normal must still be told that the rim band is '
        f'reachable at the shipped default.')

    # and no shipped source line may name a forward version while saying so
    for src_ in (surface_src, isect_src):
        assert '5.49.0' not in src_, (
            'a private docstring now names 5.49.0; the version-narrative '
            'gate (test_public_api.py) forbids a forward token in '
            'lumenairy/ -- describe the change and let the CHANGELOG carry '
            'the number.')
