"""WP-B12: the FGA reference plane -- ``ray_transfer_jacobian`` grew an explicit
output reference plane, and the four ``fga.py`` sites that add an image-side leg
ask for the exit-VERTEX one.

Added 2026-09-14.  Before this package the two differential-transfer primitives
returned the base-ray state and the Jacobian ON the last surface
(``z = sag(rho)``) while ``_fga_core``, ``_fga_coarse``, the coarse trace and
``_caustic_zone`` added ``z_image`` as if that state were on the exit-vertex
plane, so every beamlet carried a spurious ``k * sag(rho)`` of phase on any
prescription whose last surface is curved.  The tests below assert the
INVARIANTS the repair establishes -- what the two reference planes are, that
they differ by exactly the last surface's sag along the ray, that a flat last
surface makes the projection the identity bit for bit, and that FGA reproduces
an independent diffraction oracle at a caustic -- never the readings the repair
happened to produce.

Every oracle here is written in this file and shares no code with
``lumenairy.propagators.fga`` or ``lumenairy.raytrace``: the conic intersection
is a Newton solve on the implicit sag (the closed-form quadric of
``test_audit2609_b7b_caustic_routing.py`` is a DIFFERENT algorithm, deliberately),
refraction is vector Snell with the normal oriented against the incident ray,
and the diffraction reference is a brute-force Rayleigh-Sommerfeld-I sum whose
own convergence is measured inside the test that uses it.  Every bar is derived
from a quantity the running build measures, with the gap to the signal stated.
No wall-clock assertion anywhere.
"""
from __future__ import annotations

import copy
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy import raytrace as rt
from lumenairy.propagators import fga
from lumenairy.raytrace import surfaces_from_prescription
from lumenairy.raytrace.differential import (
    _exit_direction_sign,
    _last_surface_sag_vanishes,
    ray_transfer_jacobian,
    ray_transfer_jacobian_analytic,
)

# Model glasses for THIS module only: registered and removed by
# tests/conftest.py::_module_glass_registry_guard.
MODULE_GLASSES = {'_B12_N1p6': lambda wl: 1.6000}


# ===========================================================================
# The oracle -- an exact conic / even-aspheric meridional trace and a
# brute-force Rayleigh-Sommerfeld-I sum.  Nothing from the library.
# ===========================================================================
def _sag_of(h, R, k=0.0, asph=None):
    h2 = np.asarray(h, float) ** 2
    if np.isinf(R):
        z = np.zeros_like(h2)
    else:
        c = 1.0 / R
        z = c * h2 / (1.0 + np.sqrt(np.maximum(1.0 - (1.0 + k) * c * c * h2,
                                               0.0)))
    for p, a in (asph or {}).items():
        z = z + a * h2 ** (p // 2)
    return z


def _dsag_of(h, R, k=0.0, asph=None):
    h = np.asarray(h, float)
    h2 = h * h
    if np.isinf(R):
        d = np.zeros_like(h)
    else:
        c = 1.0 / R
        d = c * h / np.maximum(
            np.sqrt(np.maximum(1.0 - (1.0 + k) * c * c * h2, 0.0)), 1e-300)
    for p, a in (asph or {}).items():
        m = p // 2
        d = d + a * m * h2 ** (m - 1) * 2.0 * h
    return d


def _oracle_trace(h0, surfs, n_air=1.0):
    """Exact meridional trace of collimated rays at height ``h0``.

    ``surfs`` = list of ``(z_vertex, R, conic, asph, n_after)``.  Returns the
    state ON the last surface and on its vertex plane:
    ``(x_s, u_s, opl_s, x_v, u_v, opl_v)`` with ``u = ux/uz``.
    """
    x = np.asarray(h0, float).copy()
    z = np.zeros_like(x)
    ux = np.zeros_like(x)
    uz = np.ones_like(x)
    opl = np.zeros_like(x)
    n_cur = n_air
    for zv, R, kk, asph, n_after in surfs:
        t = (zv - z) / uz                       # seed: the flat crossing
        for _ in range(60):
            xx, zz = x + t * ux, z + t * uz
            F = (zz - zv) - _sag_of(np.abs(xx), R, kk, asph)
            dF = uz - _dsag_of(np.abs(xx), R, kk, asph) * np.sign(xx) * ux
            step = F / np.where(np.abs(dF) < 1e-300, 1e-300, dF)
            t = t - step
            if np.max(np.abs(step)) < 1e-16:
                break
        x, z = x + t * ux, z + t * uz
        assert np.max(np.abs((z - zv) - _sag_of(np.abs(x), R, kk, asph))) < 1e-14
        opl = opl + n_cur * t
        gx = -_dsag_of(np.abs(x), R, kk, asph) * np.sign(x)
        gz = np.ones_like(gx)
        g = np.sqrt(gx * gx + gz * gz)
        nx, nz = gx / g, gz / g
        ci = -(nx * ux + nz * uz)
        nx, nz = np.where(ci < 0, -nx, nx), np.where(ci < 0, -nz, nz)
        ci = np.abs(ci)
        mu = n_cur / n_after
        disc = 1.0 - mu * mu * (1.0 - ci * ci)
        assert np.all(disc > 0.0), 'premise: no total internal reflection'
        f = mu * ci - np.sqrt(disc)
        ux, uz = mu * ux + f * nx, mu * uz + f * nz
        n_cur = n_after
    x_s, u_s, opl_s = x.copy(), ux / uz, opl.copy()
    t = (surfs[-1][0] - z) / uz
    return x_s, u_s, opl_s, x + t * ux, ux / uz, opl + n_cur * t


def _rs_radial(x_v, opl_v, wgt, rho, z, lam, n_phi=384):
    """Brute-force Rayleigh-Sommerfeld-I sum over (exit ray) x (azimuth)."""
    k = 2.0 * np.pi / lam
    phi = (np.arange(n_phi) + 0.5) * (2.0 * np.pi / n_phi)
    cph = np.cos(phi)
    pre = np.exp(1j * k * opl_v) * wgt
    rho = np.asarray(rho, float)
    out = np.empty(rho.size, complex)
    for i0 in range(0, rho.size, 16):
        rr = rho[i0:i0 + 16][:, None, None]
        r2 = (z * z + rr * rr + x_v[None, :, None] ** 2
              - 2.0 * rr * x_v[None, :, None] * cph[None, None, :])
        out[i0:i0 + 16] = ((np.exp(1j * k * np.sqrt(r2)) * (z / r2))
                           * pre[None, :, None]).sum(axis=(1, 2))
    return out * (2.0 * np.pi / n_phi) / (1j * lam)


def _fid(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(abs(np.vdot(a, b)) ** 2
                 / (np.vdot(a, a).real * np.vdot(b, b).real))


# ===========================================================================
# Fixtures -- a different glass and wavelength from every other FGA test file.
# ===========================================================================
_LAM = 1.064e-6
_GLASS = 'N-BAF10'
_R, _T, _SEMI = 2.10e-3, 0.70e-3, 0.20e-3
_N, _DX, _W0 = 192, 2.55e-6, 105e-6


def _biconvex(R=_R, conic=0.0, asph=None, glass=_GLASS, t=_T, semi=_SEMI,
              R2=None):
    s0 = {'radius': R, 'conic': conic, 'thickness': t, 'glass_before': 'air',
          'glass_after': glass, 'semi_diameter': semi}
    s1 = {'radius': (-R if R2 is None else R2), 'conic': conic,
          'thickness': 0.0, 'glass_before': glass, 'glass_after': 'air',
          'semi_diameter': semi}
    if asph:
        s1['aspheric_coeffs'] = dict(asph)
    return {'name': 'b12', 'aperture_diameter': 2 * semi,
            'surfaces': [s0, s1], 'thicknesses': [t], 'stop_index': 0}


def _flat_last(R=1.45e-3, glass='N-LASF9', t=0.55e-3, semi=_SEMI):
    """Plano-convex with the CURVED side first: the last surface is flat."""
    return {'name': 'b12flat', 'aperture_diameter': 2 * semi,
            'surfaces': [
                {'radius': R, 'conic': 0.0, 'thickness': t,
                 'glass_before': 'air', 'glass_after': glass,
                 'semi_diameter': semi},
                {'radius': np.inf, 'conic': 0.0, 'thickness': 0.0,
                 'glass_before': glass, 'glass_after': 'air',
                 'semi_diameter': semi}],
            'thicknesses': [t], 'stop_index': 0}


def _surfs(presc):
    """The surface list FGA builds: a copy with the last transfer zeroed."""
    s = [copy.copy(x) for x in surfaces_from_prescription(presc)]
    s[-1].thickness = 0.0
    return s


def _fan(presc, n=121, semi=_SEMI, lam=_LAM):
    """A collimated meridional fan and the library's own two exit states."""
    surfs = _surfs(presc)
    h = np.linspace(semi / (2 * n), semi * (1.0 - 1.0 / (2 * n)), n)
    z = np.zeros(n)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        bundle = rt.RayBundle(x=h.copy(), y=z.copy(), z=z.copy(), L=z.copy(),
                              M=z.copy(), N=np.ones(n), wavelength=lam,
                              alive=np.ones(n, bool), opd=z.copy())
        res = rt.trace(bundle, surfs, lam)
        img, ex = res.image_rays, res.at_exit_vertex()
    return surfs, h, z, img, ex


def _both(presc, h, z, surfs, lam=_LAM, **kw):
    """Both backends, both reference planes."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = {}
        for nm, fn in (('fd', ray_transfer_jacobian),
                       ('analytic', ray_transfer_jacobian_analytic)):
            for ref in ('surface', 'exit_vertex'):
                out[(nm, ref)] = fn(h.copy(), z.copy(), z.copy(), z.copy(),
                                    surfs, lam, reference=ref, **kw)
    return out


# ===========================================================================
# 1. What the two reference planes ARE
# ===========================================================================
def test_the_two_reference_planes_are_the_trace_s_own_two_planes():
    """DECISION: ``reference='surface'`` returns the plane
    ``TraceResult.image_rays`` reports and ``reference='exit_vertex'`` returns
    the plane ``TraceResult.at_exit_vertex()`` reports -- for BOTH backends.

    That is the whole contract WP-B12 adds, and it is asserted against the
    library's production tracer rather than against a formula, because
    ``at_exit_vertex`` is the package's single supported definition of the
    vertex plane (``raytrace/exit_vertex.py``).

    Bars.  MEASURED here on the running build and printed in the assertion
    message; on the reference run (2026-09-14, py3.14 / numpy 2.4.4 and
    py3.12 / numpy 2.4.6, identical) the agreement is 5.4e-20 m of height and
    1.7e-18 m of optical path.  The bar is 1e-14 m -- four decades above that
    round-off and SEVEN decades below the quantity being resolved (the sag
    separating the two planes on this fixture is 8.4e-07 m of height and
    7.2e-06 m of path, asserted below), so no build can move a ray across it.
    """
    presc = _biconvex()
    surfs, h, z, img, ex = _fan(presc)
    ok = np.asarray(ex.alive, bool)
    assert ok.sum() > h.size // 2
    d = _both(presc, h, z, surfs)
    for nm in ('fd', 'analytic'):
        s, v = d[(nm, 'surface')], d[(nm, 'exit_vertex')]
        e_s = max(float(np.abs(np.asarray(s.x)[ok] - np.asarray(img.x)[ok]).max()),
                  float(np.abs(np.asarray(s.opd)[ok]
                               - np.asarray(img.opd)[ok]).max()))
        e_v = max(float(np.abs(np.asarray(v.x)[ok] - np.asarray(ex.x)[ok]).max()),
                  float(np.abs(np.asarray(v.opd)[ok]
                               - np.asarray(ex.opd)[ok]).max()))
        assert e_s < 1e-14, f'{nm} surface plane: {e_s:.3e} m'
        assert e_v < 1e-14, f'{nm} exit-vertex plane: {e_v:.3e} m'
        # the slopes are a transfer invariant: untouched, bit for bit
        assert np.array_equal(np.asarray(s.ux), np.asarray(v.ux)), nm
        assert np.array_equal(np.asarray(s.uy), np.asarray(v.uy)), nm


def test_the_two_planes_differ_by_the_last_surface_s_sag_along_the_ray():
    """DECISION: the separation between the two planes is exactly the straight
    transfer ``x - sag*ux``, ``opd - n_exit*sag*sec`` -- computed here from this
    file's OWN sag formula and this file's own trace, not from the library's.

    The premise that there is anything to measure is asserted first: on this
    N-BAF10 R = +/-2.10 mm biconvex the last surface's sag at the marginal ray
    is 6.7 waves of optical path at 1.064 um (MEASURED 2026-09-14; asserted
    here to be above 1 wave, so a build that somehow flattened the surface
    would fail loudly rather than pass a vacuous comparison).

    Bar 1e-14 m on the difference, as in the test above: seven decades under
    the 7.2e-06 m separation it is resolving.
    """
    presc = _biconvex()
    surfs, h, z, img, ex = _fan(presc)
    ok = np.asarray(ex.alive, bool)
    d = _both(presc, h, z, surfs)
    s = d[('fd', 'surface')]
    sag = _sag_of(np.hypot(np.asarray(s.x), np.asarray(s.y)), -_R)
    sec = np.sqrt(1.0 + np.asarray(s.ux) ** 2 + np.asarray(s.uy) ** 2)
    # premise: the two planes really are separated on this optic
    waves = float(np.abs(sag * sec)[ok].max() / _LAM)
    assert waves > 1.0, f'premise: sag at the rim is only {waves:.3f} waves'
    for nm in ('fd', 'analytic'):
        v = d[(nm, 'exit_vertex')]
        ds = d[(nm, 'surface')]
        ex_x = np.asarray(ds.x) - sag * np.asarray(ds.ux)
        ex_o = np.asarray(ds.opd) - 1.0 * sag * sec       # n_exit = air = 1
        assert float(np.abs(np.asarray(v.x)[ok] - ex_x[ok]).max()) < 1e-14, nm
        assert float(np.abs(np.asarray(v.opd)[ok] - ex_o[ok]).max()) < 1e-14, nm
    # ... and the library's own two states differ by the same amount
    assert float(np.abs((np.asarray(img.x) - np.asarray(ex.x))[ok]
                        - (sag * np.asarray(s.ux))[ok]).max()) < 1e-14


def test_the_oracle_reproduces_both_library_planes():
    """Premise for the diffraction test below: this file's own conic trace
    reproduces BOTH of the library's exit states.  They share no code -- the
    oracle solves the intersection by Newton on the implicit sag and refracts
    with the normal oriented against the ray.

    MEASURED (2026-09-14): 5.4e-20 m of height, 8.3e-17 of slope and 1.4e-18 m
    of path on both planes, on both builds.  Bars 1e-12 m / 1e-10 / 1e-12 m --
    decades above that and decades below anything that could move a member
    score (the members differ by MICRONS).

    The slope-vs-direction-cosine trap is checked too: if the comparison had
    silently been cosine-against-cosine it would read 0, and it reads 8.0e-04.
    """
    presc = _biconvex()
    surfs, h, z, img, ex = _fan(presc)
    ok = np.asarray(ex.alive, bool)
    ng = float(la.get_glass_index(_GLASS, _LAM))
    o = _oracle_trace(h, [(0.0, _R, 0.0, None, ng),
                          (_T, -_R, 0.0, None, 1.0)])
    x_s, u_s, opl_s, x_v, u_v, opl_v = o
    for lib, xo, uo, oo in ((img, x_s, u_s, opl_s), (ex, x_v, u_v, opl_v)):
        assert np.abs(np.asarray(lib.x)[ok] - xo[ok]).max() < 1e-12
        assert np.abs((np.asarray(lib.L) / np.asarray(lib.N))[ok]
                      - uo[ok]).max() < 1e-10
        assert np.abs(np.asarray(lib.opd)[ok] - oo[ok]).max() < 1e-12
    trap = float(np.abs((np.asarray(ex.L) / np.asarray(ex.N))[ok]
                        - np.asarray(ex.L)[ok]).max())
    assert trap > 1e-6, f'slope-vs-cosine check is vacuous: {trap:.3e}'


# ===========================================================================
# 2. The flat-last-surface identity
# ===========================================================================
def test_a_flat_last_surface_makes_the_projection_the_identity_bit_for_bit():
    """DECISION: when the last surface's sag is identically zero the
    exit-vertex projection returns the input object unchanged, so every FGA
    field on a flat-last-surface prescription is bit-for-bit what the previous
    release produced.

    This is the structural guarantee behind the Migration note, so it is
    asserted as bit-identity (``np.array_equal``) on the state AND the 4x4
    Jacobian, on both backends -- not as a tolerance.  The predicate is
    asserted two-sided on the same call, so a predicate that answered "flat"
    for everything could not pass.
    """
    flat = _flat_last()
    surfs, h, z, img, ex = _fan(flat)
    assert _last_surface_sag_vanishes(surfs[-1]) is True
    d = _both(flat, h, z, surfs)
    for nm in ('fd', 'analytic'):
        s, v = d[(nm, 'surface')], d[(nm, 'exit_vertex')]
        assert np.array_equal(np.asarray(s.x), np.asarray(v.x)), nm
        assert np.array_equal(np.asarray(s.y), np.asarray(v.y)), nm
        assert np.array_equal(np.asarray(s.opd), np.asarray(v.opd)), nm
        assert np.array_equal(np.asarray(s.jacobian),
                              np.asarray(v.jacobian)), nm
    # two-sided: every curved / departing last surface is NOT flat
    for presc in (_biconvex(), _biconvex(conic=-1.0),
                  _biconvex(asph={4: 4.0e8}),
                  _biconvex(R2=9.0e-3)):
        assert _last_surface_sag_vanishes(_surfs(presc)[-1]) is False


def test_the_flat_control_field_is_bit_identical_through_the_public_api():
    """DECISION: ``apply_real_lens_fga`` on a flat-last-surface prescription
    returns exactly the field the un-projected code path returns -- the same
    bytes, not the same to a tolerance.

    Measured by running the public entry point twice with the differential
    primitives forced back onto ``reference='surface'`` for one of the runs.
    On a flat last surface the two code paths must be indistinguishable; on the
    curved fixture the SAME comparison must differ, which is asserted here too,
    so the test cannot pass by the forcing failing to take effect.
    """
    from lumenairy.raytrace import differential as D

    def _forced(presc, force, N=128, dx=3.0e-6, w0=60e-6):
        xs = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(xs, xs)
        E = np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)
        saved = (D.ray_transfer_jacobian, D.ray_transfer_jacobian_analytic)
        try:
            if force:
                D.ray_transfer_jacobian = _pin_surface(saved[0])
                D.ray_transfer_jacobian_analytic = _pin_surface(saved[1])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                return np.asarray(la.apply_real_lens_fga(
                    E, prescription=presc, wavelength=_LAM, dx=dx,
                    output_plane_distance=0.0))
        finally:
            D.ray_transfer_jacobian, D.ray_transfer_jacobian_analytic = saved

    flat = _flat_last()
    assert np.array_equal(_forced(flat, force=False),
                          _forced(flat, force=True))
    curved = _biconvex()
    assert not np.array_equal(_forced(curved, force=False),
                              _forced(curved, force=True))


def _pin_surface(base):
    def wrapped(*a, **kw):
        kw['reference'] = 'surface'
        return base(*a, **kw)
    return wrapped


# ===========================================================================
# 3. The Jacobian really is the derivative of the projected map
# ===========================================================================
def test_the_projected_jacobian_is_the_derivative_of_the_projected_map():
    """DECISION: ``reference='exit_vertex'`` returns the Jacobian of the map
    (input state -> EXIT-VERTEX-PLANE state), not the last-surface Jacobian
    with a relabelled name.

    The reference derivative is a central finite difference of the
    exit-vertex-plane STATE, taken here by calling the primitive four more
    times with perturbed inputs -- an independent construction of the same
    derivative, with its own step ladder.

    Bar: the FD reference's own truncation, measured by halving the step.  The
    projected Jacobian must match it to a tolerance derived from that ladder
    (10x the step-halving change), and the UN-projected Jacobian must NOT --
    the second half is what makes the first non-vacuous.  MEASURED
    (2026-09-15, py3.14/numpy 2.4.4 and py3.12/numpy 2.4.6, bit identical;
    VERIFY-WP-B12 D-1 re-recorded the 2026-09-14 figures, which did not
    reproduce): projected residual 1.4648e-07, un-projected 1.4393e-02,
    ladder 3.5014e-07 (bar 3.5014e-06).
    """
    presc = _biconvex()
    surfs = _surfs(presc)
    n = 65
    h = np.linspace(_SEMI / (2 * n), _SEMI * 0.98, n)
    z = np.zeros(n)

    def _state(xx, yy, uxx, uyy):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            d = ray_transfer_jacobian(xx, yy, uxx, uyy, surfs, _LAM,
                                      reference='exit_vertex')
        return np.stack([d.x, d.y, d.ux, d.uy], axis=-1)

    def _fd_jac(step_pos, step_slope):
        cols = []
        for dim, st in enumerate((step_pos, step_pos, step_slope, step_slope)):
            args_p = [h.copy(), z.copy(), z.copy(), z.copy()]
            args_m = [h.copy(), z.copy(), z.copy(), z.copy()]
            args_p[dim] = args_p[dim] + st
            args_m[dim] = args_m[dim] - st
            cols.append((_state(*args_p) - _state(*args_m)) / (2.0 * st))
        return np.stack(cols, axis=-1)                    # (n, 4, 4)

    j_ref = _fd_jac(1e-6, 5e-5)
    j_half = _fd_jac(5e-7, 2.5e-5)
    # Per-ROW scale: the four rows of a ray-transfer Jacobian carry different
    # units (the C block is 1/f ~ 1e3 /m where A is O(1)), so one global max
    # would hide a change in the position rows behind the slope rows.
    scale = np.abs(j_ref).max(axis=(0, 2))[None, :, None]

    def _rel(a, b):
        return float((np.abs(a - b) / scale).max())

    ladder = _rel(j_ref, j_half)
    assert ladder < 1e-4, f'FD reference not in its smooth regime: {ladder:.2e}'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        j_v = ray_transfer_jacobian(h.copy(), z.copy(), z.copy(), z.copy(),
                                    surfs, _LAM,
                                    reference='exit_vertex').jacobian
        j_s = ray_transfer_jacobian(h.copy(), z.copy(), z.copy(), z.copy(),
                                    surfs, _LAM).jacobian
    r_v = _rel(j_v, j_ref)
    r_s = _rel(j_s, j_ref)
    bar = max(10.0 * ladder, 1e-7)
    assert r_v < bar, f'projected {r_v:.3e} vs bar {bar:.3e}'
    assert r_s > 10.0 * bar, (
        f'un-projected {r_s:.3e} is not distinguishable from the projected '
        f'{r_v:.3e}: the comparison is vacuous on this build')


def test_the_two_backends_agree_on_both_reference_planes_to_one_floor():
    """DECISION: the projection is backend-neutral.  The finite-difference and
    the analytic primitives already agree to the FD truncation floor on the
    last-surface plane; asking both for the exit-vertex plane must not open a
    gap between them, because both route through the one shared projection.

    The claim is a RATIO, so it carries no per-build constant: the two
    discrepancies must agree within a factor of two.  MEASURED (2026-09-15,
    both builds, bit identical): 4.68293e-07 on both planes, ratio 1.000000.

    RESTATED 2026-09-15 (VERIFY-WP-B12 D-3): the mask is the intersection of
    the two backends' own ``alive`` flags, not the base ray's.  The FD
    backend additionally kills a ray whose nine-ray companion bundle
    vignettes, and the outermost ray of this fan is one; masking on the base
    ray alone read that dead companion (97.05, both builds) and the recorded
    4.696e-07 was the reading with it removed.
    """
    presc = _biconvex()
    surfs, h, z, _img, _ex = _fan(presc)
    d = _both(presc, h, z, surfs)
    r = {}
    for ref in ('surface', 'exit_vertex'):
        ok = (np.asarray(d[('fd', ref)].alive, bool)
              & np.asarray(d[('analytic', ref)].alive, bool))
        assert ok.sum() >= 2, ok.sum()
        a = np.asarray(d[('fd', ref)].jacobian)[ok]
        b = np.asarray(d[('analytic', ref)].jacobian)[ok]
        r[ref] = float(np.abs(a - b).max()) / float(np.abs(b).max())
    assert r['surface'] > 0.0 and r['exit_vertex'] > 0.0, r
    ratio = r['exit_vertex'] / r['surface']
    assert 0.5 < ratio < 2.0, f'{r} ratio {ratio:.6f}'


def test_per_surface_projection_moves_only_the_last_local_transfer():
    """DECISION: with ``per_surface=True`` the projection re-references only the
    LAST local transfer -- the earlier ones end on their own surfaces and are
    returned bit for bit -- and the product of the locals still equals the
    composite Jacobian of the same reference plane.
    """
    presc = _biconvex()
    surfs, h, z, _img, _ex = _fan(presc, n=101)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ps = ray_transfer_jacobian(h.copy(), z.copy(), z.copy(), z.copy(),
                                   surfs, _LAM, per_surface=True)
        pv = ray_transfer_jacobian(h.copy(), z.copy(), z.copy(), z.copy(),
                                   surfs, _LAM, per_surface=True,
                                   reference='exit_vertex')
        cv = ray_transfer_jacobian(h.copy(), z.copy(), z.copy(), z.copy(),
                                   surfs, _LAM, reference='exit_vertex')
    assert np.asarray(ps.jacobian).shape[0] == len(surfs)
    assert np.array_equal(np.asarray(ps.jacobian)[:-1],
                          np.asarray(pv.jacobian)[:-1])
    assert not np.array_equal(np.asarray(ps.jacobian)[-1],
                              np.asarray(pv.jacobian)[-1])
    prod = np.asarray(pv.jacobian)[0]
    for k in range(1, np.asarray(pv.jacobian).shape[0]):
        prod = np.asarray(pv.jacobian)[k] @ prod
    # the per-surface product and the composite are two routes to one operator;
    # the gap is the FD inverse-of-cumulative round-off, measured here.
    rel = (float(np.abs(prod - np.asarray(cv.jacobian)).max())
           / float(np.abs(np.asarray(cv.jacobian)).max()))
    assert rel < 1e-6, f'{rel:.3e}'


# ===========================================================================
# 4. Surface classes the conic formula alone would get wrong
# ===========================================================================
def test_an_aspheric_last_surface_projects_with_its_polynomial_departure():
    """DECISION: the projection uses the surface's FULL sag, not its conic base.

    An even-aspheric departure on the last surface changes where the vertex
    plane is along each ray; a projection that used only ``conic_sag`` of the
    base radius would miss it.  Asserted against ``at_exit_vertex()``, with the
    conic-only projection computed here and shown to be measurably different --
    so the test cannot pass on a build that dropped the polynomial.
    """
    presc = _biconvex(asph={4: 4.0e8, 6: -8.0e17})
    surfs, h, z, _img, ex = _fan(presc)
    ok = np.asarray(ex.alive, bool)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        v = ray_transfer_jacobian(h.copy(), z.copy(), z.copy(), z.copy(),
                                  surfs, _LAM, reference='exit_vertex')
        s = ray_transfer_jacobian(h.copy(), z.copy(), z.copy(), z.copy(),
                                  surfs, _LAM)
    assert float(np.abs(np.asarray(v.opd)[ok]
                        - np.asarray(ex.opd)[ok]).max()) < 1e-12
    r = np.hypot(np.asarray(s.x), np.asarray(s.y))
    sec = np.sqrt(1.0 + np.asarray(s.ux) ** 2 + np.asarray(s.uy) ** 2)
    conic_only = np.asarray(s.opd) - _sag_of(r, -_R) * sec
    gap = float(np.abs(conic_only[ok] - np.asarray(v.opd)[ok]).max())
    assert gap > 1e-9, (
        f'the aspheric departure is not resolvable on this fixture: {gap:.3e}')


def test_a_mirror_terminated_prescription_projects_with_the_right_sign():
    """DECISION: a reflected exit ray travels back along -z, and the signed
    vertex-plane transfer follows it.  The unreduced slope ``u = L/N`` does not
    record that sign, so the primitive recovers it from the prescription
    (``_exit_direction_sign``); asserted here against ``at_exit_vertex()``,
    which reads the true ``N``.

    Two-sided: the sign helper reads +1 for the transmissive fixture and -1 for
    this one, on the same call.
    """
    semi = 0.30e-3
    presc = {'name': 'b12mirror', 'aperture_diameter': 2 * semi,
             'surfaces': [
                 {'radius': np.inf, 'conic': 0.0, 'thickness': 1.0e-3,
                  'glass_before': 'air', 'glass_after': 'air',
                  'semi_diameter': semi},
                 {'radius': -6.0e-3, 'conic': 0.0, 'thickness': 0.0,
                  'glass_before': 'air', 'glass_after': 'air',
                  'is_mirror': True, 'semi_diameter': semi}],
             'thicknesses': [1.0e-3], 'stop_index': 0}
    surfs, h, z, _img, ex = _fan(presc, n=201, semi=semi)
    assert _exit_direction_sign(surfs) == -1.0
    assert _exit_direction_sign(_surfs(_biconvex())) == 1.0
    ok = np.asarray(ex.alive, bool)
    assert ok.sum() > 100
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        v = ray_transfer_jacobian(h.copy(), z.copy(), z.copy(), z.copy(),
                                  surfs, _LAM, reference='exit_vertex')
    assert float(np.abs(np.asarray(v.x)[ok] - np.asarray(ex.x)[ok]).max()) < 1e-12
    assert float(np.abs(np.asarray(v.opd)[ok]
                        - np.asarray(ex.opd)[ok]).max()) < 1e-12


def test_reference_is_validated_on_both_primitives():
    """A misspelled reference plane is a ValueError naming the function and
    both legal values -- CONVENTIONS.md sec. 2 -- not a silent last-surface
    answer."""
    surfs = _surfs(_biconvex())
    h = np.zeros(3)
    for fn, name in ((ray_transfer_jacobian, 'ray_transfer_jacobian'),
                     (ray_transfer_jacobian_analytic,
                      'ray_transfer_jacobian_analytic')):
        with pytest.raises(ValueError) as e:
            fn(h, h, h, h, surfs, _LAM, reference='vertex')
        assert str(e.value).startswith(f'{name}: reference must be')
        assert "'exit_vertex'" in str(e.value)


# ===========================================================================
# 5. What it buys: the FGA field and the caustic zone
# ===========================================================================
def test_the_caustic_zone_is_measured_from_the_exit_vertex_plane():
    """DECISION: ``_caustic_zone`` returns axial crossings measured from the
    last surface's VERTEX plane -- the frame ``output_plane_distance`` is in --
    and not from the last surface.

    The reference is the same estimator (equally-radius-spaced meridional fan
    over the field's illuminated support, 5th-95th percentile of ``-x/u``)
    evaluated on this file's own trace.  Bar: 1e-4 relative, derived from the
    fan's own discretisation -- while the shift the projection removes is
    4.0e-03 relative on this fixture (MEASURED 2026-09-14), i.e. 40x the bar.

    Two-sided: on the flat-last-surface control the zone must be unchanged to
    the same bar, because there is no sag to remove.
    """
    N, dx, w0 = 192, 2.55e-6, 105e-6
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    E = np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)
    for presc, R2, semi, lam, glass in (
            (_biconvex(), -_R, _SEMI, _LAM, _GLASS),
            (_flat_last(), np.inf, _SEMI, _LAM, 'N-LASF9')):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            zone = fga._caustic_zone(E, dx, presc, lam)
        assert zone is not None
        # the same estimator, on this file's trace
        row = np.abs(E[N // 2, :])
        xs_h, amp_h = xs[N // 2:], row[N // 2:]
        good = amp_h > 0.05 * amp_h.max()
        rr = np.linspace(xs_h[good][0], xs_h[good][-1], 25)
        rr = rr[rr > 0]
        ng = float(la.get_glass_index(glass, lam))
        R1 = presc['surfaces'][0]['radius']
        t = presc['thicknesses'][0]
        _xs_, _us_, _ols_, x_v, u_v, _olv = _oracle_trace(
            rr, [(0.0, R1, 0.0, None, ng), (t, R2, 0.0, None, 1.0)])
        zz = -x_v / u_v
        zz = zz[zz > 0]
        want = (float(np.percentile(zz, 5)), float(np.percentile(zz, 95)))
        for got, w in zip(zone, want):
            assert abs(got - w) / w < 1e-4, (presc['name'], got, w)


@pytest.mark.slow
def test_fga_reproduces_a_diffraction_oracle_at_a_caustic_on_a_curved_lens():
    """DECISION: on a prescription whose LAST surface is curved, FGA reproduces
    an independent diffraction oracle at the caustic.

    Oracle: the brute-force Rayleigh-Sommerfeld-I sum at the top of this file,
    over this file's own exact conic trace.  Its error bound is its OWN
    convergence in the ray quadrature, measured here (n_h 251 -> 501).

    The bar is 0.99 -- a DECISION, with the whole distance between "reproduces
    the field" and "does not" beneath it.  MEASURED (2026-09-14, both builds)
    0.9998 with the repair and 0.0953 without it; the un-projected arm is
    scored here too, so the claim cannot pass on a build where the projection
    silently did nothing.  The premise that this fixture has a curved last
    surface at all is asserted as the sag in waves.
    """
    from lumenairy.raytrace import differential as D
    presc = _biconvex()
    ng = float(la.get_glass_index(_GLASS, _LAM))
    n_h = 501
    h = np.linspace(_SEMI / (2 * n_h), _SEMI * (1.0 - 1.0 / (2 * n_h)), n_h)
    _xs_, _us_, _ols_, x_v, _u_v, opl_v = _oracle_trace(
        h, [(0.0, _R, 0.0, None, ng), (_T, -_R, 0.0, None, 1.0)])
    # premise: the last surface really is curved
    sag_w = float(np.abs(_sag_of(np.abs(_xs_), -_R)).max() / _LAM)
    assert sag_w > 1.0, f'premise: last-surface sag is {sag_w:.3f} waves'
    dxe = np.gradient(x_v, h, edge_order=2)
    assert np.all(dxe > 0), 'premise: the exit plane is not a caustic'
    wgt = np.exp(-(h / _W0) ** 2) * np.sqrt(h * x_v * dxe) * (h[1] - h[0])
    # geometric best focus from the same trace (derived, never pinned)
    f0 = float(-x_v[0] / _u_v[0])
    zs = np.linspace(0.7 * f0, 1.15 * f0, 2001)
    ww = np.exp(-2.0 * (h / _W0) ** 2) * h
    xz = x_v[None, :] + _u_v[None, :] * zs[:, None]
    ctr = (ww * xz).sum(1) / ww.sum()
    var = (ww * (xz - ctr[:, None]) ** 2).sum(1) / ww.sum()
    z = float(zs[int(np.argmin(var))])
    # the readout radii: fine where the focal structure is, coarse outside
    xs_g = (np.arange(_N) - _N / 2) * _DX
    Xg, Yg = np.meshgrid(xs_g, xs_g)
    rg = np.hypot(Xg, Yg)
    na = abs(_u_v[-1]) / np.sqrt(1.0 + _u_v[-1] ** 2)
    airy = 0.61 * _LAM / na
    rho = np.concatenate([np.arange(0.0, 8.0 * airy, airy / 6.0),
                          np.arange(8.0 * airy, float(rg.max()) + _DX, _DX)])
    rho[0] = 1e-12
    hi = _rs_radial(x_v, opl_v, wgt, rho, z, _LAM)
    lo = _rs_radial(x_v[::2], opl_v[::2], wgt[::2] * 2.0, rho, z, _LAM)
    conv = float(np.linalg.norm(hi - lo) / np.linalg.norm(hi))
    assert conv < 1e-3, f'oracle not converged in its ray quadrature: {conv:.2e}'
    orc = (np.interp(rg.ravel(), rho, hi.real)
           + 1j * np.interp(rg.ravel(), rho, hi.imag)).reshape(rg.shape)
    E = np.exp(-(Xg ** 2 + Yg ** 2) / _W0 ** 2).astype(np.complex128)

    def _run(force_surface):
        saved = (D.ray_transfer_jacobian, D.ray_transfer_jacobian_analytic)
        try:
            if force_surface:
                D.ray_transfer_jacobian = _pin_surface(saved[0])
                D.ray_transfer_jacobian_analytic = _pin_surface(saved[1])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                return np.asarray(la.apply_real_lens_fga(
                    E, prescription=presc, wavelength=_LAM, dx=_DX,
                    output_plane_distance=z))
        finally:
            D.ray_transfer_jacobian, D.ray_transfer_jacobian_analytic = saved

    f_now = _fid(orc, _run(False))
    f_old = _fid(orc, _run(True))
    assert f_now > 0.99, f'fga {f_now:.4f} (oracle convergence {conv:.2e})'
    assert f_old < 0.5, (
        f'the un-projected arm scores {f_old:.4f}: this fixture does not '
        f'separate the two reference planes on this build')


def test_the_vector_and_universal_entry_points_follow_the_scalar_one():
    """DECISION: the repair reaches every public FGA entry point.

    ``apply_real_lens_universal(method='fga')`` must return the SAME BYTES as
    ``apply_real_lens_fga`` (it is a dispatcher, not a second model), and
    ``apply_real_lens_fga_vector``'s Ex channel must reproduce the scalar field
    (the Jones matrix of this air-to-air singlet is diagonal, so the two are
    the same propagation).  Bar on the vector arm: 0.999, with the scalar
    field's own agreement with the oracle at 0.9998 above it.
    """
    from lumenairy.propagators.fga import apply_real_lens_fga_vector
    presc = _biconvex()
    N, dx, w0 = 128, 3.6e-6, 80e-6
    xs = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(xs, xs)
    E = np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)
    zf = 1.2e-3
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sc = np.asarray(la.apply_real_lens_fga(
            E, prescription=presc, wavelength=_LAM, dx=dx,
            output_plane_distance=zf))
        un = np.asarray(la.apply_real_lens_universal(
            E, prescription=presc, wavelength=_LAM, dx=dx,
            output_plane_distance=zf, method='fga'))
        ve = np.asarray(apply_real_lens_fga_vector(
            np.stack([E, np.zeros_like(E)]), prescription=presc,
            wavelength=_LAM, dx=dx, output_plane_distance=zf))
    assert np.array_equal(sc, un)
    assert ve.shape == (2, N, N)
    assert _fid(ve[0], sc) > 0.999
