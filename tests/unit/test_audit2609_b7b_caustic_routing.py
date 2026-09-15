"""WP-B7b: the two default moves WP-B7 measured and escalated, and the uniform
fold completion's measured envelope.

1. ``_universal_route``'s caustic branch sends a SINGLE-VALUED field inside the
   sag-screen aberration envelope to ``'phase_screen'``, not ``'fga'``
   (WP-B7 sections 2 / 8.3, re-derived here on a second fixture).
2. ``_analytic_jacobian_applies`` is the analytic differential primitive's OWN
   domain rather than a narrower hand-kept whitelist: an even-aspheric
   prescription now reaches the exact analytic Jacobian in FGA, and a
   field-decentred conic no longer reaches it and raises at call time
   (WP-B7 section 8.2).
3. ``apply_real_lens_traced_uniform`` reports and warns when the fold's
   ``zeta(r) = kappa (r_c - r)`` is carried far past the two-branch band it was
   fitted on (WP-B7 section 8.4 / 6.2).

Every bar here is derived from the running build: the diffraction oracle is a
brute-force Rayleigh-Sommerfeld sum over an exact conic raytrace written in this
file (nothing from ``fga.py`` / ``_lens_traced*`` touches it) and carries its own
convergence estimate; the Jacobian bars come from the finite-difference step
ladder measured in the test; the routing tests assert DECISIONS, not readings.
No wall-clock assertion anywhere.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.propagators import fga
from lumenairy.propagators.fga import (
    _ABERRATION_MAX_RAD,
    _SEIDEL_SA_MAX_RAD,
    _sag_screen_aberration_rad,
    _system_na,
    _tilt_dispersion,
    _universal_route,
)
from lumenairy.raytrace import surfaces_from_prescription
from lumenairy.raytrace.differential import (
    ray_transfer_jacobian,
    ray_transfer_jacobian_analytic,
)

# Model glasses for THIS module only: registered and removed by
# tests/conftest.py::_module_glass_registry_guard.
MODULE_GLASSES = {'_B7B_ZF': lambda wl: 1.5168, '_B7B_K4': lambda wl: 1.5168}


# ===========================================================================
# The oracle: an exact conic meridional raytrace + a brute-force
# Rayleigh-Sommerfeld sum.  Imports nothing from the propagators.
# ===========================================================================
def _trace_biconvex(h, R, t_c, n_glass, conic=0.0):
    """Exact meridional trace of collimated rays through a biconvex singlet
    (vertices z = 0 and z = t_c) to the EXIT VERTEX plane.  Closed-form quadric
    intersection, vector Snell.  Returns (x_exit, slope u = L/N, OPL)."""
    x = np.asarray(h, float).copy()
    z = np.zeros_like(x)
    ux = np.zeros_like(x)
    uz = np.ones_like(x)
    opl = np.zeros_like(x)
    for zv, Rs, mu in ((0.0, R, 1.0 / n_glass), (t_c, -R, n_glass)):
        Z0 = z - zv
        a = (1.0 + conic) * uz * uz + ux * ux
        b = 2.0 * ((1.0 + conic) * Z0 * uz - Rs * uz + x * ux)
        c = (1.0 + conic) * Z0 * Z0 - 2.0 * Rs * Z0 + x * x
        s = np.sqrt(b * b - 4.0 * a * c)
        t1, t2 = (-b - s) / (2.0 * a), (-b + s) / (2.0 * a)
        tt = np.where(np.abs(Z0 + t1 * uz) <= np.abs(Z0 + t2 * uz), t1, t2)
        x = x + tt * ux
        z = z + tt * uz
        opl = opl + (1.0 if zv == 0.0 else n_glass) * tt
        gx = 2.0 * x
        gz = 2.0 * (1.0 + conic) * (z - zv) - 2.0 * Rs
        g = np.sqrt(gx * gx + gz * gz)
        nx, nz = gx / g, gz / g
        ci = -(nx * ux + nz * uz)
        flip = ci < 0
        nx, nz = np.where(flip, -nx, nx), np.where(flip, -nz, nz)
        ci = np.abs(ci)
        f = mu * ci - np.sqrt(1.0 - mu * mu * (1.0 - ci * ci))
        ux, uz = mu * ux + f * nx, mu * uz + f * nz
    tt = (t_c - z) / uz
    x = x + tt * ux
    opl = opl + tt
    return x, ux / uz, opl


def _rs_radial(R, t_c, n_glass, semi, w0, wavelength, rho, z, n_h, n_phi=512,
               conic=0.0):
    """E(rho) at z past the exit vertex, by direct RS-I summation over
    (exit ray) x (azimuth).  The change of variable to the launch height h makes
    the integrand smooth: U x_e dx_e = E_in(h) sqrt(h x_e |dx_e/dh|) dh."""
    h = np.linspace(semi / (2 * n_h), semi * (1.0 - 1.0 / (2 * n_h)), n_h)
    dh = h[1] - h[0]
    xe, _ue, opl = _trace_biconvex(h, R, t_c, n_glass, conic=conic)
    dxe = np.gradient(xe, h, edge_order=2)
    assert np.all(dxe > 0), 'premise: the exit plane must not be a caustic'
    wgt = np.exp(-(h / w0) ** 2) * np.sqrt(h * xe * dxe) * dh
    k = 2.0 * np.pi / wavelength
    phi = (np.arange(n_phi) + 0.5) * (2.0 * np.pi / n_phi)
    cph = np.cos(phi)
    pre = np.exp(1j * k * opl) * wgt
    out = np.empty(rho.size, complex)
    for i0 in range(0, rho.size, 32):
        rr = rho[i0:i0 + 32][:, None, None]
        r2 = (z * z + rr * rr + xe[None, :, None] ** 2
              - 2.0 * rr * xe[None, :, None] * cph[None, None, :])
        r = np.sqrt(r2)
        out[i0:i0 + 32] = ((np.exp(1j * k * r) * (z / r2))
                           * pre[None, :, None]).sum(axis=(1, 2))
    return out * (2.0 * np.pi / n_phi) / (1j * wavelength)


def _fid(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(abs(np.vdot(a, b)) ** 2
                 / (np.vdot(a, a).real * np.vdot(b, b).real))


# ===========================================================================
# Fixture B7b -- N-SF11 biconvex, 633 nm (nothing shared with WP-B7's fixture)
# ===========================================================================
_LAM = 0.633e-6
_AP = 0.30e-3
_T = 0.60e-3
_N, _DX = 192, 1.8e-6
_W0 = 80e-6
_R = 1.6e-3


def _grid(n=_N, dx=_DX):
    x1 = (np.arange(n) - n / 2) * dx
    return np.meshgrid(x1, x1)


def _beam(n=_N, dx=_DX, w0=_W0):
    X, Y = _grid(n, dx)
    return np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)


def _presc(R=_R, conic=0.0):
    p = la.make_singlet(R, -R, _T, 'N-SF11', aperture=_AP)
    if conic:
        p = {**p, 'surfaces': [{**s, 'conic': float(conic)}
                               for s in p['surfaces']]}
    return p


def _route(E, presc, opd, lam=_LAM, dx=_DX):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return _universal_route(E, presc, lam, dx, dx, opd, 0.12, 3.0, None,
                                0.06, _ABERRATION_MAX_RAD, _SEIDEL_SA_MAX_RAD)


def _ray_best_focus(R, conic=0.0, w0=_W0, n_h=4000):
    """The intensity-weighted geometric best focus, from the oracle's own
    raytrace -- so the readout plane is derived here, not pinned."""
    h = np.linspace(_AP / 2 / (2 * n_h), _AP / 2 * (1 - 1 / (2 * n_h)), n_h)
    xe, ue, _ = _trace_biconvex(h, R, _T, float(la.get_glass_index('N-SF11', _LAM)),
                                conic=conic)
    f0 = float(-xe[0] / ue[0])
    zs = np.linspace(0.5 * f0, 1.15 * f0, 4001)
    wgt = np.exp(-2 * (h / w0) ** 2) * h
    xz = xe[None, :] + ue[None, :] * zs[:, None]
    ctr = (wgt * xz).sum(1) / wgt.sum()
    var = (wgt * (xz - ctr[:, None]) ** 2).sum(1) / wgt.sum()
    return float(zs[int(np.argmin(var))])


# ===========================================================================
# Item 1 -- the caustic branch
# ===========================================================================
def test_b7b_oracle_agrees_with_the_library_raytrace():
    """Premise for everything below: the oracle's own conic trace reproduces
    ``lumenairy.raytrace``'s exit state.  They share no code -- the oracle
    solves the quadric in closed form and refracts by the vector Snell form
    written above; the library runs its production tracer.

    MEASURED (WP-B7b): height 1.6e-19 m, slope 1.4e-16, OPL 1.3e-18 m on this
    singlet.  The bars are 1e-12 m / 1e-10 / 1e-12 m -- decades above the
    measured agreement and decades below anything that would change a routing
    or accuracy conclusion (the members differ by MICRONS).
    """
    from lumenairy import raytrace as rt
    p = _presc()
    n = 2000
    h = np.linspace(_AP / 2 / n, _AP / 2 * 0.98, n)
    surfs = surfaces_from_prescription(p)
    bundle = rt.RayBundle(x=h.copy(), y=np.zeros(n), z=np.zeros(n),
                          L=np.zeros(n), M=np.zeros(n), N=np.ones(n),
                          wavelength=_LAM, alive=np.ones(n, bool),
                          opd=np.zeros(n))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ex = rt.trace(bundle, surfs, _LAM).at_exit_vertex()
    ok = np.asarray(ex.alive, bool)
    assert ok.sum() > n // 2
    xe, ue, opl = _trace_biconvex(h, _R, _T,
                                  float(la.get_glass_index('N-SF11', _LAM)))
    assert np.abs(np.asarray(ex.x)[ok] - xe[ok]).max() < 1e-12
    assert np.abs((np.asarray(ex.L) / np.asarray(ex.N))[ok] - ue[ok]).max() < 1e-10
    assert np.abs(np.asarray(ex.opd)[ok] - opl[ok]).max() < 1e-12


def test_b7b_single_valued_field_at_a_caustic_routes_to_phase_screen():
    """A single-valued field at a caustic, inside the sag-screen aberration
    envelope, takes ``'phase_screen'`` -- the thin screen plus the exact angular
    spectrum -- and not ``'fga'``.

    WHY, as of 2026-09-14 (WP-B12): because the thin screen is much the cheaper
    member there, not because it is the more accurate one.  Measured against a
    brute-force Rayleigh-Sommerfeld oracle on an exact conic raytrace, on this
    N-SF11 biconvex R = +/-1.6 mm singlet (NA 0.160) at its traced best focus,
    ``'fga'`` scores 0.9998 and ``'phase_screen'`` 0.9991, at 15.74 s against
    0.53 s.  (WP-B7b routed here on a 0.1251-vs-0.9991 reading; that deficit
    was the FGA reference-plane defect WP-B12 repaired.)  See
    :func:`test_b7b_both_members_reproduce_the_oracle_and_fga_is_the_closer_one`,
    which re-measures the decisive pair rather than trusting this docstring.
    Whether the route stays here is a maintainer decision (handoff sec. 4.7);
    this test pins what the router DOES, which is unchanged by WP-B12.

    The premise -- that these planes really are inside the caustic gate and
    really are single-valued and unaberrated -- is asserted, not assumed.
    """
    E = _beam()
    for R in (2.0e-3, _R, 1.3e-3, 1.05e-3):
        p = _presc(R)
        z = _ray_best_focus(R)
        na = _system_na(p, _LAM)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ab = _sag_screen_aberration_rad(E, _DX, _DX, p, _LAM)
            mv = _tilt_dispersion(E, _DX, _DX, _LAM, na)
        assert na > 0.12, (R, na)            # past the low-NA shortcut
        assert ab < _ABERRATION_MAX_RAD, (R, ab)     # inside the H2 envelope
        assert mv < 0.06, (R, mv)                    # single-valued
        assert _route(E, p, z) == 'phase_screen', (R, na, z)
        # and away from the caustic the router is untouched
        assert _route(E, p, 0.0) == 'traced', R


def test_b7b_multivalued_field_at_a_caustic_still_routes_to_fga():
    """The multi-valued branch ABOVE the caustic gate is deliberately unchanged:
    several wave components cross the same region, there is no single local
    direction, and only FGA's phase-space swarm transports them independently.
    That is what ``'fga'`` uniquely provides at a caustic."""
    X, _Y = _grid()
    k0 = 2 * np.pi / _LAM
    two = (_beam() * np.exp(1j * k0 * 0.05 * X)
           + _beam() * np.exp(-1j * k0 * 0.05 * X)).astype(np.complex128)
    p = _presc()
    na = _system_na(p, _LAM)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mv = _tilt_dispersion(two, _DX, _DX, _LAM, na)
    assert mv > 0.06, mv                     # premise: really multi-valued
    assert _route(two, p, _ray_best_focus(_R)) == 'fga'


def test_b7b_aberrated_caustic_keeps_fga_two_sided():
    """The H2 aberration gate still owns the other half of the decision: a
    prescription whose sag-screen estimate is OVER budget never reaches the thin
    screen, at a caustic either.  That class is the 2026-07-19 displaced /
    Debye-oracle regime where the analytic model is 58-123 % wrong.

    Two-sided on ONE lens: the same N-SF11 biconvex with a conic constant swept
    until the gate trips.  Under budget -> ``'phase_screen'``; over budget ->
    ``'fga'``, with nothing else about the plane changed.  The conic that trips
    it is found by a ladder here, not pinned."""
    E = _beam()
    under = over = None
    for kc in (0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 60.0):
        p = _presc(conic=kc)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ab = _sag_screen_aberration_rad(E, _DX, _DX, p, _LAM)
        z = _ray_best_focus(_R, conic=kc)
        if ab < _ABERRATION_MAX_RAD:
            under = (kc, ab, z, _route(E, p, z))
        elif over is None:
            over = (kc, ab, z, _route(E, p, z))
    assert under is not None and over is not None, 'ladder exhausted'
    assert under[3] == 'phase_screen', under
    assert over[3] == 'fga', over
    assert over[1] > _ABERRATION_MAX_RAD > under[1]


@pytest.mark.slow
def test_b7b_both_members_reproduce_the_oracle_and_fga_is_the_closer_one():
    """The measurement the caustic route rests on, re-run here.

    RESTATED 2026-09-14 (WP-B12).  This test used to assert
    ``fga < 0.5``, because on this fixture ``'fga'`` scored 0.1251 against the
    screen's 0.9991.  That deficit was a reference-plane defect, not an FGA
    model limit: the differential transfer returned the base-ray state ON the
    last surface while ``fga.py`` added the image-side leg as if it were on the
    exit-vertex plane, so every beamlet carried a spurious ``k * sag(rho)`` --
    7.79 waves at the rim of THIS singlet.  With the four sites asking for
    ``reference='exit_vertex'`` the ordering reverses, and the claim worth
    pinning is not a reading but the DECISION the route should be argued from:

    * both members reproduce the oracle at the caustic (bar 0.99 each);
    * ``'fga'`` is the closer one, by a margin stated as a RATIO so it carries
      no per-build constant: its infidelity is under half the screen's.

    MEASURED here and by ``validation/probe_wp_b12/probe_c_route.py``
    (2026-09-14, both builds): ``'fga'`` 0.9998 (infidelity 1.8e-04),
    ``'phase_screen'`` 0.9991 (infidelity 8.9e-04), ratio 0.20 against the 0.5
    bar -- 2.5x of margin, and the same ratio on the N = 192 dx = 1.8 um grid.
    The route still takes ``'phase_screen'`` here and
    :func:`test_b7b_single_valued_field_at_a_caustic_routes_to_phase_screen`
    still pins that: with the accuracy ordering reversed it is a COST choice
    (0.53 s against 15.74 s on this grid), and whether to move it is a
    maintainer decision (handoff section 4.7).

    Oracle: the brute-force Rayleigh-Sommerfeld sum above, whose error bound is
    its OWN convergence in the ray quadrature (n_h 451 -> 901, measured here
    and asserted below 1e-3).  Both members are read at the same plane by the
    same dispatcher call.  No wall-clock assertion: the costs above are
    reported, never asserted.
    """
    pytest.importorskip('numba')
    from lumenairy.propagators.asm import angular_spectrum_propagate
    from lumenairy.propagators.fga import apply_real_lens_fga
    ng = float(la.get_glass_index('N-SF11', _LAM))
    p = _presc()
    z = _ray_best_focus(_R)
    X, Y = _grid()
    rg = np.hypot(X, Y)
    rho = np.arange(0.0, float(rg.max()) + _DX, _DX / 4)
    rho[0] = 1e-12
    lo = _rs_radial(_R, _T, ng, _AP / 2, _W0, _LAM, rho, z, 451)
    hi = _rs_radial(_R, _T, ng, _AP / 2, _W0, _LAM, rho, z, 901)
    conv = float(np.linalg.norm(hi - lo) / np.linalg.norm(hi))
    assert conv < 1e-3, f'oracle not converged in its ray quadrature: {conv:.3e}'
    from scipy.interpolate import CubicSpline
    orc = np.zeros_like(X, dtype=complex)
    inside = rg <= rho[-1]
    orc[inside] = (CubicSpline(rho, hi.real)(rg[inside])
                   + 1j * CubicSpline(rho, hi.imag)(rg[inside]))
    E = _beam()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ps = angular_spectrum_propagate(
            np.asarray(la.apply_real_lens(E, prescription=p, wavelength=_LAM,
                                          dx=_DX)), z, _LAM, _DX, None)
        fg = np.asarray(apply_real_lens_fga(
            E, prescription=p, wavelength=_LAM, dx=_DX,
            output_plane_distance=z))
    f_ps, f_fga = _fid(orc, ps), _fid(orc, fg)
    assert f_ps > 0.99, f'phase_screen {f_ps:.6f} (oracle conv {conv:.2e})'
    assert f_fga > 0.99, f'fga {f_fga:.6f} (oracle conv {conv:.2e})'
    # the ordering, as a ratio of infidelities so no per-build constant enters
    ratio = (1.0 - f_fga) / (1.0 - f_ps)
    assert ratio < 0.5, (
        f'fga {f_fga:.6f} (infidelity {1 - f_fga:.2e}) is not the closer '
        f'member against phase_screen {f_ps:.6f} (infidelity {1 - f_ps:.2e}): '
        f'ratio {ratio:.3f} (oracle convergence {conv:.2e})')


# ===========================================================================
# Item 3 -- the analytic-Jacobian predicate
# ===========================================================================
def _a4(asph=None, dec=None, tilt=None, radius_y=None):
    s0 = {'radius': 51.68e-3, 'thickness': 5e-3, 'glass_before': 'air',
          'glass_after': '_B7B_ZF', 'semi_diameter': 12e-3}
    if asph is not None:
        s0['aspheric_coeffs'] = asph
    if dec is not None:
        s0['decenter'] = dec
    if tilt is not None:
        s0['tilt'] = tilt
    if radius_y is not None:
        s0['radius_y'] = radius_y
    return {'wavelength': 1.31e-6, 'aperture_diameter': 24e-3,
            'surfaces': [s0, {'radius': -51.68e-3, 'thickness': 0.0,
                              'glass_before': '_B7B_ZF', 'glass_after': 'air',
                              'semi_diameter': 12e-3}],
            'thicknesses': [5e-3], 'stop_index': 0}


_A4_CASES = {
    'conic': _a4(),
    'even asphere': _a4(asph={4: 4.0e3}),
    'A4/A6 asphere': _a4(asph={4: 4.0e3, 6: -1.2e7}),
    'biconic': _a4(radius_y=25e-3),
    'field decenter': _a4(dec=(1e-4, 0.0)),
    'field tilt': _a4(tilt=(1e-3, 0.0)),
}


def test_b7b_predicate_is_the_analytic_primitive_s_own_domain():
    """``_analytic_jacobian_applies`` must agree, surface class by surface
    class, with whether ``ray_transfer_jacobian_analytic`` actually accepts the
    prescription -- which is asked HERE by calling it, not by reading a list.

    That is the whole point of the rename: FGA dispatches on a predicate where
    ``gbd.jacobian='auto'`` dispatches on the primitive's own
    ``NotImplementedError``, so the two must not be allowed to drift.  Before
    WP-B7b the predicate rejected an even asphere the primitive accepts (WP-B9
    gave it that support) and accepted a field-decentred conic the primitive
    rejects -- so an aspheric prescription traced the 9-ray FD bundle whatever
    ``exact_jacobian`` said, and a field-decentred one raised at call time.
    """
    h = np.linspace(-11.0e-3, 11.0e-3, 33)
    zz = np.zeros(33)
    for name, presc in _A4_CASES.items():
        surfs = surfaces_from_prescription(presc)
        try:
            ray_transfer_jacobian_analytic(h, zz, zz, zz, surfs, 1.31e-6)
            accepted = True
        except NotImplementedError:
            accepted = False
        assert fga._analytic_jacobian_applies(surfs) is accepted, name
        want = (ray_transfer_jacobian_analytic if accepted
                else ray_transfer_jacobian)
        assert fga._pick_ray_transfer(surfs, None) is want, name
        assert fga._pick_ray_transfer(surfs, True) is want, name
        assert fga._pick_ray_transfer(surfs, False) is ray_transfer_jacobian, name
    # the old name still resolves, for callers that imported it
    assert fga._is_all_conic is fga._analytic_jacobian_applies


def test_b7b_field_decentred_conic_falls_back_instead_of_raising():
    """The latent bug: a field-decentred conic used to reach the analytic
    primitive from FGA and raise ``NotImplementedError`` at call time, because
    the predicate did not check ``field_decenter`` / ``field_tilt`` /
    ``field_sag_callable`` while the primitive does.  It now falls back to the
    finite-difference primitive, which carries the decenter walk-off correctly.

    Fail-before: at the parent commit this test raises on the very first call
    (``_pick_ray_transfer(..., True)`` returned the analytic primitive there).
    """
    h = np.linspace(-11.0e-3, 11.0e-3, 33)
    zz = np.zeros(33)
    for presc in (_a4(dec=(1e-4, 0.0)), _a4(tilt=(1e-3, 0.0))):
        surfs = surfaces_from_prescription(presc)
        prim = fga._pick_ray_transfer(surfs, True)
        assert prim is ray_transfer_jacobian
        out = prim(h, zz, zz, zz, surfs, 1.31e-6)     # must NOT raise
        assert np.all(np.isfinite(out.x[np.asarray(out.alive, bool)]))
        with pytest.raises(NotImplementedError):
            ray_transfer_jacobian_analytic(h, zz, zz, zz, surfs, 1.31e-6)


def test_b7b_aspheric_swap_costs_nothing_and_removes_the_fd_truncation():
    """What the aspheric default move buys, measured on the A4 singlet.

    The two primitives trace the SAME base ray, so their exit states agree to
    floating-point noise; what differs is the JACOBIAN, where the FD primitive
    carries central-difference truncation and the analytic one does not.  The
    bar on each is derived HERE: the base-ray bar from the machine epsilon of
    the traced coordinate, the Jacobian bar from the FD step ladder's own
    ``h^2`` scaling measured on this build.
    """
    surfs = surfaces_from_prescription(_a4(asph={4: 4.0e3}))
    n = 401
    h = np.linspace(-11.9e-3, 11.9e-3, n)
    zz = np.zeros(n)
    an = ray_transfer_jacobian_analytic(h, zz, zz, zz, surfs, 1.31e-6)
    fd = ray_transfer_jacobian(h, zz, zz, zz, surfs, 1.31e-6)
    ok = np.asarray(fd.alive, bool) & np.asarray(an.alive, bool)
    assert ok.sum() > n // 2
    # base ray: the same trace, to the resolution of the coordinate itself
    eps_bar = 1e3 * np.finfo(float).eps
    assert (np.abs(fd.x[ok] - an.x[ok])
            / np.maximum(np.abs(an.x[ok]), 1e-12)).max() < eps_bar
    assert np.abs(fd.ux[ok] - an.ux[ok]).max() < eps_bar
    # Jacobian: the FD side is truncation-limited, and the ladder shows it.
    # Central differences scale as h^2, so a 10x smaller step must cut the
    # disagreement by ~100x while the analytic side does not move at all.
    def _dJ(hp, hs):
        f = ray_transfer_jacobian(h, zz, zz, zz, surfs, 1.31e-6,
                                  h_pos=hp, h_slope=hs)
        return float((np.abs(f.jacobian[ok] - an.jacobian[ok])
                      / np.maximum(np.abs(an.jacobian[ok]), 1e-30)).max())
    coarse, default, fine = _dJ(1e-5, 5e-4), _dJ(1e-6, 5e-5), _dJ(1e-7, 5e-6)
    assert coarse > 10.0 * default > 100.0 * fine, (coarse, default, fine)
    assert default > 1e3 * eps_bar, default    # the FD side really is truncated
    # and the swap only changes which primitive runs, not the trace count model
    assert fga._FD_BUNDLE_RAYS == 9
    assert (fga._fga_lattice_point_bytes(49, 49, True, 1.0, True)
            > fga._fga_lattice_point_bytes(49, 49, True, 1.0, False))


# ===========================================================================
# Item 2 -- the uniform fold completion's extrapolation envelope
# ===========================================================================
def _k4(semi=0.75e-3, R=2.7e-3, t=1.0e-3):
    return {'wavelength': 1.31e-6, 'aperture_diameter': 2 * semi, 'surfaces': [
        {'radius': R, 'thickness': t, 'glass_before': 'air',
         'glass_after': '_B7B_K4', 'semi_diameter': semi},
        {'radius': float('inf'), 'thickness': 0.0, 'glass_before': '_B7B_K4',
         'glass_after': 'air', 'semi_diameter': semi}],
        'thicknesses': [t], 'stop_index': 0}


def _fold_fixture():
    """The N-LAK22 biconvex whose fold WP-B7b measured (a second singlet, not
    the one the K4 suite uses)."""
    semi = 0.55e-3
    return {'aperture_diameter': 2 * semi, 'surfaces': [
        {'radius': 6.0e-3, 'thickness': 0.9e-3, 'glass_before': 'air',
         'glass_after': 'N-LAK22', 'semi_diameter': semi},
        {'radius': -6.0e-3, 'thickness': 0.0, 'glass_before': 'N-LAK22',
         'glass_after': 'air', 'semi_diameter': semi}],
        'thicknesses': [0.9e-3], 'stop_index': 0}, 1.55e-6, 4.400e-3


def test_b7b_fold_band_is_the_two_branch_band():
    """``_trace_meridional_fold`` reports ``band``: the width of the radial
    interval reached by BOTH coalescing branches, which is the only interval on
    which the eikonal difference defining ``zeta`` exists -- and therefore the
    only interval the linear ``kappa`` fit speaks for.

    Derived here from an independent meridional trace of the same optic, not
    pinned to a number."""
    from lumenairy.elements._lens_traced_uniform import _trace_meridional_fold
    presc, wl, z = _fold_fixture()
    ap = presc['aperture_diameter']
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fold = _trace_meridional_fold(presc, wl, z, 1.0, 0.5 * ap * 0.98, 4000)
    assert fold['ok'] and fold['n_turn'] == 1, fold
    ng = float(la.get_glass_index('N-LAK22', wl))
    n = 4000
    lr = 0.5 * ap * 0.98
    h = np.linspace(lr / n, lr, n)
    xe, ue, _o = _trace_biconvex(h, 6.0e-3, 0.9e-3, ng)
    xo = xe + ue * z
    turns = np.where(np.diff(np.sign(np.diff(xo))) != 0)[0] + 1
    assert turns.size == 1
    i_f = int(turns[0])
    r_c = float(abs(xo[i_f]))
    band = r_c - float(max(np.abs(xo[:i_f + 1]).min(),
                           np.abs(xo[i_f:]).min()))
    assert fold['r_c'] == pytest.approx(r_c, rel=1e-6)
    assert fold['band'] == pytest.approx(band, rel=1e-3)
    assert 0.0 < fold['band'] < r_c


def test_b7b_uniform_warns_only_when_zeta_is_extrapolated():
    """Two-sided, and the FIELD is unchanged either way.

    The completion fits ``kappa`` on the two-branch band and then evaluates
    ``zeta`` across a fit band ``W = l_airy`` and a dark fill of
    ``_AIRY_TAIL_CELLS * l_airy``.  When the two-branch band is much narrower
    than ``W`` that is extrapolation, and MEASURED against a brute-force
    Rayleigh-Sommerfeld oracle the completed field's total power runs -2.9 % to
    +4.9 % of the truth while the ratio is under ~5 and +12.5 % to +22.8 % from
    ~10 up.  It still beats the multibranch it would fall back to at every
    measured plane (fidelity 0.93-0.97 against 0.81-0.89), so this is a warning
    and not a reroute -- the returned field is identical with and without it.

    The two arms are a factor >100 apart in the ratio, so the decision does not
    sit inside anything a build can move."""
    from lumenairy.elements._lens_traced_uniform import (
        _ZETA_EXTRAPOLATION_MAX, apply_real_lens_traced_uniform,
    )
    quiet_p, quiet_wl, quiet_z, quiet_n, quiet_dx, quiet_w = (
        _k4(), 1.31e-6, 4.3704e-3, 384, 3e-6, 0.55e-3)
    loud_p, loud_wl, loud_z = _fold_fixture()
    loud_n, loud_dx, loud_w = 320, 3.85e-6, 0.40e-3
    got = {}
    for tag, (p, wl, z, n, dx, w0) in {
            'quiet': (quiet_p, quiet_wl, quiet_z, quiet_n, quiet_dx, quiet_w),
            'loud': (loud_p, loud_wl, loud_z, loud_n, loud_dx, loud_w)}.items():
        x1 = (np.arange(n) - n / 2) * dx
        Xg, Yg = np.meshgrid(x1, x1)
        E0 = np.exp(-(Xg ** 2 + Yg ** 2) / w0 ** 2).astype(np.complex128)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            E, d = apply_real_lens_traced_uniform(
                E0, prescription=p, wavelength=wl, dx=dx,
                output_plane_distance=z, ray_subsample=2,
                return_diagnostics=True)
        assert not d['fell_back'], (tag, d['reason'])
        got[tag] = (np.asarray(E), float(d['zeta_extrapolation']),
                    sum(1 for m in rec if 'EXTRAPOLATION' in str(m.message)))
    # the module's own fold fixture is inside the envelope and silent
    assert got['quiet'][1] < _ZETA_EXTRAPOLATION_MAX, got['quiet'][1]
    assert got['quiet'][2] == 0
    # the second singlet's tight fold is an extrapolation and says so
    assert got['loud'][1] > _ZETA_EXTRAPOLATION_MAX, got['loud'][1]
    assert got['loud'][2] == 1
    # >100x between the two arms: the decision is not a near-tie
    assert got['loud'][1] > 100.0 * got['quiet'][1]
    # the warning changes nothing about the field it warns on
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        E2 = np.asarray(apply_real_lens_traced_uniform(
            np.exp(-(np.add.outer(
                ((np.arange(loud_n) - loud_n / 2) * loud_dx) ** 2,
                ((np.arange(loud_n) - loud_n / 2) * loud_dx) ** 2))
                / loud_w ** 2).astype(np.complex128),
            prescription=loud_p, wavelength=loud_wl, dx=loud_dx,
            output_plane_distance=loud_z, ray_subsample=2))
    assert np.array_equal(E2, got['loud'][0])
