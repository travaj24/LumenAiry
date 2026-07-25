"""R8 / audit F3 + F4 (2026-07-21).

F3 (P2): ``apply_real_lens_traced(tilt_aware_rays=True)`` DEGRADES a steep
explicit carrier -- the per-pixel tilt launch reads 1.8 rad rms / ~185%
exit-curvature error on the 121 S5-S7 triplet vs 0.008 rad / <1% for the
``carrier=R`` default (the N5 fix made tilt_aware collimated-safe but left it
worse than the default on a steep spherical carrier).  Fix: when BOTH an
EXPLICIT engaged carrier AND ``tilt_aware_rays`` are given, route the ray launch
+ the R7 intra-group fixes through the carrier gradient (== the ``carrier=R``
default path) and warn.  The N5 auto-carrier tilt_aware path (``carrier`` None /
``'auto'``) is untouched -- its per-pixel launch stays byte-identical.

F4 (P3): composition ergonomics.
  * ``propagate_traced_carrier_chain`` packages the per-group hand-off (carrier
    leg -> reconstruct -> ``apply_real_lens_traced(carrier=R_in)`` -> re-envelope
    with R_out), and the ELEMENT supplies R_out from its own paraxial ABCD -- no
    external q-trace.  Reproduces the repro-script hand-off in a few lines.
  * ``carrier_referenced_focus_readout`` packages the near-focus landing (stop a
    small ``standoff`` short + fine Bluestein zoom to the target plane), so
    landing at / near the carrier focus no longer collapses the co-moving window
    / clips the halo / raises at the exact focus.

These tests are SELF-CONTAINED (no Zemax): a synthetic thick triplet with a
steep spherical carrier validated against an INLINE exact meridional raytrace +
eikonal oracle (mirroring ``validation/repro_traced_carrier_121/
traced_group_oracle.py``), a synthetic multi-group carrier chain validated
against the manual hand-off pattern, and a synthetic converging Gaussian
validated against the analytic Gaussian focus.  The .zmx end-to-end acceptance
runs locally (reported in the R8 notes) but is NOT a unit test.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements import _lens_traced as LT
from lumenairy.glass import get_glass_index
from lumenairy.propagators.carrier import _paraxial_group_r_out
from lumenairy.raytrace.seidel import system_abcd_prescription

_WL = 1.31e-6
_K0 = 2.0 * np.pi / _WL


def _skip_if_low_ram(N):
    """Skip a big-N traced case when free RAM is tight (defensive)."""
    try:
        import psutil
        if psutil.virtual_memory().available < 2 * (N * N * 16) + (1 << 30):
            pytest.skip('insufficient free RAM for the traced N={} case'.format(N))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Inline exact meridional oracle (lumenairy-free), mirroring the repro / R7.
# ---------------------------------------------------------------------------
def _n_of(g):
    return 1.0 if g in (None, '', 'air', 'AIR') else float(get_glass_index(g, _WL))


def _build_group(radii, glasses, thicks, aperture, name='r8-synth'):
    gb = ['air'] + list(glasses)
    ga = list(glasses) + ['air']
    surfaces = [
        {'radius': radii[i], 'glass_before': gb[i], 'glass_after': ga[i],
         'conic': 0.0, 'radius_y': None, 'conic_y': None,
         'aspheric_coeffs': None, 'aspheric_coeffs_y': None}
        for i in range(len(radii))
    ]
    return {'name': name, 'aperture_diameter': aperture,
            'surfaces': surfaces, 'thicknesses': list(thicks)}


def _group_geometry(pres):
    surfs = []
    z = 0.0
    th = pres['thicknesses']
    for i, s in enumerate(pres['surfaces']):
        R_s = s['radius'] if s['radius'] else np.inf
        surfs.append((z, R_s, _n_of(s['glass_before']), _n_of(s['glass_after'])))
        z += th[i] if i < len(th) else 0.0
    return surfs, surfs[-1][0]


def _oracle_phase(surfs, z_exit, R_in, x_max, npts=600):
    x0s = np.linspace(1e-9, x_max, npts)
    xe = np.full(npts, np.nan)
    ph = np.full(npts, np.nan)
    for j, x0 in enumerate(x0s):
        P = np.array([x0, 0.0])
        if np.isinf(R_in):
            u = np.array([0.0, 1.0])
        else:
            u = np.array([x0, R_in]) / np.hypot(x0, R_in)
            if R_in < 0:
                u = np.array([-x0, -R_in]) / np.hypot(x0, R_in)
        opl = (np.sign(R_in) * (np.hypot(x0, R_in) - abs(R_in))
               if np.isfinite(R_in) else 0.0)
        ok = True
        for (zv, R_s, n1, n2) in surfs:
            if np.isfinite(R_s):
                C = np.array([0.0, zv + R_s])
                oc = P - C
                b = np.dot(oc, u)
                c = np.dot(oc, oc) - R_s * R_s
                disc = b * b - c
                if disc < 0:
                    ok = False
                    break
                t1, t2 = -b - np.sqrt(disc), -b + np.sqrt(disc)
                t = min((t1, t2), key=lambda tt: abs((P + tt * u)[1] - zv))
                P2 = P + t * u
                nv = np.sign(R_s) * (C - P2) / np.linalg.norm(C - P2)
            else:
                if abs(u[1]) < 1e-12:
                    ok = False
                    break
                t = (zv - P[1]) / u[1]
                P2 = P + t * u
                nv = np.array([0.0, 1.0])
            opl += n1 * t
            c1 = np.dot(u, nv)
            eta = n1 / n2
            disc2 = 1.0 - eta * eta * (1.0 - c1 * c1)
            if disc2 < 0:
                ok = False
                break
            u = eta * u + (np.sqrt(disc2) - eta * c1) * nv
            u = u / np.linalg.norm(u)
            P = P2
        if not ok:
            continue
        n_last = surfs[-1][3]
        t = (z_exit - P[1]) / u[1]
        opl += n_last * t
        xe[j] = (P + t * u)[0]
        ph[j] = _K0 * opl
    good = np.isfinite(xe)
    xe, ph = xe[good], ph[good]
    p2 = np.polyfit(xe[:20], ph[:20], 2)
    return xe, ph - np.polyval(p2, 0.0)


def _carrier_input(pres, R_in, w_, N):
    dx = 4.2 * w_ / N * 2
    x = (np.arange(N) - N // 2) * dx
    r2g = x[None, :] ** 2 + x[:, None] ** 2
    env = np.exp(-r2g / w_ ** 2)
    Sin = (np.sign(R_in) * (np.sqrt(r2g + R_in * R_in) - abs(R_in))
           if np.isfinite(R_in) else 0.0)
    E_in = (env * np.exp(1j * _K0 * Sin)).astype(np.complex128)
    return E_in, dx, r2g


def _exit_rms(pres, R_in, w_, N=1024, **kw):
    """RMS exit-phase residual (rad, piston removed) vs the meridional oracle
    over r < w -- the repro / R7 metric."""
    surfs, z_exit = _group_geometry(pres)
    E_in, dx, r2g = _carrier_input(pres, R_in, w_, N)
    xe, ph = _oracle_phase(surfs, z_exit, R_in, x_max=3.2 * w_)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        E_out = la.apply_real_lens_traced(
            E_in, prescription=pres, wavelength=_WL, dx=dx,
            ray_subsample=4, n_workers=1, parallel_amp=False, **kw)
    rr = np.sqrt(r2g)
    ph_o = np.interp(rr, xe, ph, left=ph[0], right=np.nan)
    res = np.angle(E_out * np.exp(-1j * ph_o))
    mask = (rr < w_) & np.isfinite(ph_o) & (np.abs(E_out) > 0.05 * np.abs(E_out).max())
    res0 = res - np.median(res[rr < 0.05 * w_])
    rm = res0[mask]
    m = np.angle(np.mean(np.exp(1j * rm)))
    return float(np.sqrt(np.mean(np.angle(np.exp(1j * (rm - m))) ** 2)))


# A strong, thick focusing triplet with a wide aperture (mirrors R7 _TRIPLET):
# the launch grid reaches far past the beam, so the marginal rays' high-order
# aberration is what the tilt_aware per-pixel launch mishandles on a steep
# carrier (reproduces F3 at ~1.8 rad with the guard disabled).
_TRIPLET = _build_group(
    radii=[40e-3, -35e-3, 120e-3, -45e-3],
    glasses=['N-BK7', 'N-SF2', 'N-BK7'],
    thicks=[7e-3, 5e-3, 7e-3], aperture=24e-3)


# ===========================================================================
# F3 -- tilt_aware_rays on a steep explicit carrier
# ===========================================================================
@pytest.mark.parametrize('R_in, w', [
    (55e-3, 4.0e-3),     # steep DIVERGING carrier
    (-30e-3, 3.5e-3),    # steep CONVERGING carrier
])
def test_r8_f3_tiltaware_no_worse_than_default(R_in, w):
    """Acceptance: tilt_aware_rays on a steep spherical carrier is NO WORSE than
    the default ``carrier=R`` path.  With the F3 guard the two are identical
    (the launch is rerouted to the carrier gradient)."""
    _skip_if_low_ram(1024)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        rms_default = _exit_rms(_TRIPLET, R_in, w, carrier=R_in)
        rms_tilt = _exit_rms(_TRIPLET, R_in, w, carrier=R_in,
                             tilt_aware_rays=True)
    assert rms_default < 0.1, (
        f"default carrier=R exit rms {rms_default:.4f} exceeds the F2 gate "
        f"(the R7/F2 fix must be present on this branch)")
    assert rms_tilt <= max(1.5 * rms_default, rms_default + 0.02), (
        f"tilt_aware exit rms {rms_tilt:.4f} rad worse than the default path "
        f"{rms_default:.4f} rad (F3 guard did not reroute)")


def test_r8_f3_fail_before_pass_after(monkeypatch):
    """fail-before / pass-after: with the F3 guard DISABLED the steep-carrier
    tilt_aware launch reads ~1.8 rad (reproducing F3); with it enabled it drops
    to the ``carrier=R`` default (< 0.1 rad) and matches it to the FP floor."""
    _skip_if_low_ram(1024)
    R_in, w = 55e-3, 4.0e-3
    rms_ref = _exit_rms(_TRIPLET, R_in, w, carrier=R_in)

    monkeypatch.setattr(LT, '_F3_GUARD_TILTAWARE_EXPLICIT_CARRIER', False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        rms_before = _exit_rms(_TRIPLET, R_in, w, carrier=R_in,
                               tilt_aware_rays=True)
    assert rms_before > 0.5, (
        f"pre-fix tilt_aware must reproduce the F3 degradation "
        f"(got {rms_before:.4f} rad, expected ~1.8)")

    monkeypatch.setattr(LT, '_F3_GUARD_TILTAWARE_EXPLICIT_CARRIER', True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        rms_after = _exit_rms(_TRIPLET, R_in, w, carrier=R_in,
                              tilt_aware_rays=True)
    assert rms_after < 0.1, (rms_after,)
    assert rms_after == pytest.approx(rms_ref, abs=1e-6), (rms_after, rms_ref)
    assert rms_before > 10 * rms_after, (rms_before, rms_after)


def test_r8_f3_tiltaware_field_equals_carrier_path():
    """The guarded ``tilt_aware_rays=True`` + explicit carrier field is the
    ``carrier=R`` (no tilt_aware) field to the traced FP floor -- the reroute is
    exact (the per-pixel launch is fully replaced by the carrier gradient)."""
    _skip_if_low_ram(1024)
    R_in, w = 55e-3, 4.0e-3
    E_in, dx, _ = _carrier_input(_TRIPLET, R_in, w, 1024)
    common = dict(prescription=_TRIPLET, wavelength=_WL, dx=dx,
                  ray_subsample=4, n_workers=1, parallel_amp=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        E_carrier = np.asarray(la.apply_real_lens_traced(
            E_in, carrier=R_in, **common))
        E_tilt = np.asarray(la.apply_real_lens_traced(
            E_in, carrier=R_in, tilt_aware_rays=True, **common))
    m = np.abs(E_carrier) > 1e-3 * np.abs(E_carrier).max()
    rel = (float(np.linalg.norm((E_carrier - E_tilt)[m]))
           / (float(np.linalg.norm(E_carrier[m])) + 1e-30))
    assert rel < 1e-9, f"tilt_aware+carrier not routed to carrier path (rel={rel:.2e})"


def test_r8_f3_guard_warns():
    """The F3 reroute emits a RuntimeWarning pointing the user at dropping
    tilt_aware_rays when an explicit carrier is set."""
    _skip_if_low_ram(512)
    R_in, w = 55e-3, 4.0e-3
    E_in, dx, _ = _carrier_input(_TRIPLET, R_in, w, 512)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        la.apply_real_lens_traced(
            E_in, prescription=_TRIPLET, wavelength=_WL, dx=dx,
            ray_subsample=4, n_workers=1, parallel_amp=False,
            carrier=R_in, tilt_aware_rays=True)
    assert any(issubclass(w.category, RuntimeWarning)
               and 'tilt_aware_rays' in str(w.message)
               and 'carrier' in str(w.message) for w in rec), (
        "F3 reroute must warn about tilt_aware_rays + explicit carrier")


def test_r8_f3_collimated_tiltaware_byte_identical(monkeypatch):
    """N5 regression pin (F3 must not touch it): a (near-)collimated tilt_aware
    input (``carrier=None``) is UNCHANGED by the F3 guard -- it fits W == 0, the
    guard never fires, and tilt_aware equals the untouched axial path to the FP
    floor, in both guard states."""
    N, dx, w_L = 512, 8e-6, 3e-3
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E0 = np.exp(-(X ** 2 + Y ** 2) / w_L ** 2).astype(np.complex128)
    common = dict(prescription=_TRIPLET, wavelength=_WL, dx=dx,
                  ray_subsample=4, n_workers=1, parallel_amp=False)

    def _rel(A, B):
        m = np.abs(A) > 1e-3 * np.abs(A).max()
        return (float(np.linalg.norm((A - B)[m]))
                / (float(np.linalg.norm(A[m])) + 1e-30))

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        E_axial = np.asarray(la.apply_real_lens_traced(
            E0, tilt_aware_rays=False, **common))
        E_tilt = np.asarray(la.apply_real_lens_traced(
            E0, tilt_aware_rays=True, **common))
        assert _rel(E_tilt, E_axial) < 1e-12, (
            f"collimated tilt_aware must equal the axial path "
            f"(rel={_rel(E_tilt, E_axial):.2e})")
        # the F3 guard flag must not perturb the collimated path (guard never
        # fires because carrier=None -> W == 0 -> _carrier_W is None).
        monkeypatch.setattr(LT, '_F3_GUARD_TILTAWARE_EXPLICIT_CARRIER', False)
        E_tilt_off = np.asarray(la.apply_real_lens_traced(
            E0, tilt_aware_rays=True, **common))
    assert _rel(E_tilt, E_tilt_off) < 1e-12, (
        f"F3 guard flag perturbed the collimated tilt_aware path "
        f"(rel={_rel(E_tilt, E_tilt_off):.2e})")


# ===========================================================================
# F4.1 -- chain orchestrator (element supplies R_out)
# ===========================================================================
def _singlet(R1, R2, d, glass, ap, name='s'):
    return _build_group([R1, R2], [glass], [d], ap, name=name)


def _qtrace_group_r_out(pres, R_in):
    """Independent per-group wavefront-radius q-trace (carrier_chain_121 style)
    restricted to one group -- the oracle for _paraxial_group_r_out."""
    R = R_in
    surfs = pres['surfaces']
    th = pres['thicknesses']
    for i, s in enumerate(surfs):
        n1 = _n_of(s['glass_before'])
        n2 = _n_of(s['glass_after'])
        R_s = s['radius'] if s['radius'] else np.inf
        inv = 0.0 if np.isinf(R) else 1.0 / R
        pw = 0.0 if np.isinf(R_s) else (n2 - n1) / R_s
        inv_o = (n1 * inv - pw) / n2
        R = np.inf if inv_o == 0.0 else 1.0 / inv_o
        if i < len(surfs) - 1:
            R = R + (th[i] if i < len(th) else 0.0)
    return R


def test_r8_paraxial_r_out_matches_qtrace():
    """The element-supplied R_out (paraxial ABCD Moebius) matches the
    independent wavefront-radius q-trace to full precision -- so the
    orchestrator needs no external q-trace."""
    g = _singlet(60e-3, -60e-3, 4e-3, 'N-BK7', 6e-3)
    tri = _TRIPLET
    for pres in (g, tri):
        for R_in in (80e-3, -45e-3, 153.37e-3, np.inf):
            got = _paraxial_group_r_out(pres, R_in, _WL)
            ref = _qtrace_group_r_out(pres, R_in)
            if np.isinf(ref):
                assert np.isinf(got)
            else:
                assert got == pytest.approx(ref, rel=1e-10, abs=1e-9), (
                    R_in, got, ref)


def _manual_chain(env0, groups, R0, dx0, final_distance, focus_readout=None):
    """The ~30-line manual hand-off pattern the orchestrator packages, using the
    SAME public library calls + the SAME paraxial R_out mechanism."""
    env, R, cdx = env0, R0, dx0
    for g in groups:
        presc = g['prescription']
        gap = g.get('gap_before', 0.0)
        if gap:
            cr = la.propagate_carrier_referenced(env, R, gap, _WL, cdx)
            env, R, cdx = cr.env, cr.R, cr.dx
        E_full = la.carrier_referenced_reconstruct(env, R, _WL, cdx)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            E_exit = np.asarray(la.apply_real_lens_traced(
                E_full, prescription=presc, wavelength=_WL, dx=cdx,
                carrier=R, ray_subsample=4, n_workers=1, parallel_amp=False))
        R_out = _paraxial_group_r_out(presc, R, _WL)
        env = la.carrier_referenced_envelope(E_exit, R_out, _WL, cdx)
        R = R_out
    if focus_readout is not None:
        return np.asarray(la.carrier_referenced_focus_readout(
            env, R, final_distance, _WL, cdx, **focus_readout))
    if final_distance:
        cr = la.propagate_carrier_referenced(env, R, final_distance, _WL, cdx)
        env, R, cdx = cr.env, cr.R, cr.dx
    return np.asarray(la.carrier_referenced_reconstruct(env, R, _WL, cdx))


def test_r8_chain_orchestrator_matches_manual():
    """The orchestrator reproduces the manual carrier hand-off pattern (the
    repro script's ~30 lines) to the FP floor -- the F4.1 acceptance."""
    _skip_if_low_ram(1024)
    N, dx, w0 = 1024, 3e-6, 1.2e-3
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    env0 = np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)
    groups = [
        {'prescription': _singlet(60e-3, -60e-3, 4e-3, 'N-BK7', 6e-3, 'g1'),
         'gap_before': 0.0},
        {'prescription': _singlet(80e-3, -40e-3, 3e-3, 'N-SF2', 6e-3, 'g2'),
         'gap_before': 25e-3},
    ]
    E_manual = _manual_chain(env0, groups, np.inf, dx, 15e-3)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        # v5.29 (audit S8): the chain defaults flipped to the validated
        # carrier-regime configuration; the manual pattern this test pins
        # composes the LEGACY convention, so pass it explicitly (the
        # documented escape hatch).  The flip itself is pinned in
        # test_niche_s8_sphere_carrier_reference.py.
        res = la.propagate_traced_carrier_chain(
            env0, groups, _WL, dx, r_in=np.inf, ray_subsample=4, n_workers=1,
            carrier_reference='parabola',
            traced_kwargs=dict(parallel_amp=False, amplitude_model='screen',
                               preserve_input_phase=True),
            final_distance=15e-3)
    E_orch = np.asarray(res.field)
    m = np.abs(E_orch) > 1e-3 * np.abs(E_orch).max()
    rel = (float(np.linalg.norm((E_orch - E_manual)[m]))
           / (float(np.linalg.norm(E_manual[m])) + 1e-30))
    assert rel < 1e-9, f"orchestrator vs manual mismatch rel={rel:.2e}"
    # per-group stages populated + power conserved through the (lossless,
    # unapertured) chain
    assert len(res.stages) == 2
    for s in res.stages:
        assert np.isfinite(s['R_out']) and s['power'] > 0


def test_r8_chain_orchestrator_final_focus_readout_matches_manual():
    """The orchestrator's near-focus final landing (focus_readout) reproduces
    the manual stop-short + Bluestein-zoom pattern to the FP floor."""
    _skip_if_low_ram(1024)
    N, dx, w0 = 1024, 3e-6, 1.2e-3
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    env0 = np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)
    groups = [
        {'prescription': _singlet(60e-3, -60e-3, 4e-3, 'N-BK7', 6e-3, 'g1'),
         'gap_before': 0.0}]
    fr = dict(dx_out=0.2e-6, N_out=256, standoff=1.0e-3)
    # find the group's exit focus so we land near it
    R_out = _paraxial_group_r_out(groups[0]['prescription'], np.inf, _WL)
    final = -R_out            # exit vertex -> geometric focus
    E_manual = _manual_chain(env0, groups, np.inf, dx, final, focus_readout=fr)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        # v5.29 (audit S8): legacy configuration passed explicitly -- see
        # test_r8_chain_orchestrator_matches_manual above.
        res = la.propagate_traced_carrier_chain(
            env0, groups, _WL, dx, r_in=np.inf, ray_subsample=4, n_workers=1,
            carrier_reference='parabola',
            traced_kwargs=dict(parallel_amp=False, amplitude_model='screen',
                               preserve_input_phase=True),
            final_distance=final,
            focus_readout=fr)
    E_orch = np.asarray(res.field)
    assert E_orch.shape == (256, 256)
    assert res.R is None and res.dx == 0.2e-6
    m = np.abs(E_orch) > 1e-3 * np.abs(E_orch).max()
    rel = (float(np.linalg.norm((E_orch - E_manual)[m]))
           / (float(np.linalg.norm(E_manual[m])) + 1e-30))
    assert rel < 1e-9, f"orchestrator focus readout vs manual rel={rel:.2e}"


def test_r8_chain_orchestrator_guards_input():
    """The orchestrator is a public 2-D-scalar-field entry point: a non-2-D
    input is rejected by the shared ``_check_2d_scalar_field`` first-line guard
    (the 4 dispatcher meta-audits require this)."""
    bad = np.ones(16, dtype=np.complex128)                 # 1-D
    with pytest.raises((TypeError, ValueError)):
        la.propagate_traced_carrier_chain(bad, [], _WL, 1e-6)


# ===========================================================================
# F4.2 -- near-focus landing (stop-short + Bluestein zoom)
# ===========================================================================
def _converging_gaussian(N, dx, w, R):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X ** 2 + Y ** 2) / w ** 2).astype(np.complex128)


def _analytic_focus_fwhm(w, R):
    q_in = 1.0 / (1.0 / R - 1j * _WL / (np.pi * w ** 2))
    q_f = q_in + (-R)                              # at the geometric focus z=-R
    w_f = np.sqrt(-_WL / (np.pi * np.imag(1.0 / q_f)))
    return w_f * np.sqrt(2.0 * np.log(2.0)), w_f


def _spot_metrics(E, dxo, w_ref):
    I = np.abs(E) ** 2
    n = I.shape[0]
    j, i = np.unravel_index(np.argmax(I), I.shape)
    xx = (np.arange(n) - i) * dxo
    yy = (np.arange(n) - j) * dxo
    rr = np.hypot(xx[None, :], yy[:, None])
    nb = int(0.9 * (n // 2))
    ring = np.clip((rr / dxo).astype(int), 0, nb)
    s = np.bincount(ring.ravel(), weights=I.ravel(), minlength=nb + 1)
    c = np.bincount(ring.ravel(), minlength=nb + 1)
    prof = (s[:nb] / np.maximum(c[:nb], 1)) / I[j, i]
    rb = (np.arange(nb) + 0.5) * dxo
    idx = np.where(prof < 0.5)[0]
    fwhm = 2.0 * rb[idx[0]] if len(idx) else np.nan
    far = I[rr > 3.0 * w_ref]
    sec = float(far.max() / I[j, i]) if far.size else 0.0
    offax = float(np.hypot((i - n / 2) * dxo, (j - n / 2) * dxo))
    ee = float(I[rr < 0.45 * n * dxo].sum() / I.sum())
    return fwhm, sec, offax, ee


def test_r8_focus_readout_matches_analytic_no_spurious_spots():
    """The packaged near-focus readout lands a converging Gaussian AT its focus
    with the analytic FWHM, on-axis, halo-complete, and NO spurious secondary
    spots -- the F4.2 acceptance (a sub-lambda 'spot' or off-axis peak is the
    failure the co-moving-grid collapse produced)."""
    N, dx, w, R = 1024, 6e-6, 1.2e-3, -30e-3
    assert N * dx / 2 > 2.4 * w                      # grid holds the beam
    env = _converging_gaussian(N, dx, w, R)
    fwhm_an, w_f = _analytic_focus_fwhm(w, R)
    zf = -R
    E = np.asarray(la.carrier_referenced_focus_readout(
        env, R, zf, _WL, dx, dx_out=0.15e-6, N_out=512))
    assert np.isfinite(E).all()
    fwhm, sec, offax, ee = _spot_metrics(E, 0.15e-6, w_f)
    assert fwhm == pytest.approx(fwhm_an, rel=0.10), (fwhm, fwhm_an)
    assert fwhm > 5.0 * _WL, "spot must be diffraction-limited, not sub-lambda"
    assert offax < 1.0e-6, f"focus must be on-axis (offax={offax*1e6:.2f}um)"
    assert sec < 2e-2, f"spurious secondary spot at {sec:.2e} of peak (halo clip)"
    assert ee > 0.98, f"halo clipped by the output window (EE={ee:.3f})"


def test_r8_focus_readout_survives_exact_focus():
    """fail-before / pass-after: landing EXACTLY at the carrier focus with a
    plain carrier step raises (R_out == 0, the co-moving frame is singular); the
    packaged readout returns a clean finite focused spot at the same plane."""
    N, dx, w, R = 1024, 6e-6, 1.2e-3, -30e-3
    env = _converging_gaussian(N, dx, w, R)
    zf = -R
    with pytest.raises(ValueError):
        la.propagate_carrier_referenced(env, R, zf, _WL, dx)
    E = np.asarray(la.carrier_referenced_focus_readout(
        env, R, zf, _WL, dx, dx_out=0.15e-6, N_out=512))
    assert np.isfinite(E).all() and float(np.abs(E).max()) > 0.0
    fwhm_an, w_f = _analytic_focus_fwhm(w, R)
    fwhm, sec, offax, ee = _spot_metrics(E, 0.15e-6, w_f)
    assert fwhm == pytest.approx(fwhm_an, rel=0.12), (fwhm, fwhm_an)


def test_r8_focus_readout_rejects_bad_args():
    """Guard the readout's grid inputs."""
    N, dx = 256, 6e-6
    env = _converging_gaussian(N, dx, 1.0e-3, -30e-3)
    with pytest.raises(ValueError):
        la.carrier_referenced_focus_readout(
            env, -30e-3, 30e-3, _WL, dx, dx_out=-1.0, N_out=256)
    with pytest.raises(ValueError):
        la.carrier_referenced_focus_readout(
            env, -30e-3, 30e-3, _WL, dx, dx_out=0.15e-6, N_out=256,
            standoff=-1e-3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
