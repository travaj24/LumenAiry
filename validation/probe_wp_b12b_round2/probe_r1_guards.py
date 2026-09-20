"""R1 -- the two GBD local-branch guards (VERIFY-WP-B12b D-5 and D-4).

Run in BOTH trees (PRE = a ``git archive`` of the integration tip 76019ede,
POST = this branch) and on BOTH builds.  The arm is DETECTED from the library.

What it measures
----------------
1. **Premise.**  What each fixture's exit medium resolves to, and what the
   index-free leg therefore omits, in waves -- so no refusal below is
   asserted against an unreachable premise.
2. **D-5, both sides.**  Whether an immersed exit is refused or served at
   five entry points (the beamlet function, ``apply_real_lens_gbd``,
   ``apply_real_lens_universal(method='gbd')``,
   ``propagate_gbd_through_prescription(per_surface=True)``, and the
   ``world_output_plane`` branch, which is deliberately NOT guarded), on the
   verifier's immersed fixture and on my own -- with an AIR control at every
   one of them.
3. **The boundary, bisected AT THE GBD SITE, two-sided.**  ``resolve_exit_index``
   is monkeypatched so the index is a free variable, and the bundle handed to
   the function is a TRIPWIRE whose first attribute access raises -- so the
   bisection runs the real guard at the real call site for the cost of the
   guard, and 1.01x / 0.99x of ``fga._immersed_exit_tolerance``'s own return
   are decided by the shipped code path and not by a copy of its arithmetic.
4. **D-4.**  Whether a mirror-terminated prescription is refused, on the
   verifier's mirror fixture and on mine; what the unguarded branch actually
   returned (spot RMS against a 3-D trace, and the sign of the returned
   direction); and what the ``world_output_plane`` branch does with a CURVED
   and with a FLAT terminating mirror.

Author: Andrew Traverso
"""
from __future__ import annotations

import warnings

import numpy as np
import r2_common as C


# ---------------------------------------------------------------------------
class _Served(Exception):
    """Raised by the tripwire bundle: the guards let this call through."""


class _Tripwire:
    """A beamlet bundle whose first use raises, so a guard decision costs the
    guard and not a ray trace.  ``apply_prescription_persurface_to_beamlets``
    touches ``.positions`` immediately after the guards and before anything
    expensive."""

    @property
    def positions(self):
        raise _Served()


def _guard_decision(G, presc, lam, z_image):
    """``'refused'`` / ``'served'`` at the REAL call site, cheaply."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            G.apply_prescription_persurface_to_beamlets(
                _Tripwire(), presc, lam, z_image=z_image)
    except _Served:
        return 'served', ''
    except NotImplementedError as exc:
        return 'refused', str(exc)
    return 'served', 'returned without touching the bundle'


def _bisect_site(G, ev_mod, presc, lam, z_image, tol_fn):
    """The GBD site's own boundary on ``|n_exit - 1|``, and the two-sided
    decision about the helper's return, both taken through the shipped call."""
    import lumenairy.raytrace.exit_vertex as _ev
    orig = _ev.resolve_exit_index

    def refuses(n):
        _ev.resolve_exit_index = lambda *a, **k: float(n)
        try:
            return _guard_decision(G, presc, lam, z_image)[0] == 'refused'
        finally:
            _ev.resolve_exit_index = orig

    want = float(tol_fn(lam, z_image))
    if not refuses(2.0):
        # the PRE arm: this site carries no guard at all, so there is no
        # boundary to bisect.  Recorded as the reading it is.
        return dict(z_image=z_image, helper_tolerance=want,
                    site_boundary=None, rel_gap=None,
                    refuses_1p01x=False, refuses_0p99x=False,
                    unguarded=True)
    assert not refuses(1.0), 'premise: n_exit = 1 must be served'
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if refuses(1.0 + mid):
            hi = mid
        else:
            lo = mid
    return dict(z_image=z_image,
                helper_tolerance=want,
                site_boundary=hi,
                rel_gap=abs(hi - want) / want if want else float('nan'),
                refuses_1p01x=refuses(1.0 + 1.01 * want),
                refuses_0p99x=refuses(1.0 + 0.99 * want))


# ---------------------------------------------------------------------------
def _entry_decisions(presc, E, dx, lam, z):
    """Refused / served at every public route into the local branch."""
    import lumenairy as la
    from lumenairy.propagators import gbd as G
    out = {}

    def run(label, fn):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                r = fn()
        except NotImplementedError as exc:
            out[label] = dict(decision='refused', kind='NotImplementedError',
                              message=str(exc))
            return
        except ValueError as exc:
            out[label] = dict(decision='refused', kind='ValueError',
                              message=str(exc))
            return
        a = np.asarray(r)
        out[label] = dict(decision='served',
                          digest=C.sha(a) if a.ndim >= 2 else None,
                          shape=list(a.shape) if a.ndim >= 2 else None)

    b = G.decompose_field_to_beamlets(E, dx, wavelength=lam, **C.FRAME)
    run('beamlet_function',
        lambda: G.apply_prescription_persurface_to_beamlets(
            b, presc, lam, z_image=z).positions)
    run('apply_real_lens_gbd',
        lambda: C.gbd_field(presc, E, dx, lam, z))
    run('apply_real_lens_universal_gbd',
        lambda: la.apply_real_lens_universal(
            E, prescription=presc, wavelength=lam, dx=dx,
            output_plane_distance=z, method='gbd',
            method_kwargs={'gbd': dict(C.FRAME)}))
    run('propagate_gbd_through_prescription',
        lambda: G.propagate_gbd_through_prescription(
            E, dx, presc, wavelength=lam, per_surface=True,
            z_image=z, **C.FRAME))
    run('world_output_plane_branch',
        lambda: G.apply_prescription_persurface_to_beamlets(
            b, presc, lam,
            world_output_plane=(np.array([0.0, 0.0, float(z)]),
                                np.eye(3))).positions)
    return out


def _missing_waves(presc, lam, z):
    """``max (n_exit - 1) * z * sec`` in waves, from the LIBRARY's own trace.

    The premise number: what the index-free leg omits on this fixture.  Zero
    on an air control by construction, which is the other side.
    """
    import lumenairy as la
    from lumenairy.raytrace import surfaces_from_prescription
    surfs = [s for s in surfaces_from_prescription(presc)]
    try:
        n_exit = float(la.raytrace.exit_vertex.resolve_exit_index(
            surfs, lam, fn_name='r2'))
    except ValueError:
        return None, None
    semi = float(getattr(surfs[-1], 'semi_diameter', 0.0) or 0.0) or 1e-4
    h = np.linspace(semi / 22.0, semi * 0.98, 11)
    y = np.zeros_like(h)
    nz = np.ones_like(h)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        bundle = la.raytrace.RayBundle(
            x=h.copy(), y=y.copy(), z=np.zeros_like(h),
            L=np.zeros_like(h), M=np.zeros_like(h), N=nz, wavelength=lam,
            alive=np.ones(h.size, bool), opd=np.zeros_like(h))
        s2 = [s for s in surfs]
        res = la.raytrace.trace(bundle, s2, lam)
        img = res.image_rays
    ok = np.asarray(img.alive, bool)
    sec = np.sqrt(1.0 + (np.asarray(img.L)[ok] / np.asarray(img.N)[ok]) ** 2
                  + (np.asarray(img.M)[ok] / np.asarray(img.N)[ok]) ** 2)
    return n_exit, float(np.nanmax(abs(n_exit - 1.0) * abs(z) * sec) / lam)


# ---------------------------------------------------------------------------
def _mirror_truth(fx):
    """The verifier's own 3-D tracer on a mirror fixture: the traced sign of
    ``N`` at the exit vertex and the true spot RMS at the geometric focus."""
    zf = fx.best_focus()
    m = 61
    h = np.linspace(fx.semi / (2 * m), fx.semi * 0.99, m)
    v = fx.exit_rays(h, np.zeros_like(h))
    w = np.exp(-2.0 * (h / fx.w0) ** 2)
    w = w / w.sum()
    ux, uy = v['L'] / v['N'], v['M'] / v['N']
    xz = v['x'] + ux * zf
    yz = v['y'] + uy * zf
    cx = (w * xz).sum()
    cy = (w * yz).sum()
    rms = float(np.sqrt((w * ((xz - cx) ** 2 + (yz - cy) ** 2)).sum()))
    return dict(z_focus=zf, n_sign=float(np.sign(np.median(v['N']))),
                spot_rms=rms)


def _mirror_local_arm(G, presc, E, dx, lam, z):
    """What the LOCAL branch does with a mirror-terminated prescription."""
    b = G.decompose_field_to_beamlets(E, dx, wavelength=lam, **C.FRAME)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = G.apply_prescription_persurface_to_beamlets(
                b, presc, lam, z_image=z)
    except NotImplementedError as exc:
        return dict(decision='refused', message=str(exc))
    p = np.asarray(r.positions)
    d = np.asarray(r.directions)
    a = np.abs(np.asarray(r.amplitude)) ** 2
    s = float(a.sum())
    if not np.isfinite(s) or s <= 0:
        return dict(decision='served', spot_rms=float('nan'),
                    n_sign=float(np.sign(np.median(d[:, 2]))))
    w = a / s
    cx = float((w * p[:, 0]).sum())
    cy = float((w * p[:, 1]).sum())
    rms = float(np.sqrt((w * ((p[:, 0] - cx) ** 2
                              + (p[:, 1] - cy) ** 2)).sum()))
    return dict(decision='served', spot_rms=rms, n_beamlets=int(p.shape[0]),
                n_sign=float(np.sign(np.median(d[:, 2]))))


def _mirror_world_arm(G, presc, E, dx, lam, z):
    b = G.decompose_field_to_beamlets(E, dx, wavelength=lam, **C.FRAME)
    out = {}
    for label, plane in (('auto', 'auto'),
                         ('explicit', (np.array([0.0, 0.0, float(z)]),
                                       np.eye(3)))):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                r = G.apply_prescription_persurface_to_beamlets(
                    b, presc, lam, world_output_plane=plane)
            out[label] = dict(decision='served',
                              n_beamlets=int(np.asarray(r.positions).shape[0]))
        except Exception as exc:                            # noqa: BLE001
            out[label] = dict(decision='refused',
                              kind=type(exc).__name__, message=str(exc)[:240])
    return out


# ---------------------------------------------------------------------------
def main():
    C.assert_tree()
    import lumenairy.raytrace.exit_vertex as EV
    from lumenairy.propagators import fga as F, gbd as G

    arm, tokens = C.detect_arm()
    out = dict(env=C.env_block(), arm=arm, arm_tokens=tokens)
    print(f'arm = {arm}  {tokens}')

    C.register_r2_media()
    vb_imm = C.VB.fixture('immersed')
    vb_air = C.VB.fixture('conic')
    vb_mir = C.VB.fixture('mirror')

    lam_v = vb_imm.lam
    z_v = 2.0e-3                     # the verifier's own D-5 leg
    z_m = 0.60e-3                    # my own, shorter leg

    fixtures = {
        'verifier_immersed_n172': (vb_imm.prescription(), lam_v, z_v,
                                   vb_imm.E_in(), vb_imm.dx),
        'verifier_air_conic': (vb_air.prescription(), lam_v, z_v,
                               vb_air.E_in(), vb_air.dx),
        'mine_immersed_water133': (C.r2_prescription('R2-WATER133'),
                                   C.R2_LAM, z_m, C.r2_input_field(),
                                   C.R2_DX),
        'mine_immersed_oil152': (C.r2_prescription('R2-OIL152'),
                                 C.R2_LAM, z_m, C.r2_input_field(), C.R2_DX),
        'mine_air': (C.r2_prescription('air'), C.R2_LAM, z_m,
                     C.r2_input_field(), C.R2_DX),
    }

    # -- 1. premise ---------------------------------------------------------
    prem = {}
    for key, (presc, lam, z, _E, _dx) in fixtures.items():
        n_exit, waves = _missing_waves(presc, lam, z)
        prem[key] = dict(n_exit=n_exit, z_image=z, wavelength=lam,
                         omitted_waves=waves)
        print(f'  premise {key:28s} n_exit={n_exit!r} omits {waves} waves')
    out['premise'] = prem

    # -- 2. the entry points, both sides ------------------------------------
    ent = {}
    for key, (presc, lam, z, E, dx) in fixtures.items():
        ent[key] = _entry_decisions(presc, E, dx, lam, z)
        print(f'  entries {key:28s} '
              + ' '.join(f'{k}={v["decision"]}' for k, v in ent[key].items()))
    out['entry_points'] = ent

    # -- 3. the boundary, bisected at the GBD site --------------------------
    bis = []
    presc_air = fixtures['mine_air'][0]
    for z in (0.0, C.R2_LAM, 1.0e-5, 3.5e-4, 2.0e-3, 1.0e-2):
        row = _bisect_site(G, EV, presc_air, C.R2_LAM, z,
                           F._immersed_exit_tolerance)
        bis.append(row)
        if row.get('unguarded'):
            print(f'  bisect z={z:.3e} UNGUARDED (pre arm); '
                  f'helper={row["helper_tolerance"]:.6e}')
        else:
            print(f'  bisect z={z:.3e} site={row["site_boundary"]:.6e} '
                  f'helper={row["helper_tolerance"]:.6e} '
                  f'rel={row["rel_gap"]:.2e} '
                  f'1.01x={row["refuses_1p01x"]} 0.99x={row["refuses_0p99x"]}')
    out['site_boundary'] = bis
    out['helper_identity'] = dict(
        gbd_reaches_fga_helper=True,
        tolerance_qualname=F._immersed_exit_tolerance.__qualname__,
        tolerance_module=F._immersed_exit_tolerance.__module__,
        wave_budget=F._FGA_IMAGE_LEG_WAVE_BUDGET,
        noise_floor=F._FGA_EXIT_INDEX_NOISE_FLOOR)

    # -- 4. the mirror ------------------------------------------------------
    mir = {}
    for key, fx, presc, lam, E, dx in (
            ('verifier_mirror_R15', vb_mir, vb_mir.prescription(),
             vb_mir.lam, vb_mir.E_in(), vb_mir.dx),
            ('mine_mirror_R8_curved', None,
             C.r2_prescription(mirror=True, R2=-8.0e-3), C.R2_LAM,
             C.r2_input_field(), C.R2_DX),
            ('mine_mirror_flat', None,
             C.r2_prescription(mirror=True, R2=float('inf')), C.R2_LAM,
             C.r2_input_field(), C.R2_DX)):
        truth = _mirror_truth(fx) if fx is not None else None
        z = (truth['z_focus'] if truth is not None else 4.0e-3)
        row = dict(truth=truth, z_image=z,
                   local=_mirror_local_arm(G, presc, E, dx, lam, z),
                   world=_mirror_world_arm(G, presc, E, dx, lam, z))
        if truth is not None and row['local'].get('spot_rms') is not None \
                and np.isfinite(row['local'].get('spot_rms', np.nan)) \
                and truth['spot_rms'] > 0:
            row['spot_rms_ratio'] = (row['local']['spot_rms']
                                     / truth['spot_rms'])
        mir[key] = row
        print(f'  mirror {key:24s} local={row["local"]["decision"]} '
              f'world_auto={row["world"]["auto"]["decision"]} '
              f'world_explicit={row["world"]["explicit"]["decision"]} '
              f'ratio={row.get("spot_rms_ratio")}')
    out['mirror'] = mir

    C.dump(out, 'probe_r1_guards_' + arm)


if __name__ == '__main__':
    main()
