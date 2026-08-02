# DIAGNOSIS of the halo-check firing on niche D6's exact-tilted-leg RETRACE.
# (ORACLE_ENERGY_AND_D6_HALO_2026_08_01, task B.)
#
# THE REPORT (C6_FIT_GUARD_DECISION_2026_07_31 S5.2, artefact 10):
#   ``amax_halo`` = 6.405e-01 of peak beyond 2.0202 mm against an exact-ray
#   support of 1.6161 mm, ``g_halo`` = 6.449e-04, grid reach 2.4341 mm --
#   a FULL annulus, in a fixture two green tests depend on.
#
# THE TWO HYPOTHESES, and what separates them.
#
#   (A) REAL DEFECT.  The fitted entrance->exit map is Newton-inverted outside
#       its own data support and lands on a SPURIOUS ROOT inside the bright
#       core, so the ray-density amplitude hands real power to a place no ray
#       goes.  Signature: the halo pixels pull back to an entrance point whose
#       INPUT amplitude is a sizeable fraction of the peak -- so the lobe
#       survives with ``amplitude_model='screen'`` (which is |E_in(pullback)|
#       with no ray-density gain at all).
#
#   (B) SUPPORT-DEFINITION ARTEFACT.  ``_rd_hull_r`` is measured over launch
#       nodes whose INPUT amplitude is >= ``e^-_RD_HALO_AMP_CONTOUR`` (= e^-9,
#       the r = 3w contour) of peak.  Rays FAINTER than that are still alive,
#       still traced, and still land somewhere -- and on this fixture the beam
#       is decentred inside a much larger clear aperture, so the far rim of the
#       aperture is populated by genuine (very faint) rays that the contour
#       excluded from the hull.  Their ray-density amplitude is then inflated
#       by |det J| -> 0 at the rim (the fold-caustic warning fires on the same
#       call).  Signature: real alive rays DO reach beyond 1.25 x r_hull, the
#       pullback amplitude is ~1e-5 of peak, and the lobe COLLAPSES under
#       ``amplitude_model='screen'``.
#
# Diffraction is NOT an available third explanation and this is worth stating:
# ``apply_real_lens_traced`` is a pure eikonal operator -- ``E_out`` is
# ``amp(pullback) * exp(i k0 opl_map)`` masked to the ray-covered region, with
# no propagation integral, no FFT and no angular spectrum anywhere in it.  Any
# ``E_out`` sample is nonzero only because the Newton inverse converged there.
#
# usage:  python d6halo_probe.py            (capture + classify)
#         PART=replay python d6halo_probe.py  (the screen / spline replays)
import os
import sys
import warnings

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'tests', 'unit'))

import lumenairy as la                                     # noqa: E402,I001
import lumenairy.elements as _EL                           # noqa: E402
import lumenairy.elements._lens_traced as LT               # noqa: E402
import lumenairy.raytrace as _RT                           # noqa: E402
import test_niche_d6_exact_tilted_leg as D6                # noqa: E402


class Capture(object):
    """Record every ``apply_real_lens_traced`` call of a chain run, together
    with the EXACT ray bundle its own tracer produced (post vertex correction
    and post carrier-eikonal, because the library mutates that object in
    place after ``trace`` returns and we read it afterwards)."""

    def __init__(self):
        self.calls = []
        self._orig = None
        self._otrace = None
        self._pending = None

    def __enter__(self):
        self._orig = _EL.apply_real_lens_traced
        # ``_lens_traced`` does ``from ..raytrace import ... trace`` INSIDE the
        # function, so the binding is resolved per call and patching the
        # raytrace module is what reaches it.
        self._otrace = _RT.trace

        def _tr(rays, surfaces, wavelength, **kw):
            res = self._otrace(rays, surfaces, wavelength, **kw)
            self._pending = {'h_x': np.array(rays.x, copy=True),
                             'h_y': np.array(rays.y, copy=True),
                             'result': res, 'surfaces': surfaces}
            return res

        def _w(E_in, *a, **kw):
            self._pending = None
            with warnings.catch_warnings(record=True) as wl:
                warnings.simplefilter('always')
                out = self._orig(E_in, *a, **kw)
            rec = {'args': a, 'kwargs': dict(kw),
                   'E_in': np.array(E_in, copy=True),
                   'E_out': np.array(out, copy=True),
                   'warns': [str(w.message) for w in wl],
                   'trace': self._pending}
            self.calls.append(rec)
            return out

        _EL.apply_real_lens_traced = _w
        _RT.trace = _tr
        return self

    def __exit__(self, *e):
        _EL.apply_real_lens_traced = self._orig
        _RT.trace = self._otrace
        return False


def _dx_of(rec):
    kw = rec['kwargs']
    if 'dx' in kw:
        return float(kw['dx'])
    # positional: apply_real_lens_traced(E_in, prescription, wavelength, dx)
    for v in rec['args']:
        if isinstance(v, float):
            return float(v)
    raise KeyError('dx')


def _amp_at_nodes(E_in, h_x, dx, n_launch, dy=None):
    """The library's own ``_amp`` (``_lens_traced`` ~L5425): nearest-node
    sample of |E_in| at the launch lattice, in the x-major layout."""
    xs_in = h_x.reshape(n_launch, n_launch)[:, 0]
    dyv = dx if dy is None else dy
    ix = np.clip(np.rint(xs_in / dx + E_in.shape[1] / 2).astype(int),
                 0, E_in.shape[1] - 1)
    iy = np.clip(np.rint(xs_in / dyv + E_in.shape[0] / 2).astype(int),
                 0, E_in.shape[0] - 1)
    return np.abs(E_in)[np.ix_(iy, ix)].T


def analyse(rec, tag=''):
    """Reproduce the library's hull statistic, then relax its two choices one
    at a time: (i) drop the e^-9 amplitude contour, (ii) drop nothing else."""
    tr = rec['trace']
    if tr is None:
        return None
    dx = _dx_of(rec)
    E_in = rec['E_in']
    E_out = rec['E_out']
    fin = tr['result'].image_rays
    n_launch = int(round(np.sqrt(fin.x.size)))
    amp = _amp_at_nodes(E_in, tr['h_x'], dx, n_launch,
                        dy=rec['kwargs'].get('dy'))
    alive = np.asarray(fin.alive, bool).reshape(n_launch, n_launch)
    XE = np.asarray(fin.x).reshape(n_launch, n_launch)
    YE = np.asarray(fin.y).reshape(n_launch, n_launch)
    ok = alive & np.isfinite(XE) & np.isfinite(YE)

    pk = float(amp.max())
    gate = ok & (amp >= np.exp(-LT._RD_HALO_AMP_CONTOUR) * pk)
    w = amp[gate].astype(np.float64) ** 2
    cx = float((XE[gate] * w).sum() / w.sum())
    cy = float((YE[gate] * w).sum() / w.sum())
    r_gate = float(np.hypot(XE[gate] - cx, YE[gate] - cy).max())
    r_all = float(np.hypot(XE[ok] - cx, YE[ok] - cy).max())

    n = E_out.shape[0]
    ax = (np.arange(n) - n / 2) * dx
    rr = np.hypot(ax[None, :] - cx, ax[:, None] - cy)
    aE = np.abs(E_out)
    pkE = float(aE.max())
    p_in = float((np.abs(E_in) ** 2).sum())
    edge = min(float(ax[-1]) - cx, cx - float(ax[0]),
               float(ax[-1]) - cy, cy - float(ax[0]))

    out = {'tag': tag, 'dx': dx, 'n_launch': n_launch, 'N': n,
           'r_gate': r_gate, 'r_all': r_all, 'centroid': (cx, cy),
           'edge': edge, 'peakE': pkE, 'p_in': p_in,
           'n_alive': int(ok.sum()), 'n_gate': int(gate.sum()),
           'amp_peak': pk, 'XE': XE, 'YE': YE, 'ok': ok, 'amp': amp,
           'ax': ax, 'aE': aE, 'rr': rr,
           'fold': any('fold' in t.lower() for t in rec['warns']),
           'halo_warn': any('HALO self-check' in t for t in rec['warns'])}
    for name, rh in (('gate', r_gate), ('all', r_all)):
        b = LT._RD_HALO_RADIUS_FACTOR * rh
        far = rr > b
        out[f'bound_{name}'] = b
        out[f'amax_{name}'] = (float(aE[far].max()) / pkE
                               if far.any() and pkE > 0 else 0.0)
        out[f'g_{name}'] = (float((aE[far] ** 2).sum()) / p_in
                            if far.any() and p_in > 0 else 0.0)
        out[f'nfar_{name}'] = int(far.sum())
    return out


def report(a):
    print(f"\n--- call [{a['tag']}] ---")
    print(f"  grid N={a['N']} dx={a['dx'] * 1e6:.4f} um  "
          f"(half-width {a['ax'][-1] * 1e3:.4f} mm), launch {a['n_launch']}^2")
    print(f"  traced exit centroid ({a['centroid'][0] * 1e3:+.4f}, "
          f"{a['centroid'][1] * 1e3:+.4f}) mm, grid reach {a['edge'] * 1e3:.4f}"
          f" mm")
    print(f"  alive rays {a['n_alive']} of {a['n_launch'] ** 2}; above the "
          f"e^-{LT._RD_HALO_AMP_CONTOUR:g} contour {a['n_gate']}")
    print(f"  r_hull  (e^-9 gate, = the library's) {a['r_gate'] * 1e3:.4f} mm "
          f"-> bound {a['bound_gate'] * 1e3:.4f} mm   "
          f"amax {a['amax_gate']:.3e}  g {a['g_gate']:.3e}  "
          f"pixels {a['nfar_gate']}")
    print(f"  r_hull  (ALL alive rays)            {a['r_all'] * 1e3:.4f} mm "
          f"-> bound {a['bound_all'] * 1e3:.4f} mm   "
          f"amax {a['amax_all']:.3e}  g {a['g_all']:.3e}  "
          f"pixels {a['nfar_all']}")
    print(f"  library warnings: halo={a['halo_warn']}  fold={a['fold']}")


def main():
    part = os.environ.get('PART', 'capture')
    leg = os.environ.get('LEG', 'exact')
    # The WORKING TREE carries an in-flight niche-C8 support bound on the
    # Newton inverse (``REMAP_INVERSE_SUPPORT_BOUND``, default True) that the
    # C7 record predates.  BOUND=0 restores the committed-HEAD behaviour, which
    # is the state the reported 0.641 lobe was measured in.
    bound = os.environ.get('BOUND')
    if bound is not None and hasattr(LT, 'REMAP_INVERSE_SUPPORT_BOUND'):
        LT.REMAP_INVERSE_SUPPORT_BOUND = (bound == '1')
    car = la.TiltedCarrier(np.inf, 0.0, 0.0, D6._X0, 0.0)
    print(f"D6 fixture: w={D6._W * 1e3:.3f} mm  f={D6._F * 1e3:.3f} mm  "
          f"decentre x0={D6._X0 * 1e3:.3f} mm  aperture={D6._APER * 1e3:.3f} mm"
          f"  grid {D6._NGRID} over {D6._EXTENT * 1e3:.3f} mm")
    print(f"halo constants: contour e^-{LT._RD_HALO_AMP_CONTOUR:g}  factor "
          f"{LT._RD_HALO_RADIUS_FACTOR:g}  tol {LT._RD_HALO_AMAX_TOL:.1e}")
    print(f"REMAP_INVERSE_SUPPORT_BOUND = "
          f"{getattr(LT, 'REMAP_INVERSE_SUPPORT_BOUND', '<absent: HEAD>')}"
          f"   leg={leg}")

    with Capture() as cap:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            D6._run_chain(car, final_leg=leg, centre_out=(0.0, 0.0))
    print(f"\n{len(cap.calls)} apply_real_lens_traced calls captured")

    res = []
    for i, rec in enumerate(cap.calls):
        a = analyse(rec, tag=f'{i}')
        if a is None:
            print(f"\n--- call [{i}] --- no trace captured")
            continue
        res.append((i, rec, a))
        report(a)

    # ---- the warning call, in detail -------------------------------------
    hot = [t for t in res if t[2]['halo_warn']]
    if not hot:
        print("\nNO CALL WARNED -- nothing to diagnose on this leg.")
        return 0
    i, rec, a = hot[-1]
    print(f"\n=== the warning call is [{i}] ===")
    for t in rec['warns']:
        print('  [warn] ' + t[:400].replace('\n', ' '))

    # WHICH rays reach beyond the bound, and what do they carry?
    d = np.hypot(a['XE'] - a['centroid'][0], a['YE'] - a['centroid'][1])
    beyond = a['ok'] & (d > a['bound_gate'])
    print(f"\n  ALIVE TRACED RAYS beyond the halo bound "
          f"({a['bound_gate'] * 1e3:.4f} mm): {int(beyond.sum())} of "
          f"{a['n_alive']}")
    print(f"  traced exit map, ALIVE rays: x in "
          f"[{float(a['XE'][a['ok']].min()) * 1e3:+.4f}, "
          f"{float(a['XE'][a['ok']].max()) * 1e3:+.4f}] mm, y in "
          f"[{float(a['YE'][a['ok']].min()) * 1e3:+.4f}, "
          f"{float(a['YE'][a['ok']].max()) * 1e3:+.4f}] mm; max radius from "
          f"the OPTICAL AXIS {float(np.hypot(a['XE'][a['ok']], a['YE'][a['ok']]).max()) * 1e3:.4f}"
          f" mm (aperture radius {D6._APER / 2 * 1e3:.4f} mm)")
    if beyond.any():
        rel = a['amp'][beyond] / a['amp_peak']
        print(f"    their INPUT amplitude / peak: min {rel.min():.3e}  "
              f"median {np.median(rel):.3e}  max {rel.max():.3e}")
        print(f"    they reach out to {float(d[beyond].max()) * 1e3:.4f} mm "
              f"from the traced exit centroid")
        print(f"    their launch radii from the OPTICAL AXIS: "
              f"{float(np.hypot(a['XE'][beyond], a['YE'][beyond]).min()) * 1e3:.4f}"
              f" .. "
              f"{float(np.hypot(a['XE'][beyond], a['YE'][beyond]).max()) * 1e3:.4f}"
              f" mm  (aperture radius {D6._APER / 2 * 1e3:.4f} mm)")

    # WHERE the halo pixels are
    far = a['rr'] > a['bound_gate']
    if far.any():
        aE = a['aE']
        jy, jx = np.unravel_index(int(np.argmax(np.where(far, aE, -1.0))),
                                  aE.shape)
        px, py = float(a['ax'][jx]), float(a['ax'][jy])
        print(f"\n  the BRIGHTEST halo pixel sits at ({px * 1e3:+.4f}, "
              f"{py * 1e3:+.4f}) mm, i.e. {np.hypot(px, py) * 1e3:.4f} mm from "
              f"the OPTICAL AXIS and "
              f"{np.hypot(px - a['centroid'][0], py - a['centroid'][1]) * 1e3:.4f}"
              f" mm from the traced exit centroid")
        pw = np.where(far, aE ** 2, 0.0)
        tw = float(pw.sum())
        print(f"    halo centroid (power-weighted): "
              f"({float((pw * a['ax'][None, :]).sum() / tw) * 1e3:+.4f}, "
              f"{float((pw * a['ax'][:, None]).sum() / tw) * 1e3:+.4f}) mm")
        rax = np.hypot(a['ax'][None, :], a['ax'][:, None])
        print(f"    halo pixels' distance from the OPTICAL AXIS: "
              f"{float(rax[far & (aE > 0)].min()) * 1e3:.4f} .. "
              f"{float(rax[far & (aE > 0)].max()) * 1e3:.4f} mm "
              f"({int((far & (aE > 0)).sum())} nonzero of {int(far.sum())})")

    np.savez_compressed(os.path.join(_HERE, '_d6halo_hot.npz'),
                        E_in=rec['E_in'], E_out=rec['E_out'], dx=a['dx'],
                        XE=a['XE'], YE=a['YE'], ok=a['ok'], amp=a['amp'],
                        centroid=np.array(a['centroid']))

    if part == 'replay':
        replay(rec, a)
    return 0


def replay(rec, a):
    """THE DISCRIMINATOR.  Re-run the identical element call with the
    ray-density gain removed (``amplitude_model='screen'``, i.e. the exit
    magnitude is just |E_in| at the Newton pullback) and with a different
    inverse map (``newton_fit='spline'``).

    Hypothesis (A) predicts the lobe SURVIVES in screen mode (a spurious root
    inside the core carries core amplitude).  Hypothesis (B) predicts it
    COLLAPSES to the genuine rim amplitude, ~1e-5 of peak."""
    print("\n=== REPLAY of the warning call ===")
    base = dict(rec['kwargs'])
    far = a['rr'] > a['bound_gate']
    shipped = None
    variants = [('as shipped (ray_density, remap)', {}),
                ("screen, pip=True  [PULLBACK AMPLITUDE]",
                 {'amplitude_model': 'screen', 'preserve_input_phase': True}),
                ("screen, pip=False [PULLBACK AMPLITUDE]",
                 {'amplitude_model': 'screen', 'preserve_input_phase': False}),
                ("ray_density, pip=True",
                 {'preserve_input_phase': True}),
                ("ray_density, ray_subsample=1",
                 {'ray_subsample': 1}),
                ("ray_density, fit_radius_beam_factor=1.0",
                 {'fit_radius_beam_factor': 1.0}),
                ("ray_density, fit_radius_beam_factor=3.0",
                 {'fit_radius_beam_factor': 3.0})]
    if hasattr(LT, 'REMAP_INVERSE_SUPPORT_BOUND'):
        variants.append(('as shipped + C8 support bound', {'__c8__': True}))
    for name, over in variants:
        kw = dict(base)
        c8 = kw.pop('__c8__', None) or over.pop('__c8__', None)
        kw.update(over)
        old = getattr(LT, 'REMAP_INVERSE_SUPPORT_BOUND', None)
        if c8 and old is not None:
            LT.REMAP_INVERSE_SUPPORT_BOUND = True
        try:
            with warnings.catch_warnings(record=True) as wl:
                warnings.simplefilter('always')
                out = LT.apply_real_lens_traced(rec['E_in'], *rec['args'], **kw)
        except Exception as exc:                       # noqa: BLE001
            print(f"  {name:38s} -> {type(exc).__name__}: {exc}")
            continue
        finally:
            if c8 and old is not None:
                LT.REMAP_INVERSE_SUPPORT_BOUND = old
        aE = np.abs(np.asarray(out))
        pk = float(aE.max())
        amax = float(aE[far].max()) / pk if far.any() and pk > 0 else 0.0
        g = (float((aE[far] ** 2).sum()) / a['p_in'] if far.any() else 0.0)
        nw = sum('HALO self-check' in str(w.message) for w in wl)
        if shipped is None:
            shipped = aE
            jy, jx = np.unravel_index(
                int(np.argmax(np.where(far, aE, -1.0))), aE.shape)
        nz = int((aE[far] > 0).sum())
        print(f"  {name:38s} -> amax_halo {amax:.3e}   g_halo {g:.3e}   "
              f"peak {pk:.4e}   warns {nw}")
        print(f"  {'':38s}    at the shipped hot pixel |E| = "
              f"{float(aE[jy, jx]):.6e}   nonzero far pixels {nz}")


if __name__ == '__main__':
    sys.exit(main())
