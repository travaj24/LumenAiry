# CONTROLS for the niche-D6 halo verdict.
# (ORACLE_ENERGY_AND_D6_HALO_2026_08_01, task B.)
#
# ``d6halo_probe.py`` localises the lobe and shows it moves with two purely
# NUMERICAL knobs.  This script supplies the controls that turn that into a
# verdict:
#
#   A  WHICH LEG.  The C7 record attributes the firing to "the exact-tilted-leg
#      RETRACE".  Run both legs with the C8 support bound off and see which call
#      actually warns.
#   B  WHAT CREATES IT.  Sweep the carrier DECENTRE from 0 to the fixture's
#      0.6 mm on the same optic, the same aperture, the same grid.  On axis the
#      entrance->exit map is radial and the fitted inverse has nothing to fold;
#      the lobe must be absent there and grow with the decentre.
#   C  WHAT DIFFRACTION WOULD PERMIT.  The innocent explanation is that light
#      may legitimately sit beyond a GEOMETRIC support because the true field
#      diffracts.  Two independent bounds are computed: the incident amplitude
#      at the aperture rim that casts the shadow edge in question (the boundary
#      -diffraction wave cannot exceed it), and the same element call at
#      ``ray_subsample=1``, where the physics is unchanged and the numerics are
#      not.
#   D  WHAT THE GREEN TESTS SEE.  Every quantity the two dependent tests assert,
#      measured with the lobe present and absent.
#
# usage:  PART=A|B|C|D|all  python d6halo_controls.py
import os
import sys
import warnings

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'tests', 'unit'))

import lumenairy as la                                    # noqa: E402,I001
import lumenairy.elements._lens_traced as LT              # noqa: E402
import test_niche_d6_exact_tilted_leg as D6               # noqa: E402
from d6halo_probe import Capture, analyse                 # noqa: E402

_HAS_C8 = hasattr(LT, 'REMAP_INVERSE_SUPPORT_BOUND')


def _set_c8(on):
    if _HAS_C8:
        LT.REMAP_INVERSE_SUPPORT_BOUND = bool(on)


def _run(leg, x0, c8=False, **kw):
    _set_c8(c8)
    car = la.TiltedCarrier(np.inf, 0.0, 0.0, float(x0), 0.0)
    with Capture() as cap:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res, _c = D6._run_chain(car, final_leg=leg, centre_out=(0.0, 0.0),
                                    **kw)
    return res, cap.calls


def part_A():
    print("\n=== A.  WHICH CALL WARNS (C8 support bound OFF = committed HEAD) "
          "===")
    for leg in ('paraxial', 'exact'):
        _res, calls = _run(leg, D6._X0, c8=False)
        print(f"\n  final_leg={leg!r}: {len(calls)} element call(s)")
        for i, rec in enumerate(calls):
            a = analyse(rec, tag=f'{leg}[{i}]')
            if a is None:
                continue
            declined = a['bound_gate'] > a['edge']
            print(f"    call {i}: N={a['N']} dx={a['dx'] * 1e6:.4f} um "
                  f"half-width {a['ax'][-1] * 1e3:.4f} mm  "
                  f"r_hull {a['r_gate'] * 1e3:.4f} mm  bound "
                  f"{a['bound_gate'] * 1e3:.4f} mm  grid reach "
                  f"{a['edge'] * 1e3:.4f} mm  "
                  f"{'DECLINED (no genuine annulus)' if declined else 'scored'}")
            print(f"             amax {a['amax_gate']:.3e}  g "
                  f"{a['g_gate']:.3e}  library halo warning="
                  f"{a['halo_warn']}  fold={a['fold']}")


def part_B():
    print("\n=== B.  DECENTRE SWEEP (paraxial leg, C8 OFF).  Same optic, same "
          "aperture, same grid; only the carrier decentre moves. ===")
    print(f"  {'x0 [mm]':>9} {'x0/w':>6} {'r_hull[mm]':>11} {'bound[mm]':>10} "
          f"{'reach[mm]':>10} {'amax_halo':>11} {'g_halo':>11} "
          f"{'nonzero px':>11} {'warn':>5}")
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        x0 = frac * D6._X0
        _res, calls = _run('paraxial', x0, c8=False)
        a = analyse(calls[0], tag=f'x0={x0}')
        far = a['rr'] > a['bound_gate']
        nz = int((a['aE'][far] > 0).sum()) if far.any() else 0
        print(f"  {x0 * 1e3:9.4f} {x0 / D6._W:6.2f} {a['r_gate'] * 1e3:11.4f} "
              f"{a['bound_gate'] * 1e3:10.4f} {a['edge'] * 1e3:10.4f} "
              f"{a['amax_gate']:11.3e} {a['g_gate']:11.3e} {nz:11d} "
              f"{str(a['halo_warn']):>5}", flush=True)


def part_C():
    print("\n=== C.  WHAT DIFFRACTION WOULD PERMIT THERE ===")
    _res, calls = _run('paraxial', D6._X0, c8=False)
    rec = calls[0]
    a = analyse(rec, tag='C')
    E_in = rec['E_in']
    n = E_in.shape[0]
    dx = a['dx']
    ax = (np.arange(n) - n / 2) * dx
    X, Y = np.meshgrid(ax, ax)
    rax = np.hypot(X, Y)
    ap_r = D6._APER / 2.0
    print(f"  1. The element operator itself.  apply_real_lens_traced builds "
          f"E_out = amp(pullback) * exp(i k0 opl) and masks it; there is no "
          f"propagation integral, no FFT and no angular spectrum in it, so it "
          f"carries NO diffraction at all.  Its exact ray map spans "
          f"|r| <= {float(np.hypot(a['XE'][a['ok']], a['YE'][a['ok']]).max()) * 1e3:.4f}"
          f" mm about the axis.")
    # the shadow edge on the halo's side is cast by the -x aperture rim
    rim = (np.abs(rax - ap_r) < 1.5 * dx) & (X < 0) & (np.abs(Y) < 5 * dx)
    a_rim = float(np.abs(E_in)[rim].max()) if rim.any() else 0.0
    a_pk = float(np.abs(E_in).max())
    print(f"  2. The boundary-diffraction (Young) wave.  The shadow edge on "
          f"the halo's side (-x) is cast by the aperture rim at "
          f"x = -{ap_r * 1e3:.3f} mm, where the INCIDENT amplitude is "
          f"{a_rim:.4e} = {a_rim / a_pk:.4e} of peak.")
    print(f"     A boundary wave cannot exceed the incident amplitude at its "
          f"own edge, so the physically permitted amplitude beyond that "
          f"shadow is <= {a_rim / a_pk:.3e} of peak.")
    print(f"     MEASURED lobe: {a['amax_gate']:.4e} of peak -- "
          f"{a['amax_gate'] / max(a_rim / a_pk, 1e-300):.3e} times the "
          f"ceiling.")
    # the far Gaussian tail at the halo location itself
    hot = a['rr'] > a['bound_gate']
    tail = float(np.abs(E_in)[hot].max()) / a_pk if hot.any() else 0.0
    print(f"  3. The incident field AT the halo pixels is "
          f"{tail:.4e} of peak, so even a perfect (diffraction-free) screen "
          f"model would put {tail:.3e} there.  The measured lobe is "
          f"{a['amax_gate'] / max(tail, 1e-300):.3e} times THAT.")
    print("  4. The numerical control: the same call at ray_subsample=1 "
          "(physics identical, discretisation finer) -- see "
          "d6halo_probe.py's replay: amax_halo = 0.000e+00 exactly.")


def _metrics_row(F):
    m = D6._metrics(np.asarray(F))
    return m


def part_D():
    print("\n=== D.  WHAT THE TWO DEPENDENT GREEN TESTS ACTUALLY SEE ===")
    if not _HAS_C8:
        print("  (no REMAP_INVERSE_SUPPORT_BOUND in this tree -- only the "
              "defect-present column is available)")
    orc = D6._metrics(D6._oracle_on_grid((0.0, 0.0), x0=D6._X0))
    env, dxl = D6._launch()
    p_in = float(np.sum(np.abs(env) ** 2)) * dxl * dxl
    print(f"  oracle: FWHM {orc['fwhm'] * 1e6:.4f} um  EE2 {orc['ee'][2.0]:.6f}"
          f"  EE4 {orc['ee'][4.0]:.6f}   P_in {p_in:.6e}")
    rows = {}
    for c8 in ((False, True) if _HAS_C8 else (False,)):
        ex, _ = _run('exact', D6._X0, c8=c8)
        px, _ = _run('paraxial', D6._X0, c8=c8)
        m_ex = _metrics_row(ex.field)
        m_px = _metrics_row(px.field)
        p_ex = [s['power'] for s in ex.stages if 'power' in s][-1]
        p_px = [s['power'] for s in px.stages if 'power' in s][-1]
        rows[c8] = dict(m_ex=m_ex, m_px=m_px, p_ex=p_ex, p_px=p_px)
        tag = 'lobe REMOVED (C8 on)' if c8 else 'lobe PRESENT (C8 off = HEAD)'
        print(f"\n  --- {tag} ---")
        print(f"    exact   FWHM {m_ex['fwhm'] * 1e6:.4f} um  "
              f"EE2 {m_ex['ee'][2.0]:.6f}  EE4 {m_ex['ee'][4.0]:.6f}  "
              f"peak_off_x {m_ex['peak_off'][0] * 1e6:+.4f} um  "
              f"|centroid| {np.hypot(*m_ex['centroid']) * 1e6:.4f} um")
        print(f"    paraxial FWHM {m_px['fwhm'] * 1e6:.4f} um  "
              f"EE2 {m_px['ee'][2.0]:.6f}  EE4 {m_px['ee'][4.0]:.6f}  "
              f"peak_off_x {m_px['peak_off'][0] * 1e6:+.4f} um")
        print(f"    stage power: exact {p_ex:.8e}  paraxial {p_px:.8e}  "
              f"p_ex/p_in {p_ex / p_in:.8f}  p_ex/p_px {p_ex / p_px:.8f}")
        # every assertion of the two tests, evaluated
        checks = [
            ('|FWHM_ex/FWHM_orc - 1| < 0.15',
             abs(m_ex['fwhm'] / orc['fwhm'] - 1.0), 0.15, 'lt'),
            ('EE2_ex > 0.90 EE2_orc',
             m_ex['ee'][2.0] / orc['ee'][2.0], 0.90, 'gt'),
            ('EE4_ex > 0.97 EE4_orc',
             m_ex['ee'][4.0] / orc['ee'][4.0], 0.97, 'gt'),
            ('FWHM_px > 1.70 FWHM_orc',
             m_px['fwhm'] / orc['fwhm'], 1.70, 'gt'),
            ('EE2_px < 0.25 EE2_orc',
             m_px['ee'][2.0] / orc['ee'][2.0], 0.25, 'lt'),
            ('|peak_off_x_px| > 4 um',
             abs(m_px['peak_off'][0]) * 1e6, 4.0, 'gt'),
            ('|peak_off_x_ex| < 0.5 um',
             abs(m_ex['peak_off'][0]) * 1e6, 0.5, 'lt'),
            ('FWHM_ex < 0.60 FWHM_px',
             m_ex['fwhm'] / m_px['fwhm'], 0.60, 'lt'),
            ('EE2_ex > 5 EE2_px',
             m_ex['ee'][2.0] / m_px['ee'][2.0], 5.0, 'gt'),
            ('|centroid_ex| < 0.5 um',
             np.hypot(*m_ex['centroid']) * 1e6, 0.5, 'lt'),
            ('0.95 < p_ex/p_in < 1.02', p_ex / p_in, None, 'band'),
            ('|p_ex/p_px - 1| < 0.02', abs(p_ex / p_px - 1.0), 0.02, 'lt'),
        ]
        for name, val, lim, kind in checks:
            if kind == 'band':
                ok = 0.95 < val < 1.02
            elif kind == 'lt':
                ok = val < lim
            else:
                ok = val > lim
            print(f"      {'PASS' if ok else 'FAIL'}  {name:34s} "
                  f"measured {val:.6f}")
    if len(rows) == 2:
        A, B = rows[False], rows[True]
        print("\n  --- the DEFECT'S influence on each asserted quantity ---")
        print(f"    exact FWHM   {A['m_ex']['fwhm'] * 1e6:.6f} -> "
              f"{B['m_ex']['fwhm'] * 1e6:.6f} um   "
              f"delta {((B['m_ex']['fwhm'] / A['m_ex']['fwhm']) - 1) * 100:+.4f} %")
        for r in (2.0, 4.0, 6.0):
            print(f"    exact EE{r:.0f}    {A['m_ex']['ee'][r]:.8f} -> "
                  f"{B['m_ex']['ee'][r]:.8f}   "
                  f"delta {(B['m_ex']['ee'][r] - A['m_ex']['ee'][r]) * 100:+.6f} pts")
        for r in (2.0, 4.0, 6.0):
            print(f"    paraxial EE{r:.0f} {A['m_px']['ee'][r]:.8f} -> "
                  f"{B['m_px']['ee'][r]:.8f}   "
                  f"delta {(B['m_px']['ee'][r] - A['m_px']['ee'][r]) * 100:+.6f} pts")
        print(f"    paraxial FWHM {A['m_px']['fwhm'] * 1e6:.6f} -> "
              f"{B['m_px']['fwhm'] * 1e6:.6f} um")
        print(f"    stage power exact    {A['p_ex']:.10e} -> {B['p_ex']:.10e}"
              f"   delta {((B['p_ex'] / A['p_ex']) - 1) * 100:+.6f} %")
        print(f"    stage power paraxial {A['p_px']:.10e} -> {B['p_px']:.10e}"
              f"   delta {((B['p_px'] / A['p_px']) - 1) * 100:+.6f} %")


def main():
    part = os.environ.get('PART', 'all')
    print(f"D6 controls.  REMAP_INVERSE_SUPPORT_BOUND present in this tree: "
          f"{_HAS_C8}")
    if part in ('A', 'all'):
        part_A()
    if part in ('B', 'all'):
        part_B()
    if part in ('C', 'all'):
        part_C()
    if part in ('D', 'all'):
        part_D()
    return 0


if __name__ == '__main__':
    sys.exit(main())
