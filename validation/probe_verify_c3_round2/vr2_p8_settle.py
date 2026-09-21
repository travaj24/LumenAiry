"""VERIFY-WP-C3 ROUND 2, item 4 -- SETTLE the p8 capstone's open question.

``tests/unit/test_niche_p8_capstone.py::test_stepB_composed_doublet_relay_
matches_debye`` reads EE80 = 12.3588 um on ``transport='collins'`` against
11.0139 um on ``'sziklas'``, and the test's ring-Huygens Debye oracle backs
the co-moving column.  Round 2 could not say whether the Collins FIELD is
wrong or only the LATTICE it returns is too coarse to measure EE80 on (its
pitch floors at 2 r_out/N = 6.3126 um, which puts ~2 samples inside EE80).

This probe separates the two, four independent ways:

  (1) the two arms' shared prefix is checked for bit-identity, so the ONLY
      difference is the final leg;
  (2) both final-leg envelopes are resampled onto ONE common fine lattice by
      EXACT band-limited (full-N, separable non-uniform inverse DFT)
      interpolation -- no cropping, no windowing -- and EE50/EE80 are
      recomputed there with the test's own metric;
  (3) the Collins final leg is re-run with an explicit fine ``dx_out``, which
      is the same transport evaluating the same leg on a lattice that can
      resolve the spot;
  (4) (3) is swept over a pitch ladder, so a converging sequence separates a
      sampling artefact from a wrong field.

Run with cwd = the tree root, PYTHONPATH = the tree root, VC3_TREE = the tree
root, VC3_OUT = the output directory, VC3_TAG = a tag.
"""
import importlib.util
import json
import os
import pathlib
import sys
import warnings

import numpy as np

TREE = os.path.abspath(os.environ['VC3_TREE'])
sys.path.insert(0, TREE)
import lumenairy as la                                       # noqa: E402
import lumenairy.propagators.carrier as C                    # noqa: E402

assert os.path.abspath(la.__file__).startswith(TREE), (la.__file__, TREE)

_P8 = pathlib.Path(TREE) / 'tests' / 'unit' / 'test_niche_p8_capstone.py'
_spec = importlib.util.spec_from_file_location('p8mod', _P8)
P8 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P8)
from lumenairy import glass as _g                            # noqa: E402
_g.GLASS_REGISTRY.update(P8.MODULE_GLASSES)

WL = P8._WL
N, DX = 4096, 2.0e-6


def chain_prefix():
    """Everything up to (and including) the relay -- returns env3, R3, dx2."""
    ap, w0 = 6.0e-3, 1.5e-3
    D = dict(R1=90e-3, R2=-60e-3, R3=-350e-3, d1=4e-3, d2=2e-3)
    GAP = 50e-3
    RL = dict(R1=41.3e-3, R2=-41.3e-3, d=3e-3)
    doublet = {'name': 'g1', 'aperture_diameter': ap, 'surfaces': [
        {'radius': D['R1'], 'conic': 0.0, 'aspheric_coeffs': None,
         'glass_before': 'air', 'glass_after': '_P8A'},
        {'radius': D['R2'], 'conic': 0.0, 'aspheric_coeffs': None,
         'glass_before': '_P8A', 'glass_after': '_P8B'},
        {'radius': D['R3'], 'conic': 0.0, 'aspheric_coeffs': None,
         'glass_before': '_P8B', 'glass_after': 'air'}],
        'thicknesses': [D['d1'], D['d2']]}
    relay = P8._singlet_presc(RL['R1'], RL['R2'], RL['d'], '_P8A', ap)
    full = (P8._surfs_mm_doublet(D['R1'], D['R2'], D['R3'], D['d1'], D['d2'],
                                 P8._NA, P8._NB, GAP)
            + P8._surfs_mm_singlet(RL['R1'], RL['R2'], RL['d'], P8._NA))
    img_mm, _ = P8._paraxial_img_efl(full)

    E0 = P8._gauss(N, DX, w0, None)
    E1 = np.asarray(la.apply_real_lens_traced(
        E0, prescription=doublet, wavelength=WL, dx=DX,
        on_undersample='warn'))
    R1 = la.carrier_referenced_fit_radius(E1, WL, DX)
    env1 = la.carrier_referenced_envelope(E1, R1, WL, DX)
    out = {}
    for tr in ('sziklas', 'collins'):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            env2, R2, dx2 = la.propagate_carrier_referenced(
                env1, R1, GAP, WL, DX, transport=tr)
        E_relay = np.asarray(la.carrier_referenced_reconstruct(
            np.asarray(env2), R2, WL, dx2))
        E3, method = la.apply_real_lens_universal(
            E_relay, prescription=relay, wavelength=WL, dx=dx2,
            return_method=True)
        E3 = np.asarray(E3)
        R3 = la.carrier_referenced_fit_radius(E3, WL, dx2)
        env3 = la.carrier_referenced_envelope(E3, R3, WL, dx2)
        out[tr] = dict(env3=env3, R3=R3, dx2=dx2, method=method,
                       gap_dx=dx2, gap_R=R2,
                       gap_warn=[str(w.message)[:110] for w in rec])
    return out, full, img_mm, ap, w0


def nudft_resample(env, dx_in, xo):
    """EXACT band-limited resampling of ``env`` onto the 1-D lattice ``xo``.

    ``env`` is a uniformly sampled (n x n) array of pitch ``dx_in`` centred on
    index n//2.  Its band-limited interpolant is the inverse DFT evaluated at
    arbitrary coordinates; this evaluates it with a separable non-uniform
    inverse DFT over ALL n coefficients (no cropping, no apodisation).
    """
    n = env.shape[0]
    F = np.fft.fft2(env) / (n * n)
    kk = np.fft.fftfreq(n, d=dx_in)                 # cycles / m
    x_in0 = -(n // 2) * dx_in                       # index 0 coordinate
    # the DFT was taken on index coordinates, so shift to physical ones
    ph = np.exp(2j * np.pi * np.outer(xo - x_in0, kk))
    return ph @ F @ ph.T


def ee(I, dx, win=200e-6):
    return P8._ee_metrics(I, dx, win=win)


def main():
    out = {'tree': TREE, 'lumenairy': la.__file__,
           'python': sys.version.split()[0], 'numpy': np.__version__,
           'default_transport': C.propagate_carrier_referenced.
           __defaults__ and None}
    import inspect
    out['default_transport'] = inspect.signature(
        C.propagate_carrier_referenced).parameters['transport'].default

    pre, full, img_mm, ap, w0 = chain_prefix()
    s, c = pre['sziklas'], pre['collins']
    out['prefix'] = {
        'method_sziklas': s['method'], 'method_collins': c['method'],
        'gap_dx_sziklas_um': s['dx2'] * 1e6,
        'gap_dx_collins_um': c['dx2'] * 1e6,
        'gap_R_sziklas': repr(s['gap_R']), 'gap_R_collins': repr(c['gap_R']),
        'R3_sziklas': s['R3'], 'R3_collins': c['R3'],
        'env3_bit_identical': bool(np.array_equal(s['env3'], c['env3'])),
        'env3_max_abs_diff': float(np.max(np.abs(
            np.asarray(s['env3']) - np.asarray(c['env3'])))),
        'gap_warn_collins': c['gap_warn']}

    z_img = img_mm * 1e-3
    env3, R3, dx2 = s['env3'], s['R3'], s['dx2']
    out['final_leg_geometry'] = {'z_img_m': z_img, 'R3': R3,
                                 'A': 1.0 + z_img / R3, 'dx_in_um': dx2 * 1e6}

    legs = {}
    for tr in ('sziklas', 'collins'):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            env4, R4, dx4 = la.propagate_carrier_referenced(
                env3, R3, z_img, WL, dx2, transport=tr)
        env4 = np.asarray(env4)
        dx4f = float(dx4 if not isinstance(dx4, tuple) else dx4[0])
        r2m, e50, e80 = ee(np.abs(env4) ** 2, dx4f)
        legs[tr] = {'dx_out_um': dx4f * 1e6, 'R_out': repr(R4),
                    'r2m_um': r2m * 1e6, 'EE50_um': e50 * 1e6,
                    'EE80_um': e80 * 1e6,
                    'peak': float(np.max(np.abs(env4) ** 2)),
                    'power': float(np.sum(np.abs(env4) ** 2) * dx4f ** 2),
                    'warnings': [str(w.message)[:140] for w in rec]}
        legs[tr]['_env'] = env4
        legs[tr]['_dx'] = dx4f
    out['as_shipped'] = {k: {kk: vv for kk, vv in v.items()
                             if not kk.startswith('_')}
                         for k, v in legs.items()}

    # ---- the ORACLE ------------------------------------------------------
    job = P8._oracle_job(full, WL, ap * 1e3, img_mm, w0 * 1e3, R_in_mm=None,
                         n_fan=6000, n_rho=3000, rho_max_um=200.0,
                         window_um=200.0)
    d = P8._oracle.evaluate(job)
    out['oracle'] = {'method': d['huy_method'], 'EE50_um': d['huy_EE50_um'],
                     'EE80_um': d['huy_EE80_um']}

    # ---- (2) a COMMON fine lattice, exact band-limited resampling --------
    # The band-limited interpolant of a sampled array is PERIODIC with period
    # ``n * dx``, so the common lattice may never leave the smaller array's own
    # extent.  Size it from the measured half-extents.
    half = min(0.5 * legs[tr]['_env'].shape[0] * legs[tr]['_dx']
               for tr in ('sziklas', 'collins'))
    DXC = 0.5e-6
    WIN = min(200e-6, 0.78 * half)
    NC = int(2 * min(half, WIN * 1.28) / DXC) // 2 * 2
    out['common_lattice_sizing'] = {
        'half_extent_um': half * 1e6, 'win_um': WIN * 1e6, 'NC': NC,
        'extent_sziklas_um': legs['sziklas']['_env'].shape[0]
        * legs['sziklas']['_dx'] * 1e6,
        'extent_collins_um': legs['collins']['_env'].shape[0]
        * legs['collins']['_dx'] * 1e6}
    xo = (np.arange(NC) - NC // 2) * DXC
    common = {}
    for tr in ('sziklas', 'collins'):
        f = nudft_resample(legs[tr]['_env'], legs[tr]['_dx'], xo)
        I = np.abs(f) ** 2
        r2m, e50, e80 = ee(I, DXC, win=WIN)
        common[tr] = {'r2m_um': r2m * 1e6, 'EE50_um': e50 * 1e6,
                      'EE80_um': e80 * 1e6,
                      'peak': float(I.max()),
                      'power_in_window': float(I.sum() * DXC ** 2)}
        common[tr + '_I'] = I
    # a direct field-to-field comparison on the common lattice
    A = common['sziklas_I']
    B = common['collins_I']
    out['common_lattice'] = {
        'NC': NC, 'dx_um': DXC * 1e6, 'win_um': WIN * 1e6,
        'sziklas': {k: v for k, v in common['sziklas'].items()},
        'collins': {k: v for k, v in common['collins'].items()},
        'intensity_relL2': float(np.linalg.norm(B - A) / np.linalg.norm(A)),
        'intensity_relL2_scalefree': float(
            np.linalg.norm(B / B.max() - A / A.max())
            / np.linalg.norm(A / A.max()))}

    # ---- (3)+(4) the Collins leg on a NAMED fine lattice, refined --------
    ladder = []
    for dxo in (0.25e-6, 0.5e-6, 1.0e-6, 2.0e-6, 3.0e-6, 6.3126e-6):
        try:
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter('always')
                e4, r4, d4 = la.propagate_carrier_referenced(
                    env3, R3, z_img, WL, dx2, transport='collins',
                    dx_out=dxo, on_collins_sampling='warn')
            e4 = np.asarray(e4)
            d4f = float(d4 if not isinstance(d4, tuple) else d4[0])
            r2m, e50, e80 = ee(np.abs(e4) ** 2, d4f)
            ladder.append({'dx_out_um': dxo * 1e6, 'dx_returned_um': d4f * 1e6,
                           'R_out': repr(r4), 'r2m_um': r2m * 1e6,
                           'EE50_um': e50 * 1e6, 'EE80_um': e80 * 1e6,
                           'peak': float(np.max(np.abs(e4) ** 2)),
                           'power': float(np.sum(np.abs(e4) ** 2) * d4f ** 2),
                           'warnings': [str(w.message)[:140] for w in rec]})
        except Exception as exc:                              # noqa: BLE001
            ladder.append({'dx_out_um': dxo * 1e6,
                           'raised': type(exc).__name__ + ': '
                           + str(exc)[:200]})
    out['collins_named_lattice_ladder'] = ladder

    # the same ladder for 'sziklas', as the control
    ladder_s = []
    for dxo in (0.25e-6, 0.5e-6, 1.0e-6):
        try:
            e4, r4, d4 = la.propagate_carrier_referenced(
                env3, R3, z_img, WL, dx2, transport='sziklas', dx_out=dxo)
            e4 = np.asarray(e4)
            d4f = float(d4 if not isinstance(d4, tuple) else d4[0])
            r2m, e50, e80 = ee(np.abs(e4) ** 2, d4f)
            ladder_s.append({'dx_out_um': dxo * 1e6,
                             'dx_returned_um': d4f * 1e6,
                             'EE50_um': e50 * 1e6, 'EE80_um': e80 * 1e6})
        except Exception as exc:                              # noqa: BLE001
            ladder_s.append({'dx_out_um': dxo * 1e6,
                             'raised': type(exc).__name__ + ': '
                             + str(exc)[:160]})
    out['sziklas_named_lattice_ladder'] = ladder_s

    tag = os.environ.get('VC3_TAG', 'x')
    p = os.path.join(os.environ['VC3_OUT'], f'vr2_p8_settle_{tag}.json')
    with open(p, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1, default=repr)

    print('prefix identical (env3):', out['prefix']['env3_bit_identical'],
          '| max|diff| =', out['prefix']['env3_max_abs_diff'])
    print('gap dx: sziklas', out['prefix']['gap_dx_sziklas_um'],
          'collins', out['prefix']['gap_dx_collins_um'])
    print('final leg A =', out['final_leg_geometry']['A'])
    print('ORACLE  EE50 = %.4f um  EE80 = %.4f um'
          % (d['huy_EE50_um'], d['huy_EE80_um']))
    for tr in ('sziklas', 'collins'):
        a = out['as_shipped'][tr]
        print(f"  AS SHIPPED {tr:8s} pitch {a['dx_out_um']:9.4f} um  "
              f"EE50 {a['EE50_um']:8.4f}  EE80 {a['EE80_um']:8.4f}  "
              f"peak {a['peak']:.6g}  power {a['power']:.6g}")
    for tr in ('sziklas', 'collins'):
        a = out['common_lattice'][tr]
        print(f"  COMMON     {tr:8s} pitch {DXC*1e6:9.4f} um  "
              f"EE50 {a['EE50_um']:8.4f}  EE80 {a['EE80_um']:8.4f}  "
              f"peak {a['peak']:.6g}")
    print('  common-lattice intensity relL2 =',
          out['common_lattice']['intensity_relL2'],
          '| scale-free', out['common_lattice']['intensity_relL2_scalefree'])
    print('  COLLINS named-lattice ladder:')
    for r in ladder:
        print('   ', r if 'raised' in r else
              (f"dx_out {r['dx_out_um']:7.4f} -> returned "
               f"{r['dx_returned_um']:7.4f}  EE50 {r['EE50_um']:8.4f}  "
               f"EE80 {r['EE80_um']:8.4f}  warn {len(r['warnings'])}"))
    print('  SZIKLAS named-lattice ladder:')
    for r in ladder_s:
        print('   ', r)
    print('WROTE', p)


if __name__ == '__main__':
    main()
