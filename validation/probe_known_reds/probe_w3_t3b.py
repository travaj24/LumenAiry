"""Probe for the two W3-T3b CI reds:

  test_w3_t3b_pure_lg00_default_is_bit_for_bit_unchanged   (rel 8.190e-09
      against a 1e-10 bar; the SAME quantity is barred at 1e-8 two lines up)
  test_w3_t3b_lg_merit_responds_to_a_curvature_change      (val_a off the
      frozen 2.2006679213e+09 by 1.72e+09)

What moves?  The chain is

    fit_canonical_polynomials  ->  coef_phi / coef_s1{x,y}   (LAPACK)
    -> solve_envelope_stationary -> aberration_tensor -> L[0,0]

and the fit carries ~4e5 waves of pupil phase, so the stationary phase is
O(1e6) rad: an ULP on a fit coefficient is amplified by |Phi| before it
reaches ``L``.  This probe reports, per arm:

  * a hash of the fit coefficient arrays (does the FIT move at all?)
  * ``max|coef_phi|`` and the stationary phase magnitude -- the amplifier
  * ``w_o``, ``L[0, 0]``, ``van_vleck_weight``, ``sqrt|det J|``
  * the test's own three residuals: rel(got, frozen want), rel(got, pred)
    and rel(w_o, frozen w_o)
  * the merit values ``val_a`` / ``val_b`` and their response, plus the
    reference coupling ``|L_ref|^2`` the v5.46 normalisation divides by.

Writes <out>/probe_w3_t3b_<ARM>.json.
"""
import hashlib
import json
import math
import os
import sys
import warnings

import numpy as np

import lumenairy

_WL = 1.31e-6
_WS, _WP = 20e-6, 0.02
_FITKW = dict(source_box_half=20e-6, pupil_box_half=0.02,
              n_field=6, n_pupil=6, poly_order=4)

# the literals the tests freeze
_FROZEN_OLD = {51.5e-3: 1.544807582649e+01 + 3.188059447022e+00j,
               60.0e-3: -6.570638987023e+00 - 1.439184599267e+01j}
_FROZEN = {51.5e-3: (1.0116441690e-04,
                     1.2310011487876e+07 - 5.9649449469122e+07j),
           60.0e-3: (1.0086220541e-04,
                     -5.5405030010922e+07 + 2.5295326982543e+07j)}
_FROZEN_MERIT = (2.2006679213e+09, 9.9881936194e+04)


def _singlet(R1):
    p = lumenairy.make_singlet(R1, float('inf'), 4.1e-3, 'N-BK7',
                               aperture=12.0e-3)
    p['object_distance'] = 200e-3
    return p


def _fit(R1):
    from lumenairy.propagators.asymptotic import fit_canonical_polynomials
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fit_canonical_polynomials(_singlet(R1), wavelength=_WL,
                                         **_FITKW)


def _tensor(fit, modes):
    from lumenairy.propagators.asymptotic import aberration_tensor
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return aberration_tensor(
            fit, s2_image=(fit.s2x_centre, fit.s2y_centre),
            source_point=(0.0, 0.0), source_modes=[(0, 0)],
            pupil_modes=[(0, 0)], output_modes=list(modes),
            w_s=_WS, w_p=_WP, w_o=None)


def _hash(a):
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def _one(R1):
    from lumenairy.propagators.asymptotic import solve_envelope_stationary
    from lumenairy.propagators.asymptotic_aberration_tensor import _compute_M_b
    fit = _fit(R1)
    s2 = (fit.s2x_centre, fit.s2y_centre)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        v_star, _, _ = solve_envelope_stationary(
            fit, s2, (0.0, 0.0), w_s=_WS, w_p=_WP, v2_centre=(0.0, 0.0))
    M = _compute_M_b(fit, s2[0], s2[1], v_star[0], v_star[1], 0.0, 0.0,
                     _WS, _WP, 0.0, 0.0)[0]
    legacy_w_o = 1.0 / math.sqrt(float(np.linalg.eigvalsh(np.real(M)).max()))
    res = _tensor(fit, ((0, 0),))
    got = complex(res.L[0, 0])
    w_want, want = _FROZEN[R1]
    old = _FROZEN_OLD[R1]
    sqrt_detJ = abs(complex(res.van_vleck_weight)) * _WL
    factor = -1j / (_WL * sqrt_detJ)
    pred = old * factor
    # the amplifier: the stationary phase in rad
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        phi_waves = float(fit.eval_phi(np.array([s2[0]]), np.array([s2[1]]),
                                       np.array([v_star[0]]),
                                       np.array([v_star[1]]))[0])
    return dict(
        R1=R1,
        coef_phi_hash=_hash(fit.coef_phi),
        coef_s1x_hash=_hash(fit.coef_s1x),
        coef_s1y_hash=_hash(fit.coef_s1y),
        coef_phi_absmax=float(np.max(np.abs(fit.coef_phi))),
        coef_phi_l2=float(np.linalg.norm(fit.coef_phi)),
        res_phi_rms_waves=float(fit.res_phi_rms_waves),
        s2x_centre=float(s2[0]), s2y_centre=float(s2[1]),
        v_star=[float(v_star[0]), float(v_star[1])],
        phi_stationary_waves=phi_waves,
        phi_stationary_rad=2.0 * math.pi * phi_waves,
        eps_times_phi=float(np.finfo(np.float64).eps
                            * abs(2.0 * math.pi * phi_waves)),
        w_o=float(res.w_o), legacy_w_o=legacy_w_o,
        w_o_equals_legacy=bool(res.w_o == legacy_w_o),
        w_o_rel_frozen=abs(res.w_o - w_want) / w_want,
        L00_re=got.real, L00_im=got.imag, L00_abs=abs(got),
        van_vleck_weight=[complex(res.van_vleck_weight).real,
                          complex(res.van_vleck_weight).imag],
        sqrt_detJ=sqrt_detJ,
        rel_got_want=abs(got - want) / abs(want),
        rel_got_pred=abs(got - pred) / abs(pred),
        rel_want_pred=abs(want - pred) / abs(pred),
        pred_re=pred.real, pred_im=pred.imag,
    )


def _cpl_from_warnings(recs):
    """The library's own reference-collapse reading, parsed out of the
    RuntimeWarning ``LGAberrationMerit`` raises when the aberration-free
    reference stops being a reference sphere."""
    import re
    for w in recs:
        m = re.search(r'\|L\(0,0\)/L_ref\(0,0\)\|\^2 = ([0-9.e+-]+)',
                      str(w.message))
        if m:
            r = re.search(r'reference zeroes ([0-9.e+-]+) waves',
                          str(w.message))
            return float(m.group(1)), (float(r.group(1)) if r else None)
    return None, None


def _merit():
    class _Ctx:
        wavelength = _WL
        N = 64
        dx = 20e-6

    merit = lumenairy.LGAberrationMerit(
        targets={(2, 0): 1.0}, field_points=[(0.0, 0.0)],
        w_s=_WS, w_p=_WP, fit_kwargs=_FITKW)
    ctx_a, ctx_b = _Ctx(), _Ctx()
    ctx_a.prescription = _singlet(51.5e-3)
    ctx_b.prescription = _singlet(60.0e-3)
    with warnings.catch_warnings(record=True) as ra:
        warnings.simplefilter('always')
        val_a = float(merit.evaluate(ctx_a))
    with warnings.catch_warnings(record=True) as rb:
        warnings.simplefilter('always')
        val_b = float(merit.evaluate(ctx_b))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        val_a2 = float(merit.evaluate(ctx_a))
    cpl_a, rem_a = _cpl_from_warnings(ra)
    cpl_b, rem_b = _cpl_from_warnings(rb)
    rel = abs(val_a - val_b) / max(val_a, val_b)
    out = dict(val_a=val_a, val_b=val_b, val_a_repeat=val_a2,
               in_process_spread=abs(val_a2 - val_a) / max(val_a, 1e-300),
               response=rel,
               rel_a_frozen=abs(val_a - _FROZEN_MERIT[0]) / _FROZEN_MERIT[0],
               rel_b_frozen=abs(val_b - _FROZEN_MERIT[1]) / _FROZEN_MERIT[1],
               cpl00_a=cpl_a, cpl00_b=cpl_b,
               waves_removed_a=rem_a, waves_removed_b=rem_b)
    # The REFERENCE-FREE channel ratio |L_(2,0)|^2 / |L_(0,0)|^2: the merit
    # value divided by the library's own |L(0,0)/L_ref|^2 reading, so the
    # collapsed reference cancels out of it exactly.
    if cpl_a:
        out['chan_ratio_a'] = val_a / cpl_a
    if cpl_b:
        out['chan_ratio_b'] = val_b / cpl_b
    return out


def main():
    arm = sys.argv[1] if len(sys.argv) > 1 else 'LOCAL'
    outdir = sys.argv[2] if len(sys.argv) > 2 else (
        os.path.dirname(os.path.abspath(__file__)))
    assert 'lum_reds' in lumenairy.__file__, lumenairy.__file__
    try:
        import threadpoolctl
        tpi = threadpoolctl.threadpool_info()
    except Exception as e:
        tpi = [{'error': repr(e)}]
    env = dict(arm=arm, python=sys.version.split()[0], numpy=np.__version__,
               platform=sys.platform, lumenairy_file=lumenairy.__file__,
               OPENBLAS_CORETYPE=os.environ.get('OPENBLAS_CORETYPE', ''),
               OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS', ''),
               threadpool=tpi)
    try:
        import scipy
        env['scipy'] = scipy.__version__
    except Exception:
        env['scipy'] = None
    rows = [_one(51.5e-3), _one(60.0e-3)]
    merit = _merit()
    out = dict(env=env, rows=rows, merit=merit)
    path = os.path.join(outdir, 'probe_w3_t3b_%s.json' % arm)
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)
    print('arm=%s py=%s numpy=%s blas=%s'
          % (arm, env['python'], env['numpy'],
             tpi[0].get('architecture') if tpi and 'architecture' in tpi[0]
             else '?'))
    for r in rows:
        print("R1=%.4f coef_hash=%s |phi|=%.4e rad eps*phi=%.3e w_o_rel=%.3e "
              "rel(got,want)=%.3e rel(got,pred)=%.3e rel(want,pred)=%.3e "
              "sqrt_detJ=%.12e"
              % (r['R1'], r['coef_phi_hash'], abs(r['phi_stationary_rad']),
                 r['eps_times_phi'], r['w_o_rel_frozen'], r['rel_got_want'],
                 r['rel_got_pred'], r['rel_want_pred'], r['sqrt_detJ']))
    print("merit val_a=%.10e val_b=%.10e response=%.6e rel_a_frozen=%.3e "
          "rel_b_frozen=%.3e in_process_spread=%.3e"
          % (merit['val_a'], merit['val_b'], merit['response'],
             merit['rel_a_frozen'], merit['rel_b_frozen'],
             merit['in_process_spread']))
    print("merit cpl00_a=%s cpl00_b=%s chan_ratio_a=%s chan_ratio_b=%s "
          "waves_removed=%s/%s"
          % (merit['cpl00_a'], merit['cpl00_b'], merit.get('chan_ratio_a'),
             merit.get('chan_ratio_b'), merit['waves_removed_a'],
             merit['waves_removed_b']))
    print('WROTE', path)


if __name__ == '__main__':
    main()
