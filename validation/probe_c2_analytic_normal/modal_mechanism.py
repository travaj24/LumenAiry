"""WP-C2 item 1 -- WHAT is the 1e-8 in the ModalAsymptotic arms?

WP-B9 sec. 5 and VERIFY-B9 sec. 4 describe the quantity as bimodal --
"~0 or ~1e-8 relative", one knife-edge pixel changing saddle basin.
This probe measures the per-pixel population directly and asks where the
disagreement comes from: the saddle LOCATION (v2*), or the field
evaluation at a common location.

Usage:  LUMENAIRY_ROOT=<root> python modal_mechanism.py <out.json>
"""
import importlib
import importlib.util
import json
import os
import sys

import numpy as np

_ROOT = os.environ['LUMENAIRY_ROOT']
sys.path.insert(0, _ROOT)

import lumenairy as la  # noqa: E402

_want = os.path.realpath(os.path.join(_ROOT, 'lumenairy'))
assert os.path.realpath(os.path.dirname(la.__file__)) == _want, la.__file__


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    out_path = sys.argv[1]
    tp = _load('_tp', os.path.join(_ROOT, 'tests', 'unit',
                                   'test_audit_propagation.py'))
    inst = tp.TestAuditFixesV4_14_0_agent_1_1APropagateModalAsymptoticStillBitEqual()
    fit = tp._build_singlet_fit()
    N = 32
    ax = np.linspace(-5e-6, 5e-6, N)
    S2X, S2Y = np.meshgrid(ax, ax, indexing='xy')
    src = {(0, 0): 1.0 + 0.0j}
    pup = {(0, 0): 1.0 + 0.0j}
    new = tp.propagate_modal_asymptotic(
        fit, source_point=(0.0, 0.0), source_amplitudes=src,
        pupil_amplitudes=pup, w_s=50e-6, w_p=0.02, v2_centre=(0.0, 0.0),
        s2_grid_x=S2X, s2_grid_y=S2Y)
    cold = inst._cold_start_reference_propagate_modal_asymptotic(
        fit, src, pup, 50e-6, 0.02, S2X, S2Y)
    warm = inst._reference_propagate_modal_asymptotic(
        fit, src, pup, 50e-6, 0.02, S2X, S2Y)
    peak = float(np.max(np.abs(cold)))
    out = {}
    out['cold_vs_warm_max'] = float(np.max(np.abs(cold - warm)))
    out['cold_vs_warm_rel'] = out['cold_vs_warm_max'] / peak
    out['cold_warm_identical'] = bool(np.array_equal(cold, warm))
    out['peak'] = peak
    d = np.abs(new - cold) / peak
    out['new_vs_cold_max_rel'] = float(d.max())
    out['hist'] = {f'>{e}': int(np.sum(d > 10.0 ** e))
                   for e in range(-16, -6)}
    # -- where do the saddles sit?  batch vs scalar, same start.
    asym = importlib.import_module('lumenairy.propagators.asymptotic')
    mas = importlib.import_module('lumenairy.propagators.asymptotic_maslov')
    fx, fy = S2X.ravel(), S2Y.ravel()
    vxb, vyb, convb = mas._solve_envelope_stationary_batch(
        fit, fx, fy, 0.0, 0.0, w_s=50e-6, w_p=0.02, v_cx=0.0, v_cy=0.0)
    vs = np.empty((fx.size, 2))
    ok = np.ones(fx.size, dtype=bool)
    for i in range(fx.size):
        try:
            v, _, _ = asym.solve_envelope_stationary(
                fit, (float(fx[i]), float(fy[i])), (0.0, 0.0),
                w_s=50e-6, w_p=0.02, v2_centre=(0.0, 0.0),
                v2_initial=(0.0, 0.0))
            vs[i] = v
        except Exception:
            ok[i] = False
            vs[i] = np.nan
    dv = np.hypot(vxb[ok] - vs[ok, 0], vyb[ok] - vs[ok, 1])
    vmag = np.hypot(vs[ok, 0], vs[ok, 1])
    out['n_scalar_ok'] = int(ok.sum())
    out['dv_max'] = float(np.nanmax(dv))
    out['dv_median'] = float(np.nanmedian(dv))
    out['v_mag_max'] = float(np.nanmax(vmag))
    out['v2_halfrange'] = float(fit.v2x_halfrange)
    out['dv_max_over_halfrange'] = out['dv_max'] / out['v2_halfrange']
    # correlate: is the field disagreement proportional to the saddle
    # disagreement?
    dd = d.ravel()[ok]
    if dv.size > 4 and np.nanmax(dv) > 0:
        finite = np.isfinite(dv) & np.isfinite(dd)
        out['corr_dv_dfield'] = float(
            np.corrcoef(dv[finite], dd[finite])[0, 1])
        ratio = dd[finite] / np.maximum(dv[finite], 1e-300)
        out['field_per_dv_median'] = float(np.median(ratio))
        out['field_per_dv_max'] = float(np.max(ratio))
    meta = dict(python=sys.version, numpy=np.__version__,
                lumenairy=la.__version__, platform=sys.platform)
    with open(out_path, 'w') as fh:
        json.dump(dict(meta=meta, out=out), fh, indent=1, default=str)
    for k, v in out.items():
        print(f'{k}: {v}')


if __name__ == '__main__':
    main()
