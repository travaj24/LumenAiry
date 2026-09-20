"""VERIFY-WAVE5-HYGIENE2 round 2 -- the near-focus accuracy switch, traced.

Four readings, and the first of them is a TRACE rather than a reading of the
source:

 T1  ``_GAP_KERNEL_ACCURACY_TAU`` is ``None`` in the shipped source (read from
     the file, not from the imported module, so a conftest that armed it could
     not hide).
 T2  INERT, PROVED BY EXECUTION.  ``sys.settrace`` records every line of
     ``carrier.py`` executed during a ``gap_kernel='exact'`` and a
     ``gap_kernel='auto'`` Collins leg.  The rule's own lines -- the
     ``_collins_envelope_half_angle`` call, the ``_collins_exact_kernel_
     departure`` call and the ``dep > tau`` comparison -- must not appear.
     Belt and braces: both helpers are also replaced by raising sentinels for
     the same legs, so "not executed" is asserted two independent ways.
 T3  ARMED at ``tau = 1e-4``: the hygiene-2 ladder stays ``'exact'`` at every
     rung while F3's 1 um and 10 um rungs fall back to ``'fresnel'`` and its
     100 um rung does not -- with the margins, not just the verdicts.
 T4  An EXPLICIT ``gap_kernel='exact'`` is honoured over an armed ``tau``.

    PYTHONPATH=<tree> python vh3_tau.py OUT.json
"""
import json
import pathlib
import re
import sys

import numpy as np

# ---- the hygiene-2 fixture, re-derived here from its three numbers --------
W0 = 15.915e-6
THETA = 20.0e-3
LAM = float(np.pi * W0 * THETA)
ZR = float(np.pi * W0 ** 2 / LAM)
F = 20.0e-3
N_IN = 512
DX_IN = 8e-6

# ---- VERIFY-B4 F3's fixture ----------------------------------------------
F3 = dict(lam=1.064e-6, n=1024, dx=4e-6, w=0.30e-3, R=-40e-3,
          dx_out=5.6447e-06, n_out=128)


def axis(n, d):
    return (np.arange(int(n), dtype=np.float64) - int(n) / 2) * float(d)


def gauss_env(n, dx, w):
    x = axis(n, dx)
    return np.exp(-(x[None, :] ** 2 + x[:, None] ** 2)
                  / w ** 2).astype(np.complex128)


def main(out_path):
    import lumenairy
    from lumenairy.propagators import carrier as CA

    res = {'lumenairy_file': lumenairy.__file__,
           'python': sys.version.split()[0], 'platform': sys.platform}

    # --- T1: the SOURCE, not the module ---------------------------------
    src = pathlib.Path(CA.__file__).read_text(encoding='cp1252')
    m = re.search(r'^_GAP_KERNEL_ACCURACY_TAU\s*=\s*(.+)$', src, re.M)
    res['T1'] = {'source_line': m.group(0) if m else None,
                 'source_value': m.group(1).strip() if m else None,
                 'is_None_in_source': bool(m and m.group(1).strip() == 'None'),
                 'imported_value': repr(CA._GAP_KERNEL_ACCURACY_TAU)}

    # --- the two legs the rule would act on ------------------------------
    w_in = float(np.sqrt(LAM * (F ** 2 + ZR ** 2) / (np.pi * ZR)))
    R_in = -(F ** 2 + ZR ** 2) / F
    env = gauss_env(N_IN, DX_IN, w_in)

    def h2_leg(d, gk, stats=None):
        return CA._collins_transport(
            env, R_in, F - d, LAM, DX_IN, DX_IN, dx_out=DX_IN, dy_out=DX_IN,
            N_out_x=N_IN, N_out_y=N_IN, R_ref=float('inf'), gap_kernel=gk,
            on_collins_sampling='ignore', stats_out=stats)

    env3 = gauss_env(F3['n'], F3['dx'], F3['w'])

    def f3_leg(dz, gk, stats=None):
        return CA._collins_transport(
            env3, F3['R'], -F3['R'] - dz, F3['lam'], F3['dx'], F3['dx'],
            dx_out=F3['dx_out'], dy_out=F3['dx_out'], N_out_x=F3['n_out'],
            N_out_y=F3['n_out'], R_ref=float('inf'), gap_kernel=gk,
            on_collins_sampling='ignore', stats_out=stats)

    # --- T2a: TRACE the executed lines -----------------------------------
    carrier_file = str(pathlib.Path(CA.__file__).resolve())
    # The GUARD line (``... and _GAP_KERNEL_ACCURACY_TAU is not None``) is
    # EXPECTED to execute -- a short circuit has to be evaluated to short
    # circuit.  What must not execute is the RULE it guards: the half-angle
    # measurement, the departure evaluation, the comparison against tau and
    # the extra stats key.  The two sets are kept apart so the reading is a
    # statement about the rule and not about the ``if``.
    rule_lines, guard_lines = set(), set()
    for i, line in enumerate(src.splitlines(), start=1):
        s = line.strip()
        if s.startswith('#') or s.startswith(':') or s.startswith('*'):
            continue
        if '_GAP_KERNEL_ACCURACY_TAU is not None' in s:
            guard_lines.add(i)
        elif ('_collins_envelope_half_angle(S' in s
                or '_collins_exact_kernel_departure(' in s
                or 'dep > float(_GAP_KERNEL_ACCURACY_TAU)' in s
                or "st['kernel_departure'] = dep" in s):
            rule_lines.add(i)
    executed = set()

    def tracer(frame, event, arg):
        if frame.f_code.co_filename == carrier_file:
            if event == 'line':
                executed.add(frame.f_lineno)
            return tracer
        return None

    sys.settrace(tracer)
    try:
        for gk in ('auto', 'exact', 'fresnel'):
            st = {}
            h2_leg(1e-6, gk, st)
    finally:
        sys.settrace(None)
    res['T2_trace'] = {
        'carrier_lines_executed': len(executed),
        'rule_lines': sorted(rule_lines),
        'rule_lines_executed': sorted(rule_lines & executed),
        'guard_lines': sorted(guard_lines),
        'guard_lines_executed': sorted(guard_lines & executed),
        'rule_is_inert': not (rule_lines & executed),
        # the trace is NOT vacuous: the guard that short-circuits the rule
        # did execute, so the tracer was watching the right function.
        'trace_is_live': bool(guard_lines & executed),
    }

    # --- T2b: raising sentinels, the independent second way --------------
    calls = []
    real_half = CA._collins_envelope_half_angle
    real_dep = CA._collins_exact_kernel_departure

    def boom_half(*a, **k):
        calls.append('half_angle')
        raise AssertionError('the accuracy rule measured the half-angle '
                             'while tau is None')

    def boom_dep(*a, **k):
        calls.append('departure')
        raise AssertionError('the accuracy rule evaluated the departure '
                             'while tau is None')

    CA._collins_envelope_half_angle = boom_half
    CA._collins_exact_kernel_departure = boom_dep
    try:
        sentinel = {'ok': True, 'error': None}
        for gk in ('auto', 'exact', 'fresnel'):
            st = {}
            h2_leg(1e-6, gk, st)
            if 'kernel_departure' in st:
                sentinel['ok'] = False
                sentinel['error'] = f'stats grew kernel_departure at {gk}'
    except BaseException as e:                       # noqa: BLE001
        sentinel = {'ok': False, 'error': f'{type(e).__name__}: {e}'}
    finally:
        CA._collins_envelope_half_angle = real_half
        CA._collins_exact_kernel_departure = real_dep
    res['T2_sentinel'] = {'result': sentinel, 'calls_seen': calls}

    # --- T3: ARMED at tau = 1e-4 ----------------------------------------
    CA._GAP_KERNEL_ACCURACY_TAU = 1e-4
    try:
        h2 = {}
        for d in (1e-6, 1e-5, 1e-4, 1e-3, 5e-3):
            st = {}
            h2_leg(d, 'auto', st)
            h2[f'{d:.0e}'] = {'kernel': st['kernel'],
                              'departure': st.get('kernel_departure'),
                              'k4': st.get('k4')}
        f3 = {}
        for dz in (1e-6, 1e-5, 1e-4, 1e-3):
            st = {}
            f3_leg(dz, 'auto', st)
            f3[f'{dz:.0e}'] = {'kernel': st['kernel'],
                               'departure': st.get('kernel_departure'),
                               'k4': st.get('k4')}
        # --- T4: explicit 'exact' honoured over an armed tau -------------
        st = {}
        f3_leg(1e-6, 'exact', st)
        res['T4'] = {'kernel': st['kernel'],
                     'departure': st.get('kernel_departure'),
                     'honoured': st['kernel'] == 'exact'}
        # ... and the SHIPPED default with tau armed is unchanged for
        # 'fresnel' too
        st = {}
        f3_leg(1e-6, 'fresnel', st)
        res['T4']['fresnel_still_fresnel'] = st['kernel'] == 'fresnel'
    finally:
        CA._GAP_KERNEL_ACCURACY_TAU = None
    res['T3'] = {'hygiene2': h2, 'f3': f3,
                 'h2_all_exact': all(v['kernel'] == 'exact'
                                     for v in h2.values()),
                 'h2_worst_departure': max(v['departure'] for v in
                                           h2.values()),
                 'f3_1um': f3['1e-06']['kernel'],
                 'f3_10um': f3['1e-05']['kernel'],
                 'f3_100um': f3['1e-04']['kernel']}
    res['T3_restored'] = repr(CA._GAP_KERNEL_ACCURACY_TAU)

    with open(out_path, 'w', encoding='cp1252') as fh:
        json.dump(res, fh, indent=1, sort_keys=True, default=str)
    print(f"lumenairy.__file__ = {lumenairy.__file__}")
    print(f"T1 source: {res['T1']['source_line']}  "
          f"imported {res['T1']['imported_value']}")
    print(f"T2 trace: {res['T2_trace']['carrier_lines_executed']} carrier "
          f"lines executed; rule lines {res['T2_trace']['rule_lines']} "
          f"executed {res['T2_trace']['rule_lines_executed']} -> inert "
          f"{res['T2_trace']['rule_is_inert']}; guard "
          f"{res['T2_trace']['guard_lines']} executed "
          f"{res['T2_trace']['guard_lines_executed']} -> live "
          f"{res['T2_trace']['trace_is_live']}")
    print(f"T2 sentinel: {res['T2_sentinel']}")
    print("T3 hygiene-2:", {k: (v['kernel'], f"{v['departure']:.4e}")
                            for k, v in h2.items()})
    print("T3 F3:", {k: (v['kernel'], f"{v['departure']:.4e}")
                     for k, v in f3.items()})
    print(f"T4 explicit exact honoured: {res['T4']}")
    print(f"tau restored to {res['T3_restored']}")
    print(f"-> {out_path}")


if __name__ == '__main__':
    main(sys.argv[1])
