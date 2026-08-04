# P2 DID-NOT-WARN diagnosis -- the ROUTING axis.
#
# One run of the fixture, fully instrumented.  Prints, for the
# ``test_self_check_dx_flags_a_non_convergent_chain`` fixture:
#
#   * the chain's OWN measured exit NA and how far it sits from
#     ``na_exact_threshold`` -- the ``final_leg='auto'`` cliff;
#   * which readout the final leg actually took (EXACT vs PARAXIAL);
#   * the dx self-check's per-metric drift (out of the guard's INFO log);
#   * whether the ``NOT dx-STABLE`` RuntimeWarning FIRED.
#
# WHY: the library itself warns that this fixture's measured exit NA is
# 0.14870 against ``na_exact_threshold=0.15`` -- 0.87 % BELOW a routing cliff
# that it says "flips between the exact and the PARAXIAL focus readout ... with
# no other symptom".  C11 S9.3's 10.5x margin is the PARAXIAL branch's margin.
# On the EXACT branch the same fixture drifts 8.9 % against the same 5 % bar --
# 1.78x, which is the coin-toss regime.
#
# usage:
#   python p2diag_route.py                    # shipped config
#   NA_THR=0.14 python p2diag_route.py        # force the EXACT branch
#   THREADS=1 python p2diag_route.py          # pin BLAS threads (set BEFORE numpy)
import ast
import logging
import os
import platform
import re
import sys
import warnings

_THREADS = os.environ.get('THREADS')
if _THREADS:
    for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[_v] = _THREADS

import numpy as np                                              # noqa: E402

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     '..', '..'))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lumenairy as la                                          # noqa: E402
from p2diag_ram_axis import _slow_singlet_chain, _chain_kw, _WL  # noqa: E402


class _Grab(logging.Handler):
    def __init__(self):
        logging.Handler.__init__(self)
        self.lines = []

    def emit(self, rec):
        try:
            m = rec.getMessage()
        except Exception:
            return
        if "self_check='dx'" in m:
            self.lines.append(m)


def run(which='nonconv', na_thr=None, tol=None, verbose=True):
    if which == 'stable':
        env0, groups, dx = _slow_singlet_chain()
        kw = _chain_kw()
        extra = dict(self_check_tol=1e-4 if tol is None else tol)
    else:
        env0, groups, dx = _slow_singlet_chain(N=768, dx=4e-6)
        kw = _chain_kw(r_in=3e-3)
        extra = {} if tol is None else dict(self_check_tol=tol)
    if na_thr is not None:
        extra['na_exact_threshold'] = float(na_thr)

    lg = logging.getLogger('lumenairy.propagators.carrier')
    grab = _Grab()
    lg.addHandler(grab)
    lg.setLevel(logging.INFO)

    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter('always')
        la.propagate_traced_carrier_chain(env0, groups, _WL, dx,
                                          self_check='dx', **dict(kw, **extra))
    lg.removeHandler(grab)

    texts = [' '.join(str(w.message).split()) for w in wl]
    fired = any('NOT dx-STABLE' in t for t in texts)
    na, route = None, '?'
    for t in texts:
        m = re.search(r"measured exit NA ([0-9.]+) sits within .*?"
                      r"na_exact_threshold=([0-9.]+) -- (\w+) \(routing (\w+)\)", t)
        if m:
            na, route = float(m.group(1)), m.group(4)
            break
    drift = {}
    if grab.lines:
        m = re.search(r'metrics (\{.*\}) vs (\{.*\})$', grab.lines[-1])
        if m:
            m1, m2 = ast.literal_eval(m.group(1)), ast.literal_eval(m.group(2))
            for k in sorted(set(m1) & set(m2)):
                a, b = float(m1[k]), float(m2[k])
                drift[k] = 100.0 * abs(a - b) / max(abs(a), abs(b), 1e-300)
    rd = [t for t in texts if 'P_out/P_ap' in t or 'aperture-transmitted' in t]
    rdv = []
    for t in rd:
        m = re.search(r'input power = ([0-9.]+)', t)
        if m:
            rdv.append(float(m.group(1)))
    if verbose:
        print('  measured exit NA : %s   route %s   (n warnings %d)'
              % ('%.5f' % na if na is not None else 'not reported', route, len(wl)))
        print('  ray_density P_out/P_ap : %s' % ', '.join('%.6f' % v for v in rdv))
        print('  drift %%          : %s'
              % ', '.join('%s %.4f' % (k, v) for k, v in sorted(drift.items())))
        print('  NOT dx-STABLE    : %s' % ('FIRED' if fired else '** DID NOT FIRE **'))
    return dict(na=na, route=route, drift=drift, fired=fired, rd=rdv,
                n_warn=len(wl))


def main():
    print('=' * 78)
    print('platform :', platform.platform())
    print('python   :', sys.version.split()[0], '| numpy', np.__version__)
    try:
        import scipy
        print('scipy    :', scipy.__version__)
    except Exception:
        pass
    print('threads  :', _THREADS or '(unpinned)', '| cpus',
          la.memory.available_cpus() if hasattr(la, 'memory') else '?')
    print('=' * 78)
    thr = os.environ.get('NA_THR')
    which = os.environ.get('WHICH', 'nonconv')
    print('fixture %s, na_exact_threshold=%s' % (which, thr or '0.15 (default)'))
    run(which, float(thr) if thr else None)


if __name__ == '__main__':
    main()
