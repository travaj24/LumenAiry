# P2 DID-NOT-WARN diagnosis -- AXIS 5: the RAM budget.
#
# ``tests/unit/test_niche_p2_guards.py::test_self_check_dx_flags_a_non_convergent_chain``
# demands the ``NOT dx-STABLE`` RuntimeWarning.  C11 S9.3 proved the fixture's
# drift is 52.5 % against a 5 % tolerance, five-figure-identical on MKL and
# OpenBLAS, and every physics knob inert -- so the CI DID-NOT-WARN cannot be a
# BLAS or a physics-knob axis.
#
# What C11 never varied is the quantity the guard's OWN grid sizing reads:
#
#     lumenairy.memory.get_ram_budget()  ->  psutil available PHYSICAL memory
#
# ``_memory_bounded_n_fine`` caps the exact focus readout's fine grid at
#
#     n_max = 2 ** floor(log2(floor(sqrt(0.25 * budget / 64))))
#
# i.e. n_max = 2**floor(log2(sqrt(budget)/16)).  This box reports ~67 GB
# available -> n_max = 16384.  A GitHub ubuntu-latest runner has 16 GB TOTAL
# and reports ~10-14 GB available mid-shard -> n_max = 4096 (8192 at best).
#
# The dx self-check re-runs the chain at N -> 2*round(N*sqrt2/2) at the SAME
# physical extent and compares focal metrics.  If the fine grid is CLAMPED to
# the same n_max in BOTH runs, the readout resolution stops tracking the input
# dx -- and the drift the guard exists to measure can collapse below tol, so it
# stops warning.  That is a DID-NOT-WARN with the physics unchanged, it is
# deterministic on a quantity nobody measured, and it is CI-only because CI is
# the only box with a small budget.
#
# This script sweeps the budget and reports, per budget: the fine-grid cap, the
# guard's per-metric drift, and whether the warning FIRED.
#
# LOCAL-ONLY, no library edit.  Run on BOTH builds.
#
# usage:  python p2diag_ram_axis.py
#         WHICH=stable python p2diag_ram_axis.py   # the sibling fixture
import logging
import os
import platform
import sys
import warnings

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     '..', '..'))
sys.path.insert(0, _ROOT)

import lumenairy as la                                          # noqa: E402

_WL = 1.31e-6
_GB = 1024.0 ** 3


# --- the test file's fixtures, copied verbatim -----------------------------
def _singlet(R1, R2, d, glass, ap, name='s'):
    return {'name': name, 'aperture_diameter': ap, 'thicknesses': [d],
            'surfaces': [
                {'radius': R1, 'glass_before': 'air', 'glass_after': glass,
                 'conic': 0.0, 'radius_y': None, 'conic_y': None,
                 'aspheric_coeffs': None, 'aspheric_coeffs_y': None},
                {'radius': R2, 'glass_before': glass, 'glass_after': 'air',
                 'conic': 0.0, 'radius_y': None, 'conic_y': None,
                 'aspheric_coeffs': None, 'aspheric_coeffs_y': None}]}


def _chain_kw(**over):
    kw = dict(r_in=np.inf, ray_subsample=4, n_workers=1,
              traced_kwargs=dict(parallel_amp=False, on_undersample='silent'),
              final_distance=118.3e-3,
              focus_readout=dict(dx_out=0.2e-6, N_out=256))
    kw.update(over)
    return kw


def _slow_singlet_chain(N=512, dx=6e-6, w0=0.9e-3):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    env0 = np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)
    groups = [{'prescription': _singlet(60e-3, -60e-3, 4e-3, 'N-BK7', 4e-3),
               'gap_before': 0.0}]
    return env0, groups, dx


# --- capture the guard's own m1 / m2 out of its INFO log -------------------
class _Grab(logging.Handler):
    def __init__(self):
        logging.Handler.__init__(self)
        self.lines = []

    def emit(self, rec):
        try:
            msg = rec.getMessage()
        except Exception:
            return
        if "self_check='dx'" in msg:
            self.lines.append(msg)


def _predicted_cap(budget):
    n = int(np.floor(np.sqrt(max(0.25 * budget / 64.0, 0.0))))
    if n >= 2:
        n = int(2 ** int(np.floor(np.log2(n))))
    return max(n, 256)


def main():
    which = os.environ.get('WHICH', 'nonconv')
    if which == 'stable':
        env0, groups, dx = _slow_singlet_chain()          # N=512, dx=6 um
        extra = dict(self_check_tol=1e-4)
        kw = _chain_kw()
    else:
        env0, groups, dx = _slow_singlet_chain(N=768, dx=4e-6)
        extra = {}
        kw = _chain_kw(r_in=3e-3)

    lg = logging.getLogger('lumenairy.propagators.carrier')
    grab = _Grab()
    lg.addHandler(grab)
    lg.setLevel(logging.INFO)

    print('=' * 100)
    print('P2 dx self-check vs the RAM BUDGET -- fixture:', which)
    print('platform :', platform.platform())
    print('python   :', sys.version.split()[0], '| numpy', np.__version__)
    try:
        import scipy
        print('scipy    :', scipy.__version__)
    except Exception:
        pass
    from lumenairy.memory import get_ram_budget
    la.set_max_ram(None)
    print('detected available RAM: %.2f GB -> fine-grid cap %d'
          % (get_ram_budget() / _GB, _predicted_cap(get_ram_budget())))
    print('=' * 100)

    hdr = ('%-14s %8s %8s | %9s %9s %9s | %9s %6s %6s'
           % ('budget', 'cap', 'n_req', 'power%', 'peak%', 'r50%',
              'max%', 'FIRES', 'MEMLIM'))
    print(hdr)
    print('-' * len(hdr))

    budgets = os.environ.get('BUDGETS')
    if budgets:
        sweep = [float(b) * _GB for b in budgets.split(',')]
    else:
        sweep = [None, 64 * _GB, 32 * _GB, 16 * _GB, 13 * _GB, 8 * _GB,
                 4 * _GB, 2 * _GB, 1 * _GB]

    rows = []
    for b in sweep:
        la.set_max_ram(None if b is None else int(b))
        eff = get_ram_budget()
        del grab.lines[:]
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter('always')
            la.propagate_traced_carrier_chain(env0, groups, _WL, dx,
                                              self_check='dx', **dict(kw, **extra))
        fired = any('NOT dx-STABLE' in str(w.message) for w in wl)
        memlim = [w for w in wl if 'MEMORY-LIMITED' in str(w.message)]
        n_req = ''
        if memlim:
            import re
            m = re.search(r'un-degraded requirement was (\d+)x',
                          ' '.join(str(memlim[0].message).split()))
            if m:
                n_req = m.group(1)
        # per-metric drift straight out of the guard's own log line
        drift = {}
        if grab.lines:
            import ast
            import re
            m = re.search(r'metrics (\{.*\}) vs (\{.*\})$', grab.lines[-1])
            if m:
                m1 = ast.literal_eval(m.group(1))
                m2 = ast.literal_eval(m.group(2))
                for k in sorted(set(m1) & set(m2)):
                    a, c = float(m1[k]), float(m2[k])
                    drift[k] = 100.0 * abs(a - c) / max(abs(a), abs(c), 1e-300)
        mx = max(drift.values()) if drift else float('nan')
        cap = _predicted_cap(eff)
        rows.append((eff, cap, n_req, drift, mx, fired, len(memlim)))
        print('%-14s %8d %8s | %9.4f %9.4f %9.4f | %9.4f %6s %6d'
              % (('auto %.1fGB' % (eff / _GB)) if b is None
                 else '%.0f GB' % (b / _GB),
                 cap, n_req or '-',
                 drift.get('power', float('nan')),
                 drift.get('peak', float('nan')),
                 drift.get('r50', float('nan')),
                 mx, 'YES' if fired else '** NO **', len(memlim)))
        sys.stdout.flush()

    la.set_max_ram(None)
    print('-' * len(hdr))
    bad = [r for r in rows if not r[5]]
    if bad:
        print('DID-NOT-WARN REPRODUCED at budgets: %s'
              % ', '.join('%.1f GB' % (r[0] / _GB) for r in bad))
    else:
        print('warning fired at every budget in the sweep')


if __name__ == '__main__':
    main()
