# C11 side-quest -- ADJUDICATE the niche-P2 dx self-check fixtures.
#
# ``tests/unit/test_niche_p2_guards.py::test_self_check_dx_flags_a_non_convergent_chain``
# demands the ``NOT dx-STABLE`` RuntimeWarning on a deliberately beyond-Nyquist
# fixture (N=768 / dx=4 um / r_in=+3 mm), whose drift was measured at ~50 % on
# 2026-07-25.  A DID-NOT-WARN failure means the drift fell below
# ``self_check_tol`` (0.05), i.e. the fixture stopped being unstable -- which
# would make its premise stale rather than the guard broken.
#
# ``pytest.warns`` is a BINARY instrument: it says fired / did not fire and
# never says by how much.  This measures the MARGIN instead -- the guard's own
# per-metric drift, computed with the library's own helpers so it is the same
# arithmetic the warning is raised from -- across the knobs that moved
# (``SPHERE_PARAB_CONVERSION_EXACT`` = niche C9, ``_REMAP_RESID_EIKONAL_DEGREE``
# = niche C10, and the C6 launch both sit on).
#
# A fixture whose drift sits at 50 % against a 5 % tolerance is healthy; one
# that sits at 6 % is a coin toss between BLAS builds and must be strengthened.
#
# LOCAL-ONLY, no library edit: every knob is a module attribute set inside a
# try/finally.  Run on BOTH builds.
#
# usage:  python c11_p2dx_recon.py
#         WHICH=stable python c11_p2dx_recon.py      # the sibling's fixture
import os
import platform
import sys

import numpy as np

os.environ.setdefault('LUMEN_PIN', '0')

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     '..', '..'))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'tests', 'unit'))

import lumenairy as la                                         # noqa: E402
import lumenairy.elements._lens_traced as LT                   # noqa: E402
import lumenairy.propagators.carrier as CM                     # noqa: E402

STATES = [
    ('shipped   C9=on  deg=6  C6=on', dict()),
    ('C10 off   C9=on  deg=4  C6=on', dict(deg=4)),
    ('C9  off   C9=off deg=6  C6=on', dict(c9=False)),
    ('both off  C9=off deg=4  C6=on', dict(c9=False, deg=4)),
    ('C6  off   C9=on  deg=-  C6=off', dict(c6=False)),
]


class State(object):
    def __init__(self, c9=None, deg=None, c6=None):
        self.c9, self.deg, self.c6 = c9, deg, c6

    def __enter__(self):
        self.old = (CM.SPHERE_PARAB_CONVERSION_EXACT,
                    LT._REMAP_RESID_EIKONAL_DEGREE,
                    LT.REMAP_STATIONARY_PHASE_LAUNCH)
        if self.c9 is not None:
            CM.SPHERE_PARAB_CONVERSION_EXACT = bool(self.c9)
        if self.deg is not None:
            LT._REMAP_RESID_EIKONAL_DEGREE = int(self.deg)
        if self.c6 is not None:
            LT.REMAP_STATIONARY_PHASE_LAUNCH = bool(self.c6)
        return self

    def __exit__(self, *e):
        (CM.SPHERE_PARAB_CONVERSION_EXACT,
         LT._REMAP_RESID_EIKONAL_DEGREE,
         LT.REMAP_STATIONARY_PHASE_LAUNCH) = self.old
        return False


def drift(env0, groups, dx, kw):
    """The guard's OWN comparison, per metric, without the warning.

    Mirrors ``_run_chain_dx_self_check``: refine N -> 2*round(N*sqrt(2)/2) at
    the same physical extent and compare ``_chain_result_metrics``.
    """
    full = dict(E_in=env0, groups=groups, wavelength=WL, dx=dx)
    full.update(kw)
    res = la.propagate_traced_carrier_chain(**full)
    m1 = CM._chain_result_metrics(res)
    N = int(np.asarray(env0).shape[-1])
    N2 = int(2 * round(N * np.sqrt(2.0) / 2.0))
    if N2 <= N:
        N2 = N + 2
    full2 = dict(full)
    full2['E_in'] = CM._fourier_upsample_crop(np.asarray(env0), N, N2)
    full2['dx'] = dx * N / N2
    m2 = CM._chain_result_metrics(la.propagate_traced_carrier_chain(**full2))
    out = {}
    for k in sorted(set(m1) & set(m2)):
        a, b = float(m1[k]), float(m2[k])
        out[k] = abs(a - b) / max(abs(a), abs(b), 1e-300)
    return out, N, N2


def main():
    import test_niche_p2_guards as P
    global WL
    WL = P._WL
    which = os.environ.get('WHICH', 'unstable')
    if which == 'unstable':
        env0, groups, dx, _ = P._slow_singlet_chain(N=768, dx=4e-6)
        kw = P._chain_kw(r_in=3e-3)
        tol, want = 0.05, 'MUST warn'
    else:
        env0, groups, dx, _ = P._slow_singlet_chain()
        kw = P._chain_kw()
        tol, want = 1e-4, 'MUST warn at tol=1e-4'
    kw.pop('self_check', None)
    print(f"{platform.system()} py{platform.python_version()}  "
          f"numpy {np.__version__}  lumenairy {la.__version__}")
    try:
        import numpy.__config__ as _nc
        blas = _nc.CONFIG['Build Dependencies']['blas']['name']
    except Exception:
        blas = '?'
    print(f"BLAS: {blas}")
    print(f"fixture: {which}   tolerance {tol:g}   ({want})\n")
    hdr = f"{'state':34}" + ''.join(f"{k:>12}" for k in
                                    ('power', 'peak', 'r50', 'w_env', 'R'))
    print(hdr + f"{'MAX':>10} {'x tol':>8} {'fires?':>8}")
    print('-' * (len(hdr) + 28))
    for name, st in STATES:
        with State(**st):
            d, N, N2 = drift(env0, groups, dx, kw)
        mx = max(d.values()) if d else 0.0
        row = f"{name:34}"
        for k in ('power', 'peak', 'r50', 'w_env', 'R'):
            row += (f"{d[k] * 100:11.3f}%" if k in d else f"{'--':>12}")
        print(row + f"{mx * 100:9.3f}% {mx / tol:8.2f} "
              f"{('YES' if mx > tol else 'NO'):>8}", flush=True)
    print(f"\nN {N} -> {N2}.  'x tol' is the MARGIN: a fixture at 10x is "
          f"robust across BLAS builds,\none at ~1x is a coin toss and its "
          f"premise needs strengthening, not its bar relaxing.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
