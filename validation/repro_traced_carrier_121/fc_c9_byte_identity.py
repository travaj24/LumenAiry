# Niche C9 fail-before proof: with ``SPHERE_PARAB_CONVERSION_EXACT = False``
# the library reproduces the COMMITTED ``HEAD`` BIT FOR BIT -- and with it
# True, exactly what moves and where.
#
# WHY NOT ``probe_c8_byte_identity``'s SHADOW MODULE.  That mechanism imports
# ``git show HEAD:lumenairy/elements/_lens_traced.py`` as a second module
# INSIDE the live package, which works because the element is called through
# one name.  ``propagators/carrier.py`` is not: the chain entry point, the
# element hand-off and half a dozen helpers all resolve it as
# ``lumenairy.propagators.carrier``, so a shadow copy would be reached by some
# call sites and not others.  The reference here is instead a WHOLE-PACKAGE
# ``git archive HEAD`` export in a separate PROCESS, driven through
# ``approx_common``'s existing ``LUMEN_PIN`` mechanism (which inserts the pin
# and imports ``lumenairy`` from it BEFORE ``_d121_common`` prepends the repo
# root).  Two processes, two npz dumps, ``np.array_equal``.
#
#   PIN=<dir> MODE=dump OUT=<npz> LUMEN_PIN=<dir> python fc_c9_byte_identity.py
#   MODE=dump OUT=<npz> C9=0 python fc_c9_byte_identity.py
#   MODE=cmp A=<npz> B=<npz> python fc_c9_byte_identity.py
#
# ``fc_c9_run_byte_identity.sh``-style driving is done by the caller; the
# ``MODE=all`` convenience below spawns both dumps itself.
import hashlib
import os
import subprocess
import sys

import numpy as np

os.environ.setdefault('LUMEN_PIN', '0')

import approx_common as A                                      # noqa: E402,F401
import lumenairy                                               # noqa: E402
import lumenairy.propagators.carrier as CM                      # noqa: E402
import _d121_common as C                                       # noqa: E402

LAM = C.LAM
_HERE = os.path.dirname(os.path.abspath(__file__))


# (label, order, chain kwargs) -- the same shape of matrix
# ``probe_c6_byte_identity.CASES_121`` uses: two grids, two ray_subsamples,
# 3- and 5- and 6-group runs, both final-leg routes.
def cases():
    out = []
    for order in ((0, 0), (-4, -2)):
        for lbl, rn, rs, ng, leg in (
                ('RN=1024 rs=4 paraxial', 1024, 4, 6, 'paraxial'),
                ('RN=1024 rs=2 paraxial', 1024, 2, 6, 'paraxial'),
                ('RN=2048 rs=4 paraxial', 2048, 4, 6, 'paraxial'),
                ('RN=1024 rs=4, 3 groups', 1024, 4, 3, 'paraxial'),
                ('RN=1024 rs=4, 5 groups', 1024, 4, 5, 'paraxial'),
                ('RN=1024 rs=4 final_leg=exact', 1024, 4, 6, 'exact')):
            out.append((f"({order[0]:+d},{order[1]:+d}) {lbl}", order, rn, rs,
                        ng, leg))
    return out


def run_case(order, rn, rs, ng, leg):
    _pre, post, _g, period = C.geometry()
    env, R, dx, _P = C.chain_a(n=rn, rs=rs)
    L, M = order[0] * LAM / period, order[1] * LAM / period
    kw = dict(ray_subsample=rs, n_workers=8, on_decentred_fit='ignore',
              on_gap_paraxial='ignore', on_na_proximity='ignore')
    if leg == 'exact':
        # ``on_tilt_exact_grid='ignore'``: at a TILTED order the exact leg's
        # axis-centred window is 1.48x the on-axis one, so Nyquist-sampling the
        # exit sphere would need n_fine=16384 against this probe's 8192 cap and
        # the guard REFUSES.  This is a byte-identity contract, not a physics
        # measurement -- both arms discard the same outer NA, so the comparison
        # is still exact.  (The physics runs that DO score this path --
        # ``focus_scan_121.py`` in S5.1 -- are on-axis and clear of the guard.)
        kw.update(final_distance=C.TRAILING, final_leg='exact',
                  on_tilt_exact_grid='ignore',
                  focus_readout={'dx_out': 0.1e-6, 'N_out': 96,
                                 'n_fine_cap': 8192, 'window_factor': 4.0})
    else:
        kw.update(final_distance=0.0, final_leg='paraxial')
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = C.la.propagate_traced_carrier_chain(
            env, [dict(g) for g in post[:ng]], LAM, dx,
            r_in=C.la.TiltedCarrier(R, L, M), **kw)
    return np.asarray(res.field)


def factor_matrix():
    """The conversion factor itself over a spread of (R, dx, sign, centre)."""
    out = {}
    i = 0
    for R in (-8e-3, -24.46e-3, -230e-3, 7.0e-1, np.inf):
        for dx in (33.2112e-6, 1.5e-6):
            for sg in (+1, -1):
                for cen in ((0.0, 0.0), (1.9e-3, -0.6e-3)):
                    f = CM._sphere_parab_conversion((128, 128), dx, LAM, R,
                                                    sg, centre=cen)
                    out[f'f{i}'] = (np.zeros(1, complex) if f is None
                                    else np.asarray(f))
                    i += 1
    return out


def dump(path):
    d = factor_matrix()
    for lbl, order, rn, rs, ng, leg in cases():
        d[lbl] = run_case(order, rn, rs, ng, leg)
        print(f"  dumped {lbl}", flush=True)
    np.savez(path, **d)
    print(f"wrote {path}", flush=True)


def compare(a, b, expect_equal=True):
    """``expect_equal=False`` is the WHAT-MOVES arm, where differing is the
    correct outcome.  The verdict line says so rather than printing 'FAILED'
    at a table that is supposed to differ -- the first version printed
    'FAILED -- not bit-identical' under the 'WHAT MOVES' heading, which reads
    as a broken contract at a glance and is the opposite of the truth."""
    da, db = np.load(a), np.load(b)
    keys = sorted(set(da.files) | set(db.files))
    ok = True
    print(f"{'case':>40} {'array_equal':>12} {'max|dE|':>11} {'of peak':>10} "
          f"{'dP/P':>11}")
    for kk in keys:
        if kk not in da.files or kk not in db.files:
            print(f"{kk:>40} MISSING")
            ok = False
            continue
        x, y = da[kk], db[kk]
        eq = bool(np.array_equal(x, y)) and x.dtype == y.dtype
        ok &= eq
        pk = float(np.abs(y).max()) or 1.0
        py = float((np.abs(y) ** 2).sum()) or 1.0
        if kk.startswith('f') and kk[1:].isdigit():
            continue                       # summarised below
        print(f"{kk:>40} {str(eq):>12} {float(np.abs(x - y).max()):11.3e} "
              f"{float(np.abs(x - y).max()) / pk:10.2e} "
              f"{(float((np.abs(x) ** 2).sum()) - py) / py:+11.2e}")
    fk = [kk for kk in keys if kk.startswith('f') and kk[1:].isdigit()]
    feq = all(np.array_equal(da[kk], db[kk]) for kk in fk)
    print(f"{'conversion factor matrix (%d cases)' % len(fk):>40} "
          f"{str(feq):>12}")
    ok &= feq
    print("\nOK -- SPHERE_PARAB_CONVERSION_EXACT=False IS HEAD, bit for bit"
          if ok else "\nFAILED -- not bit-identical")
    return ok


def main():
    mode = os.environ.get('MODE', 'all')
    if mode == 'dump':
        c9 = os.environ.get('C9')
        if c9 is not None:
            CM.SPHERE_PARAB_CONVERSION_EXACT = c9 not in ('0', '')
        h = hashlib.sha256(open(CM.__file__, 'rb').read()).hexdigest()[:16]
        print(f"lumenairy {lumenairy.__version__} @ {lumenairy.__file__}")
        print(f"  carrier.py {h}  EXACT="
              f"{getattr(CM, 'SPHERE_PARAB_CONVERSION_EXACT', 'ABSENT')}",
              flush=True)
        dump(os.environ['OUT'])
        return 0
    if mode == 'cmp':
        return 0 if compare(os.environ['A'], os.environ['B']) else 1
    # MODE=all: pin dump, live dump (flag off), compare; then live dump
    # (flag on) and report what moves.
    pin = os.environ['PIN']
    a = os.path.join(_HERE, '_fc_c9_head.npz')
    b = os.path.join(_HERE, '_fc_c9_off.npz')
    c = os.path.join(_HERE, '_fc_c9_on.npz')
    env = dict(os.environ, MODE='dump')
    for out, extra in ((a, {'LUMEN_PIN': pin}), (b, {'C9': '0'}),
                       (c, {'C9': '1'})):
        e = dict(env, OUT=out, **extra)
        print(f"\n--- dumping {os.path.basename(out)} "
              f"({'PINNED HEAD' if 'LUMEN_PIN' in extra else 'live, C9=' + extra['C9']}) ---",
              flush=True)
        subprocess.run([sys.executable, os.path.abspath(__file__)], env=e,
                       check=True, cwd=_HERE)
    print("\n================ (1) FAIL-BEFORE: live C9=0 vs pinned HEAD "
          "================")
    ok = compare(a, b)
    print("\n================ (2) WHAT MOVES: live C9=1 vs pinned HEAD "
          "================")
    compare(a, c)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
