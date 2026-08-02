# niche C8 fail-before proof: with ``REMAP_INVERSE_SUPPORT_BOUND = False`` the
# library reproduces the COMMITTED ``HEAD`` BIT FOR BIT -- and with it True,
# exactly what moves, and where.
#
# WHY A NEW PROBE.  ``probe_c6_byte_identity.py`` predates the commit that put
# niche C6 IN ``HEAD``.  Its contract is "C6 OFF reproduces the committed
# library", and it forces ``REMAP_STATIONARY_PHASE_LAUNCH = False`` on the live
# side while the shadow it compares against now ships that flag True -- so it
# now measures the C6 delta, not an identity, and its own header line says so
# ("HEAD has REMAP_STATIONARY_PHASE_LAUNCH: True (must be False)").  It is left
# untouched; the contract it USED to state is restated here for C8, on the same
# case matrix and against the same shadow-module mechanism:
#
#   git show HEAD:lumenairy/elements/_lens_traced.py  ->  a temp file  ->
#   imported as ``lumenairy.elements._lens_traced_head``  ->  compared
#
# so a single changed bit anywhere in the returned field shows, and the
# reference is not a re-run of the same code.
#
# EVERY OTHER FLAG IS LEFT AT ITS SHIPPED DEFAULT, which is HEAD's default --
# that is the point: this asks whether the C8 switch alone accounts for every
# difference between the working tree and the commit.
#
# usage:  python probe_c8_byte_identity.py
#         PARTS=ab python probe_c8_byte_identity.py
import os
import sys

import numpy as np
import probe_c6_byte_identity as B

import lumenairy.elements as _EL
import lumenairy.elements._lens_traced as LT


def synthetic(head, bound):
    """B.synthetic's case matrix, at the SHIPPED C6 setting."""
    n, dx, w, rc, alpha = 256, 4.0e-6, 200e-6, -0.02, 6.0
    x = (np.arange(n) - n // 2) * dx
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    S = np.sign(rc) * (np.sqrt(r2 + rc ** 2) - abs(rc))
    E = (np.exp(-r2 / w ** 2) * np.exp(1j * B.K0 * S)
         * np.exp(1j * alpha * (r2 / w ** 2) ** 2)).astype(np.complex128)
    base = dict(prescription=B._singlet(), wavelength=B.LAM, dx=dx,
                carrier=rc, parallel_amp=False, on_undersample='silent',
                on_noncollimated='silent')
    cases = []
    for pip, amod in (('remap', 'ray_density'), (True, 'ray_density'),
                      (False, 'ray_density'), (True, 'screen'),
                      (False, 'screen')):
        for rs in (1, 4):
            cases.append((f"pip={pip!r} amp={amod} rs={rs}",
                          dict(preserve_input_phase=pip,
                               amplitude_model=amod, ray_subsample=rs)))
    cases.append(("remap lattice rs=4",
                  dict(preserve_input_phase='remap',
                       amplitude_model='ray_density',
                       remap_sampling='lattice', ray_subsample=4)))
    cases.append(("no carrier, remap rs=4",
                  dict(preserve_input_phase='remap', carrier=None,
                       amplitude_model='ray_density', ray_subsample=4)))
    ok = True
    old = LT.REMAP_INVERSE_SUPPORT_BOUND
    LT.REMAP_INVERSE_SUPPORT_BOUND = bool(bound)
    try:
        for _ in range(2):        # W9 warm-up, both implementations
            LT.apply_real_lens_traced(E, **base, ray_subsample=4)
            head.apply_real_lens_traced(E, **base, ray_subsample=4)
        for lbl, kw in cases:
            k = dict(base)
            k.update(kw)
            a = np.asarray(LT.apply_real_lens_traced(E, **k))
            b = np.asarray(head.apply_real_lens_traced(E, **k))
            eq = bool(np.array_equal(a, b)) and a.dtype == b.dtype
            ok &= eq
            d = np.abs(a - b)
            pk = float(np.abs(b).max())
            print(f"  {lbl:38s} array_equal={eq}  max|dE| "
                  f"{float(d.max()):.3e}  ({float(d.max()) / max(pk, 1e-300):.2e}"
                  f" of peak)  dP/P {float((np.abs(a) ** 2 - np.abs(b) ** 2).sum()) / max(float((np.abs(b) ** 2).sum()), 1e-300):+.2e}")
    finally:
        LT.REMAP_INVERSE_SUPPORT_BOUND = old
    return ok


def chain_121(head, order, bound, cases=None):
    ok = True
    orig = _EL.apply_real_lens_traced
    old = LT.REMAP_INVERSE_SUPPORT_BOUND
    LT.REMAP_INVERSE_SUPPORT_BOUND = bool(bound)
    try:
        for lbl, kw in (cases or B.CASES_121):
            _EL.apply_real_lens_traced = orig
            a = B.run_chain(order, **kw)
            _EL.apply_real_lens_traced = head.apply_real_lens_traced
            b = B.run_chain(order, **kw)
            _EL.apply_real_lens_traced = orig
            eq = bool(np.array_equal(a, b)) and a.dtype == b.dtype
            ok &= eq
            pk = float(np.abs(b).max())
            pb = float((np.abs(b) ** 2).sum())
            print(f"  {lbl:32s} array_equal={eq}  max|dE| "
                  f"{float(np.abs(a - b).max()):.3e} "
                  f"({float(np.abs(a - b).max()) / max(pk, 1e-300):.2e} of "
                  f"peak)  dP/P "
                  f"{float((np.abs(a) ** 2).sum() - pb) / max(pb, 1e-300):+.2e}",
                  flush=True)
    finally:
        _EL.apply_real_lens_traced = orig
        LT.REMAP_INVERSE_SUPPORT_BOUND = old
    return ok


def main():
    parts = os.environ.get('PARTS', 'abc')
    head, path = B.load_head_module()
    print(f"shadow module: HEAD:lumenairy/elements/_lens_traced.py -> {path}")
    print(f"  HEAD has REMAP_INVERSE_SUPPORT_BOUND: "
          f"{hasattr(head, 'REMAP_INVERSE_SUPPORT_BOUND')} (must be False -- "
          f"C8 is uncommitted)")
    print(f"  live default REMAP_INVERSE_SUPPORT_BOUND = "
          f"{LT.REMAP_INVERSE_SUPPORT_BOUND}, feather "
          f"{LT._SUPPORT_BOUND_FEATHER_CELLS:g} cells")
    print("  every other flag is left at its shipped default (C5 on, C6 on, "
          "fit guard off) -- HEAD's own settings.")
    ok = True
    for bound in (False, True):
        tag = 'OFF -- MUST BE IDENTICAL' if not bound else 'ON -- what moves'
        if 'a' in parts:
            print(f"\n=== (a) SYNTHETIC, bound {tag} ===")
            r = synthetic(head, bound)
            ok &= r if not bound else True
        if 'b' in parts:
            print(f"\n=== (b) design 121 chain (0,0), bound {tag} ===")
            r = chain_121(head, (0, 0), bound)
            ok &= r if not bound else True
        if 'c' in parts:
            print(f"\n=== (c) design 121 chain (-4,-2), bound {tag} ===")
            r = chain_121(head, (-4, -2), bound)
            ok &= r if not bound else True
    try:
        os.unlink(path)
    except OSError:
        pass
    print("\nOK -- the bound OFF is HEAD, bit for bit" if ok
          else "\nFAILED -- the bound OFF is NOT HEAD")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
