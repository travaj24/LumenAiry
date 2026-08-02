# niche D5's ``test_dx_flatness_alone_is_not_sufficient`` pins that the
# DELIBERATELY BROKEN configuration (``carrier_reference='parabola'``) sits
# more than 3.0x wide of the independent oracle while being dx-FLAT -- the
# measured lesson that a flatness-only gate has no teeth.  On the settled tree
# the ratio reads 2.95 and the assertion fails.
#
# WHAT HAS TO BE ESTABLISHED BEFORE ANY NUMBER IS TOUCHED: whether the broken
# configuration got CLOSER to the truth (niche C6 improving even the path it
# was not aimed at) or whether the ORACLE moved.  The oracle is
# ``validation/oracles/debye_oracle_v3.py`` -- lumenairy-free, pure numpy +
# scipy.special.j0, exact meridional raytrace through the same conic/aspheric
# surface list, ring-Huygens integral, ABSOLUTELY normalised -- so it cannot
# have moved with a library change, and this runner proves that by printing it.
#
# Everything is read through D5's own ``_ladder`` / ``_oracle`` so the numbers
# are the test's numbers and not a re-implementation.
#
# usage:  python recon_d5_oracle.py
import hashlib
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'tests', 'unit'))

import lumenairy.elements._lens_traced as LT               # noqa: E402,I001
import test_niche_d5_dx_flatness_gate as D5                # noqa: E402


def _rows(tag, chain_kwargs, c6):
    old = LT.REMAP_STATIONARY_PHASE_LAUNCH
    LT.REMAP_STATIONARY_PHASE_LAUNCH = bool(c6)
    D5._CACHE.clear()
    try:
        return D5._ladder(tag, chain_kwargs, None, ((512, 2), (1024, 4)))
    finally:
        LT.REMAP_STATIONARY_PHASE_LAUNCH = old
        D5._CACHE.clear()


def main():
    print("   lib sha256 %s" % hashlib.sha256(
        open(LT.__file__, 'rb').read()).hexdigest()[:16])
    orc = D5._oracle()
    print("D5 independent oracle (debye_oracle_v3, lumenairy-free):")
    print(f"  FWHM {orc['fwhm']:.6f} um   EE1 {orc['ee1']:.4f}  "
          f"EE2 {orc['ee2']:.4f}  EE4 {orc['ee4']:.4f}  "
          f"total {orc['total']:.4f} %")
    print("  (a library change cannot move this: no lumenairy in the call)")
    print()

    hdr = (f"  {'configuration':>34} {'N':>5} {'FWHM um':>10} "
           f"{'/oracle':>8} {'EE1':>8} {'EE2':>8} {'EE4':>8} {'window':>9}")
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    out = {}
    for c6 in (True, False):
        for tag, ck in (('defaults', None),
                        ('parabola', {'carrier_reference': 'parabola'})):
            rows = _rows(tag, ck, c6)
            out[(c6, tag)] = rows
            lbl = f"{tag} / C6 {'ON (settled)' if c6 else 'OFF (pre-C6)'}"
            for (N, _rs), r in zip(((512, 2), (1024, 4)), rows):
                print(f"  {lbl:>34} {N:>5} {r['fwhm']:>10.5f} "
                      f"{r['fwhm']/orc['fwhm']:>8.4f} {r['ee1']:>8.4f} "
                      f"{r['ee2']:>8.4f} {r['ee4']:>8.4f} {r['window']:>9.4f}")
            lbl = ''
    print()
    for c6 in (True, False):
        rows = out[(c6, 'parabola')]
        fw = [r['fwhm'] for r in rows]
        spread = (max(fw) - min(fw)) / np.mean(fw)
        print(f"  parabola, C6 {'ON ' if c6 else 'OFF'}: dx-flatness spread "
              f"{spread:.3e} (bar {D5._FLAT_FWHM_REL:g}), "
              f"FWHM/oracle {rows[-1]['fwhm']/orc['fwhm']:.4f} "
              f"(pin: > 3.0), EE2/oracle "
              f"{rows[-1]['ee2']/orc['ee2']:.4f}")
    print()
    print("BETTER = the broken configuration is closer to the absolute oracle.")
    print("The DEFECT the test documents survives only if the configuration "
          "is still")
    print("plainly wrong while still passing every flatness bar.")


if __name__ == '__main__':
    sys.exit(main())
