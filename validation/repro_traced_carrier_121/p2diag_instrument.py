# P2 DID-NOT-WARN diagnosis -- VERIFY THE INSTRUMENT BY POISONING IT.
#
# ``tests/unit/test_niche_p2_guards.py`` now routes both 'NOT dx-STABLE'
# assertions through ``_expect_dx_warning``, which attaches the guard's OWN
# margin to a DID-NOT-WARN failure.  C11's standard applies: "an instrument
# that stays silent under the condition it exists to report is worse than
# none", so this script FORCES both miss modes and prints what CI would see.
#
#   case A -- the guard RUNS and decides "stable" (huge self_check_tol).
#             Expect: a per-metric table showing every metric 'inside', which
#             says the PHYSICS moved.
#   case B -- the guard never runs at all (self_check omitted).
#             Expect: the "logged NO self-check line" branch, which says the
#             chain returned before comparing.
#
# usage:  python p2diag_instrument.py
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     '..', '..'))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'tests', 'unit'))

import lumenairy as la                                          # noqa: E402
import test_niche_p2_guards as T                                # noqa: E402


def _try(label, call, tol):
    print('=' * 78)
    print(label)
    print('=' * 78)
    try:
        T._expect_dx_warning(call, tol, None)
    except AssertionError as exc:
        print(str(exc))
        print()
        print('--> INSTRUMENT REPORTED (good)')
    else:
        print('--> the warning FIRED; nothing to report for this case')
    print()


def main():
    env0, groups, dx, _ = T._slow_singlet_chain(N=768, dx=4e-6)

    _try('CASE A -- guard runs, decides stable (self_check_tol=10.0)',
         lambda: la.propagate_traced_carrier_chain(
             env0, groups, T._WL, dx, self_check='dx', self_check_tol=10.0,
             final_leg='paraxial', **T._chain_kw(r_in=3e-3)),
         10.0)

    _try('CASE B -- guard never runs (self_check omitted)',
         lambda: la.propagate_traced_carrier_chain(
             env0, groups, T._WL, dx, final_leg='paraxial',
             **T._chain_kw(r_in=3e-3)),
         0.05)


if __name__ == '__main__':
    main()
