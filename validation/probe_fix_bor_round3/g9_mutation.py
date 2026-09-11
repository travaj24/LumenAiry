"""GAP 1 -- PROVE the replacement gate can FAIL, by deleting the conjunct it
protects.

A gate that asserts a conjunct is worth nothing until the conjunct's removal is
shown to break it.  The round-2 verification did exactly this to the gate round
3 retires (it still passed with the conjunct deleted).  This does it to the
replacement, and to the OLD gate for contrast, without editing the library: the
mutation is ``_stack_is_provably_passive = _stack_media_are_passive``, which is
the predicate with its incidence-lossless conjunct removed.
"""
from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _g3  # noqa: E402

import lumenairy.elements.bor.bor_solve as bs  # noqa: E402


def run(gate, mutate):
    """``('PASS'|'FAIL: ...', )`` for one gate under / without the mutation."""
    real = bs._stack_is_provably_passive
    if mutate:
        bs._stack_is_provably_passive = bs._stack_media_are_passive
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gate()
        return "PASS"
    except AssertionError as exc:
        return "FAIL: %s" % (str(exc).strip().splitlines()[0][:110],)
    except BaseException as exc:                               # noqa: BLE001
        return "%s: %s" % (type(exc).__name__, str(exc)[:110])
    finally:
        bs._stack_is_provably_passive = real


def main():
    a = _g3.arm()
    print("ARM", a["build"], a["loaded_kernel"], "t%s" % a["threads"],
          a["lumenairy_file"])
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "tests", "unit"))
    import test_fix_bor_guards_round2 as T
    gate = T.test_the_energy_screen_disarms_on_an_absorbing_incidence_medium_but_the_ceiling_does_not
    intact = run(gate, mutate=False)
    mutated = run(gate, mutate=True)
    verdict = ("LIVE" if intact == "PASS" and mutated != "PASS"
               else "VACUOUS")
    print("  intact  ->", intact)
    print("  mutated ->", mutated)
    print("  VERDICT:", verdict)
    _g3.dump("g9_mutation", dict(summary=dict(
        gate=gate.__name__, intact=intact, mutated=mutated,
        verdict=verdict)), a)
    return 0 if verdict == "LIVE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
