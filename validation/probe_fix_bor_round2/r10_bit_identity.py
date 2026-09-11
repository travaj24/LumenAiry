"""ROUND 2 -- BIT-IDENTITY of the VERIFICATION's own 58 BOR + 15 EME fixtures,
the verify branch tip against this tree.

WHY THE VERIFICATION'S FIXTURES AND NOT NEW ONES.  The contract round 2 has to
keep is the one the verification already established and measured: 58 of 58 BOR
fixtures and 13 of 15 EME fixtures bit-identical across the whole 5.45.1 build,
on both local builds.  Re-hashing the SAME battery is what makes "only the rows
the new screen REFUSES may change" checkable rather than asserted.  The fixture
bodies are imported from ``validation/probe_verify_bor_guards`` verbatim; only
the tree check is this probe's own, because the verification's
``_vh.require_tree`` pins its author's two clone NAMES.

Usage:
    python validation/probe_fix_bor_round2/r10_bit_identity.py <pre|post> \
        [outdir]

``pre`` must be run with ``PYTHONPATH`` pointing at a checkout of
``verify/bor-multilayer-guards``; ``post`` at this tree.  The probe prints
``lumenairy.__file__`` and refuses to guess.
"""
from __future__ import annotations

import json
import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
VDIR = os.path.join(os.path.dirname(HERE), "probe_verify_bor_guards")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("pre", "post"):
        raise SystemExit(__doc__)
    build = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else HERE

    import lumenairy
    tree = os.path.abspath(os.path.dirname(os.path.dirname(
        lumenairy.__file__)))
    print("lumenairy.__file__ = %s" % (lumenairy.__file__,))
    print("tree               = %s" % (tree,))
    # the tree must be the one asked for: PRE is the branch tip's checkout,
    # POST is the tree that contains THIS probe.
    is_post = os.path.normcase(tree) == os.path.normcase(
        os.path.dirname(os.path.dirname(HERE)))
    if (build == "post") != is_post:
        raise SystemExit(
            "TREE MISMATCH: asked for build=%r but lumenairy came from %s "
            "(this probe lives under %s).  Point PYTHONPATH at the right "
            "checkout." % (build, tree, os.path.dirname(os.path.dirname(HERE))))

    sys.path.insert(0, VDIR)
    import _vh  # noqa: E402
    import v1_bit_identity as V1  # noqa: E402

    a = _vh.arm()
    print("ARM", a, flush=True)
    rows = {}
    for name, fn in _vh.bor_fixtures():
        rows[name] = _vh.solve_fixture(fn)
        print("  BOR %-30s %s" % (name, rows[name].get("raised")
                                  or rows[name]["hash"][:16]), flush=True)
    erows = {}
    for name, fn in V1.eme_fixtures():
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                erows[name] = fn()
                erows[name]["raised"] = None
            except BaseException as e:                 # noqa: BLE001
                erows[name] = dict(raised=type(e).__name__, msg=str(e)[:300])
            erows[name]["warnings"] = [str(x.message)[:160] for x in w]
        print("  EME %-30s %s" % (name, erows[name].get("raised")
                                  or erows[name].get("hash", "?")[:16]),
              flush=True)
    ker = _vh.kernel()
    if isinstance(ker, tuple):
        ker = ker[0]
    path = os.path.join(out, "r10_bit_identity_%s_%s_%s_t%s.json"
                        % (build, a["platform"], ker, a["blas_threads"]))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dict(arm=a, build=build, tree=tree, bor=rows, eme=erows),
                  fh, indent=1, sort_keys=True, default=str)
    print("[probe] wrote %s" % (path,))


if __name__ == "__main__":
    main()
