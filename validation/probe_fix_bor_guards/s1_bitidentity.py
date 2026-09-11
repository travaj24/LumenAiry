"""STEP 1 / STEP 2 instrument: the ordinary-geometry BIT-IDENTITY battery.

Runs the >= 30-fixture battery of ``_common.rt_fixtures`` and writes the
SHA-256 of the exact IEEE-754 bytes of ``R`` and ``T`` per fixture.  Run it
BEFORE a change and AFTER it, on both builds, and diff the JSON: step 1
(a pure refactor) and step 2's ORDINARY population both promise every hash is
unchanged.

Usage:
    python validation/probe_fix_bor_guards/s1_bitidentity.py <tag>

``<tag>`` names the output file, e.g. ``pre_win_t1`` / ``post_win_t1``.
"""
from __future__ import annotations

import sys

import _common as C

if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "run"
    C.pin_tree()
    res = C.rt_hash(verbose=True)
    ok = sum(1 for v in res.values() if "hash" in v)
    bad = {k: v for k, v in res.items() if "hash" not in v}
    print("FIXTURES %d  hashed %d  errored %d" % (len(res), ok, len(bad)))
    for k, v in bad.items():
        print("  ERROR", k, v)
    C.dump("s1_bitidentity_%s.json" % (tag,),
           dict(n_fixtures=len(res), n_hashed=ok, fixtures=res))
