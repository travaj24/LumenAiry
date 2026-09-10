"""W1 -- BIT-IDENTITY of every untouched path, round 2 vs the pre-round-2 tip.

sha256 over ``(dtype, shape, raw buffer)`` of every array each of the 28
fixtures returns, AND the set of warning messages it raises -- round 2 adds
two new warnings, so silence-identity is half the claim.

Run with ``--tag new`` on this worktree and ``--tag tip`` with PYTHONPATH on
the read-only ``bb0527a`` worktree, then ``--compare`` to diff the JSONs.
"""
import argparse
import hashlib
import json
import os
import sys
import traceback
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _hash(payload):
    h = hashlib.sha256()
    for k in sorted(payload):
        a = np.asarray(payload[k])
        h.update(k.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="new")
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", nargs=2, default=None)
    a = ap.parse_args()

    if a.compare:
        A = json.load(open(a.compare[0]))
        B = json.load(open(a.compare[1]))
        same = diff = err = wdiff = 0
        for k in sorted(set(A["rows"]) | set(B["rows"])):
            ra, rb = A["rows"].get(k), B["rows"].get(k)
            if ra is None or rb is None or ra.get("error") or rb.get("error"):
                err += 1
                print(f"  ERROR   {k}: {(ra or {}).get('error')} | "
                      f"{(rb or {}).get('error')}")
                continue
            if ra["hash"] == rb["hash"]:
                same += 1
            else:
                diff += 1
                print(f"  DIFFERS {k}: {ra['hash'][:16]} vs {rb['hash'][:16]}")
            if ra["warnings"] != rb["warnings"]:
                wdiff += 1
                print(f"  WARN-SET DIFFERS {k}:\n    A={ra['warnings']}\n"
                      f"    B={rb['warnings']}")
        print(f"\n{same} identical / {diff} differing / {err} errors; "
              f"{wdiff} warning-set differences")
        print(f"A: {A['lumenairy']}\nB: {B['lumenairy']}")
        return 0 if (diff == 0 and err == 0 and wdiff == 0) else 1

    from w_fixtures import BITID

    import lumenairy
    rows = {}
    for name, fn in BITID.items():
        try:
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                payload = fn()
            rows[name] = dict(
                hash=_hash(payload),
                warnings=sorted({str(w.message)[:200] for w in rec}))
        except Exception as exc:                          # noqa: BLE001
            rows[name] = dict(error=f"{type(exc).__name__}: {exc}",
                              tb=traceback.format_exc()[-600:])
    out = dict(tag=a.tag, lumenairy=lumenairy.__file__,
               python=sys.version.split()[0], numpy=np.__version__,
               rows=rows)
    path = a.out or os.path.join(HERE, f"w1_bitid_{a.tag}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    n_err = sum(1 for r in rows.values() if "error" in r)
    n_warn = sum(1 for r in rows.values() if r.get("warnings"))
    print(f"{len(rows)} fixtures, {n_err} errors, {n_warn} warn -> {path}")
    print(f"lumenairy: {lumenairy.__file__}")
    for k in sorted(rows):
        r = rows[k]
        print(f"  {k:34s} {r.get('hash', 'ERROR')[:32]} "
              f"{'WARNS' if r.get('warnings') else ''}"
              f"{r.get('error', '')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
