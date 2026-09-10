"""P10 -- BIT-IDENTITY: hash every fixture, on this tree and on the main clone.

Run it twice with different ``PYTHONPATH``s and diff the JSON.  It asserts
which ``lumenairy`` it imported so a mis-pointed run cannot pass silently.
"""
import hashlib
import json
import os
import sys
import time
import traceback

import numpy as np
from bitid_fixtures import FIXTURES

import lumenairy

HERE = os.path.dirname(os.path.abspath(__file__))


def digest(arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


if __name__ == "__main__":
    out_name = sys.argv[1] if len(sys.argv) > 1 else "p10_bitid.json"
    root = sys.argv[2] if len(sys.argv) > 2 else None
    path = os.path.abspath(lumenairy.__file__).replace("\\", "/")
    print("lumenairy:", path, lumenairy.__version__, flush=True)
    if root is not None:
        assert path.lower().startswith(root.lower().replace("\\", "/")), (
            f"expected lumenairy under {root}, imported {path}")
    rows = {}
    for name, fn in FIXTURES.items():
        t0 = time.time()
        try:
            d = digest(fn())
            rows[name] = d
            print(f"  {name:28s} {d[:32]}  ({time.time() - t0:.1f} s)",
                  flush=True)
        except Exception as exc:                              # noqa: BLE001
            rows[name] = f"ERROR {type(exc).__name__}: {exc}"
            print(f"  {name:28s} ERROR {type(exc).__name__}: {exc}",
                  flush=True)
            traceback.print_exc()
    rows["_lumenairy"] = path
    rows["_version"] = lumenairy.__version__
    json.dump(rows, open(os.path.join(HERE, out_name), "w"), indent=1)
    print("\nwrote", out_name, flush=True)
