"""VERIFY task 1 -- sha256 bit-identity of every shipped PMMStack path.

Run once against THIS tree (fix + wave2 merged) and once against the READ-ONLY
main clone at D:/... (a68a0da, 5.44.0, WITHOUT the fix).  Every array a fixture
returns is hashed over dtype, shape and raw buffer, so a one-ULP change moves
the digest.

    python validation/probe_verify_sliver/v1_bitid.py <out.json> [expect_root]
"""
import hashlib
import json
import os
import sys
import traceback

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

import lumenairy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v_fixtures  # noqa: E402


def _hash(d):
    h = hashlib.sha256()
    for k in sorted(d):
        a = np.ascontiguousarray(d[k])
        h.update(k.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def main():
    out_path = sys.argv[1]
    expect_root = sys.argv[2] if len(sys.argv) > 2 else None
    lib = os.path.abspath(lumenairy.__file__)
    if expect_root is not None:
        assert lib.lower().replace("\\", "/").startswith(
            expect_root.lower().replace("\\", "/")), (lib, expect_root)
    rows = {}
    for fn in v_fixtures.FIXTURES:
        try:
            d = fn()
            rows[fn.__name__] = {"hash": _hash(d),
                                 "keys": sorted(d),
                                 "shapes": {k: list(np.shape(d[k]))
                                            for k in sorted(d)}}
        except Exception as exc:                             # noqa: BLE001
            rows[fn.__name__] = {"error": f"{type(exc).__name__}: {exc}",
                                 "tb": traceback.format_exc()[-800:]}
        print(f"{fn.__name__:38s} "
              f"{rows[fn.__name__].get('hash', rows[fn.__name__].get('error'))[:48]}",
              flush=True)
    meta = {"lumenairy": lib, "version": lumenairy.__version__,
            "python": sys.version.split()[0], "numpy": np.__version__,
            "has_sliver_guard": None}
    try:
        import lumenairy.elements.pmm.stack as ps
        meta["has_sliver_guard"] = bool(hasattr(ps, "PMM_SLIVER_GUARD"))
    except Exception:                                        # noqa: BLE001
        pass
    with open(out_path, "w") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=1)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
