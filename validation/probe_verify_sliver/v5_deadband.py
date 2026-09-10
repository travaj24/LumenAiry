"""VERIFY follow-up 1 -- the ``min_feature`` round-off DEADBAND, two-sided.

Runs the SAME deterministic case set against whichever ``lumenairy`` is on
``PYTHONPATH``, so the pre-fix arm is the READ-ONLY main clone and the post-fix
arm is this tree.  Every case records the union widths; the comparison then
says exactly which cases moved and whether each one sits ON the threshold.

    python validation/probe_verify_sliver/v5_deadband.py <out.json>
"""
import hashlib
import json
import os
import sys
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

import lumenairy
from lumenairy.elements.pmm._core import _pmm_union_grid


def _segs(walls, eps):
    out, prev = [], 0.0
    for w, e in zip(list(walls) + [1.0], list(eps)):
        out.append((w - prev, e))
        prev = w
    return [s for s in out if s[0] > 0.0]


def cases():
    """Deterministic: 3 fixtures x 6 min_feature x 401 log deltas 0.01..100 x
    mf (the threshold included), plus 4000 pseudo-random two-layer layouts and
    200 random 6-slice tapers."""
    rng = np.random.default_rng(20260911)
    for A, B in ((0.27865, 0.62505), (0.311, 0.688), (0.1907, 0.5533)):
        for mf in (1e-6, 1e-5, 3e-5, 1e-4, 1e-3, 1e-2):
            for f in np.geomspace(1e-2, 1e2, 401):
                d = float(mf * f)
                yield (f"pair_{A}_{mf:g}_{f:.6g}",
                       [_segs([A, B], [1.0, 4.0, 1.0]),
                        _segs([A - d, B + d], [1.0, 4.0, 1.0])], mf,
                       abs(d - mf) <= 16.0 * float(np.finfo(float).eps))
    for i in range(4000):
        A = float(rng.uniform(0.05, 0.45))
        B = float(rng.uniform(0.55, 0.95))
        mf = float(10.0 ** rng.uniform(-6, -2))
        d = float(mf * 10.0 ** rng.uniform(-1.5, 1.5))
        yield (f"rand_{i}", [_segs([A, B], [1.0, 4.0, 1.0]),
                             _segs([A - d, B + d], [1.0, 4.0, 1.0])], mf,
               abs(d - mf) <= 16.0 * float(np.finfo(float).eps))
    for i in range(200):
        mf = float(10.0 ** rng.uniform(-5, -2.5))
        step = float(mf * 10.0 ** rng.uniform(-1, 1))
        a0 = float(rng.uniform(0.2, 0.35))
        segs = [_segs([a0 - step * k, 0.70 + step * k], [1.0, 4.0, 1.0])
                for k in range(6)]
        yield (f"taper_{i}", segs, mf,
               abs(step - mf) <= 16.0 * float(np.finfo(float).eps))


def main():
    out_path = sys.argv[1]
    lib = os.path.abspath(lumenairy.__file__)
    print("lumenairy:", lib, lumenairy.__version__)
    rows = {}
    h = hashlib.sha256()
    for tag, segs, mf, on_thr in cases():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            uw, _e = _pmm_union_grid(segs, mf)
        a = np.ascontiguousarray(np.asarray(uw, dtype=float))
        h.update(tag.encode())
        h.update(a.tobytes())
        rows[tag] = dict(n=int(a.size),
                         digest=hashlib.sha256(a.tobytes()).hexdigest()[:24],
                         on_threshold=bool(on_thr))
    print("cases:", len(rows), " global sha256:", h.hexdigest()[:32])
    with open(out_path, "w") as f:
        json.dump(dict(meta=dict(lumenairy=lib, python=sys.version.split()[0],
                                 numpy=np.__version__,
                                 sha=h.hexdigest()), rows=rows), f)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
