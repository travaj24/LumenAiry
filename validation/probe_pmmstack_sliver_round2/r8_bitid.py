"""ROUND 2, probe 8 -- BIT-IDENTITY against the PRE-ROUND-2 tip.

Round 2 changes what happens on stacks that trip the guard.  On every stack
that does NOT carry a manufactured sliver it must change NOTHING, so this
hashes the raw bytes of every array the two shipped fixture sets return -- the
round-1 fix's 18 (``validation/probe_pmmstack_sliver/bitid_fixtures.py``) and
the verification's 21 (``validation/probe_verify_sliver/v_fixtures.py``), 39
solves covering shared and per-layer grids, the taper with the snap dormant and
ACTIVE, Bragg, conical, slant, out-of-plane and in-plane tensors, lossy layers,
an absorbing superstrate, ``solve_vs_wavelength``, ``prepare()`` + a keyed
material sweep, ``stabilize='slices'``, ``retain_internal`` +
``internal_field``, ``layer_absorption``, ``per_order_amplitudes``, and
``_pmm_union_grid``'s 2-tuple on six geometries.

Run it TWICE -- once against this worktree, once against a read-only copy of
the pre-round-2 tip -- and diff the JSON:

    python validation/probe_pmmstack_sliver_round2/r8_bitid.py new.json
    PYTHONPATH=<tipref> python .../r8_bitid.py old.json
    python .../r8_bitid.py --compare new.json old.json
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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "probe_pmmstack_sliver"))
sys.path.insert(0, os.path.join(ROOT, "probe_verify_sliver"))


def _hash(obj):
    h = hashlib.sha256()
    for a in (obj if isinstance(obj, (list, tuple)) else [obj]):
        if isinstance(a, (list, tuple)):
            h.update(_hash(a).encode())
            continue
        if a is None:
            h.update(b"None")
            continue
        arr = np.asarray(a)
        if arr.dtype == object:
            h.update(repr(a).encode())
            continue
        h.update(str(arr.dtype).encode())
        h.update(str(arr.shape).encode())
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def collect():
    import bitid_fixtures  # round 1's 18
    import v_fixtures  # verification's 21
    out = {}
    for mod, tag in ((bitid_fixtures, "fix"), (v_fixtures, "verify")):
        book = None
        for name in ("FIXTURES", "ALL", "CASES"):
            book = getattr(mod, name, None)
            if book is not None:
                break
        if book is None:
            book = {k: v for k, v in vars(mod).items()
                    if callable(v) and k.startswith("f")}
        items = (book.items() if isinstance(book, dict)
                 else [(getattr(f, "__name__", str(i)), f)
                       for i, f in enumerate(book)])
        for name, fn in items:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out[f"{tag}:{name}"] = _hash(fn())
    return out


def collect_warnings():
    """The other half of "nothing changes": what each fixture WARNS.  Round 2
    adds two new warnings (the truncation note on the plain super-unity
    warning, and the within-layer arm), so a fixture that was quiet before must
    still be quiet."""
    import bitid_fixtures
    import v_fixtures
    out = {}
    for mod, tag in ((bitid_fixtures, "fix"), (v_fixtures, "verify")):
        book = getattr(mod, "FIXTURES")
        items = (book.items() if isinstance(book, dict)
                 else [(f.__name__, f) for f in book])
        for name, fn in items:
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                try:
                    fn()
                except Exception as exc:                 # noqa: BLE001 - probe
                    out[f"{tag}:{name}"] = ["RAISED " + str(exc)[:70]]
                    continue
            out[f"{tag}:{name}"] = sorted({str(w.message)[:70] for w in rec})
    return out


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        a = json.load(open(sys.argv[2]))["hashes"]
        b = json.load(open(sys.argv[3]))["hashes"]
        keys = sorted(set(a) | set(b))
        same = [k for k in keys if a.get(k) == b.get(k) and k in a and k in b]
        diff = [k for k in keys if a.get(k) != b.get(k)]
        wa = json.load(open(sys.argv[2])).get("warnings", {})
        wb = json.load(open(sys.argv[3])).get("warnings", {})
        wdiff = [k for k in sorted(set(wa) | set(wb)) if wa.get(k) != wb.get(k)]
        print(f"  fixtures {len(keys)}: IDENTICAL {len(same)}, differing "
              f"{len(diff)}; WARNING sets differing {len(wdiff)}")
        for k in wdiff:
            print(f"    WARNS DIFFER {k}: {wa.get(k)} vs {wb.get(k)}")
        for k in diff:
            print(f"    DIFFERS {k}: {a.get(k)} vs {b.get(k)}")
        for k in sorted(same):
            print(f"    {k:44s} {a[k][:32]}")
        return 0 if not diff else 1
    out_path = (sys.argv[1] if len(sys.argv) > 1
                else os.path.join(HERE, "r8_bitid.json"))
    lib = os.path.abspath(lumenairy.__file__)
    print("lumenairy:", lib, lumenairy.__version__)
    # This probe runs against TWO trees on purpose -- the worktree and the
    # read-only copy of the pre-round-2 tip -- so it asserts that it imported
    # one of them and records which in the JSON.
    _p = lib.replace("\\", "/")
    assert "lum_sliver2" in _p or "tipref" in _p, lib
    hashes = collect()
    warns = collect_warnings()
    print(f"  {len(hashes)} fixtures hashed, {len(warns)} warning sets "
          f"recorded ({sum(1 for v in warns.values() if v)} non-empty)")
    with open(out_path, "w") as f:
        json.dump(dict(meta=dict(lumenairy=lib, python=sys.version.split()[0],
                                 numpy=np.__version__), hashes=hashes,
                       warnings=warns), f, indent=1)
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
