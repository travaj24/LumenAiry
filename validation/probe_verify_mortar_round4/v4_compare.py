"""Flatten two ``v1_axis_identity_*.json`` trees and compare every leaf.

The two trees are the SAME probe run against ``15af675`` (pre) and ``b7239bf``
(post), in separate interpreters.  Any difference in an ANSWER leaf (the
sha256 of ``(orders, R, T)``, ``R00``, ``T00``, the lossless closure) is a
defect; the WARNING leaves are enumerated and classified (count / axis /
width).  Timings and provenance are excluded by name, not by tolerance.

Usage::

    python .../v4_compare.py --pre <tag> --post <tag> [--out <name>]
"""
from __future__ import annotations

import argparse
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent

ANSWER_KEYS = ("hash", "R00", "R00_row0", "T00", "closure")
WARNING_KEYS = ("n_warn", "axes", "widths")
#: excluded by NAME: wall-clock, provenance, and the helper that does not
#: exist on the pre tree at all
IGNORE = ("seconds", "tag", "tree", "lumenairy_file", "env", "library_axes")


def flat(o, prefix=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from flat(v, f"{prefix}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from flat(v, f"{prefix}[{i}]")
    else:
        yield prefix, o


def _kind(path):
    leaf = path.rsplit(".", 1)[-1].split("[")[0]
    if leaf in IGNORE or any(f".{g}" in path for g in ("env",)):
        return "ignored"
    if leaf in ANSWER_KEYS:
        return "answer"
    if leaf in WARNING_KEYS:
        return "warning"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True)
    ap.add_argument("--post", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    A = json.loads((HERE / f"v1_axis_identity_{a.pre}.json").read_text("utf8"))
    B = json.loads((HERE / f"v1_axis_identity_{a.post}.json").read_text("utf8"))
    common = sorted(set(A) & set(B))
    A = {k: A[k] for k in common}
    B = {k: B[k] for k in common}
    fa, fb = dict(flat(A)), dict(flat(B))
    keys = sorted(set(fa) | set(fb))
    counts = {"answer": 0, "warning": 0, "other": 0, "ignored": 0}
    compared = 0
    diffs = []
    for k in keys:
        kind = _kind(k)
        if kind == "ignored":
            continue
        compared += 1
        va, vb = fa.get(k, "<missing>"), fb.get(k, "<missing>")
        if va != vb:
            counts[kind] += 1
            diffs.append({"path": k, "kind": kind, "pre": va, "post": vb})
    res = {"pre": a.pre, "post": a.post, "sections": common,
           "leaves_compared": compared, "counts": counts, "diffs": diffs}
    dest = HERE / (a.out or f"v4_compare_{a.pre}_vs_{a.post}.json")
    dest.write_text(json.dumps(res, indent=1, sort_keys=True), encoding="utf8")
    print(f"sections compared        : {common}")
    print(f"leaves compared          : {compared}")
    print(f"ANSWER differences       : {counts['answer']}")
    print(f"warning differences      : {counts['warning']}")
    print(f"other differences        : {counts['other']}")
    for d in diffs:
        print(f"  {d['kind'].upper():7s} {d['path']}: {d['pre']} -> "
              f"{d['post']}")
    print("wrote", dest)


if __name__ == "__main__":
    main()
