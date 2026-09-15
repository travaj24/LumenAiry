"""Archive-to-archive bit-identity driver for WP-B11c.

    python validation/probe_wp_b11c/bi.py <parent-rev> <probe.py> [<probe.py> ...]

For each probe it extracts ``git archive <parent-rev> lumenairy`` into a
scratch tree, runs the probe TWICE in child processes -- once bound to that
archive, once bound to the working tree -- and compares the two JSON digest
maps key by key.  Exit status is non-zero on the first key that moves.

Every child carries ``OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1`` and a ``PYTHONPATH`` pinned to its own tree; the probe
asserts ``lumenairy.__file__`` before computing.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
ENV1 = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "PYTHONHASHSEED": "0"}


def make_archive(rev: str, dest: str) -> str:
    """Extract ``git archive <rev> lumenairy`` into ``dest``.

    ``B11C_ARCHIVE`` short-circuits this with an ALREADY EXTRACTED tree.  That
    is not a convenience: this repository is a git WORKTREE whose ``.git`` file
    names a Windows path, so ``git archive`` cannot run from inside WSL at all.
    The WSL arm therefore reuses the archive the Windows arm extracted -- the
    same bytes, checked by the probe's own ``lumenairy.__file__`` assertion.
    """
    pre = os.environ.get("B11C_ARCHIVE")
    if pre:
        assert os.path.isdir(os.path.join(pre, "lumenairy")), pre
        return pre
    os.makedirs(dest, exist_ok=True)
    tar = os.path.join(dest, "a.tar")
    with open(tar, "wb") as fh:
        subprocess.run(["git", "archive", rev, "lumenairy"], cwd=ROOT,
                       stdout=fh, check=True)
    subprocess.run([sys.executable, "-c",
                    "import tarfile,sys;tarfile.open(sys.argv[1]).extractall(sys.argv[2])",
                    tar, dest], check=True)
    return dest


def run_probe(probe: str, tree: str, extra_path: str = "") -> dict:
    env = dict(os.environ)
    env.update(ENV1)
    env["PYTHONPATH"] = (tree + os.pathsep + extra_path) if extra_path else tree
    env["LUM_PROBE_ROOT"] = tree
    out = subprocess.run([sys.executable, probe, tree], cwd=tree, env=env,
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"probe {probe} failed in {tree}:\n"
                         f"{out.stdout[-4000:]}\n{out.stderr[-6000:]}")
    txt = out.stdout[out.stdout.index("{"):]
    return json.loads(txt)


def main() -> int:
    rev = sys.argv[1]
    probes = sys.argv[2:]
    scratch = tempfile.mkdtemp(prefix="b11c_")
    arch = make_archive(rev, os.path.join(scratch, "parent"))
    # the probe file itself must be readable from either cwd: copy it in
    total = moved = 0
    report = {}
    for p in probes:
        name = os.path.basename(p)
        # The archive tree holds ONLY ``lumenairy/``, so the probe and its
        # helper are copied in beside it; the working tree runs the probe from
        # where it lives (nothing is written into the repository).
        shutil.copy(os.path.join(HERE, "probelib.py"),
                    os.path.join(arch, "probelib.py"))
        shutil.copy(p, os.path.join(arch, name))
        a = run_probe(os.path.join(arch, name), arch)
        b = run_probe(os.path.abspath(p), ROOT, extra_path=HERE)
        keys = sorted(set(a) | set(b))
        bad = [k for k in keys if a.get(k) != b.get(k)]
        total += len(keys)
        moved += len(bad)
        report[name] = {"keys": len(keys), "moved": bad,
                        "parent": a, "tree": b}
        print(f"{name}: {len(keys) - len(bad)}/{len(keys)} bit-identical"
              + (f"  MOVED: {bad}" if bad else ""))
    report["_summary"] = {"rev": rev, "keys": total, "moved": moved,
                          "archive": arch}
    out = os.path.join(HERE, os.environ.get("B11C_OUT", "hashes.json"))
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, sort_keys=True)
    print(f"\nTOTAL {total - moved}/{total} bit-identical against {rev}"
          f"   -> {out}")
    return 1 if moved else 0


if __name__ == "__main__":
    raise SystemExit(main())
