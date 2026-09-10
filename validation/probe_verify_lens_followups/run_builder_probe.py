"""Run one of the BUILDER's probes (validation/probe_fix_lens_5440/ or
validation/probe_verify_lens_5440/) UNCHANGED against an arbitrary arm.

Their ``banner()`` hard-codes the builder's worktree name in a default
argument, so this driver relaxes that one assertion to the tree given on the
command line and then executes the probe's own ``main()`` -- no other line of
the probe is touched, which is what makes the re-run comparable with their
recorded JSON.

Usage: python run_builder_probe.py <probe.py> <out.json> --tree <arm tree>
       [extra probe args...]
"""
from __future__ import annotations

import os
import runpy
import sys

probe = os.path.abspath(sys.argv[1])
out = sys.argv[2]
rest = [a for a in sys.argv[3:]]
tree = None
if '--tree' in rest:
    i = rest.index('--tree')
    tree = rest[i + 1]
    rest = rest[:i] + rest[i + 2:]

here = os.path.dirname(probe)
root = os.path.dirname(here)
sys.path.insert(0, here)
sys.path.insert(0, os.path.join(root, 'probe_verify_lens_5440'))
sys.path.insert(0, os.path.join(root, 'probe_fix_lens_5440'))

import lumenairy as la  # noqa: E402

f = os.path.abspath(la.__file__).replace(os.sep, '/')
print("# ARM lumenairy.__file__ = " + f, flush=True)
if tree:
    want = os.path.abspath(tree).replace(os.sep, '/').lower()
    if not f.lower().startswith(want + '/'):
        raise SystemExit("ARM MISMATCH: " + f + " not under " + tree)

for mod in ('_fixp', '_fix'):
    try:
        m = __import__(mod)
    except ImportError:
        continue
    if hasattr(m, 'banner') and getattr(m.banner, '__defaults__', None):
        key = os.path.basename(os.path.normpath(tree or here))
        m.banner.__defaults__ = (key,)

sys.argv = [probe, out] + rest
runpy.run_path(probe, run_name='__main__')
