"""V12: the API examples of the E3 BUILD doc (section 2.5, three examples
in one block) and of the CHANGELOG entry, EXECUTED AS WRITTEN (the code text
is read from the documents themselves, n_modes = 6 as printed), timed, on
this build; plus a NumPy cross-check of their numbers.

    python v12_api_examples.py
"""
import os
import re
import sys
import time

from _ve3 import ROOT, dump, np

out = {}


def block_after(path, anchor, lang="python"):
    txt = open(path, encoding="utf-8").read()
    i = txt.index(anchor)
    m = re.search(r"```" + lang + r"\n(.*?)```", txt[i:], re.S)
    return m.group(1)


build = os.path.join(ROOT, "docs", "audits",
                     "BUILD_PMM2D_CURVED_E3_2026_10_03.md")
chlog = os.path.join(ROOT, "CHANGELOG.md")
codes = {
    "build_2_5": block_after(build, "### 2.5 The API"),
    "changelog": block_after(chlog, "### Added -- pure 2-D PMM: a JAX twin"),
}
for name, code in codes.items():
    ns = {}
    t = time.perf_counter()
    try:
        exec(compile(code, name, "exec"), ns)
        out[name] = {"ok": True, "s": time.perf_counter() - t}
        for k in ("dT_dr", "g", "T"):
            if k in ns:
                out[name][k] = np.asarray(ns[k]).real.ravel().tolist()[:8]
    except Exception as exc:  # noqa: BLE001 -- the probe records the outcome
        out[name] = {"ok": False, "error": repr(exc)[:300],
                     "s": time.perf_counter() - t}
    print(name, out[name], flush=True)
# NumPy cross-check of dT/dr (FD on the NumPy solve, n_modes = 6)
from lumenairy.elements.pmm import Circle, PMM2DStackPure  # noqa: E402


def t_np(r):
    s = PMM2DStackPure(1.2, 1.2, n_superstrate=1.0, n_substrate=1.45,
                       n_modes=6, n_orders=2)
    s.add_layer(0.5, shapes=[Circle(0.6, 0.6, r, eps=4.0)],
                background_eps=1.0)
    s.set_source(1.0)
    return s.solve()[2][0, 12]


h = 1e-4 * 1.2
out["numpy_fd_dT_dr"] = float((t_np(0.36 + h) - t_np(0.36 - h)) / (2 * h))
print(out["numpy_fd_dT_dr"])
print(dump("v12_api_examples.json", out))
sys.exit(0 if all(v.get("ok") for k, v in out.items()
                  if isinstance(v, dict)) else 1)
