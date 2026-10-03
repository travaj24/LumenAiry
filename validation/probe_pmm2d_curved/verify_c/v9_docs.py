"""V9 -- EXECUTE every worked example of the Phase C docs as written:
the docstring examples of Rect, FilletRect, Circle, Ellipse, SinusoidalWall
and compile_shapes (shapes2d.py), the cookbook entry (docs/cookbook.md) and
the CHANGELOG release paragraph.  Each runs in a fresh namespace; the
compile_shapes example's stated eps_cell comment is checked.

  python v9_docs.py   -> v9_docs_<build>.json
"""
import os
import re
import time
import traceback
import warnings

import matplotlib

matplotlib.use("Agg")
from _vc import BUILD, ROOT, dump  # noqa: E402

from lumenairy.elements.pmm import shapes2d as SH  # noqa: E402

warnings.simplefilter("ignore")


def doc_example(obj):
    import inspect
    d = inspect.cleandoc(obj.__doc__) + "\n"
    m = re.search(r"Example[^:]*?::\n\n((?:    .*\n|[ ]*\n)+)", d)
    assert m, obj
    lines = [ln[4:] if ln.startswith("    ") else "" for ln in
             m.group(1).splitlines()]
    return "\n".join(lines).strip() + "\n"


def md_block(path, heading):
    s = open(path, encoding="utf-8").read()
    i = s.index(heading)
    j = s.index("```python", i)
    k = s.index("```", j + 9)
    return s[j + 9:k]


blocks = {nm: doc_example(getattr(SH, nm)) for nm in
          ("Rect", "FilletRect", "Circle", "Ellipse", "SinusoidalWall",
           "compile_shapes")}
blocks["cookbook"] = md_block(os.path.join(ROOT, "docs", "cookbook.md"),
                              "### Curved cells in the pure 2-D PMM")
blocks["changelog"] = md_block(os.path.join(ROOT, "CHANGELOG.md"),
                               "### Added -- pure 2-D PMM (curved cells, "
                               "Phase C)")
OUT = {}
for nm, code in blocks.items():
    ns = {}
    t0 = time.time()
    try:
        exec(compile(code, f"<{nm}>", "exec"), ns)
        rec = {"ok": True}
        if nm == "compile_shapes":
            import numpy as np
            want = np.array([[12.1, 12.1, 12.1], [12.1, 1, 12.1],
                             [12.1, 12.1, 12.1]])
            rec["eps_cell_as_documented"] = bool(np.allclose(
                ns["eps_cell"], want))
        for k in ("R", "T"):
            if k in ns:
                import numpy as np
                R, T = np.asarray(ns["R"]), np.asarray(ns["T"])
                rec["closure"] = float(np.max(np.abs(R.sum(1) + T.sum(1)
                                                     - 1)))
    except Exception as e:                         # noqa: BLE001
        rec = {"ok": False, "error": f"{type(e).__name__}: {e}"[:400],
               "tb": traceback.format_exc()[-800:]}
    rec["wall_s"] = time.time() - t0
    rec["code"] = code
    OUT[nm] = rec
    print(nm, {k: v for k, v in rec.items() if k not in ("code", "tb")},
          flush=True)
dump(f"v9_docs_{BUILD}.json", OUT)
