"""Quantify 'audit-trail' history comments vs code-describing comments."""
import os, re, io, tokenize, collections

ROOT = r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy"
FILES = ["lumenairy/elements/_lens_real.py", "lumenairy/elements/_lens_traced.py",
         "lumenairy/propagators/carrier.py", "lumenairy/elements/_lens_imap.py",
         "lumenairy/elements/lenses_maslov.py", "lumenairy/propagators/fft_infra.py"]

# a "history" marker: version stamp, audit id, dated note, "pre-fix", "used to", "v5.xx"
HIST = re.compile(
    r"\bv\d+\.\d+(\.\d+)?\b|\baudit\b|\bAUDIT\b|\bpre-?fix\b|\bpre-v\d|\bused to\b|"
    r"\bpreviously\b|\bregression\b|\bROADMAP\b|\bCHANGELOG\b|\b20\d\d[-_]\d\d[-_]\d\d\b|"
    r"\bgap\s*\d|\bG\d+\b|\bround\s*\d\b|\bwalker\b|\bcloses?\b\s+\w*\d|\bbefore the fix\b|"
    r"\bthe old\b|\bformerly\b|\bhistoric", re.I)

grand = collections.Counter()
for rel in FILES:
    p = os.path.join(ROOT, rel)
    src = open(p, encoding="utf-8", errors="replace").read()
    lines = src.splitlines()
    n = len(lines)
    comment_lines = set()
    hist_lines = set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT:
                comment_lines.add(tok.start[0])
                if HIST.search(tok.string):
                    hist_lines.add(tok.start[0])
    except Exception:
        pass
    # extend a history block: consecutive comment lines after a history-marked one
    changed = True
    while changed:
        changed = False
        for ln in sorted(comment_lines):
            if ln in hist_lines:
                continue
            if (ln - 1 in hist_lines or ln + 1 in hist_lines) and \
                    (ln - 1 in comment_lines or ln + 1 in comment_lines):
                hist_lines.add(ln)
                changed = True
    # docstrings containing history markers
    doc_hist = 0
    doc_tot = 0
    import ast
    try:
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
                b = getattr(node, "body", None)
                if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) \
                        and isinstance(b[0].value.value, str):
                    c = b[0].value
                    span = (c.end_lineno or c.lineno) - c.lineno + 1
                    doc_tot += span
                    txt = c.value
                    hm = len(HIST.findall(txt))
                    if hm >= 2:
                        doc_hist += span
    except SyntaxError:
        pass
    print(f"\n### {rel}   ({n} lines)")
    print(f"    comment lines            : {len(comment_lines):6d}  "
          f"({100*len(comment_lines)/n:5.1f}% of file)")
    print(f"    of which HISTORY comments: {len(hist_lines):6d}  "
          f"({100*len(hist_lines)/max(len(comment_lines),1):5.1f}% of comments, "
          f"{100*len(hist_lines)/n:5.1f}% of file)")
    print(f"    docstring lines          : {doc_tot:6d}")
    print(f"    of which HISTORY-heavy   : {doc_hist:6d}  "
          f"({100*doc_hist/max(doc_tot,1):5.1f}%)")
    print(f"    TOTAL history lines      : {len(hist_lines)+doc_hist:6d}  "
          f"({100*(len(hist_lines)+doc_hist)/n:5.1f}% of the file)")
    grand["lines"] += n
    grand["comment"] += len(comment_lines)
    grand["hist"] += len(hist_lines)
    grand["doc"] += doc_tot
    grand["dochist"] += doc_hist

print(f"\n\n### TOTAL over the 6 files: {grand['lines']} lines")
print(f"    history comment+docstring lines: {grand['hist']+grand['dochist']} "
      f"({100*(grand['hist']+grand['dochist'])/grand['lines']:.1f}%)")

# whole-package sweep
tot = hist = 0
per = []
for dp, dn, fn in os.walk(os.path.join(ROOT, "lumenairy")):
    dn[:] = [d for d in dn if d != "__pycache__"]
    for f in fn:
        if not f.endswith(".py"):
            continue
        p = os.path.join(dp, f)
        src = open(p, encoding="utf-8", errors="replace").read()
        n = src.count("\n") + 1
        c = h = 0
        try:
            for tok in tokenize.generate_tokens(io.StringIO(src).readline):
                if tok.type == tokenize.COMMENT:
                    c += 1
                    if HIST.search(tok.string):
                        h += 1
        except Exception:
            pass
        tot += c
        hist += h
        if c >= 40:
            per.append((h / c, h, c, n, os.path.relpath(p, ROOT).replace("\\", "/")))
print(f"\n### WHOLE PACKAGE: {tot} comment lines, {hist} carry a history marker "
      f"({100*hist/max(tot,1):.1f}%)")
per.sort(reverse=True)
print("\n### top 15 files by fraction of comments that are HISTORY:")
for r, h, c, n, f in per[:15]:
    print(f"   {100*r:5.1f}%  {h:5d}/{c:5d} comments  ({n} lines)  {f}")
