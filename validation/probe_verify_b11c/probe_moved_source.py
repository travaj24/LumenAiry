"""VERIFY-B11c: did the moved code move VERBATIM?

A refactor whose contract is bit identity has a stronger, cheaper check
available than any numeric probe: the moved definitions' SOURCE TEXT must be
byte-identical between the base file and the leaf it moved to.  This walks the
two trees with ``ast`` -- no import, no execution -- and for every name the
report says moved, compares the exact source segment (``ast.get_source_segment``
on the raw file text) at both ends, plus the module-level constant assignments.

It also reports names the report did NOT list that changed module, which is the
adversarial half: a silent passenger.

argv: --base <tree> --branch <tree> [--out F]
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os


def _defs(path):
    """{name: (kind, source-text)} for every module-level def/class/assign."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    tree = ast.parse(text, filename=path)
    out = {}
    for node in tree.body:
        seg = ast.get_source_segment(text, node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[node.name] = ("def", seg)
        elif isinstance(node, ast.ClassDef):
            out[node.name] = ("class", seg)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out[t.id] = ("assign", seg)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target,
                                                            ast.Name):
            out[node.target.id] = ("annassign", seg)
    return out


def _sha(s):
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]


class _StripDocstrings(ast.NodeTransformer):
    """Removes docstrings so a prose-only edit can be told apart from a code
    edit.  A refactor is entitled to rewrite the comment that describes where a
    name now lives; it is NOT entitled to change a statement."""

    def _strip(self, node):
        self.generic_visit(node)
        body = node.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
        return node

    visit_FunctionDef = _strip
    visit_AsyncFunctionDef = _strip
    visit_ClassDef = _strip
    visit_Module = _strip


def _code_only(seg):
    """``ast.dump`` of the segment with docstrings removed and no positions --
    equal iff the EXECUTABLE code is the same."""
    if seg is None:
        return None
    try:
        tree = ast.parse(seg)
    except SyntaxError:
        return "<unparsable>"
    tree = _StripDocstrings().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


MOVES = {
    # name: (base file, branch file)
    "_BLAS_STATE": ("rcwa/_core.py", "rcwa/_blas.py"),
    "_BLAS_WARNED_UNCONTROLLABLE": ("rcwa/_core.py", "rcwa/_blas.py"),
    "_BLAS_CONTROLLER": ("rcwa/_core.py", "rcwa/_blas.py"),
    "_BLAS_CONTROLLER_UNAVAILABLE": ("rcwa/_core.py", "rcwa/_blas.py"),
    "_BLAS_CONTROLLER_LOCK": ("rcwa/_core.py", "rcwa/_blas.py"),
    "_get_blas_threads": ("rcwa/_core.py", "rcwa/_blas.py"),
    "_threadpoolctl_available": ("rcwa/_core.py", "rcwa/_blas.py"),
    "_warn_blas_uncontrollable": ("rcwa/_core.py", "rcwa/_blas.py"),
    "set_blas_threads": ("rcwa/_core.py", "rcwa/_blas.py"),
    "rcwa_blas_threads": ("rcwa/_core.py", "rcwa/_blas.py"),
    "_blas_threads_quiet": ("rcwa/_core.py", "rcwa/_blas.py"),
    "_get_blas_controller": ("rcwa/_core.py", "rcwa/_blas.py"),
    "_blas_limit": ("rcwa/_core.py", "rcwa/_blas.py"),
    "_with_blas_limit": ("rcwa/_core.py", "rcwa/_blas.py"),
    "CUPY_AVAILABLE": ("lenses.py", "_lens_kernels.py"),
    "cp": ("lenses.py", "_lens_kernels.py"),
    "_ensure_cupy_loaded": ("lenses.py", "_lens_kernels.py"),
    "_is_cupy_array": ("lenses.py", "_lens_kernels.py"),
    "NUMEXPR_AVAILABLE": ("lenses.py", "_lens_kernels.py"),
    "_ne": ("lenses.py", "_lens_kernels.py"),
    "_ensure_numexpr_loaded": ("lenses.py", "_lens_kernels.py"),
    "_NUMBA_AVAILABLE": ("lenses.py", "_lens_kernels.py"),
    "_numba": ("lenses.py", "_lens_kernels.py"),
    "_njit": ("lenses.py", "_lens_kernels.py"),
    "_prange": ("lenses.py", "_lens_kernels.py"),
    "_NUMBA_KERNELS": ("lenses.py", "_lens_kernels.py"),
    "_load_numba": ("lenses.py", "_lens_kernels.py"),
    "_get_aspheric_sag_accum_numba": ("lenses.py", "_lens_kernels.py"),
    "surface_sag_general": ("lenses.py", "_lens_kernels.py"),
    "_surface_sag_general": ("lenses.py", "_lens_kernels.py"),
    "surface_sag_biconic": ("lenses.py", "_lens_kernels.py"),
    "_fit_normaliser": ("lenses.py", "_lens_kernels.py"),
    "_multi_indices_total_degree": ("lenses.py", "_lens_kernels.py"),
}

WATCH = ["rcwa/_core.py", "rcwa/_blas.py", "lenses.py", "_lens_kernels.py",
         "_lens_real.py", "lenses_maslov.py", "_lens_thin.py",
         "_lens_traced.py", "lenses_gbd.py"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--branch", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    def _p(tree, rel):
        return os.path.join(tree, "lumenairy", "elements", rel)

    base_defs, branch_defs = {}, {}
    for rel in WATCH:
        for tree, store in ((a.base, base_defs), (a.branch, branch_defs)):
            path = _p(tree, rel)
            store[rel] = _defs(path) if os.path.exists(path) else {}

    verdicts = {}
    for name, (src_rel, dst_rel) in MOVES.items():
        old = base_defs.get(src_rel, {}).get(name)
        new = branch_defs.get(dst_rel, {}).get(name)
        if old is None:
            verdicts[name] = "MISSING-AT-BASE"
        elif new is None:
            verdicts[name] = "MISSING-AT-BRANCH"
        elif old[1] == new[1]:
            verdicts[name] = "VERBATIM"
        elif _code_only(old[1]) == _code_only(new[1]):
            verdicts[name] = "CODE-IDENTICAL-DOCSTRING-CHANGED"
        else:
            verdicts[name] = f"CODE-CHANGED base={_sha(old[1])} branch={_sha(new[1])}"

    # the adversarial half: every module-level name in the watched files whose
    # OWNING file changed, or whose source text changed, that the report does
    # not list.
    base_owner, branch_owner = {}, {}
    for rel in WATCH:
        for nm in base_defs[rel]:
            base_owner.setdefault(nm, []).append(rel)
        for nm in branch_defs[rel]:
            branch_owner.setdefault(nm, []).append(rel)

    relocated = {}
    for nm, owners in base_owner.items():
        new_owners = branch_owner.get(nm, [])
        if sorted(owners) != sorted(new_owners):
            relocated[nm] = {"base": sorted(owners),
                             "branch": sorted(new_owners),
                             "listed": nm in MOVES}

    text_changed = {}
    for rel in WATCH:
        for nm, (_k, seg) in base_defs[rel].items():
            nb = branch_defs[rel].get(nm)
            if nb is not None and nb[1] != seg:
                text_changed.setdefault(rel, []).append(nm)

    doc = {"moved_verbatim": verdicts,
           "moved_not_verbatim": sorted(k for k, v in verdicts.items()
                                        if v != "VERBATIM"),
           "moved_code_changed": sorted(k for k, v in verdicts.items()
                                        if v.startswith("CODE-CHANGED")),
           "relocated_names": relocated,
           "relocated_unlisted": sorted(k for k, v in relocated.items()
                                        if not v["listed"]),
           "text_changed_in_place": text_changed}
    out = json.dumps(doc, indent=1)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(out)
        print(f"[moved-source] {len(verdicts)} names, "
              f"{len(doc['moved_not_verbatim'])} not verbatim, "
              f"{len(doc['relocated_unlisted'])} unlisted relocations -> "
              f"{a.out}")
    else:
        print(out)


if __name__ == "__main__":
    main()
