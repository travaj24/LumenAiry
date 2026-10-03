"""V9 -- the mutation matrix.  Builds one source-mutated COPY of the tip per
mutant under ``C:/tmp/va_mut/<name>`` (the worktree's ``lumenairy/`` is never
edited), each mutation asserted to match its anchor EXACTLY once.

usage: python v9_mutants.py make      (needs C:/tmp/va_mut/_base = git archive)
"""
import os
import shutil
import sys

BASE = r"C:\tmp\va_mut\_base"
ROOT = r"C:\tmp\va_mut"
TW = os.path.join("lumenairy", "elements", "pmm", "twod_staggered.py")
SPF = os.path.join("lumenairy", "elements", "pmm", "stack2d_pure.py")
CM = os.path.join("lumenairy", "elements", "pmm", "_curvemap.py")

MUTANTS = {
    # (1) the far-field cofactor det J J^-T replaced by J^T
    "m01_cofactor_to_JT": (TW, [(
        '''            for key, coef, xs, ys in (("xu", yv, "Bx", "Ty"),
                                      ("xv", -yu, "Tx", "By"),
                                      ("yu", -xv, "Bx", "Ty"),
                                      ("yv", xu, "Tx", "By")):''',
        '''            for key, coef, xs, ys in (("xu", xu, "Bx", "Ty"),
                                      ("xv", yu, "Tx", "By"),
                                      ("yu", xv, "Bx", "Ty"),
                                      ("yv", yv, "Tx", "By")):''')]),
    # (1b) the cofactor's two OFF-diagonal entries swapped (a transpose slip)
    "m01b_cofactor_offdiag_swap": (TW, [(
        '''                                      ("xv", -yu, "Tx", "By"),
                                      ("yu", -xv, "Bx", "Ty"),''',
        '''                                      ("xv", -xv, "Tx", "By"),
                                      ("yu", -yu, "Bx", "Ty"),''')]),
    # (2) the half-spaces' H partner through -R (plain Gram dropped)
    "m02_homog_Ginv_minusR": (TW, [(
        '''        G1g, G2g = solver.Ggram_blocks
        Ginv = np.zeros_like(G)
        Ginv[:qq, :qq] = np.linalg.inv(G1g)
        Ginv[qq:, qq:] = np.linalg.inv(G2g)''',
        '''        Ginv = np.linalg.inv(G)''')]),
    # (2b) a mapped PATTERNED layer's H partner through -R
    "m02b_region_modes_minusR": (TW, [(
        '''    if solver.Ggram_blocks is None:
        Ginv = np.linalg.inv(G)''',
        '''    if solver.Ggram_blocks is None or getattr(solver, "cmap", None) is not None:
        Ginv = np.linalg.inv(G)''')]),
    # (2c) the stack's flux Gram -R under a map
    "m02c_flux_gram_minusR": (SPF, [(
        '''            G1g, G2g = sol_h.Ggram_blocks
            G_gram = np.zeros_like(sol_h.Rmat)
            G_gram[:_qq, :_qq] = G1g
            G_gram[_qq:, _qq:] = G2g''',
        '''            G_gram = (-sol_h.Rmat).copy()''')]),
    # (3) adaptive nq pinned to 2M + 8 AT THE CALL SITE
    "m03_nq_pinned_callsite": (TW, [(
        '''            self.bx.M, _stag_map_nodes(self.bx, self.by, cmap, self.bx.M))''',
        '''            self.bx.M, 2 * self.bx.M + 8)''')]),
    # (3b) the adaptive function itself returns 2M + 8
    "m03b_nq_function_fixed": (TW, [(
        '''    from numpy.polynomial.legendre import legvander
    tol = _STAG_MAP_QUAD_TOL if tol is None else float(tol)''',
        '''    return 2 * int(M) + 8
    from numpy.polynomial.legendre import legvander
    tol = _STAG_MAP_QUAD_TOL if tol is None else float(tol)''')]),
    # (3c) the moment test watches sqrt(g) only (misses 1/sqrt(g) etc.)
    "m03c_moments_sqrtg_only": (TW, [(
        '''                    fs = (sg, 1.0 / sg, (xu * xu + yu * yu) / sg,
                          (xu * xv + yu * yv) / sg, (xv * xv + yv * yv) / sg)''',
        '''                    fs = (sg,)''')]),
    # (4) the effective tensor's off-diagonal sign flipped
    "m04_eps_offdiag_sign": (TW, [(
        '''    off = -g12 / sg''', '''    off = g12 / sg''')]),
    # (4b) the inverse-permeability off-diagonal sign flipped
    "m04b_chi_offdiag_sign": (TW, [(
        '''            "c11": g11 / sg, "c12": g12 / sg, "c21": g12 / sg,''',
        '''            "c11": g11 / sg, "c12": -g12 / sg, "c21": -g12 / sg,''')]),
    # (4c) chi33 = sqrt(g) instead of 1 / sqrt(g)
    "m04c_chi33_inverted": (TW, [(
        '''            "c22": g22 / sg, "c33": 1.0 / sg}''',
        '''            "c22": g22 / sg, "c33": sg}''')]),
    # (5) the fingerprint ignores the curve parameters
    "m05_fingerprint_no_params": (CM, [(
        '''        h.update(repr(self._key()).encode())''',
        '''        pass''')]),
    # (5b) the per-solve eig dedupe ignores the map
    "m05b_eigkey_no_map": (SPF, [(
        '''                    key = key + (cmap.fingerprint,)''',
        '''                    key = key''')]),
    # (6) the off-diagonal projector blocks never applied
    "m06_drop_P12_P21": (TW, [(
        '''    if P12 is not None:
        top = top + P12 @ Wmodes[qq:, :]
    if P21 is not None:
        bot = bot + P21 @ Wmodes[:qq, :]''',
        '''    pass''')]),
    # (7) the solver's det J <= 0 refusal removed (only non-finite refused)
    "m07_no_detJ_refusal": (TW, [(
        '''    bad = ~np.isfinite(sg) | (sg <= 0.0)''',
        '''    bad = ~np.isfinite(sg)''')]),
    # (8) validation: the boundary-onto-itself check removed
    "m08_no_boundary_check": (CM, [(
        '''            if float(np.max(np.abs(a[0]))) > tol * px:''',
        '''            if False:''')]),
    # (9) the far projector run at a FIXED 2M + 16 (phase rule dropped)
    "m09_proj_no_phase_rule": (TW, [(
        '''            nq = max(base, _stag_quad_order(M, omega))''',
        '''            nq = base''')]),
}


def make():
    for name, (rel, edits) in MUTANTS.items():
        dst = os.path.join(ROOT, name)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(BASE, dst)
        fn = os.path.join(dst, rel)
        src = open(fn, encoding="utf-8").read()
        for old, new in edits:
            n = src.count(old)
            assert n == 1, (name, n, old[:60])
            src = src.replace(old, new)
        open(fn, "w", encoding="utf-8").write(src)
        print("made", name)


if __name__ == "__main__":
    {"make": make}[sys.argv[1]]()
