"""V7 -- the verifier's MUTATION MATRIX over the Phase C tests.

  python v7_mutants.py make      -> C:/tmp/vcc_mut/<id>/ (git archive HEAD +
                                    one edit each; asserts the edit applied)
Then each tree runs (runmut.sh):
  pytest tests/unit/test_pmm2d_staggered_curved_c.py [+ verifier file]
and v7_mutants.py collect gathers the failing ids per mutant into
v7_mutation_matrix.json.
"""
import json
import os
import re
import shutil
import subprocess
import sys

BASE = "C:/tmp/vcc_mut"
HERE = os.path.dirname(os.path.abspath(__file__))
SH = "lumenairy/elements/pmm/shapes2d.py"
CMP = "lumenairy/elements/pmm/_curvemap.py"
TS = "lumenairy/elements/pmm/twod_staggered.py"
SP = "lumenairy/elements/pmm/stack2d_pure.py"

MUTANTS = {
    # (file, old, new, what)
    "m01_rot_ellipse_walls_not_preimage": (SH, '''        vrot = {(u[0], v[0]): P["BL"], (u[1], v[0]): P["BR"],
                (u[1], v[1]): P["TR"], (u[0], v[1]): P["TL"]}''', '''        vrot = {(u[0], v[0]): (u[0], v[0]), (u[1], v[0]): (u[1], v[0]),
                (u[1], v[1]): (u[1], v[1]), (u[0], v[1]): (u[0], v[1])}''',
        "rotated Ellipse: the disk corners left at their (u, v) wall "
        "positions (identity) and the edges straight -- the outline drawn "
        "on the walls instead of on their images"),
    "m02_merge_drops_layer2_curves": (SH, '''                eclaims.setdefault(ekey, []).append(
                    (_curve_piece(e, ta, tb), who))''', '''                eclaims.setdefault(ekey, []).append(
                    (None if (who.startswith("layer ")
                              and not who.startswith("layer 1:"))
                     else _curve_piece(e, ta, tb), who))''',
        "the merge drops every curve of layers >= 2 (their edges straight)"),
    "m03_ignore_background_eps": (SH, '''        bgv = _as_eps(bg, "compile_shapes: background_eps")''', '''        bgv = _as_eps(1.0, "compile_shapes: background_eps")''',
        "compile_shapes / the stack ignore background_eps (vacuum)"),
    "m04_fingerprint_ignores_radius": (CMP, '''        return ("Arc", tuple(self.center), self.radius, self.theta0,
                self.theta1)''', '''        return ("Arc", tuple(self.center), self.theta0, self.theta1)''',
        "the Arc key (map fingerprint) ignores the radius"),
    "m05_no_renormalisation": (TS, '''    C = np.linalg.solve(W0, Ginv @ B)
    if H0 is None:''', '''    C = np.linalg.solve(W0, Ginv @ B)
    if True:''', "the F-B4 order-0 renormalisation dropped (bare L2)"),
    "m06_sliver_constant_halved": (SH, '''from .twod_staggered import _STAG_MIN_SEG_FRAC
''', '''from .twod_staggered import _STAG_MIN_SEG_FRAC
_STAG_MIN_SEG_FRAC = 0.5 * _STAG_MIN_SEG_FRAC
''', "the shape layer's sliver constant halved (merge, fillet, inside)"),
    "m07_viewer_draws_uv": (SP, '''            X, Y = cm.geom_points(sx, sy, U, V)[:2]
            out[side] = np.stack([X, Y], 1)''', '''            out[side] = np.stack([U, V], 1)''',
        "the viewer draws the (u, v) cell outline"),
    "m08_identity_never": (SH, '''    identity = (not curved) and bool(np.all(V[..., 0] == U1[:, None])''', '''    identity = False and bool(np.all(V[..., 0] == U1[:, None])''',
        "rectangles never recognised as the identity map"),
    "m09_claim_tol_loose": (SH, "_CLAIM_TOL = 1e-12\n", "_CLAIM_TOL = 1e-2\n",
                            "vertex / edge claims compared at 1e-2 p"),
    "m10_no_crossing_check": (SH, '''    pts = B.boundary_points(512)
    d = np.asarray(A.signed_distance(pts[:, 0], pts[:, 1]))
    return bool(np.any(d < -tol) and np.any(d > tol))''', '''    return False''', "the plan-view CROSSING test disabled"),
    "m11_no_fold_naming": (SH, '''    from numpy.polynomial.legendre import leggauss
    xg, _ = leggauss(n)
    nx, ny = tm.shape''', '''    return
    from numpy.polynomial.legendre import leggauss
    xg, _ = leggauss(n)
    nx, ny = tm.shape''', "the fold scan that names the shapes skipped"),
    "m12_chord_interior_claims": (SH, '''    s = (t - e.a) / (e.b - e.a)
    if e.curve is None:''', '''    s = (t - e.a) / (e.b - e.a)
    if True:''', "an interior crossing of a curved edge placed on the CHORD"),
    "m13_paint_reversed": (SH, '''        for sh in shapes:
            lay = sh._layout(px, py)''', '''        for sh in reversed(shapes):
            lay = sh._layout(px, py)''',
        "painting order reversed (the first shape wins)"),
    "m14_load_no_conj": (TS, '''                vals[nm + "x"] = (ix, np.conj(cx[ix]) @ Vref)
                vals[nm + "y"] = (iy, np.conj(cy[iy]) @ Vref)''', '''                vals[nm + "x"] = (ix, cx[ix] @ Vref)
                vals[nm + "y"] = (iy, cy[iy] @ Vref)''',
        "the incident L2 load without the conj on the test functions"),
    "m15_sine_phase_dropped": (SH, '''            crv = Sinusoid(base, self.amplitude, p_run / self.period_count,
                           0.0, p_run, phase=self.phase,''', '''            crv = Sinusoid(base, self.amplitude, p_run / self.period_count,
                           0.0, p_run, phase=0.0,''',
        "SinusoidalWall: the map's curve ignores phase= (the outline "
        "methods keep it)"),
    "m16_fillet_tangency_on_wall": (SH, '''        for j in (1, 2):
            verts[(u[0], v[j])] = (lox, v[j])               # left tangency
            verts[(u[3], v[j])] = (hix, v[j])               # right tangency''', '''        for j in (1, 2):
            verts[(u[0], v[j])] = (lox, v[j])               # left tangency
            verts[(u[3], v[j])] = (hix, v[j])               # right tangency
        r = r * (1.0 + 1e-3)''',
        "FilletRect: the arcs' radius 1e-3 off the declared r (outline "
        "methods keep r) -- caught only if the endpoint check is tight"),
}


def make():
    os.makedirs(BASE, exist_ok=True)
    root = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
    for mid, (f, old, new, _w) in MUTANTS.items():
        d = BASE + "/" + mid
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)
        a = subprocess.run(["git", "-C", root, "archive", "HEAD"],
                           capture_output=True, check=True).stdout
        subprocess.run(["tar", "-x", "-C", d],
                       input=a, check=True)
        p = os.path.join(d, f)
        s = open(p, encoding="utf-8").read()
        assert s.count(old) == 1, (mid, s.count(old))
        open(p, "w", encoding="utf-8").write(s.replace(old, new))
        print("made", mid)


def collect():
    out = {}
    for mid, (_f, _o, _n, what) in MUTANTS.items():
        log = os.path.join(BASE, mid, "pytest.log")
        if not os.path.exists(log):
            continue
        txt = open(log, encoding="utf-8", errors="replace").read()
        failed = sorted(set(re.findall(r"FAILED \S+::(\S+)", txt)))
        errors = sorted(set(re.findall(r"ERROR \S+::(\S+)", txt)))
        tail = [ln for ln in txt.splitlines() if re.search(
            r"passed|failed|error", ln)][-1:]
        out[mid] = {"what": what, "failed": failed, "errors": errors,
                    "tail": tail, "caught": bool(failed or errors)}
        print(f"{mid:38s} caught={bool(failed or errors)!s:5s} "
              f"{failed + errors} | {tail}")
    with open(os.path.join(HERE, "v7_mutation_matrix.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    {"make": make, "collect": collect}[sys.argv[1]]()
