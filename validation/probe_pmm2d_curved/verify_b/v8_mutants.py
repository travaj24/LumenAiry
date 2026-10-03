"""V8 -- the mutation matrix of the Phase-B tests.

Each mutant is a FRESH ``git archive`` of the commit under test, extracted to
``C:/tmp/vcurved_b_mut/<name>``, with ONE source edit applied (asserted to
match exactly once); the Phase-B test file (and, with ``--verify``, this
verifier's decision file) is then run against that tree with PYTHONPATH
pinned to it.  A mutant is CAUGHT when at least one test id fails.

  python v8_mutants.py make <sha>          -- extract + patch every mutant
  python v8_mutants.py list                -- print the mutant names
Run the tests per mutant from the shell (run_mutants.sh).
"""
import os
import subprocess
import sys

BASE = "C:/tmp/vcurved_b_mut"
TS = "lumenairy/elements/pmm/twod_staggered.py"
CMF = "lumenairy/elements/pmm/_curvemap.py"

MUTANTS = {
    # 1a: the corner rule disabled at ONE singular corner of the c3 disk cell
    #     (that quadrant falls back to the tensor rule); cells with a single
    #     corner keep theirs
    "m1a_duffy_off_one_corner": (TS,
        "    return {k: sorted(v) for k, v in out.items()}\n\n\nclass _StagMapQuad",
        "    return {k: (sorted(v)[1:] if len(v) > 1 else sorted(v))\n"
        "            for k, v in out.items()}\n\n\nclass _StagMapQuad"),
    # 1b: the corner rule disabled in ONE whole singular cell (the
    #     lexicographically first)
    "m1b_duffy_off_one_cell": (TS,
        "    return {k: sorted(v) for k, v in out.items()}\n\n\nclass _StagMapQuad",
        "    keys = sorted(out)[1:]\n"
        "    return {k: sorted(out[k]) for k in keys}\n\n\nclass _StagMapQuad"),
    # 2: the C0 blend broken at one seam: cell (1, 0) blends the CHORD of its
    #    top edge while cell (1, 1) blends the curve
    "m2_c0_broken_one_seam": (CMF,
        '        Tv, Td = self.edge("h", sx, sy + 1)(s)      # top    (t = 1)\n',
        '        Tv, Td = self.edge("h", sx, sy + 1)(s)      # top    (t = 1)\n'
        '        if (sx, sy) == (1, 0):\n'
        '            Tv, Td = Line(*self.edge("h", sx, sy + 1).endpoints())(s)\n'),
    # 3: every circular arc replaced by its chord (value and derivative)
    "m3_arc_is_chord": (CMF,
        "        s = np.atleast_1d(np.asarray(s, dtype=float))\n"
        "        sw = self.theta1 - self.theta0\n"
        "        th = self.theta0 + s * sw\n",
        "        s = np.atleast_1d(np.asarray(s, dtype=float))\n"
        "        _c = self.center\n"
        "        _a = _c + self.radius * np.array([np.cos(self.theta0), "
        "np.sin(self.theta0)])\n"
        "        _b = _c + self.radius * np.array([np.cos(self.theta1), "
        "np.sin(self.theta1)])\n"
        "        return Line(_a, _b)(s)\n"
        "        sw = self.theta1 - self.theta0\n"
        "        th = self.theta0 + s * sw\n"),
    # 4: the fingerprint ignores the arc radius
    "m4_fingerprint_no_radius": (CMF,
        '        return ("Arc", tuple(self.center), self.radius, self.theta0,\n'
        '                self.theta1)\n',
        '        return ("Arc", tuple(self.center), self.theta0,\n'
        '                self.theta1)\n'),
    # 5: the four-fold symmetry broken by a tiny vertex offset: the 3 x 3
    #    circle's right-hand 45-degree vertices pushed out by 1e-4 r (the
    #    middle cell becomes the matching ellipse arcs, so the map is valid)
    "m5_symmetry_vertex_offset": (CMF,
        '    h = float(radius) / np.sqrt(2.0)\n'
        '    w = np.array([0.0, c[0] - h, c[0] + h, P])\n'
        '    wv = np.array([0.0, c[1] - h, c[1] + h, P])\n'
        '    r = float(radius)\n'
        '    curved = {("h", 1, 1): Arc(c, r, 225 * _DEG, 315 * _DEG),\n'
        '              ("h", 1, 2): Arc(c, r, 135 * _DEG, 45 * _DEG),\n'
        '              ("v", 1, 1): Arc(c, r, 225 * _DEG, 135 * _DEG),\n'
        '              ("v", 2, 1): Arc(c, r, -45 * _DEG, 45 * _DEG)}\n'
        '    return TransfiniteMap(w, wv, None, curved), w\n',
        '    h = float(radius) / np.sqrt(2.0)\n'
        '    ax = (float(radius) * (1 + 1e-4), float(radius))\n'
        '    w = np.array([0.0, c[0] - ax[0] / np.sqrt(2.0), '
        'c[0] + ax[0] / np.sqrt(2.0), P])\n'
        '    wv = np.array([0.0, c[1] - h, c[1] + h, P])\n'
        '    curved = {("h", 1, 1): EllipseArc(c, ax, 225 * _DEG, 315 * _DEG),\n'
        '              ("h", 1, 2): EllipseArc(c, ax, 135 * _DEG, 45 * _DEG),\n'
        '              ("v", 1, 1): EllipseArc(c, ax, 225 * _DEG, 135 * _DEG),\n'
        '              ("v", 2, 1): EllipseArc(c, ax, -45 * _DEG, 45 * _DEG)}\n'
        '    return TransfiniteMap(w, wv, None, curved), w\n'),
}


def make(sha):
    os.makedirs(BASE, exist_ok=True)
    for name, (path, old, new) in MUTANTS.items():
        d = os.path.join(BASE, name)
        if os.path.isdir(d):
            subprocess.run(["rm", "-rf", d], check=True)
        os.makedirs(d)
        arc = subprocess.run(["git", "-C", "C:/tmp/lum_vcurved_b", "archive",
                              sha], check=True, capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", d], input=arc, check=True)
        f = os.path.join(d, path)
        s = open(f, encoding="utf-8").read()
        assert s.count(old) == 1, (name, s.count(old))
        open(f, "w", encoding="utf-8").write(s.replace(old, new))
        print("made", name, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "make":
        make(sys.argv[2])
    else:
        print(" ".join(MUTANTS))
