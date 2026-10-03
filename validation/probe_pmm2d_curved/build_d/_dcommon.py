"""Phase D fixtures on top of ``_common`` (which pins ``lumenairy`` to this
tree): the tensors, the maps, the Berreman film oracle, and the ENGINEERED
MUTATIONS of the D9 matrix (each patches the real code path and restores it).
"""
from contextlib import contextmanager

import _common as C
import numpy as np
from _common import CM, SP, TS, PMM2DStackPure  # noqa: F401 -- re-exported

from lumenairy.elements.berreman import berreman_jones_1d
from lumenairy.elements.rcwa._core import uniaxial_tensor

#: the shipped anisotropic suite's G3 film fixture (period 0.40 um, lambda
#: 1 um, depth 0.55 um, n_sub 1.5, air above)
G3 = dict(P=0.40e-6, WL=1.0e-6, DEP=0.55e-6, NSUB=1.5, NSUP=1.0)
#: rotated in-plane uniaxial (LC-like): n_o 1.5, n_e 1.8, director in the
#: plane at 0.55 rad from x -- the shipped G3 tensor
LC = uniaxial_tensor(1.5, 1.8, np.pi / 2, phi=0.55)
#: LC with its director at 30 degrees to x (the D4 pillar)
LC30 = uniaxial_tensor(1.5, 1.8, np.pi / 2, phi=np.pi / 6)
#: gyrotropic (Hermitian, lossless), e12 = -e21 = +0.5i (PUBLIC gauge)
GYRO = np.array([[2.25, 0.5j, 0.0], [-0.5j, 2.25, 0.0], [0.0, 0.0, 2.0]],
                dtype=complex)


def stretch_map(Pp, a, n=3):
    """Separable sine stretch of BOTH axes (x by a, y by -a/2) on an n x n
    uniform (u, v) grid."""
    w = np.linspace(0.0, Pp, n + 1)
    return CM.SeparableStretch(w, w, fx=CM.SineStretch(a * Pp),
                               fy=CM.SineStretch(-0.5 * a * Pp))


def shear_map(Pp):
    """A 3 x 3 TransfiniteMap with straight edges and the four interior
    vertices MOVED: bilinear cells with g12 != 0 and a NON-diagonal J (the
    one map family here that separates J^-1 eps J^-T from J^-T eps J^-1)."""
    w = np.linspace(0.0, Pp, 4)
    V = np.stack(np.meshgrid(w, w, indexing="ij"), axis=-1)
    for (i, j), (dx, dy) in {(1, 1): (0.06, 0.04), (2, 1): (-0.05, 0.03),
                             (1, 2): (0.03, -0.05),
                             (2, 2): (-0.04, -0.06)}.items():
        V[i, j] += (dx * Pp, dy * Pp)
    return CM.TransfiniteMap(w, w, V, None)


def circle_map(Pp, kind="c3", r_frac=0.3):
    if kind == "c3":
        return CM._circle_map_3x3(Pp, r_frac * Pp)[0]
    return CM._circle_map_5x5(Pp, r_frac * Pp)[0]


def make_map(name, Pp):
    if name == "s05":
        return stretch_map(Pp, 0.05)
    if name == "s15":
        return stretch_map(Pp, 0.15)
    if name == "shear":
        return shear_map(Pp)
    if name in ("c3", "c5"):
        return circle_map(Pp, name)
    raise ValueError(name)


def film_vs_berreman(t33, cmap, M, theta=0.0, phi=0.0, f=None):
    """A UNIFORM block-form tensor film (a uniform layer of the mapped stack)
    against ``berreman_jones_1d``.  Returns (dRT, dJ, J): the largest |R, T|
    difference (orders summed per input), the largest complex Jones
    difference and the Jones matrix."""
    f = G3 if f is None else f
    st = PMM2DStackPure(f["P"], f["P"], n_superstrate=f["NSUP"],
                        n_substrate=f["NSUB"], n_modes=M, n_orders=2,
                        cmap=cmap)
    st.add_layer(f["DEP"], eps=t33)
    st.set_source(f["WL"], theta=theta, phi=phi)
    _o, R, T, J = st.solve(jones=True)
    Rb, Tb, jr, _jt = berreman_jones_1d([(t33, f["DEP"])], f["NSUB"],
                                        f["NSUP"], f["WL"], angle=theta,
                                        phi=phi)
    dRT = max(float(np.max(np.abs(np.asarray(R).sum(1) - Rb))),
              float(np.max(np.abs(np.asarray(T).sum(1) - Tb))))
    return dRT, float(np.max(np.abs(np.asarray(J) - jr))), np.asarray(J)


# ---- engineered mutations (the D9 matrix), through the REAL code path ------
MUTATIONS = ("transpose", "side", "no_sg_e33", "mixed_sign", "mu_after",
             "hgram_R")


@contextmanager
def mutate(kind):
    """Patch the ONE congruence kernel (or the weights / H-partner path) with
    a deliberate defect:

    * transpose  -- eps transposed inside the congruence (reverses gyration);
    * side       -- J^-T eps J^-1 instead of J^-1 eps J^-T (chi untouched);
    * no_sg_e33  -- sqrt(g) dropped from eps'_33;
    * mixed_sign -- the mixed (e12', e21') weights negated;
    * mu_after   -- chi_t = inverse of the CELL-AVERAGED mu' (the inverse
      taken after quadrature instead of at every node);
    * hgram_R    -- every mapped region recovers H through -R (the pencil's
      right-hand matrix) instead of the plain Gram."""
    orig = TS._stag_map_eff_tensor
    orig_w = TS._stag_map_weights
    patched = []

    def setp(mod, name, fn):
        patched.append((mod, name, getattr(mod, name)))
        setattr(mod, name, fn)

    if kind == "transpose":
        def f(eps, mu, xu, xv, yu, yv):
            return orig(np.swapaxes(eps, -1, -2), mu, xu, xv, yu, yv)
        setp(TS, "_stag_map_eff_tensor", f)
    elif kind == "side":
        def f(eps, mu, xu, xv, yu, yv):
            good = orig(eps, mu, xu, xv, yu, yv)
            bad = orig(eps, mu, xu, yu, xv, yv)       # adj(J^T): A^T eps A
            for k in ("e11", "e12", "e21", "e22"):
                good[k] = bad[k]
            return good
        setp(TS, "_stag_map_eff_tensor", f)
    elif kind == "no_sg_e33":
        def f(eps, mu, xu, xv, yu, yv):
            out = orig(eps, mu, xu, xv, yu, yv)
            out["e33"] = out["e33"] / (xu * yv - xv * yu)
            return out
        setp(TS, "_stag_map_eff_tensor", f)
    elif kind == "mixed_sign":
        def f(eps, mu, xu, xv, yu, yv):
            out = orig(eps, mu, xu, xv, yu, yv)
            out["e12"] = -out["e12"]
            out["e21"] = -out["e21"]
            return out
        setp(TS, "_stag_map_eff_tensor", f)
    elif kind == "mu_after":
        def w(bx, by, cmap, eps_cell, rule, mu_cell=None):
            out = orig_w(bx, by, cmap, eps_cell, rule, mu_cell=mu_cell)
            if mu_cell is None:
                return out
            return chi_after_quadrature(out, rule)
        setp(TS, "_stag_map_weights", w)
    elif kind == "hgram_R":
        # ONLY the Eq.-25 H partner: the patterned regions through
        # _region_modes (Ggram_blocks None -> inv(-R)) and the homogeneous
        # ones through _homog_region_modes (an extra inv(-R) carried in the
        # geom tuple); the incident decomposition keeps reading the PLAIN
        # Gram at geom[4] (a first version of this arm replaced that too and
        # conflated two defects)
        orig_a = TS.Granet2DTransverseE._assemble
        orig_hc = TS._homog_geom_cache
        orig_hm = TS._homog_region_modes

        def a(self):
            orig_a(self)
            if self.cmap is not None:
                self._true_gram = self.Ggram_blocks
                self.Ggram_blocks = None          # -> inv(-R) in _region_modes

        def hc(solver):
            if getattr(solver, "cmap", None) is None:
                return orig_hc(solver)
            solver.Ggram_blocks = solver._true_gram   # the plain Gram for
            try:                                       # geom[4] (incident)
                g = orig_hc(solver)
            finally:
                solver.Ggram_blocks = None
            return tuple(g) + (np.linalg.inv(-solver.Rmat),)

        def hm(geom, eps):
            if len(geom) == 7:
                geom = tuple(geom[:4]) + (geom[6], geom[5])
            return orig_hm(geom, eps)
        setp(TS.Granet2DTransverseE, "_assemble", a)
        setp(TS, "_homog_geom_cache", hc)
        setp(SP, "_homog_geom_cache", hc)
        setp(TS, "_homog_region_modes", hm)
        setp(SP, "_homog_region_modes", hm)
    else:
        raise ValueError(kind)
    try:
        yield
    finally:
        for mod, name, old in reversed(patched):
            setattr(mod, name, old)


def chi_after_quadrature(W, rule):
    """'mu_after': replace the chi node weights by the inverse of the
    QUADRATURE MEAN of mu' = [chi]^-1 over each cell (constant per cell)."""
    from lumenairy.elements.pmm.twod_staggered import _StagNodeWeight
    keys = ("c11", "c12", "c21", "c22", "c33")
    rt = rule.tensor if hasattr(rule, "tensor") else rule
    wg = rt[1]
    w2 = np.outer(wg, wg)
    tens = {k: (W[k].t if isinstance(W[k], _StagNodeWeight) else W[k])
            for k in keys}
    pts = {k: (W[k].p if isinstance(W[k], _StagNodeWeight) else {})
           for k in keys}
    cpts = rule.points if hasattr(rule, "points") else {}

    def inv2(c11, c12, c21, c22):
        d = c11 * c22 - c12 * c21
        return c22 / d, -c12 / d, -c21 / d, c11 / d

    m11, m12, m21, m22 = inv2(*(tens[k] for k in keys[:4]))
    m33 = 1.0 / tens["c33"]
    Nx, Ny = m11.shape[:2]
    new = {k: np.empty_like(tens[k]) for k in keys}
    newp = {k: {} for k in keys}
    for sx in range(Nx):
        for sy in range(Ny):
            cell = (sx, sy)
            if cell in cpts:
                wq = cpts[cell][2]
                mm = [np.sum(wq * v) / np.sum(wq) for v in
                      inv2(*(pts[k][cell] for k in keys[:4]))]
                mm.append(np.sum(wq / pts["c33"][cell]) / np.sum(wq))
            else:
                mm = [np.sum(w2 * a[sx, sy]) / np.sum(w2)
                      for a in (m11, m12, m21, m22, m33)]
            c = list(inv2(*mm[:4])) + [1.0 / mm[4]]
            for k, v in zip(keys, c):
                new[k][sx, sy] = v
                if cell in cpts:
                    newp[k][cell] = np.full_like(pts[k][cell], v)
    out = dict(W)
    for k in keys:
        if isinstance(W[k], _StagNodeWeight):
            out[k] = _StagNodeWeight(new[k], newp[k])
        else:
            out[k] = new[k]
    return out


dump = C.dump
env_record = C.env_record
