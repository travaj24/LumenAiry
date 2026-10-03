"""The verifier's engineered Phase E1 defects, each reached through the real
code path and restored afterwards.  ``vmutate(kind)`` is a context manager;
``ve1_mutplugin.py`` applies one for a whole pytest session.

kinds
  tau_J          tau = J t (the Jacobian, not its inverse) in the composite
                 slant weights (kappa = chi_t tau follows)
  tau_t          tau = t (the shear not composed with the map)
  qz_off         the QZ fallback forced off: _bgen_hermitian = True after
                 every general assembly (Cholesky on a non-Hermitian B)
  chi33_unmapped chi33 = 1 / mu33 under a map (the material's, without sg)
  chi_unmapped   chi_t and chi33 from the material mu alone under a map
                 (the map's metric dropped from the permeability blocks)
  oop_over_sg    the congruence's out-of-plane entries divided by sg (the
                 in-plane entries' 1 / sg wrongly applied to them)
  oop_adjT       e'_a3 = adj(J)_ka e_k3 (the transposed cofactor)
  double_rot     the rotation-gauge sign applied a SECOND time to a
                 slanted mapped cell's (pre-rotated) out-of-plane weights
  flux_R         the flux split whitened with B's E rows (-R) instead of
                 the plain Gram of the G rows
  mu_blocks_off  chi_t = I, chi33 = 1 in the general generator under a map
"""
import inspect
import textwrap
from contextlib import contextmanager

import numpy as np

from lumenairy.elements.pmm import stack2d_pure as SP, twod_staggered as TS

KINDS = ("tau_J", "tau_t", "qz_off", "chi33_unmapped", "chi_unmapped",
         "oop_over_sg", "oop_adjT", "double_rot", "flux_R", "mu_blocks_off")


def _set(undo, obj, nm, val):
    undo.append((obj, nm, getattr(obj, nm)))
    setattr(obj, nm, val)


def _flux_R_fn():
    src = inspect.getsource(TS._region_modes_oop)
    a = "L1 = np.linalg.cholesky(Bmat[3 * qq:, 3 * qq:]).conj().T"
    b = "L2 = np.linalg.cholesky(Bmat[2 * qq:3 * qq, 2 * qq:3 * qq]).conj().T"
    assert a in src and b in src
    src = src.replace(a, "L1 = np.linalg.cholesky(Bmat[:qq, :qq]).conj().T")
    src = src.replace(b, "L2 = np.linalg.cholesky(Bmat[qq:2 * qq, "
                         "qq:2 * qq]).conj().T")
    ns = {}
    exec(compile(textwrap.dedent(src), TS.__file__, "exec"), TS.__dict__, ns)
    return ns["_region_modes_oop"]


@contextmanager
def vmutate(kind):
    undo = []
    try:
        if kind in ("tau_J", "tau_t"):
            orig = TS._stag_map_slant_weights

            def f(o, A, sg, tvec):
                orig(o, A, sg, tvec)
                if kind == "tau_t":
                    o["t1"] = o["t1"] * 0 + tvec[0]
                    o["t2"] = o["t2"] * 0 + tvec[1]
                else:
                    xu, xv = A[1][1], -A[0][1]
                    yu, yv = -A[1][0], A[0][0]
                    o["t1"] = xu * tvec[0] + xv * tvec[1]
                    o["t2"] = yu * tvec[0] + yv * tvec[1]
                o["k1"] = o["c11"] * o["t1"] + o["c12"] * o["t2"]
                o["k2"] = o["c21"] * o["t1"] + o["c22"] * o["t2"]
            _set(undo, TS, "_stag_map_slant_weights", f)
        elif kind == "qz_off":
            orig = TS.Granet2DTransverseE._assemble_oop_general

            def g(self, *a, **k):
                r = orig(self, *a, **k)
                self._bgen_hermitian = True
                return r
            _set(undo, TS.Granet2DTransverseE, "_assemble_oop_general", g)
        elif kind in ("chi33_unmapped", "chi_unmapped", "oop_over_sg",
                      "oop_adjT"):
            orig = TS._stag_map_eff_tensor

            def h(eps, mu, xu, xv, yu, yv, **kw):
                out = orig(eps, mu, xu, xv, yu, yv, **kw)
                sg = xu * yv - xv * yu
                if kind == "chi33_unmapped":
                    m33 = 1.0 if mu is None else mu[..., 2, 2]
                    out["c33"] = out["c33"] * 0 + 1.0 / m33
                elif kind == "chi_unmapped":
                    if mu is None:
                        one = out["c11"] * 0 + 1.0
                        out["c11"], out["c22"], out["c33"] = one, one, one
                        out["c12"] = out["c21"] = out["c11"] * 0
                    else:
                        m11, m12 = mu[..., 0, 0], mu[..., 0, 1]
                        m21, m22 = mu[..., 1, 0], mu[..., 1, 1]
                        det = m11 * m22 - m12 * m21
                        z = out["c11"] * 0
                        out["c11"] = z + m22 / det
                        out["c12"] = z - m12 / det
                        out["c21"] = z - m21 / det
                        out["c22"] = z + m11 / det
                        out["c33"] = z + 1.0 / mu[..., 2, 2]
                    if "t1" in out:
                        out["k1"] = out["c11"] * out["t1"] + out["c12"] * \
                            out["t2"]
                        out["k2"] = out["c21"] * out["t1"] + out["c22"] * \
                            out["t2"]
                elif kind == "oop_over_sg" and kw.get("oop"):
                    for k in ("e13", "e23", "e31", "e32"):
                        out[k] = out[k] / sg
                elif kind == "oop_adjT" and kw.get("oop"):
                    e = eps
                    A = ((yv, -xv), (-yu, xu))
                    for i, ki in ((0, "1"), (1, "2")):
                        out["e" + ki + "3"] = (A[0][i] * e[..., 0, 2]
                                               + A[1][i] * e[..., 1, 2])
                        out["e3" + ki] = (e[..., 2, 0] * A[0][i]
                                          + e[..., 2, 1] * A[1][i])
                return out
            _set(undo, TS, "_stag_map_eff_tensor", h)
        elif kind == "double_rot":
            orig = TS._stag_scale_weight

            def d(W, s):
                return orig(W, TS._OOP_ROT_SIGN if s == 1.0 else s)
            _set(undo, TS, "_stag_scale_weight", d)
        elif kind == "flux_R":
            fn = _flux_R_fn()
            _set(undo, TS, "_region_modes_oop", fn)
            _set(undo, SP, "_region_modes_oop", fn)
        elif kind == "mu_blocks_off":
            orig = TS.Granet2DTransverseE._assemble_oop_general

            def m(self, *a, **k):
                if self.cmap is None:
                    return orig(self, *a, **k)
                W = self._mapw["c11"]

                def one(v):
                    if isinstance(W, TS._StagNodeWeight):
                        return TS._StagNodeWeight(
                            W.t * 0 + v, {k_: x * 0 + v
                                          for k_, x in W.p.items()})
                    return W * 0 + v
                self._chi_maps = lambda: (one(1.0), one(0.0), one(0.0),
                                          one(1.0), one(1.0))
                try:
                    return orig(self, *a, **k)
                finally:
                    del self._chi_maps
            _set(undo, TS.Granet2DTransverseE, "_assemble_oop_general", m)
        elif kind in ("", "none"):
            pass
        else:
            raise KeyError(kind)
        yield
    finally:
        for obj, nm, val in reversed(undo):
            setattr(obj, nm, val)


def _unused(x):          # keep numpy imported for the exec'd source
    return np.asarray(x)
