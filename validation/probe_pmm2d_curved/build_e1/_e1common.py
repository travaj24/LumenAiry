"""Phase E1 fixtures on top of ``_common`` (which pins ``lumenairy`` to this
tree): the out-of-plane tensors, the maps, the Berreman slab oracle (R, T and
BOTH Jones matrices), and the ENGINEERED MUTATIONS of the E1-9 matrix (each
patches the real code path and restores it).

Slab fixture = the shipped out-of-plane suite's uniform-slab fixture
(``tests/unit/test_pmm2d_staggered_oop.py``: period 0.9, depth 0.35,
lambda 1, air over n = 1.5), lengths in units of the wavelength.
"""
from contextlib import contextmanager

import _common as C
import numpy as np
from _common import CM, SP, TS, PMM2DStackPure  # noqa: F401 -- re-exported

from lumenairy.elements.berreman import berreman_jones_1d
from lumenairy.elements.rcwa._core import uniaxial_tensor

SLAB = dict(P=0.9, WL=1.0, DEP=0.35, NSUB=1.5, NSUP=1.0)
#: tilted-director LC, out of plane (the shipped G3 suite's _OOP): director
#: tilt 35 deg from z, azimuth 25 deg (deliberately != the 40-deg conical
#: incidence azimuth, the S6 degeneracy)
OOP = uniaxial_tensor(1.5, 1.7, np.deg2rad(35.0), phi=np.deg2rad(25.0))
#: NON-RECIPROCAL (Hermitian, lossless, NOT symmetric): the only tensor that
#: can see an e13 <-> e31 swap
NONREC = np.array(OOP, dtype=complex)
NONREC[0, 2] = OOP[0, 2] + 0.22j
NONREC[2, 0] = np.conj(NONREC[0, 2])
#: lossy (absorbing) tilted LC
LOSSY = uniaxial_tensor(1.5 + 0.02j, 1.7 + 0.02j, np.deg2rad(35.0),
                        phi=np.deg2rad(25.0))
#: LOSSY AND NON-RECIPROCAL (neither Hermitian nor symmetric)
LNONREC = NONREC + 0.03j * np.eye(3)
#: the E1-5 pillar's director: tilted 30 deg OUT OF PLANE (60 deg from z),
#: azimuth 30 deg
OOP30 = uniaxial_tensor(1.5, 1.8, np.deg2rad(60.0), phi=np.deg2rad(30.0))
#: IN-PLANE reference (Phase D's LC: director in the plane at 0.55 rad) --
#: the block-form film under the SAME map isolates what the out-of-plane
#: machinery adds from what the map's resolution costs
LC_INPLANE = uniaxial_tensor(1.5, 1.8, np.pi / 2, phi=0.55)
#: the ISOTROPIC eps-2.25 slab: slanted, it becomes an out-of-plane tensor
#: in the sheared frame (eps^13 = -t eps), the shipped slant's null test
EPS_ISO = 2.25 * np.eye(3, dtype=complex)
TENSORS = {"oop": OOP, "nonrec": NONREC, "lossy": LOSSY, "lnonrec": LNONREC,
           "lc": LC_INPLANE, "eps_iso": EPS_ISO}
ANG = {"normal": (0.0, 0.0), "oblique": (25.0, 0.0), "conical": (25.0, 40.0)}


def stretch_map(Pp, a, n=2):
    """Separable sine stretch of BOTH axes (x by a, y by -a/2) on an n x n
    uniform (u, v) grid (Phase D's s05 / s15, here on 2 x 2 by default)."""
    w = np.linspace(0.0, Pp, n + 1)
    return CM.SeparableStretch(w, w, fx=CM.SineStretch(a * Pp),
                               fy=CM.SineStretch(-0.5 * a * Pp))


def shear_map(Pp):
    """Phase D's 3 x 3 SHEARED map: straight edges, the four interior
    vertices moved by 3-6 % of the period -> bilinear cells with g12 != 0
    and a NON-diagonal J (separates J^-1 X from X J^-1 and J^-1 t from t)."""
    w = np.linspace(0.0, Pp, 4)
    V = np.stack(np.meshgrid(w, w, indexing="ij"), axis=-1)
    for (i, j), (dx, dy) in {(1, 1): (0.06, 0.04), (2, 1): (-0.05, 0.03),
                             (1, 2): (0.03, -0.05),
                             (2, 2): (-0.04, -0.06)}.items():
        V[i, j] += (dx * Pp, dy * Pp)
    return CM.TransfiniteMap(w, w, V, None)


def shear2_map(Pp):
    """A 2 x 2 sheared map (the centre vertex moved by (6, 4) % of the
    period): non-diagonal J at a quarter of the 3 x 3 cost."""
    w = np.linspace(0.0, Pp, 3)
    V = np.stack(np.meshgrid(w, w, indexing="ij"), axis=-1)
    V[1, 1] += (0.06 * Pp, 0.04 * Pp)
    return CM.TransfiniteMap(w, w, V, None)


def make_map(name, Pp):
    if name == "none":
        return None
    if name == "s05":
        return stretch_map(Pp, 0.05)
    if name == "s15":
        return stretch_map(Pp, 0.15)
    if name == "s05g3":
        return stretch_map(Pp, 0.05, n=3)
    if name == "s15g3":
        return stretch_map(Pp, 0.15, n=3)
    if name == "shear":
        return shear_map(Pp)
    if name == "shear2":
        return shear2_map(Pp)
    if name == "id3":
        # the IDENTITY map through the mapped path (3 x 3 transfinite, no
        # curve, no moved vertex): the E1-2 / E1-9 identity arm
        return CM.TransfiniteMap(np.linspace(0.0, Pp, 4),
                                 np.linspace(0.0, Pp, 4))
    if name == "c3":
        return CM._circle_map_3x3(Pp, 0.3 * Pp)[0]
    if name == "c5":
        return CM._circle_map_5x5(Pp, 0.3 * Pp)[0]
    raise ValueError(name)


def jones_t(st):
    """The order-0 TRANSMISSION Jones of the last solve (rows = transmitted
    lab [Ex; Ey], columns = incident [Ex; Ey]) from the retained per-order
    amplitudes (frame-anchor phase included)."""
    md = st._modal
    p0 = md["p0"]
    return np.array([[md["tx"][0][p0], md["tx"][1][p0]],
                     [md["ty"][0][p0], md["ty"][1][p0]]])


def slab_vs_berreman(t33, cmap, M, theta=0.0, phi=0.0, f=None, slant=None,
                     mu=None):
    """A UNIFORM tensor slab (a uniform layer of the -- possibly mapped --
    stack, optionally slanted) against ``berreman_jones_1d``.  Returns a dict:
    dRT (largest |R, T| difference, orders summed per input), dJr / dJt (the
    largest complex reflection / transmission Jones differences), closure,
    leak (largest efficiency in a non-zero order)."""
    f = SLAB if f is None else f
    st = PMM2DStackPure(f["P"], f["P"], n_superstrate=f["NSUP"],
                        n_substrate=f["NSUB"], n_modes=M, n_orders=2,
                        cmap=cmap)
    st.add_layer(f["DEP"], eps=t33, slant=slant, mu=mu)
    st.set_source(f["WL"], theta=theta, phi=phi)
    o, R, T, J = st.solve(jones=True)
    Jt = jones_t(st)
    Rb, Tb, jr, jt = berreman_jones_1d([(t33, f["DEP"])], f["NSUB"],
                                       f["NSUP"], f["WL"], angle=theta,
                                       phi=phi)
    R, T = np.asarray(R), np.asarray(T)
    p0 = st._modal["p0"]
    leak = float(max(np.max(np.abs(np.delete(R, p0, axis=1))),
                     np.max(np.abs(np.delete(T, p0, axis=1)))))
    return {"dRT": float(max(np.max(np.abs(R.sum(1) - Rb)),
                             np.max(np.abs(T.sum(1) - Tb)))),
            "dJr": float(np.max(np.abs(np.asarray(J) - jr))),
            "dJt": float(np.max(np.abs(Jt - jt))),
            "closure": float(np.max(np.abs(R.sum(1) + T.sum(1) - 1.0))),
            "berreman_closure": float(np.max(np.abs(Rb + Tb - 1.0))),
            "leak": leak}


# ---- engineered mutations (the E1-9 matrix), through the REAL code path ----
MUTATIONS = ("no_mu_blocks", "chi33_no_sg", "g3_strong", "tau_unmapped",
             "no_slant_blocks", "kappa_no_chi", "rot_flip", "hgauge_plus_i",
             "hgauge_one")


def _const_like(W, val):
    from lumenairy.elements.pmm.twod_staggered import _StagNodeWeight
    if isinstance(W, _StagNodeWeight):
        return _StagNodeWeight(W.t * 0 + val,
                               {k: v * 0 + val for k, v in W.p.items()})
    return np.asarray(W) * 0 + val


@contextmanager
def mutate(kind):
    """Patch the library with ONE deliberate defect of the Phase E1 build:

    * no_mu_blocks    -- the generator's permeability blocks dropped under a
      map (chi_t = I, chi33 = 1 in _assemble_oop_general; the eps' weights
      and the slant weights untouched) -- must be caught by the stretch /
      shear gates and NOT by the identity gate;
    * chi33_no_sg     -- chi33 = 1 / mu33 instead of 1 / (sqrt(g) mu33) in
      the out-of-plane congruence;
    * g3_strong       -- the G rows' G3 taken as the STRONG curl c3 (the
      shipped mu = 1 elimination) instead of the chi33-weighted projection;
    * tau_unmapped    -- the slant weights of the composite map replaced by
      the shipped constants (tau = t, kappa = t: the shear NOT composed with
      the map);
    * no_slant_blocks -- tau = kappa = 0 (the shear only in the congruence);
    * kappa_no_chi    -- kappa = tau (chi_t dropped from the E rows' shear
      term);
    * rot_flip        -- _OOP_ROT_SIGN = +1;
    * hgauge_plus_i / hgauge_one -- _OOP_H_GAUGE = +1j / +1."""
    G = TS.Granet2DTransverseE
    patched = []

    def setp(obj, name, val):
        patched.append((obj, name, getattr(obj, name)))
        setattr(obj, name, val)

    if kind == "no_mu_blocks":
        orig = G._assemble_oop_general

        def f(self, *a, **k):
            if self.cmap is None:
                return orig(self, *a, **k)
            w = self._mapw
            one = _const_like(w["c11"], 1.0)
            zero = _const_like(w["c11"], 0.0)
            saved = self._chi_maps
            self._chi_maps = lambda: (one, zero, zero, one, one)
            try:
                return orig(self, *a, **k)
            finally:
                self._chi_maps = saved
        setp(G, "_assemble_oop_general", f)
    elif kind == "chi33_no_sg":
        orig_t = TS._stag_map_eff_tensor

        def f(eps, mu, xu, xv, yu, yv, **kw):
            out = orig_t(eps, mu, xu, xv, yu, yv, **kw)
            if kw.get("oop"):
                out["c33"] = out["c33"] * (xu * yv - xv * yu)
            return out
        setp(TS, "_stag_map_eff_tensor", f)
    elif kind == "g3_strong":
        orig = G._chi_Gw

        def f(self, chi33):
            if self.offplane:
                return orig(self, _const_like(chi33, 1.0))
            return orig(self, chi33)
        setp(G, "_chi_Gw", f)
    elif kind in ("tau_unmapped", "no_slant_blocks", "kappa_no_chi"):
        orig_s = TS._stag_map_slant_weights

        def f(out, A, sg, tvec):
            orig_s(out, A, sg, tvec)
            if kind == "tau_unmapped":
                out["t1"] = out["t1"] * 0 + float(tvec[0])
                out["t2"] = out["t2"] * 0 + float(tvec[1])
                out["k1"], out["k2"] = out["t1"], out["t2"]
            elif kind == "no_slant_blocks":
                for k in ("t1", "t2", "k1", "k2"):
                    out[k] = out[k] * 0
            else:
                out["k1"], out["k2"] = out["t1"], out["t2"]
        setp(TS, "_stag_map_slant_weights", f)
    elif kind == "rot_flip":
        setp(TS, "_OOP_ROT_SIGN", 1.0)
    elif kind == "hgauge_plus_i":
        setp(TS, "_OOP_H_GAUGE", 1j)
    elif kind == "hgauge_one":
        setp(TS, "_OOP_H_GAUGE", 1.0)
    else:
        raise ValueError(kind)
    try:
        yield
    finally:
        for obj, name, old in reversed(patched):
            setattr(obj, name, old)


dump = C.dump
