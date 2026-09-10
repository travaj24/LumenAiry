"""V6b -- are ``_check_stack_slant``'s TWO ADMITTED READINGS actually EXACT?

The refusal admits exactly two accumulated-frame-offset shapes:

  * READING 1, ``Sh_i = 0`` -- every layer above the pattern is VERTICAL;
  * READING 2, ``Sh_i = t_i Z_i`` -- every layer above is at the SAME slant.

Reading 2 is measured by the layer-split identity (one slanted layer of ``d``
== two of ``d/2``), which V6 reads at 1e-15.  Reading 1 has no such identity,
so this probe builds the one comparison that pins BOTH at once: a UNIFORM film
above one slanted pattern, once with the film VERTICAL (reading 1) and once
with the film at the SAME slant (reading 2).

A homogeneous film is translation-invariant, so the two stacks describe the
SAME solid up to a rigid lateral translation of the grating, and a rigid
translation ``delta`` maps the scattering matrix as
``S_mn -> exp(i(alpha_m - alpha_n) . delta) S_mn`` -- which leaves EVERY
efficiency and the ZEROTH-order Jones (``m = n = 0``) unchanged.  So if both
readings are exact, the two stacks must agree, and the disagreement must fall
SPECTRALLY with ``M``; if one of them silently displaces the pattern by a
non-representable offset, the disagreement PLATEAUS.
"""
import warnings

import numpy as np
from _lib import arm, dump, mx  # noqa: I001

from lumenairy.elements.pmm import PMM2DStackPure

warnings.simplefilter("ignore")
WL = 0.68e-6
PX = PY = 1.10e-6
DEP = 0.34e-6
FILM = 0.15e-6
NSUP, NSUB = 1.0, 1.5
T35 = float(np.tan(np.deg2rad(35.0)))
SCA = np.array([[4.0, 1.0], [1.0, 1.0]], dtype=complex)


def stack(film_slant, M, theta, phi):
    st = PMM2DStackPure(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                        n_modes=M, n_orders=3)
    st.add_layer(FILM, eps=2.1 + 0j, slant=film_slant)
    st.add_layer(DEP, eps_cell=SCA, slant=(T35, 0.0))
    st.set_source(WL, theta=theta, phi=phi)
    o, R, T, J = st.solve(jones=True)
    return dict(R=R, T=T, J=J, JT=st.jones_transmission())


def main():
    out = {"ladder": {}}
    for lab, (th, ph) in (("oblique25", (np.deg2rad(25.0), 0.0)),
                          ("conical", (np.deg2rad(25.0), np.deg2rad(40.0)))):
        row = {}
        for M in (4, 5, 6, 7, 8):
            a = stack(None, M, th, ph)
            b = stack((T35, 0.0), M, th, ph)
            row[M] = dict(dR=mx(a["R"], b["R"]), dT=mx(a["T"], b["T"]),
                          dJones=mx(a["J"], b["J"]),
                          dJonesT=mx(a["JT"], b["JT"]))
            print(f"[{lab}] M={M}: dR {row[M]['dR']:.3e} dT {row[M]['dT']:.3e}"
                  f" dJ {row[M]['dJones']:.3e} dJT {row[M]['dJonesT']:.3e}")
        out["ladder"][lab] = row
    dump("v6b_readings", out)
    print("arm", arm())


if __name__ == "__main__":
    main()
