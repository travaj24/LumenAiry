"""Q1b -- the public surfaces the 200-hash table does not reach.

``internal_field`` / ``layer_absorption`` / ``cascade_stats`` on a VERTICAL
stack (must be bit-identical across the two arms -- they are the surfaces most
likely to be disturbed by a change inside ``solve``), the
``jones_field_from_orders`` bridge on a slanted and a vertical stack (it
consumes the per-order dict, so it must move exactly where the dict moves), and
the magnetic-layer question (the hybrid has no ``mu`` surface at all -- checked
rather than assumed).
"""
from __future__ import annotations

import inspect
import warnings

import _lib as L
import numpy as np
from q1_fixtures import CENTRO


def main():
    from lumenairy.elements.pmm.stack2d import PMM2DStackHybrid
    from lumenairy.elements.polarization import jones_field_from_orders
    th, ph = L.MOUNTS["oblique25"]
    out = {}

    # --- internal_field / layer_absorption / cascade_stats, VERTICAL --------
    v = PMM2DStackHybrid(L.PX, L.PY, n_superstrate=L.NSUP,
                         n_substrate=L.NSUB, n_orders=5)
    v.add_layer(0.20e-6, eps_cell=L.BASE)
    v.add_layer(0.13e-6, eps=2.25)
    v.add_layer(0.17e-6, eps_cell=CENTRO)
    v.set_source(L.WL, theta=th, phi=ph)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v.solve(retain_internal=True)
    F = v.internal_field(0.25e-6, nx=16, ny=16)
    if isinstance(F, dict):
        out["internal_field"] = {k: L.sha(np.asarray(a))
                                 for k, a in sorted(F.items())
                                 if isinstance(a, np.ndarray)}
    else:
        out["internal_field"] = dict(field=L.sha(np.asarray(F)))
    out["layer_absorption"] = L.sha(np.asarray(v.layer_absorption()))
    try:
        out["cascade_stats"] = str(sorted(v.cascade_stats().items()))
    except Exception as e:                              # noqa: BLE001
        out["cascade_stats"] = "RAISE " + type(e).__name__

    # --- the jones_field_from_orders bridge -------------------------------
    for tag, slant in (("vertical", None), ("slanted", (0.5, 0.0))):
        st = PMM2DStackHybrid(L.PX, L.PY, n_superstrate=L.NSUP,
                              n_substrate=L.NSUB, n_orders=5)
        st.add_layer(L.DTHICK, eps_cell=L.BASE, slant=slant)
        st.set_source(L.WL, theta=th, phi=ph)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            st.solve()
        for port in ("transmission", "reflection"):
            a = st.per_order_amplitudes(port)
            jf = jones_field_from_orders(a, 24, 24, L.PX / 24)
            arr = getattr(jf, "Ex", None)
            if arr is None:
                arr = np.asarray(jf)
            out["bridge_%s_%s_Ex" % (tag, port)] = L.sha(np.asarray(arr))
            ay = getattr(jf, "Ey", None)
            if ay is not None:
                out["bridge_%s_%s_Ey" % (tag, port)] = L.sha(np.asarray(ay))

    # --- the magnetic question --------------------------------------------
    sig = inspect.signature(PMM2DStackHybrid.add_layer)
    out["add_layer_params"] = list(sig.parameters)
    out["hybrid_has_mu_surface"] = any(
        p.startswith("mu") for p in sig.parameters)
    from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure
    out["pure_add_layer_params"] = list(
        inspect.signature(PMM2DStackPure.add_layer).parameters)

    for k, val in out.items():
        print("%-34s %s" % (k, val))
    L.dump("q1b_extra", out)


if __name__ == "__main__":
    main()
