"""E-JAX -- does the Berreman JAX twin take the SHARED ``_sqrt_decay``, and does
it agree with the NumPy path?

``lumenairy/elements/_berreman_jax.py:396`` calls the shared
``rcwa/_core._sqrt_decay`` WITHOUT the explicit ``xp=jnp`` that the function's
own docstring says "the JAX twins pass ... explicitly", relying instead on
``array_namespace`` detection.  Under ``jit`` the operand is a Tracer, so this
probe checks that the eager AND traced paths both run and both reproduce the
NumPy answer.

Usage: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       PYTHONPATH=. python validation/probe_scope_bor_guards/e_jax.py out.json
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e_lib as E  # noqa: E402

WL = 0.6e-6
NS = 1.5
HOST = 2.25
STRONG = 6.0


def main():
    E.pin_tree()
    out = sys.argv[1] if len(sys.argv) > 1 else "e_jax.json"
    res = {}
    try:
        import jax
        import jax.numpy as jnp
        jax.config.update("jax_enable_x64", True)
        res["jax_version"] = jax.__version__
    except Exception as exc:
        E.dump(out, {"jax": "MISSING", "error": repr(exc)[:200]})
        return

    from lumenairy.elements.berreman import berreman_jones_1d

    def tensor():
        t = np.diag([STRONG, STRONG, STRONG]).astype(complex)
        t[0, 2] = t[2, 0] = 0.6                # exz -> off-plane
        return t

    def np_arm(theta):
        lay = [(HOST * np.eye(3, dtype=complex), 0.12e-6),
               (tensor(), 0.20e-6),
               (HOST * np.eye(3, dtype=complex), 0.12e-6)]
        R, T, Jr, _Jt = berreman_jones_1d(lay, NS, NS, WL, theta=theta, phi=0.3)
        return np.asarray(R), np.asarray(T), np.asarray(Jr)

    def jx(theta):
        lay = [(jnp.asarray(HOST * np.eye(3), dtype=jnp.complex128), 0.12e-6),
               (jnp.asarray(tensor(), dtype=jnp.complex128), 0.20e-6),
               (jnp.asarray(HOST * np.eye(3), dtype=jnp.complex128), 0.12e-6)]
        return berreman_jones_1d(lay, NS, NS, WL, theta=theta, phi=0.3)

    theta = 0.35
    Rn, Tn, Jrn = np_arm(theta)
    res["numpy"] = dict(R=Rn.tolist(), T=Tn.tolist(),
                        closure=float(np.max(np.abs(Rn + Tn - 1.0))))

    for tag, fn in (("eager", jx), ("jit", jax.jit(jx))):
        try:
            Rj, Tj, Jrj, _ = fn(theta)
            Rj, Tj = np.asarray(Rj), np.asarray(Tj)
            res[tag] = dict(
                ok=True, R=Rj.tolist(), T=Tj.tolist(),
                closure=float(np.max(np.abs(Rj + Tj - 1.0))),
                max_gap_vs_numpy=float(max(np.max(np.abs(Rj - Rn)),
                                           np.max(np.abs(Tj - Tn)))),
                jones_gap=float(np.max(np.abs(np.asarray(Jrj) - Jrn))))
        except Exception as exc:
            res[tag] = dict(ok=False,
                            raised="%s: %s" % (type(exc).__name__,
                                               str(exc).replace("\n", " ")[:260]))

    # gradient through the traced path (the reason the twin exists)
    try:
        def loss(th):
            lay = [(jnp.asarray(HOST * np.eye(3), dtype=jnp.complex128), 0.12e-6),
                   (jnp.asarray(tensor(), dtype=jnp.complex128), 0.20e-6),
                   (jnp.asarray(HOST * np.eye(3), dtype=jnp.complex128), 0.12e-6)]
            R, _T, _Jr, _Jt = berreman_jones_1d(lay, NS, NS, WL, theta=th,
                                                phi=0.3)
            return jnp.real(R[0])
        g = float(jax.grad(loss)(jnp.asarray(theta)))
        h = 1e-6
        fd = (float(np_arm(theta + h)[0][0]) - float(np_arm(theta - h)[0][0])) / (2 * h)
        res["grad"] = dict(ok=True, jax_grad=g, central_difference=fd,
                           rel_gap=float(abs(g - fd) / max(abs(fd), 1e-30)))
    except Exception as exc:
        res["grad"] = dict(ok=False,
                           raised="%s: %s" % (type(exc).__name__,
                                              str(exc).replace("\n", " ")[:260]))
    E.dump(out, res)


if __name__ == "__main__":
    main()
