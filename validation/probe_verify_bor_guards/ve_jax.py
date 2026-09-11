"""TASK E / task 6 -- JAX vs NumPy parity where a twin exists (Windows only).

The EME JAX twin is ``_jax_modes`` behind ``ref_2d_modes`` /
``ref_2d_modes_vector``; the branch selector is NOT on it (both scalar call
sites pass ``xp=np`` explicitly and ``layer_modes`` refuses a JAX eps).  This
measures (i) the twin's forward parity on both builds, and (ii) whether the
shared selector, which is written against ``array_namespace``, actually WORKS
on a JAX array -- a latent claim of the new module.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); import _vh  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--build", required=True)
ap.add_argument("--tag", required=True)
a = ap.parse_args()
import lumenairy

print("lumenairy.__file__ =", lumenairy.__file__)
_vh.require_tree(a.build)
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from lumenairy.elements.eme import ref_2d_modes, ref_2d_modes_vector

out = dict(task="E", claim="3-jax", build=a.build, tag=a.tag, arm=_vh.arm(),
           jax=jax.__version__)
rows = []
Lx = Ly = 1.0
for Nx, Ny, k0, kx0, ky0 in ((6, 6, 8.0, 0.2, np.pi / 3), (8, 8, 12.0, 0.0, 0.0),
                             (10, 6, 20.0, 0.37, 0.11)):
    e = np.full((Nx, Ny), 2.0); e[:Nx // 2, :Ny // 3] = 6.0
    e[Nx - 1, Ny - 1] = 4.0
    qn = np.asarray(ref_2d_modes(e, Lx, Ly, Nx, Ny, k0, kx0=kx0, ky0=ky0))
    qj = np.asarray(ref_2d_modes(jnp.asarray(e), Lx, Ly, Nx, Ny, k0,
                                 kx0=kx0, ky0=ky0))
    vn = np.asarray(ref_2d_modes_vector(e, Lx, Ly, Nx, Ny, k0, kx0=kx0, ky0=ky0))
    vj = np.asarray(ref_2d_modes_vector(jnp.asarray(e), Lx, Ly, Nx, Ny, k0,
                                        kx0=kx0, ky0=ky0))
    rows.append(dict(Nx=Nx, Ny=Ny, k0=k0,
                     scalar_max_abs=float(np.max(np.abs(qn - qj))),
                     scalar_hash_np=_vh.hash_arrays(qn),
                     scalar_hash_jx=_vh.hash_arrays(qj),
                     vector_max_abs=float(np.max(np.abs(np.sort_complex(vn)
                                                        - np.sort_complex(vj)))),
                     vector_hash_np=_vh.hash_arrays(vn)))
    print("  Nx=%d Ny=%d k0=%g  scalar max|d| = %.3e   vector max|d| = %.3e"
          % (Nx, Ny, k0, rows[-1]["scalar_max_abs"], rows[-1]["vector_max_abs"]))
out["twin"] = rows

# the selector on a JAX array (POST only has the module; PRE has no peer)
try:
    from lumenairy.elements.eme import _branch
    z = jnp.asarray(np.array([100.0 - 1e-13j, 2.0 - 0.5j, 200.0 + 0j]))
    gj = np.asarray(_branch.forward_decaying_root(z))
    gn = np.asarray(_branch.forward_decaying_root(np.asarray(z)))
    out["selector_jax"] = dict(ok=True, max_abs=float(np.max(np.abs(gj - gn))),
                               jax_result=[[float(v.real), float(v.imag)]
                                           for v in gj])
    print("  selector on a JAX array: max|d| vs NumPy = %.3e  -> %s"
          % (out["selector_jax"]["max_abs"], np.round(gj, 12).tolist()))
except Exception as exc:  # noqa: BLE001
    out["selector_jax"] = dict(ok=False, err="%s: %s" % (type(exc).__name__, exc))
    print("  selector on a JAX array: %s" % (out["selector_jax"]["err"],))
_vh.dump(HERE / ("ve_jax_%s.json" % a.tag), out)
