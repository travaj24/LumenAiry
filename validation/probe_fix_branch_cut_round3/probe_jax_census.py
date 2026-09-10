"""ROUND 3 GATE: ``jax.grad`` against central finite differences on the hybrid
2-D PMM twin, over the regimes the branch cut touches, plus JAX/NumPy FORWARD
parity on the same fixtures.

TOLERANCE.  The reference is a CENTRAL difference at ``h = 1e-4`` in the
differentiated variable (or 1e-4 relative for a scaled one).  Central
differencing carries a truncation error ``f'''(x) h^2 / 6``; at h = 1e-4 that
is ~1e-8 relative for a smooth optical response, and the round-off floor is
``eps |f| / h`` ~ 1e-12.  The gate is therefore set at 1e-4 RELATIVE -- four
decades above the FD method error and, on the round-1/2 tree, four to five
decades BELOW the observed defect (the two failing gates read 6.18 and 0.71).
It is the same 1e-4 the two shipped gates already used.

The h-LADDER is reported alongside every row so a reading that is FD-limited
rather than AD-limited is visible rather than hidden by a single h.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import sys
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from lumenairy.elements.pmm import pmm_efficiency_2d
import lumenairy.elements.pmm.twod as _twod

_C = jnp.complex128
_P, _WL, _DEP = 0.6e-6, 0.55e-6, 0.25e-6
_XB = (0.2 * _P, 0.6 * _P)
_GATE = 1e-4


def _mk(theta=0.0, phi=0.0, eps=6.0 + 0j, pol="te", n_sub=1.5, deg=5, no=2,
        yb=None, out="T"):
    """A closure over ``theta`` plus a matching NumPy caller."""
    yb = _XB if yb is None else yb

    def f(th, xp=jnp):
        o, R, T = pmm_efficiency_2d(
            _P, _P, xp.asarray(eps, _C) if xp is jnp else eps, 1.0, _XB, yb,
            n_sub, 1.0, xp.asarray(_DEP) if xp is jnp else _DEP, _WL,
            theta=th, phi=phi, degree=deg, n_orders=no, polarization=pol)
        return (jnp.sum(T) if out == "T" else jnp.sum(R)) if xp is jnp \
            else float(np.sum(T) if out == "T" else np.sum(R))
    return f


def _central(f, x0, h):
    return (float(f(jnp.asarray(x0 + h))) - float(f(jnp.asarray(x0 - h)))) \
        / (2.0 * h)


#: (name, theta at which the gradient is taken, kwargs) -- the regimes the
#: branch cut reaches.  SLANTED is EXCLUDED by construction: the hybrid 2-D PMM
#: has no slant parameter, so there is no such fixture to differentiate.
FIXTURES = (
    ("near-normal   sum(T)", 0.0, dict()),
    ("near-normal   sum(R)", 0.0, dict(out="R")),
    ("exactly 1e-7  sum(T)", 1e-7, dict()),
    ("oblique 0.30  sum(T)", 0.30, dict()),
    ("oblique 0.30  sum(T) TM", 0.30, dict(pol="tm")),
    ("conical 0.40 phi 0.7", 0.40, dict(phi=0.7)),
    # theta EXACTLY 0 with phi != 0 is EXCLUDED: the twin blends the TE/TM
    # basis to the lab axes through a ``where(kt < 1e-12, ...)``, so d/dtheta
    # of the azimuth is taken as 0 at the measure-zero point where kt == 0.
    # That artifact is bit-identical on the PRE-ROUND-1 tree (48c8747, rel
    # 4.569e-01 on both), i.e. it is not a branch-cut defect.  1e-8 rad off
    # normal it is gone (rel 1.2e-07).
    ("conical 1e-8 off normal", 1e-8, dict(phi=0.7)),
    ("LOSSY eps 6+0.4j", 0.0, dict(eps=6.0 + 0.4j)),
    ("LOSSY oblique", 0.35, dict(eps=6.0 + 0.4j)),
    ("high-index sub 2.4", 0.0, dict(n_sub=2.4)),
    ("rectangular pillar", 0.0, dict(yb=(0.2 * _P, 0.594 * _P))),
    ("deeper degree 7", 0.0, dict(deg=7, no=3)),
)


def main():
    print(f"# py{sys.version.split()[0]}  jax {jax.__version__}  "
          f"numpy {np.__version__}  CORETYPE="
          f"{os.environ.get('OPENBLAS_CORETYPE', '-')}")
    print(f"# GATE |AD - FD| / |FD| < {_GATE:.0e}   (FD central, h = 1e-4)")
    worst_grad = 0.0
    worst_fwd = 0.0
    bad = []
    for name, th, kw in FIXTURES:
        f = _mk(**kw)
        ad = float(jax.grad(lambda t: f(t))(jnp.asarray(th)))
        fd = _central(f, th, 1e-4)
        rel = abs(ad - fd) / max(abs(fd), 1e-300)
        ladder = [_central(f, th, h) for h in (3e-4, 1e-4, 3e-5, 1e-5)]
        # forward parity: the jnp twin against the concrete NumPy path
        vj = float(f(jnp.asarray(th)))
        vn = float(_twod.pmm_efficiency_2d(
            _P, _P, kw.get("eps", 6.0 + 0j), 1.0, _XB,
            kw.get("yb", _XB) or _XB, kw.get("n_sub", 1.5), 1.0, _DEP, _WL,
            theta=th, phi=kw.get("phi", 0.0), degree=kw.get("deg", 5),
            n_orders=kw.get("no", 2),
            polarization=kw.get("pol", "te"))[2 if kw.get("out") != "R" else 1]
            .sum())
        fwd = abs(vj - vn) / max(abs(vn), 1e-300)
        worst_grad = max(worst_grad, rel)
        worst_fwd = max(worst_fwd, fwd)
        flag = "" if rel < _GATE else "   <-- OVER GATE"
        if rel >= _GATE:
            bad.append(name)
        print(f"  {name:24s} AD={ad: .6e} FD={fd: .6e} rel={rel:.3e} "
              f"fwd={fwd:.3e}{flag}")
        print(f"      FD h-ladder 3e-4/1e-4/3e-5/1e-5: "
              + "  ".join(f"{v: .6e}" for v in ladder))
    print(f"\nWORST gradient rel {worst_grad:.3e}  (gate {_GATE:.0e})")
    print(f"WORST JAX/NumPy forward rel {worst_fwd:.3e}")
    print("VERDICT:", "PASS" if not bad else f"FAIL {bad}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
