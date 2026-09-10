"""ROUND 3 probe: WHY the hybrid 2-D PMM's angle gradient broke at near-normal
incidence when round 1/2 pinned the on-cut root with ``conj(r)``.

Run:  OMP_NUM_THREADS=1 ... python probe_grad_mechanism.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from lumenairy.elements.rcwa import _core as C
from lumenairy.elements.pmm import pmm_efficiency_2d

_CJ = jnp.complex128
_P, _WL, _DEP = 0.6e-6, 0.55e-6, 0.25e-6
_XB = (0.2 * _P, 0.6 * _P)


# ---------------------------------------------------------------- variants
_ORIG = C._sqrt_decay


def _decay_conj(x, xp=None, band=C._CUT_BAND_REL):
    """round 1/2 as shipped: conj(r) on the cut (NON-holomorphic)."""
    if xp is None:
        xp = C.array_namespace(x)
    x = xp.asarray(x).astype(C._C)
    r = xp.sqrt(x)
    scale = xp.maximum(xp.max(xp.abs(r)), 1.0) if r.size else 1.0
    on_cut = xp.abs(r.real) <= band * scale
    return xp.where(on_cut & (r.imag < 0), xp.conj(r), r)


def _decay_neg(x, xp=None, band=C._CUT_BAND_REL):
    """round 3 candidate: -r on the cut (HOLOMORPHIC: a real scalar factor)."""
    if xp is None:
        xp = C.array_namespace(x)
    x = xp.asarray(x).astype(C._C)
    r = xp.sqrt(x)
    scale = xp.maximum(xp.max(xp.abs(r)), 1.0) if r.size else 1.0
    flip = (xp.abs(r.real) <= band * scale) & (r.imag < 0)
    return r * xp.where(flip, -1.0, 1.0)


def _decay_pre(x, xp=None, band=C._CUT_BAND_REL):
    """pre-round-1: the exact ``r.real == 0`` pin (never fires on eig output)."""
    if xp is None:
        xp = C.array_namespace(x)
    x = xp.asarray(x).astype(C._C)
    r = xp.sqrt(x)
    return xp.where((r.real == 0) & (r.imag < 0), -r, r)


def _decay_imagzero(x, xp=None, band=C._CUT_BAND_REL):
    """round 3 alternative: -r, then ZERO the (noise) real part on the cut."""
    if xp is None:
        xp = C.array_namespace(x)
    x = xp.asarray(x).astype(C._C)
    r = xp.sqrt(x)
    scale = xp.maximum(xp.max(xp.abs(r)), 1.0) if r.size else 1.0
    flip = (xp.abs(r.real) <= band * scale) & (r.imag < 0)
    rf = r * xp.where(flip, -1.0, 1.0)
    return xp.where(flip, 1j * xp.imag(rf), rf)


# ------------------------------------------------------------- instrument
CENSUS = []


def _instrument(fn):
    def wrapped(x, xp=None, band=C._CUT_BAND_REL):
        r_out = fn(x, xp, band)
        try:
            xn = xp if xp is not None else C.array_namespace(x)
            xa = xn.asarray(x).astype(C._C)
            r = xn.sqrt(xa)
            scale = float(np.maximum(np.max(np.abs(np.asarray(r))), 1.0)) \
                if r.size else 1.0
            fl = (np.abs(np.asarray(r).real) <= band * scale) \
                & (np.asarray(r).imag < 0)
            CENSUS.append((int(r.size), int(np.sum(fl)),
                           float(np.max(np.abs(np.asarray(r).real[fl])))
                           if np.any(fl) else 0.0,
                           float(np.max(np.abs(np.asarray(r)[fl])))
                           if np.any(fl) else 0.0))
        except Exception as exc:                # probe only, tracer-safe
            CENSUS.append(("traced", repr(exc)[:40]))
        return r_out
    return wrapped


def _patch(fn):
    C._sqrt_decay = fn
    import lumenairy.elements.pmm.twod as T
    import lumenairy.elements.pmm._jax_twod as J
    import lumenairy.elements.pmm._jax_stack2d as J2
    import lumenairy.elements.pmm._jax_twod_jones as JJ
    import lumenairy.elements.rcwa.oned as O
    import lumenairy.elements.rcwa.stack as S
    import lumenairy.elements.berreman as B
    for m in (T, J, J2, JJ, O, S, B):
        if hasattr(m, "_sqrt_decay"):
            m._sqrt_decay = fn
    # the jax twins import lazily inside the function body -> patch the source
    return fn


# ---------------------------------------------------------------- fixtures
def _f_w9(theta):
    o, R, T = pmm_efficiency_2d(_P, _P, jnp.asarray(6.0 + 0j, _CJ), 1.0,
                                _XB, _XB, 1.5, 1.0, jnp.asarray(_DEP), _WL,
                                theta=theta, degree=5, n_orders=2,
                                polarization="te")
    return jnp.sum(R)


def _f_v514(theta):
    o, R, T = pmm_efficiency_2d(_P, _P, jnp.asarray(6.0 + 0j, _CJ), 1.0,
                                _XB, _XB, 1.5, 1.0, _DEP, _WL, theta=theta,
                                degree=5, n_orders=2, polarization="te")
    return jnp.sum(T)


def _central(f, x0, h):
    return (float(f(jnp.asarray(x0 + h))) - float(f(jnp.asarray(x0 - h)))) \
        / (2 * h)


def report(name, fn):
    _patch(fn)
    out = []
    for label, f, th in (("w9  sum(R) @theta=0", _f_w9, 0.0),
                         ("v514 sum(T) @theta=0", _f_v514, 0.0),
                         ("v514 sum(T) @theta=0.3", _f_v514, 0.3)):
        CENSUS.clear()
        ad = float(jax.grad(f)(jnp.asarray(th)))
        fd = _central(f, th, 1e-6)
        rel = abs(ad - fd) / max(abs(fd), 1e-300)
        out.append((label, ad, fd, rel))
    return out


if __name__ == "__main__":
    for nm, fn in (("pre-round-1 (exact ==0 pin)", _decay_pre),
                   ("round-1/2 SHIPPED conj(r)", _decay_conj),
                   ("round-3 -r (holomorphic)", _decay_neg),
                   ("round-3 -r + imag-only", _decay_imagzero)):
        print(f"\n=== {nm} ===")
        for label, ad, fd, rel in report(nm, fn):
            print(f"  {label:24s} AD={ad: .6e}  FD={fd: .6e}  rel={rel:.3e}")
    C._sqrt_decay = _ORIG


# ---------------------------------------------------------- round-3 FINAL
def _detach(z, xp):
    """``z`` with its derivative cut, portably.  NumPy/CuPy carry no tape, so
    it is the identity there; the JAX twins get ``lax.stop_gradient``."""
    if getattr(xp, "__name__", "") .startswith("jax"):
        import jax as _j
        return _j.lax.stop_gradient(z)
    return z


def _decay_sg(x, xp=None, band=C._CUT_BAND_REL):
    """round 3 FINAL: value = conj(r) (bit-identical to round 2), derivative =
    the holomorphic ``d(-r)``.  ``-r + 2 Re(r)`` IS ``conj(r)`` exactly in
    binary FP (``2a - a == a``); the non-holomorphic ``2 Re(r)`` term is a
    band-level constant, so cutting its derivative is exact, not a hack."""
    if xp is None:
        xp = C.array_namespace(x)
    x = xp.asarray(x).astype(C._C)
    r = xp.sqrt(x)
    scale = xp.maximum(xp.max(xp.abs(r)), 1.0) if r.size else 1.0
    flip = (xp.abs(r.real) <= band * scale) & (r.imag < 0)
    r_flipped = -r + _detach(2.0 * xp.real(r), xp).astype(r.dtype)
    return xp.where(flip, r_flipped, r)
