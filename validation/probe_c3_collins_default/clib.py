"""WP-C3 probe harness -- the pieces shared by every probe in this directory.

Two things live here and nothing else: the ANCHOR (refuse to measure a tree
other than the one named on the command line, and print what bound) and the
ANALYTIC GAUSSIAN ORACLE in this library's own sign convention.

The anchor, the sha256 digest map and the build tag are NOT re-implemented:
they are :mod:`validation.probe_wave5_hyg2.hlib`, imported by path from the
sibling directory, because this campaign's bit-identity claim is the same
shape as hygiene-2's and one harness with two callers is the standing rule
(``feedback_consolidate_numerical_kernels``).  What is new here is the oracle.

THE ORACLE'S CONVENTION, WRITTEN OUT BECAUSE IT IS THE EASY THING TO GET
WRONG.  CONVENTIONS sec. 7 mandates ``exp(-i omega t)`` / ``exp(+i k z)``, so
the complex beam parameter pairs as ``1/q = 1/R + i lambda/(pi w^2)`` -- NOT
Siegman's ``- i lambda/(pi w^2)``, which belongs to ``exp(+i omega t)`` and
conjugates the Gouy phase.  The 2-D amplitude prefactor is ``q/q2`` and is
taken as that RATIO rather than as ``1/sqrt((1 + z/q)^2)``: the square's
principal branch leaves the right half-plane past the waist and the oracle
then picks up exactly ``pi`` on every leg with ``A < 0``.  Both mistakes are
recorded in VERIFY-WP-B4's opening caution; both are avoided by construction
below rather than by comment.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_HYG2 = os.path.join(os.path.dirname(_HERE), 'probe_wave5_hyg2')
if _HYG2 not in sys.path:
    sys.path.insert(0, _HYG2)

import hlib  # noqa: E402  -- the shared anchor / digest / build-tag harness

import numpy as np  # noqa: E402

anchor = hlib.anchor
build_tag = hlib.build_tag
write_json = hlib.write_json
Probe = hlib.Probe
digest = hlib.digest


# ---------------------------------------------------------------------------
# The analytic Gaussian, absolute phase included
# ---------------------------------------------------------------------------
def gaussian_field(x, y, w, R, wavelength):
    """The INPUT field of a Gaussian with ``1/e`` amplitude radius ``w`` and
    wavefront radius ``R`` (``inf`` = collimated), on the grid ``(x, y)``.

    Amplitude ``exp(-r^2/w^2)`` times ``exp(i k r^2 / 2R)``.  The envelope the
    carrier-referenced API takes is this WITHOUT the second factor, referenced
    to ``R``; the two spellings are the same field and the probes use whichever
    the entry point asks for.
    """
    r2 = x[None, :] ** 2 + y[:, None] ** 2
    amp = np.exp(-r2 / (w * w))
    if np.isfinite(R):
        k = 2.0 * np.pi / wavelength
        return (amp * np.exp(1j * k * r2 / (2.0 * R))).astype(np.complex128)
    return amp.astype(np.complex128)


def gaussian_propagated(x_out, y_out, w, R, wavelength, z):
    """The same Gaussian after a distance ``z`` of free space -- the FIELD,
    absolute phase included (piston ``exp(i k z)`` and the Gouy term), on the
    output grid ``(x_out, y_out)``.

    ``1/q = 1/R + i lambda/(pi w^2)`` at the input plane, ``q2 = q + z``, and
    the field is ``exp(i k z) * (q/q2) * exp(i k r^2 / (2 q2))`` in 2-D.  The
    prefactor is the ratio, taken once, so it is continuous through the waist.
    """
    k = 2.0 * np.pi / wavelength
    inv_q = (0.0 if not np.isfinite(R) else 1.0 / R) \
        + 1j * wavelength / (np.pi * w * w)
    q = 1.0 / inv_q
    q2 = q + z
    r2 = x_out[None, :] ** 2 + y_out[:, None] ** 2
    return (np.exp(1j * k * z) * (q / q2)
            * np.exp(1j * k * r2 / (2.0 * q2))).astype(np.complex128)


def axis(n, d):
    """The package's centred coordinate axis: ``(i - n/2) * d``."""
    return (np.arange(int(n), dtype=np.float64) - int(n) / 2.0) * float(d)


def rel_l2(a, b):
    """Relative L2 of ``a`` against ``b``, both host arrays."""
    a = np.asarray(a)
    b = np.asarray(b)
    den = float(np.linalg.norm(b))
    if den == 0.0:
        return float('inf')
    return float(np.linalg.norm(a - b) / den)


def rel_l2_piston_free(a, b):
    """``rel_l2`` after removing the single best global phase AND scale --
    i.e. the part of the disagreement that is not a piston.

    Returned beside the absolute reading everywhere, never instead of it: a
    transport that lost the Gouy phase would read perfectly here and badly
    there, which is the whole reason both are printed.
    """
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    den = float(np.vdot(a, a).real)
    if den == 0.0:
        return float('inf')
    c = np.vdot(a, b) / den
    return rel_l2(c * a, b)


def best_global_phase(a, b):
    """``arg`` of the best global phase between ``a`` and ``b``, radians."""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    return float(np.angle(np.vdot(a, b)))


def reconstruct_field(env, R, dx_out, dy_out=None, wavelength=None):
    """The FIELD from a ``CarrierReferencedField`` triple: the envelope times
    its own carrier ``exp(i k r^2 / 2R)`` on the returned lattice.

    Written here rather than taken from the library so the oracle comparison
    shares no code with the thing it measures.
    """
    env = np.asarray(env)
    ny, nx = env.shape[-2], env.shape[-1]
    dy_out = dx_out if dy_out is None else dy_out
    x = axis(nx, dx_out)
    y = axis(ny, dy_out)
    if R is None:
        return env
    Rx, Ry = (R if isinstance(R, tuple) else (R, R))
    k = 2.0 * np.pi / wavelength
    ph = np.zeros((ny, nx), dtype=np.float64)
    if np.isfinite(Rx):
        ph = ph + k * (x * x)[None, :] / (2.0 * Rx)
    if np.isfinite(Ry):
        ph = ph + k * (y * y)[:, None] / (2.0 * Ry)
    return env * np.exp(1j * ph)


def unpack_pitch(dx_out):
    """A returned pitch is a scalar or an ``(dx, dy)`` pair."""
    if isinstance(dx_out, tuple):
        return float(dx_out[0]), float(dx_out[1])
    return float(dx_out), float(dx_out)
