"""Correctness pin for the v4.13.0 BSDFModel.total_integrated_scatter
vectorisation (Tier-2 perf, audit group alpha task 3).

The pre-v4.13.0 default ``total_integrated_scatter`` integrand was
evaluated inside a nested Python loop over (theta, phi); v4.13.0
builds a 2-D meshgrid of scattered directions and evaluates the BSDF
in one broadcasted call.  This test pins the numerical result against
both a known closed-form answer and against an inline reference
scalar-loop implementation.
"""
from __future__ import annotations

import numpy as np

from lumenairy.elements.bsdf import (
    BSDFModel,
    LambertianBSDF,
    HarveyShackBSDF,
)


def _reference_scalar_tis(bsdf: BSDFModel,
                          n_theta: int = 256, n_phi: int = 128) -> float:
    """Pre-v4.13.0 nested-loop TIS, inlined here as the pin reference."""
    theta = np.linspace(1e-6, np.pi / 2, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    inc = np.array([0.0, 0.0, -1.0])
    tot = 0.0
    dth = theta[1] - theta[0]
    dph = phi[1] - phi[0]
    for i in range(n_theta):
        for j in range(n_phi):
            s = np.array([
                np.sin(theta[i]) * np.cos(phi[j]),
                np.sin(theta[i]) * np.sin(phi[j]),
                np.cos(theta[i]),
            ])
            b = float(bsdf.evaluate(inc, s))
            tot += b * np.cos(theta[i]) * np.sin(theta[i]) * dth * dph
    return tot


class _DefaultLambertian(BSDFModel):
    """Subclass that does NOT override total_integrated_scatter --
    forces use of the BSDFModel default integration path so the
    vectorised quadrature is what we actually pin."""
    kind = 'default_lambertian'

    def __init__(self, rho: float = 1.0):
        self.rho = float(rho)

    def evaluate(self, incident_dir, scattered_dir):
        sd = np.asarray(scattered_dir)
        if sd.ndim == 1:
            in_hemi = sd[2] > 0
        else:
            in_hemi = sd[..., 2] > 0
        return self.rho / np.pi * in_hemi

    def sample(self, incident_dir, n_samples, rng=None):
        raise NotImplementedError


def test_default_tis_against_lambertian_closed_form():
    """For a Lambertian-pi BSDF the analytical TIS equals rho exactly.
    Pin the vectorised default integration against rho within 1e-3
    relative."""
    rho = 0.7
    bsdf = _DefaultLambertian(rho)
    tis = bsdf.total_integrated_scatter()
    # Numerical-quadrature error from the default 256x128 grid is the
    # endpoint-rule cosine bias at theta -> pi/2; ~1e-3 relative is the
    # achievable bound without going to a Gauss-Legendre rule.
    assert abs(tis - rho) <= 1e-3 * rho, (
        f'TIS = {tis:.6f}, expected ~{rho:.6f}')


def test_vectorised_matches_scalar_loop_harvey_shack():
    """A non-degenerate Harvey-Shack BSDF: vectorised TIS must match
    the inline scalar-loop reference to within 1e-10."""
    hs = HarveyShackBSDF(b0=0.05, l=0.02, s=2.0)
    tis_vec = hs.total_integrated_scatter()  # subclass default OR override
    # Force the BSDFModel-default code path by calling it explicitly,
    # since HarveyShackBSDF doesn't override TIS.
    tis_ref = _reference_scalar_tis(hs)
    assert abs(tis_vec - tis_ref) <= 1e-10, (
        f'vectorised TIS = {tis_vec:.10e}, scalar = {tis_ref:.10e}')


def test_vectorised_matches_scalar_loop_default_lambertian():
    """Same pin for the bare-default Lambertian-pi subclass that
    forces use of BSDFModel.total_integrated_scatter."""
    b = _DefaultLambertian(rho=0.3)
    tis_vec = b.total_integrated_scatter()
    tis_ref = _reference_scalar_tis(b)
    assert abs(tis_vec - tis_ref) <= 1e-10, (
        f'vectorised TIS = {tis_vec:.10e}, scalar = {tis_ref:.10e}')
