"""G08 / S3-16 (AUDIT_V5_24_2) -- sources cleanup.

Concrete safe fix: the vestigial ``N`` keyword on the private
``_schell_phase_realizations`` generator is removed.  It duplicated
``Nx`` (callers passed ``N == Ny == Nx``) and was never read, so it only
invited an inconsistent ``N != Nx`` call.

(The broader S3-16 items -- ``seed`` -> ``rng`` kwarg migration, the
peak/power/raw normalization drift, and the ``sigma`` vs ``w0`` naming
divergence -- are coordinated public-API deprecations across the whole
source factory zoo and are tracked separately; see the run notes.)
"""
from __future__ import annotations

import inspect

import numpy as np

from lumenairy.sources.core import _schell_phase_realizations


def test_dead_N_param_removed_from_signature():
    params = inspect.signature(_schell_phase_realizations).parameters
    assert 'N' not in params, (
        'S3-16: the dead ``N`` parameter is still on '
        '_schell_phase_realizations.')
    # The grid is still fully specified by Ny / Nx.
    assert 'Ny' in params and 'Nx' in params


def test_generator_still_functions_without_N():
    """The generator must produce the documented unit-mean-intensity
    ensemble when called by grid dims alone."""
    phi = _schell_phase_realizations(
        Ny=48, Nx=48, dx=2e-6, dy=2e-6,
        coherence_length=12e-6, n_realizations=64,
        rng=np.random.default_rng(1))
    assert phi.shape == (64, 48, 48)
    ens_mean = float(np.mean(np.abs(phi) ** 2))
    assert abs(ens_mean - 1.0) < 0.06


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
