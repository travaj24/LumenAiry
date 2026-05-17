"""Pinning tests for the v4.13.1 audit fix P1-F:
``RandomState.choice`` on JAX backend silently ignored
``replace=False`` (and ``_is_jax_prng_key`` missed opaque PRNG keys).

Audit reference
---------------

``AUDIT_V4_13_0_2026_05_17.md`` P1-F found two issues in
``lumenairy/backend/random.py``:

1. ``RandomState.choice(arr, size=k, replace=False)`` on the JAX
   backend with ``p=None`` dispatched to ``jax.random.randint``,
   which is always with-replacement.  The ``replace=False`` flag
   was silently ignored, returning duplicates the caller expected
   to be unique.

2. ``_is_jax_prng_key`` only recognised legacy ``uint32`` /
   shape-trailing-``(..., 2)`` typed keys.  JAX 0.4.20+ ships
   opaque PRNG keys via ``jax.random.key()`` whose dtype has a
   custom name like ``'key<fry>'``; the old detector rejected
   them, causing ``RandomState(jax.random.key(0))`` to fall
   through to the unrecognized-type ``TypeError``.

v4.13.1 fixes:

* Dispatch ``replace=False`` to ``jax.random.choice(..., replace=False)``
  for both ``p=None`` and ``p!=None`` branches (JAX 0.4.x supports
  unweighted ``replace=False``).
* Update ``_is_jax_prng_key`` to recognise opaque keys via
  ``jax.dtypes.issubdtype(d, jax.dtypes.prng_key)`` (canonical) with a
  ``dtype.name.startswith('key<')`` fallback.

Author: Andrew Traverso -- v4.13.1
"""
from __future__ import annotations

import numpy as np
import pytest


jax = pytest.importorskip('jax')


# ============================================================================
# replace=False is honoured on JAX
# ============================================================================

class TestJaxChoiceReplaceFalse:
    """``RandomState(jax_key).choice(n, shape, replace=False)``
    returns unique indices.
    """

    def test_unweighted_replace_false_returns_unique_indices(self):
        from lumenairy.backend.random import RandomState

        key = jax.random.PRNGKey(123)
        rs = RandomState(rng=key)
        n = 20
        k = 7
        out = rs.choice(n, (k,), replace=False)
        vals = np.asarray(out).tolist()
        assert len(vals) == k
        assert len(set(vals)) == k, (
            f'Expected {k} unique values but got duplicates: '
            f'{vals} -- replace=False was silently ignored.')
        # All values must be in [0, n).
        assert all(0 <= v < n for v in vals), (
            f'Indices out of range [0, {n}): {vals}')

    def test_unweighted_replace_true_still_works(self):
        """Regression guard: don't break the with-replacement
        path."""
        from lumenairy.backend.random import RandomState

        key = jax.random.PRNGKey(0)
        rs = RandomState(rng=key)
        out = rs.choice(5, (1000,), replace=True)
        vals = np.asarray(out)
        # With 1000 draws from {0..4}, duplicates are guaranteed.
        assert len(np.unique(vals)) <= 5
        # And full coverage of {0..4} is overwhelmingly likely.
        assert len(np.unique(vals)) == 5, (
            f'Expected full coverage of [0, 5) in 1000 draws; got '
            f'{np.unique(vals)}.')

    def test_weighted_replace_false(self):
        """Weighted ``replace=False`` should also produce unique
        indices.  Pre-fix: this branch raised NotImplementedError,
        which was the documented behaviour but still broken; the
        post-fix code dispatches to
        ``jax.random.choice(..., replace=False, p=...)``.
        """
        from lumenairy.backend.random import RandomState

        key = jax.random.PRNGKey(7)
        rs = RandomState(rng=key)
        n = 10
        k = 4
        p = np.array([0.1, 0.1, 0.1, 0.1, 0.1,
                      0.1, 0.1, 0.1, 0.1, 0.1])
        out = rs.choice(n, (k,), p=p, replace=False)
        vals = np.asarray(out).tolist()
        assert len(set(vals)) == k, (
            f'Weighted replace=False produced duplicates: {vals}')


# ============================================================================
# _is_jax_prng_key recognises opaque keys (JAX 0.4.20+)
# ============================================================================

class TestIsJaxPrngKeyOpaque:
    """``_is_jax_prng_key`` returns True for both legacy uint32
    keys and the opaque keys from ``jax.random.key()``.
    """

    def test_legacy_uint32_key_detected(self):
        from lumenairy.backend.random import _is_jax_prng_key

        k = jax.random.PRNGKey(0)
        assert _is_jax_prng_key(k), (
            f'Legacy uint32 PRNGKey rejected: '
            f'dtype={k.dtype}, shape={k.shape}')

    def test_opaque_key_detected(self):
        """``jax.random.key(...)`` returns an opaque-dtype key
        (JAX 0.4.20+).  The detector must recognise it.

        Skip if the installed JAX is too old to have
        ``jax.random.key``.
        """
        if not hasattr(jax.random, 'key'):
            pytest.skip('JAX too old: jax.random.key() not available.')
        from lumenairy.backend.random import _is_jax_prng_key

        k = jax.random.key(0)
        assert _is_jax_prng_key(k), (
            f'Opaque PRNG key rejected (regression of v4.13.0 audit '
            f'P3 #20): dtype={k.dtype}, dtype.name='
            f'{getattr(k.dtype, "name", "?")}, shape={k.shape}')

    def test_non_key_array_rejected(self):
        """Sanity: regular arrays are NOT detected as keys."""
        from lumenairy.backend.random import _is_jax_prng_key

        arr = jax.numpy.arange(10, dtype=jax.numpy.float32)
        assert not _is_jax_prng_key(arr)
        arr_int = jax.numpy.arange(10, dtype=jax.numpy.int32)
        assert not _is_jax_prng_key(arr_int)
        # uint32 of wrong shape (not trailing 2) should also be
        # rejected by the legacy branch.
        arr_u32 = jax.numpy.arange(10, dtype=jax.numpy.uint32)
        assert not _is_jax_prng_key(arr_u32)

    def test_randomstate_accepts_opaque_key(self):
        """End-to-end: ``RandomState(jax.random.key(seed))``
        constructs and dispatches to the JAX backend."""
        if not hasattr(jax.random, 'key'):
            pytest.skip('JAX too old: jax.random.key() not available.')
        from lumenairy.backend.random import RandomState

        k = jax.random.key(0)
        rs = RandomState(rng=k)
        assert rs.backend == 'jax', (
            f'RandomState(opaque_key) failed to dispatch to the JAX '
            f'backend; got backend={rs.backend!r}.')
        # And a basic draw should work.
        out = rs.uniform((4,))
        assert out.shape == (4,)
