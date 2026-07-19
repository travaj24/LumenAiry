"""C1 / S4-19 (AUDIT_V5_24_2 deferred roadmap) -- faithful nested
metadata round-trip through ONE canonical, type-tagged serialization
contract honored identically by the HDF5 and Zarr backends.

Pre-fix (audit S4-19, ``storage.py`` sim-metadata path):

  * **list -> ndarray coercion** -- a native attribute cannot tell a
    Python ``list`` from a NumPy ``ndarray``; both round-tripped to the
    same type, so the distinction was lost on the h5 backend.
  * **un-reversed dict flattening** -- nested dicts were flattened to
    ``"parent.child"`` keys on write but never re-nested on read, so a
    round-trip returned a *flat* dict, not the caller's structure.

The independent oracle is a hand-constructed expected Python structure
(exact types + values), NOT the code's own output.  Back-compat: a file
written by the old (blob-less) scheme must still LOAD via the decoder's
fallback.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from lumenairy.io.storage import (
    _META_BLOB_KEY,
    _META_TAG,
    _meta_dumps,
    _meta_loads,
    read_sim_metadata,
    write_sim_metadata,
)


# ------------------------------------------------------------------ #
# A representative nested metadata mapping exercising every branch of #
# the contract.                                                       #
# ------------------------------------------------------------------ #
def _sample_metadata():
    return {
        'run': {'seed': 42, 'method': 'asm', 'nested': {'depth': 3}},
        'py_list': [1, 2, 3],
        'nd_int': np.array([1, 2, 3], dtype=np.int64),
        'nd_float': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        'nd_complex': np.array([1 + 2j, 3 - 4j], dtype=np.complex128),
        'a_complex': 1.5 + 2.5j,
        'a_tuple': (1, 'two', 3.0),
        'a_bytes': b'\x00\x01\xfe\xff',      # embedded NULL: hostile to VLEN
        'np_scalar': np.float64(3.25),
        'a_none': None,
        'a_bool': True,
        'a_str': 'hello',
    }


def _assert_faithful(out):
    """Independent hand-oracle: exact types + values after a round-trip."""
    # Nested dict structure preserved -- NOT flattened to 'run.seed'.
    assert 'run.seed' not in out
    assert isinstance(out['run'], dict)
    assert isinstance(out['run']['nested'], dict)
    assert out['run'] == {'seed': 42, 'method': 'asm',
                          'nested': {'depth': 3}}

    # The headline distinction: list stays list, ndarray stays ndarray.
    assert type(out['py_list']) is list
    assert out['py_list'] == [1, 2, 3]

    assert isinstance(out['nd_int'], np.ndarray)
    assert out['nd_int'].dtype == np.int64
    np.testing.assert_array_equal(out['nd_int'], np.array([1, 2, 3]))

    assert isinstance(out['nd_float'], np.ndarray)
    assert out['nd_float'].shape == (2, 2)
    assert out['nd_float'].dtype == np.float64
    np.testing.assert_array_equal(
        out['nd_float'], np.array([[1.0, 2.0], [3.0, 4.0]]))

    assert isinstance(out['nd_complex'], np.ndarray)
    assert out['nd_complex'].dtype == np.complex128
    np.testing.assert_array_equal(
        out['nd_complex'], np.array([1 + 2j, 3 - 4j]))

    assert isinstance(out['a_complex'], complex)
    assert out['a_complex'] == 1.5 + 2.5j

    assert isinstance(out['a_tuple'], tuple)
    assert out['a_tuple'] == (1, 'two', 3.0)

    assert isinstance(out['a_bytes'], bytes)
    assert out['a_bytes'] == b'\x00\x01\xfe\xff'

    assert out['np_scalar'] == 3.25
    assert out['a_none'] is None
    assert out['a_bool'] is True
    assert out['a_str'] == 'hello'

    # The reserved blob key is never leaked into the returned mapping.
    assert _META_BLOB_KEY not in out


# ------------------------------------------------------------------ #
# Backend-independent codec tests (hand oracle on the JSON itself).  #
# ------------------------------------------------------------------ #
def test_codec_roundtrip_direct():
    _assert_faithful(_meta_loads(_meta_dumps(_sample_metadata())))


def test_codec_json_structure_hand_oracle():
    """The on-disk JSON encodes a list as a plain array and an ndarray
    as a tagged object -- the explicit distinction that fixes the
    list->ndarray coercion."""
    md = {'lst': [1, 2], 'arr': np.array([1, 2], dtype=np.int64)}
    parsed = json.loads(_meta_dumps(md))     # raw JSON, no decode step
    assert parsed['lst'] == [1, 2]           # untagged -> decodes to list
    assert parsed['arr'][_META_TAG] == 'ndarray'
    assert parsed['arr']['dtype'] == 'int64'
    assert parsed['arr']['data'] == [1, 2]
    assert parsed['arr']['shape'] == [2]


def test_codec_preserves_dict_ordering():
    md = {'z': 1, 'a': 2, 'm': 3}
    out = _meta_loads(_meta_dumps(md))
    assert list(out.keys()) == ['z', 'a', 'm']


def test_codec_dict_with_reserved_tag_key_is_escaped():
    """A user dict that literally contains the reserved tag key must not
    be mis-read as a tagged wrapper."""
    md = {'weird': {_META_TAG: 'not_a_tag', 'x': 1}}
    out = _meta_loads(_meta_dumps(md))
    assert out['weird'] == {_META_TAG: 'not_a_tag', 'x': 1}


# ------------------------------------------------------------------ #
# HDF5 backend round-trip.                                           #
# ------------------------------------------------------------------ #
def test_h5_roundtrip_faithful(tmp_path):
    pytest.importorskip('h5py')
    path = str(tmp_path / 'meta.h5')
    write_sim_metadata(path, _sample_metadata())
    _assert_faithful(read_sim_metadata(path))


def test_h5_backcompat_no_blob_falls_back(tmp_path):
    """A pre-contract file (flat native attrs, NO blob) still loads --
    the decoder falls back and reproduces the historical flat mapping."""
    h5py = pytest.importorskip('h5py')
    path = str(tmp_path / 'legacy.h5')
    with h5py.File(path, 'w') as f:
        f.attrs['run.seed'] = 42          # old flattened nested key
        f.attrs['method'] = 'asm'
        f.attrs['count'] = 7
    out = read_sim_metadata(path)
    assert int(out['run.seed']) == 42     # flat key survives verbatim
    assert out['method'] == 'asm'
    assert int(out['count']) == 7
    assert _META_BLOB_KEY not in out


# ------------------------------------------------------------------ #
# Zarr backend round-trip (zarr>=3 requires Python>=3.11).           #
# ------------------------------------------------------------------ #
def test_zarr_roundtrip_faithful(tmp_path):
    pytest.importorskip('zarr')
    path = str(tmp_path / 'meta.zarr')
    write_sim_metadata(path, _sample_metadata())
    _assert_faithful(read_sim_metadata(path))


def test_zarr_backcompat_no_blob_falls_back(tmp_path):
    zarr = pytest.importorskip('zarr')
    path = str(tmp_path / 'legacy.zarr')
    store = zarr.open_group(path, mode='w')
    store.attrs['run.seed'] = 42
    store.attrs['method'] = 'asm'
    out = read_sim_metadata(path)
    assert int(out['run.seed']) == 42
    assert out['method'] == 'asm'
    assert _META_BLOB_KEY not in out


# ------------------------------------------------------------------ #
# The contract is stored IDENTICALLY in both backends.              #
# ------------------------------------------------------------------ #
def test_blob_identical_across_backends(tmp_path):
    h5py = pytest.importorskip('h5py')
    zarr = pytest.importorskip('zarr')
    md = _sample_metadata()

    h5_path = str(tmp_path / 'm.h5')
    zarr_path = str(tmp_path / 'm.zarr')
    write_sim_metadata(h5_path, md)
    write_sim_metadata(zarr_path, md)

    with h5py.File(h5_path, 'r') as f:
        h5_blob = f.attrs[_META_BLOB_KEY]
        if isinstance(h5_blob, bytes):
            h5_blob = h5_blob.decode()
    z_blob = zarr.open_group(zarr_path, mode='r').attrs[_META_BLOB_KEY]
    if isinstance(z_blob, bytes):
        z_blob = z_blob.decode()

    assert h5_blob == z_blob, (
        'S4-19: canonical metadata blob differs between the h5 and zarr '
        'backends -- the contract must be byte-identical across both.')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
