"""The grid-sized process-global caches scale their default byte caps with RAM.

WHAT HAPPENED.  The ASM transfer-function cache (``fft_infra._H_CACHE``), the
pyFFTW plan workspaces, the Bluestein chirp-kernel cache, the centred-grid
cache and the Zernike basis cache each carried a FIXED default cap sized on a
large workstation: 8 GiB per cache, 2 GB per plan workspace.  On a 16 GB
machine -- the hosted CI runner class -- one cache may then hold half of RAM
for the rest of the process.

MEASURED (Linux, Python 3.12, psutil's view of RAM set to 15.6 GiB, N = 8192
ASM runs from the slow lane): after two runs the H cache held 4 entries
(4 GiB) and the plan cache two double-buffered keys (4 GiB); the second run
peaked at 15.4 GiB of RSS with those 8 GiB resident, and
``clear_asm_caches()`` + ``gc`` brought the process back to 78 MiB.  The slow
shard that runs one of those tests reached a 15.0 GiB high-water mark, i.e.
the RAM of a 16 GB runner.

WHAT CHANGED.  ``memory._ram_scaled_cap(nominal, fraction)`` is
``min(nominal, fraction * total RAM)`` and every one of those defaults goes
through it: 1/8 of RAM for the 8 GiB totals, 1/16 for the 2 GB plan
workspace.  On a machine with 64 GB of RAM or more nothing changes (the
second test below pins that).  These are caches: no value depends on the cap
(the plan-buffer cap's single-workspace path is the documented byte-identical
``set_fft_double_buffer(False)`` trade).

The caps are computed at IMPORT, so the machine-size arms run in a child
interpreter whose ``psutil.virtual_memory()`` reports the size under test.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from lumenairy import memory as M

_GIB = 1024 ** 3

_CHILD = textwrap.dedent('''
    import json, sys
    sys.modules['jax'] = None            # not needed; keeps the child light
    import psutil
    _real = psutil.virtual_memory
    _TOTAL = int(sys.argv[1])

    def _vm():
        r = _real()
        return r._replace(total=_TOTAL, available=min(r.available, _TOTAL))
    psutil.virtual_memory = _vm

    import numpy as np
    from lumenairy.propagators import fft_infra as fi
    from lumenairy.propagators import _bluestein as bl
    from lumenairy.analysis import beam_stats as bs
    from lumenairy.analysis import zernike as z

    # Fill the H cache with 1 GiB entries (an N = 8192 complex128 H).  The
    # arrays are np.empty, never touched, so they cost address space only.
    fi.clear_asm_caches()
    for k in range(4):
        fi._h_cache_store(('probe', k),
                          np.empty((8192, 8192), dtype=np.complex128))
    held = sum(fi._entry_bytes(v) for v in fi._H_CACHE.values())
    n_held = len(fi._H_CACHE)
    fi.clear_asm_caches()
    print(json.dumps(dict(
        h_total=fi._H_CACHE_MAX_TOTAL_BYTES,
        plan_buf=fi._PYFFTW_PLAN_MAX_BYTES_PER_BUFFER,
        bluestein=bl._H_FFT_CACHE_MAX_TOTAL_BYTES,
        meshgrid=bs._MESHGRID_CACHE_MAX_TOTAL_BYTES,
        zernike=z._ZERNIKE_BASIS_CACHE_MAX_TOTAL_BYTES,
        n_bufs_8192=fi._plan_entry_n_bufs((8192, 8192),
                                          np.dtype('complex128')),
        n_bufs_4096=fi._plan_entry_n_bufs((4096, 4096),
                                          np.dtype('complex128')),
        h_held=held, h_n_held=n_held)))
''')


def _caps_on_a_machine_with(total_bytes):
    proc = subprocess.run(
        [sys.executable, '-c', _CHILD, str(int(total_bytes))],
        stdin=subprocess.DEVNULL, capture_output=True, text=True,
        timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_the_helper_is_the_smaller_of_the_nominal_cap_and_the_ram_share(
        monkeypatch):
    monkeypatch.setattr(M, 'total_memory_bytes', lambda: 16 * 10 ** 9)
    assert M._ram_scaled_cap(8 * _GIB, 1.0 / 8) == 2 * 10 ** 9
    assert M._ram_scaled_cap(2_000_000_000, 1.0 / 16) == 10 ** 9
    monkeypatch.setattr(M, 'total_memory_bytes', lambda: 256 * 10 ** 9)
    assert M._ram_scaled_cap(8 * _GIB, 1.0 / 8) == 8 * _GIB
    assert M._ram_scaled_cap(2_000_000_000, 1.0 / 16) == 2_000_000_000


def test_a_large_machine_keeps_every_shipped_cap():
    """64 GB of RAM and up: the five caps are the nominal values exactly, and
    an N = 8192 plan keeps its two-workspace ping-pong."""
    c = _caps_on_a_machine_with(256 * 10 ** 9)
    for k in ('h_total', 'bluestein', 'meshgrid', 'zernike'):
        assert c[k] == 8 * _GIB, (k, c[k])
    assert c['plan_buf'] == 2_000_000_000
    assert c['n_bufs_8192'] == 2 and c['n_bufs_4096'] == 2
    # four 1 GiB H entries fit an 8 GiB budget
    assert c['h_n_held'] == 4, c


def test_a_16_gb_machine_caps_each_cache_at_an_eighth_of_ram():
    """THE FIX.  On a 16 GB machine no cache may default to more than 1/8 of
    RAM, an N = 8192 plan holds ONE workspace per key, and four N = 8192 H
    arrays leave at most 2 GB in the cache instead of 4 GiB."""
    total = 16 * 10 ** 9
    c = _caps_on_a_machine_with(total)
    for k in ('h_total', 'bluestein', 'meshgrid', 'zernike'):
        assert c[k] == total // 8, (k, c[k])
    assert c['plan_buf'] == total // 16
    assert c['n_bufs_8192'] == 1, 'an 8192^2 plan still ping-pongs 2 GiB'
    assert c['n_bufs_4096'] == 2, 'a 4096^2 plan lost its ping-pong'
    assert c['h_held'] <= total // 8, c
    assert c['h_n_held'] == 1, c


@pytest.mark.parametrize('frac', [1.0 / 8, 1.0 / 16])
def test_the_cap_never_exceeds_the_nominal_value_on_this_machine(frac):
    """Whatever this box is, the helper only ever LOWERS a cap."""
    nominal = 8 * _GIB
    got = M._ram_scaled_cap(nominal, frac)
    assert 0 <= got <= nominal
    assert got == min(nominal, int(frac * M.total_memory_bytes()))
    assert isinstance(got, int) and not isinstance(got, np.integer)
