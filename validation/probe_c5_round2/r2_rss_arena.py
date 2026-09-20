"""WP-C5 round 2 -- why an in-process RSS delta reads ZERO in a long session.

`test_verify_c5_three_defaults.py::test_the_flip_bounds_the_resident_set_and_
not_only_tracemalloc` premise-gates on the two instruments agreeing within 2x.
Run alone it passes on both builds; run at the tail of a 34-file WSL session
its `'measured'` arm reads a sampled RSS delta of 0.0 MB against a
`tracemalloc` peak of 197.2 MB, and the premise gate refuses to conclude.

This probe isolates the mechanism, which is the one the id's own docstring
names: a transient that follows a LARGER one is served out of pages the
allocator already holds, so the process's resident set never grows and an
in-process high-water mark reads nothing.  It has nothing to do with the
dense loop, the accounting flip or this branch.

Two arms in ONE process, identical work:

  * COLD -- the transient runs first, in a process that has not held anything
    that size;
  * WARM -- the SAME transient runs again after a deliberately larger block
    has been allocated and freed.

Usage (BLAS pinned on the COMMAND LINE)::

    PYTHONPATH=<tree> python validation/probe_c5_round2/r2_rss_arena.py OUT.json
"""
import gc
import os
import sys
import threading

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
        __file__))), 'probe_verify_c5'))

import numpy as np                                            # noqa: E402
import psutil                                                 # noqa: E402

from _vd import write                                         # noqa: E402

_PROC = psutil.Process()


class RssPeak:
    def __init__(self, interval=0.002):
        self.interval, self.peak, self.base = float(interval), 0, 0
        self._stop, self._th = threading.Event(), None

    def __enter__(self):
        gc.collect()
        self.base = self.peak = _PROC.memory_info().rss
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        return self

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.peak = max(self.peak, _PROC.memory_info().rss)
            except Exception:
                return
            self._stop.wait(self.interval)

    def __exit__(self, *a):
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=2.0)
        return False

    @property
    def delta(self):
        return int(self.peak - self.base)


def transient(n_mb=200, block_mb=200.0):
    """Touch ``n_mb`` megabytes of fresh float64 in ``block_mb`` blocks, then
    drop them.  The BLOCK size is the variable that matters: glibc serves an
    allocation above ``M_MMAP_THRESHOLD`` with ``mmap`` (returned to the OS on
    free, so the next one maps fresh pages and RSS grows) and anything below
    it out of the retained heap (so the next one grows nothing).  That
    threshold is DYNAMIC -- freeing an mmap'd block raises it, up to 32 MB --
    which is why a long-lived process stops showing RSS growth for exactly the
    allocations a fresh one does show."""
    held, made = [], 0.0
    while made < n_mb:
        a = np.empty(int(min(block_mb, n_mb - made) * 1e6 // 8),
                     dtype=np.float64)
        a[:] = 1.0
        held.append(float(a.sum()))
        made += block_mb
        del a
    return held


def measure(n_mb, block_mb):
    with RssPeak() as rw:
        transient(n_mb, block_mb)
    return dict(transient_mb=n_mb, block_mb=block_mb,
                rss_delta_mb=rw.delta / 1e6,
                rss_peak_mb=rw.peak / 1e6, rss_base_mb=rw.base / 1e6)


def main(out_path):
    res = {}
    # (a) ONE block far above any mmap threshold: mmap either way, so the
    #     reading does not move.  This is the control.
    res['one_big_block/cold'] = measure(200, 200.0)
    big = np.empty(int(800e6 // 8), dtype=np.float64)
    big[:] = 1.0
    del big
    gc.collect()
    res['one_big_block/warm'] = measure(200, 200.0)
    # (b) the same 200 MB in 2 MB blocks -- the shape the dense loop's chunk
    #     arrays have.  Cold, glibc mmaps them; once the threshold has
    #     ratcheted up they come out of the retained heap and RSS stops
    #     growing.
    res['many_small_blocks/cold'] = measure(200, 2.0)
    res['many_small_blocks/warm'] = measure(200, 2.0)
    for tag in ('one_big_block', 'many_small_blocks'):
        res[tag + '/cold_over_warm'] = (
            res[tag + '/cold']['rss_delta_mb']
            / max(res[tag + '/warm']['rss_delta_mb'], 1e-9))
    write(out_path, res)
    for tag in ('one_big_block', 'many_small_blocks'):
        print('%-18s cold %8.1f MB   warm %8.4f MB   ratio %.1f'
              % (tag, res[tag + '/cold']['rss_delta_mb'],
                 res[tag + '/warm']['rss_delta_mb'],
                 res[tag + '/cold_over_warm']))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1
         else os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'r2_rss_arena_win.json'))
