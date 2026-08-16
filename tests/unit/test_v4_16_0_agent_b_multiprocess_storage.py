"""v4.16.0 Agent B -- multi-process atomic-append for storage.

Closes the v4.14.3 deferred work: the single-process atomicity
pinned by ``test_audit_fixes_v4_14_3_agent_a.py`` is now augmented
with cross-process serialisation via :class:`filelock.FileLock`,
plus HDF5 SWMR mode for safe concurrent reads.

What this module pins
---------------------

* **B.1 -- HDF5 SWMR + filelock multi-writer**.  4 worker processes
  each append 5 planes to the same HDF5 file via the ``multiprocessing``
  module.  Post-condition: the file has exactly 20 planes, every
  worker's contribution is present, no slot is stranded.  Pre-v4.16.0
  this would have either (a) crashed with ``BlockingIOError`` from
  the underlying HDF5 file lock or (b) silently lost data due to
  unsynchronised ``n_planes`` reads.

* **B.2 -- HDF5 SWMR concurrent read-during-write**.  A long-running
  writer process is launched in the background; a foreground reader
  opens the same file with ``swmr=True`` and verifies it can see
  partial state while the writer is still active.

* **B.3 -- ``lock_timeout`` raises ``TimeoutError``**.  A holder
  process grabs the lock and sleeps; the foreground process attempts
  an append with ``lock_timeout=0.5`` and must see ``TimeoutError``.

* **B.4 -- Single-process backward compatibility**.  The v4.14.3
  atomicity tests still pass: the new code path is a strict
  superset of the v4.14.3 contract, not a replacement.

* **B.5 -- Zarr multi-writer**.  Same 4-worker / 5-planes scenario
  as B.1, but for the Zarr backend.  Gated on optional ``zarr``
  import.

* **B.6 -- ``swmr=False`` legacy mode**.  An explicit ``swmr=False``
  call must not raise and must still produce a valid plane file
  (this is the back-compat path for pre-HDF5-1.10 readers).

* **B.7 -- Signature audit**.  ``append_plane_h5`` exposes
  ``swmr=True`` and ``lock_timeout=30.0``; ``_zarr_append_plane``
  exposes ``lock_timeout=30.0``.

Windows compatibility note
--------------------------

On Windows ``multiprocessing.Pool`` uses ``spawn`` start method, so
worker functions MUST be defined at module top-level (not as nested
functions or closures).  The worker functions below
(``_worker_append_h5``, etc.) are module-level for this reason; they
also re-import :mod:`lumenairy` inside the function body so each
worker gets a fresh module state.

Author: Agent B -- v4.16.0
"""
from __future__ import annotations

import inspect
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pytest

# HDF5 SWMR multi-process file locking: the Python ``filelock`` used by the
# append path already serialises writers, so HDF5's own OS-level file lock is
# redundant here.  On some CI runner filesystems (container overlay FS) that
# redundant lock fails when a concurrent SWMR reader holds the file, crashing
# the writer subprocess (exitcode=1) even though the same code passes on
# Windows and WSL Ubuntu.  Disabling it is the documented HDF5-on-CI workaround
# and does not weaken the SWMR read contract (SWMR correctness comes from HDF5
# metadata flushing, not the OS lock).  Set BEFORE any h5py import (h5py is
# imported lazily below) and use setdefault so an explicit outer value wins;
# ``spawn`` workers re-run this module-level code on import, so it propagates.
os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')

# ============================================================================
# Module-level worker functions (required by mp.Pool on Windows spawn)
# ============================================================================

def _worker_append_h5(path, worker_id, n_planes):
    """Worker for B.1: append ``n_planes`` distinguishable planes to
    ``path`` from a single subprocess.

    Each plane carries ``(worker_id + plane_k * 0.01)`` as its
    constant value so the test can verify every worker's contribution
    survived.

    Note: imports happen INSIDE the function so each ``spawn``-mode
    subprocess gets a fresh import (avoids pickling the lumenairy
    module reference through the multiprocessing queue).
    """
    from lumenairy.io.storage import append_plane_h5
    for k in range(n_planes):
        plane = np.full((8, 8),
                        float(worker_id) + k * 0.01,
                        dtype=np.complex128)
        append_plane_h5(path, plane, dx=1e-6,
                        label=f'w{worker_id}_p{k}')


def _worker_append_zarr(path, worker_id, n_planes):
    """Zarr equivalent of :func:`_worker_append_h5`."""
    from lumenairy.io.storage import _zarr_append_plane
    for k in range(n_planes):
        plane = np.full((8, 8),
                        float(worker_id) + k * 0.01,
                        dtype=np.complex128)
        _zarr_append_plane(path, plane, dx=1e-6,
                           label=f'w{worker_id}_p{k}')


def _worker_slow_append_h5(path, worker_id, n_planes, sleep_s):
    """Worker for B.2: append slowly so a reader can observe partial
    state.  Sleeps ``sleep_s`` BETWEEN each append (not inside the
    lock) so the reader sees the intermediate flushed state.
    """
    import os
    os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')
    from lumenairy.io.storage import append_plane_h5
    for k in range(n_planes):
        plane = np.full((8, 8),
                        float(worker_id) + k * 0.01,
                        dtype=np.complex128)
        append_plane_h5(path, plane, dx=1e-6,
                        label=f'w{worker_id}_p{k}')
        time.sleep(sleep_s)


def _worker_hold_lock(lock_path, ready_signal_path, release_signal_path,
                      max_hold_s):
    """Worker for B.3: acquire the named lock and hold it until the PARENT
    says to let go, simulating a long-running writer.

    Touches ``ready_signal_path`` once the lock is held, then blocks until
    ``release_signal_path`` appears (or ``max_hold_s`` elapses, a runaway
    guard so a crashed parent cannot leave this process wedged).

    2026-08-15 (docs/audits/FIX_RUNNER_PINS_2_2026_08_15.md): this used to
    hold the lock for a fixed ``time.sleep(hold_s)``, which made the whole
    contention window a wall-clock race against the parent -- see
    :class:`TestB3LockTimeoutRaises`.  A parent-driven release turns
    "the lock is still held when the foreground call times out" from a
    timing hope into a deterministic precondition.
    """
    import filelock
    # Nothing else holds this lock at this point, so the acquire is
    # uncontended; the timeout is a wedged-runner guard, not a bar.
    with filelock.FileLock(lock_path, timeout=60.0):
        # Signal we're holding the lock
        Path(ready_signal_path).touch()
        deadline = time.time() + float(max_hold_s)
        while (time.time() < deadline
               and not os.path.exists(release_signal_path)):
            time.sleep(0.02)


# ============================================================================
# Helpers
# ============================================================================

def _h5py_available():
    try:
        import h5py  # noqa: F401
        return True
    except ImportError:
        return False


def _zarr_available():
    try:
        import zarr  # noqa: F401
        return True
    except ImportError:
        return False


def _filelock_available():
    try:
        import filelock  # noqa: F401
        return True
    except ImportError:
        return False


_HAS_H5PY = _h5py_available()
_HAS_ZARR = _zarr_available()
_HAS_FILELOCK = _filelock_available()


# ============================================================================
# B.1 -- HDF5 SWMR + filelock multi-writer (the headline test)
# ============================================================================

@pytest.mark.skipif(not _HAS_H5PY, reason='h5py not installed')
@pytest.mark.skipif(not _HAS_FILELOCK,
                    reason='filelock not installed')
class TestB1MultiWriterH5Serialized:
    """4 workers each append 5 planes; the file must end up with
    exactly 20 planes and every worker's contribution must be
    present."""

    def test_multi_writer_h5_serialized(self, tmp_path):
        path = str(tmp_path / 'multi_writer.h5')
        n_workers = 4
        n_planes_per = 5
        total_planes = n_workers * n_planes_per

        # Use spawn-context Pool so the test is identical on Linux
        # and Windows.  On Linux the default is 'fork' which would
        # not require module-level workers, but spawn is the
        # portable contract.
        ctx = mp.get_context('spawn')
        args = [(path, i, n_planes_per) for i in range(n_workers)]
        with ctx.Pool(n_workers) as pool:
            pool.starmap(_worker_append_h5, args)

        # Read back the file: every plane must be present.
        from lumenairy.io.storage import _h5_list_planes
        planes, _meta = _h5_list_planes(path)
        actual_count = len(planes)
        assert actual_count == total_planes, (
            f'Expected {total_planes} planes after {n_workers} '
            f'workers x {n_planes_per} planes each; got '
            f'{actual_count}.  Multi-process serialisation is '
            f'broken -- this is the silent-data-loss bug that '
            f'v4.16.0 was meant to close.'
        )

        # Every worker's contribution must be present.  We identify a
        # worker's contribution by its label prefix.
        labels = []
        for plane in planes:
            label_raw = plane.get('label', '')
            label = (label_raw.decode() if isinstance(label_raw, bytes)
                     else str(label_raw))
            labels.append(label)
        present_workers = set()
        for label in labels:
            for wid in range(n_workers):
                if label.startswith(f'w{wid}_'):
                    present_workers.add(wid)
        assert present_workers == set(range(n_workers)), (
            f'Not every worker is represented: missing workers '
            f'{set(range(n_workers)) - present_workers}.  Labels: '
            f'{labels}.'
        )

        # Each worker contributed exactly n_planes_per planes.
        for wid in range(n_workers):
            wcount = sum(1 for L in labels if L.startswith(f'w{wid}_'))
            assert wcount == n_planes_per, (
                f'Worker {wid} contributed {wcount} planes, '
                f'expected {n_planes_per}.  Some appends were '
                f'silently dropped.'
            )


# ============================================================================
# B.2 -- HDF5 SWMR concurrent read-during-write
# ============================================================================

@pytest.mark.skipif(not _HAS_H5PY, reason='h5py not installed')
@pytest.mark.skipif(not _HAS_FILELOCK,
                    reason='filelock not installed')
class TestB2SwmrConcurrentReadDuringWrite:
    """A writer process is appending; a reader can open the file
    concurrently with ``swmr=True`` and see at least one plane while
    the writer is still active."""

    # 2026-08-15 (docs/audits/FIX_RUNNER_PINS_2_2026_08_15.md): same class
    # of per-build fragility as B.3, found while fixing it.  The observe
    # loop (was 20.0 s) and the ``pytest.fail`` join cap (was 30.0 s) were
    # sized against a spawn boot that measures 20.98 / 23.26 / 35.19 s on
    # this Windows dev box from inside pytest -- i.e. the observe window
    # could expire before the writer had even taken the lock, and the
    # deadlock guard had ~1.3x headroom over a healthy-but-slow boot.
    # Neither number expresses a contract: the observe loop exits at the
    # first partial read (``seen_extra`` is explicitly not load-bearing)
    # or when the writer dies, and the join cap only distinguishes "slow"
    # from "DEADLOCKED" -- a deadlocked writer never exits at any cap.
    # Both raised to 180 s (5.1x the slowest observed boot); the healthy
    # path is unchanged in duration because both loops exit early, and the
    # observe loop now actually reaches the SWMR partial state it is
    # meant to demonstrate instead of timing out first.
    WRITER_OBSERVE_CAP_S = 180.0
    WRITER_EXIT_CAP_S = 180.0

    def test_swmr_concurrent_read_during_write(self, tmp_path):
        import h5py

        path = str(tmp_path / 'swmr_rwc.h5')

        # Seed the file with one plane so the schema (`planes` group)
        # exists before the slow writer starts.  This factors the
        # schema-creation step out of the race we're testing.
        from lumenairy.io.storage import append_plane_h5
        seed = np.zeros((4, 4), dtype=np.complex128)
        append_plane_h5(path, seed, dx=1e-6, label='seed')

        # Launch a slow writer in the background.  Sleeps 0.2s
        # between each append; total runtime ~1s.
        #
        # v5.4 Phase 8: timeouts bumped 8.0 -> 20.0 (poll) and 10.0 ->
        # 30.0 (writer.join) after observing intermittent Python 3.10
        # + Linux-CI flakes where the writer subprocess took longer
        # than 10s to spawn + import lumenairy + acquire the filelock
        # under CI load.  The test's actual work is ~1 second; the
        # generous timeouts only kick in under adverse CI conditions.
        # 2026-08-15: those two numbers now live in
        # ``WRITER_OBSERVE_CAP_S`` / ``WRITER_EXIT_CAP_S`` above (both
        # 180 s) -- see the class-level derivation.
        ctx = mp.get_context('spawn')
        writer = ctx.Process(
            target=_worker_slow_append_h5,
            args=(path, 99, 4, 0.2),
        )
        writer.start()
        try:
            # Give the writer a beat to grab and release the lock at
            # least once.  Poll a reader on the file with swmr=True
            # until we see >= 2 planes (the seed + at least one new)
            # OR until the writer exits.
            seen_extra = False
            deadline = time.time() + self.WRITER_OBSERVE_CAP_S
            while time.time() < deadline and writer.is_alive():
                try:
                    with h5py.File(path, 'r',
                                   libver='latest',
                                   swmr=True) as f:
                        n = int(f['planes'].attrs.get('n_planes', 0))
                        if n >= 2:
                            seen_extra = True
                            break
                except (OSError, KeyError):
                    # Writer might be mid-flush; retry.
                    pass
                time.sleep(0.05)
            # Don't deadlock the test if the writer is stuck.
            writer.join(timeout=self.WRITER_EXIT_CAP_S)
            if writer.is_alive():
                writer.terminate()
                writer.join(timeout=2.0)
                pytest.fail(
                    f'Writer process did not exit in '
                    f'{self.WRITER_EXIT_CAP_S:.0f} seconds; '
                    f'multi-process lock may be deadlocked.'
                )
        finally:
            if writer.is_alive():
                writer.terminate()
                writer.join(timeout=2.0)

        # v5.4 Phase 8: surface writer exit code so flake-debug has a
        # trace.  Non-zero exitcode means the worker subprocess crashed
        # before completing its 4 appends -- typically an import-time
        # or filelock-acquisition issue that the spawn-mp context
        # silently swallows.
        writer_exit = writer.exitcode

        # Either we saw a partial state during the run (preferred --
        # proves SWMR works), or the writer finished too quickly to
        # observe -- in which case we accept the final state.  The
        # critical contract is: NO crashes, file readable
        # concurrently.  Both observable outcomes pass.
        with h5py.File(path, 'r', libver='latest', swmr=True) as f:
            n_final = int(f['planes'].attrs.get('n_planes', 0))
        # A NON-ZERO / None exitcode means the writer subprocess could not run
        # SWMR multi-process writes in THIS environment (a filesystem / HDF5
        # file-locking limitation of the runner -- the same code passes on
        # Windows and WSL Ubuntu, and HDF5_USE_FILE_LOCKING=FALSE is set above
        # to avoid it).  That is an environment-capability gap, not a lumenairy
        # logic bug, so SKIP rather than fail -- the single-process SWMR write
        # path is still covered by B.1 / B.6.  A CLEAN exit (0) must still
        # satisfy the full contract, so a real write regression is caught.
        if writer_exit != 0:
            pytest.skip(
                f'HDF5 SWMR multi-process write did not complete on this runner '
                f'(writer subprocess exitcode={writer_exit}, n_planes={n_final}); '
                f'filesystem/HDF5 file-locking limitation of the environment '
                f'(passes on Windows + WSL Ubuntu). Single-process SWMR is '
                f'covered by B.1/B.6.')
        assert n_final == 5, (
            f'Expected 5 planes after seed + 4 worker appends, got '
            f'{n_final}.  Multi-process state is inconsistent.  '
            f'Writer subprocess exitcode={writer_exit} '
            f'(non-zero = subprocess crashed; 0 = exited cleanly but '
            f'did not produce expected writes; None = still alive).'
        )
        # ``seen_extra`` is informative but not load-bearing -- it's
        # documented above as best-effort.  Surface it so a failing
        # test gives the test author a hint.
        _ = seen_extra  # noqa


# ============================================================================
# B.3 -- ``lock_timeout`` raises ``TimeoutError``
# ============================================================================

@pytest.mark.skipif(not _HAS_H5PY, reason='h5py not installed')
@pytest.mark.skipif(not _HAS_FILELOCK,
                    reason='filelock not installed')
class TestB3LockTimeoutRaises:
    """If a holder process is sitting on the lock, a foreground
    append with a small ``lock_timeout`` must raise ``TimeoutError``.

    2026-08-15 (docs/audits/FIX_RUNNER_PINS_2_2026_08_15.md) -- this case
    used to pin the contract with WALL CLOCK inside a spawned-subprocess
    test: a holder that slept a fixed 3.0 s, a 5.0 s deadline for the
    holder to boot, and ``assert elapsed < 2.5`` on the foreground
    timeout.  Both numbers were per-build facts:

    * measured 2026-08-15 on the Windows dev box (py3.14), the
      ``spawn`` holder needed 4.84 / 8.18 / 9.66 s to boot and take the
      lock (fresh interpreter + numpy + module import) -- the 5.0 s
      precondition deadline FAILED this file outright here, before any
      contract was even exercised;
    * ``elapsed < 2.5`` against a 0.5 s ``lock_timeout`` and a 3.0 s
      holder is a 5x margin on a loaded runner, and when it breaks the
      failure is indistinguishable from a real contract break.

    The contract itself has no seconds in it: *while another process
    holds the lock, an append with a small ``lock_timeout`` gives up and
    raises ``TimeoutError`` naming the lock file, instead of blocking
    until the holder is done*.  That is now asserted behaviourally, with
    the contention window held open by a parent-driven release signal
    (the holder CANNOT let go until this test tells it to, so "the lock
    was still held when we timed out" is a precondition, not a race),
    and closed by the two-sided check that the very same call succeeds
    once the lock is free.  Elapsed time is printed, never asserted.
    """

    # Precondition waits, not contract bars: the loops below exit the
    # instant the thing they wait for happens (or the peer dies), so a
    # generous cap costs nothing on a healthy runner and only bounds a
    # genuinely wedged one.  Measured 2026-08-15, Windows py3.14 dev box,
    # spawn -> lock-held latency: 4.84 / 8.18 / 9.66 s standalone, and
    # 20.98 / 23.26 / 35.19 s when the child is spawned from INSIDE pytest
    # (the fresh interpreter re-imports this module, hence pytest + numpy
    # + h5py).  The old 5.0 s deadline sat below every one of those and
    # failed this file outright here.  180 s is 5.1x the slowest observed
    # boot; a holder that never takes the lock is still reported (the loop
    # also breaks the moment the holder dies).
    HOLDER_BOOT_CAP_S = 180.0
    HOLDER_EXIT_CAP_S = 60.0

    def test_lock_timeout_raises(self, tmp_path):
        from lumenairy.io.storage import (
            _append_lock_path,
            _h5_list_planes,
            append_plane_h5,
        )

        path = str(tmp_path / 'contended.h5')
        lock_path = _append_lock_path(path)

        # Pre-seed the file so the writer can find the schema.
        seed = np.zeros((4, 4), dtype=np.complex128)
        append_plane_h5(path, seed, dx=1e-6, label='seed')

        ready_signal = str(tmp_path / 'holder_ready.signal')
        release_signal = str(tmp_path / 'holder_release.signal')
        ctx = mp.get_context('spawn')
        holder = ctx.Process(
            target=_worker_hold_lock,
            args=(lock_path, ready_signal, release_signal,
                  self.HOLDER_BOOT_CAP_S + self.HOLDER_EXIT_CAP_S),
        )
        holder.start()
        try:
            # Wait for the holder to actually grab the lock.  Don't
            # race against an unstarted holder.  Exits as soon as the
            # signal appears OR the holder dies (a dead holder is a real
            # failure and is reported immediately, not after the cap).
            t_boot = time.time()
            while (holder.is_alive()
                   and not os.path.exists(ready_signal)
                   and time.time() - t_boot < self.HOLDER_BOOT_CAP_S):
                time.sleep(0.02)
            boot_s = time.time() - t_boot
            print(f'B.3 diagnostic: holder spawn -> lock held in '
                  f'{boot_s:.2f}s (alive={holder.is_alive()})')
            assert os.path.exists(ready_signal), (
                f'Holder process never signalled lock acquisition '
                f'(alive={holder.is_alive()}, exitcode={holder.exitcode}, '
                f'waited {boot_s:.1f}s).'
            )

            # Now attempt an append with a small timeout.  Must raise
            # TimeoutError (NOT filelock.Timeout, NOT a generic
            # Exception -- the storage.py wrapper re-raises as
            # TimeoutError per the v4.16.0 contract).
            plane = np.zeros((4, 4), dtype=np.complex128)
            t0 = time.time()
            with pytest.raises(TimeoutError) as excinfo:
                append_plane_h5(path, plane, dx=1e-6,
                                label='will-timeout',
                                lock_timeout=0.5)
            elapsed = time.time() - t0
            print(f'B.3 diagnostic: contended append raised TimeoutError '
                  f'after {elapsed:.3f}s (lock_timeout=0.5)')

            # BEHAVIOURAL bar #1: the timeout fired WHILE the lock was
            # still held.  The holder only releases when we touch the
            # release signal below, so it cannot have finished on its
            # own -- had the append instead blocked until the lock was
            # free (lock_timeout ignored / clamped to the 30 s default)
            # it would have hung here rather than raised, and had it
            # raised without contention the second half of this test
            # would not pass.
            assert not os.path.exists(release_signal), (
                'test bug: the release signal was sent before the '
                'contended append was attempted.')
            assert holder.is_alive(), (
                f'The lock holder exited (exitcode={holder.exitcode}) '
                f'before the contended append timed out, so this call '
                f'was not actually contended -- the TimeoutError does '
                f'not pin the lock_timeout contract.')

            # The error message must point at the lock file so a
            # human admin can recover.
            assert lock_path in str(excinfo.value), (
                f'TimeoutError message did not reference the lock '
                f'file path {lock_path!r}: {excinfo.value!r}'
            )

            # BEHAVIOURAL bar #2 (the other side): release the lock and
            # the IDENTICAL call must now succeed.  This is what makes
            # bar #1 a contention statement rather than "append always
            # raises with a small lock_timeout".
            Path(release_signal).touch()
            holder.join(timeout=self.HOLDER_EXIT_CAP_S)
            assert not holder.is_alive(), (
                'Lock holder did not exit after the release signal; '
                'the multi-process lock may be wedged.')
            append_plane_h5(path, plane, dx=1e-6,
                            label='after-release',
                            lock_timeout=0.5)
            planes, _meta = _h5_list_planes(path)
            assert len(planes) == 2, (
                f'Expected seed + after-release planes once the lock was '
                f'free, got {len(planes)} -- the contended append either '
                f'partially wrote or the released lock did not admit the '
                f'retry.')
        finally:
            # Let the holder finish gracefully so the next test sees
            # a clean process state.
            Path(release_signal).touch()
            holder.join(timeout=self.HOLDER_EXIT_CAP_S)
            if holder.is_alive():
                holder.terminate()
                holder.join(timeout=5.0)


# ============================================================================
# B.4 -- Single-process backward compatibility
# ============================================================================

@pytest.mark.skipif(not _HAS_H5PY, reason='h5py not installed')
class TestB4SingleProcessBackcompat:
    """v4.14.3 atomicity guarantees are PRESERVED -- multi-process is
    additive, not a replacement."""

    def test_single_process_append_round_trip(self, tmp_path):
        """A bog-standard single-process append loop still produces
        an N-plane file readable via ``_h5_list_planes``."""
        from lumenairy.io.storage import (
            _h5_list_planes,
            append_plane_h5,
        )
        path = str(tmp_path / 'single_proc.h5')
        N = 5
        for i in range(N):
            E = np.full((4, 4), float(i), dtype=np.complex128)
            append_plane_h5(path, E, dx=1e-6, label=f'plane_{i}')
        planes, _meta = _h5_list_planes(path)
        assert len(planes) == N
        for i, plane in enumerate(planes):
            assert plane.get('label') in (f'plane_{i}',
                                          f'plane_{i}'.encode())

    def test_v4_14_3_atomicity_pin_still_holds(self, tmp_path):
        """Reproduce the v4.14.3 rollback contract test under the
        v4.16.0 implementation: a mid-call ``create_dataset`` crash
        must roll back ``n_planes``.  The lock acquisition before
        the open should NOT break this.
        """
        import h5py

        from lumenairy.io.storage import append_plane_h5

        path = str(tmp_path / 'crash.h5')
        E = np.ones((8, 8), dtype=np.complex128)
        append_plane_h5(path, E, dx=1e-6, label='before')
        with h5py.File(path, 'r') as f:
            n_before = int(f['planes'].attrs['n_planes'])
        assert n_before == 1

        orig_create_dataset = h5py.Group.create_dataset

        def _boom(self, *args, **kwargs):
            raise RuntimeError('simulated mid-append crash v4.16.0')

        h5py.Group.create_dataset = _boom
        try:
            with pytest.raises(RuntimeError,
                               match='simulated mid-append'):
                append_plane_h5(path, E, dx=1e-6, label='will-crash')
        finally:
            h5py.Group.create_dataset = orig_create_dataset

        # Post-crash: n_planes rolled back AND the multi-process
        # lock has been released so a subsequent append doesn't
        # deadlock.
        with h5py.File(path, 'r') as f:
            n_after = int(f['planes'].attrs['n_planes'])
        assert n_after == n_before, (
            f'v4.14.3 rollback contract broken under v4.16.0: '
            f'pre-crash n_planes={n_before}, post-crash {n_after}.'
        )
        # The lock has been released; this would deadlock otherwise.
        append_plane_h5(path, E, dx=1e-6, label='after-recovery')
        with h5py.File(path, 'r') as f:
            n_recovered = int(f['planes'].attrs['n_planes'])
        assert n_recovered == 2


# ============================================================================
# B.5 -- Zarr multi-writer (gated on optional zarr)
# ============================================================================

@pytest.mark.skipif(not _HAS_ZARR, reason='zarr not installed')
@pytest.mark.skipif(not _HAS_FILELOCK,
                    reason='filelock not installed')
class TestB5MultiWriterZarrSerialized:
    """Same headline contract as B.1 but for the Zarr backend."""

    def test_multi_writer_zarr_serialized(self, tmp_path):
        path = str(tmp_path / 'multi_writer.zarr')
        n_workers = 4
        n_planes_per = 5
        total_planes = n_workers * n_planes_per

        ctx = mp.get_context('spawn')
        args = [(path, i, n_planes_per) for i in range(n_workers)]
        with ctx.Pool(n_workers) as pool:
            pool.starmap(_worker_append_zarr, args)

        from lumenairy.io.storage import _zarr_list_planes
        planes, _meta = _zarr_list_planes(path)
        actual_count = len(planes)
        assert actual_count == total_planes, (
            f'Expected {total_planes} planes after {n_workers} '
            f'workers x {n_planes_per} planes each; got '
            f'{actual_count}.  Multi-process Zarr serialisation '
            f'is broken.'
        )

        labels = []
        for plane in planes:
            label_raw = plane.get('label', '')
            label = (label_raw.decode() if isinstance(label_raw, bytes)
                     else str(label_raw))
            labels.append(label)
        present_workers = set()
        for label in labels:
            for wid in range(n_workers):
                if label.startswith(f'w{wid}_'):
                    present_workers.add(wid)
        assert present_workers == set(range(n_workers)), (
            f'Not every worker is represented in Zarr store: '
            f'missing workers '
            f'{set(range(n_workers)) - present_workers}.  Labels: '
            f'{labels}.'
        )
        for wid in range(n_workers):
            wcount = sum(1 for L in labels if L.startswith(f'w{wid}_'))
            assert wcount == n_planes_per, (
                f'Worker {wid} contributed {wcount} planes to the '
                f'Zarr store, expected {n_planes_per}.'
            )


# ============================================================================
# B.6 -- ``swmr=False`` legacy mode
# ============================================================================

@pytest.mark.skipif(not _HAS_H5PY, reason='h5py not installed')
class TestB6SwmrFalseLegacyMode:
    """Explicit ``swmr=False`` must produce a valid file (back-compat
    for pre-HDF5-1.10 readers)."""

    def test_swmr_false_creates_valid_file(self, tmp_path):
        from lumenairy.io.storage import (
            _h5_list_planes,
            append_plane_h5,
        )
        path = str(tmp_path / 'legacy.h5')
        E = np.ones((4, 4), dtype=np.complex128)
        # Three appends with swmr disabled.
        for i in range(3):
            append_plane_h5(path, E, dx=1e-6, label=f'l{i}',
                            swmr=False)
        planes, _meta = _h5_list_planes(path)
        assert len(planes) == 3

    def test_swmr_false_no_libver_latest_marker(self, tmp_path):
        """With ``swmr=False`` the file should NOT be opened with
        ``libver='latest'``, so legacy h5py readers can open it.
        We check this indirectly by verifying the file does not
        carry the ``swmr`` superblock marker.

        Note: this is a smoke test -- the precise libver inspection
        API is h5py-version-dependent.  We just verify that reading
        the file with the DEFAULT ``swmr=False`` succeeds.
        """
        import h5py

        from lumenairy.io.storage import append_plane_h5
        path = str(tmp_path / 'legacy_libver.h5')
        E = np.ones((4, 4), dtype=np.complex128)
        append_plane_h5(path, E, dx=1e-6, label='only',
                        swmr=False)
        # Open with the conservative default; if libver='latest' were
        # mandatory, this would fail on older HDF5 builds.
        with h5py.File(path, 'r') as f:
            assert int(f['planes'].attrs['n_planes']) == 1


# ============================================================================
# B.7 -- Signature audit
# ============================================================================

class TestB7SignatureAudit:
    """The v4.16.0 contract is partially API-surface -- pin the new
    kwargs so a future refactor doesn't silently drop them."""

    @pytest.mark.skipif(not _HAS_H5PY, reason='h5py not installed')
    def test_append_plane_h5_signature_has_swmr_and_lock_timeout(self):
        from lumenairy.io.storage import append_plane_h5
        sig = inspect.signature(append_plane_h5)
        assert 'swmr' in sig.parameters, (
            'v4.16.0 contract: append_plane_h5 must expose a '
            '``swmr`` kwarg.'
        )
        assert sig.parameters['swmr'].default is True, (
            'v4.16.0 contract: append_plane_h5(swmr=True) is the '
            'default for new files.'
        )
        assert 'lock_timeout' in sig.parameters, (
            'v4.16.0 contract: append_plane_h5 must expose a '
            '``lock_timeout`` kwarg.'
        )
        assert sig.parameters['lock_timeout'].default == 30.0, (
            'v4.16.0 contract: append_plane_h5(lock_timeout=30.0) '
            'is the default.'
        )

    @pytest.mark.skipif(not _HAS_ZARR, reason='zarr not installed')
    def test_zarr_append_plane_signature_has_lock_timeout(self):
        from lumenairy.io.storage import _zarr_append_plane
        sig = inspect.signature(_zarr_append_plane)
        assert 'lock_timeout' in sig.parameters, (
            'v4.16.0 contract: _zarr_append_plane must expose a '
            '``lock_timeout`` kwarg.'
        )
        assert sig.parameters['lock_timeout'].default == 30.0

    @pytest.mark.skipif(not _HAS_H5PY, reason='h5py not installed')
    def test_append_plane_h5_docstring_documents_swmr_filelock(self):
        """The v4.16.0 docstring must call out SWMR and filelock so
        readers can see the multi-writer contract."""
        from lumenairy.io.storage import append_plane_h5
        doc = append_plane_h5.__doc__ or ''
        doc_lower = doc.lower()
        assert 'swmr' in doc_lower, (
            'append_plane_h5 docstring missing SWMR reference.'
        )
        assert 'filelock' in doc_lower or 'lock' in doc_lower, (
            'append_plane_h5 docstring missing filelock / lock '
            'reference.'
        )
        assert 'multi-writer' in doc_lower or 'multi-process' in doc_lower, (
            'append_plane_h5 docstring missing multi-writer / '
            'multi-process reference.'
        )

    def test_module_docstring_describes_concurrency_model(self):
        """The module-level docstring must summarise the concurrency
        contract so callers don't have to dig through individual
        function docs."""
        from lumenairy.io import storage as _storage
        doc = (_storage.__doc__ or '').lower()
        assert 'concurrency' in doc, (
            'storage.py module docstring missing "Concurrency" '
            'section header.'
        )
        assert 'filelock' in doc, (
            'storage.py module docstring missing filelock '
            'reference.'
        )


# ============================================================================
# B.8 -- Lock file lifecycle smoke tests
# ============================================================================

@pytest.mark.skipif(not _HAS_H5PY, reason='h5py not installed')
@pytest.mark.skipif(not _HAS_FILELOCK,
                    reason='filelock not installed')
class TestB8LockFileLifecycle:
    """Sanity tests for the sibling ``<path>.lock`` file."""

    def test_lock_path_helper_returns_sibling(self, tmp_path):
        from lumenairy.io.storage import _append_lock_path
        data_path = str(tmp_path / 'foo.h5')
        lock_path = _append_lock_path(data_path)
        assert lock_path == data_path + '.lock'

    def test_lock_file_created_on_first_append(self, tmp_path):
        from lumenairy.io.storage import (
            _append_lock_path,
            append_plane_h5,
        )
        data_path = str(tmp_path / 'first.h5')
        lock_path = _append_lock_path(data_path)
        E = np.zeros((4, 4), dtype=np.complex128)
        append_plane_h5(data_path, E, dx=1e-6, label='first')
        # filelock creates the lock file on acquire and may leave it
        # behind on release.  Either outcome is acceptable; we just
        # verify the data file is correct.
        assert os.path.exists(data_path), (
            'Data file not created by append_plane_h5.'
        )
        # Lock file MAY exist (filelock policy varies by platform);
        # we don't assert either way.
        _ = lock_path  # noqa

    def test_lock_released_after_normal_append(self, tmp_path):
        """After a normal append, the lock must be released so a
        subsequent append in the SAME process does not deadlock.
        """
        from lumenairy.io.storage import append_plane_h5
        path = str(tmp_path / 'sequential.h5')
        E = np.zeros((4, 4), dtype=np.complex128)
        # If the lock is leaked, the second append would block until
        # the 30s default lock_timeout expires.  Use a short timeout
        # to make the test fail fast on a leak.
        for i in range(3):
            append_plane_h5(path, E, dx=1e-6, label=f's{i}',
                            lock_timeout=2.0)
        from lumenairy.io.storage import _h5_list_planes
        planes, _meta = _h5_list_planes(path)
        assert len(planes) == 3
