"""Shared background-worker scaffolding for the designer's docks.

Every analysis dock runs its long computation on a ``QThread`` and
reports back through a Qt signal.  Sixteen hand-written copies of that
pattern had drifted into three different cancellation conventions
(``isInterruptionRequested``, ``CancellableProgress.should_stop``,
nothing at all) and two different finished-signal names -- one of which
shadowed ``QThread.finished`` itself.

The concrete problem this module fixes: ``MainWindow.closeEvent`` calls
``_shutdown_dock_workers``, which does ``requestInterruption()`` and
then ``wait(2000)`` on every running worker.  A worker that never polls
the flag cannot wind down, so the wait times out, Qt destroys the
window with threads still running, and the process aborts with
``QThread: Destroyed while thread is still running`` -- taking any
in-flight HDF5 / Zarr write with it.

Use :class:`AnalysisWorker` for new workers.  For the existing ones,
:func:`interrupt_check` is a drop-in poll that works on any ``QThread``
and also honours a ``CancellableProgress`` if the worker owns one.

Author: Andrew Traverso
"""

import weakref

from PySide6.QtCore import QThread, Signal

from ..progress import CancellableProgress


class ThreadCancellableProgress(CancellableProgress):
    """A :class:`~lumenairy.progress.CancellableProgress` that is ALSO
    tripped by Qt's ``requestInterruption()``.

    Two cancellation channels existed side by side: the docks' own Stop
    buttons called ``cancel()`` on a ``CancellableProgress``, while
    ``MainWindow._shutdown_dock_workers`` (and the Stop button on other
    docks) called ``QThread.requestInterruption()``.  A worker that
    polled only the first ignored the second, so quitting mid-run hit
    the 2 s wait timeout and Qt aborted the process.

    Giving a worker one of these makes every existing
    ``self._cancel_progress.should_stop`` poll honour BOTH -- and,
    because this is still a valid ``progress=`` callback, it also makes
    any library call the worker passes it to interruptible at the
    library's own checkpoints.

    The thread is held weakly so the progress object can outlive it.
    """

    def __init__(self, thread, parent=None):
        super().__init__(parent)
        self._thread_ref = weakref.ref(thread)

    @property
    def should_stop(self) -> bool:
        if super().should_stop:
            return True
        th = self._thread_ref()
        if th is None:
            return False
        try:
            return bool(th.isInterruptionRequested())
        except Exception:
            return False


class WorkerInterrupted(Exception):
    """Raised by :meth:`AnalysisWorker.check_interrupt` to unwind out of
    a deeply nested computation when cancellation is requested."""


def interrupt_check(worker):
    """True when ``worker`` should stop.

    Reads BOTH cancellation channels:

    * Qt's own ``requestInterruption()`` -- what the Stop button and
      ``MainWindow._shutdown_dock_workers`` use;
    * a ``CancellableProgress`` stored on ``_cancel_progress``, which
      several docks use for their in-dock Cancel.

    Safe on any object: missing attributes just read as "keep going".
    """
    try:
        if worker.isInterruptionRequested():
            return True
    except Exception:
        pass
    prog = getattr(worker, '_cancel_progress', None)
    if prog is not None:
        try:
            if prog.should_stop:
                return True
        except Exception:
            pass
    return False


class AnalysisWorker(QThread):
    """Base for the designer's background analysis workers.

    Guarantees:

    * the finished signal is called ``finished_result``, so it never
      shadows ``QThread.finished`` and the canonical
      ``worker.finished.connect(worker.deleteLater)`` idiom keeps
      working;
    * ``run()`` emits ``finished_result`` EXACTLY ONCE on every path,
      including an exception inside :meth:`work` and an exception
      inside the exception's own ``__str__``;
    * :meth:`check_interrupt` honours both cancellation channels.

    Subclasses implement :meth:`work` and return the payload; they do
    not emit or catch anything themselves.  Snapshot whatever model
    state the run needs in ``__init__`` (which executes on the GUI
    thread) -- never read the live model from :meth:`work`.
    """

    finished_result = Signal(object)

    #: Payload emitted when the run is interrupted.
    INTERRUPTED_PAYLOAD = {'error': 'Stopped by user'}

    def check_interrupt(self):
        """Raise :class:`WorkerInterrupted` if cancellation was asked
        for.  Call it at every natural loop boundary."""
        if interrupt_check(self):
            raise WorkerInterrupted()

    def interrupted(self):
        """Non-raising form of :meth:`check_interrupt`."""
        return interrupt_check(self)

    def work(self):
        """Do the computation and return the result payload."""
        raise NotImplementedError(
            f'{type(self).__name__}.work: subclasses must implement '
            f'work() and return the result payload.')

    def run(self):
        try:
            payload = self.work()
        except WorkerInterrupted:
            self.finished_result.emit(dict(self.INTERRUPTED_PAYLOAD))
            return
        except Exception as e:
            try:
                msg = f'{type(e).__name__}: {e}'
            except Exception:
                msg = type(e).__name__
            self.finished_result.emit({'error': msg})
            return
        self.finished_result.emit(payload)
