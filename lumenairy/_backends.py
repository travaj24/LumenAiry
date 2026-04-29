"""
Runtime CPU-count detection that respects affinity masks and cgroups.

Every place in the library that wants to dispatch work across worker
processes or threads should go through :func:`available_cpus` rather
than calling :func:`os.cpu_count` directly.  ``os.cpu_count`` returns
the total logical CPU count on the machine and ignores:

* Python 3.13+ ``os.process_cpu_count`` semantics (containers, cgroup
  CPU quotas, Linux ``CPU_LIMIT``).
* ``taskset`` / ``sched_setaffinity`` affinity masks on Linux.
* Windows process-affinity masks (exposed through ``psutil`` if
  installed).

Using the real "what can this process actually use" number keeps the
Newton-inversion pool, the FFT backend, and any future worker code
from oversubscribing a subset of cores.

Author: Andrew Traverso
"""

from __future__ import annotations

import os


def available_cpus() -> int:
    """Return the number of CPUs this process can actually use.

    Preference order:

    1. ``os.process_cpu_count()`` (Python 3.13+): the canonical
       "CPUs available to this process" number, respects CPU-quota
       cgroups and affinity masks.
    2. ``len(os.sched_getaffinity(0))`` (Linux / BSD): honours
       ``taskset`` restrictions.
    3. ``len(psutil.Process().cpu_affinity())`` (optional
       cross-platform path, captures Windows process affinity).
    4. ``os.cpu_count()`` fallback -- the raw logical-CPU count, used
       only when nothing above is available.

    Always returns at least 1.
    """
    if hasattr(os, 'process_cpu_count'):
        try:
            n = os.process_cpu_count()
            if n:
                return int(n)
        except Exception:
            pass

    if hasattr(os, 'sched_getaffinity'):
        try:
            n = len(os.sched_getaffinity(0))
            if n > 0:
                return int(n)
        except Exception:
            pass

    try:
        import psutil
        n = len(psutil.Process().cpu_affinity())
        if n > 0:
            return int(n)
    except Exception:
        pass

    return max(1, int(os.cpu_count() or 1))


__all__ = ['available_cpus']
