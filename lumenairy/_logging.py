"""Library-wide logging convention.

v5.3.2 (ROADMAP "logging adoption sweep"): adds per-iteration
TELEMETRY for the 3 long-running paths the audit cited
(apply_real_lens_traced, design_optimize, monte_carlo_tolerancing).

This is NOT a conversion of ``warnings.warn`` to ``logger.warning``;
the deprecation / sampling-violation warning surface is part of the
library's public API contract and stays unchanged.

Idiom: each consumer module does ``from .._logging import get_logger``
and ``logger = get_logger(__name__)``.  Per-iteration progress sites
call ``logger.info(...)``.  Users attach a handler to see progress:

    import logging
    logging.basicConfig(level=logging.INFO)
    # or per-logger:
    logging.getLogger('lumenairy').setLevel(logging.INFO)

By default the library logger has a ``NullHandler`` so library
import / use produces no log output.
"""
# v5.3.2 (ROADMAP logging adoption sweep -- per-iteration telemetry):
# Module-level root logger setup.  The ``NullHandler`` keeps library
# import / use silent by default; users opt in by attaching their own
# handler (e.g. ``logging.basicConfig(level=logging.INFO)``).
import logging

_ROOT_LOGGER_NAME = 'lumenairy'
_root = logging.getLogger(_ROOT_LOGGER_NAME)
_root.addHandler(logging.NullHandler())


def get_logger(name):
    """Return a child logger of the ``lumenairy`` root logger.

    Parameters
    ----------
    name : str
        Typically the consumer module's ``__name__``, e.g.
        ``'lumenairy.elements._lens_traced'``.  The returned logger
        is a descendant of the ``lumenairy`` root logger so messages
        propagate up through the root's ``NullHandler`` (silent by
        default) and through any handler the user has attached to
        ``'lumenairy'`` (or to a sub-logger).

    Returns
    -------
    logger : logging.Logger
    """
    return logging.getLogger(name)
