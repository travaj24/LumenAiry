"""Sweep ``carrier.py``'s three warning helpers onto ``caller_stacklevel``.

Run once.  Line-ending agnostic: the file is read with universal newlines and
written back with the newline it had.
"""
import io
import sys

P = 'lumenairy/propagators/carrier.py'

EDITS = [
    # ---- the import -------------------------------------------------------
    ("import numpy as np\n"
     "\n"
     "from .fresnel import fresnel_tf_propagate\n",
     "import numpy as np\n"
     "\n"
     "from ..elements._lens_kernels import caller_stacklevel "
     "as _caller_stacklevel\n"
     "from .fresnel import fresnel_tf_propagate\n"),

    # ---- _guard_dispose ---------------------------------------------------
    ("def _guard_dispose(action, msg, exc=RuntimeError, stacklevel=3):\n"
     '    """Apply an ``\'error\'`` / ``\'warn\'`` / ``\'ignore\'`` disposition'
     " to a\n"
     "    detected fault.  ``'warn'`` emits a ``RuntimeWarning``; ``'error'``"
     " raises\n"
     "    ``exc``; ``'ignore'`` is silent.\n"
     "\n"
     "    ``stacklevel`` is counted FROM THE CALLING GUARD SITE (this helper's"
     " own\n"
     "    frame is added internally), so a guard that used to call\n"
     "    ``warnings.warn(..., stacklevel=3)`` inline keeps pointing at the"
     " same\n"
     '    frame when it is converted to ``_guard_dispose(..., stacklevel=3)``.'
     '"""\n'
     "    if action == 'error':\n"
     "        raise exc(msg)\n"
     "    if action == 'warn':\n"
     "        import warnings\n"
     "        warnings.warn(msg, RuntimeWarning, stacklevel=int(stacklevel)"
     " + 1)\n",

     "def _guard_dispose(action, msg, exc=RuntimeError, stacklevel=None):\n"
     '    """Apply an ``\'error\'`` / ``\'warn\'`` / ``\'ignore\'`` disposition'
     " to a\n"
     "    detected fault.  ``'warn'`` emits a ``RuntimeWarning``; ``'error'``"
     " raises\n"
     "    ``exc``; ``'ignore'`` is silent.\n"
     "\n"
     "    ``stacklevel`` defaults to ``None``, which COMPUTES the level with\n"
     "    :func:`~lumenairy.elements._lens_kernels.caller_stacklevel` -- the"
     " depth\n"
     "    of the first frame outside the package, measured from this helper's"
     " own\n"
     "    frame.  That is what makes the attribution independent of how deep"
     " the\n"
     "    guard site sits: the same guard reached directly, through a chain"
     " leg or\n"
     "    through a re-entrant entry point names the USER'S frame either way,"
     "\n"
     "    where a literal is right for one of those and silently wrong for the"
     "\n"
     "    others (2026-09-14; the lens family was swept onto the same helper"
     " by\n"
     "    WP-B11).\n"
     "\n"
     "    An explicit integer is still honoured, and is still counted FROM"
     " THE\n"
     "    CALLING GUARD SITE (this helper's own frame is added internally),"
     " so an\n"
     "    existing ``_guard_dispose(..., stacklevel=3)`` keeps pointing"
     " exactly\n"
     "    where it did.  Nothing outside this module passes one today; the"
     " seam\n"
     "    is kept for a caller that genuinely wants a fixed frame.\n"
     '    """\n'
     "    if action == 'error':\n"
     "        raise exc(msg)\n"
     "    if action == 'warn':\n"
     "        import warnings\n"
     "        level = (_caller_stacklevel() if stacklevel is None\n"
     "                 else int(stacklevel) + 1)\n"
     "        warnings.warn(msg, RuntimeWarning, stacklevel=level)\n"),

    # ---- _warn_undeduped --------------------------------------------------
    ("def _warn_undeduped(msg, stacklevel=3, category=RuntimeWarning):\n",
     "def _warn_undeduped(msg, stacklevel=None, category=RuntimeWarning):\n"),

    ("    import sys\n"
     "    import warnings\n"
     "    try:\n"
     "        frame = sys._getframe(int(stacklevel))\n",

     "    import sys\n"
     "    import warnings\n"
     "    # ``stacklevel`` is counted from THIS helper's CALLER (1 = the\n"
     "    # caller), one frame in from ``caller_stacklevel``'s own counting\n"
     "    # (1 = this frame), so the computed default is its reading minus"
     " one.\n"
     "    if stacklevel is None:\n"
     "        stacklevel = max(1, _caller_stacklevel() - 1)\n"
     "    try:\n"
     "        frame = sys._getframe(int(stacklevel))\n"),

    # ---- _check_collins_sampling -----------------------------------------
    ("def _check_collins_sampling(fn, action, st, stacklevel=3,"
     " check_period=False):\n",
     "def _check_collins_sampling(fn, action, st, stacklevel=None,\n"
     "                            check_period=False):\n"),
]


def main():
    with io.open(P, encoding='cp1252', newline='') as fh:
        raw = fh.read()
    nl = '\r\n' if '\r\n' in raw else '\n'
    src = raw.replace('\r\n', '\n')
    for old, new in EDITS:
        n = src.count(old)
        if n != 1:
            print('FAILED to match uniquely (%d):' % n, old.splitlines()[0])
            return 1
        src = src.replace(old, new)
    with io.open(P, 'w', encoding='cp1252', newline='') as fh:
        fh.write(src.replace('\n', nl))
    print('patched', P, '(%d edits, newline=%r)' % (len(EDITS), nl))
    return 0


if __name__ == '__main__':
    sys.exit(main())
