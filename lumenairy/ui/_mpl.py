"""Lazy matplotlib access and shared axis styling for the designer UI.

Importing ``matplotlib.figure`` / ``matplotlib.backends.backend_qtagg``
at module scope made every dock module drag the whole matplotlib stack
in at import time -- including docks the user never opens and modules
(``model``, ``analysis``) that draw nothing.  ``python -X importtime -c
"import matplotlib.figure"`` costs ~17.5 s cold on a Windows
workstation, sub-second warm.

Use it as::

    from . import _mpl
    ...
    self.fig = _mpl.Figure(figsize=(6, 3.4), dpi=100)
    self.canvas = _mpl.FigureCanvasQTAgg(self.fig)

The attribute access is what triggers the import, so a module that is
imported but never constructs a figure pays nothing.  (A bare
``from ._mpl import Figure`` would defeat this -- PEP 562's module
``__getattr__`` runs at import time for that form.)

Author: Andrew Traverso
"""

# Dark-theme axis colours, previously copy-pasted as a
# facecolor / tick_params / spines triple into ~20 docks.
BG = '#0a0c10'
FG = '#7a94b8'
GRID = '#2a3548'

_LAZY = {
    'Figure': ('matplotlib.figure', 'Figure'),
    'FigureCanvasQTAgg': ('matplotlib.backends.backend_qtagg',
                          'FigureCanvasQTAgg'),
    'NavigationToolbar2QT': ('matplotlib.backends.backend_qtagg',
                             'NavigationToolbar2QT'),
}


def __getattr__(name):
    """PEP 562 hook: import matplotlib on first use of a drawing name."""
    try:
        mod_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(
            f'module {__name__!r} has no attribute {name!r}') from None
    import importlib
    try:
        value = getattr(importlib.import_module(mod_name), attr)
    except ImportError as exc:
        # ``__dir__`` advertises these names, so anything that walks the
        # module -- ``hasattr``, ``inspect.getmembers``, ``help()``, a
        # REPL completion -- reaches this hook.  A module __getattr__ is
        # contractually allowed to raise only AttributeError there; the
        # raw ImportError (``No module named 'shiboken6'`` on a headless
        # box with no Qt bindings) escaped ``hasattr`` instead of making
        # it return False.  The cause is chained so a caller that really
        # wanted the figure still sees why.
        raise AttributeError(
            f'module {__name__!r} cannot provide {name!r}: '
            f'{mod_name} is unavailable ({exc}).') from exc
    globals()[name] = value      # cache: later lookups skip this hook
    return value


def __dir__():
    return sorted(list(globals()) + list(_LAZY))


def style_axes(ax, *, grid=False):
    """Apply the designer's dark-theme palette to one Axes.

    Replaces the ``set_facecolor`` / ``tick_params`` / per-spine
    ``set_color`` triple that was duplicated verbatim across the
    analysis docks, so a theme change is one edit instead of twenty.
    """
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    if grid:
        ax.grid(True, color=GRID, alpha=0.4, linewidth=0.5)
    return ax
