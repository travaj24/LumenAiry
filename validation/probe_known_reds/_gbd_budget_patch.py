"""Make the dense GBD reconstruction's memory budget honest -- opt-in.

Measured (``probe_gbd_dense_budget.py``): the dense path's declared 16 B per
(cell x beamlet-column) under-counts its own working set by 4.5-6.0x, so
``mem_budget_mb`` is not a bound.  Correcting the constant changes the chunk
size, hence the summation order, hence the output bytes -- a default-path
behaviour change -- so it ships behind a module switch whose default
reproduces the previous release byte for byte, and the flip is a decision for
the maintainer.
"""
import ast
import io
import sys

P = 'lumenairy/propagators/gbd.py'

OLD = """    # v5.21: auto-shrink chunk_beamlets to the memory budget.  bytes per
    # beamlet-column of the dense working set ~ Ny*Nx*16 (the dX/dY/rho2/phase
    # buffers); keep chunk*Ny*Nx*16 under mem_budget.  Never grows the chunk
    # (so small-N default runs stay byte-identical); only shrinks when a chunk
    # would blow the budget.
    if mem_budget_mb and mem_budget_mb > 0:
        _bytes_per_col = Ny * Nx * 16.0
        _max_chunk = max(1, int(mem_budget_mb * 1e6 / max(1.0, _bytes_per_col)))
        chunk_beamlets = min(chunk_beamlets, _max_chunk)
"""

NEW = """    # v5.21: auto-shrink chunk_beamlets to the memory budget.  Never grows the
    # chunk (so small-N default runs stay byte-identical); only shrinks when a
    # chunk would blow the budget.  Which per-cell cost is used is
    # :data:`DENSE_MEM_BUDGET_ACCOUNTING` -- see its note for the measurement
    # and for why the honest figure is not yet the default.
    if mem_budget_mb and mem_budget_mb > 0:
        _cell_bytes = (_DENSE_CELL_BYTES_MEASURED
                       if DENSE_MEM_BUDGET_ACCOUNTING == 'measured'
                       else _DENSE_CELL_BYTES_LEGACY)
        _bytes_per_col = Ny * Nx * _cell_bytes
        _max_chunk = max(1, int(mem_budget_mb * 1e6 / max(1.0, _bytes_per_col)))
        chunk_beamlets = min(chunk_beamlets, _max_chunk)
"""

ANCHOR = "_WINDOWED_CELL_BYTES = 32.0\n"

CONSTS = '''_WINDOWED_CELL_BYTES = 32.0

#: Bytes per (output cell x beamlet-column) the DENSE reconstruction loop is
#: assumed to hold live, used to shrink ``chunk_beamlets`` to ``mem_budget_mb``.
#:
#: ``16.0`` is the figure shipped since v5.21.  Its comment read "the
#: dX/dY/rho2/phase buffers", but 16 B is the size of ONE complex128 element,
#: not the sum of three float64 buffers and one complex128 -- so the budget was
#: never a bound.  MEASURED with ``tracemalloc`` over a 64/128/192/256 grid
#: ladder at 512 and 64 MB budgets and 512 / 1024 beamlets
#: (``validation/probe_known_reds/probe_gbd_dense_budget.py``, 2026-09-14,
#: py3.14.6 / numpy 2.4.4): the live peak is **72.0 to 96.8 B per cell-column**,
#: i.e. the loop overruns its own budget by 1.2x to 6.0x, saturating at 6.0x
#: (= 96/16) once the chunk is the binding constraint.  Worst cell measured:
#: ``mem_budget_mb=512`` on a 256^2 grid with 1024 beamlets peaked at
#: **3 073 MB**.  A 64 MB budget peaked at 387 MB.
_DENSE_CELL_BYTES_LEGACY = 16.0

#: The honest figure: the measured maximum (96.8) with the same margin the
#: windowed sibling carries (it ships 32.0 against ~26 measured, 1.23x), rounded
#: up to a power of two -- 128 is 1.32x the worst measurement, and every
#: measured cell sits under it.  The lower side of the bar is set by the
#: measurement (below 96.8 the budget stops bounding the loop); the upper side
#: by cost, because a larger constant only shrinks the chunk and buys wall time
#: for nothing: at 128 the chunk is 8x smaller than at 16.
_DENSE_CELL_BYTES_MEASURED = 128.0

#: ``'legacy'`` (default) or ``'measured'``.  Which of the two constants above
#: the dense chunk sizing uses.
#:
#: WHY THIS IS A SWITCH AND NOT A REPAIR IN PLACE.  The constant sets the chunk
#: boundary, the chunk boundary sets the order the per-chunk ``einsum``
#: reductions are summed in, and floating-point addition is not associative --
#: so correcting it MOVES THE OUTPUT BYTES on a default path.  Under the house
#: rule, a default moves only with a Migration note and a measurement beside it,
#: and everything else ships opt-in behind a switch whose default reproduces the
#: previous release exactly.  ``'legacy'`` does that.
#:
#: WHAT THE DEFECT COSTS TODAY.  ``mem_budget_mb`` does not bound this loop, so
#: a caller who sets it to fit a machine can still be handed a multi-gigabyte
#: transient (measured 3 073 MB against a 512 MB request).  Handoff section 5
#: records a long pytest run on the maintainer's box dying twice with
#: ``Windows fatal exception: access violation``, once inside this dense path,
#: "which passes alone in 30 s" -- the signature of an allocation that only
#: fails beside other heavy jobs.  This measurement BOUNDS that: the transient
#: is up to 6x what was asked for.  It does not prove the fault, and no fault
#: was reproduced here.
#:
#: MITIGATION WITHOUT FLIPPING THE SWITCH: pass ``window=5.0`` (the bounded-
#: support scatter-add, whose own accounting IS correct), or divide
#: ``mem_budget_mb`` by 6.
#:
#: THE FLIP IS A DECISION RESERVED FOR THE MAINTAINER (handoff 4.7 list).
DENSE_MEM_BUDGET_ACCOUNTING = 'legacy'
'''


def main():
    with io.open(P, encoding='cp1252', newline='') as fh:
        raw = fh.read()
    nl = '\r\n' if '\r\n' in raw else '\n'
    src = raw.replace('\r\n', '\n')
    for old, new in ((ANCHOR, CONSTS), (OLD, NEW)):
        if src.count(old) != 1:
            print('no unique match (%d):' % src.count(old),
                  old.splitlines()[0])
            return 1
        src = src.replace(old, new)
    ast.parse(src)
    with io.open(P, 'w', encoding='cp1252', newline='') as fh:
        fh.write(src.replace('\n', nl))
    print('patched', P)
    return 0


if __name__ == '__main__':
    sys.exit(main())
