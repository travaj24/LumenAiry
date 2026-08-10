# THE RECORD-RUN GRID CHOICE, MADE LOUD (docs/audits/FIX_PERF_PARALLEL_2026_08_10.md)
#
# WHY THIS EXISTS.  ADJUDICATION_NFC_8192_2026_08_10.md sec 2.1 found that on
# a 137 GB box the shipped ``n_fine_cap = 16384`` -- the setting of record for
# design 121 -- ALREADY RAN AT 8192.  ``carrier._memory_bounded_n_fine``
# degraded it, said so in a ``RuntimeWarning`` labelled "RESOLUTION-LIMITED
# (non-converged)", and the run then printed a full acceptance banner that
# passed.  Nothing in the runner compared the grid it asked for against the
# grid it got, so a 4.7-hour reference run and a 1.3-hour exploration run were
# the same run with different labels.  A naive dual-arm comparison would have
# reported "8192 and 16384 agree exactly".
#
# The remedy is not a louder warning.  It is that the runner must state its
# INTENT (which grid, on which budget), PROVE before the chain starts that the
# clamp cannot bind, and FAIL if it would -- because a warning is a thing a
# production log swallows and an exit code is not.
#
# THREE THINGS ARE CHECKED, in this order, all before any chain runs:
#
#   1. CLAMP.  ``carrier._fine_grid_ceiling(budget)`` is the largest fine grid
#      the RAM clamp will approve, as a pure function of the budget.  The leg
#      asks for ``min(n_fine_req, n_fine_cap)``, so ``ceiling >= n_fine_cap``
#      PROVES the clamp cannot bind, for any request, without running it.
#   2. BOX.  With ``ram_budget=inf`` the clamp is switched off, so something
#      else has to refuse an impossible run.  ``k`` congruence workers each
#      hold ``carrier._fine_grid_peak_bytes(...)``; the parent additionally
#      holds the common-grid accumulator.  That total is checked against the
#      free memory the box actually reports, with a stated reserve.
#   3. AFTER THE RUN.  Every warning raised during the chain is recorded (and
#      still shown), then scanned for the clamp's own vocabulary.  This is the
#      only check that survives ``congruence_workers > 1``: the clamp then
#      runs INSIDE the workers, where no parent-side wrapper can see it, and
#      the multi entry point re-emits the workers' warnings in the parent.
#
# THE WORKER BUDGET IS DIVIDED, NOT COPIED.  ``_multi_capture_worker_state``
# hands each congruence worker ``get_ram_budget() // k`` (carrier.py, and it is
# right to: K workers each sizing themselves for the whole box is K-fold
# over-subscription).  So the effective budget on the leg is k times SMALLER
# than the box's, and the same ``n_fine_cap`` that is honoured at k=1 can be
# silently degraded at k=4.  That is why ``workers`` is an argument here and
# why the record runs pass an explicit ``ram_budget``.
#
# Import-safe by construction: no module-level work beyond the imports, so a
# spawn child that re-imports the runner pays nothing for it.
import os
import sys
import warnings

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from lumenairy.memory import get_ram_budget  # noqa: E402
from lumenairy.propagators import carrier as _C  # noqa: E402

#: Free memory left unclaimed by the BOX check.  Defaults to the same 8 GB
#: reserve ``_multi_resolve_workers`` uses for congruence workers, so the
#: runner-side refusal and the library-side clamp agree about what "fits"
#: means.  ``RAMRES=<GB>`` lowers it for a run whose operator has decided to
#: spend the margin -- a DELIBERATE, logged choice, which is the whole point
#: of this module; it is not a way to make the check go away, because the
#: modelled peak still has to fit what is left.
RESERVE_GB = float(os.environ.get('RAMRES', '8') or 8.0)

#: Substrings of the clamp's own guard messages.  A run of record must not see
#: any of them.  Kept as data rather than one regex so the failure report can
#: name which one fired.
_DEGRADE_MARKS = (
    'MEMORY-LIMITED',            # _memory_bounded_n_fine
    'COUNT-LIMITED',             # readout n_fine_cap bind
    'RESOLUTION-LIMITED',        # both, the shared verdict phrase
)


def resolve_ram_budget(env_value, workers=1):
    """Turn the ``RAMB`` env string into (override, effective, description).

    ``override`` is what to put in ``output_grid['ram_budget']`` /
    ``focus_readout['ram_budget']`` -- ``None`` means "pass nothing, use the
    box".  ``effective`` is the budget the clamp will actually see ON THE LEG,
    which is where the division by ``workers`` is applied.
    """
    workers = max(1, int(workers))
    v = str(env_value).strip().lower()
    if v in ('', 'auto', 'box', 'none'):
        box = float(get_ram_budget())
        return None, box / workers, (
            'auto (the box: get_ram_budget() = %.2f GB%s)'
            % (box / 1e9, '' if workers == 1
               else ', DIVIDED by %d workers -> %.2f GB each'
                    % (workers, box / workers / 1e9)))
    if v == 'inf':
        return float('inf'), float('inf'), 'inf (RAM clamp DISABLED, explicitly)'
    try:
        gb = float(v)
    except ValueError:
        raise SystemExit(
            "RAMB must be 'auto', 'inf' or a number of GB, got %r" % env_value)
    if not (gb > 0.0):
        raise SystemExit("RAMB must be > 0 GB, got %r" % env_value)
    return gb * 1e9, gb * 1e9, '%.2f GB, per worker, explicitly' % gb


def preflight(n_fine_cap, ramb_env='auto', workers=1, n_px=0, n_out=0,
              label='run', paraxial=False, stream=None):
    """PROVE the run gets ``n_fine_cap``, or refuse it loudly.

    Returns the ``ram_budget`` value to pass through (possibly ``None``).
    Raises ``SystemExit(2)`` -- not a warning -- when either check fails.

    ``paraxial=True`` (``final_leg='paraxial'``) builds no fine grid, so both
    checks reduce to the chain's own footprint and ``n_fine_cap`` is not
    asserted.
    """
    out = stream or sys.stdout
    workers = max(1, int(workers))
    override, effective, how = resolve_ram_budget(ramb_env, workers)

    def _say(s):
        print(s, file=out, flush=True)

    _say("")
    _say("GRID INTENT -- %s" % label)
    _say("  n_fine_cap requested : %s" % ('n/a (paraxial leg)' if paraxial
                                          else str(int(n_fine_cap))))
    _say("  ram_budget           : %s" % how)
    _say("  congruence workers   : %d" % workers)

    if paraxial:
        _say("  paraxial leg: no fine grid is built, so neither the clamp nor "
             "the box check applies.")
        return override

    n_fine_cap = int(n_fine_cap)
    # ---- 1. CLAMP -------------------------------------------------------
    ceiling = _C._fine_grid_ceiling(effective)
    need_b = (_C._FINE_GRID_WORK_ARRAYS * 16.0 * float(n_fine_cap) ** 2
              / _C._FINE_GRID_RAM_FRAC)
    _say("  clamp ceiling        : %s  (the largest grid %s approves; %d "
         "needs %.1f GB of budget)"
         % (('unbounded' if effective == float('inf') else '%d' % ceiling),
            ('an unbounded budget' if effective == float('inf')
             else 'a %.2f GB budget' % (effective / 1e9)),
            n_fine_cap, need_b / 1e9))
    if ceiling < n_fine_cap:
        _refuse(out, label,
                "the RAM clamp would DEGRADE the fine grid from %d to %d."
                % (n_fine_cap, ceiling),
                [("this run would have produced a %dx%d answer wearing a %d "
                  "label -- the silent-8192 failure of "
                  "ADJUDICATION_NFC_8192_2026_08_10.md sec 2.1."
                  % (ceiling, ceiling, n_fine_cap)),
                 ("the clamp needs %.1f GB of budget for %d "
                  "(%d work arrays x 16 B x %d^2 over a %.0f%% reserve); the "
                  "leg will see %.2f GB."
                  % (need_b / 1e9, n_fine_cap, _C._FINE_GRID_WORK_ARRAYS,
                     n_fine_cap, 100.0 * _C._FINE_GRID_RAM_FRAC,
                     effective / 1e9))],
                ["%-20s-- disable the clamp, having read that one order is "
                 "modelled at %.1f GB"
                 % ('RAMB=inf',
                    _C._fine_grid_peak_bytes(n_fine_cap, n_px) / 1e9),
                 "%-20s-- pin the budget (>= %.1f)" % ('RAMB=<GB>',
                                                       need_b / 1e9),
                 "%-20s-- ask for the grid the box can hold"
                 % ('NFC=%d' % ceiling),
                 ("%-20s-- stop dividing the budget by %d"
                  % ('CW=1', workers)) if workers > 1 else
                 "%-20s-- close what is holding it" % 'free memory'])

    # ---- 2. BOX ---------------------------------------------------------
    import psutil
    avail = float(psutil.virtual_memory().available)
    per_worker = _C._fine_grid_peak_bytes(n_fine_cap, n_px)
    acc = 16.0 * float(n_out) ** 2 if n_out else 0.0
    # THE ACCUMULATOR IS CHARGED ONLY AT k > 1, and that is measured, not a
    # convenience.  At k=1 the congruence runs IN the parent, so the fine
    # leg's peak and the common-grid accumulator are SEQUENTIAL: the leg's
    # arrays are gone before the accumulator is filled.  ADJUDICATION_NFC_8192
    # sec 5.3's arm B is the direct evidence -- 32 orders at NFC=16384 on the
    # NOUT=32768 grid (a 17.2 GB accumulator) peaked at 71.9-80.5 GiB = 77-86
    # GB, against a leg peak of ~83.5 GB on its own; i.e. +3 GB, not +17.2.
    # AUDIT_TRACED_SPEED sec 3.3 recorded the same thing from the other side
    # ("the N_out=32768 accumulator is NOT yet allocated at that instant").
    # At k > 1 they DO co-occur: the parent holds the accumulator while the
    # workers hold their legs, which is the whole shape of the k-way peak.
    parent = (acc + _C._FINE_GRID_BASE_BYTES) if workers > 1 else 0.0
    total = workers * per_worker + parent
    _say("  modelled peak        : %d x %.1f GB per worker + %.1f GB parent "
         "= %.1f GB   (free %.1f GB, reserve %.0f GB)"
         % (workers, per_worker / 1e9, parent / 1e9, total / 1e9,
            avail / 1e9, RESERVE_GB))
    if total > avail - RESERVE_GB * 1e9:
        _refuse(out, label,
                "the box cannot hold this run: %.1f GB modelled against "
                "%.1f GB free (%.0f GB reserved)."
                % (total / 1e9, avail / 1e9, RESERVE_GB),
                [("%d worker(s) x %.1f GB (fine grid %d^2 at %d work arrays + "
                  "a %.1f GB process floor) + %.1f GB parent."
                  % (workers, per_worker / 1e9, n_fine_cap,
                     _C._FINE_GRID_WORK_ARRAYS,
                     _C._FINE_GRID_BASE_BYTES / 1e9, parent / 1e9)),
                 ("the model is MEASURED, not allowed for: see "
                  "FIX_PERF_PARALLEL_2026_08_10.md sec 3.")],
                ["%-20s-- the k this box can actually hold"
                 % ('CW=%d' % max(1, int((avail - RESERVE_GB * 1e9 - parent)
                                         // max(per_worker, 1.0)))),
                 "%-20s-- a lighter grid" % ('NFC=%d' % (n_fine_cap // 2)),
                 "%-20s-- close what is holding it" % 'free memory',
                 "%-20s-- spend the reserve, deliberately (currently %.0f GB)"
                 % ('RAMRES=<GB>', RESERVE_GB)])
    _say("  VERDICT: the clamp cannot bind and the box can hold it.")
    return override


def _refuse(out, label, headline, why, remedies, code=2):
    bar = '!' * 74
    print("", file=out)
    print(bar, file=out)
    print("REFUSED -- %s" % label, file=out)
    print(bar, file=out)
    print("  %s" % headline, file=out)
    for w in why:
        print("  * %s" % w, file=out)
    print("  Remedies:", file=out)
    for r in remedies:
        print("    %s" % r, file=out)
    print(bar, file=out, flush=True)
    raise SystemExit(code)


class record_warnings:
    """Record every warning raised inside the block AND still show it.

    ``warnings.catch_warnings(record=True)`` would swallow them, and these
    runners' warnings are diagnostics the campaign reads (AUDIT_TRACED_SPEED
    sec 8.4).  Teeing ``showwarning`` keeps both.  Congruence workers'
    warnings arrive here too: the multi entry point re-raises them in the
    parent, tagged with the congruence name.
    """

    def __init__(self):
        self.records = []

    def __enter__(self):
        self._orig = warnings.showwarning

        def _tee(message, category, filename, lineno, file=None, line=None):
            self.records.append({
                'msg': str(message), 'cat': getattr(category, '__name__', '?'),
                'where': '%s:%d' % (os.path.basename(str(filename)), lineno)})
            return self._orig(message, category, filename, lineno, file, line)

        warnings.showwarning = _tee
        return self

    def __exit__(self, *exc):
        warnings.showwarning = self._orig
        return False

    def degradations(self):
        return [r for r in self.records
                if any(m in r['msg'] for m in _DEGRADE_MARKS)]


def assert_no_grid_degradation(rec, n_fine_cap, label='run', stream=None):
    """Post-run half of the intent check: the clamp must not have fired.

    Returns silently, or raises ``SystemExit(3)``.  Separate exit code from
    the pre-flight's 2 on purpose: a 2 means "this box cannot do what you
    asked", a 3 means "the pre-flight said it could and it did not", which is
    a defect in the model, not in the request.
    """
    out = stream or sys.stdout
    bad = rec.degradations()
    if not bad:
        print("  grid check: no MEMORY/COUNT/RESOLUTION-LIMITED warning was "
              "raised -- the leg ran at the %d it was asked for."
              % int(n_fine_cap), file=out, flush=True)
        return
    _refuse(out, label,
            "the fine grid WAS degraded during the run, after the pre-flight "
            "proved it could not be.",
            ['%s  [%s]' % (b['msg'][:220], b['where']) for b in bad[:4]],
            ["re-run with RAMB=inf or a pinned RAMB=<GB>",
             "report this: the pre-flight model and the clamp disagree"],
            code=3)
