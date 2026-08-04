# P2 DID-NOT-WARN diagnosis -- AXIS 3: read a RUNNING shard's verdict per test.
#
# CI runs the fast gate as ``pytest-split --splits 3 --group 2``.  Reproducing
# that shard takes hours, and pytest's ``-q`` progress stream is one CHARACTER
# per test (``.`` pass, ``F`` fail, ``s`` skip, ``E`` error, ``x`` xfail ...)
# with the node names only printed in the summary AT THE END.  This maps the
# characters back onto the collected node IDs so the shard can be read while it
# is still running -- and in particular so
# ``test_niche_p2_guards.py::test_self_check_dx_flags_a_non_convergent_chain``
# (position 1744 of 3424 on the py3.13 / Linux collection, vs position 291 on
# the Windows / py3.14 one C11 measured) can be adjudicated the moment the run
# passes it.
#
# usage:
#   # 1. collect the shard's node IDs, in order
#   python -m pytest tests/unit -m "not integration and not slow" \
#       --collect-only -q -p no:cacheprovider --splits 3 --group 2 \
#       --splitting-algorithm least_duration --durations-path .test_durations \
#       | grep '::' > ids.txt
#   # 2. run the shard with -q, tee'ing to a log, then
#   python p2diag_shardmap.py ids.txt shard.log [substring-to-report]
import sys

_OUTCOME = {'.': 'PASSED', 'F': 'FAILED', 'E': 'ERROR', 's': 'skipped',
            'x': 'xfail', 'X': 'XPASS', 'u': 'xfail', 'U': 'XPASS'}


def parse(log_text):
    """Return the progress characters, in order, from a ``-q`` pytest log."""
    chars = []
    for line in log_text.splitlines():
        # progress lines are runs of outcome chars optionally ending in
        # ' [ NN%]'; anything else (headers, summaries, tracebacks) is skipped
        body = line.split('[')[0].rstrip() if line.rstrip().endswith(']') \
            else line.rstrip()
        if not body:
            continue
        if all(c in _OUTCOME for c in body):
            chars.extend(body)
    return chars


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    ids = [ln.strip() for ln in open(sys.argv[1]) if '::' in ln]
    chars = parse(open(sys.argv[2], errors='replace').read())
    print('collected %d node ids, %d outcomes so far (%.1f %%)'
          % (len(ids), len(chars), 100.0 * len(chars) / max(len(ids), 1)))
    bad = [(i, c) for i, c in enumerate(chars) if c in ('F', 'E')]
    print('non-passing so far: %d' % len(bad))
    for i, c in bad:
        print('  %-8s #%-5d %s' % (_OUTCOME[c], i + 1,
                                   ids[i] if i < len(ids) else '(past end)'))
    want = sys.argv[3] if len(sys.argv) > 3 else 'niche_p2_guards'
    print()
    print('reporting on %r:' % want)
    hit = False
    for i, nid in enumerate(ids):
        if want in nid:
            hit = True
            if i < len(chars):
                print('  #%-5d %-9s %s' % (i + 1, _OUTCOME.get(chars[i], chars[i]),
                                           nid))
            else:
                print('  #%-5d %-9s %s' % (i + 1, 'NOT YET', nid))
    if not hit:
        print('  (no node id matched)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
