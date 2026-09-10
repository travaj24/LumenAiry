"""Line-based splice of measured per-test durations into ``.test_durations``.

Follows CHORE_TEST_HYGIENE_2026_08_16 (d)/(e): every RETAINED entry's
``"key": value`` text is copied verbatim, so the only lines that differ are
the ones added and removed for the files measured.  The result keeps the
committed file's own formatting (2-space indent, sorted keys, LF).

Usage:  python splice_durations.py <measured.json> [<measured.json> ...]
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
TARGET = os.path.join(ROOT, '.test_durations')


def main():
    measured = {}
    for p in sys.argv[1:]:
        with open(p, encoding='utf-8') as fh:
            measured.update(json.load(fh))
    if not measured:
        raise SystemExit('no measured durations given')
    touched_files = sorted({k.split('::', 1)[0] for k in measured})
    print('measured %d node ids over %d files' % (len(measured),
                                                  len(touched_files)))

    with open(TARGET, encoding='utf-8', newline='') as fh:
        raw = fh.read()
    nl = '\r\n' if '\r\n' in raw else '\n'
    lines = raw.split(nl)
    assert lines[0] == '{', lines[0]
    body = [ln for ln in lines[1:] if ln.strip() not in ('}', '')]

    dec = json.JSONDecoder()
    kept = []
    dropped = 0
    for ln in body:
        # the key can contain ': ' (parametrized ids), so decode it as JSON
        # from the start of the line rather than splitting on a separator.
        key, _ = dec.raw_decode(ln.strip())
        if key.split('::', 1)[0] in touched_files:
            dropped += 1
            continue
        kept.append((key, ln.strip().rstrip(',')))
    added = [(k, '%s: %r' % (json.dumps(k), float(v)))
             for k, v in measured.items()]
    allrows = sorted(kept + added, key=lambda kv: kv[0])
    out = ['{']
    for i, (_, text) in enumerate(allrows):
        out.append('  ' + text + (',' if i + 1 < len(allrows) else ''))
    out.append('}')
    out.append('')
    with open(TARGET, 'w', encoding='utf-8', newline='') as fh:
        fh.write(nl.join(out))
    print('retained %d, removed %d, added %d -> %d entries'
          % (len(kept), dropped, len(added), len(allrows)))

    # verification: retained entries byte-identical, JSON still parses
    with open(TARGET, encoding='utf-8', newline='') as fh:
        new_raw = fh.read()
    new_lines = set(l.strip().rstrip(',') for l in new_raw.split(nl))
    bad = [k for k, t in kept if t not in new_lines]
    assert not bad, bad[:5]
    d = json.loads(new_raw)
    print('re-parsed %d entries; %d retained lines byte-identical'
          % (len(d), len(kept)))


if __name__ == '__main__':
    main()
