"""VERIFY-WP-C2 ROUND 2, item D7 -- abuse the re-anchoring override.

The round-1 defect was that ``EDITED_IN_PLACE`` compared only the text
BEFORE the first ``=``, so it re-anchored a citation whose CLAIM had become
false (the default silently reverted, set to nonsense, or left behind as a
stale copy).  Round 2 pins a SHA-256 of the expected line content and the
release the entry was recorded for.

This probe abuses the new guard on its own terms.  Nothing on disk is
edited: the script's own line cache is doctored, which is exactly the file
content the tool would have read, and the tool's real
``_edited_in_place`` is then called.

Cases, each reported with the override's answer and the refusal record:

    1  shipped        the real content at the mapped line          -> FIRE
    2  reverted       default silently back to 'generic'           -> refuse
    3  nonsense       default set to an unknown route              -> refuse
    4  unrelated      a different declaration at the line          -> refuse
    5  truncated      the file too short for the coordinate        -> refuse
    6  stale-copy     the declaration MOVED and the OLD-content
                      copy left at the mapped line                 -> refuse
    7  STALE-COPY-NEW the declaration MOVED and a copy with the
                      EXACT EXPECTED CONTENT left at the mapped
                      line, so the digest still matches            -> ?
    8  reindented     the same content, indented differently       -> FIRE
                      (documented: the digest strips whitespace)
    9  version-ahead  the package already past the recorded
                      release                                      -> refuse
   10  version-equal  the package exactly at the recorded release  -> FIRE
   11  digest-rerecorded  a REVERTED default whose digest was
                      re-recorded to match it (an attacker with
                      commit access to the map)                    -> ?

Case 7 and case 11 are the two this verification wanted a number on.

Usage:  python vr2_reanchor_abuse.py <out.json>
"""
import importlib.util
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
REPO = HERE.parents[2]


def load_tool():
    spec = importlib.util.spec_from_file_location(
        'vr2_reanchor_tool', REPO / 'scripts' / 'reanchor_citations.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(out_path):
    T = load_tool()
    path = 'lumenairy/raytrace/trace.py'
    base_num = 61
    base = 'f4f18851'

    real_lines = list(T.lines(path))
    want, _ctx = T.base_line(path, base_num, base)
    shipped = real_lines[base_num - 1]
    print('base line       :', repr(want))
    print('shipped line    :', repr(shipped))
    print('expected digest :', T.EDITED_IN_PLACE[(path, base_num)][2])
    print('content digest  :', T.content_digest(shipped))

    def run(label, mutate, version=None):
        T._cache.clear()
        T.EDITED_IN_PLACE_REFUSALS.clear()
        hay = list(real_lines)
        hay = mutate(hay)
        T._cache[(path, None)] = hay
        T._cache[(path, base)] = list(T.lines(path, base))
        real_ver = T._source_version
        if version is not None:
            T._source_version = lambda: version
        try:
            num, how = T._edited_in_place(path, base_num, base)
        finally:
            T._source_version = real_ver
        ref = list(T.EDITED_IN_PLACE_REFUSALS)
        return {'label': label, 'fired': num is not None,
                'new_num': num, 'how': how,
                'n_refusals': len(ref),
                'why': ref[0]['why'] if ref else None,
                'refusal_prints_both_lines': bool(
                    ref and ref[0].get('base_line')
                    and ref[0].get('found_line')),
                'found_line': ref[0]['found_line'] if ref else None}

    cases = []
    cases.append(run('1 shipped', lambda h: h))
    cases.append(run('2 reverted', lambda h: _sub(h, base_num,
                                                  "sphere_normal: str = 'generic',")))
    cases.append(run('3 nonsense', lambda h: _sub(h, base_num,
                                                  "sphere_normal: str = 'not-a-route',")))
    cases.append(run('4 unrelated', lambda h: _sub(h, base_num,
                                                   "renormalize: str = 'exit',")))
    cases.append(run('5 truncated', lambda h: h[:30]))

    def _stale_old(h):
        h = list(h)
        # the declaration MOVES three lines down, an OLD-content copy stays
        h.insert(base_num + 2, shipped)
        h[base_num - 1] = "    sphere_normal: str = 'generic',"
        return h
    cases.append(run('6 stale-copy-old', _stale_old))

    def _stale_new(h):
        h = list(h)
        # the REAL declaration moves elsewhere and a copy with the EXACT
        # expected content is left at the mapped coordinate
        h.insert(base_num + 40, shipped)
        return h
    cases.append(run('7 stale-copy-with-expected-content', _stale_new))

    cases.append(run('8 reindented',
                     lambda h: _sub(h, base_num,
                                    '        ' + shipped.strip())))
    def _wrong_owner(h):
        h = list(h)
        # the citation's own function loses the parameter entirely; a line
        # with the EXACT expected text now belongs to a different (private)
        # helper that happens to sit at the mapped coordinate.
        h[base_num - 1] = shipped          # same text, different owner
        for i, ln in enumerate(h[:base_num - 1][::-1]):
            if ln.startswith('def '):
                h[base_num - 2 - i] = 'def _trace_debug_shim('
                break
        return h
    cases.append(run('12 same content, different enclosing def',
                     _wrong_owner))

    cases.append(run('9 version-ahead', lambda h: h, version='5.50.0'))
    cases.append(run('10 version-equal', lambda h: h, version='5.49.0'))

    # 11: the digest itself re-recorded to match a reverted default
    T._cache.clear()
    T.EDITED_IN_PLACE_REFUSALS.clear()
    reverted = "    sphere_normal: str = 'generic',"
    saved = T.EDITED_IN_PLACE[(path, base_num)]
    T.EDITED_IN_PLACE[(path, base_num)] = (
        saved[0], saved[1], T.content_digest(reverted), saved[3])
    hay = _sub(list(real_lines), base_num, reverted)
    T._cache[(path, None)] = hay
    T._cache[(path, base)] = list(T.lines(path, base))
    num, how = T._edited_in_place(path, base_num, base)
    T.EDITED_IN_PLACE[(path, base_num)] = saved
    cases.append({'label': '11 digest re-recorded for a reverted default',
                  'fired': num is not None, 'new_num': num, 'how': how,
                  'n_refusals': len(T.EDITED_IN_PLACE_REFUSALS),
                  'why': None, 'refusal_prints_both_lines': False,
                  'found_line': reverted.strip()})

    out = {'python': sys.version.split()[0], 'repo': str(REPO),
           'source_version': T._source_version(),
           'shipped_line': shipped.strip(),
           'expected_digest': T.EDITED_IN_PLACE[(path, base_num)][2],
           'n_map_entries': len(T.EDITED_IN_PLACE),
           'cases': cases}
    pathlib.Path(out_path).write_text(json.dumps(out, indent=1),
                                      encoding='utf-8')
    print()
    for c in cases:
        print('%-46s fired=%-6s refusals=%d  %s'
              % (c['label'], c['fired'], c['n_refusals'],
                 (c['why'] or '')[:70]))
    print('wrote', out_path)


def _sub(h, num, text):
    h = list(h)
    h[num - 1] = text
    return h


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'vr2_reanchor_abuse.json')
