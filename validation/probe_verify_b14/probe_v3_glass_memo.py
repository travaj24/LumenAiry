"""VERIFY-B14 arm 3 -- the glass validity one-shot pin, without pytest.

Shows the coupled-state mechanism directly: ``get_glass_index`` memoises the
``(name, wavelength)`` evaluation and returns BEFORE the validity warning, so
clearing ``_validity_warned`` alone leaves a state in which the warn-once set
says "not warned" and no warning can ever be produced.  Also answers the
question the WP report's section 3.1 asserts an answer to -- whether the
poison needs a SECOND FILE at all.
"""
from __future__ import annotations

import json
import sys
import warnings

import lumenairy as la
from lumenairy import glass as G


def _count(n=5, name='N-BK7', wl=200e-9):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for _ in range(n):
            la.get_glass_index(name, wl)
    return len([w for w in caught
                if issubclass(w.category, UserWarning)
                and 'validity' in str(w.message)])


def main():
    out = {'lumenairy_file': la.__file__, 'version': la.__version__,
           'python': sys.version.split()[0], 'steps': []}

    def step(label, **kw):
        out['steps'].append(dict(label=label, **kw))
        print(label, kw, flush=True)

    # 1. cold process: the one-shot claim holds.
    step('cold_5_calls', warnings_seen=_count(),
         value_cache=len(G._glass_value_cache),
         warned=len(G._validity_warned))

    # 2. PARTIAL reset (what the pre-fix autouse fixture did).
    G._validity_warned.clear()
    step('after_partial_reset', warnings_seen=_count(),
         value_cache=len(G._glass_value_cache),
         warned=len(G._validity_warned))

    # 3. FULL drain through the registered drain.
    la.clear_asm_caches()
    G._validity_warned.clear()
    step('after_full_drain', warnings_seen=_count(),
         value_cache_after_drain_was=0,
         warned=len(G._validity_warned))

    # 4. Does clear_asm_caches() really empty BOTH?
    _count(1)
    before = (len(G._glass_value_cache), len(G._validity_warned))
    la.clear_asm_caches()
    after = (len(G._glass_value_cache), len(G._validity_warned))
    step('clear_asm_caches_empties_both', before=before, after=after)

    with open(sys.argv[1] if len(sys.argv) > 1 else 'v3_glass_memo.json',
              'w', encoding='cp1252') as fh:
        json.dump(out, fh, indent=1)


if __name__ == '__main__':
    main()
