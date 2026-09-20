"""VERIFY-WP-B12b -- pair probe W2's ``pre`` and ``post`` ladder JSONs for one
build and print the archive-to-archive table at full precision.

Usage::

    python validation/probe_verify_b12b/compare_w2.py <build_tag>

Author: VERIFY-WP-B12b
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def load(arm, tag):
    with open(os.path.join(_HERE, f'probe_w2_ladder_{arm}_{tag}.json'),
              encoding='cp1252') as fh:
        return json.load(fh)


def main(tag):
    pre, post = load('pre', tag), load('post', tag)
    print(f"PRE  {pre['env']['lumenairy_file']}")
    print(f"POST {post['env']['lumenairy_file']}")
    print(f"{'fixture':15s} {'plane':11s} {'pre fid':>12s} {'post fid':>12s} "
          f"{'pre relL2':>10s} {'post relL2':>10s} {'bytes':>10s} "
          f"{'pre sha':>26s} {'post sha':>26s}")
    out = []
    for a, b in zip(pre['rows'], post['rows']):
        assert a['key'] == b['key'] and a['plane'] == b['plane']
        same = a['sha'] == b['sha']
        out.append(dict(key=a['key'], plane=a['plane'], pre=a['fidelity'],
                        post=b['fidelity'], pre_l2=a['rel_l2'],
                        post_l2=b['rel_l2'], pre_sha=a['sha'],
                        post_sha=b['sha'], identical=same,
                        pre_peak=a['peak'], post_peak=b['peak'],
                        airy=a['airy'], dx_over_airy=a['dx_over_airy'],
                        oracle_src=a['oracle_src_halved_infid'],
                        oracle_grid=a['oracle_grid_halved_infid']))
        print(f"{a['key']:15s} {a['plane']:11s} {a['fidelity']:12.8f} "
              f"{b['fidelity']:12.8f} {a['rel_l2']:10.5f} {b['rel_l2']:10.5f} "
              f"{'IDENTICAL' if same else 'changed':>10s} {a['sha']:>26s} "
              f"{b['sha']:>26s}")
    print()
    for a, b in zip(pre.get('forced', []), post.get('forced', [])):
        print(f"FORCED-SURFACE {a['key']:15s} PRE shipped "
              f"{a['shipped']['fidelity']:.8f} forced "
              f"{a['forced_surface']['fidelity']:.8f} identical "
              f"{a['bit_identical']} | POST shipped "
              f"{b['shipped']['fidelity']:.8f} forced "
              f"{b['forced_surface']['fidelity']:.8f} relL2 "
              f"{b['forced_vs_shipped_rel_l2']:.4f} identical "
              f"{b['bit_identical']}")
        print(f"    cross-check: PRE shipped sha {a['shipped']['sha']} vs "
              f"POST forced sha {b['forced_surface']['sha']} -> "
              f"{'SAME BYTES' if a['shipped']['sha'] == b['forced_surface']['sha'] else 'different'}")
    print()
    for a, b in zip(pre['entries'], post['entries']):
        print(f"ENTRY {a['entry']:48s} pre sha {a['sha']} post sha {b['sha']} "
              f"{'IDENTICAL' if a['sha'] == b['sha'] else 'changed':>10s}"
              + (f"  same_bytes_as_gbd pre={a.get('same_bytes_as_gbd')} "
                 f"post={b.get('same_bytes_as_gbd')}"
                 if 'same_bytes_as_gbd' in a else ''))
    with open(os.path.join(_HERE, f'compare_w2_{tag}.json'), 'w',
              encoding='cp1252') as fh:
        json.dump(dict(tag=tag, rows=out,
                       pre_file=pre['env']['lumenairy_file'],
                       post_file=post['env']['lumenairy_file'],
                       pre_forced=pre.get('forced'),
                       post_forced=post.get('forced'),
                       pre_entries=pre['entries'],
                       post_entries=post['entries']), fh, indent=1)
    print('WROTE', os.path.join(_HERE, f'compare_w2_{tag}.json'))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'win32_314')
