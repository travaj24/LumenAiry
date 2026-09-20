"""WP-C2 ROUND 3, VR2-D1 -- compare a PRE and a POST run of
``r3_apply_real_lens_wayback.py`` and print the byte-identity table.

Usage:  python r3_wayback_compare.py <pre.json> <post.json> [out.json]
"""
import json
import pathlib
import sys


def main(pre_path, post_path, out_path=None):
    pre = json.loads(pathlib.Path(pre_path).read_text(encoding='utf-8'))
    post = json.loads(pathlib.Path(post_path).read_text(encoding='utf-8'))
    assert pre['tag'] == 'pre' and post['tag'] == 'post', (pre['tag'],
                                                           post['tag'])
    assert not pre['apply_real_lens_carries_the_pair'], (
        'the PRE tree already carries the pair, so this comparison has no '
        'before side')
    assert post['apply_real_lens_carries_the_pair']
    assert pre['field_sha256'] == post['field_sha256'], (
        'the two runs were handed different input fields')

    rows = []
    n_same = n_moved = n_control = 0
    for key in sorted(pre['digests']):
        a, b = pre['digests'][key], post['digests'][key]
        way_back = (a['seidel'] == b['seidel_old'])
        moved = (a['seidel'] != b['seidel'])
        none_is_omitted = (b['seidel'] == b['seidel_none'])
        control_same = (a['control'] == b['control'] == b['control_kw'])
        n_same += int(way_back)
        n_moved += int(moved)
        n_control += int(control_same)
        rows.append({
            'prescription': key,
            'pre_seidel': a['seidel'][:8],
            'post_seidel_default': b['seidel'][:8],
            'post_seidel_old_keywords': b['seidel_old'][:8],
            'way_back_is_byte_identical': way_back,
            'default_moved': moved,
            'none_equals_omitted': none_is_omitted,
            'seidel_false_control_identical_with_and_without_keywords':
                control_same,
            'pre_control': a['control'][:8],
            'post_control': b['control'][:8],
            'post_control_with_old_keywords': b['control_kw'][:8],
        })
    out = {
        'pre': {'lumenairy_file': pre['lumenairy_file'],
                'python': pre['python'], 'numpy': pre['numpy']},
        'post': {'lumenairy_file': post['lumenairy_file'],
                 'python': post['python'], 'numpy': post['numpy'],
                 'signature_defaults': post['signature_defaults']},
        'n_prescriptions': len(rows),
        'n_way_back_identical': n_same,
        'n_default_moved': n_moved,
        'n_control_identical': n_control,
        'rows': rows,
    }
    text = json.dumps(out, indent=2, sort_keys=True)
    if out_path:
        pathlib.Path(out_path).write_text(text, encoding='utf-8')
    print(text)
    return 0 if (n_same == n_moved == n_control == len(rows)) else 1


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
