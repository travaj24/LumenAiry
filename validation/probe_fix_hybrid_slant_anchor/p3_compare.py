"""P3b -- diff two ``p3_bit_identity`` JSONs, hash by hash.

Usage (from this directory)::

    python p3_compare.py results/p3_bit_identity_post_fix_win.json \
                         results/p3_bit_identity_pre_base_win.json

Prints one line per differing hash and a verdict.  The ONLY rows allowed to
differ are ``slanted_patterned_*`` / ``slanted_tensor_*`` on the
``jones_transmission`` and ``per_order_transmission`` keys.
"""
import json
import sys

ALLOWED_KEYS = ("jones_transmission", "per_order_transmission")


def main(a_path, b_path):
    a = json.loads(open(a_path, encoding="cp1252").read())
    b = json.loads(open(b_path, encoding="cp1252").read())
    print(f"A = {a['_env']['lumenairy']}  ({a['_env']['arm']}/"
          f"{a['_env']['stage']}/{a['_env']['tag']})")
    print(f"B = {b['_env']['lumenairy']}  ({b['_env']['arm']}/"
          f"{b['_env']['stage']}/{b['_env']['tag']})")
    rows = sorted(set(a) & set(b) - {"_env"})
    n_diff = n_same = 0
    unexpected = []
    for r in rows:
        for k in sorted(set(a[r]) & set(b[r])):
            if not isinstance(a[r][k], str):
                continue
            if a[r][k] == b[r][k]:
                n_same += 1
                continue
            n_diff += 1
            ok = (r.startswith(("slanted_patterned", "slanted_tensor"))
                  and k in ALLOWED_KEYS)
            print(f"  {'EXPECTED' if ok else 'UNEXPECTED'}  {r}.{k}: "
                  f"{a[r][k][:16]} vs {b[r][k][:16]}")
            if not ok:
                unexpected.append(f"{r}.{k}")
    print(f"rows {len(rows)}  hashes identical {n_same}  differing {n_diff}  "
          f"unexpected {len(unexpected)}")
    return 1 if unexpected else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
