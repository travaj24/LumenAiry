"""Compare p4_bytes PRE vs POST (byte equality per entry) and the traced
'li' route against NumPy 'li' (POST).  python p4_compare.py <build>"""
import json
import sys

import numpy as np

b = sys.argv[1]
pre = dict(np.load(f"p4_bytes_pre_{b}.npz"))
post = dict(np.load(f"p4_bytes_post_{b}.npz"))
npre = json.load(open(f"p4_notes_pre_{b}.json"))
npost = json.load(open(f"p4_notes_post_{b}.json"))
same, diff = [], {}
for k in sorted(set(pre) | set(post)):
    if k not in pre or k not in post:
        diff[k] = "missing in " + ("pre" if k not in pre else "post")
        continue
    if pre[k].tobytes() == post[k].tobytes():
        same.append(k)
    else:
        diff[k] = float(np.max(np.abs(pre[k] - post[k]))
                        / max(np.max(np.abs(post[k])), 1e-300))
li = {}
for kind in ("iso", "aniso", "tilt"):
    for th in ("0.0", "0.2"):
        base = f"jones2d|{kind}|li|{th}|"
        lau = f"jones2d|{kind}|laurent|{th}|"
        if base + "np" in post:
            ref = post[base + "np"]
            li[f"{kind}/{th}"] = dict(
                jit_vs_np_post=float(np.max(np.abs(post[base + "jit"] - ref))),
                jit_vs_np_pre=float(np.max(np.abs(pre[base + "jit"] - ref))),
                eager_vs_np_post=float(np.max(np.abs(post[base + "jax"]
                                                     - ref))),
                np_li_vs_laurent=float(np.max(np.abs(post[lau + "np"]
                                                     - ref))))
notes_diff = {k: (npre.get(k), npost.get(k)) for k in set(npre) | set(npost)
              if npre.get(k) != npost.get(k)}
li_notes = {k: v for k, v in npost.items() if "|li|" in k.replace("/", "|")
            and v}
out = dict(n_same=len(same), n_total=len(set(pre) | set(post)), diff=diff,
           li=li, notes_diff=notes_diff, li_notes_post=li_notes)
print(json.dumps(out, indent=1))
json.dump(out, open(f"p4_compare_{b}.json", "w"), indent=1)
