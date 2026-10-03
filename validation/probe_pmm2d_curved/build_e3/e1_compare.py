"""Compare e1_bytes_pre.json and e1_bytes_post.json (gate E3-1)."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
a = json.load(open(os.path.join(HERE, "e1_bytes_pre.json")))["sha"]
b = json.load(open(os.path.join(HERE, "e1_bytes_post.json")))["sha"]
keys = sorted(set(a) | set(b))
diff = [k for k in keys if a.get(k) != b.get(k)]
out = {"n_pre": len(a), "n_post": len(b), "n_equal": len(keys) - len(diff),
       "differ": diff}
json.dump(out, open(os.path.join(HERE, "e1_compare.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
sys.exit(1 if diff else 0)
