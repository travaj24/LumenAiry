"""Compare the v1_bytes_*.json key sets: pre vs post, pre vs posttrap, and
pre vs postchitrap on the keys each produced."""
import sys

import _ve1common as V

labs = sys.argv[1:] or ["pre", "post", "posttrap", "postchitrap"]
D = {lab: V.load(f"v1_bytes_{lab}.json") for lab in labs}
ref = D[labs[0]]["keys"]
out = {"ref": labs[0], "n_ref": len(ref), "cmp": {}}
for lab in labs[1:]:
    k = D[lab]["keys"]
    common = sorted(set(ref) & set(k))
    diff = [x for x in common if ref[x] != k[x]]
    out["cmp"][lab] = {"produced": len(k), "common": len(common),
                       "identical": len(common) - len(diff), "differ": diff,
                       "missing": sorted(set(ref) - set(k)),
                       "errors": D[lab]["errors"]}
    print(lab, "produced", len(k), "common", len(common), "identical",
          len(common) - len(diff), "missing", len(set(ref) - set(k)))
V.dump(f"v1_compare_{'_'.join(labs)}.json", out)
