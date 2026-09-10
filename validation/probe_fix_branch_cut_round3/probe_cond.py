import probe_sign_census as P
import numpy as np, os, sys
print(f"# CORETYPE={os.environ.get('OPENBLAS_CORETYPE','-')}")
for nm, fn in (("A spacer COINCIDENT", P.census_spacer),
               ("A spacer DETUNED   ", P.census_spacer_detuned),
               ("B thin ladder TE   ", P.census_thin)):
    for arm, body in (("POST", None), ("PRE ", P._pre_body)):
        rc, n = P.rcond_census(fn, body)
        c = P.__dict__  # keep linters quiet
        print(f"  {nm} {arm}: worst rcond(a+b) {rc:.4e} over {n} inverses")
