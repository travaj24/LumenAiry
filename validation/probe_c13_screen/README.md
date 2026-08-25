# probe_c13_screen -- ITEM 2 of `docs/audits/FIX_P4_TRACED_CLOSEOUT_2026_08_24.md`

Is `_LSTSQ_GRAM_RCOND_MIN = 1e-8` the right screen for the traced fits?

The captured design matrices (`fits/*.npz`, ~300 MB) are NOT committed -- they
are regenerated in a couple of minutes by `capture.py`, and the numbers derived
from them are in `fits/adjudication.json` and the `ladder_*.json` files.

```sh
export LUMENAIRY_ROOT=/path/to/this/checkout
for f in singlet singlet_nosub singlet_big biconcave fast_decentred; do
    python capture.py $f fits
done
python adjudicate.py fits                 # the 39-solve table (S2.2 / S2.4)
python post.py                            # adds r*/||b|| to it
mkdir -p fits_small && cp fits/singlet_nosub_0[123].npz fits_small/
python adjudicate.py fits_small --mp      # + the mpmath oracle cross-check (S2.1)
python branch.py fits_small               # which solver branch fires (S2.3)
python basis.py fits/singlet_nosub_03.npz # what conditioning would buy (S2.2)
python ladder.py fits/singlet_nosub_03.npz ladder_m28.json    # the derived bar (S2.5)
python ladder.py fits/singlet_nosub_04.npz ladder_m120.json
```

`ladder_m28_wsl.json` is the same ladder on the second build (WSL py3.12.3 /
numpy 2.4.6 / scipy 1.17.1) and agrees with `ladder_m28.json` to every printed
digit.
