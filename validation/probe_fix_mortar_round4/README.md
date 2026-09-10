# probe_fix_mortar_round4

Measurements behind `docs/audits/FIX_PMM2D_MORTAR_ROUND4_2026_09_11.md` --
round 4 of the PURE staggered 2-D PMM per-layer L2 mortar work, closing the two
P3 defects and the two durability flags of
`docs/audits/VERIFY_PMM2D_MORTAR_ROUND3_2026_09_11.md`.

| file | what it measures |
|---|---|
| `_path.py` | pins THIS worktree's `lumenairy` on `sys.path` and REFUSES to run if the import resolves anywhere else.  A probe run as `python validation/probe_.../x.py` gets the SCRIPT's directory as `sys.path[0]`, not the working directory -- the trap the round-3 verification recorded, which cost it one probe run against a checkout on `D:`. |
| `p1_axis_band.py` | DEFECT 2, the per-axis band warning, in FOUR parts: (a) the no-mortar-axis fixture, (b) the warnings that fire for a real reason, (c) the mixed case, (d) the ordinary-geometry census.  Runs BOTH arms -- round 3's collapsed stack-level rule, restored in process, and round 4's per-axis rule -- in ONE interpreter, then compares every leaf.  Answer hashes must be IDENTICAL; only warning fields may move. |
| `p2_durability.py` | S14: part `p` is the near-null PARTICIPATION population over every generalized-mortar class the two fixture families build (33 operands), part `l` is the degradation LADDER on the round-3 gate's own fixture at `M` = 5..8 against a degree-12/14 1-D `PMMStack` oracle. |

## Running

Every command starts with the `cd`, and pins ONE BLAS thread on the COMMAND
LINE -- a `os.environ.setdefault` inside a module is a no-op once numpy is
imported, and a pytest launched from elsewhere prints "no tests ran in 0.02s"
and exits 0.

```sh
# Windows (py3.14, OpenBLAS Haswell)
cd /c/tmp/lum_mortar4 && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 python validation/probe_fix_mortar_round4/p1_axis_band.py \
  --tag win --parts abcd
cd /c/tmp/lum_mortar4 && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 python validation/probe_fix_mortar_round4/p2_durability.py \
  --tag win --parts p
cd /c/tmp/lum_mortar4 && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 python validation/probe_fix_mortar_round4/p2_durability.py \
  --tag win_ladder --parts l --ladder-M 5,6,7,8

# WSL (py3.12, OpenBLAS SkylakeX) -- same, through the venv
wsl -e bash -lc 'cd /mnt/c/tmp/lum_mortar4 && OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ~/lumvenv/bin/python \
  validation/probe_fix_mortar_round4/p1_axis_band.py --tag wsl --parts abcd'
```

## Units

`p1`'s band fixtures are DIMENSIONLESS (period 1.07, wavelength 0.79) and
reproduce the round-3 verification's DEFECT-2 fixture knob for knob; the
round-2 battery it reuses is period 1.2 / wavelength 0.85.  `p2`'s
participation fixtures come from `probe_verify_mortar_round3/_vfix.py` and are
in METRES (period 0.87e-6).  The period and the wavelength must be in the SAME
units; nothing else in these fixtures is dimensional.

## Cost

`p1 --parts abcd` is about 14 minutes a build, and part `d` is nearly all of
it: it SOLVES the eight ordinary census geometries closest to the band edge,
twice (once per arm), and an 8-cell uniform lattice at `M` = 4 is a 1152-
dimension dense region eigenproblem -- 200 s a solve.  `--parts abc` is under
two minutes and carries the whole DEFECT-2 argument; part `d` is the census.
`p2 --parts l` at `M` = 8 is ~10 minutes a rung.
