# Lumenairy examples

Five short, runnable scripts that walk through the library's core
workflows from simplest to most advanced.  Each one is independent and
prints diagnostic output -- no plotting required.

| File | What it shows |
|------|---------------|
| `01_basic_propagation.py` | Build a Source, propagate through free space and a singlet, read out a Strehl ratio. |
| `02_design_optimization.py` | Drive `design_optimize` with a focal-length + Strehl merit on an N-BK7 singlet. |
| `03_high_fidelity_wave.py` | Compare ASM vs GBD vs HF as the wave-leg propagator on the same prescription. |
| `04_jax_differentiable.py` | JaxMeritTerm + `jac='auto'` -- analytic JAX gradients flowing into SciPy. |
| `05_mhs_pipeline_with_replay.py` | MhsPipeline.from_prescription with checkpoint logging + `replay_run` to read the planes back. |

To run:

```bash
python examples/01_basic_propagation.py
```

These are not part of the validation suite -- the goal is to read like
a tutorial.  Each script is < 100 lines.
