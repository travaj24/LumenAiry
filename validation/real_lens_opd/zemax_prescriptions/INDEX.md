# Zemax-compatible prescriptions for validation cases

Each case has two files:
- `<name>.txt` -- human-readable lens-data table for manual entry into Zemax's Lens Data Editor.
- `<name>.zmx` -- minimal Zemax sequential file for File > Open.  After loading, verify the wavelength, aperture type, and stop index match what is written in the `.txt` header.

| Case | Wavelength [um] | Aperture [mm] | EFL [mm] | BFL [mm] |
|---|---:|---:|---:|---:|
| `plano_convex_R50_BK7` | 1.3100 | 10.00 | 99.29 | 96.63 |
| `equi_convex_R50_BK7` | 1.3100 | 10.00 | 50.49 | 48.80 |
| `biconcave_R50_BK7` | 1.3100 | 10.00 | -49.23 | -50.06 |
| `meniscus_positive_BK7` | 1.3100 | 10.00 | 115.29 | 111.42 |
| `meniscus_negative_BK7` | 1.3100 | 10.00 | -123.27 | -121.21 |
| `plano_convex_R50_SF6` | 1.3100 | 10.00 | 65.17 | 62.90 |
| `plano_convex_thick_BK7` | 1.3100 | 10.00 | 99.29 | 92.64 |
| `LA1050_C` | 1.3100 | 20.00 | 102.27 | 99.54 |
| `LA1509_C` | 1.3100 | 20.00 | 205.11 | 202.72 |
| `LA1301_C` | 1.3100 | 20.00 | 256.56 | 254.30 |
| `AC254_050_C` | 1.3100 | 20.00 | 45.11 | 39.14 |
| `AC254_100_C` | 1.3100 | 20.00 | 84.14 | 80.03 |
| `AC254_200_C` | 1.3100 | 20.00 | 139.19 | 136.50 |
| `fnum_sweep_f20_R200` | 1.3100 | 8.00 | 397.15 | 395.16 |
| `fnum_sweep_f10_R100` | 1.3100 | 8.00 | 198.58 | 196.58 |
| `fnum_sweep_f5_R50` | 1.3100 | 8.00 | 99.29 | 97.29 |
| `fnum_sweep_f3_R30` | 1.3100 | 8.00 | 59.57 | 57.58 |
| `fnum_sweep_f2_R20` | 1.3100 | 8.00 | 39.72 | 37.72 |
| `AC254_100_C_1064nm` | 1.0640 | 20.00 | 83.82 | 79.71 |
| `AC254_100_C_1310nm` | 1.3100 | 20.00 | 84.14 | 80.03 |
| `AC254_100_C_1550nm` | 1.5500 | 20.00 | 84.49 | 80.38 |
