# Spin-polarized currents through bends and junctions of V2Se2O-family altermagnet nanoribbons: code and data

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24238844.svg)](https://doi.org/10.5281/zenodo.24238844)

This repository holds the code and data behind the paper

> T. Rojas, *Design rules for spin-polarized currents through bends and junctions of V2Se2O-family altermagnet
> nanoribbons* (submitted to Physics Letters A).

It contains:
- the tight-binding model, built from the four-band model of Cheng et al., Phys. Rev. Materials 10, 114407 (2026),
  doi:[10.1103/8czs-74vv](https://doi.org/10.1103/8czs-74vv);
- the Kwant transport calculations, the two-current resistor network and the Boltzmann, Kubo–Greenwood and Elliott
  estimates;
- the disorder ensembles;
- every output file the paper's numbers come from;
- the scripts that draw all figures.

## Layout

```
calc/        all scripts (run from the repository root)
data/        outputs: CSV, JSON, JSONL and NPY files, with the run logs
figures/     complete per-stage figures (s0-s5); figures/ovito/ holds the device renders of Fig. 1(c)
matplotlibrc figure defaults (vector PDFs with embedded TrueType fonts)
```

Shared modules:

| Module | What it is |
|---|---|
| `calc/v2x2o_tb.py` | Model: the four decoupled spin × orbital bands of Cheng et al. (Table 1, KV2Se2O, no spin–orbit coupling), the single-band reference and the isotropic reference; channel counts (Sharvin and the k∥-conserving count, Eq. (2)); [100] and [110] lead modes |
| `calc/network2c.py` | Two-current resistor network with floating probes and an on-site spin-relaxation conductance |
| `calc/soc_corner.py` | Two-sublattice Kwant builder with the Table 2 spin–orbit terms inside the corner of an L-bend |

## Pipeline

The stages build on each other. Every output is already in `data/`, so any later stage can be rerun alone.

| Stage | Script(s) | Writes | Paper |
|---|---|---|---|
| S0 model validation | `s0_digitize_cheng_fig2e.py` (needs the published PDF of Cheng et al., not included), `s0_validation.py` | `s0_cheng_fig2e_digitized.csv`, `s0_validation.json` | Sec. 2.1, Fig. 1(a), SI S1 |
| S1 channel counting | `s1_channels.py`, `s1_analysis.py` | `s1_kwant_modes.csv`, `s1_fermi_surface_counts.csv`, `s1_summary.json` | Sec. 3.1, Fig. 2, SI S2 |
| S2 corner efficiency | `s2_corners.py`, `s2_analysis.py`, `s2b_holdout.py`, `s2c_si_numbers.py` | `s2_corners.csv`, `s2_soc.csv`, `s2_summary.json`, `s2b_holdout.csv`, `s2b_summary.json`, `s2c_si_numbers.json` | Sec. 3.2, Fig. 3, Table S1, SI S3 |
| S3 diffusive rules | `s3_diffusive.py`, `s3_analysis.py`, `s3b_lambda.py`, `s3c_elliott.py`, `s3d_kubo.py`, `s3e_hotspots.py`, `s3f_lambda_correction.py` | `s3_boltzmann.csv`, `s3_network.csv`, `s3_summary.json`, `s3_spin_accumulation.npy`, `s3b_lambda.csv`, `s3d_kubo.json`, `s3e_hotspots.json`, `s3f_lambda_correction.json` | Secs. 2.3 and 3.3, Fig. 4, SI S4 |
| S4 readout | `s4_readout.py`, `s4_analysis.py`, `s4b_ishe.py` | `s4_diffusive.csv`, `s4_ballistic.csv`, `s4_summary.json`, `s4_mu_s_with_probes.npy`, `s4b_ishe.json` | Sec. 3.4, Fig. 5(a,b), SI S5 |
| S5 disorder | `s5_disorder.py` (checkpointed; the longest run), `s5_analysis.py` | `s5_results.jsonl`, `s5_summary.json` | Sec. 3.5, Fig. 5(c), Table S2, SI S6 |
| Figures | `paper_figures.py` (Figs. 1–5), `si_figures.py` (Figs. S1–S6), `figures/ovito/render_devices.py` (Fig. 1(c) renders) | `manuscript/figures/` (created on first run) | all figures |

Table 2 collects numbers from `s1_summary.json`, `s2_summary.json`, `s3_summary.json`, `s3f_lambda_correction.json`
and `s4b_ishe.json`.

## Environment

The calculations were run with:
- Python 3.12, NumPy 1.26, SciPy 1.17, Matplotlib 3.11, Kwant 1.5.0;
- PyMuPDF, only for the S0 digitization;
- OVITO 3.17.2, only for the Fig. 1(c) renders (`bash figures/ovito/render.sh figures/ovito/render_devices.py`).

```bash
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
export OMP_NUM_THREADS=1          # threaded BLAS slows the many small Kwant solves by orders of magnitude
python calc/s1_channels.py        # e.g. one stage; run from the repository root
```

Kwant ran with SciPy's sparse solver (no MUMPS). A standalone [110] lead does not finalize in Kwant 1.5, so the [110]
leads are attached to a short scattering region (`v2x2o_tb.lead_modes_110`). `s5_disorder.py` uses at most two worker
processes, checks free memory and checkpoints every task, so an interrupted run resumes where it stopped.

## Conventions

- **Energies:** in eV, measured from the first-principles Fermi level. The Fermi level matched to photoemission
  is E_F = −0.09 eV.
- **Lengths:** in lattice sites (a = 0.3952 nm), or in atomic rows along [110]. Network cells are about 1 nm.
- **Spins:** spin up lives on sublattice A (d_xz, d_xy) and spin down on sublattice B (d_yz, d_xy), with
  x ↔ y between them.
- **Polarization:** P = (T↑ − T↓)/(T↑ + T↓).

## License and citation

The code is released under the MIT License (`LICENSE`) and the data under CC BY 4.0. If you use them, please cite
the paper and this archive: doi:[10.5281/zenodo.24238844](https://doi.org/10.5281/zenodo.24238844) (all versions;
v1.0.0 is [10.5281/zenodo.24238845](https://doi.org/10.5281/zenodo.24238845)).
