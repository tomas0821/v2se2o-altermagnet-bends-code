"""S1 channel counting (2026-11-14).

N_sigma(E, W) along [100] and [110]: Fermi-surface (wide-limit) predictions against Kwant lead mode counts.
  - widths 5-20 nm: [100] W = 13, 19, 25, 38, 51 sites; [110] Wd = 18, 27, 36, 54, 72 atomic rows
    (perpendicular width Wd a / sqrt2 = 5.0 ... 20.1 nm), a = 0.3952 nm;
  - E = -0.25 ... +0.10 eV in 5 meV steps (E_F = -0.09 +- 0.15 eV and the -0.15 ... +0.10 sweep);
  - orbital-resolved (the bands decouple without SOC), so the [110] d_xz sheets and d_xy pockets are separated;
  - pocket-shift error bar: d_xy on-site +0.08 eV (TB pocket top +0.13 eV vs DFT +0.21 eV, S0);
  - the single-band fit (arXiv:2602.11694) on the same footing.
Two wide-limit predictions per band (see v2x2o_tb.wide_limit_counts): the bulk Sharvin count, and the
k_par-conserving 'projected' count. They differ when edges retro-reflect carriers, which is what happens to the
quasi-1D sheets in a [110] strip.

Run (project root): OMP_NUM_THREADS=1 ~/venvs/crn-te/bin/python calc/s1_channels.py
Outputs: data/s1_kwant_modes.csv, data/s1_fermi_surface_counts.csv, data/s1_summary.json, figures/s1_channels.png
"""
import csv, json, os, sys, time, warnings
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import v2x2o_tb as tb

ROOT = Path(__file__).resolve().parent.parent
ES = [round(float(e), 4) for e in np.arange(-0.25, 0.10001, 0.005)]
W100 = [13, 19, 25, 38, 51]
W110 = [18, 27, 36, 54, 72]
CASES = [("cheng", 0.0), ("cheng", 0.08), ("single_band", 0.0)]


def param_sets():
    """(model, xy_shift, spin, orbital, params); the +0.08 case only re-runs the shifted d_xy band."""
    out = []
    for model, shift in CASES:
        m = tb.shifted(model, shift) if shift else tb.MODELS[model]
        for spin in ("up", "dn"):
            for orb, p in tb.orbital_params(m, spin):
                if shift and orb != "xy":
                    continue
                out.append((model, shift, spin, orb, p))
    return out


def job_kwant(args):
    kind, (model, shift, spin, orb, p), W = args
    f = tb.lead_modes_100 if kind == "100" else tb.lead_modes_110
    return [(model, shift, kind, spin, orb, W, E, f(p, W, E)) for E in ES]


def job_fs(args):
    kind, (model, shift, spin, orb, p) = args
    rows = []
    for E in ES:
        sh, pr = tb.wide_limit_counts(p, E, kind, n_par=720, n_perp=2000)
        if kind == "110":   # per unit perpendicular width -> per atomic row (row spacing a / sqrt2)
            sh, pr = sh / np.sqrt(2), pr / np.sqrt(2)
        rows.append((model, shift, kind, spin, orb, E, sh, pr))
    return rows


def main():
    t0 = time.time()
    ps = param_sets()
    kjobs = [("100", s, W) for s in ps for W in W100]
    # [110]: N_up = N_dn by the diagonal mirror (it maps the strip direction onto itself and swaps the spins);
    # spin up at every width, spin down at one width as an explicit check
    kjobs += [("110", s, W) for s in ps for W in W110 if s[2] == "up" or W == 36]
    fjobs = [(k, s) for s in ps for k in ("100", "110")]
    with Pool(6) as pool:
        fs = [r for rows in pool.map(job_fs, fjobs, chunksize=1) for r in rows]
        print(f"Fermi-surface counts done ({time.time() - t0:.0f} s)", flush=True)
        km = [r for rows in pool.map(job_kwant, kjobs, chunksize=1) for r in rows]
        print(f"Kwant mode counts done ({time.time() - t0:.0f} s)", flush=True)
    with open(ROOT / "data/s1_kwant_modes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "xy_shift_eV", "direction", "spin", "orbital", "width_sites_or_rows", "width_nm", "E_eV", "N"])
        for model, shift, kind, spin, orb, W, E, N in km:
            wnm = W * tb.A_KV2Se2O * (1 if kind == "100" else 1 / np.sqrt(2))
            w.writerow([model, shift, kind, spin, orb, W, round(wnm, 2), E, N])
    with open(ROOT / "data/s1_fermi_surface_counts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "xy_shift_eV", "direction", "spin", "orbital", "E_eV", "sharvin_per_row", "projected_per_row"])
        for r in fs:
            w.writerow(list(r[:6]) + [round(r[6], 5), round(r[7], 5)])
    print(f"wrote CSVs ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
