"""S3d (2026-11-14): Kubo-Greenwood check of the constant-tau Boltzmann
anisotropy r = sigma_xx/sigma_yy at k_F l ~ 1-2. Intraband Kubo-Greenwood with a constant Lorentzian broadening Gamma
(no vertex corrections) on the Cheng KV2Se2O model; without spin-orbit coupling the four bands decouple, so there
are no interband velocity terms. Gamma -> 0 reproduces the Boltzmann value.
Run (project root): OMP_NUM_THREADS=1 ~/venvs/crn-te/bin/python calc/s3d_kubo.py   ->  data/s3d_kubo.json
"""
import json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import v2x2o_tb as tb

ROOT = Path(__file__).resolve().parent.parent
KB, T = 8.617333e-5, 300.0
kT = KB * T
N = 500
HBAR_EVS = 6.582119569e-16
S3 = json.load(open(ROOT / "data/s3_summary.json"))

k = (np.arange(N) + 0.5) * 2 * np.pi / N - np.pi
kx, ky = np.meshgrid(k, k, indexing="ij")


def bands(p):
    e, t1x, t1y, t2, t3x, t3y = p
    E = tb.band(p, kx, ky)
    vx = -2 * t1x * np.sin(kx) - 4 * t2 * np.sin(kx) * np.cos(ky) - 4 * t3x * np.sin(2 * kx)
    vy = -2 * t1y * np.sin(ky) - 4 * t2 * np.cos(kx) * np.sin(ky) - 4 * t3y * np.sin(2 * ky)
    return E, vx * vx, vy * vy


B = {o: bands(p) for o, p in tb.MODELS["cheng"].items()}          # spin up; spin down is x <-> y
xs = np.linspace(-12, 12, 49); dx = xs[1] - xs[0]
wf = 1.0 / (4 * np.cosh(xs / 2) ** 2) * dx                          # -f'(E) dE in units of kT


def sig(mu, G):
    out = {}
    for o, (E, vx2, vy2) in B.items():
        if G == 0:                                                  # Boltzmann: -f'(E_k)
            xx = (E - mu) / kT; ww = 1.0 / (4 * kT * np.cosh(np.clip(xx, -60, 60) / 2) ** 2)
            out[o] = ((ww * vx2).mean(), (ww * vy2).mean())
            continue
        sxx = syy = 0.0
        for x, w in zip(xs, wf):
            A = (G / np.pi) / ((mu + x * kT - E) ** 2 + G ** 2)
            sxx += w * (vx2 * A * A).mean(); syy += w * (vy2 * A * A).mean()
        out[o] = (sxx * 2 * np.pi * G, syy * 2 * np.pi * G)         # A^2 -> delta / (2 pi G) as G -> 0
    return out


def ratio(s):
    return sum(v[0] for v in s.values()) / sum(v[1] for v in s.values())


P = lambda r: (r - 1) / (r + 1)


def main():
    tau = S3["mean free path, rho(300 K) = 0.5 mOhm cm (Cs1-dV2Te2O), c = 7.37 A"]["tau_fs"] * 1e-15
    G_half, G_full = HBAR_EVS / (2 * tau), HBAR_EVS / tau
    out = dict(method="intraband Kubo-Greenwood, constant Lorentzian Gamma, no vertex corrections; 500^2 k-grid; "
                      "Fermi smearing at 300 K", tau_fs=round(tau * 1e15, 3), hbar_over_tau_eV=round(G_full, 3),
               Gamma_half_eV=round(G_half, 3), scan=[], sweep_at_hbar_over_2tau=[])
    for G in (0.0, 0.05, 0.10, 0.15, round(G_half, 3), 0.30, round(G_full, 3)):
        s = sig(tb.E_F, G); r = ratio(s)
        row = dict(Gamma_eV=G, r=round(r, 3), P_str=round(P(r), 4), r_sheets=round(s["xz"][0] / s["xz"][1], 2),
                   r_pockets=round(s["xy"][0] / s["xy"][1], 2))
        out["scan"].append(row); print(row, flush=True)
    for mu in np.round(tb.E_F + np.array([-0.10, -0.05, 0.0, 0.05, 0.10]), 3):
        rk, rb = ratio(sig(mu, G_half)), ratio(sig(mu, 0.0))
        row = dict(E_eV=float(mu), r_Kubo=round(rk, 3), P_Kubo=round(P(rk), 4), r_Boltzmann=round(rb, 3), P_Boltzmann=round(P(rb), 4))
        out["sweep_at_hbar_over_2tau"].append(row); print(row, flush=True)
    Pk = [r["P_Kubo"] for r in out["sweep_at_hbar_over_2tau"]]
    out["P_str_Kubo_range_over_EF_pm_0.1"] = [min(Pk), max(Pk)]
    json.dump(out, open(ROOT / "data/s3d_kubo.json", "w"), indent=1)


if __name__ == "__main__":
    main()
