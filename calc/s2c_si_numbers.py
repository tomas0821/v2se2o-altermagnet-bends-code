"""S2c (2026-11-14): corner-efficiency numbers tabulated in the Supplementary Material -- spin-resolved T efficiencies with their ranges, the isotropic W/4 chamfer, the T-arm polarization by
width (scan and held-out grid), the largest |T - kappa N_min| in channels, and the 45-degree polarization at E_F.
Reads data/s2_corners.csv and data/s2b_holdout.csv through s2_analysis. No new Kwant runs.
Run (project root): ~/venvs/crn-te/bin/python calc/s2c_si_numbers.py  ->  data/s2c_si_numbers.json
"""
import csv, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import s2_analysis as A2

ROOT = Path(__file__).resolve().parent.parent
A = 0.3952
EF = -0.09


def stats(v):
    v = np.array([x for x in v if np.isfinite(x)])
    return dict(mean=round(float(v.mean()), 3), min=round(float(v.min()), 3), max=round(float(v.max()), 3),
                max_dev_pct=round(float(np.max(np.abs(v / v.mean() - 1)) * 100), 1))


def Parm(tot_fn, W, E):
    u, d = tot_fn("up", W, E)["T1"], tot_fn("dn", W, E)["T1"]
    return (u - d) / (u + d)


def main():
    WS, ES = A2.WS, A2.ES
    out = {}
    for g in ("L", "Lc4", "Lc2", "T", "B45"):
        out[f"kappa {g}"] = {m: stats([A2.kappa(g, m, sp, W, E) for sp in ("up", "dn") for W in WS for E in ES])
                             for m in ("cheng", "iso", "single_band") if not (m == "single_band" and g in ("Lc4", "Lc2"))}
    out["kappa T by spin (cheng)"] = {sp: stats([A2.kappa("T", "cheng", sp, W, E) for W in WS for E in ES]) for sp in ("up", "dn")}
    out["kappa T by spin (cheng), W >= 20 sites"] = {sp: stats([A2.kappa("T", "cheng", sp, W, E) for W in WS if W >= 20 for E in ES])
                                                     for sp in ("up", "dn")}
    # largest |T - kappa N_min| in channels with the quoted constants
    kL = out["kappa L"]["cheng"]["mean"]
    ku, kd = out["kappa T by spin (cheng)"]["up"]["mean"], out["kappa T by spin (cheng)"]["dn"]["mean"]
    devL = max(abs(A2.total("L", "cheng", sp, W, E)["T1"] - kL * A2.bound("L", A2.total("L", "cheng", sp, W, E)))
               for sp in ("up", "dn") for W in WS for E in ES)
    devT = max(abs(A2.total("T", "cheng", sp, W, E)["T1"] - k * A2.bound("T", A2.total("T", "cheng", sp, W, E)))
               for sp, k in (("up", ku), ("dn", kd)) for W in WS for E in ES)
    out["max |T - kappa N_min| (channels), scan"] = dict(L_sharp=round(devL, 2), T_arm_spin_resolved=round(devT, 2))
    # T-arm polarization by width
    tot = lambda sp, W, E: A2.total("T", "cheng", sp, W, E)
    out["T-arm P by width (scan)"] = {f"W={W} ({W * A:.1f} nm)": [round(min(Parm(tot, W, E) for E in ES), 3), round(max(Parm(tot, W, E) for E in ES), 3)]
                                      for W in WS}
    P6 = [Parm(tot, W, E) for W in WS if W * A >= 5.5 for E in ES]
    out["T-arm P, W >= 6 nm (scan)"] = [round(min(P6), 3), round(max(P6), 3)]
    H = {}
    for r in csv.DictReader(open(ROOT / "data/s2b_holdout.csv")):
        if r["geometry"] == "T" and r["model"] == "cheng" and float(r["xy_shift_eV"]) == 0.0:
            H.setdefault((int(r["W_sites"]), float(r["E_eV"]), r["spin"]), 0.0)
            H[(int(r["W_sites"]), float(r["E_eV"]), r["spin"])] += float(r["T_1from0"])
    Ph = [(H[(W, E, "up")] - H[(W, E, "dn")]) / (H[(W, E, "up")] + H[(W, E, "dn")]) for (W, E, s) in H if s == "up" and (W, E, "dn") in H]
    out["T-arm P, held-out grid"] = [round(min(Ph), 3), round(max(Ph), 3)]
    # 45-degree bend at E_F, by width, with and without the pocket shift
    b45 = {}
    for sh in (0.0, 0.08):
        t = lambda sp, W, E: A2.total("B45", "cheng", sp, W, E, sh)
        b45[f"xy+{sh}"] = {f"{W * A:.1f} nm": round(Parm(t, W, EF), 3) for W in WS}
    out["45-degree bend P at E_F by width"] = b45
    print(json.dumps(out, indent=1))
    json.dump(out, open(ROOT / "data/s2c_si_numbers.json", "w"), indent=1)


if __name__ == "__main__":
    main()
