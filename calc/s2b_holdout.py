"""S2b (2026-11-14): out-of-sample check of the restated ballistic rules (GATE option b), pre-registered in the
record ("Holdout criterion for the restated rules") before this ran. Realistic model only; widths and energies
disjoint from the S2 grid. Writes data/s2b_holdout.csv and data/s2b_summary.json.
Run (project root): OMP_NUM_THREADS=1 ~/venvs/crn-te/bin/python calc/s2b_holdout.py
"""
import csv, json, os, sys, warnings
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import s2_corners as S2

ROOT = Path(__file__).resolve().parent.parent
WS = [18, 22, 35, 50]
ES = [round(-0.1775 + 0.025 * i, 4) for i in range(8)]
RULE = {"L": 0.876, "Lc4": 0.921, "Lc2": 0.957, "T_up": 0.792, "T_dn": 0.896}   # S2 means at W >= 20
TOL = 0.06


def main():
    jobs = [(g, "cheng", 0.0, o, sp, W, ES) for g in ("L", "Lc4", "Lc2", "T") for W in WS for o in ("xz", "xy") for sp in ("up", "dn")]
    with Pool(6) as pool:
        res = pool.map(S2.job, sorted(jobs, key=lambda j: -j[5]))
    rows = [r for rr, _ in res for r in rr]
    with open(ROOT / "data/s2b_holdout.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["geometry", "model", "xy_shift_eV", "orbital", "spin", "W_sites", "E_eV",
                    "N_lead0_in", "N_lead1", "N_lead2", "T_1from0", "T_2from0"])
        w.writerows(sorted(rows, key=lambda r: tuple(str(x) for x in r[:7])))
    D = {}
    for r in rows:
        k = (r[0], r[4], r[5], r[6]); d = D.setdefault(k, dict(n0=0, n1=0, T=0.0))
        d["n0"] += r[7]; d["n1"] += r[8]; d["T"] += r[10]
    out = {}
    for g in ("L", "Lc4", "Lc2", "T"):
        for sp in ("up", "dn"):
            key = g if g != "T" else f"T_{sp}"
            for W in WS:
                for E in ES:
                    d = D[(g, sp, W, E)]
                    b = min(d["n0"], 2 * d["n1"]) / 2 if g == "T" else min(d["n0"], d["n1"])
                    out.setdefault(key, []).append((W, E, d["T"] / b))
    S = {"criterion": f"every kappa at W in (22, 35, 50) within ±{TOL:.0%} of the S2 (W >= 20) rule value; W = 18 reported only"}
    allpass = True
    for key, vals in out.items():
        gated = [abs(k / RULE[key] - 1) for W, E, k in vals if W >= 20]
        w18 = [abs(k / RULE[key] - 1) for W, E, k in vals if W == 18]
        ok = max(gated) <= TOL
        allpass &= ok
        worst = max(((abs(k / RULE[key] - 1), W, E, k) for W, E, k in vals if W >= 20))
        S[key] = dict(rule=RULE[key], holdout_mean=round(float(np.mean([k for W, E, k in vals if W >= 20])), 4),
                      max_rel_dev_W_ge_20=round(max(gated), 4), verdict="PASS" if ok else "FAIL",
                      worst=dict(W=worst[1], E=worst[2], kappa=round(worst[3], 4)),
                      W18_max_rel_dev=round(max(w18), 4))
    # T-arm P from the spin-resolved rule vs holdout Kwant
    Pk = []
    for W in WS:
        for E in ES:
            u, d = D[("T", "up", W, E)]["T"], D[("T", "dn", W, E)]["T"]
            Pk.append((W, (u - d) / (u + d)))
    S["T-arm P holdout range (W >= 20)"] = [round(min(p for W, p in Pk if W >= 20), 3), round(max(p for W, p in Pk if W >= 20), 3)]
    S["T-arm P spin-resolved rule"] = round((2 * RULE["T_up"] - RULE["T_dn"]) / (2 * RULE["T_up"] + RULE["T_dn"]), 3)
    S["overall"] = "PASS" if allpass else "FAIL"
    json.dump(S, open(ROOT / "data/s2b_summary.json", "w"), indent=1)
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
