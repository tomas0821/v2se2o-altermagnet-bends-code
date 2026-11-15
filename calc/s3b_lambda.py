"""S3b (2026-11-14): diffusive design charts with the spin-relaxation length as an explicit parameter
(lambda_sf / W = inf, 5, 2, 1), with the spin-relaxation length as an explicit parameter.
P is the spin polarisation of the current delivered into the output contact (L: the [010] arm; T: one arm).
Writes data/s3b_lambda.csv. Run (project root): OMP_NUM_THREADS=1 ~/venvs/crn-te/bin/python calc/s3b_lambda.py
"""
import csv, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import network2c as N

ROOT = Path(__file__).resolve().parent.parent
S3 = json.load(open(ROOT / "data/s3_summary.json"))
r = S3["network: r used (KV2Se2O, E_F, 300 K)"]
W = 20
cond = {"up": (r, 1.0), "dn": (1.0, r)}
P = lambda d: (d["up"] - d["dn"]) / (d["up"] + d["dn"])
rows = []
for lam in (None, 5, 2, 1):
    g = 0.0 if lam is None else np.sqrt(r) / (2 * (lam * W) ** 2)
    for ratio in (1.0, 4 / 3, 2, 3, 4, 6):
        tot = 4 * W; a = int(round(tot * ratio / (1 + ratio))); b = tot - a
        cells, cin, cout = N.L_cells(W, a, b)
        res = N.solve(cells, cond, fixed=[(cin, 1.0), (cout, 0.0)], g_sf=g)
        rows.append(["L", "inf" if lam is None else lam, a, b, round(a / b, 4), P(res["I_fixed"][1]), P(res["I_fixed"][0])])
    for ratio in (0.25, 0.5, 1, 2, 3, 4):
        La = 2 * W; Ls = int(round(ratio * La))
        cells, cs_, ca1, ca2 = N.T_cells(W, Ls, La)
        res = N.solve(cells, cond, fixed=[(cs_, 1.0), (ca1, 0.0), (ca2, 0.0)], g_sf=g)
        rows.append(["T", "inf" if lam is None else lam, Ls, La, round(Ls / La, 4), P(res["I_fixed"][1]), P(res["I_fixed"][0])])
with open(ROOT / "data/s3b_lambda.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["geometry", "lambda_sf_over_W", "arm1_or_stem", "arm2_or_arm", "ratio", "P_output", "P_input"])
    w.writerows(rows)
for r_ in rows:
    print(r_[0], r_[1], f"ratio {r_[4]:.3g}: P_out {r_[5]:+.3f}  P_in {r_[6]:+.3f}")
