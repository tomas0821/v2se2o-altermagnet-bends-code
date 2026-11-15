"""S3f (2026-11-14): how far spin relaxation moves
the network away from the spin-conserving closed forms, and the network's spin-diffusion length convention.
Reads data/s3b_lambda.csv (network with on-site spin flip g = sqrt(r)/(2 lambda^2), W = 20 cells) and
data/s3_summary.json. Path L = arm lengths (L1 + L2 for the L-bend, L_s + L_a for the T), corner square excluded.
Run (project root): ~/venvs/crn-te/bin/python calc/s3f_lambda_correction.py  ->  data/s3f_lambda_correction.json
"""
import csv, json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
W = 20
S3 = json.load(open(ROOT / "data/s3_summary.json"))
r = S3["network: r used (KV2Se2O, E_F, 300 K)"]


def main():
    rows = list(csv.DictReader(open(ROOT / "data/s3b_lambda.csv")))
    base = {(x["geometry"], x["arm1_or_stem"], x["arm2_or_arm"]): float(x["P_output"]) for x in rows if x["lambda_sf_over_W"] == "inf"}
    pts = []
    for x in rows:
        if x["lambda_sf_over_W"] == "inf":
            continue
        lam = float(x["lambda_sf_over_W"]) * W
        L = int(x["arm1_or_stem"]) + int(x["arm2_or_arm"])
        dP = float(x["P_output"]) - base[(x["geometry"], x["arm1_or_stem"], x["arm2_or_arm"])]
        pts.append(dict(geometry=x["geometry"], arms=[int(x["arm1_or_stem"]), int(x["arm2_or_arm"])], lambda_over_W=float(x["lambda_sf_over_W"]),
                        L_over_lambda=round(L / lam, 3), dP=round(dP, 4), c=round(-dP / (L / lam) ** 2, 3)))
    sr = S3["spin relaxation (W = 20)"]["lambda_sf/W = 5"]
    small = [p for p in pts if p["L_over_lambda"] <= 0.8]
    cs = [p["c"] for p in small]
    cmax = max(cs)
    out = dict(
        note="dP = P_output(lambda_sf) - P_output(no spin flip); fit dP = -c (L/lambda_sf)^2 for L/lambda_sf <= 0.8",
        points=sorted(pts, key=lambda p: p["L_over_lambda"]),
        c_range_small_L=[min(cs), max(cs)], c_mean_small_L=round(float(np.mean(cs)), 3),
        L_over_lambda_for_absdP_le_0p05=round(float(np.sqrt(0.05 / cmax)), 2),
        L_over_lambda_for_absdP_le_0p03=round(float(np.sqrt(0.03 / cmax)), 2),
        L_40_20_at_lambda_5W=dict(P_output=sr.get("P_L_40_20_at_output"), note="arms 40/20 = 3W, L/lambda = 0.6"),
        network_decay_length_over_lambda=round(float(np.sqrt(2 * np.sqrt(r) / (r + 1))), 3),
        decay_note="with g = sqrt(r)/(2 lambda^2) and bonds (r,1)/(1,r), mu_s decays as exp(-x/lambda_eff) along either axis, "
                   "lambda_eff = lambda sqrt(2 sqrt(r)/(r+1))")
    print(json.dumps({k: v for k, v in out.items() if k != "points"}, indent=1))
    json.dump(out, open(ROOT / "data/s3f_lambda_correction.json", "w"), indent=1)


if __name__ == "__main__":
    main()
