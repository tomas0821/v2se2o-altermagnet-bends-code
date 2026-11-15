"""S4b (2026-11-14): readout numbers at the working spin-relaxation length.
(i) The Pt inverse-spin-Hall estimate of s4_readout.ishe() redone at lambda_sf = 25 nm (it used 100 nm) with the
corner accumulation of the lambda_sf = W network (W = 20 cells, about 20-25 nm), against the probe pair at the same
lambda_sf/W. (ii) Probe-pair ranges per lambda_sf/W, for all probe widths and for w_p >= 4 cells.
Reads data/s4_diffusive.csv, data/s3_summary.json. No new transport runs.
Run (project root): ~/venvs/crn-te/bin/python calc/s4b_ishe.py  ->  data/s4b_ishe.json
"""
import csv, json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
S3 = json.load(open(ROOT / "data/s3_summary.json"))
D = list(csv.DictReader(open(ROOT / "data/s4_diffusive.csv")))


def ishe(lambda_sf_nm, mu_s_over_V, w_nm=20.0, rho_Pt=2.0e-7, lam_Pt_nm=1.5, t_Pt_nm=5.0, theta=0.1, bias_mV=10.0):
    """same spin-resistance model as s4_readout.ishe(): single altermagnet layer under a Pt pad covering the corner."""
    Rsq = S3["mean free path, rho(300 K) = 0.5 mOhm cm (Cs1-dV2Te2O), c = 7.37 A"]["sheet_R_per_layer_Ohm"]
    w, lam, lPt, tPt = w_nm * 1e-9, lambda_sf_nm * 1e-9, lam_Pt_nm * 1e-9, t_Pt_nm * 1e-9
    A = w * w
    Rs_AM = Rsq * lam / w
    Rs_Pt = rho_Pt * lPt / (A * np.tanh(tPt / lPt))
    Is = mu_s_over_V * bias_mV * 1e-3 / (Rs_AM + Rs_Pt)
    jc = theta * (Is / A) * (2 * lPt / tPt) * np.tanh(tPt / (2 * lPt))
    return dict(parameters=dict(lambda_sf_nm=lambda_sf_nm, mu_s_over_V=mu_s_over_V, w_nm=w_nm, rho_Pt_Ohm_m=rho_Pt,
                                lambda_Pt_nm=lam_Pt_nm, t_Pt_nm=t_Pt_nm, theta_SH=theta, bias_mV=bias_mV),
                R_s_AM_Ohm=round(Rs_AM, 1), R_s_Pt_Ohm=round(Rs_Pt, 2), V_ISHE_uV=round(rho_Pt * jc * w * 1e6, 4))


def main():
    out = {"probe pair dV/V (total, dV_AM) by lambda_sf/W": {}, "probe pair dV/V, w_p >= 4 cells": {}}
    for lam in ("inf", "5", "2", "1"):
        v = [float(r["dV_AM"]) for r in D if r["lambda_sf_over_W"] == lam]
        v4 = [float(r["dV_AM"]) for r in D if r["lambda_sf_over_W"] == lam and int(r["wp"]) >= 4]
        out["probe pair dV/V (total, dV_AM) by lambda_sf/W"][lam] = [round(min(v), 3), round(max(v), 3)]
        out["probe pair dV/V, w_p >= 4 cells"][lam] = [round(min(v4), 3), round(max(v4), 3)]
    mu1 = S3["spin relaxation (W = 20)"]["lambda_sf/W = 1"]["max_abs_mu_s"]
    est = ishe(25.0, round(mu1, 3))
    bias = est["parameters"]["bias_mV"]
    probe = [float(r["dV_spin"]) * bias * 1e3 for r in D if r["lambda_sf_over_W"] == "1"]       # uV
    est["probe_pair_lambda_W_uV"] = [round(min(probe), 0), round(max(probe), 0)]
    est["ratio_ISHE_over_probe"] = [float(f"{est['V_ISHE_uV'] / max(probe):.2e}"), float(f"{est['V_ISHE_uV'] / min(probe):.2e}")]
    est["note"] = "lambda_sf = 25 nm (working value), mu_s/V = max corner accumulation of the lambda_sf = W network (no probes)"
    out["ISHE estimate at lambda_sf = 25 nm vs probe pair at lambda_sf = W"] = est
    print(json.dumps(out, indent=1))
    json.dump(out, open(ROOT / "data/s4b_ishe.json", "w"), indent=1)


if __name__ == "__main__":
    main()
