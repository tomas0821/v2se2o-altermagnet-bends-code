"""S4 readout (2026-11-14).

Readout of the corner spin accumulation of a symmetric L (net P = 0) with nonmagnetic contacts only.
Detector: two floating probe arms of the same altermagnet at the outer corner, one along [100] (to +x from the
right edge) and one along [010] (to -y from the bottom edge), each wp wide and ending in a nonmagnetic contact.
A [100] arm conducts spin up r times better than spin down, a [010] arm the reverse, so
    dV = V_[100] - V_[010] ~ P_str * mu_s(corner) + (charge background).
The charge background is computed exactly with an isotropic, spin-degenerate device of the same total
conductivity (the altermagnet's total sigma_x = sigma_y, so its charge problem is that of the isotropic device).
A straight ribbon has mu_s = 0 identically (S4 check), so it gives no spin signal.

(a) Diffusive, room temperature: two-current network (network2c), W = 20, arms 40/40, probe length 10,
    wp = 1, 2, 4, 8; lambda_sf / W = inf, 5, 2, 1 (S3c Elliott estimate: lambda_sf ~ 25 nm -> ~W for 20 nm ribbons).
(b) Ballistic clean limit: Kwant, four terminals (input [100], output [010], two probe leads), Buettiker
    floating probes with nonmagnetic reservoirs; orbital- and spin-resolved conductance matrices summed;
    W = 20, 40; wp = 2, 4, 6; E_F +- 0.1 eV.
(c) ISHE estimate (order of magnitude, stated parameters): Pt detector on the corner, spin-resistance model.

Run (project root): OMP_NUM_THREADS=1 ~/venvs/crn-te/bin/python calc/s4_readout.py
Outputs: data/s4_diffusive.csv, data/s4_ballistic.csv, data/s4_summary.json, figures/s4_readout.png
"""
import csv, json, os, sys, warnings
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import network2c as N
import v2x2o_tb as tb

ROOT = Path(__file__).resolve().parent.parent
S3 = json.load(open(ROOT / "data/s3_summary.json"))
R = S3["network: r used (KV2Se2O, E_F, 300 K)"]
PSTR = (R - 1) / (R + 1)


# ---------------------------------------------------------------- (a) diffusive
def L_with_probes(W, Larm, wp, Lp):
    cells, cin, cout = N.L_cells(W, Larm, Larm)
    px = {(i, j) for i in range(W, W + Lp) for j in range(wp)}
    py = {(i, j) for i in range(W - wp, W) for j in range(-Lp, 0)}
    fx = {(W + Lp - 1, j) for j in range(wp)}
    fy = {(i, -Lp) for i in range(W - wp, W)}
    return cells | px | py, cin, cout, fx, fy


def diffusive():
    rows = []
    W, Larm, Lp = 20, 40, 10
    # corner mu_s without probes (normalisation)
    cells, cin, cout = N.L_cells(W, Larm, Larm)
    mu0 = {}
    for lam in (None, 5, 2, 1):
        g = 0.0 if lam is None else np.sqrt(R) / (2 * (lam * W) ** 2)
        res = N.solve(cells, {"up": (R, 1.0), "dn": (1.0, R)}, fixed=[(cin, 1.0), (cout, 0.0)], g_sf=g)
        mu0[lam] = res["phi"][((W - 1, 0), "up")] - res["phi"][((W - 1, 0), "dn")]
    for lam in (None, 5, 2, 1):
        for wp in (1, 2, 4, 8):
            cs, cin, cout, fx, fy = L_with_probes(W, Larm, wp, Lp)
            out = {}
            for model, cond in (("AM", {"up": (R, 1.0), "dn": (1.0, R)}),
                                ("iso", {"up": ((R + 1) / 2, (R + 1) / 2), "dn": ((R + 1) / 2, (R + 1) / 2)})):
                g = 0.0 if lam is None else np.sqrt(R) / (2 * (lam * W) ** 2)
                if model == "iso" and lam is not None:
                    g = (R + 1) / 2 / (2 * (lam * W) ** 2)
                res = N.solve(cs, cond, fixed=[(cin, 1.0), (cout, 0.0)], floating=[fx, fy], g_sf=g)
                Vx, Vy = res["V_float"]
                mus = res["phi"][((W - 1, 0), "up")] - res["phi"][((W - 1, 0), "dn")]
                out[model] = (Vx, Vy, mus)
            if lam is None and wp == 4:   # map of mu_s with the probes attached, for the figure
                res = N.solve(cs, {"up": (R, 1.0), "dn": (1.0, R)}, fixed=[(cin, 1.0), (cout, 0.0)], floating=[fx, fy])
                M = np.full((W + Larm + Lp, W + Larm + Lp), np.nan)    # rows y + Lp, cols x + Larm
                for c in cs:
                    M[c[1] + Lp, c[0] + Larm] = res["phi"][(c, "up")] - res["phi"][(c, "dn")]
                np.save(ROOT / "data/s4_mu_s_with_probes.npy", M)
            dV, dV0 = out["AM"][0] - out["AM"][1], out["iso"][0] - out["iso"][1]
            rows.append(dict(lambda_sf_over_W="inf" if lam is None else lam, wp=wp, Vx=out["AM"][0], Vy=out["AM"][1],
                             dV_AM=dV, dV_iso=dV0, dV_spin=dV - dV0, mu_s_corner_with_probes=out["AM"][2],
                             mu_s_corner_no_probes=mu0[lam]))
    return rows


# ---------------------------------------------------------------- (b) ballistic (Kwant)
def kwant_job(args):
    import kwant
    model, orb, spin, W, wp, energies = args
    p = dict(tb.orbital_params(tb.MODELS[model], spin))[orb]
    lat = kwant.lattice.square(a=1, norbs=1)

    def lead(direction, cells):
        ld = kwant.Builder(kwant.TranslationalSymmetry(direction)); tb._fill(ld, lat, p, cells); return ld
    b = kwant.Builder()
    tb._fill(b, lat, p, [(x, y) for x in range(W) for y in range(W)])
    b.attach_lead(lead((-2, 0), [(x, y) for x in range(2) for y in range(W)]))          # 0 input  [100]
    b.attach_lead(lead((0, 2), [(x, y) for x in range(W) for y in range(2)]))           # 1 output [010]
    b.attach_lead(lead((2, 0), [(x, y) for x in range(2) for y in range(wp)]))          # 2 probe  [100] at the outer corner
    b.attach_lead(lead((0, -2), [(x, y) for x in range(W - wp, W) for y in range(2)]))  # 3 probe  [010] at the outer corner
    f = b.finalized()
    return [(model, orb, spin, W, wp, E, kwant.smatrix(f, E).conductance_matrix()) for E in energies]


def ballistic():
    ES = [round(-0.19 + 0.025 * i, 4) for i in range(9)]
    jobs = [(m, o, s, W, wp, ES) for W in (20, 40) for wp in (2, 4, 6)
            for m, sp in (("cheng", ("up", "dn")), ("iso", ("up",))) for o in ("xz", "xy") for s in sp]
    with Pool(6) as pool:
        res = [r for rr in pool.map(kwant_job, sorted(jobs, key=lambda j: -j[3])) for r in rr]
    G = {}
    for model, orb, spin, W, wp, E, g in res:
        k = (model, W, wp, E)
        G[k] = G.get(k, 0) + np.asarray(g) * (2 if model == "iso" else 1)
    rows = []
    for (model, W, wp, E), g in sorted(G.items()):
        # V0 = 1, V1 = 0, I2 = I3 = 0
        A = g[2:4, 2:4]; bvec = -g[2:4, 0] * 1.0
        V2, V3 = np.linalg.solve(A, bvec)
        rows.append(dict(model=model, W=W, wp=wp, E=E, V100=V2, V010=V3, dV=V2 - V3))
    return rows


# ---------------------------------------------------------------- (c) ISHE estimate
def ishe(dV_spin_per_V):
    """order-of-magnitude, stated parameters; single altermagnet layer under a Pt pad covering the corner."""
    Rsq = S3["mean free path, rho(300 K) = 0.5 mOhm cm (Cs1-dV2Te2O), c = 7.37 A"]["sheet_R_per_layer_Ohm"]
    par = dict(w_nm=20.0, lambda_sf_nm=100.0, rho_Pt_Ohm_m=2.0e-7, lambda_Pt_nm=1.5, t_Pt_nm=5.0, theta_SH=0.1,
               bias_mV=10.0, mu_s_over_V=0.8)
    w, lam = par["w_nm"] * 1e-9, par["lambda_sf_nm"] * 1e-9
    lPt, tPt = par["lambda_Pt_nm"] * 1e-9, par["t_Pt_nm"] * 1e-9
    A = w * w
    Rs_AM = Rsq * lam / w                                    # spin resistance of the AM channel feeding the pad
    Rs_Pt = par["rho_Pt_Ohm_m"] * lPt / (A * np.tanh(tPt / lPt))
    mu0 = par["mu_s_over_V"] * par["bias_mV"] * 1e-3         # V
    Is = mu0 / (Rs_AM + Rs_Pt)                               # A (charge units)
    js = Is / A
    jc = par["theta_SH"] * js * (2 * lPt / tPt) * np.tanh(tPt / (2 * lPt))
    V_ishe = par["rho_Pt_Ohm_m"] * jc * w
    V_probe = dV_spin_per_V * par["bias_mV"] * 1e-3
    return dict(parameters=par, R_s_AM_Ohm=round(Rs_AM, 1), R_s_Pt_Ohm=round(Rs_Pt, 1),
                mu_s_suppression_under_Pt=round(Rs_Pt / (Rs_AM + Rs_Pt), 4),
                V_ISHE_uV=round(V_ishe * 1e6, 4), V_probe_pair_uV=round(V_probe * 1e6, 1),
                ratio_probe_over_ISHE=round(V_probe / V_ishe, 0))


def main():
    S = {}
    D = diffusive()
    with open(ROOT / "data/s4_diffusive.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(D[0].keys())); w.writeheader(); w.writerows(D)
    S["diffusive (W=20, arms 40, probes 10 long)"] = {
        f"lambda={d['lambda_sf_over_W']}, wp={d['wp']}": dict(
            dV_AM=round(d["dV_AM"], 4), dV_iso=round(d["dV_iso"], 4), dV_spin=round(d["dV_spin"], 4),
            dV_spin_over_Pstr_mu_s=round(d["dV_spin"] / (PSTR * d["mu_s_corner_no_probes"]), 3)) for d in D}
    # straight ribbon check with one [010] probe at mid-length: spin signal must vanish
    W = 20
    cs = {(i, j) for i in range(100) for j in range(W)} | {(i, j) for i in range(48, 52) for j in range(-10, 0)}
    res = N.solve(cs, {"up": (R, 1.0), "dn": (1.0, R)}, fixed=[({(0, j) for j in range(W)}, 1.0), ({(99, j) for j in range(W)}, 0.0)],
                  floating=[{(i, -10) for i in range(48, 52)}])
    S["straight ribbon: [010] side probe potential - local mid-edge potential"] = float(
        f"{res['V_float'][0] - np.mean([res['phi'][((i, 0), 'up')] for i in range(48, 52)]):.2e}")
    B = ballistic()
    with open(ROOT / "data/s4_ballistic.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(B[0].keys())); w.writeheader(); w.writerows(B)
    bal = {}
    for W in (20, 40):
        for wp in (2, 4, 6):
            am = [r["dV"] for r in B if r["model"] == "cheng" and r["W"] == W and r["wp"] == wp]
            iso = [r["dV"] for r in B if r["model"] == "iso" and r["W"] == W and r["wp"] == wp]
            ef = [r["dV"] for r in B if r["model"] == "cheng" and r["W"] == W and r["wp"] == wp and r["E"] == -0.09][0]
            bal[f"W={W}, wp={wp}"] = dict(dV_AM_at_EF=round(ef, 4), dV_AM_range=[round(min(am), 4), round(max(am), 4)],
                                         dV_iso_range=[round(min(iso), 4), round(max(iso), 4)])
    S["ballistic (Kwant, NM reservoirs), dV/V"] = bal
    best = max((d for d in D if d["lambda_sf_over_W"] == 5), key=lambda d: abs(d["dV_spin"]))
    S["ISHE estimate vs probe pair (lambda_sf = 5W)"] = ishe(abs(best["dV_spin"]))
    json.dump(S, open(ROOT / "data/s4_summary.json", "w"), indent=1)
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
