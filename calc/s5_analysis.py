"""S5 analysis + figure (2026-11-14). Reads data/s5_results.jsonl (from s5_disorder.py).
Writes data/s5_summary.json and figures/s5_disorder.png. compute() is also used by calc/si_figures.py.
Ensembles are labelled "non-localized" (the criterion tests
non-localization, not diffusion), with L/ell for both spins, standard errors and Eq. (5) for the T geometry.
Run (project root): ~/venvs/crn-te/bin/python calc/s5_analysis.py
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
A = 0.3952
UP, DN, ALT, GREY, INK, MUTED = "#e34948", "#2a78d6", "#eda100", "#a3a29d", "#0b0b0b", "#52514e"
VIOLET, AQUA = "#6250d6", "#1baf7a"

R = defaultdict(dict)       # (part, geom, W, Ls, U, spin, mode) -> {orb: record}
for l in open(ROOT / "data/s5_results.jsonl"):
    r = json.loads(l); t = r["task"]
    R[(t["part"], t["geom"], t["W"], t["Ls"], t["U"], t["spin"], t["mode"])][t["orb"]] = r


def tot(part, geom, W, Ls, U, spin, mode="ind", arm=1):
    rec = R[(part, geom, W, Ls, U, spin, mode)]
    if set(rec) != {"xz", "xy"}:
        return None, None
    key = "T1" if arm == 1 else "T2"
    T = np.array(rec["xz"][key]) + np.array(rec["xy"][key])      # same salts -> same disorder on both orbitals
    N0 = rec["xz"]["N"][0] + rec["xy"]["N"][0]
    return T, N0


def compute():
    S = {}
    # ---------------- calibration per spin sector
    cal = {}
    for W in (20, 38):
        for U in (0.25, 0.30, 0.35):
            for spin, lab in (("up", "fast"), ("dn", "slow")):
                rows = []
                for L in (20, 40, 80, 160, 320):
                    T, N0 = tot("calib", "straight", W, L, U, spin)
                    if T is None:
                        continue
                    rows.append((L, float(T.mean()), float(np.log(np.maximum(T, 1e-300)).mean()), N0))
                if len(rows) < 5:
                    continue
                Ls = np.array([r[0] for r in rows]); Tm = np.array([r[1] for r in rows]); lnT = np.array([r[2] for r in rows])
                N0 = rows[0][3]
                ell = [L * T / (N0 - T) for L, T in zip(Ls[:3], Tm[:3])]          # short lengths, before localisation
                sl = np.polyfit(Ls[2:], lnT[2:], 1)[0]                              # L >= 80
                xi = -2.0 / sl if sl < 0 else float("inf")
                cal[(W, U, lab)] = dict(N0=N0, L=Ls.tolist(), meanT=np.round(Tm, 3).tolist(), mean_lnT=np.round(lnT, 3).tolist(),
                                        ell_sites=round(float(np.mean(ell)), 1), ell_nm=round(float(np.mean(ell)) * A, 1),
                                        xi_sites=round(xi, 0), xi_nm=round(xi * A, 0),
                                        xi_is_lower_bound=bool(Tm[-1] > 1.0))
    S["calibration (straight ribbons; per spin sector along [100])"] = {f"W={k[0]} U={k[1]} {k[2]}": v for k, v in cal.items()}
    # ---------------- ensembles
    def P_stats(geom, W, U, mode="ind", part="ens"):
        Ls = W if part == "ens" else 20
        Tu, _ = tot(part, geom, W, Ls, U, "up", mode); Td, _ = tot(part, geom, W, Ls, U, "dn", mode)
        if Tu is None or Td is None:
            return None
        P = (Tu - Td) / (Tu + Td)
        out = dict(n=len(P), P_mean=round(float(P.mean()), 4), P_rms=round(float(P.std()), 4),
                   P_median=round(float(np.median(P)), 4), frac_absP_gt_0p5=round(float(np.mean(np.abs(P) > 0.5)), 3),
                   T_up_mean=round(float(Tu.mean()), 3), T_dn_mean=round(float(Td.mean()), 3),
                   lnT_dn_mean=round(float(np.log(np.maximum(Td, 1e-300)).mean()), 3),
                   corr_Tup_Tdn=round(float(np.corrcoef(Tu, Td)[0, 1]), 3))
        if geom == "T":
            Tu2, _ = tot(part, geom, W, Ls, U, "up", mode, arm=2); Td2, _ = tot(part, geom, W, Ls, U, "dn", mode, arm=2)
            P2 = (Tu2 - Td2) / (Tu2 + Td2)
            out["P_mean_bottom_arm"] = round(float(P2.mean()), 4)
        return out, P
    clean = {}
    for W in (20, 38):
        for g in ("straight", "L", "T"):
            Tu, _ = tot("clean", g, W, W, 0.0, "up"); Td, _ = tot("clean", g, W, W, 0.0, "dn")
            if Tu is not None:
                clean[(g, W)] = float((Tu[0] - Td[0]) / (Tu[0] + Td[0]))
    S["clean P (U = 0, same geometry)"] = {f"{g} W={W}": round(v, 4) for (g, W), v in clean.items()}
    ens, samples = {}, {}
    for U in (0.25, 0.30, 0.35):
        for W in (20, 38):
            for g in ("straight", "L", "T"):
                r = P_stats(g, W, U)
                if r is None:
                    continue
                st, P = r
                c = cal.get((W, U, "slow"))
                devlen = 3 * W      # disordered path length in sites: Ls + W + Ls with Ls = W, for all three geometries
                st["device_length_sites"] = devlen
                st["label"] = ("non-localized" if c and st["T_dn_mean"] >= 1.0 and devlen < c["xi_sites"] else "localised/marginal")
                cf = cal.get((W, U, "fast"))
                st["L_over_ell_slow"] = round(devlen / c["ell_sites"], 2) if c else None
                st["L_over_ell_fast"] = round(devlen / cf["ell_sites"], 2) if cf else None
                st["P_mean_standard_error"] = round(st["P_rms"] / np.sqrt(st["n"]), 4)
                ens[(U, W, g)] = st; samples[(U, W, g)] = P
    S["ensembles (independent A/B disorder, 200 seeds)"] = {f"U={k[0]} W={k[1]} {k[2]}": v for k, v in ens.items()}
    corr = {}
    for g in ("straight", "L", "T"):
        r = P_stats(g, 20, 0.30, mode="corr", part="corr")
        if r:
            corr[g] = r[0]; samples[("corr", 20, g)] = r[1]
    S["correlated A/B disorder (U = 0.30, W = 20, 200 seeds)"] = corr
    loc = {}
    for g in ("straight", "L", "T"):
        r = P_stats(g, 20, 1.2, part="loc")
        if r:
            loc[g] = r[0]; samples[("loc", 20, g)] = r[1]
    S["localised contrast (U = 1.2 eV, W = 20, 100 seeds)"] = loc
    s3 = json.load(open(ROOT / "data/s3_summary.json"))
    pstr = s3["KV2Se2O 300 K, E_F: r, P_str"][1]
    dT, LT = s3["T-junction terms"]["d_T_over_W"], s3["T-junction terms"]["L_T_over_W"]
    S["Eq. (5) for the ensemble T (L_s = L_a = W)"] = dict(
        P_str=pstr, P_T=round(pstr * (2 - 1 + dT) / (2 + 1 + LT), 4),
        P_T_with_ensemble_straight_P={f"U={k[0]} W={k[1]}": round(v["P_mean"] * (2 - 1 + dT) / (2 + 1 + LT), 4)
                                      for k, v in ens.items() if k[2] == "straight"})
    nl = [v for v in ens.values()]
    S["non-localized window: ranges"] = dict(
        L_over_ell_fast=[min(v["L_over_ell_fast"] for v in nl), max(v["L_over_ell_fast"] for v in nl)],
        L_over_ell_slow=[min(v["L_over_ell_slow"] for v in nl), max(v["L_over_ell_slow"] for v in nl)],
        slow_T_mean=[min(v["T_dn_mean"] for v in nl), max(v["T_dn_mean"] for v in nl)],
        all_non_localized=all(v["label"] == "non-localized" for v in nl))
    json.dump(S, open(ROOT / "data/s5_summary.json", "w"), indent=1)
    print(json.dumps(S, indent=1))
    return S, cal, ens, samples, clean


def main():
    S, cal, ens, samples, clean = compute()

    # ---------------- figure
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.linewidth": 0.8, "legend.frameon": False})
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.8))
    # (a) calibration at U = 0.30
    a = ax[0, 0]
    for W, ls in ((20, "-"), (38, "--")):
        for lab, col in (("fast", UP), ("slow", DN)):
            c = cal.get((W, 0.30, lab))
            if not c:
                continue
            a.plot(np.array(c["L"]) * A, c["mean_lnT"], ls, marker="o", color=col, ms=4, lw=1.3,
                   label=f"{'↑ fast' if lab == 'fast' else '↓ slow'}, W = {W * A:.0f} nm: ℓ = {c['ell_nm']} nm, ξ ≈ {c['xi_nm']:.0f} nm"
                         + (" (≥)" if c["xi_is_lower_bound"] else ""))
    a.axhline(0, color=GREY, lw=0.6)
    a.set_xlabel("disordered length L (nm)"); a.set_ylabel(r"$\langle \ln T_\sigma \rangle$")
    a.set_title("(a) per-spin calibration along [100], U = 0.30 eV", loc="left", fontsize=8.5)
    a.legend(fontsize=6.5, loc="lower left")
    # (b) P distributions at U = 0.30
    a = ax[0, 1]
    pos, labels = [], []
    k = 0
    for g in ("straight", "L", "T"):
        for W, col in ((20, MUTED), (38, INK)):
            P = samples.get((0.30, W, g))
            if P is None:
                continue
            vp = a.violinplot([P], positions=[k], widths=0.8, showextrema=False)
            for b in vp["bodies"]:
                b.set_facecolor(col); b.set_alpha(0.35); b.set_edgecolor(col)
            a.plot([k], [P.mean()], "o", color=col, ms=5)
            if (g, W) in clean:
                a.plot([k - 0.35, k + 0.35], [clean[(g, W)]] * 2, "-", color=UP, lw=1.8)
            pos.append(k); labels.append(f"{g}\n{W * A:.0f} nm"); k += 1
        k += 0.5
    s3 = json.load(open(ROOT / "data/s3_summary.json"))["KV2Se2O 300 K, E_F: r, P_str"][1]
    a.axhline(s3, color=UP, lw=1.0, ls=":", xmax=0.3)
    a.text(1.6, s3 + 0.02, f"diffusive P$_{{str}}$ = {s3:.2f} (S3)", fontsize=7, color=UP)
    a.plot([], [], "-", color=UP, lw=1.8, label="clean (U = 0)")
    a.plot([], [], "o", color=INK, ms=5, label="ensemble mean")
    a.axhline(0, color=GREY, lw=0.6)
    a.set_xticks(pos, labels, fontsize=7.5); a.set_ylabel("P (T-junction: one arm)")
    a.set_title("(b) diffusive-window ensembles, U = 0.30 eV, 200 seeds", loc="left", fontsize=8.5)
    a.legend(fontsize=7, loc="upper right")
    # (c) mean +- rms vs U
    a = ax[1, 0]
    for g, col, mk in (("straight", INK, "o"), ("L", VIOLET, "s"), ("T", AQUA, "^")):
        for W, ls in ((20, "-"), (38, "--")):
            pts = [(U, ens[(U, W, g)]["P_mean"], ens[(U, W, g)]["P_rms"]) for U in (0.25, 0.30, 0.35) if (U, W, g) in ens]
            if not pts:
                continue
            U_, m, s = zip(*pts)
            off = {20: -0.004, 38: 0.004}[W]
            a.errorbar(np.array(U_) + off, m, yerr=s, fmt=mk + ls, color=col, ms=4.5, lw=1.1, capsize=2,
                       label=f"{g}, W = {W * A:.0f} nm")
            if (g, W) in clean:
                a.plot([0.24, 0.36], [clean[(g, W)]] * 2, ls, color=col, lw=0.7, alpha=0.6)
    a.axhline(0, color=GREY, lw=0.6)
    a.set_xlabel("Anderson disorder U (eV)"); a.set_ylabel("⟨P⟩ ± rms")
    a.set_title("(c) ensemble mean and rms vs disorder (thin lines: clean)", loc="left", fontsize=8.5)
    a.legend(fontsize=6.5, ncol=3, loc="center", bbox_to_anchor=(0.5, 0.56))
    # (d) independent vs correlated vs localised (W = 20)
    a = ax[1, 1]
    k = 0; pos, labels = [], []
    for g in ("straight", "L", "T"):
        for lab, key, col in (("ind. 0.30", (0.30, 20, g), MUTED), ("corr. 0.30", ("corr", 20, g), VIOLET), ("loc. 1.2", ("loc", 20, g), ALT)):
            P = samples.get(key)
            if P is None:
                continue
            vp = a.violinplot([P], positions=[k], widths=0.8, showextrema=False)
            for b in vp["bodies"]:
                b.set_facecolor(col); b.set_alpha(0.4); b.set_edgecolor(col)
            a.plot([k], [np.median(P)], "o", color=INK, ms=3.5)
            pos.append(k); labels.append(lab); k += 1
        a.text(k - 2, 1.12, g, ha="center", fontsize=8, color=INK)
        k += 0.6
    a.set_xticks(pos, labels, fontsize=6.5, rotation=45); a.set_ylim(-1.05, 1.22)
    a.axhline(0, color=GREY, lw=0.6)
    a.set_ylabel("P"); a.set_title("(d) independent vs correlated vs localised (U = 1.2), W = 8 nm", loc="left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(ROOT / "figures/s5_disorder.png", dpi=200)
    fig.savefig(ROOT / "figures/s5_disorder.pdf", dpi=600)   # vector, for the SI
    print("wrote figures/s5_disorder.png")


if __name__ == "__main__":
    main()
