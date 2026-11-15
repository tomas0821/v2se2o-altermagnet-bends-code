"""S2 analysis, GATE verdict and figure (2026-11-14). Applies the GATE criterion exactly as pre-registered in the
record ("GATE criterion: fixed 2026-11-14, before the scan"). Reads data/s2_corners.csv and data/s2_soc.csv.
Writes data/s2_summary.json and figures/s2_corners.png.
Run (project root): ~/venvs/crn-te/bin/python calc/s2_analysis.py
"""
import csv, json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
E_F = -0.09
A = 0.3952
UP, DN, ALT, GREY, INK, MUTED = "#e34948", "#2a78d6", "#eda100", "#a3a29d", "#0b0b0b", "#52514e"
POCKET, SHEET = "#6250d6", "#1baf7a"
GATED = ["L", "Lc4", "Lc2", "T"]
GNAME = {"L": "L sharp", "Lc4": "L chamfer W/4", "Lc2": "L chamfer W/2", "T": "T-junction", "B45": "45° bend"}

R = {}
for r in csv.DictReader(open(ROOT / "data/s2_corners.csv")):
    k = (r["geometry"], r["model"], float(r["xy_shift_eV"]), r["orbital"], r["spin"], int(r["W_sites"]), float(r["E_eV"]))
    R[k] = dict(n0=int(r["N_lead0_in"]), n1=int(r["N_lead1"]), n2=int(r["N_lead2"]) if r["N_lead2"] else None,
                T1=float(r["T_1from0"]), T2=float(r["T_2from0"]) if r["T_2from0"] else None)
WS = sorted({k[5] for k in R})
ES = sorted({k[6] for k in R})
ORBS = {"cheng": ["xz", "xy"], "iso": ["xz", "xy"], "single_band": ["xz"]}


def get(geom, model, orb, spin, W, E, shift=0.0):
    sp = "up" if model == "iso" else spin
    s = shift if orb == "xy" else 0.0
    return R[(geom, model, s, orb, sp, W, E)]


def total(geom, model, spin, W, E, shift=0.0, orbs=None):
    rows = [get(geom, model, o, spin, W, E, shift) for o in (orbs or ORBS[model])]
    return dict(n0=sum(r["n0"] for r in rows), n1=sum(r["n1"] for r in rows), T1=sum(r["T1"] for r in rows),
                T2=sum(r["T2"] for r in rows) if geom == "T" else None)


def bound(geom, t):
    """bottleneck: min(N_in, N_out) for two-terminal bends; per-arm min(N_stem, 2 N_arm)/2 for the T."""
    return min(t["n0"], 2 * t["n1"]) / 2 if geom == "T" else min(t["n0"], t["n1"])


def kappa(geom, model, spin, W, E, shift=0.0, orbs=None):
    t = total(geom, model, spin, W, E, shift, orbs)
    b = bound(geom, t)
    return t["T1"] / b if b > 0 else float("nan")


def main():
    S = {}
    # ---------------- GATE (pre-registered)
    gate = {}
    for g in GATED:
        ks = np.array([kappa(g, "cheng", sp, W, E) for sp in ("up", "dn") for W in WS for E in ES])
        dev = float(np.max(np.abs(ks / ks.mean() - 1)))
        verdict = "PASS" if dev <= 0.10 else ("REFRAME" if dev > 0.20 else "GREY")
        # where is the worst point?
        pts = [(sp, W, E) for sp in ("up", "dn") for W in WS for E in ES]
        worst = pts[int(np.argmax(np.abs(ks / ks.mean() - 1)))]
        gate[GNAME[g]] = dict(kappa_mean=round(float(ks.mean()), 4), kappa_min=round(float(ks.min()), 4),
                              kappa_max=round(float(ks.max()), 4), max_rel_dev=round(dev, 4), verdict=verdict,
                              worst_point=dict(spin=worst[0], W=worst[1], E=worst[2]), n_points=len(ks))
    overall = "PASS" if all(v["verdict"] == "PASS" for v in gate.values()) else (
        "REFRAME" if any(v["verdict"] == "REFRAME" for v in gate.values()) else "GREY")
    S["GATE"] = dict(per_geometry=gate, overall=overall)
    # deviation without the smallest width, for the record (not part of the verdict)
    S["GATE diagnostics: max |kappa/mean-1| excluding W=10"] = {
        GNAME[g]: round(float(np.max(np.abs((lambda a: a / a.mean() - 1)(
            np.array([kappa(g, "cheng", sp, W, E) for sp in ("up", "dn") for W in WS if W > 10 for E in ES]))))), 4)
        for g in GATED}
    # ---------------- references
    ref = {}
    for g in ("L", "Lc2", "T", "B45"):
        for model in ("cheng", "iso", "single_band"):
            if model == "single_band" and g == "Lc2":
                continue
            ks = np.array([kappa(g, model, sp, W, E) for sp in ("up", "dn") for W in WS for E in ES])
            nz = int(np.sum(~np.isfinite(ks)))   # points with no channel at all in the bottleneck arm
            ks = ks[np.isfinite(ks)]
            ref[f"{GNAME[g]} / {model}"] = [round(float(ks.mean()), 3), round(float(ks.min()), 3), round(float(ks.max()), 3)] + (
                [f"{nz} of {nz + ks.size} points have N_min = 0"] if nz else [])
    S["kappa mean [min, max] by geometry and model"] = ref
    # orbital-resolved kappa (cheng): each band against its own bottleneck
    orb = {}
    for g in ("L", "T", "B45"):
        for o in ("xz", "xy"):
            ks = np.array([kappa(g, "cheng", sp, W, E, orbs=[o]) for sp in ("up", "dn") for W in WS for E in ES])
            ks = ks[np.isfinite(ks)]
            orb[f"{GNAME[g]} / {o}"] = [round(float(ks.mean()), 3), round(float(ks.min()), 3), round(float(ks.max()), 3)]
    S["orbital-resolved kappa, cheng: mean [min, max]"] = orb
    # ---------------- exact zeros and polarisations
    for g in ("L", "Lc4", "Lc2"):
        P = [abs((total(g, "cheng", "up", W, E)["T1"] - total(g, "cheng", "dn", W, E)["T1"]) /
                 (total(g, "cheng", "up", W, E)["T1"] + total(g, "cheng", "dn", W, E)["T1"])) for W in WS for E in ES]
        S[f"symmetric {GNAME[g]}: max |P|"] = float(f"{max(P):.2e}")
    tP = []
    for W in WS:
        for E in ES:
            tu, td = total("T", "cheng", "up", W, E), total("T", "cheng", "dn", W, E)
            P = (tu["T1"] - td["T1"]) / (tu["T1"] + td["T1"])
            bu, bd = bound("T", tu), bound("T", td)
            Pr = (bu - bd) / (bu + bd)
            tP.append((W, E, P, Pr, abs(tu["T1"] - tu["T2"])))
    S["T-arm P: Kwant range"] = [round(min(x[2] for x in tP), 3), round(max(x[2] for x in tP), 3)]
    S["T-arm P: counting-rule range"] = [round(min(x[3] for x in tP), 3), round(max(x[3] for x in tP), 3)]
    S["T-arm P: max |P - P_rule|"] = round(max(abs(x[2] - x[3]) for x in tP), 3)
    S["T-junction top/bottom arm symmetry: max |T_top - T_bot|"] = float(f"{max(x[4] for x in tP):.2e}")
    p45 = {}
    for sh in (0.0, 0.08):
        P = [(total("B45", "cheng", "up", W, E, sh)["T1"] - total("B45", "cheng", "dn", W, E, sh)["T1"]) /
             (total("B45", "cheng", "up", W, E, sh)["T1"] + total("B45", "cheng", "dn", W, E, sh)["T1"]) for W in WS for E in ES]
        Pef = [p for (W, E), p in zip([(W, E) for W in WS for E in ES], P) if E == E_F]
        ks = [kappa("B45", "cheng", sp, W, E, sh) for sp in ("up", "dn") for W in WS for E in ES]
        p45[f"xy+{sh}"] = dict(P_range=[round(min(P), 3), round(max(P), 3)], P_at_EF_over_W=[round(min(Pef), 3), round(max(Pef), 3)],
                               kappa_mean=round(float(np.mean(ks)), 3), kappa_range=[round(min(ks), 3), round(max(ks), 3)])
    S["45-degree bend (cheng)"] = p45
    # pocket-shift check on L and T at W = 20
    S["pocket shift +0.08 at W=20: max |d kappa|"] = {
        GNAME[g]: round(max(abs(kappa(g, "cheng", sp, 20, E, 0.08) - kappa(g, "cheng", sp, 20, E)) for sp in ("up", "dn") for E in ES), 4)
        for g in ("L", "T")}
    # SOC
    soc = [r for r in csv.DictReader(open(ROOT / "data/s2_soc.csv"))]
    on = [r for r in soc if r["soc_on"] == "True"]; off = [r for r in soc if r["soc_on"] == "False"]
    S["SOC bound (Table 2 SOC in the corner)"] = dict(
        max_abs_P_soc_on=float(f"{max(abs(float(r['P_out'])) for r in on):.2e}"),
        max_abs_P_soc_off=float(f"{max(abs(float(r['P_out'])) for r in off):.2e}"),
        max_spin_flip_T=float(f"{max(max(float(r['T_up_to_dn']), float(r['T_dn_to_up'])) for r in on):.2e}"))
    json.dump(S, open(ROOT / "data/s2_summary.json", "w"), indent=1)
    print(json.dumps(S, indent=1))

    # ---------------- figure
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.linewidth": 0.8, "legend.frameon": False})
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.6))
    # (a) GATE strip plot
    a = ax[0, 0]
    rng = np.random.default_rng(1)
    cats = GATED + ["B45"]
    for i, g in enumerate(cats):
        ks = np.array([kappa(g, "cheng", sp, W, E) for sp in ("up", "dn") for W in WS for E in ES])
        m = ks.mean()
        if g in GATED:
            a.fill_between([i - 0.38, i + 0.38], 0.8 * m, 1.2 * m, color=GREY, alpha=0.18, lw=0)
            a.fill_between([i - 0.38, i + 0.38], 0.9 * m, 1.1 * m, color=GREY, alpha=0.35, lw=0)
        a.scatter(i + rng.uniform(-0.28, 0.28, ks.size), ks, s=9, color=INK, lw=0, zorder=3,
                  label="realistic (Cheng)" if i == 0 else None)
        for model, col, off in (("iso", GREY, 0.0), ("single_band", ALT, 0.0)):
            if model == "single_band" and g in ("Lc4", "Lc2"):
                continue
            kr = np.array([kappa(g, model, sp, W, E) for sp in ("up", "dn") for W in WS for E in ES])
            kr = kr[np.isfinite(kr)]
            a.errorbar(i + 0.42, kr.mean(), yerr=[[kr.mean() - kr.min()], [kr.max() - kr.mean()]], fmt="D" if model == "iso" else "s",
                       color=col, ms=5, lw=1.2, capsize=2, label=("isotropic reference" if model == "iso" else "single-band fit") if i == 0 else None)
        a.text(i, 1.06, gate[GNAME[g]]["verdict"] if g in GATED else "not gated", ha="center", fontsize=7.5,
               color=INK if g in GATED else MUTED, transform=a.get_xaxis_transform() if False else a.transData)
    a.set_xticks(range(len(cats)), [GNAME[g].replace(" chamfer ", "\nchamfer ") for g in cats], fontsize=7.5)
    a.set_ylabel(r"$\kappa = T_\sigma / N_{\min}$"); a.set_ylim(0.3, 1.12)
    a.set_title(f"(a) GATE: κ over W = 10–40, E$_F$ ± 0.1 eV (bands ±10 %, ±20 %) → {overall}", loc="left", fontsize=8.5)
    a.legend(fontsize=7, loc="lower left")
    # (b) orbital-resolved kappa vs E, L sharp, W = 40
    a = ax[0, 1]
    E = np.array(ES); W = WS[-1]
    a.plot(E, [kappa("L", "cheng", "up", W, e) for e in ES], "-o", color=INK, ms=3.5, lw=1.4, label="realistic, total")
    a.plot(E, [kappa("L", "cheng", "up", W, e, orbs=["xz"]) for e in ES], "-", color=SHEET, lw=1.4, label="d$_{xz}$ sheets only")
    a.plot(E, [kappa("L", "cheng", "up", W, e, orbs=["xy"]) for e in ES], "-", color=POCKET, lw=1.4, label="d$_{xy}$ pockets only")
    a.plot(E, [kappa("L", "iso", "up", W, e) for e in ES], "--", color=GREY, lw=1.4, label="isotropic reference")
    a.plot(E, [kappa("L", "single_band", "up", W, e) for e in ES], "--", color=ALT, lw=1.4, label="single-band fit")
    a.axvline(E_F, color=INK, lw=0.7, ls=":")
    a.set_xlabel("E (eV)"); a.set_ylabel(r"$\kappa$"); a.set_ylim(0.3, 1.05)
    a.set_title(f"(b) L sharp, W = {W} sites ({W*A:.0f} nm): which band turns the corner", loc="left", fontsize=8.5)
    a.legend(fontsize=7, loc="lower left", ncol=2)
    # (c) T-arm polarisation vs counting rule
    a = ax[1, 0]
    for W, mk in ((20, "o"), (40, "s")):
        Pk = [(total("T", "cheng", "up", W, e)["T1"] - total("T", "cheng", "dn", W, e)["T1"]) /
              (total("T", "cheng", "up", W, e)["T1"] + total("T", "cheng", "dn", W, e)["T1"]) for e in ES]
        a.plot(E, Pk, mk, color=INK if W == 40 else MUTED, ms=4.5, label=f"Kwant, W = {W} ({W*A:.0f} nm)")
    # the counting rule with one kappa: per-arm bounds b_up = N_y, b_dn = N_y / 2 whenever N_x >= 2 N_y -> exactly 1/3
    a.axhline(1 / 3, color=GREY, lw=1.2, label="counting rule, one κ (= ⅓)")
    ku = np.mean([kappa("T", "cheng", "up", W, e) for W in WS if W >= 20 for e in ES])
    kd = np.mean([kappa("T", "cheng", "dn", W, e) for W in WS if W >= 20 for e in ES])
    a.axhline((2 * ku - kd) / (2 * ku + kd), color=INK, lw=1.0, ls="--",
              label=f"spin-resolved κ (κ↑ = {ku:.2f}, κ↓ = {kd:.2f}; post hoc)")
    a.axvline(E_F, color=INK, lw=0.7, ls=":")
    a.set_xlabel("E (eV)"); a.set_ylabel(r"T-arm $P$"); a.set_ylim(0.25, 0.345)
    a.set_title("(c) T-junction arm polarisation vs the counting rule", loc="left", fontsize=8.5)
    a.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0.0, 0.84))
    # (d) 45-degree bend: polarisation with pocket-shift band
    a = ax[1, 1]
    for W, col in ((20, MUTED), (40, INK)):
        P0 = [(lambda u, d: (u - d) / (u + d))(total("B45", "cheng", "up", W, e)["T1"], total("B45", "cheng", "dn", W, e)["T1"]) for e in ES]
        P8 = [(lambda u, d: (u - d) / (u + d))(total("B45", "cheng", "up", W, e, 0.08)["T1"], total("B45", "cheng", "dn", W, e, 0.08)["T1"]) for e in ES]
        a.fill_between(E, P0, P8, color=POCKET, alpha=0.15, lw=0)
        a.plot(E, P0, "-o", color=col, ms=3.5, lw=1.3, label=f"W = {W} ({W*A:.0f} nm)")
        a.plot(E, P8, "--", color=col, lw=1.0)
    a.plot([], [], "--", color=MUTED, lw=1.0, label="d$_{xy}$ +0.08 eV (band shaded)")
    a.axvline(E_F, color=INK, lw=0.7, ls=":")
    a.set_xlabel("E (eV)"); a.set_ylabel(r"$P = (T_\uparrow - T_\downarrow)/(T_\uparrow + T_\downarrow)$")
    a.set_title("(d) 45° [100]→[110] bend polarisation", loc="left", fontsize=8.5)
    a.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(ROOT / "figures/s2_corners.png", dpi=200)
    fig.savefig(ROOT / "figures/s2_corners.pdf")   # vector, for the manuscript
    print("wrote figures/s2_corners.png")


if __name__ == "__main__":
    main()
