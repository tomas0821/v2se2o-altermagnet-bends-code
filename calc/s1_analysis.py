"""S1 analysis + figure (2026-11-14). Reads data/s1_kwant_modes.csv and data/s1_fermi_surface_counts.csv
(written by s1_channels.py). Writes data/s1_summary.json and figures/s1_channels.png.
Run (project root): ~/venvs/crn-te/bin/python calc/s1_analysis.py
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
POCKET, SHEET = "#6250d6", "#1baf7a"   # violet / aqua; validated all-pairs with ALT (dataviz validator)

K = {}   # (model, shift, dir, spin, orb, W, E) -> N
for r in csv.DictReader(open(ROOT / "data/s1_kwant_modes.csv")):
    K[(r["model"], float(r["xy_shift_eV"]), r["direction"], r["spin"], r["orbital"], int(r["width_sites_or_rows"]),
       float(r["E_eV"]))] = int(r["N"])
F = {}   # (model, shift, dir, spin, orb, E) -> (sharvin, projected) per row
for r in csv.DictReader(open(ROOT / "data/s1_fermi_surface_counts.csv")):
    F[(r["model"], float(r["xy_shift_eV"]), r["direction"], r["spin"], r["orbital"], float(r["E_eV"]))] = (
        float(r["sharvin_per_row"]), float(r["projected_per_row"]))
ES = sorted({k[-1] for k in K})
W100 = sorted({k[5] for k in K if k[2] == "100"})
W110 = sorted({k[5] for k in K if k[2] == "110"})


def orbs(model):
    return ["xz", "xy"] if model == "cheng" else ["xz"]


def kw(model, shift, d, spin, W, E, orb=None):
    """total (or one orbital's) Kwant count; for the shifted case the d_xz band is the unshifted one."""
    tot = 0
    for o in ([orb] if orb else orbs(model)):
        s = shift if o == "xy" else 0.0
        tot += K[(model, s, d, spin, o, W, E)]
    return tot


def fs(model, shift, d, spin, E, which=1, orb=None):
    tot = 0.0
    for o in ([orb] if orb else orbs(model)):
        s = shift if o == "xy" else 0.0
        tot += F[(model, s, d, spin, o, E)][which]
    return tot


def P(nu, nd):
    return (nu - nd) / (nu + nd) if nu + nd > 0 else float("nan")


def main():
    S = {}
    ef = min(ES, key=lambda e: abs(e - E_F))
    # 1. [100] Kwant vs Fermi-surface prediction
    rows = []
    for model in ("cheng", "single_band"):
        for spin in ("up", "dn"):
            for W in W100:
                for E in ES:
                    n = kw(model, 0.0, "100", spin, W, E)
                    rows.append((model, n, n - fs(model, 0.0, "100", spin, E) * W, n - fs(model, 0.0, "100", spin, E, 0) * W))
    for model in ("cheng", "single_band"):
        d_pr = np.array([r[2] for r in rows if r[0] == model]); d_sh = np.array([r[3] for r in rows if r[0] == model])
        S[f"[100] {model}: Kwant - FS prediction (all W, E)"] = dict(
            projected_mean_abs=round(float(np.abs(d_pr).mean()), 3), projected_max_abs=round(float(np.abs(d_pr).max()), 3),
            sharvin_mean_abs=round(float(np.abs(d_sh).mean()), 3), sharvin_max_abs=round(float(np.abs(d_sh).max()), 3),
            n_points=len(d_pr))
    S["[100] cheng at E_F"] = {f"W={W} ({W*A:.1f} nm)": dict(
        N_up=kw("cheng", 0, "100", "up", W, ef), N_dn=kw("cheng", 0, "100", "dn", W, ef),
        N_up_xz=kw("cheng", 0, "100", "up", W, ef, "xz"), N_up_xy=kw("cheng", 0, "100", "up", W, ef, "xy"),
        N_dn_xz=kw("cheng", 0, "100", "dn", W, ef, "xz"), N_dn_xy=kw("cheng", 0, "100", "dn", W, ef, "xy"),
        P=round(P(kw("cheng", 0, "100", "up", W, ef), kw("cheng", 0, "100", "dn", W, ef)), 3)) for W in W100}
    S["[100] wide limit per row at E_F"] = {m: dict(
        up=round(fs(m, 0, "100", "up", ef), 4), dn=round(fs(m, 0, "100", "dn", ef), 4),
        P=round(P(fs(m, 0, "100", "up", ef), fs(m, 0, "100", "dn", ef)), 3)) for m in ("cheng", "single_band")}
    S["[100] wide limit per row at E_F"]["cheng xy+0.08"] = dict(
        up=round(fs("cheng", 0.08, "100", "up", ef), 4), dn=round(fs("cheng", 0.08, "100", "dn", ef), 4),
        P=round(P(fs("cheng", 0.08, "100", "up", ef), fs("cheng", 0.08, "100", "dn", ef)), 3))
    for lo, hi, lab in ((-0.19, 0.01, "E_F +- 0.10"), (-0.15, 0.10, "-0.15 ... +0.10")):
        sel = [E for E in ES if lo - 1e-9 <= E <= hi + 1e-9]
        for m, sh in (("cheng", 0.0), ("cheng", 0.08), ("single_band", 0.0)):
            ps = [P(fs(m, sh, "100", "up", E), fs(m, sh, "100", "dn", E)) for E in sel]
            S[f"[100] wide-limit P range, {m} xy+{sh}, E in {lab}"] = [round(min(ps), 3), round(max(ps), 3)]
    # 2. [110]
    S["[110] symmetry check: N_dn == N_up at Wd=36 for all E and bands"] = all(
        K[(m, s, "110", "dn", o, 36, E)] == K[(m, s, "110", "up", o, 36, E)]
        for (m, s, d, sp, o, W, E) in K if d == "110" and sp == "dn")
    d110 = defaultdict(list)
    for m, s in (("cheng", 0.0), ("cheng", 0.08), ("single_band", 0.0)):
        for o in orbs(m):
            if s and o != "xy":
                continue
            for W in W110:
                for E in ES:
                    n = K[(m, s, "110", "up", o, W, E)]
                    d110[(m, s, o, "proj")].append(n - F[(m, s, "110", "up", o, E)][1] * W)
                    d110[(m, s, o, "sharvin")].append(n - F[(m, s, "110", "up", o, E)][0] * W)
    S["[110] Kwant - FS prediction, mean |diff| (all Wd, E)"] = {
        f"{m} xy+{s} {o} {w}": round(float(np.abs(v).mean()), 2) for (m, s, o, w), v in d110.items()}
    S["[110] at E_F per spin"] = {f"Wd={W} ({W*A/np.sqrt(2):.1f} nm)": dict(
        N=kw("cheng", 0, "110", "up", W, ef), N_sheets_xz=kw("cheng", 0, "110", "up", W, ef, "xz"),
        N_pockets_xy=kw("cheng", 0, "110", "up", W, ef, "xy"), N_xy_plus_0p08=kw("cheng", 0.08, "110", "up", W, ef),
        N_single_band=kw("single_band", 0, "110", "up", W, ef),
        N100_same_width=dict(up=None)) for W in W110}
    for W in W110:  # [100] counts at the matching width for comparison
        W1 = min(W100, key=lambda w: abs(w * A - W * A / np.sqrt(2)))
        S["[110] at E_F per spin"][f"Wd={W} ({W*A/np.sqrt(2):.1f} nm)"]["N100_same_width"] = dict(
            W100=W1, up=kw("cheng", 0, "100", "up", W1, ef), dn=kw("cheng", 0, "100", "dn", W1, ef))
    pr = {o: fs("cheng", 0, "110", "up", ef, 1, o) for o in ("xz", "xy")}
    sh = {o: fs("cheng", 0, "110", "up", ef, 0, o) for o in ("xz", "xy")}
    S["[110] wide limit per row at E_F (per spin)"] = dict(
        projected_xz_sheets=round(pr["xz"], 4), projected_xy_pockets=round(pr["xy"], 4),
        sharvin_xz_sheets=round(sh["xz"], 4), sharvin_xy_pockets=round(sh["xy"], 4),
        pocket_fraction_projected=round(pr["xy"] / (pr["xy"] + pr["xz"]), 3),
        xy_plus_0p08_projected=round(fs("cheng", 0.08, "110", "up", ef, 1, "xy"), 4),
        single_band_projected=round(fs("single_band", 0, "110", "up", ef), 4),
        single_band_sharvin=round(fs("single_band", 0, "110", "up", ef, 0), 4),
        cheng_100_up_per_row=round(fs("cheng", 0, "100", "up", ef), 4),
        cheng_100_dn_per_row=round(fs("cheng", 0, "100", "dn", ef), 4))
    pf = [fs("cheng", 0, "110", "up", E, 1, "xy") / max(fs("cheng", 0, "110", "up", E), 1e-12) for E in ES if -0.19 <= E <= 0.01]
    S["[110] wide-limit pocket fraction over E_F +- 0.10"] = [round(min(pf), 3), round(max(pf), 3)]
    json.dump(S, open(ROOT / "data/s1_summary.json", "w"), indent=1)
    print(json.dumps(S, indent=1))

    # ---------------------------------------------------------------- figure
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.linewidth": 0.8, "legend.frameon": False})
    fig, ax = plt.subplots(2, 2, figsize=(9.6, 7.4))
    E = np.array(ES)
    Wbig = W100[-1]
    a = ax[0, 0]
    for spin, col, lab in (("up", UP, "↑"), ("dn", DN, "↓")):
        a.step(E, [kw("cheng", 0, "100", spin, Wbig, e) for e in ES], where="mid", color=col, lw=1.6, label=f"Kwant {lab}")
        a.plot(E, [fs("cheng", 0, "100", spin, e) * Wbig for e in ES], color=col, lw=0.9, ls="--", label=f"Fermi surface {lab}")
    a.axvline(E_F, color=INK, lw=0.7, ls=":")
    a.text(E_F + 0.004, 2, "$E_F$", fontsize=8)
    a.set_xlabel("E (eV)"); a.set_ylabel(r"channels $N_\sigma$")
    a.set_title(f"(a) [100] ribbon, W = {Wbig*A:.0f} nm ({Wbig} sites)", loc="left", fontsize=9)
    a.legend(fontsize=7.5, ncol=2, loc="center right")
    a.set_ylim(bottom=0)
    # (b) convergence with width at E_F
    a = ax[0, 1]
    w1 = np.array(W100) * A; w2 = np.array(W110) * A / np.sqrt(2)
    a.plot(w1, [kw("cheng", 0, "100", "up", W, ef) / W for W in W100], "o", color=UP, ms=6, label="[100] ↑")
    a.plot(w1, [kw("cheng", 0, "100", "dn", W, ef) / W for W in W100], "s", color=DN, ms=6, label="[100] ↓")
    a.plot(w2, [kw("cheng", 0, "110", "up", W, ef) / W for W in W110], "^", color=INK, ms=6, label="[110] ↑ = ↓")
    for val, col in ((fs("cheng", 0, "100", "up", ef), UP), (fs("cheng", 0, "100", "dn", ef), DN),
                     (fs("cheng", 0, "110", "up", ef), INK)):
        a.axhline(val, color=col, lw=0.9, ls="--")
    a.axhline(fs("cheng", 0, "110", "up", ef, 0), color=GREY, lw=1.2, ls=(0, (1, 1.5)))
    a.text(20.5, fs("cheng", 0, "110", "up", ef, 0) + 0.015, "[110] naive Sharvin", fontsize=7.5, color=MUTED, ha="right")
    a.set_xlabel("width (nm)"); a.set_ylabel("channels per row at $E_F$")
    a.set_title("(b) per-row counts converge to the k$_\\parallel$-conserving limit (dashed)", loc="left", fontsize=8.5)
    a.legend(fontsize=7.5, loc="upper right", bbox_to_anchor=(1.0, 0.86))
    a.set_ylim(0, 1.5)
    # (c) [110] decomposition
    a = ax[1, 0]
    Wd = W110[-1]
    xy0 = [K[("cheng", 0.0, "110", "up", "xy", Wd, e)] / Wd for e in ES]
    xy8 = [K[("cheng", 0.08, "110", "up", "xy", Wd, e)] / Wd for e in ES]
    xz = [K[("cheng", 0.0, "110", "up", "xz", Wd, e)] / Wd for e in ES]
    a.fill_between(E, xy0, xy8, step="mid", color=POCKET, alpha=0.18, lw=0, label="pockets, d$_{xy}$ +0.08 eV")
    a.step(E, xy0, where="mid", color=POCKET, lw=1.6, label="d$_{xy}$ pockets (Kwant)")
    a.step(E, xz, where="mid", color=SHEET, lw=1.6, label="d$_{xz}$ sheets (Kwant)")
    a.plot(E, [fs("cheng", 0, "110", "up", e, 1, "xy") for e in ES], color=POCKET, lw=0.9, ls="--")
    a.plot(E, [fs("cheng", 0, "110", "up", e, 1, "xz") for e in ES], color=SHEET, lw=0.9, ls="--",
           label="k$_\\parallel$-conserving limit")
    a.plot(E, [fs("cheng", 0, "110", "up", e, 0, "xz") for e in ES], color=GREY, lw=1.2, ls=(0, (1, 1.5)),
           label="d$_{xz}$ naive Sharvin")
    a.step(E, [K[("single_band", 0.0, "110", "up", "xz", Wd, e)] / Wd for e in ES], where="mid", color=ALT, lw=1.3,
           ls=(0, (4, 2)), label="single-band fit (Kwant)")
    a.axvline(E_F, color=INK, lw=0.7, ls=":")
    a.set_xlabel("E (eV)"); a.set_ylabel("channels per row, per spin")
    a.set_title(f"(c) [110] ribbon, {Wd*A/np.sqrt(2):.0f} nm ({Wd} rows): pockets dominate", loc="left", fontsize=9)
    a.legend(fontsize=7, loc="upper left", ncol=2)
    a.set_ylim(0, 0.75)
    # (d) polarisation of a straight [100] ribbon
    a = ax[1, 1]
    for W, mk in ((W100[2], "o"), (Wbig, "s")):
        a.plot(E, [P(kw("cheng", 0, "100", "up", W, e), kw("cheng", 0, "100", "dn", W, e)) for e in ES], mk,
               color=INK if W == Wbig else MUTED, ms=3.2, label=f"Kwant, {W*A:.0f} nm")
    a.plot(E, [P(fs("cheng", 0, "100", "up", e), fs("cheng", 0, "100", "dn", e)) for e in ES], color=INK, lw=1.2,
           label="wide limit")
    a.plot(E, [P(fs("cheng", 0.08, "100", "up", e), fs("cheng", 0.08, "100", "dn", e)) for e in ES], color=POCKET,
           lw=1.2, ls="--", label="wide limit, d$_{xy}$ +0.08 eV")
    a.plot(E, [P(fs("single_band", 0, "100", "up", e), fs("single_band", 0, "100", "dn", e)) for e in ES], color=ALT,
           lw=1.3, ls=(0, (4, 2)), label="single-band fit")
    a.axvline(E_F, color=INK, lw=0.7, ls=":")
    a.set_xlabel("E (eV)"); a.set_ylabel(r"$P = (N_\uparrow - N_\downarrow)/(N_\uparrow + N_\downarrow)$")
    a.set_title("(d) straight [100] ribbon polarisation", loc="left", fontsize=9)
    a.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(ROOT / "figures/s1_channels.png", dpi=200)
    fig.savefig(ROOT / "figures/s1_channels.pdf")   # vector, for the manuscript
    print("wrote figures/s1_channels.png")


if __name__ == "__main__":
    main()
