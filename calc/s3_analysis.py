"""S3 figure (2026-11-14). Reads data/s3_boltzmann.csv, data/s3_network.csv, data/s3_summary.json,
data/s3_spin_accumulation.npy (written by s3_diffusive.py). Writes figures/s3_diffusive.png.
Run (project root): ~/venvs/crn-te/bin/python calc/s3_analysis.py
"""
import csv, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import NullFormatter, NullLocator

ROOT = Path(__file__).resolve().parent.parent
UP, DN, ALT, GREY, INK, MUTED = "#e34948", "#2a78d6", "#eda100", "#a3a29d", "#0b0b0b", "#52514e"
POCKET = "#6250d6"
E_F = -0.09
S = json.load(open(ROOT / "data/s3_summary.json"))
m75 = S["mean-field m(T/T_N = 0.75), S = 1"]
Pstr = lambda r: (r - 1) / (r + 1)


def main():
    B = [r for r in csv.DictReader(open(ROOT / "data/s3_boltzmann.csv")) if r["compound"] == "KV2Se2O"]
    curve = {}
    for r in B:
        curve.setdefault(float(r["tau_xy_over_tau_xz"]), []).append((float(r["E_eV"]), float(r["r"])))
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.linewidth": 0.8, "legend.frameon": False})
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.8))
    # (a) straight-ribbon diffusive polarisation with sensitivity bands
    a = ax[0, 0]
    E = np.array([e for e, _ in sorted(curve[1.0])])
    P1 = np.array([Pstr(r) for _, r in sorted(curve[1.0])])
    Plo = np.array([Pstr(r) for _, r in sorted(curve[2.0])]); Phi = np.array([Pstr(r) for _, r in sorted(curve[0.5])])
    a.fill_between(E, Plo, Phi, color=GREY, alpha=0.35, lw=0, label="τ$_{xy}$/τ$_{xz}$ = 0.5–2")
    a.plot(E, P1, "-o", color=INK, ms=3.5, lw=1.5, label="constant τ, 300 K")
    a.plot(E, m75 * P1, "--", color=POCKET, lw=1.3, label=f"reduced order m = {m75:.2f} (T/T$_N$ = 0.75)")
    a.axvline(E_F, color=INK, lw=0.7, ls=":")
    a.set_xlabel("E (eV)"); a.set_ylabel(r"$P_{\rm str} = (r-1)/(r+1)$,  $r = \sigma_{xx}/\sigma_{yy}$")
    a.set_ylim(0.4, 0.95); a.set_xticks([-0.2, -0.15, -0.1, -0.05, 0.0])
    a.set_title(f"(a) diffusive straight-ribbon polarisation, KV$_2$Se$_2$O (r = {S['KV2Se2O 300 K, E_F: r, P_str'][0]:.1f} at E$_F$)",
                loc="left", fontsize=8.5)
    a.legend(fontsize=7, loc="upper left")
    a.text(0.98, 0.04, "RbV$_2$Te$_2$O check: Cheng Fig. 4c (r = 4.32)\nreproduced at E = −0.10 eV (E$_F$ not stated)",
           transform=a.transAxes, ha="right", va="bottom", fontsize=7, color=MUTED)
    # network rows
    N = [r for r in csv.DictReader(open(ROOT / "data/s3_network.csv"))]
    r0 = S["network: r used (KV2Se2O, E_F, 300 K)"]
    Lc = S["L-bend corner term"]["L_c_over_W"]
    dT, LT = S["T-junction terms"]["d_T_over_W"], S["T-junction terms"]["L_T_over_W"]
    # (b) L-bend design chart
    a = ax[0, 1]
    ratios = np.logspace(0, np.log10(6), 100)
    for totW, mk, col in ((4, "o", MUTED), (8, "s", INK)):
        pts = [(int(r["arm1_or_stem"]) / int(r["arm2_or_arm"]), float(r["P_network"])) for r in N
               if r["geometry"] == "Lchart" and int(r["arm1_or_stem"]) + int(r["arm2_or_arm"]) == totW * 20]
        a.plot(*zip(*pts), mk, color=col, ms=5, label=f"network, L1+L2 = {totW}W")
        lam = (ratios - 1) / (ratios + 1)
        a.plot(ratios, Pstr(r0) * lam / (1 + Lc / totW), "-", color=col, lw=1.1)
    a.plot(ratios, Pstr(r0) * (ratios - 1) / (ratios + 1), ":", color=GREY, lw=1.3, label="long-arm limit")
    a.plot([], [], "-", color=MUTED, lw=1.1, label=f"closed form, L$_c$ = {Lc:.2f} W")
    lam_rows = [r for r in csv.DictReader(open(ROOT / "data/s3b_lambda.csv"))]
    for lam, col, mk in (("5", POCKET, "v"), ("2", "#1baf7a", "^"), ("1", ALT, "D")):
        pts = sorted((float(r["ratio"]), float(r["P_output"])) for r in lam_rows if r["geometry"] == "L" and r["lambda_sf_over_W"] == lam)
        a.plot(*zip(*pts), "--" + mk, color=col, ms=4.5, lw=1.1, label=f"λ$_{{sf}}$ = {lam}W, L1+L2 = 4W (output P)")
    a.axhline(0, color=GREY, lw=0.6)
    a.set_xscale("log"); a.xaxis.set_minor_locator(NullLocator()); a.set_xticks([1, 4 / 3, 2, 3, 4, 6], ["1", "4/3", "2", "3", "4", "6"])
    a.set_xlabel("arm ratio L1 / L2  ([100] arm / [010] arm)"); a.set_ylabel("$P_L$")
    a.set_title("(b) L-bend: P$_L$ = P$_{str}$ (L1 − L2)/(L1 + L2 + L$_c$) for path ≪ λ$_{sf}$", loc="left", fontsize=8.5)
    a.legend(fontsize=6.5, loc="upper left"); a.set_ylim(-0.75, 0.75)
    # (c) T-junction design chart
    a = ax[1, 0]
    rr = np.logspace(np.log10(0.25), np.log10(4), 100)
    for LaW, mk, col in ((2, "o", MUTED), (4, "s", INK)):
        pts = [(int(r["arm1_or_stem"]) / int(r["arm2_or_arm"]), float(r["P_network"])) for r in N
               if r["geometry"] == "Tchart" and int(r["arm2_or_arm"]) == LaW * 20]
        a.plot(*zip(*pts), mk, color=col, ms=5, label=f"network, L$_a$ = {LaW}W")
        Ls = rr * LaW
        a.plot(rr, Pstr(r0) * (2 * Ls - LaW + dT) / (2 * Ls + LaW + LT), "-", color=col, lw=1.1)
    a.plot(rr, Pstr(r0) * (2 * rr - 1) / (2 * rr + 1), ":", color=GREY, lw=1.3, label="long-arm limit")
    for lam, col, mk in (("5", POCKET, "v"), ("2", "#1baf7a", "^"), ("1", ALT, "D")):
        pts = sorted((float(r["ratio"]), float(r["P_output"])) for r in lam_rows if r["geometry"] == "T" and r["lambda_sf_over_W"] == lam)
        a.plot(*zip(*pts), "--" + mk, color=col, ms=4.5, lw=1.1, label=f"λ$_{{sf}}$ = {lam}W, L$_a$ = 2W (arm P)")
    a.axhline(0, color=GREY, lw=0.6)
    a.set_xscale("log"); a.xaxis.set_minor_locator(NullLocator()); a.set_xticks([0.25, 0.5, 1, 2, 4], ["1/4", "1/2", "1", "2", "4"])
    a.set_xlabel("stem / arm length  L$_s$ / L$_a$"); a.set_ylabel("T-arm $P_T$")
    a.set_title(f"(c) T-junction: P$_T$ = P$_{{str}}$ (2L$_s$ − L$_a$ + {dT:.2f}W)/(2L$_s$ + L$_a$ + {LT:.2f}W)",
                loc="left", fontsize=8.5)
    a.legend(fontsize=6.5, loc="upper left"); a.set_ylim(-0.75, 0.85)
    # (d) spin accumulation map
    a = ax[1, 1]
    M = np.load(ROOT / "data/s3_spin_accumulation.npy")
    cmap = LinearSegmentedColormap.from_list("spin", [DN, "#e9e9e6", UP])
    vmax = np.nanmax(np.abs(M))
    im = a.imshow(M, origin="lower", cmap=cmap, vmin=-vmax, vmax=vmax, extent=(-40, 20, 0, 60))
    a.set_aspect("equal")
    a.set_xlabel("x (cells; W = 20)"); a.set_ylabel("y (cells)")
    cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.03)
    cb.set_label(r"$\mu_s/V = (\phi_\uparrow - \phi_\downarrow)/V$")
    sf = S["spin relaxation (W = 20)"]
    a.set_title("(d) symmetric L (net P = 0): corner spin accumulation", loc="left", fontsize=8.5)
    a.text(-38, 56, f"max |μ$_s$|/V = {sf['no spin flip']['max_abs_mu_s']:.2f} (no spin flip)\n"
                    f"{sf['lambda_sf/W = 5']['max_abs_mu_s']:.2f} for λ$_{{sf}}$ = 5W, {sf['lambda_sf/W = 1']['max_abs_mu_s']:.2f} for λ$_{{sf}}$ = W",
           fontsize=7, va="top", color=INK)
    a.text(-38, 3, "[100] arm (contact V)", fontsize=7, color=MUTED)
    a.text(1, 57, "[010] arm\n(contact 0)", fontsize=7, color=MUTED, va="top")
    fig.tight_layout()
    fig.savefig(ROOT / "figures/s3_diffusive.png", dpi=200)
    fig.savefig(ROOT / "figures/s3_diffusive.pdf", dpi=600)   # vector (the mu_s map is embedded at 600 dpi)
    print("wrote figures/s3_diffusive.png")


if __name__ == "__main__":
    main()
