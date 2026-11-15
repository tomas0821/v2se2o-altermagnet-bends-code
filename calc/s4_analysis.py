"""S4 figure (2026-11-14). Reads data/s4_*.{csv,json,npy}. Writes figures/s4_readout.png.
Run (project root): ~/venvs/crn-te/bin/python calc/s4_analysis.py
"""
import csv, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
UP, DN, ALT, GREY, INK, MUTED = "#e34948", "#2a78d6", "#eda100", "#a3a29d", "#0b0b0b", "#52514e"
VIOLET, AQUA = "#6250d6", "#1baf7a"
E_F = -0.09


def main():
    S = json.load(open(ROOT / "data/s4_summary.json"))
    D = [r for r in csv.DictReader(open(ROOT / "data/s4_diffusive.csv"))]
    B = [r for r in csv.DictReader(open(ROOT / "data/s4_ballistic.csv"))]
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.linewidth": 0.8, "legend.frameon": False})
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.8))
    # (a) geometry + mu_s map with probes attached
    a = ax[0, 0]
    M = np.load(ROOT / "data/s4_mu_s_with_probes.npy")
    W, Larm, Lp = 20, 40, 10
    cmap = LinearSegmentedColormap.from_list("spin", [DN, "#e9e9e6", UP])
    vmax = np.nanmax(np.abs(M))
    im = a.imshow(M, origin="lower", cmap=cmap, vmin=-vmax, vmax=vmax,
                  extent=(-Larm, W + Lp, -Lp, W + Larm))
    a.set_aspect("equal")
    cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.03); cb.set_label(r"$\mu_s/V$")
    a.annotate("V$_{[100]}$ probe", xy=(W + Lp - 1, 2), xytext=(W - 2, 26), fontsize=7.5, color=INK,
               arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    a.annotate("V$_{[010]}$ probe", xy=(W - 2, -Lp + 1), xytext=(-30, -7), fontsize=7.5, color=INK,
               arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    a.text(-38, 3, "in (V)", fontsize=7, color=MUTED); a.text(1, W + Larm - 4, "out (0)", fontsize=7, color=MUTED)
    a.set_xlabel("x (cells)"); a.set_ylabel("y (cells)")
    a.set_title("(a) symmetric L (net P = 0) with two orthogonal corner probes", loc="left", fontsize=8.5)
    # (b) diffusive signal vs probe width
    a = ax[0, 1]
    for lam, col, mk in (("inf", INK, "o"), ("5", VIOLET, "v"), ("2", AQUA, "^"), ("1", ALT, "D")):
        rows = sorted((int(r["wp"]), float(r["dV_AM"])) for r in D if r["lambda_sf_over_W"] == lam)
        a.plot(*zip(*rows), "-" + mk, color=col, ms=5, lw=1.3,
               label="no spin flip" if lam == "inf" else f"λ$_{{sf}}$ = {lam}W")
    iso = sorted((int(r["wp"]), float(r["dV_iso"])) for r in D if r["lambda_sf_over_W"] == "inf")
    a.plot(*zip(*iso), "--s", color=GREY, ms=4, lw=1.1, label="isotropic device (charge background)")
    a.axhline(0, color=GREY, lw=0.6)
    a.text(8, 0.015, "straight ribbon: 0 (exact)", fontsize=7, color=MUTED, ha="right")
    a.set_xscale("log", base=2); a.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
    a.set_xlabel("probe width w$_p$ (cells; W = 20)"); a.set_ylabel(r"$\Delta V/V = (V_{[100]} - V_{[010]})/V$")
    a.set_ylim(-0.05, 0.55)
    a.set_title("(b) diffusive (room temperature): probe-pair signal", loc="left", fontsize=8.5)
    a.legend(fontsize=7, loc="lower left")
    # (c) ballistic signal vs E
    a = ax[1, 0]
    for wp, col, mk in ((2, INK, "o"), (4, VIOLET, "s")):
        am = sorted((float(r["E"]), float(r["dV"])) for r in B if r["model"] == "cheng" and r["W"] == "40" and int(r["wp"]) == wp)
        iso = sorted((float(r["E"]), float(r["dV"])) for r in B if r["model"] == "iso" and r["W"] == "40" and int(r["wp"]) == wp)
        a.plot(*zip(*am), "-" + mk, color=col, ms=4.5, lw=1.3, label=f"altermagnet, w$_p$ = {wp}")
        a.plot(*zip(*iso), "--" + mk, color=col, ms=3.5, lw=1.0, mfc="none", label=f"isotropic, w$_p$ = {wp}")
    a.axvline(E_F, color=INK, lw=0.7, ls=":")
    a.set_xlabel("E (eV)"); a.set_ylabel(r"$\Delta V/V$")
    a.set_title("(c) ballistic (Kwant, W = 40 sites): signal and line-of-sight background", loc="left", fontsize=8.5)
    a.legend(fontsize=7, loc="upper left", ncol=2); a.set_ylim(-0.1, 1.15); a.set_xticks([-0.2, -0.15, -0.1, -0.05, 0.0])
    # (d) signal at 10 mV bias: probe pair vs ISHE
    a = ax[1, 1]
    ish = S["ISHE estimate vs probe pair (lambda_sf = 5W)"]
    bias = ish["parameters"]["bias_mV"]
    d5 = [float(r["dV_spin"]) * bias * 1e3 for r in D if r["lambda_sf_over_W"] == "5"]
    bal = [float(r["dV"]) * bias * 1e3 for r in B if r["model"] == "cheng" and r["E"] == str(E_F)]
    items = [("probe pair, diffusive\n(λ$_{sf}$ = 5W, w$_p$ = 1–8)", min(d5), max(d5), INK),
             ("probe pair, ballistic\n(E$_F$, W = 20–40)", min(bal), max(bal), VIOLET),
             ("Pt ISHE pad\n(θ$_{SH}$ = 0.1, 5 nm Pt)", ish["V_ISHE_uV"], ish["V_ISHE_uV"], ALT)]
    for i, (lab, lo, hi, col) in enumerate(items):
        a.plot([lo, hi], [i, i], "-", color=col, lw=6, solid_capstyle="round")
        a.plot([lo, hi], [i, i], "o", color=col, ms=6)
        fmt = (lambda v: f"{v / 1e3:.1f}") if lo >= 1e3 else (lambda v: f"{v:.2g}")
        unit = "mV" if lo >= 1e3 else "µV"
        a.text(hi * 1.8, i, (f"{fmt(lo)}–{fmt(hi)} {unit}" if hi != lo else f"{fmt(lo)} {unit}"), va="center", fontsize=7.5, color=INK)
    a.set_xscale("log"); a.set_xlim(0.03, 3e5)
    a.set_yticks(range(len(items)), [it[0] for it in items], fontsize=7.5); a.set_ylim(-0.6, len(items) - 0.4)
    a.set_xlabel(f"readout signal at {bias:.0f} mV bias (µV)")
    a.set_title(f"(d) probe pair vs Pt ISHE pad: ~{ish['ratio_probe_over_ISHE'] / 1e4:.0f}×10$^4$ larger", loc="left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(ROOT / "figures/s4_readout.png", dpi=200)
    fig.savefig(ROOT / "figures/s4_readout.pdf", dpi=600)   # vector, for the SI
    print("wrote figures/s4_readout.png")


if __name__ == "__main__":
    main()
