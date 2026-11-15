"""Supplementary-Material figures (2026-11-14): only the panels the main figures do not show. Fig. S1 is the
validation figure written by s0_validation.py (figures/s0_validation.pdf).
Run (project root): ~/venvs/crn-te/bin/python calc/si_figures.py   ->  manuscript/figures/si/figS1..S6 (.pdf)
"""
import csv, json, shutil, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).parent))
import s1_analysis as A1
import s2_analysis as A2
import s5_analysis as A5

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "manuscript/figures/si"
UP, DN, ALT, GREY, INK, MUTED = "#e34948", "#2a78d6", "#eda100", "#a3a29d", "#0b0b0b", "#52514e"
VIOLET, AQUA = "#6250d6", "#1baf7a"
A, E_F = 0.3952, -0.09
SW = 6.7                        # SI text width (in)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "legend.fontsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6, "lines.linewidth": 1.1,
                     "lines.markersize": 3.5, "legend.frameon": False})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02, dpi=600)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)


def title(a, s):
    a.set_title(s, loc="left", pad=3)


def figS2():
    fig, a = plt.subplots(figsize=(SW * 0.55, 2.5))
    E = np.array(A1.ES); P = A1.P
    for W, mk in ((A1.W100[2], "o"), (A1.W100[-1], "s")):
        a.plot(E, [P(A1.kw("cheng", 0, "100", "up", W, e), A1.kw("cheng", 0, "100", "dn", W, e)) for e in A1.ES], mk,
               color=INK if W == A1.W100[-1] else MUTED, label=f"Kwant, {W * A:.0f} nm")
    a.plot(E, [P(A1.fs("cheng", 0, "100", "up", e), A1.fs("cheng", 0, "100", "dn", e)) for e in A1.ES], color=INK, label="wide limit, Eq. (2)")
    a.plot(E, [P(A1.fs("cheng", 0.08, "100", "up", e), A1.fs("cheng", 0.08, "100", "dn", e)) for e in A1.ES], color=VIOLET, ls="--",
           label="wide limit, d$_{xy}$ +0.08 eV")
    a.plot(E, [P(A1.fs("single_band", 0, "100", "up", e), A1.fs("single_band", 0, "100", "dn", e)) for e in A1.ES], color=ALT,
           ls=(0, (4, 2)), label="single-band fit")
    a.axvline(E_F, color=INK, lw=0.6, ls=":")
    a.set_xlabel("E (eV)"); a.set_ylabel(r"$P=(N_\uparrow-N_\downarrow)/(N_\uparrow+N_\downarrow)$")
    a.legend(loc="upper left")
    save(fig, "figS2_straight_P")


def figS3():
    fig, ax = plt.subplots(1, 2, figsize=(SW, 2.7), gridspec_kw=dict(wspace=0.3, width_ratios=[1.25, 1]))
    WS, ES = A2.WS, A2.ES
    a = ax[0]; rng = np.random.default_rng(1)
    cats = [("L", None), ("Lc4", None), ("Lc2", None), ("T", "up"), ("T", "dn"), ("B45", None)]
    names = ["L\nsharp", "L\nW/4", "L\nW/2", "T arm\n↑", "T arm\n↓", "45°\nbend"]
    for i, (g, sp) in enumerate(cats):
        spins = (sp,) if sp else ("up", "dn")
        ks = np.array([A2.kappa(g, "cheng", s, W, E) for s in spins for W in WS for E in ES]); m = ks.mean()
        if g != "B45":
            a.fill_between([i - 0.36, i + 0.36], 0.9 * m, 1.1 * m, color=GREY, alpha=0.3, lw=0)
        a.scatter(i + rng.uniform(-0.26, 0.26, ks.size), ks, s=4, color=UP if sp == "up" else DN if sp == "dn" else INK, lw=0,
                  zorder=3, label="realistic" if i == 0 else None)
        for model, col, mk in (("iso", GREY, "D"), ("single_band", ALT, "s")):
            if model == "single_band" and g in ("Lc4", "Lc2"):
                continue
            sp_m = ("up",) if model == "iso" else spins
            kr = np.array([A2.kappa(g, model, s, W, E) for s in sp_m for W in WS for E in ES]); kr = kr[np.isfinite(kr)]
            a.errorbar(i + 0.42, kr.mean(), yerr=[[kr.mean() - kr.min()], [kr.max() - kr.mean()]], fmt=mk, color=col, ms=3,
                       lw=0.8, capsize=1.5, label=("isotropic" if model == "iso" else "single-band") if i == 0 else None)
    a.set_xticks(range(len(cats)), names, fontsize=7); a.set_ylim(0.0, 1.08)
    a.set_ylabel(r"$\kappa=T_\sigma/N_{\min}$"); title(a, "(a) κ, all geometries (grey: ±10 % of mean)")
    a.legend(loc="lower left", ncol=3)
    a = ax[1]
    W = np.array(WS)
    for sh, col, lab in ((0.0, INK, "realistic"), (0.08, VIOLET, "d$_{xy}$ +0.08 eV")):
        P = [(lambda u, d: (u - d) / (u + d))(A2.total("B45", "cheng", "up", w, E_F, sh)["T1"], A2.total("B45", "cheng", "dn", w, E_F, sh)["T1"]) for w in WS]
        a.plot(W * A, P, "-o", color=col, label=lab)
    a.set_xlabel("width (nm)"); a.set_ylabel("P at $E_F$"); a.set_ylim(0, 0.35)
    title(a, "(b) 45° [100]→[110] bend"); a.legend(loc="upper right")
    save(fig, "figS3_corners")


def figS4():
    S3 = json.load(open(ROOT / "data/s3_summary.json")); C = json.load(open(ROOT / "data/s3f_lambda_correction.json"))
    fig, ax = plt.subplots(1, 2, figsize=(SW, 2.8), gridspec_kw=dict(wspace=0.75, width_ratios=[1, 1.05]))
    a = ax[0]
    M = np.load(ROOT / "data/s3_spin_accumulation.npy")
    cmap = LinearSegmentedColormap.from_list("spin", [DN, "#e9e9e6", UP]); vmax = np.nanmax(np.abs(M))
    im = a.imshow(M, origin="lower", cmap=cmap, vmin=-vmax, vmax=vmax, extent=(-40, 20, 0, 60))
    a.set_aspect("equal"); a.set_xlabel("x (cells; W = 20)"); a.set_ylabel("y (cells)")
    cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.03); cb.set_label(r"$\mu_s/V$")
    sf = S3["spin relaxation (W = 20)"]
    a.text(-38, 57, f"max |μ$_s$|/V: {sf['no spin flip']['max_abs_mu_s']:.2f} (no spin flip)\n"
                    f"{sf['lambda_sf/W = 5']['max_abs_mu_s']:.2f} (λ$_{{sf}}$ = 5W), {sf['lambda_sf/W = 1']['max_abs_mu_s']:.2f} (λ$_{{sf}}$ = W)",
           fontsize=6.8, va="top", color=INK)
    title(a, "(a) symmetric L, no spin relaxation")
    a = ax[1]
    for geom, mk in (("L", "o"), ("T", "^")):
        for lam, col in ((5.0, VIOLET), (2.0, AQUA), (1.0, ALT)):
            pts = [(p["Ltot_over_lambda"], p["dP"]) for p in C["points"] if p["geometry"] == geom and p["lambda_over_W"] == lam]
            if pts:
                a.plot(*zip(*pts), mk, color=col, mfc="none" if geom == "T" else col,
                       label=f"λ$_{{sf}}$ = {lam:g}W" if geom == "L" else ("T (open)" if lam == 5.0 else None))
    x = np.linspace(0, 1.5, 50)
    a.plot(x, -0.13 * x ** 2, "-", color=INK, lw=0.9, label=r"$-0.13\,(L_{\rm tot}/\lambda_{\rm sf})^2$")
    a.axhspan(-0.05, 0.05, color=GREY, alpha=0.25, lw=0)
    a.set_xscale("log"); a.set_xlim(0.5, 12); a.set_ylim(-1.3, 0.05)
    a.set_xlabel(r"path (arms + corner) / $\lambda_{\rm sf}$"); a.set_ylabel(r"$\Delta P$ (network − Eq. (5))")
    title(a, "(b) shift of the delivered P (L: filled)"); a.legend(loc="center left", bbox_to_anchor=(0.0, 0.42))
    save(fig, "figS4_diffusive")


def figS5():
    B = list(csv.DictReader(open(ROOT / "data/s4_ballistic.csv")))
    fig, a = plt.subplots(figsize=(SW * 0.6, 2.7))
    for wp, col, mk in ((2, INK, "o"), (4, VIOLET, "s")):
        am = sorted((float(r["E"]), float(r["dV"])) for r in B if r["model"] == "cheng" and r["W"] == "40" and int(r["wp"]) == wp)
        iso = sorted((float(r["E"]), float(r["dV"])) for r in B if r["model"] == "iso" and r["W"] == "40" and int(r["wp"]) == wp)
        a.plot(*zip(*am), "-" + mk, color=col, label=f"altermagnet, w$_p$ = {wp}")
        a.plot(*zip(*iso), "--" + mk, color=col, mfc="none", lw=0.8, label=f"isotropic, w$_p$ = {wp}")
    a.axvline(E_F, color=INK, lw=0.6, ls=":"); a.set_ylim(-0.1, 1.45)
    a.set_xlabel("E (eV)"); a.set_ylabel(r"$\Delta V/V$"); title(a, "clean limit (Kwant, 16 nm): signal and background")
    a.legend(loc="upper left", ncol=2, fontsize=6.5)
    save(fig, "figS5_readout")


def figS6():
    S, cal, ens, samples, clean = A5.compute()
    fig, ax = plt.subplots(1, 3, figsize=(SW, 2.6), gridspec_kw=dict(wspace=0.42))
    a = ax[0]
    for W, ls in ((20, "-"), (38, "--")):
        for lab, col in (("fast", UP), ("slow", DN)):
            c = cal.get((W, 0.30, lab))
            if c:
                a.plot(np.array(c["L"]) * A, c["mean_lnT"], ls, marker="o", color=col, ms=3,
                       label=f"{'↑ fast' if lab == 'fast' else '↓ slow'}, {W * A:.0f} nm")
    a.axhline(0, color=GREY, lw=0.5); a.set_xlabel("disordered length L (nm)"); a.set_ylabel(r"$\langle\ln T_\sigma\rangle$")
    title(a, "(a) calibration, U = 0.30 eV"); a.legend(loc="lower left", fontsize=6.5)
    a = ax[1]
    for g, col, mk in (("straight", INK, "o"), ("L", VIOLET, "s"), ("T", AQUA, "^")):
        for W, ls in ((20, "-"), (38, "--")):
            pts = [(U, ens[(U, W, g)]["P_mean"], ens[(U, W, g)]["P_rms"]) for U in (0.25, 0.30, 0.35) if (U, W, g) in ens]
            U_, m, s = zip(*pts)
            a.errorbar(np.array(U_) + (-0.004 if W == 20 else 0.004), m, yerr=s, fmt=mk + ls, color=col, ms=3, lw=0.9, capsize=1.5,
                       label=g if W == 20 else None)
            if (g, W) in clean:
                a.plot([0.24, 0.36], [clean[(g, W)]] * 2, ls, color=col, lw=0.6, alpha=0.6)
    a.axhline(0, color=GREY, lw=0.5); a.set_xlabel("U (eV)"); a.set_ylabel("⟨P⟩ ± rms")
    title(a, "(b) mean ± rms"); a.legend(fontsize=6.5, ncol=3, loc="center", bbox_to_anchor=(0.5, 0.5), handlelength=1.2, columnspacing=0.6)
    a = ax[2]
    k = 0; pos, labels = [], []
    for g in ("straight", "L", "T"):
        for lab, key, col in (("ind.", (0.30, 20, g), MUTED), ("corr.", ("corr", 20, g), VIOLET), ("U=1.2", ("loc", 20, g), ALT)):
            P = samples.get(key)
            if P is None:
                continue
            vp = a.violinplot([P], positions=[k], widths=0.8, showextrema=False)
            for b in vp["bodies"]:
                b.set_facecolor(col); b.set_alpha(0.4); b.set_edgecolor(col)
            a.plot([k], [np.median(P)], "o", color=INK, ms=2.5)
            pos.append(k); labels.append(lab); k += 1
        a.text(k - 2, 1.17, g, ha="center", fontsize=7.5, color=INK)
        k += 0.6
    a.set_xticks(pos, labels, fontsize=6.5, rotation=60); a.set_ylim(-1.05, 1.32); a.axhline(0, color=GREY, lw=0.5)
    a.set_ylabel("P"); title(a, "(c) disorder types, 8 nm")
    save(fig, "figS6_disorder")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("figS*"):
        old.unlink()
    shutil.copy(ROOT / "figures/s0_validation.pdf", OUT / "figS1_s0_validation.pdf")
    figS2(); figS3(); figS4(); figS5(); figS6()
    print("wrote", sorted(p.name for p in OUT.glob("*.pdf")))
