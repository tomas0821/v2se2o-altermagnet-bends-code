"""Manuscript figures at print size (2026-11-14): each figure is one 190 mm row of three panels, fonts 6.5-7.5 pt,
vector PDF (+ PNG preview). The stage scripts write their own complete figures to figures/s0-s5.
Run (project root): ~/venvs/crn-te/bin/python calc/paper_figures.py
Outputs: manuscript/figures/fig1_model ... fig5_readout_disorder (.pdf, .png); manuscript/figures/si/figS*.pdf
"""
import csv, json, shutil, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).parent))
import s0_validation as S0
import s1_analysis as A1
import s2_analysis as A2
import s5_analysis as A5
from v2x2o_tb import MODELS, E_F, band, orbital_params

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "manuscript/figures"
UP, DN, ALT, GREY, INK, MUTED = "#e34948", "#2a78d6", "#eda100", "#a3a29d", "#0b0b0b", "#52514e"
VIOLET, AQUA = "#6250d6", "#1baf7a"
PW = 7.48            # 190 mm, full page width
A = 0.3952
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 6.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6, "lines.linewidth": 1.1,
                     "lines.markersize": 3.2, "legend.frameon": False, "legend.handlelength": 1.6})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", dpi=250, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def title(a, s):
    a.set_title(s, loc="left", pad=3)


# ---------------------------------------------------------------- Fig. 1
def fig1():
    fig = plt.figure(figsize=(PW, 2.55))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 0.95, 1.35], wspace=0.32)
    a1, a2, a3 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    dft = S0.load_dots()["DFT"]
    s = np.linspace(0, 4, 1601)
    a1.scatter(dft[:, 0], dft[:, 1], s=1.2, color=GREY, lw=0, label="DFT")
    for col, spin, lab in ((UP, "up", "↑ TB"), (DN, "dn", "↓ TB")):
        for i, b in enumerate(S0.bands_on_path("cheng", spin, s, True)):
            a1.plot(s, b, color=col, lw=1.1, label=lab if i == 0 else None)
    a1.axhline(E_F, color=INK, lw=0.6, ls="--"); a1.text(4.04, E_F, "$E_F$", va="center", fontsize=6.5)
    for x in (1, 2, 3):
        a1.axvline(x, color=GREY, lw=0.4)
    a1.set_xticks([0, 1, 2, 3, 4], ["M", "X", "Γ", "Y", "M"]); a1.set_xlim(0, 4); a1.set_ylim(-1.1, 1.1)
    a1.set_ylabel("E (eV)"); title(a1, "(a) bands (red ↑, blue ↓ TB; gray DFT)")
    kg = np.linspace(-np.pi, np.pi, 401)
    KX, KY = np.meshgrid(kg, kg, indexing="xy")
    for col, spin in ((UP, "up"), (DN, "dn")):
        for orb, p in orbital_params(MODELS["cheng"], spin):
            a2.contour(KX / np.pi, KY / np.pi, band(p, KX, KY), levels=[E_F], colors=col,
                       linewidths=1.2 if orb == "xz" else 0.9, linestyles="solid" if orb == "xz" else "dotted")
    a2.set_aspect("equal"); a2.set_xlabel(r"$k_x/\pi$"); a2.set_ylabel(r"$k_y/\pi$", labelpad=1)
    a2.set_xticks([-1, 0, 1]); a2.set_yticks([-1, 0, 1]); title(a2, "(b) Fermi surface at $E_F$")
    a2.text(0.58, 0.78, "↑ d$_{xz}$", color=UP, fontsize=6.5); a2.text(-0.97, 0.66, "↓ d$_{yz}$", color=DN, fontsize=6.5)
    a2.text(0.915, 0.0, "↑ d$_{xy}$", rotation=90, color=UP, fontsize=6.5, ha="center", va="center")   # pocket at X
    a2.text(0.0, 0.915, "↓ d$_{xy}$", color=DN, fontsize=6.5, ha="center", va="center")                 # pocket at Y
    info = json.load(open(ROOT / "figures/ovito/devices.json"))
    a3.imshow(plt.imread(ROOT / "figures/ovito/devices_all.png"), extent=info["extent"], interpolation="lanczos")
    a3.set_aspect("equal"); a3.axis("off"); a3.set_title("(c) devices on the V$_2$Se$_2$O layer", loc="left", pad=16)
    for lab in info["labels"].values():
        a3.text(lab["x"], lab["y"], lab["text"], fontsize=6.5, ha="center", va="bottom")
    x0, x1, y0, y1 = info["extent"]
    for k, (col, txt) in enumerate(((UP, "V$_A$: ↑, easy [100]"), (DN, "V$_B$: ↓, easy [010]"))):
        xx = x0 + (0.02 + 0.5 * k) * (x1 - x0); yy = y0 - 0.05 * (y1 - y0)
        a3.plot([xx], [yy], "o", color=col, ms=3.5, clip_on=False)
        a3.text(xx + 0.03 * (x1 - x0), yy, txt, fontsize=6.5, va="center", clip_on=False)
    save(fig, "fig1_model")


# ---------------------------------------------------------------- Fig. 2
def fig2():
    fig, ax = plt.subplots(1, 3, figsize=(PW, 2.25), gridspec_kw=dict(wspace=0.32))
    E = np.array(A1.ES); W = A1.W100[-1]; ef = -0.09
    a = ax[0]
    for spin, col, lab in (("up", UP, "↑"), ("dn", DN, "↓")):
        a.step(E, [A1.kw("cheng", 0, "100", spin, W, e) for e in A1.ES], where="mid", color=col, lw=1.1, label=f"Kwant {lab}")
        a.plot(E, [A1.fs("cheng", 0, "100", spin, e) * W for e in A1.ES], color=col, lw=0.7, ls="--", label=f"Eq. (2) {lab}")
    a.axvline(E_F, color=INK, lw=0.5, ls=":"); a.set_ylim(0, 82)
    a.set_xlabel("E (eV)"); a.set_ylabel(r"channels $N_\sigma$"); title(a, f"(a) [100], W = {W * A:.0f} nm")
    a.legend(ncol=2, loc="center right", columnspacing=0.8)
    a = ax[1]
    w1 = np.array(A1.W100) * A; w2 = np.array(A1.W110) * A / np.sqrt(2)
    a.plot(w1, [A1.kw("cheng", 0, "100", "up", w, ef) / w for w in A1.W100], "o", color=UP, label="[100] ↑")
    a.plot(w1, [A1.kw("cheng", 0, "100", "dn", w, ef) / w for w in A1.W100], "s", color=DN, label="[100] ↓")
    a.plot(w2, [A1.kw("cheng", 0, "110", "up", w, ef) / w for w in A1.W110], "^", color=INK, label="[110] ↑ = ↓")
    for v, col in ((A1.fs("cheng", 0, "100", "up", ef), UP), (A1.fs("cheng", 0, "100", "dn", ef), DN), (A1.fs("cheng", 0, "110", "up", ef), INK)):
        a.axhline(v, color=col, lw=0.7, ls="--")
    a.axhline(A1.fs("cheng", 0, "110", "up", ef, 0), color=GREY, lw=0.9, ls=(0, (1, 1.5)))
    a.text(20.6, A1.fs("cheng", 0, "110", "up", ef, 0) + 0.03, "[110] Sharvin", fontsize=6.5, color=MUTED, ha="right")
    a.set_ylim(0, 1.5); a.set_xlabel("width (nm)"); a.set_ylabel("channels per row at $E_F$")
    title(a, "(b) per-row counts vs Eq. (2)"); a.legend(loc="upper right", bbox_to_anchor=(1.0, 0.9))
    a = ax[2]
    Wd = A1.W110[-1]
    xy0 = [A1.K[("cheng", 0.0, "110", "up", "xy", Wd, e)] / Wd for e in A1.ES]
    xy8 = [A1.K[("cheng", 0.08, "110", "up", "xy", Wd, e)] / Wd for e in A1.ES]
    xz = [A1.K[("cheng", 0.0, "110", "up", "xz", Wd, e)] / Wd for e in A1.ES]
    a.fill_between(E, xy0, xy8, step="mid", color=VIOLET, alpha=0.18, lw=0, label="pockets, +0.08 eV")
    a.step(E, xy0, where="mid", color=VIOLET, lw=1.1, label="d$_{xy}$ pockets")
    a.step(E, xz, where="mid", color=AQUA, lw=1.1, label="d$_{xz}$ sheets")
    a.plot(E, [A1.fs("cheng", 0, "110", "up", e, 1, "xz") for e in A1.ES], color=AQUA, lw=0.7, ls="--", label="Eq. (2)")
    a.plot(E, [A1.fs("cheng", 0, "110", "up", e, 1, "xy") for e in A1.ES], color=VIOLET, lw=0.7, ls="--")
    a.plot(E, [A1.fs("cheng", 0, "110", "up", e, 0, "xz") for e in A1.ES], color=GREY, lw=0.9, ls=(0, (1, 1.5)), label="sheets, Sharvin")
    a.step(E, [A1.K[("single_band", 0.0, "110", "up", "xz", Wd, e)] / Wd for e in A1.ES], where="mid", color=ALT, lw=0.9,
           ls=(0, (4, 2)), label="single-band fit")
    a.axvline(E_F, color=INK, lw=0.5, ls=":"); a.set_ylim(0, 0.78)
    a.set_xlabel("E (eV)"); a.set_ylabel("channels per row, per spin"); title(a, f"(c) [110], {Wd * A / np.sqrt(2):.0f} nm")
    a.legend(ncol=2, loc="upper left", columnspacing=0.8)
    save(fig, "fig2_channels")


# ---------------------------------------------------------------- Fig. 3
def fig3():
    fig, ax = plt.subplots(1, 3, figsize=(PW, 2.25), gridspec_kw=dict(wspace=0.34, width_ratios=[1.15, 1, 1]))
    WS, ES = A2.WS, A2.ES
    a = ax[0]
    rng = np.random.default_rng(1)
    cats = ["L", "Lc4", "Lc2", "T"]
    names = {"L": "L sharp", "Lc4": "L, W/4", "Lc2": "L, W/2", "T": "T arm"}
    for i, g in enumerate(cats):
        ks = np.array([A2.kappa(g, "cheng", sp, W, E) for sp in ("up", "dn") for W in WS for E in ES]); m = ks.mean()
        a.fill_between([i - 0.36, i + 0.36], 0.8 * m, 1.2 * m, color=GREY, alpha=0.15, lw=0)
        a.fill_between([i - 0.36, i + 0.36], 0.9 * m, 1.1 * m, color=GREY, alpha=0.3, lw=0)
        if g == "T":       # the T arm needs two efficiencies: split by spin
            for sp, col, off in (("up", UP, -0.13), ("dn", DN, 0.13)):
                kk = np.array([A2.kappa(g, "cheng", sp, W, E) for W in WS for E in ES])
                a.scatter(i + off + rng.uniform(-0.12, 0.12, kk.size), kk, s=3, color=col, lw=0, zorder=3)
        else:
            a.scatter(i + rng.uniform(-0.26, 0.26, ks.size), ks, s=3, color=INK, lw=0, zorder=3, label="realistic" if i == 0 else None)
        for model, col, mk in (("iso", GREY, "D"), ("single_band", ALT, "s")):
            if model == "single_band" and g in ("Lc4", "Lc2"):
                continue
            kr = np.array([A2.kappa(g, model, sp, W, E) for sp in ("up", "dn") for W in WS for E in ES]); kr = kr[np.isfinite(kr)]
            a.errorbar(i + 0.42, kr.mean(), yerr=[[kr.mean() - kr.min()], [kr.max() - kr.mean()]], fmt=mk, color=col, ms=3,
                       lw=0.8, capsize=1.5, label=("isotropic" if model == "iso" else "single-band") if i == 0 else None)
    a.set_xticks(range(len(cats)), [names[g] for g in cats], fontsize=6.5); a.set_ylim(0.4, 1.08)
    a.set_ylabel(r"$\kappa = T_\sigma/N_{\min}$"); title(a, "(a) corner efficiency (±10 %, ±20 %)")
    a.legend(loc="lower left", ncol=3, columnspacing=0.6, handletextpad=0.2)
    a = ax[1]
    E = np.array(ES); W = WS[-1]
    a.plot(E, [A2.kappa("L", "cheng", "up", W, e) for e in ES], "-o", color=INK, ms=2.5, label="realistic")
    a.plot(E, [A2.kappa("L", "cheng", "up", W, e, orbs=["xz"]) for e in ES], "-", color=AQUA, label="d$_{xz}$ sheets")
    a.plot(E, [A2.kappa("L", "cheng", "up", W, e, orbs=["xy"]) for e in ES], "-", color=VIOLET, label="d$_{xy}$ pockets")
    a.plot(E, [A2.kappa("L", "iso", "up", W, e) for e in ES], "--", color=GREY, label="isotropic")
    a.plot(E, [A2.kappa("L", "single_band", "up", W, e) for e in ES], "--", color=ALT, label="single-band")
    a.axvline(E_F, color=INK, lw=0.5, ls=":"); a.set_ylim(0.15, 1.05)
    a.set_xlabel("E (eV)"); a.set_ylabel(r"$\kappa$"); title(a, f"(b) L sharp, W = {W * A:.0f} nm")
    a.legend(loc="lower left", ncol=2, columnspacing=0.5, handlelength=1.4)
    a = ax[2]
    for W, mk, col in ((20, "o", MUTED), (40, "s", INK)):
        Pk = [(A2.total("T", "cheng", "up", W, e)["T1"] - A2.total("T", "cheng", "dn", W, e)["T1"]) /
              (A2.total("T", "cheng", "up", W, e)["T1"] + A2.total("T", "cheng", "dn", W, e)["T1"]) for e in ES]
        a.plot(E, Pk, mk, color=col, label=f"Kwant, {W * A:.0f} nm")
    a.axhline(1 / 3, color=GREY, lw=1.0, label="counting, one κ (⅓)")
    ku = np.mean([A2.kappa("T", "cheng", "up", W, e) for W in WS if W >= 20 for e in ES])
    kd = np.mean([A2.kappa("T", "cheng", "dn", W, e) for W in WS if W >= 20 for e in ES])
    a.axhline((2 * ku - kd) / (2 * ku + kd), color=INK, lw=0.8, ls="--", label=f"Eq. (4): κ↑ {ku:.3f}, κ↓ {kd:.3f}")
    a.axvline(E_F, color=INK, lw=0.5, ls=":"); a.set_ylim(0.25, 0.345)
    a.set_xlabel("E (eV)"); a.set_ylabel("T-arm P"); title(a, "(c) T-junction arm polarization")
    a.legend(loc="upper left", bbox_to_anchor=(0, 0.86))
    save(fig, "fig3_corners")


# ---------------------------------------------------------------- Fig. 4
def fig4():
    S3 = json.load(open(ROOT / "data/s3_summary.json"))
    m75 = S3["mean-field m(T/T_N = 0.75), S = 1"]; r0 = S3["network: r used (KV2Se2O, E_F, 300 K)"]
    Lc = S3["L-bend corner term"]["L_c_over_W"]; dT = S3["T-junction terms"]["d_T_over_W"]; LT = S3["T-junction terms"]["L_T_over_W"]
    Pstr = lambda r: (r - 1) / (r + 1)
    curve = {}
    for r in csv.DictReader(open(ROOT / "data/s3_boltzmann.csv")):
        if r["compound"] == "KV2Se2O":
            curve.setdefault(float(r["tau_xy_over_tau_xz"]), []).append((float(r["E_eV"]), float(r["r"])))
    N = [r for r in csv.DictReader(open(ROOT / "data/s3_network.csv"))]
    LAM = [r for r in csv.DictReader(open(ROOT / "data/s3b_lambda.csv"))]
    fig, ax = plt.subplots(1, 3, figsize=(PW, 2.25), gridspec_kw=dict(wspace=0.38))
    a = ax[0]
    E = np.array([e for e, _ in sorted(curve[1.0])]); P1 = np.array([Pstr(r) for _, r in sorted(curve[1.0])])
    a.fill_between(E, [Pstr(r) for _, r in sorted(curve[2.0])], [Pstr(r) for _, r in sorted(curve[0.5])], color=GREY,
                   alpha=0.35, lw=0, label="τ$_{xy}$/τ$_{xz}$ = 0.5–2")
    a.plot(E, P1, "-o", color=INK, ms=2.2, label="constant τ, 300 K")
    a.plot(E, m75 * P1, "--", color=VIOLET, label=f"m = {m75:.2f} (T/T$_N$ = 0.75)")
    a.axvline(E_F, color=INK, lw=0.5, ls=":"); a.set_ylim(0.3, 0.95); a.set_xticks([-0.2, -0.15, -0.1, -0.05, 0.0])
    a.set_xlabel("E (eV)"); a.set_ylabel(r"$P_{\rm str}=(r-1)/(r+1)$"); title(a, f"(a) straight ribbon, r = {r0:.1f}")
    a.legend(loc="lower right")
    a = ax[1]
    ratios = np.logspace(0, np.log10(6), 100)
    for totW, mk, col in ((4, "o", MUTED), (8, "s", INK)):
        pts = [(int(r["arm1_or_stem"]) / int(r["arm2_or_arm"]), float(r["P_network"])) for r in N
               if r["geometry"] == "Lchart" and int(r["arm1_or_stem"]) + int(r["arm2_or_arm"]) == totW * 20]
        a.plot(*zip(*pts), mk, color=col, label=f"$L_1+L_2$ = {totW}W")
        a.plot(ratios, Pstr(r0) * (ratios - 1) / (ratios + 1) / (1 + Lc / totW), "-", color=col, lw=0.8)
    for lam, col, mk in (("5", VIOLET, "v"), ("2", AQUA, "^"), ("1", ALT, "D")):
        pts = sorted((float(r["ratio"]), float(r["P_output"])) for r in LAM if r["geometry"] == "L" and r["lambda_sf_over_W"] == lam)
        a.plot(*zip(*pts), "--" + mk, color=col, ms=2.5, lw=0.8, label=f"λ$_{{sf}}$ = {lam}W")
    a.axhline(0, color=GREY, lw=0.5)
    a.set_xscale("log"); a.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    a.set_xticks([1, 4 / 3, 2, 3, 4, 6], ["1", "4/3", "2", "3", "4", "6"]); a.set_ylim(-0.75, 0.98); a.set_yticks([-0.6, -0.3, 0, 0.3, 0.6])
    a.set_xlabel("arm ratio $L_1/L_2$"); a.set_ylabel("$P_L$ (output)"); title(a, "(b) L-bend: Eq. (5) (lines)")
    a.legend(loc="upper left", ncol=2, columnspacing=0.6)
    a = ax[2]
    rr = np.logspace(np.log10(0.25), np.log10(4), 100)
    for LaW, mk, col in ((2, "o", MUTED), (4, "s", INK)):
        pts = [(int(r["arm1_or_stem"]) / int(r["arm2_or_arm"]), float(r["P_network"])) for r in N
               if r["geometry"] == "Tchart" and int(r["arm2_or_arm"]) == LaW * 20]
        a.plot(*zip(*pts), mk, color=col, label=f"$L_a$ = {LaW}W", gid="net")
        Ls = rr * LaW
        a.plot(rr, Pstr(r0) * (2 * Ls - LaW + dT) / (2 * Ls + LaW + LT), "-", color=col, lw=0.8)
    for lam, col, mk in (("5", VIOLET, "v"), ("2", AQUA, "^"), ("1", ALT, "D")):
        pts = sorted((float(r["ratio"]), float(r["P_output"])) for r in LAM if r["geometry"] == "T" and r["lambda_sf_over_W"] == lam)
        a.plot(*zip(*pts), "--" + mk, color=col, ms=2.5, lw=0.8, label=f"λ$_{{sf}}$ = {lam}W")
    a.axhline(0, color=GREY, lw=0.5)
    a.set_xscale("log"); a.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    a.set_xticks([0.25, 0.5, 1, 2, 4], ["1/4", "1/2", "1", "2", "4"]); a.set_ylim(-0.75, 0.98); a.set_yticks([-0.6, -0.3, 0, 0.3, 0.6])
    a.set_xlabel("stem/arm $L_s/L_a$"); a.set_ylabel("$P_T$ (arm)"); title(a, "(c) T-junction: Eq. (5) (lines)")
    h, l = a.get_legend_handles_labels()
    a.legend([x for x, y in zip(h, l) if y.startswith("$L_a$")], [y for y in l if y.startswith("$L_a$")],
             loc="upper left", title="λ$_{sf}$ curves as in (b)", title_fontsize=6.5)
    save(fig, "fig4_diffusive")


# ---------------------------------------------------------------- Fig. 5
def fig5():
    D = [r for r in csv.DictReader(open(ROOT / "data/s4_diffusive.csv"))]
    S5 = json.load(open(ROOT / "data/s5_summary.json"))
    fig, ax = plt.subplots(1, 3, figsize=(PW, 2.3), gridspec_kw=dict(wspace=0.55, width_ratios=[1.1, 1, 1.05]))
    a = ax[0]
    M = np.load(ROOT / "data/s4_mu_s_with_probes.npy"); W, Larm, Lp = 20, 40, 10
    cmap = LinearSegmentedColormap.from_list("spin", [DN, "#e9e9e6", UP]); vmax = np.nanmax(np.abs(M))
    im = a.imshow(M, origin="lower", cmap=cmap, vmin=-vmax, vmax=vmax, extent=(-Larm, W + Lp, -Lp, W + Larm))
    a.set_aspect("equal"); cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.16, ticks=[-0.6, -0.3, 0, 0.3, 0.6]); cb.ax.tick_params(pad=1)
    cb.ax.tick_params(labelsize=6)
    a.annotate("V$_{[100]}$", xy=(W + Lp - 1, 2), xytext=(W + 1, 22), fontsize=6.5, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
    a.annotate("V$_{[010]}$", xy=(W - 2, -Lp + 1), xytext=(-26, -7), fontsize=6.5, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
    a.text(-38, 3, "in", fontsize=6.5, color=MUTED); a.text(1, W + Larm - 6, "out", fontsize=6.5, color=MUTED)
    a.set_xlabel("x (cells)"); a.set_ylabel("y (cells)"); title(a, "(a) $\\mu_s/V$, symmetric L")
    a = ax[1]
    for lam, col, mk in (("inf", INK, "o"), ("5", VIOLET, "v"), ("2", AQUA, "^"), ("1", ALT, "D")):
        rows = sorted((int(r["wp"]), float(r["dV_AM"])) for r in D if r["lambda_sf_over_W"] == lam)
        a.plot(*zip(*rows), "-" + mk, color=col, label="no spin flip" if lam == "inf" else f"λ$_{{sf}}$ = {lam}W")
    iso = sorted((int(r["wp"]), float(r["dV_iso"])) for r in D if r["lambda_sf_over_W"] == "inf")
    a.plot(*zip(*iso), "--s", color=GREY, ms=2.5, lw=0.8, label="isotropic")
    a.axhline(0, color=GREY, lw=0.5)
    a.set_xscale("log", base=2); a.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"]); a.minorticks_off()
    a.set_xlabel("probe width $w_p$ (cells; W = 20)"); a.set_ylabel(r"$\Delta V/V$"); a.set_ylim(-0.04, 0.55)
    title(a, "(b) probe-pair signal (diffusive)"); a.legend(loc="lower left", bbox_to_anchor=(0.0, 0.05), ncol=2, columnspacing=0.5, handlelength=1.3)
    a = ax[2]
    clean = {(k.split(" W=")[0], int(k.split(" W=")[1])): v for k, v in S5["clean P (U = 0, same geometry)"].items()}
    k = 0; pos, labels = [], []
    for g in ("straight", "L", "T"):
        for Wc, col in ((20, MUTED), (38, INK)):
            Tu, _ = A5.tot("ens", g, Wc, Wc, 0.30, "up"); Td, _ = A5.tot("ens", g, Wc, Wc, 0.30, "dn")
            P = (Tu - Td) / (Tu + Td)
            vp = a.violinplot([P], positions=[k], widths=0.8, showextrema=False)
            for b in vp["bodies"]:
                b.set_facecolor(col); b.set_alpha(0.35); b.set_edgecolor(col); b.set_linewidth(0.5)
            a.plot([k], [P.mean()], "o", color=col, ms=2.8)
            a.plot([k - 0.35, k + 0.35], [clean[(g, Wc)]] * 2, "-", color=UP, lw=1.2)
            pos.append(k); labels.append(f"{Wc * A:.0f}"); k += 1
        a.text(k - 1.5, 0.97, g, ha="center", va="top", fontsize=6.5, color=INK, transform=a.get_xaxis_transform())
        k += 0.4
    a.axhline(0, color=GREY, lw=0.5)
    a.plot([], [], "-", color=UP, lw=1.2, label="clean"); a.plot([], [], "o", color=INK, ms=2.8, label="mean")
    a.set_xticks(pos, labels, fontsize=6.5); a.set_xlabel("width (nm)"); a.set_ylabel("P"); a.set_ylim(-0.32, 1.02)
    title(a, "(c) disorder, U = 0.30 eV"); a.legend(loc="upper right", bbox_to_anchor=(1.0, 0.88))
    save(fig, "fig5_readout_disorder")


def si_figures():
    (OUT / "si").mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(("s0_validation", "s1_channels", "s2_corners", "s3_diffusive", "s4_readout", "s5_disorder"), 1):
        shutil.copy(ROOT / "figures" / f"{src}.pdf", OUT / "si" / f"figS{i}_{src}.pdf")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig1(); fig2(); fig3(); fig4(); fig5()     # SI figures: calc/si_figures.py
    print("wrote", sorted(p.name for p in OUT.glob("fig*.pdf")), sorted(p.name for p in (OUT / "si").glob("*.pdf")))
