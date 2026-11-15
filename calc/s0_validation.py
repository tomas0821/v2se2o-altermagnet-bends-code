"""S0 validation (2026-11-14).

1. Implementation check: our H(k) built from Cheng Table 1 (KV2Se2O, no SOC) against Cheng's own TB curves in
   Fig. 2(e) (red/blue dots, extracted from the PDF vector paths by s0_digitize_cheng_fig2e.py).
2. Fit quality: the same bands against the first-principles (grey) dots of Fig. 2(e) at Gamma/X/Y/M and in the
   Fermi window; the d_xy pocket top at X.
3. Filling at E_F = -0.09 eV and at E = 0.
4. The published single-band fit (arXiv:2602.11694) on the same footing.

Run: OMP_NUM_THREADS=1 ~/venvs/crn-te/bin/python calc/s0_validation.py   (from the project root)
Outputs: data/s0_validation.json, figures/s0_validation.png
"""
import csv, json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from v2x2o_tb import MODELS, E_F, band, orbital_params

ROOT = Path(__file__).resolve().parent.parent
UP, DN, ALT, GREY, INK, MUTED = "#e34948", "#2a78d6", "#eda100", "#a3a29d", "#0b0b0b", "#52514e"


def path_k(s):
    """s in [0,4]: M(pi,pi) -> X -> Gamma -> Y -> M, four segments of length pi.
    X is (pi,0) or (0,pi); chosen below by the fit to Cheng's TB dots."""
    s = np.asarray(s, float)
    kx = np.select([s <= 1, s <= 2, s <= 3], [np.full_like(s, np.pi), np.pi * (2 - s), 0 * s], np.pi * (s - 3))
    ky = np.select([s <= 1, s <= 2, s <= 3], [np.pi * (1 - s), 0 * s, np.pi * (s - 2)], np.full_like(s, np.pi))
    return kx, ky


def bands_on_path(model, spin, s, x_is_100=True):
    kx, ky = path_k(s)
    if not x_is_100:
        kx, ky = ky, kx
    return np.array([band(p, kx, ky) for _, p in orbital_params(MODELS[model], spin)])


def load_dots():
    d = {"TB_up": [], "TB_dn": [], "DFT": []}
    with open(ROOT / "data/s0_cheng_fig2e_digitized.csv") as f:
        for r in csv.DictReader(f):
            d[r["series"]].append((float(r["s_over_pi_path_M_X_G_Y_M"]), float(r["E_eV"])))
    return {k: np.array(v) for k, v in d.items()}


def resid(dots, model, spin, x_is_100):
    b = bands_on_path(model, spin, dots[:, 0], x_is_100)
    return np.min(np.abs(b - dots[:, 1][None, :]), axis=0)


def main():
    dots = load_dots()
    out = {}
    # ---- 1. implementation check: which colour is our 'up', and is X = (pi,0)?
    best = None
    for x100 in (True, False):
        for red_is_up in (True, False):
            r_red = resid(dots["TB_up"], "cheng", "up" if red_is_up else "dn", x100)
            r_blue = resid(dots["TB_dn"], "cheng", "dn" if red_is_up else "up", x100)
            rms = np.sqrt(np.mean(np.r_[r_red, r_blue] ** 2))
            # (X=(0,pi), red=down) is the same assignment as (X=(pi,0), red=up) because H_dn = H_up(x<->y);
            # keep the first (conventional) one unless another is genuinely better
            if best is None or rms < best[0] - 1e-6:
                best = (rms, x100, red_is_up, r_red, r_blue)
    rms, x100, red_is_up, r_red, r_blue = best
    out["implementation_check"] = dict(
        X_point="(pi,0)" if x100 else "(0,pi)", red_dots_are="our spin up" if red_is_up else "our spin down",
        n_red=len(r_red), n_blue=len(r_blue),
        rms_meV=1e3 * rms, max_meV=1e3 * max(r_red.max(), r_blue.max()),
        p99_meV=1e3 * np.percentile(np.r_[r_red, r_blue], 99),
        note="Residual = distance of each Cheng TB dot to the nearest of our two same-spin bands at that k. "
             "Digitisation uncertainty ~1.5 meV.")
    # ---- 2. TB vs DFT
    dft = dots["DFT"]
    hs = {"M": 0.0, "X": 1.0, "Gamma": 2.0, "Y": 3.0, "M'": 4.0}
    hs_rows = []
    for name, s0 in hs.items():
        near = dft[np.abs(dft[:, 0] - s0) < 0.012]
        for spin in ("up", "dn"):
            for (orb, _), E in zip(orbital_params(MODELS["cheng"], spin), bands_on_path("cheng", spin, [s0], x100)[:, 0]):
                if abs(E) > 1.08 or near.size == 0:
                    continue
                j = np.argmin(np.abs(near[:, 1] - E))
                hs_rows.append(dict(point=name, spin=spin, orbital=orb, E_TB=round(float(E), 4),
                                    E_DFT_nearest=round(float(near[j, 1]), 4), dE_meV=round(1e3 * float(E - near[j, 1]), 1)))
    out["high_symmetry_TB_vs_DFT"] = hs_rows

    def window_stats(model, lo, hi, exclude_pocket=False):
        m = (dft[:, 1] > lo) & (dft[:, 1] < hi)
        if exclude_pocket:   # d_xy hole pockets around X (and Y for spin down): 0.6 < s < 1.4 and 2.6 < s < 3.4
            m &= ~(((dft[:, 0] > 0.6) & (dft[:, 0] < 1.4)) | ((dft[:, 0] > 2.6) & (dft[:, 0] < 3.4)))
        sel = dft[m]
        r = np.minimum(resid(sel, model, "up", x100), resid(sel, model, "dn", x100))
        return dict(window_eV=[lo, hi], excluding_xy_pockets=exclude_pocket, n_dots=int(m.sum()),
                    rms_meV=round(1e3 * float(np.sqrt(np.mean(r ** 2))), 1), max_meV=round(1e3 * float(r.max()), 1),
                    median_meV=round(1e3 * float(np.median(r)), 1))
    def tb_crossings(model, level, x100=True):
        """path positions s where any band (both spins) crosses `level`, and the slope dE/ds there."""
        sg = np.linspace(0, 4, 40001)
        res = []
        for spin in ("up", "dn"):
            for b in bands_on_path(model, spin, sg, x100):
                e = b - level
                for i in np.where(np.sign(e[:-1]) != np.sign(e[1:]))[0]:
                    s_c = sg[i] - e[i] * (sg[i + 1] - sg[i]) / (e[i + 1] - e[i])
                    res.append((s_c, (b[i + 1] - b[i]) / (sg[i + 1] - sg[i]), spin))
        return res

    def kf_check(model, level, half_window=0.05):
        """horizontal (momentum) offset of the DFT dots from the TB Fermi crossings: for each DFT dot with
        |E - level| < half_window, the signed distance in s (units of pi/a) to the nearest TB crossing of the
        dot's own energy; grouped by the TB crossing at `level`."""
        sel = dft[np.abs(dft[:, 1] - level) < half_window]
        rows = []
        for s_c, slope, spin in tb_crossings(model, level, x100):
            ds = []
            for s_d, E_d in sel:
                cr = [c[0] for c in tb_crossings_cache(model, round(E_d, 3))]
                j = np.argmin(np.abs(np.array(cr) - s_d))
                if abs(cr[j] - s_c) < 0.08 and abs(s_d - cr[j]) < 0.12:
                    ds.append(s_d - cr[j])
            if ds:
                med = float(np.median(ds))
                rows.append(dict(s_TB=round(s_c, 4), spin=spin, n_dft_dots=len(ds),
                                 dk_over_pi=round(med, 4), equivalent_dE_meV=round(1e3 * abs(slope * med), 1)))
        return rows

    _cache = {}

    def tb_crossings_cache(model, level):
        if (model, level) not in _cache:
            _cache[(model, level)] = tb_crossings(model, level, x100)
        return _cache[(model, level)]

    out["kF_TB_vs_DFT_at_E0"] = {m: kf_check(m, 0.0) for m in ("cheng", "single_band")}
    out["fermi_window_TB_vs_DFT"] = {
        "cheng": [window_stats("cheng", -0.25, 0.25), window_stats("cheng", -0.25, 0.25, True)],
        "single_band": [window_stats("single_band", -0.25, 0.25), window_stats("single_band", -0.25, 0.25, True)],
    }
    # d_xy pocket top at X: TB value vs highest DFT dot near X inside 0..0.4 eV
    nearX = dft[(np.abs(dft[:, 0] - 1.0) < 0.012) & (dft[:, 1] > 0.0) & (dft[:, 1] < 0.4)]
    E_xy_X = float(bands_on_path("cheng", "up", [1.0], x100)[1, 0])
    out["xy_pocket_top_at_X"] = dict(TB_eV=round(E_xy_X, 4),
                                     DFT_dots_near_X_0_to_0p4_eV=[round(float(e), 4) for e in sorted(nearX[:, 1])],
                                     note="the DFT dot nearest above the TB value is the d_xy top in DFT")
    # ---- 3. filling (electrons per cell in the four-band model; both spin sectors)
    N = 1200
    k = (np.arange(N) + 0.5) * 2 * np.pi / N - np.pi
    KX, KY = np.meshgrid(k, k, indexing="ij")
    fill = {}
    for model in ("cheng", "single_band"):
        for E in (E_F, 0.0, -0.15, 0.10):
            n = sum(float((band(p, KX, KY) < E).mean()) for spin in ("up", "dn")
                    for _, p in orbital_params(MODELS[model], spin))
            fill[f"{model} E={E:+.2f}"] = round(n, 4)
    # pocket area fraction at E_F (holes in the d_xy band per spin)
    pxy = MODELS["cheng"]["xy"]
    fill["xy_holes_per_spin_at_EF"] = round(float((band(pxy, KX, KY) > E_F).mean()), 4)
    out["filling_e_per_cell"] = fill
    # Fermi momenta along Gamma-X for the spin-up d_xz sheet at E_F (cheng vs single band)
    kk = np.linspace(0, np.pi, 200001)
    kf = {}
    for model in ("cheng", "single_band"):
        for ky0, lab in ((0.0, "ky=0"), (np.pi, "ky=pi")):
            e = band(MODELS[model]["xz"], kk, ky0) - E_F
            i = np.where(np.sign(e[:-1]) != np.sign(e[1:]))[0]
            kf[f"{model} {lab}"] = [round(float(kk[j] / np.pi), 4) for j in i]
    out["kF_over_pi_spin_up_xz_along_kx"] = kf
    json.dump(out, open(ROOT / "data/s0_validation.json", "w"), indent=1)
    print(json.dumps(out["implementation_check"], indent=1))
    print(json.dumps(out["fermi_window_TB_vs_DFT"], indent=1))
    print(json.dumps(out["xy_pocket_top_at_X"], indent=1))
    print(json.dumps(out["kF_TB_vs_DFT_at_E0"], indent=1))
    print(json.dumps(out["filling_e_per_cell"], indent=1))
    print(json.dumps(out["kF_over_pi_spin_up_xz_along_kx"], indent=1))
    for r in hs_rows:
        print(r)

    # ---- figure
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.linewidth": 0.8})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 4.5), gridspec_kw=dict(width_ratios=[1.45, 1]))
    s = np.linspace(0, 4, 1601)
    a1.scatter(dft[:, 0], dft[:, 1], s=3, color=GREY, lw=0, label="DFT (Cheng Fig. 2e)", zorder=1)
    for col, spin, lab in ((UP, "up", "spin ↑ (this work, Table 1)"), (DN, "dn", "spin ↓ (this work, Table 1)")):
        for i, b in enumerate(bands_on_path("cheng", spin, s, x100)):
            a1.plot(s, b, color=col, lw=1.6, label=lab if i == 0 else None, zorder=3)
    for spin in ("up", "dn"):
        b = bands_on_path("single_band", spin, s, x100)[0]
        a1.plot(s, b, color=ALT, lw=1.4, ls=(0, (4, 2)), label="single-band fit (arXiv:2602.11694)" if spin == "up" else None, zorder=4)
    a1.axhline(0, color=MUTED, lw=0.6, ls=":")
    a1.axhline(E_F, color=INK, lw=0.8, ls="--")
    a1.text(4.02, E_F, r"$E_F$ = −0.09", va="center", fontsize=8, color=INK)
    a1.text(4.02, 0.0, "DFT $E_F$", va="center", fontsize=8, color=MUTED)
    for x in (1, 2, 3):
        a1.axvline(x, color=GREY, lw=0.5)
    a1.set_xticks([0, 1, 2, 3, 4], ["M", "X", "Γ", "Y", "M"])
    a1.set_xlim(0, 4); a1.set_ylim(-1.1, 1.1); a1.set_ylabel("E (eV)")
    a1.set_title(f"(a) KV$_2$Se$_2$O: Table 1 TB (lines) vs DFT (grey); = Cheng's TB to {1e3*rms:.1f} meV rms",
                 fontsize=8.5, loc="left", color=INK)
    a1.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, frameon=False)
    # Fermi surface
    n = 401
    kg = np.linspace(-np.pi, np.pi, n)
    KX2, KY2 = np.meshgrid(kg, kg, indexing="xy")
    for col, spin in ((UP, "up"), (DN, "dn")):
        for orb, p in orbital_params(MODELS["cheng"], spin):
            a2.contour(KX2 / np.pi, KY2 / np.pi, band(p, KX2, KY2), levels=[E_F], colors=col,
                       linewidths=1.6 if orb == "xz" else 1.2, linestyles="solid" if orb == "xz" else "dotted")
        for _, p in orbital_params(MODELS["single_band"], spin):
            a2.contour(KX2 / np.pi, KY2 / np.pi, band(p, KX2, KY2), levels=[E_F], colors=ALT, linewidths=1.2,
                       linestyles="dashed")
    a2.set_aspect("equal"); a2.set_xlabel(r"$k_x/\pi$"); a2.set_ylabel(r"$k_y/\pi$")
    a2.set_xticks([-1, 0, 1]); a2.set_yticks([-1, 0, 1])
    a2.set_title("(b) Fermi surface at $E_F$ = −0.09 eV", fontsize=9, loc="left", color=INK)
    a2.text(0.60, 0.80, "↑ d$_{xz}$", color=UP, fontsize=8, ha="left")
    a2.text(-0.97, 0.66, "↓ d$_{yz}$", color=DN, fontsize=8, ha="left")
    a2.text(0.0, 0.0, "d$_{xy}$ pockets\n(dotted)", fontsize=7.5, color=INK, ha="center", va="center")
    a2.annotate("", xy=(0.80, 0.0), xytext=(0.30, 0.0), arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    a2.annotate("", xy=(0.0, 0.83), xytext=(0.0, 0.12), arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    a2.text(-0.98, -0.98, "dashed: single-band fit", fontsize=7, color=MUTED)
    fig.tight_layout()
    fig.savefig(ROOT / "figures/s0_validation.png", dpi=200)
    fig.savefig(ROOT / "figures/s0_validation.pdf", dpi=600)   # vector, for the SI
    print("wrote figures/s0_validation.png")


if __name__ == "__main__":
    main()
