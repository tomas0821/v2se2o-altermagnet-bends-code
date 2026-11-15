"""S3 diffusive room-temperature rules (2026-11-14).

1. Constant-tau Boltzmann conductivity tensors per spin and per orbital (sigma_ab / tau), Fermi-Dirac smearing.
   Spin down = spin up with x <-> y. Implementation check: RbV2Te2O against Cheng et al. Fig. 4(c), whose
   normalised longitudinal conductivities at theta = 0 were read from the PDF vector paths:
   sigma_up,L = 0.812, sigma_dn,L = 0.188 (sum 1.000) -> sigma_xx/sigma_yy = 4.32 (temperature not stated by Cheng).
2. KV2Se2O at 300 K over E_F +- 0.1 eV: r = sigma_xx/sigma_yy per spin, P_str = (r - 1)/(r + 1).
   Sensitivity bands: orbital-dependent tau (tau_xy/tau_xz = 0.5 ... 2); E_F sweep; reduced order m at
   T/T_N = 0.75 (mean-field S = 1) in the spin-mixing picture: each sublattice-locked band carries spin
   polarisation m, so sigma_up = (1+m)/2 sigma_A + (1-m)/2 sigma_B and P_str -> m P_str exactly (r -> r_eff).
   (A first attempt scaled the hoppings toward their x/y average, iso + m (cheng - iso). It was rejected: the
   x/y anisotropy is orbital, not magnetic, and that interpolation drives t1y of d_xz through zero, so P_str
   *rose* to 0.84 at m = 0.74.)
3. Mean free path from the measured rho(300 K) = 0.5 mOhm cm of Cs1-dV2Te2O (arXiv:2511.06865) with the KV2Se2O
   Drude weight and interlayer spacing c = 7.37 A .
4. Two-current resistor network (no spin flip; P is an upper bound) for L-bends and T-junctions, and the closed
   forms derived from series resistance:
     P_L = P_str (L1 - L2) / (L1 + L2 + L_c)
     P_T = P_str (2 L_s - L_a + d_T) / (2 L_s + L_a + L_T)
   L_c, d_T, L_T come from the corner/junction resistances, which are scale-invariant (W-independent in units of
   1/sigma). They are fitted at W = 20 and tested on held-out arm lengths and on W = 10, 30, 40.
5. Corner spin accumulation mu_s = phi_up - phi_dn (in units of the applied bias) for the symmetric L, without
   spin flip (upper bound) and with spin relaxation lambda_sf / W = 5, 2, 1 (on-site up-down conductance
   g_sf = sigma_bar / (2 lambda_sf^2), Valet-Fert form; lambda_sf is a parameter).

Run (project root): OMP_NUM_THREADS=1 ~/venvs/crn-te/bin/python calc/s3_diffusive.py
Outputs: data/s3_boltzmann.csv, data/s3_network.csv, data/s3_summary.json, data/s3_spin_accumulation.npy
"""
import csv, json, os, sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).parent))
import v2x2o_tb as tb

ROOT = Path(__file__).resolve().parent.parent
KB = 8.617333e-5
RB_XZ = (-0.04629229, -0.24988718, 0.094201, 0.04609694, 0.01695743, -0.00710368)   # Cheng Table 1, RbV2Te2O
RB_XY = (-0.54598791, -0.08869994, -0.03574438, -0.1130422, 0.06412028, 0.01420836)
CHENG_FIG4C = dict(sigma_up_L=0.812, sigma_dn_L=0.188, r=0.812 / 0.188)


# ---------------------------------------------------------------- Boltzmann
def _bands(p, kx, ky):
    e, t1x, t1y, t2, t3x, t3y = p
    E = tb.band(p, kx, ky)
    vx = -2 * t1x * np.sin(kx) - 4 * t2 * np.sin(kx) * np.cos(ky) - 4 * t3x * np.sin(2 * kx)
    vy = -2 * t1y * np.sin(ky) - 4 * t2 * np.cos(kx) * np.sin(ky) - 4 * t3y * np.sin(2 * ky)
    return E, vx, vy


_K = {}


def kgrid(n):
    if n not in _K:
        k = (np.arange(n) + 0.5) * 2 * np.pi / n - np.pi
        _K[n] = np.meshgrid(k, k, indexing="ij")
    return _K[n]


def sigma_over_tau(p, E, T, n=900):
    """(sxx, syy, sxy, speed-weight, weight) per tau, one band, lattice units (a = 1, hbar = 1, e = 1), per cell."""
    kx, ky = kgrid(n)
    Ek, vx, vy = _bands(p, kx, ky)
    kT = KB * T
    x = (Ek - E) / kT
    w = 1.0 / (4 * kT * np.cosh(np.clip(x, -60, 60) / 2) ** 2)      # -df/dE
    return (float((w * vx * vx).mean()), float((w * vy * vy).mean()), float((w * vx * vy).mean()),
            float((w * np.hypot(vx, vy)).mean()), float(w.mean()))


def spin_tensor(model_dict, E, T, tau_ratio=1.0):
    """spin-up (sxx, syy) summed over orbitals, d_xy weighted by tau_xy/tau_xz."""
    sxx = syy = 0.0
    for o, p in model_dict.items():
        s = sigma_over_tau(p, E, T)
        f = tau_ratio if o == "xy" else 1.0
        sxx += f * s[0]; syy += f * s[1]
    return sxx, syy


def m_meanfield(t, S=1.0):
    """reduced sublattice magnetisation from the mean-field Brillouin equation."""
    def B(x):
        a, b = (2 * S + 1) / (2 * S), 1 / (2 * S)
        return a / np.tanh(a * x) - b / np.tanh(b * x)
    if t >= 1:
        return 0.0
    return brentq(lambda m: m - B(3 * S / (S + 1) * m / t), 1e-6, 1.0)


def r_eff(r, m):
    """spin-mixing picture: sigma_up = (1+m)/2 sigma_A + (1-m)/2 sigma_B, sigma_B,xx = sigma_A,yy."""
    return ((1 + m) * r + (1 - m)) / ((1 + m) + (1 - m) * r)


# ---------------------------------------------------------------- resistor network
def network(cells, sx, sy, contacts, return_potential=False):
    cells = sorted(cells)
    idx = {c: i for i, c in enumerate(cells)}
    N = len(cells)
    fixed = np.full(N, np.nan)
    for cs, V in contacts:
        for c in cs:
            fixed[idx[c]] = V
    rows, cols, vals = [], [], []
    b = np.zeros(N)
    for c, i in idx.items():
        if not np.isnan(fixed[i]):
            rows.append(i); cols.append(i); vals.append(1.0); b[i] = fixed[i]; continue
        d = 0.0
        for (di, dj), g in (((1, 0), sx), ((-1, 0), sx), ((0, 1), sy), ((0, -1), sy)):
            j = idx.get((c[0] + di, c[1] + dj))
            if j is not None:
                rows.append(i); cols.append(j); vals.append(-g); d += g
        rows.append(i); cols.append(i); vals.append(d)
    V = spla.spsolve(sp.csr_matrix((vals, (rows, cols)), shape=(N, N)), b)
    cur = []
    for cs, Vc in contacts:
        I = 0.0
        for c in cs:
            for (di, dj), g in (((1, 0), sx), ((-1, 0), sx), ((0, 1), sy), ((0, -1), sy)):
                j = idx.get((c[0] + di, c[1] + dj))
                if j is not None and (c[0] + di, c[1] + dj) not in cs:
                    I += g * (V[idx[c]] - V[j])
        cur.append(I)
    if return_potential:
        return cur, {c: V[i] for c, i in idx.items()}
    return cur


def network_sf(cells, su, sd, g_sf, contacts):
    """two-current network with on-site spin relaxation; su, sd = (sx, sy) per spin. Contacts fix both spins.
    Returns (I_up, I_dn) into the first contact and the potentials."""
    cells = sorted(cells)
    idx = {c: i for i, c in enumerate(cells)}
    N = len(cells)
    fixed = {}
    for cs, V in contacts:
        for c in cs:
            fixed[idx[c]] = V
    rows, cols, vals = [], [], []
    b = np.zeros(2 * N)
    for c, i in idx.items():
        for s, (sx, sy) in enumerate((su, sd)):
            I = i + s * N
            if i in fixed:
                rows.append(I); cols.append(I); vals.append(1.0); b[I] = fixed[i]; continue
            d = 0.0
            for (di, dj), g in (((1, 0), sx), ((-1, 0), sx), ((0, 1), sy), ((0, -1), sy)):
                j = idx.get((c[0] + di, c[1] + dj))
                if j is not None:
                    rows.append(I); cols.append(j + s * N); vals.append(-g); d += g
            if g_sf > 0:
                rows.append(I); cols.append(i + (1 - s) * N); vals.append(-g_sf); d += g_sf
            rows.append(I); cols.append(I); vals.append(d)
    V = spla.spsolve(sp.csr_matrix((vals, (rows, cols)), shape=(2 * N, 2 * N)), b)
    out = []   # out[k] = (I_up, I_dn) leaving contact k into the device
    for cs, _ in contacts:
        pair = []
        for s, (sx, sy) in enumerate((su, sd)):
            I = 0.0
            for c in cs:
                for (di, dj), g in (((1, 0), sx), ((-1, 0), sx), ((0, 1), sy), ((0, -1), sy)):
                    nb = (c[0] + di, c[1] + dj)
                    j = idx.get(nb)
                    if j is not None and nb not in cs:
                        I += g * (V[idx[c] + s * N] - V[j + s * N])
            pair.append(I)
        out.append(tuple(pair))
    return out, {c: (V[i], V[i + N]) for c, i in idx.items()}


def L_geom(W, L1, L2):
    cells = {(i, j) for i in range(-L1, W) for j in range(W)} | {(i, j) for i in range(W) for j in range(W, W + L2)}
    con = [({(-L1, j) for j in range(W)}, 1.0), ({(i, W + L2 - 1) for i in range(W)}, 0.0)]
    return cells, con


def T_geom(W, Ls, La):
    cells = ({(i, j) for i in range(-Ls, W) for j in range(W)} | {(i, j) for i in range(W) for j in range(W, W + La)}
             | {(i, j) for i in range(W) for j in range(-La, 0)})
    con = [({(-Ls, j) for j in range(W)}, 1.0), ({(i, W + La - 1) for i in range(W)}, 0.0),
           ({(i, -La) for i in range(W)}, 0.0)]
    return cells, con


def R_of(geom, W, a, b, sx, sy):
    cells, con = (L_geom if geom == "L" else T_geom)(W, a, b)
    I = network(cells, sx, sy, con)[0]
    return 1.0 / I        # V = 1


def fit_corner(geom, r, W=20, train=None):
    """intercepts c_up, c_dn (units 1/sigma_y) of R = arm terms + c; arm terms exact for uniform arms."""
    train = train or ([(20, 20), (40, 20), (20, 40), (60, 30), (30, 60), (40, 40)] if geom == "L"
                      else [(20, 20), (40, 20), (20, 40), (60, 30), (30, 60), (40, 40)])
    out = {}
    for spin, (sx, sy) in (("up", (r, 1.0)), ("dn", (1.0, r))):
        X, y = [], []
        for a, b in train:
            R = R_of(geom, W, a, b, sx, sy)
            arm = (a / (sx * W) + b / (sy * W)) if geom == "L" else (a / (sx * W) + b / (2 * sy * W))
            X.append(1.0); y.append(R - arm)
        out[spin] = float(np.mean(y)); out[spin + "_spread"] = float(np.ptp(y))
    return out


def closed_form(geom, r, W, a, b, c):
    """P from series resistance with fitted corner/junction terms c (W-independent)."""
    if geom == "L":
        Ru = a / (r * W) + b / W + c["up"]; Rd = a / W + b / (r * W) + c["dn"]
    else:
        Ru = a / (r * W) + b / (2 * W) + c["up"]; Rd = a / W + b / (2 * r * W) + c["dn"]
    return (Rd - Ru) / (Rd + Ru)


def network_P(geom, r, W, a, b):
    Ru, Rd = R_of(geom, W, a, b, r, 1.0), R_of(geom, W, a, b, 1.0, r)
    return (Rd - Ru) / (Rd + Ru)


def main():
    S = {}
    rows = []
    # 1. implementation check against Cheng Fig. 4(c)
    rb = {"xz": RB_XZ, "xy": RB_XY}
    chk = {}
    for T in (10, 100, 300):
        sxx, syy = spin_tensor(rb, 0.0, T)
        chk[f"T={T} K"] = round(sxx / syy, 3)
        rows.append(["RbV2Te2O", "cheng", 0.0, T, 1.0, 1.0, sxx, syy, sxx / syy])
    S["RbV2Te2O r = sxx/syy at E = 0 (Cheng Fig. 4c: 4.32)"] = chk
    E432 = {T: brentq(lambda E: np.divide(*spin_tensor(rb, E, T)) - CHENG_FIG4C["r"], -0.2, 0.0) for T in (10, 300)}
    S["RbV2Te2O: energy where r = 4.32 (Cheng Fig. 4 E_F not stated)"] = {f"T={T} K": round(E * 1e3, 1) for T, E in E432.items()}
    # 2. KV2Se2O at 300 K
    ES = [round(-0.19 + 0.0125 * i, 4) for i in range(17)]
    m75 = m_meanfield(0.75)
    S["mean-field m(T/T_N = 0.75), S = 1"] = round(m75, 4)
    variants = [("cheng", 1.0, 1.0), ("cheng", 0.5, 1.0), ("cheng", 2.0, 1.0)]
    rE = {}
    for name, tr, m in variants:
        md = tb.MODELS["cheng"]
        for E in ES:
            sxx, syy = spin_tensor(md, E, 300.0, tr)
            rE[(name, tr, E)] = sxx / syy
            rows.append(["KV2Se2O", name, E, 300.0, tr, m, sxx, syy, sxx / syy])
    # per-orbital at E_F
    orb = {}
    for o, p in tb.MODELS["cheng"].items():
        s = sigma_over_tau(p, tb.E_F, 300.0)
        orb[o] = dict(sxx=s[0], syy=s[1], r=round(s[0] / s[1], 3))
    tot_xx = sum(v["sxx"] for v in orb.values()); tot_xy = sum(v["syy"] for v in orb.values())
    S["KV2Se2O 300 K, E_F: per-orbital r and share of sigma_xx"] = {
        o: dict(r=v["r"], share_xx=round(v["sxx"] / tot_xx, 3), share_yy=round(v["syy"] / tot_xy, 3)) for o, v in orb.items()}
    rF = rE[("cheng", 1.0, -0.09)]
    Pstr = lambda r: (r - 1) / (r + 1)
    S["KV2Se2O 300 K, E_F: r, P_str"] = [round(rF, 3), round(Pstr(rF), 4)]
    S["P_str range over E_F +- 0.1 eV (cheng)"] = [round(min(Pstr(rE[("cheng", 1.0, E)]) for E in ES), 4),
                                                    round(max(Pstr(rE[("cheng", 1.0, E)]) for E in ES), 4)]
    S["P_str at E_F, sensitivity"] = {
        "tau_xy/tau_xz = 0.5": round(Pstr(rE[("cheng", 0.5, -0.09)]), 4),
        "tau_xy/tau_xz = 2": round(Pstr(rE[("cheng", 2.0, -0.09)]), 4),
        f"m = {m75:.3f} (T/T_N = 0.75), spin-mixing: P_str -> m P_str": round(Pstr(r_eff(rE[("cheng", 1.0, -0.09)], m75)), 4)}
    S["P_str full band over E_F +- 0.1 eV (tau 0.5-2, m 0.74-1)"] = [
        round(min(Pstr(r_eff(rE[("cheng", tr, E)], m75)) for tr in (0.5, 1.0, 2.0) for E in ES), 4),
        round(max(Pstr(rE[("cheng", tr, E)]) for tr in (0.5, 1.0, 2.0) for E in ES), 4)]
    # 3. mean free path from rho
    a_m, c_m, rho = 0.3952e-9, 7.37e-10, 0.5e-5      # m, m, Ohm m
    e, hbar = 1.602176634e-19, 1.054571817e-34
    tot = 0.0; vw = ww = 0.0
    for spin in ("up", "dn"):
        for o, p in tb.orbital_params(tb.MODELS["cheng"], spin):
            s = sigma_over_tau(p, tb.E_F, 300.0)
            tot += s[0]; vw += s[3]; ww += s[4]
    # lattice units -> SI: v = (dE/dk) a / hbar with E in eV -> v[m/s] = v_lat * e * a / hbar ; -df/dE in 1/eV -> 1/J = /e
    vfac = e * a_m / hbar
    sig_tau = e ** 2 * tot * vfac ** 2 / e / a_m ** 2          # S/s per layer (sigma_xx / tau), both spins
    sig2d = c_m / rho
    tau = sig2d / sig_tau
    vF = vw / ww * vfac
    S["mean free path, rho(300 K) = 0.5 mOhm cm (Cs1-dV2Te2O), c = 7.37 A"] = dict(
        sheet_R_per_layer_Ohm=round(1 / sig2d, 1), tau_fs=round(tau * 1e15, 2), vF_1e5_m_per_s=round(vF / 1e5, 3),
        l_nm=round(vF * tau * 1e9, 3))
    with open(ROOT / "data/s3_boltzmann.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["compound", "variant", "E_eV", "T_K", "tau_xy_over_tau_xz", "m", "sxx_over_tau", "syy_over_tau", "r"])
        w.writerows(rows)
    # 4. network vs closed form
    r = rF
    S["network: r used (KV2Se2O, E_F, 300 K)"] = round(r, 4)
    netrows = []
    fits = {}
    for geom in ("L", "T"):
        c = fit_corner(geom, r)
        fits[geom] = c
        if geom == "L":
            Lc = 2 * 20 * c["up"] / (1 / r + 1)     # effective corner length at W = 20 (units of sites)
            S["L-bend corner term"] = dict(c_up=round(c["up"], 5), c_dn=round(c["dn"], 5),
                                           fit_spread=round(max(c["up_spread"], c["dn_spread"]), 6),
                                           L_c_over_W=round(Lc / 20, 4))
        else:
            dT = 2 * 20 * (c["dn"] - c["up"]) / ((1 + 1 / r) * Pstr(r))
            LT = 2 * 20 * (c["dn"] + c["up"]) / (1 + 1 / r)
            S["T-junction terms"] = dict(c_up=round(c["up"], 5), c_dn=round(c["dn"], 5),
                                         fit_spread=round(max(c["up_spread"], c["dn_spread"]), 6),
                                         d_T_over_W=round(dT / 20, 4), L_T_over_W=round(LT / 20, 4))
    tests = {"L": [(W, a, b) for W in (10, 20, 30, 40) for a, b in ((50, 15), (15, 50), (80, 20), (30, 70), (45, 45), (25, 10))],
             "T": [(W, a, b) for W in (10, 20, 30, 40) for a, b in ((50, 15), (15, 50), (80, 20), (30, 70), (45, 45), (10, 40))]}
    for geom, T in tests.items():
        errs = []
        for W, a, b in T:
            # scale arm lengths with W so the test covers the same aspect ratios at every width
            A_, B_ = a * W // 20, b * W // 20
            Pn = network_P(geom, r, W, A_, B_)
            Pc = closed_form(geom, r, W, A_, B_, fits[geom])
            P0 = Pstr(r) * ((A_ - B_) / (A_ + B_) if geom == "L" else (2 * A_ - B_) / (2 * A_ + B_))
            errs.append(abs(Pn - Pc))
            netrows.append([geom, W, A_, B_, r, Pn, Pc, P0])
        S[f"{geom}: closed form vs network on held-out geometries"] = dict(
            n=len(T), max_abs_dP=float(f"{max(errs):.2e}"), mean_abs_dP=float(f"{np.mean(errs):.2e}"))
    # design-chart grids (network, for the figure)
    for ratio in (1.0, 4 / 3, 2, 3, 4, 6):
        for tot_W in (4, 8):
            W = 20; tot = tot_W * W
            a = int(round(tot * ratio / (1 + ratio))); b = tot - a
            netrows.append(["Lchart", W, a, b, r, network_P("L", r, W, a, b), closed_form("L", r, W, a, b, fits["L"]),
                            Pstr(r) * (a - b) / (a + b)])
    for ratio in (0.25, 0.5, 1, 2, 3, 4):
        for La_W in (2, 4):
            W = 20; La = La_W * W; Ls = int(round(ratio * La))
            netrows.append(["Tchart", W, Ls, La, r, network_P("T", r, W, Ls, La), closed_form("T", r, W, Ls, La, fits["T"]),
                            Pstr(r) * (2 * Ls - La) / (2 * Ls + La)])
    with open(ROOT / "data/s3_network.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["geometry", "W", "arm1_or_stem", "arm2_or_arm", "r", "P_network", "P_closed_form", "P_long_arm_limit"])
        w.writerows(netrows)
    S["L: P for arm ratios 4/3, 2, 3 (total 4W, network)"] = [
        round(r_[5], 3) for r_ in netrows if r_[0] == "Lchart" and r_[2] + r_[3] == 80 and r_[2] / r_[3] in (60 / 20, 53 / 27, 46 / 34)]
    # 5. spin accumulation, symmetric L
    W, Larm = 20, 40
    cells, con = L_geom(W, Larm, Larm)
    _, Vu = network(cells, r, 1.0, con, return_potential=True)
    _, Vd = network(cells, 1.0, r, con, return_potential=True)
    xs = range(-Larm, W); ys = range(0, W + Larm)
    M = np.full((len(ys), len(xs)), np.nan)
    for (i, j) in cells:
        M[j - 0, i + Larm] = Vu[(i, j)] - Vd[(i, j)]
    np.save(ROOT / "data/s3_spin_accumulation.npy", M)
    k = np.nanargmax(np.abs(M)); jj, ii = np.unravel_index(k, M.shape)
    S["symmetric L (W=20, arms 40): spin accumulation mu_s/V"] = dict(
        max_abs=round(float(np.nanmax(np.abs(M))), 4), at_cell=dict(x=int(ii - Larm), y=int(jj)),
        net_P=float(f"{(lambda u, d: (u - d) / (u + d))(network(cells, r, 1.0, con)[0], network(cells, 1.0, r, con)[0]):.1e}"))
    # 6. spin relaxation: corner accumulation and an asymmetric L
    sbar = np.sqrt(r * 1.0)
    sf = {}
    for lamW in (None, 5, 2, 1):
        g = 0.0 if lamW is None else sbar / (2 * (lamW * W) ** 2)
        _, Vs = network_sf(cells, (r, 1.0), (1.0, r), g, con)
        mus = max(abs(u - d) for u, d in Vs.values())
        c2, k2 = L_geom(W, 40, 20)
        (ci, co), _ = network_sf(c2, (r, 1.0), (1.0, r), g, k2)
        Pin = (ci[0] - ci[1]) / (ci[0] + ci[1])
        Pout = (co[0] - co[1]) / (co[0] + co[1])     # current delivered into the output ([010]) contact
        sf["no spin flip" if lamW is None else f"lambda_sf/W = {lamW}"] = dict(
            max_abs_mu_s=round(float(mus), 4), P_L_40_20_at_input=round(Pin, 4), P_L_40_20_at_output=round(Pout, 4))
    S["spin relaxation (W = 20)"] = sf
    json.dump(S, open(ROOT / "data/s3_summary.json", "w"), indent=1)
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
