"""S2 corner efficiency (2026-11-14). The +-10 % acceptance criterion for kappa was fixed before this ran (SI Sec. S3).

Geometries (hard walls, single-orbital builders; the d_xz and d_xy bands decouple without SOC, so every
transmission is orbital-resolved and the total is the sum):
  L      : 90-degree bend, [100] lead in (from -x), [010] lead out (to +y); W x W corner; chamfer c removes the
           outer-corner sites with (W-1-x) + y < c
  T      : stem from -x, arms to +y and -y; W x W centre
  B45    : 45-degree [100] -> [110] bend, [110] arm of Wd = round(W sqrt2) rows (same physical width)
           (arm lengths Ls = Ld = 10)
Models: cheng (Table 1), iso (x/y-averaged, spin-degenerate), single_band (arXiv:2602.11694); plus the cheng
d_xy band shifted +0.08 eV (45-degree everywhere; L and T at W = 20 as a check).
Each solve also returns the leads' propagating-mode counts, so N_min comes from the same calculation.
SOC bound: soc_corner.soc_bend (two-sublattice builder with Table 2 SOC inside the corner; its k-convention
mapping is an assumption, so the result is an order-of-magnitude bound).

Run (project root): OMP_NUM_THREADS=1 ~/venvs/crn-te/bin/python calc/s2_corners.py
Outputs: data/s2_corners.csv, data/s2_soc.csv
"""
import csv, os, sys, time, warnings
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")
import numpy as np
import kwant

sys.path.insert(0, str(Path(__file__).parent))
import v2x2o_tb as tb

ROOT = Path(__file__).resolve().parent.parent
WS = [10, 15, 20, 25, 30, 40]
ES = [round(-0.19 + 0.025 * i, 4) for i in range(9)]
LAT = kwant.lattice.square(a=1, norbs=1)


def _lead(p, direction, cells):
    ld = kwant.Builder(kwant.TranslationalSymmetry(direction))
    tb._fill(ld, LAT, p, cells)
    return ld


def build(geom, p, W):
    b = kwant.Builder()
    if geom.startswith("L"):
        c = {"L": 0, "Lc4": round(W / 4), "Lc2": round(W / 2)}[geom]
        tb._fill(b, LAT, p, [(x, y) for x in range(W) for y in range(W) if (W - 1 - x) + y >= c])
        b.attach_lead(_lead(p, (-2, 0), [(x, y) for x in range(2) for y in range(W)]))
        b.attach_lead(_lead(p, (0, 2), [(x, y) for x in range(W) for y in range(2)]))
    elif geom == "T":
        tb._fill(b, LAT, p, [(x, y) for x in range(W) for y in range(W)])
        b.attach_lead(_lead(p, (-2, 0), [(x, y) for x in range(2) for y in range(W)]))
        b.attach_lead(_lead(p, (0, 2), [(x, y) for x in range(W) for y in range(2)]))
        b.attach_lead(_lead(p, (0, -2), [(x, y) for x in range(W) for y in range(2)]))
    elif geom == "B45":
        Wd, Ls, Ld = int(round(W * np.sqrt(2))), 10, 10
        inD = lambda x, y: 0 <= (y - x) < Wd and y >= 0
        S_end = Wd + 2 * Ld
        sites = [(x, y) for x in range(-Ls, Wd + 2 * Ld + 2) for y in range(0, Wd + 2 * Ld + 2)
                 if x + y < S_end and ((x <= 0 and y < W) or inD(x, y))]
        tb._fill(b, LAT, p, sites)
        b.attach_lead(_lead(p, (-2, 0), [(x, y) for x in range(2) for y in range(W)]))
        b.attach_lead(_lead(p, (1, 1), [(x, y) for x in range(-Wd - 2, Wd + 2) for y in range(-Wd - 2, Wd + 2)
                                         if 0 <= (y - x) < Wd and 0 <= x + y < 2]))
    else:
        raise ValueError(geom)
    return b.finalized()


def job(args):
    geom, model, shift, orb, spin, W, energies = args
    m = tb.shifted(model, shift) if shift else tb.MODELS[model]
    p = dict(tb.orbital_params(m, spin))[orb]
    t0 = time.time()
    f = build(geom, p, W)
    rows = []
    for E in energies:
        sm = kwant.smatrix(f, E)
        n = [sm.num_propagating(i) for i in range(len(f.leads))]
        T1 = sm.transmission(1, 0)
        T2 = sm.transmission(2, 0) if geom == "T" else ""
        rows.append([geom, model, shift, orb, spin, W, E, n[0], n[1], n[2] if geom == "T" else "",
                     round(T1, 6), round(T2, 6) if T2 != "" else ""])
    return rows, time.time() - t0


def jobs():
    out = []
    for geom in ("L", "Lc4", "Lc2", "T", "B45"):
        for W in WS:
            for model in ("cheng", "iso", "single_band"):
                if model == "single_band" and geom in ("Lc4", "Lc2"):
                    continue
                for orb in tb.MODELS[model]:
                    for spin in (("up",) if model == "iso" else ("up", "dn")):   # iso is spin-degenerate
                        out.append((geom, model, 0.0, orb, spin, W, ES))
            for spin in ("up", "dn"):   # pocket shift: 45 deg everywhere, L and T at W = 20
                if geom == "B45" or (geom in ("L", "T") and W == 20):
                    out.append((geom, "cheng", 0.08, "xy", spin, W, ES))
    # big jobs first for load balance
    return sorted(out, key=lambda j: -(j[5] ** 2) * (3 if j[0] == "B45" else 1))


def soc_bound():
    from soc_corner import soc_bend, T2
    rows = []
    for W in (20, 40):
        for soc in (False, True):
            f = soc_bend(T2["KV2Se2O"], W, soc_on=soc)
            for E in (-0.19, -0.09, 0.01):
                sm = kwant.smatrix(f, E)
                uu, ud = sm.transmission((1, 0), (0, 0)), sm.transmission((1, 1), (0, 0))
                du, dd = sm.transmission((1, 0), (0, 1)), sm.transmission((1, 1), (0, 1))
                P = ((uu + du) - (ud + dd)) / max(uu + ud + du + dd, 1e-12)
                rows.append([W, soc, E, uu, ud, du, dd, P])
    return rows


def main():
    t0 = time.time()
    J = jobs()
    print(f"{len(J)} jobs x {len(ES)} energies = {len(J) * len(ES)} solves", flush=True)
    rows, done = [], 0
    with Pool(6) as pool:
        soc_async = pool.apply_async(soc_bound)
        for r, dt in pool.imap_unordered(job, J):
            rows += r; done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(J)} jobs ({time.time() - t0:.0f} s)", flush=True)
        soc_rows = soc_async.get()
    rows.sort(key=lambda r: (r[0], r[1], r[2], r[3], r[4], r[5], r[6]))
    with open(ROOT / "data/s2_corners.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["geometry", "model", "xy_shift_eV", "orbital", "spin", "W_sites", "E_eV",
                    "N_lead0_in", "N_lead1", "N_lead2", "T_1from0", "T_2from0"])
        w.writerows(rows)
    with open(ROOT / "data/s2_soc.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["W_sites", "soc_on", "E_eV", "T_upup", "T_up_to_dn", "T_dn_to_up", "T_dndn", "P_out"])
        w.writerows(soc_rows)
    print(f"wrote data/s2_corners.csv ({len(rows)} rows) and data/s2_soc.csv in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
