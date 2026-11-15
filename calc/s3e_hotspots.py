"""S3e (2026-11-14): convergence and hot-spot check of the Elliott spin
admixture <b^2> of calc/s3c_elliott.py, plus the Elliott lambda_sf with the paper's own mean free path.
The SOC of the model couples only bands 0-3 (A d_xz up, B d_xy dn) and 1-2 (A d_xy up, B d_yz dn) for a Neel vector
along z (the convention of s3c_elliott.py), so <b^2> follows from exact 2x2 blocks; this allows fine grids. Also
locates the closest approach of coupled Fermi lines (hot spots) and the crossings of the up-xz / dn-yz sheets, which
this SOC form does not couple.
Run (project root): OMP_NUM_THREADS=1 ~/venvs/crn-te/bin/python calc/s3e_hotspots.py  ->  data/s3e_hotspots.json
"""
import json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import s3c_elliott as el

ROOT = Path(__file__).resolve().parent.parent
pxz, pxy = el.P["KV2Se2O"]
L1, L2 = pxz[6], pxy[6]
kT = 8.617e-5 * 300
ELL = json.load(open(ROOT / "data/s3_summary.json"))["mean free path, rho(300 K) = 0.5 mOhm cm (Cs1-dV2Te2O), c = 7.37 A"]["l_nm"]


def blocks(KX, KY):
    E0, E1 = el.band(pxz, KX, KY), el.band(pxy, KX, KY)
    E2, E3 = el.band(pxz, KY, KX), el.band(pxy, KY, KX)
    c, s = np.cos(KX / 2) * np.cos(KY / 2), np.sin(KX / 2) * np.sin(KY / 2)
    h03 = np.abs(4j * L1 * c - 4 * L2 * s); h12 = np.abs(4 * L1 * c - 4j * L2 * s)
    return (E0, E3, h03), (E1, E2, h12), (E0, E2)


def b2avg(N, EF, chunk=250):
    k = (np.arange(N) + 0.5) / N * 2 * np.pi - np.pi
    num = den = 0.0
    for i0 in range(0, N, chunk):
        KX, KY = np.meshgrid(k[i0:i0 + chunk], k, indexing="ij")
        for Ea, Eb, h in blocks(KX, KY)[:2]:
            d = Ea - Eb; R = np.sqrt(d * d + 4 * h * h); b2 = (1 - np.abs(d) / R) / 2
            for Eband in ((Ea + Eb) / 2 + R / 2, (Ea + Eb) / 2 - R / 2):
                w = 1 / (4 * kT * np.cosh(np.clip((Eband - EF) / kT, -50, 50) / 2) ** 2)
                num += (w * b2).sum(); den += w.sum()
    return num / den


def main():
    out = dict(ell_nm=ELL, convergence_at_EF=[], lambda_sf=[], crossings={})
    for N in (800, 2000, 4000):
        b = b2avg(N, -0.09)
        out["convergence_at_EF"].append(dict(N=N, b2=float(f"{b:.4e}"))); print(N, b, flush=True)
    for EF in (-0.15, -0.09, 0.0, 0.10):
        b = b2avg(2000, EF); lam = ELL / np.sqrt(8 * b)
        out["lambda_sf"].append(dict(E_eV=EF, b2=float(f"{b:.4e}"), lambda_over_ell=round(1 / np.sqrt(8 * b), 1), lambda_nm=round(lam, 1)))
    N = 2000; k = (np.arange(N) + 0.5) / N * 2 * np.pi - np.pi
    KX, KY = np.meshgrid(k, k, indexing="ij"); EF = -0.09
    (E0, E3, h03), (E1, E2, h12), (Es1, Es2) = blocks(KX, KY)
    for name, Ea, Eb, h in (("up-xz/dn-xy (coupled)", E0, E3, h03), ("up-xy/dn-yz (coupled)", E1, E2, h12),
                            ("up-xz/dn-yz sheets (not coupled for Neel along z)", Es1, Es2, None)):
        m = np.abs(Ea - EF) + np.abs(Eb - EF); i = np.unravel_index(np.argmin(m), m.shape)
        out["crossings"][name] = dict(min_sum_dist_meV=round(float(m[i]) * 1e3, 1), k_over_pi=[round(float(KX[i] / np.pi), 3), round(float(KY[i] / np.pi), 3)],
                                      coupling_there_meV=None if h is None else round(float(h[i]) * 1e3, 2))
    print(json.dumps(out, indent=1))
    json.dump(out, open(ROOT / "data/s3e_hotspots.json", "w"), indent=1)


if __name__ == "__main__":
    main()
