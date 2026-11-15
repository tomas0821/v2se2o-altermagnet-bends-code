"""S3c (2026-11-14): Elliott spin-relaxation estimate for KV2Se2O from Cheng Table 2 (SOC) -- the lambda_sf input.
No V2X2O value exists in the literature. The printed lambda_sf uses l = 0.5 nm; the paper uses l = 0.468 nm
(calc/s3e_hotspots.py: 24 nm at E_F, 18-34 nm over -0.15...+0.10 eV). SOC terms use the same k-convention as
calc/soc_corner.py (an assumption). Only the SOC inside the 4-band model is included (no Se 4p, no magnon spin
flip), so the estimate is an order of magnitude; both omissions would shorten lambda_sf.
Run: python3 calc/s3c_elliott.py KV2Se2O 800
"""
import sys
import numpy as np

P = {
    "KV2Se2O": [(-0.02692227, -0.29104709, 0.09206813, 0.04028453, 0.0067143, -0.00456873, 0.00288065),
                (-0.72424086, -0.18424356, -0.06059493, -0.11643046, 0.055216, 0.02809002, 0.00003625)],
    "RbV2Te2O": [(-0.05223832, -0.25078344, 0.0961765, 0.04660479, 0.01985274, -0.00953854, 0.00038615),
                 (-0.52960812, -0.07193622, -0.02131625, -0.10328106, 0.0721981, 0.02303562, 0.00116775)],
}


def band(p, kx, ky):
    e, t1x, t1y, t2, t3x, t3y = p[:6]
    return (e + 2 * t1x * np.cos(kx) + 2 * t1y * np.cos(ky) + 4 * t2 * np.cos(kx) * np.cos(ky)
            + 2 * t3x * np.cos(2 * kx) + 2 * t3y * np.cos(2 * ky))


def run(mat, N, EFs, T, soc_scale=1.0):
    pxz, pxy = P[mat]
    l1, l2 = pxz[6] * soc_scale, pxy[6] * soc_scale
    k = (np.arange(N) + 0.5) / N * 2 * np.pi - np.pi
    KX, KY = np.meshgrid(k, k, indexing="ij")
    KX, KY = KX.ravel(), KY.ravel()
    H = np.zeros((KX.size, 4, 4), complex)
    H[:, 0, 0] = band(pxz, KX, KY)
    H[:, 1, 1] = band(pxy, KX, KY)
    H[:, 2, 2] = band(pxz, KY, KX)  # x<->y
    H[:, 3, 3] = band(pxy, KY, KX)
    c = np.cos(KX / 2) * np.cos(KY / 2)
    s = np.sin(KX / 2) * np.sin(KY / 2)
    h03 = 4j * l1 * c - 4 * l2 * s
    h12 = 4 * l1 * c - 4j * l2 * s
    H[:, 0, 3] = h03; H[:, 3, 0] = np.conj(h03)
    H[:, 1, 2] = h12; H[:, 2, 1] = np.conj(h12)
    E, V = np.linalg.eigh(H)
    sz = (np.abs(V[:, 0, :]) ** 2 + np.abs(V[:, 1, :]) ** 2 - np.abs(V[:, 2, :]) ** 2 - np.abs(V[:, 3, :]) ** 2)
    b2 = (1 - np.abs(sz)) / 2
    kT = 8.617e-5 * T
    out = []
    for EF in EFs:
        x = (E - EF) / kT
        w = 1.0 / (4 * kT * np.cosh(np.clip(x, -50, 50) / 2) ** 2)  # -df/dE
        b2avg = (w * b2).sum() / w.sum()
        # fraction of FS weight with b2 > 0.1 (hot spots)
        hot = (w * (b2 > 0.1)).sum() / w.sum()
        out.append((EF, b2avg, hot))
    return out


if __name__ == "__main__":
    mat = sys.argv[1] if len(sys.argv) > 1 else "KV2Se2O"
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 1200
    EFs = [-0.15, -0.09, 0.0, 0.10]
    for T in (300.0, 30.0):
        for EF, b2, hot in run(mat, N, EFs, T):
            lam_over_l = 1 / np.sqrt(8 * b2)
            print(f"{mat} N={N} T={T:.0f}K EF={EF:+.2f} <b2>={b2:.3e} hotfrac={hot:.3e} "
                  f"tau_s/tau={1/(4*b2):.3e} lambda_sf/l={lam_over_l:.1f} lambda_sf(l=0.5nm)={0.5*lam_over_l:.1f} nm")
