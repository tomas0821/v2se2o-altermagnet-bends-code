"""Shared model code for the production calculations (S0 onward), 2026-11-14.

Model: Cheng et al., PRMaterials 10, 114407 (2026) [chengRealisticTightbindingModel2026], four-band TB without SOC.
The four bands decouple into spin x orbital blocks:
  spin up   = sublattice A, orbitals {d_xz, d_xy}, parameters as in Table 1;
  spin down = sublattice B, orbitals {d_yz, d_xy}, the same parameters with x <-> y (H_dn = M^-1 H_up M).
Each orbital band:  E(k) = e + 2 t1x cos kx + 2 t1y cos ky + 4 t2 cos kx cos ky + 2 t3x cos 2kx + 2 t3y cos 2ky.
Energies in eV, measured from the DFT Fermi level; the ARPES-matched Fermi level is E_F = -0.09 eV
(Jiang et al. Nat. Phys. 21, 754 (2025), Fig. 2e) [jiangMetallicRoomtemperatureDwave2025].

Minimal-model reference: the single-band KV2Se2O fit of Yang, Zhao, Yan & Chen, arXiv:2602.11694v2,
  xi_{k,s} = -2t(cos kx + cos ky) - 4t' cos kx cos ky - 2 t_j s_z (cos kx - cos ky) - mu,
  t = 90.375 meV, t' = -36.9375 meV, t_j = 194.24 meV, mu = 75.25 meV.
Rewritten in the same orbital form (spin up, s_z = +1):
  e = -mu, t1x = -(t + t_j), t1y = -(t - t_j), t2 = -t', t3x = t3y = 0.
Its zero is the DFT Fermi level (the paper fits the DFT bands), so it is compared at the same E.

Lattice constant a = 3.952 Angstrom for KV2Se2O (Cheng Table 1 header values).
Run everything with OMP_NUM_THREADS=1.
"""
import numpy as np

A_KV2Se2O = 0.3952  # nm
E_F = -0.09         # eV

# Cheng Table 1 (no SOC), KV2Se2O: (e, t1x, t1y, t2, t3x, t3y)
CHENG_XZ = (-0.02378745, -0.29092429, 0.09126757, 0.04004989, 0.00558104, -0.00366236)
CHENG_XY = (-0.71549718, -0.2089599, -0.0856611, -0.13369163, 0.02983547, 0.00226622)

_t, _tp, _tj, _mu = 0.090375, -0.0369375, 0.19424, 0.07525
SINGLE_BAND = (-_mu, -(_t + _tj), -(_t - _tj), -_tp, 0.0, 0.0)

def _iso(p):
    """non-magnetic isotropic reference : x and y hoppings averaged, so both spins share one band."""
    e, t1x, t1y, t2, t3x, t3y = p
    return (e, 0.5 * (t1x + t1y), 0.5 * (t1x + t1y), t2, 0.5 * (t3x + t3y), 0.5 * (t3x + t3y))


MODELS = {
    "cheng": {"xz": CHENG_XZ, "xy": CHENG_XY},
    "single_band": {"xz": SINGLE_BAND},
    "iso": {"xz": _iso(CHENG_XZ), "xy": _iso(CHENG_XY)},
}


def shifted(model, xy_shift):
    """Cheng model with the d_xy on-site energy shifted (pocket-shift error bar)."""
    m = dict(MODELS[model])
    if "xy" in m:
        p = list(m["xy"]); p[0] += xy_shift; m["xy"] = tuple(p)
    return m


def swap(p):
    e, t1x, t1y, t2, t3x, t3y = p
    return (e, t1y, t1x, t2, t3y, t3x)


def orbital_params(model_dict, spin):
    """list of (orbital name, params) for one spin; spin 'up' or 'dn'."""
    return [(o, p if spin == "up" else swap(p)) for o, p in model_dict.items()]


def band(p, kx, ky):
    e, t1x, t1y, t2, t3x, t3y = p
    return (e + 2 * t1x * np.cos(kx) + 2 * t1y * np.cos(ky) + 4 * t2 * np.cos(kx) * np.cos(ky)
            + 2 * t3x * np.cos(2 * kx) + 2 * t3y * np.cos(2 * ky))


# ---------------------------------------------------------------- Fermi-surface channel counts (wide limit)
def _frac_below(e1, e2):
    """fraction of a straight segment between band values e1, e2 (relative to E) lying below E."""
    both = (e1 < 0) & (e2 < 0)
    mixed = (e1 < 0) != (e2 < 0)
    f = np.where(both, 1.0, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        fm = np.where(e1 < 0, e1 / (e1 - e2), e2 / (e2 - e1))
    return np.where(mixed, fm, f)


def wide_limit_counts(p, E, direction, n_par=1440, n_perp=4000):
    """Channels per unit perpendicular width (lattice units) of one orbital band at energy E.

    Returns (sharvin, projected):
      sharvin   = (1/2pi) * integral dk_perp n_+(k_perp): every right-moving Fermi crossing counts
                  (bulk Sharvin count; ignores the edges).
      projected = (1/4pi) * integral dk_par |d ell / d k_par|, where ell(k_par) is the length of the
                  transverse line at fixed k_par lying below E. This is the wide-strip limit of the
                  number of subbands crossing E when the edges conserve k_par (Weyl count of a hard-wall
                  strip). It differs from Sharvin when the transverse occupied length gains and loses at
                  the same k_par, i.e. when edge reflection sends a carrier back (retro-reflection).
    direction '100': k_par = kx (period 2pi), transverse ky in [-pi, pi).
    direction '110': lead phase phi = kx + ky in [-pi, pi); transverse line k = (phi/2 + v, phi/2 - v),
                     v in [0, 2pi), length element sqrt(2) dv; per unit perpendicular width.
    """
    if direction == "100":
        kpar = np.linspace(-np.pi, np.pi, n_par, endpoint=False)
        kp = np.linspace(-np.pi, np.pi, n_perp + 1)
        KX, KY = np.meshgrid(kpar, kp, indexing="ij")
        e = band(p, KX, KY) - E
        ell = _frac_below(e[:, :-1], e[:, 1:]).sum(1) * (2 * np.pi / n_perp)
        tv = np.abs(np.diff(np.r_[ell, ell[0]])).sum()
        projected = tv / (4 * np.pi)
        # Sharvin: lines of fixed ky, sweep kx over a full period; count up-crossings
        s = np.linspace(0, 2 * np.pi, 4 * n_par, endpoint=False)
        ky = np.linspace(-np.pi, np.pi, n_perp // 2, endpoint=False)
        S, KY2 = np.meshgrid(s, ky, indexing="ij")
        e2 = band(p, S, KY2) - E
        up = ((e2 < 0) & (np.roll(e2, -1, 0) >= 0)).sum()
        sharvin = up * (2 * np.pi / (n_perp // 2)) / (2 * np.pi)
        return sharvin, projected
    if direction == "110":
        phi = np.linspace(-np.pi, np.pi, n_par, endpoint=False)
        v = np.linspace(0, 2 * np.pi, n_perp + 1)
        PH, V = np.meshgrid(phi, v, indexing="ij")
        e = band(p, PH / 2 + V, PH / 2 - V) - E
        ell = _frac_below(e[:, :-1], e[:, 1:]).sum(1) * (2 * np.pi / n_perp) * np.sqrt(2)
        tv = np.abs(np.diff(np.r_[ell, ell[0]])).sum()
        # d(phi) -> d(k_par): k_par = phi / sqrt2 ... the per-width count uses phi directly:
        # M(phi) = (W_perp / 2pi) ell(phi);  N = (1/2) * total variation of M over one period of phi.
        projected = tv / (4 * np.pi)
        # Sharvin along (1,1): lines kx = s, ky = s + d; d in [0, 2pi) -> k_perp = d / sqrt2
        s = np.linspace(0, 2 * np.pi, 4 * n_par, endpoint=False)
        d = np.linspace(0, 2 * np.pi, n_perp // 2, endpoint=False)
        S, D = np.meshgrid(s, d, indexing="ij")
        e2 = band(p, S, S + D) - E
        up = ((e2 < 0) & (np.roll(e2, -1, 0) >= 0)).sum()
        sharvin = up * (2 * np.pi / (n_perp // 2)) / np.sqrt(2) / (2 * np.pi)
        return sharvin, projected
    raise ValueError(direction)


# ---------------------------------------------------------------- Kwant builders (one orbital, norbs = 1)
def _kwant():
    import kwant  # imported lazily so the band-only scripts run without Kwant
    return kwant


def _hops(p):
    e, t1x, t1y, t2, t3x, t3y = p
    return {(1, 0): t1x, (0, 1): t1y, (1, 1): t2, (1, -1): t2, (2, 0): t3x, (0, 2): t3y}


def _fill(b, lat, p, sites):
    kwant = _kwant()
    for s in sites:
        b[lat(*s)] = p[0]
    for d, t in _hops(p).items():
        if t != 0.0:
            b[kwant.builder.HoppingKind(d, lat)] = t


def lead_modes_100(p, W, E):
    """number of right-moving modes of a [100] lead W sites wide (one orbital band)."""
    kwant = _kwant()
    lat = kwant.lattice.square(a=1, norbs=1)
    ld = kwant.Builder(kwant.TranslationalSymmetry((-2, 0)))   # period 2 so the (2,0) hop is nearest-cell
    _fill(ld, lat, p, [(x, y) for x in range(2) for y in range(W)])
    pm, _ = ld.finalized().modes(energy=E)
    return pm.velocities.size // 2


_DIAG_CACHE = {}


def lead_modes_110(p, Wd, E):
    """right-moving modes of a [110] lead with Wd atomic rows (perpendicular width Wd/sqrt2 lattice units).
    A standalone [110] lead fails to finalise in Kwant 1.5, so it is attached to a short scattering region
    and read back from the finalised system."""
    kwant = _kwant()
    key = (p, Wd)
    if key not in _DIAG_CACHE:
        lat = kwant.lattice.square(a=1, norbs=1)
        inside = lambda x, y: 0 <= (y - x) < Wd
        L = 4
        b = kwant.Builder()
        _fill(b, lat, p, [(x, y) for x in range(-Wd, L + Wd) for y in range(-Wd, L + 2 * Wd)
                          if inside(x, y) and 0 <= x + y < 2 * L])
        ld = kwant.Builder(kwant.TranslationalSymmetry((-1, -1)))
        _fill(ld, lat, p, [(x, y) for x in range(-Wd, Wd + 2) for y in range(-Wd, 2 * Wd + 2)
                           if inside(x, y) and 0 <= x + y < 2])
        b.attach_lead(ld)
        _DIAG_CACHE[key] = b.finalized().leads[0]
    pm, _ = _DIAG_CACHE[key].modes(energy=E)
    return pm.velocities.size // 2
