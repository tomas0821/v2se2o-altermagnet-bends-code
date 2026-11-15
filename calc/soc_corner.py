"""Two-sublattice Kwant builders for the spin-orbit bound of the L-bend (Cheng et al., PRMaterials 10, 114407
(2026), Table 2), used by s2_corners.soc_bound().

Model (no SOC): spin up = sublattice A orbitals {xz, xy}, spin down = sublattice B {yz, xy} (x <-> y).
H_orb(k) = e + 2 t1x cos kx + 2 t1y cos ky + 4 t2 cos kx cos ky + 2 t3x cos 2kx + 2 t3y cos 2ky.
SOC (Table 2 fit): H_03 = 4i l1 c - 4 l2 s, H_12 = 4 l1 c - 4i l2 s, c = cos(kx/2) cos(ky/2), s = sin(kx/2) sin(ky/2),
basis order 0 xz^A_up, 1 xy^A_up, 2 yz^B_dn, 3 xy^B_dn (Neel vector along z). Real-space form: A at (1/2, 0),
B at (0, 1/2); c and s expanded over the four A-B half-diagonal vectors d (c -> 1/4 each; s -> -1/4 for
d = +-(1/2, 1/2), +1/4 for d = +-(1/2, -1/2)). This phase convention is an assumption (the source does not state it),
so the result is used only as an order-of-magnitude bound.
"""
import numpy as np
import kwant

T1 = {  # Table 1 (no SOC): e, t1x, t1y, t2, t3x, t3y ; rows xz, xy
    "KV2Se2O": [(-0.02378745, -0.29092429, 0.09126757, 0.04004989, 0.00558104, -0.00366236),
                (-0.71549718, -0.2089599, -0.0856611, -0.13369163, 0.02983547, 0.00226622)],
}
T2 = {  # Table 2 (SOC fit): same + lambda
    "KV2Se2O": [(-0.02692227, -0.29104709, 0.09206813, 0.04028453, 0.0067143, -0.00456873, 0.00288065),
                (-0.72424086, -0.18424356, -0.06059493, -0.11643046, 0.055216, 0.02809002, 0.00003625)],
}

# ------------------------------------------------------------------ spin-separated (no SOC) builder
latS = kwant.lattice.square(a=1, norbs=2)


def hop_terms(pars, swap):
    terms = {}
    for (dx, dy), i in (((1, 0), 1), ((0, 1), 2), ((2, 0), 4), ((0, 2), 5)):
        if swap:
            dx, dy = dy, dx
        terms[(dx, dy)] = np.diag([p[i] for p in pars])
    terms[(1, 1)] = terms[(1, -1)] = np.diag([p[3] for p in pars])
    return terms


def fill(b, pars, swap, sites):
    E0 = np.diag([p[0] for p in pars])
    for s in sites:
        b[latS(*s)] = E0
    for d, M in hop_terms(pars, swap).items():
        b[kwant.builder.HoppingKind(d, latS)] = M


def lead(pars, swap, direction, cells):
    ld = kwant.Builder(kwant.TranslationalSymmetry(direction))
    fill(ld, pars, swap, cells)
    return ld


def straight(pars, swap, W, L=8):
    b = kwant.Builder()
    fill(b, pars, swap, [(x, y) for x in range(L) for y in range(W)])
    l = lead(pars, swap, (-2, 0), [(x, y) for x in range(2) for y in range(W)])
    b.attach_lead(l); b.attach_lead(l.reversed())
    return b.finalized()


def lbend(pars, swap, W, chamfer=0):
    sites = [(x, y) for x in range(W) for y in range(W)
             if not (chamfer and (x - 0) + (W - 1 - y) < 0)]  # placeholder (no-op), replaced below
    # outer corner of an L from the left lead (x<0) to the top lead (y>=W) is at (0, 0)? No: the left arm
    # occupies y in [0,W), the top arm x in [0,W); the OUTER corner is (W-1, 0) side? The bend turns from +x
    # travel to +y travel, so the outer corner is (W-1, 0). Chamfer removes sites with (W-1-x) + y < chamfer.
    sites = [(x, y) for x in range(W) for y in range(W) if (W - 1 - x) + y >= chamfer]
    b = kwant.Builder()
    fill(b, pars, swap, sites)
    left = lead(pars, swap, (-2, 0), [(x, y) for x in range(2) for y in range(W)])
    top = lead(pars, swap, (0, 2), [(x, y) for x in range(W) for y in range(2)])
    b.attach_lead(left); b.attach_lead(top)
    return b.finalized()


def tjunction(pars, swap, W):
    b = kwant.Builder()
    fill(b, pars, swap, [(x, y) for x in range(W) for y in range(W)])
    left = lead(pars, swap, (-2, 0), [(x, y) for x in range(2) for y in range(W)])
    top = lead(pars, swap, (0, 2), [(x, y) for x in range(W) for y in range(2)])
    bot = lead(pars, swap, (0, -2), [(x, y) for x in range(W) for y in range(2)])
    b.attach_lead(left); b.attach_lead(top); b.attach_lead(bot)
    return b.finalized()


def iso(pars):
    """non-magnetic reference: average x/y hoppings so both spins see the same isotropic band."""
    out = []
    for p in pars:
        e, t1x, t1y, t2, t3x, t3y = p[:6]
        out.append((e, 0.5 * (t1x + t1y), 0.5 * (t1x + t1y), t2, 0.5 * (t3x + t3y), 0.5 * (t3x + t3y)))
    return out


def T(f, E, i=1, j=0):
    return kwant.smatrix(f, E).transmission(i, j)


# ------------------------------------------------------------------ two-sublattice SOC builder (L-bend)
latAB = kwant.lattice.general([(1, 0), (0, 1)], [(0.5, 0.0), (0.0, 0.5)], norbs=2)
SA, SB = latAB.sublattices
D_HALF = {  # B(x,y) at (x, y+1/2) -> A at B + d ; A(x', y') at (x'+1/2, y')
    (0.5, 0.5): (0, 1), (-0.5, 0.5): (-1, 1), (0.5, -0.5): (0, 0), (-0.5, -0.5): (-1, 0)}


def soc_bend(pars2, W, soc_on=True):
    """L-bend on the two-sublattice lattice. A carries (xz_up, xy_up), B carries (yz_dn, xy_dn).
    SOC only inside the W x W corner (leads spin-conserving, so spin-resolved transmissions are defined)."""
    pA = [p[:6] for p in pars2]
    l1, l2 = pars2[0][6], pars2[1][6]

    def intra(b, sites_xy):
        EA = np.diag([p[0] for p in pA]); EB = EA
        for (x, y) in sites_xy:
            b[SA(x, y)] = EA
            b[SB(x, y)] = EB
        for d, M in hop_terms(pA, False).items():
            b[kwant.builder.HoppingKind(d, SA, SA)] = M
        for d, M in hop_terms(pA, True).items():
            b[kwant.builder.HoppingKind(d, SB, SB)] = M

    b = kwant.Builder()
    corner = [(x, y) for x in range(W) for y in range(W)]
    intra(b, corner)
    if soc_on:
        cset = set(corner)
        for (x, y) in corner:
            for (dx, dy), (ox, oy) in D_HALF.items():
                a = (x + ox, y + oy)
                if a not in cset:
                    continue
                cf = 0.25
                sf = -0.25 if dx * dy > 0 else 0.25
                # B->A block: rows A(xz,xy), cols B(yz,xy). H_03: A-xz <- B-xy ; H_12: A-xy <- B-yz.
                M = np.zeros((2, 2), complex)
                M[0, 1] = 4j * l1 * cf - 4 * l2 * sf
                M[1, 0] = 4 * l1 * cf - 4j * l2 * sf
                b[SA(*a), SB(x, y)] = M

    def mk_lead(direction, cells):
        ld = kwant.Builder(kwant.TranslationalSymmetry(direction),
                           conservation_law=lambda site: -np.eye(2) if site.family == SA else np.eye(2))
        intra(ld, cells)
        return ld

    b.attach_lead(mk_lead((-2, 0), [(x, y) for x in range(2) for y in range(W)]))
    b.attach_lead(mk_lead((0, 2), [(x, y) for x in range(W) for y in range(2)]))
    return b.finalized()
