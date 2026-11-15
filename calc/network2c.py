"""Two-current diffusive network solver (2026-11-14), shared by S3b and S4.

Square grid of cells; each cell carries a spin-up and a spin-down electrochemical potential. Bonds between
neighbouring cells have conductance sx (horizontal) or sy (vertical), separately per spin (the altermagnet's
spin-split anisotropy). Optional on-site spin relaxation: conductance g_sf between the up and down potential of
each cell, g_sf = sigma_bar / (2 lambda_sf^2) with lambda_sf in cells (Valet-Fert form for equal sigma_bar).
Contacts are nonmagnetic: a fixed contact pins both spins of its cells to V; a floating contact (a voltage probe)
merges all its cells and both spins into one unknown node, so it draws no net current.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

NB = ((1, 0, 0), (-1, 0, 0), (0, 1, 1), (0, -1, 1))   # (di, dj, axis) axis 0 = x bond, 1 = y bond


def solve(cells, cond, fixed=(), floating=(), g_sf=0.0):
    """cells: iterable of (i, j). cond: {"up": (sx, sy), "dn": (sx, sy)}.
    fixed: list of (set_of_cells, V). floating: list of set_of_cells.
    Returns dict(phi={(cell, spin): V}, V_float=[...], I_fixed=[{"up": I, "dn": I}, ...]) where I_fixed is the
    current flowing from each fixed contact into the device, per spin."""
    cells = sorted(set(cells))
    cset = set(cells)
    spins = ("up", "dn")
    fixed_of = {}
    for k, (cs, V) in enumerate(fixed):
        for c in cs:
            fixed_of[c] = k
    float_of = {}
    for k, cs in enumerate(floating):
        for c in cs:
            float_of[c] = k
    # unknown numbering
    node = {}
    n = 0
    for k in range(len(floating)):
        node[("F", k)] = n; n += 1
    for c in cells:
        if c in fixed_of or c in float_of:
            continue
        for s in spins:
            node[(c, s)] = n; n += 1

    def key(c, s):
        if c in fixed_of:
            return ("D", fixed_of[c])
        if c in float_of:
            return node[("F", float_of[c])]
        return node[(c, s)]

    rows, cols, vals = [], [], []
    b = np.zeros(n)

    def add(a, bb, g):
        """conductance g between endpoints a and bb (int unknown or ("D", k) Dirichlet)."""
        if a == bb:
            return
        for x, y in ((a, bb), (bb, a)):
            if isinstance(x, tuple):
                continue
            rows.append(x); cols.append(x); vals.append(g)
            if isinstance(y, tuple):
                b[x] += g * fixed[y[1]][1]
            else:
                rows.append(x); cols.append(y); vals.append(-g)

    for c in cells:
        for di, dj, ax in NB[::2]:      # each bond once (+x and +y)
            nb = (c[0] + di, c[1] + dj)
            if nb not in cset:
                continue
            for s in spins:
                add(key(c, s), key(nb, s), cond[s][ax])
        if g_sf > 0:
            add(key(c, "up"), key(c, "dn"), g_sf)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    x = spla.spsolve(A, b) if n else np.zeros(0)

    def pot(c, s):
        k = key(c, s)
        return fixed[k[1]][1] if isinstance(k, tuple) else x[k]

    phi = {(c, s): pot(c, s) for c in cells for s in spins}
    I_fixed = []
    for k, (cs, V) in enumerate(fixed):
        d = {}
        for s in spins:
            I = 0.0
            for c in cs:
                for di, dj, ax in NB:
                    nb = (c[0] + di, c[1] + dj)
                    if nb in cset and nb not in cs:
                        I += cond[s][ax] * (V - phi[(nb, s)])
            d[s] = I
        I_fixed.append(d)
    return dict(phi=phi, V_float=[x[node[("F", k)]] for k in range(len(floating))], I_fixed=I_fixed)


def L_cells(W, L1, L2):
    """L-bend: [100] arm from x = -L1 (contact) into a W x W corner, [010] arm up to y = W + L2 - 1 (contact)."""
    cells = {(i, j) for i in range(-L1, W) for j in range(W)} | {(i, j) for i in range(W) for j in range(W, W + L2)}
    return cells, {(-L1, j) for j in range(W)}, {(i, W + L2 - 1) for i in range(W)}


def T_cells(W, Ls, La):
    cells = ({(i, j) for i in range(-Ls, W) for j in range(W)} | {(i, j) for i in range(W) for j in range(W, W + La)}
             | {(i, j) for i in range(W) for j in range(-La, 0)})
    return cells, {(-Ls, j) for j in range(W)}, {(i, W + La - 1) for i in range(W)}, {(i, -La) for i in range(W)}
