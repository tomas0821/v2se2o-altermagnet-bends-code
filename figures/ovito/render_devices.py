"""OVITO renders of the device geometries on the single V2Se2O layer, for Fig. 1(c) (draft 1, 2026-11-14).

Schematic layer geometry (P4/mmm, anti-CuO2 / inverse-Lieb V2O plane): a = 3.952 A; per cell O at (0, 0, 0),
V_A at (a/2, 0, 0) (spin up, d_xz), V_B at (0, a/2, 0) (spin down, d_yz), Se at (a/2, a/2, +-1.84 A) (V-Se ~ 2.7 A;
schematic). Kwant site (x, y) = one cell. Devices use W = 5 cells so the atoms stay visible; the calculations use
W = 10-51. Top view (camera along -z), transparent background; image extents (in A) go to devices.json for the
matplotlib panel. Usage (project root): bash figures/ovito/render.sh figures/ovito/render_devices.py [--preview]
"""
import json, sys
import numpy as np
from ovito.data import DataCollection, ParticleType
from ovito.pipeline import Pipeline, StaticSource
from ovito.vis import TachyonRenderer, Viewport

A, ZSE = 3.952, 1.84
PREVIEW = "--preview" in sys.argv
PX_PER_A = 14 if PREVIEW else 28
W, L, LP = 5, 6, 3


def cells(kind):
    if kind == "straight":
        return {(x, y) for x in range(-L, W + L) for y in range(W)}, set()
    if kind == "L":
        c = {(x, y) for x in range(-L, W) for y in range(W)} | {(x, y) for x in range(W) for y in range(W, W + L)}
        probes = {(x, 0) for x in range(W, W + LP)} | {(W - 1, y) for y in range(-LP, 0)}
        return c, probes
    if kind == "T":
        return ({(x, y) for x in range(-L, W) for y in range(W)} | {(x, y) for x in range(W) for y in range(W, W + L)}
                | {(x, y) for x in range(W) for y in range(-L, 0)}), set()
    if kind == "45":
        Wd = int(round(W * np.sqrt(2)))
        c = {(x, y) for x in range(-L, 1) for y in range(W)}
        c |= {(x, y) for x in range(0, Wd + 2 * L) for y in range(0, Wd + 2 * L) if 0 <= (y - x) < Wd and x + y < Wd + 2 * L}
        return c, set()
    raise ValueError(kind)


def atoms(cs, probe=False):
    pos, typ = [], []
    for (x, y) in sorted(cs):
        X, Y = x * A, y * A
        pos += [(X, Y, 0.0), (X + A / 2, Y, 0.0), (X, Y + A / 2, 0.0), (X + A / 2, Y + A / 2, ZSE), (X + A / 2, Y + A / 2, -ZSE)]
        typ += [1, 5 if probe else 2, 5 if probe else 3, 4, 4]
    return pos, typ


def render_all(out, gap=3):
    """all four devices in one scene, 2 x 2, same scale; returns the image extent and label anchors (A)."""
    order = [("straight", "straight [100]"), ("L", "L-bend + probe pair"), ("T", "T-junction"), ("45", "45° [100]→[110]")]
    boxes = {}
    for k, _ in order:
        c, pr = cells(k); allc = np.array(sorted(c | pr))
        boxes[k] = (allc[:, 0].min(), allc[:, 0].max() + 1, allc[:, 1].min(), allc[:, 1].max() + 1)
    colw = [max(boxes[order[i][0]][1] - boxes[order[i][0]][0], boxes[order[i + 2][0]][1] - boxes[order[i + 2][0]][0]) for i in (0, 1)]
    rowh = [max(boxes[order[j][0]][3] - boxes[order[j][0]][2] for j in r) for r in ((0, 1), (2, 3))]
    pos, typ, labels = [], [], {}
    for idx, (k, lab) in enumerate(order):
        col, row = idx % 2, idx // 2
        x0 = sum(colw[:col]) + gap * col; y0 = (rowh[1] + gap + 2) if row == 0 else 0
        bx = boxes[k]
        dx, dy = x0 - bx[0] + (colw[col] - (bx[1] - bx[0])) / 2, y0 - bx[2] + (rowh[row] - (bx[3] - bx[2])) / 2
        c, pr = cells(k)
        p1, t1 = atoms({(x + dx, y + dy) for x, y in c}); p2, t2 = atoms({(x + dx, y + dy) for x, y in pr}, probe=True)
        pos += p1 + p2; typ += t1 + t2
        labels[k] = dict(text=lab, x=float((x0 + colw[col] / 2) * A), y=float((y0 + rowh[row] + 0.6) * A))
    pos, typ = np.array(pos), np.array(typ)
    dc = DataCollection()
    dc.create_cell(np.array([[500.0, 0, 0, -250], [0, 500.0, 0, -250], [0, 0, 50.0, -25]]), pbc=(False, False, False)).vis.enabled = False
    p = dc.create_particles(count=len(pos))
    p.create_property("Position", data=pos)
    t = p.create_property("Particle Type", data=typ)
    t.types.append(ParticleType(id=1, name="O", color=(0.25, 0.25, 0.27), radius=0.45))
    t.types.append(ParticleType(id=2, name="V_A", color=(0.89, 0.29, 0.28), radius=0.85))     # spin up, #e34948
    t.types.append(ParticleType(id=3, name="V_B", color=(0.16, 0.47, 0.84), radius=0.85))     # spin down, #2a78d6
    t.types.append(ParticleType(id=4, name="Se", color=(0.86, 0.84, 0.76), radius=0.55))
    t.types.append(ParticleType(id=5, name="V_probe", color=(0.93, 0.63, 0.0), radius=0.85))  # probe arms, #eda100
    pipe = Pipeline(source=StaticSource(data=dc)); pipe.add_to_scene()
    lo, hi = pos[:, :2].min(0) - 1.5, pos[:, :2].max(0) + 1.5
    ctr, half = (lo + hi) / 2, (hi - lo) / 2
    vp = Viewport(type=Viewport.Type.Ortho, camera_dir=(0, 0, -1), camera_pos=(ctr[0], ctr[1], 30.0), fov=half[1])
    vp.camera_up = (0, 1, 0)
    rend = (TachyonRenderer(ambient_occlusion=False, antialiasing_samples=4) if PREVIEW else
            TachyonRenderer(ambient_occlusion=True, ambient_occlusion_brightness=0.8, ambient_occlusion_samples=12,
                            direct_light_intensity=0.9, shadows=False, antialiasing_samples=12))
    size = (int(round(2 * half[0] * PX_PER_A)), int(round(2 * half[1] * PX_PER_A)))
    vp.render_image(size=size, filename=out, background=(1, 1, 1), alpha=True, renderer=rend)
    pipe.remove_from_scene()
    return dict(extent=[float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])], labels=labels)


if __name__ == "__main__":
    info = render_all("figures/ovito/devices_all.png")
    json.dump(info, open("figures/ovito/devices.json", "w"), indent=1)
    print(info["extent"])
