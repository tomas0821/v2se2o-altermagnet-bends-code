"""S0 (2026-11-14): extract the band dots of Cheng et al. PRMaterials 10, 114407 (2026), Fig. 2(e) (KV2Se2O, no SOC)
straight from the vector paths of the arXiv v1 PDF (page 3). No manual reading of the figure.

Calibration (from the PDF itself):
  axes background rect x 248.49..374.30 pt  -> path M-X-Gamma-Y-M, four equal segments of length pi (a = b);
  y tick labels '1', '0', '-1' centred at 184.29, 221.40, 258.52 pt -> 37.115 pt/eV, E = 0 at 221.405 pt.
  (Cross-check: axes rect 180.49..262.15 pt -> E range [-1.098, +1.102] eV, i.e. ylim = +-1.1.)
Colours: red (1,0,0) = TB spin up, blue (0,0,1) = TB spin down, grey (0.5,0.5,0.5) = first principles.
Dot centre = centre of its 1.17 pt marker. Digitisation uncertainty ~ +-0.05 pt = +-1.5 meV in E.

Clip paths matter: PyMuPDF returns every path, including dots of the neighbouring panels (b), (d), (f) that
matplotlib drew past their own axes and the PDF clips away. A first pass without clip tracking picked up 171
such red/blue dots per spin (rms 168 meV against the model). Only dots drawn under panel (e)'s own axes clip
(248.4, 180.5, 374.3, 262.1) with the centre inside it are kept; the legend markers are dropped too.

Run with system python3 (needs PyMuPDF):  python3 calc/s0_digitize_cheng_fig2e.py <cheng.pdf>
"""
import csv, sys
import fitz

PDF = sys.argv[1] if len(sys.argv) > 1 else "cheng2026_prmaterials.pdf"   # the published PDF of 10.1103/8czs-74vv
X0, X1 = 248.49, 374.30          # axes left/right (M ... M)
Y_ZERO, PT_PER_EV = 221.405, 37.115
AX = fitz.Rect(X0 - 0.7, 180.49 - 0.7, X1 + 0.7, 262.15 + 0.7)
COL = {(1.0, 0.0, 0.0): "TB_up", (0.0, 0.0, 1.0): "TB_dn", (0.5, 0.5, 0.5): "DFT"}

PANEL_E_CLIP = fitz.Rect(247.9, 179.9, 374.9, 262.7)   # panel (e) axes clip, +-0.5 pt
pg = fitz.open(PDF)[2]
rows, stack = [], {}
for d in pg.get_drawings(extended=True):
    lev = d.get("level", 0)
    if d["type"] == "clip":
        stack = {k: v for k, v in stack.items() if k < lev}
        stack[lev] = fitz.Rect(d["scissor"])
        continue
    if d["type"] == "group":
        continue
    r = d["rect"]
    if not AX.contains(r) or r.width > 1.3 or r.height > 1.3:
        continue
    key = tuple(round(c, 2) for c in (d.get("fill") or ()))
    if key not in COL:
        continue
    clip = fitz.Rect(-1e9, -1e9, 1e9, 1e9)
    for k, v in stack.items():
        if k < lev:
            clip &= v
    xc, yc = 0.5 * (r.x0 + r.x1), 0.5 * (r.y0 + r.y1)
    if not (clip.width < 130 and PANEL_E_CLIP.contains(clip) and clip.contains(fitz.Point(xc, yc))):
        continue
    s = 4.0 * (xc - X0) / (X1 - X0)          # path coordinate in units of pi: 0=M 1=X 2=Gamma 3=Y 4=M
    E = (Y_ZERO - yc) / PT_PER_EV
    rows.append((COL[key], round(s, 5), round(E, 5)))
out = "data/s0_cheng_fig2e_digitized.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["series", "s_over_pi_path_M_X_G_Y_M", "E_eV"]); w.writerows(rows)
from collections import Counter
print(Counter(r[0] for r in rows), "->", out)
