"""Palette, matplotlib style, color helpers and the layout audit used by make_fig0.py."""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import numpy as np

matplotlib.use("Agg")

ORACLE = "#7030A0"
MIX    = "#C55A11"
ROUTER = "#3178C6"
FLOOR  = "#2E5E1C"
BEST   = "#8A8A8A"
MEMBER = "#2E7D8C"
INK, MUT = "#1A1A1A", "#6B6B6B"
GRID, SPINE = "#E6E6E6", "#B0B0B0"
RULE = "#4A4A4A"


def tint(hexcol, f):
    """Mix a color with white; f=1 is the color itself."""
    r, g, b = (int(hexcol[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(int(round(f * v + (1 - f) * 255))
                                   for v in (r, g, b))


def shade(hexcol, f=0.82):
    """Darken a color toward black, for text that must carry more contrast."""
    r, g, b = (int(hexcol[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(int(round(f * v)) for v in (r, g, b))

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 6.5,
    "text.color": INK,
    "axes.edgecolor": SPINE,
    "axes.linewidth": 0.55,
    "axes.labelcolor": INK,
    "xtick.color": SPINE,
    "ytick.color": SPINE,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "pdf.fonttype": 42,
})


def audit(fig):
    """Measure real rendered text extents; count overlaps and escapes."""
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    boxes = []
    for t in fig.findobj(mtext.Text):
        if not t.get_text().strip() or not t.get_visible():
            continue
        try:
            boxes.append((t, t.get_window_extent(ren)))
        except Exception:
            pass
    n = 0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            (ta, a), (tb, b) = boxes[i], boxes[j]
            ix = min(a.x1, b.x1) - max(a.x0, b.x0)
            iy = min(a.y1, b.y1) - max(a.y0, b.y0)
            if ix > 1.5 and iy > 1.5:
                print(f"  TEXT/TEXT  {ta.get_text()!r} X {tb.get_text()!r}")
                n += 1
    for i, ax in enumerate(fig.axes):
        b = ax.get_position()
        if b.x0 < -1e-6 or b.x1 > 1 + 1e-6 or b.y0 < -1e-6 or b.y1 > 1 + 1e-6:
            print(f"  AXES/BOUNDS axes {i} spans x[{b.x0:.4f},{b.x1:.4f}] "
                  f"y[{b.y0:.4f},{b.y1:.4f}] -- outside the figure, so its "
                  f"content is clipped")
            n += 1

    W, H = fig.get_figwidth() * fig.dpi, fig.get_figheight() * fig.dpi
    for t, bb in boxes:
        if bb.x0 < -2 or bb.y0 < -2 or bb.x1 > W + 2 or bb.y1 > H + 2:
            print(f"  TEXT/BOUNDS {t.get_text()!r} escapes the figure")
            n += 1
    # text placed in DATA coordinates belongs to its panel: catch anything
    # spilling past the panel edge, which a figure-level check cannot see
    for ax in fig.axes:
        ab = ax.get_window_extent(ren)
        for t in ax.texts:
            if not t.get_text().strip() or t.get_transform() is not ax.transData:
                continue
            bb = t.get_window_extent(ren)
            if bb.x0 < ab.x0 - 1 or bb.x1 > ab.x1 + 1:
                print(f"  TEXT/PANEL  {t.get_text()[:28]!r} spills outside "
                      f"its panel")
                n += 1
    # decorations in ANY axes (key rules, mastheads) against text in ANY other:
    # these live in different axes, so the per-axes checks below cannot see them
    rules = []
    for ax in fig.axes:
        for l in ax.lines:
            if l.get_clip_on():
                continue                      # confined to its own axes
            try:
                rules.append((l, l.get_transform().transform(l.get_xydata())))
            except Exception:
                pass
    for t, bb in boxes:
        for l, disp in rules:
            hit = ((disp[:, 0] >= bb.x0) & (disp[:, 0] <= bb.x1) &
                   (disp[:, 1] >= bb.y0) & (disp[:, 1] <= bb.y1))
            if hit.any():
                print(f"  TEXT/RULE   {t.get_text()[:30]!r} sits on a rule")
                n += 1
                break

    for ax in fig.axes:
        lines = [l for l in ax.lines if l.get_gid() != "deco"]
        leg = ax.get_legend()
        ax_texts = list(ax.texts) + (list(leg.get_texts()) if leg else [])
        for t in ax_texts:
            if not t.get_text().strip():
                continue
            bb = t.get_window_extent(ren)
            for l in lines:
                xy = l.get_xydata()
                if len(xy) < 2:
                    continue
                seg = np.concatenate([
                    np.linspace(xy[k], xy[k + 1], 24)
                    for k in range(len(xy) - 1)])
                disp = ax.transData.transform(seg)
                hit = ((disp[:, 0] >= bb.x0 - 1) & (disp[:, 0] <= bb.x1 + 1) &
                       (disp[:, 1] >= bb.y0 - 1) & (disp[:, 1] <= bb.y1 + 1))
                if hit.any():
                    print(f"  TEXT/CURVE {t.get_text()!r} lies on a curve")
                    n += 1
                    break
    return n
