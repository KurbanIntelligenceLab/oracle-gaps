"""Figure 1: the re-deal null on one test prompt."""
import pathlib as _pl, sys as _sys
_REPO_ROOT = next(p for p in _pl.Path(__file__).resolve().parents if (p / "lib").is_dir())
_sys.path.insert(0, str(_REPO_ROOT / "lib"))  # shared modules
import argparse, json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import numpy as np

ROOT = _REPO_ROOT
sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import audit, tint, shade, ORACLE, MIX, ROUTER, MEMBER, BEST, INK, MUT, GRID, SPINE, RULE  # noqa: E402

OUTPDF = Path(__file__).resolve().parent / "out" / "fig0.pdf"
TAU = 0.20
N_PERM = 2000
SEED = 0
FS, FST = 7.0, 8.0
plt.rcParams.update({"font.size": FS})


def load_prompt():
    d = np.load(ROOT / "data" / "geometry3k" / "verdicts.npz")
    V, spl, pids = d["verdicts"], d["split"], d["prompt_ids"]
    test = np.where(spl == "test")[0]
    Vt, pt = V[test], pids[test]
    rep = json.load(open(ROOT / "results" / "e7_routers" / "sh_0000.json"))
    asg = np.array(rep["routers"]["tfidf_logreg"]["per_init"][0]["assignment"])
    assert len(asg) == len(pt)
    rng = np.random.default_rng(SEED)
    best_i, best_score = None, 9
    for i in range(len(pt)):
        S = int(Vt[i].sum())
        if not 8 <= S <= 12:
            continue
        obs = Vt[i].sum(1).max()
        flat = Vt[i].ravel()
        mx = np.array([rng.permutation(flat).reshape(5, 16).sum(1).max() for _ in range(400)])
        p = float(np.mean(mx >= obs))
        if abs(p - 0.5) < best_score:
            best_i, best_score = i, abs(p - 0.5)
    i = best_i
    return pt[i], Vt[i].astype(int), int(asg[i]), rng


def build():
    pid, G, r_pick, rng = load_prompt()
    M, k = G.shape
    flat = G.ravel()
    perm = rng.permutation(flat.size)                            # one re-deal: new cell i holds old cell perm[i]
    G2 = flat[perm].reshape(M, k)
    mx = np.array([rng.permutation(flat).reshape(M, k).sum(1).max() for _ in range(N_PERM)])
    obs = int(G.sum(1).max()); p_null = float(np.mean(mx >= obs))
    counts, counts2 = G.sum(1), G2.sum(1)
    imax, imax2 = int(np.argmax(counts)), int(np.argmax(counts2)); mean = float(counts.mean() / k)

    # one axes, coordinates in printed points: 396pt wide (5.5 in), H_PT tall
    W_PT, H_PT, Y_LO = 396.0, 108.0, 5.0                        # the box ends 3 pt under the key line
    fig = plt.figure(figsize=(W_PT / 72, (H_PT - Y_LO) / 72), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W_PT); ax.set_ylim(Y_LO, H_PT); ax.axis("off")
    cw, ch, TS = 5.6, 7.5, 4.6                                   # column pitch, row pitch, tile side (pt)
    GW = k * cw                                                  # grid width
    RW, RS = 46.0, 31.2 / 0.4                                    # rate area width; points per unit rate
    y0 = 35.0; ytop = y0 + M * ch                                # rows span y0..ytop
    ry = lambda m: ytop - (m + 1) * ch                           # bottom y of row m
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    from matplotlib.patches import FancyBboxPatch

    HOLLOW = "#D4D4D4"                                           # edge of an unsolved tile

    def tile(x, y, solved, hl=None):
        """One draw, one tile: a small square at the centre of its cell, the same shape in every
        state. Fill says whether the draw was solved (filled) or not (hollow). Hue says whose row
        it is: the members' teal, or the oracle's purple and the router's blue on the rows those
        rules pick. Filled tiles take the darker edge the bars use."""
        x0, y0t = x + (cw - TS) / 2, y + (ch - TS) / 2
        if hl is not None:
            if solved:
                return Rectangle((x0, y0t), TS, TS, fc=hl, ec=shade(hl, 0.7), lw=0.35, zorder=2.3)
            return Rectangle((x0, y0t), TS, TS, fc="white", ec=hl, lw=0.5, zorder=2.2)
        if solved:
            return Rectangle((x0, y0t), TS, TS, fc=MEMBER, ec=shade(MEMBER, 0.75), lw=0.35, zorder=2.3)
        return Rectangle((x0, y0t), TS, TS, fc="white", ec=HOLLOW, lw=0.35, zorder=2.2)

    def hbar(x0, y, w, h, r=1.0):
        """Bar silhouette for a horizontal bar: square on the baseline, rounded on the right."""
        r = max(0.0, min(r, w / 2.0, h / 2.0)); kk = 0.5523 * r; x1 = x0 + w
        verts = [(x0, y), (x1 - r, y), (x1 - r + kk, y), (x1, y + r - kk), (x1, y + r), (x1, y + h - r),
                 (x1, y + h - r + kk), (x1 - r + kk, y + h), (x1 - r, y + h), (x0, y + h), (x0, y)]
        codes = [Path.MOVETO, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO,
                 Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.CLOSEPOLY]
        return Path(verts, codes)

    HB, HK = 90.5, 100.5                                         # headline baseline, kicker baseline

    def tracked(x, y, text, color, size, spacing=0.9):
        """Letter-spaced text, drawn glyph by glyph: a kicker in small capitals."""
        ren = fig.canvas.get_renderer()
        for ch in text:
            ax.text(x, y, ch, fontsize=size, color=color, ha="left", va="baseline")
            w, _, _ = ren.get_text_width_height_descent(ch, matplotlib.font_manager.FontProperties(size=size), ismath=False)
            x += w * 72.0 / fig.dpi + spacing
        return x

    def masthead(x0, x1, letter, text, accent, kicker=""):
        """Panel header as typography alone: a two-line drop cap in the panel's colour, a tracked
        kicker in that colour on the upper line, the headline in ink on the lower."""
        tl = ax.text(x0, HB, letter, fontsize=20, fontweight="bold", color=tint(accent, 0.62), ha="left", va="baseline")
        fig.canvas.draw(); bb = tl.get_window_extent(fig.canvas.get_renderer())
        lx1 = ax.transData.inverted().transform((bb.x1, 0))[0] + 3.0
        tracked(lx1, HK, kicker, shade(accent, 0.9), FS)
        ax.text(lx1, HB, text, fontsize=FST, color=INK, ha="left", va="baseline")

    def panel(xr, xg, V, best, frames=()):
        """A bar graph beside the draw grid, row-aligned. Bars are thin, flat and strokeless, so
        they read as measurements against the axis and do not compete with the tiles. The mean
        rule (the mixture's rate) and the tau rule cross the graph under the bars; the best row's
        bar turns dark purple past the mean, its edge over the mixture."""
        cnt = V.sum(1)
        hl_of = {m: col for m, col in frames}
        for m in range(M):
            for jcell in range(k):
                ax.add_patch(tile(xg + jcell * cw, ry(m), bool(V[m, jcell]), hl=hl_of.get(m)))
        PS = 8.0                                                 # points per success along the bars
        BH = 3.0                                                 # bar height
        cx = lambda c: xr + c * PS
        ax.plot([xr, xr], [y0, ytop], color=SPINE, lw=0.6, zorder=3.2, gid="deco")            # axis above the bars
        ax.plot([xr, cx(5)], [y0, y0], color=SPINE, lw=0.55, zorder=3.2, gid="deco")
        for c in range(0, 6):                                    # a tick at every success; labels at 0, 2 and 4
            ax.plot([cx(c), cx(c)], [y0, y0 - (1.8 if c % 2 == 0 else 1.1)], color=SPINE, lw=0.5, gid="deco")
            if c in (0, 2, 4):
                ax.text(cx(c), y0 - 3.2, str(c), fontsize=FS, ha="center", va="top", color=MUT)
        xt, xm = cx(TAU * k), cx(mean * k)                       # tau and the mean at their exact rates
        for m in range(M):
            if cnt[m] == 0:
                continue
            yb = ry(m) + (ch - BH) / 2
            if m == best:
                ax.add_patch(Rectangle((xr, yb), min(cnt[m], mean * k) * PS, BH, fc=tint(ORACLE, 0.45), ec="none", zorder=2.3))
                if cnt[m] > mean * k:
                    ax.add_patch(Rectangle((xm, yb), (cnt[m] - mean * k) * PS, BH, fc=ORACLE, ec="none", zorder=2.3))
            else:
                ax.add_patch(Rectangle((xr, yb), cnt[m] * PS, BH, fc=MEMBER, ec="none", zorder=2.3))
        ax.plot([xt, xt], [y0, ytop + 1.5], color=RULE, lw=0.6, ls=(0, (2.0, 1.4)), zorder=0.95, gid="deco")
        ax.plot([xt, xt], [y0, y0 - 1.8], color=RULE, lw=0.6, gid="deco")                    # tau named on the axis
        ax.text(xt, y0 - 3.2, "τ", fontsize=FS, ha="center", va="top", color=RULE)
        ax.plot([xm, xm], [y0, ytop + 1.0], color=MIX, lw=0.9, zorder=0.95, gid="deco")
        ax.text(xm, ytop + 2.5, "mean", fontsize=FS, ha="center", va="bottom", color=shade(MIX))
        ax.text(xr + 2.5 * PS, y0 - 11.5, "successes", fontsize=FS, ha="center", va="top", color=MUT)     # one word, centred under the axis: the caption says of how many draws

    # ---------------- (a) observed: labels | rates | grid ----------------
    xra = 30.0; xga = xra + 44.0                                # grid starts right after the axis and the count label
    masthead(xra, xga + GW, "a", "Five seeds on one prompt", MEMBER, kicker="OBSERVED")
    for m in range(M):                                          # the picked rows' labels take their colors
        col = shade(ORACLE) if m == imax else (ROUTER if m == r_pick else INK)
        ax.text(xra - 4.0, ry(m) + ch / 2, f"seed {m+1}", fontsize=FS, ha="right", va="center", color=col)
    panel(xra, xga, G, imax, frames=((imax, ORACLE), (r_pick, ROUTER)))
    # ---------------- flows: the solved draws move from row to row under the re-deal ----------------
    # One ribbon per (source row, destination row) with a positive count, its width the number of
    # draws that moved that way. Ribbons leave and land at the row centres, stacked into ports of
    # height proportional to each row's count, and ordered so that they cross as little as possible.
    # A ribbon takes the hue of the row it leaves and fades into the hue of the row it lands in,
    # so the oracle's draws can be watched scattering and the new best row assembling.
    from matplotlib.colors import to_rgb
    xgb = xga + GW + 28.0
    src_x, dst_x = xga + GW - (cw - TS) / 2, xgb + (cw - TS) / 2   # snapped to the last and first tiles' edges
    F = np.zeros((M, M), int)                                    # F[i, j] = draws that moved from row i to row j
    for i_new in range(flat.size):
        i_old = perm[i_new]
        if flat[i_old] == 1:
            F[i_old // k, i_new // k] += 1
    UNIT = TS / 5.0                                              # ribbon width per draw: a five-draw port is one tile tall
    STUB = 3.0                                                   # straight run at each end before the curve
    rc = lambda m: ry(m) + ch / 2                                # row centre
    hue_src = lambda m: ORACLE if m == imax else (ROUTER if m == r_pick else MEMBER)
    hue_dst = lambda m: ORACLE if m == imax2 else MEMBER
    # no separate port bars: the ribbons leaving a row all start in its hue and stack without gaps,
    # so their ends form the port themselves, flush with the grid
    src_top = {m: rc(m) + F[m].sum() * UNIT / 2 for m in range(M)}          # ports, filled top-down
    dst_top = {m: rc(m) + F[:, m].sum() * UNIT / 2 for m in range(M)}
    for i in range(M):                                           # sources top-down, destinations top-down
        for jdst in range(M):
            n = F[i, jdst]
            if n == 0:
                continue
            w = n * UNIT
            ya1, yb1 = src_top[i], dst_top[jdst]; ya0, yb0 = ya1 - w, yb1 - w
            src_top[i] -= w; dst_top[jdst] -= w
            # a true stroke: the centreline is a cubic with horizontal tangents at both ports, and the
            # outline is that curve offset along its normal by half the width, so the ribbon is the
            # same width everywhere along its path, on the diagonal as at the ports
            ya_c, yb_c = (ya0 + ya1) / 2, (yb0 + yb1) / 2; half = (ya1 - ya0) / 2 + 0.12   # a hair of overlap: no seam between stacked ribbons
            xs0, xs1 = src_x + STUB, dst_x - STUB                 # the curve runs between the two stubs
            tt = np.linspace(0, 1, 120)
            P0, P1, P2, P3 = np.array([xs0, ya_c]), np.array([xs0 + 0.42 * (xs1 - xs0), ya_c]), \
                             np.array([xs0 + 0.58 * (xs1 - xs0), yb_c]), np.array([xs1, yb_c])
            Cc = ((1 - tt) ** 3)[:, None] * P0 + (3 * (1 - tt) ** 2 * tt)[:, None] * P1 + (3 * (1 - tt) * tt ** 2)[:, None] * P2 + (tt ** 3)[:, None] * P3
            stub_in = np.stack([np.linspace(src_x, xs0, 12, endpoint=False), np.full(12, ya_c)], axis=1)
            stub_out = np.stack([np.linspace(xs1, dst_x, 13)[1:], np.full(12, yb_c)], axis=1)
            C = np.vstack([stub_in, Cc, stub_out])                 # evenly sampled, so the normals are clean at the joins
            T = np.gradient(C, axis=0); T /= np.linalg.norm(T, axis=1)[:, None]
            Nn = np.stack([-T[:, 1], T[:, 0]], axis=1)
            top, bot = C + half * Nn, C - half * Nn
            # vector gradient: one quad per sample along the ribbon, colour interpolated from the
            # source hue to the destination hue and pre-blended to the ribbon's translucency, each
            # quad overlapping the next by one sample so no seam can open between them
            c0, c1 = np.array(to_rgb(hue_src(i))), np.array(to_rgb(hue_dst(jdst)))
            A = 0.55
            for q in range(len(C) - 1):
                tq = q / (len(C) - 2)
                col = (1 - A) * np.ones(3) + A * ((1 - tq) * c0 + tq * c1)
                q1 = min(q + 2, len(C) - 1)
                ax.add_patch(Polygon([top[q], top[q1], bot[q1], bot[q]], closed=True, fc=col, ec="none", lw=0, zorder=1.5))
    ax.text((src_x + dst_x) / 2, ytop + 2.5, "re-dealt", fontsize=FS, ha="center", va="bottom", color=MUT, style="italic")
    ax.text((src_x + dst_x) / 2, y0 - 4.5, f"{int(flat.sum())} draws", fontsize=FS, ha="center", va="top", color=MUT, style="italic")
    # ---------------- (b) re-dealt: grid | rates ----------------
    xrb = xgb + GW + 5.0
    masthead(xgb, xrb + RW, "b", "The same draws, new rows", RULE, kicker="ONE RE-DEAL")
    panel(xrb, xgb, G2, imax2, frames=((imax2, ORACLE),))
    # ---------------- (c) the best row's null, as a unit chart of tiles ----------------
    # Each tile is 50 of the 2,000 re-deals, stacked by the best row's successes. The axis is in
    # successes of 16, as the bars are labelled; the tau rule crosses it where a row clears the
    # threshold; the observed column is full purple; p is the share of re-deals at or above it,
    # drawn as a bracket over those columns.
    xc0, xc1, yc0 = xrb + RW - 2.0, W_PT - 4.0, y0
    masthead(xc0, xc1, "c", "2,000 re-deals", ORACLE, kicker="THE NULL")
    vals, cnt = np.unique(mx, return_counts=True); frac = cnt / N_PERM
    TQ, GQ = 3.6, 0.5                                            # square tile side and gap; two tiles per row (a waffle)
    PY = TQ + GQ
    TW = 2 * TQ + GQ                                             # a column's width
    n_tiles = np.round(frac * 40).astype(int)                    # one tile = 2.5% of the re-deals, 50 of 2,000
    drawn = [int(v) for v, n in zip(vals, n_tiles) if n > 0]     # counts rarer than half a tile draw nothing
    c_lo, c_hi = min(drawn), max(drawn)                          # the drawn columns span the panel edge to edge:
    pitch = (xc1 - xc0 - TW) / (c_hi - c_lo)                     # first column's left edge at xc0, last's right edge at xc1
    sx = lambda c: xc0 + TW / 2 + (c - c_lo) * pitch
    tops = {}
    for v, n in zip(vals, n_tiles):
        col_fc, col_ec = (ORACLE, shade(ORACLE, 0.7)) if v == obs else (tint(ORACLE, 0.42), tint(ORACLE, 0.75))
        for q in range(n):                                       # fill left to right, bottom up
            row, colq = divmod(q, 2)
            ax.add_patch(Rectangle((sx(v) - TW / 2 + colq * (TQ + GQ), yc0 + 1.0 + row * PY), TQ, TQ,
                                   fc=col_fc, ec=col_ec, lw=0.3, zorder=2.3))
        tops[int(v)] = yc0 + 1.0 + ((n + 1) // 2) * PY
    ax.plot([xc0, xc1], [yc0, yc0], color=SPINE, lw=0.55, zorder=1, gid="deco")
    for c in range(c_lo, c_hi + 1):
        ax.plot([sx(c), sx(c)], [yc0, yc0 - 1.8], color=SPINE, lw=0.5, gid="deco")
        ax.text(sx(c), yc0 - 3.2, str(c), fontsize=FS, ha="center", va="top",
                color=shade(ORACLE) if c == obs else INK, fontweight="bold" if c == obs else "normal")
    # successes are integers, so the threshold is a boundary between columns: a row clears tau = .2
    # from four successes up. The rule sits in the gap between the 3 and 4 columns, on top.
    xt = sx(int(np.ceil(TAU * k - 1e-9)) - 0.5)
    ax.plot([xt, xt], [yc0, ytop + 5.0], color=RULE, lw=0.8, ls=(0, (2.2, 1.4)), zorder=0.95, gid="deco")   # behind the axis, up to its label
    ax.text(xt, ytop + 6.0, "τ = .2", fontsize=FS, ha="center", va="bottom", color=RULE)
    xcm = (xc0 + xc1) / 2                                        # the axis centre, under the observed column
    ax.text(xcm, yc0 - 11.5, "successes", fontsize=FS, ha="center", va="top", color=MUT)      # the same word as under (a) and (b)
    above = [v for v in drawn if v >= obs]                       # the drawn columns at or above the observed best row
    yb = max(tops[v] for v in above) + 2.0
    xl, xr = sx(min(above)) - TW / 2, min(sx(max(above)) + TW / 2, xc1)   # the wash ends at the last column
    # the share of re-deals at or above the observed count, as a wash under the bracket: that share is p
    NS = 120                                                     # the wash fades upward, as Figure 1's bands do:
    cp = np.array(to_rgb(ORACLE)); hq = (yb - yc0) / NS         # opaque slices pre-blended toward white and
    for q in range(NS):                                          # overlapping, so the fade is smooth and seamless
        a = 0.16 * (1 - q / NS) + 0.09                              # deeper at the baseline, still purple at the top
        ax.add_patch(Rectangle((xl - 1.0, yc0 + q * hq), xr - xl + 1.0, hq + 0.15, fc=(1 - a) * np.ones(3) + a * cp,
                               ec="none", lw=0, zorder=0.9))      # under the axis line
    # p labels the region it measures, inside the wash; the observed count hangs from a leader
    # rising out of its own column
    ax.text(sx(obs) + TW / 2 + 2.5, yb - 1.8, r"$p$ = " + f"{p_null:.2f}".replace("0.", "."), fontsize=FS, ha="left", va="top", color=shade(ORACLE))
    # the observed count is marked by its column (full purple) and its bold purple tick; no label repeats it
    # the re-deal drawn in (b) is one of these; the caption says which column, no marker
    # ---------------- key: a footer line set from the figure's own marks ----------------
    # Two groups, each opened by a tracked kicker in the mastheads' manner. DRAW: what one tile
    # says, filled for solved and hollow for not. ROW: what a row's hue says, shown as a strip
    # of three tiles in that hue with the role word in the colour its row label carries. Every
    # advance is measured from the rendered glyphs, so the gaps are equal by construction.
    ren = fig.canvas.get_renderer(); px = 72.0 / fig.dpi
    def width(text, size=FS):
        w, _, _ = ren.get_text_width_height_descent(text, matplotlib.font_manager.FontProperties(size=size), ismath=False)
        return w * px
    _, lp_h, lp_d = ren.get_text_width_height_descent("lp", matplotlib.font_manager.FontProperties(size=FS), ismath=False)
    KTOP = y0 - 11.5 - 9.0                                       # one line pitch under the axis titles
    KB = KTOP - (lp_h - lp_d) * px                               # its baseline
    yt = KB + 1.6 - TS / 2                                       # tiles centred on the x-height
    GAP, GROUP, PAD, KS = 7.0, 13.0, 2.5, FS - 1.0
    def ktile(x, solved, hl=None):
        ax.add_patch(tile(x - (cw - TS) / 2, yt - (ch - TS) / 2, solved, hl))   # the grid's own tile, left edge at x
        return x + TS
    def strip(x, hl, states):                                    # tiles at the grid's pitch, in the row's hue: one per
        for i, s in enumerate(states):                           # state that row has in the figure
            ktile(x + i * cw, s, hl)
        return x + (len(states) - 1) * cw + TS
    def kword(x, text, color=INK):
        ax.text(x, KB, text, fontsize=FS, ha="left", va="baseline", color=color)
        return x + width(text)
    n_t, n_p = len(ax.texts), len(ax.patches)                    # laid out from x = 0, then centred in the figure
    x = tracked(0.0, KB, "DRAW", MUT, KS, spacing=0.8) + 3.0
    x = kword(ktile(x, True) + PAD, "solved") + GAP
    x = kword(ktile(x, False) + PAD, "unsolved") + GROUP
    x = tracked(x, KB, "ROW", MUT, KS, spacing=0.8) + 3.0
    x = kword(kword(strip(x, ORACLE, (False,)) + PAD, "oracle", shade(ORACLE)), ": the best row") + GAP
    x = kword(kword(strip(x, ROUTER, (False,)) + PAD, "router", ROUTER), ": its pick")
    dx = (W_PT - x) / 2
    for t in ax.texts[n_t:]:
        t.set_x(t.get_position()[0] + dx)
    for r in ax.patches[n_p:]:
        r.set_x(r.get_x() + dx)
    fig._facts = dict(null_counts={int(v): int(c) for v, c in zip(vals, cnt)}, prompt=str(pid), successes=int(flat.sum()), obs_best=obs, p_null=p_null, mean=mean,
                      router_row=r_pick + 1, best_row=imax + 1, redeal_best=int(counts2.max()), redeal_best_row=imax2 + 1,
                      counts=[int(c) for c in counts], redeal_counts=[int(c) for c in counts2])
    return fig


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--png", default=None); ap.add_argument("--audit-only", action="store_true")
    a = ap.parse_args()
    fig = build()
    n = audit(fig)
    print("facts:", json.dumps(fig._facts))
    print(f"layout collisions: {n}")
    if a.audit_only:
        return 1 if n else 0
    OUTPDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPDF, format="pdf")
    if a.png:
        fig.savefig(a.png, dpi=220)
    print(f"wrote {OUTPDF}")
    return 1 if n else 0


if __name__ == "__main__":
    sys.exit(main())
