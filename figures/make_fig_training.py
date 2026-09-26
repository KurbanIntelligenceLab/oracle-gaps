#!/usr/bin/env python3
"""Figure 2: cross-dataset pools under light and stronger training."""
from __future__ import annotations
import pathlib as _pl, sys as _sys
_REPO_ROOT = next(p for p in _pl.Path(__file__).resolve().parents if (p / "lib").is_dir())
_sys.path.insert(0, str(_REPO_ROOT / "lib"))  # shared modules

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "csm-figure-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "csm-figure-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
ROOT = _REPO_ROOT
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.facecolor": "white",
})


def read_results(path: Path):
    return json.loads(path.read_text())


def check_pool(pool, name):
    if (pool.get("M"), pool.get("n"), pool.get("k")) != (2, 424, 16):
        raise ValueError(f"{name} must have M=2, n=424, k=16")
    if pool.get("router_fit") != "validation":
        raise ValueError(f"{name} lacks validation-fitted routers")
    if pool.get("members_dropped"):
        raise ValueError(f"{name} dropped members")
    for member in pool["members"]:
        seed = re.search(r"(\d+)L?\s*$", member)  # long-recipe members are named seed1L, seed2L
        if seed is None or int(seed.group(1)) != 1:
            raise ValueError(f"Expected seed 1, found {member!r}")


def load_long(path: Path | None):
    if path is None:
        return None
    result = read_results(path)
    if result.get("dropped"):
        raise ValueError("The long run still has incomplete adapters")
    all_pool = result["pools"]["all_long"]
    if (all_pool.get("M"), all_pool.get("n"), all_pool.get("k")) != (4, 424, 16):
        raise ValueError("The long run must include all four intended adapters")
    if all_pool.get("router_fit") != "validation":
        raise ValueError("Long-run validation coverage is incomplete")
    members = all_pool["members"]
    for dataset in ("geometry3k", "mathvista"):
        if sum(m.startswith(dataset + " ") for m in members) != 2:
            raise ValueError(f"Expected two long adapters for {dataset}")
    pool = result["pools"]["two_long"]
    check_pool(pool, "Long pool")
    pool["_all_long"] = all_pool
    return pool


def at_threshold(pool, tau):
    key = next((k for k in pool["nulls"] if abs(float(k) - tau) < 1e-9), None)
    if key is None:
        raise ValueError(f"Threshold {tau} missing from results")
    nulls = pool["nulls"][key]["L"]
    routing = pool["routing"][key]
    for family in ("domain", "embedding nearest-neighbor"):
        if family not in routing:
            raise ValueError(f"Missing router {family}")
    assert math.isclose(nulls["redeal"]["obs"], routing["L"], abs_tol=1e-10)
    assert math.isclose(nulls["margin"]["obs"], routing["L"], abs_tol=1e-10)
    return nulls, routing


def illustrative_long():
    """Every value here is invented solely to demonstrate the layout."""
    n = 424
    pool = {"_dummy": True, "n": n, "M": 2, "k": 16, "nulls": {}, "routing": {}}
    for key, gap_count, plain_count, margin_count, identity_count, nearest_count in [
        ("0.05", 54, 27.5, 24.5, 28, 34), ("0.1", 51, 26.8, 23.6, 26, 32),
        ("0.2", 42, 22, 20, 20, 26), ("0.3", 32, 20, 18, 13, 18),
    ]:
        gap = gap_count / n
        nulls = {}
        for family, mean_count in [("redeal", plain_count), ("margin", margin_count)]:
            nulls[family] = {"obs": gap, "null_mean": mean_count / n,
                             "null_lo": (mean_count - 6) / n, "null_hi": (mean_count + 6) / n,
                             "p": 0.0005, "excess": (gap_count - mean_count) / n}
        row = {"L": gap, "D": 100 / n, "L_over_D": gap_count / 100}
        for family, gain_count in [("domain", identity_count), ("embedding nearest-neighbor", nearest_count)]:
            row[family] = {"gain": gain_count / n, "lo": (gain_count - 15) / n,
                           "hi": (gain_count + 13) / n, "eD": (gap_count - gain_count) / 100}
        pool["nulls"][key] = {"L": nulls}; pool["routing"][key] = row
    import copy
    all_pool = copy.deepcopy(pool)
    all_pool["M"] = 4
    row = all_pool["routing"]["0.1"]
    row.update({"L": 68 / n, "D": 140 / n, "L_over_D": 68 / 140})
    for family, gain_count in [("domain", 36), ("embedding nearest-neighbor", 44)]:
        row[family] = {"gain": gain_count / n, "lo": (gain_count - 14) / n,
                       "hi": (gain_count + 12) / n, "eD": (68 - gain_count) / 140}
    for cell in all_pool["nulls"]["0.1"]["L"].values():
        cell.update({"obs": 68 / n, "null_mean": 42 / n,
                     "null_lo": 35 / n, "null_hi": 49 / n, "excess": 26 / n})
    pool["_all_long"] = all_pool
    return pool


def render(short, long, tau, outdir):
    """Three compact panels inspired by the removed results figure."""
    from matplotlib import font_manager
    from matplotlib.colors import to_rgb, to_hex
    from matplotlib.lines import Line2D
    from matplotlib.legend_handler import HandlerBase
    import numpy as np

    oracle, router = "#7030A0", "#3178C6"
    ink, muted, spine = "#1A1A1A", "#6B6B6B", "#B0B0B0"
    def tint(color, amount):
        return to_hex(tuple(1 - amount + amount * v for v in to_rgb(color)))

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
        "mathtext.fontset": "stix", "font.size": 7.5, "text.color": ink,
        "axes.edgecolor": spine, "axes.linewidth": 0.55,
        "axes.labelcolor": ink, "xtick.color": spine, "ytick.color": spine,
        "xtick.labelcolor": muted, "ytick.labelcolor": muted,
        "pdf.fonttype": 42,
    })
    width, height = 396, 130
    plot_height = 46
    plot_top = 45 + plot_height
    fig = plt.figure(figsize=(5.5, height / 72), dpi=400, facecolor="white")
    positions = [(23, 45, 99, plot_height), (157, 45, 99, plot_height),
                 (293, 45, 99, plot_height)]
    axes = [fig.add_axes([x / width, y / height, w / width, h / height]) for x, y, w, h in positions]
    dummy = long is not None and long.get("_dummy", False)
    keys = ["0.05", "0.1", "0.2", "0.3"]
    thresholds = np.array([float(k) for k in keys])
    pools = [short, long]
    measured = [p for p in pools if p is not None]
    gap_max = max(max(p["nulls"][k]["L"][family][field] * 100
                      for k in keys for family in ("redeal", "margin")
                      for field in ("obs", "null_hi")) for p in measured)
    ymax = max(8, math.ceil((gap_max + 1) / 4) * 4)

    for idx, (ax, pool) in enumerate(zip(axes[:2], pools)):
        ax.set_xlim(0.035, 0.315)
        ax.set_ylim(0, ymax)
        ax.set_xticks(thresholds, [".05", ".10", ".20", ".30"])
        ax.set_yticks([0, ymax / 2, ymax])
        ax.tick_params(length=2, width=0.55, pad=1.6, labelsize=7.5)
        for edge in ("top", "right", "left"):
            ax.spines[edge].set_visible(False)
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.45)
        ax.set_axisbelow(True)
        ax.set_xlabel(r"reliability threshold $\tau$", fontsize=7.5, labelpad=2.3)
        ax.axvline(tau, color="#C9C9C9", linestyle=(0, (1, 2)), linewidth=0.6, zorder=1)
        if pool is None:
            ax.text(0.5, 0.5, "pending", transform=ax.transAxes, ha="center", va="center", fontsize=8, color=muted)
            continue
        data = [pool["nulls"][k]["L"] for k in keys]
        for family, amount, style in [("redeal", 0.19, (0, (3, 1.8))), ("margin", 0.35, (0, (1, 1.2)))]:
            lo = np.array([r[family]["null_lo"] for r in data]) * 100
            hi = np.array([r[family]["null_hi"] for r in data]) * 100
            mean = np.array([r[family]["null_mean"] for r in data]) * 100
            ax.fill_between(thresholds, lo, hi, facecolor=tint(oracle, amount), linewidth=0, zorder=1.5)
            ax.plot(thresholds, mean, color=oracle, linestyle=style, linewidth=0.85, zorder=3)
        observed = np.array([r["redeal"]["obs"] for r in data]) * 100
        ax.plot(thresholds, observed, color=oracle, linewidth=1.4, marker="o", markersize=2.5,
                markeredgecolor="white", markeredgewidth=0.45, zorder=5)
        j = int(np.argmin(abs(thresholds - tau)))
        ax.plot(tau, observed[j], "D", markersize=4.5, color=oracle,
                markeredgecolor="white", markeredgewidth=0.8, zorder=6)
        p = data[j]["margin"]["p"]
        ptext = r"$p_{\mathrm{m}}<.001$" if p < 0.001 else rf"$p_{{\mathrm{{m}}}}={p:.2f}$"
        ax.text(0.98, 0.98, ptext, transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color=oracle)

    # Panel c: the stronger two-member pool across every reported threshold. Bars are the plug-in
    # oracle gap (the ceiling any router can reach); points are router gains over the best member.
    ax = axes[2]
    pool_c = long
    if pool_c is None:
        ax.set_xlim(0, 1); ax.set_ylim(-4, 8); ax.set_yticks([0, 4, 8]); ax.set_xticks([])
        for edge in ("top", "right", "left"):
            ax.spines[edge].set_visible(False)
        ax.text(0.5, 0.5, "pending", transform=ax.transAxes, ha="center", va="center", fontsize=8, color=muted)
    else:
        keys_c = sorted(pool_c["routing"], key=float)
        xs = np.arange(len(keys_c))
        fams = [("domain", "o", -0.17, "white"), ("embedding nearest-neighbor", "s", 0.17, router)]
        rows_c = [pool_c["routing"][k] for k in keys_c]
        for k, row in zip(keys_c, rows_c):
            for fam, _, _, _ in fams:
                if fam not in row:
                    raise ValueError(f"Missing router {fam} at threshold {k}")
        gaps = np.array([r["L"] * 100 for r in rows_c])
        hi_all = max(gaps.max(), max(r[f]["hi"] * 100 for r in rows_c for f, _, _, _ in fams))
        lo_all = min(r[f]["lo"] * 100 for r in rows_c for f, _, _, _ in fams)
        rmax = max(10, math.ceil((hi_all + 3) / 5) * 5)   # headroom for the band labels
        rmin = min(-4, math.floor((lo_all - 0.5) / 4) * 4)
        ax.set_xlim(-0.6, len(keys_c) - 0.4)
        ax.set_ylim(rmin, rmax)
        ax.set_xticks(xs, [f"{float(k):.2f}"[1:] for k in keys_c])
        ax.set_yticks([0, rmax / 2, rmax])
        ax.tick_params(length=2, width=0.55, pad=1.6, labelsize=7.5)
        for edge in ("top", "right", "left"):
            ax.spines[edge].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.45)
        ax.set_xlabel(r"reliability threshold $\tau$", fontsize=7.5, labelpad=2.3)
        ax.axhspan(rmin, 0, facecolor="#F3F3F3", zorder=0)
        ax.axhline(0, color="#8A8A8A", linewidth=0.7, zorder=2)
        j_tau = int(np.argmin([abs(float(k) - tau) for k in keys_c]))
        ax.axvline(j_tau, color="#C9C9C9", linestyle=(0, (1, 2)), linewidth=0.6, zorder=1)
        # Each dataset's reliability band, shaded above zero so it never competes with the loss region.
        y0 = (0 - rmin) / (rmax - rmin)
        bands = [("Geometry3K", [i for i, k in enumerate(keys_c) if 0.05 <= float(k) <= 0.30], "#F4EDE2"),
                 ("MathVista", [i for i, k in enumerate(keys_c) if 0.50 <= float(k) <= 0.90], tint(router, 0.13))]
        for label, idx, color in bands:
            if not idx:
                continue
            x0 = min(idx) - 0.5 if min(idx) > 0 else -0.6
            x1 = max(idx) + 0.5 if max(idx) < len(keys_c) - 1 else len(keys_c) - 0.4
            ax.axvspan(x0, x1, ymin=y0, ymax=1, facecolor=color, edgecolor="none", zorder=0.5)
            ax.text((x0 + x1) / 2, rmax - 0.3, label, ha="center", va="top", fontsize=6.2, color=muted)
        ax.bar(xs, gaps, width=0.7, color=tint(oracle, 0.16), edgecolor="none", zorder=1)
        for x, gap in zip(xs, gaps):
            ax.plot([x - 0.35, x + 0.35], [gap, gap], color=oracle, linewidth=1.1, zorder=3)
        for fam, marker, offset, fill in fams:
            for x, row in zip(xs, rows_c):
                cell = row[fam]
                gain, lo, hi = [100 * cell[k] for k in ("gain", "lo", "hi")]
                ax.errorbar(x + offset, gain, yerr=[[gain - lo], [hi - gain]], fmt=marker,
                            color=router, markerfacecolor=fill, markeredgewidth=0.7,
                            markersize=2.9, linewidth=0.8, capsize=1.1, capthick=0.6, zorder=5)

    # Mastheads follow the compact results figure, with Figure 1's semantic palette.
    titles = ["Light training", "Stronger training", "Router gain"]
    subtitles = ["oracle gap · 2 members", "oracle gap · dummy" if dummy else "oracle gap · pending" if long is None else "oracle gap · 2 members",
                 "router gain · dummy" if dummy else "router gain · pending" if long is None else "stronger training · 2 members"]
    for i, (left, title, subtitle) in enumerate(zip([2, 136, 272], titles, subtitles)):
        color = oracle if i < 2 else router
        fig.add_artist(Line2D([left / width, (left + 122) / width], [(height - 8) / height] * 2,
                              transform=fig.transFigure, color="#E6E6E6", linewidth=0.55))
        fig.add_artist(Line2D([left / width, (left + 11) / width], [(height - 8) / height] * 2,
                              transform=fig.transFigure, color=color, linewidth=1.8))
        fig.text(left / width, (height - 19) / height, "abc"[i], fontsize=9, color=color, fontweight="bold", va="center")
        fig.text((left + 9) / width, (height - 19) / height, title, fontsize=8.5, va="center")
        fig.text(left / width, (height - 31) / height, subtitle, fontsize=7.5, va="center", color=muted)

    fig.text(2 / width, (plot_top - 5) / height, "pp", fontsize=7.5, color=muted, va="center")
    fig.text(136 / width, (plot_top - 5) / height, "pp", fontsize=7.5, color=muted, va="center")
    fig.text(272 / width, (plot_top - 5) / height, "pp", fontsize=7.5, color=muted, va="center")

    # Reproduce both parts of the highlighted threshold in the legend.
    class RoutingThresholdHandler(HandlerBase):
        def create_artists(self, legend, orig_handle, xdescent, ydescent,
                           width, height, fontsize, trans):
            cx, cy = width / 2 - xdescent, height / 2 - ydescent
            guide = Line2D([cx, cx], [cy - 4.4, cy + 4.4],
                           color="#C9C9C9", linestyle=(0, (1, 2)),
                           linewidth=0.6, transform=trans)
            curve = Line2D([-xdescent, width - xdescent], [cy, cy],
                           color=oracle, linewidth=1.4, transform=trans)
            dot = Line2D([cx], [cy], marker="D", linestyle="none",
                         markersize=4.5, color=oracle, markeredgecolor="white",
                         markeredgewidth=0.8, transform=trans)
            return [guide, curve, dot]

    class NullBandHandler(HandlerBase):
        def __init__(self, amount, style):
            super().__init__()
            self.amount, self.style = amount, style

        def create_artists(self, legend, orig_handle, xdescent, ydescent,
                           width, height, fontsize, trans):
            band = Rectangle((-xdescent, -ydescent), width, height,
                             facecolor=tint(oracle, self.amount), edgecolor="none",
                             transform=trans)
            mean = Line2D([-xdescent, width - xdescent],
                          [height / 2 - ydescent] * 2, color=oracle,
                          linestyle=self.style, linewidth=0.85, transform=trans)
            return [band, mean]

    class LossRegionHandler(HandlerBase):
        def create_artists(self, legend, orig_handle, xdescent, ydescent,
                           width, height, fontsize, trans):
            box = Rectangle((-xdescent, -ydescent), width, height,
                            facecolor="#F3F3F3", edgecolor="none", transform=trans)
            zero = Line2D([-xdescent, width - xdescent], [height - ydescent] * 2,
                          color="#8A8A8A", linewidth=0.7, transform=trans)
            return [box, zero]

    class OracleBarHandler(HandlerBase):
        def create_artists(self, legend, orig_handle, xdescent, ydescent,
                           width, height, fontsize, trans):
            left, right = width * 0.15 - xdescent, width * 0.85 - xdescent
            bar = Rectangle((left, -ydescent), right - left, height,
                            facecolor=tint(oracle, 0.16), edgecolor="none",
                            transform=trans)
            top = Line2D([left, right], [height - ydescent] * 2,
                         color=oracle, linewidth=1.1, transform=trans)
            return [bar, top]

    threshold_key, redeal_key, margin_key, bar_key, loss_key = [object() for _ in range(5)]
    # Give the gap curves and the routing comparison their own aligned keys.
    # Shaded keys reproduce the plotted ranges, and the bar key reproduces
    # the oracle bar, avoiding an unexplained reuse of the line-only symbol.
    gap_legend = fig.legend(
        handles=[Line2D([0], [0], color=oracle, linewidth=1.4), threshold_key,
                 redeal_key, margin_key],
        labels=["Oracle gap", rf"Routing threshold ($\tau={tau:.2f}$)",
                "Re-deal null (90%)", "Margin null (90%)"],
        handler_map={threshold_key: RoutingThresholdHandler(),
                     redeal_key: NullBandHandler(0.19, (0, (3, 1.8))),
                     margin_key: NullBandHandler(0.35, (0, (1, 1.2)))},
        loc="lower center", bbox_to_anchor=(132 / width, 0.003), ncol=2,
        frameon=False, fontsize=7.5, handlelength=2.3, handletextpad=0.45,
        columnspacing=1.7, labelspacing=0.55, borderaxespad=0)
    router_legend = fig.legend(
        handles=[bar_key,
                 Line2D([0], [0], color=router, marker="o", markerfacecolor="white",
                        markersize=3.1, linewidth=0),
                 loss_key,
                 Line2D([0], [0], color=router, marker="s", markersize=3.1, linewidth=0)],
        labels=["Oracle gap", "Identity", "Router loses", "Embedding NN"],
        handler_map={bar_key: OracleBarHandler(), loss_key: LossRegionHandler()},
        loc="lower center", bbox_to_anchor=(334 / width, 0.003), ncol=2,
        frameon=False, fontsize=7.5, handlelength=1.35, handletextpad=0.4,
        columnspacing=1.0, labelspacing=0.55, borderaxespad=0)
    fig.add_artist(Line2D([267 / width] * 2, [2 / height, 23 / height],
                         transform=fig.transFigure, color="#E6E6E6", linewidth=0.55))

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for legend in (gap_legend, router_legend):
        bounds = legend.get_window_extent(renderer)
        if not (bounds.x0 >= 0 and bounds.x1 <= fig.bbox.width and
                bounds.y0 >= 0 and bounds.y1 <= fig.bbox.height):
            raise ValueError("Legend extends outside the figure")
    if gap_legend.get_window_extent(renderer).overlaps(router_legend.get_window_extent(renderer)):
        raise ValueError("The gap and routing legends overlap")
    outdir.mkdir(parents=True, exist_ok=True)
    stem = outdir / "fig_training"
    for ext in ("pdf", "svg", "png"):
        fig.savefig(stem.with_suffix("." + ext), dpi=400, facecolor="white",
                    metadata={"Creator": "Matplotlib; short real, stronger explicitly dummy" if dummy else "Matplotlib; measured CSM results"}
                    if ext in ("pdf", "svg") else None)
    plt.close(fig)
    return {"threshold": tau, "M": 2, "n": 424, "k": 16,
            "units": "percentage points", "long_results_pending": long is None or dummy,
            "stronger_row_is_dummy": dummy, "layout": "three wider panels, separate alternative",
            "plot_aspect_ratio": 99 / plot_height,
            "real_panels": ["a", "c: short / 2"] + ([] if (dummy or long is None) else ["b", "c: strong / 2", "c: strong / 4"]),
            "dummy_panels": ["b", "c: strong / 2", "c: strong / 4"] if dummy else [],
            "width_inches": 5.5, "height_inches": height / 72, "smallest_text_points": 7.5,
            "font": font_manager.findfont(font_manager.FontProperties(family="Times New Roman")),
            "highlighted_diamond": "Routing threshold, tau=" + str(tau),
            "purple_bars": "Plug-in oracle coverage minus best-member coverage, in percentage points",
            "band_shades": {"lighter": "Re-deal null 90% reference range",
                            "darker": "Margin-preserving null 90% reference range"},
            "null_ranges": "5th–95th percentile of null statistics",
            "router_intervals": "95% prompt-bootstrap percentile intervals"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--short-results", type=Path, default=ROOT / "analysis" / "specialist_pool.json")
    long_args = parser.add_mutually_exclusive_group()
    long_args.add_argument("--long-results", type=Path, help="Completed specialist_pool_long.json; omit to leave blank")
    long_args.add_argument("--dummy-long", action="store_true", help="Show an explicitly labeled, invented positive example")
    parser.add_argument("--tau", type=float, default=0.10)
    parser.add_argument("--outdir", type=Path, default=HERE / "out")
    args = parser.parse_args()
    short = read_results(args.short_results)["pools"]["two_specialists"]
    check_pool(short, "Short pool")
    long = illustrative_long() if args.dummy_long else load_long(args.long_results)
    report = render(short, long, args.tau, args.outdir)
    report["short_source_sha256"] = hashlib.sha256(args.short_results.read_bytes()).hexdigest()
    if args.long_results:
        report["long_source_sha256"] = hashlib.sha256(args.long_results.read_bytes()).hexdigest()
    if args.dummy_long:
        report["dummy_values"] = long
    (args.outdir / "figure_provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
