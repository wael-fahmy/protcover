"""Main paper figures, read only from results/<tag>/*.csv.

Encoding used in every figure: color and marker = method (variant-level split,
family-level HCP, group-weighted, ProtCOVER-W, ProtCOVER); filled marker and solid
line = class-conditional (Mondrian) version; hollow marker and dashed line =
marginal version. Palette: reference categorical slots; the four family-level
methods (blue, aqua, yellow, green) and the Fig. 1 pair (violet, yellow) pass the
dataviz validator on all pairs in light mode. Aqua and yellow are below 3:1
contrast, so every series also has its own marker shape and a legend entry.

Usage: python scripts/make_figures.py --tag main
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd

from protcover.config import ROOT, results_dir

PRED_ORDER = ["ESM1b", "EVE", "GEMME", "TranceptEVE_L", "PoET"]
PRED_LABEL = {"ESM1b": "ESM-1b", "EVE": "EVE", "GEMME": "GEMME", "TranceptEVE_L": "TranceptEVE", "PoET": "PoET"}
COLOR = {"B1_split": "#6250d6", "B3_pool": "#eda100", "B4_gwcp": "#1baf7a", "P_w": "#008300", "C_ratio": "#2a78d6"}
MARKER = {"B1_split": "v", "B3_pool": "s", "B4_gwcp": "D", "P_w": "^", "C_ratio": "o"}
LABEL = {("B1_split", False): "Split (B1)", ("B1_split", True): "Mondrian split (B2)",
         ("B3_pool", False): "HCP (B3)", ("B3_pool", True): "HCP",
         ("B4_gwcp", False): "GWCP (B4)", ("B4_gwcp", True): "GWCP",
         ("P_w", False): "ProtCOVER-W", ("P_w", True): "ProtCOVER-W",
         ("C_ratio", False): "ProtCOVER", ("C_ratio", True): "ProtCOVER"}
FAMILY_METHODS = ["B3_pool", "B4_gwcp", "P_w", "C_ratio"]
# Fig. 1 shows marginal and class-conditional versions side by side, so its labels name both.
LABEL_S1 = {("B1_split", False): "Split (B1)", ("B1_split", True): "Mondrian split (B2)",
            ("B3_pool", False): "HCP (B3)", ("B3_pool", True): "HCP, class-conditional"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
RISK = {"ess": ("#2a78d6", "o", "ESS gate"), "support": ("#eb6834", "s", "Support only (max identity)"),
        "random": ("#1baf7a", "D", "Random"), "mass": ("#eda100", "v", "Test-point mass")}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.2, "pdf.fonttype": 42,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.4, "axes.axisbelow": True,
})


def style(method, cc):
    return dict(color=COLOR[method], marker=MARKER[method], linestyle="-" if cc else "--",
                markerfacecolor=COLOR[method] if cc else "white", markeredgecolor=COLOR[method],
                markersize=4, linewidth=1.2 if cc else 1.0)


def target_line(ax, y):
    ax.axhline(y, color=MUTED, linewidth=0.7, linestyle=":", zorder=1)


def sel(df, **kw):
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        if k == "beta" and v is None:
            m &= df["beta"].isna().to_numpy()
        elif isinstance(v, float):
            m &= np.isclose(df[k].to_numpy(dtype=float), v)
        else:
            m &= (df[k] == v).to_numpy()
    return df[m]


def fig_s1(summ, disp, out, alpha=0.1):
    """Fig. 1: S1 (exchangeable families). (a) family-weighted pathogenic coverage;
    (b) fraction of families below 1 - alpha - 0.05 against the binomial-noise expectation."""
    methods = [("B1_split", False), ("B1_split", True), ("B3_pool", False), ("B3_pool", True)]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.2))
    x = np.arange(len(PRED_ORDER))
    offs = np.linspace(-0.27, 0.27, len(methods))
    ax = axes[0]
    for (m, cc), o in zip(methods, offs):
        d = sel(summ, scenario="S1", alpha=alpha, method=m, cc=cc, metric="cov_path_family").set_index("predictor").loc[PRED_ORDER]
        st = style(m, cc)
        ax.errorbar(x + o, d["mean"], yerr=[d["mean"] - d["ci_lo"], d["ci_hi"] - d["mean"]], fmt=st["marker"],
                    color=st["color"], mfc=st["markerfacecolor"], mec=st["markeredgecolor"], ms=4, elinewidth=0.8,
                    capsize=0, label=LABEL_S1[(m, cc)])
    target_line(ax, 1 - alpha)
    ax.set_xticks(x, [PRED_LABEL[p] for p in PRED_ORDER])
    ax.tick_params(axis="x", labelsize=6.5)
    ax.set_ylabel("Pathogenic coverage\n(family-weighted)")
    ax.set_title("(a) Pathogenic coverage, S1", loc="left")
    ax = axes[1]
    for (m, cc), o in zip(methods, offs):
        d = sel(disp, scenario="S1", alpha=alpha, method=m, cc=cc).set_index("predictor").loc[PRED_ORDER]
        st = style(m, cc)
        ax.errorbar(x + o, d["frac_fam_below_mean"], yerr=d["frac_fam_below_std"], fmt=st["marker"], color=st["color"],
                    mfc=st["markerfacecolor"], mec=st["markeredgecolor"], ms=4, elinewidth=0.8, capsize=0)
    null = sel(disp, scenario="S1", alpha=alpha, method="B1_split", cc=False).set_index("predictor").loc[PRED_ORDER]
    ax.scatter(x, null["null_frac_fam_below_mean"], marker="_", s=120, color=MUTED, zorder=3, label="Binomial noise only")
    ax.set_xticks(x, [PRED_LABEL[p] for p in PRED_ORDER])
    ax.tick_params(axis="x", labelsize=6.5)
    ax.set_ylabel(f"Families below {1 - alpha - 0.05:.2f}")
    ax.set_title("(b) Under-covered families (≥10 variants), S1", loc="left")
    h1, l1 = axes[0].get_legend_handles_labels()
    h2, l2 = axes[1].get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, out, "fig1_s1_family_coverage")


def fig_shift(summ, out, alpha=0.1):
    """Fig. 2: class-conditional coverage of both classes against shift strength beta in S2 (beta = 0 is S1)."""
    # Mondrian split (B2) is far below the family-level methods (Fig. 1, Table 6); leaving it out
    # lets the axis resolve the differences among the family-level methods.
    rows = [("cov_path_family", True, "Pathogenic $\\mathrm{Cov}_1$"), ("cov_benign_family", True, "Benign $\\mathrm{Cov}_0$")]
    dodge = dict(zip(FAMILY_METHODS, (-0.075, -0.025, 0.025, 0.075)))  # keep intervals apart
    fig, axes = plt.subplots(2, 5, figsize=(6.5, 3.0), sharex=True, sharey="row")
    for r, (metric, cc, ylab) in enumerate(rows):
        for c, pred in enumerate(PRED_ORDER):
            ax = axes[r, c]
            for m in FAMILY_METHODS:
                d = summ[(summ["predictor"] == pred) & np.isclose(summ["alpha"], alpha) & (summ["method"] == m)
                         & (summ["cc"] == cc) & (summ["metric"] == metric) & summ["scenario"].isin(["S1", "S2"])].sort_values("beta")
                st = style(m, cc)
                ax.errorbar(d["beta"] + dodge[m], d["mean"], yerr=[d["mean"] - d["ci_lo"], d["ci_hi"] - d["mean"]],
                            color=st["color"], marker=st["marker"], linestyle=st["linestyle"], mfc=st["markerfacecolor"],
                            mec=st["markeredgecolor"], ms=3.5, lw=st["linewidth"], elinewidth=0.7, capsize=0,
                            label=LABEL[(m, cc)])
            target_line(ax, 1 - alpha)
            if r == 0:
                ax.set_title(PRED_LABEL[pred])
            if r == 1:
                ax.set_xlabel(r"Shift strength $\beta$")
            if c == 0:
                ax.set_ylabel(ylab)
            ax.set_xticks([0, 0.5, 1, 2])
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, out, "fig2_shift_sweep")


def fig_tradeoff(summ, out):
    """Fig. 3: pathogenic coverage against mean set size across alpha in {0.2, 0.1, 0.05} on the two
    shifts that the descriptors were not chosen for; a tick on the 0.90 line marks each method's set
    size at matched pathogenic coverage 0.90 (linear interpolation)."""
    settings = [("S3", None, "S3 (orphans)"), ("S4", None, "S4 (regions)")]
    fig, axes = plt.subplots(2, 5, figsize=(6.5, 3.0))
    for r, (scen, beta, rlab) in enumerate(settings):
        for c, pred in enumerate(PRED_ORDER):
            ax = axes[r, c]
            for m in FAMILY_METHODS:
                cov = sel(summ, scenario=scen, beta=beta, predictor=pred, method=m, cc=True, metric="cov_path_family").sort_values("alpha")
                size = sel(summ, scenario=scen, beta=beta, predictor=pred, method=m, cc=True, metric="set_size").sort_values("alpha")
                st = style(m, True)
                x, y = size["mean"].to_numpy(), cov["mean"].to_numpy()
                # GWCP and the ProtCOVER variants often coincide: larger GWCP markers stay visible underneath.
                ax.plot(x, y, color=st["color"], marker=st["marker"], ms=5.5 if m == "B4_gwcp" else 3.5, lw=1.0,
                        label=LABEL[(m, True)])
                o = np.argsort(y)
                if y[o][0] <= 0.9 <= y[o][-1]:
                    ax.plot([np.interp(0.9, y[o], x[o])] * 2, [0.885, 0.915], color=st["color"], lw=1.6,
                            solid_capstyle="butt", zorder=4)
            for a in (0.05, 0.1, 0.2):
                target_line(ax, 1 - a)
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            if r == 0:
                ax.set_title(PRED_LABEL[pred])
            if c == 0:
                ax.set_ylabel(f"{rlab}\npathogenic cov.")
            if r == 1:
                ax.set_xlabel("Mean set size")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, out, "fig3_coverage_size")


def fig_risk(rs, out, alpha=0.1):
    """Fig. 4: risk-coverage of abstention rules applied to ProtCOVER-W class-conditional sets."""
    settings = [("S2", 1.0, r"S2, $\beta$=1"), ("S3", None, "S3")]
    grid_cols = sorted([c for c in rs.columns if c.startswith("r") and c.endswith("_mean") and c[1].isdigit()])
    grid = np.array([float(c[1:-5]) for c in grid_cols])
    fig, axes = plt.subplots(2, 5, figsize=(6.5, 3.0), sharex=True)
    for r, (scen, beta, rlab) in enumerate(settings):
        for c, pred in enumerate(PRED_ORDER):
            ax = axes[r, c]
            for score, (col, mk, lab) in RISK.items():
                d = sel(rs, scenario=scen, beta=beta, predictor=pred, alpha=alpha, cc=True, score=score)
                if d.empty:
                    continue
                ax.plot(grid, d[grid_cols].to_numpy()[0], color=col, marker=mk, ms=2.5, markevery=4, lw=1.0, label=lab)
            if r == 0:
                ax.set_title(PRED_LABEL[pred])
            if c == 0:
                ax.set_ylabel(f"{rlab}\nsingleton error")
            if r == 1:
                ax.set_xlabel("Retained fraction")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, out, "fig4_risk_coverage")


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    d = results_dir() / args.tag
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    summ = pd.read_csv(d / "summary.csv")
    disp = pd.read_csv(d / "dispersion.csv")
    rs = pd.read_csv(d / "risk_summary.csv")
    fig_s1(summ, disp, out)
    fig_shift(summ, out)
    fig_tradeoff(summ, out)
    fig_risk(rs, out)
    print(f"wrote figures to {out}")


if __name__ == "__main__":
    main()
