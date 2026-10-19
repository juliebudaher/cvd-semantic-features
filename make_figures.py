"""Publication figures (300 dpi PNG + PDF) for the CBM submission."""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = "figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#8a8a85",
                     "axes.labelcolor": "#2b2b29", "xtick.color": "#55554f", "ytick.color": "#55554f"})
INK, QUIET, EDGE, GRID = "#2b2b29", "#6b6b66", "#8a8a85", "#e4e3de"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------- Figure 1: pipeline ----------------
def fig1():
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.set_xlim(0, 760); ax.set_ylim(380, 8); ax.axis("off")

    def box(x, y, w, h, name, l1, l2, accent=False):
        fc = "#e6effa" if accent else "white"
        ec = CAT[0] if accent else EDGE
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=8",
                                    fc=fc, ec=ec, lw=1.6 if accent else 1.0))
        ax.text(x + w / 2, y + 22, name, ha="center", va="center", fontsize=7.6, weight="bold", color=INK)
        ax.text(x + w / 2, y + 42, l1, ha="center", va="center", fontsize=6.4, color=QUIET)
        ax.text(x + w / 2, y + 58, l2, ha="center", va="center", fontsize=6.4, color=QUIET)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=1.0, mutation_scale=9))

    rowA, hA, wA = 20, 72, 166
    xs = [24, 206, 388, 570]
    box(xs[0], rowA, wA, hA, "Two datasets", "Kaggle CVD, n = 68,557", "Framingham, n = 3,750")
    box(xs[1], rowA, wA, hA, "Cleaning", "Fixed plausibility limits", "nothing fitted to data")
    box(xs[2], rowA, wA, hA, "Semantic layer", "ACC/AHA, WHO, NCEP,", "ADA guideline rules", accent=True)
    box(xs[3], rowA, wA, hA, "Six representations", "R1 raw, R2 semantic,", "R3-R6 combinations")
    for i in range(3):
        arrow(xs[i] + wA, rowA + hA / 2, xs[i + 1], rowA + hA / 2)
    cB = 124
    ax.add_patch(FancyBboxPatch((24, cB), 712, 132, boxstyle="round,pad=0,rounding_size=8",
                                fc="#f4f3ef", ec=EDGE, lw=1.0))
    ax.text(40, cB + 20, "Repeated stratified 5 × 5 CV: the same 25 folds for every model and representation",
            fontsize=7.4, weight="bold", color=INK, va="center")
    arrow(653, rowA + hA, 653, cB)
    rowB, wB = 164, 210
    xb = [40, 274, 508]
    box(xb[0], rowB, wB, hA, "Nested tuning", "8 configurations, inner 3-fold", "on the training fold only")
    box(xb[1], rowB, wB, hA, "Four learners", "Logistic regression, tree,", "random forest, XGBoost")
    box(xb[2], rowB, wB, hA, "Validation fold", "ROC-AUC (primary), F1,", "PR-AUC, Brier score")
    arrow(xb[0] + wB, rowB + hA / 2, xb[1], rowB + hA / 2)
    arrow(xb[1] + wB, rowB + hA / 2, xb[2], rowB + hA / 2)
    rowC = 296
    box(xb[0], rowC, wB, hA, "Main comparison", "6 pre-specified pairs,", "corrected t-test, Holm")
    box(xb[1], rowC, wB, hA, "Ablation, learning curve", "Groups removed from R3;", "1%-100% of training data")
    box(xb[2], rowC, wB, hA, "Explainability", "Concept-grouped SHAP,", "compactness, stability")
    for x in xb:
        arrow(x + wB / 2, cB + 132, x + wB / 2, rowC)
    save(fig, "Figure1_pipeline")


# ---------------- Figure 2: concept shares ----------------
def fig2():
    rows = []
    for d, path in [("Kaggle CVD (XGBoost)", "results_cardio"), ("Framingham (logistic regression)", "results_framingham")]:
        c = pd.read_csv(f"{path}/exp4_concept_shares.csv")
        c["dataset"] = d
        rows.append(c)
    c = pd.concat(rows)
    order = ["Blood pressure", "Age", "Metabolic", "Lifestyle", "Body composition", "Sex", "Other clinical", "Composite (CSRI)"]
    col = dict(zip(order, CAT))
    reps = ["R1 Raw", "R2 Semantic", "R3 Raw+Semantic"]
    labels = {"R1 Raw": "R1 Raw", "R2 Semantic": "R2 Semantic only", "R3 Raw+Semantic": "R3 Raw + semantic"}
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.2), sharex=True)
    for ax, d in zip(axes, ["Kaggle CVD (XGBoost)", "Framingham (logistic regression)"]):
        sub = c[c.dataset == d]
        for i, r in enumerate(reps):
            left = 0
            s = sub[sub.representation == r].set_index("concept")["mean"]
            s = s / s.sum() * 100
            for con in order:
                if con not in s.index:
                    continue
                v = s[con]
                ax.barh(i, v - 0.4, left=left + 0.2, height=0.62, color=col[con], edgecolor="none")
                if v >= 6:
                    ax.text(left + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=7.5,
                            color="white" if con in ("Blood pressure", "Composite (CSRI)", "Sex") else INK, weight="bold")
                left += v
        ax.set_yticks(range(3)); ax.set_yticklabels([labels[r] for r in reps])
        ax.invert_yaxis(); ax.set_xlim(0, 100)
        ax.set_title(d, loc="left", fontsize=9, weight="bold", color=INK)
        ax.tick_params(axis="y", length=0)
    axes[-1].set_xlabel("Share of total mean |SHAP| attribution (%)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=col[k]) for k in order]
    fig.legend(handles, ["Blood pressure", "Age", "Metabolic", "Lifestyle", "Body composition", "Sex",
                         "Other clinical", "CSRI"], loc="lower center", ncol=8, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, -0.06), handlelength=1.0, columnspacing=0.9)
    fig.tight_layout()
    save(fig, "Figure2_concept_shares")


# ---------------- Figure 3: learning curve ----------------
def fig3():
    st = pd.read_csv("results_learning_curve/lc_stats.csv")
    names = {"LogReg": "Logistic regression", "DecisionTree": "Decision tree", "XGBoost": "XGBoost"}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
    for ax, mname in zip(axes, ["LogReg", "DecisionTree", "XGBoost"]):
        s = st[st.model == mname]
        ax.axhline(0, color=EDGE, lw=0.9)
        for comp, colr, lab in [("R2 Semantic vs R1 Raw", CAT[0], "R2 semantic only − R1"),
                                ("R3 Raw+Semantic vs R1 Raw", CAT[1], "R3 raw + semantic − R1")]:
            g = s[s.comparison == comp].sort_values("n_train")
            ax.fill_between(g.n_train, g.ci95_low, g.ci95_high, color=colr, alpha=0.15, lw=0)
            ax.plot(g.n_train, g.mean_diff, color=colr, lw=2, marker="o", ms=4, label=lab)
        ax.set_xscale("log")
        ticks = sorted(s.n_train.unique())
        ax.set_xticks(ticks, [f"{t/1000:.1f}k" if t < 10000 else f"{t/1000:.0f}k" for t in ticks], fontsize=7)
        ax.minorticks_off()
        ax.set_ylim(-0.065, 0.03)
        ax.set_title(names[mname], loc="left", fontsize=9, weight="bold", color=INK)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_xlabel("Training records (log scale)")
    axes[0].set_ylabel("Difference in ROC-AUC vs R1")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.1), fontsize=8)
    fig.tight_layout()
    save(fig, "Figure3_learning_curve")


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3"]
    if "1" in which: fig1()
    if "2" in which: fig2()
    if "3" in which: fig3()
