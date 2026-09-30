import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SCORES_PATH = "outputs/results/detection_scores.json"

LABELS = {
    "random": "Random",
    "heuristic": "Speed-change heuristic",
    "single_step": "Single-step (as published)",
    "total": "Rollout, total divergence",
    "speed_shortfall": "Rollout, speed shortfall",
}
STYLE = {
    "random":          dict(color="#9aa0a6", ls=":",  lw=1.4, zorder=2),
    "heuristic":       dict(color="#e8a33d", ls="-.", lw=1.7, zorder=3),
    "single_step":     dict(color="#d1495b", ls="--", lw=2.0, zorder=4),
    "total":           dict(color="#4a7fb5", ls="-",  lw=1.7, zorder=3),
    "speed_shortfall": dict(color="#2a9d5c", ls="-",  lw=3.0, zorder=6),
}
FILL = "#2a9d5c"


def _tie_aware_curve(benign, attacked):
    """Steps the curve once per distinct score, so tied values do not get an
    arbitrary ordering. Shortfall is zero for many benign sequences by
    construction, so ties are common here and breaking them by input order
    would bias the result."""
    y = np.concatenate([np.zeros(len(benign)), np.ones(len(attacked))])
    s = np.concatenate([benign, attacked])
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    boundaries = np.where(np.diff(s) != 0)[0]
    idx = np.concatenate([boundaries, [len(s) - 1]])
    tp = np.cumsum(y)[idx]
    fp = np.cumsum(1 - y)[idx]
    return tp, fp, y.sum(), len(y) - y.sum()


def auc_mann_whitney(benign, attacked):
    """Rank-based AUC, which assigns tied scores their average rank. This is
    the correct estimator in the presence of ties."""
    s = np.concatenate([benign, attacked])
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n_b, n_a = len(benign), len(attacked)
    r_pos = ranks[n_b:].sum()
    return float((r_pos - n_a * (n_a + 1) / 2) / (n_a * n_b))


def roc_points(benign, attacked):
    tp, fp, n_pos, n_neg = _tie_aware_curve(benign, attacked)
    tpr = np.concatenate([[0.0], tp / max(n_pos, 1), [1.0]])
    fpr = np.concatenate([[0.0], fp / max(n_neg, 1), [1.0]])
    return fpr, tpr, auc_mann_whitney(benign, attacked)


def pr_points(benign, attacked):
    tp, fp, n_pos, _ = _tie_aware_curve(benign, attacked)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(n_pos, 1)
    ap = float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))
    return recall, precision, ap


def main():
    path = REPO_ROOT / SCORES_PATH
    if not path.exists():
        print(f"[abort] {SCORES_PATH} not found.")
        print("[abort] Run scripts/evaluate_detection_paired.py first -- it now")
        print("[abort] writes the per-sequence scores this figure needs.")
        return

    d = json.load(open(path))
    benign, attacked = d["benign"], d["attacked"]
    measures = [m for m in LABELS if m in benign and m in attacked]
    n_b = len(benign[measures[0]])
    n_a = len(attacked[measures[0]])
    print(f"[setup] {n_b} benign, {n_a} attacked sequences, {len(measures)} measures")
    print(f"[setup] each curve therefore has at most {n_b + n_a} distinct points\n")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    summary = {}

    for m in measures:
        b = np.asarray(benign[m], dtype=float)
        a = np.asarray(attacked[m], dtype=float)

        fpr, tpr, auc = roc_points(b, a)
        axes[0].plot(fpr, tpr, label=f"{LABELS[m]} ({auc:.3f})", **STYLE[m])

        rec, prec, ap = pr_points(b, a)
        axes[1].plot(rec, prec, label=f"{LABELS[m]} ({ap:.3f})", **STYLE[m])

        summary[m] = {"auc": auc, "average_precision": ap}
        print(f"  {LABELS[m]:<30} AUC {auc:.4f}   AP {ap:.4f}")

    axes[0].plot([0, 1], [0, 1], color="#c9ccd1", lw=1.0, ls=(0, (4, 3)), zorder=1)
    axes[0].set_xlabel("False-alarm rate")
    axes[0].set_ylabel("Detection rate")
    axes[0].set_title("ROC", fontsize=10)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.02)

    axes[1].axhline(n_a / (n_a + n_b), color="#c9ccd1", lw=1.0, ls=(0, (4, 3)), zorder=1)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-recall", fontsize=10)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1.02)

    for ax in axes:
        ax.set_facecolor("#fbfbfc")
        ax.tick_params(labelsize=8, length=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c9ccd1")

    axes[0].legend(fontsize=6.5, loc="lower right", frameon=False)
    fig.tight_layout()

    out = REPO_ROOT / "outputs" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "detection_curves.pdf", bbox_inches="tight")
    fig.savefig(out / "detection_curves.png", dpi=200, bbox_inches="tight")

    fig2, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.set_facecolor("#fbfbfc")
    for gl in np.arange(0.2, 1.01, 0.2):
        ax.axhline(gl, color="white", lw=1.0, zorder=0)
        ax.axvline(gl, color="white", lw=1.0, zorder=0)
    ax.plot([0, 1], [0, 1], color="#c9ccd1", lw=1.0, ls=(0, (4, 3)), zorder=1)

    for m in measures:
        b = np.asarray(benign[m], dtype=float)
        a = np.asarray(attacked[m], dtype=float)
        fpr, tpr, auc = roc_points(b, a)
        ax.plot(fpr, tpr, label=f"{LABELS[m]}  {auc:.3f}", **STYLE[m])
    ax.set_xlabel("False-alarm rate", fontsize=9)
    ax.set_ylabel("Detection rate", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.tick_params(labelsize=8, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9ccd1")
    leg = ax.legend(fontsize=6.4, loc="lower right", frameon=True,
                    framealpha=0.94, edgecolor="#dfe1e5", borderpad=0.5,
                    handlelength=2.2, labelspacing=0.45)
    leg.get_frame().set_linewidth(0.6)
    fig2.tight_layout()
    fig2.savefig(out / "detection_roc_single_column.pdf", bbox_inches="tight")
    fig2.savefig(out / "detection_roc_single_column.png", dpi=220, bbox_inches="tight")

    with open(REPO_ROOT / "outputs" / "results" / "detection_curves.json", "w") as f:
        json.dump({"n_benign": n_b, "n_attacked": n_a, "summary": summary},
                  f, indent=2)

    print()
    print("=" * 74)
    print("FIGURES WRITTEN")
    print("=" * 74)
    print(f"  outputs/figures/detection_curves.pdf              two panels, full width")
    print(f"  outputs/figures/detection_roc_single_column.pdf   ROC only, one column")
    print()
    print(f"  With {n_b} benign and {n_a} attacked sequences each curve is a")
    print(f"  staircase of at most {n_b + n_a} steps. That is honest for this sample")
    print(f"  size but looks coarse; the single-column ROC is the cheaper option")
    print(f"  if the page budget is tight.")


if __name__ == "__main__":
    main()