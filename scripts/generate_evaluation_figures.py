#!/usr/bin/env python
"""
Generate statistical summaries and figures for the dissertation Testing & Evaluation chapter.
Metrics are taken from the project's evaluation screenshots (baseline, DistilBERT, hybrid,
ablation, cross-dataset). Run from project root:

  pip install matplotlib seaborn scipy
  python scripts/generate_evaluation_figures.py

Outputs:
  docs/evaluation_figures/*.png
  docs/evaluation_figures/summary_statistics.csv
"""

from __future__ import annotations

import math
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# -----------------------------------------------------------------------------
# Wilson score 95% CI for a binomial proportion (appropriate for accuracy)
# -----------------------------------------------------------------------------


def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return p, max(0.0, centre - margin), min(1.0, centre + margin)


def accuracy_ci_from_cm(cm: np.ndarray) -> Tuple[float, float, float]:
    """cm rows=true, cols=pred; returns (acc, lo, hi)."""
    n = int(cm.sum())
    correct = int(np.trace(cm))
    return wilson_ci(correct, n)


# =============================================================================
# Numbers keyed to dissertation figures (images 1–11)
# =============================================================================

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "docs",
    "evaluation_figures",
)

# Image 1 — Fakeddit baseline (TF-IDF + LR)
FAKEDDIT_BASELINE_VAL = dict(acc=0.7885, p1=0.7791, r1=0.8073, f1=0.7929)
FAKEDDIT_BASELINE_TEST = dict(acc=0.7961, p1=0.7866, r1=0.8144, f1=0.8003)
FAKEDDIT_BASELINE_TEST_CM = np.array([[4652, 1330], [1117, 4902]])

# Image 2 — LIAR baseline
LIAR_BASELINE_VAL = dict(acc=0.6293, p1=0.6151, r1=0.6071, f1=0.6111)
LIAR_BASELINE_TEST = dict(acc=0.6346, p1=0.5795, r1=0.5931, f1=0.5862)
LIAR_BASELINE_TEST_CM = np.array([[476, 238], [225, 328]])

# Image 3 — Combined baseline comparison (alternate run / larger Fakeddit split in screenshot)
IMG3_FAKEDDIT_VAL = dict(acc=0.8069, p1=0.7993, r1=0.8184, f1=0.8087)
IMG3_FAKEDDIT_TEST = dict(acc=0.8048, p1=0.7967, r1=0.8170, f1=0.8068)

# Image 4 — LIAR DistilBERT (final test from HF Trainer eval)
LIAR_DISTIL_TEST = dict(acc=0.6258879242304657, f1=0.6154233328722701, p1=0.6178490683714041, r1=0.614855)  # noqa: E501

# Image 5 — Fakeddit DistilBERT (28k train, 4 epochs, test eval)
FAKEDDIT_DISTIL_TEST = dict(acc=0.852798, f1=0.852779, p1=0.853054, r1=0.852832)

# Image 6 — Quick run ~2k Fakeddit DistilBERT
QUICK2K = dict(acc=0.7676, p1=0.7543, r1=0.7920, f1=0.7727)

# Image 7 — In-domain LIAR vs Fakeddit (same model family bar chart)
IMG7_LIAR = dict(acc=0.625888, f1=0.615423, p1=0.617849, r1=0.614855)
IMG7_FAKEDDIT = dict(acc=0.852798, f1=0.852779, p1=0.853054, r1=0.852832)

# Image 8 — Cross-dataset
CROSS_L_ON_F = dict(acc=0.490355, f1=0.418303, p1=0.482538, r1=0.491252)
CROSS_F_ON_L = dict(acc=0.430939, f1=0.375983, p1=0.451581, r1=0.476341)

# Image 9 — Ablation (accuracy; F1 from same run in thesis)
ABLATION = [
    ("LIAR", "Full", 0.625888, 0.615423),
    ("LIAR", "Text-only", 0.627466, 0.610424),
    ("LIAR", "Features-only", 0.533544, 0.427163),
    ("Fakeddit", "Full", 0.852798, 0.852779),
    ("Fakeddit", "Text-only", 0.857464, 0.857464),
    ("Fakeddit", "Features-only", 0.668139, 0.668131),
]

# Image 10 — LIAR hybrid (test)
LIAR_HYBRID_TEST = dict(acc=0.6290, p1=0.5793, r1=0.5479, f1=0.5632)

# Image 11 — Fakeddit hybrid (test)
FAKEDDIT_HYBRID_TEST = dict(acc=0.8399, mean_conf=0.7787, std_conf=0.1542)


def plot_style():
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.85)
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 200


def fig_baseline_val_test():
    """Grouped bars: validation vs test for Fakeddit and LIAR baselines (images 1–2)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    metrics = ["Accuracy", "Precision (cls 1)", "Recall (cls 1)", "F1 (cls 1)"]
    x = np.arange(len(metrics))
    w = 0.18

    def pack(d):
        return [d["acc"], d["p1"], d["r1"], d["f1"]]

    fv, ft = pack(FAKEDDIT_BASELINE_VAL), pack(FAKEDDIT_BASELINE_TEST)
    lv, lt = pack(LIAR_BASELINE_VAL), pack(LIAR_BASELINE_TEST)

    ax.bar(x - 1.5 * w, fv, w, label="Fakeddit — Val", color="#2c7fb8")
    ax.bar(x - 0.5 * w, ft, w, label="Fakeddit — Test", color="#253494")
    ax.bar(x + 0.5 * w, lv, w, label="LIAR — Val", color="#7fcdbb")
    ax.bar(x + 1.5 * w, lt, w, label="LIAR — Test", color="#41b6c4")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=12, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Baseline (TF–IDF + balanced LR): validation vs test")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig01_baseline_val_test_four_metrics.png"))
    plt.close(fig)


def fig_confusion_heatmaps():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    cms = [
        ("Fakeddit test (baseline)", FAKEDDIT_BASELINE_TEST_CM),
        ("LIAR test (baseline)", LIAR_BASELINE_TEST_CM),
    ]
    for ax, (title, cm) in zip(axes, cms):
        row_sums = cm.sum(axis=1, keepdims=True)
        norm = np.divide(cm, row_sums, where=row_sums != 0, out=np.zeros_like(cm, dtype=float))
        sns.heatmap(
            norm,
            annot=cm,
            fmt="d",
            cmap="Blues",
            vmin=0,
            vmax=1,
            ax=ax,
            cbar_kws={"label": "Row-normalised"},
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"],
        )
        ax.set_title(title)
    fig.suptitle("Confusion matrices (rows = true, cols = pred)", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig02_baseline_confusion_matrices.png"))
    plt.close(fig)


def fig_models_per_dataset():
    """In-domain: Baseline vs DistilBERT vs Hybrid for each dataset."""
    datasets = ["Fakeddit", "LIAR"]
    baseline = [FAKEDDIT_BASELINE_TEST["acc"], LIAR_BASELINE_TEST["acc"]]
    distil = [FAKEDDIT_DISTIL_TEST["acc"], LIAR_DISTIL_TEST["acc"]]
    hybrid = [FAKEDDIT_HYBRID_TEST["acc"], LIAR_HYBRID_TEST["acc"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(datasets))
    w = 0.25
    ax.bar(x - w, baseline, w, label="Baseline (TF–IDF+LR)", color="#888888")
    ax.bar(x, distil, w, label="DistilBERT", color="#3182bd")
    ax.bar(x + w, hybrid, w, label="Hybrid", color="#31a354")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("In-domain test accuracy by model family")
    for i, ds in enumerate(datasets):
        for k, (h, vals) in enumerate(
            [
                (baseline[i], "baseline"),
                (distil[i], "distil"),
                (hybrid[i], "hybrid"),
            ]
        ):
            ax.text(
                x[i] + (k - 1) * w,
                h + 0.02,
                f"{h:.3f}",
                ha="center",
                fontsize=9,
            )
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig03_in_domain_accuracy_baseline_distil_hybrid.png"))
    plt.close(fig)


def fig_quick_vs_full_distil():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = ["Quick run\n(~2k Fakeddit)", "Full DistilBERT\n(Fakeddit test)"]
    accs = [QUICK2K["acc"], FAKEDDIT_DISTIL_TEST["acc"]]
    cols = ["#fdae61", "#3182bd"]
    bars = ax.bar(labels, accs, color=cols)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("DistilBERT on Fakeddit: pilot run vs full training evaluation")
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.02, f"{a:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig04_fakeddit_quick_run_vs_full_distilbert.png"))
    plt.close(fig)


def fig_in_domain_liar_vs_fakeddit_bar():
    """Image 7 style — four metrics."""
    metrics = ["Accuracy", "F1", "Precision", "Recall"]
    liar = [IMG7_LIAR[k] for k in ["acc", "f1", "p1", "r1"]]
    fake = [IMG7_FAKEDDIT[k] for k in ["acc", "f1", "p1", "r1"]]
    x = np.arange(len(metrics))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, liar, w, label="LIAR", color="#2b6a8e")
    ax.bar(x + w / 2, fake, w, label="Fakeddit", color="#31a354")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_title("In-domain comparison: LIAR vs Fakeddit (same model configuration)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig05_in_domain_liar_vs_fakeddit_four_metrics.png"))
    plt.close(fig)


def fig_cross_dataset():
    metrics = ["Accuracy", "F1", "Precision", "Recall"]
    m1 = [CROSS_L_ON_F["acc"], CROSS_L_ON_F["f1"], CROSS_L_ON_F["p1"], CROSS_L_ON_F["r1"]]
    m2 = [CROSS_F_ON_L["acc"], CROSS_F_ON_L["f1"], CROSS_F_ON_L["p1"], CROSS_F_ON_L["r1"]]
    x = np.arange(len(metrics))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, m1, w, label="Trained LIAR → test Fakeddit", color="#6a51a3")
    ax.bar(x + w / 2, m2, w, label="Trained Fakeddit → test LIAR", color="#fb6a4a")
    ax.axhline(0.5, color="crimson", ls="--", lw=1.2, label="0.5 reference line")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 0.65)
    ax.set_title("Cross-dataset evaluation (domain shift)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig06_cross_dataset_domain_shift.png"))
    plt.close(fig)


def fig_ablation():
    rows = []
    for ds, mode, acc, f1 in ABLATION:
        rows.append({"Dataset": ds, "Mode": mode, "Accuracy": acc, "F1": f1})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, x="Mode", y="Accuracy", hue="Dataset", ax=ax, palette="Set2")
    ax.set_title("Ablation: accuracy by input mode (Full vs Text-only vs Features-only)")
    ax.set_ylim(0.4, 1.0)
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig07_ablation_accuracy_by_mode.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, x="Mode", y="F1", hue="Dataset", ax=ax, palette="Set2")
    ax.set_title("Ablation: F1 by input mode")
    ax.set_ylim(0.35, 1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig08_ablation_f1_by_mode.png"))
    plt.close(fig)


def fig_hybrid_confidence():
    """Image 11: show hybrid mean confidence on Fakeddit vs LIAR hybrid accuracy."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        ["Fakeddit hybrid\n(test accuracy)", "Fakeddit hybrid\n(mean max prob)"],
        [FAKEDDIT_HYBRID_TEST["acc"], FAKEDDIT_HYBRID_TEST["mean_conf"]],
        color=["#31a354", "#a1d99b"],
    )
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Value")
    ax.set_title("Fakeddit hybrid: accuracy vs mean confidence (test set)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig09_fakeddit_hybrid_accuracy_vs_mean_confidence.png"))
    plt.close(fig)


def write_summary_csv():
    rows = []

    def add_row(name: str, n: int, acc_reported: float) -> None:
        k = int(round(acc_reported * n))
        p, lo, hi = wilson_ci(k, n)
        rows.append(
            {
                "experiment": name,
                "n": n,
                "accuracy": acc_reported,
                "wilson_95_ci_low": lo,
                "wilson_95_ci_high": hi,
            }
        )

    for label, cm in [
        ("Fakeddit baseline test (CM)", FAKEDDIT_BASELINE_TEST_CM),
        ("LIAR baseline test (CM)", LIAR_BASELINE_TEST_CM),
    ]:
        n = int(cm.sum())
        k = int(np.trace(cm))
        p, lo, hi = wilson_ci(k, n)
        rows.append(
            {
                "experiment": label,
                "n": n,
                "accuracy": p,
                "wilson_95_ci_low": lo,
                "wilson_95_ci_high": hi,
            }
        )

    # Approximate Wilson CI using test set sizes from prepare_datasets splits
    add_row("Fakeddit DistilBERT test", 24001, FAKEDDIT_DISTIL_TEST["acc"])
    add_row("LIAR DistilBERT test", 1267, LIAR_DISTIL_TEST["acc"])
    add_row("Fakeddit hybrid test", 12000, FAKEDDIT_HYBRID_TEST["acc"])
    add_row("LIAR hybrid test", 1267, LIAR_HYBRID_TEST["acc"])
    add_row("Quick run DistilBERT (~2k)", 2000, QUICK2K["acc"])

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "summary_statistics.csv"), index=False)
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    plot_style()

    fig_baseline_val_test()
    fig_confusion_heatmaps()
    fig_models_per_dataset()
    fig_quick_vs_full_distil()
    fig_in_domain_liar_vs_fakeddit_bar()
    fig_cross_dataset()
    fig_ablation()
    fig_hybrid_confidence()

    df = write_summary_csv()
    print("Wrote figures to:", OUT_DIR)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
