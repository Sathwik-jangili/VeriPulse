#!/usr/bin/env python
"""Build figures from docs/user_evaluation_figures/user_evaluation_responses.csv"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "docs", "user_evaluation_figures", "user_evaluation_responses.csv")
OUT = os.path.join(ROOT, "docs", "user_evaluation_figures")

Q16_SHORT = [
    "Supportive",
    "Easy",
    "Efficient",
    "Clear",
    "Exciting",
    "Interesting",
    "Inventive",
    "Leading-edge",
]


def cronbach_alpha(df, cols):
    X = df[cols].astype(float).values
    k = X.shape[1]
    if k < 2 or len(df) < 2:
        return float("nan")
    item_vars = X.var(axis=0, ddof=1)
    total_var = X.sum(axis=1).var(ddof=1)
    if total_var <= 0:
        return float("nan")
    return (k / (k - 1)) * (1 - item_vars.sum() / total_var)


def main():
    df = pd.read_csv(CSV)
    n = len(df)
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams["figure.dpi"] = 120

    # Fig 1: Age
    fig, ax = plt.subplots(figsize=(6.5, 4))
    df["age"].value_counts().sort_index().plot(kind="bar", ax=ax, color="#3182bd", edgecolor="white")
    ax.set_ylabel("Count")
    ax.set_xlabel("Age group")
    ax.set_title(f"Participant age (N = {n})")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ue_fig01_age.png"), dpi=200)
    plt.close()

    # Fig 2: Frequency
    fig, ax = plt.subplots(figsize=(7, 4))
    df["social_freq"].value_counts().plot(kind="barh", ax=ax, color="#31a354")
    ax.set_xlabel("Count")
    ax.set_title("Social media use frequency")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ue_fig02_social_frequency.png"), dpi=200)
    plt.close()

    # Fig 3: Platform
    fig, ax = plt.subplots(figsize=(7, 4))
    df["platform"].value_counts().plot(kind="barh", ax=ax, color="#756bb1")
    ax.set_xlabel("Count")
    ax.set_title("Most frequent platform")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ue_fig03_platform.png"), dpi=200)
    plt.close()

    def cat_bar(col, title, fname, order=None):
        fig, ax = plt.subplots(figsize=(8, 4))
        vc = df[col].value_counts()
        if order:
            vc = vc.reindex([x for x in order if x in vc.index], fill_value=0)
        vc.plot(kind="bar", ax=ax, color="#fd8d3c", edgecolor="white")
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=18)
        for i, (k, v) in enumerate(vc.items()):
            if v > 0:
                ax.text(i, v + 0.05, str(int(v)), ha="center", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, fname), dpi=200)
        plt.close()

    cat_bar("q4_pred_understand", "Q4: Prediction easy to understand?", "ue_fig04_q4.png", ["Yes", "No"])
    cat_bar(
        "q5_explain_clear",
        "Q5: Clarity of explanation",
        "ue_fig05_q5_clarity.png",
        ["Very clear", "Somewhat clear", "Neutral", "Somewhat unclear", "Very unclear"],
    )
    cat_bar("q6_help_why", "Q6: Explanation helped understand label?", "ue_fig06_q6.png", ["Yes", "No", "Not sure"])
    cat_bar("q7_confidence_up", "Q7: Explanation increased confidence?", "ue_fig07_q7.png", ["Yes", "No", "Not sure"])
    cat_bar(
        "q8_trust",
        "Q8: Trust in predictions",
        "ue_fig08_q8_trust.png",
        ["Strongly trust", "Somewhat trust", "Neutral", "Somewhat distrust", "Strongly distrust"],
    )
    cat_bar("q9_would_use", "Q9: Would use when checking reliability?", "ue_fig09_q9.png", ["Yes", "No", "Maybe"])
    cat_bar(
        "q10_ai_reduce_misinfo",
        "Q10: AI can help reduce misinformation?",
        "ue_fig10_q10.png",
        ["Yes", "No", "Not sure"],
    )
    cat_bar(
        "q11_help_mislead",
        "Q11: System helped assess misleading content?",
        "ue_fig11_q11.png",
        ["Yes", "No", "Not sure"],
    )
    cat_bar(
        "q12_explain_important",
        "Q12: Explanations important when AI decides?",
        "ue_fig12_q12.png",
        ["Yes", "No", "Not sure"],
    )
    cat_bar(
        "q13_prefer_explain",
        "Q13: Prefer explain + prediction vs prediction only?",
        "ue_fig13_q13.png",
        ["Yes", "No"],
    )
    cat_bar("q14_confusing", "Q14: Found part confusing?", "ue_fig14_q14.png", ["Yes", "No"])
    cat_bar(
        "q17_recommend",
        "Q17: Recommend for misinformation identification?",
        "ue_fig15_q17.png",
        ["Definitely yes", "Probably yes", "Not sure", "Probably no", "Definitely no"],
    )

    q16cols = [c for c in df.columns if c.startswith("q16_")]
    means = df[q16cols].mean()
    stds = df[q16cols].std(ddof=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(Q16_SHORT))
    ax.bar(x, means, yerr=stds, capsize=4, color="#74c476", ecolor="#333", alpha=0.92, width=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(Q16_SHORT, rotation=30, ha="right")
    ax.set_ylabel("Mean (1 = negative, 5 = positive pole)")
    ax.set_ylim(0, 5.5)
    ax.axhline(3, color="#888", ls="--", lw=1, label="Scale midpoint (3)")
    ax.set_title("Q16: Overall experience (semantic differentials, mean ± SD)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ue_fig16_q16_differentials.png"), dpi=200)
    plt.close()

    alpha = cronbach_alpha(df, q16cols)
    lines = [f"User evaluation summary (N = {n})", "=" * 44, ""]
    lines.append("Percentages (valid % of respondents)\n")
    for col in [
        "q4_pred_understand",
        "q5_explain_clear",
        "q6_help_why",
        "q7_confidence_up",
        "q8_trust",
        "q9_would_use",
        "q10_ai_reduce_misinfo",
        "q11_help_mislead",
        "q12_explain_important",
        "q13_prefer_explain",
        "q14_confusing",
        "q17_recommend",
    ]:
        lines.append(f"\n{col}:")
        vc = df[col].value_counts()
        for k, v in vc.items():
            lines.append(f"  {k}: {int(v)} ({100 * v / n:.1f}%)")

    lines.append("\n\nQ16 — Mean (SD) per dimension (1–5, higher = more positive end)")
    for name, col in zip(Q16_SHORT, q16cols):
        lines.append(f"  {name}: M = {df[col].mean():.2f}, SD = {df[col].std(ddof=1):.2f}")

    lines.append(f"\nCronbach's α (Q16, 8 items): {alpha:.3f}")
    lines.append(f"Grand mean Q16 (all items): {df[q16cols].values.mean():.3f}")

    with open(os.path.join(OUT, "user_evaluation_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Wrote figures and user_evaluation_summary.txt to", OUT)


if __name__ == "__main__":
    main()
