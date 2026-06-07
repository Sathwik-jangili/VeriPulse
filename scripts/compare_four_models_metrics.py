#!/usr/bin/env python3
"""
Compare neural checkpoints (2 hybrids + 2 DistilBERT) and TF-IDF + LR baselines
on their respective test CSVs (same row count per dataset).

Metrics (definitions match verify_* scripts for neural models):
  - Accuracy: fraction correct
  - Mean confidence: neural = max softmax; baseline = max predict_proba
  - Label bias: P(pred=1) - P(true=1)
  - MCC: Matthews correlation
  - Conf. bias (diff): mean(conf|correct) - mean(conf|wrong)
  - r(conf, correct): Pearson correlation between confidence and 1{correct}

Run from project root:
  python scripts/compare_four_models_metrics.py --limit 400
  python scripts/compare_four_models_metrics.py --limit 0
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

# Keep console readable (LIAR hybrid loads many scalers; sklearn version mismatch noise).
warnings.filterwarnings("ignore", message=".*Trying to unpickle estimator StandardScaler.*")
warnings.filterwarnings("ignore", category=UserWarning, module="src.liar_hybrid_checkpoint")
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, matthews_corrcoef
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MAX_LENGTH = 128

_DATA = os.path.join(PROJECT_ROOT, "data_processed")
FAKEDDIT_TEST = os.path.join(_DATA, "fakeddit_test.csv")
FAKEDDIT_TRAIN = os.path.join(_DATA, "fakeddit_train.csv")
LIAR_TEST = os.path.join(_DATA, "liar_test.csv")
LIAR_TRAIN = os.path.join(_DATA, "liar_train.csv")
DISTIL_FAKEDDIT = os.path.join(PROJECT_ROOT, "models", "distilbert_fakeddit_30k_full")
DISTIL_LIAR = os.path.join(PROJECT_ROOT, "models", "distilbert_liar_trained")


@dataclass
class Row:
    name: str
    dataset: str
    n: int
    ok: bool
    note: str
    accuracy: float
    mean_confidence: float
    label_bias: float
    mcc: float
    conf_bias_delta: float
    r_conf_correct: float


def _aggregate(y_true: np.ndarray, y_pred: np.ndarray, confidences: np.ndarray) -> Dict[str, float]:
    acc = float(accuracy_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    mean_c = float(np.mean(confidences))
    true1 = float((y_true == 1).mean())
    pred1 = float((y_pred == 1).mean())
    label_bias = pred1 - true1
    correct = (y_true == y_pred).astype(float)
    mc = confidences[correct == 1]
    mw = confidences[correct == 0]
    mean_mc = float(mc.mean()) if len(mc) else float("nan")
    mean_mw = float(mw.mean()) if len(mw) else float("nan")
    conf_bias_delta = mean_mc - mean_mw if not (np.isnan(mean_mc) or np.isnan(mean_mw)) else float("nan")
    if np.std(confidences) > 1e-12 and np.std(correct) > 1e-12:
        r_pb = float(np.corrcoef(correct, confidences)[0, 1])
    else:
        r_pb = float("nan")
    return {
        "accuracy": acc,
        "mean_confidence": mean_c,
        "label_bias": label_bias,
        "mcc": mcc,
        "conf_bias_delta": conf_bias_delta,
        "r_conf_correct": r_pb,
    }


def _fmt(x: float, d: int = 4) -> str:
    if x != x:  # NaN
        return "nan"
    return f"{x:.{d}f}"


def eval_fakeddit_hybrid(texts: List[str], y_true: np.ndarray) -> Tuple[bool, str, np.ndarray, np.ndarray]:
    from src.veripulse_predictor import get_predictor

    pred = get_predictor()
    st = pred.load()
    if not st.model_loaded:
        return False, st.message or "not loaded", np.array([]), np.array([])

    y_pred: List[int] = []
    confidences: List[float] = []
    for i, t in enumerate(texts):
        out = pred.predict(t)
        if not out.get("model_loaded"):
            return False, "Hybrid fell back to heuristic mid-run", np.array([]), np.array([])
        y_pred.append(int(out["prediction"]))
        confidences.append(float(out["confidence"]))
        if (i + 1) % 500 == 0:
            print(f"  [Fakeddit hybrid] {i + 1}/{len(texts)} ...")

    return True, "", np.array(y_pred), np.array(confidences)


def eval_liar_hybrid(texts: List[str], y_true: np.ndarray) -> Tuple[bool, str, np.ndarray, np.ndarray]:
    from src.liar_hybrid_checkpoint import find_liar_hybrid_checkpoint, linguistic_batch_for_liar, load_liar_hybrid

    d, wp = find_liar_hybrid_checkpoint()
    if not wp:
        return False, "No LIAR hybrid .bin found", np.array([]), np.array([])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model, info = load_liar_hybrid(device=device)
    variant = info["variant"]
    model_dir = info["model_dir"]

    y_pred: List[int] = []
    confidences: List[float] = []
    for i, t in enumerate(texts):
        inputs = tokenizer(
            t,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        meta = linguistic_batch_for_liar([t], model_dir, variant).to(device)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                meta_features=meta,
            )
            logits = out["logits"]
            probs = F.softmax(logits, dim=-1)
            pred = int(torch.argmax(probs, dim=-1).item())
            conf = float(torch.max(probs, dim=-1).values.item())
        y_pred.append(pred)
        confidences.append(conf)
        if (i + 1) % 500 == 0:
            print(f"  [LIAR hybrid] {i + 1}/{len(texts)} ...")

    return True, "", np.array(y_pred), np.array(confidences)


def eval_baseline_fakeddit(texts: List[str], y_true: np.ndarray) -> Tuple[bool, str, np.ndarray, np.ndarray]:
    """TF-IDF + LogisticRegression (src.baseline_fakeddit); train on fakeddit_train only."""
    if not os.path.isfile(FAKEDDIT_TRAIN):
        return False, f"Missing {FAKEDDIT_TRAIN}", np.array([]), np.array([])
    try:
        from src.baseline_fakeddit import fit_tfidf_logreg, predict_labels_and_confidence
    except ImportError:
        from baseline_fakeddit import fit_tfidf_logreg, predict_labels_and_confidence  # type: ignore

    print("  [Baseline Fakeddit TF-IDF+LR] fitting on train split ...")
    vectorizer, clf = fit_tfidf_logreg()
    y_pred, confidences = predict_labels_and_confidence(vectorizer, clf, texts)
    return True, "", y_pred, confidences


def eval_baseline_liar(texts: List[str], y_true: np.ndarray) -> Tuple[bool, str, np.ndarray, np.ndarray]:
    """TF-IDF + LogisticRegression (src.baseline_liar); train on liar_train only."""
    if not os.path.isfile(LIAR_TRAIN):
        return False, f"Missing {LIAR_TRAIN}", np.array([]), np.array([])
    try:
        from src.baseline_liar import fit_tfidf_logreg, predict_labels_and_confidence
    except ImportError:
        from baseline_liar import fit_tfidf_logreg, predict_labels_and_confidence  # type: ignore

    print("  [Baseline LIAR TF-IDF+LR] fitting on train split ...")
    vectorizer, clf = fit_tfidf_logreg()
    y_pred, confidences = predict_labels_and_confidence(vectorizer, clf, texts)
    return True, "", y_pred, confidences


def _has_weights(model_dir: str) -> bool:
    if not os.path.isdir(model_dir):
        return False
    return any(
        os.path.isfile(os.path.join(model_dir, n))
        for n in ("model.safetensors", "pytorch_model.bin", "model.bin")
    )


def eval_distilbert(model_dir: str, texts: List[str], y_true: np.ndarray) -> Tuple[bool, str, np.ndarray, np.ndarray]:
    if not _has_weights(model_dir):
        return False, f"Missing weights in {model_dir}", np.array([]), np.array([])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    y_pred: List[int] = []
    confidences: List[float] = []
    for i, t in enumerate(texts):
        inputs = tokenizer(
            t,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits
            probs = F.softmax(logits, dim=-1)
            pred = int(torch.argmax(probs, dim=-1).item())
            conf = float(torch.max(probs, dim=-1).values.item())
        y_pred.append(pred)
        confidences.append(conf)
        if (i + 1) % 500 == 0:
            print(f"  [DistilBERT {os.path.basename(model_dir)}] {i + 1}/{len(texts)} ...")

    return True, "", np.array(y_pred), np.array(confidences)


def _run_one(
    name: str,
    dataset: str,
    y_true: np.ndarray,
    texts: List[str],
    runner,
) -> Row:
    ok, note, y_pred, conf = runner()
    n = len(texts)
    if not ok or len(y_pred) != n:
        return Row(
            name, dataset, n, False, note[:80] if note else "failed",
            float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"),
        )
    m = _aggregate(y_true, y_pred, conf)
    return Row(
        name,
        dataset,
        n,
        True,
        "",
        m["accuracy"],
        m["mean_confidence"],
        m["label_bias"],
        m["mcc"],
        m["conf_bias_delta"],
        m["r_conf_correct"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=400, help="Rows per dataset (0 = full CSV)")
    args = parser.parse_args()

    if not os.path.isfile(FAKEDDIT_TEST):
        print(f"Missing {FAKEDDIT_TEST}")
        sys.exit(2)
    if not os.path.isfile(LIAR_TEST):
        print(f"Missing {LIAR_TEST}")
        sys.exit(2)
    if not os.path.isfile(FAKEDDIT_TRAIN):
        print(f"Warning: missing {FAKEDDIT_TRAIN} — Fakeddit baseline row will fail.", file=sys.stderr)
    if not os.path.isfile(LIAR_TRAIN):
        print(f"Warning: missing {LIAR_TRAIN} — LIAR baseline row will fail.", file=sys.stderr)

    df_f = pd.read_csv(FAKEDDIT_TEST)
    df_l = pd.read_csv(LIAR_TEST)
    if args.limit and args.limit > 0:
        df_f = df_f.head(args.limit)
        df_l = df_l.head(args.limit)

    texts_f = df_f["text"].astype(str).fillna("").tolist()
    y_f = df_f["label"].astype(int).values
    texts_l = df_l["text"].astype(str).fillna("").tolist()
    y_l = df_l["label"].astype(int).values

    rows: List[Row] = []

    rows.append(
        _run_one(
            "Hybrid (Fakeddit 60k)",
            "fakeddit_test.csv",
            y_f,
            texts_f,
            lambda: eval_fakeddit_hybrid(texts_f, y_f),
        )
    )
    rows.append(
        _run_one(
            "DistilBERT (Fakeddit 30k)",
            "fakeddit_test.csv",
            y_f,
            texts_f,
            lambda: eval_distilbert(DISTIL_FAKEDDIT, texts_f, y_f),
        )
    )
    rows.append(
        _run_one(
            "Baseline TF-IDF+LR (Fakeddit)",
            "fakeddit_test.csv",
            y_f,
            texts_f,
            lambda: eval_baseline_fakeddit(texts_f, y_f),
        )
    )
    rows.append(
        _run_one(
            "Hybrid (LIAR)",
            "liar_test.csv",
            y_l,
            texts_l,
            lambda: eval_liar_hybrid(texts_l, y_l),
        )
    )
    rows.append(
        _run_one(
            "DistilBERT (LIAR)",
            "liar_test.csv",
            y_l,
            texts_l,
            lambda: eval_distilbert(DISTIL_LIAR, texts_l, y_l),
        )
    )
    rows.append(
        _run_one(
            "Baseline TF-IDF+LR (LIAR)",
            "liar_test.csv",
            y_l,
            texts_l,
            lambda: eval_baseline_liar(texts_l, y_l),
        )
    )

    # Markdown table
    print()
    print("### Model comparison: hybrids, DistilBERT, TF-IDF baselines (same N per dataset)")
    print()
    print(
        "| Model | Test CSV | N | Accuracy | Mean conf. | Label bias (diff) | MCC | Conf. bias (diff) | r(conf, correct) |"
    )
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in rows:
        if r.ok:
            print(
                f"| {r.name} | `{r.dataset}` | {r.n} | {_fmt(r.accuracy)} | {_fmt(r.mean_confidence)} | "
                f"{_fmt(r.label_bias)} | {_fmt(r.mcc)} | {_fmt(r.conf_bias_delta)} | {_fmt(r.r_conf_correct)} |"
            )
        else:
            print(f"| {r.name} | `{r.dataset}` | {r.n} | - | - | - | - | - | - | *{r.note}* |")

    print()
    print(
        "**Label bias (diff)** = P(pred=1) - P(true=1).  "
        "**Conf. bias (diff)** = mean(conf|correct) - mean(conf|wrong).  "
        "**r(conf, correct)** = Pearson correlation of confidence with 1{prediction = label} "
        "(neural: max softmax; baseline: max predict_proba)."
    )
    print()

    # TSV for copy-paste
    print("--- TSV ---")
    print("model\tdataset\tn\taccuracy\tmean_conf\tlabel_bias\tmcc\tconf_bias_delta\tr_conf_correct\tok\tnote")
    for r in rows:
        print(
            f"{r.name}\t{r.dataset}\t{r.n}\t{_fmt(r.accuracy)}\t{_fmt(r.mean_confidence)}\t"
            f"{_fmt(r.label_bias)}\t{_fmt(r.mcc)}\t{_fmt(r.conf_bias_delta)}\t{_fmt(r.r_conf_correct)}\t"
            f"{int(r.ok)}\t{r.note.replace(chr(9), ' ')}"
        )


if __name__ == "__main__":
    main()
