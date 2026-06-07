"""
Baseline: TF-IDF + Logistic Regression on LIAR (binary fake-news classification).
Run from project root: python -m src.baseline_liar

Also used by scripts/compare_four_models_metrics.py (same train setup, metrics on a test slice).
"""

import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Paths (anchored to repo root)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PROCESSED = os.path.join(_PROJECT_ROOT, "data_processed")
LIAR_TRAIN = os.path.join(_PROCESSED, "liar_train.csv")
LIAR_VAL = os.path.join(_PROCESSED, "liar_val.csv")
LIAR_TEST = os.path.join(_PROCESSED, "liar_test.csv")

# Model settings (same as Fakeddit baseline)
TFIDF_MAX_FEATURES = 50_000
TFIDF_NGRAM_RANGE = (1, 2)
LOGISTIC_SOLVER = "liblinear"
LOGISTIC_CLASS_WEIGHT = "balanced"


def load_splits():
    """Load train, val, and test CSVs; return (X_train, y_train, X_val, y_val, X_test, y_test)."""
    train = pd.read_csv(LIAR_TRAIN)
    val = pd.read_csv(LIAR_VAL)
    test = pd.read_csv(LIAR_TEST)
    return (
        train["text"].astype(str).fillna(""),
        train["label"].values,
        val["text"].astype(str).fillna(""),
        val["label"].values,
        test["text"].astype(str).fillna(""),
        test["label"].values,
    )


def fit_tfidf_logreg(X_train=None, y_train=None):
    """
    Fit TF-IDF + balanced LogisticRegression on LIAR train split only.
    Returns (vectorizer, classifier).

    If X_train/y_train are omitted, loads them via load_splits().
    """
    if X_train is None or y_train is None:
        X_train, y_train, _, _, _, _ = load_splits()
    vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=TFIDF_NGRAM_RANGE)
    X_train_tf = vectorizer.fit_transform(X_train)
    clf = LogisticRegression(
        solver=LOGISTIC_SOLVER, class_weight=LOGISTIC_CLASS_WEIGHT, random_state=42
    )
    clf.fit(X_train_tf, y_train)
    return vectorizer, clf


def predict_labels_and_confidence(vectorizer, clf, texts):
    """
    Batch predict. Confidence = max predicted class probability.

    Args:
        texts: iterable of str

    Returns:
        y_pred: np.ndarray int, shape (n,)
        confidences: np.ndarray float, shape (n,)
    """
    X = vectorizer.transform(texts)
    proba = clf.predict_proba(X)
    y_pred = np.asarray(proba.argmax(axis=1), dtype=int)
    confidences = np.asarray(proba.max(axis=1), dtype=float)
    return y_pred, confidences


def main():
    print("=" * 60)
    print("LIAR baseline: TF-IDF (unigrams+bigrams) + Logistic Regression")
    print("=" * 60)

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits()
    print(f"\nTrain: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")

    print(
        "\nFitting TfidfVectorizer + LogisticRegression (max_features=%d, ngram_range=%s, solver=%s) ..."
        % (TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, LOGISTIC_SOLVER)
    )
    vectorizer, clf = fit_tfidf_logreg(X_train, y_train)
    X_train_tf = vectorizer.transform(X_train)
    X_val_tf = vectorizer.transform(X_val)
    X_test_tf = vectorizer.transform(X_test)
    print("Done. Train matrix shape:", X_train_tf.shape)

    def evaluate(name, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        print("\n" + "-" * 40)
        print("  %s" % name)
        print("-" * 40)
        print("  Accuracy:  %.4f" % acc)
        print("  Precision (class 1): %.4f" % prec)
        print("  Recall (class 1):    %.4f" % rec)
        print("  F1 (class 1):       %.4f" % f1)
        print("  Confusion matrix (rows=true, cols=pred):")
        print("    %s" % str(cm))
        return acc, prec, rec, f1, cm

    # Validation
    y_val_pred = clf.predict(X_val_tf)
    evaluate("Validation", y_val, y_val_pred)

    # Test
    y_test_pred = clf.predict(X_test_tf)
    evaluate("Test", y_test, y_test_pred)

    print("\n" + "=" * 60)
    print("Summary (copy for report)")
    print("=" * 60)
    for name, y_true, y_pred in [("Validation", y_val, y_val_pred), ("Test", y_test, y_test_pred)]:
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        print("%s — Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f" % (name, acc, prec, rec, f1))
    print("=" * 60)


if __name__ == "__main__":
    main()
