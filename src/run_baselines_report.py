"""
Run TF-IDF + Logistic Regression on Fakeddit and LIAR, then print a summary report.
Assumes data_processed/ CSVs exist (run python -m src.prepare_datasets first).
Run from project root:
  python -m src.run_baselines_report           # both datasets
  python -m src.run_baselines_report fakeddit  # Fakeddit scores only
"""

import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

PROCESSED_DIR = "data_processed"
TFIDF_MAX_FEATURES = 50_000
TFIDF_NGRAM_RANGE = (1, 2)
LOGISTIC_SOLVER = "liblinear"
LOGISTIC_CLASS_WEIGHT = "balanced"


def run_baseline(train_path, val_path, test_path):
    """Load splits, fit TF-IDF on train, train LR, return (n_train, n_val, n_test, val_metrics, test_metrics)."""
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    test = pd.read_csv(test_path)
    X_train = train["text"].astype(str).fillna("")
    y_train = train["label"].values
    X_val = val["text"].astype(str).fillna("")
    y_val = val["label"].values
    X_test = test["text"].astype(str).fillna("")
    y_test = test["label"].values

    vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=TFIDF_NGRAM_RANGE)
    X_train_tf = vectorizer.fit_transform(X_train)
    X_val_tf = vectorizer.transform(X_val)
    X_test_tf = vectorizer.transform(X_test)

    clf = LogisticRegression(solver=LOGISTIC_SOLVER, class_weight=LOGISTIC_CLASS_WEIGHT, random_state=42)
    clf.fit(X_train_tf, y_train)

    def metrics(y_true, y_pred):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
            "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
            "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        }

    val_m = metrics(y_val, clf.predict(X_val_tf))
    test_m = metrics(y_test, clf.predict(X_test_tf))
    return len(y_train), len(y_val), len(y_test), val_m, test_m


def main():
    import sys
    only_fakeddit = len(sys.argv) > 1 and sys.argv[1].lower() == "fakeddit"

    print("Running baseline (TF-IDF + Logistic Regression)%s ..." % (" [Fakeddit only]" if only_fakeddit else ""))
    print()

    # Fakeddit
    f_train, f_val, f_test, f_val_m, f_test_m = run_baseline(
        os.path.join(PROCESSED_DIR, "fakeddit_train.csv"),
        os.path.join(PROCESSED_DIR, "fakeddit_val.csv"),
        os.path.join(PROCESSED_DIR, "fakeddit_test.csv"),
    )

    if only_fakeddit:
        print()
        print("=" * 72)
        print("FAKEDDIT — TF-IDF (unigrams+bigrams) + Logistic Regression")
        print("=" * 72)
        print()
        print("Dataset sizes:  Train: %d  |  Val: %d  |  Test: %d" % (f_train, f_val, f_test))
        print()
        print("-" * 72)
        print("  %-12s  %-10s  %-10s  %-10s  %-10s" % ("Split", "Accuracy", "Precision", "Recall", "F1 (cls 1)"))
        print("-" * 72)
        print("  %-12s  %10.4f  %10.4f  %10.4f  %10.4f" % ("Validation", f_val_m["accuracy"], f_val_m["precision"], f_val_m["recall"], f_val_m["f1"]))
        print("  %-12s  %10.4f  %10.4f  %10.4f  %10.4f" % ("Test", f_test_m["accuracy"], f_test_m["precision"], f_test_m["recall"], f_test_m["f1"]))
        print("-" * 72)
        print()
        print("Copy for report:")
        print("  Validation: Acc %.4f | P %.4f | R %.4f | F1 %.4f" % (f_val_m["accuracy"], f_val_m["precision"], f_val_m["recall"], f_val_m["f1"]))
        print("  Test:       Acc %.4f | P %.4f | R %.4f | F1 %.4f" % (f_test_m["accuracy"], f_test_m["precision"], f_test_m["recall"], f_test_m["f1"]))
        print("=" * 72)
        return

    # LIAR
    l_train, l_val, l_test, l_val_m, l_test_m = run_baseline(
        os.path.join(PROCESSED_DIR, "liar_train.csv"),
        os.path.join(PROCESSED_DIR, "liar_val.csv"),
        os.path.join(PROCESSED_DIR, "liar_test.csv"),
    )

    # Summary report (both)
    print()
    print("=" * 72)
    print("SUMMARY REPORT — TF-IDF (unigrams+bigrams) + Logistic Regression")
    print("=" * 72)
    print()
    print("Dataset sizes:")
    print("  Fakeddit — Train: %d  |  Val: %d  |  Test: %d" % (f_train, f_val, f_test))
    print("  LIAR     — Train: %d  |  Val: %d  |  Test: %d" % (l_train, l_val, l_test))
    print()
    print("-" * 72)
    print("  %-10s  %-12s  %-10s  %-10s  %-10s  %-10s" % ("Dataset", "Split", "Accuracy", "Precision", "Recall", "F1 (cls 1)"))
    print("-" * 72)
    print("  %-10s  %-12s  %10.4f  %10.4f  %10.4f  %10.4f" % ("Fakeddit", "Validation", f_val_m["accuracy"], f_val_m["precision"], f_val_m["recall"], f_val_m["f1"]))
    print("  %-10s  %-12s  %10.4f  %10.4f  %10.4f  %10.4f" % ("Fakeddit", "Test", f_test_m["accuracy"], f_test_m["precision"], f_test_m["recall"], f_test_m["f1"]))
    print("  %-10s  %-12s  %10.4f  %10.4f  %10.4f  %10.4f" % ("LIAR", "Validation", l_val_m["accuracy"], l_val_m["precision"], l_val_m["recall"], l_val_m["f1"]))
    print("  %-10s  %-12s  %10.4f  %10.4f  %10.4f  %10.4f" % ("LIAR", "Test", l_test_m["accuracy"], l_test_m["precision"], l_test_m["recall"], l_test_m["f1"]))
    print("-" * 72)
    print()
    print("Copy for report:")
    print("  Fakeddit — Validation: Acc %.4f | P %.4f | R %.4f | F1 %.4f" % (f_val_m["accuracy"], f_val_m["precision"], f_val_m["recall"], f_val_m["f1"]))
    print("  Fakeddit — Test:       Acc %.4f | P %.4f | R %.4f | F1 %.4f" % (f_test_m["accuracy"], f_test_m["precision"], f_test_m["recall"], f_test_m["f1"]))
    print("  LIAR     — Validation: Acc %.4f | P %.4f | R %.4f | F1 %.4f" % (l_val_m["accuracy"], l_val_m["precision"], l_val_m["recall"], l_val_m["f1"]))
    print("  LIAR     — Test:       Acc %.4f | P %.4f | R %.4f | F1 %.4f" % (l_test_m["accuracy"], l_test_m["precision"], l_test_m["recall"], l_test_m["f1"]))
    print("=" * 72)


if __name__ == "__main__":
    main()
