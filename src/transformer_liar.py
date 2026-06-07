"""
Fine-tune DistilBERT on LIAR for binary text classification (0=reliable, 1=unreliable).
Run from project root: python -m src.transformer_liar
"""

import os
import sys

import numpy as np
import pandas as pd
try:
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
    from transformers import EvalPrediction
except ImportError as e:
    print("Missing dependency:", e)
    print("Install with: pip install torch transformers datasets")
    sys.exit(1)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

PROCESSED_DIR = "data_processed"
MODEL_SAVE_DIR = "models/distilbert_liar_trained"
MODEL_NAME = "distilbert-base-uncased"

# --- Hyperparameters for LIAR dataset ---
NUM_EPOCHS = 4
PER_DEVICE_TRAIN_BATCH_SIZE = 16
PER_DEVICE_EVAL_BATCH_SIZE = 32
LEARNING_RATE = 2e-5
MAX_LENGTH = 128
# Save best checkpoint by validation F1 (class 1)
METRIC_FOR_BEST_MODEL = "f1"
# Full training on all LIAR samples
MAX_TRAIN_SAMPLES = None  # Use all samples


def load_datasets():
    """Load LIAR train/val/test CSVs into HuggingFace Dataset with 'text' and 'label'."""
    train = pd.read_csv(os.path.join(PROCESSED_DIR, "liar_train.csv"))
    val = pd.read_csv(os.path.join(PROCESSED_DIR, "liar_val.csv"))
    test = pd.read_csv(os.path.join(PROCESSED_DIR, "liar_test.csv"))
    
    print(f"LIAR Dataset Loaded:")
    print(f"Train: {len(train)} samples")
    print(f"Val: {len(val)} samples") 
    print(f"Test: {len(test)} samples")
    
    train_ds = Dataset.from_pandas(train[["text", "label"]].astype({"text": str, "label": int}))
    val_ds = Dataset.from_pandas(val[["text", "label"]].astype({"text": str, "label": int}))
    test_ds = Dataset.from_pandas(test[["text", "label"]].astype({"text": str, "label": int}))
    return train_ds, val_ds, test_ds


def tokenize(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors=None,
    )


def compute_metrics(eval_pred: EvalPrediction):
    """Return accuracy and P/R/F1 for the positive class (label 1)."""
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, pos_label=1, zero_division=0)
    rec = recall_score(labels, preds, pos_label=1, zero_division=0)
    f1 = f1_score(labels, preds, pos_label=1, zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def main():
    print("=" * 60)
    print("LIAR Dataset: DistilBERT Fine-tuning (Full Training)")
    print("=" * 60)
    
    print("Loading LIAR splits from %s ..." % PROCESSED_DIR)
    train_ds, val_ds, test_ds = load_datasets()
    
    print("Loading tokenizer: %s" % MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Tokenizing (max_length=%d) ..." % MAX_LENGTH)
    train_ds = train_ds.map(
        lambda x: tokenize(x, tokenizer),
        batched=True,
        remove_columns=["text"],
        desc="Tokenize train",
    )
    train_ds = train_ds.rename_column("label", "labels")
    val_ds = val_ds.map(
        lambda x: tokenize(x, tokenizer),
        batched=True,
        remove_columns=["text"],
        desc="Tokenize val",
    )
    val_ds = val_ds.rename_column("label", "labels")
    test_ds = test_ds.map(
        lambda x: tokenize(x, tokenizer),
        batched=True,
        remove_columns=["text"],
        desc="Tokenize test",
    )
    test_ds = test_ds.rename_column("label", "labels")

    print("Loading model: %s (num_labels=2)" % MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # Training configuration for LIAR
    training_args = TrainingArguments(
        output_dir=os.path.join(MODEL_SAVE_DIR, "checkpoints"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=METRIC_FOR_BEST_MODEL,
        greater_is_better=True,
        save_total_limit=1,
    )

    # Trainer setup
    import inspect
    sig = inspect.signature(Trainer.__init__)
    trainer_kw = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    if "tokenizer" in sig.parameters:
        trainer_kw["tokenizer"] = tokenizer
    elif "processing_class" in sig.parameters:
        trainer_kw["processing_class"] = tokenizer
    trainer = Trainer(**trainer_kw)
    
    print("Starting training...")
    trainer.train()

    print("\nEvaluating on test set ...")
    test_out = trainer.predict(test_ds)
    preds = np.argmax(test_out.predictions, axis=-1)
    labels = test_out.label_ids
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, pos_label=1, zero_division=0)
    rec = recall_score(labels, preds, pos_label=1, zero_division=0)
    f1 = f1_score(labels, preds, pos_label=1, zero_division=0)
    
    print("-" * 50)
    print("LIAR Test Results:")
    print("Accuracy:  %.4f" % acc)
    print("Precision (class 1): %.4f" % prec)
    print("Recall (class 1):    %.4f" % rec)
    print("F1 (class 1):       %.4f" % f1)
    print("-" * 50)

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    trainer.save_model(MODEL_SAVE_DIR)
    tokenizer.save_pretrained(MODEL_SAVE_DIR)
    print("Model and tokenizer saved to %s" % MODEL_SAVE_DIR)


if __name__ == "__main__":
    main()
