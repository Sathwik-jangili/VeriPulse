"""
Colab-export Fakeddit hybrid: matches hybrid_fakeddit_model.bin on disk.

Checkpoint keys: distilbert.*, linguistic_projection.*, fusion_bottleneck.*
This is NOT the same as AdvancedHybridTransformerModel (text_encoder / feature_processor / fusion_layer).

Optional: linguistic_scaler.joblib (sklearn StandardScaler fit on 60k) in the model directory.
If missing, raw 6-D features are passed through — add the scaler from Colab for parity.
"""

from __future__ import annotations

import os
import warnings
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel

# Salience set from your Colab / audit doc (case-insensitive substring match).
SALIENCE_KEYWORDS_COLAB = (
    "breaking",
    "shocking",
    "unbelievable",
    "urgent",
    "must-see",
    "official",
)

_SCALER_WARNED = False


def linguistic_feature_dict_for_api(text: str) -> dict:
    """Human-readable raw features for /predict JSON (not necessarily scaled)."""
    row = _raw_linguistic_row(text)
    keys = (
        "word_count",
        "exclamation_count",
        "question_count",
        "uppercase_ratio",
        "avg_word_length",
        "salience_hits",
    )
    return dict(zip(keys, [float(x) for x in row]))


def _raw_linguistic_row(text: str) -> np.ndarray:
    """Six raw features (same semantics as typical Colab pipeline; scales via StandardScaler in training)."""
    words = text.split()
    wc = float(len(words))
    excl = float(text.count("!"))
    quest = float(text.count("?"))
    upper = sum(1 for w in words if w.isupper())
    upper_ratio = upper / max(len(words), 1)
    avg_len = float(np.mean([len(w) for w in words])) if words else 0.0
    tl = text.lower()
    sal = float(sum(1 for kw in SALIENCE_KEYWORDS_COLAB if kw in tl))
    return np.array([wc, excl, quest, upper_ratio, avg_len, sal], dtype=np.float64)


def colab_linguistic_features(
    texts: List[str],
    model_dir: Optional[str] = None,
) -> torch.Tensor:
    """
    Build [B, 6] linguistic tensor. Applies linguistic_scaler.joblib from model_dir if present.
    """
    global _SCALER_WARNED
    X = np.stack([_raw_linguistic_row(t or "") for t in texts], axis=0)
    scaler_path = None
    if model_dir:
        for name in ("linguistic_scaler.joblib", "linguistic_scaler.pkl"):
            p = os.path.join(model_dir, name)
            if os.path.isfile(p):
                scaler_path = p
                break
    if scaler_path:
        try:
            import joblib

            scaler = joblib.load(scaler_path)
            X = scaler.transform(X)
        except Exception as e:
            warnings.warn(f"Failed to load linguistic scaler {scaler_path}: {e}", stacklevel=2)
    elif model_dir and not _SCALER_WARNED:
        _SCALER_WARNED = True
        warnings.warn(
            "linguistic_scaler.joblib not found next to the checkpoint; using raw 6-D linguistic "
            "features. Export the StandardScaler from Colab for metrics to match training.",
            stacklevel=2,
        )
    return torch.tensor(X, dtype=torch.float32)


def is_colab_distil_hybrid_state_dict(state: dict) -> bool:
    return any(
        isinstance(k, str) and k.startswith("fusion_bottleneck.") for k in state.keys()
    )


class ColabDistilHybridFakeddit(nn.Module):
    """
    Mirrors the Colab class used with hybrid_fakeddit_model.bin:
    DistilBERT CLS (768) + Linear(6,16) linguistic + fusion bottleneck -> 2 logits.
    """

    def __init__(self, linguistic_input_dim: int = 6, linguistic_proj_dim: int = 16, num_labels: int = 2):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained(
            "distilbert-base-uncased", output_attentions=False
        )
        self.linguistic_projection = nn.Linear(linguistic_input_dim, linguistic_proj_dim)
        fusion_input_dim = 768 + linguistic_proj_dim
        self.fusion_bottleneck = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        meta_features: Optional[torch.Tensor] = None,
        linguistic_features: Optional[torch.Tensor] = None,
    ):
        ling = linguistic_features if linguistic_features is not None else meta_features
        if ling is None:
            raise ValueError("meta_features / linguistic_features required")
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        ling_proj = F.relu(self.linguistic_projection(ling))
        combined = torch.cat([cls_embedding, ling_proj], dim=-1)
        logits = self.fusion_bottleneck(combined)
        return {"logits": logits}
