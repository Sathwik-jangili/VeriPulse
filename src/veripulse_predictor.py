"""
VeriPulse production predictor: Advanced Hybrid Fakeddit (60k / Colab export).

Convention (matches training / Fakeddit CSV in this repo):
  - prediction 0 = Reliable (real)
  - prediction 1 = Unreliable (fake)

Weights are expected as state_dict in one of:
  advanced_hybrid_model.bin | fakeddit_model.bin | model.bin

Set VERIPULSE_MODEL_DIR to override the search directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

try:
    from src.rich_prediction_explanation import build_rich_explanation
except ImportError:
    from rich_prediction_explanation import build_rich_explanation

# Project root = parent of src/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Canonical Colab export (zip extract). Put advanced_hybrid_model.bin (or fakeddit_model.bin) here.
FAKEDDIT_60K_ZIP_ROOT = os.path.join(
    PROJECT_ROOT,
    "models",
    "advanced_hybrid_fakeddit_60k-20260322T162137Z-3-001",
)
FAKEDDIT_60K_INNER = os.path.join(FAKEDDIT_60K_ZIP_ROOT, "advanced_hybrid_fakeddit_60k")

DEFAULT_MODEL_CANDIDATES = [
    os.environ.get("VERIPULSE_MODEL_DIR", "").strip(),
    # Prefer your actual 60k export first (inner folder, then zip root if you dropped .bin there)
    FAKEDDIT_60K_INNER,
    FAKEDDIT_60K_ZIP_ROOT,
    os.path.join(PROJECT_ROOT, "models", "advanced_hybrid_fakeddit_60k"),
    os.path.join(PROJECT_ROOT, "models", "advanced_hybrid_fakeddit_trained"),
    # Older / alternate checkpoint — only used if nothing above contains a .bin
    os.path.join(PROJECT_ROOT, "models", "fakeddit_hybrid_model", "advanced_hybrid_fakeddit_trained"),
]

WEIGHT_FILENAMES = (
    "advanced_hybrid_model.bin",
    "hybrid_fakeddit_model.bin",
    "fakeddit_model.bin",
    "model.bin",
    "pytorch_model.bin",
)

BASE_TOKENIZER = "distilbert-base-uncased"
MAX_LENGTH = 128


def _find_model_dir() -> Optional[str]:
    for d in DEFAULT_MODEL_CANDIDATES:
        if not d or not os.path.isdir(d):
            continue
        for name in WEIGHT_FILENAMES:
            if os.path.isfile(os.path.join(d, name)):
                return d
    return None


def _find_weight_path(model_dir: str) -> Optional[str]:
    for name in WEIGHT_FILENAMES:
        p = os.path.join(model_dir, name)
        if os.path.isfile(p):
            return p
    return None


def _linguistic_feature_dict(text: str) -> Dict[str, float]:
    """Six features aligned with AdvancedHybridTransformerModel.extract_linguistic_features."""
    from src.advanced_hybrid_transformer import AdvancedHybridTransformerModel

    t = AdvancedHybridTransformerModel.extract_linguistic_features([text])
    v = t[0].tolist()
    keys = [
        "length_norm",
        "exclaim_norm",
        "question_norm",
        "uppercase_ratio",
        "avg_word_len_norm",
        "salience_norm",
    ]
    return dict(zip(keys, v))


def _heuristic_predict(text: str) -> Tuple[int, float, List[float], str]:
    """Lightweight fallback when no checkpoint is on disk."""
    t = text.lower()
    hits = sum(
        1
        for w in (
            "breaking",
            "shocking",
            "miracle",
            "cure",
            "secret",
            "conspiracy",
            "doctors hate",
            "one weird",
            "aliens",
            "5g",
        )
        if w in t
    )
    excl = t.count("!")
    score = min(1.0, hits * 0.15 + excl * 0.08)
    p_unreliable = float(np.clip(0.35 + score * 0.55, 0.08, 0.92))
    pred = 1 if p_unreliable >= 0.5 else 0
    conf = max(p_unreliable, 1.0 - p_unreliable)
    probs = [1.0 - p_unreliable, p_unreliable] if pred == 1 else [p_unreliable, 1.0 - p_unreliable]
    # normalize
    s = sum(probs)
    probs = [p / s for p in probs]
    conf = max(probs)
    return pred, conf, probs, "heuristic"


@dataclass
class PredictorStatus:
    model_loaded: bool
    model_dir: Optional[str]
    weight_file: Optional[str]
    message: str


class VeriPulsePredictor:
    """Lazy-loading singleton-style predictor for Flask / scripts."""

    def __init__(self) -> None:
        self._tokenizer = None
        self._model = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model_dir: Optional[str] = None
        self._weight_file: Optional[str] = None
        self._load_error: Optional[str] = None
        self._checkpoint_variant: Optional[str] = None  # colab_distil_hybrid | advanced_hybrid_legacy

    def load(self) -> PredictorStatus:
        """Load weights from disk if available; returns current status."""
        self._ensure_loaded()
        return self.status()

    def status(self) -> PredictorStatus:
        if self._model is not None and self._tokenizer is not None:
            return PredictorStatus(
                True,
                self._model_dir,
                self._weight_file,
                "Advanced hybrid weights loaded.",
            )
        if self._load_error:
            return PredictorStatus(False, self._model_dir, self._weight_file, self._load_error)
        d = _find_model_dir()
        if not d:
            return PredictorStatus(
                False,
                None,
                None,
                "No weight file found. Copy your Colab export (.bin) into "
                "models/advanced_hybrid_fakeddit_60k/ (or set VERIPULSE_MODEL_DIR).",
            )
        wf = _find_weight_path(d)
        if not wf:
            return PredictorStatus(
                False,
                d,
                None,
                f"Directory exists but no {WEIGHT_FILENAMES} present.",
            )
        return PredictorStatus(False, d, wf, "Not loaded yet; call predict() to load.")

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        try:
            try:
                from src.fakeddit_hybrid_checkpoint import load_advanced_hybrid_fakeddit
            except ImportError:
                from fakeddit_hybrid_checkpoint import load_advanced_hybrid_fakeddit

            tok, model, info = load_advanced_hybrid_fakeddit(device=str(self._device))
            self._tokenizer = tok
            self._model = model
            self._model_dir = info["model_dir"]
            self._weight_file = info["weight_path"]
            self._checkpoint_variant = info.get("variant", "advanced_hybrid_legacy")
            self._load_error = None
        except FileNotFoundError as e:
            self._load_error = str(e)
            self._model = None
            self._tokenizer = None
            self._checkpoint_variant = None
        except Exception as e:
            self._load_error = str(e)
            self._model = None
            self._tokenizer = None
            self._checkpoint_variant = None

    def predict(self, text: str) -> Dict[str, Any]:
        text = (text or "").strip()
        if not text:
            return {
                "prediction": 0,
                "label": "Reliable",
                "confidence": 0.0,
                "probabilities": [0.5, 0.5],
                "model_loaded": False,
                "fallback": "empty",
                "detail": "Empty text",
                "explanation": "No text was entered, so no content could be scored.",
                "explanation_source": "n/a",
            }

        self._ensure_loaded()

        if self._model is None or self._tokenizer is None:
            pred, conf, probs, fb = _heuristic_predict(text)
            feats = _linguistic_feature_dict(text)
            expl = build_rich_explanation(text, pred, conf, probs, feats, False, fb)
            return {
                "prediction": pred,
                "label": "Unreliable" if pred == 1 else "Reliable",
                "confidence": round(conf, 4),
                "probabilities": [round(p, 4) for p in probs],
                "model_loaded": False,
                "fallback": fb,
                "detail": self._load_error or "Weights missing; heuristic used.",
                "features": feats,
                "explanation": expl,
                "explanation_source": "template",
            }

        try:
            from src.advanced_hybrid_transformer import AdvancedHybridTransformerModel
            from src.fakeddit_colab_hybrid_model import (
                colab_linguistic_features,
                linguistic_feature_dict_for_api,
            )
        except ImportError:
            from advanced_hybrid_transformer import AdvancedHybridTransformerModel
            from fakeddit_colab_hybrid_model import (
                colab_linguistic_features,
                linguistic_feature_dict_for_api,
            )

        inputs = self._tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        if self._checkpoint_variant == "colab_distil_hybrid":
            meta = colab_linguistic_features([text], self._model_dir).to(self._device)
            feat_dict = linguistic_feature_dict_for_api(text)
        else:
            meta = AdvancedHybridTransformerModel.extract_linguistic_features([text]).to(self._device)
            feat_dict = _linguistic_feature_dict(text)
        input_ids = inputs["input_ids"].to(self._device)
        attention_mask = inputs["attention_mask"].to(self._device)

        with torch.no_grad():
            out = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                meta_features=meta,
            )
            logits = out["logits"]
            probs = F.softmax(logits, dim=-1)
            pred = int(torch.argmax(probs, dim=-1).item())
            confidence = float(torch.max(probs, dim=-1).values.item())
            prob_list = probs[0].cpu().tolist()

        expl = build_rich_explanation(
            text, pred, confidence, prob_list, feat_dict, True, "none"
        )
        return {
            "prediction": pred,
            "label": "Unreliable" if pred == 1 else "Reliable",
            "confidence": round(confidence, 4),
            "probabilities": [round(float(p), 4) for p in prob_list],
            "model_loaded": True,
            "fallback": "none",
            "checkpoint_variant": self._checkpoint_variant,
            "fusion_text_only": bool(getattr(self._model, "fusion_text_only", False)),
            "model_dir": self._model_dir,
            "weight_file": self._weight_file,
            "features": feat_dict,
            "explanation": expl,
            "explanation_source": "template",
        }


_GLOBAL: Optional[VeriPulsePredictor] = None


def get_predictor() -> VeriPulsePredictor:
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = VeriPulsePredictor()
    return _GLOBAL


def predict_text(text: str) -> Tuple[int, float]:
    """Backward-compatible (prediction, confidence) for live feed; 0=reliable, 1=unreliable."""
    r = get_predictor().predict(text)
    return int(r["prediction"]), float(r["confidence"])
