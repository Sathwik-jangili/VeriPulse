"""
Route /predict to Fakeddit or LIAR, DistilBERT (sequence) or Hybrid checkpoint.

Convention: prediction 0 = Reliable, 1 = Unreliable.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DISTILBERT_FAKEDDIT_DIR = os.path.join(PROJECT_ROOT, "models", "distilbert_fakeddit_30k_full")
DISTILBERT_LIAR_DIR = os.path.join(PROJECT_ROOT, "models", "distilbert_liar_trained")
MAX_LENGTH = 128

try:
    from src.rich_prediction_explanation import build_rich_explanation
except ImportError:
    from rich_prediction_explanation import build_rich_explanation


def _linguistic_feature_dict(text: str) -> Dict[str, float]:
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


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _distil_forward(tokenizer, model, text: str, device: torch.device) -> Tuple[int, float, List[float]]:
    inputs = tokenizer(
        text,
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
        prob_list = [float(x) for x in probs[0].cpu().tolist()]
    return pred, conf, prob_list


def _load_distil_pair(model_dir: str, device: torch.device) -> Optional[Tuple[Any, Any]]:
    if not os.path.isdir(model_dir):
        return None
    has_w = any(
        os.path.isfile(os.path.join(model_dir, n))
        for n in ("model.safetensors", "pytorch_model.bin", "model.bin")
    )
    if not has_w:
        return None
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return tokenizer, model


class MultiModelRouter:
    """Lazy-loaded bundle of four model paths."""

    def __init__(self) -> None:
        self._device = _device()
        self._distil_fakeddit: Optional[Tuple[Any, Any]] = None
        self._distil_liar: Optional[Tuple[Any, Any]] = None
        self._liar_bundle: Optional[Tuple[Any, Any, Dict[str, Any]]] = None

    def predict(self, text: str, dataset: str, arch: str) -> Dict[str, Any]:
        dataset = (dataset or "fakeddit").strip().lower()
        arch = (arch or "hybrid").strip().lower()
        if dataset not in ("fakeddit", "liar"):
            return {"error": f"Invalid dataset: {dataset}. Use fakeddit or liar."}
        if arch not in ("distilbert", "hybrid"):
            return {"error": f"Invalid arch: {arch}. Use distilbert or hybrid."}

        if dataset == "fakeddit" and arch == "hybrid":
            try:
                from src.veripulse_predictor import get_predictor

                out = dict(get_predictor().predict(text))
            except Exception as e:
                return {"error": str(e), "model_loaded": False}
            out["dataset"] = "fakeddit"
            out["arch"] = "hybrid"
            return out

        if dataset == "fakeddit" and arch == "distilbert":
            if self._distil_fakeddit is None:
                self._distil_fakeddit = _load_distil_pair(DISTILBERT_FAKEDDIT_DIR, self._device)
            ok = self._distil_fakeddit is not None
            if ok:
                tok, mdl = self._distil_fakeddit
                pred, conf, probs = _distil_forward(tok, mdl, text, self._device)
            else:
                pred, conf, probs = 0, 0.5, [0.5, 0.5]
            feat = _linguistic_feature_dict(text)
            expl = build_rich_explanation(
                text, pred, conf, probs, feat, ok, "none" if ok else "missing_weights"
            )
            return {
                "prediction": pred,
                "label": "Unreliable" if pred == 1 else "Reliable",
                "confidence": round(conf, 4),
                "probabilities": [round(p, 4) for p in probs],
                "model_loaded": ok,
                "fallback": "none" if ok else "missing_weights",
                "features": feat,
                "explanation": expl,
                "explanation_source": "template",
                "dataset": "fakeddit",
                "arch": "distilbert",
                "model_dir": DISTILBERT_FAKEDDIT_DIR,
            }

        if dataset == "liar" and arch == "distilbert":
            if self._distil_liar is None:
                self._distil_liar = _load_distil_pair(DISTILBERT_LIAR_DIR, self._device)
            ok = self._distil_liar is not None
            if ok:
                tok, mdl = self._distil_liar
                pred, conf, probs = _distil_forward(tok, mdl, text, self._device)
            else:
                pred, conf, probs = 0, 0.5, [0.5, 0.5]
            feat = _linguistic_feature_dict(text)
            expl = build_rich_explanation(
                text, pred, conf, probs, feat, ok, "none" if ok else "missing_weights"
            )
            return {
                "prediction": pred,
                "label": "Unreliable" if pred == 1 else "Reliable",
                "confidence": round(conf, 4),
                "probabilities": [round(p, 4) for p in probs],
                "model_loaded": ok,
                "fallback": "none" if ok else "missing_weights",
                "features": feat,
                "explanation": expl,
                "explanation_source": "template",
                "dataset": "liar",
                "arch": "distilbert",
                "model_dir": DISTILBERT_LIAR_DIR,
            }

        # liar + hybrid
        try:
            from src.liar_hybrid_checkpoint import (
                find_liar_hybrid_checkpoint,
                linguistic_batch_for_liar,
                load_liar_hybrid,
            )
            try:
                from src.fakeddit_colab_hybrid_model import linguistic_feature_dict_for_api
            except ImportError:
                from fakeddit_colab_hybrid_model import linguistic_feature_dict_for_api
        except ImportError as e:
            return {"error": str(e), "model_loaded": False, "dataset": "liar", "arch": "hybrid"}

        _, wp = find_liar_hybrid_checkpoint()
        if not wp:
            feat = _linguistic_feature_dict(text)
            expl = build_rich_explanation(text, 0, 0.5, [0.5, 0.5], feat, False, "missing_weights")
            return {
                "prediction": 0,
                "label": "Reliable",
                "confidence": 0.5,
                "probabilities": [0.5, 0.5],
                "model_loaded": False,
                "fallback": "missing_weights",
                "detail": "LIAR hybrid weights not found",
                "features": feat,
                "explanation": expl,
                "explanation_source": "template",
                "dataset": "liar",
                "arch": "hybrid",
            }

        if self._liar_bundle is None:
            tokenizer, model, info = load_liar_hybrid(device=str(self._device))
            self._liar_bundle = (tokenizer, model, info)
        tokenizer, model, info = self._liar_bundle
        variant = info["variant"]
        model_dir = info["model_dir"]

        inputs = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        meta = linguistic_batch_for_liar([text], model_dir, variant).to(self._device)
        input_ids = inputs["input_ids"].to(self._device)
        attention_mask = inputs["attention_mask"].to(self._device)
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
            prob_list = probs[0].cpu().tolist()

        if variant == "colab_distil_hybrid":
            feat_dict = linguistic_feature_dict_for_api(text)
        else:
            feat_dict = _linguistic_feature_dict(text)
        expl = build_rich_explanation(
            text, pred, conf, [float(x) for x in prob_list], feat_dict, True, "none"
        )
        return {
            "prediction": pred,
            "label": "Unreliable" if pred == 1 else "Reliable",
            "confidence": round(conf, 4),
            "probabilities": [round(float(p), 4) for p in prob_list],
            "model_loaded": True,
            "fallback": "none",
            "checkpoint_variant": variant,
            "features": feat_dict,
            "explanation": expl,
            "explanation_source": "template",
            "dataset": "liar",
            "arch": "hybrid",
            "model_dir": model_dir,
            "weight_file": info.get("weight_path"),
        }


_GLOBAL_ROUTER: Optional[MultiModelRouter] = None


def get_multi_router() -> MultiModelRouter:
    global _GLOBAL_ROUTER
    if _GLOBAL_ROUTER is None:
        _GLOBAL_ROUTER = MultiModelRouter()
    return _GLOBAL_ROUTER


MODEL_COMBOS: List[Tuple[str, str]] = [
    ("fakeddit", "distilbert"),
    ("fakeddit", "hybrid"),
    ("liar", "distilbert"),
    ("liar", "hybrid"),
]


def predict_all_models(text: str, router: Optional[MultiModelRouter] = None) -> Dict[str, Dict[str, Any]]:
    """Run all four combinations; keys like 'fakeddit:distilbert'."""
    r = router or get_multi_router()
    out: Dict[str, Dict[str, Any]] = {}
    for ds, ar in MODEL_COMBOS:
        key = f"{ds}:{ar}"
        try:
            out[key] = r.predict(text, ds, ar)
        except Exception as e:
            out[key] = {"error": str(e), "dataset": ds, "arch": ar}
    return out
