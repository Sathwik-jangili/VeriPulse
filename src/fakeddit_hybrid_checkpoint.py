"""
Find and load Advanced Hybrid Fakeddit checkpoints (60k / Colab export).

Used by test_fakeddit_three_step.py, test_advanced_hybrid_fakeddit.py, and mirrors
src/veripulse_predictor.py search order.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

ZIP_ROOT = os.path.join(
    PROJECT_ROOT,
    "models",
    "advanced_hybrid_fakeddit_60k-20260322T162137Z-3-001",
)

MODEL_SEARCH_DIRS = [
    os.environ.get("VERIPULSE_MODEL_DIR", "").strip(),
    os.path.join(ZIP_ROOT, "advanced_hybrid_fakeddit_60k"),
    ZIP_ROOT,
    os.path.join(PROJECT_ROOT, "models", "advanced_hybrid_fakeddit_60k"),
    os.path.join(PROJECT_ROOT, "models", "advanced_hybrid_fakeddit_trained"),
    os.path.join(
        PROJECT_ROOT,
        "models",
        "fakeddit_hybrid_model",
        "advanced_hybrid_fakeddit_trained",
    ),
]

WEIGHT_NAMES = (
    "advanced_hybrid_model.bin",
    "hybrid_fakeddit_model.bin",
    "fakeddit_model.bin",
    "model.bin",
    "pytorch_model.bin",
)

BASE_TOKENIZER = "distilbert-base-uncased"


def find_fakeddit_hybrid_checkpoint() -> Tuple[Optional[str], Optional[str]]:
    """Return (model_dir, weight_path) or (None, None)."""
    for d in MODEL_SEARCH_DIRS:
        if not d or not os.path.isdir(d):
            continue
        for name in WEIGHT_NAMES:
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return d, p
    return None, None


def read_class_weights_json(model_dir: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(model_dir, "class_weights.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _load_state_dict_raw(weight_path: str) -> dict:
    try:
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(weight_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return state


def load_advanced_hybrid_fakeddit(device: str = "cpu"):
    """
    Load tokenizer + hybrid model from the first matching checkpoint.

    Colab export (hybrid_fakeddit_model.bin): distilbert + linguistic_projection + fusion_bottleneck.
    Legacy repo class: AdvancedHybridTransformerModel (text_encoder + feature_processor + fusion_layer).

    Returns:
        tokenizer, model, info dict with keys: model_dir, weight_path, variant, ...
    """
    try:
        from src.advanced_hybrid_transformer import AdvancedHybridTransformerModel
        from src.fakeddit_colab_hybrid_model import (
            ColabDistilHybridFakeddit,
            is_colab_distil_hybrid_state_dict,
        )
    except ImportError:
        from advanced_hybrid_transformer import AdvancedHybridTransformerModel
        from fakeddit_colab_hybrid_model import (
            ColabDistilHybridFakeddit,
            is_colab_distil_hybrid_state_dict,
        )

    model_dir, weight_path = find_fakeddit_hybrid_checkpoint()
    if not weight_path or not model_dir:
        raise FileNotFoundError(
            "No Fakeddit hybrid .bin found. Set VERIPULSE_MODEL_DIR or add "
            "hybrid_fakeddit_model.bin under models/.../advanced_hybrid_fakeddit_60k/"
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(BASE_TOKENIZER)

    state = _load_state_dict_raw(weight_path)

    if is_colab_distil_hybrid_state_dict(state):
        model = ColabDistilHybridFakeddit()
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError:
            model.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()
        info = {
            "model_dir": model_dir,
            "weight_path": weight_path,
            "variant": "colab_distil_hybrid",
            "fusion_text_only": False,
            "class_weights": read_class_weights_json(model_dir),
        }
        return tokenizer, model, info

    fusion_text_only = AdvancedHybridTransformerModel.infer_fusion_text_only_from_state_dict(
        state
    )
    model = AdvancedHybridTransformerModel(mode="full", fusion_text_only=fusion_text_only)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    info = {
        "model_dir": model_dir,
        "weight_path": weight_path,
        "variant": "advanced_hybrid_legacy",
        "fusion_text_only": fusion_text_only,
        "class_weights": read_class_weights_json(model_dir),
    }
    return tokenizer, model, info
