"""
Load LIAR hybrid checkpoints (Colab zip export or local training folder).

Colab-style keys: distilbert.*, linguistic_projection.*, fusion_bottleneck.*
Repo HybridTransformerModel keys: text_encoder.*, feature_processor.*, classifier.*

Set VERIPULSE_LIAR_MODEL_DIR to override the search directory.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

LIAR_ZIP_INNER = os.path.join(
    PROJECT_ROOT,
    "models",
    "hybrid_liar_model-20260322T181556Z-3-001",
    "hybrid_liar_model",
)
LIAR_ZIP_ROOT = os.path.join(
    PROJECT_ROOT,
    "models",
    "hybrid_liar_model-20260322T181556Z-3-001",
)

MODEL_SEARCH_DIRS = [
    os.environ.get("VERIPULSE_LIAR_MODEL_DIR", "").strip(),
    LIAR_ZIP_INNER,
    LIAR_ZIP_ROOT,
    os.path.join(PROJECT_ROOT, "models", "hybrid_liar_trained"),
    os.path.join(PROJECT_ROOT, "models", "advanced_hybrid_liar_trained"),
]

WEIGHT_NAMES = (
    "hybrid_liar_model.bin",
    "hybrid_model.bin",
    "advanced_hybrid_model.bin",
    "model.bin",
    "pytorch_model.bin",
)

BASE_TOKENIZER = "distilbert-base-uncased"


def find_liar_hybrid_checkpoint() -> Tuple[Optional[str], Optional[str]]:
    for d in MODEL_SEARCH_DIRS:
        if not d or not os.path.isdir(d):
            continue
        for name in WEIGHT_NAMES:
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return d, p
    return None, None


def _load_state(path: str) -> dict:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return state


def load_liar_hybrid(device: str = "cpu"):
    """
    Returns tokenizer, model, info dict with variant, model_dir, weight_path.
    """
    try:
        from src.fakeddit_colab_hybrid_model import (
            ColabDistilHybridFakeddit,
            is_colab_distil_hybrid_state_dict,
        )
        from src.models.hybrid_transformer import HybridTransformerModel
    except ImportError:
        from fakeddit_colab_hybrid_model import (
            ColabDistilHybridFakeddit,
            is_colab_distil_hybrid_state_dict,
        )
        from models.hybrid_transformer import HybridTransformerModel

    model_dir, weight_path = find_liar_hybrid_checkpoint()
    if not weight_path or not model_dir:
        raise FileNotFoundError(
            "No LIAR hybrid weights found. Copy e.g. hybrid_liar_model.bin into "
            f"{LIAR_ZIP_INNER} or set VERIPULSE_LIAR_MODEL_DIR."
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(BASE_TOKENIZER)

    state = _load_state(weight_path)

    if is_colab_distil_hybrid_state_dict(state):
        model = ColabDistilHybridFakeddit()
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError:
            model.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()
        info: Dict[str, Any] = {
            "model_dir": model_dir,
            "weight_path": weight_path,
            "variant": "colab_distil_hybrid",
        }
        return tokenizer, model, info

    model = HybridTransformerModel(model_name=BASE_TOKENIZER, mode="full")
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return tokenizer, model, {
        "model_dir": model_dir,
        "weight_path": weight_path,
        "variant": "hybrid_transformer_repo",
    }


def linguistic_batch_for_liar(texts: List[str], model_dir: str, variant: str):
    """Return linguistic feature tensor appropriate for variant."""
    if variant == "colab_distil_hybrid":
        try:
            from src.fakeddit_colab_hybrid_model import colab_linguistic_features
        except ImportError:
            from fakeddit_colab_hybrid_model import colab_linguistic_features

        return colab_linguistic_features(texts, model_dir)
    try:
        from src.models.hybrid_transformer import HybridTransformerModel
    except ImportError:
        from models.hybrid_transformer import HybridTransformerModel

    return HybridTransformerModel.extract_linguistic_features(texts)
