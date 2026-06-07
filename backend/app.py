from flask import Flask, jsonify, request
from flask_cors import CORS
import sys
import os

# Add the project root to sys.path to import from backend.live_data and src
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from backend.live_data.reddit_fetcher import fetch_reddit_posts
from backend.live_data.mastodon_fetcher import fetch_mastodon_posts
from backend.database import dashboard_summary, persist_live_post_with_models
from src.veripulse_predictor import get_predictor
from src.multi_model_router import get_multi_router, predict_all_models

app = Flask(__name__)
# Allow browser requests from Vite (any port), file preview, or production origin
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    supports_credentials=False,
)


def _primary_from_all_outputs(all_out: dict) -> dict:
    """Prefer fakeddit hybrid for display; else first valid."""
    for key in ("fakeddit:hybrid", "fakeddit:distilbert", "liar:hybrid", "liar:distilbert"):
        o = all_out.get(key)
        if o and not o.get("error") and "prediction" in o:
            return dict(o)
    return {
        "prediction": 0,
        "label": "Reliable",
        "confidence": 0.5,
        "explanation": "No model produced a score.",
        "model_loaded": False,
    }


@app.route("/health", methods=["GET"])
def health():
    st = get_predictor().status()
    return jsonify(
        {
            "ok": True,
            "model_loaded": st.model_loaded,
            "model_dir": st.model_dir,
            "weight_file": st.weight_file,
            "detail": st.message,
        }
    )


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    """
    JSON body:
      { "text": "...", "dataset": "fakeddit"|"liar", "arch": "distilbert"|"hybrid",
        "persist": true }
    Defaults: fakeddit + hybrid (VeriPulse advanced hybrid). persist saves all 4 model scores.
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        text = (data.get("text") or "").strip()
        dataset = (data.get("dataset") or "fakeddit").strip().lower()
        arch = (data.get("arch") or "hybrid").strip().lower()
        persist = bool(data.get("persist"))

        if not text:
            return jsonify(
                {
                    "prediction": 0,
                    "label": "Reliable",
                    "confidence": 0.0,
                    "probabilities": [0.5, 0.5],
                    "model_loaded": False,
                    "fallback": "empty",
                    "detail": "Empty text",
                    "explanation": "No text was entered, so no content could be scored.",
                    "explanation_source": "n/a",
                    "dataset": dataset,
                    "arch": arch,
                }
            )

        router = get_multi_router()
        out = router.predict(text, dataset, arch)

        if persist and "error" not in out:
            try:
                all_o = predict_all_models(text, router)
                persist_live_post_with_models("Web", text, "analyze_tab", all_o)
            except Exception as ex:
                out["persist_warning"] = str(ex)

        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/dashboard/summary", methods=["GET"])
def dashboard_summary_endpoint():
    try:
        return jsonify(dashboard_summary())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/model/status", methods=["GET"])
def model_status():
    st = get_predictor().load()
    return jsonify(
        {
            "model_loaded": st.model_loaded,
            "model_dir": st.model_dir,
            "weight_file": st.weight_file,
            "message": st.message,
        }
    )


def _live_limit():
    return request.args.get("limit", default=10, type=int) or 10


def _live_sentence_bounds():
    min_s = request.args.get("min_sentences", default=1, type=int) or 1
    max_s = request.args.get("max_sentences", default=2, type=int) or 2
    min_s = min(max(min_s, 1), 10)
    max_s = min(max(max_s, 1), 10)
    if max_s < min_s:
        min_s, max_s = max_s, min_s
    return min_s, max_s


@app.route("/live/reddit", methods=["GET"])
def get_reddit_live():
    try:
        limit = min(max(_live_limit(), 1), 25)
        min_s, max_s = _live_sentence_bounds()
        posts = fetch_reddit_posts(
            limit=limit, min_sentences=min_s, max_sentences=max_s
        )
        results = []
        router = get_multi_router()
        for post in posts:
            all_out = predict_all_models(post, router)
            try:
                persist_live_post_with_models("Reddit", post, "live_feed", all_out)
            except Exception as ex:
                print(f"DB persist reddit: {ex}")
            primary = _primary_from_all_outputs(all_out)
            results.append(
                {
                    "text": post,
                    "prediction": int(primary.get("prediction", 0)),
                    "confidence": float(primary.get("confidence", 0.5)),
                    "explanation": primary.get("explanation", ""),
                    "source": "Reddit",
                    "dataset": primary.get("dataset", "fakeddit"),
                    "arch": primary.get("arch", "hybrid"),
                }
            )
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/live/mastodon", methods=["GET"])
def get_mastodon_live():
    try:
        limit = min(max(_live_limit(), 1), 40)
        min_s, max_s = _live_sentence_bounds()
        instance = (request.args.get("instance") or "").strip() or None
        posts = fetch_mastodon_posts(
            instance_url=instance,
            limit=limit,
            min_sentences=min_s,
            max_sentences=max_s,
        )
        results = []
        router = get_multi_router()
        for post in posts:
            all_out = predict_all_models(post, router)
            try:
                persist_live_post_with_models("Mastodon", post, "live_feed", all_out)
            except Exception as ex:
                print(f"DB persist mastodon: {ex}")
            primary = _primary_from_all_outputs(all_out)
            results.append(
                {
                    "text": post,
                    "prediction": int(primary.get("prediction", 0)),
                    "confidence": float(primary.get("confidence", 0.5)),
                    "explanation": primary.get("explanation", ""),
                    "source": "Mastodon",
                    "dataset": primary.get("dataset", "fakeddit"),
                    "arch": primary.get("arch", "hybrid"),
                }
            )
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("VeriPulse API listening on http://127.0.0.1:5000  (GET /health, POST /predict)")
    # 127.0.0.1 avoids Windows resolving "localhost" to IPv6 ::1 while proxy targets IPv4
    app.run(debug=True, host="127.0.0.1", port=5000, use_reloader=False)
