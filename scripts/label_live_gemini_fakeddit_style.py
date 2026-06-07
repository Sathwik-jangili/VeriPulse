#!/usr/bin/env python3
"""
Label live posts with Gemini using the same binary scheme as Fakeddit test CSV:
  0 = reliable (factual, benign, clear satire, legitimate science/news framing)
  1 = unreliable (misleading, false or unverified claims as fact, manipulative
      conspiracy framing, hoax-style headlines, etc.)

Requires: pip install google-generativeai
Auth:     set GEMINI_API_KEY or GOOGLE_API_KEY (never commit keys).

Free tier has a low daily request cap; use a large --batch-size (e.g. 100) so the
full 1000 rows need only ~10 calls. If you hit 429, run again with --resume.

Usage:
  $env:GEMINI_API_KEY='...'   # PowerShell
  python scripts/label_live_gemini_fakeddit_style.py --batch-size 100 --resume
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
except ImportError:
    pass

try:
    import google.generativeai as genai
except ImportError:
    print("Install: pip install google-generativeai", file=sys.stderr)
    sys.exit(2)
DEFAULT_INPUT = os.path.join(PROJECT_ROOT, "data_processed", "live_balanced_proxy_labeled.csv")
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "data_processed", "live_1k_gemini_labeled.csv")
FAKEDDIT_TEST = os.path.join(PROJECT_ROOT, "data_processed", "fakeddit_test.csv")


def _few_shot_block() -> str:
    df = pd.read_csv(FAKEDDIT_TEST)
    z = df[df["label"] == 0].sample(4, random_state=42)
    u = df[df["label"] == 1].sample(4, random_state=42)
    lines = ["### Fakeddit-style reference examples\n"]
    for _, r in z.iterrows():
        lines.append(f'0: "{str(r["text"])[:160]}"')
    for _, r in u.iterrows():
        lines.append(f'1: "{str(r["text"])[:160]}"')
    return "\n".join(lines)


def _parse_json_array(text: str) -> list[dict]:
    text = text.strip()
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        text = m.group(0)
    return json.loads(text)


def _retry_seconds(err: str) -> float | None:
    m = re.search(r"retry in ([0-9.]+)\s*s", err, re.I)
    if m:
        return float(m.group(1)) + 2.0
    m2 = re.search(r"seconds:\s*(\d+)", err)
    if m2:
        return float(m2.group(1)) + 2.0
    return None


def _is_daily_quota_exhausted(err: str) -> bool:
    """True when the API says the per-day request budget for this model is gone."""
    return "GenerateRequestsPerDayPerProjectPerModel" in err


def _load_checkpoint(path: str) -> dict[int, int]:
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): int(v) for k, v in data.get("labels", {}).items()}


def _save_checkpoint(path: str, labels: dict[int, int], meta: dict) -> None:
    payload = {"labels": {str(k): v for k, v in sorted(labels.items())}, **meta}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUT)
    parser.add_argument("--checkpoint", default="", help="Default: <output>.checkpoint.json")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds between batches")
    parser.add_argument("--limit", type=int, default=0, help="Max rows (0 = all)")
    parser.add_argument("--resume", action="store_true", help="Continue from checkpoint")
    parser.add_argument("--text-max-chars", type=int, default=500)
    args = parser.parse_args()

    ckpt = args.checkpoint or (args.output + ".checkpoint.json")

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY or GOOGLE_API_KEY in the environment.", file=sys.stderr)
        sys.exit(2)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        args.model,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.2,
        ),
    )

    df = pd.read_csv(args.input)
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    n = len(df)
    labels: dict[int, int] = {}
    if args.resume or os.path.isfile(ckpt):
        labels = _load_checkpoint(ckpt)
        print(f"Resume: loaded {len(labels)} labels from {ckpt}")

    few = _few_shot_block()

    for start in range(0, n, args.batch_size):
        end = min(start + args.batch_size, n)
        need = [i for i in range(start, end) if i not in labels]
        if not need:
            print(f"  skip {start}-{end} (already labeled)")
            continue

        batch_items = []
        for i in need:
            t = str(df.iloc[i]["text"])
            t = " ".join(t.split())[: args.text_max_chars]
            batch_items.append({"i": i, "text": t})

        prompt = f"""You label social posts for a binary dataset matching Fakeddit.
{few}

Rules:
- 0 RELIABLE: Straight news, real study summaries, ordinary opinion, clear fiction/book promo, activism with factual framing, war updates, weather data, humor marked /s, discussing conspiracy theories without asserting them as true.
- 1 UNRELIABLE: False or unverified claims presented as fact, hoaxes, satire as real news, dangerous health misinfo, election-fraud-as-fact, disinfo tropes, obvious fabrications.

Return a JSON array only. Each element: {{"i": <integer index>, "label": 0 or 1}}

Posts to label:
{json.dumps(batch_items, ensure_ascii=False)}
"""

        max_attempts = 12
        ok = False
        for attempt in range(max_attempts):
            try:
                resp = model.generate_content(prompt)
                raw = (resp.text or "").strip()
                arr = _parse_json_array(raw)
                for obj in arr:
                    idx = int(obj["i"])
                    lab = int(obj["label"])
                    if lab not in (0, 1):
                        raise ValueError(f"bad label {lab}")
                    labels[idx] = lab
                missing = [i for i in need if i not in labels]
                if missing:
                    raise ValueError(f"missing indices {missing}")
                ok = True
                break
            except Exception as e:
                err = str(e)
                if "429" in err or "quota" in err.lower() or "resource" in err.lower():
                    if _is_daily_quota_exhausted(err):
                        _save_checkpoint(
                            ckpt,
                            labels,
                            {"input": os.path.abspath(args.input), "model": args.model},
                        )
                        print(
                            "Daily request quota for this model is used up (free tier).\n"
                            "Checkpoint saved. Options:\n"
                            "  • Tomorrow:  python scripts/label_live_gemini_fakeddit_style.py --resume "
                            f"--output {args.output}\n"
                            "  • Or enable billing in Google AI Studio / Cloud for higher limits.\n"
                            "  • Or try another model:  --model gemini-flash-latest",
                            file=sys.stderr,
                        )
                        sys.exit(4)
                    wait = _retry_seconds(err) or 35.0
                    print(
                        f"  rate limit (not daily): sleep {wait:.0f}s "
                        f"({attempt + 1}/{max_attempts})"
                    )
                    time.sleep(min(wait, 120.0))
                    continue
                if attempt >= max_attempts - 1:
                    print(f"Batch {start}-{end} failed: {e}", file=sys.stderr)
                    _save_checkpoint(
                        ckpt,
                        labels,
                        {"input": os.path.abspath(args.input), "model": args.model},
                    )
                    sys.exit(3)
                time.sleep(2.0)

        if not ok:
            _save_checkpoint(
                ckpt,
                labels,
                {"input": os.path.abspath(args.input), "model": args.model},
            )
            print(
                "Checkpoint saved. Re-run with --resume when quota allows:\n"
                f"  python scripts/label_live_gemini_fakeddit_style.py --resume --output {args.output}",
                file=sys.stderr,
            )
            sys.exit(3)

        _save_checkpoint(
            ckpt,
            labels,
            {"input": os.path.abspath(args.input), "model": args.model},
        )
        print(f"  labeled through index {end - 1} ({len(labels)}/{n})")
        if end < n and args.sleep > 0:
            time.sleep(args.sleep)

    if len(labels) < n:
        print(f"Incomplete: {len(labels)}/{n} labels", file=sys.stderr)
        sys.exit(3)

    out = df.copy()
    if "label" in out.columns:
        out["proxy_label"] = out["label"]
    out["label"] = [labels[i] for i in range(n)]
    out["labeler"] = f"gemini:{args.model}"

    d = os.path.dirname(args.output) or "."
    os.makedirs(d, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {n} rows -> {args.output}")
    print(out["label"].value_counts().sort_index().to_dict())
    if os.path.isfile(ckpt):
        try:
            os.remove(ckpt)
        except OSError:
            pass


if __name__ == "__main__":
    main()
