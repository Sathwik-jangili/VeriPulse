#!/usr/bin/env python3
"""
Build a roughly balanced live CSV (reliable vs unreliable) from Reddit + Mastodon.

IMPORTANT — Labels are PROXIES for testing only, not ground truth:
  label=0 (reliable proxy): mainstream-style subreddits / neutral-ish news tags on Mastodon.
  label=1 (unreliable proxy): e.g. satire (r/TheOnion) and conspiracy-style sources/tags.
  Satire is not the same as “harmful misinformation”; document this in your report.

Same text hygiene as the live feed by default (1–2 sentences, English, no URLs, letters required).
Use --no-text-filters if you need more rows.

Run from project root:
  python scripts/collect_balanced_live_mix.py
  python scripts/collect_balanced_live_mix.py --per-class 400 --output data_processed/live_balanced_proxy_800.csv

Environment: MASTODON_ACCESS_TOKEN, MASTODON_INSTANCES (optional, same as backend).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import re
import sys
import time
from datetime import datetime, timezone

import requests

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.live_data.mastodon_fetcher import _instance_list  # noqa: E402
from backend.live_data.text_filters import (  # noqa: E402
    contains_url,
    lacks_english_letters,
    passes_live_post_filters,
)

# Mainstream / factual-style (proxy label 0)
REDDIT_RELIABLE = ["news", "worldnews", "science", "upliftingnews", "technology", "space"]

# Satire + conspiracy-adjacent (proxy label 1) — document limitations in write-up
REDDIT_UNRELIABLE_PROXY = [
    "TheOnion",
    "conspiracy",
    "HighStrangeness",
    "UFOs",
    "flatearth",
]

# Hashtag timelines (no leading # in API). May be sparse on some instances.
MASTODON_TAGS_RELIABLE = ["science", "breakingnews", "worldnews", "climate", "health"]
MASTODON_TAGS_UNRELIABLE = ["conspiracy", "qanon", "misinformation", "chemtrails", "deepstate"]

REDDIT_HEADERS = {
    "User-Agent": "VeriPulse/1.0 (Academic research; balanced live test collection)",
}
HTML_TAG_RE = re.compile(r"<.*?>")
MASTODON_HEADERS = {
    "User-Agent": REDDIT_HEADERS["User-Agent"],
    "Accept": "application/json",
}


def _norm_key(text: str) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def _relaxed_ok(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 10 or len(t) > 2000:
        return False
    if lacks_english_letters(t):
        return False
    if contains_url(t):
        return False
    return True


def _text_ok(text: str, use_filters: bool, min_s: int, max_s: int) -> bool:
    if use_filters:
        return passes_live_post_filters(text, min_sentences=min_s, max_sentences=max_s)
    return _relaxed_ok(text)


def reddit_collect_for_label(
    subs: list[str],
    label: int,
    target: int,
    seen: set[str],
    *,
    proxy_note: str,
    use_filters: bool,
    min_s: int,
    max_s: int,
    sleep_s: float,
    max_pages_per_sub: int,
) -> list[dict]:
    out: list[dict] = []
    subs_shuffled = list(subs)
    random.shuffle(subs_shuffled)
    for sub in subs_shuffled:
        if len(out) >= target:
            break
        after = None
        for _ in range(max_pages_per_sub):
            if len(out) >= target:
                break
            params: dict = {"limit": 100, "raw_json": 1}
            if after:
                params["after"] = after
            try:
                r = requests.get(
                    f"https://old.reddit.com/r/{sub}/new.json",
                    headers=REDDIT_HEADERS,
                    params=params,
                    timeout=20,
                )
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                print(f"  [reddit r/{sub}] {e}")
                break
            children = data.get("data", {}).get("children", [])
            if not children:
                break
            after = data.get("data", {}).get("after")
            for child in children:
                if len(out) >= target:
                    break
                pd = child.get("data", {}) if isinstance(child, dict) else {}
                title = (pd.get("title") or "").strip()
                body = (pd.get("selftext") or "").strip()
                combined = f"{title}\n{body}".strip()
                if not combined or not _text_ok(combined, use_filters, min_s, max_s):
                    continue
                k = _norm_key(combined)
                if k in seen:
                    continue
                seen.add(k)
                out.append(
                    {
                        "text": combined,
                        "label": label,
                        "source": "reddit",
                        "origin": f"r/{sub}",
                        "proxy_note": proxy_note,
                        "collected_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
            if not after:
                break
            time.sleep(sleep_s)
        time.sleep(sleep_s)
    return out


def mastodon_tag_collect(
    tags: list[str],
    label: int,
    target: int,
    seen: set[str],
    *,
    proxy_note: str,
    use_filters: bool,
    min_s: int,
    max_s: int,
    sleep_s: float,
    max_pages_per_tag: int,
) -> list[dict]:
    token = os.environ.get("MASTODON_ACCESS_TOKEN", "").strip()
    headers = dict(MASTODON_HEADERS)
    if token:
        headers["Authorization"] = f"Bearer {token}"

    out: list[dict] = []
    tags_shuffled = list(tags)
    random.shuffle(tags_shuffled)

    for base in _instance_list():
        if len(out) >= target:
            break
        for tag in tags_shuffled:
            if len(out) >= target:
                break
            max_id = None
            for _ in range(max_pages_per_tag):
                if len(out) >= target:
                    break
                params: dict = {"limit": 40}
                if max_id is not None:
                    params["max_id"] = max_id
                try:
                    url = f"{base.rstrip('/')}/api/v1/timelines/tag/{tag}"
                    r = requests.get(url, headers=headers, params=params, timeout=25)
                    if r.status_code == 404:
                        break
                    if r.status_code in (401, 403):
                        print(f"  [mastodon #{tag} @ {base}] HTTP {r.status_code}")
                        break
                    r.raise_for_status()
                    items = r.json()
                except Exception as e:
                    print(f"  [mastodon #{tag} @ {base}] {e}")
                    break
                if not isinstance(items, list) or not items:
                    break
                for status in items:
                    if len(out) >= target:
                        break
                    if not isinstance(status, dict):
                        continue
                    raw = status.get("content") or ""
                    clean = HTML_TAG_RE.sub("", raw).strip()
                    if not clean:
                        reblog = status.get("reblog") or {}
                        if isinstance(reblog, dict):
                            c2 = reblog.get("content") or ""
                            clean = HTML_TAG_RE.sub("", c2).strip()
                    if not clean or not _text_ok(clean, use_filters, min_s, max_s):
                        continue
                    k = _norm_key(clean)
                    if k in seen:
                        continue
                    seen.add(k)
                    host = base.replace("https://", "").replace("http://", "").split("/")[0]
                    out.append(
                        {
                            "text": clean,
                            "label": label,
                            "source": "mastodon",
                            "origin": f"{host}#{tag}",
                            "proxy_note": proxy_note,
                            "collected_at": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                max_id = items[-1].get("id")
                time.sleep(sleep_s)
            time.sleep(sleep_s)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--per-class", type=int, default=500, help="Target rows per label (0 and 1)")
    p.add_argument(
        "--output",
        default=os.path.join(PROJECT_ROOT, "data_processed", "live_balanced_proxy_labeled.csv"),
    )
    p.add_argument("--reddit-fraction", type=float, default=0.5, help="Fraction of EACH class from Reddit (0-1)")
    p.add_argument("--min-sentences", type=int, default=1)
    p.add_argument("--max-sentences", type=int, default=2)
    p.add_argument("--no-text-filters", action="store_true")
    p.add_argument("--sleep", type=float, default=1.2)
    p.add_argument("--max-pages-reddit", type=int, default=35)
    p.add_argument("--max-pages-mastodon-tag", type=int, default=50)
    args = p.parse_args()

    if not 0 <= args.reddit_fraction <= 1:
        print("reddit-fraction must be in [0,1]")
        sys.exit(2)

    per = max(1, int(args.per_class))
    n_r = int(round(per * args.reddit_fraction))
    use_filters = not args.no_text_filters
    seen: set[str] = set()

    print(f"Target per class: {per} (first pass reddit target per class: {n_r})")
    print("Collecting label=0 (mainstream proxy) ...")
    zeros = reddit_collect_for_label(
        REDDIT_RELIABLE,
        0,
        n_r,
        seen,
        proxy_note="reddit_mainstream_subs_proxy_0",
        use_filters=use_filters,
        min_s=args.min_sentences,
        max_s=args.max_sentences,
        sleep_s=args.sleep,
        max_pages_per_sub=args.max_pages_reddit,
    )
    if len(zeros) < per:
        need = per - len(zeros)
        print(f"  reddit got {len(zeros)}; fetching up to {need} more from Mastodon ...")
        zeros.extend(
            mastodon_tag_collect(
                MASTODON_TAGS_RELIABLE,
                0,
                need,
                seen,
                proxy_note="mastodon_newsish_tags_proxy_0",
                use_filters=use_filters,
                min_s=args.min_sentences,
                max_s=args.max_sentences,
                sleep_s=args.sleep,
                max_pages_per_tag=args.max_pages_mastodon_tag,
            )
        )
    zeros = zeros[:per]
    print(f"  label=0 count: {len(zeros)} (target {per})")

    print("Collecting label=1 (satire/conspiracy proxy) ...")
    ones = reddit_collect_for_label(
        REDDIT_UNRELIABLE_PROXY,
        1,
        n_r,
        seen,
        proxy_note="reddit_satire_conspiracy_proxy_1",
        use_filters=use_filters,
        min_s=args.min_sentences,
        max_s=args.max_sentences,
        sleep_s=args.sleep,
        max_pages_per_sub=args.max_pages_reddit,
    )
    if len(ones) < per:
        need = per - len(ones)
        print(f"  reddit got {len(ones)}; fetching up to {need} more from Mastodon ...")
        ones.extend(
            mastodon_tag_collect(
                MASTODON_TAGS_UNRELIABLE,
                1,
                need,
                seen,
                proxy_note="mastodon_conspiracy_tags_proxy_1",
                use_filters=use_filters,
                min_s=args.min_sentences,
                max_s=args.max_sentences,
                sleep_s=args.sleep,
                max_pages_per_tag=args.max_pages_mastodon_tag,
            )
        )
    ones = ones[:per]
    print(f"  label=1 count: {len(ones)} (target {per})")

    final = zeros + ones
    random.shuffle(final)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    fields = ["text", "label", "source", "origin", "proxy_note", "collected_at"]
    with open(args.output, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in final:
            w.writerow({k: r[k] for k in fields})

    n0 = sum(1 for r in final if r["label"] == 0)
    n1 = sum(1 for r in final if r["label"] == 1)
    print(f"Wrote {len(final)} rows to {args.output} (label=0: {n0}, label=1: {n1})")
    if n0 < per or n1 < per:
        print("Warning: below target for at least one class — try --no-text-filters or increase --max-pages-*")


if __name__ == "__main__":
    main()
