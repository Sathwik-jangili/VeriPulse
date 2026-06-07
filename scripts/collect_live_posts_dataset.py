#!/usr/bin/env python3
"""
Collect Reddit + Mastodon posts into a CSV for external labeling (e.g. Gemini) and later model testing.

Uses the same text rules as the live feed by default (1-2 sentences, English, no URLs, letters required).
Paginates Reddit (after=...) and Mastodon (max_id=...) until targets are met or sources exhaust.

Run from project root:
  python scripts/collect_live_posts_dataset.py
  python scripts/collect_live_posts_dataset.py --reddit 600 --mastodon 400 --output data_processed/my_live_set.csv
  python scripts/collect_live_posts_dataset.py --no-text-filters --reddit 200
    (looser: only dedupe, non-empty, strip HTML; still skips obvious http/www for CSV cleanliness)

Environment (Mastodon): MASTODON_ACCESS_TOKEN, MASTODON_INSTANCES — same as backend.
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
from backend.live_data.reddit_fetcher import SUBREDDITS  # noqa: E402
from backend.live_data.text_filters import (  # noqa: E402
    contains_url,
    lacks_english_letters,
    passes_live_post_filters,
)

REDDIT_HEADERS = {
    "User-Agent": "VeriPulse/1.0 (Academic research; dataset collection; contact: student project)",
}
HTML_TAG_RE = re.compile(r"<.*?>")
MASTODON_HEADERS = {
    "User-Agent": REDDIT_HEADERS["User-Agent"],
    "Accept": "application/json",
}


def _norm_dedupe_key(text: str) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def _relaxed_ok(text: str) -> bool:
    """Minimal cleanup row: not empty, has letters, no URL, reasonable length."""
    t = (text or "").strip()
    if len(t) < 10 or len(t) > 2000:
        return False
    if lacks_english_letters(t):
        return False
    if contains_url(t):
        return False
    return True


def collect_reddit(
    target: int,
    *,
    min_sentences: int,
    max_sentences: int,
    use_filters: bool,
    sleep_s: float,
    max_pages_per_sub: int,
) -> list[dict]:
    if target <= 0:
        return []
    out: list[dict] = []
    seen: set[str] = set()
    subs = list(SUBREDDITS)
    random.shuffle(subs)

    for sub in subs:
        if len(out) >= target:
            break
        after: str | None = None
        for _page in range(max_pages_per_sub):
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
                print(f"  [reddit r/{sub}] page error: {e}")
                break

            children = data.get("data", {}).get("children", [])
            if not children:
                break
            after = data.get("data", {}).get("after")

            for child in children:
                if len(out) >= target:
                    break
                pd = child.get("data", {}) if isinstance(child, dict) else {}
                title = pd.get("title", "") or ""
                body = pd.get("selftext", "") or ""
                combined = f"{title}\n{body}".strip()
                if not combined:
                    continue
                ok = (
                    passes_live_post_filters(
                        combined,
                        min_sentences=min_sentences,
                        max_sentences=max_sentences,
                    )
                    if use_filters
                    else _relaxed_ok(combined)
                )
                if not ok:
                    continue
                k = _norm_dedupe_key(combined)
                if k in seen:
                    continue
                seen.add(k)
                out.append(
                    {
                        "source": "reddit",
                        "origin": f"r/{sub}",
                        "text": combined,
                        "collected_at": datetime.now(timezone.utc).isoformat(),
                    }
                )

            if not after:
                break
            time.sleep(sleep_s)

        time.sleep(sleep_s)

    return out[:target]


def collect_mastodon(
    target: int,
    *,
    min_sentences: int,
    max_sentences: int,
    use_filters: bool,
    sleep_s: float,
    max_pages_per_instance: int,
) -> list[dict]:
    if target <= 0:
        return []
    token = os.environ.get("MASTODON_ACCESS_TOKEN", "").strip()
    headers = dict(MASTODON_HEADERS)
    if token:
        headers["Authorization"] = f"Bearer {token}"

    out: list[dict] = []
    seen: set[str] = set()

    for base in _instance_list():
        if len(out) >= target:
            break
        max_id = None
        host = base.replace("https://", "").replace("http://", "").split("/")[0]

        for _page in range(max_pages_per_instance):
            if len(out) >= target:
                break
            params: dict = {"limit": 40}
            if max_id is not None:
                params["max_id"] = max_id
            try:
                r = requests.get(
                    f"{base.rstrip('/')}/api/v1/timelines/public",
                    headers=headers,
                    params=params,
                    timeout=25,
                )
                if r.status_code in (401, 403):
                    print(f"  [mastodon {host}] HTTP {r.status_code} — try MASTODON_ACCESS_TOKEN or another instance")
                    break
                r.raise_for_status()
                items = r.json()
            except Exception as e:
                print(f"  [mastodon {host}] error: {e}")
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
                if not clean:
                    continue
                ok = (
                    passes_live_post_filters(
                        clean,
                        min_sentences=min_sentences,
                        max_sentences=max_sentences,
                    )
                    if use_filters
                    else _relaxed_ok(clean)
                )
                if not ok:
                    continue
                k = _norm_dedupe_key(clean)
                if k in seen:
                    continue
                seen.add(k)
                out.append(
                    {
                        "source": "mastodon",
                        "origin": host,
                        "text": clean,
                        "collected_at": datetime.now(timezone.utc).isoformat(),
                    }
                )

            max_id = items[-1].get("id")
            time.sleep(sleep_s)

        time.sleep(sleep_s)

    return out[:target]


def main() -> None:
    p = argparse.ArgumentParser(description="Collect live posts to CSV for labeling / evaluation.")
    p.add_argument("--reddit", type=int, default=500, help="Target count from Reddit (default 500)")
    p.add_argument("--mastodon", type=int, default=500, help="Target count from Mastodon (default 500)")
    p.add_argument(
        "--output",
        default=os.path.join(PROJECT_ROOT, "data_processed", "live_unlabeled_posts.csv"),
        help="Output CSV path",
    )
    p.add_argument("--min-sentences", type=int, default=1)
    p.add_argument("--max-sentences", type=int, default=2)
    p.add_argument(
        "--no-text-filters",
        action="store_true",
        help="Skip English/sentence filters; only dedupe, length, letters, no URL",
    )
    p.add_argument("--sleep", type=float, default=1.5, help="Seconds between paginated requests")
    p.add_argument("--max-pages-reddit", type=int, default=30, help="Max listing pages per subreddit")
    p.add_argument("--max-pages-mastodon", type=int, default=80, help="Max pages per Mastodon instance")
    args = p.parse_args()

    use_filters = not args.no_text_filters
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    print("Collecting Reddit ...")
    reddit_rows = collect_reddit(
        args.reddit,
        min_sentences=args.min_sentences,
        max_sentences=args.max_sentences,
        use_filters=use_filters,
        sleep_s=args.sleep,
        max_pages_per_sub=args.max_pages_reddit,
    )
    print(f"  -> {len(reddit_rows)} posts")

    print("Collecting Mastodon ...")
    m_rows = collect_mastodon(
        args.mastodon,
        min_sentences=args.min_sentences,
        max_sentences=args.max_sentences,
        use_filters=use_filters,
        sleep_s=args.sleep,
        max_pages_per_instance=args.max_pages_mastodon,
    )
    print(f"  -> {len(m_rows)} posts")

    combined = reddit_rows + m_rows
    # stable sample_id
    fieldnames = [
        "sample_id",
        "source",
        "origin",
        "text",
        "collected_at",
        "label_gemini",
        "label_human",
        "notes",
    ]
    with open(args.output, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for i, row in enumerate(combined, start=1):
            row_out = {
                "sample_id": i,
                "source": row["source"],
                "origin": row["origin"],
                "text": row["text"],
                "collected_at": row["collected_at"],
                "label_gemini": "",
                "label_human": "",
                "notes": "",
            }
            w.writerow(row_out)

    print(f"Wrote {len(combined)} rows to {args.output}")
    if len(combined) < args.reddit + args.mastodon:
        print(
            "Warning: fewer rows than requested — relax filters (--no-text-filters), "
            "increase --max-pages-*, or fix Mastodon auth (MASTODON_ACCESS_TOKEN)."
        )


if __name__ == "__main__":
    main()
