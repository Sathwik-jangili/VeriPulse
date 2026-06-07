import os
import re
import requests

try:
    from .text_filters import passes_live_post_filters, is_war_or_conflict_heavy
except ImportError:
    from text_filters import passes_live_post_filters, is_war_or_conflict_heavy

# Many instances throttle or block datacenter IPs; try several. Override with env.
_DEFAULT_INSTANCES = [
    "https://mastodon.social",
    "https://mastodon.online",
    "https://mastodon.world",
    "https://fosstodon.org",
    "https://mstdn.social",
]


def _instance_list() -> list[str]:
    raw = os.environ.get("MASTODON_INSTANCES", "").strip()
    if raw:
        return [u.strip().rstrip("/") for u in raw.split(",") if u.strip()]
    single = os.environ.get("MASTODON_INSTANCE_URL", "").strip().rstrip("/")
    if single:
        return [single]
    return list(_DEFAULT_INSTANCES)


def fetch_mastodon_posts(
    instance_url=None,
    limit=10,
    min_sentences=1,
    max_sentences=2,
):
    """
    Public timeline via Mastodon REST API.

    Same text filters as Reddit: 1-2 sentences (by default), English, no URLs, not numbers-only.

    Requests up to 40 statuses (API max) to allow filtering; may return fewer than `limit`.
    """
    want = min(max(int(limit), 1), 40)
    min_s = max(1, int(min_sentences))
    max_s = max(min_s, int(max_sentences))
    api_limit = min(40, max(want * 5, 20))

    token = os.environ.get("MASTODON_ACCESS_TOKEN", "").strip()

    headers_base = {
        "User-Agent": "VeriPulse/1.0 (Academic research; misinformation detection FYP)",
        "Accept": "application/json",
    }
    if token:
        headers_base["Authorization"] = f"Bearer {token}"

    instances = [instance_url.rstrip("/")] if instance_url else _instance_list()
    html_re = re.compile(r"<.*?>")
    last_error = None

    for base in instances:
        url = f"{base}/api/v1/timelines/public"
        try:
            response = requests.get(
                url,
                headers=headers_base,
                params={"limit": api_limit},
                timeout=20,
            )
            if response.status_code in (401, 403):
                last_error = f"{base}: HTTP {response.status_code}"
                continue
            response.raise_for_status()
            items = response.json()
            if not isinstance(items, list):
                last_error = f"{base}: non-list JSON"
                continue

            posts = []
            for status in items:
                if not isinstance(status, dict):
                    continue
                content = status.get("content") or ""
                clean_text = html_re.sub("", content).strip()
                if not clean_text:
                    reblog = status.get("reblog") or {}
                    if isinstance(reblog, dict):
                        c2 = reblog.get("content") or ""
                        clean_text = html_re.sub("", c2).strip()
                if not clean_text:
                    continue
                if is_war_or_conflict_heavy(clean_text):
                    continue
                if passes_live_post_filters(
                    clean_text, min_sentences=min_s, max_sentences=max_s
                ):
                    posts.append(clean_text)
                if len(posts) >= want:
                    break

            if posts:
                return posts[:want]
            last_error = f"{base}: no posts passed filters (English / 1-2 sentences / no URL)"
        except Exception as e:
            last_error = f"{base}: {e}"
            continue

    if last_error:
        print(f"Error fetching Mastodon posts (all instances failed). Last: {last_error}")
    return []


if __name__ == "__main__":
    sample_posts = fetch_mastodon_posts()
    for i, post in enumerate(sample_posts):
        print(f"Post {i+1}: {post[:100]}...")
