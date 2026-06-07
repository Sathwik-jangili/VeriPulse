import random
import requests

try:
    from .text_filters import passes_live_post_filters, is_war_or_conflict_heavy
except ImportError:
    from text_filters import passes_live_post_filters, is_war_or_conflict_heavy

# General-interest subs closer to Fakeddit-style headlines (avoid news/worldnews war domination)
FAKEDDIT_STYLE_SUBREDDITS = [
    "todayilearned",
    "science",
    "technology",
    "mildlyinteresting",
    "nottheonion",
    "explainlikeimfive",
    "UpliftingNews",
    "space",
    "LifeProTips",
    "askscience",
    "Showerthoughts",
    "InternetIsBeautiful",
]

# Legacy name kept for imports; prefer Fakeddit-style rotation
SUBREDDITS = FAKEDDIT_STYLE_SUBREDDITS


def fetch_reddit_posts(
    subreddit=None,
    limit=10,
    min_sentences=1,
    max_sentences=2,
    fetch_cap=100,
    skip_war_heavy=True,
):
    """
    Fetches recent posts from one or more subreddits (Old Reddit JSON).

    Default: rotates through FAKEDDIT_STYLE_SUBREDDITS until `limit` posts pass filters.
    If `subreddit` is set, only that sub is used.

    Filters: sentence count, English, no URLs, letters present.
    Optionally drops conflict/war-heavy titles (similar to general social headline mix).
    """
    want = min(max(int(limit), 1), 25)
    min_s = max(1, int(min_sentences))
    max_s = max(min_s, int(max_sentences))
    per_request_cap = min(max(int(fetch_cap), want), 100)

    headers = {
        "User-Agent": "VeriPulse/1.0 (Academic research; misinformation detection FYP)",
    }

    def pull_from_sub(sub: str) -> list:
        sub = sub.strip().replace("/", "")
        url = f"https://old.reddit.com/r/{sub}/new.json"
        params = {"limit": per_request_cap, "raw_json": 1}
        try:
            response = requests.get(url, headers=headers, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Error fetching Reddit r/{sub}: {e}")
            return []

        out = []
        for post in data.get("data", {}).get("children", []):
            post_data = post.get("data", {})
            title = post_data.get("title", "")
            selftext = post_data.get("selftext", "")
            combined_text = f"{title}\n{selftext}".strip()
            if not combined_text:
                continue
            if skip_war_heavy and is_war_or_conflict_heavy(combined_text):
                continue
            if passes_live_post_filters(
                combined_text, min_sentences=min_s, max_sentences=max_s
            ):
                out.append(combined_text)
        return out

    posts: list = []
    seen: set = set()

    def add_unique(t: str) -> bool:
        key = t[:240].strip().lower()
        if key in seen:
            return False
        seen.add(key)
        posts.append(t)
        return True

    if subreddit:
        for t in pull_from_sub(subreddit):
            add_unique(t)
            if len(posts) >= want:
                break
        return posts[:want]

    subs = list(FAKEDDIT_STYLE_SUBREDDITS)
    random.shuffle(subs)
    for sub in subs:
        if len(posts) >= want:
            break
        for t in pull_from_sub(sub):
            add_unique(t)
            if len(posts) >= want:
                break
    return posts[:want]


if __name__ == "__main__":
    sample_posts = fetch_reddit_posts()
    for i, post in enumerate(sample_posts):
        print(f"Post {i+1}: {post[:100]}...")
