"""Shared helpers for live-feed text: length, English, no URLs, no numbers-only."""

from __future__ import annotations

import re
# Split after . ? ! when followed by whitespace (naive; good enough for feed filtering)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

# http(s), www., markdown [label](url)
_URL_PATTERN = re.compile(
    r"https?://[^\s\]\)]+|www\.[^\s\]\)]+|\[[^\]]+\]\([^)]+\)",
    re.IGNORECASE,
)

# Non-Latin scripts (reject for "English only")
_NON_LATIN_SCRIPTS = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF"  # Arabic
    r"\u0400-\u04FF\u0500-\u052F"  # Cyrillic
    r"\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF"  # JP/CJK
    r"\u0E00-\u0E7F"  # Thai
    r"\u0590-\u05FF"  # Hebrew
    r"]"
)


def sentence_count(text: str) -> int:
    """Number of sentences (naive; title-only counts as one sentence)."""
    t = (text or "").strip()
    if not t:
        return 0
    parts = [p for p in _SENT_SPLIT.split(t) if p.strip()]
    return len(parts)


def contains_url(text: str) -> bool:
    """True if text has a URL (http(s), www., or markdown link)."""
    return bool(_URL_PATTERN.search(text or ""))


def lacks_english_letters(text: str) -> bool:
    """True if there are no Latin letters (digits-only, emoji-only, etc.)."""
    return not bool(re.search(r"[a-zA-Z]", text or ""))


def is_probably_english(text: str, min_lang_prob: float = 0.85) -> bool:
    """
    Detect English. Short snippets use a conservative ASCII / script heuristic.
    Uses langdetect when available and text is long enough.
    """
    sample = (text or "").strip()
    if not sample:
        return False
    if _NON_LATIN_SCRIPTS.search(sample):
        return False

    if len(sample) < 20:
        return _short_snippet_english_heuristic(sample)

    try:
        from langdetect import LangDetectException, detect_langs
    except ImportError:
        return _short_snippet_english_heuristic(sample)

    try:
        scores = detect_langs(sample)
        if not scores:
            return _short_snippet_english_heuristic(sample)
        best = scores[0]
        return best.lang == "en" and best.prob >= min_lang_prob
    except LangDetectException:
        return _short_snippet_english_heuristic(sample)


def _short_snippet_english_heuristic(text: str) -> bool:
    """For very short strings: Latin letters, mostly printable ASCII, no obvious non-English scripts."""
    if lacks_english_letters(text):
        return False
    if _NON_LATIN_SCRIPTS.search(text):
        return False
    # Allow common punctuation and digits; block high ratio of non-ASCII letters (e.g. accented words ok)
    ascii_printable_ratio = sum(1 for c in text if 32 <= ord(c) < 127 or c in "\n\t") / max(len(text), 1)
    return ascii_printable_ratio >= 0.75


# Down-rank conflict / breaking-war posts for feeds meant to resemble general Fakeddit-style text
_WAR_TOPIC = re.compile(
    r"\b(ukraine|ukrainian|ukrain|kiev|kyiv|russia|russian|putin|zelensky|kremlin|"
    r"donbas|crimea|nato|mobilization|mobilisation|"
    r"gaza|gazan|hamas|idf|israel defense|west bank|rafah|"
    r"ceasefire|artillery|howitzer|drone strike|missile barrage|"
    r"war crime|war crimes|invasion of|frontline|front line|battlefield|troops in)\b",
    re.IGNORECASE,
)


def is_war_or_conflict_heavy(text: str) -> bool:
    return bool(_WAR_TOPIC.search(text or ""))


def passes_live_post_filters(
    text: str,
    min_sentences: int = 1,
    max_sentences: int = 2,
) -> bool:
    """
    Live feed rules:
    - Between min and max sentences (inclusive).
    - No URL-like substrings.
    - Not numbers/symbols-only (must contain at least one letter).
    - Predominantly English.
    """
    t = (text or "").strip()
    if not t:
        return False
    if contains_url(t):
        return False
    if lacks_english_letters(t):
        return False

    n = sentence_count(t)
    if n < min_sentences or n > max_sentences:
        return False

    if not is_probably_english(t):
        return False

    return True


def within_max_sentences(text: str, max_sentences: int = 2) -> bool:
    """Backward-compatible: 1..max sentences, no other checks."""
    n = sentence_count(text)
    return 0 < n <= max_sentences
