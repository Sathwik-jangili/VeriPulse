"""
Plain-language explanations: varied phrasing per input (deterministic hash).
No percentages in user text. No LLM.
"""

from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Optional

_ATTENTION_LEXICON = (
    "breaking",
    "shocking",
    "unbelievable",
    "urgent",
    "must-see",
    "official",
    "miracle",
    "cure",
    "secret",
    "conspiracy",
    "doctors hate",
    "one weird",
    "aliens",
    "5g",
    "deep state",
    "cover-up",
    "coverup",
    "wake up",
    "sheeple",
    "hoax",
    "destroyed",
    "slams",
)

_ATTRIBUTION_HINTS = (
    "according to",
    "published",
    "study",
    "studies",
    "research",
    "researchers",
    "scientists",
    "journal",
    "peer-reviewed",
    "peer reviewed",
    "reported",
    "evidence suggests",
)


def _find_lexicon_hits(text: str, max_hits: int = 8) -> List[str]:
    t = (text or "").lower()
    return [w for w in _ATTENTION_LEXICON if w in t][:max_hits]


def _word_count(text: str) -> int:
    return len((text or "").split())


def _exclaim_count_text(text: str) -> int:
    return (text or "").count("!")


def _uppercase_ratio_text(text: str) -> float:
    t = text or ""
    letters = [c for c in t if c.isalpha()]
    if not letters:
        return 0.0
    up = sum(1 for c in letters if c.isupper())
    return up / len(letters)


def _variant_seed(text: str, pred: int, confidence: float) -> int:
    raw = (text or "").strip()
    h = hashlib.sha256(
        f"{pred}|{confidence:.5f}|{raw[:2000]}".encode("utf-8", errors="replace")
    ).digest()
    return int.from_bytes(h[:8], "big")


def _pick(seed: int, index: int, options: List[str]) -> str:
    if not options:
        return ""
    x = (seed >> (index * 7)) & 0xFFFFFFFF
    return options[x % len(options)]


def _lexicon_plain(hits: List[str], seed: int) -> str:
    if not hits:
        return ""
    health = {"miracle", "cure", "doctors hate", "one weird"}
    alarm = {"breaking", "shocking", "unbelievable", "urgent", "must-see", "slams", "destroyed"}
    fringe = {"conspiracy", "deep state", "cover-up", "coverup", "hoax", "sheeple", "wake up", "aliens", "5g"}
    bits = []
    if any(h in health for h in hits):
        bits.append("health or remedy hype")
    if any(h in alarm for h in hits):
        bits.append("alarm-style or clickbait wording")
    if any(h in fringe for h in hits):
        bits.append("conspiracy or fringe-topic framing")
    if "secret" in hits or "official" in hits:
        bits.append("secrecy or faux-official hype")
    if not bits:
        bits.append("emotionally loaded wording")

    a = _pick(
        seed,
        1,
        [
            "You can feel {0} in how it is written.",
            "The wording leans into {0}.",
            "It pushes {0}.",
            "The line carries {0}.",
        ],
    )
    b = _pick(seed, 2, [" and ", ", plus ", " together with ", " mixed with "])
    if len(bits) == 1:
        return a.format(bits[0])
    if len(bits) == 2:
        return a.format(bits[0] + b + bits[1])
    return a.format(", ".join(bits[:-1]) + b + bits[-1])


def _read_the_post(
    text: str,
    feat_dict: Optional[Dict[str, float]],
    model_loaded: bool,
    fallback: str,
    lex_hits: List[str],
    seed: int,
) -> str:
    raw = (text or "").strip()
    if not raw:
        return "There is no text here to interpret."

    low = raw.lower()
    wc = _word_count(raw)
    ex = int(feat_dict.get("exclamation_count", _exclaim_count_text(raw))) if feat_dict else _exclaim_count_text(raw)
    ur = float(feat_dict.get("uppercase_ratio", _uppercase_ratio_text(raw))) if feat_dict else _uppercase_ratio_text(raw)

    chunks: List[str] = []

    # Length band — several phrasings each
    if wc <= 3:
        chunks.append(
            _pick(
                seed,
                3,
                [
                    "Only a few words show up, so there is little context beyond the bare sentence.",
                    "This is barely a sentence, so context is thin.",
                    "So little text is here that the tool has almost no narrative to work with.",
                    "It is extremely short, almost a fragment.",
                ],
            )
        )
    elif wc < 12:
        chunks.append(
            _pick(
                seed,
                4,
                [
                    "It reads like a snippet or title, not a full argument.",
                    "Length-wise it is closer to a headline than an article.",
                    "You get a short blurb, not a developed claim.",
                    "It is compact, more teaser than explanation.",
                ],
            )
        )
    elif wc < 40:
        chunks.append(
            _pick(
                seed,
                5,
                [
                    "There is enough text to judge tone, but not a deep essay.",
                    "It sits in the middle: not tiny, not long-form.",
                    "You can scan the vibe in a couple of breaths.",
                    "It is a medium-length post in everyday social style.",
                ],
            )
        )
    else:
        chunks.append(
            _pick(
                seed,
                6,
                [
                    "There is a longer stretch of text, so phrasing and pacing actually matter.",
                    "This is a fuller message, not a one-liner.",
                    "Enough lines appear that tone can build across the piece.",
                    "The writer had room to shape how this comes across.",
                ],
            )
        )

    if re.search(r"https?://|www\.", low):
        chunks.append(
            _pick(
                seed,
                7,
                [
                    "A link is embedded, so the honest next step is to open it and see the source.",
                    "Because a URL is present, the story partly lives outside this text.",
                    "Links shift the focus toward whatever site sits behind them.",
                    "It points outward on the web, not only inward at these words.",
                ],
            )
        )
    if "@" in raw and re.search(r"\@\w+", raw):
        chunks.append(
            _pick(
                seed,
                8,
                [
                    "It tags or mentions accounts, typical of social posts.",
                    "Handles show up, which usually means social context.",
                    "Someone is being called out or tagged directly.",
                ],
            )
        )
    if "#" in raw:
        chunks.append(
            _pick(
                seed,
                9,
                [
                    "Hashtags mark it as social-native content.",
                    "Topic tags steer how readers find the post.",
                    "It uses hashtags, so it is clearly aimed at a feed audience.",
                ],
            )
        )

    if any(h in low for h in _ATTRIBUTION_HINTS):
        chunks.append(
            _pick(
                seed,
                10,
                [
                    "It cites studies or reporting, which often tracks with careful language.",
                    "Science or attribution language appears, which can signal seriousness.",
                    "There is a nod to research or sources, which is not proof, but it is a tone.",
                    "References to research or outlets show up.",
                ],
            )
        )

    if lex_hits:
        chunks.append(_lexicon_plain(lex_hits, seed))
    else:
        chunks.append(
            _pick(
                seed,
                11,
                [
                    "Nothing here screams obvious clickbait catchphrases from the usual list.",
                    "It avoids the loudest sensational buzzwords we scan for.",
                    "No strong hit on the usual hype-word checklist.",
                    "The flashy keyword list is quiet here.",
                ],
            )
        )

    if ex >= 3:
        chunks.append(
            _pick(
                seed,
                12,
                [
                    "Multiple exclamation marks push the tone toward urgent or loud.",
                    "The punctuation stack feels shouty.",
                    "Exclamation marks pile up, which often reads as hype.",
                    "It yells a little through punctuation.",
                ],
            )
        )
    elif ex == 2:
        chunks.append(
            _pick(
                seed,
                13,
                [
                    "Two exclamation marks add emphasis or excitement.",
                    "A pair of exclamation marks nudges the energy up.",
                ],
            )
        )
    elif ex == 1:
        chunks.append(
            _pick(
                seed,
                14,
                [
                    "One exclamation mark adds a bit of punch.",
                    "A single exclamation mark lifts the tone slightly.",
                ],
            )
        )

    if ur > 0.18:
        chunks.append(
            _pick(
                seed,
                15,
                [
                    "Heavy capitals read like shouting or cheap ads.",
                    "ALL CAPS style shows up more than usual.",
                    "Capital letters dominate, which can feel aggressive.",
                ],
            )
        )
    elif ur > 0.1:
        chunks.append(
            _pick(
                seed,
                16,
                [
                    "Some words are shouted in capitals, which can feel pushy.",
                    "Caps jump out here and there.",
                ],
            )
        )

    qn = low.count("?")
    if qn >= 3:
        chunks.append(
            _pick(
                seed,
                17,
                [
                    "Several questions in a row can sound rhetorical or provocative.",
                    "Question after question can bait a reaction.",
                    "The question stack invites outrage or curiosity.",
                ],
            )
        )
    elif qn >= 1:
        chunks.append(
            _pick(
                seed,
                18,
                [
                    "At least one question invites the reader to engage.",
                    "A question opens the door to back-and-forth.",
                ],
            )
        )

    if feat_dict and "word_count" not in feat_dict and feat_dict.get("salience_norm") is not None:
        sn = float(feat_dict["salience_norm"])
        if sn > 0.55:
            chunks.append(
                _pick(
                    seed,
                    19,
                    [
                        "Overall delivery feels closer to loud viral posts than to a dry note.",
                        "Style signals skew toward attention-grabbing.",
                    ],
                )
            )
        elif sn < 0.35:
            chunks.append(
                _pick(
                    seed,
                    20,
                    [
                        "Overall delivery feels calmer than a typical viral blast.",
                        "Style signals look mild compared with shouty posts.",
                    ],
                )
            )

    if not model_loaded or fallback != "none":
        chunks.append(
            _pick(
                seed,
                21,
                [
                    "The backup scorer leans on simple keywords and punctuation, not the full neural model.",
                    "Without full weights, scoring sticks to basic cues.",
                    "Only a lightweight fallback is active here.",
                ],
            )
        )

    # Keep 3–6 sentences: trim from middle if too long, vary by seed
    max_keep = 4 + (seed % 3)
    if len(chunks) > max_keep:
        step = max(1, len(chunks) // max_keep)
        keep_idx = list(range(0, len(chunks), step))[:max_keep]
        chunks = [chunks[i] for i in keep_idx]

    return " ".join(chunks)


def _verdict_paragraph(pred: int, confidence: float, wc: int, lex_hits: List[str], seed: int) -> str:
    if confidence >= 0.82:
        cert = _pick(
            seed,
            30,
            [
                "The model is quite sure about this label.",
                "Confidence here looks strong.",
                "The score lands clearly on one side.",
                "The model is acting decisive.",
            ],
        )
    elif confidence >= 0.62:
        cert = _pick(
            seed,
            31,
            [
                "The model leans that way, but borderline cases still happen.",
                "This is not a slam-dunk score.",
                "Confidence is only moderate, so treat it as a hint.",
                "The model wavers enough that you should stay skeptical.",
            ],
        )
    else:
        cert = _pick(
            seed,
            32,
            [
                "The model is unsure, similar posts often split both ways.",
                "Scores are close, so this is a weak signal.",
                "Treat this label as soft, not final.",
                "This one could flip with a small wording change.",
            ],
        )

    short_and_thin = wc < 10 and len(lex_hits) == 0

    if pred == 0:
        body = _pick(
            seed,
            40,
            [
                "Taken together, this reads more like calm sharing or everyday language than a packaged hoax.",
                "The vibe matches posts that usually looked trustworthy in training.",
                "It lands on the side of ordinary, less sensational posting.",
                "Nothing here screams manipulative headline craft.",
            ],
        )
    else:
        body = _pick(
            seed,
            41,
            [
                "Taken together, this lines up more with posts that were usually flagged as unreliable.",
                "The pattern matches misleading or noisy content the model saw before.",
                "It sits closer to the unreliable bucket than to calm news tone.",
                "The model groups this with posts that often failed the trust bar.",
            ],
        )
        if short_and_thin:
            body += " " + _pick(
                seed,
                42,
                [
                    "Because the text is so thin, that call is mostly from tiny cues, and a person might disagree.",
                    "With so few words, the label is a guess from small signals.",
                    "Short lines are easy to misread, so take this with extra salt.",
                ],
            )

    return f"{body} {cert}"


def build_rich_explanation(
    text: str,
    pred: int,
    confidence: float,
    probs: List[float],
    feat_dict: Optional[Dict[str, float]],
    model_loaded: bool,
    fallback: str,
) -> str:
    _ = probs
    seed = _variant_seed(text, pred, confidence)
    lex_hits = _find_lexicon_hits(text)
    wc = _word_count(text)
    label = "reliable" if pred == 0 else "unreliable"

    reading = _read_the_post(text, feat_dict, model_loaded, fallback, lex_hits, seed)
    verdict = _verdict_paragraph(pred, confidence, wc, lex_hits, seed)

    headline = _pick(
        seed,
        50,
        [
            f"This post reads as {label}.",
            f"We mark this as {label}.",
            f"VeriPulse scores this as {label}.",
            f"The outcome here is {label}.",
            f"Label: {label}.",
            f"Overall classification: {label}.",
        ],
    )

    caveat = _pick(
        seed,
        60,
        [
            "This is an automatic read of wording and tone, not proof that a claim is true or false.",
            "Use it as a triage hint, not as a substitute for checking sources yourself.",
            "Scores reflect patterns in training data, not a courtroom verdict.",
            "Double-check anything important before you act on it.",
            "Treat this as a second opinion on style, not ground truth.",
        ],
    )

    return "\n\n".join([headline, reading, verdict, caveat])
