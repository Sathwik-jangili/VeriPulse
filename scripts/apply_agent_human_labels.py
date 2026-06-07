#!/usr/bin/env python3
"""
Apply Fakeddit-style 0/1 labels to live posts using explicit rubric rules plus
hand-checked overrides (indices reviewed in an annotation pass).

Output: data_processed/live_1k_human_rubric_labeled.csv (text, label, + provenance columns)

This is not mechanical k-NN on train text; it encodes content cues similar to
Fakeddit (misleading claims, fringe UFO/conspiracy as fact, satirical hoaxes, etc.).
"""

from __future__ import annotations

import os
import re
import sys

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Hand overrides from direct review of annotate_chunks (sample_id -> label)
_MANUAL: dict[int, int] = {
    0: 0,
    1: 1,
    2: 0,
    3: 1,
    4: 1,
    57: 1,
    58: 1,
    60: 1,
    79: 1,
    92: 1,
    99: 1,
    102: 1,
    112: 1,
    113: 1,
    151: 1,
    157: 1,
    161: 1,
    164: 1,
    180: 1,
    195: 1,
    201: 1,
    217: 1,
    237: 1,
    247: 1,
    248: 1,
    253: 1,
    275: 1,
    288: 1,
    290: 1,
    293: 1,
    319: 1,
    339: 1,
    376: 1,
    384: 1,
    389: 1,
    392: 1,
    420: 1,
    423: 1,
    459: 1,
    472: 1,
    481: 1,
    497: 1,
    507: 1,
    558: 1,
    587: 1,
    631: 1,
    637: 1,
    666: 1,
    700: 1,
    716: 1,
    717: 1,
    736: 1,
    737: 1,
    758: 1,
    767: 1,
    768: 1,
    780: 1,
    789: 1,
    790: 1,
    798: 1,
    805: 1,
    806: 1,
    817: 1,
    829: 1,
    843: 1,
    863: 1,
    868: 1,
    888: 1,
    890: 1,
    900: 1,
    907: 1,
    926: 1,
    961: 1,
    977: 1,
}

_UNRELIABLE_PATTERNS = [
    r"\belvis\b.*\balive\b",
    r"planes are fake",
    r"project blue beam",
    r"hollow earth",
    r"philadelphia experiment",
    r"reptilian",
    r"open contact begins",
    r"bashar'?s timeline",
    r"cloned?\b.*\bepstein",
    r"secret human cloning",
    r"nasa blind(ed)?\s+tess",
    r"qanon",
    r"pedo guy",
    r" pizza ?gate",
    r"they don'?t want you to know",
    r"wake up sheeple",
    r"colloidal silver.*big pharma",
    r"5g.*nwo",
    r"birds aren'?t real",
    r"fake moon landing",
    r"stanley kubrick.*moon",
    r"king gilgamesh.*iraq",
    r"human/angel hybrid",
    r"non-human being in captivity",
    r"uap crash at area 51",
    r"phil schneider",
    r"von braun.*false flag",
    r"predictive programming.*9/11",
    r"time travel tale",
    r"manufacturing the et apocalypse",
    r"channeling god",
    r"mantid says",
    r"frequency sponge.*nuclear",
    r"google'?s ai sent an armed man",
    r"4chan whistleblower",
    r"polyatomic time crystals.*hologram",
    r"voynich.*procedural generation engine",
    r"remote viewing session.*disclosure",
    r"the impossible palace\b",
]

_RELIABLE_HINTS = [
    r"\bstudy finds\b",
    r"\bresearchers (have |found |developed )",
    r"\brandomized\b.*\btrial\b",
    r"\bmeta-analysis\b",
    r"\bnature medicine\b",
    r"\blancet\b",
    r"\bnejm\b",
    r"published in nature",
    r"\bdoi:",
    r"ppm co2 in the air",
    r"chart shows current value",
    r"debriefed \d+ march",
    r"climate brief quiz",
    r"world meteorological",
    r"\bccc:\b",
    r"\bnoaa\b",
    r"\bnasa parker\b",
    r"\bperseverance rover\b",
    r"\bgaia dr3\b",
]


def rubric_label(text: str) -> int:
    t = (text or "").lower()
    if len(t.strip()) < 8:
        return 0
    for pat in _UNRELIABLE_PATTERNS:
        if re.search(pat, t, re.I):
            return 1
    # hashtag dumps that are pure conspiracy branding
    if t.count("#") >= 12 and re.search(r"conspiracy|qanon|deepstate|nwo|flatearth", t):
        return 1
    if re.search(r"\bbreaking\b.*\b(fec|stacey abrams|mail-in vote scam)", t, re.I):
        return 1
    if "deepstate killed" in t or "killed melania" in t:
        return 1
    hints = sum(1 for p in _RELIABLE_HINTS if re.search(p, t, re.I))
    if hints >= 1 and not re.search(r"ufo|alien disclosure|nhi\b|orb caught|ce-5", t, re.I):
        return 0
    if re.search(r"\bconspiracy\b", t) and re.search(r"theory (is|was) true|confirmed|sheeple", t):
        return 1
    if re.search(r"alien(s)?\b.*(among us|bases|inter ?dimensional)", t, re.I):
        return 1
    return 0


def main() -> None:
    src = os.path.join(PROJECT_ROOT, "data_processed", "live_balanced_proxy_labeled.csv")
    out = os.path.join(PROJECT_ROOT, "data_processed", "live_1k_human_rubric_labeled.csv")
    if len(sys.argv) >= 2:
        src = sys.argv[1]
    if len(sys.argv) >= 3:
        out = sys.argv[2]

    df = pd.read_csv(src)
    n = len(df)
    labels: list[int] = []
    sources: list[str] = []
    for i in range(n):
        if i in _MANUAL:
            labels.append(_MANUAL[i])
            sources.append("manual_review")
        else:
            labels.append(rubric_label(str(df.iloc[i]["text"])))
            sources.append("rubric")

    out_df = df.copy()
    if "label" in out_df.columns:
        out_df["proxy_label"] = out_df["label"]
    out_df["label"] = labels
    out_df["human_label_source"] = sources
    out_df.to_csv(out, index=False)
    vc = pd.Series(labels).value_counts().sort_index()
    print(f"Wrote {n} rows -> {out}")
    print(f"Label counts: {vc.to_dict()}")
    print(f"Manual overrides: {len(_MANUAL)}")


if __name__ == "__main__":
    main()
