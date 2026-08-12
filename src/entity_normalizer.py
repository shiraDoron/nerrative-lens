"""
Entity normalization / alias merging for the Narrative Fingerprint profiler.

Raw NER output produces many surface variants of the same real-world actor
(e.g. "US", "U.S.", "United States", "America" should all become one canonical
actor). Without merging these, narrative-level aggregation (entity mention
counts, role distributions) becomes badly fragmented across surface forms of
the same entity.

This module provides:
  1. A small curated alias table (ALIAS_GROUPS) for the actors that recur most
     across the project's own narratives/sources (see NARRATIVES_ACCOUNTS in
     build_twitter_dataset.py / NARRATIVES_CHANNELS in build_telegram_dataset.py)
     plus common geopolitical actors that show up across the 7 narratives.
  2. Light, deterministic textual normalization rules (case, punctuation,
     leading "the", possessive 's) for anything not covered by the table, so
     at minimum pure surface-form variants of the SAME unseen entity still
     merge together even without a known alias.

This is intentionally a curated/rule-based MVP, not automatic entity-linking
(e.g. via Wikidata) - easy to extend by adding more entries to ALIAS_GROUPS as
gaps are found during manual validation of the profiler's output.
"""

import re

# canonical_name -> list of surface aliases (matched case-insensitively, after
# basic punctuation/whitespace cleanup - see _basic_clean below).
ALIAS_GROUPS = {
    "United States": [
        "us", "u.s.", "u.s.a.", "usa", "america", "american", "washington",
        "united states", "united states of america",
    ],
    "Israel": ["israel", "israeli", "state of israel"],
    "IDF": [
        "idf", "israeli military", "israeli army", "israeli defense forces",
        "israel defense forces", "israeli defence forces",
    ],
    "Hamas": ["hamas"],
    "Hezbollah": ["hezbollah", "hizbullah", "hizballah"],
    "Iran": ["iran", "iranian", "islamic republic of iran"],
    "Russia": ["russia", "russian", "russian federation", "moscow", "kremlin"],
    "Ukraine": ["ukraine", "ukrainian", "kyiv", "kiev"],
    "European Union": ["eu", "european union"],
    "United Nations": ["un", "united nations", "u.n."],
    "NATO": ["nato", "north atlantic treaty organization"],
    "China": ["china", "chinese", "beijing", "prc"],
    "United Kingdom": ["uk", "u.k.", "united kingdom", "britain", "british", "london"],
    # NOTE: deliberately does NOT include "palestine" here - that's a place
    # name (like "Gaza"), not the demonym/people-group "Palestinians". Merging
    # them conflated a LOC mention with a PEOPLE mention (found during manual
    # calibration review: "occupied Palestine" was wrongly canonicalized to
    # "Palestinians"). "Palestine" alone now falls through to
    # fallback_titlecase, keeping it distinct.
    "Palestinians": ["palestinians", "palestinian"],
    "Houthis": ["houthi", "houthis", "ansar allah"],
}

# lowercased alias -> canonical name, built once at import time.
_ALIAS_TO_CANONICAL = {}
for _canonical, _aliases in ALIAS_GROUPS.items():
    for _alias in _aliases:
        _ALIAS_TO_CANONICAL[_alias.lower()] = _canonical
    _ALIAS_TO_CANONICAL[_canonical.lower()] = _canonical

_PUNCT_RE = re.compile(r"[.\u2019']")
_LEADING_THE_RE = re.compile(r"^\s*the\s+", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


def _basic_clean(raw_text):
    """Strip punctuation dots/apostrophes, collapse whitespace, drop a leading 'the'."""
    cleaned = raw_text.strip()
    cleaned = _PUNCT_RE.sub("", cleaned)
    cleaned = _WS_RE.sub(" ", cleaned).strip()
    cleaned_no_the = _LEADING_THE_RE.sub("", cleaned)
    return cleaned_no_the.lower(), cleaned_no_the


def normalize_entity(raw_text):
    """
    Normalize a raw NER entity surface string to a canonical actor name.

    Returns a dict: {"canonical": str, "method": str, "matched_alias": str | None}
      method is one of: "alias_table" | "alias_table_possessive" | "fallback_titlecase"
    """
    cleaned_lower, cleaned_original = _basic_clean(raw_text)

    if cleaned_lower in _ALIAS_TO_CANONICAL:
        return {
            "canonical": _ALIAS_TO_CANONICAL[cleaned_lower],
            "method": "alias_table",
            "matched_alias": cleaned_lower,
        }

    # possessive suffix, e.g. "Israels" after punctuation-stripping of "Israel's"
    if cleaned_lower.endswith("s") and cleaned_lower[:-1] in _ALIAS_TO_CANONICAL:
        return {
            "canonical": _ALIAS_TO_CANONICAL[cleaned_lower[:-1]],
            "method": "alias_table_possessive",
            "matched_alias": cleaned_lower[:-1],
        }

    # fallback: no known alias - normalize case/whitespace so pure surface variants
    # of the same unseen entity still merge, even without a cross-reference.
    fallback_canonical = " ".join(
        w if w.isupper() else w.capitalize() for w in cleaned_original.split()
    )
    return {
        "canonical": fallback_canonical or raw_text.strip(),
        "method": "fallback_titlecase",
        "matched_alias": None,
    }


if __name__ == "__main__":
    for sample in ["US", "U.S.", "United States", "America", "Israel's", "IDF",
                   "Israeli military", "Kyiv", "unknown organization"]:
        print(f"{sample!r:30s} -> {normalize_entity(sample)}")
