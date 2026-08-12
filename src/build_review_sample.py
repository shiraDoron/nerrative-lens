"""
Builds a small, human-friendly REVIEW SAMPLE (5 texts per narrative) from an
ALREADY-RUN build_profile_prototype.py output (e.g. the "calibration" batch),
for a first manual look BEFORE any extraction-rule tuning or building the full
narrative_profiler.py.

Deliberately does NOT run any model (no NER/spaCy/etc. re-run) - it only reads
the existing data/profiles/text_profiles_<prefix>.json and formats it. The one
exception is "agendas": that facet was not part of build_profile_prototype.py's
output, so this script adds it here via the EXISTING, already-built lexicon-
regex AGENDA_PATTERNS (from analyze_agendas.py, same approach as values_hits) -
this is a plain regex lookup, not a new model, and is computed directly on the
text already stored in the JSON.

Output: ONE ROW PER TEXT (wide, easy-to-read-in-Excel format), with each
predicted facet (actors/roles, relations, values, agendas) rendered as a
multi-line human-readable cell, and a blank free-text column per facet for the
annotator to note what's right/wrong, per the "quick look before tuning"
workflow (the separate long-format entity/relation review CSVs already built
by build_profile_prototype.py remain the source for the later, more rigorous
per-item evaluation via evaluate_profile_extraction.py).

Run (from repo root):
    python src/build_review_sample.py --prefix calibration --n-per-narrative 5
"""

import argparse
import json
import os

import pandas as pd

from analyze_agendas import AGENDA_PATTERNS, clean_text as clean_lexicon_text, MENTION_RE, HASHTAG_RE

PROFILES_DIR = "data/profiles"
REPORT_DIR = "reports/profiler_prototype"


def _extract_agenda_hits(text):
    """Same evidence-bearing pattern as build_profile_prototype.py's
    _extract_values_hits, applied to AGENDA_PATTERNS instead of VALUES_PATTERNS.
    Also strips @mentions/#hashtags (not stripped by the shared
    analyze_agendas.clean_text) before matching - a word embedded in a handle
    or hashtag (e.g. "@media.somehandle", "#VoicesFromCentralAsia") can
    otherwise spuriously match a lexicon category term."""
    cleaned = clean_lexicon_text(text)
    cleaned = MENTION_RE.sub(" ", cleaned)
    cleaned = HASHTAG_RE.sub(" ", cleaned)
    hits = []
    for category, pattern in AGENDA_PATTERNS.items():
        match = pattern.search(cleaned)
        if match is None:
            continue
        start = max(0, match.start() - 30)
        end = min(len(cleaned), match.end() + 30)
        hits.append({
            "category": category,
            "matched_text": match.group(0),
            "evidence": cleaned[start:end].strip(),
        })
    return hits


def _format_actors_roles(entities):
    if not entities:
        return "(none detected)"
    lines = []
    for e in entities:
        role = e["predicted_role"]
        extra = f" [candidate: {e['candidate_role']}]" if role in ("unknown", "uncertain") else ""
        frag = " (reconstructed)" if e.get("was_reconstructed") else ""
        lines.append(
            f"{e['canonical']}{frag} = {role} (conf {e['role_confidence']}, agency {e['predicted_agency']}){extra}"
        )
    return "\n".join(lines)


def _format_relations(relations):
    if not relations:
        return "(none detected)"
    lines = []
    for r in relations:
        rel = r["relation"]
        extra = f" [candidate: {r['candidate_relation']}]" if rel == "uncertain" else ""
        lines.append(f"{r['source']} --{rel}--> {r['target']} (conf {r['confidence']}){extra}")
    return "\n".join(lines)


def _format_hits(hits):
    if not hits:
        return "(none detected)"
    return "\n".join(f"{h['category']}: {h['matched_text']!r}" for h in hits)


def build_review_sample(text_profiles, n_per_narrative=5):
    rows = []
    seen_per_narrative = {}
    for tp in text_profiles:
        nar = tp["narrative_name"]
        count = seen_per_narrative.get(nar, 0)
        if count >= n_per_narrative:
            continue
        seen_per_narrative[nar] = count + 1

        agenda_hits = _extract_agenda_hits(tp["text"])

        rows.append({
            "text_id": tp["text_id"],
            "narrative_name": nar,
            "text": tp["text"],
            "system_actors_roles": _format_actors_roles(tp["entities"]),
            "system_relations": _format_relations(tp["relations"]),
            "system_values": _format_hits(tp["values_hits"]),
            "system_agendas": _format_hits(agenda_hits),
            "manual_actors_roles_feedback": "",
            "manual_relations_feedback": "",
            "manual_values_feedback": "",
            "manual_agendas_feedback": "",
            "annotator_notes": "",
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prefix", type=str, default="calibration",
                         help="Prefix of the already-run build_profile_prototype.py batch to read from.")
    parser.add_argument("--n-per-narrative", type=int, default=5)
    parser.add_argument("--out", type=str, default=None,
                         help="Output CSV path (default: reports/profiler_prototype/<prefix>_review_sample.csv)")
    args = parser.parse_args()

    text_profiles_path = os.path.join(PROFILES_DIR, f"text_profiles_{args.prefix}.json")
    if not os.path.exists(text_profiles_path):
        raise FileNotFoundError(
            f"{text_profiles_path} not found - run build_profile_prototype.py --prefix {args.prefix} first."
        )

    with open(text_profiles_path, "r", encoding="utf-8") as f:
        text_profiles = json.load(f)

    rows = build_review_sample(text_profiles, args.n_per_narrative)

    out_path = args.out or os.path.join(REPORT_DIR, f"{args.prefix}_review_sample.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")

    n_narratives = len({r["narrative_name"] for r in rows})
    print(f"Wrote {len(rows)} texts across {n_narratives} narratives -> {out_path}")


if __name__ == "__main__":
    main()
