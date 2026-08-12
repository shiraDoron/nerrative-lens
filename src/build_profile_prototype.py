"""
Prototype runner for the Narrative Fingerprint pipeline - Step 1 of the
"Narrative Fingerprint" roadmap: raw entity extraction + normalization,
Entity->Role MVP, relation extraction MVP.

Deliberately scoped to a SMALL SAMPLE so the heuristics can be spot-checked by
hand BEFORE running on the full corpus or building the full aggregating
narrative_profiler.py. Does NOT touch any of the classification models
(fusion.py's NarrativeDetector / SBERTOnlyDetector / HybridNarrativeDetector) -
those remain untouched baselines/side experiments.

Pipeline stages, kept SEPARATE in the output (per the validation requirement
that we must be able to tell at which stage an error was introduced):
  1. raw_entities        - straight out of the HF NER pipeline (dslim/bert-base-NER),
                            only clean "##"-continuation word-pieces merged.
  2. reconstructed_entities - raw_entities after a GENERIC, name-agnostic
                            offset-based merge of adjacent same-type fragments
                            (see ner.reconstruct_fragmented_entities) - addresses
                            NER fragmentation (e.g. "Ran Goyili" -> "Ra"/"n"/"Go")
                            WITHOUT any per-name alias hand-fixing.
  3. normalized_entities  - reconstructed_entities after alias-table canonicalization
                            (entity_normalizer.normalize_entity).
  4. role-tagged entities - normalized_entities after Entity->Role MVP
                            (entity_role_tagger.tag_entity_role): role is one of
                            hero/victim/aggressor/betrayer/savior, or "unknown"
                            (no signal at all) / "uncertain" (weak signal) - we
                            deliberately do NOT force every entity into one of
                            the 5 substantive roles.
  Relations (extract_relations) are similarly either a specific RELATION_TYPES
  value, or "uncertain" when confidence is below threshold.

Outputs:
  - data/profiles/text_profiles_<prefix>.json
        One TextProfile dict per sampled text, with all 4 entity stages kept
        separately (see keys above), plus relations and values_hits (each with
        confidence/evidence/method for explainability).
  - data/profiles/narrative_profiles_<prefix>.json
        A PRELIMINARY, PARTIAL NarrativeProfile per narrative - only the
        actors/roles/relations/values facets covered by this prototype.
        Explicitly split into "observed_statistics" (raw counts/prevalence)
        vs. "inferred_interpretation" (role distributions, relation examples).
        Full facet aggregation (agenda/ideology/rhetoric/stance-per-entity) is
        deferred to the full narrative_profiler.py once these MVPs are validated.
  - reports/profiler_prototype/<prefix>_entities.csv
        LONG format, one row per entity mention, predicted_* columns + empty
        gold_* columns + a blank correct_incorrect_uncertain column, for manual
        annotation. gold_* columns are NEVER pre-filled from system predictions.
  - reports/profiler_prototype/<prefix>_relations.csv
        LONG format, one row per extracted relation, same predicted/gold split.
  - reports/profiler_prototype/<prefix>_values.csv
        LONG format, one row per values-lexicon hit, same predicted/gold split.

Run:
    # small calibration batch (~5-10 texts/narrative) to review and tune rules first:
    python src/build_profile_prototype.py --n-per-narrative 8 --prefix calibration

    # later, larger (still NOT full-corpus) validation batch:
    python src/build_profile_prototype.py --n-per-narrative 25 --prefix sample25
(from the repo root, so the data/... and reports/... relative paths resolve correctly)
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict

import pandas as pd
import spacy

from ner import EntityAnalysisPipeline, reconstruct_fragmented_entities
from entity_normalizer import normalize_entity
from entity_role_tagger import tag_entity_role, find_conceptual_actor_spans
from relation_extractor import extract_relations
from analyze_agendas import (
    VALUES_PATTERNS, clean_text as clean_lexicon_text,
    MENTION_RE, HASHTAG_RE,
)

# Minimum alphabetic-character length for a promoted entity mention. Found
# during manual calibration review that the HF NER pipeline sometimes emits
# tiny single-CHARACTER tokenization fragments (e.g. standalone "U", "M", "O")
# that carry no real signal and only add noise to the role/relation stages
# downstream. Set to 2 (not 3) so legitimate short abbreviations common in
# this project's domain (US, UK, EU, UN, PA, ...) are NOT filtered out -
# general, length-based rule, not tied to any specific name.
MIN_ENTITY_TEXT_LEN = 2

_LEADING_DET_RE = re.compile(r"^\s*(the|a|an)\s+", re.IGNORECASE)


def _strip_social_noise_for_lexicon(cleaned_text):
    """Additionally strips @mentions and #hashtags (kept as-is by the shared
    analyze_agendas.clean_text, which only strips URLs/known boilerplate) before
    values/agenda lexicon matching for THIS profiler only - found during manual
    calibration review that a word embedded in a handle or hashtag (e.g.
    "@media.somehandle", "#VoicesFromCentralAsia") can spuriously match a
    lexicon category term. Kept local to this file rather than changing the
    shared analyze_agendas.clean_text(), since that function also feeds the
    already-published narrative-level reports."""
    text = MENTION_RE.sub(" ", cleaned_text)
    text = HASHTAG_RE.sub(" ", text)
    return text

RAW_DATASETS = [
    "data/raw/twitter_natural_dataset.csv",
    "data/raw/telegram_natural_dataset.csv",
    "data/raw/gemini_natural_dataset.csv",
    "data/raw/gpt_natural_dataset.csv",
]

PROFILES_DIR = "data/profiles"
REPORT_DIR = "reports/profiler_prototype"


def load_sample(n_per_narrative=25, seed=42):
    """Loads all 4 natural datasets, concatenates, and samples up to
    n_per_narrative texts per narrative_name (fewer if a narrative has less data)."""
    frames = []
    for path in RAW_DATASETS:
        if os.path.exists(path):
            df = pd.read_csv(path)
            frames.append(df[["text", "narrative_name"]])
        else:
            print(f"WARNING: dataset not found, skipping: {path}")

    full = pd.concat(frames, ignore_index=True).dropna(subset=["text", "narrative_name"])
    sampled = (
        full.groupby("narrative_name", group_keys=False)
        .apply(lambda g: g.sample(n=min(n_per_narrative, len(g)), random_state=seed))
        .reset_index(drop=True)
    )
    return sampled


def _extract_values_hits(text):
    """Returns a list of {"category", "matched_text", "evidence"} for each
    VALUES_LEXICON category found in the (cleaned) text - each hit carries the
    actual matched span (+ a bit of surrounding context) as evidence, per the
    project's "every prediction needs evidence" rule."""
    cleaned = _strip_social_noise_for_lexicon(clean_lexicon_text(text))
    hits = []
    for category, pattern in VALUES_PATTERNS.items():
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


def build_text_profile(text_id, text, narrative_name, ner_analyzer, nlp):
    """Runs the full 4-stage entity pipeline (raw -> reconstructed -> normalized
    -> role-tagged) + relation extraction + values-lexicon hits for a single
    text, returning a TextProfile dict with every stage kept separately."""
    doc = nlp(text)

    raw_entities = ner_analyzer.extract_raw_entities(text)
    reconstructed_entities = reconstruct_fragmented_entities(raw_entities)
    # Drop entities that are STILL too short after the fragment-merge pass (a
    # real single wordpiece fragment like standalone "M"/"Se"/"Al" that never
    # found a neighbor to merge with) - filtering pre-merge would break the
    # legitimate "U" + "S" -> "U.S." repair, so this must run AFTER reconstruction.
    reconstructed_entities = [
        ent for ent in reconstructed_entities
        if len(re.sub(r"[^A-Za-z]", "", ent["text"])) >= MIN_ENTITY_TEXT_LEN
    ]

    entity_spans_by_token_i = {}
    normalized_entities = []
    span_and_norm = []  # (span, reconstructed_ent, norm) kept for the role-tagging pass

    for ent in reconstructed_entities:
        span = doc.char_span(ent["start"], ent["end"], alignment_mode="expand")
        if span is None:
            continue
        norm = normalize_entity(ent["text"])
        normalized_entities.append({
            "raw_text": ent["text"],
            "entity_group": ent["entity_group"],
            "ner_score": round(ent["score"], 3),
            "was_reconstructed": ent["was_reconstructed"],
            "n_fragments_merged": ent["n_fragments_merged"],
            "canonical": norm["canonical"],
            "normalization_method": norm["method"],
            "matched_alias": norm["matched_alias"],
        })
        for tok in span:
            entity_spans_by_token_i[tok.i] = norm["canonical"]
        span_and_norm.append((span, ent, norm))

    role_tagged_entities = []
    for span, ent, norm in span_and_norm:
        role_info = tag_entity_role(span)
        role_tagged_entities.append({
            "raw_text": ent["text"],
            "canonical": norm["canonical"],
            "normalization_method": norm["method"],
            "entity_group": ent["entity_group"],
            "ner_score": round(ent["score"], 3),
            "was_reconstructed": ent["was_reconstructed"],
            "predicted_role": role_info["role"],
            "role_confidence": role_info["confidence"],
            "role_evidence": role_info["evidence_text"],
            "role_method": role_info["method"],
            "candidate_role": role_info["candidate_role"],
            "predicted_agency": role_info["agency"],
        })

    # Conceptual actors: noun-phrase / in-group-pronoun candidates that AREN'T
    # proper-noun NER entities (see find_conceptual_actor_spans docstring - the
    # single biggest actor-coverage gap found during manual calibration
    # review). Folded into the SAME entity_spans_by_token_i map (so relation
    # extraction can reference them too) and appended to role_tagged_entities
    # with a distinguishing entity_group ("CONCEPT"/"IN_GROUP") rather than a
    # separate list, so every downstream consumer (aggregation, review CSVs,
    # build_review_sample.py) picks them up with zero extra changes.
    for span, span_type in find_conceptual_actor_spans(doc, set(entity_spans_by_token_i.keys())):
        if span_type == "in_group_pronoun":
            # Canonicalize every surface form (we/us/our/ours/ourselves, any
            # case) to a single fixed "we" - they all denote the SAME narrator
            # in-group referent within a text, so keeping surface casing would
            # wrongly fragment one actor into several near-duplicate rows.
            canonical = "we"
        else:
            canonical = _LEADING_DET_RE.sub("", span.text.strip())
        for tok in span:
            entity_spans_by_token_i[tok.i] = canonical
        role_info = tag_entity_role(span)
        role_tagged_entities.append({
            "raw_text": span.text,
            "canonical": canonical,
            "normalization_method": span_type,
            "entity_group": "IN_GROUP" if span_type == "in_group_pronoun" else "CONCEPT",
            "ner_score": None,
            "was_reconstructed": False,
            "predicted_role": role_info["role"],
            "role_confidence": role_info["confidence"],
            "role_evidence": role_info["evidence_text"],
            "role_method": role_info["method"],
            "candidate_role": role_info["candidate_role"],
            "predicted_agency": role_info["agency"],
        })

    relations = []
    for sent in doc.sents:
        relations.extend(extract_relations(sent, entity_spans_by_token_i))

    values_hits = _extract_values_hits(text)

    return {
        "text_id": text_id,
        "text": text,
        "narrative_name": narrative_name,
        "raw_entities": raw_entities,
        "reconstructed_entities": reconstructed_entities,
        "normalized_entities": normalized_entities,
        "entities": role_tagged_entities,
        "relations": relations,
        "values_hits": values_hits,
    }


def aggregate_narrative_profiles(text_profiles):
    """Builds a preliminary, PARTIAL NarrativeProfile per narrative from the
    TextProfiles produced by this prototype (actors/roles/relations/values only)."""
    raw = defaultdict(lambda: {
        "n_documents": 0,
        "actor_mentions": Counter(),
        "actor_role_counts": defaultdict(Counter),
        "relation_counts": Counter(),
        "relation_examples": defaultdict(list),
        "values_hit_counts": Counter(),
    })

    for tp in text_profiles:
        nar = tp["narrative_name"]
        bucket = raw[nar]
        bucket["n_documents"] += 1
        for ent in tp["entities"]:
            bucket["actor_mentions"][ent["canonical"]] += 1
            bucket["actor_role_counts"][ent["canonical"]][ent["predicted_role"]] += 1
        for rel in tp["relations"]:
            key = rel["relation"]
            bucket["relation_counts"][key] += 1
            if len(bucket["relation_examples"][key]) < 5:
                bucket["relation_examples"][key].append({
                    "source": rel["source"], "target": rel["target"],
                    "confidence": rel["confidence"], "evidence": rel["evidence"],
                    "candidate_relation": rel.get("candidate_relation"),
                })
        for hit in tp["values_hits"]:
            bucket["values_hit_counts"][hit["category"]] += 1

    narrative_profiles = {}
    for nar, bucket in raw.items():
        n_docs = bucket["n_documents"]

        main_actors = []
        for actor, mentions in bucket["actor_mentions"].most_common(10):
            role_counts = bucket["actor_role_counts"][actor]
            total_votes = sum(role_counts.values())
            role_distribution = (
                {role: round(count / total_votes, 3) for role, count in role_counts.items()}
                if total_votes else {}
            )
            main_actors.append({
                "entity": actor,
                "observed": {
                    "mentions": mentions,
                    "prevalence_pct": round(mentions / n_docs * 100, 1),
                },
                "inferred": {"role_distribution": role_distribution},
            })

        narrative_profiles[nar] = {
            "narrative": nar,
            "n_documents": n_docs,
            "observed_statistics": {
                "relation_counts": dict(bucket["relation_counts"]),
                "values_hit_prevalence_pct": {
                    cat: round(count / n_docs * 100, 1)
                    for cat, count in bucket["values_hit_counts"].items()
                },
            },
            "inferred_interpretation": {
                "main_actors": main_actors,
                "relation_examples": dict(bucket["relation_examples"]),
            },
        }

    return narrative_profiles


ENTITY_REVIEW_COLUMNS = [
    "text_id", "narrative_name", "text",
    "predicted_entity", "predicted_entity_type", "was_reconstructed", "n_fragments_merged",
    "predicted_role", "role_confidence", "role_evidence", "role_method", "candidate_role",
    "predicted_agency",
    "gold_entity", "gold_role", "gold_agency", "correct_incorrect_uncertain", "annotator_notes",
]

RELATION_REVIEW_COLUMNS = [
    "text_id", "narrative_name", "text",
    "predicted_source", "predicted_relation", "predicted_target",
    "relation_confidence", "relation_evidence", "relation_method", "candidate_relation",
    "gold_source", "gold_relation", "gold_target", "correct_incorrect_uncertain", "annotator_notes",
]

VALUES_REVIEW_COLUMNS = [
    "text_id", "narrative_name", "text",
    "predicted_value", "matched_text", "value_evidence",
    "gold_value", "correct_incorrect_uncertain", "annotator_notes",
]


def write_entity_review(text_profiles, path):
    rows = []
    for tp in text_profiles:
        if not tp["entities"]:
            rows.append({
                "text_id": tp["text_id"], "narrative_name": tp["narrative_name"], "text": tp["text"],
                "predicted_entity": "(none detected)", "predicted_entity_type": "", "was_reconstructed": "",
                "n_fragments_merged": "", "predicted_role": "", "role_confidence": "", "role_evidence": "",
                "role_method": "", "candidate_role": "", "predicted_agency": "",
                "gold_entity": "", "gold_role": "", "gold_agency": "", "correct_incorrect_uncertain": "",
                "annotator_notes": "",
            })
            continue
        for ent in tp["entities"]:
            rows.append({
                "text_id": tp["text_id"], "narrative_name": tp["narrative_name"], "text": tp["text"],
                "predicted_entity": ent["canonical"], "predicted_entity_type": ent["entity_group"],
                "was_reconstructed": ent["was_reconstructed"], "n_fragments_merged": ent.get("n_fragments_merged", 1),
                "predicted_role": ent["predicted_role"], "role_confidence": ent["role_confidence"],
                "role_evidence": ent["role_evidence"], "role_method": ent["role_method"],
                "candidate_role": ent["candidate_role"], "predicted_agency": ent["predicted_agency"],
                "gold_entity": "", "gold_role": "", "gold_agency": "", "correct_incorrect_uncertain": "",
                "annotator_notes": "",
            })
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows, columns=ENTITY_REVIEW_COLUMNS).to_csv(path, index=False, encoding="utf-8-sig")


def write_relation_review(text_profiles, path):
    rows = []
    for tp in text_profiles:
        if not tp["relations"]:
            rows.append({
                "text_id": tp["text_id"], "narrative_name": tp["narrative_name"], "text": tp["text"],
                "predicted_source": "", "predicted_relation": "(none detected)", "predicted_target": "",
                "relation_confidence": "", "relation_evidence": "", "relation_method": "",
                "candidate_relation": "", "gold_source": "", "gold_relation": "", "gold_target": "",
                "correct_incorrect_uncertain": "", "annotator_notes": "",
            })
            continue
        for rel in tp["relations"]:
            rows.append({
                "text_id": tp["text_id"], "narrative_name": tp["narrative_name"], "text": tp["text"],
                "predicted_source": rel["source"], "predicted_relation": rel["relation"],
                "predicted_target": rel["target"], "relation_confidence": rel["confidence"],
                "relation_evidence": rel["evidence"], "relation_method": rel["method"],
                "candidate_relation": rel.get("candidate_relation"),
                "gold_source": "", "gold_relation": "", "gold_target": "",
                "correct_incorrect_uncertain": "", "annotator_notes": "",
            })
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows, columns=RELATION_REVIEW_COLUMNS).to_csv(path, index=False, encoding="utf-8-sig")


def write_values_review(text_profiles, path):
    rows = []
    for tp in text_profiles:
        if not tp["values_hits"]:
            rows.append({
                "text_id": tp["text_id"], "narrative_name": tp["narrative_name"], "text": tp["text"],
                "predicted_value": "(none detected)", "matched_text": "", "value_evidence": "",
                "gold_value": "", "correct_incorrect_uncertain": "", "annotator_notes": "",
            })
            continue
        for hit in tp["values_hits"]:
            rows.append({
                "text_id": tp["text_id"], "narrative_name": tp["narrative_name"], "text": tp["text"],
                "predicted_value": hit["category"], "matched_text": hit["matched_text"],
                "value_evidence": hit["evidence"],
                "gold_value": "", "correct_incorrect_uncertain": "", "annotator_notes": "",
            })
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows, columns=VALUES_REVIEW_COLUMNS).to_csv(path, index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-per-narrative", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", type=str, default="sample",
                         help="Output filename prefix, e.g. 'calibration' or 'sample25'. "
                              "Keeps different-sized runs from overwriting each other.")
    args = parser.parse_args()

    os.makedirs(PROFILES_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    sample_df = load_sample(args.n_per_narrative, args.seed)
    print(f"Sampled {len(sample_df)} texts across {sample_df['narrative_name'].nunique()} narratives.")

    print("Loading NER pipeline...")
    ner_analyzer = EntityAnalysisPipeline()
    print("Loading spaCy dependency parser (en_core_web_trf)...")
    nlp = spacy.load("en_core_web_trf")

    text_profiles = []
    total = len(sample_df)
    for i, row in enumerate(sample_df.itertuples(index=False), 1):
        text_id = f"{args.prefix}_{i:04d}"
        print(f"[{i}/{total}] {row.narrative_name}: {row.text[:60]!r}...")
        tp = build_text_profile(text_id, row.text, row.narrative_name, ner_analyzer, nlp)
        text_profiles.append(tp)

    text_profiles_path = os.path.join(PROFILES_DIR, f"text_profiles_{args.prefix}.json")
    narrative_profiles_path = os.path.join(PROFILES_DIR, f"narrative_profiles_{args.prefix}.json")
    entities_csv_path = os.path.join(REPORT_DIR, f"{args.prefix}_entities.csv")
    relations_csv_path = os.path.join(REPORT_DIR, f"{args.prefix}_relations.csv")
    values_csv_path = os.path.join(REPORT_DIR, f"{args.prefix}_values.csv")

    with open(text_profiles_path, "w", encoding="utf-8") as f:
        json.dump(text_profiles, f, ensure_ascii=False, indent=2)

    narrative_profiles = aggregate_narrative_profiles(text_profiles)
    with open(narrative_profiles_path, "w", encoding="utf-8") as f:
        json.dump(narrative_profiles, f, ensure_ascii=False, indent=2)

    write_entity_review(text_profiles, entities_csv_path)
    write_relation_review(text_profiles, relations_csv_path)
    write_values_review(text_profiles, values_csv_path)

    n_fragmented = sum(
        1 for tp in text_profiles for ent in tp["reconstructed_entities"] if ent["was_reconstructed"]
    )

    print("\nDone.")
    print(f"- {len(text_profiles)} text profiles -> {text_profiles_path}")
    print(f"- {len(narrative_profiles)} narrative profiles -> {narrative_profiles_path}")
    print(f"- entity review (fill gold_* columns) -> {entities_csv_path}")
    print(f"- relation review (fill gold_* columns) -> {relations_csv_path}")
    print(f"- values review (fill gold_* columns) -> {values_csv_path}")
    print(f"- entity fragments reconstructed (possible NER fragmentation cases): {n_fragmented}")


if __name__ == "__main__":
    main()
