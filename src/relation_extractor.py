"""
Relation extraction MVP for the Narrative Fingerprint profiler.

Extracts a small, fixed set of explicit, measurable relation types between
entities/actors in a sentence, using spaCy dependency parsing plus small
curated verb/connective lexicons (one per relation type). This is NOT a full
semantic-role-labeling system - it is enough to answer "who does what to whom"
for the relation types the NarrativeProfile schema needs:

    causes, threatens, blames, protects, attacks, supports, opposes, proposes_action

Every extracted relation carries `confidence` (higher when both source and
target are recognized entities, lower for looser/connective-based patterns)
and `evidence` (the source sentence), plus `method` documenting which
lexicon/pattern fired - so every relation is explainable and checkable against
manual annotation (see build_profile_prototype.py).
"""

import re

RELATION_TYPES = (
    "causes", "threatens", "blames", "protects",
    "attacks", "supports", "opposes", "proposes_action",
)

# Below this confidence, we don't commit to the specific relation type - it's
# demoted to "uncertain" (the originally-guessed type is kept in
# "candidate_relation" for error-analysis only, not as a prediction).
RELATION_CONFIDENCE_THRESHOLD = 0.4

# Direct verb-based relations: entity-as-subject "verb"s entity-as-object.
VERB_RELATION_LEXICON = {
    "attacks": {"attack", "strike", "bomb", "invade", "shell", "raid", "assault", "storm"},
    "threatens": {"threaten", "warn", "intimidate", "menace"},
    "protects": {"protect", "defend", "shield", "safeguard", "rescue", "save"},
    "supports": {"support", "back", "endorse", "aid", "assist", "help", "fund", "champion"},
    "opposes": {"oppose", "condemn", "denounce", "reject", "resist", "criticize", "slam"},
    "blames": {"blame", "accuse", "fault"},
}

# Phrase-based fallback for "blames" when there's no explicit blaming subject
# (e.g. "X is responsible for Y" attributes fault to X without naming who blames it).
BLAME_PHRASE_TRIGGERS = ("responsible for", "to blame for", "at fault for")

# Causal connectives/phrases linking two clauses. Direction matters: some
# phrases put the cause BEFORE the phrase, others put it AFTER.
CAUSAL_PHRASE_DIRECTIONS = {
    "because of": "cause_after",
    "due to": "cause_after",
    "as a result of": "cause_after",
    "led to": "cause_before",
    "leads to": "cause_before",
    "resulted in": "cause_before",
    "results in": "cause_before",
}
CAUSAL_MARK_CONNECTIVES = {"because", "since", "as", "so"}

# Modal/action-proposal cues ("we must act", "leaders call for a ceasefire").
PROPOSAL_MODALS = {"must", "should", "ought"}
PROPOSAL_VERB_TRIGGERS = {"call", "urge", "demand", "propose", "insist", "push", "need"}


def _entity_text_for_token(token, entity_spans_by_token_i):
    """Return the normalized entity text covering `token`, if any; else the token's own text."""
    return entity_spans_by_token_i.get(token.i, token.text)


def _get_subject(verb_token):
    """Returns (subject_token_or_None, is_passive)."""
    for child in verb_token.children:
        if child.dep_ in ("nsubj", "nsubjpass"):
            return child, (child.dep_ == "nsubjpass")
    return None, False


def _get_object(verb_token):
    for child in verb_token.children:
        if child.dep_ in ("dobj", "attr", "oprd"):
            return child
    for child in verb_token.children:
        if child.dep_ == "prep":
            for grandchild in child.children:
                if grandchild.dep_ == "pobj":
                    return grandchild
    return None


def _get_passive_agent(verb_token):
    for child in verb_token.children:
        if child.dep_ == "prep" and child.lemma_.lower() == "by":
            for grandchild in child.children:
                if grandchild.dep_ == "pobj":
                    return grandchild
    return None


def _extract_verb_relations(sent, entity_spans_by_token_i):
    relations = []
    sent_text = sent.text.strip()

    for token in sent:
        if token.pos_ not in ("VERB", "AUX"):
            continue
        lemma = token.lemma_.lower()

        matched_relation = None
        for relation, verbs in VERB_RELATION_LEXICON.items():
            if lemma in verbs:
                matched_relation = relation
                break
        if matched_relation is None:
            continue

        subj, is_passive = _get_subject(token)
        obj = _get_object(token)
        passive_agent = _get_passive_agent(token) if is_passive else None

        if is_passive and passive_agent is not None and subj is not None:
            source_tok, target_tok = passive_agent, subj
        elif not is_passive and subj is not None and obj is not None:
            source_tok, target_tok = subj, obj
        else:
            continue

        source = _entity_text_for_token(source_tok, entity_spans_by_token_i)
        target = _entity_text_for_token(target_tok, entity_spans_by_token_i)
        both_known_entities = (source_tok.i in entity_spans_by_token_i and
                                target_tok.i in entity_spans_by_token_i)
        relations.append({
            "source": source, "relation": matched_relation, "target": target,
            "confidence": 0.7 if both_known_entities else 0.45,
            "evidence": sent_text,
            "method": f"verb_lexicon:{lemma}+dependency:{'passive' if is_passive else 'active'}",
        })

    return relations


def _extract_proposal_relations(sent, entity_spans_by_token_i):
    relations = []
    sent_text = sent.text.strip()

    for token in sent:
        lemma = token.lemma_.lower()
        is_trigger_verb = token.pos_ == "VERB" and lemma in PROPOSAL_VERB_TRIGGERS
        is_trigger_modal = token.pos_ == "AUX" and lemma in PROPOSAL_MODALS
        if not (is_trigger_verb or is_trigger_modal):
            continue

        main_verb = token if token.pos_ != "AUX" else token.head
        subj, _ = _get_subject(main_verb)
        if subj is None:
            continue

        action_tokens = [t.text for t in main_verb.subtree if t.i >= main_verb.i]
        action_phrase = " ".join(action_tokens)[:120]
        source = _entity_text_for_token(subj, entity_spans_by_token_i)
        relations.append({
            "source": source, "relation": "proposes_action", "target": action_phrase,
            "confidence": 0.55 if subj.i in entity_spans_by_token_i else 0.35,
            "evidence": sent_text,
            "method": f"proposal_trigger:{lemma}",
        })

    return relations


def _strip_trailing_copula(text):
    return re.sub(r"\s+(is|was|are|were|be|being|been)\s*$", "", text, flags=re.IGNORECASE)


def _extract_blame_phrase_relations(sent):
    sent_text = sent.text.strip()
    lower_sent = sent_text.lower()
    relations = []
    for phrase in BLAME_PHRASE_TRIGGERS:
        idx = lower_sent.find(phrase)
        if idx == -1:
            continue
        target_part = _strip_trailing_copula(sent_text[:idx].strip(" ,."))
        if target_part:
            relations.append({
                "source": "(unattributed)", "relation": "blames", "target": target_part[:120],
                "confidence": 0.35, "evidence": sent_text,
                "method": f"blame_phrase:{phrase}",
            })
        break
    return relations


def _extract_causal_phrase_relations(sent):
    sent_text = sent.text.strip()
    lower_sent = sent_text.lower()
    relations = []
    for phrase, direction in CAUSAL_PHRASE_DIRECTIONS.items():
        idx = lower_sent.find(phrase)
        if idx == -1:
            continue
        before = sent_text[:idx].strip(" ,.")
        after = sent_text[idx + len(phrase):].strip(" ,.")
        if direction == "cause_after":
            cause_part, effect_part = after, before
        else:
            cause_part, effect_part = before, after
        if cause_part and effect_part:
            relations.append({
                "source": cause_part[:120], "relation": "causes", "target": effect_part[:120],
                "confidence": 0.4, "evidence": sent_text,
                "method": f"causal_phrase:{phrase}",
            })
        break
    return relations


def _extract_causal_connective_relations(sent, entity_spans_by_token_i):
    relations = []
    sent_text = sent.text.strip()
    for token in sent:
        if token.dep_ != "mark" or token.lemma_.lower() not in CAUSAL_MARK_CONNECTIVES:
            continue
        advcl_verb = token.head
        main_verb = advcl_verb.head
        if advcl_verb == main_verb:
            continue
        cause_subj, _ = _get_subject(advcl_verb)
        effect_subj, _ = _get_subject(main_verb)
        if cause_subj is None or effect_subj is None:
            continue
        relations.append({
            "source": _entity_text_for_token(cause_subj, entity_spans_by_token_i),
            "relation": "causes",
            "target": _entity_text_for_token(effect_subj, entity_spans_by_token_i),
            "confidence": 0.5, "evidence": sent_text,
            "method": f"causal_connective:{token.lemma_.lower()}+dependency:advcl",
        })
    return relations


_ALPHA_RUN_RE = re.compile(r"[A-Za-z]{2,}")


def _is_meaningful_span(text):
    """Rejects source/target spans with no real alphabetic content (e.g. a bare
    "%" or other punctuation/number token grabbed as a dependency-parse subject
    when the true head noun isn't a recognized entity) - found during manual
    calibration review to produce meaningless relations like '% -> causes -> %'.
    General, content-based check - not tied to any specific name/token."""
    return bool(text) and bool(_ALPHA_RUN_RE.search(text))


def extract_relations(sent, entity_spans_by_token_i):
    """
    sent: a spaCy sentence Span (from doc.sents), belonging to a Doc parsed with
          a dependency parser (e.g. en_core_web_trf).
    entity_spans_by_token_i: dict mapping token.i -> normalized entity text, for
          every token that is part of a detected+normalized entity mention
          (built by build_profile_prototype.py from the NER + entity_normalizer output).

    Returns a list of relation dicts:
      {"source": str, "relation": str, "target": str, "confidence": float,
       "evidence": str, "method": str, "candidate_relation": str}
    "relation" is one of RELATION_TYPES or "uncertain" (confidence below
    RELATION_CONFIDENCE_THRESHOLD - we don't force a specific relation type when
    the evidence is this weak). "candidate_relation" always holds the raw
    originally-matched type, even when relation=="uncertain", for error analysis.
    """
    relations = []
    relations.extend(_extract_verb_relations(sent, entity_spans_by_token_i))
    relations.extend(_extract_proposal_relations(sent, entity_spans_by_token_i))
    relations.extend(_extract_blame_phrase_relations(sent))
    relations.extend(_extract_causal_phrase_relations(sent))
    relations.extend(_extract_causal_connective_relations(sent, entity_spans_by_token_i))

    relations = [
        rel for rel in relations
        if _is_meaningful_span(rel["source"]) and _is_meaningful_span(rel["target"])
    ]

    for rel in relations:
        rel["candidate_relation"] = rel["relation"]
        if rel["confidence"] < RELATION_CONFIDENCE_THRESHOLD:
            rel["relation"] = "uncertain"

    return relations


if __name__ == "__main__":
    import spacy

    nlp = spacy.load("en_core_web_trf")
    samples = [
        "Hamas attacked Israeli civilians near the border.",
        "The IDF protects Israeli citizens from rocket attacks.",
        "World leaders condemn the invasion of Ukraine.",
        "Iran is responsible for the escalation in the region.",
        "The economy collapsed because of the sanctions.",
        "The sanctions led to a severe economic crisis.",
        "We must act now to stop the violence.",
    ]
    for text in samples:
        doc = nlp(text)
        print(f"\nText: {text}")
        for sent in doc.sents:
            for rel in extract_relations(sent, {}):
                print(f"  {rel}")
