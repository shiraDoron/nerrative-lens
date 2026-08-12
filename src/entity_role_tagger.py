"""
Entity -> Role tagging MVP for the Narrative Fingerprint profiler.

Assigns one of a fixed role taxonomy (hero, victim, aggressor, betrayer, savior,
neutral) to each entity mention found in a spaCy-parsed sentence, by combining
FOUR independent signals - never a single proximity-to-keyword heuristic:

  1. dependency relation of the entity's syntactic head relative to its
     governing verb (nsubj/agent -> the entity is doing the action;
     nsubjpass/dobj/pobj -> the entity is receiving the action)
  2. the governing verb's own lemma, checked against small curated per-role
     verb lexicons (AGENT_ROLE_VERBS / PATIENT_VICTIM_VERBS below)
  3. active/passive voice of the entity's OWN clause (passive subject ->
     leans victim), analogous to emotion.py's get_agency_voice() but scoped to
     the entity's mention, not the whole sentence
  4. proximity of RHETORIC_LEXICON hits (victim-framing / delegitimization /
     pride-heroism categories, imported directly from analyze_agendas.py, same
     lexicons already used elsewhere in the project) within the entity's
     containing sentence

Each signal that fires contributes a weighted vote to a role and is recorded in
`method`; the containing sentence is kept as `evidence_text`; the combined,
capped vote score becomes `confidence`. This makes every role assignment
explainable and checkable against manual annotation (see
build_profile_prototype.py) - it is a rule-based MVP, not a trained classifier,
and is expected to be validated (and iterated on) against a small hand-labeled
sample before any larger-scale run or LLM-assisted refinement.

We deliberately do NOT force every entity mention into one of the 5 narrative
roles: if NO signal fires at all, the role is "unknown" (we have no evidence to
say anything); if signals fire but the total weighted vote is weak (below
ROLE_CONFIDENCE_THRESHOLD - i.e. only one loose signal, no corroboration), the
role is "uncertain" rather than a committed label. The highest-scoring label is
still kept in `candidate_role` in both cases, purely for error-analysis (so we
can see what the system WOULD have guessed, and which signal drove it).
"""

from collections import Counter

from analyze_agendas import RHETORIC_PATTERNS, clean_text as clean_rhetoric_text

ROLE_LABELS = ("hero", "victim", "aggressor", "betrayer", "savior", "unknown", "uncertain")

# Below this weighted-vote score, we don't commit to the top-scoring role label.
ROLE_CONFIDENCE_THRESHOLD = 0.4

# Verb lemmas that, when the ENTITY IS THE AGENT (subject), suggest a given role.
AGENT_ROLE_VERBS = {
    "aggressor": {
        "attack", "invade", "bomb", "strike", "kill", "shell", "occupy", "oppress",
        "terrorize", "destroy", "assault", "raid", "massacre", "assassinate",
        "threaten", "annex", "bombard", "besiege",
    },
    "savior": {
        "defend", "protect", "save", "rescue", "shield", "liberate", "evacuate",
        "aid", "relieve",
    },
    "hero": {
        "triumph", "win", "achieve", "overcome", "resist", "prevail", "sacrifice",
    },
    "betrayer": {
        "betray", "abandon", "deceive", "violate", "desert",
    },
}

# Verb lemmas that, when the entity is the PATIENT (object / passive subject),
# suggest it is being victimized.
PATIENT_VICTIM_VERBS = {
    "attack", "kill", "bomb", "strike", "injure", "wound", "target", "oppress",
    "displace", "starve", "massacre", "assassinate", "shell", "besiege", "betray",
    "abandon", "deceive", "occupy", "terrorize", "threaten",
}

# Generic nouns describing an adversarial/threat actor (NOT narrative-specific
# names - a small closed lexicon of role-nouns, same spirit as AGENT_ROLE_VERBS).
# When an AGENT_ROLE_VERBS["aggressor"] verb's direct object is ITSELF described
# with one of these nouns, the subject is more plausibly acting defensively/
# responsively (neutralizing a threat) rather than as the initiating aggressor -
# e.g. "The IAF struck the terrorist" frames removing a threat, not unprovoked
# aggression. Found necessary during manual calibration review (an entity was
# wrongly tagged "aggressor" at conf 0.85 for exactly this pattern).
ADVERSARY_ROLE_NOUNS = {
    "terrorist", "militant", "insurgent", "attacker", "invader", "aggressor",
    "extremist", "gunman", "hijacker", "bomber", "assailant",
}

# Verb lemmas (subset of AGENT_ROLE_VERBS["aggressor"]) eligible for the
# defensive-context override above - only verbs that plausibly denote a
# discrete strike/removal action (not e.g. "occupy"/"oppress", which don't fit
# a single-target "neutralize the threat" framing).
_DEFENSIVE_OVERRIDE_VERBS = {"strike", "kill", "shell", "assault", "raid", "bombard"}

# First-person plural pronouns - treated as an explicit "in-group" candidate
# actor (the narrator's own collective voice), since HF NER never tags pronouns.
FIRST_PERSON_PLURAL_PRONOUNS = {"we", "us", "our", "ours", "ourselves"}

_ALL_AGENT_ROLE_VERBS = set().union(*AGENT_ROLE_VERBS.values())
# Union of every verb lemma this module already treats as role-bearing (agent
# OR patient side) - used to find CANDIDATE conceptual actors (common nouns
# that aren't proper-noun NER entities) via the EXACT SAME dependency+verb-
# lexicon signal already used by tag_entity_role(), rather than a new/separate
# per-word heuristic.
_ROLE_BEARING_VERB_LEMMAS = _ALL_AGENT_ROLE_VERBS | PATIENT_VICTIM_VERBS

# Which RHETORIC_LEXICON category (from analyze_agendas.py) supports which role
# if it fires in the same sentence as the entity mention.
_ROLE_TO_RHETORIC_CATEGORY = {
    "victim": "מסגור קורבנות וסבל",
    "aggressor": "דה-לגיטימציה של היריב",
    "hero": "גאווה, הישג והרואיות",
    "savior": "גאווה, הישג והרואיות",
    "betrayer": "דה-לגיטימציה של היריב",
}

SIGNAL_WEIGHTS = {
    "dependency": 0.35,
    "verb_lexicon": 0.35,
    "agency_voice": 0.15,
    "rhetoric_lexicon": 0.15,
}

AGENT_DEPS = ("nsubj", "agent")
PATIENT_DEPS = ("nsubjpass", "dobj", "pobj", "iobj", "dative")


def _find_governing_verb(token, max_hops=5):
    """Walk up the dependency tree from `token` to find the nearest verb/aux it depends on."""
    head = token.head
    hops = 0
    while head.pos_ not in ("VERB", "AUX") and head.head != head and hops < max_hops:
        head = head.head
        hops += 1
    return head if head.pos_ in ("VERB", "AUX") else None


def _agency_from_dep(dep):
    """Maps a dependency label to a coarse observed grammatical agency for reporting."""
    if dep in AGENT_DEPS:
        return "active"
    if dep in PATIENT_DEPS:
        return "passive"
    return "unknown"


def _get_object(verb_token):
    """Returns the direct object (or prepositional object) token of a verb, if
    any. Also checks conjoined verbs sharing the same object (e.g. "struck
    and eliminated the terrorist" - "terrorist" attaches syntactically only
    to "eliminated", but "struck" needs to see it too) - a general fix for
    coordinate VP structures, not specific to any narrative/verb."""
    verbs = [verb_token] + [c for c in verb_token.children if c.dep_ == "conj"]
    if verb_token.dep_ == "conj":
        verbs.append(verb_token.head)
    for v in verbs:
        for child in v.children:
            if child.dep_ in ("dobj", "attr", "oprd"):
                return child
    for v in verbs:
        for child in v.children:
            if child.dep_ == "prep":
                for grandchild in child.children:
                    if grandchild.dep_ == "pobj":
                        return grandchild
    return None


def _verb_object_is_adversary(verb_token):
    """True if `verb_token`'s direct object noun phrase is headed by (or
    contains) one of ADVERSARY_ROLE_NOUNS - see comment on that lexicon above."""
    obj = _get_object(verb_token)
    if obj is None:
        return False
    for tok in obj.subtree:
        if tok.lemma_.lower() in ADVERSARY_ROLE_NOUNS:
            return True
    return False


def find_conceptual_actor_spans(doc, existing_token_is):
    """Finds noun-phrase / pronoun spans that plausibly denote a narrative
    actor but are NOT proper-noun NER entities. The HF NER model only tags
    PER/ORG/LOC/MISC proper nouns, missing collective/abstract actors central
    to a lot of narrative framing ("unions", "politicians", "the media",
    "checkpoints") and the narrator's own in-group ("we"/"us"/"our"). Found to
    be the single biggest actor-coverage gap during manual calibration review
    (many texts had zero detected actors despite clear role-bearing language).

    Candidates (both syntax-driven, NOT a per-word/per-name lookup):
      1. First-person plural pronouns (we/us/our/ours/ourselves).
      2. Any NOUN/PROPN noun-chunk whose root token is in agent-position
         (nsubj/agent) or DIRECT patient-position (nsubjpass/dobj - NOT the
         looser pobj/iobj/dative used elsewhere in this module) of a verb
         already in this module's OWN AGENT_ROLE_VERBS/PATIENT_VICTIM_VERBS
         lexicons. Deliberately narrower than tag_entity_role()'s own
         PATIENT_DEPS: a bare prepositional object is often a temporal/
         locative adjunct rather than a true affected entity (found as a real
         false positive during validation - "attack... at night" tagged
         "night" itself as a victim via the "pobj" of "attack").

    `existing_token_is`: a set of token.i already covered by a recognized NER
    entity, so this doesn't produce duplicates of what NER already found.

    Returns a list of (span, span_type) tuples, span_type one of
    "in_group_pronoun" | "conceptual_noun_phrase".
    """
    _CONCEPTUAL_PATIENT_DEPS = ("nsubjpass", "dobj")
    candidates = []

    for token in doc:
        if token.i in existing_token_is:
            continue
        if token.pos_ == "PRON" and token.lower_ in FIRST_PERSON_PLURAL_PRONOUNS:
            candidates.append((doc[token.i:token.i + 1], "in_group_pronoun"))

    seen_root_is = set()
    for chunk in doc.noun_chunks:
        root = chunk.root
        if root.pos_ not in ("NOUN", "PROPN"):
            continue
        if root.i in seen_root_is or any(tok.i in existing_token_is for tok in chunk):
            continue
        verb = _find_governing_verb(root)
        if verb is None:
            continue
        dep = root.dep_
        if dep not in AGENT_DEPS and dep not in _CONCEPTUAL_PATIENT_DEPS:
            continue
        if verb.lemma_.lower() not in _ROLE_BEARING_VERB_LEMMAS:
            continue
        candidates.append((chunk, "conceptual_noun_phrase"))
        seen_root_is.add(root.i)

    return candidates


def tag_entity_role(entity_span):
    """
    entity_span: a spaCy Span covering the entity mention (must belong to a Doc
                 that has been parsed - i.e. produced by a pipeline with a parser,
                 like en_core_web_trf).

    Returns a dict:
      {"role": str, "confidence": float, "evidence_text": str, "method": str,
       "agency": str, "candidate_role": str}
    "role" is one of hero/victim/aggressor/betrayer/savior/unknown/uncertain -
    "unknown" means no signal fired at all, "uncertain" means signals fired but
    too weak/uncorroborated to commit to a label (see ROLE_CONFIDENCE_THRESHOLD).
    "candidate_role" is always the raw top-scoring label (even when role is
    unknown/uncertain), kept for error-analysis only - do NOT treat it as a
    prediction on its own.
    """
    if entity_span is None or len(entity_span) == 0:
        return {
            "role": "unknown", "confidence": 0.0, "evidence_text": "", "method": "no_span",
            "agency": "unknown", "candidate_role": "unknown",
        }

    head_token = entity_span.root
    sent = entity_span.sent
    dep = head_token.dep_
    agency = _agency_from_dep(dep)

    verb = _find_governing_verb(head_token)
    role_votes = Counter()
    signals_fired = []

    is_agent_position = dep in AGENT_DEPS
    is_patient_position = dep in PATIENT_DEPS

    if verb is not None:
        verb_lemma = verb.lemma_.lower()

        if is_agent_position:
            for role, verbs in AGENT_ROLE_VERBS.items():
                if verb_lemma in verbs:
                    if (role == "aggressor" and verb_lemma in _DEFENSIVE_OVERRIDE_VERBS
                            and _verb_object_is_adversary(verb)):
                        # Defensive-context override: the verb's own object is
                        # itself framed as the adversary/threat (e.g. "struck
                        # the terrorist") - treat the subject as responding to
                        # a threat, not initiating aggression. See
                        # ADVERSARY_ROLE_NOUNS comment above.
                        role_votes["savior"] += SIGNAL_WEIGHTS["dependency"] + SIGNAL_WEIGHTS["verb_lexicon"]
                        signals_fired.append(
                            f"dependency:{dep}+verb_lexicon:aggressor_defensive_override({verb_lemma})"
                        )
                    else:
                        role_votes[role] += SIGNAL_WEIGHTS["dependency"] + SIGNAL_WEIGHTS["verb_lexicon"]
                        signals_fired.append(f"dependency:{dep}+verb_lexicon:{role}({verb_lemma})")

        if is_patient_position and verb_lemma in PATIENT_VICTIM_VERBS:
            role_votes["victim"] += SIGNAL_WEIGHTS["dependency"] + SIGNAL_WEIGHTS["verb_lexicon"]
            signals_fired.append(f"dependency:{dep}+verb_lexicon:victim({verb_lemma})")

    # agency-voice signal, scoped to this entity's own clause (not the whole sentence).
    if dep == "nsubjpass":
        role_votes["victim"] += SIGNAL_WEIGHTS["agency_voice"]
        signals_fired.append("agency_voice:passive_subject")
    elif dep == "nsubj" and role_votes:
        for role in ("aggressor", "hero", "savior"):
            if role in role_votes:
                role_votes[role] += SIGNAL_WEIGHTS["agency_voice"]
        signals_fired.append("agency_voice:active_subject")

    # rhetoric-lexicon proximity within the containing sentence.
    cleaned_sent = clean_rhetoric_text(sent.text)
    for role, category in _ROLE_TO_RHETORIC_CATEGORY.items():
        pattern = RHETORIC_PATTERNS.get(category)
        if pattern is not None and pattern.search(cleaned_sent):
            role_votes[role] += SIGNAL_WEIGHTS["rhetoric_lexicon"]
            signals_fired.append(f"rhetoric_lexicon:{category}")

    if not role_votes:
        return {
            "role": "unknown",
            "confidence": 0.0,
            "evidence_text": sent.text.strip(),
            "method": "no_signal",
            "agency": agency,
            "candidate_role": "unknown",
        }

    best_role, best_score = max(role_votes.items(), key=lambda kv: kv[1])
    confidence = round(min(best_score, 1.0), 3)
    final_role = best_role if best_score >= ROLE_CONFIDENCE_THRESHOLD else "uncertain"
    return {
        "role": final_role,
        "confidence": confidence,
        "evidence_text": sent.text.strip(),
        "method": "+".join(signals_fired),
        "agency": agency,
        "candidate_role": best_role,
    }


if __name__ == "__main__":
    import spacy

    nlp = spacy.load("en_core_web_trf")
    samples = [
        "Hamas attacked innocent civilians in a brutal massacre.",
        "The IDF defended the border and rescued the hostages.",
        "Ukrainian civilians were displaced by the invasion.",
    ]
    for text in samples:
        doc = nlp(text)
        print(f"\nText: {text}")
        for ent in doc.ents:
            print(f"  {ent.text} ({ent.label_}) -> {tag_entity_role(ent)}")
        # also tag common nouns like "civilians" which spaCy NER may not tag as an entity
        for token in doc:
            if token.lemma_.lower() == "civilians" or token.text.lower() == "civilians":
                span = doc[token.i:token.i + 1]
                print(f"  {span.text} (NOUN) -> {tag_entity_role(span)}")
