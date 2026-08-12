import torch
import torch.nn as nn
from transformers import pipeline
from config import NUM_NARRATIVES


# שכבת הלמידה של הישויות
class NarrativeEntityLayer(nn.Module):
    def __init__(self, all_entities_vocab):
        super(NarrativeEntityLayer, self).__init__()

        self.entity_vocab = all_entities_vocab
        self.vocab_size = len(all_entities_vocab)
        self.num_narratives = NUM_NARRATIVES

        # יצירת טבלת המשקולות: כל ישות מקבלת וקטור ציונים לנרטיבים
        self.entity_embeddings = nn.Embedding(self.vocab_size + 1, self.num_narratives)

        # אתחול אקראי ראשוני
        nn.init.uniform_(self.entity_embeddings.weight, 0.0, 1.0)

    def forward(self, entity_indices):
        if entity_indices.numel() == 0:
            return torch.zeros(self.num_narratives)

        # שליפת הוקטורים וחישוב ממוצע הנרטיבים של הישויות בטקסט
        vectors = self.entity_embeddings(entity_indices)
        return torch.mean(vectors, dim=0)


# ניהול חילוץ הישויות מהטקסט
class EntityAnalysisPipeline:
    def __init__(self):
        print("Loading BERT-NER Model...")
        # שימוש באסטרטגיית simple כדי לחבר חלקי שמות לישות אחת
        self.ner_pipe = pipeline(
            "ner",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple"
        )

    def extract_entities(self, text, vocab):
        """ מחלצת רשימת שמות של ישויות מהטקסט ומנקה שברים """
        results = self.ner_pipe(text)

        entities = []
        for res in results:
            word = res['word']
            # אם המילה מתחילה ב-##, אנחנו מדביקים אותה למילה הקודמת
            if word.startswith("##") and entities:
                entities[-1] = entities[-1] + word[2:]
            else:
                entities.append(word)

        # ניקוי רווחים מיותרים ש-BERT לפעמים מוסיף בחיבורים
        entities = [e.replace(" ", "") for e in entities]

        # המרה של המילים שנמצאו לאינדקסים מספריים בעזרת המילון
        indices = [vocab[ent.lower()] for ent in entities if ent.lower() in vocab]

        # הגנה חיונית: אם המשפט לא כלל ישויות מוכרות, נשים אינדקס 0 כדי שהטנזור לא יקרוס
        if len(indices) == 0:
            indices = [0]

        return torch.tensor(indices, dtype=torch.long)

    def extract_raw_entities(self, text):
        """
        Additive method (does not change extract_entities()'s existing behavior/output):
        returns the raw NER hits WITHOUT collapsing them into vocab indices, keeping the
        character offsets so callers (entity_role_tagger.py / relation_extractor.py) can
        align each entity mention to spaCy tokens for dependency-based role/relation
        extraction. Used by the Narrative Fingerprint profiler (build_profile_prototype.py),
        not by the classification models (fusion.py) which keep using extract_entities().

        Returns a list of dicts: {"text": str, "entity_group": str, "start": int, "end": int, "score": float}
        """
        results = self.ner_pipe(text)

        merged = []
        for res in results:
            word = res["word"]
            # merge continuation word-pieces ("##...") into the previous entity, same
            # defensive logic as extract_entities(), but keeping character offsets.
            if word.startswith("##") and merged:
                merged[-1]["text"] = merged[-1]["text"] + word[2:]
                merged[-1]["end"] = int(res["end"])
                merged[-1]["score"] = min(merged[-1]["score"], float(res["score"]))
            else:
                merged.append({
                    "text": word.replace(" ", ""),
                    "entity_group": res.get("entity_group", "MISC"),
                    "start": int(res["start"]),
                    "end": int(res["end"]),
                    "score": float(res["score"]),
                })

        return merged


def reconstruct_fragmented_entities(raw_entities, max_gap_chars=1):
    """
    Generic, name-agnostic post-processing pass to address NER FRAGMENTATION -
    e.g. a single proper noun (often a rare/foreign/transliterated name, like
    "Ran Goyili") gets split by dslim/bert-base-NER into multiple adjacent
    entity chunks ("Ra", "n", "Go", "yili", ...) because the token-classifier's
    B-/I- tag confidence flips mid-name at the sub-word level. This is NOT the
    same failure mode as the "##"-continuation merging already done in
    extract_raw_entities() (that handles clean word-piece continuations the HF
    pipeline's own aggregation_strategy="simple" already keeps together);
    this instead merges separately-returned entity dicts that are of the SAME
    entity_group and are adjacent in the original text (at most `max_gap_chars`
    characters apart - i.e. touching or separated by a single space), using
    ONLY character offsets. Deliberately contains no name-specific / alias
    lookups (per project convention: general, methodological fixes only, not
    hand-fixes for individual names encountered in a particular sample).

    Input: raw_entities as returned by extract_raw_entities(text) (must be
    sorted or will be sorted here by `start`).
    Returns a NEW list of merged entity dicts (same shape: text/entity_group/
    start/end/score), plus a "was_reconstructed" bool flag and
    "n_fragments_merged" int on each dict, so downstream error-analysis can
    flag/inspect cases where reconstruction fired (a proxy for likely
    fragmentation in the original NER output).
    """
    if not raw_entities:
        return []

    ordered = sorted(raw_entities, key=lambda e: e["start"])
    reconstructed = [dict(ordered[0], was_reconstructed=False, n_fragments_merged=1)]

    for ent in ordered[1:]:
        prev = reconstructed[-1]
        gap = ent["start"] - prev["end"]
        if gap <= max_gap_chars and ent["entity_group"] == prev["entity_group"]:
            separator = " " if gap > 0 else ""
            prev["text"] = prev["text"] + separator + ent["text"]
            prev["end"] = ent["end"]
            prev["score"] = min(prev["score"], ent["score"])
            prev["was_reconstructed"] = True
            prev["n_fragments_merged"] += 1
        else:
            reconstructed.append(dict(ent, was_reconstructed=False, n_fragments_merged=1))

    return reconstructed


# --- קוד בדיקה (Test) ---
if __name__ == "__main__":
    analyzer = EntityAnalysisPipeline()

    # בדיקה על משפט לדוגמה מהקונגרס
    test_text = "Zelenskyy met with Biden in Washington regarding the war in Ukraine."

    found = analyzer.extract_entities(test_text)

    print(f"\nמשפט לבדיקה: {test_text}")
    print(f"ישויות שנמצאו: {found}")