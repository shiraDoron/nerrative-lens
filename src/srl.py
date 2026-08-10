import spacy
import torch
import torch.nn as nn
from config import NUM_NARRATIVES


class SRLNarrativeLayer(nn.Module):
    """
    שכבה לומדת המקשרת בין מילים המייצגות סיבה/מטרה לבין וקטור הנרטיבים.
    """

    def __init__(self, reason_vocab):
        super(SRLNarrativeLayer, self).__init__()
        self.num_narratives = NUM_NARRATIVES

        # יצירת מטריצה לומדת בגודל (מספר המילים במילון הסיבות + 1 עבור UNK) X (10 נרטיבים)
        self.reason_embeddings = nn.Embedding(len(reason_vocab) + 1, self.num_narratives)

        nn.init.xavier_uniform_(self.reason_embeddings.weight)

    def forward(self, reason_indices):
        # אם אין סיבות או מטרות במשפט, ההשפעה של הרכיב הזה היא אפס
        if reason_indices.numel() == 0:
            return torch.zeros(self.num_narratives)

        # שליפת הוקטורים עבור הסיבות שנמצאו וביצוע ממוצע
        reason_vectors = self.reason_embeddings(reason_indices)
        return torch.mean(reason_vectors, dim=0)


class SRLProcessor:
    """
    מנוע לחילוץ מילים המשמשות כסיבה או מטרה במשפט באמצעות ניתוח תלויות.
    """

    def __init__(self):
        print("Loading spaCy Transformer Model for Reason Extraction...")
        self.nlp = spacy.load("en_core_web_trf")

    def extract_features(self, text, reason_vocab):
        doc = self.nlp(text)
        reason_ids = []

        for token in doc:
            # המודל מחפש תגיות של adverbial clause modifier (סיבה/מטרה)
            if token.dep_ == "advcl":
                lemma = token.lemma_.lower()
                # משיכת האינדקס מהמילון, או אינדקס UNK (אחרון) אם המילה חדשה
                r_id = reason_vocab.get(lemma, len(reason_vocab))
                reason_ids.append(r_id)

        return torch.tensor(reason_ids, dtype=torch.long)


# --- קוד בדיקה ---
if __name__ == "__main__":
    processor = SRLProcessor()

    # מילון דוגמה (במערכת האמיתית ייבנה מראש מהדאטה-סט)
    sample_vocab = {"liberate": 0, "protect": 1, "destroy": 2}
    layer = SRLNarrativeLayer(sample_vocab)

    test_sentences = [
        "The army attacked to liberate the city.",
        "They protested because they wanted freedom.",
        "The government simply resigned."
    ]

    print("\n--- הרצת חילוץ מטרות וסיבות ---")
    for text in test_sentences:
        r_ids = processor.extract_features(text, sample_vocab)
        vector = layer(r_ids).detach().numpy()

        print(f"טקסט: {text}")
        print(f"אינדקסים שנמצאו: {r_ids.tolist()}")
        if r_ids.numel() > 0:
            print(f"וקטור נרטיב (ראשיתו): {vector[:5]}...\n")
        else:
            print("לא נמצאו מטרות/סיבות. הוחזר וקטור אפסים.\n")