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

# --- קוד בדיקה (Test) ---
if __name__ == "__main__":
    analyzer = EntityAnalysisPipeline()

    # בדיקה על משפט לדוגמה מהקונגרס
    test_text = "Zelenskyy met with Biden in Washington regarding the war in Ukraine."

    found = analyzer.extract_entities(test_text)

    print(f"\nמשפט לבדיקה: {test_text}")
    print(f"ישויות שנמצאו: {found}")