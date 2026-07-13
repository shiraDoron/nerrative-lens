import torch
import torch.nn as nn
from transformers import pipeline
import spacy
from config import NUM_NARRATIVES

# פונקציה זו בעצם מזהה
# אלו נרטיבים יותר מתאפיינים בקורבנות
# אלו יותר בדחיפה לפעולה וכדומה


class EmotionAgencyLayer(nn.Module):
    def __init__(self):
        super(EmotionAgencyLayer, self).__init__()

        # המודל מחזיר 6 רגשות
        self.emotions_map = ["joy", "sadness", "anger", "fear", "surprise", "neutral"]
        self.num_emotions = len(self.emotions_map)
        self.agency_types = 2  # 0: Passive, 1: Active

        # גודל מטריצה: 12 שילובים אפשריים (6 רגשות * 2 סוגי פעילות)
        self.input_size = self.num_emotions * self.agency_types
        self.num_narratives = NUM_NARRATIVES

        self.emotion_agency_matrix = nn.Embedding(self.input_size, self.num_narratives)
        nn.init.xavier_uniform_(self.emotion_agency_matrix.weight)

    def forward(self, emotion_indices, agency_flags):
        """
        חישוב אינדקס משולב: (Emotion * 2) + Agency
        """
        # וידוא שהקלט הוא Tensor
        indices = (emotion_indices * self.agency_types) + agency_flags.long()
        return self.emotion_agency_matrix(indices)


class EmotionAgencyProcessor:
    def __init__(self):
        print("Loading Emotion Classifier & spaCy for Voice Detection...")
        # טעינת מודל הרגשות
        self.emotion_pipe = pipeline("text-classification",
                                     model="michellejieli/emotion_text_classifier")
        # שימוש במודל spaCy שכבר יש לנו מה-SRL
        self.nlp = spacy.load("en_core_web_trf")

        self.emotions_map = ["joy", "sadness", "anger", "fear", "surprise", "neutral"]

    def get_agency_voice(self, text):
        """
        מזהה האם המשפט הוא Active (1) או Passive (0)
        לפי נוכחות של nsubjpass (נושא סביל) בעץ התחבירי.
        """
        doc = self.nlp(text)
        for token in doc:
            if token.dep_ in ("nsubjpass", "auxpass"):
                return 0  # Passive (קורבן)
        return 1  # Active (פועל)

    def extract_features(self, text):
        # 1. זיהוי רגש
        pred = self.emotion_pipe(text, truncation=True, max_length=512)[0]
        label = pred['label']
        emotion_idx = self.emotions_map.index(label) if label in self.emotions_map else 5

        # 2. זיהוי סוכנות (פעיל/סביל)
        agency_flag = self.get_agency_voice(text)

        return torch.tensor([emotion_idx]), torch.tensor([agency_flag])


# --- טסט מהיר לווידוא תקינות ---
if __name__ == "__main__":
    processor = EmotionAgencyProcessor()
    layer = EmotionAgencyLayer()

    test_texts = [
        "The peaceful protesters were attacked by the guards.",  # סביל + עצב/פחד?
        "We will celebrate our great victory with joy!",  # פעיל + שמחה
        "The government failed to protect the citizens."  # פעיל + כעס/עצב
    ]

    print("\n--- ניתוח רגשות וסוכנות ---")
    for text in test_texts:
        e_idx, a_flag = processor.extract_features(text)
        vector = layer(e_idx, a_flag).detach().numpy()

        voice = "Active" if a_flag.item() == 1 else "Passive"
        emotion = processor.emotions_map[e_idx.item()]

        print(f"טקסט: {text}")
        print(f"זיהוי: {emotion} | קול: {voice}")
        print(f"וקטור נרטיבי (ראשיתו): {vector[0][:5]}...\n")