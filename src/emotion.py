import torch
import torch.nn as nn
from transformers import pipeline
import spacy
from config import NUM_NARRATIVES

# This module essentially identifies which narratives are more characterized
# by victimhood framing, and which lean more toward a call-to-action framing,
# and so on.


class EmotionAgencyLayer(nn.Module):
    def __init__(self):
        super(EmotionAgencyLayer, self).__init__()

        # The model returns 6 emotions
        self.emotions_map = ["joy", "sadness", "anger", "fear", "surprise", "neutral"]
        self.num_emotions = len(self.emotions_map)
        self.agency_types = 2  # 0: Passive, 1: Active

        # Matrix size: 12 possible combinations (6 emotions * 2 agency types)
        self.input_size = self.num_emotions * self.agency_types
        self.num_narratives = NUM_NARRATIVES

        self.emotion_agency_matrix = nn.Embedding(self.input_size, self.num_narratives)
        nn.init.xavier_uniform_(self.emotion_agency_matrix.weight)

    def forward(self, emotion_indices, agency_flags):
        """
        Computes a combined index: (Emotion * 2) + Agency
        """
        # Ensure the input is a Tensor
        indices = (emotion_indices * self.agency_types) + agency_flags.long()
        return self.emotion_agency_matrix(indices)


class EmotionAgencyProcessor:
    def __init__(self):
        print("Loading Emotion Classifier & spaCy for Voice Detection...")
        # Load the emotion classification model
        self.emotion_pipe = pipeline("text-classification",
                                     model="michellejieli/emotion_text_classifier")
        # Reuse the spaCy model we already have from the SRL module
        self.nlp = spacy.load("en_core_web_trf")

        self.emotions_map = ["joy", "sadness", "anger", "fear", "surprise", "neutral"]

    def get_agency_voice(self, text):
        """
        Determines whether the sentence is Active (1) or Passive (0),
        based on the presence of nsubjpass (passive subject) in the parse tree.
        """
        doc = self.nlp(text)
        for token in doc:
            if token.dep_ in ("nsubjpass", "auxpass"):
                return 0  # Passive (victim)
        return 1  # Active (agent)

    def extract_features(self, text):
        # 1. Emotion detection
        pred = self.emotion_pipe(text, truncation=True, max_length=512)[0]
        label = pred['label']
        emotion_idx = self.emotions_map.index(label) if label in self.emotions_map else 5

        # 2. Agency detection (active/passive)
        agency_flag = self.get_agency_voice(text)

        return torch.tensor([emotion_idx]), torch.tensor([agency_flag])


# --- Quick sanity-check test ---
if __name__ == "__main__":
    processor = EmotionAgencyProcessor()
    layer = EmotionAgencyLayer()

    test_texts = [
        "The peaceful protesters were attacked by the guards.",  # Passive + sadness/fear?
        "We will celebrate our great victory with joy!",  # Active + joy
        "The government failed to protect the citizens."  # Active + anger/sadness
    ]

    print("\n--- Emotion and agency analysis ---")
    for text in test_texts:
        e_idx, a_flag = processor.extract_features(text)
        vector = layer(e_idx, a_flag).detach().numpy()

        voice = "Active" if a_flag.item() == 1 else "Passive"
        emotion = processor.emotions_map[e_idx.item()]

        print(f"Text: {text}")
        print(f"Detected: {emotion} | Voice: {voice}")
        print(f"Narrative vector (start): {vector[0][:5]}...\n")