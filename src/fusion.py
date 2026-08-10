#import torch
#import torch.nn as nn
#import json
#from bertopic import BERTopic
#from tqdm import tqdm
#import transformers
#import logging
#
## ייבוא כל הרכיבים שבנינו מהקבצים השונים
#from ner import NarrativeEntityLayer, EntityAnalysisPipeline
#from srl import SRLProcessor, SRLNarrativeLayer
#from emotion import EmotionAgencyProcessor, EmotionAgencyLayer
#from reliability import ReliabilityProcessor, ReliabilityLayer
#from stance import TopicStanceLayer, TopicAnalysisPipeline
#from config import NUM_NARRATIVES, NARRATIVES
#
#transformers.logging.set_verbosity_error()
#logging.getLogger("transformers").setLevel(logging.ERROR)
#
#
## רשת MLP
#class NarrativeFusionNetwork(nn.Module):
#    """
#    רשת האיחוד המשודרגת (MLP): משרשרת את הוקטורים מכל מודל,
#    לומדת קשרים לא-ליניאריים ביניהם, ומפעילה את פקטור האמינות לקבלת החלטה סופית.
#    """
#
#    def __init__(self):
#        super(NarrativeFusionNetwork, self).__init__()
#
#        # אנחנו מקבלים 4 וקטורים, כל אחד בגודל של מספר הנרטיבים
#        input_size = 4 * NUM_NARRATIVES
#
#        # גודל השכבה הנסתרת
#        hidden_size = 64
#
#        # בניית הרשת העצבית
#        self.mlp = nn.Sequential(
#            nn.Linear(input_size, hidden_size),
#            nn.ReLU(),
#            nn.Dropout(0.3),
#            nn.Linear(hidden_size, NUM_NARRATIVES)
#        )
#
#        self.softmax = nn.Softmax(dim=-1)
#
#    def forward(self, vec_ner, vec_stance, vec_srl, vec_emotion, weight_factor):
#        # שרשור כל הוקטורים לטנזור אחד
#        combined_features = torch.cat(
#            [vec_ner.squeeze(), vec_stance.squeeze(), vec_srl.squeeze(), vec_emotion.squeeze()], dim=-1)
#
#        # העברה דרך הרשת העצבית הלומדת
#        fused_logits = self.mlp(combined_features)
#
#        # הפעלת פקטור האמינות
#        amplified_signal = fused_logits * weight_factor
#
#        # המרה להסתברויות
#        return self.softmax(amplified_signal)
#
#
#class NarrativeDetector(nn.Module):
#    """
#    המערכת המלאה (Pipeline) שאורזת את כל הרכיבים הלומדים והמעבדים.
#    """
#
#    def __init__(self, ner_vocab, srl_vocab):
#        super(NarrativeDetector, self).__init__()
#
#        print("Initializing Full Narrative Detection Pipeline...")
#
#        pbar = tqdm(total=5, desc="Loading Models",
#                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
#
#        self.ner_processor = EntityAnalysisPipeline()
#        pbar.update(1)
#        self.srl_processor = SRLProcessor()
#        pbar.update(1)
#        self.emotion_processor = EmotionAgencyProcessor()
#        pbar.update(1)
#        self.stance_processor = TopicAnalysisPipeline()  # מודול הנושאים המעודכן
#        pbar.update(1)
#        self.reliability_processor = ReliabilityProcessor()
#        pbar.update(1)
#
#        pbar.close()
#
#        # השכבות הלומדות בתוך PyTorch
#        self.ner_layer = NarrativeEntityLayer(ner_vocab)
#        self.srl_layer = SRLNarrativeLayer(srl_vocab)
#        self.emotion_layer = EmotionAgencyLayer()
#        self.stance_layer = TopicStanceLayer()  # השכבה שקולטת כעת נושאים בלבד
#        self.reliability_layer = ReliabilityLayer()
#
#        # רשת האיחוד
#        self.fusion_network = NarrativeFusionNetwork()
#
#        self.ner_vocab = ner_vocab
#        self.srl_vocab = srl_vocab
#
#    def extract_features(self, text):
#        return {
#            "ner": self.ner_processor.extract_entities(text, self.ner_vocab),
#            "srl": self.srl_processor.extract_features(text, self.srl_vocab),
#            "emotion": self.emotion_processor.extract_features(text),
#            "stance": self.stance_processor.process_text(text),  # מחזיר כעת רק topic_id
#            "reliability": self.reliability_processor.extract_features(text)
#        }
#
#    def classify_features(self, features):
#        vec_ner = self.ner_layer(features["ner"])
#        vec_srl = self.srl_layer(features["srl"])
#
#        emotion_idx, agency_flag = features["emotion"]
#        vec_emotion = self.emotion_layer(emotion_idx, agency_flag)
#
#        # --- התיקון המרכזי למניעת קריסה ---
#        topic_id = features["stance"]  # קבלת מזהה הנושא בלבד (ללא stance_label)
#        topic_tensor = torch.tensor([topic_id], dtype=torch.long)
#
#        # העברת ה-topic_tensor בלבד לשכבה המעודכנת
#        vec_stance = self.stance_layer(topic_tensor)
#
#        weight_factor = self.reliability_layer(features["reliability"])
#
#        final_probs = self.fusion_network(vec_ner, vec_stance, vec_srl, vec_emotion, weight_factor)
#        return final_probs
#
#    def forward(self, text):
#        features = self.extract_features(text)
#        return self.classify_features(features)
#
#
## --- סימולציה של ריצת המערכת ---
#if __name__ == "__main__":
#    with open("shared_vocab.json", "r", encoding="utf-8") as f:
#        shared_vocab = json.load(f)
#
#    detector = NarrativeDetector(ner_vocab=shared_vocab, srl_vocab=shared_vocab)
#    detector.load_state_dict(torch.load("best_narrative_model_hybrid.pth"))
#
#    # תיקון שם קובץ ה-BERTopic לשם התואם לקוד האימון
#    loaded_topic_model = BERTopic.load("saved_topic_model")
#    detector.stance_processor.topic_model = loaded_topic_model
#    detector.eval()
#
#    # משפט דוגמה
#    sample_text = "In my opinion, the UN might condemn the attack to protect the civilians."
#
#    print(f"\nAnalyzing Text: '{sample_text}'")
#    with torch.no_grad():
#        results = detector(sample_text).squeeze()
#
#    print("\nFinal Narrative Probabilities (Top 5):")
#    top_probs, top_indices = torch.topk(results, 5)
#
#    for i in range(5):
#        print(f"Narrative {NARRATIVES[top_indices[i].item()]}: {top_probs[i].item() * 100:.2f}%")

import torch
import torch.nn as nn
import json
from bertopic import BERTopic
from tqdm import tqdm
import transformers
import logging

# ייבוא כל הרכיבים שבנינו מהקבצים השונים
from ner import NarrativeEntityLayer, EntityAnalysisPipeline
from srl import SRLProcessor, SRLNarrativeLayer
from emotion import EmotionAgencyProcessor, EmotionAgencyLayer
from reliability import ReliabilityProcessor, ReliabilityLayer
from stance import TopicStanceLayer, TopicAnalysisPipeline  # משמש כעת כמודול נושאים בלבד
from config import NUM_NARRATIVES, NARRATIVES

transformers.logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)


# =====================================================================
# רשת איחוד לינארית (Linear Fusion Network) לשקיפות וחילול משקלים
# =====================================================================
class NarrativeFusionNetwork(nn.Module):
    """
    רשת איחוד לינארית לבדיקת חשיבות מודולים:
    לומדת משקל בודד עבור כל מודול (NER, Topics, SRL, Emotion)
    ומבצעת סכום משוקלל שלהם כדי לאפשר פרשנות שקופה של חשיבות הרכיבים.
    """

    def __init__(self):
        super(NarrativeFusionNetwork, self).__init__()

        # הגדרת 4 משקלים לומדים (אחד לכל מודול) שיאותחלו בערך שווה
        self.module_weights = nn.Parameter(torch.ones(4))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, vec_ner, vec_topics, vec_srl, vec_emotion, weight_factor):
        # נרנול המשקלים באמצעות Softmax כך שסכומם יהיה תמיד 1 (100%)
        weights = torch.softmax(self.module_weights, dim=0)

        # חישוב סכום משוקלל: כל וקטור מוכפל במשקל הספציפי שהמודל למד עבורו
        fused_logits = (weights[0] * vec_ner.squeeze() +
                        weights[1] * vec_topics.squeeze() +
                        weights[2] * vec_srl.squeeze() +
                        weights[3] * vec_emotion.squeeze())

        # הפעלת פקטור האמינות על התוצאה המשוקללת
        amplified_signal = fused_logits * weight_factor

        # המרה להסתברויות
        return self.softmax(amplified_signal)


# =====================================================================
# המערכת המלאה (Pipeline) המותאמת למודול נושאים בלבד
# =====================================================================
class NarrativeDetector(nn.Module):
    """
    המערכת המלאה (Pipeline) שאורזת את כל הרכיבים הלומדים והמעבדים.
    מותאמת למודול נושאים בלבד (ללא רכיב העמדות - Stance).
    """

    def __init__(self, ner_vocab, srl_vocab):
        super(NarrativeDetector, self).__init__()

        print("Initializing Full Narrative Detection Pipeline (Topics Only)...")

        # מחלצי המאפיינים - לא לומדים (כבדים)
        pbar = tqdm(total=5, desc="Loading Models",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

        self.ner_processor = EntityAnalysisPipeline()
        pbar.update(1)
        self.srl_processor = SRLProcessor()
        pbar.update(1)
        self.emotion_processor = EmotionAgencyProcessor()
        pbar.update(1)
        self.stance_processor = TopicAnalysisPipeline()
        pbar.update(1)
        self.reliability_processor = ReliabilityProcessor()
        pbar.update(1)

        pbar.close()

        # השכבות הלומדות בתוך PyTorch (מהירות)
        self.ner_layer = NarrativeEntityLayer(ner_vocab)
        self.srl_layer = SRLNarrativeLayer(srl_vocab)
        self.emotion_layer = EmotionAgencyLayer()
        self.stance_layer = TopicStanceLayer()  # מתנהג כעת כ-Topic Layer בלבד
        self.reliability_layer = ReliabilityLayer()

        # רשת האיחוד הלינארית לבדיקת משקלים
        self.fusion_network = NarrativeFusionNetwork()

        self.ner_vocab = ner_vocab
        self.srl_vocab = srl_vocab

    def extract_features(self, text):
        return {
            "ner": self.ner_processor.extract_entities(text, self.ner_vocab),
            "srl": self.srl_processor.extract_features(text, self.srl_vocab),
            "emotion": self.emotion_processor.extract_features(text),
            "stance": self.stance_processor.process_text(text),  # מחזיר tuple של (topic_id, stance_label)
            "reliability": self.reliability_processor.extract_features(text)
        }

    def classify_features(self, features):
        # 1. עיבוד מודול ישויות ומטרות
        vec_ner = self.ner_layer(features["ner"])
        vec_srl = self.srl_layer(features["srl"])

        # 2. עיבוד מודול דפוסים (רגש ואקטיביות)
        emotion_idx, agency_flag = features["emotion"]
        vec_emotion = self.emotion_layer(emotion_idx, agency_flag)

        # 3. התאמה למודול נושאים בלבד: שליחת ה-topic_id בלבד לשכבה הלומדת
        topic_id = features["stance"]  # התעלמות מודעת מה-stance_label (הורדת הרעש)
        topic_tensor = torch.tensor([topic_id], dtype=torch.long)

        # הערה: שכבת ה-stance_layer המקורית מופעלת כאן רק על טנזור הנושא
        vec_topics = self.stance_layer(topic_tensor)

        # 4. חילוץ פקטור אמינות
        weight_factor = self.reliability_layer(features["reliability"])

        # 5. איחוד לינארי משוקלל
        final_probs = self.fusion_network(vec_ner, vec_topics, vec_srl, vec_emotion, weight_factor)
        return final_probs

    def forward(self, text):
        features = self.extract_features(text)
        return self.classify_features(features)


# --- סימולציה של ריצת המערכת ---
if __name__ == "__main__":
    # טעינת נתוני המודל
    with open("data/cache/shared_vocab.json", "r", encoding="utf-8") as f:
        shared_vocab = json.load(f)

    detector = NarrativeDetector(ner_vocab=shared_vocab, srl_vocab=shared_vocab)
    detector.load_state_dict(torch.load("models/best_narrative_model_hybrid.pth"))

    loaded_topic_model = BERTopic.load("models/best_bertopic_model")
    detector.stance_processor.topic_model = loaded_topic_model
    detector.eval()

    # משפט דוגמה
    sample_text = "In my opinion, the UN might condemn the attack to protect the civilians."

    print(f"\nAnalyzing Text: '{sample_text}'")
    with torch.no_grad():
        results = detector(sample_text).squeeze()

    print("\nFinal Narrative Probabilities (Top 5):")
    top_probs, top_indices = torch.topk(results, 5)

    for i in range(5):
        print(f"Narrative {NARRATIVES[top_indices[i].item()]}: {top_probs[i].item() * 100:.2f}%")