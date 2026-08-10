import torch
import torch.nn as nn
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from bertopic import BERTopic
from config import NUM_TOPICS, NUM_NARRATIVES
from transformers import AutoModel
from peft import PeftModel
from sentence_transformers import models

# הלמידה מהעמדות
class TopicStanceLayer(nn.Module):
    def __init__(self):
        super(TopicStanceLayer, self).__init__()

        self.num_topics = NUM_TOPICS
        self.num_narratives = NUM_NARRATIVES

        # מתנגד 0, ניטרלי 1, תומך 2
        self.num_stances = 3

        # גודל המילון = מספר הנושאים * 3
        self.matrix_size = self.num_topics * self.num_stances
        self.weights = nn.Embedding(self.matrix_size, self.num_narratives)

        # אתחול
        nn.init.uniform_(self.weights.weight, 0, 1)

    # פונקציית הלמידה
    # מקבלת נושאים ועמדות
    def forward(self, topic_ids, stance_labels):
        # הגדרת אינדקס ה-OOV כנושא האחרון (499)
        oov_topic_index = self.num_topics - 1

        # הגנה: החלפת -1 או חריגות (מעל 499) באינדקס ה-OOV
        # torch.clamp ו-torch.where עוזרים לנו להישאר בתוך גבולות הגזרה
        safe_topic_ids = torch.where(
            (topic_ids == -1) | (topic_ids >= self.num_topics),
            torch.tensor(oov_topic_index, device=topic_ids.device),
            topic_ids
        )

        # חישוב האינדקס המורכב (בין 0 ל-1499)
        indices = (safe_topic_ids * self.num_stances) + stance_labels

        # וידוא סופי שהאינדקס לא חורג מ-matrix_size (למשל אם stance_label לא תקין)
        indices = torch.clamp(indices, 0, self.matrix_size - 1)

        # שליפת הוקטורים המתאימים (האחוזים לנרטיבים)
        narrative_vectors = self.weights(indices)

        return narrative_vectors


def load_merged_sentence_transformer(base_model_name, peft_model_id):
    base_model = AutoModel.from_pretrained(base_model_name)
    peft_model = PeftModel.from_pretrained(base_model, peft_model_id)
    merged_model = peft_model.merge_and_unload()

    word_embedding_model = models.Transformer(base_model_name)
    word_embedding_model.auto_model = merged_model
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

    return SentenceTransformer(modules=[word_embedding_model, pooling_model])

# הרצת המודלים לזיהוי העמדות
class TopicAnalysisPipeline:
    def __init__(self):
        print("Loading Stance Model...")
        # טעינה תקינה של מודל הבסיס + השכבה המכווננת
        self.stance_model = load_merged_sentence_transformer(
            "sentence-transformers/all-mpnet-base-v2",
            "vahidthegreat/StanceAware-SBERT"
        )

        print("Loading pre-trained Topic Model...")
        # טעינת המודל המוכן מהתיקייה ששמרנו אליה קודם
        self.topic_model = BERTopic.load("saved_topic_model")
        print("Pipeline is ready!")

    # מבצעת סיווג עמדה באמצעות משפטי עוגן
    def detect_stance(self, text, topic_text):
        anchor_pro = f"I support {topic_text}"
        anchor_con = f"I oppose {topic_text}"

        embeddings = self.stance_model.encode([text, anchor_pro, anchor_con])

        sim_pro = util.cos_sim(embeddings[0], embeddings[1]).item()
        sim_con = util.cos_sim(embeddings[0], embeddings[2]).item()

        diff = sim_pro - sim_con

        if diff > 0.02:
            return 2  # Support
        elif diff < -0.02:
            return 0  # Oppose
        else:
            return 1  # Neutral

    # מזהה נושא מרכזי, ומחזירה אותו עם העמדה כלפיו
    def process_text(self, text):
        topics, probs = self.topic_model.transform([text])
        main_topic_id = topics[0]

        if main_topic_id == -1:
            return -1, 1

        topic_info = self.topic_model.get_topic(main_topic_id)

        if isinstance(topic_info, list) and len(topic_info) > 0:
            topic_name = topic_info[0][0]
        else:
            topic_name = "General"

        stance = self.detect_stance(text, topic_name)

        return main_topic_id, stance


# --- קוד בדיקה (טסט) ---
if __name__ == "__main__":
    print("מתחיל אתחול מודלים (זה עשוי לקחת כמה שניות)...")
    pipeline = TopicAnalysisPipeline()

    # מאגר נתונים שרירותי כדי ש-BERTopic יוכל לייצר "קלסרים"
    with open("gemini_natural_dataset.csv", "r", encoding="utf-8") as f:
        test_full_data = [line.strip() for line in f if line.strip()]

    # משפטי בדיקה שנרצה לנתח מול הקלסרים שנוצרו
    test_sentences = test_full_data[:10]

    print("\n--- תוצאות הניתוח ---")
    for sentence in test_sentences:
        topic_id, test_stance = pipeline.process_text(sentence, pipeline.topic_model)

        stance_text = {0: "נגד", 1: "ניטרלי", 2: "בעד"}.get(test_stance)
        test_topic_name = "לא זוהה"

        if topic_id != -1:
            test_topic_info = pipeline.topic_model.get_topic(topic_id)
            if test_topic_info:
                topic_name = test_topic_info[0][0]  # המילה החזקה ביותר

        print(f"טקסט: '{sentence}'")
        print(f"נושא שזוהה: {test_topic_name} (ID: {topic_id})")
        print(f"עמדה כלפי הנושא: {stance_text} (קוד: {test_stance})\n")