import torch
import torch.nn as nn
import pandas as pd
from bertopic import BERTopic
from config import NUM_TOPICS, NUM_NARRATIVES

# שכבת הלמידה מהנושאים (ללא ממד העמדות)
class TopicStanceLayer(nn.Module):
    def __init__(self):
        super(TopicStanceLayer, self).__init__()

        self.num_topics = NUM_TOPICS
        self.num_narratives = NUM_NARRATIVES

        # גודל המילון מבוסס כעת על מספר הנושאים בלבד
        self.matrix_size = self.num_topics
        self.weights = nn.Embedding(self.matrix_size, self.num_narratives)

        # אתחול המשקולות באופן אחיד
        nn.init.uniform_(self.weights.weight, 0, 1)

    # פונקציית הלמידה - מקבלת נושאים בלבד
    def forward(self, topic_ids):
        # הגדרת אינדקס ה-OOV כנושא האחרון (499)
        oov_topic_index = self.num_topics - 1

        # הגנה: החלפת -1 או חריגות (מעל 499) באינדקס ה-OOV
        safe_topic_ids = torch.where(
            (topic_ids == -1) | (topic_ids >= self.num_topics),
            torch.tensor(oov_topic_index, device=topic_ids.device),
            topic_ids
        )

        # ויดוא סופי שהאינדקס לא חורג מטווח המטריצה
        indices = torch.clamp(safe_topic_ids, 0, self.matrix_size - 1)

        # שליפת הוקטורים המתאימים ישירות לפי אינדקס הנושא
        narrative_vectors = self.weights(indices)

        return narrative_vectors

# הרצת המודל לזיהוי נושאים בלבד
class TopicAnalysisPipeline:
    def __init__(self):
        print("Loading pre-trained Topic Model...")
        # טעינת המודל המוכן מהתיקייה ששמרנו אליה קודם
        self.topic_model = BERTopic.load("saved_topic_model")
        # תוויות שכוללו ע"י LLM (llm_topic_refiner.py), אם קיימות - אופציונלי,
        # לא משפיע על הסיווג עצמו, רק על פרשנות אנושית של הנושא.
        try:
            from llm_topic_refiner import load_refined_labels
            self.llm_labels = load_refined_labels()
        except Exception:
            self.llm_labels = {}
        print("Pipeline is ready!")

    def get_topic_label(self, topic_id):
        """מחזיר תווית קריאה לנרטיב: תווית LLM אם קיימת, אחרת המילה הראשונה
        של BERTopic."""
        if topic_id == -1:
            return "לא זוהה"
        refined = self.llm_labels.get(topic_id)
        if refined and refined.get("label"):
            return refined["label"]
        info = self.topic_model.get_topic(topic_id)
        return info[0][0] if info else "לא זוהה"

    # מזהה נושא מרכזי בלבד ומחזירה אותו
    def process_text(self, text):
        topics, probs = self.topic_model.transform([text])
        main_topic_id = topics[0]

        # החזרת מזהה הנושא בלבד (ללא ערך עמדה)
        if main_topic_id == -1:
            return -1

        return main_topic_id


# --- קוד בדיקה (טסט) מעודכן ---
if __name__ == "__main__":
    print("מתחיל אתחול מודלים (זה עשוי לקחת כמה שניות)...")
    pipeline = TopicAnalysisPipeline()

    # טעינת מאגר נתוני הבדיקה
    with open("gemini_natural_dataset.csv", "r", encoding="utf-8") as f:
        test_full_data = [line.strip() for line in f if line.strip()]

    # משפטי בדיקה שנרצה לנתח
    test_sentences = test_full_data[:10]

    print("\n--- תוצאות הניתוח ---")
    for sentence in test_sentences:
        topic_id = pipeline.process_text(sentence)
        test_topic_name = pipeline.get_topic_label(topic_id)

        print(f"טקסט: '{sentence}'")
        print(f"נושא שזוהה: {test_topic_name} (ID: {topic_id})\n")