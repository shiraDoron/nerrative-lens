import torch
import torch.nn as nn
from transformers import pipeline
import spacy


class ReliabilityProcessor:
    """
    מחלץ את נתוני הגלם (הסתברויות וספירות) ללא למידה.
    """

    def __init__(self):
        print("Loading Reliability Assessment Models...")
        self.fake_news_pipe = pipeline("text-classification", model="XSY/albert-base-v2-fakenews-discriminator")
        self.subjectivity_pipe = pipeline("text-classification", model="lighteternal/fact-or-opinion-xlmr-el")

        print("Loading spaCy for Modality & Doubt Extraction...")
        self.nlp = spacy.load("en_core_web_trf")

        self.doubt_aux_lemmas = {"might", "may", "could", "would"}
        self.doubt_adv_lemmas = {"possibly", "maybe", "perhaps", "probably"}

    def extract_features(self, text):
        # הרצת המודלים
        fake_res = self.fake_news_pipe(text, truncation=True, max_length=512)[0]
        subj_res = self.subjectivity_pipe(text, truncation=True, max_length=512)[0]

        prob_fake = fake_res['score'] if fake_res['label'] == 'LABEL_0' else (1 - fake_res['score'])
        prob_subjective = subj_res['score'] if subj_res['label'] == 'LABEL_0' else (1 - subj_res['score'])

        # הרצת מנתח התלויות לאיתור ספק
        doc = self.nlp(text)
        doubt_count = 0.0

        for token in doc:
            lemma = token.lemma_.lower()
            if token.dep_ == "aux" and lemma in self.doubt_aux_lemmas:
                doubt_count += 1.0
            elif token.dep_ == "advmod" and lemma in self.doubt_adv_lemmas:
                doubt_count += 1.0

        # מחזיר טנזור של שלושת נתוני הגלם
        return torch.tensor([prob_fake, prob_subjective, doubt_count], dtype=torch.float32)


class ReliabilityLayer(nn.Module):
    """
    השכבה הלומדת: מקבלת את נתוני הגלם ומוצאת את המשקולות המדויקות
    כדי לחשב את פקטור ההכפלה האולטימטיבי.
    """

    def __init__(self):
        super(ReliabilityLayer, self).__init__()

        # שכבה ליניארית שמקבלת 3 קלטים ומחזירה סקלר 1 (הפקטור)
        self.linear = nn.Linear(3, 1)

        # אתחול חכם: אנחנו נותנים למודל "נקודת פתיחה" הגיונית,
        # אבל PyTorch ישנה את המספרים האלו תוך כדי הלמידה מול הדאטה שלך.
        # ההנחה ההתחלתית: פייק וסובייקטיביות מוסיפים ערך (חיובי), ספק מוריד ערך (שלילי)
        nn.init.constant_(self.linear.weight[0][0], 0.5)  # משקל התחלתי לפייק ניוז
        nn.init.constant_(self.linear.weight[0][1], 0.5)  # משקל התחלתי לסובייקטיביות
        nn.init.constant_(self.linear.weight[0][2], -0.15)  # משקל התחלתי לכמות הספק
        nn.init.constant_(self.linear.bias, 1.0)  # משקל הבסיס (מתחיל מ-1.0)

    def forward(self, features):
        # חישוב הפקטור לפי המשקולות הלומדות
        weight_factor = self.linear(features)

        # שימוש ב-ReLU כדי לוודא שפקטור ההכפלה לעולם לא יירד מתחת לאפס
        # (אנחנו לא רוצים שפקטור שלילי יהפוך פתאום נרטיב הפוך לנרטיב מוביל)
        return torch.relu(weight_factor)


# --- קוד בדיקה להרצה ---
if __name__ == "__main__":
    print("\n--- מתחילים בדיקת אמינות לומדת ---")

    # אתחול שני הרכיבים
    processor = ReliabilityProcessor()
    layer = ReliabilityLayer()

    test_texts = [
        "The sky is blue today.",
        "Rumors say the moon is made of green cheese!",
        "In my humble opinion, this is a terrible idea.",
        "The government might possibly resign today."
    ]

    for text in test_texts:
        # 1. חילוץ וקטור התכונות (פייק, סובייקטיביות, ספירת ספק)
        features = processor.extract_features(text)

        # 2. חישוב פקטור ההכפלה דרך השכבה הלומדת (ללא חישוב גרדיאנטים בטסט)
        with torch.no_grad():
            weight_factor = layer(features)

        print(f"טקסט: '{text}'")
        # מדפיסים את וקטור הגלם כדי לראות מה המודל "ראה"
        print(f"נתוני גלם [פייק, סובייקטיביות, ספק]: {features.numpy().round(3)}")
        # מדפיסים את התוצאה הסופית שהשכבה הליניארית חישבה
        print(f"פקטור הכפלה (Weight Factor): {weight_factor.item():.4f}\n")