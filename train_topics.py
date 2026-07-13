import os

import pandas as pd
from bertopic import BERTopic

from llm_topic_refiner import refine_topics_with_llm


def build_and_save_topics():
    print("טוען את קבצי הנתונים...")
    df_llm = pd.read_csv("sample_data/gemini_natural_dataset.csv")
    df_little = pd.read_csv("sample_data/gpt_natural_dataset.csv")
    df_twitter = pd.read_csv("sample_data/twitter_natural_dataset.csv")
    df_telegram = pd.read_csv("sample_data/telegram_natural_dataset.csv")

    # איחוד הקבצים
    full_data = pd.concat([df_llm, df_little, df_twitter, df_telegram], ignore_index=True)
    texts_list = full_data['text'].dropna().astype(str).tolist()

    print(f"בונה אשכולות נושאים מתוך {len(texts_list)} משפטים (התהליך עשוי לקחת מספר דקות)...")
    topic_model = BERTopic()
    topic_model.fit(texts_list)

    print("שומר את המודל לתיקייה מקומית...")
    # שימוש בפורמט safetensors המומלץ והמאובטח לשמירת מודלים
    topic_model.save("saved_topic_model", serialization="safetensors")

    # שמירת משקלי המודל ישירות ל-Google Drive
    drive_save_path = "/content/drive/MyDrive/saved_topic_model"
    topic_model.save(drive_save_path, serialization="safetensors")
    print(f">>> Best Hybrid MLP Model Saved to Drive at: {drive_save_path}")
    print("השמירה הושלמה בהצלחה!")

    # שיפור פרשנות הנושאים בעזרת LLM (גרסה קלה בהשראת LLM-ITL, ראו llm_topic_refiner.py).
    # דורש משתנה סביבה GEMINI_API_KEY; מדלג בשקט אם הוא לא מוגדר כדי לא לשבור
    # את זרימת האימון הרגילה.
    if os.environ.get("GEMINI_API_KEY"):
        print("משכלל תוויות נושאים בעזרת Gemini...")
        refine_topics_with_llm(topic_model)
    else:
        print("[i] GEMINI_API_KEY לא מוגדר - מדלג על שכלול תוויות הנושאים ב-LLM.")

if __name__ == "__main__":
    build_and_save_topics()