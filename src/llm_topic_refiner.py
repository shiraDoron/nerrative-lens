# -*- coding: utf-8 -*-
"""
שיפור פרשנות נושאי BERTopic בעזרת LLM (גרסה קלה, בהשראת LLM-ITL)
==================================================================
אחרי אימון BERTopic (train_topics.py), כל "נושא" מיוצג ע"י רשימת מילות מפתח
סטטיסטית (c-TF-IDF) שלא תמיד קוהרנטית לבן אדם (למשל מילים לא קשורות תחת
אותו נושא). הסקריפט הזה שולח את מילות המפתח של כל נושא ל-Gemini ומבקש:
  1. תווית קצרה, קריאה, לנושא.
  2. אילו מהמילים (אם בכלל) לא שייכות תמטית לשאר ("רעש").
  3. רמת ביטחון (0-1) של ה-LLM בתווית שהציע.

חשוב: זו גרסה קלה בהשראת LLM-ITL (https://github.com/Xiaohao-Yang/LLM-ITL),
לא מימוש מלא שלו - אין כאן Neural Topic Model מותאם ואין יישור
Optimal-Transport בין הצעות ה-LLM למשקלי המודל; המטרה כאן היא רק לשפר
את הפרשנות/התיוג האנושי של הנושאים ש-BERTopic כבר הפיק, בעלות ומורכבות
נמוכות בהרבה, ולא לשנות את אימון המודל עצמו.

רץ רק ב-Colab / בסביבה שבה מותקן bertopic (לא בסביבה המקומית).
דורש משתנה סביבה GEMINI_API_KEY (אל תשתמשו במפתח hardcoded בקוד!).

הרצה עצמאית (אחרי שכבר יש saved_topic_model/ שמור):
    python llm_topic_refiner.py
"""
import json
import os
import time

DEFAULT_MODEL = "gemini-2.5-flash"
FALLBACK_MODEL = "gemini-flash-latest"

SYSTEM_PROMPT = """
You are an expert in topic modeling interpretability.
For each numbered topic below, you are given its top statistical keywords
(from a BERTopic / c-TF-IDF model). These keyword lists are sometimes noisy
or incoherent.
For EACH topic, return:
  - "label": a short, human-readable 2-5 word label capturing the topic's theme.
  - "noise_words": a list of the given words (verbatim) that do NOT thematically
    fit the rest of the group (empty list if all words fit).
  - "confidence": your confidence (0.0-1.0) that the label correctly captures
    a coherent theme in the words.
Respond ONLY with a valid JSON array of objects, one per topic, in the same
order as given, each with fields: "topic_id", "label", "noise_words", "confidence".
"""


def _get_client():
    """יוצר לקוח Gemini מתוך משתנה סביבה - לעולם לא hardcoded בקוד."""
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "לא הוגדר משתנה הסביבה GEMINI_API_KEY. "
            "הגדר אותו לפני הרצת שיפור הנושאים (למשל: os.environ['GEMINI_API_KEY']=... "
            "או export GEMINI_API_KEY=... בטרמינל). אין להטמיע מפתח בקוד."
        )
    return genai.Client(api_key=api_key)


def _build_batch_prompt(batch):
    lines = [f"Topic {topic_id}: {', '.join(words)}" for topic_id, words in batch]
    return "\n".join(lines)


def _call_llm(client, batch, model_id):
    from google.genai import types

    prompt = f"{SYSTEM_PROMPT}\n\n{_build_batch_prompt(batch)}"
    response = client.models.generate_content(
        model=model_id,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
        ),
    )
    data = json.loads(response.text)
    return data if isinstance(data, list) else []


def refine_topics_with_llm(topic_model, top_n_words=10, batch_size=15,
                            out_path="saved_topic_model/topics_llm_refined.json"):
    """
    topic_model: BERTopic מאומן (אחרי .fit()/.load()).
    שולח את מילות המפתח של כל נושא (למעט הנושא החריג -1) ל-Gemini בבאצ'ים,
    ומחזיר + שומר dict:
        {topic_id: {"top_words": [...], "label": ..., "noise_words": [...],
                    "confidence": ...}}
    תומך בהמשכה אוטומטית אם הריצה נקטעה (טוען out_path אם קיים).
    """
    client = _get_client()
    current_model = DEFAULT_MODEL
    consecutive_429s = 0

    topic_info = topic_model.get_topic_info()
    topic_ids = [t for t in topic_info["Topic"].tolist() if t != -1]

    topics_words = {tid: [w for w, _ in topic_model.get_topic(tid)[:top_n_words]]
                     for tid in topic_ids}

    results = {}
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            results = {int(k): v for k, v in json.load(f).items()}
        print(f"[i] נמצאו {len(results)} נושאים משוכללים קיימים, ממשיך משם...")

    pending = [tid for tid in topic_ids if tid not in results]
    batches = [pending[i:i + batch_size] for i in range(0, len(pending), batch_size)]

    for batch_ids in batches:
        batch = [(tid, topics_words[tid]) for tid in batch_ids]
        print(f"  [i] שולח נושאים {batch_ids[0]}-{batch_ids[-1]} ({len(batch)}) ל-{current_model}...")
        try:
            parsed = _call_llm(client, batch, current_model)
            consecutive_429s = 0
            returned_ids = set()
            for item in parsed:
                tid = int(item.get("topic_id"))
                results[tid] = {
                    "top_words": topics_words.get(tid, []),
                    "label": item.get("label", ""),
                    "noise_words": item.get("noise_words", []),
                    "confidence": item.get("confidence", None),
                }
                returned_ids.add(tid)
            missing = set(batch_ids) - returned_ids
            if missing:
                print(f"  [!] ה-LLM לא החזיר תשובה עבור נושאים {sorted(missing)}, ינוסה שוב בריצה הבאה.")

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            time.sleep(3)

        except Exception as e:
            err_str = str(e)
            if "429" in err_str:
                consecutive_429s += 1
                if consecutive_429s >= 2 and current_model == DEFAULT_MODEL:
                    print("  [!] מכסת gemini-2.5-flash הסתיימה, עובר למודל גיבוי...")
                    current_model = FALLBACK_MODEL
                    consecutive_429s = 0
                    time.sleep(5)
                else:
                    print("  [!] הגבלת קצב. ממתין 65 שניות...")
                    time.sleep(65)
            else:
                print(f"  [!] שגיאה: {err_str}. מנסה שוב בעוד 20 שניות...")
                time.sleep(20)

    print(f"[✓] שוכללו {len(results)}/{len(topic_ids)} נושאים. נשמר ב-{out_path}")
    return results


def load_refined_labels(path="models/saved_topic_model/topics_llm_refined.json"):
    """טוען את קובץ התוויות המשוכללות (אם קיים) כ-dict {topic_id(int): {...}}."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


if __name__ == "__main__":
    from bertopic import BERTopic

    print("טוען מודל BERTopic קיים מ-saved_topic_model...")
    model = BERTopic.load("models/saved_topic_model")
    refine_topics_with_llm(model)
