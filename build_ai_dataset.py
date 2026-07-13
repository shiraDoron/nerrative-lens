import os
import json
import csv
import time
from google import genai
from google.genai import types
from google.genai.errors import APIError
from config import NARRATIVES

# אתחול הלקוח עם המפתח שלך - נקרא ממשתנה סביבה GEMINI_API_KEY, אף פעם לא מוטבע בקוד.
_GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not _GEMINI_API_KEY:
    raise RuntimeError(
        "חסרה משתנת סביבה GEMINI_API_KEY. הגדר: "
        "$env:GEMINI_API_KEY = '<your-key>' (PowerShell) לפני הרצת הסקריפט."
    )
client = genai.Client(api_key=_GEMINI_API_KEY)

SYSTEM_PROMPT = """
You are an expert data generator for machine learning pipelines.
Your task is to generate highly realistic, diverse, and natural text samples representing specific political, social, or religious narratives.
Include natural imperfections and conversational fillers to make it sound like real social media posts, comments, or speeches. 
Respond ONLY with a valid JSON array of objects.
"""

# הוספת מילון ההקשרים המדויק שימנע מהמודל להתבלבל
NARRATIVE_CONTEXTS = {
    "Zionist": "The Israeli/Zionist narrative, focusing on the historical right to the Jewish homeland, self-defense against terrorism, the resilience of the IDF, and the survival of the democratic State of Israel.",
    "Western": "The Western democratic narrative, focusing on upholding the global rules-based order, NATO alliances, human rights, free markets, and defending against authoritarian regimes.",
    "Resistance": "The geopolitical 'Axis of Resistance' narrative, focusing on anti-imperialism, armed struggle, opposition to Western hegemony and imperialism in the Middle East, and the Palestinian liberation movement.",
    "Russian": "The Russian state narrative, focusing on protecting traditional spheres of influence, opposing NATO expansion, establishing a multipolar world order, and defending the Motherland.",
    "Ukrainian": "The Ukrainian national narrative, focusing on defending territorial integrity against unprovoked foreign invasion, integration into European democracies, and fighting for absolute sovereignty and freedom.",
    "Right-wing": "The conservative Right-wing narrative, focusing on strict border security, traditional family values, nationalism, free market capitalism, and opposition to progressive or woke ideologies.",
    "Left-wing": "The progressive Left-wing narrative, focusing on dismantling systemic inequality, combating the climate crisis, universal healthcare, taxing the ultra-rich, and fighting for marginalized communities.",
}


def generate_samples_for_narrative(narrative, num_samples, model_id):
    # משיכת ההקשר הספציפי למניעת אי-הבנות
    context = NARRATIVE_CONTEXTS.get(narrative, "")

    # בניית הפרומפט עם ההקשר המדויק
    user_prompt = (
        f"Generate exactly {num_samples} distinct sentences for the '{narrative}' narrative.\n"
        f"Context and themes to focus on: {context}\n"
        f"Format: [{{\"text\": \"...\"}}]"
    )

    response = client.models.generate_content(
        model=model_id,
        contents=f"{SYSTEM_PROMPT}\n\n{user_prompt}",
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.8
        )
    )

    data = json.loads(response.text)
    return data if isinstance(data, list) else []


def build_llm_dataset(total_samples_per_narrative=1200):
    output_filename = "gemini_natural_dataset.csv"

    current_model = 'gemini-2.5-flash'
    fallback_model = 'gemini-flash-latest'
    consecutive_429s = 0

    existing_counts = {n: 0 for n in NARRATIVES}
    if os.path.exists(output_filename):
        with open(output_filename, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 3:
                    narrative = row[2]
                    if narrative in existing_counts:
                        existing_counts[narrative] += 1
        print(f"Resuming progress from {sum(existing_counts.values())} samples...")
    else:
        with open(output_filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["text", "label", "narrative_name"])

    batch_size = 50

    for label_idx, narrative in enumerate(NARRATIVES):
        samples_collected = existing_counts[narrative]

        while samples_collected < total_samples_per_narrative:
            print(f"  Requesting batch for {narrative} using {current_model}...")
            try:
                batch = generate_samples_for_narrative(narrative, batch_size, current_model)
                consecutive_429s = 0

                with open(output_filename, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    for item in batch:
                        if isinstance(item, dict) and "text" in item:
                            # ניקוי מהיר של ירידות שורה כדי לא לשבור את ה-CSV
                            clean_text = item["text"].replace("\n", " ").replace("\r", "")
                            writer.writerow([clean_text, label_idx, narrative])
                            samples_collected += 1

                print(f"  [+] Saved. Total for {narrative}: {samples_collected}/{total_samples_per_narrative}")
                time.sleep(5)

            except Exception as e:
                err_str = str(e)
                if "429" in err_str:
                    consecutive_429s += 1
                    if consecutive_429s >= 2 and current_model == 'gemini-2.5-flash':
                        print("  [!] 2.5-flash quota hit. Switching to Flash fallback...")
                        current_model = fallback_model
                        consecutive_429s = 0
                        time.sleep(5)
                    else:
                        print("  [!] Rate limit. Sleeping 65s...")
                        time.sleep(65)
                else:
                    print(f"  [!] Error: {err_str}. Retrying in 20s...")
                    time.sleep(20)

    print("\nSuccess! Dataset is complete.")


if __name__ == "__main__":
    build_llm_dataset(total_samples_per_narrative=1200)