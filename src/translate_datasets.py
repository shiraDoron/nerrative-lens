import pandas as pd
import time
import re
import os
from deep_translator import GoogleTranslator


def needs_translation(text):
    """בודק אם הטקסט מכיל אותיות בעברית, רוסית/אוקראינית או ערבית"""
    text_str = str(text)
    # הטווחים: עברית, קירילית (רוסית), וערבית
    if re.search(r'[\u0590-\u05FF\u0400-\u04FF\u0600-\u06FF]', text_str):
        return True
    return False


def translate_inplace_final(file_path):
    if not os.path.exists(file_path):
        print(f"הקובץ {file_path} לא נמצא. מדלג...")
        return

    try:
        df = pd.read_csv(file_path)
        print(f"\nמעבד את הקובץ: {file_path} ({len(df)} שורות)")

        translator = GoogleTranslator(source='auto', target='en')
        updated_count = 0
        skipped_count = 0
        error_count = 0

        for index, row in df.iterrows():
            text = str(row['text'])

            # בדיקה האם השורה דורשת תרגום (עברית/ערבית/רוסית)
            if pd.notna(text) and text.strip() != "" and needs_translation(text):
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        translated_text = translator.translate(text)
                        df.at[index, 'text'] = translated_text
                        updated_count += 1
                        time.sleep(0.3)  # השהיה למניעת חסימה
                        break
                    except Exception:
                        if attempt < max_retries - 1:
                            time.sleep(2)
                        else:
                            print(f"שגיאה בשורה {index}, הושארה במקור.")
                            error_count += 1
            else:
                # טקסט באנגלית או ריק - הסקריפט לא פונה לגוגל וחוסך זמן
                skipped_count += 1

            # שמירת ביניים כל 100 שורות כדי שלא נאבד מידע לעולם
            if (index + 1) % 100 == 0:
                print(f"הושלמו {index + 1} שורות... (תורגמו: {updated_count}, דולגו: {skipped_count})")
                df.to_csv(file_path, index=False, encoding='utf-8')

        # שמירה סופית על הקובץ המקורי
        df.to_csv(file_path, index=False, encoding='utf-8')
        print(f"סיום! הקובץ {file_path} עודכן בהצלחה.")
        print(f"סיכום: {updated_count} תורגמו, {skipped_count} דולגו (כבר באנגלית), {error_count} שגיאות.")

    except Exception as e:
        print(f"שגיאה כללית בקובץ {file_path}: {e}")


if __name__ == "__main__":
    # רשימת הקבצים המקוריים שלך
    files = ['data/raw/telegram_natural_dataset.csv', 'data/raw/twitter_natural_dataset.csv']

    print("מתחיל בתרגום חכם ובטוח (עברית, ערבית, רוסית -> אנגלית)...")
    for f in files:
        translate_inplace_final(f)
    print("\nכל הדטה-סטים שלך מעודכנים כעת באנגלית בתוך הקבצים המקוריים.")