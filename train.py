import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report  # ייבוא הספרייה לחישוב מדדי הביצועים

# ייבוא הרכיבים מהקבצים שלך
from config import NARRATIVES
from fusion import NarrativeDetector

# שם קובץ חדש לקובץ המשולב כדי לא לטעון בטעות את הקאש הישן
FEATURES_CACHE_FILE = "cached_features_hybrid.pt"

# מיפוי הנרטיבים לחישוב מדדים
label_to_index = {narrative: i for i, narrative in enumerate(NARRATIVES)}
index_to_label = {i: narrative for i, narrative in enumerate(NARRATIVES)}


def extract_all(data, desc, detector):
    features_list = []
    for _, row in tqdm(data.iterrows(), total=len(data), desc=desc):
        text = str(row['text'])
        feat = detector.extract_features(text)
        features_list.append((feat, int(row['label'])))
    return features_list


if __name__ == "__main__":
    print("Loading datasets...")
    # טעינת הדטה-סט של ג'מיני
    df_llm = pd.read_csv("gemini_natural_dataset.csv")

    # דטה-סט קטן של ג'י-פי-טי (580 משפטים)
    df_little = pd.read_csv("gpt_natural_dataset.csv")

    # דטה-סט מטוויטר
    df_twitter = pd.read_csv("twitter_natural_dataset.csv")

    # דטה-סט מטלגרם
    df_telegram = pd.read_csv("telegram_natural_dataset.csv")

    # איחוד קבצי הנתונים
    df = pd.concat([df_llm, df_little, df_twitter, df_telegram], ignore_index=True)

    # ערבוב חשוב כדי שהמודל יראה משפטים מכל המקורות באופן רנדומלי
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Total samples for training (Hybrid Dataset): {len(df)}")

    # חלוקה ל-Train ו-Val
    train_data, val_data = train_test_split(df, test_size=0.2, random_state=42)

    print("Cleaning and Truncating text for Hybrid Dataset...")

    # שינוי 1: חיתוך טקסטים ארוכים מדי לפני הכל כדי למנוע את קריסת מודל הרגשות
    train_data['text'] = train_data['text'].astype(str).str.slice(0, 3000)
    val_data['text'] = val_data['text'].astype(str).str.slice(0, 3000)

    print("Building and saving new deterministic vocabulary...")


    # שינוי 2: הוספת סינון תווים מיותרים כדי שה-Vocabulary לא יתנפח למימדים עצומים
    def clean_for_vocab(text):
        return re.sub(r'[^\w\s]', '', text.lower())


    all_text = " ".join(train_data['text'])
    all_words_set = set(clean_for_vocab(all_text).split())
    sorted_words = sorted(list(all_words_set))
    shared_vocab = {word: i for i, word in enumerate(sorted_words)}

    with open("shared_vocab.json", "w", encoding="utf-8") as f:
        json.dump(shared_vocab, f, ensure_ascii=False, indent=4)

    # אתחול המודל
    detector = NarrativeDetector(ner_vocab=shared_vocab, srl_vocab=shared_vocab)

    # --- מנגנון השמירה לדיסק (Cache) ---
    if os.path.exists(FEATURES_CACHE_FILE):
        print(f"\nFound cached features at '{FEATURES_CACHE_FILE}'. Loading instantly...")
        cached_data = torch.load(FEATURES_CACHE_FILE, weights_only=False)
        train_features = cached_data['train']
        val_features = cached_data['val']
        print("Loaded successfully! Skipping extraction.")
    else:
        print("\nNo cache found. Starting Feature Extraction (This will take a while for the hybrid dataset)...")
        train_features = extract_all(train_data, "Extracting Train Features", detector)
        val_features = extract_all(val_data, "Extracting Val Features", detector)

        print(f"\nSaving extracted features to '{FEATURES_CACHE_FILE}'...")
        torch.save({'train': train_features, 'val': val_features}, FEATURES_CACHE_FILE)
        print("Features saved successfully! Next time it will load in seconds.")

    print("\n--- Starting Fast Training ---")

    optimizer = optim.Adam(detector.parameters(), lr=0.001)
    loss_fn = nn.NLLLoss()

    epochs = 20
    batch_size = 16
    best_val_loss = float('inf')

    # הגדרות מנגנון העצירה המוקדמת
    patience = 3
    epochs_no_improve = 0

    for epoch in range(epochs):
        detector.train()
        total_train_loss = 0.0
        train_correct = 0

        # איפוס מחוץ ללולאת המשפטים כדי להתחיל בצבירה נקייה
        optimizer.zero_grad()

        for i, (features, label_idx) in enumerate(train_features):
            label = torch.tensor([label_idx], dtype=torch.long)
            probs = detector.classify_features(features)

            if probs.dim() == 1:
                probs = probs.unsqueeze(0)

            predicted = torch.argmax(probs, dim=-1)
            if predicted.item() == label.item():
                train_correct += 1

            loss = loss_fn(torch.log(probs + 1e-8), label)
            loss = loss / batch_size

            # צבירת הגרדיאנטים
            loss.backward()

            if (i + 1) % batch_size == 0 or (i + 1) == len(train_features):
                optimizer.step()
                optimizer.zero_grad()

            total_train_loss += loss.item() * batch_size

        # שלב אימות (Validation) ואיסוף מדדים בזמן אמת
        detector.eval()
        val_correct = 0
        total_val_loss = 0.0

        # רשימות זמניות לאופק הנוכחי
        epoch_true_labels = []
        epoch_predicted_labels = []

        with torch.no_grad():
            for features, label_idx in val_features:
                label = torch.tensor([label_idx], dtype=torch.long)
                probs = detector.classify_features(features)

                if probs.dim() == 1:
                    probs = probs.unsqueeze(0)

                predicted = torch.argmax(probs, dim=-1)
                if predicted.item() == label.item():
                    val_correct += 1

                total_val_loss += loss_fn(torch.log(probs + 1e-8), label).item()

                # איסוף עבור המדדים האקדמיים בסוף האימון
                epoch_true_labels.append(label_idx)
                epoch_predicted_labels.append(predicted.item())

        avg_val_loss = total_val_loss / len(val_features)
        print(
            f"Epoch {epoch + 1}: Train Acc: {100 * train_correct / len(train_features):.2f}% | Val Loss: {avg_val_loss:.4f}")

        # שמירת המודל הטוב ביותר והנתונים הסטטיסטיים שלו
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(detector.state_dict(), "best_narrative_model_hybrid.pth")

            # שמירת תוצאות ה-Validation הטובות ביותר להדפסה הסופית
            best_true_labels = epoch_true_labels.copy()
            best_predicted_labels = epoch_predicted_labels.copy()
        else:
            epochs_no_improve += 1
            print(f">>> No improvement in Validation Loss for {epochs_no_improve} epoch(s).")

            if epochs_no_improve >= patience:
                print(f"\n[!] Early Stopping Triggered! Training halted at epoch {epoch + 1}.")
                break

    # ==========================================================
    # חישוב והדפסת מדדי הביצועים האקדמיים על סט ה-Val הנקי
    # ==========================================================
    print("\n" + "=" * 60)
    print("FINAL EVALUATION RESULTS (Best Model on Unseen Validation Split)")
    print("=" * 60)

    # טעינת המודל הטוב ביותר שנשמר לפני החישוב הסופי (ליתר ביטחון)
    detector.load_state_dict(torch.load("best_narrative_model_hybrid.pth"))
    detector.eval()

    # חילוץ שמות הנרטיבים הרלוונטיים שנמצאו בסט האימות
    target_names = [index_to_label[i] for i in range(len(NARRATIVES)) if
                    i in best_true_labels or i in best_predicted_labels]

    # יצירת הדו"ח המלא
    report = classification_report(
        best_true_labels,
        best_predicted_labels,
        target_names=target_names,
        digits=4,
        zero_division=0
    )
    print(report)
    print("=" * 60)

    # הדפסת משקלי המודולים שנלמדו במודל הטוב ביותר
    if hasattr(detector.fusion_network, 'module_weights'):
        print("\n--- Module Importance (Learned Weights) ---")
        learned_weights = torch.softmax(detector.fusion_network.module_weights, dim=0)
        print(f"NER Weight: {learned_weights[0].item() * 100:.1f}%")
        print(f"Stance Weight: {learned_weights[1].item() * 100:.1f}%")
        print(f"SRL Weight: {learned_weights[2].item() * 100:.1f}%")
        print(f"Emotion Weight: {learned_weights[3].item() * 100:.1f}%")
    print("=" * 60)