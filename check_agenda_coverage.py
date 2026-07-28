# -*- coding: utf-8 -*-
"""
בדיקת כיסוי אג'נדות ב-DB
=========================
בודק, לכל נרטיב, כמה מתוך 16 קטגוריות האג'נדה (AGENDA_LEXICON ב-analyze_agendas.py)
מופיעות בפועל בדאטהסטים (לפחות מסמך אחד תואם), וכמה מהן מופיעות בנפח משמעותי
(לפחות MIN_DOCS מסמכים תואמים). מדגיש נרטיבים שלא מגיעים ל-10 קטגוריות.

הרצה:
    python check_agenda_coverage.py
    python check_agenda_coverage.py --files twitter_natural_dataset.csv telegram_natural_dataset.csv ...
"""
import argparse
import os

import numpy as np
import pandas as pd

from analyze_agendas import AGENDA_PATTERNS, clean_text

MIN_DOCS = 10  # ספי "נוכחות משמעותית" של אג'נדה בנרטיב


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", default=[
        "twitter_natural_dataset.csv",
        "telegram_natural_dataset.csv",
        "gemini_natural_dataset.csv",
        "gpt_natural_dataset.csv",
    ])
    ap.add_argument("--min-len", type=int, default=15)
    args = ap.parse_args()

    frames = []
    for f in args.files:
        if not os.path.exists(f):
            print(f"[!] קובץ לא נמצא, מדלג: {f}")
            continue
        d = pd.read_csv(f)
        d["source_file"] = os.path.basename(f)
        frames.append(d[["text", "narrative_name"]])
    df = pd.concat(frames, ignore_index=True)

    df = df.dropna(subset=["text", "narrative_name"])
    df["clean"] = df["text"].apply(clean_text)
    df = df[df["clean"].str.len() >= args.min_len]
    df = df.drop_duplicates(subset=["clean"]).reset_index(drop=True)
    print(f"[i] סה\"כ {len(df)} טקסטים ייחודיים מתוך {len(args.files)} קבצים.\n")

    # ספירת מסמכים לכל (נרטיב, קטגוריה)
    hit_frames = {}
    for cat, pat in AGENDA_PATTERNS.items():
        hit_frames[cat] = df["clean"].apply(lambda s: bool(pat.search(s)))
    hits = pd.DataFrame(hit_frames, index=df.index)
    counts = hits.groupby(df["narrative_name"]).sum().astype(int)  # ספירת מסמכים, לא %

    total_docs = df.groupby("narrative_name").size()

    print(f"{'נרטיב':<15}{'טקסטים':>9}{'קטגוריות עם >=1 מסמך':>24}{'קטגוריות עם >=' + str(MIN_DOCS) + ' מסמכים':>26}")
    print("-" * 80)
    problems = []
    for nar in counts.index:
        row = counts.loc[nar]
        n_any = int((row >= 1).sum())
        n_min = int((row >= MIN_DOCS).sum())
        flag = ""
        if n_min < 10:
            flag = "  <-- פחות מ-10 אג'נדות עם נוכחות משמעותית!"
            problems.append((nar, n_min))
        print(f"{nar:<15}{total_docs[nar]:>9}{n_any:>24}{n_min:>26}{flag}")

    print()
    if problems:
        print("[!] נרטיבים שדורשים תשומת לב (פחות מ-10 אג'נדות עם >= %d מסמכים):" % MIN_DOCS)
        for nar, n_min in problems:
            weak = counts.loc[nar].sort_values()
            weak_cats = weak[weak < MIN_DOCS]
            print(f"\n  {nar} ({n_min}/16 עומדות בסף):")
            for cat, c in weak_cats.items():
                print(f"      • {cat:<38} {c} מסמכים")
    else:
        print(f"[OK] לכל הנרטיבים יש לפחות 10 קטגוריות אג'נדה עם {MIN_DOCS}+ מסמכים תואמים.")

    counts.to_csv("agenda_coverage_counts.csv", encoding="utf-8-sig")
    print("\n[i] טבלת ספירות מלאה נשמרה ב-agenda_coverage_counts.csv")


if __name__ == "__main__":
    main()
