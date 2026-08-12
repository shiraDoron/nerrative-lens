# -*- coding: utf-8 -*-
"""
Agenda coverage check against the dataset
=========================
Checks, for each narrative, how many of the 16 agenda categories (AGENDA_LEXICON in
analyze_agendas.py) actually appear in the datasets (at least one matching document),
and how many appear in significant volume (at least MIN_DOCS matching documents).
Highlights narratives that don't reach 10 categories.

Usage:
    python check_agenda_coverage.py
    python check_agenda_coverage.py --files data/raw/twitter_natural_dataset.csv data/raw/telegram_natural_dataset.csv ...
"""
import argparse
import os

import numpy as np
import pandas as pd

from analyze_agendas import AGENDA_PATTERNS, clean_text

MIN_DOCS = 10  # threshold for "significant presence" of an agenda in a narrative


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", default=[
        "data/raw/twitter_natural_dataset.csv",
        "data/raw/telegram_natural_dataset.csv",
        "data/raw/gemini_natural_dataset.csv",
        "data/raw/gpt_natural_dataset.csv",
    ])
    ap.add_argument("--min-len", type=int, default=15)
    args = ap.parse_args()

    frames = []
    for f in args.files:
        if not os.path.exists(f):
            print(f"[!] File not found, skipping: {f}")
            continue
        d = pd.read_csv(f)
        d["source_file"] = os.path.basename(f)
        frames.append(d[["text", "narrative_name"]])
    df = pd.concat(frames, ignore_index=True)

    df = df.dropna(subset=["text", "narrative_name"])
    df["clean"] = df["text"].apply(clean_text)
    df = df[df["clean"].str.len() >= args.min_len]
    df = df.drop_duplicates(subset=["clean"]).reset_index(drop=True)
    print(f"[i] Total {len(df)} unique texts out of {len(args.files)} files.\n")

    # Count documents for each (narrative, category)
    hit_frames = {}
    for cat, pat in AGENDA_PATTERNS.items():
        hit_frames[cat] = df["clean"].apply(lambda s: bool(pat.search(s)))
    hits = pd.DataFrame(hit_frames, index=df.index)
    counts = hits.groupby(df["narrative_name"]).sum().astype(int)  # document count, not %

    total_docs = df.groupby("narrative_name").size()

    print(f"{'Narrative':<15}{'Texts':>9}{'Categories with >=1 doc':>24}{'Categories with >=' + str(MIN_DOCS) + ' docs':>26}")
    print("-" * 80)
    problems = []
    for nar in counts.index:
        row = counts.loc[nar]
        n_any = int((row >= 1).sum())
        n_min = int((row >= MIN_DOCS).sum())
        flag = ""
        if n_min < 10:
            flag = "  <-- fewer than 10 agendas with significant presence!"
            problems.append((nar, n_min))
        print(f"{nar:<15}{total_docs[nar]:>9}{n_any:>24}{n_min:>26}{flag}")

    print()
    if problems:
        print("[!] Narratives that need attention (fewer than 10 agendas with >= %d docs):" % MIN_DOCS)
        for nar, n_min in problems:
            weak = counts.loc[nar].sort_values()
            weak_cats = weak[weak < MIN_DOCS]
            print(f"\n  {nar} ({n_min}/16 meet the threshold):")
            for cat, c in weak_cats.items():
                print(f"      • {cat:<38} {c} docs")
    else:
        print(f"[OK] All narratives have at least 10 agenda categories with {MIN_DOCS}+ matching docs.")

    os.makedirs("reports", exist_ok=True)
    counts.to_csv("reports/agenda_coverage_counts.csv", encoding="utf-8-sig")
    print("\n[i] Full counts table saved to reports/agenda_coverage_counts.csv")


if __name__ == "__main__":
    main()
