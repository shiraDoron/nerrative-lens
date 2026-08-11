import argparse
import json
import os
import re

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# import project components
from config import NARRATIVES, MODEL_TYPE, MODEL_TYPES, VAL_SIZE, TEST_SIZE, EPOCHS, BATCH_SIZE, LEARNING_RATE
from fusion import NarrativeDetector, SBERTOnlyDetector, HybridNarrativeDetector
# AGENDA_PATTERNS/clean_text: safe import (only pandas/numpy/re at import time, the CLI is
# guarded by `if __name__ == "__main__"`) - already used by fusion.py's
# AgendaIdeologyFeatureExtractor. Used here by assign_topic_ids_lexicon (fixed, leak-free
# topic labels for LOTO).
from analyze_agendas import AGENDA_PATTERNS, clean_text

# ==========================================================================
# Split modes supported by train.py:
#   "random"              - ordinary random train/val/test split (default)
#   "leave_one_topic"     - Leave-One-Topic-Out (LOTO): one whole topic is held
#                           out entirely for test, to test generalization to
#                           unseen topics. See --topic-source ("lexicon"
#                           [default, leak-free] or "bertopic" [best-effort,
#                           see assign_topic_ids_lexicon /
#                           split_leave_one_topic_bertopic below for full detail]).
#   "leave_one_author"    - Leave-One-Author-Out (LOAO): a single account/channel
#                           is held out entirely for test, to check the model
#                           doesn't just recognize source style.
#   "leave_group_authors" - Group split: several accounts/channels (possibly from
#                           multiple narratives) are held out together for test -
#                           a stronger generalization test than leave_one_author
#                           (single account).
# ==========================================================================
SPLIT_MODES = ("random", "leave_one_topic", "leave_one_author", "leave_group_authors")

# Possible sources for topic_id used by leave_one_topic - see assign_topic_ids_lexicon /
# split_leave_one_topic_bertopic:
#   "lexicon"  - (default, recommended) the dominant AGENDA_LEXICON category in the text -
#                defined a priori (hand-authored regex), not learned from the corpus - no
#                leakage whatsoever.
#   "bertopic" - BERTopic clustering, best-effort: the model is fit only on an internal
#                training pool, and test only receives topic_id via transform() (no
#                re-fitting) - see the limitations note in the function itself.
TOPIC_SOURCES = ("lexicon", "bertopic")

# Placeholder suffix marking a synthetic author_source (gemini/gpt, see
# _derive_author_source) - never a legitimate held-out author in LOAO/Group-split
# experiments (see _validate_real_authors).
SYNTHETIC_AUTHOR_SUFFIX = "_synthetic"

# ==========================================================================
# Model-dependent settings: a feature-cache file + a checkpoint file for each of
# the three model types separately (each model extracts different features, so
# each needs its own cache; but all of them use the exact same train/val/test
# split - see load_raw_data/split_data).
# Important: the cache/checkpoint also depend on split_mode (see
# get_cache_file/get_checkpoint_file) - a different split means different
# samples in train/val/test, so separate cache/checkpoint files are required,
# otherwise we'd silently reuse features computed for a different split.
# ==========================================================================
CACHE_FILES = {
    # The historical filename "hybrid" refers to the combined DATASET
    # (twitter+telegram+gemini+gpt), not to the new Hybrid architecture - kept as-is
    # to avoid invalidating an already-computed cache.
    # (Applies only to split_mode="random" - see get_cache_file.)
    "baseline_fusion": "data/cache/cached_features_hybrid.pt",
    "sbert_only": "data/cache/cached_features_sbert_only.pt",
    "hybrid": "data/cache/cached_features_hybrid_model.pt",
}

CHECKPOINT_FILES = {
    # Historical filename, kept as-is because fusion.py's __main__ and other places load it.
    # (Applies only to split_mode="random" - see get_checkpoint_file.)
    "baseline_fusion": "models/best_narrative_model_hybrid.pth",
    "sbert_only": "models/best_model_sbert_only.pth",
    "hybrid": "models/best_model_hybrid_architecture.pth",
}

# Shared JSON file that accumulates results from every run, for a research comparison
# between the three models.
RESULTS_FILE = "reports/model_comparison_results.json"

# Narrative <-> index mapping for computing metrics
label_to_index = {narrative: i for i, narrative in enumerate(NARRATIVES)}
index_to_label = {i: narrative for i, narrative in enumerate(NARRATIVES)}


def _safe_filename_part(value):
    """Turns an arbitrary value (e.g. an account name) into a filename-safe string."""
    return re.sub(r'[^\w\-]', '_', str(value))


def is_synthetic_author(author_source):
    """gemini_synthetic / gpt_synthetic (see _derive_author_source) - not a real author,
    must not be chosen as held-out in LOAO/Group-split (see _validate_real_authors)."""
    return str(author_source).endswith(SYNTHETIC_AUTHOR_SUFFIX)


def _validate_real_authors(authors):
    """Ensures none of the requested held-out authors is a synthetic placeholder
    (gemini_synthetic/gpt_synthetic) - per user request: an author-generalization test is
    only meaningful for Twitter/Telegram (which have a real account/channel per row)."""
    synthetic = [a for a in authors if is_synthetic_author(a)]
    if synthetic:
        raise ValueError(
            f"held-out author(s) {synthetic} are synthetic placeholders (gemini/gpt have no "
            f"real per-row author), not real accounts - not valid for author-generalization "
            f"experiments. Use a real Twitter/Telegram account/channel instead."
        )


def run_key_for(split_mode, held_out_topic=None, held_out_author=None, held_out_authors=None,
                topic_source="lexicon"):
    """Unique textual identifier for a run, based on the split mode + held-out value (if
    relevant). Used both for cache/checkpoint filenames (split_mode != random) and as a
    key within reports/model_comparison_results.json."""
    if split_mode == "random":
        return "random"
    elif split_mode == "leave_one_topic":
        return f"loto_{topic_source}_topic{held_out_topic}"
    elif split_mode == "leave_one_author":
        return f"loao_{_safe_filename_part(held_out_author)}"
    elif split_mode == "leave_group_authors":
        joined = "_".join(_safe_filename_part(a) for a in sorted(held_out_authors))
        return f"loago_{joined}"
    else:
        raise ValueError(f"Unknown split_mode '{split_mode}'. Expected one of {SPLIT_MODES}.")


def get_cache_file(model_type, split_mode, held_out_topic=None, held_out_author=None,
                   held_out_authors=None, topic_source="lexicon"):
    if split_mode == "random":
        return CACHE_FILES[model_type]
    run_key = run_key_for(split_mode, held_out_topic, held_out_author, held_out_authors, topic_source)
    return f"data/cache/cached_features_{model_type}_{run_key}.pt"


def get_checkpoint_file(model_type, split_mode, held_out_topic=None, held_out_author=None,
                        held_out_authors=None, topic_source="lexicon"):
    if split_mode == "random":
        return CHECKPOINT_FILES[model_type]
    run_key = run_key_for(split_mode, held_out_topic, held_out_author, held_out_authors, topic_source)
    return f"models/best_model_{model_type}_{run_key}.pth"


def build_model(model_type, ner_vocab, srl_vocab):
    """
    Factory that creates the model for the given model_type. This is the single
    extension point to touch when adding another model variant in the future
    (e.g. a Hybrid version with SBERT encoder fine-tuning).
    """
    if model_type == "baseline_fusion":
        return NarrativeDetector(ner_vocab=ner_vocab, srl_vocab=srl_vocab)
    elif model_type == "sbert_only":
        return SBERTOnlyDetector()
    elif model_type == "hybrid":
        return HybridNarrativeDetector(ner_vocab=ner_vocab, srl_vocab=srl_vocab)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Expected one of {MODEL_TYPES}.")


def _derive_author_source(df, dataset_source_name):
    """
    author_source = the specific account/channel the text came from.
    - twitter/telegram: a real 'account' column already exists in the CSV files
      (build_twitter_dataset.py / build_telegram_dataset.py already write it).
    - gemini/gpt: there's no real "account" (synthetic text, not from a single source) -
      gets a fixed placeholder for the whole dataset (e.g. "gemini_synthetic"). Important:
      LOAO on a synthetic source is effectively equivalent to holding out the entire
      synthetic dataset, not testing a real "account style" - see the limitations summary
      given to the user.
    """
    if "account" in df.columns:
        return df["account"].astype(str)
    return pd.Series([f"{dataset_source_name}_synthetic"] * len(df), index=df.index)


def load_raw_data():
    """
    Loads and concatenates (without splitting) all four datasets, with two separate
    provenance columns:
      - dataset_source: twitter / telegram / gpt / gemini
      - author_source:  the specific account/channel (or a synthetic placeholder for
                        gemini/gpt)
    This code is shared by every model_type and split_mode - the actual split happens
    in split_data.
    """
    print("Loading datasets...")

    df_llm = pd.read_csv("data/raw/gemini_natural_dataset.csv")
    df_llm["dataset_source"] = "gemini"
    df_llm["author_source"] = _derive_author_source(df_llm, "gemini")

    df_little = pd.read_csv("data/raw/gpt_natural_dataset.csv")
    df_little["dataset_source"] = "gpt"
    df_little["author_source"] = _derive_author_source(df_little, "gpt")

    df_twitter = pd.read_csv("data/raw/twitter_natural_dataset.csv")
    df_twitter["dataset_source"] = "twitter"
    df_twitter["author_source"] = _derive_author_source(df_twitter, "twitter")

    df_telegram = pd.read_csv("data/raw/telegram_natural_dataset.csv")
    df_telegram["dataset_source"] = "telegram"
    df_telegram["author_source"] = _derive_author_source(df_telegram, "telegram")

    # concatenate the datasets
    df = pd.concat([df_llm, df_little, df_twitter, df_telegram], ignore_index=True)

    # shuffling matters so the model sees sentences from all sources in random order
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Total samples loaded: {len(df)}")

    return df


def assign_topic_ids_lexicon(df):
    """
    LEAK-FREE topic assignment for LOTO (priority-1 method, see documentation in
    TOPIC_SOURCES).

    Assigns each row a 'topic_id' = the category (from AGENDA_LEXICON/AGENDA_PATTERNS in
    analyze_agendas.py) with the most regex matches in the text (or "no_agenda_match" if
    there's no match at all). These categories are defined a priori (hand-authored) and are
    not "learned" in any way from the experiment's corpus - unlike the BERTopic model (see
    train_topics.py: fit on all four concatenated datasets, INCLUDING the texts that would
    later become this experiment's test set, before any split - this is exactly the leakage
    the user identified). Therefore this topic_id can be computed on the whole df before
    splitting, with zero leakage risk: there's no "fit" at all, just deterministic pattern
    matching that doesn't depend on which rows end up in train/val/test.
    """
    df = df.copy()
    cleaned = df["text"].astype(str).apply(clean_text)

    def dominant_category(text):
        counts = {cat: len(pattern.findall(text)) for cat, pattern in AGENDA_PATTERNS.items()}
        best_cat = max(counts, key=lambda c: counts[c])
        if counts[best_cat] == 0:
            return "no_agenda_match"
        return best_cat

    print("Assigning fixed lexicon-based topic ids (AGENDA_LEXICON categories) "
          "for Leave-One-Topic-Out splitting - deterministic, no fitting, no leakage...")
    df["topic_id"] = cleaned.apply(dominant_category)
    return df


def split_leave_one_topic_bertopic(df, held_out_topic):
    """
    BERTopic-based LOTO (fallback method, see TOPIC_SOURCES) - best-effort leakage
    avoidance: unlike using the global saved BERTopic model
    (models/saved_topic_model/, which is fit by train_topics.py on all four
    concatenated datasets - INCLUDING what would become this experiment's test set - and
    therefore constitutes real leakage), this function fits a new, separate BERTopic
    instance, within the current run, ONLY on a "FIT_POOL": a random portion of the data
    (at the same proportion as an ordinary train split - 1 - (VAL_SIZE+TEST_SIZE)).

    The remaining rows ("CANDIDATE_POOL") only get a topic_id via
    topic_model.transform(), without any re-fitting on them - exactly as requested.

    To guarantee a test set fully clean of leakage, the final test is built ONLY from
    CANDIDATE_POOL rows matching held_out_topic (these texts were never seen by the fit).
    FIT_POOL rows that happen to also match that topic_id (assigned by the fit itself) are
    dropped entirely (neither train nor test) - so that train also doesn't actually contain
    texts from that same topic, and so we don't "leak" samples that were seen during fit
    into test.

    Important limitation to document for the user: topic ids here are numbers (int)
    reassigned on every run (the model is refit every time) - they are NOT stable/comparable
    across different runs, unlike the lexicon-based topic_id (based on a fixed category
    name).
    """
    from bertopic import BERTopic

    print("Fitting a fresh BERTopic model ONLY on a designated FIT_POOL "
          "(fit-on-train-only, to avoid the global saved_topic_model's full-corpus "
          "leakage) - this is a Colab-only, heavy step...")

    fit_pool, candidate_pool = train_test_split(df, test_size=(VAL_SIZE + TEST_SIZE), random_state=42)

    topic_model = BERTopic()
    fit_texts = fit_pool["text"].astype(str).tolist()
    fit_topics, _ = topic_model.fit_transform(fit_texts)

    candidate_texts = candidate_pool["text"].astype(str).tolist()
    candidate_topics, _ = topic_model.transform(candidate_texts)

    fit_pool = fit_pool.copy()
    fit_pool["topic_id"] = fit_topics
    candidate_pool = candidate_pool.copy()
    candidate_pool["topic_id"] = candidate_topics

    available_topics = sorted(set(fit_topics) | set(candidate_topics))
    if held_out_topic not in available_topics:
        raise ValueError(
            f"held_out_topic={held_out_topic} not found in data. "
            f"Available topic ids (this run's fresh BERTopic fit): {available_topics}"
        )

    # test = only CANDIDATE_POOL - never seen by the fit
    test_data = candidate_pool[candidate_pool["topic_id"] == held_out_topic].reset_index(drop=True)

    # FIT_POOL rows of the same topic are dropped entirely (neither train nor test) - keeps
    # train clean of the held-out topic, and prevents "leaking" samples that were already
    # seen during fit into test.
    fit_pool_clean = fit_pool[fit_pool["topic_id"] != held_out_topic]
    candidate_pool_remaining = candidate_pool[candidate_pool["topic_id"] != held_out_topic]

    n_dropped_from_fit_pool = (fit_pool["topic_id"] == held_out_topic).sum()
    if n_dropped_from_fit_pool:
        print(f"[i] Dropped {n_dropped_from_fit_pool} FIT_POOL row(s) matching held_out_topic "
              f"{held_out_topic} entirely (not train, not test) to keep the split leak-free.")

    remaining = pd.concat([fit_pool_clean, candidate_pool_remaining], ignore_index=True)
    train_data, val_data = train_test_split(remaining, test_size=VAL_SIZE, random_state=42)

    verify_no_leakage(train_data, val_data, test_data, key_col="topic_id")
    print(f"[i] BERTopic LOTO test set built from CANDIDATE_POOL only "
          f"({len(test_data)} samples) - guaranteed never seen during BERTopic fit.")
    return train_data, val_data, test_data


def verify_no_leakage(train_data, val_data, test_data, key_col):
    """
    Safety check: ensures no value (account/topic_id) is shared between test and
    train/val - so we can trust that the dedicated split (LOTO/LOAO) genuinely tests
    generalization and doesn't "leak" the same source/topic to both sides.
    """
    test_values = set(test_data[key_col].unique())
    train_val_values = set(train_data[key_col].unique()) | set(val_data[key_col].unique())
    overlap = test_values & train_val_values
    if overlap:
        raise AssertionError(
            f"Data leakage detected! '{key_col}' value(s) {overlap} appear in BOTH "
            f"test and train/val splits. This should never happen for a "
            f"leave-one-{key_col}-out split."
        )
    print(f"Leakage check passed: no '{key_col}' value overlaps between test and train/val.")


def _print_narrative_distribution_and_warn(test_data, context_label, dominance_threshold=0.9):
    """
    Prints the narrative distribution in test, and explicitly warns (WARNING, not
    silent) if a single narrative makes up more than dominance_threshold of test - a
    situation that can naturally occur in LOAO/Group-split (an account usually belongs
    to one narrative), but must be stated explicitly and not "silently swallowed" (per
    user requirement).
    """
    counts = test_data["narrative_name"].value_counts()
    shares = (counts / len(test_data)) if len(test_data) else counts
    print(f"[{context_label}] Test narrative distribution: {counts.to_dict()}")

    if len(test_data) and shares.iloc[0] >= dominance_threshold:
        dominant_narrative = shares.index[0]
        print(
            f"*** WARNING: test set for '{context_label}' is dominated by a SINGLE narrative "
            f"('{dominant_narrative}', {shares.iloc[0] * 100:.1f}% of test) - this is expected "
            f"when the held-out author/group belongs mostly to one narrative, but it means test "
            f"Macro-F1 here mostly reflects performance on that one narrative, not a balanced "
            f"7-narrative evaluation. Reported/documented explicitly, not silently. ***"
        )


def split_random(df):
    """Ordinary random split into train/val/test (the default, as before)."""
    train_data, temp_data = train_test_split(df, test_size=(VAL_SIZE + TEST_SIZE), random_state=42)
    relative_test_size = TEST_SIZE / (VAL_SIZE + TEST_SIZE)
    val_data, test_data = train_test_split(temp_data, test_size=relative_test_size, random_state=42)
    return train_data, val_data, test_data


def split_leave_one_topic(df, held_out_topic, topic_source="lexicon"):
    """
    Leave-One-Topic-Out (lexicon method only - the leak-free, priority-1 method):
    all samples assigned to held_out_topic (by assign_topic_ids_lexicon, which must be
    called beforehand) go entirely to test; the rest is split into train/val.

    Note: when topic_source="bertopic", an early rejection happens in split_data -
    that method (BERTopic) needs access to the full df (fit/transform), so it's handled
    directly by split_leave_one_topic_bertopic rather than through this function.
    """
    if topic_source != "lexicon":
        raise ValueError(
            f"split_leave_one_topic (this function) only supports topic_source='lexicon'. "
            f"For topic_source='bertopic', call split_leave_one_topic_bertopic directly."
        )

    if "topic_id" not in df.columns:
        raise ValueError(
            "split_leave_one_topic requires a 'topic_id' column - "
            "call assign_topic_ids_lexicon(df) first."
        )

    available_topics = sorted(df["topic_id"].unique().tolist())
    if held_out_topic not in available_topics:
        raise ValueError(
            f"held_out_topic={held_out_topic!r} not found in data. "
            f"Available topic ids (lexicon categories): {available_topics}"
        )

    test_data = df[df["topic_id"] == held_out_topic].reset_index(drop=True)
    remaining = df[df["topic_id"] != held_out_topic]

    train_data, val_data = train_test_split(remaining, test_size=VAL_SIZE, random_state=42)

    verify_no_leakage(train_data, val_data, test_data, key_col="topic_id")
    return train_data, val_data, test_data


def split_leave_one_author(df, held_out_author):
    """
    Leave-One-Author-Out: all samples from held_out_author (a real Twitter account
    or Telegram channel only - see _validate_real_authors) go entirely to test; the rest
    is split into train/val.
    """
    _validate_real_authors([held_out_author])

    available_authors = sorted(a for a in df["author_source"].unique().tolist()
                                if not is_synthetic_author(a))
    if held_out_author not in available_authors:
        raise ValueError(
            f"held_out_author='{held_out_author}' not found in data (or is a real-author-only "
            f"list excluding synthetic placeholders). Available real authors "
            f"({len(available_authors)}): {available_authors}"
        )

    held_out_rows = df[df["author_source"] == held_out_author]
    narratives = sorted(held_out_rows["narrative_name"].unique().tolist())
    print(f"[i] Held-out author '{held_out_author}': {len(held_out_rows)} sample(s), "
          f"narrative(s): {narratives}")

    test_data = held_out_rows.reset_index(drop=True)
    remaining = df[df["author_source"] != held_out_author]

    train_data, val_data = train_test_split(remaining, test_size=VAL_SIZE, random_state=42)

    verify_no_leakage(train_data, val_data, test_data, key_col="author_source")
    _print_narrative_distribution_and_warn(test_data, context_label=f"leave_one_author={held_out_author}")
    return train_data, val_data, test_data


def split_leave_group_authors(df, held_out_authors):
    """
    Group split by authors (a stronger generalization test than a single held-out
    author): several real accounts/channels (from the same or different narratives) are
    held out together for test; the rest is split into train/val.
    """
    if not held_out_authors:
        raise ValueError("split_leave_group_authors requires a non-empty list of held_out_authors.")

    _validate_real_authors(held_out_authors)

    available_authors = set(a for a in df["author_source"].unique().tolist()
                             if not is_synthetic_author(a))
    missing = [a for a in held_out_authors if a not in available_authors]
    if missing:
        raise ValueError(
            f"held_out_authors {missing} not found in data (or are synthetic placeholders). "
            f"Available real authors ({len(available_authors)}): {sorted(available_authors)}"
        )

    held_out_rows = df[df["author_source"].isin(held_out_authors)]
    for author in held_out_authors:
        author_rows = held_out_rows[held_out_rows["author_source"] == author]
        narratives = sorted(author_rows["narrative_name"].unique().tolist())
        print(f"[i] Held-out author '{author}': {len(author_rows)} sample(s), narrative(s): {narratives}")

    test_data = held_out_rows.reset_index(drop=True)
    remaining = df[~df["author_source"].isin(held_out_authors)]

    train_data, val_data = train_test_split(remaining, test_size=VAL_SIZE, random_state=42)

    verify_no_leakage(train_data, val_data, test_data, key_col="author_source")
    _print_narrative_distribution_and_warn(
        test_data, context_label=f"leave_group_authors={held_out_authors}"
    )
    return train_data, val_data, test_data


def split_data(df, split_mode, held_out_topic=None, held_out_author=None, held_out_authors=None,
               topic_source="lexicon"):
    """Dispatcher: picks the split function based on split_mode (see SPLIT_MODES)."""
    if split_mode == "random":
        return split_random(df)
    elif split_mode == "leave_one_topic":
        if topic_source == "bertopic":
            return split_leave_one_topic_bertopic(df, held_out_topic)
        return split_leave_one_topic(df, held_out_topic, topic_source=topic_source)
    elif split_mode == "leave_one_author":
        return split_leave_one_author(df, held_out_author)
    elif split_mode == "leave_group_authors":
        return split_leave_group_authors(df, held_out_authors)
    else:
        raise ValueError(f"Unknown split_mode '{split_mode}'. Expected one of {SPLIT_MODES}.")


def print_and_save_split_summary(train_data, val_data, test_data, model_type, run_key):
    """
    Prints and saves: the number of samples per narrative in each split, and the number
    of distinct sources/authors in each split (dataset_source + author_source).
    """
    summary = {}
    print("\n=== Split Summary ===")
    for split_name, split_df in (("train", train_data), ("val", val_data), ("test", test_data)):
        narrative_counts = split_df["narrative_name"].value_counts().to_dict()
        n_dataset_sources = int(split_df["dataset_source"].nunique())
        n_authors = int(split_df["author_source"].nunique())

        print(f"[{split_name}] total={len(split_df)} | narrative_counts={narrative_counts} | "
              f"distinct_dataset_sources={n_dataset_sources} | distinct_authors={n_authors}")

        summary[split_name] = {
            "total": int(len(split_df)),
            "narrative_counts": {str(k): int(v) for k, v in narrative_counts.items()},
            "distinct_dataset_sources": n_dataset_sources,
            "distinct_authors": n_authors,
        }

    summary_file = f"reports/split_summary_{model_type}_{run_key}.json"
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved split summary to '{summary_file}'.")


def build_shared_vocab(train_data):
    """
    Builds a shared vocabulary (NER/SRL) from train_data only. Shared by all three
    models (even though sbert_only doesn't really need it, for consistency/simplicity).
    """
    print("Building and saving new deterministic vocabulary...")

    def clean_for_vocab(text):
        return re.sub(r'[^\w\s]', '', text.lower())

    all_text = " ".join(train_data['text'])
    all_words_set = set(clean_for_vocab(all_text).split())
    sorted_words = sorted(list(all_words_set))
    shared_vocab = {word: i for i, word in enumerate(sorted_words)}

    with open("data/cache/shared_vocab.json", "w", encoding="utf-8") as f:
        json.dump(shared_vocab, f, ensure_ascii=False, indent=4)

    return shared_vocab


def extract_all(data, desc, detector):
    features_list = []
    for _, row in tqdm(data.iterrows(), total=len(data), desc=desc):
        text = str(row['text'])
        feat = detector.extract_features(text)
        features_list.append((feat, int(row['label'])))
    return features_list


def get_or_extract_features(model_type, split_mode, held_out_topic, held_out_author,
                             held_out_authors, topic_source, detector, train_data, val_data, test_data):
    """
    Cache mechanism: extracts features (slow, once) and saves to disk in a dedicated
    file per (model_type, split_mode, held_out_*) - see get_cache_file.
    """
    cache_file = get_cache_file(model_type, split_mode, held_out_topic, held_out_author,
                                 held_out_authors, topic_source)

    if os.path.exists(cache_file):
        print(f"\nFound cached features at '{cache_file}'. Loading instantly...")
        cached_data = torch.load(cache_file, weights_only=False)
        print("Loaded successfully! Skipping extraction.")
        return cached_data['train'], cached_data['val'], cached_data['test']

    print(f"\nNo cache found at '{cache_file}'. Starting Feature Extraction "
          f"(this will take a while)...")
    train_features = extract_all(train_data, "Extracting Train Features", detector)
    val_features = extract_all(val_data, "Extracting Val Features", detector)
    test_features = extract_all(test_data, "Extracting Test Features", detector)

    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    print(f"\nSaving extracted features to '{cache_file}'...")
    torch.save({'train': train_features, 'val': val_features, 'test': test_features}, cache_file)
    print("Features saved successfully! Next time it will load in seconds.")

    return train_features, val_features, test_features


def compute_metrics(true_labels, predicted_labels):
    """
    Accuracy + Macro-Precision/Recall/F1 (for comparing models) + per-class metrics
    (precision/recall/f1/support per narrative) + confusion matrix - all required for the
    research reports.
    """
    accuracy = accuracy_score(true_labels, predicted_labels)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average='macro', zero_division=0
    )

    labels_present = sorted(set(true_labels) | set(predicted_labels))
    per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
        true_labels, predicted_labels, labels=labels_present, zero_division=0
    )
    per_class = {
        index_to_label[label_idx]: {
            "precision": float(per_class_precision[i]),
            "recall": float(per_class_recall[i]),
            "f1": float(per_class_f1[i]),
            "support": int(per_class_support[i]),
        }
        for i, label_idx in enumerate(labels_present)
    }

    cm = confusion_matrix(true_labels, predicted_labels, labels=labels_present).tolist()
    cm_labels = [index_to_label[label_idx] for label_idx in labels_present]

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": cm,
        "confusion_matrix_labels": cm_labels,
    }


def save_confusion_matrix_csv(metrics, path):
    """Saves the confusion matrix as a readable CSV (rows=true, cols=predicted)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    labels = metrics["confusion_matrix_labels"]
    cm_df = pd.DataFrame(metrics["confusion_matrix"], index=labels, columns=labels)
    cm_df.to_csv(path, encoding="utf-8-sig")
    print(f"Saved confusion matrix to '{path}'.")


def evaluate(detector, features_labels, loss_fn=None):
    """
    Runs the model (classify_features only - fast, no re-extraction of features) on a
    set of already-extracted features, and returns metrics (including per-class +
    confusion matrix) + (optionally) average loss.

    This generic function is also the natural future extension point for an ablation
    study: it can be called with features_labels where one feature group is
    zeroed-out/removed (e.g. zero out features["agenda_ideology"] before the call) in
    order to measure the drop in macro_f1 caused by removing that feature group.
    """
    detector.eval()
    true_labels, predicted_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for features, label_idx in features_labels:
            label = torch.tensor([label_idx], dtype=torch.long)
            probs = detector.classify_features(features)

            if probs.dim() == 1:
                probs = probs.unsqueeze(0)

            predicted = torch.argmax(probs, dim=-1)
            true_labels.append(label_idx)
            predicted_labels.append(predicted.item())

            if loss_fn is not None:
                total_loss += loss_fn(torch.log(probs + 1e-8), label).item()

    metrics = compute_metrics(true_labels, predicted_labels)
    avg_loss = (total_loss / len(features_labels)) if loss_fn is not None else None
    return metrics, avg_loss, true_labels, predicted_labels


def save_comparison_result(model_type, run_key, split_name, metrics, extra=None):
    """
    Accumulates the results of every run into one shared JSON file (RESULTS_FILE), so
    that in the end it's easy to compare accuracy/macro-F1/precision/recall (also
    per-class + confusion matrix) between baseline_fusion / sbert_only / hybrid,
    and between different split_modes (random / leave_one_topic / leave_one_author).
    Structure: {model_type: {run_key: {split_name: {...metrics}}}}
    """
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

    results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            results = json.load(f)

    results.setdefault(model_type, {}).setdefault(run_key, {})
    entry = dict(metrics)
    if extra:
        entry.update(extra)
    results[model_type][run_key][split_name] = entry

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved '{split_name}' results for model '{model_type}' (run '{run_key}') to '{RESULTS_FILE}'.")


def train(model_type, split_mode="random", held_out_topic=None, held_out_author=None,
          held_out_authors=None, topic_source="lexicon",
          epochs=EPOCHS, batch_size=BATCH_SIZE, patience=3, lr=LEARNING_RATE):
    run_key = run_key_for(split_mode, held_out_topic, held_out_author, held_out_authors, topic_source)
    print(f"\n=== Split mode: '{split_mode}' (run_key='{run_key}') ===")

    df = load_raw_data()

    if split_mode == "leave_one_topic" and topic_source == "lexicon":
        # The bertopic method (if topic_source="bertopic") performs fit+transform inside
        # split_leave_one_topic_bertopic itself (see split_data) - not here, since it
        # needs to control the fit/transform split (FIT_POOL/CANDIDATE_POOL) itself.
        df = assign_topic_ids_lexicon(df)

    train_data, val_data, test_data = split_data(
        df, split_mode, held_out_topic, held_out_author, held_out_authors, topic_source
    )
    print(f"Split sizes -> train: {len(train_data)}, val: {len(val_data)}, test: {len(test_data)}")

    print("Cleaning and Truncating text...")
    # Truncate overly long texts first, to prevent the emotion model from crashing
    for split_df in (train_data, val_data, test_data):
        split_df["text"] = split_df["text"].astype(str).str.slice(0, 3000)

    print_and_save_split_summary(train_data, val_data, test_data, model_type, run_key)

    shared_vocab = build_shared_vocab(train_data)

    print(f"\n=== Building model: '{model_type}' ===")
    detector = build_model(model_type, ner_vocab=shared_vocab, srl_vocab=shared_vocab)

    train_features, val_features, test_features = get_or_extract_features(
        model_type, split_mode, held_out_topic, held_out_author, held_out_authors, topic_source,
        detector, train_data, val_data, test_data
    )

    print("\n--- Starting Training ---")
    optimizer = optim.Adam(detector.parameters(), lr=lr)
    loss_fn = nn.NLLLoss()

    best_val_macro_f1 = -1.0
    epochs_no_improve = 0
    checkpoint_file = get_checkpoint_file(model_type, split_mode, held_out_topic, held_out_author,
                                          held_out_authors, topic_source)
    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)

    best_val_metrics = None

    for epoch in range(epochs):
        detector.train()
        total_train_loss = 0.0
        train_correct = 0

        # Reset outside the sentence loop, to start with a clean accumulation
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

            # Accumulate gradients
            loss.backward()

            if (i + 1) % batch_size == 0 or (i + 1) == len(train_features):
                optimizer.step()
                optimizer.zero_grad()

            total_train_loss += loss.item() * batch_size

        # Validation step - checkpoint is chosen based on Macro-F1 (not loss)
        val_metrics, avg_val_loss, _, _ = evaluate(detector, val_features, loss_fn)

        print(
            f"Epoch {epoch + 1}: Train Acc: {100 * train_correct / len(train_features):.2f}% | "
            f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_metrics['accuracy'] * 100:.2f}% | "
            f"Val Macro-F1: {val_metrics['macro_f1']:.4f}"
        )

        if val_metrics['macro_f1'] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics['macro_f1']
            best_val_metrics = val_metrics
            epochs_no_improve = 0
            torch.save(detector.state_dict(), checkpoint_file)
            print(f">>> New best model saved (Val Macro-F1: {best_val_macro_f1:.4f}) -> '{checkpoint_file}'")
        else:
            epochs_no_improve += 1
            print(f">>> No improvement in Validation Macro-F1 for {epochs_no_improve} epoch(s).")

            if epochs_no_improve >= patience:
                print(f"\n[!] Early Stopping Triggered! Training halted at epoch {epoch + 1}.")
                break

    # ==========================================================
    # Final evaluation on the Test set (unseen, used neither for training nor for model selection)
    # ==========================================================
    print("\n" + "=" * 60)
    print(f"FINAL TEST EVALUATION - model_type='{model_type}' split_mode='{split_mode}' "
          f"(Best Checkpoint by Val Macro-F1)")
    print("=" * 60)

    detector.load_state_dict(torch.load(checkpoint_file))
    detector.eval()

    test_metrics, _, test_true_labels, test_predicted_labels = evaluate(detector, test_features)

    target_names = [index_to_label[i] for i in range(len(NARRATIVES))
                     if i in test_true_labels or i in test_predicted_labels]

    report = classification_report(
        test_true_labels, test_predicted_labels,
        target_names=target_names, digits=4, zero_division=0
    )
    print(report)
    print(f"Test Accuracy: {test_metrics['accuracy'] * 100:.2f}% | "
          f"Test Macro-Precision: {test_metrics['macro_precision']:.4f} | "
          f"Test Macro-Recall: {test_metrics['macro_recall']:.4f} | "
          f"Test Macro-F1: {test_metrics['macro_f1']:.4f}")
    print("\nConfusion Matrix (rows=true, cols=predicted):")
    cm_df = pd.DataFrame(
        test_metrics["confusion_matrix"],
        index=test_metrics["confusion_matrix_labels"],
        columns=test_metrics["confusion_matrix_labels"],
    )
    print(cm_df)
    print("=" * 60)

    save_comparison_result(model_type, run_key, "validation", best_val_metrics)
    save_comparison_result(model_type, run_key, "test", test_metrics)
    save_confusion_matrix_csv(test_metrics, f"reports/confusion_matrix_{model_type}_{run_key}_test.csv")

    # Print the learned module weights - relevant only for baseline_fusion (NarrativeDetector)
    if hasattr(detector, "fusion_network") and hasattr(detector.fusion_network, 'module_weights'):
        print("\n--- Module Importance (Learned Weights) ---")
        learned_weights = torch.softmax(detector.fusion_network.module_weights, dim=0)
        print(f"NER Weight: {learned_weights[0].item() * 100:.1f}%")
        print(f"Stance Weight: {learned_weights[1].item() * 100:.1f}%")
        print(f"SRL Weight: {learned_weights[2].item() * 100:.1f}%")
        print(f"Emotion Weight: {learned_weights[3].item() * 100:.1f}%")
        print("=" * 60)

    return detector, test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train one of the three narrative-detection models.")
    parser.add_argument(
        "--model", choices=MODEL_TYPES, default=MODEL_TYPE,
        help=f"Which model to train (default from config.py: '{MODEL_TYPE}')."
    )
    parser.add_argument(
        "--split", choices=SPLIT_MODES, default="random",
        help="Split strategy: 'random' (default), 'leave_one_topic', 'leave_one_author', "
             "or 'leave_group_authors'."
    )
    parser.add_argument(
        "--topic-source", choices=TOPIC_SOURCES, default="lexicon",
        help="How topic_id is assigned for --split leave_one_topic: 'lexicon' (default, "
             "leak-free fixed AGENDA_LEXICON categories) or 'bertopic' (fits a fresh BERTopic "
             "model on a train-only pool per run - best-effort, Colab-only, topic ids not "
             "comparable across runs)."
    )
    parser.add_argument(
        "--held-out-topic", type=str, default=None,
        help="Topic id to hold out entirely for test (required for --split leave_one_topic). "
             "A lexicon category name when --topic-source lexicon (default), or an integer "
             "BERTopic topic id when --topic-source bertopic."
    )
    parser.add_argument(
        "--held-out-author", type=str, default=None,
        help="Real account/channel (author_source) to hold out entirely for test "
             "(required for --split leave_one_author). Synthetic gemini/gpt placeholders "
             "are not allowed."
    )
    parser.add_argument(
        "--held-out-authors", type=str, default=None,
        help="Comma-separated list of real accounts/channels to hold out together for test "
             "(required for --split leave_group_authors), e.g. 'IDF,khamenei_ir'. Synthetic "
             "gemini/gpt placeholders are not allowed."
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    args = parser.parse_args()

    if args.split == "leave_one_topic" and args.held_out_topic is None:
        parser.error("--split leave_one_topic requires --held-out-topic <topic_id>")
    if args.split == "leave_one_author" and args.held_out_author is None:
        parser.error("--split leave_one_author requires --held-out-author <account_or_channel>")
    if args.split == "leave_group_authors" and args.held_out_authors is None:
        parser.error("--split leave_group_authors requires --held-out-authors <a,b,c>")

    # bertopic topic ids are integers assigned by this run's fresh fit; lexicon topic ids are
    # fixed category name strings (see assign_topic_ids_lexicon) - cast accordingly.
    held_out_topic = args.held_out_topic
    if held_out_topic is not None and args.topic_source == "bertopic":
        held_out_topic = int(held_out_topic)

    held_out_authors_list = None
    if args.held_out_authors is not None:
        held_out_authors_list = [a.strip() for a in args.held_out_authors.split(",") if a.strip()]

    print(f"Selected model type: '{args.model}' | split mode: '{args.split}'")
    train(
        model_type=args.model,
        split_mode=args.split,
        held_out_topic=held_out_topic,
        held_out_author=args.held_out_author,
        held_out_authors=held_out_authors_list,
        topic_source=args.topic_source,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        lr=args.lr,
    )

# ==========================================================================
# Remaining future extension points (not implemented right now, intentionally - just
# infrastructure):
#
# - Leave-One-Source-Out (LOSO) by dataset_source (twitter/telegram/gemini/gpt
#   held out entirely, unlike leave_one_author/leave_group_authors which hold out a
#   single account/channel or group): could add split_leave_one_source(df, held_out_source)
#   which filters by dataset_source using the exact same logic as split_leave_one_author.
# - Ablation study: evaluate() above accepts a ready-made features_labels - could add
#   ablate_feature_group(features_labels, group_name) which zeroes out one feature group
#   (e.g. "agenda_ideology" or "emotion") before calling evaluate, and measures the
#   drop in macro_f1 relative to the full run.
# ==========================================================================