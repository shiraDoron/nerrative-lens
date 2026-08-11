"""
Builds a single cross-model comparison table from reports/model_comparison_results.json
(the shared results file written by train.py's save_comparison_result() after every run).

Usage (from repo root, after running the 3 models for a given run):
    python src/compare_models.py --run-key random --split test
    python src/compare_models.py --run-key loto_lexicon_topic<category_name> --split test
    python src/compare_models.py --run-key loao_IDF --split test

Only needs pandas/json (no torch/transformers) - safe to run locally even though the
actual training runs happen on Colab; just copy reports/model_comparison_results.json
back locally (or run this script on Colab too).
"""
import argparse
import json
import os

import pandas as pd

from config import MODEL_TYPES, NARRATIVES

RESULTS_FILE = "reports/model_comparison_results.json"


def load_results(results_file=RESULTS_FILE):
    if not os.path.exists(results_file):
        raise FileNotFoundError(
            f"'{results_file}' not found. Run train.py for each model first "
            f"(it calls save_comparison_result() automatically after every run)."
        )
    with open(results_file, "r", encoding="utf-8") as f:
        return json.load(f)


def build_comparison_table(results, run_key, split_name="test", model_types=MODEL_TYPES):
    """
    One row per model_type, with the shared summary metrics + one column per narrative's
    F1 score (from per_class). Models missing this run_key/split_name are skipped with a
    printed warning (so a partial comparison - e.g. only 2 of 3 models run so far - still
    works instead of crashing).
    """
    rows = []
    for model_type in model_types:
        entry = results.get(model_type, {}).get(run_key, {}).get(split_name)
        if entry is None:
            print(f"[!] No '{split_name}' results for model_type='{model_type}' run_key='{run_key}' "
                  f"- skipping (run train.py for this model/run first).")
            continue

        row = {
            "model_type": model_type,
            "accuracy": entry["accuracy"],
            "macro_precision": entry["macro_precision"],
            "macro_recall": entry["macro_recall"],
            "macro_f1": entry["macro_f1"],
        }
        per_class = entry.get("per_class", {})
        for narrative in NARRATIVES:
            f1 = per_class.get(narrative, {}).get("f1")
            row[f"f1_{narrative}"] = f1
        rows.append(row)

    if not rows:
        raise ValueError(
            f"No results found for run_key='{run_key}' split_name='{split_name}' for any of "
            f"{model_types}. Check the run_key spelling and that train.py has been run."
        )
    return pd.DataFrame(rows)


def dataframe_to_markdown(df):
    """Minimal Markdown table writer (avoids a hard dependency on the optional
    'tabulate' package, which pandas.DataFrame.to_markdown() requires)."""
    header = "| " + " | ".join(df.columns) + " |"
    separator = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = [header, separator]
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Build a cross-model comparison table from reports/model_comparison_results.json."
    )
    parser.add_argument("--run-key", default="random",
                        help="run_key to compare across models, e.g. 'random', "
                             "'loto_lexicon_topic<category>', 'loao_<account>', "
                             "'loago_<account1>_<account2>'.")
    parser.add_argument("--split", default="test", choices=("validation", "test"),
                        help="Which split's metrics to compare (default: test).")
    parser.add_argument("--results-file", default=RESULTS_FILE)
    parser.add_argument("--out-prefix", default=None,
                        help="If set, also saves the table to '<out-prefix>.csv' and "
                             "'<out-prefix>.md'. Default: reports/comparison_<run_key>_<split>.")
    args = parser.parse_args()

    results = load_results(args.results_file)
    table = build_comparison_table(results, args.run_key, args.split)

    print(f"\n=== Model comparison: run_key='{args.run_key}' split='{args.split}' ===")
    print(table.to_string(index=False))

    out_prefix = args.out_prefix or f"reports/comparison_{args.run_key}_{args.split}"
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    table.to_csv(f"{out_prefix}.csv", index=False)
    with open(f"{out_prefix}.md", "w", encoding="utf-8") as f:
        f.write(dataframe_to_markdown(table))
    print(f"\nSaved comparison table to '{out_prefix}.csv' and '{out_prefix}.md'.")


if __name__ == "__main__":
    main()
