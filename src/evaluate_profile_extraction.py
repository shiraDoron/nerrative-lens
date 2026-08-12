"""
Evaluation script for the Narrative Fingerprint entity/role/relation MVP - to be
run AFTER a human annotator has filled in the gold_* columns in the review CSVs
produced by build_profile_prototype.py (<prefix>_entities.csv / _relations.csv).

Does NOT touch any classification model (fusion.py) and does NOT run any new
extraction - it only reads already-produced CSVs and computes metrics.

Computes:
  - Entity->Role: per-class Precision/Recall/F1 + Macro-F1 (over annotated rows
    only, i.e. rows where gold_role is filled in)
  - Confusion matrix of roles (gold vs predicted)
  - Per relation-type Precision/Recall/F1 (over annotated rows)
  - Agency accuracy (gold_agency vs predicted_agency)
  - Coverage: % of ALL rows (annotated or not) where the system committed to a
    substantive (non "unknown"/"uncertain"/"(none detected)") prediction - this
    can be computed even BEFORE annotation, since it doesn't need gold labels.
  - Error analysis by signal type (dependency / rhetoric cue / agency /
    predicate-action), based on parsing the predicted_*_method column: for each
    signal bucket, what fraction of rows where that signal fired were judged
    "incorrect" by the annotator (correct_incorrect_uncertain column) - lets us
    see which signal is driving mistakes.

If no rows have been annotated yet (gold_* columns all empty), the script still
prints the coverage stats and a clear message that no P/R/F1 can be computed
until annotation is done - it will NOT crash or silently fabricate metrics.

Run (after filling in gold_* columns in Excel/Sheets and re-saving as CSV):
    python src/evaluate_profile_extraction.py \\
        --entities-csv reports/profiler_prototype/calibration_entities.csv \\
        --relations-csv reports/profiler_prototype/calibration_relations.csv
"""

import argparse
import os

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from entity_role_tagger import ROLE_LABELS
from relation_extractor import RELATION_TYPES

NON_SUBSTANTIVE_ROLES = {"unknown", "uncertain"}
NON_SUBSTANTIVE_RELATIONS = {"uncertain", "(none detected)"}


def _non_empty(series):
    return series.notna() & (series.astype(str).str.strip() != "")


def _signal_buckets_from_method(method):
    """Maps a predicted_*_method string (e.g. 'dependency:nsubj+verb_lexicon:
    aggressor(attack)+agency_voice:active_subject+rhetoric_lexicon:...') to the
    coarse signal buckets the user wants error analysis grouped by."""
    buckets = set()
    if not isinstance(method, str):
        return buckets
    if "dependency" in method:
        buckets.add("dependency")
    if "rhetoric" in method:
        buckets.add("rhetoric_cue")
    if "agency" in method:
        buckets.add("agency")
    if any(k in method for k in (
        "verb_lexicon", "causal_phrase", "causal_connective", "blame_phrase", "proposal_trigger",
    )):
        buckets.add("predicate_action")
    return buckets


def compute_coverage(df, predicted_col, non_substantive_values):
    """% of ALL rows (no annotation needed) where the system committed to a
    substantive (non-neutral/unknown/uncertain) prediction."""
    total = len(df)
    if total == 0:
        return 0.0
    committed = (~df[predicted_col].isin(non_substantive_values)).sum()
    return round(committed / total * 100, 1)


def evaluate_entity_roles(entities_df, out_dir):
    annotated = entities_df[_non_empty(entities_df["gold_role"])].copy()
    coverage_pct = compute_coverage(entities_df, "predicted_role", NON_SUBSTANTIVE_ROLES)

    print(f"\n=== Entity -> Role ===")
    print(f"Coverage (predicted a substantive role, not unknown/uncertain): {coverage_pct}% "
          f"of {len(entities_df)} entity mentions.")

    if annotated.empty:
        print("No annotated rows (gold_role is empty for all rows) - fill in gold_role to "
              "compute Precision/Recall/F1/confusion matrix.")
        return {"coverage_pct": coverage_pct, "n_annotated": 0}

    labels = list(ROLE_LABELS)
    y_true = annotated["gold_role"].astype(str)
    y_pred = annotated["predicted_role"].astype(str)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_class = pd.DataFrame({
        "role": labels, "precision": precision, "recall": recall, "f1": f1, "support": support,
    })

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"gold_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])

    print(f"Annotated rows: {len(annotated)}")
    print(per_class.to_string(index=False))
    print(f"Macro-F1: {round(macro_f1, 3)} (macro-precision {round(macro_precision, 3)}, "
          f"macro-recall {round(macro_recall, 3)})")

    os.makedirs(out_dir, exist_ok=True)
    per_class.to_csv(os.path.join(out_dir, "eval_entity_role_per_class.csv"), index=False)
    cm_df.to_csv(os.path.join(out_dir, "eval_entity_role_confusion_matrix.csv"))

    agency_result = None
    if "gold_agency" in annotated.columns:
        agency_annotated = annotated[_non_empty(annotated["gold_agency"])]
        if not agency_annotated.empty:
            accuracy = (agency_annotated["gold_agency"] == agency_annotated["predicted_agency"]).mean()
            agency_result = round(accuracy, 3)
            print(f"Agency accuracy: {agency_result} (n={len(agency_annotated)})")

    return {
        "coverage_pct": coverage_pct,
        "n_annotated": len(annotated),
        "macro_f1": round(macro_f1, 3),
        "per_class": per_class,
        "confusion_matrix": cm_df,
        "agency_accuracy": agency_result,
    }


def evaluate_relations(relations_df, out_dir):
    annotated = relations_df[_non_empty(relations_df["gold_relation"])].copy()
    coverage_pct = compute_coverage(relations_df, "predicted_relation", NON_SUBSTANTIVE_RELATIONS)

    print(f"\n=== Relation extraction ===")
    print(f"Coverage (predicted a specific relation type, not uncertain/none): {coverage_pct}% "
          f"of {len(relations_df)} relation candidates.")

    if annotated.empty:
        print("No annotated rows (gold_relation is empty for all rows) - fill in gold_relation "
              "to compute per-type Precision/Recall/F1.")
        return {"coverage_pct": coverage_pct, "n_annotated": 0}

    labels = list(RELATION_TYPES) + ["uncertain"]
    y_true = annotated["gold_relation"].astype(str)
    y_pred = annotated["predicted_relation"].astype(str)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_type = pd.DataFrame({
        "relation": labels, "precision": precision, "recall": recall, "f1": f1, "support": support,
    })

    print(f"Annotated rows: {len(annotated)}")
    print(per_type.to_string(index=False))

    os.makedirs(out_dir, exist_ok=True)
    per_type.to_csv(os.path.join(out_dir, "eval_relation_per_type.csv"), index=False)

    return {"coverage_pct": coverage_pct, "n_annotated": len(annotated), "per_type": per_type}


def error_analysis_by_signal(df, method_col, out_path):
    """For rows the annotator marked 'incorrect', tallies which signal buckets
    (dependency/rhetoric_cue/agency/predicate_action) were involved in the
    prediction's method string, vs. how often that bucket appears overall -
    to see which signal correlates with mistakes."""
    if "correct_incorrect_uncertain" not in df.columns:
        return None
    judged = df[_non_empty(df["correct_incorrect_uncertain"])].copy()
    if judged.empty:
        print("No correct_incorrect_uncertain judgments filled in yet - skipping error-by-signal analysis.")
        return None

    bucket_stats = {}
    for _, row in judged.iterrows():
        buckets = _signal_buckets_from_method(row.get(method_col))
        judgment = str(row["correct_incorrect_uncertain"]).strip().lower()
        for bucket in buckets:
            stats = bucket_stats.setdefault(bucket, {"n_total": 0, "n_incorrect": 0, "n_uncertain": 0})
            stats["n_total"] += 1
            if judgment == "incorrect":
                stats["n_incorrect"] += 1
            elif judgment == "uncertain":
                stats["n_uncertain"] += 1

    rows = []
    for bucket, stats in bucket_stats.items():
        error_rate = round(stats["n_incorrect"] / stats["n_total"], 3) if stats["n_total"] else 0.0
        rows.append({
            "signal": bucket, "n_rows_with_signal": stats["n_total"],
            "n_judged_incorrect": stats["n_incorrect"], "n_judged_uncertain": stats["n_uncertain"],
            "error_rate": error_rate,
        })
    result_df = pd.DataFrame(rows).sort_values("error_rate", ascending=False)

    print(f"\nError analysis by signal ({method_col}):")
    print(result_df.to_string(index=False))
    result_df.to_csv(out_path, index=False)
    return result_df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entities-csv", type=str, default="reports/profiler_prototype/calibration_entities.csv")
    parser.add_argument("--relations-csv", type=str, default="reports/profiler_prototype/calibration_relations.csv")
    parser.add_argument("--out-dir", type=str, default="reports/profiler_prototype")
    args = parser.parse_args()

    if not os.path.exists(args.entities_csv):
        raise FileNotFoundError(f"Entities CSV not found: {args.entities_csv}")
    if not os.path.exists(args.relations_csv):
        raise FileNotFoundError(f"Relations CSV not found: {args.relations_csv}")

    entities_df = pd.read_csv(args.entities_csv)
    relations_df = pd.read_csv(args.relations_csv)

    evaluate_entity_roles(entities_df, args.out_dir)
    evaluate_relations(relations_df, args.out_dir)

    error_analysis_by_signal(
        entities_df, "role_method", os.path.join(args.out_dir, "error_analysis_entities_by_signal.csv")
    )
    error_analysis_by_signal(
        relations_df, "relation_method", os.path.join(args.out_dir, "error_analysis_relations_by_signal.csv")
    )

    print("\nDone. (Re-run this script anytime after adding more gold annotations.)")


if __name__ == "__main__":
    main()
