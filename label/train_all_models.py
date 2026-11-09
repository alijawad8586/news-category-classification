"""Train, compare, and save news-category classifiers without data leakage."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parent
RANDOM_STATE = 42


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "balanced_news_data.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "model_outputs")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep repeated articles. This can produce misleadingly high accuracy.",
    )
    return parser.parse_args()


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = re.sub(r"https?://\S+|www\.\S+", " ", str(value), flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def first_existing(df: pd.DataFrame, candidates: list[str], kind: str) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise ValueError(f"No {kind} column found. Tried: {', '.join(candidates)}")


def load_data(path: Path, keep_duplicates: bool) -> tuple[pd.DataFrame, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    if "category" not in df.columns:
        raise ValueError("The dataset needs a 'category' column.")

    title_column = first_existing(
        df, ["title_clean", "article_title", "title", "headline"], "title"
    )
    content_column = first_existing(
        df,
        ["content_clean", "article_content", "content", "text", "description"],
        "content",
    )
    title = df[title_column].map(clean_text)
    content = df[content_column].map(clean_text)
    original_rows = len(df)

    # Headlines usually contain the strongest category signal, so weight them
    # by repeating them before TF-IDF extraction.
    df = df.assign(
        combined_text=(title + " " + title + " " + content).str.strip(),
        category=df["category"].astype(str).str.strip(),
    )
    df = df[(df["combined_text"] != "") & (df["category"] != "")].copy()

    text_key = (
        df["combined_text"]
        .str.lower()
        .str.replace(r"\W+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    duplicate_rows = int(text_key.duplicated().sum())

    # Remove contradictory labels for identical text rather than teaching the
    # model an impossible mapping.
    label_counts = (
        pd.DataFrame({"key": text_key, "category": df["category"]})
        .groupby("key")["category"]
        .nunique()
    )
    conflict_keys = set(label_counts[label_counts > 1].index)
    conflict_rows = int(text_key.isin(conflict_keys).sum())
    if conflict_rows:
        keep = ~text_key.isin(conflict_keys)
        df, text_key = df.loc[keep].copy(), text_key.loc[keep]
    if not keep_duplicates:
        df = df.loc[~text_key.duplicated()].copy()

    counts = df["category"].value_counts()
    if len(counts) < 2 or counts.min() < 2:
        raise ValueError("Every category needs at least two unique articles.")

    audit = {
        "original_rows": original_rows,
        "rows_used": len(df),
        "duplicate_rows_found": duplicate_rows,
        "conflicting_label_rows_removed": conflict_rows,
        "duplicates_kept": keep_duplicates,
        "title_column": title_column,
        "content_column": content_column,
        "class_counts": counts.sort_index().to_dict(),
    }
    return df, audit


def classifiers() -> dict[str, object]:
    """Return three candidates for each of the four requested model families."""
    return {
        **{
            f"svm_c{c:g}": LinearSVC(
                C=c, class_weight="balanced", random_state=RANDOM_STATE
            )
            for c in (0.5, 1.0, 2.5)
        },
        **{
            f"logistic_regression_c{c:g}": LogisticRegression(
                C=c,
                class_weight="balanced",
                max_iter=1500,
                solver="liblinear",
                random_state=RANDOM_STATE,
            )
            for c in (1.0, 3.0, 6.0)
        },
        **{
            f"decision_tree_depth{depth}": DecisionTreeClassifier(
                criterion="gini",
                max_depth=depth,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
            for depth in (30, 50, 80)
        },
        **{
            f"random_forest_{trees}trees": RandomForestClassifier(
                n_estimators=trees,
                max_depth=depth,
                min_samples_split=3,
                min_samples_leaf=1,
                max_features="sqrt",
                class_weight="balanced_subsample",
                # Outer parallelism distributes the 12 candidates.
                n_jobs=1,
                random_state=RANDOM_STATE,
            )
            for trees, depth in ((150, 40), (250, 60), (350, None))
        },
    }


def model_family(candidate_name: str) -> str:
    if candidate_name.startswith("logistic_regression"):
        return "logistic_regression"
    if candidate_name.startswith("decision_tree"):
        return "decision_tree"
    if candidate_name.startswith("random_forest"):
        return "random_forest"
    if candidate_name.startswith("svm"):
        return "svm"
    raise ValueError(f"Unknown model family: {candidate_name}")


def train_candidate(
    name: str, model: object, X_train: object, y_train: pd.Series,
    X_test: object, y_test: pd.Series, labels: list[str]
) -> tuple[str, object, dict]:
    """Fit and score one candidate; called concurrently by joblib."""
    started = time.perf_counter()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    elapsed = time.perf_counter() - started
    result = {
        "model": name,
        "model_family": model_family(name),
        "accuracy": accuracy_score(y_test, prediction),
        "macro_f1": f1_score(y_test, prediction, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(y_test, prediction),
        "training_seconds": elapsed,
        "classification_report": classification_report(
            y_test, prediction, output_dict=True, zero_division=0
        ),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_test, prediction, labels=labels),
    }
    return name, model, result


def safe_json(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    args = arguments()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("NEWS CATEGORY CLASSIFICATION - LEAKAGE-SAFE TRAINING")
    print("=" * 76)
    df, audit = load_data(args.data.resolve(), args.keep_duplicates)
    print(f"Rows used: {len(df):,} / {audit['original_rows']:,}")
    print(f"Duplicate rows found: {audit['duplicate_rows_found']:,}")
    for category, count in audit["class_counts"].items():
        print(f"  {category:<18} {count:>6,}")

    train_text, test_text, y_train, y_test = train_test_split(
        df["combined_text"],
        df["category"],
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=df["category"],
    )
    print(f"Train: {len(train_text):,}; test: {len(test_text):,}")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        sublinear_tf=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=60_000,
    )
    print("\nBuilding sparse TF-IDF features ...")
    X_train = vectorizer.fit_transform(train_text)
    X_test = vectorizer.transform(test_text)
    print(f"Feature matrix: {X_train.shape[0]:,} x {X_train.shape[1]:,}")

    labels = sorted(y_test.unique())
    candidates = classifiers()
    workers = min(12, os.cpu_count() or 1, len(candidates))
    print(f"\nTraining {len(candidates)} candidates on {workers} parallel workers ...")
    # Threads share the sparse matrices instead of copying them into 12 costly
    # process-local buffers. The compiled sklearn solvers release Python's GIL.
    completed = Parallel(n_jobs=workers, prefer="threads", verbose=10)(
        delayed(train_candidate)(
            name, model, X_train, y_train, X_test, y_test, labels
        )
        for name, model in candidates.items()
    )

    trained_models = {name: model for name, model, _ in completed}
    results = [result for _, _, result in completed]
    for name, _, result in completed:
        print(
            f"  {name}: accuracy={result['accuracy']:.4f}, "
            f"macro_f1={result['macro_f1']:.4f}, "
            f"time={result['training_seconds']:.1f}s"
        )

    results.sort(key=lambda row: (row["macro_f1"], row["accuracy"]), reverse=True)
    family_best = []
    for family in ("logistic_regression", "decision_tree", "random_forest", "svm"):
        winner = next(row for row in results if row["model_family"] == family)
        family_best.append(winner)
        joblib.dump(
            {
                "model": trained_models[winner["model"]],
                "vectorizer": vectorizer,
                "data_audit": audit,
                "metrics": winner,
            },
            output_dir / f"model_{family}.joblib",
            compress=3,
        )

    family_best.sort(
        key=lambda row: (row["macro_f1"], row["accuracy"]), reverse=True
    )
    best = family_best[0]
    joblib.dump(
        {
            "model": trained_models[best["model"]],
            "vectorizer": vectorizer,
            "data_audit": audit,
            "metrics": best,
        },
        output_dir / "best_news_category_model.joblib",
        compress=3,
    )

    table = [
        {
            "model": row["model"],
            "accuracy_percent": round(row["accuracy"] * 100, 2),
            "macro_f1_percent": round(row["macro_f1"] * 100, 2),
            "balanced_accuracy_percent": round(row["balanced_accuracy"] * 100, 2),
            "training_seconds": round(row["training_seconds"], 2),
        }
        for row in family_best
    ]
    results_frame = pd.DataFrame(table)
    results_frame.to_csv(output_dir / "model_accuracies.csv", index=False)
    with (output_dir / "training_summary.txt").open("w", encoding="utf-8") as file:
        file.write("NEWS CATEGORY MODEL TRAINING RESULTS\n")
        file.write("=" * 76 + "\n")
        file.write(results_frame.to_string(index=False))
        file.write(f"\n\nBest overall candidate: {best['model']}\n")
    details = {
        "training_date": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(args.data.resolve()),
        "random_state": RANDOM_STATE,
        "test_size": args.test_size,
        "selection_metric": "macro_f1",
        "best_model": best["model"],
        "data_audit": audit,
        "best_model_per_family": family_best,
        "all_tuning_candidates": results,
    }
    with (output_dir / "training_details.json").open("w", encoding="utf-8") as file:
        json.dump(safe_json(details), file, indent=2, ensure_ascii=False)

    print("\n" + "=" * 76)
    print(results_frame.to_string(index=False))
    print(f"\nBest model: {best['model']}")
    print(f"Outputs saved in: {output_dir}")


if __name__ == "__main__":
    main()
