"""
Predict news categories for an Excel/CSV file using the saved Logistic Regression pipeline.

Required files:
  1. This script: predict_categories.py
  2. Trained model: news_category_logreg_pipeline.pkl
  3. Input Excel/CSV file containing at least: title, content, date
     Optional: author
     Default input: label/news_final_clean.csv

The script creates a new output file with a predicted "category" column.
It does NOT train a new model.
"""

import argparse
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
UNKNOWN_AUTHOR = "unknown author"


def find_column(df, possible_names, required=True):
    """Find a column without being sensitive to upper/lower case or extra spaces."""
    lookup = {str(col).strip().lower(): col for col in df.columns}

    for name in possible_names:
        found = lookup.get(name.lower())
        if found is not None:
            return found

    if required:
        raise ValueError(
            f"Could not find any of these columns: {possible_names}\n"
            f"Available columns: {list(df.columns)}"
        )
    return None


def clean_text(series):
    """Match the notebook's URL-removal step and return text strings."""
    return (
        series.fillna("")
        .astype(str)
        .map(lambda value: URL_PATTERN.sub(" ", value))
        .str.strip()
    )


def prepare_model_input(df):
    """
    Build exactly the feature columns used by the notebook's pipeline:
    title_text, content_text, published_day, published_day_num,
    is_weekend, published_hour, published_month, author_missing,
    author_model.
    """
    title_col = find_column(df, ["title", "title_clean", "headline"])
    content_col = find_column(
        df, ["content", "content_clean", "text", "article_content", "description"]
    )
    date_col = find_column(
        df, ["date", "published_at_utc", "published_at", "publish_date", "published"]
    )
    author_col = find_column(df, ["author", "author_clean", "byline"], required=False)

    features = pd.DataFrame(index=df.index)

    features["title_text"] = clean_text(df[title_col])
    features["content_text"] = clean_text(df[content_col])

    # The notebook derives these fields from the publication date.
    dates = pd.to_datetime(df[date_col], errors="coerce", utc=True)

    features["published_day"] = dates.dt.day_name()
    features["published_day_num"] = dates.dt.dayofweek
    features["is_weekend"] = dates.dt.dayofweek.isin([5, 6]).astype("float")
    features.loc[dates.isna(), "is_weekend"] = np.nan
    features["published_hour"] = dates.dt.hour
    features["published_month"] = dates.dt.month

    if author_col is None:
        author = pd.Series(pd.NA, index=df.index, dtype="string")
    else:
        author = df[author_col].astype("string").str.strip()

    author = author.mask(author.str.len().fillna(0) == 0)
    features["author_missing"] = author.isna().astype(int)
    features["author_model"] = author.fillna(UNKNOWN_AUTHOR).astype(str)

    # Rows without usable title/content cannot provide meaningful text predictions.
    no_text = (
        ~features["title_text"].str.contains(r"\S", regex=True)
        | ~features["content_text"].str.contains(r"\S", regex=True)
    )
    return features, no_text


def read_input(path):
    suffix = path.suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError("Input file must be .xlsx, .xls, or .csv")


def main():
    parser = argparse.ArgumentParser(
        description="Add predicted news categories using an existing trained model."
    )
    parser.add_argument(
        "--input",
        default=r"S:\Project_task\label\news_final_clean.csv",
        help="Input Excel or CSV file (default: S:\\Project_task\\label\\news_final_clean.csv)",
    )
    parser.add_argument(
        "--model",
        default="news_category_logreg_pipeline.pkl",
        help="Path to saved .pkl model (default: same folder/current directory)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path. Default: input filename with _categorized suffix",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    model_path = Path(args.model)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Put news_category_logreg_pipeline.pkl beside this script, "
            "or pass its full path with --model."
        )

    print("Loading input file...")
    data = read_input(input_path)
    print(f"Rows loaded: {len(data):,}")

    print("Loading already-trained model...")
    model = joblib.load(model_path)

    model_features, no_text = prepare_model_input(data)
    valid_mask = ~no_text

    # Keep every original row. Rows with no title/content receive a clear marker.
    data["category"] = "Not predicted - missing title/content"

    if valid_mask.any():
        print(f"Predicting categories for {int(valid_mask.sum()):,} rows...")
        predictions = model.predict(model_features.loc[valid_mask])
        data.loc[valid_mask, "category"] = predictions

        # Optional confidence score, if the saved pipeline supports predict_proba.
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(model_features.loc[valid_mask])
            data["category_confidence"] = np.nan
            data.loc[valid_mask, "category_confidence"] = probabilities.max(axis=1)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(
            f"{input_path.stem}_categorized.xlsx"
        )

    if output_path.suffix.lower() == ".csv":
        data.to_csv(output_path, index=False, encoding="utf-8-sig")
    else:
        data.to_excel(output_path, index=False)

    print("\nDone!")
    print(f"Total rows: {len(data):,}")
    print(f"Predicted rows: {int(valid_mask.sum()):,}")
    print(f"Skipped rows (missing title/content): {int(no_text.sum()):,}")
    print("\nPredicted category counts:")
    print(data["category"].value_counts(dropna=False).to_string())
    print(f"\nOutput saved to: {output_path.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
