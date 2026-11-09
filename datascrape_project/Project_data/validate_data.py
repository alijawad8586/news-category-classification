import sys
import csv
import re
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from urllib.parse import urlparse

# ✅ CRITICAL FIX: Large article content handle karne ke liye
csv.field_size_limit(sys.maxsize)

# ============================== DATA CONTRACT ==============================
CONTRACT = {
    "required_columns": ["news_url_clean", "title_clean", "content_clean"],
    "optional_columns": ["published_at_utc", "author_clean", "domain"],
    "min_title_chars_warning": 8,
    "min_content_chars_warning": 200,
    "known_error_patterns": [
        r"access denied", r"enable javascript", r"captcha",
        r"robot check", r"403 forbidden", r"page not found"
    ],
}
FATAL_RULES = {"FIELD_001", "URL_001"}

# ============================== HELPERS ==============================
def is_missing(value):
    if pd.isna(value): return True
    return isinstance(value, str) and not value.strip()

def is_valid_http_url(value):
    if not isinstance(value, str) or not value.strip(): return False
    try:
        parsed = urlparse(value.strip())
        return parsed.scheme.lower() in {"http", "https"} and bool(parsed.hostname) and "." in parsed.hostname
    except Exception: return False

def add_issue(issues, rule_id, severity, row_index, field, message):
    issues.append({"rule_id": rule_id, "severity": severity, "scope": "row",
                   "row_index": int(row_index), "field": field, "message": message})

# ============================== MAIN PIPELINE ==============================
def main(input_file, output_dir):
    print(f"🔍 Validating data from {input_file}...")
    
    # ✅ FIXED: low_memory=False hata diya hai kyunke python engine usay support nahi karta
    try:
        print(f"📖 Reading CSV file... (this may take 1-3 minutes for large files)")
        data = pd.read_csv(
            input_file, 
            encoding="utf-8-sig",
            engine='python',           # ✅ Python engine stable hai bari files ke liye
            on_bad_lines='skip'        # ✅ Kharab lines skip kare
        )
        print(f"✅ Loaded {len(data)} rows successfully!")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return

    df = data.copy(deep=True).reset_index(drop=False).rename(columns={"index": "source_row_index"})
    issues = []

    # 1. Schema Validation
    for col in CONTRACT["required_columns"]:
        if col not in df.columns:
            add_issue(issues, "SCHEMA_002", "ERROR", -1, col, f"Missing required column: {col}")

    # 2. Row-level Validation (Optimized for speed)
    print(f"⚙️  Running validation rules on {len(df)} rows...")
    
    # Vectorized checks (bohat tez hain)
    df['_url_valid'] = df.get("news_url_clean", pd.Series([pd.NA]*len(df))).apply(
        lambda x: is_valid_http_url(x) if not is_missing(x) else True
    )
    df['_content_len'] = df.get("content_clean", pd.Series(['']*len(df))).apply(
        lambda x: len(x.strip()) if isinstance(x, str) else 0
    )
    
    error_regex = re.compile("|".join(CONTRACT["known_error_patterns"]), flags=re.I)
    df['_has_error_pattern'] = df.get("content_clean", pd.Series(['']*len(df))).apply(
        lambda x: bool(error_regex.search(x)) if isinstance(x, str) else False
    )
    
    # Duplicate URL check (vectorized)
    if "news_url_clean" in df.columns:
        url_counts = df["news_url_clean"].value_counts()
        df['_is_dup_url'] = df["news_url_clean"].map(lambda x: url_counts.get(x, 0) > 1 if not is_missing(x) else False)
    
    # Ab row-wise issues collect karein
    for idx, row in df.iterrows():
        source_idx = row["source_row_index"]
        
        # Required Fields
        for field in CONTRACT["required_columns"]:
            if field in df.columns and is_missing(row[field]):
                add_issue(issues, "FIELD_001", "ERROR", source_idx, field, f"Required field '{field}' is missing.")

        # URL Validation
        url = row.get("news_url_clean", pd.NA)
        if not is_missing(url) and not row.get('_url_valid', True):
            add_issue(issues, "URL_001", "ERROR", source_idx, "news_url_clean", "Invalid HTTP(S) URL.")

        # Content Quality
        if row.get('_content_len', 0) > 0:
            if row['_content_len'] < CONTRACT["min_content_chars_warning"]:
                add_issue(issues, "TEXT_002", "WARNING", source_idx, "content_clean", "Content is too short.")
            if row.get('_has_error_pattern', False):
                add_issue(issues, "TEXT_003", "WARNING", source_idx, "content_clean", "Possible CAPTCHA/Block/Error page.")

        # Duplicates
        if row.get('_is_dup_url', False):
            add_issue(issues, "DUP_001", "WARNING", source_idx, "news_url_clean", "Duplicate URL detected.")
    
    # Cleanup temp columns
    df = df.drop(columns=[c for c in df.columns if c.startswith('_')], errors='ignore')

    issues_df = pd.DataFrame(issues, columns=["rule_id", "severity", "scope", "row_index", "field", "message"])

    # 3. Assign Dispositions
    row_issue_map = {}
    for _, issue in issues_df.iterrows():
        if pd.notna(issue["row_index"]) and issue["row_index"] >= 0:
            row_issue_map.setdefault(int(issue["row_index"]), []).append(issue["rule_id"])

    def disposition(row_id):
        rules = row_issue_map.get(int(row_id), [])
        if any(rule in FATAL_RULES for rule in rules): return "QUARANTINE"
        if rules: return "WARNING"
        return "ACCEPT"

    df["validation_disposition"] = df["source_row_index"].map(disposition)
    df["validation_issue_codes"] = df["source_row_index"].map(lambda x: ";".join(row_issue_map.get(int(x), [])))

    accepted_df = df[df["validation_disposition"] == "ACCEPT"].copy()
    warning_df = df[df["validation_disposition"] == "WARNING"].copy()
    quarantine_df = df[df["validation_disposition"] == "QUARANTINE"].copy()

    # 4. Export Artifacts
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"💾 Saving validation outputs...")
    df.to_csv(out_path / "news_validation_all.csv", index=False, encoding="utf-8-sig")
    accepted_df.to_csv(out_path / "news_accepted.csv", index=False, encoding="utf-8-sig")
    warning_df.to_csv(out_path / "news_warning_review.csv", index=False, encoding="utf-8-sig")
    quarantine_df.to_csv(out_path / "news_quarantine.csv", index=False, encoding="utf-8-sig")
    issues_df.to_csv(out_path / "news_validation_issues.csv", index=False, encoding="utf-8-sig")

    summary = {
        "total": len(df), "accepted": len(accepted_df), 
        "warning": len(warning_df), "quarantine": len(quarantine_df)
    }
    with open(out_path / "validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ VALIDATION COMPLETE!")
    print(f"🟢 Accepted: {len(accepted_df)} | 🟡 Warning: {len(warning_df)} | 🔴 Quarantine: {len(quarantine_df)}")
    print(f"📂 Files saved to: {out_path.absolute()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate scraped news data against a contract.")
    parser.add_argument("--input", default=r"I:\Amazon\Playwright\preprocessed_output\news_clean.csv", help="Input CSV file")
    parser.add_argument("--output", default=r"I:\Amazon\Playwright\validation_output", help="Output directory")
    args = parser.parse_args()
    main(args.input, args.output)