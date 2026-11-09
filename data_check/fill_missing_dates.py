import pandas as pd
import numpy as np
import re
import csv
import sys
from pathlib import Path
from datetime import datetime

csv.field_size_limit(sys.maxsize)

# ============================== CONFIG ==============================
INPUT_PATH = Path(r"S:\Project_task\datascrape_project\preprocessed_output\news_enriched.csv")
OUTPUT_PATH = Path(r"S:\Project_task\datascrape_project\preprocessed_output\news_final_clean.csv")

# Date range validation
DATE_MIN = pd.Timestamp("1995-01-01", tz="UTC")
DATE_MAX = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=2)

MISSING_TOKENS = {"", "n/a", "na", "null", "none", "nan", "-", "--", "unknown", "not available", "no date"}

# Common date formats
DATE_FORMATS = [
    "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%d %H:%M:%S", "%d/%m/%Y", "%m/%d/%Y",
    "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y",
    "%Y/%m/%d", "%Y-%m", "%B %Y", "%b %Y",
    "%d %B %Y", "%B %d %Y", "%d %b %Y", "%b %d %Y",
    "%Y%m%d", "%d-%m-%Y", "%m-%d-%Y",
]


# ============================== HELPER FUNCTIONS ==============================

def normalize_missing(value):
    """Missing/invalid values ko pd.NA mein convert karta hai"""
    if pd.isna(value):
        return pd.NA
    if isinstance(value, str):
        value = value.strip()
        if value.casefold() in MISSING_TOKENS:
            return pd.NA
    return value


def parse_date_safe(value):
    """Date string ko safely parse karta hai - multiple formats try karta hai"""
    if pd.isna(value):
        return pd.NaT
    
    value = str(value).strip()
    if value.casefold() in MISSING_TOKENS:
        return pd.NaT
    
    # Try specific formats first
    for fmt in DATE_FORMATS:
        try:
            dt = datetime.strptime(value, fmt)
            return pd.Timestamp(dt, tz="UTC")
        except (ValueError, TypeError):
            continue
    
    # Pandas auto-parse as last resort
    try:
        parsed = pd.to_datetime(value, errors="coerce", utc=True, format="mixed")
        if pd.notna(parsed):
            return parsed
    except Exception:
        pass
    
    try:
        parsed = pd.to_datetime(value, errors="coerce", utc=True)
        if pd.notna(parsed):
            return parsed
    except Exception:
        pass
    
    return pd.NaT


def extract_date_from_url(url):
    """URL se date extract karta hai using multiple regex patterns"""
    if pd.isna(url) or not isinstance(url, str):
        return pd.NaT
    
    url = str(url)
    
    # Pattern 1: Full date YYYY/MM/DD or YYYY-MM-DD
    match = re.search(r"/(19|20)(\d{2})[/\-](\d{1,2})[/\-](\d{1,2})(?:/|$|\?|#)", url)
    if match:
        year = int(match.group(1) + match.group(2))
        month = int(match.group(3))
        day = int(match.group(4))
        try:
            ts = pd.Timestamp(year, month, day, tz="UTC")
            if DATE_MIN <= ts <= DATE_MAX:
                return ts
        except ValueError:
            pass
    
    # Pattern 2: Year/Month only (e.g., /archive/2019/09 or /2021/08/)
    match = re.search(r"/(19|20)(\d{2})[/\-](\d{1,2})(?:/|$|\?|#)", url)
    if match:
        year = int(match.group(1) + match.group(2))
        month = int(match.group(3))
        if 1 <= month <= 12:
            try:
                ts = pd.Timestamp(year, month, 1, tz="UTC")
                if DATE_MIN <= ts <= DATE_MAX:
                    return ts
            except ValueError:
                pass
    
    # Pattern 3: Year only (e.g., /2019/ or /news/2021)
    match = re.search(r"/(19|20)(\d{2})(?:/|$|\?|#)", url)
    if match:
        year = int(match.group(1) + match.group(2))
        try:
            ts = pd.Timestamp(year, 1, 1, tz="UTC")
            if DATE_MIN <= ts <= DATE_MAX:
                return ts
        except ValueError:
            pass
    
    # Pattern 4: Date in filename (e.g., article-2021-08-15.html)
    match = re.search(r"[-_](19|20)(\d{2})[-_](\d{1,2})[-_](\d{1,2})", url)
    if match:
        year = int(match.group(1) + match.group(2))
        month = int(match.group(3))
        day = int(match.group(4))
        try:
            ts = pd.Timestamp(year, month, day, tz="UTC")
            if DATE_MIN <= ts <= DATE_MAX:
                return ts
        except ValueError:
            pass
    
    return pd.NaT


def extract_date_from_content(content):
    """Content ke pehle 500 characters se date extract karta hai"""
    if pd.isna(content) or not isinstance(content, str):
        return pd.NaT
    
    # Sirf pehle 500 chars check karein (dates usually shuru mein hoti hain)
    text = content[:500]
    
    # Pattern: "Published on August 15, 2021" or "August 15, 2021"
    patterns = [
        r"(?:published|posted|updated|dated)\s+(?:on\s+)?(\w+\s+\d{1,2},?\s+\d{4})",
        r"(\d{1,2}\s+\w+\s+\d{4})",
        r"(\w+\s+\d{1,2},?\s+\d{4})",
        r"(\d{4}[-/]\d{1,2}[-/]\d{1,2})",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            date_str = match.group(1)
            parsed = parse_date_safe(date_str)
            if pd.notna(parsed) and DATE_MIN <= parsed <= DATE_MAX:
                return parsed
    
    return pd.NaT


def validate_date_range(series):
    """Date series ko valid range mein filter karta hai"""
    if series.dtype != "datetime64[ns, UTC]":
        try:
            series = pd.to_datetime(series, errors="coerce", utc=True)
        except Exception:
            return pd.Series(pd.NaT, index=series.index)
    
    valid = series.notna() & series.between(DATE_MIN, DATE_MAX)
    return series.where(valid)


# ============================== MAIN PIPELINE ==============================

def main():
    print("=" * 70)
    print("🔄 COMPLETE DATE FILLING & PREPROCESSING PIPELINE")
    print("=" * 70)

    # ──────────────────────────────────────────────
    # STEP 1: Load Data
    # ──────────────────────────────────────────────
    print(f"\n📂 Loading data from: {INPUT_PATH}")
    if not INPUT_PATH.exists():
        print(f"❌ File not found: {INPUT_PATH}")
        return

    df = pd.read_csv(INPUT_PATH, low_memory=False, encoding="utf-8-sig")
    print(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
    print(f"📋 Columns: {list(df.columns)}")

    # ──────────────────────────────────────────────
    # STEP 2: Standardize Column Names
    # ──────────────────────────────────────────────
    print("\n🔧 Standardizing column names...")
    
    # "publish date" (space) ko "publish_date" (underscore) mein convert
    rename_map = {}
    for col in df.columns:
        standardized = re.sub(r"[^a-z0-9]+", "_", col.strip().lower()).strip("_")
        if standardized != col:
            rename_map[col] = standardized
    
    if rename_map:
        df = df.rename(columns=rename_map)
        print(f"   Renamed: {rename_map}")
    else:
        print("   No renaming needed")

    # ──────────────────────────────────────────────
    # STEP 3: Identify Date Columns & Set Target
    # ──────────────────────────────────────────────
    print("\n📅 Identifying date columns...")
    
    date_keywords = ["date", "publish", "time", "pub_date", "published"]
    date_cols = [c for c in df.columns if any(k in c.lower() for k in date_keywords)]
    print(f"   Found date columns: {date_cols}")

    # Primary target column
    target_col = None
    for candidate in ["published_at_utc", "publish_date", "published_date", "date"]:
        if candidate in df.columns:
            target_col = candidate
            break
    
    if target_col is None:
        target_col = "published_at_utc"
        df[target_col] = pd.NaT
        print(f"   ⚠️ No date column found, creating '{target_col}'")
    else:
        print(f"   Target column: '{target_col}'")

    # Ensure target is datetime
    if df[target_col].dtype == object:
        df[target_col] = df[target_col].apply(parse_date_safe)
    else:
        df[target_col] = pd.to_datetime(df[target_col], errors="coerce", utc=True)
    
    df[target_col] = validate_date_range(df[target_col])

    # Create date_source tracking column
    df["date_source"] = df[target_col].notna().map({True: "original", False: None})

    # Show initial missing count
    missing_before = df[target_col].isna().sum()
    print(f"\n❌ Missing dates BEFORE: {missing_before:,} / {len(df):,} ({missing_before/len(df)*100:.2f}%)")

    # ──────────────────────────────────────────────
    # STRATEGY 1: Coalesce from Other Date Columns
    # ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("📅 STRATEGY 1: Coalesce from Other Date Columns")
    print("=" * 70)

    source_cols = [c for c in date_cols if c != target_col and c != "published_date_only" and c != "date_source"]
    print(f"   Source columns: {source_cols}")

    for col in source_cols:
        if col not in df.columns:
            continue
        
        # Parse the column
        if df[col].dtype == object:
            parsed = df[col].apply(normalize_missing).apply(parse_date_safe)
        else:
            parsed = pd.to_datetime(df[col], errors="coerce", utc=True)
        
        # Validate range
        parsed = validate_date_range(parsed)

        # Fill only where target is missing
        mask = df[target_col].isna() & parsed.notna()
        filled_count = mask.sum()
        
        if filled_count > 0:
            df.loc[mask, target_col] = parsed[mask]
            df.loc[mask, "date_source"] = f"coalesce_{col}"
            print(f"   ✔ From '{col}': filled {filled_count:,} rows")
        else:
            print(f"   - From '{col}': filled 0 rows")

    missing_after_s1 = df[target_col].isna().sum()
    print(f"\n   📊 After Strategy 1: {missing_after_s1:,} still missing (filled {missing_before - missing_after_s1:,})")

    # ──────────────────────────────────────────────
    # STRATEGY 2: Extract Date from URL
    # ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("🔗 STRATEGY 2: Extract Date from URL")
    print("=" * 70)

    url_col = None
    for candidate in ["url_clean", "article_url", "news_url_clean", "news_url", "url"]:
        if candidate in df.columns:
            url_col = candidate
            break

    if url_col:
        print(f"   Using URL column: '{url_col}'")
        
        mask = df[target_col].isna()
        missing_count = mask.sum()
        print(f"   Rows to process: {missing_count:,}")
        
        if missing_count > 0:
            url_dates = df.loc[mask, url_col].apply(extract_date_from_url)
            valid_url_dates = url_dates[url_dates.notna()]
            
            filled_count = len(valid_url_dates)
            if filled_count > 0:
                df.loc[valid_url_dates.index, target_col] = valid_url_dates
                df.loc[valid_url_dates.index, "date_source"] = "url_extract"
                print(f"   ✔ Extracted from URL: filled {filled_count:,} rows")
            else:
                print(f"   - Extracted from URL: filled 0 rows")
    else:
        print("   ⚠️ No URL column found!")

    missing_after_s2 = df[target_col].isna().sum()
    print(f"\n   📊 After Strategy 2: {missing_after_s2:,} still missing (filled {missing_after_s1 - missing_after_s2:,})")

    # ──────────────────────────────────────────────
    # STRATEGY 3: Extract Date from Content
    # ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("📝 STRATEGY 3: Extract Date from Content")
    print("=" * 70)

    content_col = None
    for candidate in ["content_clean", "article_content", "content", "article_content_raw"]:
        if candidate in df.columns:
            content_col = candidate
            break

    if content_col:
        print(f"   Using content column: '{content_col}'")
        
        mask = df[target_col].isna()
        missing_count = mask.sum()
        print(f"   Rows to process: {missing_count:,}")
        
        if missing_count > 0:
            content_dates = df.loc[mask, content_col].apply(extract_date_from_content)
            valid_content_dates = content_dates[content_dates.notna()]
            
            filled_count = len(valid_content_dates)
            if filled_count > 0:
                df.loc[valid_content_dates.index, target_col] = valid_content_dates
                df.loc[valid_content_dates.index, "date_source"] = "content_extract"
                print(f"   ✔ Extracted from content: filled {filled_count:,} rows")
            else:
                print(f"   - Extracted from content: filled 0 rows")
    else:
        print("   ⚠️ No content column found!")

    missing_after_s3 = df[target_col].isna().sum()
    print(f"\n   📊 After Strategy 3: {missing_after_s3:,} still missing (filled {missing_after_s2 - missing_after_s3:,})")

    # ──────────────────────────────────────────────
    # STRATEGY 4: Forward/Backward Fill (by Domain)
    # ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("⏩ STRATEGY 4: Forward/Backward Fill (by Domain)")
    print("=" * 70)

    domain_col = "domain" if "domain" in df.columns else None

    if domain_col and df[target_col].isna().sum() > 0:
        # Original index preserve karein
        original_index = df.index.copy()
        
        # Sort by domain and date for better fill accuracy
        df = df.sort_values([domain_col, target_col], na_position="last")

        # Forward fill within each domain
        mask_before = df[target_col].isna()
        df[target_col] = df.groupby(domain_col, group_keys=False)[target_col].ffill()
        filled_ffill = mask_before.sum() - df[target_col].isna().sum()
        
        # Mark newly filled rows
        newly_filled = mask_before & df[target_col].notna()
        df.loc[newly_filled, "date_source"] = "domain_ffill"
        print(f"   ✔ Forward fill: filled {filled_ffill:,} rows")

        # Backward fill within each domain (for remaining)
        mask_before = df[target_col].isna()
        df[target_col] = df.groupby(domain_col, group_keys=False)[target_col].bfill()
        filled_bfill = mask_before.sum() - df[target_col].isna().sum()
        
        newly_filled = mask_before & df[target_col].notna()
        df.loc[newly_filled, "date_source"] = "domain_bfill"
        print(f"   ✔ Backward fill: filled {filled_bfill:,} rows")

        # Restore original order
        df = df.loc[original_index]
    elif df[target_col].isna().sum() > 0:
        print("   ⚠️ No domain column found, applying global ffill/bfill")
        
        mask_before = df[target_col].isna()
        df[target_col] = df[target_col].ffill()
        filled_ffill = mask_before.sum() - df[target_col].isna().sum()
        newly_filled = mask_before & df[target_col].notna()
        df.loc[newly_filled, "date_source"] = "global_ffill"
        print(f"   ✔ Forward fill: filled {filled_ffill:,} rows")

        mask_before = df[target_col].isna().sum()
        df[target_col] = df[target_col].bfill()
        filled_bfill = mask_before - df[target_col].isna().sum()
        newly_filled = mask_before & df[target_col].notna()
        df.loc[newly_filled, "date_source"] = "global_bfill"
        print(f"   ✔ Backward fill: filled {filled_bfill:,} rows")
    else:
        print("   ✅ No missing dates remaining, skipping fill")

    missing_after_s4 = df[target_col].isna().sum()
    print(f"\n   📊 After Strategy 4: {missing_after_s4:,} still missing")

    # ──────────────────────────────────────────────
    # STRATEGY 5: Fill Remaining with Median Date
    # ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("📊 STRATEGY 5: Fill Remaining with Median Date")
    print("=" * 70)

    remaining_missing = df[target_col].isna().sum()
    if remaining_missing > 0:
        # Calculate median date from valid dates
        valid_dates = df[target_col].dropna()
        if len(valid_dates) > 0:
            median_date = valid_dates.median()
            df.loc[df[target_col].isna(), target_col] = median_date
            df.loc[df["date_source"].isna(), "date_source"] = "median_fill"
            print(f"   ✔ Filled {remaining_missing:,} rows with median date: {median_date}")
        else:
            print("   ⚠️ No valid dates available for median calculation")
    else:
        print("   ✅ No missing dates remaining")

    missing_final = df[target_col].isna().sum()

    # ──────────────────────────────────────────────
    # FINAL: Update Related Columns & Save
    # ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("💾 FINAL RESULTS & SAVE")
    print("=" * 70)

    # Update published_date_only column
    if "published_date_only" in df.columns:
        df["published_date_only"] = pd.to_datetime(df[target_col], errors="coerce").dt.date
    else:
        df["published_date_only"] = pd.to_datetime(df[target_col], errors="coerce").dt.date

    # Update publish_date column
    if "publish_date" in df.columns and target_col != "publish_date":
        df["publish_date"] = pd.to_datetime(df[target_col], errors="coerce").dt.strftime("%Y-%m-%d")

    # Ensure datetime format is consistent
    df[target_col] = pd.to_datetime(df[target_col], errors="coerce", utc=True)

    # Final stats
    total_filled = missing_before - missing_final

    print(f"\n📊 SUMMARY:")
    print(f"   Total rows:                {len(df):,}")
    print(f"   Missing BEFORE:            {missing_before:,} ({missing_before/len(df)*100:.2f}%)")
    print(f"   Missing AFTER:             {missing_final:,} ({missing_final/len(df)*100:.2f}%)")
    print(f"   Total FILLED:              {total_filled:,} ({total_filled/max(missing_before,1)*100:.2f}%)")
    print(f"   Strategy 1 (Coalesce):     {missing_before - missing_after_s1:,}")
    print(f"   Strategy 2 (URL):          {missing_after_s1 - missing_after_s2:,}")
    print(f"   Strategy 3 (Content):      {missing_after_s2 - missing_after_s3:,}")
    print(f"   Strategy 4 (FFill/BFill):  {missing_after_s3 - missing_after_s4:,}")
    print(f"   Strategy 5 (Median):       {missing_after_s4 - missing_final:,}")

    if df[target_col].notna().any():
        print(f"\n   📆 Date range: {df[target_col].min()} → {df[target_col].max()}")

    # Date source distribution
    print(f"\n   📋 Date Source Distribution:")
    source_dist = df["date_source"].value_counts()
    for source, count in source_dist.items():
        pct = count / len(df) * 100
        print(f"      {source}: {count:,} ({pct:.1f}%)")

    # ──────────────────────────────────────────────
    # REMOVE ARCHIVE PAGES (Optional Cleanup)
    # ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("🗑️ ARCHIVE PAGE DETECTION")
    print("=" * 70)

    if url_col:
        archive_mask = df[url_col].astype(str).str.contains(
            r"/archive|/archives|/category|/tag/|/page/\d+",
            case=False, na=False, regex=True
        )
        archive_count = archive_mask.sum()
        print(f"   Archive pages detected: {archive_count:,}")
        
        if archive_count > 0:
            df["is_archive_page"] = archive_mask
            # Archive pages ko alag file mein save karein
            df_archive = df[archive_mask].copy()
            df_clean = df[~archive_mask].copy()
            
            archive_path = OUTPUT_PATH.parent / "news_archive_pages.csv"
            df_archive.to_csv(archive_path, index=False, encoding="utf-8-sig")
            print(f"   📄 Archive pages saved to: {archive_path}")
            
            df = df_clean
            print(f"   ✅ Clean data rows: {len(df):,}")
    else:
        df["is_archive_page"] = False

    # ──────────────────────────────────────────────
    # FINAL SAVE
    # ──────────────────────────────────────────────
    print(f"\n💾 Saving final clean data to: {OUTPUT_PATH}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove null bytes (Windows Errno 22 fix)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].map(lambda x: x.replace("\x00", "") if isinstance(x, str) else x)
    
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    file_size = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"✅ Done! File size: {file_size:.2f} MB")
    print(f"📊 Final rows: {len(df):,}")
    print(f"📊 Final columns: {len(df.columns)}")
    print("=" * 70)


if __name__ == "__main__":
    main()