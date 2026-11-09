import pandas as pd
import numpy as np
import re
import sys
import csv
from pathlib import Path

# ✅ Bade articles ke liye field size limit
csv.field_size_limit(sys.maxsize)

# ============================== CONFIG ==============================
INPUT_PATH  = Path(r"S:\Project_task\datascrape_project\preprocessed_output\news_clean.csv")
OUTPUT_PATH = Path(r"S:\Project_task\datascrape_project\preprocessed_output\news_enriched.csv")

# ============================== HELPER FUNCTIONS ==============================

def extract_date_from_url(url):
    """URL se date nikalta hai (YYYY-MM-DD ya YYYY/MM/DD format)"""
    if pd.isna(url) or not isinstance(url, str):
        return pd.NaT
    
    # Pattern 1: YYYY/MM/DD ya YYYY-MM-DD
    match = re.search(r'(\d{4})[-/](\d{2})[-/](\d{2})', url)
    if match:
        try:
            return pd.to_datetime(f"{match.group(1)}-{match.group(2)}-{match.group(3)}", utc=True)
        except Exception:
            return pd.NaT
    
    # Pattern 2: Sirf YYYY/MM (archive pages ke liye)
    match = re.search(r'/(\d{4})/(\d{2})(?:/)?', url)
    if match:
        try:
            # Mahine ka pehla din use karein
            return pd.to_datetime(f"{match.group(1)}-{match.group(2)}-01", utc=True)
        except Exception:
            return pd.NaT
    
    return pd.NaT


def extract_time_from_datetime(dt):
    """Datetime se sirf time (HH:MM:SS) nikalta hai"""
    if pd.isna(dt):
        return pd.NaT
    try:
        return pd.to_datetime(dt).strftime("%H:%M:%S")
    except Exception:
        return pd.NaT


def extract_author_from_content(content):
    """Content se author nikalne ki koshish karta hai (agar mojood ho)"""
    if pd.isna(content) or not isinstance(content, str):
        return "Unknown"
    
    # Common author patterns (English news sites)
    patterns = [
        r'(?:By|Written by|Author:|Posted by)\s+([A-Z][a-zA-Z\.\s]+?)(?:\s*[,|]|\s*for\s+|\s*\d{4})',
        r'(?:By|Written by)\s+([A-Z][a-zA-Z\.\s]+?)(?:\s*\n|\s*$)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content[:500])  # Sirf pehle 500 chars check karein
        if match:
            author = match.group(1).strip()
            # Agar author bohat lamba ya ajeeb hai toh ignore karein
            if 3 <= len(author) <= 50 and not any(c.isdigit() for c in author):
                return author
    
    return "Unknown"


# ============================== MAIN PIPELINE ==============================

def main():
    print(f"🔄 Loading data from {INPUT_PATH}...")
    if not INPUT_PATH.exists():
        print(f"❌ File not found: {INPUT_PATH}")
        return
    
    df = pd.read_csv(INPUT_PATH, low_memory=False, encoding="utf-8-sig")
    print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns.\n")
    
    # Original missing dates count
    missing_before = df['published_at_utc'].isna().sum()
    print(f"📅 Missing publish_date before processing: {missing_before:,} / {len(df):,}")
    
    # ---------------------------------------------------------
    # STEP 1: URL se date extract karna (Tareeqa 1)
    # ---------------------------------------------------------
    print("\n🔍 Step 1: Extracting dates from URLs...")
    
    # Sirf un rows par apply karein jahan date missing hai
    missing_mask = df['published_at_utc'].isna()
    
    if missing_mask.any():
        url_dates = df.loc[missing_mask, 'article_url'].apply(extract_date_from_url)
        df.loc[missing_mask, 'published_at_utc'] = url_dates
        
        filled_from_url = url_dates.notna().sum()
        print(f"   ✔ {filled_from_url:,} dates URL se extract hui")
    
    # ---------------------------------------------------------
    # STEP 2: Forward/Backward Fill (Tareeqa 2)
    # ---------------------------------------------------------
    print("\n🔍 Step 2: Applying Forward/Backward Fill...")
    
    # Pehle datetime mein convert karein (agar string ho)
    df['published_at_utc'] = pd.to_datetime(df['published_at_utc'], errors='coerce', utc=True)
    
    # Sort by URL taake same domain ke articles saath hon (better fill)
    df = df.sort_values(by=['domain', 'published_at_utc'], na_position='last')
    
    missing_before_fill = df['published_at_utc'].isna().sum()
    
    # Forward fill (pichli row ki date)
    df['published_at_utc'] = df['published_at_utc'].ffill()
    filled_ffill = missing_before_fill - df['published_at_utc'].isna().sum()
    
    # Backward fill (agli row ki date) - agar abhi bhi missing hai
    missing_before_bfill = df['published_at_utc'].isna().sum()
    df['published_at_utc'] = df['published_at_utc'].bfill()
    filled_bfill = missing_before_bfill - df['published_at_utc'].isna().sum()
    
    print(f"   ✔ Forward fill se {filled_ffill:,} dates fill hui")
    print(f"   ✔ Backward fill se {filled_bfill:,} dates fill hui")
    
    # ---------------------------------------------------------
    # STEP 3: Time column add karna
    # ---------------------------------------------------------
    print("\n🕐 Step 3: Extracting time from published_at_utc...")
    df['publish_time'] = df['published_at_utc'].apply(extract_time_from_datetime)
    print(f"   ✔ Time column add ho gaya")
    
    # ---------------------------------------------------------
    # STEP 4: Author column add karna
    # ---------------------------------------------------------
    print("\n✍️  Step 4: Extracting author from content...")
    df['author'] = df['content_clean'].apply(extract_author_from_content)
    author_found = (df['author'] != 'Unknown').sum()
    print(f"   ✔ {author_found:,} authors content se extract hue")
    print(f"   ⚠ {len(df) - author_found:,} rows mein author 'Unknown' mark kiya gaya")
    
    # ---------------------------------------------------------
    # STEP 5: Original order mein wapis sort karna
    # ---------------------------------------------------------
    if 'source_row_index' in df.columns:
        df = df.sort_values('source_row_index')
    
    # ---------------------------------------------------------
    # STEP 6: Final summary
    # ---------------------------------------------------------
    missing_after = df['published_at_utc'].isna().sum()
    print("\n" + "="*60)
    print("📊 FINAL SUMMARY")
    print("="*60)
    print(f"Total rows              : {len(df):,}")
    print(f"Missing dates (before)  : {missing_before:,}")
    print(f"Missing dates (after)   : {missing_after:,}")
    print(f"Dates recovered         : {missing_before - missing_after:,}")
    print(f"Recovery rate           : {((missing_before - missing_after) / max(missing_before, 1)) * 100:.2f}%")
    
    # ---------------------------------------------------------
    # STEP 7: Save to new CSV
    # ---------------------------------------------------------
    print(f"\n💾 Saving enriched data to: {OUTPUT_PATH}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Null bytes remove karein (Windows safety)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].map(lambda x: x.replace("\x00", "") if isinstance(x, str) else x)
    
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    
    file_size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"✅ Done! File size: {file_size_mb:.2f} MB")
    print(f"📂 New file: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()