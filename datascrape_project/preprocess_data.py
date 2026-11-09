import sys
import csv
import re
import html
import json
import unicodedata
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
from html.parser import HTMLParser

# ✅ FIX: Prevent "field larger than field limit" error for long articles
csv.field_size_limit(sys.maxsize)

# ============================== CONFIG ==============================
ALIASES = {
    "news_url": ["url", "article_url", "news_url", "link", "source_url"],
    "title": ["headline", "article_title", "news_title", "title"],
    "published_time": ["publish_date", "published_date", "time", "date", "datetime", "timestamp"],
    "author": ["writer", "byline", "article_author", "authors", "author"],
    "content": ["text", "article_content", "body", "article_text", "full_text", "content"],
}

TRACKING_PARAMS = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "fbclid", "gclid"}
MISSING_TOKENS = {"", "n/a", "na", "null", "none", "nan", "-", "--", "unknown"}
INVISIBLE_CHARS = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad"), None)
MIN_TITLE_CHARS = 8
MIN_CONTENT_CHARS = 200

# ============================== HELPERS ==============================
def standardize_column_name(name):
    name = unicodedata.normalize("NFKC", str(name)).strip().lower()
    name = re.sub(r"[^a-z0-9]+", " ", name)
    return name.strip(" ")

def is_present(series):
    series = series.astype(object)
    return series.notna() & series.map(lambda v: not (isinstance(v, str) and v.strip() == ""))

def coalesce_columns(frame, canonical, aliases):
    names = [canonical] + aliases
    positions = [i for i, c in enumerate(frame.columns) if c in names]
    if not positions:
        return frame, None
    positions.sort(key=lambda i: names.index(frame.columns[i]))
    block = frame.iloc[:, positions].astype(object)
    merged = block.iloc[:, 0]
    conflicts = 0
    for k in range(1, block.shape[1]):
        other = block.iloc[:, k]
        both = is_present(merged) & is_present(other)
        conflicts += int((both & (merged.astype(str) != other.astype(str))).sum())
        merged = merged.where(is_present(merged), other)
    rest = frame.iloc[:, [i for i in range(frame.shape[1]) if i not in positions]]
    out = pd.concat([pd.DataFrame({canonical: merged}), rest], axis=1)
    return out, {"merged_from": [frame.columns[i] for i in positions], "conflicts": conflicts}

def normalize_missing(value):
    if pd.isna(value): return pd.NA
    if isinstance(value, str):
        value = value.replace("\u00a0", " ").strip()
        if value.casefold() in MISSING_TOKENS: return pd.NA
    return value

def normalize_whitespace(text):
    if pd.isna(text): return pd.NA
    text = str(text).replace("\u00a0", " ")
    text = re.sub(r"[\t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()

class TextExtractor(HTMLParser):
    SKIP_TAGS = {"script", "style", "noscript"}
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self._skip_depth = 0
    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS: self._skip_depth += 1
    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS and self._skip_depth: self._skip_depth -= 1
    def handle_data(self, data):
        if self._skip_depth == 0 and data and data.strip(): self.parts.append(data.strip())

def strip_html(value):
    if pd.isna(value): return pd.NA
    parser = TextExtractor()
    try:
        parser.feed(str(value))
        parser.close()
        extracted = " ".join(parser.parts)
    except Exception:
        extracted = re.sub(r"<[^>]+>", " ", str(value))
    extracted = html.unescape(extracted)
    extracted = unicodedata.normalize("NFKC", extracted)
    extracted = re.sub(r"\s+", " ", extracted).strip()
    return extracted if extracted else pd.NA

def normalize_typography(value):
    if pd.isna(value): return pd.NA
    value = str(value).translate(INVISIBLE_CHARS)
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    value = re.sub(r"\s+", " ", value).strip()
    return value if value else pd.NA

def normalize_url(value, remove_tracking=False):
    if pd.isna(value): return pd.NA
    value = str(value).strip()
    if not value or not re.match(r"^https?://", value, flags=re.I) or re.search(r"\s", value): return pd.NA
    try:
        p = urlparse(value)
        if not p.hostname: return pd.NA
        scheme, hostname = p.scheme.lower(), p.hostname.lower()
        netloc = f"{hostname}:{p.port}" if p.port and not ((scheme=="http" and p.port==80) or (scheme=="https" and p.port==443)) else hostname
        path = re.sub(r"/{2,}", "/", p.path or "/")
        query = p.query
        if remove_tracking and query:
            pairs = [(k, v) for k, v in parse_qsl(query, keep_blank_values=True) if k.lower() not in TRACKING_PARAMS]
            query = urlencode(pairs, doseq=True)
        return urlunparse((scheme, netloc, path, "", query, ""))
    except Exception: return pd.NA

def parse_timestamps(series):
    try: return pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
    except Exception: return pd.to_datetime(series, errors="coerce", utc=True)

def safe_len(value): return len(value) if isinstance(value, str) else 0
def word_count(value): return len(re.findall(r"\b[\w'-]+\b", value, flags=re.UNICODE)) if isinstance(value, str) else 0

# ============================== MAIN PIPELINE ==============================
def main(input_file, output_dir):
    print(f"🔄 Loading data from {input_file}...")
    raw_df = pd.read_csv(input_file, encoding="utf-8-sig", low_memory=False)
    print(f"✅ Loaded {len(raw_df)} rows.")

    df = raw_df.copy(deep=True)
    df.columns = [standardize_column_name(c) for c in df.columns]

    # 1. Map Columns
    for canonical, aliases in ALIASES.items():
        df, _ = coalesce_columns(df, canonical, aliases)
    for col in ["news_url", "title", "published_time", "author", "content"]:
        if col not in df.columns: df[col] = pd.NA

    # 2. Preserve Raw & Normalize
    for col in ["news_url", "title", "published_time", "author", "content"]:
        df[f"{col}_raw"] = df[col]
        df[col] = df[col].map(normalize_missing)
    for col in ["title", "author", "content"]:
        df[col] = df[col].map(normalize_whitespace)

    # 3. Clean HTML & Typography
    df["title_clean"] = df["title"].map(strip_html).map(normalize_typography)
    df["author_clean"] = df["author"].map(strip_html).map(normalize_typography)
    df["content_clean"] = df["content"].map(strip_html).map(normalize_typography)

    # 4. URL Normalization
    df["news_url_clean"] = df["news_url"].map(lambda x: normalize_url(x, remove_tracking=True))
    df["url_valid"] = df["news_url_clean"].notna()
    df["domain"] = df["news_url_clean"].map(lambda x: urlparse(x).hostname if pd.notna(x) else pd.NA)

    # 5. Timestamp Parsing
    df["published_at_utc"] = parse_timestamps(df["published_time"])
    df["published_date_utc"] = df["published_at_utc"].dt.date

    # 6. Text Lengths & Error Patterns
    df["title_char_count"] = df["title_clean"].map(safe_len)
    df["content_char_count"] = df["content_clean"].map(safe_len)
    df["content_word_count"] = df["content_clean"].map(word_count)
    
    error_regex = re.compile(r"access denied|enable javascript|captcha|robot check|403 forbidden|page not found", flags=re.I)
    df["possible_block_or_paywall"] = df["content_clean"].map(lambda x: bool(error_regex.search(x)) if isinstance(x, str) else False)
    df["title_too_short"] = df["title_char_count"] < MIN_TITLE_CHARS
    df["content_too_short"] = df["content_char_count"] < MIN_CONTENT_CHARS

    # 7. Duplicates
    df["is_exact_duplicate"] = df.duplicated(subset=["news_url_clean", "title_clean", "published_at_utc", "content_clean"], keep="first")
    df["is_duplicate_url"] = df["news_url_clean"].notna() & df.duplicated(subset=["news_url_clean"], keep="first")

    # 8. Quality Flags & Splitting
    df["reject_for_nlp"] = (df["news_url_clean"].isna() | df["title_clean"].isna() | df["content_clean"].isna() | df["is_exact_duplicate"])
    
    clean_df = df[~df["reject_for_nlp"]].copy()
    rejected_df = df[df["reject_for_nlp"]].copy()
    review_df = df[(df["content_too_short"] | df["possible_block_or_paywall"]) & ~df["reject_for_nlp"]].copy()

    # 9. Export
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    clean_df.to_csv(out_path / "news_clean.csv", index=False, encoding="utf-8-sig")
    review_df.to_csv(out_path / "news_review.csv", index=False, encoding="utf-8-sig")
    rejected_df.to_csv(out_path / "news_rejected.csv", index=False, encoding="utf-8-sig")
    
    summary = {
        "total_rows": len(df), "clean_rows": len(clean_df), 
        "review_rows": len(review_df), "rejected_rows": len(rejected_df)
    }
    with open(out_path / "preprocessing_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ PREPROCESSING COMPLETE!")
    print(f"📁 Clean: {len(clean_df)} | Review: {len(review_df)} | Rejected: {len(rejected_df)}")
    print(f"📂 Files saved to: {out_path.absolute()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess scraped news data.")
    parser.add_argument("--input", default=r"I:\Amazon\Playwright\all_scrap_data.csv", help="Input CSV file")
    parser.add_argument("--output", default=r"I:\Amazon\Playwright\preprocessed_output", help="Output directory")
    args = parser.parse_args()
    main(args.input, args.output)