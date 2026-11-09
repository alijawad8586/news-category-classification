#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys
import csv
import re
import json
import time
import argparse
from pathlib import Path
from collections import Counter
from urllib.parse import urlparse

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except Exception:
    HAS_PLOT = False

csv.field_size_limit(sys.maxsize)
pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 200)


# ==============================================================
# CONFIG
# ==============================================================
DEFAULT_INPUT  = Path(r"S:\Project_task\datascrape_project\preprocessed_output\news_final_clean.csv")
DEFAULT_OUTPUT = Path(r"S:\Project_task\datascrape_project\features_output")

# Sirf zaroori columns load karein (1.5 GB file par RAM bachane ke liye)
USEFUL_COLUMN_HINTS = [
    "url", "title", "content", "text", "body", "publish", "date", "time",
    "scrape", "domain", "author", "category", "niche", "topic", "section",
    "archive", "word_count", "char_count", "duplicate", "paywall",
    "valid", "clean", "source", "reject", "review",
]

# Ratio-based metrics ke liye text cap (speed optimization)
TEXT_CAP_FOR_RATIOS = 6000

# Top-N domains ke liye one-hot encoding
TOP_N_DOMAINS = 20

# Simple sentiment lexicons
POSITIVE_WORDS = {
    "surge", "surges", "gain", "gains", "rise", "rises", "rally", "rallies",
    "beat", "beats", "growth", "grow", "strong", "record", "approve",
    "approved", "success", "successful", "boost", "boosted", "win", "wins",
    "improve", "improved", "profit", "profits", "advance", "advanced",
    "optimistic", "breakthrough", "positive", "best", "leader", "leads",
}
NEGATIVE_WORDS = {
    "fall", "falls", "drop", "drops", "decline", "declines", "loss", "losses",
    "crash", "crashes", "fear", "fears", "warning", "warns", "risk", "risks",
    "bankruptcy", "fraud", "scandal", "concern", "concerns", "problem",
    "problems", "fail", "fails", "failure", "cut", "cuts", "layoff",
    "layoffs", "recession", "inflation", "crisis", "negative", "worst",
    "lawsuit", "sued", "investigation", "breach", "attack", "attacks",
}

# ==============================================================
# KEYWORD GROUPS  (topic_guess banane ke liye)
# ==============================================================
KEYWORD_GROUPS = {
    "finance": [
        "earnings", "revenue", "profit", "loss", "stock", "stocks", "shares",
        "market", "markets", "forecast", "acquisition", "merger", "bankruptcy",
        "dividend", "inflation", "interest rate", "investment", "investor",
        "investors", "debt", "valuation", "ipo", "bond", "bonds", "currency",
        "dollar", "trading", "hedge fund", "equity", "quarterly results",
    ],
    "technology": [
        "artificial intelligence", "ai", "machine learning", "chip", "chips",
        "semiconductor", "software", "hardware", "cloud", "data center",
        "gpu", "algorithm", "startup", "app", "cybersecurity", "encryption",
        "quantum", "robot", "robotics", "api", "open source", "model",
        "large language model", "generative", "network", "5g", "device",
    ],
    "health": [
        "vaccine", "clinical trial", "fda", "patient", "patients", "disease",
        "treatment", "therapy", "doctor", "doctors", "hospital", "hospitals",
        "health", "medical", "research", "drug", "drugs", "symptoms",
        "diagnosis", "cancer", "flu", "pandemic", "public health", "study",
    ],
    "politics": [
        "senate", "congress", "parliament", "lawmakers", "election",
        "elections", "president", "minister", "government", "policy",
        "policies", "bill", "legislation", "vote", "voters", "campaign",
        "democrat", "republican", "opposition", "white house", "trade policy",
        "sanctions", "diplomat", "regulation",
    ],
    "business": [
        "company", "companies", "ceo", "executive", "revenue", "sales",
        "customers", "product", "launch", "expansion", "retail", "supply chain",
        "manufacturing", "jobs", "employment", "wages", "economy", "economic",
        "gdp", "consumer", "brand", "corporate", "board", "shareholders",
    ],
    "science": [
        "scientists", "researchers", "university", "journal", "peer-reviewed",
        "experiment", "physics", "chemistry", "biology", "astronomy", "nasa",
        "space", "climate", "environment", "emissions", "species", "genome",
        "telescope", "discovery",
    ],
    "world": [
        "international", "global", "united nations", "war", "conflict",
        "military", "troops", "ceasefire", "refugees", "summit", "treaty",
        "ukraine", "china", "russia", "europe", "middle east", "aid",
        "humanitarian",
    ],
    "entertainment": [
        "movie", "movies", "film", "films", "netflix", "series", "celebrity",
        "music", "album", "concert", "award", "oscars", "box office",
        "streaming", "game", "games", "gaming", "sports", "football",
        "cricket", "nba", "fifa",
    ],
}

# Flat keyword list (combined regex ke liye)
ALL_KEYWORDS = sorted(
    {kw for group in KEYWORD_GROUPS.values() for kw in group},
    key=len, reverse=True,          # longer first -> better alternation match
)

KEYWORD_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in ALL_KEYWORDS) + r")\b",
    flags=re.IGNORECASE,
)

KEYWORD_TO_GROUP = {}
for grp, kws in KEYWORD_GROUPS.items():
    for kw in kws:
        KEYWORD_TO_GROUP.setdefault(kw.lower(), set()).add(grp)


# ==============================================================
# HELPERS
# ==============================================================
def timer(label):
    print(f"\n{'─'*70}\n⏳ {label}\n{'─'*70}")
    return time.time()


def elapsed(t0):
    print(f"   ⏱️  Time: {time.time() - t0:.1f}s")


def pick(df, candidates):
    """Column auto-detect (case/space insensitive)"""
    lookup = {str(c).strip().lower().replace(" ", "_"): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower().replace(" ", "_")
        if key in lookup:
            return lookup[key]
        if cand in df.columns:
            return cand
    return None


def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def downcast_numeric(df):
    """Memory kam karne ke liye numeric columns downcast"""
    for c in df.select_dtypes(include=["integer"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    for c in df.select_dtypes(include=["float"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="float")
    return df


# ==============================================================
# MAIN
# ==============================================================
def main():
    parser = argparse.ArgumentParser(description="Feature engineering for scraped news data")
    parser.add_argument("--input",  type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit",  type=int, default=None,
                        help="Quick test ke liye sirf N rows process karein")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    INPUT_PATH, OUTPUT_DIR = args.input, args.output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR = OUTPUT_DIR / "plots"
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    FEATURES = {}   # feature_name -> {"dtype":..., "group":..., "description":...}

    def reg(name, group, desc):
        FEATURES[name] = {"group": group, "description": desc}

    print("=" * 70)
    print("🔧 FEATURE ENGINEERING PIPELINE")
    print("=" * 70)

    # ----------------------------------------------------------
    # STEP 1: Header sniff + selective load
    # ----------------------------------------------------------
    t = timer("STEP 1: Loading data (selective columns)")
    if not INPUT_PATH.exists():
        print(f"❌ File not found: {INPUT_PATH}")
        return

    header = pd.read_csv(INPUT_PATH, nrows=0, encoding="utf-8-sig").columns.tolist()
    print(f"📋 File ke columns ({len(header)}): {header}")

    usecols = [c for c in header
               if any(h in str(c).lower() for h in USEFUL_COLUMN_HINTS)]
    if not usecols:
        usecols = header
    print(f"🎯 Loading {len(usecols)} useful columns: {usecols}")

    df = pd.read_csv(INPUT_PATH, usecols=usecols,
                     low_memory=False, encoding="utf-8-sig")

    if args.limit:
        df = df.head(args.limit).copy()
        print(f"   ✂️  Limited to {len(df):,} rows (--limit)")

    df = df.reset_index(drop=True)
    N = len(df)
    print(f"✅ Loaded {N:,} rows × {len(df.columns)} columns")
    print(f"💾 Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    elapsed(t)

    # ----------------------------------------------------------
    # STEP 2: Column resolution
    # ----------------------------------------------------------
    t = timer("STEP 2: Column resolution")
    COL = {
        "url":       pick(df, ["url_clean", "article_url", "news_url_clean", "url", "link"]),
        "title":     pick(df, ["title_clean", "article_title", "title", "headline"]),
        "content":   pick(df, ["content_clean", "article_content", "content", "body", "text"]),
        "published": pick(df, ["published_at_utc", "publish_date", "published_date_only",
                               "published_at", "date"]),
        "scraped":   pick(df, ["scraped_at_utc", "scraped_at", "scrape_time"]),
        "domain":    pick(df, ["domain", "source_domain", "website"]),
        "author":    pick(df, ["author", "author_clean", "byline"]),
        "category":  pick(df, ["category", "article_niche", "topic", "section", "news_category"]),
        "archive":   pick(df, ["is_archive_page"]),
    }
    for k, v in COL.items():
        print(f"   {k:<10} → {v}")

    if COL["title"] is None or COL["content"] is None:
        print("❌ title/content column nahi mila — feature engineering possible nahi")
        return

    # Agar domain column nahi, URL se bana lo
    if COL["domain"] is None and COL["url"]:
        df["domain"] = df[COL["url"]].map(
            lambda u: (urlparse(str(u)).hostname or "unknown") if pd.notna(u) else "unknown")
        COL["domain"] = "domain"
        print("   ✔ 'domain' URL se derive kar diya")
    elapsed(t)

    # ----------------------------------------------------------
    # STEP 3: Text normalization
    # ----------------------------------------------------------
    t = timer("STEP 3: Text normalization")
    df["title_norm"] = df[COL["title"]].map(clean_text)
    df["text_norm"]  = df[COL["content"]].map(clean_text)
    df["title_lower"] = df["title_norm"].str.lower()
    df["text_lower"]  = df["text_norm"].str.lower()
    reg("title_norm", "text", "Cleaned & whitespace-normalized title")
    reg("title_lower", "text", "Lowercased title")
    print(f"   ✔ title_norm / text_norm ready")
    elapsed(t)

    # ----------------------------------------------------------
    # STEP 4: Article statistics (vectorized + single apply)
    # ----------------------------------------------------------
    t = timer("STEP 4: Article statistics")

    txt = df["text_norm"]
    ttl = df["title_norm"]

    # --- Vectorized (fast, C-level) ---
    df["body_char_count"]      = txt.str.len()
    df["body_word_count"]      = txt.str.count(r"\b\w+\b")
    df["body_sentence_count"]  = txt.str.count(r"[.!?]+")
    df["body_number_count"]    = txt.str.count(r"\b\d+(?:[.,]\d+)*%?\b")
    df["body_url_count"]       = txt.str.count(r"https?://|www\.")
    df["body_comma_count"]     = txt.str.count(r",")
    df["body_quote_count"]     = txt.str.count(r'["\']')
    df["title_word_count"]     = ttl.str.count(r"\b\w+\b")
    df["title_char_count"]     = ttl.str.len()

    # --- Ratio metrics: capped text par ek hi apply pass ---
    capped = txt.str.slice(0, TEXT_CAP_FOR_RATIOS)

    def ratio_metrics(s):
        if not s:
            return (0.0, 0.0, 0.0)
        words = s.split()
        n = len(words)
        if n == 0:
            return (0.0, 0.0, 0.0)
        total_len = sum(len(w) for w in words)
        uniq = len(set(w.lower() for w in words))
        sents = max(len([x for x in re.split(r"[.!?]+", s) if x.strip()]), 1)
        return (total_len / n, uniq / n, n / sents)

    ratios = capped.map(ratio_metrics)
    df["body_avg_word_length"]    = ratios.map(lambda x: x[0]).astype("float32")
    df["body_unique_word_ratio"]  = ratios.map(lambda x: x[1]).astype("float32")
    df["body_avg_sentence_len"]   = ratios.map(lambda x: x[2]).astype("float32")

    # Derived ratios
    df["content_to_title_ratio"] = (
        df["body_word_count"] / df["title_word_count"].replace(0, np.nan)
    ).astype("float32")

    df["body_readability_est"] = np.clip(
        206.835
        - 1.015 * df["body_avg_sentence_len"]
        - 84.6 * (df["body_avg_word_length"] / 5.0),
        0, 120
    ).astype("float32")   # simplified Flesch-style proxy

    for c in ["body_char_count", "body_word_count", "body_sentence_count",
              "body_number_count", "body_url_count", "body_comma_count",
              "body_quote_count", "title_word_count", "title_char_count"]:
        reg(c, "length", f"Count-based statistic: {c}")
    for c in ["body_avg_word_length", "body_unique_word_ratio",
              "body_avg_sentence_len", "content_to_title_ratio",
              "body_readability_est"]:
        reg(c, "length", f"Ratio / density feature: {c}")

    print(df[["body_word_count", "body_sentence_count",
              "body_unique_word_ratio", "body_avg_sentence_len"]].describe().round(2).to_string())
    elapsed(t)

    # ----------------------------------------------------------
    # STEP 5: Keyword features + topic_guess
    # ----------------------------------------------------------
    t = timer(f"STEP 5: Keyword extraction ({len(ALL_KEYWORDS)} keywords)")

    def count_kw(s):
        if not s:
            return {}
        return Counter(m.lower() for m in KEYWORD_RE.findall(s))

    kw_counters = capped.map(count_kw)      # ek hi pass over text

    # Individual keyword indicator columns (binary presence)
    kw_names = []
    for kw in ALL_KEYWORDS:
        col = "kw_" + re.sub(r"[^a-z0-9]+", "_", kw).strip("_")
        kw_names.append((kw, col))

    kw_matrix = np.zeros((N, len(kw_names)), dtype=np.int16)
    for i, cnt in enumerate(kw_counters):
        if cnt:
            for j, (kw, _) in enumerate(kw_names):
                if kw in cnt:
                    kw_matrix[i, j] = 1

    kw_df = pd.DataFrame(kw_matrix, columns=[c for _, c in kw_names], index=df.index)
    df = pd.concat([df, kw_df], axis=1)

    for kw, col in kw_names:
        reg(col, "keyword", f"Binary presence of keyword '{kw}' in body")

    # Group-level counts
    group_counts = {g: np.zeros(N, dtype=np.int32) for g in KEYWORD_GROUPS}
    for i, cnt in enumerate(kw_counters):
        for kw, n in cnt.items():
            for g in KEYWORD_TO_GROUP.get(kw, ()):
                group_counts[g][i] += n

    for g, arr in group_counts.items():
        df[f"topic_{g}_count"] = arr
        reg(f"topic_{g}_count", "keyword_group", f"Total occurrences of {g} keywords")

    grp_df = pd.DataFrame(group_counts, index=df.index)
    df["topic_total_kw"] = grp_df.sum(axis=1)

    has_any = df["topic_total_kw"] > 0
    df["topic_guess"] = np.where(has_any, grp_df.idxmax(axis=1), "unknown")
    df["topic_confidence"] = np.where(
        has_any,
        (grp_df.max(axis=1) / df["topic_total_kw"].replace(0, np.nan)).fillna(0),
        0.0
    ).astype("float32")

    reg("topic_total_kw", "keyword_group", "Sum of all topic keyword hits")
    reg("topic_guess", "category", "Derived topic label (argmax of keyword groups)")
    reg("topic_confidence", "category", "Share of dominant topic in total keyword hits")

    print("\n📊 Topic distribution:")
    print(df["topic_guess"].value_counts().to_string())
    print(f"\n📈 Keyword coverage: {has_any.mean()*100:.2f}% rows mein kam-az-kam 1 keyword")
    elapsed(t)

    # ----------------------------------------------------------
    # STEP 6: Title features
    # ----------------------------------------------------------
    t = timer("STEP 6: Title features")

    df["title_has_question"]  = ttl.str.contains(r"\?", regex=True).astype(np.int8)
    df["title_has_number"]    = ttl.str.contains(r"\b\d+\b", regex=True).astype(np.int8)
    df["title_has_percent"]   = ttl.str.contains(r"%", regex=False).astype(np.int8)
    df["title_has_colon"]     = ttl.str.contains(r":", regex=False).astype(np.int8)
    df["title_exclamation"]   = ttl.str.count(r"!").clip(0, 5).astype(np.int8)
    df["title_caps_ratio"]    = (
        ttl.str.count(r"\b[A-Z]{2,}\b") / ttl.str.count(r"\b\w+\b").replace(0, np.nan)
    ).fillna(0).astype("float32")
    df["title_starts_with_verb_hint"] = ttl.str.match(
        r"^(How|Why|What|When|Where|Who)\b", case=False).astype(np.int8)
    df["title_is_clickbait_ish"] = (
        ttl.str.contains(r"\b(you won'?t|shocking|unbelievable|secret|revealed|"
                         r"must[- ]see|watch|leaked)\b", case=False, regex=True)
    ).astype(np.int8)

    # Simple lexicon sentiment on title
    def lexicon_score(text):
        if not text:
            return (0, 0)
        words = re.findall(r"[a-z]+", text.lower())
        if not words:
            return (0, 0)
        pos = sum(1 for w in words if w in POSITIVE_WORDS)
        neg = sum(1 for w in words if w in NEGATIVE_WORDS)
        return (pos, neg)

    title_sent = df["title_norm"].map(lexicon_score)
    df["title_pos_words"] = title_sent.map(lambda x: x[0]).astype(np.int8)
    df["title_neg_words"] = title_sent.map(lambda x: x[1]).astype(np.int8)
    df["title_sentiment"] = (df["title_pos_words"] - df["title_neg_words"]).astype(np.int8)

    for c in ["title_has_question", "title_has_number", "title_has_percent",
              "title_has_colon", "title_exclamation", "title_caps_ratio",
              "title_starts_with_verb_hint", "title_is_clickbait_ish",
              "title_pos_words", "title_neg_words", "title_sentiment"]:
        reg(c, "title", f"Title-based signal: {c}")

    print(df[["title_has_question", "title_has_number",
              "title_sentiment"]].describe().round(3).to_string())
    elapsed(t)

    # ----------------------------------------------------------
    # STEP 7: URL & domain features
    # ----------------------------------------------------------
    t = timer("STEP 7: URL / domain features")

    if COL["url"]:
        urls = df[COL["url"]].fillna("").astype(str)

        parsed = urls.map(lambda u: urlparse(u if "://" in u else "https://" + u))

        df["url_length"]       = urls.str.len().astype(np.int16)
        df["url_https"]        = parsed.map(lambda p: int(p.scheme.lower() == "https")).astype(np.int8)
        df["url_has_query"]    = parsed.map(lambda p: int(bool(p.query))).astype(np.int8)
        df["url_has_fragment"] = parsed.map(lambda p: int(bool(p.fragment))).astype(np.int8)
        df["url_path_depth"]   = parsed.map(
            lambda p: len([x for x in p.path.split("/") if x])).astype(np.int8)
        df["url_slug"]         = parsed.map(
            lambda p: [x for x in p.path.split("/") if x][-1] if any(p.path.split("/")) else "")
        df["url_slug_length"]  = df["url_slug"].str.len().astype(np.int16)
        df["url_has_date_pattern"] = urls.str.contains(
            r"/(19|20)\d{2}/", regex=True).astype(np.int8)
        df["url_is_index"]     = parsed.map(
            lambda p: int(p.path.strip("/") in ("", "index.html", "index.php"))).astype(np.int8)

        for c in ["url_length", "url_https", "url_has_query", "url_has_fragment",
                  "url_path_depth", "url_slug_length", "url_has_date_pattern", "url_is_index"]:
            reg(c, "url", f"URL structural feature: {c}")

        df["domain_feature"] = parsed.map(lambda p: p.netloc.lower().removeprefix("www."))
        reg("domain_feature", "url", "Normalized registrable host from URL")
    else:
        df["domain_feature"] = df[COL["domain"]].fillna("unknown") if COL["domain"] else "unknown"

    print(f"🌐 Unique domains: {df['domain_feature'].nunique():,}")
    print(df["domain_feature"].value_counts().head(10).to_string())
    elapsed(t)

    # ----------------------------------------------------------
    # STEP 8: Datetime features
    # ----------------------------------------------------------
    t = timer("STEP 8: Datetime features")

    if COL["published"]:
        raw = df[COL["published"]]
        try:
            dt = pd.to_datetime(raw, errors="coerce", utc=True, format="mixed")
        except Exception:
            dt = pd.to_datetime(raw, errors="coerce", utc=True)

        df["published_at"] = dt
    else:
        df["published_at"] = pd.NaT

    valid_dt = df["published_at"].notna()
    df["published_is_valid"] = valid_dt.astype(np.int8)
    reg("published_is_valid", "time", "1 = usable publication timestamp")

    if valid_dt.any():
        dt = df["published_at"]
        df["pub_year"]      = dt.dt.year.astype("Int16")
        df["pub_month"]     = dt.dt.month.astype("Int8")
        df["pub_day"]       = dt.dt.day.astype("Int8")
        df["pub_hour"]      = dt.dt.hour.astype("Int8")
        df["pub_minute"]    = dt.dt.minute.astype("Int8")
        df["pub_dayofweek"] = dt.dt.dayofweek.astype("Int8")
        df["pub_dayofyear"] = dt.dt.dayofyear.astype("Int16")
        df["pub_quarter"]   = dt.dt.quarter.astype("Int8")
        df["pub_weekofyear"]= dt.dt.isocalendar().week.astype("Int8")

        df["pub_is_weekend"] = df["pub_dayofweek"].isin([5, 6]).astype(np.int8)
        df["pub_is_night"]   = df["pub_hour"].between(0, 5).astype(np.int8)
        df["pub_is_business_hours"] = (
            df["pub_hour"].between(8, 17) & (df["pub_is_weekend"] == 0)
        ).astype(np.int8)
        df["pub_is_month_start"] = df["pub_day"].le(3).astype(np.int8)
        df["pub_is_month_end"]   = df["pub_day"].ge(28).astype(np.int8)

        # Cyclical encoding
        df["pub_hour_sin"] = np.sin(2 * np.pi * df["pub_hour"] / 24).astype("float32")
        df["pub_hour_cos"] = np.cos(2 * np.pi * df["pub_hour"] / 24).astype("float32")
        df["pub_dow_sin"]  = np.sin(2 * np.pi * df["pub_dayofweek"] / 7).astype("float32")
        df["pub_dow_cos"]  = np.cos(2 * np.pi * df["pub_dayofweek"] / 7).astype("float32")
        df["pub_month_sin"]= np.sin(2 * np.pi * df["pub_month"] / 12).astype("float32")
        df["pub_month_cos"]= np.cos(2 * np.pi * df["pub_month"] / 12).astype("float32")

        # Recency
        latest = dt.max()
        df["days_since_latest"] = ((latest - dt).dt.total_seconds() / 86400).astype("float32")
        df["hours_since_latest"] = ((latest - dt).dt.total_seconds() / 3600).astype("float32")

        # Ordinal timestamp (trend-friendly numeric)
        df["pub_ordinal_days"] = (dt.astype("int64") / 86400e9).astype("float32")

        for c in ["pub_year", "pub_month", "pub_day", "pub_hour", "pub_minute",
                  "pub_dayofweek", "pub_dayofyear", "pub_quarter", "pub_weekofyear",
                  "pub_is_weekend", "pub_is_night", "pub_is_business_hours",
                  "pub_is_month_start", "pub_is_month_end",
                  "pub_hour_sin", "pub_hour_cos", "pub_dow_sin", "pub_dow_cos",
                  "pub_month_sin", "pub_month_cos",
                  "days_since_latest", "hours_since_latest", "pub_ordinal_days"]:
            reg(c, "time", f"Temporal feature derived from published_at: {c}")

        print(f"📅 Valid timestamps: {valid_dt.sum():,} / {N:,} ({valid_dt.mean()*100:.2f}%)")
        print(f"   Range: {dt.min()} → {dt.max()}")
        print(f"   Year spread:\n{df['pub_year'].value_counts().sort_index().to_string()}")
    else:
        print("⚠️ Koi valid date nahi — temporal features skip")

    # Article age (agar scraped_at mojood ho)
    if COL["scraped"]:
        try:
            sc = pd.to_datetime(df[COL["scraped"]], errors="coerce", utc=True)
            df["article_age_hours"] = ((sc - df["published_at"]).dt.total_seconds() / 3600).astype("float32")
            df["article_age_days"] = (df["article_age_hours"] / 24).astype("float32")
            reg("article_age_hours", "time", "Scrape lag in hours (scraped_at - published_at)")
            reg("article_age_days", "time", "Scrape lag in days")
            print(f"   ✔ article_age computed")
        except Exception as e:
            print(f"   ⚠️ scraped_at parse failed: {e}")
    else:
        print("   ℹ️ scraped_at column nahi hai → article_age features skip")
    elapsed(t)

    # ----------------------------------------------------------
    # STEP 9: Author features
    # ----------------------------------------------------------
    t = timer("STEP 9: Author features")

    if COL["author"]:
        auth = df[COL["author"]].map(clean_text).replace("", np.nan)
        auth = auth.where(~auth.str.lower().isin(["unknown", "n/a", "na", "none", "null"]), np.nan)
    else:
        auth = pd.Series(np.nan, index=df.index)

    df["author_clean"] = auth
    df["has_author"] = auth.notna().astype(np.int8)
    reg("has_author", "author", "1 = author name present")

    if auth.notna().any():
        vc = auth.value_counts()
        df["author_article_count"] = auth.map(vc).fillna(0).astype("float32")
        df["author_freq"] = (df["author_article_count"] / N).astype("float32")
        df["author_is_prolific"] = (df["author_article_count"] >= 50).astype(np.int8)
        df["author_name_word_count"] = auth.fillna("").map(
            lambda s: len(s.split())).astype(np.int8)
        for c in ["author_article_count", "author_freq",
                  "author_is_prolific", "author_name_word_count"]:
            reg(c, "author", f"Author-derived feature: {c}")
        print(f"✍️  Known authors: {auth.notna().sum():,} ({auth.notna().mean()*100:.2f}%)")
        print(f"   Unique authors: {auth.nunique():,}")
        print(vc.head(10).to_string())
    else:
        print("⚠️ Koi author nahi mila")
    elapsed(t)

    # ----------------------------------------------------------
    # STEP 10: Categorical encodings
    # ----------------------------------------------------------
    t = timer("STEP 10: Categorical encoding")

    # (a) Native category column (agar ho)
    cat_col = COL["category"]
    if cat_col:
        df["category_clean"] = df[cat_col].map(clean_text).replace("", "Unknown")
        df["category_encoded"] = df["category_clean"].astype("category").cat.codes.astype(np.int16)
        reg("category_clean", "category", "Original category label from source")
        reg("category_encoded", "category", "Label-encoded original category")
        print(f"📁 Native category column mila: '{cat_col}'")
        print(df["category_clean"].value_counts().head(10).to_string())
    else:
        print("ℹ️ Native 'category' column nahi — 'topic_guess' use hoga")
        df["category_clean"] = df["topic_guess"]
        df["category_encoded"] = df["category_clean"].astype("category").cat.codes.astype(np.int16)
        reg("category_clean", "category", "Derived topic label (keyword-based)")
        reg("category_encoded", "category", "Label-encoded derived topic")

    # (b) Domain: frequency + label + top-N one-hot
    dom = df["domain_feature"].fillna("unknown").astype(str)
    dom_counts = dom.value_counts()
    df["domain_freq"] = (dom.map(dom_counts) / N).astype("float32")
    df["domain_encoded"] = dom.astype("category").cat.codes.astype(np.int16)
    df["domain_is_rare"] = (dom.map(dom_counts) < 20).astype(np.int8)
    reg("domain_freq", "categorical", "Domain frequency encoding (count / N)")
    reg("domain_encoded", "categorical", "Label-encoded domain")
    reg("domain_is_rare", "categorical", "1 = domain appears < 20 times")

    top_domains = dom_counts.head(TOP_N_DOMAINS).index.tolist()
    for d in top_domains:
        safe = re.sub(r"[^a-z0-9]+", "_", d.lower()).strip("_")
        col = f"domain_is_{safe}"
        df[col] = (dom == d).astype(np.int8)
        reg(col, "categorical", f"One-hot: article from {d}")
    df["domain_is_other"] = (~dom.isin(top_domains)).astype(np.int8)
    reg("domain_is_other", "categorical", f"1 = domain not in top {TOP_N_DOMAINS}")
    print(f"🌐 One-hot for top {TOP_N_DOMAINS} domains + 'other'")

    # (c) Topic one-hot
    for g in list(KEYWORD_GROUPS) + ["unknown"]:
        col = f"topic_is_{g}"
        df[col] = (df["topic_guess"] == g).astype(np.int8)
        reg(col, "categorical", f"One-hot: derived topic == {g}")
    elapsed(t)

    # ----------------------------------------------------------
    # STEP 11: Quality / meta flags carry-through
    # ----------------------------------------------------------
    t = timer("STEP 11: Quality & meta flags")

    flag_src = ["is_archive_page", "is_duplicate_url", "is_exact_duplicate",
                "title_too_short", "content_too_short",
                "possible_block_or_paywall", "url_valid",
                "date_out_of_range", "needs_review", "reject_for_nlp"]
    carried = []
    for f in flag_src:
        if f in df.columns:
            col = f"flag_{f}"
            df[col] = df[f].map(lambda v: int(bool(v)) if pd.notna(v) else 0).astype(np.int8)
            carried.append(col)
            reg(col, "quality_flag", f"Carried quality flag: {f}")

    # Missing-indicator features (missingness informative ho sakti hai)
    for key in ["published", "author", "domain", "url"]:
        src = COL.get(key)
        if src:
            col = f"missing_{key}"
            df[col] = df[src].isna().astype(np.int8)
            carried.append(col)
            reg(col, "missingness", f"1 = {key} value missing")

    print(f"   ✔ {len(carried)} flag/missingness features add hue")
    elapsed(t)

    # ----------------------------------------------------------
    # STEP 12: Assemble final feature matrix
    # ----------------------------------------------------------
    t = timer("STEP 12: Assembling feature matrix")

    DROP_COLS = {
        "title_norm", "text_norm", "title_lower", "text_lower",
        "url_slug", "published_at", "author_clean",
    }
    # Original raw columns (jinke clean versions already features hain)
    for k, v in COL.items():
        if v:
            DROP_COLS.add(v)
    DROP_COLS |= set(flag_src)

    keep = [c for c in df.columns if c not in DROP_COLS]
    # Non-numeric text columns drop (model mein nahi jayenge)
    numeric_like = df[keep].select_dtypes(include=[np.number, "Int8", "Int16", "Int32", "Int64"]).columns
    categorical_like = ["domain_feature", "category_clean", "topic_guess"]
    categorical_like = [c for c in categorical_like if c in keep]

    feature_cols = list(numeric_like) + categorical_like
    feat = df[feature_cols].copy()
    feat = downcast_numeric(feat)

    print(f"🧮 Total features: {len(feat.columns):,}")
    print(f"   Numeric:     {len(numeric_like):,}")
    print(f"   Categorical: {len(categorical_like)}")
    print(f"💾 Feature matrix memory: {feat.memory_usage(deep=True).sum()/1024**2:.1f} MB")
    elapsed(t)

    # ----------------------------------------------------------
    # STEP 13: Save
    # ----------------------------------------------------------
    t = timer("STEP 13: Saving outputs")

    # Parquet (preferred)
    parquet_path = OUTPUT_DIR / "news_features.parquet"
    try:
        feat.to_parquet(parquet_path, index=False, compression="snappy")
        SAVED = [parquet_path]
        print(f"   💾 {parquet_path.name} ({parquet_path.stat().st_size/1024**2:.2f} MB)")
    except Exception as e:
        print(f"   ⚠️ Parquet save failed ({e}) — pyarrow/fastparquet install karein")
        SAVED = []

    # CSV.gz (compatibility)
    csv_path = OUTPUT_DIR / "news_features.csv.gz"
    feat.to_csv(csv_path, index=False, compression="gzip", encoding="utf-8-sig")
    SAVED.append(csv_path)
    print(f"   💾 {csv_path.name} ({csv_path.stat().st_size/1024**2:.2f} MB)")

    # Feature dictionary
    for c in feat.columns:
        if c not in FEATURES:
            FEATURES[c] = {"group": "other", "description": ""}
        FEATURES[c]["dtype"] = str(feat[c].dtype)
        FEATURES[c]["missing_pct"] = round(float(feat[c].isna().mean() * 100), 3)

    dict_payload = {
        "source_file": str(INPUT_PATH),
        "rows": int(N),
        "total_features": int(len(feat.columns)),
        "feature_groups": {},
        "features": FEATURES,
    }
    for name, meta in FEATURES.items():
        g = meta.get("group", "other")
        dict_payload["feature_groups"].setdefault(g, []).append(name)

    dict_path = OUTPUT_DIR / "feature_dictionary.json"
    with open(dict_path, "w", encoding="utf-8") as f:
        json.dump(dict_payload, f, indent=2, ensure_ascii=False)
    print(f"   💾 {dict_path.name}")

    print("\n📚 Feature groups:")
    for g, cols in sorted(dict_payload["feature_groups"].items()):
        print(f"   {g:<16}: {len(cols):>4} features")
    elapsed(t)

    # ----------------------------------------------------------
    # STEP 14: Plots
    # ----------------------------------------------------------
    if HAS_PLOT and not args.no_plots:
        t = timer("STEP 14: Feature visualization")
        sns.set_theme(style="whitegrid")

        def save(fig, name):
            p = PLOT_DIR / name
            fig.savefig(p, dpi=110, bbox_inches="tight")
            plt.close(fig)
            print(f"   🖼️  plots/{name}")

        # (1) Key distributions
        plot_cols = [c for c in ["body_word_count", "title_word_count",
                                 "body_unique_word_ratio", "body_avg_sentence_len",
                                 "content_to_title_ratio"] if c in feat]
        if plot_cols:
            fig, axes = plt.subplots(1, len(plot_cols), figsize=(5 * len(plot_cols), 4))
            axes = np.atleast_1d(axes)
            for ax, c in zip(axes, plot_cols):
                s = pd.to_numeric(feat[c], errors="coerce").dropna()
                if len(s) == 0:
                    continue
                s = s[s <= s.quantile(0.99)]
                sns.histplot(s, bins=50, kde=True, color="teal", ax=ax)
                ax.set_title(c, fontsize=11, fontweight="bold")
            fig.tight_layout(); save(fig, "01_text_distributions.png")

        # (2) Topic distribution
        if "topic_guess" in feat:
            tc = feat["topic_guess"].value_counts()
            fig, ax = plt.subplots(figsize=(11, 5))
            ax.bar(tc.index.astype(str), tc.values,
                   color=sns.color_palette("viridis", len(tc)))
            ax.set_title("Derived Topic Distribution", fontsize=13, fontweight="bold")
            ax.set_ylabel("Articles")
            plt.xticks(rotation=40, ha="right")
            for i, v in enumerate(tc.values):
                ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
            fig.tight_layout(); save(fig, "02_topic_distribution.png")

        # (3) Topic confidence
        if "topic_confidence" in feat:
            fig, ax = plt.subplots(figsize=(9, 4.5))
            sns.histplot(pd.to_numeric(feat["topic_confidence"], errors="coerce").dropna(),
                         bins=40, kde=True, color="darkorange", ax=ax)
            ax.set_title("Topic Confidence Distribution", fontsize=13, fontweight="bold")
            fig.tight_layout(); save(fig, "03_topic_confidence.png")

        # (4) Publication hour / weekday
        if {"pub_hour", "pub_dayofweek"} <= set(feat.columns):
            fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
            h = pd.to_numeric(feat["pub_hour"], errors="coerce").value_counts().reindex(range(24), fill_value=0)
            axes[0].bar(h.index, h.values, color="orange", alpha=.85)
            axes[0].set_title("Articles by Hour (UTC)", fontsize=12, fontweight="bold")
            axes[0].set_xlabel("Hour")
            d = (pd.to_numeric(feat["pub_dayofweek"], errors="coerce")
                 .value_counts().reindex(range(7), fill_value=0))
            axes[1].bar(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], d.values,
                        color="teal", alpha=.85)
            axes[1].set_title("Articles by Day of Week", fontsize=12, fontweight="bold")
            fig.tight_layout(); save(fig, "04_time_patterns.png")

        # (5) Top domains
        if "domain_feature" in feat:
            dc = feat["domain_feature"].value_counts().head(15).sort_values()
            fig, ax = plt.subplots(figsize=(11, 6))
            ax.barh(dc.index.astype(str), dc.values,
                    color=sns.color_palette("viridis", len(dc)))
            ax.set_title("Top 15 Domains", fontsize=13, fontweight="bold")
            ax.set_xlabel("Articles")
            fig.tight_layout(); save(fig, "05_top_domains.png")

        # (6) Word count by topic (boxplot)
        if {"body_word_count", "topic_guess"} <= set(feat.columns):
            sub = feat[feat["topic_guess"] != "unknown"]
            tops = sub["topic_guess"].value_counts().head(8).index
            sub = sub[sub["topic_guess"].isin(tops)].copy()
            sub["wc"] = pd.to_numeric(sub["body_word_count"], errors="coerce")
            sub = sub[sub["wc"] <= sub["wc"].quantile(0.98)]
            fig, ax = plt.subplots(figsize=(12, 5.5))
            sns.boxplot(data=sub, x="topic_guess", y="wc", palette="Set2", ax=ax)
            ax.set_title("Content Word Count by Topic", fontsize=13, fontweight="bold")
            ax.set_ylabel("Words"); ax.set_xlabel("Topic")
            plt.xticks(rotation=35, ha="right")
            fig.tight_layout(); save(fig, "06_wordcount_by_topic.png")

        # (7) Correlation heatmap (top variance numeric)
        num = feat.select_dtypes(include=[np.number, "Int8", "Int16", "Int32", "Int64"])
        num = num.astype("float32", errors="ignore")
        variances = num.var(numeric_only=True).fillna(0)
        top = variances.sort_values(ascending=False).head(30).index.tolist()
        if len(top) >= 3:
            corr = num[top].corr(numeric_only=True).fillna(0)
            fig, ax = plt.subplots(figsize=(14, 12))
            sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1,
                        square=True, linewidths=.4, cbar_kws={"shrink": .7}, ax=ax)
            ax.set_title("Correlation — Top 30 Features by Variance",
                         fontsize=13, fontweight="bold")
            ax.tick_params(labelsize=7)
            fig.tight_layout(); save(fig, "07_correlation_heatmap.png")

        elapsed(t)
    else:
        print("\n⚠️ Plots skipped (matplotlib/seaborn unavailable ya --no-plots)")

    # ----------------------------------------------------------
    # STEP 15: Final summary
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("✅ FEATURE ENGINEERING COMPLETE")
    print("=" * 70)
    print(f"   Rows:            {N:,}")
    print(f"   Features:        {len(feat.columns):,}")
    print(f"   Output dir:      {OUTPUT_DIR.resolve()}")
    print(f"\n📂 Files:")
    for p in sorted(OUTPUT_DIR.iterdir()):
        if p.is_file():
            print(f"   📄 {p.name:<34} {p.stat().st_size/1024**2:>8.2f} MB")
    if PLOT_DIR.exists():
        pngs = sorted(PLOT_DIR.glob("*.png"))
        print(f"\n🖼️  Plots ({len(pngs)}):")
        for p in pngs:
            print(f"   {p.name:<34} {p.stat().st_size/1024:>8.1f} KB")
    print("=" * 70)

    # Preview
    print("\n🔎 Feature preview (first 3 rows, first 18 cols):")
    print(feat.iloc[:3, :18].to_string())


if __name__ == "__main__":
    main()