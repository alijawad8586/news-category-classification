import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ============================== CONFIG ==============================
INPUT_PATH = r"S:\Project_task\datascrape_project\news_final_clean.csv"
OUTPUT_DIR = Path(r"S:\Project_task\datascrape_project\eda_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================== LOAD DATA ==============================
print(f"🔄 Loading data from {INPUT_PATH}...")
df = pd.read_csv(INPUT_PATH, low_memory=False)
print(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns.\n")

# ============================== 1. DATASET OVERVIEW ==============================
print("=" * 60)
print("1. DATASET OVERVIEW & SCHEMA")
print("=" * 60)
overview = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "non_null": df.notna().sum(),
    "missing": df.isna().sum(),
    "missing_pct": (df.isna().mean() * 100).round(2),
    "unique_values": df.nunique(dropna=True)
})
print(overview.sort_values("missing_pct", ascending=False))
overview.to_csv(OUTPUT_DIR / "01_column_profile.csv")

# ============================== 2. MISSING VALUES ==============================
print("\n" + "=" * 60)
print("2. MISSING VALUES ANALYSIS")
print("=" * 60)
missing_summary = overview[['missing', 'missing_pct']].sort_values("missing_pct", ascending=False)
missing_summary = missing_summary[missing_summary['missing_pct'] > 0]

if not missing_summary.empty:
    print(missing_summary)
    missing_summary.to_csv(OUTPUT_DIR / "02_missingness_summary.csv")
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=missing_summary['missing_pct'].values, y=missing_summary.index, palette='magma')
    plt.title("Missing Values Percentage by Column", fontsize=14, fontweight='bold')
    plt.xlabel("Missing (%)")
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_missing_values_barplot.png")
    plt.close()
else:
    print("✅ No missing values found in the dataset!")

# ============================== 3. CORE FIELDS CHECK ==============================
print("\n" + "=" * 60)
print("3. CORE FIELDS ANALYSIS")
print("=" * 60)
core_fields = ['article_url', 'article_title', 'article_content', 'published_at_utc', 'domain', 'author', 'publish_time']
core_analysis = pd.DataFrame({
    'Field': core_fields,
    'Total': [len(df)] * len(core_fields),
    'Present': [df[f].notna().sum() if f in df.columns else 0 for f in core_fields],
    'Missing': [df[f].isna().sum() if f in df.columns else len(df) for f in core_fields],
    'Present_%': [(df[f].notna().sum() / len(df) * 100).round(2) if f in df.columns else 0 for f in core_fields]
})
print(core_analysis)
core_analysis.to_csv(OUTPUT_DIR / "03_core_fields_analysis.csv", index=False)

plt.figure(figsize=(10, 5))
plt.bar(core_analysis['Field'], core_analysis['Present_%'], color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22'])
plt.title("Core Fields - Present Percentage (%)", fontsize=14, fontweight='bold')
plt.ylabel("Present (%)")
plt.ylim(0, 105)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_core_fields_barplot.png")
plt.close()

# ============================== 4. DOMAIN ANALYSIS ==============================
print("\n" + "=" * 60)
print("4. DOMAIN / SOURCE ANALYSIS")
print("=" * 60)
if 'domain' in df.columns:
    domain_counts = df['domain'].value_counts()
    print(f"🌐 Total Unique Domains: {df['domain'].nunique()}")
    print(f"\n🏆 Top 15 Domains:\n{domain_counts.head(15)}")
    
    # Top 15 Domains Bar Chart
    plt.figure(figsize=(12, 7))
    sns.barplot(x=domain_counts.head(15).values, y=domain_counts.head(15).index, palette="viridis")
    plt.title("Top 15 Domains by Article Count", fontsize=14, fontweight='bold')
    plt.xlabel("Number of Articles")
    plt.ylabel("Domain")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_top_domains.png")
    plt.close()
    
    # Domain Share %
    domain_share = (domain_counts / domain_counts.sum() * 100).round(2).to_frame(name='share_pct')
    domain_share.to_csv(OUTPUT_DIR / "04_domain_distribution.csv")
    
    # Domain share pie chart (top 10)
    plt.figure(figsize=(10, 8))
    top10 = domain_counts.head(10)
    others = domain_counts.iloc[10:].sum()
    pie_data = pd.concat([top10, pd.Series({'Others': others})])
    plt.pie(pie_data.values, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
    plt.title("Domain Distribution (Top 10 + Others)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_domain_pie_chart.png")
    plt.close()
else:
    print("⚠️ 'domain' column not found.")

# ============================== 5. TEXT LENGTH ANALYSIS ==============================
print("\n" + "=" * 60)
print("5. TEXT LENGTH DISTRIBUTIONS")
print("=" * 60)
length_cols = ['title_char_count', 'content_char_count', 'content_word_count']
available_lengths = [c for c in length_cols if c in df.columns]

if available_lengths:
    length_stats = df[available_lengths].describe(percentiles=[.05, .25, .5, .75, .95]).T
    print(length_stats.round(2))
    length_stats.to_csv(OUTPUT_DIR / "05_text_length_stats.csv")
    
    # Histograms
    fig, axes = plt.subplots(1, len(available_lengths), figsize=(18, 5))
    if len(available_lengths) == 1:
        axes = [axes]
    for ax, col in zip(axes, available_lengths):
        data = df[col].dropna()
        # Clip extreme values for better visualization
        clip_val = data.quantile(0.99)
        data_clipped = data[data <= clip_val]
        sns.histplot(data_clipped, bins=50, ax=ax, kde=True, color='teal')
        ax.set_title(f"Distribution of {col}", fontsize=12, fontweight='bold')
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_text_length_distributions.png")
    plt.close()
    
    # Content Length by Top Domains (Boxplot)
    if 'domain' in df.columns and 'content_word_count' in df.columns:
        top_domains = df['domain'].value_counts().head(10).index
        df_top = df[df['domain'].isin(top_domains)]
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='domain', y='content_word_count', data=df_top, palette="Set2")
        plt.title("Content Length (Words) Distribution by Top 10 Domains", fontsize=14, fontweight='bold')
        plt.xlabel("Domain")
        plt.ylabel("Content Word Count")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "05_content_length_by_domain.png")
        plt.close()

# ============================== 6. TIME ANALYSIS ==============================
print("\n" + "=" * 60)
print("6. PUBLICATION TIME PATTERNS")
print("=" * 60)

if 'published_at_utc' in df.columns:
    df['published_dt'] = pd.to_datetime(df['published_at_utc'], errors='coerce', utc=True)
    valid_dates = df['published_dt'].dropna()
    print(f"📅 Valid publication timestamps: {len(valid_dates):,} / {len(df):,}")
    
    if len(valid_dates) > 0:
        print(f"   Earliest: {valid_dates.min()}")
        print(f"   Latest:   {valid_dates.max()}")
        print(f"   Coverage: {(valid_dates.max() - valid_dates.min()).days} days")
        
        # Daily counts
        df['published_date'] = df['published_dt'].dt.date
        daily_counts = df.groupby('published_date').size()
        
        plt.figure(figsize=(14, 5))
        daily_counts.plot(color='blue', alpha=0.7, linewidth=0.8)
        plt.title("Articles Published Per Day (UTC)", fontsize=14, fontweight='bold')
        plt.xlabel("Date")
        plt.ylabel("Number of Articles")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "06_daily_publications.png")
        plt.close()
        
        # Monthly counts
        df['published_month'] = df['published_dt'].dt.to_period('M')
        monthly_counts = df.groupby('published_month').size()
        
        plt.figure(figsize=(14, 5))
        monthly_counts.plot(kind='bar', color='purple', alpha=0.7)
        plt.title("Articles Published Per Month (UTC)", fontsize=14, fontweight='bold')
        plt.xlabel("Month")
        plt.ylabel("Number of Articles")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "06_monthly_publications.png")
        plt.close()
        
        # Yearly counts
        df['published_year'] = df['published_dt'].dt.year
        yearly_counts = df.groupby('published_year').size()
        
        plt.figure(figsize=(12, 5))
        yearly_counts.plot(kind='bar', color='darkgreen', alpha=0.7)
        plt.title("Articles Published Per Year", fontsize=14, fontweight='bold')
        plt.xlabel("Year")
        plt.ylabel("Number of Articles")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "06_yearly_publications.png")
        plt.close()
        
        # Hour of day
        df['published_hour'] = df['published_dt'].dt.hour
        hourly_counts = df.groupby('published_hour').size()
        
        plt.figure(figsize=(12, 5))
        hourly_counts.plot(kind='bar', color='orange', alpha=0.7)
        plt.title("Articles by Hour of Day (UTC)", fontsize=14, fontweight='bold')
        plt.xlabel("Hour (UTC)")
        plt.ylabel("Number of Articles")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "06_hourly_publications.png")
        plt.close()
        
        # Day of week
        df['published_day_name'] = df['published_dt'].dt.day_name()
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        weekday_counts = df['published_day_name'].value_counts().reindex(order, fill_value=0)
        
        plt.figure(figsize=(10, 5))
        weekday_counts.plot(kind='bar', color='teal', alpha=0.7)
        plt.title("Articles by Day of Week", fontsize=14, fontweight='bold')
        plt.xlabel("Day")
        plt.ylabel("Number of Articles")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "06_weekday_publications.png")
        plt.close()
        
        print(f"\n📊 Yearly Distribution:")
        print(yearly_counts)
        print(f"\n📊 Weekday Distribution:")
        print(weekday_counts)

# ============================== 7. AUTHOR ANALYSIS ==============================
print("\n" + "=" * 60)
print("7. AUTHOR ANALYSIS")
print("=" * 60)

if 'author' in df.columns:
    author_counts = df['author'].value_counts()
    known_authors = author_counts[author_counts.index != 'Unknown']
    unknown_count = author_counts.get('Unknown', 0)
    
    print(f"✍️  Total Unique Authors: {df['author'].nunique()}")
    print(f"   Known Authors: {len(known_authors)}")
    print(f"   Unknown: {unknown_count:,} ({unknown_count/len(df)*100:.2f}%)")
    
    if len(known_authors) > 0:
        print(f"\n🏆 Top 10 Known Authors:")
        print(known_authors.head(10))
            
    
    
# ============================== 8. QUALITY FLAGS ==============================
print("\n" + "=" * 60)
print("8. QUALITY FLAGS & EXTRACTION ISSUES")
print("=" * 60)
flag_cols = ['url_valid', 'possible_block_or_paywall', 'title_too_short', 
             'content_too_short', 'is_exact_duplicate', 'is_duplicate_url']
available_flags = [c for c in flag_cols if c in df.columns]

if available_flags:
    flag_summary = df[available_flags].sum().to_frame(name='count')
    flag_summary['percentage'] = (flag_summary['count'] / len(df) * 100).round(2)
    print(flag_summary)
    flag_summary.to_csv(OUTPUT_DIR / "08_quality_flags_summary.csv")
    
    plt.figure(figsize=(10, 5))
    sns.barplot(x=flag_summary.index, y=flag_summary['percentage'], palette='rocket')
    plt.title("Percentage of Articles with Quality Flags", fontsize=14, fontweight='bold')
    plt.ylabel("Percentage (%)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "08_quality_flags_barplot.png")
    plt.close()

# ============================== 9. TITLE ANALYSIS ==============================
print("\n" + "=" * 60)
print("9. TITLE ANALYSIS")
print("=" * 60)

if 'title_clean' in df.columns:
    # Most common title words
    all_words = ' '.join(df['title_clean'].dropna().astype(str)).lower().split()
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                  'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                  'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
                  'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                  'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
                  'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
                  'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each',
                  'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
                  'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
                  'and', 'but', 'or', 'nor', 'yet', 'it', 'its', 'this', 'that', 'these',
                  'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
                  'you', 'your', 'yours', 'yourself', 'he', 'him', 'his', 'himself',
                  'she', 'her', 'hers', 'herself', 'they', 'them', 'their', 'theirs',
                  'what', 'which', 'who', 'whom', 'if', 'while', 'about', 'against'}
    
    filtered_words = [w for w in all_words if w not in stop_words and len(w) > 3]
    word_freq = pd.Series(filtered_words).value_counts()
    
    print(f"📝 Top 20 Most Common Words in Titles:")
    print(word_freq.head(20))
    
    plt.figure(figsize=(12, 6))
    word_freq.head(20).plot(kind='bar', color='steelblue', alpha=0.8)
    plt.title("Top 20 Most Common Words in Article Titles", fontsize=14, fontweight='bold')
    plt.xlabel("Word")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "09_title_word_frequency.png")
    plt.close()

# ============================== 10. DATE COVERAGE BY DOMAIN ==============================
print("\n" + "=" * 60)
print("10. DOMAIN-WISE DATE COVERAGE")
print("=" * 60)

if 'domain' in df.columns and 'published_at_utc' in df.columns:
    domain_date_stats = df.groupby('domain').agg(
        total_articles=('published_at_utc', 'count'),
        articles_with_date=('published_at_utc', lambda x: x.notna().sum()),
        earliest_date=('published_dt', 'min'),
        latest_date=('published_dt', 'max')
    ).sort_values('total_articles', ascending=False)
    
    domain_date_stats['date_coverage_pct'] = (domain_date_stats['articles_with_date'] / domain_date_stats['total_articles'] * 100).round(2)
    
    print(domain_date_stats.head(20))
    domain_date_stats.to_csv(OUTPUT_DIR / "10_domain_date_coverage.csv")

# ============================== 11. SUMMARY JSON ==============================
print("\n" + "=" * 60)
print("11. GENERATING EDA SUMMARY")
print("=" * 60)

eda_summary = {
    "total_rows": int(len(df)),
    "total_columns": int(len(df.columns)),
    "unique_domains": int(df['domain'].nunique()) if 'domain' in df.columns else 0,
    "valid_dates_count": int(df['published_at_utc'].notna().sum()) if 'published_at_utc' in df.columns else 0,
    "date_coverage_pct": round(float(df['published_at_utc'].notna().mean() * 100), 2) if 'published_at_utc' in df.columns else 0,
    "unique_authors": int(df['author'].nunique()) if 'author' in df.columns else 0,
    "known_authors": int((df['author'] != 'Unknown').sum()) if 'author' in df.columns else 0,
    "title_stats": {
        "mean_length": round(float(df['title_char_count'].mean()), 2) if 'title_char_count' in df.columns else None,
        "median_length": round(float(df['title_char_count'].median()), 2) if 'title_char_count' in df.columns else None,
    },
    "content_stats": {
        "mean_words": round(float(df['content_word_count'].mean()), 2) if 'content_word_count' in df.columns else None,
        "median_words": round(float(df['content_word_count'].median()), 2) if 'content_word_count' in df.columns else None,
    },
    "outputs_directory": str(OUTPUT_DIR.resolve()),
}

with open(OUTPUT_DIR / "eda_summary.json", "w", encoding="utf-8") as f:
    json.dump(eda_summary, f, indent=2, ensure_ascii=False)

print(json.dumps(eda_summary, indent=2, ensure_ascii=False))

# ============================== FINAL OUTPUT ==============================
print("\n" + "=" * 60)
print(f"✅ EDA COMPLETE! All plots and summaries saved to: {OUTPUT_DIR.resolve()}")
print("📂 Generated files:")
for p in sorted(OUTPUT_DIR.iterdir()):
    print(f"   - {p.name}")
print("=" * 60)