import pandas as pd

# File load karein
df = pd.read_csv("balanced_news_data.csv")

print("=" * 60)
print("📊 DATA LABELS DISTRIBUTION")
print("=" * 60)

# Total rows
print(f"\n📁 Total Rows: {len(df):,}")

# Har category ki count
print("\n📋 Category-wise Distribution:")
print("-" * 60)
category_counts = df['category'].value_counts()
for category, count in category_counts.items():
    percentage = (count / len(df)) * 100
    print(f"  {category:15} : {count:5,} rows ({percentage:5.1f}%)")

print("-" * 60)
print(f"  {'TOTAL':15} : {len(df):5,} rows")
print("=" * 60)

# Agar aap dekhna chahte hain ke kitne unique labels hain
print(f"\n🏷️  Total Unique Labels: {df['category'].nunique()}")
print(f"📝 Label Names: {list(df['category'].unique())}")