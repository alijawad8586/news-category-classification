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

# Column jo ignore karni hai
IGNORE_COLUMN = 'publish_date_raw'

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================== LOAD DATA ==============================
print(f"🔄 Loading data from {INPUT_PATH}...")
df = pd.read_csv(INPUT_PATH, low_memory=False)
print(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns.\n")

# Ignore column ko remove karein agar exist karti hai
if IGNORE_COLUMN in df.columns:
    df = df.drop(columns=[IGNORE_COLUMN])
    print(f"⚠️ Ignored column: {IGNORE_COLUMN}")

# ============================== MISSING VALUE ANALYSIS ==============================
print("\n" + "=" * 60)
print("MISSING VALUE ANALYSIS")
print("=" * 60)

# Har column ka missing value count
missing_data = {
    'column': df.columns,
    'total_rows': [len(df)] * len(df.columns),
    'missing_count': df.isna().sum().values,
    'missing_pct': (df.isna().mean() * 100).round(2).values,
    'present_count': df.notna().sum().values,
    'present_pct': (df.notna().mean() * 100).round(2).values
}

missing_df = pd.DataFrame(missing_data)
missing_df = missing_df.sort_values('missing_pct', ascending=False)

# Summary print karein
print("\n📊 Missing Value Summary:")
print(missing_df.to_string(index=False))

# CSV save karein
missing_df.to_csv(OUTPUT_DIR / "missing_value_summary.csv", index=False)
print(f"\n✅ Summary saved to: {OUTPUT_DIR / 'missing_value_summary.csv'}")

# ============================== PLOT ==============================
plt.figure(figsize=(12, 8))

# Bar chart banayein
bars = plt.barh(missing_df['column'], missing_df['missing_pct'], color='coral', alpha=0.7)

plt.title("Missing Values Percentage by Column", fontsize=14, fontweight='bold')
plt.xlabel("Missing (%)")
plt.ylabel("Column Name")
plt.xlim(0, max(100, missing_df['missing_pct'].max() + 5))

# Har bar par value label lagayein
for i, (bar, val) in enumerate(zip(bars, missing_df['missing_pct'])):
    plt.text(val + 0.5, bar.get_y() + bar.get_height()/2, f'{val}%', 
             va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "missing_values_plot.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Plot saved to: {OUTPUT_DIR / 'missing_values_plot.png'}")

# ============================== JSON SUMMARY ==============================
total_missing = int(df.isna().sum().sum())
total_cells = len(df) * len(df.columns)

summary_json = {
    "analysis_type": "missing_value_check",
    "total_rows": int(len(df)),
    "total_columns": int(len(df.columns)),
    "total_cells": total_cells,
    "total_missing_values": total_missing,
    "missing_percentage": round((total_missing / total_cells * 100), 4) if total_cells > 0 else 0,
    "status": "NO_MISSING_VALUES" if total_missing == 0 else "HAS_MISSING_VALUES",
    "ignored_columns": [IGNORE_COLUMN] if IGNORE_COLUMN in pd.read_csv(INPUT_PATH, nrows=0).columns else [],
    "columns_with_missing": int((missing_df['missing_count'] > 0).sum()),
    "columns_without_missing": int((missing_df['missing_count'] == 0).sum()),
    "conclusion": "Dataset is complete with no missing values in any column" if total_missing == 0 else "Dataset contains missing values",
    "output_files": {
        "summary_csv": str((OUTPUT_DIR / "missing_value_summary.csv").resolve()),
        "plot_png": str((OUTPUT_DIR / "missing_values_plot.png").resolve())
    }
}

# JSON save karein
with open(OUTPUT_DIR / "missing_value_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary_json, f, indent=2, ensure_ascii=False)

print(f"\n✅ JSON summary saved to: {OUTPUT_DIR / 'missing_value_summary.json'}")

# ============================== FINAL OUTPUT ==============================
print("\n" + "=" * 60)
print("📋 FINAL SUMMARY")
print("=" * 60)
print(json.dumps(summary_json, indent=2, ensure_ascii=False))
print("\n" + "=" * 60)
print(f"✅ Analysis Complete! All files saved to: {OUTPUT_DIR.resolve()}")
print("=" * 60)