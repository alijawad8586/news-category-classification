import pandas as pd
from pathlib import Path

BASE = Path(r"S:\Project_task\datascrape_project")

# Saari CSV files dhoondo
print("="*70)
print("CSV FILES FOUND:")
print("="*70)
csvs = list(BASE.rglob("*.csv"))
for p in csvs:
    print(f"{p}  ({p.stat().st_size/1e6:.1f} MB)")

# Pehli bari file (raw ya preprocessed) uthao
target = None
for p in csvs:
    if "all_scrap" in p.name.lower():
        target = p
        break
if target is None and csvs:
    target = max(csvs, key=lambda p: p.stat().st_size)

if target is None:
    print("\n❌ Koi CSV nahi mili. Path check karein.")
    raise SystemExit

print(f"\n📂 Reading: {target}")
df = pd.read_csv(target, encoding="utf-8-sig", low_memory=False, nrows=2000)

print("\n" + "="*70)
print("EXACT COLUMN NAMES (repr shows spaces):")
print("="*70)
for i, c in enumerate(df.columns):
    print(f"{i:3d}: {repr(c)}")

print("\n" + "="*70)
print("DATE COLUMNS DETAIL:")
print("="*70)
date_cols = [c for c in df.columns if any(k in c.lower() for k in ("date", "time", "publish"))]
for c in date_cols:
    s = df[c]
    print(f"\nColumn {repr(c)} | non-null: {s.notna().sum()}/{len(s)}")
    print("Sample values:", s.dropna().head(5).tolist())
    print("-"*70)