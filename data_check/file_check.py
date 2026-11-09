import pandas as pd

# CSV file ka path (r" ka matlab raw string, backslash ka masla nahi hota)
file_path = r"S:\Project_task\data_check\news_final_clean.csv"

# CSV load karo
df = pd.read_csv(file_path)

# 1) Total rows aur columns
rows, cols = df.shape
print("=" * 50)
print(f"Total Rows    : {rows}")
print(f"Total Columns : {cols}")
print("=" * 50)

# 2) Column names
print("\nColumn Names:")
for i, col in enumerate(df.columns, start=1):
    print(f"{i}. {col}")

# 3) Har column mein missing values
missing = df.isnull().sum()
missing_percent = (missing / rows * 100).round(2)

missing_report = pd.DataFrame({
    "Missing Count": missing,
    "Missing %": missing_percent
})

print("\nMissing Values in Each Column:")
print(missing_report)

# 4) Total missing values poore dataset mein
print("\nTotal Missing Values (all columns):", int(missing.sum()))

# 5) Kitni rows mein kam az kam 1 missing value hai
rows_with_missing = df.isnull().any(axis=1).sum()
print("Rows with at least one missing value:", int(rows_with_missing))