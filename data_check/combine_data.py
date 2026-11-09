import pandas as pd

# 1. Define your file paths (using 'r' before the string handles Windows backslashes)
file1 = r"S:\Project_task\data_check\tech_articles_101k.csv"
file2 = r"S:\Project_task\data_check\news_final_clean.csv"
output_file = r"S:\Project_task\data_check\combined_data.csv"

# 2. Read both CSV files
# Note: If you get an encoding error, add encoding='utf-8' or encoding='latin1' inside the read_csv()
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# 3. Combine the dataframes
# ignore_index=True resets the row numbers so they flow continuously
combined_df = pd.concat([df1, df2], ignore_index=True, sort=False)

# 4. Save the combined data to a new CSV file
combined_df.to_csv(output_file, index=False, encoding='utf-8')

print(f"Success! Combined file saved to: {output_file}")