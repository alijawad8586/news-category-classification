
import os

import pandas as pd
from sqlalchemy import create_engine

# 1. CSV file path
csv_path = r"S:\Project_task\datascrape_project\news_final_clean.csv"

# 2. Read CSV
df = pd.read_csv(csv_path)

print("CSV Loaded Successfully!")
print("Total Rows:", len(df))
print("Columns:", df.columns.tolist())

# 3. PostgreSQL connection
USER = "postgres"
PASSWORD = os.environ.get("DB_PASSWORD")
HOST = "localhost"
PORT = "5432"
DATABASE = "scrap_db"

if not PASSWORD:
    raise RuntimeError("Set the DB_PASSWORD environment variable before running.")

connection_string = (
    f"postgresql+psycopg2://{USER}:{PASSWORD}"
    f"@{HOST}:{PORT}/{DATABASE}"
)

engine = create_engine(connection_string)

# 4. Push CSV data into PostgreSQL
df.to_sql(
    name="news_final_clean",
    con=engine,
    if_exists="replace",
    index=False,
    chunksize=5000,
    method="multi"
)

print("Data successfully uploaded to PostgreSQL!")

# 5. Verify
with engine.connect() as conn:
    from sqlalchemy import text

    result = conn.execute(
        text("SELECT COUNT(*) FROM news_final_clean")
    )

    print("Total Records in PostgreSQL:", result.scalar())

engine.dispose()
