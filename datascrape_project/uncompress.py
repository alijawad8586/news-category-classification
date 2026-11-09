import gzip
import shutil
from pathlib import Path

input_file = Path(
    r"S:\Project_task\datascrape_project\features_output\news_features.csv.gz"
)

output_file = input_file.with_suffix("")

with gzip.open(input_file, "rb") as f_in:
    with open(output_file, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

print("Uncompression completed!")
print(f"Output file: {output_file}")