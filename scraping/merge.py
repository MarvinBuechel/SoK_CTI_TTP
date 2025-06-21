import pandas as pd
import glob
import os
import html
import string

punct_to_remove = string.punctuation.replace("&", "")
translator = str.maketrans('', '', punct_to_remove)

folder_path = "./data"

csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

print(csv_files)

dfs = []

for file in csv_files:
    df = pd.read_csv(file)
    df['title'] = (
    df['title']
        .apply(html.unescape)  # Convert HTML entities
        .str.lower()           # Convert to lowercase
        .str.translate(translator)  # Remove punctuation
    )
    dfs.append(df)

merged_df = pd.concat(dfs, ignore_index=True)
deduplicated_df = merged_df.drop_duplicates(subset='title', keep='first')
deduplicated_df = deduplicated_df.sort_values(by=["year", "title"])
deduplicated_df.to_csv("merged_papers.csv", index=False)
