import pandas as pd

df = pd.read_csv("merged_papers.csv")

df['year'] = pd.to_numeric(df['year'], errors='coerce')

with open("keywords.txt", "r") as f:
    kws = [s.strip().lower() for s in f.readlines() if s.strip()]

filtered_df = df[(df['year'] >= 2015)]

filtered_df = filtered_df[~filtered_df['venue'].fillna('').str.lower().apply(
    lambda venue: any(kw in venue for kw in kws)
)]

filtered_df.to_csv("merged_papers_after_2015.csv", index=False)

