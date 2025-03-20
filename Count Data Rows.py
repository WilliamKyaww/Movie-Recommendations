import pandas as pd
import os

# Define folder and file paths
folder = "Cleaned Datasets"
tsv_file = os.path.join(folder, "imdb_movielens_matched.tsv")
csv_file = os.path.join(folder, "ratings_imdb_matched.csv")

# Function to count rows excluding headers
def count_rows(file_path, delimiter):
    df = pd.read_csv(file_path, delimiter=delimiter)
    return len(df)

# Get row counts
try:
    tsv_rows = count_rows(tsv_file, '\t')
    csv_rows = count_rows(csv_file, ',')
    print(f"Rows in {tsv_file}: {tsv_rows}")
    print(f"Rows in {csv_file}: {csv_rows}")
except FileNotFoundError as e:
    print(f"Error: {e}")
