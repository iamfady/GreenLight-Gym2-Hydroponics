import pandas as pd

df = pd.read_csv("Geneina2024.csv")
print("Columns in Geneina2024.csv:", df.columns.tolist())
print("First 5 rows:\n", df.head())