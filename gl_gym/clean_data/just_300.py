import pandas as pd
import numpy as np

# Open CSV file
df = pd.read_csv("Geneina2024.csv")  

# Keep rows with 'time' column value that is a multiple of 300 while using float
df_filtered = df[np.isclose(df['time'] % 300, 0)]

# Save the file after filtering.
df_filtered.to_csv("filtered_file.csv", index=False)
