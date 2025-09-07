
import pandas as pd

csv_path = "/Users/fady/Desktop/test/GreenLight-Gym2/gl_gym/environments/weather/Egypt/Geneina2025.csv"
try:
    df = pd.read_csv(csv_path)
    print("Columns:", df.columns.tolist())
    print("Data Types:\n", df.dtypes)
    print("First 5 rows :\n", df.head())
    print(" First 5 rows   'Datetime':\n", df['Datetime'].head())
    print("Unique values in 'Hour':\n", df['Hour'].unique()[:10])
    print("Unique values in 'time':\n", df['time'].unique()[:10])
    print("Unique values in 'Day/Night':\n", df['Day/Night'].unique())
    print("Unique values in 'External Total Solar Radiation':\n", df['External Total Solar Radiation'].unique()[:10])
    print("Are there any missing values ?:\n", df.isnull().sum())
except FileNotFoundError:
    print(f"CSV file not found at path: {csv_path}")
except Exception as e:
    print(f"An error occurred while reading the file: {e}")





