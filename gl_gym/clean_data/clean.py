import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator
from os.path import join

#Define data types to read a CSV file
dtypes = {
    'Date': str,
    'time': str,  # Read as string to handle 'HH:MM'
    'Day/Night': str,
    'Hour': str,  # Read as string to handle 'YYYY-MM-DD HH:MM:SS'
    'External Total Solar Radiation': float,
    'External Temperature': float,
    'External Relative Humidity (RH)': float,
    'Wind Speed': float,
    'Datetime': str
}

# Read CSV file
csv_path = "/Users/fady/Desktop/test/GreenLight-Gym2/gl_gym/environments/weather/Egypt/Geneina2024.csv"
try:
    df = pd.read_csv(csv_path, dtype=dtypes, low_memory=False)
except FileNotFoundError:
    raise FileNotFoundError(f"CSV file not found in path: {csv_path}")
except Exception as e:
    raise Exception(f"An error occurred while reading the file: {e}")

df.columns = [col.strip() for col in df.columns]  # Remove spaces from column names

# Print columns for verification
print("Columns in file:", df.columns.tolist())
print("Data types:\n", df.dtypes)


## Convert 'Hour' column from 'YYYY-MM-DD HH:MM:SS' to decimal numbers
def convert_hour_to_float(hour):
    try:
        if pd.isna(hour):
            return np.nan
        if isinstance(hour, str):
            # Extract hour and minutes from 'YYYY-MM-DD HH:MM:SS'
            time_part = hour.split(' ')[-1]  # Take the part 'HH:MM:SS'
            hours, minutes, _ = map(int, time_part.split(':'))
            return hours + minutes / 60.0
        return float(hour)
    except (ValueError, AttributeError) as e:
        print(f"Failed to convert value '{hour}' in column 'Hour': {e}")
        return np.nan

if 'Hour' in df.columns:
    df['Hour'] = df['Hour'].apply(convert_hour_to_float)
    invalid_hours = df['Hour'].isna()
    if invalid_hours.any():
        print("Invalid values in 'Hour':", df.loc[invalid_hours, 'Hour'].index.tolist())
        print("Sample of original invalid values:", df.loc[invalid_hours, 'Hour'].head().tolist())
        raise ValueError(f"Invalid or missing values in column 'Hour' after conversion. Unique values: {df['Hour'].unique()}")
else:
    raise KeyError(f"The column 'Hour' does not exist in the CSV file. Available columns: {df.columns.tolist()}")

# Convert 'time' column from 'HH:MM' to seconds since the beginning of the day
def convert_time_to_seconds(time_str):
    try:
        if pd.isna(time_str):
            return np.nan
        if isinstance(time_str, str):
            hours, minutes = map(int, time_str.split(':'))
            return hours * 3600 + minutes * 60
        return float(time_str)
    except (ValueError, AttributeError) as e:
        print(f"Failed to convert value '{time_str}' in column 'time': {e}")
        return np.nan

if 'time' in df.columns:
    df['time'] = df['time'].apply(convert_time_to_seconds)
    if df['time'].isna().any():
        raise ValueError(f"Invalid or missing values in column 'time'. Unique values: {df['time'].unique()}")

# Create 'sky temperature' column if missing
if 'sky temperature' not in df.columns:
    df['sky temperature'] = df['External Temperature'] - 5.0
    print("The 'sky temperature' column was created with default values")

# Set column map
column_mapping = {
    'External Total Solar Radiation': 'External Total Solar Radiation',
    'External Temperature': 'External Temperature',
    'External Relative Humidity (RH)': 'External Relative Humidity (RH)',
    'Wind Speed': 'Wind Speed',
    'sky temperature': 'sky temperature',
    'Datetime': 'Datetime'
}

# Check for columns
for input_col, output_col in column_mapping.items():
    if input_col != 'Datetime' and input_col not in df.columns:
        raise KeyError(f"Column '{input_col}' does not exist in CSV file. Available columns: {df.columns.tolist()}")

# Convert 'Datetime' to seconds
time = pd.to_datetime(df['Datetime'], errors='coerce')
if time.isnull().any():
    raise ValueError(f"Invalid date and time values in column 'Datetime'. Unique values: {df['Datetime'].unique()}")

# Print the minimum and maximum dates to check
print(f"Minimum date for 'Datetime': {time.min()}")
print(f"Maximum date for 'Datetime': {time.max()}")

# Correct dates to start from 2024-01-01
if time.min().year != 2024:
    print(f"WARNING: Dates in 'Datetime' are not in the year 2024. Minimum: {time.min()}")
    time_offset = pd.Timestamp("2024-01-01") - time.min()
    time = time + time_offset
    print(f"Dates corrected to start from 2024-01-01. New minimum: {time.min()}")

# Convert dates to seconds since the beginning of 2024
time_seconds = (time - pd.Timestamp("2024-01-01")).dt.total_seconds()

# Print the minimum and maximum time_seconds to check
print(f"Minimum time_seconds: {time_seconds.min()}")
print(f"Maximum time_seconds: {time_seconds.max()}")

# Check that time_seconds contains positive values
if (time_seconds < 0).any():
    raise ValueError(f"The 'time_seconds' values contain negative values after correction. Minimum: {time_seconds.min()}")

# Create a new time grid (5 minute intervals)
start_time = max(time_seconds.iloc[0], 0)  # Ensure start_time is not negative
end_time = time_seconds.iloc[-1]
new_time = np.arange(start_time, end_time + 300, 300)

# Print the minimum and maximum new_time for verification
print(f"Minimum new_time: {new_time.min()}")
print(f"Maximum new_time: {new_time.max()}")

# Completion procedure
interp_df = pd.DataFrame(index=new_time)
for input_col, output_col in column_mapping.items():
    if input_col != 'Datetime':
        print(f"Column processing: {input_col} -> {output_col}")
        if input_col in df.columns:
            interp_df[output_col] = PchipInterpolator(time_seconds, df[input_col])(new_time)
        else:
            raise KeyError(f"Column '{input_col}' does not exist in df during interpolation")



# Print the columns in interp_df for verification
print("Columns in interp_df after interpolation:", interp_df.columns.tolist())


# Add the remaining columns
interp_df['Datetime'] = pd.to_datetime(new_time, unit='s', origin=pd.Timestamp("2024-01-01"))
print(f"Minimum 'Datetime' date in interp_df: {interp_df['Datetime'].min()}")
print(f"Maximum 'Datetime' date in interp_df: {interp_df['Datetime'].max()}")
if interp_df['Datetime'].dt.year.min() != 2024:
    raise ValueError(f"'Datetime' dates in interp_df do not start from 2024. Minimum: {interp_df['Datetime'].dt.year.min()}")


interp_df['time'] = new_time
interp_df['Date'] = interp_df['Datetime'].dt.date.astype(str)
interp_df['Hour'] = interp_df['Datetime'].dt.hour + interp_df['Datetime'].dt.minute / 60.0


# Check if 'External Total Solar Radiation' exists before creating 'Day/Night'
if 'External Total Solar Radiation' not in interp_df.columns:
    raise KeyError(f"Column 'External Total Solar Radiation' does not exist in interp_df. Available columns: {interp_df.columns.tolist()}")
interp_df['Day/Night'] = np.where(interp_df['External Total Solar Radiation'] > 0, 'Day', 'Night')

# Save clean file
output_path = "/Users/fady/Desktop/test/GreenLight-Gym2/gl_gym/environments/weather/Egypt/Geneina2024_clean.csv"
interp_df.to_csv(output_path, index=False)
print(f"Clean data saved to {output_path} with {len(interp_df)} rows")


# Create a Geneina2025.csv file
df_2025 = interp_df.copy()
df_2025['Datetime'] = df_2025['Datetime'] + pd.Timedelta(days=365)
print(f"Minimum 'Datetime' date in df_2025: {df_2025['Datetime'].min()}")
print(f"Maximum 'Datetime' date in df_2025: {df_2025['Datetime'].max()}")
if df_2025['Datetime'].dt.year.min() != 2025:
    raise ValueError(f"'Datetime' dates in df_2025 do not start from 2025. Minimum: {df_2025['Datetime'].dt.year.min()}")
df_2025['Date'] = df_2025['Datetime'].dt.date.astype(str)
df_2025['time'] = (df_2025['Datetime'] - pd.Timestamp("2025-01-01")).dt.total_seconds()
df_2025['Hour'] = df_2025['Datetime'].dt.hour + df_2025['Datetime'].dt.minute / 60.0
df_2025.to_csv("/Users/fady/Desktop/test/GreenLight-Gym2/gl_gym/environments/weather/Egypt/Geneina2025.csv", index=False)
print("Genina2025.csv created")