import pandas as pd

# Read the original file 2024.csv
df_2024 = pd.read_csv("2024.csv")

# Prepare the new data in the same format as 2001.csv
df_mapped = pd.DataFrame()
df_mapped["time"] = df_2024["time"]
df_mapped["global radiation"] = df_2024["External Total Solar Radiation"]
df_mapped["wind speed"] = df_2024["Wind Speed"]
df_mapped["air temperature"] = df_2024["External Temperature"]
df_mapped["sky temperature"] = df_2024["sky temperature"]
df_mapped["??"] = 0.0
df_mapped["CO2 concentration"] = 400.0
df_mapped["day number"] = pd.to_datetime(df_2024["Date"]).dt.dayofyear.astype(float)
df_mapped["RH"] = df_2024["External Relative Humidity (RH)"]

# We keep the columns arranged like 2001
df_mapped = df_mapped[
    ["time", "global radiation", "wind speed", "air temperature",
     "sky temperature", "??", "CO2 concentration", "day number", "RH"]
]

# Save file
df_mapped.to_csv("2024_mapped.csv", index=False)
print("File created: 2024_mapped.csv")