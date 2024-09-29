import pandas as pd
import glob
import os
from datetime import datetime

# Change to your directory containing the CSV files
os.chdir(r'C:\Users\custo\OneDrive\Desktop\medscan\!lovely\Lepera\bills\lepera-gsmrc51-216')

# Get a list of all CSV files in the directory
csv_files = glob.glob('*.csv')

# List to hold dataframes
df_list = []

for file in csv_files:
    # Read the CSV file, skipping empty lines
    df = pd.read_csv(file, skip_blank_lines=True, header=0, dtype=str)
    
    # Remove any rows after 'Confidence Scores % (Table Cell)' if present
    if df.iloc[:, 0].str.contains('Confidence Scores %', na=False).any():
        idx = df[df.iloc[:, 0].str.contains('Confidence Scores %', na=False)].index[0]
        df = df.iloc[:idx]
    
    # Append the dataframe to the list
    df_list.append(df)

# Concatenate all dataframes
combined_df = pd.concat(df_list, ignore_index=True)

# Clean up column names and data by removing leading apostrophes
combined_df.columns = combined_df.columns.str.lstrip("'")
for col in combined_df.columns:
    combined_df[col] = combined_df[col].str.lstrip("'")

# Parse 'DATE OF SERVICE' column to datetime
combined_df['DATE OF SERVICE'] = pd.to_datetime(combined_df['DATE OF SERVICE'], format='%m%d%y', errors='coerce')

# Drop rows with invalid dates
combined_df = combined_df.dropna(subset=['DATE OF SERVICE'])

# Sort the dataframe by 'DATE OF SERVICE'
combined_df = combined_df.sort_values(by='DATE OF SERVICE')

# Reset the index after sorting
combined_df.reset_index(drop=True, inplace=True)

# Save the combined and sorted data to a new CSV file
combined_df.to_csv('combined_sorted.csv', index=False)
