######################### Data Dump for Blocks #########################
########################################################################

import requests
import datetime
import pandas as pd
from tqdm import tqdm  # Optional, for progress bar support

# Define the start and end dates (YYYY, MM, DD)
start_date = datetime.date(2019, 11, 11)
end_date = datetime.date(2020, 11, 09)

# Generate a list of dates
date_range = pd.date_range(start_date, end_date)

# Base URL for downloading the files
base_url = 'https://gz.blockchair.com/bitcoin/blocks/'

# Directory to save the downloaded compressed gz_files
save_dir = '/Path/gz_files/'  # Update this to your desired directory

for single_date in tqdm(date_range):  
    file_date = single_date.strftime("%Y%m%d")  # Format the date as YYYYMMDD
    file_name = f'blockchair_bitcoin_blocks_{file_date}.tsv.gz'  # Construct the file name
    file_url = base_url + file_name  # Construct the file URL
    
    # Send GET request to download file
    response = requests.get(file_url, stream=True)
    
    # Check if request was successful (status code 200)
    if response.status_code == 200:
        # Open a local file with the same name as the remote file
        with open(save_dir + file_name, 'wb') as f:
            for chunk in response.iter_content(chunk_size=128):
                f.write(chunk)
        print(f"Downloaded {file_name}")
    else:
        print(f"Failed to download {file_name}, status code: {response.status_code}")

print("All downloads completed.")

######################### Decompress each file #########################
########################################################################
import gzip
import os

# Directory containing the downloaded .gz files
source_dir = '/Path/gz_files/'

# Directory to save the decompressed .tsv files
output_dir = '/Path/tsv_files/'

# Make sure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through all files in the source directory
for filename in os.listdir(source_dir):
    if filename.endswith('.gz'):
        # Construct the full path to the source file
        gz_path = os.path.join(source_dir, filename)
        
        # Construct the full path to the output file (remove .gz extension)
        tsv_path = os.path.join(output_dir, filename[:-3])  # Removes the last 3 characters ('.gz')
        
        # Open the compressed file, read its contents, and write them to a new .tsv file
        with gzip.open(gz_path, 'rb') as f_in:
            with open(tsv_path, 'wb') as f_out:
                f_out.write(f_in.read())
        
        print(f"Decompressed: {filename} to {tsv_path}")


######################## Combine to Single tsv File ####################
########################################################################
import datetime
import pandas as pd

# Define start and end date (YYYY, MM, DD)
start_date = datetime.date(2019, 11, 11)
end_date = datetime.date(2020, 11, 09)

# Generate list of dates
date_range = pd.date_range(start_date, end_date)

# Base path
base_path = '/Path/tsv_files/'

# List to hold DataFrames
dfs = []

# Loop through each date, generate file path, and read file
for single_date in date_range:
    file_date = single_date.strftime("%Y%m%d")  # Format the date as YYYYMMDD
    file_name = f'blockchair_bitcoin_blocks_{file_date}.tsv.gz'  # Construct the file name
    file_path = base_path + file_name  # Construct the full file path
    try:
        df = pd.read_csv(file_path, sep='\t', compression='gzip')  # Read the TSV file
        dfs.append(df)  # Append the DataFrame to the list
    except FileNotFoundError:
        print(f"File not found: {file_path}, skipping...")

# Concatenate all DataFrames in the list
combined_df = pd.concat(dfs, ignore_index=True)

# Write the combined DataFrame to a new TSV file
combined_df.to_csv('/Path/New_File_Name/', sep='\t', index=False)
