import os
import pandas as pd
import requests
from tqdm import tqdm

# Base URL where the files are hosted
BASE_URL = "https://www.ncei.noaa.gov/data/oceans/argo/gadr/"

# Path to your txt/csv file
index_file = "data_store.txt"

# Output directory for downloads
output_dir = "argo_nc_files"
os.makedirs(output_dir, exist_ok=True)

# Read the file (comma-separated)
df = pd.read_csv(index_file)

# Iterate through file paths and download
for file_path in tqdm(df["file_path"][1500:], desc="Downloading files"):
    # Construct full URL
    url = BASE_URL + file_path
    
    # Extract filename (last part of path)
    filename = os.path.basename(file_path)
    
    # Full local path
    local_path = os.path.join(output_dir, filename)
    
    # Skip if already downloaded
    if os.path.exists(local_path):
        continue
    
    try:
        # Download the file
        r = requests.get(url, stream=True, timeout=60)
        if r.status_code == 200:
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        else:
            print(f"Failed to download {url} (status {r.status_code})")
    except Exception as e:
        print(f"Error downloading {url}: {e}")
