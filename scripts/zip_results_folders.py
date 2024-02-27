## This script finds the folders matching the UUIDs from a CSV file and creates a ZIP file with the folders. 
## I used it to create the zip files for download for auditing.

import os
import zipfile
import pandas
from tqdm import tqdm

def find_folders_and_zip(root_folder, csv_file, output_zip):
    # Read UUIDs from CSV
    uuids = []
    df_uuid = pandas.read_csv(csv_file)
    uuids.extend(df_uuid['UUID'].tolist())

    # Find folders matching UUIDs
    folders_to_zip = []
    for uuid in tqdm(uuids):
        folder_path = os.path.join(root_folder, uuid)
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            folders_to_zip.append(folder_path)

    # Create zip file with collected folders
    with zipfile.ZipFile(output_zip, 'w') as zipf:
        for folder in folders_to_zip:
            for root, dirs, files in os.walk(folder):
                
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, root_folder))

if __name__ == "__main__":
    root_folder = input("Enter the root folder path: ")
    csv_file = input("Enter the path to the CSV file: ")
    output_zip = input("Enter the path for the output ZIP file: ")

    find_folders_and_zip(root_folder, csv_file, output_zip)
    print("Zip file created successfully.")
