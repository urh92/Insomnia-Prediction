# Import necessary libraries
import os
from zipfile import ZipFile
import pandas as pd
import numpy as np

# Define a class to handle operations on zip files containing EDF files
class ZFile:
    def __init__(self, file_path, batch='batch1', csv_path=None):
        # Initialize the ZFile object with file paths and batch information
        self.file_path = file_path  # Base path where zip files are located
        self.batch = batch  # Batch name or folder within the base path
        self.file_path_batch = os.path.join(self.file_path, self.batch)  # Full path to the batch folder
        self.csv_path = csv_path  # Path to CSV file containing metadata or IDs (optional)
        self.missing_edf = []  # List to keep track of zip files missing EDF files
        self.zip_files = sorted(os.listdir(self.file_path_batch))  # Sorted list of all zip files in the batch folder

        # If a CSV path is provided, filter zip files based on IDs listed in the CSV
        if self.csv_path:
            df = pd.read_csv(csv_path)
            self.zip_files = [z for z in self.zip_files if int(z[:-4]) in df['ID'].unique()]

    def get_content(self, zip_file, which='edf'):
        # Retrieve content of a zip file, filtering for EDF files or non-EDF files based on the 'which' parameter
        zip_file_path = os.path.join(self.file_path_batch, zip_file)
        with ZipFile(zip_file_path, 'r') as zip:
            zip_content = zip.namelist()
            # Check if the content list is a single file (string) or multiple files (list)
            if isinstance(zip_content, str):
                # Single file handling
                if which == 'edf' and not zip_content.lower().endswith(".EDF".lower()):
                    files = np.nan
                elif which == 'non_edf' and not zip_content.lower().endswith(".EDF".lower()):
                    files = zip_content
                else:
                    files = np.nan
            elif isinstance(zip_content, list):
                # Multiple files handling
                files = [z for z in zip_content if z.lower().endswith(".EDF".lower())] if which == 'edf' else [z for z in zip_content if not z.lower().endswith(".EDF".lower())]
                if not files:
                    files = np.nan
        return zip_file, files

    def extract_edfs(self, zip_file_path, output_path):
        # Extract EDF files from a zip file and rename them for clarity
        zip_file, edf_files = self.get_content(zip_file_path)
        with ZipFile(os.path.join(self.file_path_batch, zip_file_path), 'r') as zip:
            # Handle both single and multiple EDF file cases
            if isinstance(edf_files, str):
                zip.extract(edf_files, output_path)
                os.rename(os.path.join(output_path, edf_files), os.path.join(output_path, zip_file + edf_files[-4:]))
            elif isinstance(edf_files, list):
                for i, edf_file in enumerate(sorted(edf_files)):
                    zip.extract(edf_file, output_path)
                    # Rename files to include the zip file name for uniqueness
                    new_name = os.path.join(output_path, zip_file[:-4] + ('' if i == 0 else '_' + str(i)) + edf_file[-4:])
                    os.rename(os.path.join(output_path, edf_file), new_name)

    def get_n_edfs(self, save_dir):
        # Generate a report of the number of EDF files contained within each zip file in the batch
        names, n_edfs = [], []
        for i, zip_file in enumerate(self.zip_files, start=1):
            # Progress update for every 500 files processed
            if (i % 500) == 0:
                print("File {} out of {}".format(i, len(self.zip_files)))
            name, files = self.get_content(zip_file, which='edf')
            names.append(name)
            n_edfs.append(len(files) if not isinstance(files, float) else 0)
        dic = {'Zip Files': names, 'N EDFs': n_edfs}
        df = pd.DataFrame(dic)
        # Save the report as an Excel file in the specified directory
        df.to_excel(os.path.join(save_dir, "zipfiles_info.xlsx"))


if __name__ == '__main__':
    # Example usage of the ZFile class
    path = "/oak/stanford/groups/mignot/psg/Bioserenity"  # Base path to zip files
    c_path = '/scratch/users/umaer/Insomnia/Excel/Subjects_All.csv'  # CSV path for filtering
    output_path = '/scratch/users/umaer/Morpheus_Datalake/Raw PSGs 2'  # Output path for extracted EDFs
    z = ZFile(path, 'batch4/20230206/202111Studies')  # Instantiate ZFile with specific batch
    file_names, contents = [], []
    # Loop through each zip file, extract EDF content information, and store it
    for file in z.zip_files:
        file_name, content = z.get_content(file, which='edf')
        file_names.append(file_name)
        contents.append(content)
    # Save the content information to a CSV file
    pd.DataFrame(list(zip(file_names, contents)), columns=['Files', 'Contents']).to_csv('/scratch/users/umaer/Insomnia/Excel/batch4.csv')
